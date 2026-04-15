#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod jit_page;
mod probe;
mod signal_handler;

use jit_page::JitPage;
use probe::Probe;
use signal_handler::{
    clear_probe_flags, clear_sigill_flag, did_sigill_fire, install_sigill_handler,
    set_escape_address,
};

// ╔══════════════════════════════════════╗
// ║  AArch64 Instruction Constants       ║
// ╚══════════════════════════════════════╝

/// AArch64 NOP — no operation.
#[allow(dead_code)]
const NOP: u32 = 0xD503_201F;

/// AArch64 RET — return to caller via x30 (LR).
const RET: u32 = 0xD65F_03C0;

/// AArch64 UDF #0 — permanently undefined, guaranteed SIGILL.
#[allow(dead_code)]
const UDF_0: u32 = 0x0000_0000;

// ╔══════════════════════════════════════╗
// ║  AArch64 Instruction Encoding Helpers║
// ╚══════════════════════════════════════╝

/// Encode `MOVZ Xd, #imm16` — move a 16-bit immediate into a 64-bit register,
/// zeroing the rest.
///
/// Encoding: `1 10 100101 00 <imm16:16> <Rd:5>`
///
/// # Panics
/// Panics if `imm16 > 0xFFFF` or `rd > 30` (x31 is SP/ZR, not a GPR target).
const fn encode_movz_x(rd: u8, imm16: u16) -> u32 {
    assert!(rd <= 30, "rd must be 0..=30 (x31 is SP/ZR)");
    // sf=1 (64-bit), opc=10 (MOVZ), hw=00 (no shift)
    0xD280_0000 | ((imm16 as u32) << 5) | (rd as u32)
}

/// Encode `ADD Xd, Xn, Xm` — 64-bit register add.
///
/// Encoding: `1 00 01011 00 0 <Rm:5> 000000 <Rn:5> <Rd:5>`
///
/// # Panics
/// Panics if any register index > 30.
const fn encode_add_x(rd: u8, rn: u8, rm: u8) -> u32 {
    assert!(rd <= 30 && rn <= 30 && rm <= 30, "registers must be 0..=30");
    0x8B00_0000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

// ╔══════════════════════════════════════╗
// ║  Gate 0 — Minimal Viable Toolchain   ║
// ╚══════════════════════════════════════╝

fn gate_0() {
    println!("── gate 0: inline asm toolchain ──");
    // SAFETY: `nop` has no side effects and no operands. It exists purely
    // to prove the toolchain can compile and execute AArch64 inline assembly.
    unsafe {
        std::arch::asm!("nop");
    }
    println!("alive\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 1 — Allocate an Executable Page║
// ╚══════════════════════════════════════╝

fn gate_1() {
    println!("── gate 1: allocate a JIT page ──");

    // 1. Allocate a MAP_JIT page.
    let page = JitPage::alloc(4096).expect("failed to mmap a JIT page");
    println!("allocated: {page}");
    println!("  debug:   {page:?}");

    // 2. Toggle to writable and write a RET instruction at offset 0.
    page.make_writable();
    page.write_instruction(0, RET);
    println!("wrote RET (0x{RET:08X}) at offset 0");

    // 3. Toggle to executable (also flushes icache).
    page.make_executable();
    println!("flipped to executable + flushed icache");

    // 4. Read back the bytes to confirm the write stuck.
    let readback = page.read_instruction(0);
    assert_eq!(
        readback, RET,
        "readback mismatch: expected 0x{RET:08X}, got 0x{readback:08X}"
    );

    // Pretty-print the raw bytes (little-endian on AArch64).
    let bytes = readback.to_le_bytes();
    println!(
        "readback:  0x{:08X}  [{:02x} {:02x} {:02x} {:02x}]",
        readback, bytes[0], bytes[1], bytes[2], bytes[3]
    );
    println!("✓ page allocated, written, flipped, and verified — no crash, no SIGKILL\n");

    // Page is dropped here → munmap cleans up.
}

// ╔══════════════════════════════════════╗
// ║  Gate 2 — Execute a RET             ║
// ╚══════════════════════════════════════╝

fn gate_2() {
    println!("── gate 2: execute a RET from JIT buffer ──");

    // 1. Allocate a JIT page and write a single RET instruction.
    let page = JitPage::alloc(4096).expect("failed to mmap a JIT page");
    page.make_writable();
    page.write_instruction(0, RET);
    page.make_executable();
    println!("wrote RET at offset 0, page is executable");

    // 2. Call into the JIT buffer. RET will pop LR (x30) and return here.
    println!("jumping into JIT buffer...");
    // SAFETY: The page is in executable mode and contains a single RET at
    // offset 0. The AArch64 `BLR` used by the call sets x30 (LR) to our
    // return address, so RET brings us right back.
    unsafe {
        page.call_void();
    }
    println!("returned from JIT!");
    println!("✓ Rust → JIT buffer → Rust round-trip succeeded\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 3 — Execute Something Useful   ║
// ╚══════════════════════════════════════╝

fn gate_3() {
    println!("── gate 3: execute instructions that do something ──");

    let page = JitPage::alloc(4096).expect("failed to mmap a JIT page");

    // ── Test A: MOV X0, #42 ; RET ──
    {
        let mov_x0_42 = encode_movz_x(0, 42);
        let bytes = mov_x0_42.to_le_bytes();
        println!(
            "MOVZ X0, #42  = 0x{mov_x0_42:08X}  [{:02x} {:02x} {:02x} {:02x}]",
            bytes[0], bytes[1], bytes[2], bytes[3]
        );

        page.make_writable();
        page.write_instruction(0, mov_x0_42); // MOV X0, #42
        page.write_instruction(4, RET);        // RET
        page.make_executable();

        // SAFETY: Page is executable and contains MOVZ X0,#42 + RET.
        // Per the AArch64 calling convention, the return value is in x0.
        let result = unsafe { page.call_ret_u64() };
        assert_eq!(result, 42, "expected x0=42, got x0={result}");
        println!("  called → x0 = {result} ✓");
    }

    // ── Test B: MOV X0, #10 ; MOV X1, #32 ; ADD X0, X0, X1 ; RET ──
    //    (pass two values through registers, add them, return the sum)
    {
        let mov_x0_10 = encode_movz_x(0, 10);
        let mov_x1_32 = encode_movz_x(1, 32);
        let add_x0_x0_x1 = encode_add_x(0, 0, 1);

        println!(
            "MOVZ X0, #10  = 0x{mov_x0_10:08X}\n\
             MOVZ X1, #32  = 0x{mov_x1_32:08X}\n\
             ADD  X0,X0,X1 = 0x{add_x0_x0_x1:08X}"
        );

        page.make_writable();
        page.write_instruction(0, mov_x0_10);     // MOV X0, #10
        page.write_instruction(4, mov_x1_32);     // MOV X1, #32
        page.write_instruction(8, add_x0_x0_x1);  // ADD X0, X0, X1
        page.write_instruction(12, RET);           // RET
        page.make_executable();

        // SAFETY: Page contains a valid 4-instruction sequence ending with RET.
        let result = unsafe { page.call_ret_u64() };
        assert_eq!(result, 42, "expected x0=10+32=42, got x0={result}");
        println!("  called → x0 = {result} (10 + 32) ✓");
    }

    // ── Test C: Use arguments — the caller passes values in x0, x1 ──
    //    fn(a: u64, b: u64) -> u64 { a + b }
    //    Sequence: ADD X0, X0, X1 ; RET
    {
        println!("ADD X0, X0, X1 (args from caller) + RET");

        page.make_writable();
        page.write_instruction(0, encode_add_x(0, 0, 1)); // ADD X0, X0, X1
        page.write_instruction(4, RET);                     // RET
        page.make_executable();

        // SAFETY: Page contains ADD X0,X0,X1 + RET. We cast to a function
        // that takes two u64 args (x0, x1) and returns u64 (x0).
        let f: extern "C" fn(u64, u64) -> u64 = unsafe {
            core::mem::transmute(page.as_ptr())
        };
        let result = f(100, 23);
        assert_eq!(result, 123, "expected 100+23=123, got {result}");
        println!("  called(100, 23) → x0 = {result} ✓");

        let result2 = f(0xFFFF_FFFF, 1);
        println!("  called(0xFFFFFFFF, 1) → x0 = {result2} (0x{result2:X}) ✓");
    }

    println!("✓ arbitrary runtime-emitted instructions execute and return values\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 4 — SIGILL Recovery            ║
// ╚══════════════════════════════════════╝

fn gate_4() {
    println!("── gate 4: SIGILL recovery ──");

    install_sigill_handler();
    println!("installed SIGILL handler");

    let page = JitPage::alloc(4096).expect("failed to mmap a JIT page");

    // ── Test A: UDF #0 should trigger SIGILL, handler skips it, RET returns ──
    {
        page.make_writable();
        page.write_instruction(0, UDF_0);
        page.write_instruction(4, RET);
        page.make_executable();

        // Escape address = the RET at offset 4.
        set_escape_address(page.as_ptr() as u64 + 4);
        clear_sigill_flag();
        println!("executing UDF #0 (0x{UDF_0:08X}) + RET ...");

        // SAFETY: UDF + RET. The SIGILL handler redirects PC to RET,
        // which returns control here.
        unsafe {
            page.call_void();
        }

        let fired = did_sigill_fire();
        assert!(fired, "SIGILL should have fired for UDF #0");
        println!("  SIGILL fired: {fired} ✓  (recovered, continued to RET)");
    }

    // ── Test B: NOP should NOT trigger SIGILL ──
    {
        page.make_writable();
        page.write_instruction(0, NOP);
        page.write_instruction(4, RET);
        page.make_executable();

        set_escape_address(page.as_ptr() as u64 + 4);
        clear_sigill_flag();
        println!("executing NOP (0x{NOP:08X}) + RET ...");

        // SAFETY: NOP + RET is perfectly valid.
        unsafe {
            page.call_void();
        }

        let fired = did_sigill_fire();
        assert!(!fired, "NOP should not trigger SIGILL");
        println!("  SIGILL fired: {fired} ✓  (NOP executed cleanly)");
    }

    // ── Test C: Multiple UDFs in a row ──
    {
        page.make_writable();
        page.write_instruction(0, UDF_0);   // SIGILL → escape
        page.write_instruction(4, UDF_0);
        page.write_instruction(8, UDF_0);
        page.write_instruction(12, RET);    // return
        page.make_executable();

        // Escape to the final RET.
        set_escape_address(page.as_ptr() as u64 + 12);
        clear_sigill_flag();
        println!("executing UDF ; UDF ; UDF ; RET ...");

        // SAFETY: Three UDFs + RET. First UDF triggers SIGILL, handler
        // redirects to escape (RET at offset 12), which returns.
        unsafe {
            page.call_void();
        }

        let fired = did_sigill_fire();
        assert!(fired, "SIGILL should have fired");
        println!("  SIGILL fired: {fired} ✓  (survived UDFs via escape)");
    }

    // ── Test D: Interleaved — MOV X0,#99 ; UDF ; MOV X0,#42 ; RET ──
    //    The UDF fires SIGILL, handler escapes to the final RET.
    //    Since we escape to RET, x0 retains whatever value it had when
    //    UDF faulted (99, from the first MOV).
    {
        page.make_writable();
        page.write_instruction(0, encode_movz_x(0, 99));   // MOV X0, #99
        page.write_instruction(4, UDF_0);                    // SIGILL → escape
        page.write_instruction(8, encode_movz_x(0, 42));   // MOV X0, #42
        page.write_instruction(12, RET);                     // return
        page.make_executable();

        // Escape address = RET at offset 12.
        set_escape_address(page.as_ptr() as u64 + 12);
        clear_probe_flags();
        println!("executing MOV X0,#99 ; UDF ; MOV X0,#42 ; RET ...");

        // SAFETY: MOV X0,#99 runs, UDF faults → handler escapes to RET.
        // x0 = 99 because the escape skips the second MOV.
        let result = unsafe { page.call_ret_u64() };

        let fired = did_sigill_fire();
        assert!(fired, "SIGILL should have fired for the UDF");
        assert_eq!(result, 99, "expected x0=99 (escape skipped second MOV), got {result}");
        println!("  SIGILL fired: {fired}, x0 = {result} ✓  (escaped directly to RET)");
    }

    println!("✓ SIGILL handler installed, UDF survived, execution continues\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 5 — The Execution Harness      ║
// ╚══════════════════════════════════════╝

fn gate_5() {
    println!("── gate 5: the execution harness ──");

    let probe = Probe::new();

    // ── Known-good / known-bad spot checks ──
    println!("spot checks:");
    for &opcode in &[NOP, RET, UDF_0] {
        let result = probe.run(opcode);
        println!("  {result}");
    }

    // ── Timeout spot check: B . (branch to self = 0x14000000) ──
    {
        const BRANCH_TO_SELF: u32 = 0x1400_0000;
        println!("  probing B . (branch-to-self, 0x{BRANCH_TO_SELF:08X}) — should timeout...");
        let result = probe.run(BRANCH_TO_SELF);
        println!("  {result}");
        assert!(result.timed_out, "branch-to-self should time out");
    }
    println!();

    // ── Sweep: UDF range (0x00000000..0x00000100) — all should fault ──
    {
        println!("sweep: UDF range 0x00000000..0x000000FF (256 opcodes)");
        let (_, summary) = probe.sweep(0x0000_0000..0x0000_0100);
        println!("  {summary}");
        assert_eq!(summary.faulted, 256, "all UDF-range opcodes should fault");
        println!("  all faulted ✓");
        println!();
    }

    // ── Sweep: NOP neighbourhood — explore around the NOP encoding ──
    //    NOP = 0xD503201F. Let's vary the low 5 bits (Rt field in HINT space).
    //    HINT instructions (0xD503201F with Rt=0..31) should all be valid.
    {
        println!("sweep: HINT space 0xD5032000..0xD503201F (32 opcodes)");
        let (results, summary) = probe.sweep(0xD503_2000..0xD503_2020);
        println!("  {summary}");
        println!("  sample results:");
        for r in results.iter().take(8) {
            println!("    {r}");
        }
        if summary.ok > 8 {
            println!("    ... ({} more ok)", summary.ok - 8);
        }
        println!();
    }

    // ── Sweep: wider scan from a few interesting regions ──
    //    ALU region: ADD Xd, Xn, Xm (base 0x8B000000).
    //    These opcodes write to various registers including callee-saved
    //    ones, which previously caused infinite loops. The sigsetjmp/siglongjmp
    //    recovery handles this correctly.
    {
        use std::io::Write;
        let base: u32 = 0x8B00_0000;
        let count: u32 = 4096;
        println!("sweep: ALU region 0x{base:08X}..0x{:08X} ({count} opcodes)", base + count);
        
        for i in 0..count {
            let opcode = base + i;
            if i % 256 == 0 {
                print!("\n  0x{opcode:08X}..0x{:08X} ", base + i + 256);
            }
            std::io::stdout().flush().unwrap();
            probe.run(opcode);
            print!(".");
        }
        println!("\n  all ALU opcodes passed (no crash)");
        println!();
    }

    // ── Throughput test: 10,000 SIGILL probes ──
    {
        println!("throughput: 10,000 UDF probes (all SIGILL)");
        let (_, summary) = probe.sweep(0..10_000);
        println!("  {summary}");
        println!();
    }

    // ── Throughput test: 10,000 NOP probes (no SIGILL) ──
    {
        println!("throughput: 10,000 NOP probes (no SIGILL)");
        let (_, summary) = probe.sweep(std::iter::repeat(NOP).take(10_000));
        println!("  {summary}");
        println!();
    }

    // ── Mixed region sweep (includes possible timeouts) ──
    {
        // Branch/exception region — likely to contain hangs.
        // The timeout mechanism handles these gracefully.
        let base: u32 = 0x1400_0000; // Unconditional branch space
        let count: u32 = 256;
        println!("sweep: 0x{base:08X}..0x{:08X} ({count} opcodes, may include timeouts)", base + count);
        let (_, summary) = probe.sweep(base..base + count);
        println!("  {summary}");
        println!();
    }

    println!("✓ probe harness operational — can sweep arbitrary opcode ranges\n");
}

// ╔══════════════════════════════════════╗
// ║  Main                                ║
// ╚══════════════════════════════════════╝

fn main() {
    gate_0();
    gate_1();
    gate_2();
    gate_3();
    gate_4();
    gate_5();
}
