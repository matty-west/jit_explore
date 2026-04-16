#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod cpu_state;
mod emitter;
mod jit_page;
mod probe;
mod signal_handler;
mod sink;

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

#[allow(dead_code)]
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
        let base: u32 = 0x8B00_0000;
        let count: u32 = 4096;
        println!("sweep: ALU region 0x{base:08X}..0x{:08X} ({count} opcodes)", base + count);
        let (_, summary) = probe.sweep(base..base + count);
        println!("  {summary}");
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
// ║  Gate 6 — Register Snapshot Observer ║
// ╚══════════════════════════════════════╝

fn gate_6() {
    println!("── gate 6: register snapshot observer ──");

    let probe = Probe::new();

    // ── Test A: MOV X5, #42 — should mutate exactly one register ──
    {
        // MOVZ X5, #42 = 0xD2800545
        let mov_x5_42: u32 = 0xD280_0545;
        println!("observed probe: MOV X5, #42 (0x{mov_x5_42:08X})");
        let result = probe.run_observed(mov_x5_42);
        println!("  status: {}", result.base.status());
        assert!(!result.base.faulted, "MOV X5,#42 should not fault");
        assert!(!result.snapshot_corrupted, "snapshot should be intact");

        if let Some(ref pre) = result.pre {
            println!("  pre  x5 = 0x{:016X}", pre.reg(5));
        }
        if let Some(ref post) = result.post {
            println!("  post x5 = 0x{:016X}", post.reg(5));
        }

        println!("  mutations:");
        for d in &result.diff {
            println!("    {d}");
        }

        // Verify x5 changed to 42.
        assert!(
            result.diff.iter().any(|d| d.index == 5 && d.post == 42),
            "expected x5 to be mutated to 42, diff: {:?}",
            result.diff
        );
        println!("  ✓ x5 mutated to 42");
    }
    println!();

    // ── Test B: NOP — should mutate no registers ──
    {
        println!("observed probe: NOP (0x{NOP:08X})");
        let result = probe.run_observed(NOP);
        assert!(!result.base.faulted, "NOP should not fault");
        assert!(!result.snapshot_corrupted, "snapshot should be intact");
        assert!(
            result.diff.is_empty(),
            "NOP should not change any registers, but got: {:?}",
            result.diff
        );
        println!("  ✓ no GPR changes");
    }
    println!();

    // ── Test C: ADD X0, X0, X1 — should mutate X0 ──
    {
        // ADD X0, X0, X1 = 0x8B010000
        let add_x0_x0_x1: u32 = 0x8B01_0000;
        println!("observed probe: ADD X0, X0, X1 (0x{add_x0_x0_x1:08X})");
        let result = probe.run_observed(add_x0_x0_x1);
        assert!(!result.base.faulted, "ADD should not fault");
        assert!(!result.snapshot_corrupted, "snapshot should be intact");

        println!("  mutations:");
        for d in &result.diff {
            println!("    {d}");
        }

        // X0 should change (it's the sum of seeded X0 + seeded X1).
        assert!(
            result.diff.iter().any(|d| d.index == 0),
            "expected x0 to be mutated by ADD, diff: {:?}",
            result.diff
        );
        println!("  ✓ x0 mutated by ADD");
    }
    println!();

    // ── Test D: UDF — should fault, no post snapshot ──
    {
        println!("observed probe: UDF #0 (0x{UDF_0:08X})");
        let result = probe.run_observed(UDF_0);
        assert!(result.base.faulted, "UDF should fault");
        assert!(result.post.is_none(), "post snapshot should be None on fault");
        assert!(result.diff.is_empty(), "diff should be empty on fault");
        println!("  ✓ faulted, no post snapshot");
    }
    println!();

    println!("✓ register snapshot observer operational\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 7 — The Data Sink              ║
// ╚══════════════════════════════════════╝

fn gate_7() {
    use std::path::Path;
    use crate::signal_handler::{install_sigint_handler, was_interrupted, clear_interrupted};
    use crate::sink::ResultSink;

    println!("── gate 7: the data sink ──");

    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir).expect("failed to create output/ directory");
    let jsonl_path = output_dir.join("gate7_test.jsonl");

    // ── Test A: Write a small observed sweep to JSONL ──
    {
        // Clean slate for the test.
        let _ = std::fs::remove_file(&jsonl_path);

        let probe = Probe::new();
        let mut sink = ResultSink::new(&jsonl_path).expect("failed to open sink");

        // Sweep the HINT space (NOP neighbourhood): 0xD5032000..0xD5032020 (32 opcodes)
        // plus some UDF range: 0x00000000..0x00000020 (32 opcodes).
        // Total: 64 opcodes — quick, predictable results.
        let opcodes = (0xD503_2000u32..0xD503_2020).chain(0x0000_0000u32..0x0000_0020);

        println!("  writing observed sweep to {}", jsonl_path.display());
        let (summary, _interrupted) = probe.observed_sweep(opcodes, &mut sink, 32);
        println!("  {summary}");
        println!("  total written: {} records", sink.total_written());

        // Verify file exists and has the right number of lines.
        let line_count = std::fs::read_to_string(&jsonl_path)
            .expect("read jsonl")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        assert_eq!(
            line_count, 64,
            "expected 64 JSONL lines, got {line_count}"
        );
        println!("  ✓ {line_count} JSON lines written");

        // Verify the last opcode.
        let last = ResultSink::last_opcode(&jsonl_path);
        assert_eq!(last, Some(0x1F), "expected last opcode 0x1F, got {last:?}");
        println!("  ✓ last opcode: 0x{:08X}", last.unwrap());

        // Spot-check: parse the first line.
        let first_line = std::fs::read_to_string(&jsonl_path)
            .expect("read")
            .lines()
            .next()
            .expect("at least one line")
            .to_string();
        let record: serde_json::Value =
            serde_json::from_str(&first_line).expect("parse first line");
        assert_eq!(record["v"], 1, "schema version should be 1");
        assert_eq!(record["opcode"], "0xD5032000", "first opcode");
        println!("  ✓ first record: v={}, opcode={}", record["v"], record["opcode"]);
    }
    println!();

    // ── Test B: Resume from an existing file ──
    {
        // The file from Test A has 64 records (last opcode = 0x1F).
        // Simulate a resume by reading last_opcode and continuing.
        let resume_from = ResultSink::last_opcode(&jsonl_path)
            .map(|op| op + 1)
            .unwrap_or(0);
        println!("  resume test: last_opcode + 1 = 0x{resume_from:08X}");

        let probe = Probe::new();
        let mut sink = ResultSink::new(&jsonl_path).expect("failed to open sink for resume");

        // Append 32 more UDF opcodes.
        let opcodes = resume_from..(resume_from + 32);
        let (summary, _) = probe.observed_sweep(opcodes, &mut sink, 0);
        println!("  appended {} more records", summary.total);

        // Verify total lines.
        let line_count = std::fs::read_to_string(&jsonl_path)
            .expect("read jsonl")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        assert_eq!(
            line_count, 96,
            "expected 96 JSONL lines after resume, got {line_count}"
        );

        let last = ResultSink::last_opcode(&jsonl_path).unwrap();
        assert_eq!(last, resume_from + 31);
        println!("  ✓ resume works: {line_count} total lines, last = 0x{last:08X}");
    }
    println!();

    // ── Test C: SIGINT handling ──
    {
        install_sigint_handler();
        clear_interrupted();

        // We can't easily send ourselves SIGINT in an automated test without
        // disrupting the terminal, so just verify the flag mechanism works.
        assert!(!was_interrupted(), "should not be interrupted yet");
        println!("  ✓ SIGINT handler installed, flag is clear");
        println!("  (manual test: run a large sweep and press Ctrl+C to verify graceful stop)");
    }
    println!();

    // ── Test D: Validate JSONL with jq-style parsing ──
    {
        // Parse every line to verify the entire file is valid JSON.
        let content = std::fs::read_to_string(&jsonl_path).expect("read jsonl");
        let mut valid = 0;
        let mut invalid = 0;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<crate::sink::SinkRecord>(line) {
                Ok(_) => valid += 1,
                Err(e) => {
                    eprintln!("  invalid line: {e}");
                    invalid += 1;
                }
            }
        }
        assert_eq!(invalid, 0, "all lines should be valid SinkRecords");
        println!("  ✓ all {valid} lines are valid SinkRecords");
    }
    println!();

    // Print file size.
    let metadata = std::fs::metadata(&jsonl_path).expect("file metadata");
    let size_kb = metadata.len() as f64 / 1024.0;
    println!("  output file: {} ({size_kb:.1} KiB)", jsonl_path.display());

    println!("✓ data sink operational — JSONL logging with resume\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 8 — Waking the Matrix          ║
// ╚══════════════════════════════════════╝

fn gate_8() {
    use crate::emitter::{SMSTART, SMSTOP};

    println!("── gate 8: waking the matrix (SME enable) ──");

    let probe = Probe::new();

    // ── Test A: Probe SMSTART directly ──
    //    SMSTART = MSR SVCRSM, #1 = 0xD503437F
    //    sysctl says FEAT_SME=1, so this should NOT fault.
    {
        println!("  probing SMSTART (0x{SMSTART:08X})...");
        let result = probe.run(SMSTART);
        println!("    {result}");
        if result.faulted {
            println!("  ✗ SMSTART faulted — SME may require entitlements or kernel enable.");
            println!("    sysctl reports FEAT_SME=1 but macOS may be guarding access.");
            println!("    STOPPING gate 8 — investigate with Frida/Accelerate before retrying.");
            return;
        }
        assert!(!result.timed_out, "SMSTART should not timeout");
        assert!(!result.segfaulted, "SMSTART should not segfault");
        println!("    ✓ SMSTART executed without fault — streaming mode entered");
    }
    println!();

    // ── Test B: Probe SMSTOP ──
    //    SMSTOP = MSR SVCRSM, #0 = 0xD503427F
    //    Should always work after SMSTART.
    {
        println!("  probing SMSTOP (0x{SMSTOP:08X})...");
        let result = probe.run(SMSTOP);
        println!("    {result}");
        assert!(!result.faulted, "SMSTOP should not fault");
        println!("    ✓ SMSTOP executed — streaming mode exited");
    }
    println!();

    // ── Test C: SMSTART in observed mode ──
    //    Use run_observed_with_prefix to probe SMSTART *with* SMSTOP as a
    //    suffix. Without the suffix the postlude executes in streaming SVE
    //    mode — STP of X-regs technically still works, but on M4 the CPU
    //    hangs (possibly an SVL-dependent pipeline stall). Adding SMSTOP as
    //    suffix ensures the postlude runs in normal (non-streaming) mode.
    {
        println!("  observed probe: SMSTART (with SMSTOP suffix)");
        let result = probe.run_observed_with_prefix(SMSTART, &[], &[SMSTOP]);
        println!("    status: {}", result.base.status());
        assert!(!result.base.faulted, "SMSTART should not fault in observed mode");

        if result.diff.is_empty() {
            println!("    no GPR changes (expected — SMSTART modifies PSTATE, not GPRs)");
        } else {
            println!("    GPR mutations (unexpected!):");
            for d in &result.diff {
                println!("      {d}");
            }
        }
        println!("    ✓ SMSTART observed");
    }
    println!();

    // ── Test D: Probe a known SME instruction ──
    //    FMOPA ZA0.S, P0/M, Z0.S, Z0.S  =  0x80800000
    //    This is a Streaming SVE outer product — requires streaming mode.
    //
    //    Step 1: Bare probe (no SMSTART) → should SIGILL.
    //    Step 2: Manual 4-instruction sequence via run() to test raw execution.
    //    Step 3: Observed probe with prefix/suffix (if step 2 passes).
    {
        const FMOPA_ZA0_S: u32 = 0x8080_0000;

        // Step 1: without SMSTART — should fault.
        println!("  probing FMOPA ZA0.S (0x{FMOPA_ZA0_S:08X}) WITHOUT SMSTART...");
        let result_bare = probe.run(FMOPA_ZA0_S);
        println!("    {result_bare}");

        // Step 2: Manual sequence — write SMSTART + FMOPA + SMSTOP + RET
        //         directly into the page and execute via call_void.
        //         This bypasses the observed prelude/postlude entirely.
        println!("  manual sequence: SMSTART → FMOPA → SMSTOP → RET...");
        {
            use crate::signal_handler::{
                enable_longjmp, disable_longjmp, clear_probe_flags,
                arm_alarm, disarm_alarm, did_sigill_fire, did_timeout,
                did_segfault, did_trap, sigsetjmp, JMP_BUF,
            };

            let page = jit_page::JitPage::alloc(4096)
                .expect("alloc page for manual FMOPA test");
            crate::signal_handler::set_probe_bounds(
                page.as_ptr() as u64,
                page.as_ptr() as u64 + page.size() as u64,
            );

            page.make_writable();
            page.write_instruction(0, SMSTART);
            page.write_instruction(4, FMOPA_ZA0_S);
            page.write_instruction(8, SMSTOP);
            page.write_instruction(12, 0xD65F_03C0); // RET
            page.make_executable();

            set_escape_address(page.as_ptr() as u64 + 16);
            clear_probe_flags();
            arm_alarm(5_000); // 5ms timeout

            let ret = sigsetjmp(JMP_BUF.as_mut_ptr(), 1);
            if ret == 0 {
                enable_longjmp();
                // SAFETY: Page contains SMSTART + FMOPA + SMSTOP + RET.
                // Faults/hangs recover via siglongjmp.
                unsafe { page.call_void(); }
                disable_longjmp();
            } else {
                disable_longjmp();
            }
            disarm_alarm();

            let faulted = did_sigill_fire();
            let timed_out = did_timeout();
            let segfaulted = did_segfault();
            let trapped = did_trap();

            if faulted {
                println!("    SIGILL — FMOPA faults even with SMSTART");
            } else if timed_out {
                println!("    TIMEOUT — FMOPA hangs (5ms alarm fired)");
            } else if segfaulted {
                println!("    SEGV — FMOPA segfaults");
            } else if trapped {
                println!("    TRAP — FMOPA trapped");
            } else {
                println!("    ✓ FMOPA executed: SMSTART → FMOPA → SMSTOP → RET succeeded");
            }

            // Restore probe bounds to the shared probe's page.
            crate::signal_handler::set_probe_bounds(
                probe.page_ptr() as u64,
                probe.page_ptr() as u64 + probe.page_size() as u64,
            );
        }

        // Step 3: Observed probe with prefix/suffix — only if step 2 worked.
        // (Deferred until we know the basic sequence works.)
    }
    println!();

    // ── Test E: Mini sweep of SME encoding space ──
    //    Use simple run() with manual SMSTART/SMSTOP wrapping.
    //    SME outer products live around 0x80800000..0x80FFFFFF.
    {
        use crate::signal_handler::{
            enable_longjmp, disable_longjmp, clear_probe_flags,
            arm_alarm, disarm_alarm, did_sigill_fire, did_timeout,
            did_segfault, did_trap, sigsetjmp, JMP_BUF,
        };

        let base: u32 = 0x8080_0000;
        let count: u32 = 64;
        println!("  mini sweep: SME space 0x{base:08X}..0x{:08X} ({count} opcodes)", base + count);

        let page = jit_page::JitPage::alloc(4096)
            .expect("alloc page for SME sweep");
        crate::signal_handler::set_probe_bounds(
            page.as_ptr() as u64,
            page.as_ptr() as u64 + page.size() as u64,
        );

        let mut ok = 0u32;
        let mut faulted = 0u32;
        let mut timed_out_count = 0u32;
        let mut other = 0u32;

        for i in 0..count {
            let opcode = base + i;

            page.make_writable();
            page.write_instruction(0, SMSTART);
            page.write_instruction(4, opcode);
            page.write_instruction(8, SMSTOP);
            page.write_instruction(12, 0xD65F_03C0); // RET
            page.make_executable();

            set_escape_address(page.as_ptr() as u64 + 16);
            clear_probe_flags();
            arm_alarm(5_000);

            let ret = sigsetjmp(JMP_BUF.as_mut_ptr(), 1);
            if ret == 0 {
                enable_longjmp();
                // SAFETY: Page contains SMSTART + opcode + SMSTOP + RET.
                unsafe { page.call_void(); }
                disable_longjmp();
            } else {
                disable_longjmp();
            }
            disarm_alarm();

            if did_sigill_fire() {
                faulted += 1;
            } else if did_timeout() {
                timed_out_count += 1;
            } else if did_segfault() || did_trap() {
                other += 1;
            } else {
                ok += 1;
            }
        }
        println!("    {count} probed: {ok} ok, {faulted} SIGILL, {timed_out_count} timeout, {other} other");
        if ok > 0 {
            println!("    ✓ {ok} SME instructions executed with SMSTART/SMSTOP wrapping!");
        }

        // Restore probe bounds.
        crate::signal_handler::set_probe_bounds(
            probe.page_ptr() as u64,
            probe.page_ptr() as u64 + probe.page_size() as u64,
        );
    }
    println!();

    println!("✓ matrix coprocessor probing operational\n");
}

// ╔══════════════════════════════════════╗
// ║  Gate 9 — Autonomous AMX/SME Sweep   ║
// ╚══════════════════════════════════════╝

fn gate_9() {
    use std::path::Path;
    use crate::emitter::{SMSTART, SMSTOP, encode_amx_store_tile};
    use crate::probe::{Probe, ProbeClassification};
    use crate::signal_handler::install_sigint_handler;
    use crate::sink::ResultSink;

    println!("── gate 9: autonomous AMX/SME sweep ──");

    install_sigint_handler();

    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir).expect("failed to create output/");

    // ── Step 1: Validate AMX store encoding ──────────────────────────────────
    // The corsix/amx reverse-engineered encoding has never been executed on this
    // M4. Probe it in a fork before trusting it in the postlude.
    println!("\n  [1/4] Validating AMX store tile encoding...");
    let amx_encoding_valid = {
        use crate::jit_page::JitPage;

        // Allocate a 4KB scratch buffer in shared memory (mmap MAP_ANON | MAP_SHARED).
        let scratch_size = 4096usize;
        let scratch_buf = unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                scratch_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_SHARED,
                -1,
                0,
            );
            assert!(ptr != libc::MAP_FAILED, "mmap scratch failed");
            ptr as *mut u8
        };

        // Emit: SMSTART → load scratch addr into X0 → AMX_ST_TILE(T0, X0) → SMSTOP → RET
        let page = JitPage::alloc(4096).expect("alloc validation page");
        page.make_writable();
        let mut off = 0usize;

        // SMSTART
        page.write_instruction(off, SMSTART); off += 4;

        // MOVZ X0, #lo16(scratch_buf_ptr) ; MOVK for upper bits
        let ptr_val = scratch_buf as u64;
        let emit_movz = |rd: u8, imm16: u16, shift: u8| -> u32 {
            let hw = (shift / 16) as u32;
            0xD280_0000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32)
        };
        let emit_movk = |rd: u8, imm16: u16, shift: u8| -> u32 {
            let hw = (shift / 16) as u32;
            0xF280_0000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32)
        };
        // Load all 64 bits of scratch_buf into X0
        let chunks: [u16; 4] = [
            (ptr_val & 0xFFFF) as u16,
            ((ptr_val >> 16) & 0xFFFF) as u16,
            ((ptr_val >> 32) & 0xFFFF) as u16,
            ((ptr_val >> 48) & 0xFFFF) as u16,
        ];
        page.write_instruction(off, emit_movz(0, chunks[0], 0));  off += 4;
        if chunks[1] != 0 { page.write_instruction(off, emit_movk(0, chunks[1], 16)); off += 4; }
        if chunks[2] != 0 { page.write_instruction(off, emit_movk(0, chunks[2], 32)); off += 4; }
        if chunks[3] != 0 { page.write_instruction(off, emit_movk(0, chunks[3], 48)); off += 4; }

        // AMX_ST_TILE(T0, X0)  — store tile 0 to [X0]
        page.write_instruction(off, encode_amx_store_tile(0, 0)); off += 4;

        // SMSTOP
        page.write_instruction(off, SMSTOP); off += 4;

        // RET
        page.write_instruction(off, RET);
        // off += 4; — last instruction, no need to advance offset further
        page.make_executable();

        let mut ok = false;
        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                libc::signal(libc::SIGILL,  libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS,  libc::SIG_DFL);
                page.make_executable();
                page.call_void();
                libc::_exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                let start = std::time::Instant::now();
                loop {
                    let ret = libc::waitpid(pid, &mut status, libc::WNOHANG);
                    if ret == pid { break; }
                    if start.elapsed() > std::time::Duration::from_millis(500) {
                        libc::kill(pid, libc::SIGKILL);
                        libc::waitpid(pid, &mut status, 0);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0 {
                    ok = true;
                }
            } else {
                panic!("fork failed");
            }
        }

        // Free scratch buffer.
        unsafe { libc::munmap(scratch_buf as *mut libc::c_void, scratch_size); }

        ok
    };

    if amx_encoding_valid {
        println!("    ✓ AMX store tile encoding validated — streaming postlude will capture tiles");
    } else {
        println!("    ✗ AMX store tile encoding INVALID (SIGILL or timeout)");
        println!("      Streaming sweeps will still run but AMX tile capture is disabled.");
        println!("      GPR-only observation is still valid for Gate 9.");
    }

    // ── Step 2: Define opcode ranges ─────────────────────────────────────────
    // ALU baseline (streaming=false) — ADD variants.
    let alu_range   = 0x8B00_0000u32..0x8B00_FFFFu32; // 64 K opcodes
    // SME outer product space (streaming=true).
    let sme_range   = 0x8080_0000u32..0x80FF_FFFFu32; // ~8 M opcodes
    // Apple AMX instruction space (streaming=true, if encoding validated).
    let amx_range   = 0x0020_0000u32..0x002F_FFFFu32; // ~1 M opcodes

    let probe = Probe::new();

    // ── Step 3: ALU baseline sweep (streaming=false) ─────────────────────────
    println!("\n  [2/4] ALU baseline sweep (streaming=false): {:?} ({} opcodes)",
        alu_range, alu_range.end - alu_range.start);
    let alu_path = output_dir.join("gate9_alu.jsonl");
    let _ = std::fs::remove_file(&alu_path);
    {
        let mut sink = ResultSink::new(&alu_path).expect("open alu sink");
        let (summary, interrupted) = probe.observed_sweep(alu_range.clone(), &mut sink, 4096);
        println!("    {summary}");
        if interrupted { println!("    (interrupted)"); }
        // Classification breakdown.
        println!("    output: {}", alu_path.display());
    }

    // ── Step 4: SME sweep (streaming=true) ───────────────────────────────────
    println!("\n  [3/4] SME sweep (streaming=true): {:?} (~{:.0}K opcodes)",
        sme_range, (sme_range.end - sme_range.start) as f64 / 1000.0);
    let sme_path = output_dir.join("gate9_sme.jsonl");
    let resume_from = ResultSink::last_opcode(&sme_path)
        .map(|op| op.saturating_add(1))
        .unwrap_or(sme_range.start);
    println!("    resuming from 0x{resume_from:08X}");
    {
        let mut sink = ResultSink::new(&sme_path).expect("open sme sink");
        let opcodes = resume_from..sme_range.end;
        let (summary, interrupted) = probe.observed_sweep_streaming(opcodes, &mut sink, 50_000);
        println!("    {summary}");
        if interrupted { println!("    (interrupted — resume supported)"); }
        println!("    output: {}", sme_path.display());
    }

    // ── Step 5: AMX sweep (streaming=true, only if encoding validated) ────────
    if amx_encoding_valid {
        println!("\n  [4/4] AMX sweep (streaming=true): {:?} (~{:.0}K opcodes)",
            amx_range, (amx_range.end - amx_range.start) as f64 / 1000.0);
        let amx_path = output_dir.join("gate9_amx.jsonl");
        let resume_from = ResultSink::last_opcode(&amx_path)
            .map(|op| op.saturating_add(1))
            .unwrap_or(amx_range.start);
        println!("    resuming from 0x{resume_from:08X}");
        {
            let mut sink = ResultSink::new(&amx_path).expect("open amx sink");
            let opcodes = resume_from..amx_range.end;
            let (summary, interrupted) = probe.observed_sweep_streaming(opcodes, &mut sink, 50_000);
            println!("    {summary}");
            if interrupted { println!("    (interrupted — resume supported)"); }
            println!("    output: {}", amx_path.display());
        }
    } else {
        println!("\n  [4/4] AMX sweep SKIPPED (encoding unvalidated)");
    }

    // ── Step 6: Post-sweep classification summary ─────────────────────────────
    println!("\n  ── Classification summary ──");

    let mut classify_file = |label: &str, path: &std::path::PathBuf| {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut undefined = 0u64;
        let mut nop_like = 0u64;
        let mut gpr_mutating = 0u64;
        let mut amx_mutating = 0u64;
        let mut gpr_amx = 0u64;
        let mut mem_fault = 0u64;
        let mut trapped_count = 0u64;
        let mut hung = 0u64;
        let mut interesting: Vec<String> = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            let Ok(rec) = serde_json::from_str::<crate::sink::SinkRecord>(line) else { continue };

            // Re-classify from the record fields.
            let classification = match rec.status.as_str() {
                "SIGILL"  => ProbeClassification::Undefined,
                "SEGV"    => ProbeClassification::MemoryFault,
                "TRAP"    => ProbeClassification::Trapped,
                "TIMEOUT" => ProbeClassification::Hung,
                "ok"      => {
                    let gpr_changed = !rec.diff.is_empty();
                    let amx_ch = rec.amx_changed.unwrap_or(false);
                    match (gpr_changed, amx_ch) {
                        (false, false) => ProbeClassification::NopLike,
                        (true,  false) => ProbeClassification::GprMutating,
                        (false, true)  => ProbeClassification::AmxMutating,
                        (true,  true)  => ProbeClassification::GprAndAmxMutating,
                    }
                }
                _ => ProbeClassification::Undefined,
            };

            match classification {
                ProbeClassification::Undefined        => undefined      += 1,
                ProbeClassification::NopLike          => nop_like       += 1,
                ProbeClassification::GprMutating      => { gpr_mutating  += 1; interesting.push(rec.opcode.clone()); }
                ProbeClassification::AmxMutating      => { amx_mutating  += 1; interesting.push(rec.opcode.clone()); }
                ProbeClassification::GprAndAmxMutating=> { gpr_amx       += 1; interesting.push(rec.opcode.clone()); }
                ProbeClassification::MemoryFault      => mem_fault      += 1,
                ProbeClassification::Trapped          => trapped_count  += 1,
                ProbeClassification::Hung             => hung           += 1,
            }
        }
        let total = undefined + nop_like + gpr_mutating + amx_mutating + gpr_amx + mem_fault + trapped_count + hung;
        println!("  {label}: {total} total");
        println!("    SIGILL={undefined}  NOP-like={nop_like}  GPR={gpr_mutating}  AMX={amx_mutating}  GPR+AMX={gpr_amx}  SEGV={mem_fault}  TRAP={trapped_count}  HANG={hung}");
        if !interesting.is_empty() {
            let show: Vec<_> = interesting.iter().take(20).collect();
            println!("    interesting ({} total): {}", interesting.len(), show.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" "));
        }
    };

    classify_file("ALU baseline", &output_dir.join("gate9_alu.jsonl"));
    classify_file("SME sweep",    &output_dir.join("gate9_sme.jsonl"));
    if amx_encoding_valid {
        classify_file("AMX sweep", &output_dir.join("gate9_amx.jsonl"));
    }

    println!("\n✓ gate 9 complete — sweep data in output/gate9_*.jsonl\n");
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
    // gate_5 omitted — it validated the probe harness (sweeps, timeouts, throughput)
    // and hangs for ~1.3s on the deliberate branch-to-self sweep. Everything it
    // proved is now implicitly exercised by gate_6+.
    gate_6();
    gate_7();
    gate_8();
    gate_9();
}
