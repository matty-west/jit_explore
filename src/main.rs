#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::arch::asm;

// ╔══════════════════════════════════════╗
// ║  AArch64 Instruction Constants       ║
// ╚══════════════════════════════════════╝

/// AArch64 NOP — no operation.
#[allow(dead_code)]
const NOP: u32 = 0xD503_201F;

/// AArch64 RET — return to caller via x30 (LR).
#[allow(dead_code)]
const RET: u32 = 0xD65F_03C0;

/// AArch64 UDF #0 — permanently undefined, guaranteed SIGILL.
#[allow(dead_code)]
const UDF_0: u32 = 0x0000_0000;

// ╔══════════════════════════════════════╗
// ║  Gate 0 — Minimal Viable Toolchain   ║
// ╚══════════════════════════════════════╝

fn main() {
    // SAFETY: `nop` has no side effects and no operands. It exists purely
    // to prove the toolchain can compile and execute AArch64 inline assembly.
    unsafe {
        asm!("nop");
    }
    println!("alive");
}
