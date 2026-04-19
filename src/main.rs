#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod cpu_state;
mod crucible;
mod emitter;
mod jit_page;
mod probe;
mod signal_handler;
mod sink;

use signal_handler::install_sigill_handler;

// ═══════════════════════════════════════════════════════════════════════════════
// Gate 14d: Full 16×16 SGEMM — Correctness vs Accelerate
// ═══════════════════════════════════════════════════════════════════════════════

fn gate_14d() {
    use crate::crucible::Crucible;
    use crate::emitter::build_sme_sgemm_16x16;
    use crate::probe::SharedMemory;

    println!("── gate 14d: full 16×16 SME SGEMM vs Accelerate ──");
    println!();

    let m = 16_usize;
    let n = 16_usize;
    let k = 16_usize;

    // ── Input matrices (all-ones → C[i][j] = K) ──
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    // ── Output (SharedMemory for fork visibility) ──
    let c_shared = SharedMemory::<[f32; 256]>::new();
    let c_ptr = c_shared.as_mut_ptr() as *mut f32;
    // SAFETY: c_ptr points to SharedMemory of 256 f32s.
    unsafe { std::ptr::write_bytes(c_ptr, 0, 256); }

    // ── Build the kernel ──
    let block = build_sme_sgemm_16x16(k);
    println!("  Kernel: {} instructions (K={}, M=N=16)", block.len(), k);
    println!("    Setup:   2 (PTRUE + ZERO_ZA)");
    println!("    Compute: {} ({}× [LD1W + LD1W + FMOPA + ADD + ADD])", 5 * k, k);
    println!("    Store:   {} (16× [ST1W + ADD W12 + ADD X2])", 3 * 16);
    println!();

    // ── Register overrides ──
    let overrides: Vec<(u8, u64)> = vec![
        (2,  c_ptr as u64),
        (3,  0_u64),
        (4,  a.as_ptr() as u64),
        (5,  b.as_ptr() as u64),
        (12, 0_u64),
    ];

    // ── Execute ──
    let crucible = Crucible::new();
    println!("  Executing {}-instruction SME kernel...", block.len());
    let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

    println!();
    if result.faulted {
        println!("  [✗] FAULT: {}", result.status());
        return;
    }

    // ── Read results ──
    // SAFETY: c_ptr points to SharedMemory, fork child wrote via ST1W.
    let c_jit: &[f32] = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
    let nonzero = c_jit.iter().filter(|&&v| v != 0.0).count();

    println!("  ✓ Kernel executed without fault!");
    println!("  c_jit[0..8]   = {:?}", &c_jit[0..8]);
    println!("  c_jit[248..256] = {:?}", &c_jit[248..256]);
    println!("  non-zero: {}/{}", nonzero, m * n);
    println!();

    // ── Compare vs Accelerate ──
    let c_accel = Crucible::run_accelerate(m, n, k, &a, &b);
    let max_diff = Crucible::max_abs_diff(&c_accel, c_jit);

    println!("  Accelerate c[0..8] = {:?}", &c_accel[0..8]);
    println!("  max_diff = {:.2e} (target < 1e-4)", max_diff);
    println!();

    if max_diff < 1e-4 {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🏆  GOLDEN — SME SGEMM MATCHES ACCELERATE ON M4  🏆   █");
        println!("  █                                                          █");
        println!("  █   max_diff = {:.2e}                                 █", max_diff);
        println!("  █   Tile: {}×{}, K={}                                    █", m, n, k);
        println!("  █   {} FMOPA ops × 256 MACs = {} total MACs           █", k, k * 256);
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else if nonzero == m * n {
        println!("  [!] All {} elements non-zero but diff too large.", m * n);
        println!("  c_jit[0] = {}, c_accel[0] = {}", c_jit[0], c_accel[0]);
    } else {
        println!("  [!] Only {}/{} elements stored.", nonzero, m * n);
    }

    println!();
    println!("✓ gate 14d complete\n");
}

fn main() {
    install_sigill_handler();
    gate_14d();
}
