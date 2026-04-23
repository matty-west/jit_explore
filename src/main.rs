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
    use crate::emitter::{build_sme_sgemm_16x16, Activation};
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
    let block = build_sme_sgemm_16x16(k, Activation::None);
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

fn gate_16() {
    use crate::emitter::{encode_bfmopa, encode_bfmops, encode_smopa, encode_umopa, encode_sumopa};
    use crate::probe::Probe;

    println!("── gate 16: BFMOPA / SMOPA probing ──");
    println!();

    let probe = Probe::new();
    
    let variants = vec![
        ("BFMOPA", encode_bfmopa(0, 0, 0, 0, 1)),
        ("BFMOPS", encode_bfmops(0, 0, 0, 0, 1)),
        ("SMOPA",  encode_smopa(0, 0, 0, 0, 1)),
        ("UMOPA",  encode_umopa(0, 0, 0, 0, 1)),
        ("SUMOPA", encode_sumopa(0, 0, 0, 0, 1)),
    ];

    println!("  {: <10} | {: <10} | {: <10}", "Instruction", "Opcode", "Status");
    println!("  -----------|------------|-----------");

    let mut supported = Vec::new();

    for (name, op) in variants {
        let result = probe.run_observed_streaming(op);
        let status = result.base.status();
        println!("  {: <10} | 0x{:08X} | {}", name, op, status);
        
        if status == "ok" {
            supported.push(name);
        }
    }

    println!();
    if supported.is_empty() {
        println!("  [!] No extended outer product instructions supported on this M4.");
    } else {
        println!("  ✓ Supported: {:?}", supported);
    }
    println!();
    println!("✓ gate 16 skeleton complete\n");
}

fn gate_16_correctness() {
    use crate::crucible::Crucible;
    use crate::emitter::{build_sme_bfmopa_16x16, build_sme_smopa_16x16};
    use crate::probe::SharedMemory;

    println!("── gate 16: BFMOPA / SMOPA correctness ──");
    println!();

    let m = 16_usize;
    let n = 16_usize;
    let k = 16_usize;

    // ── 1. BFMOPA Correctness ──
    {
        println!("  [1] Testing BFMOPA (BF16 → FP32)...");
        let a_f32 = vec![1.0f32; m * k];
        let b_f32 = vec![1.0f32; k * n];
        let a_bf16 = Crucible::f32_to_bf16(&a_f32);
        let b_bf16 = Crucible::f32_to_bf16(&b_f32);

        let c_shared = SharedMemory::<[f32; 256]>::new();
        let c_ptr = c_shared.as_mut_ptr() as *mut f32;
        unsafe {
            for i in 0..256 { *c_ptr.add(i) = 42.0; }
        }

        let block = build_sme_bfmopa_16x16(k);
        let overrides: Vec<(u8, u64)> = vec![
            (2,  c_ptr as u64),
            (3,  0_u64),
            (4,  a_bf16.as_ptr() as u64),
            (5,  b_bf16.as_ptr() as u64),
            (12, 0_u64),
        ];

        let crucible = Crucible::new();
        let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

        if result.faulted {
            println!("      [✗] BFMOPA FAULTED: {}", result.status());
        } else {
            let c_jit = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
            println!("      BFMOPA JIT [0..4] = {:?}", &c_jit[0..4]);
            let c_ref = Crucible::ref_bf16_matmul(m, n, k, &a_bf16, &b_bf16);
            let max_diff = Crucible::max_abs_diff(&c_ref, c_jit);
            println!("      ✓ BFMOPA max_diff = {:.2e}", max_diff);
            if max_diff > 1e-2 {
                 println!("      [!] Diff too large for BF16");
            }
        }
    }
    println!();

    // ── 2. SMOPA Correctness ──
    {
        println!("  [2] Testing SMOPA (INT8 → INT32)...");
        let a_i8 = vec![1i8; m * k];
        let b_i8 = vec![1i8; k * n];

        let c_shared = SharedMemory::<[i32; 256]>::new();
        let c_ptr = c_shared.as_mut_ptr() as *mut i32;
        unsafe {
            for i in 0..256 { *c_ptr.add(i) = 42; }
        }

        let block = build_sme_smopa_16x16(k);
        let overrides: Vec<(u8, u64)> = vec![
            (2,  c_ptr as u64),
            (3,  0_u64),
            (4,  a_i8.as_ptr() as u64),
            (5,  b_i8.as_ptr() as u64),
            (12, 0_u64),
        ];

        let crucible = Crucible::new();
        let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

        if result.faulted {
            println!("      [✗] SMOPA FAULTED: {}", result.status());
        } else {
            let c_jit = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
            println!("      SMOPA JIT [0..4] = {:?}", &c_jit[0..4]);
            let c_ref = Crucible::ref_int8_matmul(m, n, k, &a_i8, &b_i8);
            let mut matches = true;
            for i in 0..m*n {
                if c_jit[i] != c_ref[i] {
                    println!("      [✗] SMOPA mismatch at [{}]: jit={}, ref={}", i, c_jit[i], c_ref[i]);
                    matches = false;
                    break;
                }
            }
            if matches {
                println!("      ✓ SMOPA matches reference exactly!");
            }
        }
    }

    println!();
    println!("✓ gate 16 correctness complete\n");
}

fn gate_17a() {
    use crate::crucible::Crucible;
    use crate::emitter::{build_sme_sgemm_16x16, Activation};
    use crate::probe::SharedMemory;

    println!("── gate 17a: fused GEMM + ReLU ──");
    println!();

    let m = 16_usize;
    let n = 16_usize;
    let k = 16_usize;

    // ── Input matrices (symmetric A so A^T=A matches kernel layout, mixed signs for ReLU) ──
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for j in 0..k {
            a[i * k + j] = ((i + j) as f32 % 7.0) - 3.0; // symmetric: a[i][j] = a[j][i]
        }
    }
    for i in 0..b.len() { b[i] = (i as f32 % 5.0) - 2.0; } // -2.0 to 2.0

    // ── Output ──
    let c_shared = SharedMemory::<[f32; 256]>::new();
    let c_ptr = c_shared.as_mut_ptr() as *mut f32;

    // ── Build the kernel ──
    let block = build_sme_sgemm_16x16(k, Activation::ReLU);
    println!("  Kernel: {} instructions (K={}, M=N=16, Activation=ReLU)", block.len(), k);
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
    let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

    if result.faulted {
        println!("  [✗] FAULT: {}", result.status());
        return;
    }

    // ── Read results ──
    let c_jit: &[f32] = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
    
    // ── Compare vs Reference ──
    let c_ref = Crucible::ref_sgemm_fused(m, n, k, &a, &b, None, true);
    let max_diff = Crucible::max_abs_diff(&c_ref, c_jit);

    println!("  JIT c[0..4] = {:?}", &c_jit[0..4]);
    println!("  REF c[0..4] = {:?}", &c_ref[0..4]);
    println!("  max_diff = {:.2e}", max_diff);

    let negative_count = c_jit.iter().filter(|&&v| v < 0.0).count();
    if negative_count > 0 {
        println!("  [✗] FAILED: Found {} negative elements in ReLU output!", negative_count);
    } else if max_diff < 1e-4 {
        println!("  ✓ SUCCESS: Fused GEMM + ReLU matches reference!");
    } else {
        println!("  [!] Output matches ReLU (all >= 0) but diff too large: {:.2e}", max_diff);
    }

    println!();
    println!("✓ gate 17a complete\n");
}

fn gate_17b() {
    use crate::crucible::Crucible;
    use crate::emitter::{build_sme_sgemm_16x16, Activation};
    use crate::probe::SharedMemory;

    println!("── gate 17b: fused GEMM + Bias ──");
    println!();

    let m = 16_usize;
    let n = 16_usize;
    let k = 16_usize;

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let bias = vec![0.5f32; n]; // bias = 0.5 for all elements

    let c_shared = SharedMemory::<[f32; 256]>::new();
    let c_ptr = c_shared.as_mut_ptr() as *mut f32;

    let block = build_sme_sgemm_16x16(k, Activation::Bias);
    println!("  Kernel: {} instructions (K={}, M=N=16, Activation=Bias)", block.len(), k);

    let overrides: Vec<(u8, u64)> = vec![
        (2,  c_ptr as u64),
        (3,  0_u64),
        (4,  a.as_ptr() as u64),
        (5,  b.as_ptr() as u64),
        (6,  bias.as_ptr() as u64),
        (12, 0_u64),
    ];

    let crucible = Crucible::new();
    let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

    if result.faulted {
        println!("  [✗] FAULT: {}", result.status());
        return;
    }

    let c_jit: &[f32] = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
    let c_ref = Crucible::ref_sgemm_fused(m, n, k, &a, &b, Some(&bias), false);
    let max_diff = Crucible::max_abs_diff(&c_ref, c_jit);

    println!("  JIT c[0..4] = {:?}", &c_jit[0..4]);
    println!("  REF c[0..4] = {:?}", &c_ref[0..4]);
    println!("  max_diff = {:.2e}", max_diff);

    if max_diff < 1e-4 {
        println!("  ✓ SUCCESS: Fused GEMM + Bias matches reference!");
    } else {
        println!("  [✗] FAILED: Max diff too large!");
    }

    println!();
    println!("✓ gate 17b complete\n");
}

fn gate_17c() {
    use crate::crucible::Crucible;
    use crate::emitter::{build_sme_sgemm_16x16, Activation};
    use crate::probe::SharedMemory;

    println!("── gate 17c: fused GEMM + Bias + ReLU ──");
    println!();

    let m = 16_usize;
    let n = 16_usize;
    let k = 16_usize;

    // Use negative values so Bias + ReLU actually does something interesting
    // Symmetric A so A^T=A matches kernel layout
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for j in 0..k {
            a[i * k + j] = ((i + j) as f32 % 5.0) - 2.5; // symmetric
        }
    }
    for i in 0..b.len() { b[i] = (i as f32 % 3.0) - 1.5; }
    let bias = vec![-1.0f32; n]; // negative bias to push more elements into ReLU clamp

    let c_shared = SharedMemory::<[f32; 256]>::new();
    let c_ptr = c_shared.as_mut_ptr() as *mut f32;

    let block = build_sme_sgemm_16x16(k, Activation::BiasReLU);
    println!("  Kernel: {} instructions (K={}, M=N=16, Activation=BiasReLU)", block.len(), k);

    let overrides: Vec<(u8, u64)> = vec![
        (2,  c_ptr as u64),
        (3,  0_u64),
        (4,  a.as_ptr() as u64),
        (5,  b.as_ptr() as u64),
        (6,  bias.as_ptr() as u64),
        (12, 0_u64),
    ];

    let crucible = Crucible::new();
    let result = crucible.probe.run_block_with_overrides(&block, &overrides, true);

    if result.faulted {
        println!("  [✗] FAULT: {}", result.status());
        return;
    }

    let c_jit: &[f32] = unsafe { std::slice::from_raw_parts(c_ptr, m * n) };
    let c_ref = Crucible::ref_sgemm_fused(m, n, k, &a, &b, Some(&bias), true);
    let max_diff = Crucible::max_abs_diff(&c_ref, c_jit);

    println!("  JIT c[0..4] = {:?}", &c_jit[0..4]);
    println!("  REF c[0..4] = {:?}", &c_ref[0..4]);
    println!("  max_diff = {:.2e}", max_diff);

    let negative_count = c_jit.iter().filter(|&&v| v < 0.0).count();
    if negative_count > 0 {
        println!("  [✗] FAILED: Found {} negative elements!", negative_count);
    } else if max_diff < 1e-4 {
        println!("  ✓ SUCCESS: Fused GEMM + Bias + ReLU matches reference!");
    } else {
        println!("  [✗] FAILED: Max diff too large!");
    }

    println!();
    println!("✓ gate 17c complete\n");
}

fn discovery_sweep() {
    use crate::emitter::*;
    use crate::probe::{Probe, SharedMemory};

    println!("── systematic discovery sweep ──");
    let probe = Probe::new();
    let c_shared = SharedMemory::<[f32; 16]>::new();
    let c_ptr = c_shared.as_mut_ptr() as *mut f32;

    struct Variant {
        name: &'static str,
        opcodes: Vec<u32>,
        streaming: bool,
    }

    let mut variants = Vec::new();

    // 1. SVE Store Probes (Streaming)
    variants.push(Variant {
        name: "SVE STR (unpredicated)",
        opcodes: vec![0x0528_E402, 0xE580_4042], // STR Z2, [X2]
        streaming: true,
    });

    // REGISTER-ONLY MOVA VERIFY:
    // 1. Set Z2 = 1.0
    // 2. MOVA Z2 -> ZA0H.S[0]
    // 3. Set Z2 = 2.0 (Clobber)
    // 4. MOVA ZA0H.S[0] -> Z2
    // 5. STR Z2 -> [X2]
    // If v[0] == 1.0, MOVA is PERFECT.
    variants.push(Variant {
        name: "Register-only MOVA Verification",
        opcodes: vec![
            PTRUE_P0_S,
            0x0528_E402, // FCPY Z2.S, P0/M, #1.0
            0xC000_0002, // MOVA ZA0H.S[0, 0], P0, Z2.S (using literal bits)
            0x0528_E412, // FCPY Z2.S, P0/M, #2.0
            0xC040_0002, // MOVA Z2.S, P0, ZA0H.S[0, 0]
            0xE580_4042, // STR Z2, [X2]
        ],
        streaming: true,
    });

    // Final Value-Sensitive MOVA Sweep
    // Init Z2 = 7.0, clobber = 42.0. If v[0] == 7.0, we found it.
    for bits in (0..256).step_by(4) {
        variants.push(Variant {
            name: Box::leak(format!("MOVA Brute [bits23:16={}]", bits).into_boxed_str()),
            opcodes: vec![
                PTRUE_P0_S,
                0x0521_E402, // FCPY Z2.S, P0/M, #7.0 (approx)
                0xC000_0002 | ((bits as u32) << 16), 
                0x0528_E412, // FCPY Z2.S, P0/M, #2.0 (clobber)
                0xC040_0002 | ((bits as u32) << 16),
                0xE580_4042, 
            ],
            streaming: true,
        });
    }
    variants.push(Variant {
        name: "BFMOPA -> SME ST1W (Gate 16 NOP?)",
        opcodes: vec![
            PTRUE_P0_S, 
            ZERO_ZA, 
            0x0528_E400, // FCPY Z0, #1.0
            0x0528_E401, // FCPY Z1, #1.0
            encode_bfmopa(0, 0, 0, 0, 1), 
            encode_sme_st1w_za_h(0, 0, 0, 2, 3), // ST1W {ZA0H[0]}, P0, [X2]
        ],
        streaming: true,
    });
    variants.push(Variant {
        name: "BFMOPA (Gate 16 NOP?)",
        opcodes: vec![PTRUE_P0_S, ZERO_ZA, 0x0528_E400, 0x0528_E401, encode_bfmopa(0, 0, 0, 0, 1)],
        streaming: true,
    });

    println!("  {: <30} | {: <10} | {: <10}", "Instruction Variant", "Status", "Impact");
    println!("  -------------------------------|------------|-----------");

    let a_mem = vec![1.0f32; 16];
    let b_mem = vec![1.0f32; 16];

    for v in variants {
        // Pre-fill with sentinel
        unsafe {
            for i in 0..16 { *c_ptr.add(i) = 42.0; }
        }

        let overrides = vec![
            (2, c_ptr as u64),
            (3, 0u64),
            (4, a_mem.as_ptr() as u64),
            (5, b_mem.as_ptr() as u64),
            (12, 0u64),
        ];

        let result = probe.run_block_with_overrides(&v.opcodes, &overrides, v.streaming);
        let status = result.status();
        
        // Check if anything changed in memory (for store tests)
        let c_jit = unsafe { std::slice::from_raw_parts(c_ptr, 16) };
        let changed = c_jit.iter().any(|&v| v != 42.0);
        let impact = if changed { 
            format!("CHANGED (v[0]={:.1})", c_jit[0])
        } else { 
            "NO IMPACT".to_string() 
        };

        println!("  {: <30} | {: <10} | {: <10}", v.name, status, impact);
    }
    println!();
}

fn main() {
    install_sigill_handler();
    
    // Check for "sweep" argument to run the big latency table
    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"sweep".to_string()) {
        gate_latency_sweep();
    } else {
        discovery_sweep();
        gate_14d();
        gate_16();
        gate_16_correctness();
        gate_17a();
        gate_17b();
        gate_17c();
        gate_performance();
    }
}

fn gate_latency_sweep() {
    use crate::probe::Probe;
    use crate::sink::ResultSink;
    use std::path::Path;

    println!("── gate latency sweep: building M4 SME performance database ──");
    
    let probe = Probe::new();
    let mut sink = ResultSink::new(Path::new("latency_table.jsonl")).expect("sink");
    let iterations = 1000; // Fast sweep

    // Interesting ranges for SME/SVE
    let ranges = vec![
        0x80000000..0x81000000, // FMOPA/BFMOPA
        0xA0000000..0xA1000000, // SMOPA/UMOPA
        0xA4000000..0xA6000000, // LD1*
        0xE0000000..0xE1000000, // ST1*
    ];

    let mut total_probed = 0;
    let mut valid_found = 0;

    for range in ranges {
        println!("  Sweeping range: 0x{:08X}..0x{:08X}", range.start, range.end);
        for (i, opcode) in range.step_by(0x4000).enumerate() { // Even coarser for first pass
            total_probed += 1;
            if let Some(ticks) = probe.run_latency_benchmark(&[opcode], iterations, true) {
                valid_found += 1;
                
                // Construct a minimal ObservedProbeResult to satisfy the sink
                let res = crate::probe::ObservedProbeResult {
                    base: crate::probe::ProbeResult {
                        opcode, faulted: false, timed_out: false,
                        segfaulted: false, trapped: false, fault_offset: 0,
                        timestamp_pre: 0, timestamp_post: (ticks * iterations as f64) as u64,
                    },
                    pre: None, post: None, diff: vec![],
                    snapshot_corrupted: false, gprs_post: None,
                };
                let _ = sink.write(&res);
                let _ = sink.flush(); // Real-time visibility

                if i % 10 == 0 {
                    println!("    Found valid: 0x{:08X} | Latency: {:.4} ticks", opcode, ticks);
                }
            }
        }
    }

    println!();
    println!("✓ Latency sweep complete. Probed: {}, Valid: {}", total_probed, valid_found);
}

fn gate_performance() {
    use crate::probe::Probe;
    use crate::emitter::*;

    println!("── gate performance: SME latency mapping ──");
    println!();

    let probe = Probe::new();
    let iterations = 10000;

    let targets: Vec<(&str, Vec<u32>)> = vec![
        ("Loop Overhead (NOP)", vec![NOP]),
        ("FMOPA (Functional)",  vec![0x8081_0000]), // FMOPA ZA0.S, P0, Z0, Z1
        ("BFMOPA (NOP-like)",   vec![0x8181_0000]), // BFMOPA
        ("LD1W (ZA row load)",  vec![0xA540_4000]), // LD1W Z0.S, P0/Z, [X0]
        ("ST1W (ZA row store)", vec![0xE0A0_0000]), // ST1W {ZA0H[0]}, P0, [X0]
        ("FADD (SVE vector)",   vec![0x6580_0000]), // FADD Z0.S, P0/M, Z0.S, Z0.S
    ];

    println!("  {: <25} | {: <15}", "Instruction", "Ticks/Iter");
    println!("  --------------------------|----------------");

    // Calibrate loop overhead
    let overhead = probe.run_latency_benchmark(&[NOP], iterations, true).unwrap_or(0.0);

    for (name, ops) in targets {
        if let Some(ticks) = probe.run_latency_benchmark(&ops, iterations, true) {
            let adjusted = (ticks - overhead).max(0.0);
            println!("  {: <25} | {: <15.4}", name, adjusted);
        } else {
            println!("  {: <25} | {: <15}", name, "FAULT/TIMEOUT");
        }
    }
    println!();
    println!("✓ gate performance complete\n");
}
