//! Crucible benchmark suite — SME SGEMM vs Accelerate.
//!
//! Measures three execution paths for 16×16 matrix multiplication:
//!
//! | Group        | What is measured |
//! |--------------|-----------------|
//! | `accelerate` | Raw `cblas_sgemm` via Accelerate.framework — the floor. |
//! | `jit_cold`   | SME SGEMM kernel through the fork-based safety harness. |
//! | `jit_hot`    | SME SGEMM kernel via direct JIT page call (bare-metal). |
//!
//! ## Running
//! ```sh
//! cargo bench
//! cargo bench -- accelerate    # just Accelerate baseline
//! cargo bench -- jit_hot       # just bare-metal JIT
//! ```
//!
//! HTML reports land in `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use sme_jit_core::emitter::{build_sme_sgemm_16x16, build_sme_sgemm_page, Activation};
use sme_jit_core::probe::{Probe, SharedMemory};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// K dimensions to benchmark (all use M=N=16 tile).
const K_SIZES: &[usize] = &[1, 4, 16, 64];

/// Wake the SME hardware via a cheap Accelerate call.
fn wake_hardware() {
    let a = [1.0f32; 16];
    let b = [1.0f32; 16];
    let mut c = [0.0f32; 16];
    // SAFETY: valid pointers, sizes correct, Accelerate always present.
    unsafe {
        sme_jit_core::crucible::cblas_sgemm(
            sme_jit_core::crucible::CblasOrder::RowMajor,
            sme_jit_core::crucible::CblasTranspose::NoTrans,
            sme_jit_core::crucible::CblasTranspose::NoTrans,
            4, 4, 4,
            1.0, a.as_ptr(), 4, b.as_ptr(), 4, 0.0, c.as_mut_ptr(), 4,
        );
    }
    let _ = black_box(c);
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Accelerate baseline (cblas_sgemm, 16×16×K)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_accelerate(c: &mut Criterion) {
    wake_hardware();

    let mut group = c.benchmark_group("accelerate");
    group.sample_size(200);

    for &k in K_SIZES {
        let m = 16;
        let n = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];

        group.bench_with_input(BenchmarkId::new("16x16", k), &k, |b_iter, &k| {
            let mut result = vec![0.0f32; m * n];
            b_iter.iter(|| {
                // SAFETY: valid pointers, correct sizes.
                unsafe {
                    sme_jit_core::crucible::cblas_sgemm(
                        sme_jit_core::crucible::CblasOrder::RowMajor,
                        sme_jit_core::crucible::CblasTranspose::NoTrans,
                        sme_jit_core::crucible::CblasTranspose::NoTrans,
                        m as i32, n as i32, k as i32,
                        1.0,
                        a.as_ptr(), k as i32,
                        b.as_ptr(), n as i32,
                        0.0,
                        result.as_mut_ptr(), n as i32,
                    );
                }
                black_box(&result);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: JIT Cold (fork-based safety harness)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_jit_cold(c: &mut Criterion) {
    wake_hardware();

    let probe = Probe::new();
    let mut group = c.benchmark_group("jit_cold");
    // Fork overhead is ~1 ms per call — keep sample count low.
    group.sample_size(20);

    for &k in K_SIZES {
        let m = 16;
        let n = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let block = build_sme_sgemm_16x16(k, Activation::None);

        group.bench_with_input(BenchmarkId::new("16x16", k), &k, |b_iter, &_k| {
            let c_shared = SharedMemory::<[f32; 256]>::new();
            let c_ptr = c_shared.as_mut_ptr() as *mut f32;

            let overrides: Vec<(u8, u64)> = vec![
                (2,  c_ptr as u64),
                (3,  0u64),
                (4,  a.as_ptr() as u64),
                (5,  b.as_ptr() as u64),
                (12, 0u64),
            ];

            b_iter.iter(|| {
                // SAFETY: c_ptr points to valid SharedMemory.
                unsafe { std::ptr::write_bytes(c_ptr, 0, 256); }
                let result = probe.run_block_with_overrides(
                    black_box(&block),
                    black_box(&overrides),
                    true,
                );
                black_box(result.faulted);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: JIT Hot (direct JIT page call — bare-metal throughput)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_jit_hot(c: &mut Criterion) {
    wake_hardware();

    let mut group = c.benchmark_group("jit_hot");
    group.sample_size(500);

    for &k in K_SIZES {
        let m = 16;
        let n = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c_out = vec![0.0f32; m * n];

        let page = match build_sme_sgemm_page(
            k,
            Activation::None,
            a.as_ptr() as u64,
            b.as_ptr() as u64,
            c_out.as_mut_ptr() as u64,
            0,
        ) {
            Some(p) => p,
            None => {
                eprintln!("[bench] jit_hot K={k}: skipped (page alloc failed)");
                continue;
            }
        };

        // Pre-verify: execute once and check output.
        // SAFETY: page contains SMSTART + kernel + SMSTOP + RET.
        // Pointers baked in are valid for the lifetime of this scope.
        unsafe { page.call_void(); }
        let expected = k as f32; // all-ones input → C[i][j] = K
        if (c_out[0] - expected).abs() > 1e-4 {
            eprintln!(
                "[bench] jit_hot K={k}: skipped — output c[0]={} (expected {})",
                c_out[0], expected
            );
            continue;
        }

        group.bench_with_input(BenchmarkId::new("16x16", k), &k, |b_iter, &_k| {
            b_iter.iter(|| {
                // SAFETY: verified above that the page executes correctly.
                unsafe { page.call_void(); }
                black_box(&c_out);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: Fused JIT (GEMM + ReLU, GEMM + Bias + ReLU)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_fused(c: &mut Criterion) {
    wake_hardware();

    let mut group = c.benchmark_group("fused");
    group.sample_size(500);

    let k = 16;
    let m = 16;
    let n = 16;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let bias = vec![0.5f32; n];
    let mut c_out = vec![0.0f32; m * n];

    // Benchmark GEMM + ReLU
    let page_relu = build_sme_sgemm_page(
        k, Activation::ReLU,
        a.as_ptr() as u64, b.as_ptr() as u64, c_out.as_mut_ptr() as u64, 0
    ).unwrap();

    group.bench_function("sgemm_relu_16x16", |b_iter| {
        b_iter.iter(|| {
            unsafe { page_relu.call_void(); }
            black_box(&c_out);
        });
    });

    // Benchmark GEMM + Bias + ReLU
    let page_bias_relu = build_sme_sgemm_page(
        k, Activation::BiasReLU,
        a.as_ptr() as u64, b.as_ptr() as u64, c_out.as_mut_ptr() as u64, bias.as_ptr() as u64
    ).unwrap();

    group.bench_function("sgemm_bias_relu_16x16", |b_iter| {
        b_iter.iter(|| {
            unsafe { page_bias_relu.call_void(); }
            black_box(&c_out);
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion entry points
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_accelerate,
    bench_jit_cold,
    bench_jit_hot,
    bench_fused,
);
criterion_main!(benches);
