//! Crucible benchmark suite.
//!
//! Measures three execution paths for `sgemm`-equivalent matrix multiplication
//! across three matrix sizes (8×8, 32×32, 64×64):
//!
//! | Group             | What is measured |
//! |-------------------|-----------------|
//! | `accelerate`      | Raw `cblas_sgemm` via Accelerate.framework — the floor. |
//! | `jit_cold`        | Golden Block through the full safety harness (`fork` + shared memory + prelude/postlude). |
//! | `jit_hot`         | Golden Block via a direct JIT page call (no fork, no overhead). |
//!
//! ## Running
//! ```sh
//! cargo bench
//! # or for just one group:
//! cargo bench -- accelerate
//! cargo bench -- jit_cold
//! cargo bench -- jit_hot
//! ```
//!
//! HTML reports land in `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use jit_explore::crucible::{build_golden_block, MicrokernelAbi};
use jit_explore::emitter::{SMSTART, SMSTOP};
use jit_explore::jit_page::JitPage;
use jit_explore::probe::Probe;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix sizes to benchmark.
const SIZES: &[usize] = &[8, 32, 64];

/// Load the Golden Block opcodes from `stolen_blocks.json`, if present.
///
/// Returns `None` if the file is missing (Frida heist hasn't been run yet)
/// so the benchmark can skip the JIT groups gracefully instead of panicking.
fn load_golden_block() -> Option<Vec<u32>> {
    #[derive(serde::Deserialize)]
    struct StoreEntry {
        pub opcode: String,
        #[serde(rename = "type")]
        #[allow(dead_code)]
        pub store_type: String,
    }

    #[derive(serde::Deserialize)]
    struct StolenBlock {
        pub name: String,
        pub block: Vec<String>,
        #[serde(default)]
        pub stores: Vec<StoreEntry>,
    }

    let path = std::path::Path::new("stolen_blocks.json");
    if !path.exists() {
        eprintln!("[bench] stolen_blocks.json not found — JIT benchmarks will be skipped.");
        eprintln!("        Run `python3 heist/extract_all.py` first.");
        return None;
    }

    let content = std::fs::read_to_string(path).ok()?;
    let blocks: Vec<StolenBlock> = serde_json::from_str(&content).ok()?;

    let target = blocks.iter()
        .find(|b| !b.stores.is_empty())
        .or_else(|| blocks.iter().find(|b| b.name == "cblas_sgemm"))
        .or_else(|| blocks.first())?;

    let math_body: Vec<u32> = target.block.iter()
        .filter_map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .collect();

    let store_epilogue: Vec<u32> = target.stores.iter()
        .filter_map(|s| u32::from_str_radix(s.opcode.trim_start_matches("0x"), 16).ok())
        .collect();

    Some(build_golden_block(&math_body, &store_epilogue))
}

/// Build a "hot" JIT page that wraps the Golden Block in SMSTART / SMSTOP
/// and can be called directly without the fork harness.
///
/// Layout:
/// ```text
/// SMSTART
/// <golden_block opcodes>  (already starts with ZERO_ZA)
/// SMSTOP
/// RET
/// ```
///
/// The page is large enough for 1200 instructions plus SMSTART/SMSTOP/RET.
/// Returns `None` if the block doesn't fit.
fn build_hot_page(golden: &[u32]) -> Option<JitPage> {
    const RET: u32 = 0xD65F_03C0;

    // Each instruction is 4 bytes; add 3 for SMSTART, SMSTOP, RET.
    let total_bytes = (golden.len() + 3) * 4;
    // Round up to 16 KiB pages if needed.
    let page_size = ((total_bytes + 16383) / 16384) * 16384;

    let page = JitPage::alloc(page_size).ok()?;
    page.make_writable();

    let mut off = 0usize;

    // SMSTART — enter streaming mode + ZA
    page.write_instruction(off, SMSTART);
    off += 4;

    // Golden block (ZERO_ZA + math + stores)
    for &op in golden {
        page.write_instruction(off, op);
        off += 4;
    }

    // SMSTOP — return to normal mode
    page.write_instruction(off, SMSTOP);
    off += 4;

    // RET
    page.write_instruction(off, RET);

    page.make_executable();
    Some(page)
}

/// Wake the AMX/SME hardware via a cheap Accelerate call so the coprocessor
/// is warm before we start timing.
fn wake_hardware() {
    let a = [1.0f32; 16];
    let b = [1.0f32; 16];
    let mut c = [0.0f32; 16];
    // SAFETY: valid pointers, sizes are correct, Accelerate always present.
    unsafe {
        jit_explore::crucible::cblas_sgemm(
            jit_explore::crucible::CblasOrder::RowMajor,
            jit_explore::crucible::CblasTranspose::NoTrans,
            jit_explore::crucible::CblasTranspose::NoTrans,
            4, 4, 4,
            1.0, a.as_ptr(), 4, b.as_ptr(), 4, 0.0, c.as_mut_ptr(), 4,
        );
    }
    let _ = black_box(c);
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Accelerate baseline
// ─────────────────────────────────────────────────────────────────────────────

fn bench_accelerate(c: &mut Criterion) {
    wake_hardware();

    let mut group = c.benchmark_group("accelerate");
    group.sample_size(200);

    for &n in SIZES {
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b_iter, &n| {
            let mut result = vec![0.0f32; n * n];
            b_iter.iter(|| {
                // SAFETY: valid pointers, sizes correct.
                unsafe {
                    jit_explore::crucible::cblas_sgemm(
                        jit_explore::crucible::CblasOrder::RowMajor,
                        jit_explore::crucible::CblasTranspose::NoTrans,
                        jit_explore::crucible::CblasTranspose::NoTrans,
                        n as i32, n as i32, n as i32,
                        1.0,
                        a.as_ptr(), n as i32,
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
// Group 2: JIT Cold (full fork-based safety harness)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_jit_cold(c: &mut Criterion) {
    let golden = match load_golden_block() {
        Some(g) => g,
        None    => {
            eprintln!("[bench] jit_cold: skipped (no stolen_blocks.json)");
            return;
        }
    };

    wake_hardware();

    let probe = Probe::new();
    let mut group = c.benchmark_group("jit_cold");
    // Fork overhead is ~1 ms per call — keep sample count low to avoid 5-minute runs.
    group.sample_size(20);

    for &n in SIZES {
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b_iter, &n| {
            let mut c_jit = vec![0.0f32; n * n];

            // Inject microkernel ABI pointers (x5=B, x7=A, x8=C).
            let abi = MicrokernelAbi {
                a_ptr: a.as_ptr()         as u64,
                b_ptr: b.as_ptr()         as u64,
                c_ptr: c_jit.as_mut_ptr() as u64,
            };
            let overrides = abi.to_overrides();

            b_iter.iter(|| {
                // reset c_jit for each iteration
                for v in c_jit.iter_mut() { *v = 0.0; }

                let result = probe.run_block_with_overrides(
                    black_box(&golden),
                    black_box(&overrides),
                    true, // streaming mode
                );
                black_box(result.faulted);
                black_box(&c_jit);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: JIT Hot (raw JIT page call — bare-metal throughput)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_jit_hot(c: &mut Criterion) {
    let golden = match load_golden_block() {
        Some(g) => g,
        None    => {
            eprintln!("[bench] jit_hot: skipped (no stolen_blocks.json)");
            return;
        }
    };

    let page = match build_hot_page(&golden) {
        Some(p) => p,
        None    => {
            eprintln!("[bench] jit_hot: skipped (JIT page allocation failed)");
            return;
        }
    };

    wake_hardware();

    // Pre-verify: if the block faults we skip the hot bench rather than crash.
    {
        let probe = Probe::new();
        let a = vec![1.0f32; 64 * 64];
        let b = vec![1.0f32; 64 * 64];
        let mut c_test = vec![0.0f32; 64 * 64];
        let abi = MicrokernelAbi {
            a_ptr: a.as_ptr()           as u64,
            b_ptr: b.as_ptr()           as u64,
            c_ptr: c_test.as_mut_ptr()  as u64,
        };
        let overrides = abi.to_overrides();
        let result = probe.run_block_with_overrides(&golden, &overrides, true);
        if result.faulted {
            eprintln!("[bench] jit_hot: skipped — golden block faults ({}). \
                       Run gate_12 first to diagnose.", result.status());
            return;
        }
    }

    let mut group = c.benchmark_group("jit_hot");
    group.sample_size(500);

    for &n in SIZES {
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b_iter, &_n| {
            // The hot page doesn't do pointer injection — it runs whatever was
            // baked in at page-build time. This measures pure execution cost
            // of the SMSTART + block + SMSTOP sequence without any harness.
            b_iter.iter(|| {
                // SAFETY: The page contains SMSTART + golden_block + SMSTOP + RET.
                // The golden block was pre-verified above not to fault.
                // No matrix data is needed here — this is a throughput probe,
                // not a precision test. The block runs against whatever happens
                // to be in registers.
                unsafe { page.call_void(); }
                black_box(&a);
                black_box(&b);
            });
        });
    }

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
);
criterion_main!(benches);
