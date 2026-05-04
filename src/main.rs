//! Gate runner — tiny dispatcher for active research gates.
//!
//! All shared infrastructure lives in the library crate (`sme_jit_core`).
//! Historical gates have been retired; only the current research front
//! (Gate 26: predicated memory) is wired up here.

use sme_jit_core::emitter::{
    build_gate26_page, build_gate27_page,
    build_gate27p5_row_edge, build_gate27p5_col_edge,
};
use sme_jit_core::signal_handler::install_sigill_handler;

/// Current active research gate.
fn gate_26() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Gate 26: Predicated Memory & Generation — Edge Bounds");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let limit = 20_usize;
    let guard_val = -1.0f32;

    let src: Vec<f32> = (0..limit).map(|i| i as f32).collect();
    let mut dst = vec![guard_val; 32];

    println!("  [1] Building predicated copy kernel (limit={})...", limit);
    let page = build_gate26_page(limit).expect("Failed to build gate 26 page");

    println!("  [2] Executing copy...");
    // SAFETY: page contains valid SVE kernel, src and dst pointers are valid.
    unsafe {
        page.call_with_args(src.as_ptr() as u64, dst.as_mut_ptr() as u64);
    }

    println!("  [3] Verifying results...");
    let mut errors = 0;
    for i in 0..limit {
        if dst[i] != src[i] {
            println!("      [✗] Mismatch at index {}: expected {}, got {}", i, src[i], dst[i]);
            errors += 1;
        }
    }

    let mut guard_violations = 0;
    for i in limit..32 {
        if dst[i] != guard_val {
            println!("      [✗] Guard violation at index {}: expected {}, got {}", i, guard_val, dst[i]);
            guard_violations += 1;
        }
    }

    if errors == 0 && guard_violations == 0 {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🛡️  GATE 26 — PREDICATED MEMORY SUCCESS  🛡️           █");
        println!("  █                                                          █");
        println!("  █   Copied: {}/20 elements correctly                    █", limit);
        println!("  █   Guard:  12/12 elements untouched                       █");
        println!("  █                                                          █");
        println!("  █   SVE WHILELT generated correct masks for 20 elements.   █");
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else {
        println!("  [!] Gate 26 FAILED: {} errors, {} guard violations", errors, guard_violations);
    }

    println!();
    println!("✓ gate 26 complete\n");
}

fn gate_27p5() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Gate 27.5: Separate-Predicate FMOPA Probe");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let k_vals: &[(usize, &str)] = &[(1, "trivial"), (7, "prime"), (16, "SVE width"), (31, "odd")];

    // ── Probe A: P1=row, P0=col, P1 NOT refreshed ──────────────────────────
    println!("  Probe A — FMOPA P1/M(row), P0/M(col), P1 set ONCE:");
    println!("  Question: does FMOPA corrupt P1 when it is ONLY the row predicate?");
    println!();
    let mut probe_a_ok = true;
    for &(k, label) in k_vals {
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let b: Vec<f32> = (0..k * 16).map(|i| (i as f32) * 0.05 + 0.01).collect();
        let expected: Vec<f32> = (0..16).map(|j| {
            (0..k).map(|i| a[i] * b[i * 16 + j]).sum::<f32>()
        }).collect();

        let mut c = vec![0.0f32; 16];
        let page = build_gate27p5_row_edge(
            k, a.as_ptr() as u64, b.as_ptr() as u64, c.as_mut_ptr() as u64,
        ).expect("build_gate27p5_row_edge failed");
        unsafe { page.call_void(); }

        let max_diff = c.iter().zip(&expected).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let ok = max_diff < 1e-3;
        if !ok { probe_a_ok = false; }
        println!(
            "  K={:<4} ({:<10}) │ max_diff={:.2e}  P1_corrupted={}",
            k, label, max_diff, if ok { "NO  ✓" } else { "YES ✗" }
        );
    }

    // ── Probe B: P0=row, P1=col, P1 NOT refreshed + ZA store test ──────────
    println!();
    println!("  Probe B — FMOPA P0/M(row), P1/M(col), P1 set ONCE + predicated ZA store:");
    println!("  Questions: (1) does FMOPA corrupt P1 as col pred?  (2) does P1-ST1W work?");
    println!();
    let sentinel = -999.0f32;
    let mut probe_b_ok = true;
    let mut p1_store_ok_all = true;
    for &(k, label) in k_vals {
        let a: Vec<f32> = (0..k * 16).map(|i| (i as f32) * 0.1 + 0.2).collect();
        let b: Vec<f32> = (0..k).map(|i| (i as f32) * 0.5 + 1.0).collect();
        // c[0] = Σ a[i*16+0]*b[i], c[1..15] = 0.0  (only ZA col 0 touched)
        let expected_c0: f32 = (0..k).map(|i| a[i * 16] * b[i]).sum();

        let mut c = vec![0.0f32; 16];
        let mut d = vec![sentinel; 16];
        let page = build_gate27p5_col_edge(
            k,
            a.as_ptr() as u64,
            b.as_ptr() as u64,
            c.as_mut_ptr() as u64,
            d.as_mut_ptr() as u64,
        ).expect("build_gate27p5_col_edge failed");
        unsafe { page.call_void(); }

        let c_side_zero = c[1..16].iter().all(|&x| x == 0.0);
        // P1-col masking verdict: ZA cols 1-15 must be exactly 0.0 (proves FMOPA respected P1)
        //   AND something was accumulated (c[0] > 0 rules out all-false predicate)
        let p0_ok = c_side_zero && c[0].abs() > 1e-6;
        // Predicated ZA store verdict: d[1..15] must be sentinel AND d[0] must match c[0]
        let d_sides_sentinel = d[1..16].iter().all(|&x| x == sentinel);
        let d0_matches = (d[0] - c[0]).abs() < 1e-3 * c[0].abs().max(1e-6);
        let p1_ok = d_sides_sentinel && d0_matches;

        let exp_note = format!("exp≈{:.4}", expected_c0);

        if !p0_ok { probe_b_ok = false; }
        if !p1_ok { p1_store_ok_all = false; }

        println!(
            "  K={:<4} ({:<10}) │ P0-store c[0]={:>9.4} ({}) side={}  │  P1-store d[0]={:>9.4} side={}",
            k, label,
            c[0], exp_note, if c_side_zero { "✓" } else { "✗" },
            d[0], if d_sides_sentinel { "sentinel ✓" } else { "OVERWRITTEN ✗" },
        );
    }

    println!();
    println!("  ── Summary ──────────────────────────────────────────────────");
    println!(
        "  P1 as ROW predicate (probe A):  {}",
        if probe_a_ok { "NOT corrupted — Gate 28 can use P_row without refresh ✓" }
        else          { "CORRUPTED — per-iter WHILELT refresh needed for row pred ✗" }
    );
    println!(
        "  P1 as COL predicate (probe B):  {}",
        if probe_b_ok { "NOT corrupted — Gate 28 col masking works ✓" }
        else          { "CORRUPTED — per-iter refresh needed for col pred ✗" }
    );
    println!(
        "  Predicated ZA store (P1-ST1W):  {}",
        if p1_store_ok_all { "WORKS — Gate 28 can use predicated ZA stores directly ✓" }
        else               { "BROKEN — Gate 28 needs P0-store + trim workaround ✗" }
    );
    println!();
}

/// Current active research gate.
fn gate_27() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Gate 27: Predicated Outer Products — Odd-K Dot Products");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let cases: &[(usize, &str)] = &[
        (1,   "trivial"),
        (7,   "odd, prime"),
        (13,  "prime"),
        (16,  "full SVE width"),
        (31,  "odd, prime"),
        (100, "larger"),
    ];

    let mut all_ok = true;
    for &(k, label) in cases {
        // Deterministic inputs: a[i] = i*0.1 - 0.1, b[i] = i+1.0
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 0.1).collect();
        let b: Vec<f32> = (0..k).map(|i| i as f32 + 1.0).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let mut c = vec![0.0f32; 16];
        let page = build_gate27_page(
            k,
            a.as_ptr() as u64,
            b.as_ptr() as u64,
            c.as_mut_ptr() as u64,
        ).expect("build_gate27_page failed");

        // SAFETY: all pointers baked in; page ends with RET.
        unsafe { page.call_void(); }

        let diff = (c[0] - expected).abs();
        let cols_masked = c[1..16].iter().all(|&x| x == 0.0);
        let pass = diff <= 1.6e-5 && cols_masked;
        if !pass { all_ok = false; }

        println!(
            "  K={:<4} ({:<14}) │ expected={:>10.4}  c[0]={:>10.4}  diff={:.2e}  cols_masked={}  {}",
            k, label, expected, c[0], diff,
            if cols_masked { "✓" } else { "✗" },
            if pass { "✓" } else { "✗" },
        );
    }

    println!();
    if all_ok {
        println!("  ████████████████████████████████████████████████████████████");
        println!("  █                                                          █");
        println!("  █   🛡️  GATE 27 — PREDICATED OUTER PRODUCTS SUCCESS  🛡️   █");
        println!("  █                                                          █");
        println!("  █   6/6 test cases pass (K=1,7,13,16,31,100)              █");
        println!("  █   ZA[0][1..15] = 0.0 for all K — FMOPA P1/M verified   █");
        println!("  █                                                          █");
        println!("  ████████████████████████████████████████████████████████████");
    } else {
        println!("  [!] Gate 27 FAILED — see above for mismatches");
    }
    println!();
    println!("✓ gate 27 complete\n");
}

fn main() {
    install_sigill_handler();

    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"gate27p5".to_string()) {
        gate_27p5();
    } else if args.contains(&"gate27".to_string()) {
        gate_27();
    } else if args.contains(&"gate26".to_string()) {
        gate_26();
    } else if args.contains(&"all".to_string()) {
        println!("Historical gates are disabled by default. Run specifically (e.g., cargo run -- gate27).");
    } else {
        println!("sme-jit-core gate runner");
        println!("Usage: cargo run --release -- [gate27p5|gate27|gate26]");
        println!();
        println!("Running latest research (Gate 27.5)...");
        gate_27p5();
    }
}
