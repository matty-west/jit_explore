<p align="center">
  <img src="sme-jit-core-banner.png" alt="Dawn State — sme-jit-core" width="100%" />
</p>

<p align="center">
  <strong>A <a href="https://github.com/dawn-state">Dawn State</a> research project</strong> · Silicon Exploitation & Bare-Metal Optimization
</p>

---

# sme-jit-core

**A zero-overhead Rust JIT harness for reverse-engineering and executing matrix coprocessor instructions on Apple Silicon M4 — beats Accelerate.framework by 5× at tile-sized problems, runs a full MNIST classifier 1.93× faster than Accelerate.**

Empirical opcode probing, fault-tolerant sandboxing, and raw SME instruction emission straight to the silicon.

Two execution modes: a **fork-isolated probe harness** for safely exploring undocumented instructions (fault recovery via SIGILL/SEGV), and a **direct-execution JIT path** that emits machine code into `MAP_JIT` pages and calls it as a native function pointer — zero dispatch, zero abstraction, just `BLR` into your FMOPA sequence. The benchmarks below measure the direct path.

## Benchmark Results

### Tiled GEMM — JIT vs Accelerate (with warmup, 2000 samples, median)

```
          Size |      JIT (min/med/max) |    Accel (min/med/max) | Speedup
  -------------|------------------------|------------------------|--------
   16× 16× 16 |     0/    41/   167 ns |   125/   208/   375 ns |   5.1×
   32× 32× 32 |     0/    83/   208 ns |   167/   250/   417 ns |   3.0×
   48× 48× 48 |   250/   333/   500 ns |   416/   459/ 24583 ns |   1.4×
   64× 64× 64 |   750/   834/  1166 ns |   583/   667/   917 ns |   0.8×
  128×128×128  |  7625/  7750/  ... ns  |  2916/  3000/  ... ns  |   0.4×
```

**JIT dominates at ≤48×48** (zero dispatch overhead — Accelerate pays ~200ns fixed tax per call). Accelerate wins at ≥64×64 (optimized cache blocking, multi-core). The crossover at ~48×48 defines our niche: **tiny-model, latency-critical inference**.

### Full MNIST Inference — 3-Layer MLP (784→16→16→10)

```
  Path                          Latency       vs Accelerate
  Accelerate (cblas_sgemm)      3.8 μs        1.00×
  Cached JIT (with transpose)   9.1 μs        0.42×
  Cached JIT (pretransposed)    2.0 μs        1.93×
```

16/16 images classified correctly. Bit-exact match vs Accelerate reference (`max_diff = 0.00e0`). Zero frameworks, zero dispatch, pure silicon.

**Differential correctness:** `max_diff = 0.0` vs `cblas_sgemm` across all tested configurations. Every result is bit-identical to Apple's reference implementation.

## M4 SME Parameters (Empirically Determined)

| Parameter | Value |
|:---|:---|
| SVL (Scalable Vector Length) | **512 bits** |
| float32 per Z register | 16 |
| ZA tile dimensions | 16×16 = 256 float32 |
| MACs per `FMOPA` | **256** |
| BFMOPA/SMOPA/UMOPA | Decode OK, execute as NOP (disabled in firmware?) |

These parameters were discovered through empirical instruction probing — not from Apple documentation, which does not publicly disclose M4 SME configuration details.

## What This Does

1. **Probes** — Executes arbitrary AArch64 opcodes in a fault-tolerant JIT sandbox (fork + SIGILL/SEGV/SIGBUS recovery)
2. **Discovers** — Proved M4 uses ARM SME through empirical instruction probing when Apple's own documentation remained silent
3. **Computes** — JIT-emits SME instruction sequences (LD1W → FMOPA → ST1W) producing correct SGEMM results, with fused activation (ReLU, Bias)
4. **Tiles** — Emits tiled GEMM kernels for arbitrary M×N (multiples of 16) up to 128×128 with branched K-loops
5. **Infers** — Runs a full 3-layer MNIST MLP entirely through JIT'd fused SME kernels, 1.93× faster than Accelerate
6. **Verifies** — Differential correctness via Crucible: `max_diff = 0.0` vs `cblas_sgemm`
7. **Benchmarks** — Rigorous microbenchmarks with warmup, percentile timing, and per-iteration measurement

## Milestones

| Gate | Goal | Status |
|:-----|:-----|:-------|
| 0–9 | JIT page allocation, fault recovery, fork probing, GPR snapshots | ✅ |
| 10–13 | Frida heist, AMX leaf execution (exploration phase — archived) | ✅ |
| 14a–b | Synthetic AMX tests + operand bit-walk (all zeros — dead end) | ✅ |
| 14c | **SME pivot: FMOPA produces first non-zero result** | ✅ |
| 14d | **Full 16×16×K SGEMM, max_diff = 0.0 vs Accelerate** | ✅ |
| 15 | **Benchmark: JIT beats Accelerate 1.8–2.5× at tile size** | ✅ |
| 16 | BFMOPA/SMOPA probing — all decode OK but execute as NOPs | ✅ |
| 17 | Fused GEMM+Activation kernels (ReLU, Bias, BiasReLU) | ✅ |
| 18 | Tiny inference engine — MNIST MLP via chained fused kernels | ✅ |
| 19 | Cached inference engine — 5× faster than uncached, 0.46× Accelerate | ✅ |
| 20 | Pre-transposed input — **1.93× faster than Accelerate** | ✅ |
| 21 | **Tiled GEMM — arbitrary M×N up to 128×128, 5× at 16×16** | ✅ |

See [ROADMAP.md](ROADMAP.md) for detailed findings and architecture notes.

## Project Structure

```
src/
  main.rs              Gate runner (entry point)
  emitter.rs           AArch64/SME instruction encoding + SGEMM kernel builder
  probe.rs             Fork-based opcode probing and block execution
  crucible.rs          Differential correctness testing (Accelerate vs JIT)
  jit_page.rs          MAP_JIT page allocation with W^X toggle
  cpu_state.rs         GPR snapshot capture before/after execution
  signal_handler.rs    Fault recovery (SIGILL/SEGV/SIGBUS/SIGALRM)
  sink.rs              JSONL result logging with resume support
  inference.rs         MNIST inference engine (probed, direct, cached paths)
  weights.rs           Weight loading and test batch management
  lib.rs               Library crate re-exports for benches

scripts/
  train_mnist.py       Train MNIST MLP, export weights as raw f32 binaries

benches/
  crucible_bench.rs    Criterion benchmarks (accelerate vs jit_cold vs jit_hot)
```

## Quick Start

```sh
# Run all gates
cargo run --release

# Run specific gates
cargo run --release -- gate21     # Tiled GEMM correctness + benchmark
cargo run --release -- gate20     # Pre-transposed inference
cargo run --release -- gate18     # MNIST inference engine

# Rigorous microbenchmark (warmup + percentiles)
cargo run --release -- bench21

# Run benchmarks (Accelerate vs JIT)
cargo bench
```

## Requirements

- Apple Silicon Mac (M4 tested; M1–M3 lack SME support)
- macOS Sequoia 15+
- Rust nightly

## Why This Matters

Modern ML infrastructure is built on layers of abstraction — frameworks dispatching to libraries dispatching to drivers dispatching to silicon. Each layer adds latency, memory overhead, and opaque scheduling decisions. At scale, these costs compound. At the edge, they dominate.

`sme-jit-core` strips away every layer and proves what the hardware can actually do when you let it. By reverse-engineering Apple's undocumented M4 matrix coprocessor and emitting instructions directly, we expose the true performance ceiling — and demonstrate that even Apple's own Accelerate framework doesn't reach it for small, latency-critical workloads.

This is the founding artifact of **[Dawn State](https://github.com/dawn-state)** — an independent research lab exploring the full AI compute stack from silicon to swarms. We believe that pushing the ML frontier requires understanding every layer of the stack, starting at the instruction level. `sme-jit-core` is where that work begins.

---

<p align="center">
  <sub>Dawn State · Silicon → Models → Swarms · <a href="https://github.com/dawn-state">github.com/dawn-state</a></sub>
</p>
