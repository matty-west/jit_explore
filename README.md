<p align="center">
  <img src="sme-jit-core-banner.png" alt="Dawn State — sme-jit-core" width="100%" />
</p>

<p align="center">
  <strong>A <a href="https://github.com/dawn-state">Dawn State</a> research project</strong> · Silicon Exploitation & Bare-Metal Optimization
</p>

---

# sme-jit-core

**A zero-overhead Rust JIT harness for reverse-engineering and executing matrix coprocessor instructions on Apple Silicon M4 — beats Accelerate.framework by 2.5×.**

Empirical opcode probing, fault-tolerant sandboxing, and raw SME instruction emission straight to the silicon.

Two execution modes: a **fork-isolated probe harness** for safely exploring undocumented instructions (fault recovery via SIGILL/SEGV), and a **direct-execution JIT path** that emits machine code into `MAP_JIT` pages and calls it as a native function pointer — zero dispatch, zero abstraction, just `BLR` into your FMOPA sequence. The benchmarks below measure the direct path.

## Benchmark Results

JIT-emitted SME SGEMM kernel vs `cblas_sgemm` (Accelerate.framework), 16×16 tile:

```
                    Accelerate    JIT (direct)        Speedup
K=4  (1K MACs)       117.8 ns        65.0 ns          1.81×
K=16 (4K MACs)       164.5 ns        66.0 ns          2.49×
K=64 (16K MACs)      236.5 ns       101.5 ns          2.33×
```

**Why the JIT kernel wins:** Zero dispatch overhead — no argument validation, no size routing, no thread pool setup. At small tile sizes, Accelerate's dispatch logic dominates its runtime. The JIT kernel bypasses all of it and talks directly to the silicon.

**Differential correctness:** `max_diff = 0.0` vs `cblas_sgemm` across all tested configurations. Every result is bit-identical to Apple's reference implementation.

## M4 SME Parameters (Empirically Determined)

| Parameter | Value |
|:---|:---|
| SVL (Scalable Vector Length) | **512 bits** |
| float32 per Z register | 16 |
| ZA tile dimensions | 16×16 = 256 float32 |
| MACs per `FMOPA` | **256** |

These parameters were discovered through empirical instruction probing — not from Apple documentation, which does not publicly disclose M4 SME configuration details.

## What This Does

1. **Probes** — Executes arbitrary AArch64 opcodes in a fault-tolerant JIT sandbox (fork + SIGILL/SEGV/SIGBUS recovery)
2. **Discovers** — Proved M4 uses ARM SME through empirical instruction probing when Apple's own documentation remained silent
3. **Computes** — JIT-emits SME instruction sequences (LD1W → FMOPA → ST1W) producing correct SGEMM results
4. **Verifies** — Differential correctness via Crucible: `max_diff = 0.0` vs `cblas_sgemm`
5. **Benchmarks** — Criterion benchmarks comparing JIT throughput against Accelerate

## Milestones

| Gate | Goal | Status |
|:-----|:-----|:-------|
| 0–9 | JIT page allocation, fault recovery, fork probing, GPR snapshots | ✅ |
| 10–13 | Frida heist, AMX leaf execution (exploration phase — archived) | ✅ |
| 14a–b | Synthetic AMX tests + operand bit-walk (all zeros — dead end) | ✅ |
| 14c | **SME pivot: FMOPA produces first non-zero result** | ✅ |
| 14d | **Full 16×16×K SGEMM, max_diff = 0.0 vs Accelerate** | ✅ |
| 15 | **Benchmark: JIT beats Accelerate 1.8–2.5×** | ✅ |
| 16 | BFMOPA/SMOPA probing — discover BF16/INT8 outer product support | ✅ |
| 17 | Fused GEMM+Activation kernels (matmul+ReLU/GELU in one kernel) | ✅ |
| 18 | Tiny inference engine demo (MLP via chained fused kernels) | ⬜ |

See [ROADMAP.md](ROADMAP.md) for detailed next-step plans.

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
  lib.rs               Library crate re-exports for benches

benches/
  crucible_bench.rs    Criterion benchmarks (accelerate vs jit_cold vs jit_hot)
```

## Quick Start

```sh
# Run the current gate (16×16 SGEMM correctness test)
cargo run --release

# Run benchmarks (Accelerate vs JIT)
cargo bench

# Just the direct-execution JIT benchmark
cargo bench -- jit_hot

# Just the Accelerate baseline
cargo bench -- accelerate
```

## Requirements

- Apple Silicon Mac (M4 tested; M1–M3 lack SME support)
- macOS Sequoia 15+
- Rust nightly

## Why This Matters

Modern ML infrastructure is built on layers of abstraction — frameworks dispatching to libraries dispatching to drivers dispatching to silicon. Each layer adds latency, memory overhead, and opaque scheduling decisions. At scale, these costs compound. At the edge, they dominate.

`sme-jit-core` strips away every layer and proves what the hardware can actually do when you let it. By reverse-engineering Apple's undocumented M4 matrix coprocessor and emitting instructions directly, we expose the true performance ceiling — and demonstrate that even Apple's own Accelerate framework doesn't reach it.

This is the founding artifact of **[Dawn State](https://github.com/dawn-state)** — an independent research lab exploring the full AI compute stack from silicon to swarms. We believe that pushing the ML frontier requires understanding every layer of the stack, starting at the instruction level. `sme-jit-core` is where that work begins.

---

<p align="center">
  <sub>Dawn State · Silicon → Models → Swarms · <a href="https://github.com/dawn-state">github.com/dawn-state</a></sub>
</p>
