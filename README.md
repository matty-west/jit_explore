# jit_explore — Bare-Metal Apple M4 SME Exploration

A Rust JIT harness for reverse-engineering and executing matrix coprocessor instructions on Apple Silicon that **beats Accelerate.framework by 2.5×** at small tile sizes.

**M4 SME parameters:**
- SVL (Scalable Vector Length) = **512 bits**
- 16 float32 per Z register
- ZA tile = 16×16 = 256 float32 values
- One `FMOPA` computes **256 multiply-accumulates** per instruction

## Benchmark Results

JIT-emitted SME SGEMM kernel vs `cblas_sgemm` (Accelerate.framework), 16×16 tile:

```
                    Accelerate    JIT (bare-metal)    Speedup
K=4  (1K MACs)       117.8 ns        65.0 ns          1.81×
K=16 (4K MACs)       164.5 ns        66.0 ns          2.49×
K=64 (16K MACs)      236.5 ns       101.5 ns          2.33×
```

The JIT kernel wins because it has **zero dispatch overhead** — no argument validation, no size routing, no thread pool setup. At small tile sizes, Accelerate's dispatch logic dominates its runtime.

## What This Does

1. **Probes** — Executes arbitrary AArch64 opcodes in a fault-tolerant JIT sandbox (fork + SIGILL/SEGV/SIGBUS recovery)
2. **Discovers** — Proved M4 uses ARM SME through empirical instruction probing
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
| 16 | Multi-tile tiling (M,N > 16) | ⬜ |
| 17 | OS scheduler bypass (P-core pinning, mlock) | ⬜ |
| 18 | Double-buffered loads + BFMOPA exploration | ⬜ |

See [ROADMAP.md](ROADMAP.md) for detailed next-step plans.

## Project Structure

```
src/
  main.rs              Gate functions (progressive capability tests)
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

# Just the bare-metal JIT benchmark
cargo bench -- jit_hot

# Just the Accelerate baseline
cargo bench -- accelerate
```

## Requirements

- Apple Silicon Mac (M4 tested; M1–M3 lack SME support)
- macOS Sequoia 15+
- Rust nightly
