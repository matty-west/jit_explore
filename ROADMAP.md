# OPCODE Roadmap

## Where We Are

Gates 0–15 are complete. We have:
- A working JIT harness (MAP_JIT, fork isolation, GPR snapshots)
- Proof that M4 uses ARM SME (not AMX)
- A 16×16 SGEMM kernel (PTRUE → ZERO ZA → [LD1W×2 + FMOPA + ADD×2]×K → [ST1W + ADD×2]×16)
- Differential correctness: `max_diff = 0.0` vs Accelerate
- Benchmark: 1.8–2.5× faster than Accelerate at tile-sized problems

## Gate 16: Multi-Tile Tiling (M,N > 16)

**Goal**: Handle matrices larger than one ZA tile (16×16).

The current kernel computes exactly one 16×16 output tile. For larger M×N, we need
an outer loop that walks tiles across the output matrix:

```
for tile_m in (0..M).step_by(16):
  for tile_n in (0..N).step_by(16):
    ZERO {ZA}
    for kk in 0..K:
      LD1W Z0, [A + tile_m*K + kk*16]
      LD1W Z1, [B + kk*N + tile_n]
      FMOPA ZA0.S, P0/M, Z0.S, Z1.S
    ST1W ZA0H[0..15] → C[tile_m..tile_m+16, tile_n..tile_n+16]
```

**Key decisions**:
- Emit the tile loop in JIT (unrolled or with branch-back)?
- Pointer arithmetic: stride-aware addressing for row-major layout
- Edge tiles: predicate masking when M%16 ≠ 0 or N%16 ≠ 0

**Correctness target**: `max_diff < 1e-4` vs Accelerate for 64×64, 128×128, 256×256.

## Gate 17: OS Scheduler Bypass

**Goal**: Pin execution to P-cores and lock memory for deterministic benchmarks.

- `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` for P-core affinity
- `mlock()` on JIT pages + input/output buffers
- Measure variance reduction in Criterion benchmarks

## Gate 18: Microarchitecture Optimization

**Goal**: Extract more throughput from the M4 SME pipeline.

Candidates:
- **Double-buffered loads**: Interleave Z0/Z2 and Z1/Z3 loads with FMOPA to hide load latency
- **FMOPA chaining**: Use multiple ZA tiles (ZA0–ZA3) if M4 supports concurrent accumulation
- **BFMOPA**: BF16 outer product — 2× MACs per FMOPA if the hardware supports it
- **SMOPA (I8)**: Integer 8-bit outer product for quantized inference

## Gate 19: Larger Kernel Library

**Goal**: Build out a suite of JIT-emitted kernels beyond SGEMM.

- DGEMM (FP64 outer product via FMOPA.D)
- SYRK / TRSM building blocks
- Batched small-matrix multiply
- Activation function fusion (ReLU/GELU after store)

## Non-Goals (Archived)

- ~~AMX instruction encoding~~ — dead on M4
- ~~Frida heist scripts~~ — exploration complete, data preserved in git history
- ~~Planning documents~~ — replaced by this roadmap
