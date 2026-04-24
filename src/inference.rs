//! Tiny MNIST inference engine — runs a 3-layer MLP entirely through
//! JIT'd fused SME kernels. Zero frameworks, zero dispatch overhead.
//!
//! Architecture: 784 → 16 (BiasReLU) → 16 (BiasReLU) → 16 (Bias, padded from 10)
//!
//! Each layer is one call to a pre-compiled JIT page containing:
//! SMSTART → PTRUE → ZERO_ZA → [LD1W×2 + FMOPA + ADD×2]×K → ST1W×16 → activation → SMSTOP → RET
//!
//! Batch=16: all three layers operate on full 16×16 ZA tiles.

use crate::emitter::{build_sme_sgemm_page, Activation};
use crate::jit_page::JitPage;
use crate::weights::MnistWeights;

/// A pre-compiled 3-layer MLP inference engine.
///
/// Each layer is a single JIT page callable via `page.call_void()`.
/// Pointers to weights, biases, and intermediate buffers are baked
/// into the machine code at construction time.
pub struct InferenceEngine {
    /// Layer 1: input[16×784] × W1[784×16] + b1 → ReLU → hidden1[16×16]
    _page1: JitPage,
    /// Layer 2: hidden1[16×16] × W2[16×16] + b2 → ReLU → hidden2[16×16]
    _page2: JitPage,
    /// Layer 3: hidden2[16×16] × W3[16×16] + b3 → logits[16×16]
    _page3: JitPage,

    /// Input buffer: 16 images × 784 features = 12544 floats.
    /// Must be written before calling `run()`.
    input_buf: Vec<f32>,

    /// Intermediate buffer after layer 1: 16×16 = 256 floats.
    hidden1_buf: Vec<f32>,

    /// Intermediate buffer after layer 2: 16×16 = 256 floats.
    hidden2_buf: Vec<f32>,

    /// Output buffer: 16×16 = 256 floats (only first 10 cols are logits).
    output_buf: Vec<f32>,
}

impl InferenceEngine {
    /// Build the inference engine by JIT-compiling all three layers.
    ///
    /// The `weights` struct must outlive the engine (weights are referenced
    /// by pointer from the JIT pages). In practice we keep them alive in
    /// the same scope.
    pub fn build(weights: &MnistWeights) -> Result<Self, String> {
        // Allocate intermediate buffers
        let input_buf = vec![0.0f32; 16 * 784];
        let mut hidden1_buf = vec![0.0f32; 16 * 16];
        let mut hidden2_buf = vec![0.0f32; 16 * 16];
        let mut output_buf = vec![0.0f32; 16 * 16];

        // Layer 1: 16×784 × 784×16, K=49
        // Input A = input_buf (16 images, 784 features each, stored row-major)
        //   The kernel loads A as K panels of 16 floats. With row-major [16×784],
        //   panel k reads input_buf[k*16..(k+1)*16] across the batch dimension.
        //   But wait — our kernel loads A[k] from X4+k*64. For a 16×784 input,
        //   we need the kernel to iterate over 784/16 = 49 chunks of the input.
        //
        //   FMOPA computes ZA += Z0 ⊗ Z1 (outer product).
        //   Z0 comes from A (input), Z1 comes from B (weights).
        //   After K iterations: ZA[i][j] = Σ_k A_panel[k][i] × B_panel[k][j]
        //
        //   For correct matmul C = Input × Weights:
        //   - A_panel[k][i] should be Input[i][k] (the k-th feature of image i)
        //   - B_panel[k][j] should be Weights[k][j] (the k-th row of weights)
        //
        //   Input is [16×784] row-major, so Input[i][k] is at offset i*784+k.
        //   But the kernel loads 16 contiguous floats per panel.
        //   We need Input transposed to [784×16] so panel k = Input_T[k*16..(k+1)*16]
        //   gives us [Input[0][k], Input[1][k], ..., Input[15][k]].
        //
        //   So: A pointer = transposed input buffer (784×16)
        //       B pointer = W1 (784×16, already row-major)
        //
        //   We'll transpose the input at runtime (cheap: just a memory shuffle).
        //   Actually, let's pre-allocate a transpose buffer.

        let input_t_buf = vec![0.0f32; 784 * 16]; // Will be filled at runtime

        let page1 = build_sme_sgemm_page(
            49, // K = 784/16
            Activation::BiasReLU,
            input_t_buf.as_ptr() as u64,     // A = transposed input
            weights.w1.as_ptr() as u64,       // B = W1
            hidden1_buf.as_mut_ptr() as u64,  // C = hidden1
            weights.b1.as_ptr() as u64,       // bias
        )
        .ok_or("Failed to build Layer 1 JIT page")?;

        // Layer 2: 16×16 × 16×16, K=1
        // hidden1 is already [16×16] row-major.
        // For FMOPA: A_panel[0] = column 0 of hidden1 across all 16 rows.
        // hidden1[i][0] is at offset i*16. Contiguous stride=16, not 1.
        // So we need hidden1 transposed too... OR we use K=1 where:
        //   Z0 = 16 floats from hidden1 starting at offset 0 (first 16 floats = row 0)
        //   Z1 = 16 floats from W2 starting at offset 0 (first 16 floats = row 0)
        //   ZA += Z0 ⊗ Z1 → ZA[i][j] = hidden1[0][i] × W2[0][j]
        //
        // That's wrong — it only uses row 0. For K=1 with 16×16:
        //   We need K=16, loading one row at a time!
        //   Wait, let me re-examine the kernel...
        //
        // The kernel does:
        //   for k in 0..K:
        //     Z0 = LD1W from A + k*64
        //     Z1 = LD1W from B + k*64
        //     ZA += Z0 ⊗ Z1
        //
        // For C = A × B where A is [16×16] and B is [16×16]:
        //   ZA[i][j] = Σ_k Z0_k[i] × Z1_k[j]
        //
        // With K=16 and stride 64 (16 floats):
        //   Z0_k = A[k*16..(k+1)*16] → row k of a [16×16] matrix stored row-major
        //     But Z0_k[i] should be A[i][k] (column k), not A[k][i] (row k).
        //
        // So for [16×16] × [16×16], we need:
        //   A stored as [16×16] COLUMN-major (or transposed row-major)
        //   B stored as [16×16] ROW-major (already correct)
        //   K = 16
        //
        // hidden1 is output from layer 1 as row-major [16×16].
        // We need it transposed for layer 2's input.
        //
        // Same issue for layer 3: hidden2 needs transposing.
        //
        // Solution: Transpose each intermediate buffer before the next layer.
        // For 16×16, transpose is trivial (256 float swaps, ~1 cache line).

        let hidden1_t_buf = vec![0.0f32; 16 * 16]; // transposed hidden1

        let page2 = build_sme_sgemm_page(
            16, // K = 16 (full 16×16 matmul)
            Activation::BiasReLU,
            hidden1_t_buf.as_ptr() as u64,    // A = transposed hidden1
            weights.w2.as_ptr() as u64,        // B = W2
            hidden2_buf.as_mut_ptr() as u64,   // C = hidden2
            weights.b2.as_ptr() as u64,        // bias
        )
        .ok_or("Failed to build Layer 2 JIT page")?;

        let hidden2_t_buf = vec![0.0f32; 16 * 16]; // transposed hidden2

        // Layer 3: 16×16 × 16×16 (padded), K=16, no ReLU (just bias)
        let page3 = build_sme_sgemm_page(
            16,
            Activation::Bias,
            hidden2_t_buf.as_ptr() as u64,     // A = transposed hidden2
            weights.w3.as_ptr() as u64,         // B = W3 (zero-padded to 16×16)
            output_buf.as_mut_ptr() as u64,     // C = output
            weights.b3.as_ptr() as u64,         // bias (zero-padded)
        )
        .ok_or("Failed to build Layer 3 JIT page")?;

        Ok(Self {
            _page1: page1,
            _page2: page2,
            _page3: page3,
            input_buf,
            hidden1_buf,
            hidden2_buf,
            output_buf,
        })
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize { 3 }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Simpler approach: Use the fork-based probe for correctness first,
// then optimize with direct JIT calls.
// ═══════════════════════════════════════════════════════════════════════════════

/// Run a complete 3-layer MLP inference using the fork-based probe harness.
///
/// This is the **correctness path** — each layer runs in a forked child process
/// with fault recovery. Slower than direct JIT, but safe for development.
///
/// Returns: (predictions[16], hidden1[16×16], hidden2[16×16], logits[16×16])
pub fn run_inference_probed(
    weights: &MnistWeights,
    images: &[f32],  // 16×784 row-major
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
    use crate::emitter::{build_sme_sgemm_16x16, Activation};
    use crate::probe::SharedMemory;
    use crate::crucible::Crucible;

    assert_eq!(images.len(), 16 * 784, "Expected 16×784 input");

    // ── Transpose input: [16×784] row-major → [784×16] row-major ──
    // input_t[k][i] = images[i][k] = images[i*784 + k]
    // input_t stored row-major: input_t[k*16 + i] = images[i*784 + k]
    let mut input_t = vec![0.0f32; 784 * 16];
    for i in 0..16 {
        for k in 0..784 {
            input_t[k * 16 + i] = images[i * 784 + k];
        }
    }

    // ── Layer 1: input_t[784×16] × W1[784×16] → hidden1[16×16], K=784 ──
    // Each FMOPA is one rank-1 outer product = one feature.
    // 784 features → 784 outer products.
    let hidden1_shared = SharedMemory::<[f32; 256]>::new();
    let h1_ptr = hidden1_shared.as_mut_ptr() as *mut f32;
    unsafe { std::ptr::write_bytes(h1_ptr, 0, 256); }

    let block1 = build_sme_sgemm_16x16(784, Activation::BiasReLU);
    let overrides1: Vec<(u8, u64)> = vec![
        (2,  h1_ptr as u64),
        (3,  0u64),
        (4,  input_t.as_ptr() as u64),
        (5,  weights.w1.as_ptr() as u64),
        (6,  weights.b1.as_ptr() as u64),
        (12, 0u64),
    ];

    let crucible = Crucible::new();
    let result1 = crucible.probe.run_block_with_overrides(&block1, &overrides1, true);
    if result1.faulted {
        return Err(format!("Layer 1 FAULT: {}", result1.status()));
    }

    let hidden1: Vec<f32> = unsafe { std::slice::from_raw_parts(h1_ptr, 256).to_vec() };

    // ── Transpose hidden1: [16×16] row-major → [16×16] transposed ──
    let mut hidden1_t = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            hidden1_t[j * 16 + i] = hidden1[i * 16 + j];
        }
    }

    // ── Layer 2: hidden1_t[16×16] × W2[16×16] → hidden2[16×16], K=16 ──
    let hidden2_shared = SharedMemory::<[f32; 256]>::new();
    let h2_ptr = hidden2_shared.as_mut_ptr() as *mut f32;
    unsafe { std::ptr::write_bytes(h2_ptr, 0, 256); }

    let block2 = build_sme_sgemm_16x16(16, Activation::BiasReLU);
    let overrides2: Vec<(u8, u64)> = vec![
        (2,  h2_ptr as u64),
        (3,  0u64),
        (4,  hidden1_t.as_ptr() as u64),
        (5,  weights.w2.as_ptr() as u64),
        (6,  weights.b2.as_ptr() as u64),
        (12, 0u64),
    ];

    let result2 = crucible.probe.run_block_with_overrides(&block2, &overrides2, true);
    if result2.faulted {
        return Err(format!("Layer 2 FAULT: {}", result2.status()));
    }

    let hidden2: Vec<f32> = unsafe { std::slice::from_raw_parts(h2_ptr, 256).to_vec() };

    // ── Transpose hidden2 ──
    let mut hidden2_t = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            hidden2_t[j * 16 + i] = hidden2[i * 16 + j];
        }
    }

    // ── Layer 3: hidden2_t[16×16] × W3[16×16] → output[16×16], K=16 ──
    let output_shared = SharedMemory::<[f32; 256]>::new();
    let out_ptr = output_shared.as_mut_ptr() as *mut f32;
    unsafe { std::ptr::write_bytes(out_ptr, 0, 256); }

    let block3 = build_sme_sgemm_16x16(16, Activation::Bias);
    let overrides3: Vec<(u8, u64)> = vec![
        (2,  out_ptr as u64),
        (3,  0u64),
        (4,  hidden2_t.as_ptr() as u64),
        (5,  weights.w3.as_ptr() as u64),
        (6,  weights.b3.as_ptr() as u64),
        (12, 0u64),
    ];

    let result3 = crucible.probe.run_block_with_overrides(&block3, &overrides3, true);
    if result3.faulted {
        return Err(format!("Layer 3 FAULT: {}", result3.status()));
    }

    let output: Vec<f32> = unsafe { std::slice::from_raw_parts(out_ptr, 256).to_vec() };

    // ── Argmax over first 10 columns of each row ──
    let mut predictions = Vec::with_capacity(16);
    for i in 0..16 {
        let row = &output[i * 16..i * 16 + 10]; // only first 10 (actual classes)
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0);
        predictions.push(pred);
    }

    Ok((predictions, hidden1, hidden2, output))
}

/// Run inference using direct JIT page calls (no fork overhead).
///
/// This is the **performance path** — used for benchmarking.
/// Requires pre-built JIT pages with baked pointers.
///
/// Returns: predictions[16]
pub fn run_inference_direct(
    weights: &MnistWeights,
    images: &[f32],  // 16×784 row-major
) -> Result<Vec<u8>, String> {
    use crate::emitter::{build_sme_sgemm_page, Activation};

    assert_eq!(images.len(), 16 * 784, "Expected 16×784 input");

    // ── Transpose input ──
    let mut input_t = vec![0.0f32; 784 * 16];
    for i in 0..16 {
        for k in 0..784 {
            input_t[k * 16 + i] = images[i * 784 + k];
        }
    }

    // ── Buffers ──
    let mut hidden1 = vec![0.0f32; 256];
    let mut hidden2 = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 256];

    // ── Build pages ──
    let page1 = build_sme_sgemm_page(
        784, Activation::BiasReLU,
        input_t.as_ptr() as u64,
        weights.w1.as_ptr() as u64,
        hidden1.as_mut_ptr() as u64,
        weights.b1.as_ptr() as u64,
    ).ok_or("Failed to build Layer 1 page")?;

    // Execute layer 1
    unsafe { page1.call_void(); }

    // Transpose hidden1
    let mut hidden1_t = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            hidden1_t[j * 16 + i] = hidden1[i * 16 + j];
        }
    }

    let page2 = build_sme_sgemm_page(
        16, Activation::BiasReLU,
        hidden1_t.as_ptr() as u64,
        weights.w2.as_ptr() as u64,
        hidden2.as_mut_ptr() as u64,
        weights.b2.as_ptr() as u64,
    ).ok_or("Failed to build Layer 2 page")?;

    unsafe { page2.call_void(); }

    // Transpose hidden2
    let mut hidden2_t = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            hidden2_t[j * 16 + i] = hidden2[i * 16 + j];
        }
    }

    let page3 = build_sme_sgemm_page(
        16, Activation::Bias,
        hidden2_t.as_ptr() as u64,
        weights.w3.as_ptr() as u64,
        output.as_mut_ptr() as u64,
        weights.b3.as_ptr() as u64,
    ).ok_or("Failed to build Layer 3 page")?;

    unsafe { page3.call_void(); }

    // ── Argmax ──
    let mut predictions = Vec::with_capacity(16);
    for i in 0..16 {
        let row = &output[i * 16..i * 16 + 10];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0);
        predictions.push(pred);
    }

    Ok(predictions)
}

/// Run the reference MLP in pure Rust (for differential testing).
///
/// Uses cblas_sgemm for each layer's matmul, then applies bias + activation manually.
pub fn run_inference_reference(
    weights: &MnistWeights,
    images: &[f32],  // 16×784 row-major
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use crate::crucible::Crucible;

    // Layer 1: images[16×784] × W1[784×16] + b1, ReLU
    let h1_raw = Crucible::run_accelerate(16, 16, 784, images, &weights.w1);
    let mut hidden1 = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let val = h1_raw[i * 16 + j] + weights.b1[j];
            hidden1[i * 16 + j] = val.max(0.0); // ReLU
        }
    }

    // Layer 2: hidden1[16×16] × W2[16×16] + b2, ReLU
    let h2_raw = Crucible::run_accelerate(16, 16, 16, &hidden1, &weights.w2);
    let mut hidden2 = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let val = h2_raw[i * 16 + j] + weights.b2[j];
            hidden2[i * 16 + j] = val.max(0.0); // ReLU
        }
    }

    // Layer 3: hidden2[16×16] × W3[16×16] + b3 (no ReLU)
    let out_raw = Crucible::run_accelerate(16, 16, 16, &hidden2, &weights.w3);
    let mut output = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            output[i * 16 + j] = out_raw[i * 16 + j] + weights.b3[j];
        }
    }

    // Argmax over first 10 columns
    let mut predictions = Vec::with_capacity(16);
    for i in 0..16 {
        let row = &output[i * 16..i * 16 + 10];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0);
        predictions.push(pred);
    }

    (predictions, hidden1, hidden2, output)
}
