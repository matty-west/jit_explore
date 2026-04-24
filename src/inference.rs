//! Tiny MNIST inference engine — runs a 3-layer MLP entirely through
//! JIT'd fused SME kernels. Zero frameworks, zero dispatch overhead.
//!
//! Architecture: 784 → 16 (BiasReLU) → 16 (BiasReLU) → 16 (Bias, padded from 10)
//!
//! Each layer is one call to a pre-compiled JIT page containing:
//! SMSTART → PTRUE → ZERO_ZA → [LD1W×2 + FMOPA + ADD×2]×K → ST1W×16 → activation → SMSTOP → RET
//!
//! Batch=16: all three layers operate on full 16×16 ZA tiles.

use crate::emitter::{build_sme_sgemm_page, build_sme_sgemm_page_cached, build_sme_tiled_sgemm_page_cached, Activation};
use crate::jit_page::JitPage;
use crate::weights::{MnistWeights, MnistWeightsWide};

/// A **cached** 3-layer MLP inference engine (Gate 19).
///
/// JIT pages are compiled once at construction time. Per-inference calls
/// pass only the input/output pointers — no mmap, no munmap, no icache flush.
///
/// Buffers are pre-allocated and reused across calls.
pub struct CachedInferenceEngine {
    /// Layer 1: input_t[784×16] × W1[784×16] + b1 → ReLU → hidden1[16×16]
    page1: JitPage,
    /// Layer 2: hidden1_t[16×16] × W2[16×16] + b2 → ReLU → hidden2[16×16]
    page2: JitPage,
    /// Layer 3: hidden2_t[16×16] × W3[16×16] + b3 → output[16×16]
    page3: JitPage,

    /// Pre-allocated transpose buffer for input: 784×16
    input_t: Vec<f32>,
    /// Pre-allocated output from layer 1: 16×16
    hidden1: Vec<f32>,
    /// Pre-allocated transposed hidden1: 16×16
    hidden1_t: Vec<f32>,
    /// Pre-allocated output from layer 2: 16×16
    hidden2: Vec<f32>,
    /// Pre-allocated transposed hidden2: 16×16
    hidden2_t: Vec<f32>,
    /// Pre-allocated output: 16×16
    output: Vec<f32>,
}

impl CachedInferenceEngine {
    /// Build the cached inference engine. JIT pages are compiled once here.
    ///
    /// The `weights` struct must outlive the engine (weight pointers are
    /// baked into the JIT pages).
    pub fn build(weights: &MnistWeights) -> Result<Self, String> {
        // Build cached pages — B (weights) and bias are baked in,
        // A (input) and C (output) are passed via X0/X1 at call time.
        let page1 = build_sme_sgemm_page_cached(
            784, Activation::BiasReLU,
            weights.w1.as_ptr() as u64,
            weights.b1.as_ptr() as u64,
        ).ok_or("Failed to build cached Layer 1 page")?;

        let page2 = build_sme_sgemm_page_cached(
            16, Activation::BiasReLU,
            weights.w2.as_ptr() as u64,
            weights.b2.as_ptr() as u64,
        ).ok_or("Failed to build cached Layer 2 page")?;

        let page3 = build_sme_sgemm_page_cached(
            16, Activation::Bias,
            weights.w3.as_ptr() as u64,
            weights.b3.as_ptr() as u64,
        ).ok_or("Failed to build cached Layer 3 page")?;

        Ok(Self {
            page1, page2, page3,
            input_t: vec![0.0f32; 784 * 16],
            hidden1: vec![0.0f32; 256],
            hidden1_t: vec![0.0f32; 256],
            hidden2: vec![0.0f32; 256],
            hidden2_t: vec![0.0f32; 256],
            output: vec![0.0f32; 256],
        })
    }

    /// Run inference on a batch of 16 images.
    ///
    /// This is the **performance path** — no allocations, no page rebuilds.
    /// Only the transpose + 3 kernel calls + argmax.
    ///
    /// Returns: predictions[16]
    pub fn run(&mut self, images: &[f32]) -> Vec<u8> {
        assert_eq!(images.len(), 16 * 784, "Expected 16×784 input");

        // ── Transpose input: [16×784] row-major → [784×16] column-major ──
        for i in 0..16 {
            for k in 0..784 {
                self.input_t[k * 16 + i] = images[i * 784 + k];
            }
        }

        // ── Layer 1: call_with_args(A=input_t, C=hidden1) ──
        // SAFETY: page1 is a valid cached kernel, input_t and hidden1 are valid buffers.
        unsafe {
            self.page1.call_with_args(
                self.input_t.as_ptr() as u64,
                self.hidden1.as_mut_ptr() as u64,
            );
        }

        // ── Transpose hidden1 ──
        for i in 0..16 {
            for j in 0..16 {
                self.hidden1_t[j * 16 + i] = self.hidden1[i * 16 + j];
            }
        }

        // ── Layer 2: call_with_args(A=hidden1_t, C=hidden2) ──
        unsafe {
            self.page2.call_with_args(
                self.hidden1_t.as_ptr() as u64,
                self.hidden2.as_mut_ptr() as u64,
            );
        }

        // ── Transpose hidden2 ──
        for i in 0..16 {
            for j in 0..16 {
                self.hidden2_t[j * 16 + i] = self.hidden2[i * 16 + j];
            }
        }

        // ── Layer 3: call_with_args(A=hidden2_t, C=output) ──
        unsafe {
            self.page3.call_with_args(
                self.hidden2_t.as_ptr() as u64,
                self.output.as_mut_ptr() as u64,
            );
        }

        // ── Argmax over first 10 columns ──
        let mut predictions = Vec::with_capacity(16);
        for i in 0..16 {
            let row = &self.output[i * 16..i * 16 + 10];
            let pred = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0);
            predictions.push(pred);
        }
        predictions
    }

    /// Run inference with a **pre-transposed** input (Gate 20).
    ///
    /// Skips the 784×16 transpose entirely — the input must already be
    /// in [784×16] column-major layout (as exported by train_mnist.py).
    ///
    /// Returns: predictions[16]
    pub fn run_pretransposed(&mut self, images_t: &[f32]) -> Vec<u8> {
        assert_eq!(images_t.len(), 784 * 16, "Expected 784×16 pre-transposed input");

        // ── Layer 1: NO TRANSPOSE — use pre-transposed input directly ──
        unsafe {
            self.page1.call_with_args(
                images_t.as_ptr() as u64,
                self.hidden1.as_mut_ptr() as u64,
            );
        }

        // ── Transpose hidden1 (16×16 — only 256 floats, trivial) ──
        for i in 0..16 {
            for j in 0..16 {
                self.hidden1_t[j * 16 + i] = self.hidden1[i * 16 + j];
            }
        }

        // ── Layer 2 ──
        unsafe {
            self.page2.call_with_args(
                self.hidden1_t.as_ptr() as u64,
                self.hidden2.as_mut_ptr() as u64,
            );
        }

        // ── Transpose hidden2 ──
        for i in 0..16 {
            for j in 0..16 {
                self.hidden2_t[j * 16 + i] = self.hidden2[i * 16 + j];
            }
        }

        // ── Layer 3 ──
        unsafe {
            self.page3.call_with_args(
                self.hidden2_t.as_ptr() as u64,
                self.output.as_mut_ptr() as u64,
            );
        }

        // ── Argmax ──
        let mut predictions = Vec::with_capacity(16);
        for i in 0..16 {
            let row = &self.output[i * 16..i * 16 + 10];
            let pred = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0);
            predictions.push(pred);
        }
        predictions
    }

    /// Get the output buffer (16×16, first 10 cols are logits).
    pub fn output(&self) -> &[f32] { &self.output }
    /// Get the hidden1 buffer (16×16).
    pub fn hidden1(&self) -> &[f32] { &self.hidden1 }
    /// Get the hidden2 buffer (16×16).
    pub fn hidden2(&self) -> &[f32] { &self.hidden2 }
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

// ═══════════════════════════════════════════════════════════════════════════════
// Gate 22: Tiled Inference Engine — Wider Hidden Layers
// ═══════════════════════════════════════════════════════════════════════════════

/// A tiled inference engine for wider models (Gate 22).
///
/// Uses `build_sme_tiled_sgemm_page_cached` for arbitrary M×N×K dimensions.
/// Architecture: 784 → H → H → 10 (H = hidden_dim, must be multiple of 16).
///
/// Layer dimensions (batch=16):
///   Layer 1: M=16, N=H, K=784 → output [16×H]
///   Layer 2: M=16, N=H, K=H   → output [16×H]
///   Layer 3: M=16, N=16, K=H  → output [16×16] (padded from 10)
pub struct TiledInferenceEngine {
    hidden_dim: usize,
    page1: JitPage,
    page2: JitPage,
    page3: JitPage,
    /// Pre-allocated buffers
    hidden1: Vec<f32>,      // 16 × H (row-major output from layer 1)
    hidden1_t: Vec<f32>,    // H × 16 (column-major input for layer 2)
    hidden2: Vec<f32>,      // 16 × H
    hidden2_t: Vec<f32>,    // H × 16
    output: Vec<f32>,       // 16 × 16 (padded)
}

impl TiledInferenceEngine {
    /// Build the tiled inference engine.
    ///
    /// Weight pointers are baked into the JIT pages — `weights` must outlive the engine.
    pub fn build(weights: &MnistWeightsWide) -> Result<Self, String> {
        let h = weights.hidden_dim;
        assert!(h % 16 == 0, "hidden_dim must be multiple of 16, got {}", h);

        // Layer 1: M=16, N=H, K=784, BiasReLU
        let page1 = build_sme_tiled_sgemm_page_cached(
            16, h, 784, Activation::BiasReLU,
            weights.w1.as_ptr() as u64,
            weights.b1.as_ptr() as u64,
        ).ok_or("Failed to build tiled Layer 1 page")?;

        // Layer 2: M=16, N=H, K=H, BiasReLU
        let page2 = build_sme_tiled_sgemm_page_cached(
            16, h, h, Activation::BiasReLU,
            weights.w2.as_ptr() as u64,
            weights.b2.as_ptr() as u64,
        ).ok_or("Failed to build tiled Layer 2 page")?;

        // Layer 3: M=16, N=16, K=H, Bias (output padded from 10→16)
        let page3 = build_sme_tiled_sgemm_page_cached(
            16, 16, h, Activation::Bias,
            weights.w3.as_ptr() as u64,
            weights.b3.as_ptr() as u64,
        ).ok_or("Failed to build tiled Layer 3 page")?;

        Ok(Self {
            hidden_dim: h,
            page1, page2, page3,
            hidden1: vec![0.0f32; 16 * h],
            hidden1_t: vec![0.0f32; h * 16],
            hidden2: vec![0.0f32; 16 * h],
            hidden2_t: vec![0.0f32; h * 16],
            output: vec![0.0f32; 16 * 16],
        })
    }

    /// Run inference with pre-transposed input [784×16].
    ///
    /// Returns: predictions[16]
    pub fn run_pretransposed(&mut self, images_t: &[f32]) -> Vec<u8> {
        assert_eq!(images_t.len(), 784 * 16, "Expected 784×16 pre-transposed input");
        let h = self.hidden_dim;

        // ── Layer 1: images_t[784×16] → hidden1[16×H] ──
        // SAFETY: page1 is a valid tiled kernel, pointers are valid.
        unsafe {
            self.page1.call_with_args(
                images_t.as_ptr() as u64,
                self.hidden1.as_mut_ptr() as u64,
            );
        }

        // ── Transpose hidden1: [16×H] row-major → [H×16] column-major ──
        for i in 0..16 {
            for j in 0..h {
                self.hidden1_t[j * 16 + i] = self.hidden1[i * h + j];
            }
        }

        // ── Layer 2: hidden1_t[H×16] → hidden2[16×H] ──
        unsafe {
            self.page2.call_with_args(
                self.hidden1_t.as_ptr() as u64,
                self.hidden2.as_mut_ptr() as u64,
            );
        }

        // ── Transpose hidden2: [16×H] → [H×16] ──
        for i in 0..16 {
            for j in 0..h {
                self.hidden2_t[j * 16 + i] = self.hidden2[i * h + j];
            }
        }

        // ── Layer 3: hidden2_t[H×16] → output[16×16] ──
        unsafe {
            self.page3.call_with_args(
                self.hidden2_t.as_ptr() as u64,
                self.output.as_mut_ptr() as u64,
            );
        }

        // ── Argmax over first 10 columns ──
        let mut predictions = Vec::with_capacity(16);
        for i in 0..16 {
            let row = &self.output[i * 16..i * 16 + 10];
            let pred = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0);
            predictions.push(pred);
        }
        predictions
    }

    /// Run inference with row-major input [16×784] (includes transpose).
    pub fn run(&mut self, images: &[f32]) -> Vec<u8> {
        assert_eq!(images.len(), 16 * 784, "Expected 16×784 input");
        let mut input_t = vec![0.0f32; 784 * 16];
        for i in 0..16 {
            for k in 0..784 {
                input_t[k * 16 + i] = images[i * 784 + k];
            }
        }
        self.run_pretransposed(&input_t)
    }

    pub fn output(&self) -> &[f32] { &self.output }
    pub fn hidden1(&self) -> &[f32] { &self.hidden1 }
    pub fn hidden2(&self) -> &[f32] { &self.hidden2 }
}

/// Reference inference for wide model using Accelerate (cblas_sgemm).
///
/// Returns: (predictions, hidden1[16×H], hidden2[16×H], output[16×16])
pub fn run_inference_reference_wide(
    weights: &MnistWeightsWide,
    images: &[f32],  // 16×784 row-major
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use crate::crucible::Crucible;
    let h = weights.hidden_dim;

    // Layer 1: images[16×784] × W1[784×H] + b1, ReLU
    let h1_raw = Crucible::run_accelerate(16, h, 784, images, &weights.w1);
    let mut hidden1 = vec![0.0f32; 16 * h];
    for i in 0..16 {
        for j in 0..h {
            let val = h1_raw[i * h + j] + weights.b1[j];
            hidden1[i * h + j] = val.max(0.0);
        }
    }

    // Layer 2: hidden1[16×H] × W2[H×H] + b2, ReLU
    let h2_raw = Crucible::run_accelerate(16, h, h, &hidden1, &weights.w2);
    let mut hidden2 = vec![0.0f32; 16 * h];
    for i in 0..16 {
        for j in 0..h {
            let val = h2_raw[i * h + j] + weights.b2[j];
            hidden2[i * h + j] = val.max(0.0);
        }
    }

    // Layer 3: hidden2[16×H] × W3[H×16] + b3 (no ReLU, padded output)
    let out_raw = Crucible::run_accelerate(16, 16, h, &hidden2, &weights.w3);
    let mut output = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        for j in 0..16 {
            output[i * 16 + j] = out_raw[i * 16 + j] + weights.b3[j];
        }
    }

    // Argmax over first 10 columns
    let mut predictions = Vec::with_capacity(16);
    for i in 0..16 {
        let row = &output[i * 16..i * 16 + 10];
        let pred = row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0);
        predictions.push(pred);
    }

    (predictions, hidden1, hidden2, output)
}
