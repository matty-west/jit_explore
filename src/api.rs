//! # sme-jit-core Public API
//!
//! Ergonomic wrappers around the raw JIT infrastructure. Users interact with
//! `SmeGemm` (single matmul kernel) and `SmeMlp` (fused multi-layer inference)
//! without touching pointer arithmetic, column-major layouts, or ZA tile details.
//!
//! ## Quick Example — Single GEMM
//!
//! ```rust,no_run
//! use sme_jit_core::api::{SmeGemm, Activation};
//!
//! let m = 16; let n = 16; let k = 32;
//! let weights = vec![1.0f32; k * n]; // row-major [K×N]
//! let input   = vec![1.0f32; m * k]; // row-major [M×K] (transposed internally)
//! let mut output = vec![0.0f32; m * n];
//!
//! let kernel = SmeGemm::new(m, n, k, &weights, None, Activation::None).unwrap();
//! kernel.run(&input, &mut output);
//! // output[i][j] == K for all i,j (all-ones × all-ones)
//! ```
//!
//! ## Quick Example — Fused MLP
//!
//! ```rust,no_run
//! use sme_jit_core::api::{SmeMlp, LayerConfig, Activation};
//!
//! let mlp = SmeMlp::new(784, &[
//!     LayerConfig { n: 48, weights: w1, bias: b1, activation: Activation::BiasReLU },
//!     LayerConfig { n: 48, weights: w2, bias: b2, activation: Activation::BiasReLU },
//!     LayerConfig { n: 16, weights: w3, bias: b3, activation: Activation::Bias },
//! ]).unwrap();
//!
//! let mut output = vec![0.0f32; 16 * 16];
//! mlp.run(&input_col_major, &mut output);
//! ```

use crate::emitter::{
    build_sme_tiled_sgemm_page_cached, build_monolithic_inference_page,
    MonolithicLayerConfig,
};
use crate::jit_page::JitPage;

// Re-export Activation so users don't need to reach into emitter
pub use crate::emitter::Activation;

// ═══════════════════════════════════════════════════════════════════════════════
// Error Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors from kernel construction or execution.
#[derive(Debug)]
pub enum SmeError {
    /// Dimension is not a positive multiple of 16.
    BadDimension { name: &'static str, value: usize },
    /// Weight or bias slice has wrong length.
    WrongLength { name: &'static str, expected: usize, got: usize },
    /// JIT page allocation failed (mmap error).
    PageAllocFailed,
    /// K dimension out of range (must be 1..65535).
    KOutOfRange(usize),
    /// Too many layers (max 4).
    TooManyLayers(usize),
    /// Need at least one layer.
    NoLayers,
}

impl std::fmt::Display for SmeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SmeError::BadDimension { name, value } =>
                write!(f, "{name} must be a positive multiple of 16, got {value}"),
            SmeError::WrongLength { name, expected, got } =>
                write!(f, "{name}: expected {expected} elements, got {got}"),
            SmeError::PageAllocFailed =>
                write!(f, "JIT page allocation failed (mmap error)"),
            SmeError::KOutOfRange(k) =>
                write!(f, "K must be 1..65535, got {k}"),
            SmeError::TooManyLayers(n) =>
                write!(f, "max 4 layers supported, got {n}"),
            SmeError::NoLayers =>
                write!(f, "need at least one layer"),
        }
    }
}

impl std::error::Error for SmeError {}

// ═══════════════════════════════════════════════════════════════════════════════
// SmeGemm — Single Tiled GEMM Kernel
// ═══════════════════════════════════════════════════════════════════════════════

/// A JIT-compiled tiled SGEMM kernel for Apple Silicon M4 SME.
///
/// Built once, called many times. Weights (and optional bias) are baked into
/// the instruction stream. Input and output pointers are passed at call time.
///
/// **Performance**: 5× faster than Accelerate at 16×16, 1.4× at 48×48.
/// Accelerate wins at ≥64×64.
///
/// **Constraints**: M, N must be positive multiples of 16 (max 128).
/// K must be 1..65535.
pub struct SmeGemm {
    page: JitPage,
    m: usize,
    n: usize,
    k: usize,
    /// Owned copies of weights/bias to guarantee lifetime
    _weights: Vec<f32>,
    _bias: Vec<f32>,
}

impl SmeGemm {
    /// Build a tiled SGEMM kernel.
    ///
    /// - `m`, `n`: output dimensions (must be multiples of 16, max 128)
    /// - `k`: inner dimension (1..65535)
    /// - `weights`: row-major `[K × N]` weight matrix
    /// - `bias`: optional `N`-element bias vector
    /// - `act`: fused activation function
    ///
    /// The kernel computes: `C = A × weights [+ bias] [+ activation]`
    pub fn new(
        m: usize, n: usize, k: usize,
        weights: &[f32],
        bias: Option<&[f32]>,
        act: Activation,
    ) -> Result<Self, SmeError> {
        // Validate dimensions
        if m == 0 || m % 16 != 0 || m > 128 {
            return Err(SmeError::BadDimension { name: "M", value: m });
        }
        if n == 0 || n % 16 != 0 || n > 128 {
            return Err(SmeError::BadDimension { name: "N", value: n });
        }
        if k == 0 || k > 65535 {
            return Err(SmeError::KOutOfRange(k));
        }
        if weights.len() != k * n {
            return Err(SmeError::WrongLength {
                name: "weights", expected: k * n, got: weights.len(),
            });
        }

        let needs_bias = act == Activation::Bias || act == Activation::BiasReLU;
        let owned_weights = weights.to_vec();
        let owned_bias = if needs_bias {
            let b = bias.ok_or(SmeError::WrongLength {
                name: "bias", expected: n, got: 0,
            })?;
            if b.len() != n {
                return Err(SmeError::WrongLength {
                    name: "bias", expected: n, got: b.len(),
                });
            }
            b.to_vec()
        } else {
            vec![]
        };

        let bias_ptr = if needs_bias { owned_bias.as_ptr() as u64 } else { 0 };

        let page = build_sme_tiled_sgemm_page_cached(
            m, n, k, act,
            owned_weights.as_ptr() as u64,
            bias_ptr,
        ).ok_or(SmeError::PageAllocFailed)?;

        Ok(Self { page, m, n, k, _weights: owned_weights, _bias: owned_bias })
    }

    /// Execute the kernel: `C = A × W [+ bias] [+ act]`
    ///
    /// - `a`: column-major input `[K × M]` (K columns of M floats each)
    /// - `c`: row-major output buffer `[M × N]`
    ///
    /// # Panics
    /// Panics if `a` or `c` have wrong lengths.
    pub fn run(&self, a: &[f32], c: &mut [f32]) {
        assert_eq!(a.len(), self.k * self.m,
            "input A must be {} elements (K={} × M={}), got {}",
            self.k * self.m, self.k, self.m, a.len());
        assert_eq!(c.len(), self.m * self.n,
            "output C must be {} elements (M={} × N={}), got {}",
            self.m * self.n, self.m, self.n, c.len());

        unsafe {
            self.page.call_with_args(a.as_ptr() as u64, c.as_mut_ptr() as u64);
        }
    }

    /// Execute with row-major input `[M × K]` — transposes to column-major internally.
    ///
    /// Convenience method for when your data is already in standard row-major layout.
    /// Adds ~0.5-2 μs for the transpose depending on dimensions.
    pub fn run_row_major(&self, a_row_major: &[f32], c: &mut [f32]) {
        assert_eq!(a_row_major.len(), self.m * self.k);
        let mut a_col = vec![0.0f32; self.k * self.m];
        for i in 0..self.m {
            for j in 0..self.k {
                a_col[j * self.m + i] = a_row_major[i * self.k + j];
            }
        }
        self.run(&a_col, c);
    }

    /// Returns (M, N, K) dimensions.
    pub fn dims(&self) -> (usize, usize, usize) { (self.m, self.n, self.k) }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SmeMlp — Fused Multi-Layer Inference Engine
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for one layer of an MLP.
#[derive(Clone)]
pub struct LayerConfig {
    /// Output width (must be multiple of 16).
    pub n: usize,
    /// Weight matrix, row-major `[K × N]` where K is the previous layer's output width
    /// (or the input width for the first layer).
    pub weights: Vec<f32>,
    /// Bias vector, `N` floats.
    pub bias: Vec<f32>,
    /// Fused activation function.
    pub activation: Activation,
}

/// A fused multi-layer perceptron compiled into a single JIT page.
///
/// All layers execute in one SMSTART/SMSTOP pair with zero inter-layer
/// transposes (column-major stores via ST1W vertical slices). This is the
/// fastest inference path — **1.55× faster than Accelerate** for 784→48→48→10.
///
/// Batch size is fixed at 16 (one ZA tile height).
///
/// **Constraints**:
/// - Input must be column-major `[K × 16]`
/// - All layer widths must be multiples of 16
/// - Maximum 4 layers
pub struct SmeMlp {
    page: JitPage,
    input_k: usize,
    output_n: usize,
    /// Intermediate buffer 1 — pointer baked into JIT page, must outlive page
    #[allow(dead_code)]
    buf1: Vec<f32>,
    /// Intermediate buffer 2 — pointer baked into JIT page, must outlive page
    #[allow(dead_code)]
    buf2: Vec<f32>,
    /// Output buffer — kept for lifetime guarantee
    #[allow(dead_code)]
    output: Vec<f32>,
    /// Owned weight/bias data — pointers baked into the JIT page
    _layer_data: Vec<(Vec<f32>, Vec<f32>)>,
}

impl SmeMlp {
    /// Build a fused MLP from layer configs.
    ///
    /// - `input_k`: input dimension (number of features, e.g. 784 for MNIST)
    /// - `layers`: 1–4 layer configurations
    ///
    /// All weight/bias data is copied and owned by the engine. The JIT page
    /// bakes pointers to the owned copies, so the engine is fully self-contained.
    pub fn new(input_k: usize, layers: &[LayerConfig]) -> Result<Self, SmeError> {
        if layers.is_empty() { return Err(SmeError::NoLayers); }
        if layers.len() > 4 { return Err(SmeError::TooManyLayers(layers.len())); }

        // Validate dimensions
        let mut prev_n = input_k;
        for (i, layer) in layers.iter().enumerate() {
            if layer.n == 0 || layer.n % 16 != 0 {
                return Err(SmeError::BadDimension {
                    name: if i == 0 { "layer1.n" } else if i == 1 { "layer2.n" } else { "layer3.n" },
                    value: layer.n,
                });
            }
            let expected_w = prev_n * layer.n;
            if layer.weights.len() != expected_w {
                return Err(SmeError::WrongLength {
                    name: "weights", expected: expected_w, got: layer.weights.len(),
                });
            }
            let needs_bias = layer.activation == Activation::Bias
                || layer.activation == Activation::BiasReLU;
            if needs_bias && layer.bias.len() != layer.n {
                return Err(SmeError::WrongLength {
                    name: "bias", expected: layer.n, got: layer.bias.len(),
                });
            }
            prev_n = layer.n;
        }

        // Own all layer data
        let layer_data: Vec<(Vec<f32>, Vec<f32>)> = layers.iter()
            .map(|l| (l.weights.clone(), l.bias.clone()))
            .collect();

        // Find max hidden dim for buffer sizing
        let max_n = layers.iter().map(|l| l.n).max().unwrap();
        let buf1 = vec![0.0f32; max_n * 16];
        let buf2 = vec![0.0f32; max_n * 16];
        let output_n = layers.last().unwrap().n;
        let output = vec![0.0f32; 16 * output_n];

        // Build MonolithicLayerConfig list
        let mut prev_k = input_k;
        let mut configs = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            configs.push(MonolithicLayerConfig {
                m: 16,
                n: layer.n,
                k: prev_k,
                act: layer.activation,
                w_ptr: layer_data[i].0.as_ptr() as u64,
                b_ptr: layer_data[i].1.as_ptr() as u64,
            });
            prev_k = layer.n;
        }

        let page = build_monolithic_inference_page(
            &configs,
            buf1.as_ptr() as u64,
            buf2.as_ptr() as u64,
        ).ok_or(SmeError::PageAllocFailed)?;

        Ok(Self {
            page,
            input_k,
            output_n,
            buf1,
            buf2,
            output,
            _layer_data: layer_data,
        })
    }

    /// Run inference with column-major input `[K × 16]`.
    ///
    /// Returns a reference to the output buffer `[16 × N_last]` (row-major).
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.input_k * 16,
            "input must be {} elements (K={} × 16), got {}",
            self.input_k * 16, self.input_k, input.len());
        assert_eq!(output.len(), 16 * self.output_n,
            "output must be {} elements (16 × N={}), got {}",
            16 * self.output_n, self.output_n, output.len());

        unsafe {
            self.page.call_with_args(
                input.as_ptr() as u64,
                output.as_mut_ptr() as u64,
            );
        }
    }

    /// Run inference with row-major input `[16 × K]` — transposes internally.
    pub fn run_row_major(&mut self, input_row_major: &[f32], output: &mut [f32]) {
        assert_eq!(input_row_major.len(), 16 * self.input_k);
        let mut input_col = vec![0.0f32; self.input_k * 16];
        for i in 0..16 {
            for k in 0..self.input_k {
                input_col[k * 16 + i] = input_row_major[i * self.input_k + k];
            }
        }
        self.run(&input_col, output);
    }

    /// Returns (input_k, output_n) dimensions.
    pub fn dims(&self) -> (usize, usize) { (self.input_k, self.output_n) }
}
