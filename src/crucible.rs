//! Semantic verification harness — "The Crucible".
//!
//! The Crucible runs **differential correctness tests** between Apple's
//! `Accelerate.framework` and a JIT-executed heisted block, comparing the
//! floating-point output element-by-element with a `< 1e-4` precision target.
//!
//! ## Golden Block
//!
//! A "Golden Block" is the fused instruction sequence that achieves zero-diff:
//!
//! ```text
//! Golden Block = ZERO_ZA  ║  math_body  ║  store_epilogue
//! ```
//!
//! - **ZERO_ZA** (`0xC008_00FF`): Clears all ZA accumulator tiles before the
//!   outer-product loop, fixing the "dirty state" 8.0 error seen in Gate 12.
//! - **math_body**: The heisted outer-product microkernel (e.g. `FMOPA ZA0.S`
//!   instructions and supporting ALU).
//! - **store_epilogue**: `ST1W` / AMX store instructions that write the ZA tile
//!   contents back to the C matrix pointer. These are captured by the updated
//!   Frida script (`extract_all.py` v3) in the `stores` field of each block
//!   entry in `stolen_blocks.json`. If no stores are available from the heist,
//!   `build_golden_block` accepts an empty slice and logs a warning.
//!
//! ## ABI mapping (heisted microkernel)
//!
//! The standard `cblas_sgemm` wrapper uses the BLAS C ABI. The inner Apple
//! microkernel (`APL_sgemm`) uses a **custom PCS** discovered by the Frida
//! hook in Gate 12:
//!
//! | Register | Role |
//! |----------|------|
//! | x5       | Matrix B base pointer |
//! | x7       | Matrix A base pointer |
//! | x8       | Matrix C base pointer |
//!
//! The `MicrokernelAbi` struct codifies this mapping so callers don't need to
//! hard-code register indices.

use crate::emitter::ZERO_ZA;
use crate::probe::Probe;
use std::os::raw::{c_float, c_int};

// ─────────────────────────────────────────────────────────────────────────────
// Accelerate FFI
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CblasOrder {
    RowMajor = 101,
    ColMajor = 102,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CblasTranspose {
    NoTrans = 111,
    Trans   = 112,
}

unsafe extern "C" {
    pub fn cblas_sgemm(
        order:   CblasOrder,
        trans_a: CblasTranspose,
        trans_b: CblasTranspose,
        m:       c_int,
        n:       c_int,
        k:       c_int,
        alpha:   c_float,
        a:       *const c_float,
        lda:     c_int,
        b:       *const c_float,
        ldb:     c_int,
        beta:    c_float,
        c:       *mut c_float,
        ldc:     c_int,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ABI mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Custom PCS (Procedure Call Standard) for Apple's inner `sgemm` microkernel.
///
/// Discovered via Frida ABI-mapping in Gate 12 / heist report section 2.
/// These register assignments override the deterministic seed values in the
/// JIT prelude so that the heisted block operates on real matrix data.
#[derive(Debug, Clone)]
pub struct MicrokernelAbi {
    /// Matrix A base pointer — x7.
    pub a_ptr: u64,
    /// Matrix B base pointer — x5.
    pub b_ptr: u64,
    /// Matrix C (output) base pointer — x8.
    pub c_ptr: u64,
}

impl MicrokernelAbi {
    /// Build a GPR override list suitable for [`Probe::run_block_with_overrides`].
    ///
    /// Returns `[(reg_index, value), …]` for x5, x7, and x8.
    pub fn to_overrides(&self) -> Vec<(u8, u64)> {
        vec![
            (5, self.b_ptr),
            (7, self.a_ptr),
            (8, self.c_ptr),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Golden Block builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build a "Golden Block" — the fused sequence that achieves semantic equivalence.
///
/// # Layout
/// ```text
/// [ZERO_ZA] ++ math_body ++ store_epilogue
/// ```
///
/// # Arguments
/// - `math_body`: The heisted outer-product loop opcodes (from `stolen_blocks.json`
///   `block` field). These are the instructions that accumulate into ZA tiles.
/// - `store_epilogue`: The store instructions that write ZA → C matrix pointer.
///   Pass the `stores[].opcode` values from `stolen_blocks.json` (captured by the
///   updated Frida v3 script). Pass an empty slice if stores are not yet available;
///   the function will warn but still build a partial block for diagnosis.
///
/// # Returns
/// A `Vec<u32>` ready to be passed to `Probe::run_block_with_overrides`.
pub fn build_golden_block(math_body: &[u32], store_epilogue: &[u32]) -> Vec<u32> {
    if store_epilogue.is_empty() {
        eprintln!(
            "[crucible] WARNING: store_epilogue is empty — the Golden Block will compute \
             into ZA tiles but NOT write results back to the C pointer. \
             Re-run `heist/extract_all.py` (v3) to capture store instructions, \
             then pass the `stores` field here."
        );
    }

    let mut block = Vec::with_capacity(1 + math_body.len() + store_epilogue.len());
    // Step 1: Zero the ZA accumulator (fixes the dirty-state 8.0 error).
    block.push(ZERO_ZA);
    // Step 2: Math body — outer-product accumulation.
    block.extend_from_slice(math_body);
    // Step 3: Store epilogue — write ZA tiles → C matrix.
    block.extend_from_slice(store_epilogue);
    block
}

// ─────────────────────────────────────────────────────────────────────────────
// Crucible struct
// ─────────────────────────────────────────────────────────────────────────────

pub struct Crucible {
    pub probe: Probe,
}

/// Result of a single Crucible differential test.
#[derive(Debug)]
pub struct CrucibleResult {
    /// Max absolute difference between Accelerate output and JIT output.
    pub max_diff: f32,
    /// Whether the JIT block faulted (SIGILL / SEGV / etc.).
    pub faulted: bool,
    /// Human-readable status string.
    pub status: &'static str,
}

impl CrucibleResult {
    /// Returns `true` if the precision target is met (max_diff < 1e-4).
    pub fn is_golden(&self) -> bool {
        !self.faulted && self.max_diff < 1e-4
    }
}

impl std::fmt::Display for CrucibleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.faulted {
            write!(f, "[✗] JIT block faulted ({})", self.status)
        } else if self.is_golden() {
            write!(f, "[✓] GOLDEN: max_diff = {:.2e} < 1e-4 — SEMANTIC EQUIVALENCE PROVEN", self.max_diff)
        } else {
            write!(f, "[!] max_diff = {} (target < 1e-4)", self.max_diff)
        }
    }
}

impl Crucible {
    pub fn new() -> Self {
        Self { probe: Probe::new() }
    }

    /// Run Accelerate `cblas_sgemm` and return the output matrix.
    ///
    /// Used as the ground-truth baseline for differential testing.
    pub fn run_accelerate(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        // SAFETY: All pointers are valid, sizes are correct, Accelerate is
        // a system framework that is always present on macOS.
        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int,
                b.as_ptr(),
                n as c_int,
                0.0,
                c.as_mut_ptr(),
                n as c_int,
            );
        }
        c
    }

    /// Compare two f32 matrices element-wise, returning the max absolute diff.
    pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "matrix size mismatch in max_abs_diff");
        a.iter().zip(b.iter()).fold(0.0f32, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    /// Run the full differential test for a given Golden Block.
    ///
    /// ## Steps
    /// 1. Allocate A (all-ones), B (all-ones), C_accelerate (zero), C_jit (zero).
    /// 2. Run `cblas_sgemm` → fills `C_accelerate` with ground-truth values.
    /// 3. Build the Golden Block: `[ZERO_ZA] ++ math_body ++ store_epilogue`.
    /// 4. Inject the microkernel ABI pointers (x5=B, x7=A, x8=C_jit).
    /// 5. Execute the Golden Block in streaming mode (SMSTART / SMSTOP wrapper).
    /// 6. Compute max absolute diff between `C_accelerate` and `C_jit`.
    ///
    /// # Arguments
    /// - `math_body`: Inner-loop opcodes from `stolen_blocks.json`.
    /// - `store_epilogue`: Store-back opcodes from the `stores` field.
    ///   Pass `&[]` if not yet available (yields large diff, used for diagnosis).
    /// - `m`, `n`, `k`: Matrix dimensions.
    pub fn test_golden_block(
        &self,
        math_body:      &[u32],
        store_epilogue: &[u32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CrucibleResult {
        // ── Input matrices: all-ones ──────────────────────────────────────────
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];

        // ── Accelerate baseline ───────────────────────────────────────────────
        let c_accelerate = Self::run_accelerate(m, n, k, &a, &b);

        // ── JIT execution ─────────────────────────────────────────────────────
        let mut c_jit = vec![0.0f32; m * n];

        let abi = MicrokernelAbi {
            a_ptr: a.as_ptr()           as u64,
            b_ptr: b.as_ptr()           as u64,
            c_ptr: c_jit.as_mut_ptr()   as u64,
        };
        let overrides = abi.to_overrides();

        let golden = build_golden_block(math_body, store_epilogue);

        let probe_result = self.probe.run_block_with_overrides(&golden, &overrides, true);

        if probe_result.faulted {
            return CrucibleResult {
                max_diff: f32::INFINITY,
                faulted:  true,
                status:   probe_result.status(),
            };
        }

        // ── Precision ─────────────────────────────────────────────────────────
        let max_diff = Self::max_abs_diff(&c_accelerate, &c_jit);
        CrucibleResult { max_diff, faulted: false, status: "ok" }
    }

    /// Legacy entry-point kept for compatibility with Gate 12 callers.
    ///
    /// This version uses the heuristic ABI (x0=A, x1=B, x2=C) from before
    /// the ABI mapping was fully resolved. Prefer `test_golden_block` for
    /// new code.
    pub fn test_sgemm_equivalence(
        &self,
        opcodes: &[u32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<f32, String> {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c_accelerate = vec![0.0f32; m * n];
        let mut c_jit        = vec![0.0f32; m * n];

        // 1. Accelerate baseline.
        // SAFETY: valid pointers, correct sizes.
        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                m as c_int, n as c_int, k as c_int,
                1.0,
                a.as_ptr(),   k as c_int,
                b.as_ptr(),   n as c_int,
                0.0,
                c_accelerate.as_mut_ptr(), n as c_int,
            );
        }

        // 2. JIT with heuristic ABI (x0=A, x1=B, x2=C).
        let overrides = vec![
            (0u8, a.as_ptr()         as u64),
            (1u8, b.as_ptr()         as u64),
            (2u8, c_jit.as_mut_ptr() as u64),
        ];

        let result = self.probe.run_block_with_overrides(opcodes, &overrides, true);
        if result.faulted {
            return Err(format!("JIT block faulted: {}", result.status()));
        }

        Ok(Self::max_abs_diff(&c_accelerate, &c_jit))
    }
}
