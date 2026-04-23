//! Semantic verification harness — "The Crucible".
//!
//! Runs **differential correctness tests** between Apple's `Accelerate.framework`
//! and JIT-emitted SME kernels, comparing floating-point output element-by-element
//! with a `< 1e-4` precision target.

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
// Crucible
// ─────────────────────────────────────────────────────────────────────────────

pub struct Crucible {
    pub probe: Probe,
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

    /// Convert a slice of f32 to BF16 (top 16 bits of FP32).
    pub fn f32_to_bf16(data: &[f32]) -> Vec<u16> {
        data.iter()
            .map(|&f| {
                let bits = f.to_bits();
                // Simple truncation for BF16 (bfloat16)
                (bits >> 16) as u16
            })
            .collect()
    }

    /// Convert a slice of BF16 to f32.
    pub fn bf16_to_f32(data: &[u16]) -> Vec<f32> {
        data.iter()
            .map(|&h| {
                let bits = (h as u32) << 16;
                f32::from_bits(bits)
            })
            .collect()
    }

    /// Reference matmul for BF16 (converted to f32 internally).
    pub fn ref_bf16_matmul(m: usize, n: usize, k: usize, a_bf16: &[u16], b_bf16: &[u16]) -> Vec<f32> {
        let a_f32 = Self::bf16_to_f32(a_bf16);
        let b_f32 = Self::bf16_to_f32(b_bf16);
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_f32[i * k + l] * b_f32[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Reference matmul for INT8 → INT32.
    pub fn ref_int8_matmul(m: usize, n: usize, k: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i32;
                for l in 0..k {
                    sum += (a[i * k + l] as i32) * (b[l * n + j] as i32);
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Reference matmul + activation (ReLU/Bias).
    pub fn ref_sgemm_fused(
        m: usize, n: usize, k: usize,
        a: &[f32], b: &[f32], bias: Option<&[f32]>,
        relu: bool,
    ) -> Vec<f32> {
        let mut c = Self::run_accelerate(m, n, k, a, b);
        for i in 0..m {
            for j in 0..n {
                let mut val = c[i * n + j];
                if let Some(b_vec) = bias {
                    val += b_vec[j];
                }
                if relu {
                    val = val.max(0.0);
                }
                c[i * n + j] = val;
            }
        }
        c
    }
}
