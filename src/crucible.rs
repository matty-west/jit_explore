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
}
