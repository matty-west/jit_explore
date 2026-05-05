//! Accelerate.framework FFI bindings.
//!
//! Used by [`crate::probe`] (hardware wake-up) and the benchmark suite
//! (`cblas_sgemm` baseline). Higher-level differential testing utilities
//! were retired alongside the Gate 18–22 inference paths they served.

use std::os::raw::{c_float, c_int};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CblasOrder {
    RowMajor = 101,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CblasTranspose {
    NoTrans = 111,
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

pub struct Accelerate;

impl Accelerate {
    pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
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
    }
}
