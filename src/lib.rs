//! `sme-jit-core` — bare-metal ARM SME JIT harness for Apple Silicon M4.
//!
//! Re-exports the harness modules so that `benches/` and integration tests
//! can import them without duplicating the `main.rs` binary.
//!
//! All modules carry the same `#![deny(unsafe_op_in_unsafe_fn)]` posture
//! as the binary; the lint is set at the module level inside each file.

pub mod cpu_state;
pub mod crucible;
pub mod emitter;
pub mod inference;
pub mod jit_page;
pub mod probe;
pub mod signal_handler;
pub mod sink;
pub mod weights;
