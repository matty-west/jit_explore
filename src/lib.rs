//! # sme-jit-core
//!
//! A zero-overhead Rust JIT harness for ARM SME on Apple Silicon M4.
//!
//! **Start here**: [`api`] module provides ergonomic wrappers:
//! - [`SmeGemm`] — JIT-compiled tiled SGEMM kernel (5× faster than Accelerate at 16×16)
//! - [`SmeMlp`] — Fused multi-layer inference (1.55× faster than Accelerate)
//!
//! **Advanced**: The lower-level modules ([`emitter`], [`jit_page`], [`probe`]) are
//! public for power users who need direct control over instruction emission.
//!
//! ## Requirements
//! - Apple Silicon M4 (M1–M3 lack SME)
//! - macOS Sequoia 15+
//! - Rust nightly

// ── Public API (start here) ─────────────────────────────────────────────────
pub mod api;
pub use api::{SmeGemm, SmeMlp, LayerConfig, Activation, SmeError};

// ── Internal modules (public for advanced usage and benchmarks) ─────────────
pub mod cpu_state;
pub mod crucible;
pub mod emitter;
pub mod inference;
pub mod jit_page;
pub mod probe;
pub mod signal_handler;
pub mod sink;
pub mod weights;
