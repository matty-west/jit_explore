//! Leaf kernel extractor — Step 3 of the heist pipeline.
//!
//! Parses `stolen_blocks.json` (produced by `heist/extract_all.py`) to find
//! **self-contained leaf sub-functions** inside large heisted blocks like
//! `APL_sgemm`.
//!
//! ## What is a "leaf kernel"?
//!
//! Large optimised functions such as `APL_sgemm` are internally structured as
//! a collection of inline sub-functions, each delimited by a `RET`. Each
//! sub-function handles a specific matrix tile size (8×K, 16×K, 32×K, …).
//! The innermost ones — **leaves** — contain the dense AMX multiply-accumulate
//! loops with:
//!
//! * **No calls** (`BL`) to external functions.
//! * **No out-of-block branches** — every `B`, `B.cond`, `CBZ`, `TBZ` target
//!   lands inside the same RET-bounded region.
//!
//! These properties mean that PC-relative branch offsets are **automatically
//! preserved** when the leaf is copied linearly into a JIT page — no branch
//! relocation engine is required.
//!
//! ## PC-relative hazards (ADRP / ADR)
//!
//! A leaf may contain `ADRP` or `ADR` instructions that compute page-relative
//! or PC-relative addresses. These break when relocated to a JIT page because
//! the absolute address they compute changes. The extractor records their
//! positions in [`LeafKernel::adrp_indices`] / [`LeafKernel::adr_indices`] so
//! callers can NOP or patch them with [`crate::emitter::nop_pc_relative_hazards`].
//!
//! ## Literal pools
//!
//! Some leaves are preceded by a small constant table (jump-offset data in the
//! `0x0000xxxx` range). These are detected heuristically and their length is
//! reported as [`LeafKernel::literal_pool_len`].  The pool words are **kept**
//! in [`LeafKernel::opcodes`] so that intra-leaf PC-relative offsets remain
//! intact — the surrounding branch structure ensures the CPU never executes the
//! pool as instructions.

use std::path::Path;

use serde::Deserialize;

// ─────────────────────────────────────────────────────────────────────────────
// JSON wire types  (mirroring stolen_blocks.json schema)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct JsonStoreEntry {
    pub offset:     u64,
    pub opcode:     String,
    #[serde(rename = "type")]
    pub store_type: String,
}

#[derive(Deserialize, Debug)]
struct JsonBlock {
    pub name:    String,
    pub address: String,
    pub block:   Vec<String>,
    #[serde(default)]
    pub stores:  Vec<JsonStoreEntry>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single store instruction captured by the Frida heist.
///
/// Store instructions write the AMX accumulator tiles back to the C-matrix
/// pointer and are required to complete the Golden Block.
#[derive(Debug, Clone)]
pub struct StoreInsn {
    /// Byte offset from the start of the heisted function.
    pub offset:     u64,
    /// Raw opcode.
    pub opcode:     u32,
    /// Heist-assigned type tag (`"amx_store"` or `"sme_store"`).
    pub store_type: String,
}

/// A self-contained leaf sub-function extracted from a heisted block.
///
/// All PC-relative branches within the leaf target other instructions inside
/// the leaf — no out-of-block jumps — so this slice can be copied linearly
/// into a JIT page without any branch relocation.
#[derive(Debug, Clone)]
pub struct LeafKernel {
    /// Name of the heisted function this leaf came from (e.g. `"APL_sgemm"`).
    pub source_name:      String,
    /// Original base address of the source function (from Frida capture).
    pub source_base:      u64,
    /// Byte offset within the source function where this leaf starts.
    pub leaf_byte_offset: u64,
    /// Instruction words of the leaf (including any leading literal pool).
    pub opcodes:          Vec<u32>,
    /// Number of Apple AMX instructions detected (`0x0020_xxxx` prefix).
    pub amx_count:        usize,
    /// Indices into `opcodes` where `ADRP` instructions live.  These compute
    /// page-relative addresses and will produce wrong values after relocation.
    pub adrp_indices:     Vec<usize>,
    /// Indices into `opcodes` where `ADR` instructions live.
    pub adr_indices:      Vec<usize>,
    /// How many leading words look like literal-pool data (not instructions).
    /// The executable entry point is `opcodes[literal_pool_len..]`.
    pub literal_pool_len: usize,
    /// Store instructions from the parent block's `stores` field (if any).
    /// These may be needed to write the AMX accumulator result back to memory.
    pub stores:           Vec<StoreInsn>,
}

impl LeafKernel {
    /// Returns the instruction index where executable code begins, i.e. the
    /// first instruction after any leading literal pool.
    #[inline]
    pub fn code_start(&self) -> usize {
        self.literal_pool_len
    }

    /// Score used to rank leaf candidates.  Higher is better.
    ///
    /// Rewards AMX density; penalises PC-relative hazards.
    pub fn score(&self) -> i64 {
        let amx      = self.amx_count as i64;
        let hazards  = (self.adrp_indices.len() + self.adr_indices.len()) as i64;
        amx * 10 - hazards * 50
    }

    /// Pretty-print a one-line summary suitable for gate output.
    pub fn summary(&self) -> String {
        format!(
            "{} +0x{:06x}: {} insns  ({} AMX, {} branches)  \
             literal_pool={}  ADRP/ADR hazards={}  score={}",
            self.source_name,
            self.leaf_byte_offset,
            self.opcodes.len(),
            self.amx_count,
            self.opcodes.iter().filter(|&&op| decode_branch(op).is_some()).count(),
            self.literal_pool_len,
            self.adrp_indices.len() + self.adr_indices.len(),
            self.score(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch / instruction decoders
// ─────────────────────────────────────────────────────────────────────────────

/// PC-relative branch delta in bytes, or `None` if the opcode is not a branch.
///
/// Handles: `B`, `BL`, `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ`.
pub(crate) fn decode_branch(op: u32) -> Option<i64> {
    // B   [31:26] = 000101
    if (op >> 26) == 0b000101 {
        return Some(sign_extend26(op & 0x3FF_FFFF) as i64 * 4);
    }
    // BL  [31:26] = 100101
    if (op >> 26) == 0b100101 {
        return Some(sign_extend26(op & 0x3FF_FFFF) as i64 * 4);
    }
    // B.cond  [31:24] = 0x54
    if (op >> 24) == 0x54 {
        return Some(sign_extend19((op >> 5) & 0x7_FFFF) as i64 * 4);
    }
    // CBZ/CBNZ  [31:24] & 0xFE = 0x34
    if (op >> 24) & 0xFE == 0x34 {
        return Some(sign_extend19((op >> 5) & 0x7_FFFF) as i64 * 4);
    }
    // TBZ/TBNZ  [31:24] & 0xFE = 0x36
    if (op >> 24) & 0xFE == 0x36 {
        return Some(sign_extend14((op >> 5) & 0x3FFF) as i64 * 4);
    }
    None
}

#[inline]
fn sign_extend26(raw: u32) -> i32 {
    let raw = (raw & 0x3FF_FFFF) as i32;
    if raw & 0x200_0000 != 0 { raw - 0x400_0000 } else { raw }
}
#[inline]
fn sign_extend19(raw: u32) -> i32 {
    let raw = (raw & 0x7_FFFF) as i32;
    if raw & 0x4_0000 != 0 { raw - 0x8_0000 } else { raw }
}
#[inline]
fn sign_extend14(raw: u32) -> i32 {
    let raw = (raw & 0x3FFF) as i32;
    if raw & 0x2000 != 0 { raw - 0x4000 } else { raw }
}

pub(crate) fn is_ret(op: u32)  -> bool { (op & 0xFFFF_FC1F) == 0xD65F_0000 }
pub(crate) fn is_amx(op: u32)  -> bool { (op & 0xFFF0_0000) == 0x0020_0000 }
pub(crate) fn is_adrp(op: u32) -> bool { (op >> 24) & 0x9F   == 0x90 }
pub(crate) fn is_adr(op: u32)  -> bool { (op >> 24) & 0x9F   == 0x10 }

/// Returns `true` if the word value looks like literal-pool data rather than
/// a valid AArch64 instruction.  Heuristic: values below 0x1000_0000 that are
/// not AMX ops fall in the `UDF` / permanently-undefined range.
fn looks_like_data(op: u32) -> bool {
    op < 0x1000_0000 && !is_amx(op)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Determine whether all branches in `opcodes` target addresses within the
/// slice, and return the number of branches found.
///
/// A branch at instruction index `i` with delta `d` is "self-contained" if
/// `0 ≤ (i*4 + d) < len*4`.
fn all_branches_self_contained(opcodes: &[u32]) -> (bool, usize) {
    let len_bytes = (opcodes.len() * 4) as i64;
    let mut all_ok   = true;
    let mut n_branches = 0usize;

    for (i, &op) in opcodes.iter().enumerate() {
        if let Some(delta) = decode_branch(op) {
            n_branches += 1;
            let target = (i as i64) * 4 + delta;
            if target < 0 || target >= len_bytes {
                all_ok = false;
                // Don't break early — we want the full count.
            }
        }
    }
    (all_ok, n_branches)
}

/// Split `opcodes` into RET-delimited regions, returning
/// `(start_index, end_index_inclusive)` pairs.
fn ret_boundaries(opcodes: &[u32]) -> Vec<(usize, usize)> {
    let mut out   = Vec::new();
    let mut start = 0usize;
    for (i, &op) in opcodes.iter().enumerate() {
        if is_ret(op) {
            out.push((start, i));
            start = i + 1;
        }
    }
    if start < opcodes.len() {
        out.push((start, opcodes.len() - 1));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `stolen_blocks.json` at `path`.
pub fn load_stolen_blocks_json(path: &Path) -> Result<Vec<serde_json::Value>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("cannot parse {}: {e}", path.display()))
}

/// Find and score all self-contained leaf kernels in a flat opcode slice.
///
/// Only regions where **every branch is self-contained** and that contain
/// at least one AMX instruction are returned.  Results are sorted best-first.
pub fn find_leaves(
    source_name: &str,
    source_base: u64,
    all_opcodes:  &[u32],
    stores:       &[StoreInsn],
) -> Vec<LeafKernel> {
    let mut leaves = Vec::new();

    for (start, end) in ret_boundaries(all_opcodes) {
        let region = &all_opcodes[start..=end];
        if region.len() < 16 {
            continue;
        }

        let (self_contained, _n) = all_branches_self_contained(region);
        if !self_contained {
            continue;
        }

        let amx_count = region.iter().filter(|&&op| is_amx(op)).count();
        if amx_count == 0 {
            continue;
        }

        let adrp_indices: Vec<usize> = region.iter().enumerate()
            .filter(|&(_, op)| is_adrp(*op))
            .map(|(i, _)| i)
            .collect();

        let adr_indices: Vec<usize> = region.iter().enumerate()
            .filter(|&(_, op)| is_adr(*op))
            .map(|(i, _)| i)
            .collect();

        let literal_pool_len = region.iter()
            .take_while(|&&op| looks_like_data(op))
            .count();

        leaves.push(LeafKernel {
            source_name:      source_name.to_string(),
            source_base,
            leaf_byte_offset: (start * 4) as u64,
            opcodes:          region.to_vec(),
            amx_count,
            adrp_indices,
            adr_indices,
            literal_pool_len,
            stores:           stores.to_vec(),
        });
    }

    leaves.sort_by(|a, b| b.score().cmp(&a.score()));
    leaves
}

/// High-level entry point: find the single best leaf kernel in
/// `stolen_blocks.json`.
///
/// Block selection priority:
/// 1. `APL_sgemm` — the real hardware math kernel.
/// 2. Any block with non-empty stores and at least one self-contained leaf.
/// 3. Any block with any self-contained leaf.
///
/// Returns `None` if no suitable leaf is found.
pub fn best_leaf_from_file(path: &Path) -> Option<LeafKernel> {
    let text   = std::fs::read_to_string(path).ok()?;
    let blocks: Vec<JsonBlock> = serde_json::from_str(&text).ok()?;

    // Decode JSON block → (opcodes, stores, name, base).
    let decode = |b: &JsonBlock| -> Option<(Vec<u32>, Vec<StoreInsn>)> {
        let opcodes: Vec<u32> = b.block.iter()
            .filter_map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .collect();
        if opcodes.is_empty() { return None; }
        let stores: Vec<StoreInsn> = b.stores.iter()
            .filter_map(|s| {
                let op = u32::from_str_radix(s.opcode.trim_start_matches("0x"), 16).ok()?;
                Some(StoreInsn { offset: s.offset, opcode: op, store_type: s.store_type.clone() })
            })
            .collect();
        Some((opcodes, stores))
    };

    let base_of = |b: &JsonBlock| -> u64 {
        u64::from_str_radix(b.address.trim_start_matches("0x"), 16).unwrap_or(0)
    };

    // 1. APL_sgemm preferred.
    if let Some(b) = blocks.iter().find(|b| b.name == "APL_sgemm") {
        if let Some((opcodes, stores)) = decode(b) {
            let leaves = find_leaves(&b.name, base_of(b), &opcodes, &stores);
            if let Some(leaf) = leaves.into_iter().next() {
                return Some(leaf);
            }
        }
    }

    // 2. Any block with stores.
    for b in blocks.iter().filter(|b| !b.stores.is_empty()) {
        if let Some((opcodes, stores)) = decode(b) {
            let leaves = find_leaves(&b.name, base_of(b), &opcodes, &stores);
            if let Some(leaf) = leaves.into_iter().next() {
                return Some(leaf);
            }
        }
    }

    // 3. Anything at all.
    for b in &blocks {
        if let Some((opcodes, stores)) = decode(b) {
            let leaves = find_leaves(&b.name, base_of(b), &opcodes, &stores);
            if let Some(leaf) = leaves.into_iter().next() {
                return Some(leaf);
            }
        }
    }

    None
}

/// Return every self-contained leaf from `APL_sgemm` (all candidates, ranked).
///
/// Useful for diagnostics: prints all leaves so we can pick the right one.
pub fn all_leaves_from_file(path: &Path) -> Vec<LeafKernel> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let blocks: Vec<JsonBlock> = match serde_json::from_str(&text) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };

    let base_of = |b: &JsonBlock| -> u64 {
        u64::from_str_radix(b.address.trim_start_matches("0x"), 16).unwrap_or(0)
    };

    for b in &blocks {
        if b.name != "APL_sgemm" { continue; }
        let opcodes: Vec<u32> = b.block.iter()
            .filter_map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .collect();
        let stores: Vec<StoreInsn> = b.stores.iter()
            .filter_map(|s| {
                let op = u32::from_str_radix(s.opcode.trim_start_matches("0x"), 16).ok()?;
                Some(StoreInsn { offset: s.offset, opcode: op, store_type: s.store_type.clone() })
            })
            .collect();
        return find_leaves(&b.name, base_of(b), &opcodes, &stores);
    }
    Vec::new()
}
