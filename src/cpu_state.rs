//! Types for capturing and diffing AArch64 CPU register state.
//!
//! The [`GprSnapshot`] type holds the 31 general-purpose registers (x0–x30)
//! and supports diffing two snapshots to identify which registers changed.

use std::fmt;

// --- Constants ---

/// Number of general-purpose registers captured (x0–x30).
pub const GPR_COUNT: usize = 31;

/// Canary value written at the start of a snapshot buffer.
/// If this is clobbered after a probe, the snapshot is corrupt.
pub const CANARY_HEAD: u64 = 0xCAFE_BABE_DEAD_F00D;

/// Canary value written at the end of a snapshot buffer.
pub const CANARY_TAIL: u64 = 0xF00D_CAFE_BABE_DEAD;

/// AArch64 register names for display.
pub const GPR_NAMES: [&str; GPR_COUNT] = [
    "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
    "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30",
];

/// Memory layout for a single register dump, including canary values.
///
/// The assembly prelude/postlude writes directly into this struct via
/// a pointer. The layout is:
///
/// ```text
/// offset 0x000: canary_head  (u64)
/// offset 0x008: gprs[0]      (x0)
/// offset 0x010: gprs[1]      (x1)
/// ...
/// offset 0x0F8: gprs[30]     (x30)
/// offset 0x100: canary_tail  (u64)
/// ```
///
/// Total size: 8 + (31 × 8) + 8 = 264 bytes.
#[repr(C, align(16))]
pub struct SnapshotBuffer {
    /// Head canary — detect corruption from wild stores.
    pub canary_head: u64,
    /// The 31 GPRs: x0–x30.
    pub gprs: [u64; GPR_COUNT],
    /// Virtual counter timestamp (CNTVCT_EL0).
    pub timestamp: u64,
    /// Tail canary — detect corruption from wild stores.
    pub canary_tail: u64,
}

impl SnapshotBuffer {
    /// Create a new snapshot buffer with canary values pre-filled.
    pub fn new() -> Self {
        SnapshotBuffer {
            canary_head: CANARY_HEAD,
            gprs: [0u64; GPR_COUNT],
            timestamp: 0,
            canary_tail: CANARY_TAIL,
        }
    }

    /// Check whether the canary values are intact.
    ///
    /// Returns `true` if the buffer was not corrupted by the probed instruction.
    pub fn canaries_intact(&self) -> bool {
        self.canary_head == CANARY_HEAD && self.canary_tail == CANARY_TAIL
    }

    /// Extract a [`GprSnapshot`] from this buffer.
    ///
    /// Returns `None` if the canaries are corrupted.
    pub fn to_snapshot(&self) -> Option<GprSnapshot> {
        if self.canaries_intact() {
            Some(GprSnapshot { gprs: self.gprs })
        } else {
            None
        }
    }

    /// Returns the byte offset of `gprs[0]` from the start of the struct.
    ///
    /// The prelude/postlude use this to know where to write registers.
    pub const fn gprs_offset() -> usize {
        std::mem::offset_of!(SnapshotBuffer, gprs)
    }

    /// Returns the byte offset of `timestamp` from the start of the struct.
    pub const fn timestamp_offset() -> usize {
        std::mem::offset_of!(SnapshotBuffer, timestamp)
    }

    /// Returns a mutable pointer to the start of this buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        (self as *mut SnapshotBuffer) as *mut u8
    }
}

impl Default for SnapshotBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// A snapshot of the 31 general-purpose registers (x0–x30).
#[derive(Clone, PartialEq, Eq)]
pub struct GprSnapshot {
    /// Register values: `gprs[0]` = x0, ..., `gprs[30]` = x30.
    pub gprs: [u64; GPR_COUNT],
}

impl GprSnapshot {
    /// Create a snapshot with all registers zeroed.
    #[allow(dead_code)]
    pub fn zeroed() -> Self {
        GprSnapshot { gprs: [0u64; GPR_COUNT] }
    }

    /// Get the value of register `xN`.
    ///
    /// # Panics
    /// Panics if `n > 30`.
    pub fn reg(&self, n: usize) -> u64 {
        assert!(n < GPR_COUNT, "register index {n} out of range (0..=30)");
        self.gprs[n]
    }

    /// Compute the diff between `self` (pre) and `other` (post).
    ///
    /// Returns a list of `(register_index, pre_value, post_value)` for every
    /// register that changed.
    pub fn diff(&self, other: &GprSnapshot) -> Vec<RegDiff> {
        self.gprs
            .iter()
            .zip(other.gprs.iter())
            .enumerate()
            .filter(|(i, (a, b))| {
                // Filter out noisy platform registers:
                // x16, x17: IP0/IP1 (linker veneers)
                // x18: Darwin platform register (TEB pointer)
                *i != 16 && *i != 17 && *i != 18 && a != b
            })
            .map(|(i, (&pre, &post))| RegDiff { index: i, pre, post })
            .collect()
    }
}

impl fmt::Debug for GprSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GprSnapshot {{")?;
        for (i, &val) in self.gprs.iter().enumerate() {
            writeln!(f, "  {}: 0x{:016X}", GPR_NAMES[i], val)?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for GprSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nonzero: Vec<_> = self.gprs.iter().enumerate()
            .filter(|(_, v)| **v != 0)
            .collect();
        if nonzero.is_empty() {
            write!(f, "(all zero)")
        } else {
            for (i, (idx, val)) in nonzero.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}=0x{:X}", GPR_NAMES[*idx], val)?;
            }
            Ok(())
        }
    }
}

/// A single register that changed between two snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegDiff {
    /// Register index (0 = x0, ..., 30 = x30).
    pub index: usize,
    /// Value before the probed instruction.
    pub pre: u64,
    /// Value after the probed instruction.
    pub post: u64,
}

impl fmt::Display for RegDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: 0x{:016X} → 0x{:016X}",
            GPR_NAMES[self.index], self.pre, self.post
        )
    }
}

// --- Seed values ---

/// Generate a deterministic seed value for register `xN`.
///
/// The pattern `0xDEAD_0000_0000_00XX` makes it easy to identify which
/// register a value came from, and unlikely to collide with real results.
pub const fn seed_value(reg_index: u8) -> u64 {
    0xDEAD_0000_0000_0000 | (reg_index as u64)
}

/// Build a [`GprSnapshot`] containing the expected seed values.
///
/// Registers x0–x27 are seeded with [`seed_value(i)`]. x28 is the scratch
/// register (holds the buffer pointer, not seeded), x29 and x30 are the
/// caller's FP/LR (not seeded). These three are set to 0 in the snapshot
/// to indicate "don't compare."
pub fn seeded_snapshot() -> GprSnapshot {
    let mut gprs = [0u64; GPR_COUNT];
    for i in 0..28 {
        gprs[i] = seed_value(i as u8);
    }
    GprSnapshot { gprs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_buffer_layout() {
        assert_eq!(SnapshotBuffer::gprs_offset(), 8);
        // 8 (canary_head) + 31*8 (gprs) + 8 (canary_tail) = 264, rounded to 16 = 272
        let size = std::mem::size_of::<SnapshotBuffer>();
        assert!(size >= 264, "SnapshotBuffer too small: {size}");
    }

    #[test]
    fn canary_detection() {
        let mut buf = SnapshotBuffer::new();
        assert!(buf.canaries_intact());

        buf.canary_head = 0;
        assert!(!buf.canaries_intact());
        assert!(buf.to_snapshot().is_none());

        buf.canary_head = CANARY_HEAD;
        buf.canary_tail = 0;
        assert!(!buf.canaries_intact());
    }

    #[test]
    fn diff_detects_changes() {
        let pre = GprSnapshot { gprs: [0u64; GPR_COUNT] };
        let mut post = pre.clone();
        post.gprs[5] = 0x2A;
        post.gprs[12] = 0xFF;

        let diffs = pre.diff(&post);
        assert_eq!(diffs.len(), 2);
        assert_eq!(diffs[0].index, 5);
        assert_eq!(diffs[0].post, 0x2A);
        assert_eq!(diffs[1].index, 12);
        assert_eq!(diffs[1].post, 0xFF);
    }

    #[test]
    fn diff_identical_is_empty() {
        let a = GprSnapshot { gprs: [42u64; GPR_COUNT] };
        assert!(a.diff(&a).is_empty());
    }

    #[test]
    fn seed_values_are_distinct() {
        for i in 0..GPR_COUNT {
            let v = seed_value(i as u8);
            assert_eq!(v & 0xFF, i as u64);
            assert_eq!(v >> 48, 0xDEAD);
        }
    }

    #[test]
    fn display_reg_diff() {
        let d = RegDiff {
            index: 5,
            pre: 0xDEAD_0000_0000_0005,
            post: 0x0000_0000_0000_002A,
        };
        let s = format!("{d}");
        assert!(s.contains("x5"));
        assert!(s.contains("DEAD"));
        assert!(s.contains("002A"));
    }
}
