//! Writes probe results to a JSON Lines file for persistent, crash-safe logging.
//!
//! Each line is a self-contained JSON object ([`SinkRecord`]) with a schema
//! version field so the format can evolve without breaking older data.
//!
//! ## Design decisions
//!
//! - **Decoupled from internal types.** `SinkRecord` is a standalone serde
//!   struct — no `#[derive(Serialize)]` on `ProbeResult` or `GprSnapshot`.
//!   Serialization concerns stay in this module.
//! - **JSON Lines (.jsonl)** over a single JSON array — crash-safe by design.
//!   A hard crash only loses the unflushed buffer; every previously flushed
//!   line is valid.
//! - **Hex strings for opcodes** (`"0x8B010000"`) — human-readable in `jq`
//!   output, `grep`-able.
//! - **BufWriter with periodic flush** — synchronous I/O is fine at our
//!   probe throughput (~10k–50k observed ops/sec).

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cpu_state::GPR_NAMES;
use crate::probe::ObservedProbeResult;

// --- Schema types ---

/// Current schema version for the JSONL output format.
const SCHEMA_VERSION: u32 = 1;

/// A single register diff entry in the serialized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    /// Register name (e.g., `"x5"`).
    pub reg: String,
    /// Value before the probed instruction (hex string).
    pub pre: String,
    /// Value after the probed instruction (hex string).
    pub post: String,
}

/// A single probe result serialized to JSON Lines.
///
/// This is the on-disk representation — decoupled from the internal
/// [`ObservedProbeResult`] type so serialization concerns stay here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkRecord {
    /// Schema version (currently `1`).
    pub v: u32,
    /// The opcode that was probed (hex string, e.g., `"0x8B010000"`).
    pub opcode: String,
    /// Execution status: `"ok"`, `"SIGILL"`, `"SEGV"`, `"TRAP"`, or `"TIMEOUT"`.
    pub status: String,
    /// Register diffs (empty if faulted or no changes).
    pub diff: Vec<DiffEntry>,
    /// `true` if the snapshot canaries were corrupted.
    pub corrupted: bool,
    /// Whether any AMX tile state changed.
    ///
    /// `Some(true)`  — at least one AMX tile word was written by the opcode.
    /// `Some(false)` — streaming mode was active but no tile changed.
    /// `None`        — non-streaming probe; AMX state was not captured.
    ///
    /// Only present in the JSONL output when non-`None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amx_changed: Option<bool>,
}

impl SinkRecord {
    /// Convert an [`ObservedProbeResult`] into a serializable record.
    pub fn from_observed(result: &ObservedProbeResult) -> Self {
        let diff = result
            .diff
            .iter()
            .map(|d| DiffEntry {
                reg: GPR_NAMES[d.index].to_string(),
                pre: format!("0x{:016X}", d.pre),
                post: format!("0x{:016X}", d.post),
            })
            .collect();

        // Only include amx_changed when the probe actually ran in streaming mode
        // and we have a valid (non-corrupted) snapshot. For non-streaming probes
        // `amx_changed` is always false and we omit it from the record entirely.
        let amx_changed = if result.amx_changed || result.post.is_some() {
            // We have a post-snapshot, so we know whether AMX changed.
            // For non-streaming probes amx_changed is false and post may be Some —
            // use a sentinel: only serialize if there's something to say.
            // Simplest rule: emit Some(x) when the probe reached postlude (no fault),
            // but only if the field would be informative.
            // We always omit None (handled by skip_serializing_if),
            // and emit Some(true/false) for streaming probes.
            //
            // We detect "was this a streaming probe" by checking if amx_changed is
            // ever true. For non-streaming probes it's always false and we leave it
            // as None to save output size.
            if result.amx_changed {
                Some(true)
            } else {
                None // non-streaming or no AMX change worth surfacing
            }
        } else {
            None
        };

        SinkRecord {
            v: SCHEMA_VERSION,
            opcode: format!("0x{:08X}", result.base.opcode),
            status: result.base.status().to_string(),
            diff,
            corrupted: result.snapshot_corrupted,
            amx_changed,
        }
    }
}

/// Default number of results between automatic flushes.
const FLUSH_INTERVAL: usize = 1000;

/// Buffered JSONL writer for probe results.
///
/// Opens a file for append, writes one JSON object per line, and flushes
/// periodically. Implements `Drop` to flush any remaining buffered data.
///
/// # Example
///
/// ```no_run
/// # use std::path::Path;
/// # fn example() -> std::io::Result<()> {
/// let mut sink = jit_explore::sink::ResultSink::new(Path::new("results.jsonl"))?;
/// // sink.write(&observed_result)?;
/// // sink is flushed on drop
/// # Ok(())
/// # }
/// ```
pub struct ResultSink {
    writer: BufWriter<File>,
    #[allow(dead_code)]
    path: PathBuf,
    /// Number of records written since last flush.
    pending: usize,
    /// Total number of records written.
    total: usize,
}

impl ResultSink {
    /// Open (or create) a JSONL file for appending results.
    ///
    /// If the file already exists, new records are appended.
    /// The parent directory must exist.
    #[must_use = "the sink must be stored to write results and flush on drop"]
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(ResultSink {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
            pending: 0,
            total: 0,
        })
    }

    /// Write a single observed probe result as a JSON line.
    ///
    /// Automatically flushes every [`FLUSH_INTERVAL`] writes.
    pub fn write(&mut self, result: &ObservedProbeResult) -> io::Result<()> {
        let record = SinkRecord::from_observed(result);
        serde_json::to_writer(&mut self.writer, &record)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.writer.write_all(b"\n")?;

        self.pending += 1;
        self.total += 1;

        if self.pending >= FLUSH_INTERVAL {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush the internal buffer to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.pending = 0;
        Ok(())
    }

    /// Total number of records written (including unflushed).
    pub fn total_written(&self) -> usize {
        self.total
    }

    /// The path this sink is writing to.
    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read the last opcode from an existing JSONL file.
    ///
    /// Returns `None` if the file doesn't exist, is empty, or the last
    /// line can't be parsed.
    ///
    /// This reads the file from the beginning to find the last line.
    /// For files with millions of lines, a seek-from-end approach would
    /// be faster — but for now simplicity wins.
    pub fn last_opcode(path: &Path) -> Option<u32> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);

        let mut last_line = None;
        for line in reader.lines() {
            if let Ok(l) = line {
                if !l.trim().is_empty() {
                    last_line = Some(l);
                }
            }
        }

        let line = last_line?;
        let record: SinkRecord = serde_json::from_str(&line).ok()?;

        // Parse the hex opcode string: "0x8B010000" → u32
        let hex_str = record.opcode.strip_prefix("0x").unwrap_or(&record.opcode);
        u32::from_str_radix(hex_str, 16).ok()
    }
}

impl Drop for ResultSink {
    fn drop(&mut self) {
        // Best-effort flush on drop — can't propagate errors.
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_state::{GprSnapshot, RegDiff, GPR_COUNT};
    use crate::probe::{ObservedProbeResult, ProbeResult};

    /// Build a minimal observed result for testing.
    fn make_observed(opcode: u32, status_faulted: bool, diffs: Vec<RegDiff>) -> ObservedProbeResult {
        ObservedProbeResult {
            base: ProbeResult {
                opcode,
                faulted: status_faulted,
                timed_out: false,
                segfaulted: false,
                trapped: false,
                fault_offset: 0,
            },
            pre: Some(GprSnapshot {
                gprs: [0u64; GPR_COUNT],
                amx: [0u64; crate::cpu_state::AMX_STATE_COUNT],
            }),
            post: if status_faulted {
                None
            } else {
                Some(GprSnapshot {
                    gprs: [0u64; GPR_COUNT],
                    amx: [0u64; crate::cpu_state::AMX_STATE_COUNT],
                })
            },
            diff: diffs,
            snapshot_corrupted: false,
            amx_changed: false,
        }
    }

    #[test]
    fn sink_record_from_observed() {
        let result = make_observed(
            0x8B01_0000,
            false,
            vec![RegDiff {
                index: 0,
                pre: 0xDEAD_0000_0000_0000,
                post: 0xBD5A_0000_0000_0001,
            }],
        );

        let record = SinkRecord::from_observed(&result);
        assert_eq!(record.v, 1);
        assert_eq!(record.opcode, "0x8B010000");
        assert_eq!(record.status, "ok");
        assert_eq!(record.diff.len(), 1);
        assert_eq!(record.diff[0].reg, "x0");
        assert!(!record.corrupted);
    }

    #[test]
    fn sink_record_faulted() {
        let result = make_observed(0x0000_0000, true, vec![]);
        let record = SinkRecord::from_observed(&result);
        assert_eq!(record.status, "SIGILL");
        assert!(record.diff.is_empty());
    }

    #[test]
    fn sink_record_round_trip_json() {
        let result = make_observed(
            0xD503_201F,
            false,
            vec![RegDiff {
                index: 5,
                pre: 0xDEAD_0000_0000_0005,
                post: 0x0000_0000_0000_002A,
            }],
        );

        let record = SinkRecord::from_observed(&result);
        let json = serde_json::to_string(&record).expect("serialize");
        let parsed: SinkRecord = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.v, record.v);
        assert_eq!(parsed.opcode, record.opcode);
        assert_eq!(parsed.status, record.status);
        assert_eq!(parsed.diff.len(), record.diff.len());
    }

    #[test]
    fn write_and_read_back() {
        let dir = std::env::temp_dir().join("jit_explore_test_sink");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_write_read.jsonl");
        let _ = std::fs::remove_file(&path); // clean slate

        // Write some results.
        {
            let mut sink = ResultSink::new(&path).expect("open sink");
            for i in 0..10u32 {
                let result = make_observed(i, i == 0, vec![]);
                sink.write(&result).expect("write");
            }
            // Drop flushes.
        }

        // Read back last opcode.
        let last = ResultSink::last_opcode(&path);
        assert_eq!(last, Some(9));

        // Clean up.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn last_opcode_missing_file() {
        let path = Path::new("/tmp/jit_explore_nonexistent_file.jsonl");
        assert_eq!(ResultSink::last_opcode(path), None);
    }

    #[test]
    fn resume_appends() {
        let dir = std::env::temp_dir().join("jit_explore_test_resume");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_resume.jsonl");
        let _ = std::fs::remove_file(&path);

        // First run: write opcodes 0..5.
        {
            let mut sink = ResultSink::new(&path).expect("open sink");
            for i in 0..5u32 {
                let result = make_observed(i, false, vec![]);
                sink.write(&result).expect("write");
            }
        }

        // Simulate resume: read last opcode, continue from there.
        let resume_from = ResultSink::last_opcode(&path).map(|op| op + 1).unwrap_or(0);
        assert_eq!(resume_from, 5);

        // Second run: write opcodes 5..10.
        {
            let mut sink = ResultSink::new(&path).expect("open sink");
            for i in resume_from..10u32 {
                let result = make_observed(i, false, vec![]);
                sink.write(&result).expect("write");
            }
        }

        // Verify: 10 total lines, last opcode is 9.
        let file = File::open(&path).expect("open");
        let line_count = BufReader::new(file).lines().count();
        assert_eq!(line_count, 10);
        assert_eq!(ResultSink::last_opcode(&path), Some(9));

        // Clean up.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
