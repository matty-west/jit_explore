//! Writes probe results to a JSON Lines file for persistent, crash-safe logging.
//!
//! Each line is a self-contained JSON object ([`SinkRecord`]) with a schema
//! version field so the format can evolve without breaking older data.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cpu_state::GPR_NAMES;
use crate::probe::ObservedProbeResult;

// --- Schema types ---

const SCHEMA_VERSION: u32 = 2;

/// A single register diff entry in the serialized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub reg: String,
    pub pre: String,
    pub post: String,
}

/// A single probe result serialized to JSON Lines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkRecord {
    pub v: u32,
    pub opcode: String,
    pub status: String,
    pub diff: Vec<DiffEntry>,
    pub corrupted: bool,
}

impl SinkRecord {
    /// Convert an [`ObservedProbeResult`] into a serializable record.
    pub fn from_observed(result: &ObservedProbeResult) -> Self {
        let diff = result.diff.iter()
            .map(|d| DiffEntry {
                reg: GPR_NAMES[d.index].to_string(),
                pre: format!("0x{:016X}", d.pre),
                post: format!("0x{:016X}", d.post),
            })
            .collect();

        SinkRecord {
            v: SCHEMA_VERSION,
            opcode: format!("0x{:08X}", result.base.opcode),
            status: result.base.status().to_string(),
            diff,
            corrupted: result.snapshot_corrupted,
        }
    }
}

const FLUSH_INTERVAL: usize = 1000;

/// Buffered JSONL writer for probe results.
pub struct ResultSink {
    writer: BufWriter<File>,
    #[allow(dead_code)]
    path: PathBuf,
    pending: usize,
    total: usize,
}

impl ResultSink {
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

    pub fn write(&mut self, result: &ObservedProbeResult) -> io::Result<()> {
        let record = SinkRecord::from_observed(result);
        serde_json::to_writer(&mut self.writer, &record)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.writer.write_all(b"\n")?;
        self.pending += 1;
        self.total += 1;
        if self.pending >= FLUSH_INTERVAL { self.flush()?; }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.pending = 0;
        Ok(())
    }

    pub fn total_written(&self) -> usize { self.total }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path { &self.path }

    /// Read the last opcode from an existing JSONL file.
    pub fn last_opcode(path: &Path) -> Option<u32> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);
        let mut last_line = None;
        for line in reader.lines() {
            if let Ok(l) = line {
                if !l.trim().is_empty() { last_line = Some(l); }
            }
        }
        let line = last_line?;
        let record: SinkRecord = serde_json::from_str(&line).ok()?;
        let hex_str = record.opcode.strip_prefix("0x").unwrap_or(&record.opcode);
        u32::from_str_radix(hex_str, 16).ok()
    }
}

impl Drop for ResultSink {
    fn drop(&mut self) { let _ = self.flush(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_state::{GprSnapshot, RegDiff, GPR_COUNT};
    use crate::probe::{ObservedProbeResult, ProbeResult};

    fn make_observed(opcode: u32, status_faulted: bool, diffs: Vec<RegDiff>) -> ObservedProbeResult {
        ObservedProbeResult {
            base: ProbeResult {
                opcode, faulted: status_faulted, timed_out: false,
                segfaulted: false, trapped: false, fault_offset: 0,
            },
            pre: Some(GprSnapshot { gprs: [0u64; GPR_COUNT] }),
            post: if status_faulted { None } else { Some(GprSnapshot { gprs: [0u64; GPR_COUNT] }) },
            diff: diffs,
            snapshot_corrupted: false,
            gprs_post: None,
        }
    }

    #[test]
    fn sink_record_from_observed() {
        let result = make_observed(0x8B01_0000, false, vec![RegDiff { index: 0, pre: 0xDEAD_0000_0000_0000, post: 0xBD5A_0000_0000_0001 }]);
        let record = SinkRecord::from_observed(&result);
        assert_eq!(record.v, 2);
        assert_eq!(record.opcode, "0x8B010000");
        assert_eq!(record.status, "ok");
        assert_eq!(record.diff.len(), 1);
        assert_eq!(record.diff[0].reg, "x0");
    }

    #[test]
    fn sink_record_round_trip_json() {
        let result = make_observed(0xD503_201F, false, vec![RegDiff { index: 5, pre: 0xDEAD_0000_0000_0005, post: 0x2A }]);
        let record = SinkRecord::from_observed(&result);
        let json = serde_json::to_string(&record).expect("serialize");
        let parsed: SinkRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.opcode, record.opcode);
    }

    #[test]
    fn write_and_read_back() {
        let dir = std::env::temp_dir().join("jit_explore_test_sink");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_write_read.jsonl");
        let _ = std::fs::remove_file(&path);
        {
            let mut sink = ResultSink::new(&path).expect("open sink");
            for i in 0..10u32 {
                let result = make_observed(i, i == 0, vec![]);
                sink.write(&result).expect("write");
            }
        }
        let last = ResultSink::last_opcode(&path);
        assert_eq!(last, Some(9));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
