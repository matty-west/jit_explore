// ╔══════════════════════════════════════╗
// ║  probe — opcode execution harness    ║
// ╚══════════════════════════════════════╝
//
//! Reusable harness for probing arbitrary AArch64 opcodes.
//!
//! Given a `u32` opcode, the harness writes it into a JIT page followed by
//! `RET`, executes it, and reports whether the CPU accepted it or raised
//! `SIGILL`.
//!
//! ## Recovery strategy
//!
//! The probed instruction can corrupt **any** register (SP, LR, x0-x28).
//! We use `sigsetjmp` / `siglongjmp` to recover:
//!
//! 1. Before each probe, `sigsetjmp` saves the entire CPU state.
//! 2. If the instruction faults (SIGILL, SIGSEGV, SIGBUS) or hangs (SIGALRM),
//!    the signal handler calls `siglongjmp` to restore the full state and
//!    return to the `sigsetjmp` call site with a nonzero value.
//! 3. Signal handlers run on an alternate signal stack (SA_ONSTACK) so they
//!    work even when the probed instruction has corrupted SP.

use std::fmt;
use std::time::Instant;

use crate::cpu_state::{GprSnapshot, RegDiff, SnapshotBuffer, seeded_snapshot};
use crate::emitter;
use crate::jit_page::JitPage;
use crate::signal_handler::{
    arm_alarm, clear_probe_flags, did_segfault, did_sigill_fire, did_timeout, did_trap,
    disarm_alarm, disable_longjmp, enable_longjmp, install_signal_handlers, set_escape_address,
    sigsetjmp, JMP_BUF,
};

// ╔══════════════════════════════════════╗
// ║  Constants                           ║
// ╚══════════════════════════════════════╝

/// AArch64 RET — return to caller via x30 (LR).
const RET: u32 = 0xD65F_03C0;

// ╔══════════════════════════════════════╗
// ║  ProbeResult                         ║
// ╚══════════════════════════════════════╝

/// The outcome of probing a single opcode.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// The opcode that was tested.
    pub opcode: u32,
    /// `true` if the instruction raised `SIGILL` (illegal/undefined).
    pub faulted: bool,
    /// `true` if the instruction hung and was killed by the timeout timer.
    pub timed_out: bool,
    /// `true` if the instruction caused a SIGSEGV or SIGBUS (memory fault).
    pub segfaulted: bool,
    /// `true` if the instruction caused a SIGTRAP (BRK / debug trap).
    pub trapped: bool,
}

impl ProbeResult {
    /// Human-readable status.
    pub fn status(&self) -> &'static str {
        if self.timed_out {
            "TIMEOUT"
        } else if self.faulted {
            "SIGILL"
        } else if self.segfaulted {
            "SEGV"
        } else if self.trapped {
            "TRAP"
        } else {
            "ok"
        }
    }
}

impl fmt::Display for ProbeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let le = self.opcode.to_le_bytes();
        write!(
            f,
            "0x{:08X}  [{:02x} {:02x} {:02x} {:02x}]  {}",
            self.opcode,
            le[0], le[1], le[2], le[3],
            self.status()
        )
    }
}

// ╔══════════════════════════════════════╗
// ║  SweepSummary                        ║
// ╚══════════════════════════════════════╝

/// Summary statistics from a sweep over a range of opcodes.
#[derive(Debug, Clone)]
pub struct SweepSummary {
    /// Total number of opcodes probed.
    pub total: usize,
    /// Number that executed without any fault.
    pub ok: usize,
    /// Number that raised SIGILL.
    pub faulted: usize,
    /// Number that timed out (hung).
    pub timed_out: usize,
    /// Number that raised SIGSEGV/SIGBUS.
    pub segfaulted: usize,
    /// Number that raised SIGTRAP.
    pub trapped: usize,
    /// Wall-clock duration of the sweep.
    pub elapsed: std::time::Duration,
}

impl fmt::Display for SweepSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rate = if self.elapsed.as_secs_f64() > 0.0 {
            self.total as f64 / self.elapsed.as_secs_f64()
        } else {
            0.0
        };
        write!(
            f,
            "{} probed: {} ok, {} SIGILL, {} SEGV, {} TRAP, {} TIMEOUT ({:.0} ops/sec, {:.3?})",
            self.total, self.ok, self.faulted, self.segfaulted, self.trapped, self.timed_out,
            rate, self.elapsed
        )
    }
}

// ╔══════════════════════════════════════╗
// ║  ObservedProbeResult                 ║
// ╚══════════════════════════════════════╝

/// The outcome of a probe with full register snapshot.
///
/// Extends [`ProbeResult`] with before/after GPR state and a diff.
#[derive(Debug, Clone)]
pub struct ObservedProbeResult {
    /// Basic probe result (opcode, faulted, timed_out, segfaulted).
    pub base: ProbeResult,
    /// GPR state just before the probed instruction executed.
    /// `None` if the snapshot was corrupted.
    pub pre: Option<GprSnapshot>,
    /// GPR state just after the probed instruction executed.
    /// `None` if the instruction faulted/timed out, or snapshot was corrupted.
    pub post: Option<GprSnapshot>,
    /// Registers that changed between pre and post.
    /// Empty if either snapshot is unavailable.
    pub diff: Vec<RegDiff>,
    /// `true` if the snapshot canaries were corrupted (unreliable diff).
    pub snapshot_corrupted: bool,
}

impl fmt::Display for ObservedProbeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        if self.snapshot_corrupted {
            write!(f, "  [snapshot corrupted]")?;
        } else if !self.diff.is_empty() {
            write!(f, "  mutations:")?;
            for d in &self.diff {
                write!(f, "\n    {d}")?;
            }
        } else if self.base.status() == "ok" {
            write!(f, "  (no GPR changes)")?;
        }
        Ok(())
    }
}

// ╔══════════════════════════════════════╗
// ║  Probe                               ║
// ╚══════════════════════════════════════╝

/// Reusable opcode probe harness.
///
/// Owns a single [`JitPage`] and reuses it for each probe to avoid
/// repeated `mmap`/`munmap` overhead. The signal handlers must be installed
/// before calling [`Probe::run`] (the constructor does this for you).
pub struct Probe {
    page: JitPage,
    /// Per-probe timeout in microseconds. 0 = no timeout.
    timeout_micros: u64,
}

/// Default probe timeout: 5ms.
/// Most valid instructions complete in nanoseconds. If we're still running
/// after 5ms, the instruction is almost certainly stuck (branch-to-self, WFI, etc.).
const DEFAULT_TIMEOUT_MICROS: u64 = 5_000;

impl Probe {
    /// Create a new probe harness with the default 5ms timeout.
    ///
    /// Installs the signal handlers (idempotent) and allocates a JIT page.
    ///
    /// # Panics
    /// Panics if the JIT page allocation fails.
    pub fn new() -> Self {
        install_signal_handlers();
        let page = JitPage::alloc(4096).expect("failed to alloc JIT page for probe");
        crate::signal_handler::set_probe_bounds(
            page.as_ptr() as u64,
            page.as_ptr() as u64 + page.size() as u64,
        );
        Probe {
            page,
            timeout_micros: DEFAULT_TIMEOUT_MICROS,
        }
    }

    /// Create a probe with no timeout (use with care — may hang on bad opcodes).
    #[allow(dead_code)]
    pub fn new_no_timeout() -> Self {
        install_signal_handlers();
        let page = JitPage::alloc(4096).expect("failed to alloc JIT page for probe");
        crate::signal_handler::set_probe_bounds(
            page.as_ptr() as u64,
            page.as_ptr() as u64 + page.size() as u64,
        );
        Probe {
            page,
            timeout_micros: 0,
        }
    }

    /// Raw pointer to the start of the JIT page.
    pub fn page_ptr(&self) -> *const u8 {
        self.page.as_ptr()
    }

    /// Size of the JIT page in bytes.
    pub fn page_size(&self) -> usize {
        self.page.size()
    }

    /// Probe a single opcode: write it + RET, execute, report.
    pub fn run(&self, opcode: u32) -> ProbeResult {
        // 1. Write the opcode under test at offset 0, RET at offset 4.
        self.page.make_writable();
        self.page.write_instruction(0, opcode);
        self.page.write_instruction(4, RET);
        self.page.make_executable();

        // 2. Set escape address (needed by the SIGALRM handler's legacy
        //    path, just in case, but longjmp takes priority).
        set_escape_address(self.page.as_ptr() as u64 + 4);

        // 3. Clear flags, arm timeout.
        clear_probe_flags();

        if self.timeout_micros > 0 {
            arm_alarm(self.timeout_micros);
        }

        // 4. Use sigsetjmp / siglongjmp for recovery.
        //
        //    sigsetjmp(JMP_BUF, 1) saves the entire CPU state:
        //      - All general-purpose registers (x0-x30/SP)
        //      - Program counter
        //      - Signal mask (the `1` argument)
        //
        //    If the probed instruction faults or hangs, the signal handler
        //    calls siglongjmp(JMP_BUF, 1), which restores ALL of the above
        //    and makes sigsetjmp return 1 instead of 0.
        //
        //    This elegantly handles ANY register clobbering.
        let ret = sigsetjmp(JMP_BUF.as_mut_ptr(), 1);
        let _signal_fired = if ret == 0 {
            // First return from sigsetjmp — execute the JIT code.
            // Enable the siglongjmp path in signal handlers.
            enable_longjmp();

            // SAFETY: The page is in executable mode and contains the
            // probed opcode + RET. If the instruction faults or hangs,
            // the signal handler calls siglongjmp to recover.
            unsafe { self.page.call_void(); }

            disable_longjmp();
            false
        } else {
            // Nonzero return: we got here via siglongjmp from a signal
            // handler (fault or timeout). All registers restored by longjmp.
            disable_longjmp();
            true
        };

        if self.timeout_micros > 0 {
            disarm_alarm();
        }

        ProbeResult {
            opcode,
            faulted: did_sigill_fire(),
            timed_out: did_timeout(),
            segfaulted: did_segfault(),
            trapped: did_trap(),
        }
    }

    /// Sweep a range of opcodes and return all results plus a summary.
    pub fn sweep(&self, opcodes: impl Iterator<Item = u32>) -> (Vec<ProbeResult>, SweepSummary) {
        let start = Instant::now();
        let results: Vec<ProbeResult> = opcodes.map(|op| self.run(op)).collect();
        let elapsed = start.elapsed();

        let ok = results.iter().filter(|r| !r.faulted && !r.timed_out && !r.segfaulted && !r.trapped).count();
        let faulted = results.iter().filter(|r| r.faulted).count();
        let timed_out = results.iter().filter(|r| r.timed_out).count();
        let segfaulted = results.iter().filter(|r| r.segfaulted).count();
        let trapped = results.iter().filter(|r| r.trapped).count();

        let summary = SweepSummary {
            total: results.len(),
            ok,
            faulted,
            timed_out,
            segfaulted,
            trapped,
            elapsed,
        };

        (results, summary)
    }

    /// Probe a single opcode with full GPR snapshot (observed mode).
    ///
    /// Emits a prelude (save + seed GPRs) → probed opcode → postlude (save + restore + RET)
    /// into the JIT page, then executes the sequence and diffs the register state.
    ///
    /// This is slower than [`Probe::run`] due to the prelude/postlude overhead,
    /// but captures exactly which registers the instruction modified.
    pub fn run_observed(&self, opcode: u32) -> ObservedProbeResult {
        // Allocate snapshot buffers on the stack.
        let mut buf_pre = SnapshotBuffer::new();
        let mut buf_post = SnapshotBuffer::new();

        // Emit the full observed sequence into the JIT page:
        //   prelude → opcode → postlude
        self.page.make_writable();

        let opcode_offset = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr());
        self.page.write_instruction(opcode_offset, opcode);
        let postlude_offset = opcode_offset + 4;
        let _end = emitter::emit_postlude(
            &self.page,
            postlude_offset,
            buf_post.as_mut_ptr(),
            buf_pre.as_mut_ptr(),
        );

        self.page.make_executable();

        // Set escape address past the sequence (just in case the legacy path fires).
        set_escape_address(self.page.as_ptr() as u64 + _end as u64);

        // Clear flags, arm timeout.
        clear_probe_flags();
        if self.timeout_micros > 0 {
            arm_alarm(self.timeout_micros);
        }

        // Use sigsetjmp / siglongjmp for recovery.
        let ret = sigsetjmp(JMP_BUF.as_mut_ptr(), 1);
        let _signal_fired = if ret == 0 {
            enable_longjmp();
            // SAFETY: The page is in executable mode and contains the full
            // prelude → opcode → postlude → RET sequence. If the probed
            // instruction faults or hangs, siglongjmp recovers.
            unsafe { self.page.call_void(); }
            disable_longjmp();
            false
        } else {
            disable_longjmp();
            true
        };

        if self.timeout_micros > 0 {
            disarm_alarm();
        }

        let faulted = did_sigill_fire();
        let timed_out = did_timeout();
        let segfaulted = did_segfault();
        let trapped = did_trap();

        let base = ProbeResult {
            opcode,
            faulted,
            timed_out,
            segfaulted,
            trapped,
        };

        // Extract snapshots and compute diff.
        // The "pre" for diffing is the known seed values (what the prelude loaded),
        // NOT the caller's register state saved in buf_pre.
        let pre_corrupted = !buf_pre.canaries_intact();
        let post_corrupted = !buf_post.canaries_intact();
        let snapshot_corrupted = pre_corrupted || post_corrupted;

        let seeds = seeded_snapshot();
        let post = if faulted || timed_out || segfaulted || trapped {
            // Post-snapshot is unreliable if the instruction didn't complete normally.
            None
        } else {
            buf_post.to_snapshot()
        };

        // Diff seeds vs post, but skip x28 (scratch), x29 (FP), x30 (LR)
        // since those are framework registers, not probed state.
        let diff = match &post {
            Some(post_snap) => {
                seeds.diff(post_snap)
                    .into_iter()
                    .filter(|d| d.index < 28) // skip x28, x29, x30
                    .collect()
            }
            _ => Vec::new(),
        };

        ObservedProbeResult {
            base,
            pre: Some(seeds),
            post,
            diff,
            snapshot_corrupted,
        }
    }

    /// Probe a single opcode with a prefix/suffix instruction sequence.
    ///
    /// Layout: `prelude → prefix[] → opcode → suffix[] → postlude`
    ///
    /// The prefix runs *after* seeds are loaded (e.g., `SMSTART` to enable
    /// streaming mode before the probed instruction). The suffix runs *before*
    /// the postlude captures registers (e.g., `SMSTOP` to disable streaming
    /// mode so the STP instructions in the postlude work correctly in
    /// non-streaming mode).
    ///
    /// # Arguments
    /// - `opcode`: The instruction under test.
    /// - `prefix`: Instructions emitted before the opcode (e.g., `&[SMSTART]`).
    /// - `suffix`: Instructions emitted after the opcode (e.g., `&[SMSTOP]`).
    pub fn run_observed_with_prefix(
        &self,
        opcode: u32,
        prefix: &[u32],
        suffix: &[u32],
    ) -> ObservedProbeResult {
        let mut buf_pre = SnapshotBuffer::new();
        let mut buf_post = SnapshotBuffer::new();

        self.page.make_writable();

        // Emit: prelude → prefix → opcode → suffix → postlude
        let mut off = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr());

        for &inst in prefix {
            self.page.write_instruction(off, inst);
            off += 4;
        }

        self.page.write_instruction(off, opcode);
        off += 4;

        for &inst in suffix {
            self.page.write_instruction(off, inst);
            off += 4;
        }

        let _end = emitter::emit_postlude(
            &self.page,
            off,
            buf_post.as_mut_ptr(),
            buf_pre.as_mut_ptr(),
        );

        self.page.make_executable();

        set_escape_address(self.page.as_ptr() as u64 + _end as u64);

        clear_probe_flags();
        if self.timeout_micros > 0 {
            arm_alarm(self.timeout_micros);
        }

        let ret = sigsetjmp(JMP_BUF.as_mut_ptr(), 1);
        let _signal_fired = if ret == 0 {
            enable_longjmp();
            // SAFETY: The page contains a valid prelude → prefix → opcode →
            // suffix → postlude → RET sequence. Faults recover via siglongjmp.
            unsafe { self.page.call_void(); }
            disable_longjmp();
            false
        } else {
            disable_longjmp();
            true
        };

        if self.timeout_micros > 0 {
            disarm_alarm();
        }

        let faulted = did_sigill_fire();
        let timed_out = did_timeout();
        let segfaulted = did_segfault();
        let trapped = did_trap();

        let base = ProbeResult {
            opcode,
            faulted,
            timed_out,
            segfaulted,
            trapped,
        };

        let pre_corrupted = !buf_pre.canaries_intact();
        let post_corrupted = !buf_post.canaries_intact();
        let snapshot_corrupted = pre_corrupted || post_corrupted;

        let seeds = seeded_snapshot();
        let post = if faulted || timed_out || segfaulted || trapped {
            None
        } else {
            buf_post.to_snapshot()
        };

        let diff = match &post {
            Some(post_snap) => {
                seeds.diff(post_snap)
                    .into_iter()
                    .filter(|d| d.index < 28)
                    .collect()
            }
            _ => Vec::new(),
        };

        ObservedProbeResult {
            base,
            pre: Some(seeds),
            post,
            diff,
            snapshot_corrupted,
        }
    }

    /// Sweep opcodes with full GPR snapshots, writing each result to a sink.
    ///
    /// Returns a [`SweepSummary`] and whether the sweep was interrupted by Ctrl+C.
    /// The sweep checks [`was_interrupted`] after each probe and stops early if set.
    ///
    /// Progress is printed every `progress_interval` opcodes.
    pub fn observed_sweep(
        &self,
        opcodes: impl Iterator<Item = u32>,
        sink: &mut crate::sink::ResultSink,
        progress_interval: usize,
    ) -> (SweepSummary, bool) {
        use crate::signal_handler::was_interrupted;

        let start = Instant::now();
        let mut ok = 0usize;
        let mut faulted = 0usize;
        let mut timed_out = 0usize;
        let mut segfaulted = 0usize;
        let mut trapped = 0usize;
        let mut total = 0usize;
        let mut interrupted = false;

        for opcode in opcodes {
            if was_interrupted() {
                interrupted = true;
                break;
            }

            let result = self.run_observed(opcode);

            // Classify.
            if result.base.timed_out {
                timed_out += 1;
            } else if result.base.faulted {
                faulted += 1;
            } else if result.base.segfaulted {
                segfaulted += 1;
            } else if result.base.trapped {
                trapped += 1;
            } else {
                ok += 1;
            }

            // Write to sink (best-effort — don't abort sweep on I/O error).
            if let Err(e) = sink.write(&result) {
                eprintln!("warning: sink write failed: {e}");
            }

            total += 1;

            if progress_interval > 0 && total % progress_interval == 0 {
                let elapsed = start.elapsed();
                let rate = total as f64 / elapsed.as_secs_f64();
                eprint!(
                    "\r  {total} probed ({ok} ok, {faulted} SIGILL, {segfaulted} SEGV, {trapped} TRAP, {timed_out} TIMEOUT) [{rate:.0} ops/sec]"
                );
            }
        }

        // Final progress line.
        if total > 0 {
            eprintln!();
        }

        // Flush the sink.
        if let Err(e) = sink.flush() {
            eprintln!("warning: sink flush failed: {e}");
        }

        let summary = SweepSummary {
            total,
            ok,
            faulted,
            timed_out,
            segfaulted,
            trapped,
            elapsed: start.elapsed(),
        };

        (summary, interrupted)
    }
}

impl fmt::Debug for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Probe {{ page: {:?} }}", self.page)
    }
}

// ╔══════════════════════════════════════╗
// ║  Tests                               ║
// ╚══════════════════════════════════════╝

#[cfg(test)]
mod tests {
    use super::*;

    const NOP: u32 = 0xD503_201F;
    const RET_INST: u32 = 0xD65F_03C0;
    const UDF_0: u32 = 0x0000_0000;

    /// B . (branch to self — infinite loop).
    const BRANCH_TO_SELF: u32 = 0x1400_0000;

    #[test]
    fn probe_nop_is_ok() {
        let probe = Probe::new();
        let result = probe.run(NOP);
        assert!(!result.faulted, "NOP should not fault: {result}");
        assert!(!result.timed_out, "NOP should not time out: {result}");
    }

    #[test]
    fn probe_ret_is_ok() {
        let probe = Probe::new();
        let result = probe.run(RET_INST);
        assert!(!result.faulted, "RET should not fault: {result}");
        assert!(!result.timed_out, "RET should not time out: {result}");
    }

    #[test]
    fn probe_udf_faults() {
        let probe = Probe::new();
        let result = probe.run(UDF_0);
        assert!(result.faulted, "UDF #0 should fault: {result}");
        assert!(!result.timed_out, "UDF should not time out: {result}");
    }

    #[test]
    fn probe_branch_to_self_times_out() {
        let probe = Probe::new();
        let result = probe.run(BRANCH_TO_SELF);
        assert!(!result.faulted, "branch-to-self should not SIGILL: {result}");
        assert!(result.timed_out, "branch-to-self should time out: {result}");
    }

    #[test]
    fn sweep_known_opcodes() {
        let probe = Probe::new();
        let opcodes = vec![NOP, RET_INST, UDF_0, NOP, UDF_0];
        let (results, summary) = probe.sweep(opcodes.into_iter());

        assert_eq!(results.len(), 5);
        assert!(!results[0].faulted); // NOP
        assert!(!results[1].faulted); // RET
        assert!(results[2].faulted);  // UDF
        assert!(!results[3].faulted); // NOP
        assert!(results[4].faulted);  // UDF

        assert_eq!(summary.ok, 3);
        assert_eq!(summary.faulted, 2);
        assert_eq!(summary.timed_out, 0);
        assert_eq!(summary.total, 5);
    }

    #[test]
    fn sweep_256_opcodes() {
        let probe = Probe::new();
        let (results, summary) = probe.sweep(0u32..256);

        // All opcodes 0x00000000..0x000000FF are in the UDF encoding space.
        assert_eq!(summary.total, 256);
        assert_eq!(summary.faulted, 256, "all UDF-range opcodes should fault");
        assert_eq!(summary.ok, 0);

        assert!(results[0].faulted);   // 0x00000000 = UDF #0
        assert!(results[255].faulted); // 0x000000FF = UDF #255
    }
}
