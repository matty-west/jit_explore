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

use crate::jit_page::JitPage;
use crate::signal_handler::{
    arm_alarm, clear_probe_flags, did_segfault, did_sigill_fire, did_timeout, disarm_alarm,
    disable_longjmp, enable_longjmp, install_signal_handlers, set_escape_address,
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
            "{} probed: {} ok, {} SIGILL, {} SEGV, {} TIMEOUT ({:.0} ops/sec, {:.3?})",
            self.total, self.ok, self.faulted, self.segfaulted, self.timed_out, rate, self.elapsed
        )
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
        }
    }

    /// Sweep a range of opcodes and return all results plus a summary.
    pub fn sweep(&self, opcodes: impl Iterator<Item = u32>) -> (Vec<ProbeResult>, SweepSummary) {
        let start = Instant::now();
        let results: Vec<ProbeResult> = opcodes.map(|op| self.run(op)).collect();
        let elapsed = start.elapsed();

        let ok = results.iter().filter(|r| !r.faulted && !r.timed_out && !r.segfaulted).count();
        let faulted = results.iter().filter(|r| r.faulted).count();
        let timed_out = results.iter().filter(|r| r.timed_out).count();
        let segfaulted = results.iter().filter(|r| r.segfaulted).count();

        let summary = SweepSummary {
            total: results.len(),
            ok,
            faulted,
            timed_out,
            segfaulted,
            elapsed,
        };

        (results, summary)
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
