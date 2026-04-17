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

struct SharedMemory<T> {
    ptr: *mut T,
}

impl<T> SharedMemory<T> {
    fn new() -> Self {
        let size = std::mem::size_of::<T>();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_SHARED,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "Failed to mmap shared memory");
        
        // Zero initialize the memory (not strictly required if MAP_ANON zeros it, but good practice)
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }

        Self { ptr: ptr as *mut T }
    }

    fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    fn get(&self) -> &T {
        unsafe { &*self.ptr }
    }
}

impl<T> Drop for SharedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, std::mem::size_of::<T>());
        }
    }
}

/// AArch64 RET — return to caller via x30 (LR).
const RET: u32 = 0xD65F_03C0;

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
    /// The offset of the faulting instruction from the start of the batch (0 if not batched).
    pub fault_offset: u32,
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

/// The outcome of probing a batch of opcodes.
#[derive(Debug, Clone)]
pub struct BatchProbeResult {
    /// The opcodes that were tested.
    pub opcodes: Vec<u32>,
    /// Results for each opcode in the batch.
    pub results: Vec<ProbeResult>,
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
    /// `true` if any AMX tile state changed between pre and post snapshots.
    /// Always `false` when the probe ran in non-streaming mode (no ZA state
    /// is captured in that case). `None`-like sentinel — treat as `false`
    /// unless the sweep was explicitly run in streaming mode.
    pub amx_changed: bool,
}

/// High-level classification of a single probe result.
///
/// Used by Gate 9 sweep summaries to identify candidate AMX/SME instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeClassification {
    /// SIGILL — the opcode is not a real instruction (or is restricted).
    Undefined,
    /// Ok, no GPR diff, no AMX change — behaves like NOP.
    NopLike,
    /// Ok, GPR(s) changed but no AMX tile change.
    GprMutating,
    /// Ok, no GPR diff but AMX tile state changed.
    AmxMutating,
    /// Ok, both GPRs and AMX tiles changed.
    GprAndAmxMutating,
    /// SIGSEGV / SIGBUS — the instruction touches memory.
    MemoryFault,
    /// SIGTRAP — BRK or debug trap instruction.
    Trapped,
    /// TIMEOUT — instruction hung (infinite loop, WFI, etc.).
    Hung,
}

impl ObservedProbeResult {
    /// Classify this probe result into a [`ProbeClassification`].
    pub fn classify(&self) -> ProbeClassification {
        if self.base.timed_out {
            ProbeClassification::Hung
        } else if self.base.faulted {
            ProbeClassification::Undefined
        } else if self.base.segfaulted {
            ProbeClassification::MemoryFault
        } else if self.base.trapped {
            ProbeClassification::Trapped
        } else {
            let gpr_changed = !self.diff.is_empty();
            match (gpr_changed, self.amx_changed) {
                (false, false) => ProbeClassification::NopLike,
                (true, false)  => ProbeClassification::GprMutating,
                (false, true)  => ProbeClassification::AmxMutating,
                (true, true)   => ProbeClassification::GprAndAmxMutating,
            }
        }
    }
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
        let page = JitPage::alloc(4096).expect("failed to alloc JIT page for probe");
        Probe {
            page,
            timeout_micros: DEFAULT_TIMEOUT_MICROS,
        }
    }

    /// Create a probe with no timeout (use with care — may hang on bad opcodes).
    #[allow(dead_code)]
    pub fn new_no_timeout() -> Self {
        let page = JitPage::alloc(4096).expect("failed to alloc JIT page for probe");
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

        let mut faulted = false;
        let mut timed_out = false;
        let mut segfaulted = false;
        let mut trapped = false;

        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                // Child process
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::signal(libc::SIGTRAP, libc::SIG_DFL);
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                self.page.make_executable(); // Ensure thread-local JIT state
                
                self.page.call_void();
                libc::_exit(0);
            } else if pid > 0 {
                // Parent process
                let mut status: libc::c_int = 0;
                
                // Increase timeout for debugging if it's too low
                let effective_timeout = if self.timeout_micros > 0 && self.timeout_micros < 100_000 {
                    100_000 // 100ms min for debug
                } else {
                    self.timeout_micros
                };

                if effective_timeout > 0 {
                    let start = Instant::now();
                    let timeout = std::time::Duration::from_micros(effective_timeout);
                    
                    loop {
                        let ret = libc::waitpid(pid, &mut status, libc::WNOHANG | libc::WUNTRACED);
                        if ret == pid {
                            break;
                        } else if ret == -1 {
                            break;
                        }
                        
                        if start.elapsed() > timeout {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                            timed_out = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                } else {
                    libc::waitpid(pid, &mut status, 0);
                }

                if !timed_out {
                    if libc::WIFSIGNALED(status) {
                        let sig = libc::WTERMSIG(status);
                        match sig {
                            libc::SIGILL => faulted = true,
                            libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                            libc::SIGTRAP => trapped = true,
                            _ => faulted = true, // Treat other signals as faults
                        }
                    }
                }
            } else {
                panic!("fork failed");
            }
        }

        ProbeResult {
            opcode,
            faulted,
            timed_out,
            segfaulted,
            trapped,
            fault_offset: 0,
        }
    }

    /// Probe a batch of opcodes: write them + RET, execute, report.
    pub fn run_batch(&self, opcodes: &[u32]) -> BatchProbeResult {
        let count = opcodes.len();
        self.page.make_writable();
        for (i, &op) in opcodes.iter().enumerate() {
            self.page.write_instruction(i * 4, op);
        }
        self.page.write_instruction(count * 4, RET);
        self.page.make_executable();

        let mut faulted = false;
        let mut timed_out = false;
        let mut segfaulted = false;
        let mut trapped = false;

        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::signal(libc::SIGTRAP, libc::SIG_DFL);
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                self.page.make_executable();
                
                self.page.call_void();
                libc::_exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                let effective_timeout = if self.timeout_micros > 0 && self.timeout_micros < 100_000 {
                    100_000
                } else {
                    self.timeout_micros
                };

                if effective_timeout > 0 {
                    let start = Instant::now();
                    let timeout = std::time::Duration::from_micros(effective_timeout);
                    loop {
                        let ret = libc::waitpid(pid, &mut status, libc::WNOHANG | libc::WUNTRACED);
                        if ret == pid {
                            break;
                        } else if ret == -1 {
                            break;
                        }
                        if start.elapsed() > timeout {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                            timed_out = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                } else {
                    libc::waitpid(pid, &mut status, 0);
                }

                if !timed_out && libc::WIFSIGNALED(status) {
                    let sig = libc::WTERMSIG(status);
                    match sig {
                        libc::SIGILL => faulted = true,
                        libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                        libc::SIGTRAP => trapped = true,
                        _ => faulted = true,
                    }
                    // For batching, we would ideally get the PC from siginfo.
                    // But in a separate process, we'd need ptrace or a signal handler
                    // that communicates the PC back via shared memory.
                    // For now, we assume the first instruction faulted if we can't get more info.
                }
            } else {
                panic!("fork failed");
            }
        }

        let mut results = Vec::with_capacity(count);
        for (i, &opcode) in opcodes.iter().enumerate() {
            let mut res = ProbeResult {
                opcode,
                faulted: false,
                timed_out: false,
                segfaulted: false,
                trapped: false,
                fault_offset: 0,
            };
            if i == 0 {
                res.faulted = faulted;
                res.timed_out = timed_out;
                res.segfaulted = segfaulted;
                res.trapped = trapped;
            }
            results.push(res);
        }

        BatchProbeResult {
            opcodes: opcodes.to_vec(),
            results,
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
        // Allocate snapshot buffers in shared memory.
        let buf_pre = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();
        unsafe {
            *buf_pre.as_mut_ptr() = SnapshotBuffer::new();
            *buf_post.as_mut_ptr() = SnapshotBuffer::new();
        }

        // Emit the full observed sequence into the JIT page:
        //   prelude → opcode → postlude
        self.page.make_writable();

        // streaming=false: standard ALU/HINT sweeps — no SMSTART, no AMX capture.
        let opcode_offset = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, false);
        self.page.write_instruction(opcode_offset, opcode);
        let postlude_offset = opcode_offset + 4;
        let _end = emitter::emit_postlude(
            &self.page,
            postlude_offset,
            buf_post.as_mut_ptr() as *mut u8,
            buf_pre.as_mut_ptr() as *mut u8,
            false,
        );

        self.page.make_executable();

        let mut faulted = false;
        let mut timed_out = false;
        let mut segfaulted = false;
        let mut trapped = false;

        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                // Child process
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::signal(libc::SIGTRAP, libc::SIG_DFL);
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                self.page.make_executable(); // Ensure thread-local JIT state
                self.page.call_void();
                std::process::exit(0);
            } else if pid > 0 {
                // Parent process
                let mut status: libc::c_int = 0;
                
                if self.timeout_micros > 0 {
                    let wait_options = libc::WNOHANG;
                    let start = Instant::now();
                    let timeout = std::time::Duration::from_micros(self.timeout_micros);
                    
                    loop {
                        let ret = libc::waitpid(pid, &mut status, wait_options);
                        if ret == pid {
                            break;
                        }
                        if start.elapsed() > timeout {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                            timed_out = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    }
                } else {
                    libc::waitpid(pid, &mut status, 0);
                }

                if !timed_out {
                    if libc::WIFSIGNALED(status) {
                        let sig = libc::WTERMSIG(status);
                        match sig {
                            libc::SIGILL => faulted = true,
                            libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                            libc::SIGTRAP => trapped = true,
                            _ => faulted = true,
                        }
                    }
                }
            } else {
                panic!("fork failed");
            }
        }

        let base = ProbeResult {
            opcode,
            faulted,
            timed_out,
            segfaulted,
            trapped,
            fault_offset: 0,
        };

        // Extract snapshots and compute diff.
        let pre_corrupted = !buf_pre.get().canaries_intact();
        let post_corrupted = !buf_post.get().canaries_intact();
        let snapshot_corrupted = pre_corrupted || post_corrupted;

        let seeds = seeded_snapshot();
        let post = if faulted || timed_out || segfaulted || trapped {
            None
        } else {
            buf_post.get().to_snapshot()
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
            amx_changed: false, // non-streaming: no AMX state captured
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
        let buf_pre = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();
        unsafe {
            *buf_pre.as_mut_ptr() = SnapshotBuffer::new();
            *buf_post.as_mut_ptr() = SnapshotBuffer::new();
        }

        self.page.make_writable();

        // Emit: prelude(streaming=false) → prefix → opcode → suffix → postlude(streaming=false)
        // Callers that need streaming mode should use run_observed_streaming() instead.
        let mut off = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, false);

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
            buf_post.as_mut_ptr() as *mut u8,
            buf_pre.as_mut_ptr() as *mut u8,
            false,
        );

        self.page.make_executable();

        let mut faulted = false;
        let mut timed_out = false;
        let mut segfaulted = false;
        let mut trapped = false;

        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::signal(libc::SIGTRAP, libc::SIG_DFL);
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                self.page.make_executable(); // Ensure thread-local JIT state
                self.page.call_void();
                std::process::exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                let mut wait_options = 0;
                
                if self.timeout_micros > 0 {
                    wait_options = libc::WNOHANG;
                    let start = Instant::now();
                    let timeout = std::time::Duration::from_micros(self.timeout_micros);
                    
                    loop {
                        let ret = libc::waitpid(pid, &mut status, wait_options);
                        if ret == pid {
                            break;
                        }
                        if start.elapsed() > timeout {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                            timed_out = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    }
                } else {
                    libc::waitpid(pid, &mut status, 0);
                }

                if !timed_out {
                    if libc::WIFSIGNALED(status) {
                        let sig = libc::WTERMSIG(status);
                        match sig {
                            libc::SIGILL => faulted = true,
                            libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                            libc::SIGTRAP => trapped = true,
                            _ => faulted = true,
                        }
                    }
                }
            } else {
                panic!("fork failed");
            }
        }

        let base = ProbeResult {
            opcode,
            faulted,
            timed_out,
            segfaulted,
            trapped,
            fault_offset: 0,
        };

        let pre_corrupted = !buf_pre.get().canaries_intact();
        let post_corrupted = !buf_post.get().canaries_intact();
        let snapshot_corrupted = pre_corrupted || post_corrupted;

        let seeds = seeded_snapshot();
        let post = if faulted || timed_out || segfaulted || trapped {
            None
        } else {
            buf_post.get().to_snapshot()
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
            amx_changed: false, // non-streaming: no AMX state captured
        }
    }

    /// Probe a single opcode in **streaming SVE mode** with full GPR + AMX snapshot.
    ///
    /// Layout: `prelude(streaming=true) → opcode → postlude(streaming=true)`
    ///
    /// The prelude emits `SMSTART` before the opcode; the postlude stores all
    /// 8 AMX tiles and emits `SMSTOP` before restoring caller state.
    /// Use this for SME/AMX sweeps — not for standard ALU/HINT sweeps where
    /// `streaming=false` is correct.
    pub fn run_observed_streaming(&self, opcode: u32) -> ObservedProbeResult {
        let buf_pre  = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();
        unsafe {
            *buf_pre.as_mut_ptr()  = SnapshotBuffer::new();
            *buf_post.as_mut_ptr() = SnapshotBuffer::new();
        }

        self.page.make_writable();

        // streaming=true: SMSTART before opcode, AMX capture + SMSTOP in postlude.
        let opcode_offset = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, true);
        self.page.write_instruction(opcode_offset, opcode);
        let postlude_offset = opcode_offset + 4;
        let _end = emitter::emit_postlude(
            &self.page,
            postlude_offset,
            buf_post.as_mut_ptr() as *mut u8,
            buf_pre.as_mut_ptr()  as *mut u8,
            true,
        );

        self.page.make_executable();

        let mut faulted = false;
        let mut timed_out = false;
        let mut segfaulted = false;
        let mut trapped = false;

        unsafe {
            let pid = libc::fork();
            if pid == 0 {
                libc::signal(libc::SIGILL,  libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS,  libc::SIG_DFL);
                libc::signal(libc::SIGTRAP, libc::SIG_DFL);
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                self.page.make_executable();
                self.page.call_void();
                std::process::exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;

                if self.timeout_micros > 0 {
                    let start = Instant::now();
                    let timeout = std::time::Duration::from_micros(self.timeout_micros);
                    loop {
                        let ret = libc::waitpid(pid, &mut status, libc::WNOHANG);
                        if ret == pid { break; }
                        if start.elapsed() > timeout {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                            timed_out = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    }
                } else {
                    libc::waitpid(pid, &mut status, 0);
                }

                if !timed_out && libc::WIFSIGNALED(status) {
                    match libc::WTERMSIG(status) {
                        libc::SIGILL             => faulted    = true,
                        libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                        libc::SIGTRAP            => trapped    = true,
                        _                        => faulted    = true,
                    }
                }
            } else {
                panic!("fork failed");
            }
        }

        let base = ProbeResult { opcode, faulted, timed_out, segfaulted, trapped, fault_offset: 0 };

        let pre_corrupted  = !buf_pre.get().canaries_intact();
        let post_corrupted = !buf_post.get().canaries_intact();
        let snapshot_corrupted = pre_corrupted || post_corrupted;

        let seeds = seeded_snapshot();
        let post = if faulted || timed_out || segfaulted || trapped {
            None
        } else {
            buf_post.get().to_snapshot()
        };

        // GPR diff (skip x28/x29/x30 — framework registers).
        let diff = match &post {
            Some(post_snap) => seeds.diff(post_snap)
                .into_iter()
                .filter(|d| d.index < 28)
                .collect(),
            _ => Vec::new(),
        };

        // AMX diff — compare tile arrays between seeds (all-zero) and post snapshot.
        // The seeds snapshot has amx=[0; AMX_STATE_COUNT], so any non-zero tile byte
        // in the post snapshot means the opcode wrote to a tile.
        let amx_changed = match (&post, snapshot_corrupted) {
            (Some(post_snap), false) => seeds.amx_changed(post_snap),
            _ => false,
        };

        ObservedProbeResult {
            base,
            pre: Some(seeds),
            post,
            diff,
            snapshot_corrupted,
            amx_changed,
        }
    }

    /// Sweep opcodes in **streaming mode** with full GPR + AMX snapshots.
    ///
    /// Like [`observed_sweep`] but calls [`run_observed_streaming`] for each opcode.
    /// Use for SME and AMX encoding sweeps.
    pub fn observed_sweep_streaming(
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

            let result = self.run_observed_streaming(opcode);

            if result.base.timed_out       { timed_out  += 1; }
            else if result.base.faulted    { faulted    += 1; }
            else if result.base.segfaulted { segfaulted += 1; }
            else if result.base.trapped    { trapped    += 1; }
            else                           { ok         += 1; }

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

        if total > 0 { eprintln!(); }
        if let Err(e) = sink.flush() { eprintln!("warning: sink flush failed: {e}"); }

        let summary = SweepSummary {
            total, ok, faulted, timed_out, segfaulted, trapped,
            elapsed: start.elapsed(),
        };
        (summary, interrupted)
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
