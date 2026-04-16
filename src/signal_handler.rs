// ╔══════════════════════════════════════╗
// ║  signal_handler — SIGILL recovery    ║
// ╚══════════════════════════════════════╝
//
//! Installs signal handlers for recovering from illegal instructions and
//! hung probes:
//!
//! - **`SIGILL`**: The CPU hit an undefined/illegal instruction.
//! - **`SIGALRM`**: A per-probe timer expired — the instruction hung.
//! - **`SIGSEGV`/`SIGBUS`**: The instruction performed a bad memory access.
//!
//! ## Recovery strategy
//!
//! All handlers use `siglongjmp` to restore the full CPU state (including SP,
//! LR, callee-saved registers) from a `sigsetjmp` checkpoint. This handles
//! **any** register clobbering by the probed instruction.
//!
//! The handlers run on an alternate signal stack (`SA_ONSTACK`) so they work
//! even when the probed instruction has corrupted SP.
//!
//! ## Legacy support
//!
//! The PC-redirect approach (`redirect_pc_to_escape`) is kept for the Gate 4
//! tests and the `install_sigill_handler` / `set_escape_address` API.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ╔══════════════════════════════════════╗
// ║  FFI: sigsetjmp / siglongjmp         ║
// ╚══════════════════════════════════════╝

/// On macOS aarch64, `sigjmp_buf` = `int[49]` (196 bytes, align 4).
/// The libc crate doesn't expose these on macOS, so we declare them manually.
#[repr(C, align(4))]
pub struct SigJmpBuf {
    _buf: [i32; 49],
}

// SAFETY: SigJmpBuf is only accessed by the single probe thread and signal
// handlers (which run on the same thread). We wrap it in SyncUnsafeCell to
// avoid `static mut` and the associated UB of mutable references.
unsafe impl Sync for SigJmpBufWrapper {}

/// Wrapper to make `SigJmpBuf` usable in a `static`.
pub struct SigJmpBufWrapper {
    inner: std::cell::UnsafeCell<SigJmpBuf>,
}

impl SigJmpBufWrapper {
    const fn new() -> Self {
        SigJmpBufWrapper {
            inner: std::cell::UnsafeCell::new(SigJmpBuf { _buf: [0; 49] }),
        }
    }

    /// Get a raw mutable pointer to the inner `SigJmpBuf`.
    pub fn as_mut_ptr(&self) -> *mut SigJmpBuf {
        self.inner.get()
    }
}

unsafe extern "C" {
    /// Save the current CPU state + signal mask into `env`.
    /// Returns 0 on direct call; returns `val` when reached via `siglongjmp`.
    pub safe fn sigsetjmp(env: *mut SigJmpBuf, savemask: libc::c_int) -> libc::c_int;

    /// Restore the CPU state saved by `sigsetjmp`, making `sigsetjmp` return `val`.
    /// `val` must be nonzero (if 0 is passed, the implementation substitutes 1).
    pub safe fn siglongjmp(env: *mut SigJmpBuf, val: libc::c_int) -> !;
}

// ╔══════════════════════════════════════╗
// ║  Global state                        ║
// ╚══════════════════════════════════════╝

/// Set to `true` by the SIGILL handler when it fires.
static SIGILL_FIRED: AtomicBool = AtomicBool::new(false);

/// Set to `true` by the SIGALRM handler when a probe times out.
static TIMED_OUT: AtomicBool = AtomicBool::new(false);

/// Set to `true` by the SIGSEGV/SIGBUS handler when a memory fault occurs.
static SEGFAULT_FIRED: AtomicBool = AtomicBool::new(false);

/// Set to `true` by the SIGTRAP handler when a debug/breakpoint trap occurs.
static TRAP_FIRED: AtomicBool = AtomicBool::new(false);

/// Captured PC where the fault occurred.
static FAULT_PC: AtomicU64 = AtomicU64::new(0);

/// The "escape address" — points to a RET instruction in the JIT page.
/// Used by the legacy PC-redirect path (Gate 4 tests).
static ESCAPE_PC: AtomicU64 = AtomicU64::new(0);

/// Bounds of the active JIT page. We only recover if the fault occurred
/// within these bounds. This prevents delayed SIGALRM from corrupting Rust control flow.
static PROBE_START: AtomicU64 = AtomicU64::new(0);
static PROBE_END: AtomicU64 = AtomicU64::new(0);

/// When `true`, signal handlers use `siglongjmp` to recover.
/// When `false`, they fall back to the PC-redirect approach.
static USE_LONGJMP: AtomicBool = AtomicBool::new(false);

/// The `sigsetjmp` / `siglongjmp` buffer.
///
/// # Safety
/// - Only written by `sigsetjmp` in `Probe::run` (single-threaded probe path).
/// - Only read by signal handlers via `siglongjmp_from_handler`.
/// - Zero-initialised; `sigsetjmp` fills it before any longjmp can occur.
pub static JMP_BUF: SigJmpBufWrapper = SigJmpBufWrapper::new();

// ╔══════════════════════════════════════╗
// ║  Handler installation                ║
// ╚══════════════════════════════════════╝

/// Install the `SIGILL`, `SIGALRM`, `SIGSEGV`, and `SIGBUS` handlers.
///
/// Must be called once before any probing. Calling it multiple times is safe
/// (it just re-installs the same handlers).
///
/// # Panics
/// Panics if `sigaction` or `sigaltstack` fails.
pub fn install_signal_handlers() {
    install_alt_stack();
    install_handler(libc::SIGILL, sigill_handler);
    install_handler(libc::SIGALRM, sigalrm_handler);
    install_handler(libc::SIGSEGV, sigsegv_handler);
    install_handler(libc::SIGBUS, sigsegv_handler);
    install_handler(libc::SIGTRAP, sigtrap_handler);
}

/// Also expose the original name for backward compatibility with Gate 4.
pub fn install_sigill_handler() {
    install_signal_handlers();
}

/// Enable the `siglongjmp` recovery path. Call this after `sigsetjmp`
/// has initialised `JMP_BUF`.
pub fn enable_longjmp() {
    USE_LONGJMP.store(true, Ordering::Relaxed);
}

/// Disable the `siglongjmp` recovery path.
pub fn disable_longjmp() {
    USE_LONGJMP.store(false, Ordering::Relaxed);
}

/// Called from signal handlers to `siglongjmp` back to the `sigsetjmp`
/// checkpoint in `Probe::run`.
///
/// # Safety
/// `JMP_BUF` must have been initialised by a prior `sigsetjmp` call, and
/// `enable_longjmp()` must have been called.
pub fn siglongjmp_from_handler() {
    // SAFETY: Caller guarantees JMP_BUF is initialised. We pass `val=1`
    // so `sigsetjmp` returns 1 to indicate signal recovery.
    siglongjmp(JMP_BUF.as_mut_ptr(), 1);
}

/// Helper: install an alternate signal stack so handlers work even when
/// the probed instruction has corrupted SP.
///
/// Uses a 64 KiB heap-allocated buffer (leaked — lives for the process lifetime).
fn install_alt_stack() {
    use std::sync::atomic::AtomicBool;
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return; // already installed
    }

    const ALT_STACK_SIZE: usize = 64 * 1024; // 64 KiB
    let buf = vec![0u8; ALT_STACK_SIZE].into_boxed_slice();
    let ptr = Box::into_raw(buf) as *mut libc::c_void;

    let ss = libc::stack_t {
        ss_sp: ptr,
        ss_flags: 0,
        ss_size: ALT_STACK_SIZE,
    };

    // SAFETY: `sigaltstack` with a valid buffer pointer and size.
    let ret = unsafe { libc::sigaltstack(&ss, std::ptr::null_mut()) };
    assert!(ret == 0, "sigaltstack failed: {}", std::io::Error::last_os_error());
}

/// Helper: install a single `SA_SIGINFO | SA_ONSTACK` handler for the given signal.
fn install_handler(
    sig: libc::c_int,
    handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void),
) {
    // SAFETY: We're constructing a `sigaction` struct and installing it.
    // The handler functions are async-signal-safe (atomic ops + pointer write only).
    // `SA_SIGINFO` gives us access to the `ucontext_t` for PC modification.
    // `SA_ONSTACK` makes the handler run on the alternate signal stack, so it
    // works even if the probed instruction has corrupted SP.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&raw mut sa.sa_mask);

        let ret = libc::sigaction(sig, &sa, std::ptr::null_mut());
        assert!(
            ret == 0,
            "sigaction({sig}) failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

// ╔══════════════════════════════════════╗
// ║  Handler functions                   ║
// ╚══════════════════════════════════════╝

/// Helper: check if `__pc` is inside the JIT page bounds.
fn is_inside_probe(ucontext: *mut libc::c_void) -> bool {
    let start = PROBE_START.load(Ordering::Relaxed);
    let end = PROBE_END.load(Ordering::Relaxed);

    if start == 0 && end == 0 {
        // If bounds aren't set (e.g. older tests), assume we're inside.
        return true;
    }

    unsafe {
        let uc = ucontext as *mut libc::ucontext_t;
        let mctx = (*uc).uc_mcontext;
        let current_pc = (*mctx).__ss.__pc;
        current_pc >= start && current_pc < end
    }
}

/// `SIGILL` handler: set the faulted flag and recover.
///
/// # Signal safety
/// Only performs atomic stores and either a `siglongjmp` or pointer write.
extern "C" fn sigill_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let pc = capture_fault_pc(ucontext);
    if USE_LONGJMP.load(Ordering::Relaxed) {
        // Active siglongjmp probe — catch all faults even if PC is wild.
        SIGILL_FIRED.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        siglongjmp_from_handler();
    } else {
        // Legacy path (Gate 4) or spurious fault. Must check bounds.
        if !is_inside_probe(ucontext) { return; }
        SIGILL_FIRED.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        redirect_pc_to_escape(ucontext);
    }
}

/// `SIGALRM` handler: set the timeout flag and recover.
extern "C" fn sigalrm_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let pc = capture_fault_pc(ucontext);
    if USE_LONGJMP.load(Ordering::Relaxed) {
        TIMED_OUT.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        siglongjmp_from_handler();
    } else {
        if !is_inside_probe(ucontext) { return; }
        TIMED_OUT.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        redirect_pc_to_escape(ucontext);
    }
}

/// `SIGSEGV` / `SIGBUS` handler: set the segfault flag and recover.
extern "C" fn sigsegv_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let pc = capture_fault_pc(ucontext);
    if USE_LONGJMP.load(Ordering::Relaxed) {
        SEGFAULT_FIRED.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        siglongjmp_from_handler();
    } else {
        if !is_inside_probe(ucontext) { return; }
        SEGFAULT_FIRED.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        redirect_pc_to_escape(ucontext);
    }
}

/// `SIGTRAP` handler: set the trap flag and recover.
///
/// SIGTRAP is raised by BRK instructions (software breakpoints) and
/// certain debug-related opcodes.
extern "C" fn sigtrap_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let pc = capture_fault_pc(ucontext);
    if USE_LONGJMP.load(Ordering::Relaxed) {
        TRAP_FIRED.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        siglongjmp_from_handler();
    } else {
        if !is_inside_probe(ucontext) { return; }
        TRAP_FIRED.store(true, Ordering::Relaxed);
        FAULT_PC.store(pc, Ordering::Relaxed);
        redirect_pc_to_escape(ucontext);
    }
}

/// Helper: extract the PC from a `ucontext_t`.
fn capture_fault_pc(ucontext: *mut libc::c_void) -> u64 {
    unsafe {
        let uc = ucontext as *mut libc::ucontext_t;
        let mctx = (*uc).uc_mcontext;
        (*mctx).__ss.__pc
    }
}

/// Redirect the saved PC in the ucontext to the escape address.
///
/// This is the **legacy** recovery path used by Gate 4 tests (which call
/// `page.call_void()` directly). The `Probe` harness uses `siglongjmp` instead.
///
/// # Signal safety
/// Atomic loads + pointer writes. Fully signal-safe.
fn redirect_pc_to_escape(ucontext: *mut libc::c_void) {
    let escape = ESCAPE_PC.load(Ordering::Relaxed);
    debug_assert!(escape != 0, "ESCAPE_PC not set before probe");

    // SAFETY: When `SA_SIGINFO` is set, the third argument to the handler is
    // a `*mut ucontext_t`. On macOS aarch64:
    //   ucontext_t.uc_mcontext -> __darwin_mcontext64
    //   __darwin_mcontext64.__ss -> __darwin_arm_thread_state64
    //   __darwin_arm_thread_state64.__pc -> u64
    //
    // We set __pc to the escape address (a RET instruction in the JIT page).
    unsafe {
        let uc = ucontext as *mut libc::ucontext_t;
        let mctx = (*uc).uc_mcontext;
        (*mctx).__ss.__pc = escape;
    }
}

// ╔══════════════════════════════════════╗
// ║  Public interface                    ║
// ╚══════════════════════════════════════╝

/// Set the escape address — the address of a `RET` instruction that the
/// legacy signal handlers will jump to when recovering from a fault or timeout.
///
/// Must be called before each probe (or once if the JIT page doesn't move).
pub fn set_escape_address(addr: u64) {
    ESCAPE_PC.store(addr, Ordering::Relaxed);
}

/// Clear all flags. Call this **before** executing a probe.
pub fn clear_probe_flags() {
    SIGILL_FIRED.store(false, Ordering::Relaxed);
    TIMED_OUT.store(false, Ordering::Relaxed);
    SEGFAULT_FIRED.store(false, Ordering::Relaxed);
    TRAP_FIRED.store(false, Ordering::Relaxed);
    FAULT_PC.store(0, Ordering::Relaxed);
}

/// Also expose the original name for backward compatibility with Gate 4.
pub fn clear_sigill_flag() {
    clear_probe_flags();
}

/// Check whether `SIGILL` fired since the last call to [`clear_probe_flags`].
pub fn did_sigill_fire() -> bool {
    SIGILL_FIRED.load(Ordering::Relaxed)
}

/// Check whether the probe timed out since the last call to [`clear_probe_flags`].
pub fn did_timeout() -> bool {
    TIMED_OUT.load(Ordering::Relaxed)
}

/// Check whether `SIGSEGV` or `SIGBUS` fired since the last call to [`clear_probe_flags`].
pub fn did_segfault() -> bool {
    SEGFAULT_FIRED.load(Ordering::Relaxed)
}

/// Check whether `SIGTRAP` fired since the last call to [`clear_probe_flags`].
pub fn did_trap() -> bool {
    TRAP_FIRED.load(Ordering::Relaxed)
}

/// Get the captured faulting PC.
pub fn get_fault_pc() -> u64 {
    FAULT_PC.load(Ordering::Relaxed)
}

/// Set the bounds of the active JIT page.
///
/// We use this to ensure that `SIGALRM` only triggers a recovery if the timer fired
/// while we were actually executing the probe, avoiding control flow corruption
/// if the signal is delivered delayed.
pub fn set_probe_bounds(start: u64, end: u64) {
    PROBE_START.store(start, Ordering::Relaxed);
    PROBE_END.store(end, Ordering::Relaxed);
}

// ╔══════════════════════════════════════╗
// ║  SIGINT (Ctrl+C) support             ║
// ╚══════════════════════════════════════╝

/// Set to `true` by the SIGINT handler when Ctrl+C is pressed.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Install a `SIGINT` handler that sets the [`INTERRUPTED`] flag.
///
/// The handler does *not* longjmp or abort — it simply sets a flag that
/// sweep loops can poll via [`was_interrupted`].
///
/// # Panics
/// Panics if `sigaction` fails.
pub fn install_sigint_handler() {
    // SAFETY: We install a minimal signal handler that only writes to an
    // atomic bool. The handler is async-signal-safe.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigint_handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);

        let ret = libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        assert!(ret == 0, "sigaction(SIGINT) failed: {}", *libc::__error());
    }
}

/// Minimal `SIGINT` handler — just sets the flag.
extern "C" fn sigint_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ucontext: *mut libc::c_void,
) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

/// Check whether Ctrl+C has been pressed since the last call to [`clear_interrupted`].
pub fn was_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Clear the interrupted flag.
pub fn clear_interrupted() {
    INTERRUPTED.store(false, Ordering::Relaxed);
}

// ╔══════════════════════════════════════╗
// ║  Timer helpers                       ║
// ╚══════════════════════════════════════╝

/// Arm a one-shot real-time alarm after `micros` microseconds.
///
/// When the timer fires, `SIGALRM` is delivered to the process, and
/// our handler redirects execution to the escape address.
pub fn arm_alarm(micros: u64) {
    let secs = micros / 1_000_000;
    let usecs = (micros % 1_000_000) as i32;

    let it = libc::itimerval {
        it_interval: libc::timeval { tv_sec: 0, tv_usec: 0 }, // no repeat
        it_value: libc::timeval {
            tv_sec: secs as _,
            tv_usec: usecs as _,
        },
    };

    // SAFETY: `setitimer(ITIMER_REAL, ...)` is safe to call. It arms a
    // one-shot alarm that delivers SIGALRM after the specified duration.
    let ret = unsafe { libc::setitimer(libc::ITIMER_REAL, &it, std::ptr::null_mut()) };
    debug_assert!(ret == 0, "setitimer failed");
}

/// Disarm the alarm timer. Call this after a probe returns successfully.
pub fn disarm_alarm() {
    let it = libc::itimerval {
        it_interval: libc::timeval { tv_sec: 0, tv_usec: 0 },
        it_value: libc::timeval { tv_sec: 0, tv_usec: 0 }, // zero = disarm
    };

    // SAFETY: `setitimer` with zero values disarms the timer.
    let ret = unsafe { libc::setitimer(libc::ITIMER_REAL, &it, std::ptr::null_mut()) };
    debug_assert!(ret == 0, "setitimer (disarm) failed");
}

// ╔══════════════════════════════════════╗
// ║  Tests                               ║
// ╚══════════════════════════════════════╝

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_page::JitPage;

    /// AArch64 UDF #0 — permanently undefined.
    const UDF_0: u32 = 0x0000_0000;
    /// AArch64 RET.
    const RET: u32 = 0xD65F_03C0;
    /// AArch64 NOP.
    const NOP: u32 = 0xD503_201F;

    /// Helper: set up a page with `opcode` at offset 0, RET at offset 4,
    /// configure escape address, and return the page.
    fn setup_probe_page(opcode: u32) -> JitPage {
        let page = JitPage::alloc(4096).expect("mmap should succeed");
        page.make_writable();
        page.write_instruction(0, opcode);
        page.write_instruction(4, RET);
        page.make_executable();

        // Escape address = address of the RET at offset 4.
        set_escape_address(page.as_ptr() as u64 + 4);
        page
    }

    #[test]
    fn sigill_handler_recovers_from_udf() {
        install_signal_handlers();
        let page = setup_probe_page(UDF_0);

        clear_probe_flags();
        // SAFETY: UDF + RET. Handler redirects PC to RET.
        unsafe { page.call_void(); }

        assert!(did_sigill_fire(), "SIGILL should have fired for UDF #0");
        assert!(!did_timeout(), "should not have timed out");
    }

    #[test]
    fn nop_does_not_trigger_sigill() {
        install_signal_handlers();
        let page = setup_probe_page(NOP);

        clear_probe_flags();
        // SAFETY: NOP + RET is perfectly valid.
        unsafe { page.call_void(); }

        assert!(!did_sigill_fire(), "NOP should not trigger SIGILL");
        assert!(!did_timeout(), "should not have timed out");
    }

    #[test]
    fn multiple_udf_in_sequence() {
        install_signal_handlers();
        let page = JitPage::alloc(4096).expect("mmap should succeed");

        // Emit: UDF ; UDF ; UDF ; RET
        page.make_writable();
        page.write_instruction(0, UDF_0);
        page.write_instruction(4, UDF_0);
        page.write_instruction(8, UDF_0);
        page.write_instruction(12, RET);
        page.make_executable();

        // Escape address points to the final RET.
        set_escape_address(page.as_ptr() as u64 + 12);

        clear_probe_flags();
        // SAFETY: Three UDFs + RET. Each UDF triggers SIGILL, handler
        // redirects to escape (RET at offset 12).
        unsafe { page.call_void(); }

        assert!(did_sigill_fire(), "SIGILL should have fired for the UDFs");
    }

    #[test]
    #[ignore = "SIGALRM is process-directed; can be delivered to the wrong thread \
                in the cargo test harness and hang. Run with: cargo test -- --ignored"]
    fn timeout_recovers_from_hang() {
        install_signal_handlers();

        let page = JitPage::alloc(4096).expect("mmap should succeed");

        // Emit: B . (branch to self — infinite loop) ; RET
        // B . = 0x14000000 (branch +0, i.e., to self)
        const BRANCH_TO_SELF: u32 = 0x1400_0000;
        page.make_writable();
        page.write_instruction(0, BRANCH_TO_SELF);
        page.write_instruction(4, RET);
        page.make_executable();

        set_escape_address(page.as_ptr() as u64 + 4);
        clear_probe_flags();

        // Arm a 10ms timeout.
        arm_alarm(10_000);

        // SAFETY: Branch-to-self will loop forever, but SIGALRM will fire
        // after 10ms and redirect PC to the RET.
        unsafe { page.call_void(); }

        disarm_alarm();

        assert!(!did_sigill_fire(), "should not have SIGILLed");
        assert!(did_timeout(), "should have timed out on branch-to-self");
    }
}

