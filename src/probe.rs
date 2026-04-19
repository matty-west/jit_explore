//! Reusable harness for probing arbitrary AArch64 opcodes.

use std::fmt;
use std::time::Instant;

use crate::cpu_state::{GprSnapshot, RegDiff, SnapshotBuffer, seeded_snapshot};
use crate::emitter;
use crate::jit_page::JitPage;

#[cfg(target_os = "macos")]
mod accelerate {
    use crate::crucible::{CblasOrder, CblasTranspose};
    use std::os::raw::{c_float, c_int};

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: CblasOrder,
            trans_a: CblasTranspose,
            trans_b: CblasTranspose,
            m: c_int, n: c_int, k: c_int,
            alpha: c_float,
            a: *const c_float, lda: c_int,
            b: *const c_float, ldb: c_int,
            beta: c_float,
            c: *mut c_float, ldc: c_int,
        );
    }

    /// Wake the SME/matrix hardware via a cheap Accelerate call.
    pub fn wake_hardware() {
        let a = [1.0f32];
        let b = [1.0f32];
        let mut c = [0.0f32];
        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor, CblasTranspose::NoTrans, CblasTranspose::NoTrans,
                1, 1, 1, 1.0, a.as_ptr(), 1, b.as_ptr(), 1, 0.0, c.as_mut_ptr(), 1,
            );
            let a2 = [1.0f32; 16];
            let b2 = [1.0f32; 16];
            let mut c2 = [0.0f32; 16];
            cblas_sgemm(
                CblasOrder::RowMajor, CblasTranspose::NoTrans, CblasTranspose::NoTrans,
                4, 4, 4, 1.0, a2.as_ptr(), 4, b2.as_ptr(), 4, 0.0, c2.as_mut_ptr(), 4,
            );
        }
    }
}

pub struct SharedMemory<T> {
    ptr: *mut T,
}

impl<T> SharedMemory<T> {
    pub fn new() -> Self {
        let size = std::mem::size_of::<T>();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(), size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_SHARED,
                -1, 0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "Failed to mmap shared memory");
        unsafe { std::ptr::write_bytes(ptr, 0, size); }
        Self { ptr: ptr as *mut T }
    }
    pub fn as_mut_ptr(&self) -> *mut T { self.ptr }
    pub fn get(&self) -> &T { unsafe { &*self.ptr } }
}

impl<T> Drop for SharedMemory<T> {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, std::mem::size_of::<T>()); }
    }
}

const RET: u32 = 0xD65F_03C0;

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub opcode: u32,
    pub faulted: bool,
    pub timed_out: bool,
    pub segfaulted: bool,
    pub trapped: bool,
    pub fault_offset: u32,
}

impl ProbeResult {
    pub fn status(&self) -> &'static str {
        if self.timed_out { "TIMEOUT" }
        else if self.faulted { "SIGILL" }
        else if self.segfaulted { "SEGV" }
        else if self.trapped { "TRAP" }
        else { "ok" }
    }
}

impl fmt::Display for ProbeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let le = self.opcode.to_le_bytes();
        write!(f, "0x{:08X}  [{:02x} {:02x} {:02x} {:02x}]  {}",
            self.opcode, le[0], le[1], le[2], le[3], self.status())
    }
}

#[derive(Debug, Clone)]
pub struct SweepSummary {
    pub total: usize,
    pub ok: usize,
    pub faulted: usize,
    pub timed_out: usize,
    pub segfaulted: usize,
    pub trapped: usize,
    pub elapsed: std::time::Duration,
}

impl fmt::Display for SweepSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rate = if self.elapsed.as_secs_f64() > 0.0 { self.total as f64 / self.elapsed.as_secs_f64() } else { 0.0 };
        write!(f, "{} probed: {} ok, {} SIGILL, {} SEGV, {} TRAP, {} TIMEOUT ({:.0} ops/sec, {:.3?})",
            self.total, self.ok, self.faulted, self.segfaulted, self.trapped, self.timed_out, rate, self.elapsed)
    }
}

#[derive(Debug, Clone)]
pub struct ObservedProbeResult {
    pub base: ProbeResult,
    pub pre: Option<GprSnapshot>,
    pub post: Option<GprSnapshot>,
    pub diff: Vec<RegDiff>,
    pub snapshot_corrupted: bool,
    pub gprs_post: Option<GprSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeClassification {
    Undefined,
    NopLike,
    GprMutating,
    MemoryFault,
    Trapped,
    Hung,
}

impl ObservedProbeResult {
    pub fn classify(&self) -> ProbeClassification {
        if self.base.timed_out { ProbeClassification::Hung }
        else if self.base.faulted { ProbeClassification::Undefined }
        else if self.base.segfaulted { ProbeClassification::MemoryFault }
        else if self.base.trapped { ProbeClassification::Trapped }
        else if !self.diff.is_empty() { ProbeClassification::GprMutating }
        else { ProbeClassification::NopLike }
    }
}

impl fmt::Display for ObservedProbeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        if self.snapshot_corrupted { write!(f, "  [snapshot corrupted]")?; }
        else if !self.diff.is_empty() {
            write!(f, "  mutations:")?;
            for d in &self.diff { write!(f, "\n    {d}")?; }
        } else if self.base.status() == "ok" { write!(f, "  (no GPR changes)")?; }
        Ok(())
    }
}

pub struct Probe {
    page: JitPage,
    pub timeout_micros: u64,
}

const DEFAULT_TIMEOUT_MICROS: u64 = 5_000;

impl Probe {
    pub fn new() -> Self {
        let page = JitPage::alloc(1024 * 1024).expect("failed to alloc JIT page for probe");
        Probe { page, timeout_micros: DEFAULT_TIMEOUT_MICROS }
    }

    pub fn page_ptr(&self) -> *const u8 { self.page.as_ptr() }
    pub fn page_size(&self) -> usize { self.page.size() }

    pub fn run_block(&self, opcodes: &[u32]) -> ProbeResult {
        self.run_block_with_overrides(opcodes, &[], false)
    }

    pub fn run_block_with_overrides(&self, opcodes: &[u32], gpr_overrides: &[(u8, u64)], streaming: bool) -> ProbeResult {
        self.run_block_with_overrides_ext(opcodes, gpr_overrides, streaming, true)
    }

    pub fn run_block_with_overrides_ext(&self, opcodes: &[u32], gpr_overrides: &[(u8, u64)], streaming: bool, wake: bool) -> ProbeResult {
        let buf_pre = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();
        unsafe {
            *buf_pre.as_mut_ptr() = SnapshotBuffer::new();
            *buf_post.as_mut_ptr() = SnapshotBuffer::new();
        }
        self.page.make_writable();
        let mut off = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, streaming, gpr_overrides);
        for &op in opcodes {
            self.page.write_instruction(off, op);
            off += 4;
        }
        emitter::emit_postlude(&self.page, off, buf_post.as_mut_ptr() as *mut u8, buf_pre.as_mut_ptr() as *mut u8, streaming);
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
                #[cfg(target_os = "macos")]
                if wake {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static WOKEN: AtomicBool = AtomicBool::new(false);
                    if !WOKEN.swap(true, Ordering::SeqCst) { accelerate::wake_hardware(); }
                }
                self.page.call_void();
                libc::_exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                let start = Instant::now();
                let timeout = std::time::Duration::from_micros(if self.timeout_micros > 0 { self.timeout_micros } else { 100_000 });
                loop {
                    let ret = libc::waitpid(pid, &mut status, libc::WNOHANG);
                    if ret == pid { break; }
                    if start.elapsed() > timeout {
                        libc::kill(pid, libc::SIGKILL);
                        libc::waitpid(pid, &mut status, 0);
                        timed_out = true;
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                if !timed_out && libc::WIFSIGNALED(status) {
                    match libc::WTERMSIG(status) {
                        libc::SIGILL => faulted = true,
                        libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                        libc::SIGTRAP => trapped = true,
                        _ => faulted = true,
                    }
                }
            } else { panic!("fork failed"); }
        }
        ProbeResult { opcode: opcodes[0], faulted, timed_out, segfaulted, trapped, fault_offset: 0 }
    }

    pub fn run(&self, opcode: u32) -> ProbeResult {
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
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::signal(libc::SIGTRAP, libc::SIG_DFL);
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                self.page.make_executable();
                #[cfg(target_os = "macos")]
                accelerate::wake_hardware();
                self.page.call_void();
                libc::_exit(0);
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                let effective_timeout = if self.timeout_micros > 0 && self.timeout_micros < 100_000 { 100_000 } else { self.timeout_micros };
                if effective_timeout > 0 {
                    let start = Instant::now();
                    let timeout = std::time::Duration::from_micros(effective_timeout);
                    loop {
                        let ret = libc::waitpid(pid, &mut status, libc::WNOHANG | libc::WUNTRACED);
                        if ret == pid || ret == -1 { break; }
                        if start.elapsed() > timeout {
                            libc::kill(pid, libc::SIGKILL);
                            libc::waitpid(pid, &mut status, 0);
                            timed_out = true;
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                } else { libc::waitpid(pid, &mut status, 0); }
                if !timed_out && libc::WIFSIGNALED(status) {
                    match libc::WTERMSIG(status) {
                        libc::SIGILL => faulted = true,
                        libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                        libc::SIGTRAP => trapped = true,
                        _ => faulted = true,
                    }
                }
            } else { panic!("fork failed"); }
        }
        ProbeResult { opcode, faulted, timed_out, segfaulted, trapped, fault_offset: 0 }
    }

    pub fn run_observed(&self, opcode: u32) -> ObservedProbeResult {
        self.run_observed_with_prefix(opcode, &[], &[])
    }

    pub fn run_observed_with_prefix(&self, opcode: u32, prefix: &[u32], suffix: &[u32]) -> ObservedProbeResult {
        self.run_observed_with_prefix_ext(opcode, prefix, suffix, &[])
    }

    pub fn run_observed_with_prefix_ext(&self, opcode: u32, prefix: &[u32], suffix: &[u32], gpr_overrides: &[(u8, u64)]) -> ObservedProbeResult {
        let buf_pre = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();
        unsafe {
            *buf_pre.as_mut_ptr() = SnapshotBuffer::new();
            *buf_post.as_mut_ptr() = SnapshotBuffer::new();
        }
        self.page.make_writable();
        let mut off = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, false, gpr_overrides);
        for &inst in prefix { self.page.write_instruction(off, inst); off += 4; }
        self.page.write_instruction(off, opcode); off += 4;
        for &inst in suffix { self.page.write_instruction(off, inst); off += 4; }
        emitter::emit_postlude(&self.page, off, buf_post.as_mut_ptr() as *mut u8, buf_pre.as_mut_ptr() as *mut u8, false);
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
                #[cfg(target_os = "macos")]
                accelerate::wake_hardware();
                self.page.call_void();
                libc::_exit(0);
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
                } else { libc::waitpid(pid, &mut status, 0); }
                if !timed_out && libc::WIFSIGNALED(status) {
                    match libc::WTERMSIG(status) {
                        libc::SIGILL => faulted = true,
                        libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                        libc::SIGTRAP => trapped = true,
                        _ => faulted = true,
                    }
                }
            } else { panic!("fork failed"); }
        }
        let base = ProbeResult { opcode, faulted, timed_out, segfaulted, trapped, fault_offset: 0 };
        let post = if faulted || timed_out || segfaulted || trapped { None } else { buf_post.get().to_snapshot() };
        let diff = match &post {
            Some(post_snap) => seeded_snapshot().diff(post_snap).into_iter().filter(|d| d.index < 28).collect(),
            _ => Vec::new(),
        };
        ObservedProbeResult { base, pre: Some(seeded_snapshot()), post: post.clone(), diff, snapshot_corrupted: !buf_pre.get().canaries_intact() || !buf_post.get().canaries_intact(), gprs_post: post }
    }

    pub fn run_observed_streaming(&self, opcode: u32) -> ObservedProbeResult {
        self.run_observed_streaming_with_overrides(opcode, &[])
    }

    pub fn run_observed_streaming_with_overrides(&self, opcode: u32, gpr_overrides: &[(u8, u64)]) -> ObservedProbeResult {
        let buf_pre = SharedMemory::<SnapshotBuffer>::new();
        let buf_post = SharedMemory::<SnapshotBuffer>::new();
        unsafe {
            *buf_pre.as_mut_ptr() = SnapshotBuffer::new();
            *buf_post.as_mut_ptr() = SnapshotBuffer::new();
        }
        self.page.make_writable();
        let opcode_offset = emitter::emit_prelude(&self.page, buf_pre.as_mut_ptr() as *mut u8, true, gpr_overrides);
        self.page.write_instruction(opcode_offset, opcode);
        emitter::emit_postlude(&self.page, opcode_offset + 4, buf_post.as_mut_ptr() as *mut u8, buf_pre.as_mut_ptr() as *mut u8, true);
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
                #[cfg(target_os = "macos")]
                {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static WOKEN: AtomicBool = AtomicBool::new(false);
                    if !WOKEN.swap(true, Ordering::SeqCst) { accelerate::wake_hardware(); }
                }
                self.page.call_void();
                libc::_exit(0);
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
                } else { libc::waitpid(pid, &mut status, 0); }
                if !timed_out && libc::WIFSIGNALED(status) {
                    match libc::WTERMSIG(status) {
                        libc::SIGILL => faulted = true,
                        libc::SIGSEGV | libc::SIGBUS => segfaulted = true,
                        libc::SIGTRAP => trapped = true,
                        _ => faulted = true,
                    }
                }
            } else { panic!("fork failed"); }
        }
        let base = ProbeResult { opcode, faulted, timed_out, segfaulted, trapped, fault_offset: 0 };
        let post = if faulted || timed_out || segfaulted || trapped { None } else { buf_post.get().to_snapshot() };
        let diff = match &post {
            Some(post_snap) => seeded_snapshot().diff(post_snap).into_iter().filter(|d| d.index < 28).collect(),
            _ => Vec::new(),
        };
        ObservedProbeResult { base, pre: Some(seeded_snapshot()), post: post.clone(), diff, snapshot_corrupted: !buf_pre.get().canaries_intact() || !buf_post.get().canaries_intact(), gprs_post: post }
    }

    pub fn observed_sweep_streaming(&self, opcodes: impl Iterator<Item = u32>, sink: &mut crate::sink::ResultSink, progress_interval: usize) -> (SweepSummary, bool) {
        self.observed_sweep_streaming_with_overrides(opcodes, sink, progress_interval, &[])
    }

    pub fn observed_sweep_streaming_with_overrides(&self, opcodes: impl Iterator<Item = u32>, sink: &mut crate::sink::ResultSink, progress_interval: usize, gpr_overrides: &[(u8, u64)]) -> (SweepSummary, bool) {
        use crate::signal_handler::was_interrupted;
        let start = Instant::now();
        let (mut ok, mut faulted, mut timed_out, mut segfaulted, mut trapped, mut total) = (0, 0, 0, 0, 0, 0);
        let mut interrupted = false;
        for opcode in opcodes {
            if was_interrupted() { interrupted = true; break; }
            let result = self.run_observed_streaming_with_overrides(opcode, gpr_overrides);
            if result.base.timed_out { timed_out += 1; }
            else if result.base.faulted { faulted += 1; }
            else if result.base.segfaulted { segfaulted += 1; }
            else if result.base.trapped { trapped += 1; }
            else { ok += 1; }
            let _ = sink.write(&result);
            total += 1;
            if progress_interval > 0 && total % progress_interval == 0 {
                let elapsed = start.elapsed();
                let rate = total as f64 / elapsed.as_secs_f64();
                eprint!("\r  {total} probed ({ok} ok, {faulted} SIGILL, {segfaulted} SEGV, {trapped} TRAP, {timed_out} TIMEOUT) [{rate:.0} ops/sec]");
            }
        }
        if total > 0 { eprintln!(); }
        let _ = sink.flush();
        (SweepSummary { total, ok, faulted, timed_out, segfaulted, trapped, elapsed: start.elapsed() }, interrupted)
    }

    pub fn observed_sweep(&self, opcodes: impl Iterator<Item = u32>, sink: &mut crate::sink::ResultSink, progress_interval: usize) -> (SweepSummary, bool) {
        use crate::signal_handler::was_interrupted;
        let start = Instant::now();
        let (mut ok, mut faulted, mut timed_out, mut segfaulted, mut trapped, mut total) = (0, 0, 0, 0, 0, 0);
        let mut interrupted = false;
        for opcode in opcodes {
            if was_interrupted() { interrupted = true; break; }
            let result = self.run_observed(opcode);
            if result.base.timed_out { timed_out += 1; }
            else if result.base.faulted { faulted += 1; }
            else if result.base.segfaulted { segfaulted += 1; }
            else if result.base.trapped { trapped += 1; }
            else { ok += 1; }
            let _ = sink.write(&result);
            total += 1;
            if progress_interval > 0 && total % progress_interval == 0 {
                let elapsed = start.elapsed();
                let rate = total as f64 / elapsed.as_secs_f64();
                eprint!("\r  {total} probed ({ok} ok, {faulted} SIGILL, {segfaulted} SEGV, {trapped} TRAP, {timed_out} TIMEOUT) [{rate:.0} ops/sec]");
            }
        }
        if total > 0 { eprintln!(); }
        let _ = sink.flush();
        (SweepSummary { total, ok, faulted, timed_out, segfaulted, trapped, elapsed: start.elapsed() }, interrupted)
    }
}

impl fmt::Debug for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Probe {{ page: {:?} }}", self.page) }
}
