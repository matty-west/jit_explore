//! Safe wrapper around macOS `MAP_JIT` pages.
//!
//! This module handles the full lifecycle of a JIT-capable memory page:
//! 1. Allocate via `mmap` with `MAP_JIT`
//! 2. Toggle write/execute permissions via `pthread_jit_write_protect_np`
//! 3. Invalidate the instruction cache after writes
//! 4. Deallocate via `munmap` on drop

use std::fmt;

// --- FFI declarations ---

// NOTE: `pthread_jit_write_protect_np` and `sys_icache_invalidate` are not
// exposed by the `libc` crate as of 0.2.x. We declare them manually.
unsafe extern "C" {
    /// Toggle JIT page write protection for the current thread.
    /// - `false` → page is **writable** (not executable)
    /// - `true`  → page is **executable** (not writable)
    fn pthread_jit_write_protect_np(enabled: libc::c_int);

    /// Invalidate the instruction cache for a region of memory.
    /// Must be called after writing code and before executing it.
    fn sys_icache_invalidate(addr: *mut libc::c_void, len: libc::size_t);
}

// --- Constants ---

/// macOS `MAP_JIT` flag — allows toggling W^X permissions at runtime.
/// Not defined in the `libc` crate for all targets, so we define it here.
///
/// Value from `<sys/mman.h>`: `#define MAP_JIT 0x0800`
const MAP_JIT: libc::c_int = 0x0800;

// --- Error type ---

/// Errors that can occur during JIT page operations.
#[derive(Debug)]
pub enum JitError {
    /// `mmap` returned `MAP_FAILED`.
    MmapFailed(std::io::Error),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::MmapFailed(e) => write!(f, "mmap(MAP_JIT) failed: {e}"),
        }
    }
}

impl std::error::Error for JitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JitError::MmapFailed(e) => Some(e),
        }
    }
}

/// A page-aligned region of memory allocated with `MAP_JIT`.
///
/// Wraps the raw `mmap` pointer and manages the W^X lifecycle:
/// write mode → write instructions → executable mode → call.
///
/// Dropped automatically via `munmap`.
pub struct JitPage {
    /// Pointer to the start of the mmap'd region.
    ptr: *mut u8,
    /// Size of the allocation in bytes (always a whole number of pages).
    size: usize,
}

/// `JitPage` manages a raw mmap'd pointer that is not inherently `Send`/`Sync`.
/// We only access it from the thread that created it, and the W^X toggle is
/// per-thread, so this is safe for our single-threaded usage.
///
/// SAFETY: We never share the raw pointer across threads.
unsafe impl Send for JitPage {}

impl JitPage {
    /// Allocate a new JIT page of at least `size` bytes.
    ///
    /// The actual allocation will be rounded up to the system page size.
    /// The page starts in **executable** mode (W^X default after mmap).
    ///
    /// # Errors
    /// Returns [`JitError::MmapFailed`] if `mmap` fails.
    #[must_use = "leaking executable pages is a bad look"]
    pub fn alloc(size: usize) -> Result<Self, JitError> {
        let page_size = unsafe {
            // SAFETY: `sysconf(_SC_PAGESIZE)` is always safe to call and
            // returns the system page size (typically 16384 on Apple Silicon).
            libc::sysconf(libc::_SC_PAGESIZE)
        } as usize;

        // Round up to the nearest page boundary.
        let alloc_size = (size + page_size - 1) & !(page_size - 1);

        let ptr = unsafe {
            // SAFETY: We request anonymous, private, JIT-capable memory.
            // - `addr` is null (let the kernel choose).
            // - `PROT_READ | PROT_WRITE | PROT_EXEC` is required for MAP_JIT pages.
            // - `MAP_ANON | MAP_PRIVATE | MAP_JIT` gives us a private JIT page.
            // - `fd = -1` and `offset = 0` because it's anonymous.
            libc::mmap(
                std::ptr::null_mut(),
                alloc_size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_ANON | libc::MAP_PRIVATE | MAP_JIT,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(JitError::MmapFailed(std::io::Error::last_os_error()));
        }

        Ok(JitPage {
            ptr: ptr as *mut u8,
            size: alloc_size,
        })
    }

    /// Returns the base address of the JIT page.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Returns the size of the allocation in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Switch the current thread's JIT pages to **writable** mode.
    ///
    /// After this call, you can write machine code into the page,
    /// but you **must not** execute it until [`Self::make_executable`] is called.
    pub fn make_writable(&self) {
        // SAFETY: `pthread_jit_write_protect_np(0)` is safe to call at any time.
        // It affects only the calling thread's view of MAP_JIT pages.
        // Passing 0 (false) makes the pages writable.
        unsafe {
            pthread_jit_write_protect_np(0);
        }
    }

    /// Switch the current thread's JIT pages to **executable** mode
    /// and invalidate the instruction cache for this page.
    ///
    /// After this call, the page is executable but **not writable**.
    pub fn make_executable(&self) {
        // SAFETY: `pthread_jit_write_protect_np(1)` makes MAP_JIT pages executable.
        // It affects only the calling thread.
        unsafe {
            pthread_jit_write_protect_np(1);
        }

        // SAFETY: `sys_icache_invalidate` flushes the instruction cache for the
        // given range. `self.ptr` is valid for `self.size` bytes (from mmap).
        // This must be called after writing instructions and before executing them,
        // because AArch64 has separate I-cache and D-cache.
        unsafe {
            sys_icache_invalidate(self.ptr as *mut libc::c_void, self.size);
        }
    }

    /// Write a single 32-bit instruction at the given **byte offset**.
    ///
    /// # Panics
    /// Panics if `offset + 4 > self.size` (out of bounds) or if `offset`
    /// is not 4-byte aligned.
    pub fn write_instruction(&self, offset: usize, instruction: u32) {
        assert!(
            offset % 4 == 0,
            "instruction offset must be 4-byte aligned, got {offset}"
        );
        assert!(
            offset + 4 <= self.size,
            "write at offset {offset} would overflow page (size = {})",
            self.size
        );

        // SAFETY: `self.ptr` is valid for `self.size` bytes. We checked
        // alignment and bounds above. The page must be in writable mode
        // (caller's responsibility to call `make_writable` first).
        unsafe {
            let dest = self.ptr.add(offset) as *mut u32;
            dest.write(instruction);
        }
    }

    /// Read back the raw bytes at the given byte offset (4 bytes).
    ///
    /// Useful for verifying that a write stuck.
    ///
    /// # Panics
    /// Panics if `offset + 4 > self.size`.
    pub fn read_instruction(&self, offset: usize) -> u32 {
        assert!(
            offset + 4 <= self.size,
            "read at offset {offset} would overflow page (size = {})",
            self.size
        );

        // SAFETY: `self.ptr` is valid for `self.size` bytes. Bounds checked above.
        unsafe {
            let src = self.ptr.add(offset) as *const u32;
            src.read()
        }
    }

    // ── Execution ──────────────────────────────────────────────

    /// Call the JIT buffer as `extern "C" fn()` (no args, no return value).
    ///
    /// The page **must** be in executable mode ([`Self::make_executable`])
    /// and must contain a valid instruction sequence ending with `RET`.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The page is in executable mode.
    /// - The instruction sequence at offset 0 is valid and ends with `RET`.
    /// - No write is in progress on another thread.
    pub unsafe fn call_void(&self) {
        // SAFETY: The caller guarantees the page is executable and contains
        // a valid instruction sequence starting at `self.ptr`. We cast the
        // pointer to `extern "C" fn()` which matches the AArch64 calling
        // convention — `BLR` will set LR (x30) so `RET` returns here.
        unsafe {
            let f: extern "C" fn() = core::mem::transmute(self.ptr);
            f();
        }
    }

    /// Call the JIT buffer as `extern "C" fn() -> u64`.
    ///
    /// The return value is whatever the instruction sequence leaves in `x0`.
    ///
    /// # Safety
    /// Same requirements as [`Self::call_void`], plus the instruction
    /// sequence must leave a meaningful value in `x0`.
    pub unsafe fn call_ret_u64(&self) -> u64 {
        // SAFETY: Same as `call_void`, but we interpret x0 as the return
        // value per the AArch64 calling convention.
        unsafe {
            let f: extern "C" fn() -> u64 = core::mem::transmute(self.ptr);
            f()
        }
    }
}

impl Drop for JitPage {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by `mmap` and `self.size` is the
        // exact allocation size. We only call this once (in Drop).
        let ret = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.size) };
        if ret != 0 {
            eprintln!(
                "warning: munmap({:p}, {}) failed: {}",
                self.ptr,
                self.size,
                std::io::Error::last_os_error()
            );
        }
    }
}

impl fmt::Debug for JitPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JitPage {{ ptr: {:p}, size: {} (0x{:X}) }}",
            self.ptr, self.size, self.size
        )
    }
}

impl fmt::Display for JitPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JitPage @ {:p} ({} bytes)", self.ptr, self.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AArch64 RET instruction encoding.
    const RET_INST: u32 = 0xD65F_03C0;

    #[test]
    fn alloc_and_write_ret() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");

        // Page should be at least 4096 bytes (likely 16384 on Apple Silicon).
        assert!(page.size() >= 4096);
        assert!(!page.as_ptr().is_null());

        // Write a RET instruction.
        page.make_writable();
        page.write_instruction(0, RET_INST);

        // Flip to executable (also flushes icache).
        page.make_executable();

        // Read back and verify.
        let readback = page.read_instruction(0);
        assert_eq!(
            readback, RET_INST,
            "expected RET (0x{RET_INST:08X}), got 0x{readback:08X}"
        );
    }

    #[test]
    fn write_out_of_bounds_panics() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");
        page.make_writable();

        // Writing beyond the page should panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            page.write_instruction(page.size(), 0xDEAD_BEEF);
        }));
        assert!(result.is_err(), "expected panic on out-of-bounds write");
    }

    #[test]
    fn unaligned_write_panics() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");
        page.make_writable();

        // Writing at a non-4-byte-aligned offset should panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            page.write_instruction(2, 0xDEAD_BEEF);
        }));
        assert!(result.is_err(), "expected panic on unaligned write");
    }

    // ── Gate 2 tests: execute from JIT buffer ──

    #[test]
    fn execute_ret_returns_cleanly() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");

        page.make_writable();
        page.write_instruction(0, RET_INST);
        page.make_executable();

        // SAFETY: The page contains a single RET instruction at offset 0.
        // RET pops the return address from x30 (LR), which was set by the
        // call instruction — so control returns here.
        unsafe {
            page.call_void();
        }
        // If we reach this point, the round-trip worked.
    }

    // ── Gate 3 tests: execute instructions that produce values ──

    /// Encode `MOVZ Xd, #imm16` for use in tests.
    const fn encode_movz_x(rd: u8, imm16: u16) -> u32 {
        assert!(rd <= 30, "rd must be 0..=30");
        0xD280_0000 | ((imm16 as u32) << 5) | (rd as u32)
    }

    /// Encode `ADD Xd, Xn, Xm` for use in tests.
    const fn encode_add_x(rd: u8, rn: u8, rm: u8) -> u32 {
        assert!(rd <= 30 && rn <= 30 && rm <= 30);
        0x8B00_0000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
    }

    #[test]
    fn mov_x0_42_returns_42() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");

        page.make_writable();
        page.write_instruction(0, encode_movz_x(0, 42));
        page.write_instruction(4, RET_INST);
        page.make_executable();

        // SAFETY: MOVZ X0,#42 + RET — valid sequence, returns 42 in x0.
        let result = unsafe { page.call_ret_u64() };
        assert_eq!(result, 42);
    }

    #[test]
    fn add_two_immediates_returns_sum() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");

        page.make_writable();
        page.write_instruction(0, encode_movz_x(0, 10));
        page.write_instruction(4, encode_movz_x(1, 32));
        page.write_instruction(8, encode_add_x(0, 0, 1));
        page.write_instruction(12, RET_INST);
        page.make_executable();

        // SAFETY: 4-instruction sequence ending with RET. x0 = 10 + 32 = 42.
        let result = unsafe { page.call_ret_u64() };
        assert_eq!(result, 42);
    }

    #[test]
    fn call_with_args_add() {
        let page = JitPage::alloc(4096).expect("mmap should succeed");

        page.make_writable();
        page.write_instruction(0, encode_add_x(0, 0, 1)); // ADD X0, X0, X1
        page.write_instruction(4, RET_INST);
        page.make_executable();

        // SAFETY: ADD X0,X0,X1 + RET — caller passes args in x0, x1.
        let f: extern "C" fn(u64, u64) -> u64 = unsafe {
            core::mem::transmute(page.as_ptr())
        };
        assert_eq!(f(100, 23), 123);
        assert_eq!(f(0, 0), 0);
        assert_eq!(f(u64::MAX, 1), 0); // wrapping overflow
    }
}
