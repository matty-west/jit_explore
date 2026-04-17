//! Emits AArch64 instruction sequences into a [`JitPage`].
//!
//! The key sequences are the **prelude** and **postlude** that surround a
//! probed opcode to capture register state before and after execution.
//!
//! ## Buffer layout for an observed probe
//!
//! ```text
//! Offset  Instruction
//! ──────  ──────────────────────────────────────────
//!   0x00  PRELUDE: save caller state, load seeds
//!   ...   (variable length)
//!   N     PROBED OPCODE (1 instruction, 4 bytes)
//!   N+4   POSTLUDE: save post-probe state, restore, RET
//!   ...   (variable length)
//! ```
//!
//! ## Register convention
//!
//! - **X28** is reserved as the scratch register. It holds a pointer to the
//!   [`SnapshotBuffer`] pair during the prelude/postlude.
//! - All other GPRs (x0–x27) are loaded with deterministic seed values before
//!   the probed opcode runs, and dumped afterward.
//! - **X29** (FP) and **X30** (LR) are saved/restored to ensure RET works.

use crate::cpu_state::{SnapshotBuffer, seed_value};
use crate::jit_page::JitPage;

// --- AArch64 encoding helpers ---

/// AArch64 RET — return to caller via x30 (LR).
const RET: u32 = 0xD65F_03C0;

// --- SME enable / disable ---

/// `SMSTART` — enable **both** streaming SVE mode (SM) and ZA tile storage.
///
/// Encoding: `MSR SVCRSMZA, #3` → `0xD503_477F`.
///
/// After this instruction, both SVE streaming instructions and ZA tile
/// operations (FMOPA, etc.) can execute.
/// Must be paired with [`SMSTOP`] to restore normal mode.
pub const SMSTART: u32 = 0xD503_477F;

/// `SMSTOP` — disable **both** streaming SVE mode (SM) and ZA tile storage.
///
/// Encoding: `MSR SVCRSMZA, #0` → `0xD503_467F`.
/// Returns the CPU to non-streaming mode with ZA disabled.
pub const SMSTOP: u32 = 0xD503_467F;

/// `SMSTART SM` — enable streaming SVE mode only (no ZA).
///
/// Encoding: `MSR SVCRSM, #1` → `0xD503_437F`.
#[allow(dead_code)]
pub const SMSTART_SM: u32 = 0xD503_437F;

/// `SMSTOP SM` — disable streaming SVE mode only (no ZA).
///
/// Encoding: `MSR SVCRSM, #0` → `0xD503_427F`.
#[allow(dead_code)]
pub const SMSTOP_SM: u32 = 0xD503_427F;

/// `SMSTART ZA` — enable ZA tile storage only (no streaming mode).
///
/// Encoding: `MSR SVCRZA, #1` → `0xD503_457F`.
#[allow(dead_code)]
pub const SMSTART_ZA: u32 = 0xD503_457F;

/// `SMSTOP ZA` — disable ZA tile storage only.
///
/// Encoding: `MSR SVCRZA, #0` → `0xD503_447F`.
#[allow(dead_code)]
pub const SMSTOP_ZA: u32 = 0xD503_447F;

/// Encode `STP Xt1, Xt2, [Xn, #imm7*8]` (64-bit, offset variant).
///
/// Stores two 64-bit registers to `[Xn + imm7*8]`.
///
/// Encoding: `10 101 0 010 0 <imm7:7> <Rt2:5> <Rn:5> <Rt:5>`
///
/// `imm7` is a signed 7-bit value scaled by 8 (range: -512..+504, step 8).
///
/// # Panics
/// Panics if `offset` is not a multiple of 8, or out of the ±512 range.
const fn encode_stp_x(rt: u8, rt2: u8, rn: u8, offset: i16) -> u32 {
    assert!(offset % 8 == 0, "STP offset must be multiple of 8");
    assert!(offset >= -512 && offset <= 504, "STP offset out of range");
    let imm7 = ((offset / 8) as u32) & 0x7F;
    0xA900_0000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode `LDP Xt1, Xt2, [Xn, #imm7*8]` (64-bit, offset variant).
///
/// Loads two 64-bit registers from `[Xn + imm7*8]`.
///
/// Encoding: `10 101 0 010 1 <imm7:7> <Rt2:5> <Rn:5> <Rt:5>`
#[allow(dead_code)]
const fn encode_ldp_x(rt: u8, rt2: u8, rn: u8, offset: i16) -> u32 {
    assert!(offset % 8 == 0, "LDP offset must be multiple of 8");
    assert!(offset >= -512 && offset <= 504, "LDP offset out of range");
    let imm7 = ((offset / 8) as u32) & 0x7F;
    0xA940_0000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode `MOVZ Xd, #imm16, LSL #shift` — move wide immediate, zeroing.
///
/// Encoding: `1 10 100101 <hw:2> <imm16:16> <Rd:5>`
///
/// `shift` must be 0, 16, 32, or 48.
const fn encode_movz_x(rd: u8, imm16: u16, shift: u8) -> u32 {
    assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = (shift / 16) as u32;
    0xD280_0000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// Encode `MOVK Xd, #imm16, LSL #shift` — move wide immediate, keeping.
///
/// Encoding: `1 11 100101 <hw:2> <imm16:16> <Rd:5>`
///
/// Inserts `imm16` at the specified 16-bit lane without clearing other bits.
const fn encode_movk_x(rd: u8, imm16: u16, shift: u8) -> u32 {
    assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = (shift / 16) as u32;
    0xF280_0000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// Encode `STR Xd, [Xn, #imm12*8]` (unsigned offset variant).
///
/// Encoding: `11 111 00100 <imm12:12> <Rn:5> <Rt:5>`
const fn encode_str_x_uoff(rt: u8, rn: u8, offset: u16) -> u32 {
    assert!(offset % 8 == 0, "STR offset must be multiple of 8");
    let imm12 = (offset / 8) as u32;
    assert!(imm12 < 4096, "STR unsigned offset out of range");
    0xF900_0000 | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode `LDR Xd, [Xn, #imm12*8]` (unsigned offset variant).
///
/// Encoding: `11 111 00101 <imm12:12> <Rn:5> <Rt:5>`
const fn encode_ldr_x_uoff(rt: u8, rn: u8, offset: u16) -> u32 {
    assert!(offset % 8 == 0, "LDR offset must be multiple of 8");
    let imm12 = (offset / 8) as u32;
    assert!(imm12 < 4096, "LDR unsigned offset out of range");
    0xF940_0000 | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode `MOV Xd, SP` (alias: `ADD Xd, SP, #0`).
///
/// Encoding: `1 00 10001 00 000000000000 11111 <Rd:5>`
#[allow(dead_code)]
const fn encode_mov_from_sp(rd: u8) -> u32 {
    0x9100_03E0 | (rd as u32)
}

/// Encode `MOV SP, Xn` (alias: `ADD SP, Xn, #0`).
///
/// Encoding: `1 00 10001 00 000000000000 <Rn:5> 11111`
#[allow(dead_code)]
const fn encode_mov_to_sp(rn: u8) -> u32 {
    0x9100_001F | ((rn as u32) << 5)
}

// --- Immediate loading ---

/// Emit instructions to load a full 64-bit immediate into register `rd`.
///
/// Uses MOVZ for the lowest non-zero 16-bit chunk, then MOVK for each
/// subsequent non-zero chunk. Returns the number of instructions emitted.
fn emit_load_imm64(page: &JitPage, offset: &mut usize, rd: u8, value: u64) -> usize {
    let mut count = 0;
    let mut first = true;

    for shift in (0..4).map(|i| i * 16u8) {
        let chunk = ((value >> shift) & 0xFFFF) as u16;
        if chunk == 0 && !first {
            continue; // skip zero chunks after the first MOVZ
        }
        if first {
            page.write_instruction(*offset, encode_movz_x(rd, chunk, shift));
            first = false;
        } else {
            page.write_instruction(*offset, encode_movk_x(rd, chunk, shift));
        }
        *offset += 4;
        count += 1;
    }

    // If value is 0, we still emitted one MOVZ with imm16=0.
    count
}

// --- Prelude / Postlude emission ---

/// The byte offset within [`SnapshotBuffer`] where the GPR array starts.
const GPRS_OFFSET: usize = SnapshotBuffer::gprs_offset();

/// The byte offset within [`SnapshotBuffer`] where the AMX state starts.
const AMX_OFFSET: usize = SnapshotBuffer::amx_offset();

/// Encode `AMX_ST_TILE_T(n)` — store AMX tile `n` to `[Xn]`.
///
/// Encoding: `0x0020_8000 | (tile_index << 0) | (rn << 5)`
///
/// This encoding is from community reverse-engineering (corsix/amx) and must
/// be validated before use — see Gate 9 AMX validation probe.
pub const fn encode_amx_store_tile(tile_index: u8, rn: u8) -> u32 {
    0x0020_8000 | (tile_index as u32) | ((rn as u32) << 5)
}

/// Emit the **prelude** sequence into the JIT page.
///
/// The prelude:
/// 1. Saves the caller's X28 into `buf_pre.gprs[28]` (so we can use X28 as scratch).
/// 2. Loads the `buf_pre` base address into X28.
/// 3. Dumps all GPRs (x0–x27, x29, x30) into `buf_pre.gprs[]`.
/// 4. Saves SP into `buf_pre.gprs[28]` position is already taken — we store SP
///    separately if needed. For now, SP is saved via the sigsetjmp path.
/// 5. Loads all GPRs (x0–x27) with deterministic seed values.
/// 6. Optionally enables streaming SVE mode + ZA tile storage (`SMSTART`) if
///    `streaming == true`. This must be matched by `streaming == true` in
///    [`emit_postlude`] so the CPU is back in normal mode before the postlude
///    runs its STP instructions.
///
/// Returns the byte offset of the next free instruction slot.
///
/// # Arguments
/// - `page`: The JIT page (must be in writable mode).
/// - `buf_pre_ptr`: Pointer to the `SnapshotBuffer` for the pre-probe dump.
/// - `streaming`: If `true`, emit `SMSTART` at the end of the prelude so the
///   probed instruction runs in streaming SVE mode with ZA enabled. Use
///   `false` for all non-SME sweeps (standard ALU, HINT, etc.) to avoid
///   false-positive SIGILLs caused by the streaming-mode instruction subset.
pub fn emit_prelude(page: &JitPage, buf_pre_ptr: *mut u8, streaming: bool) -> usize {
    let mut off = 0usize;
    
    // Step 1: Push X28, X30 to stack (pre-indexed: SP -= 16, then store).
    //   STP X28, X30, [SP, #-16]!
    //   Encoding: pre-index STP = 10 101 0 011 0 <imm7> <Rt2> <Rn=SP(31)> <Rt>
    const STP_X28_X30_PUSH: u32 = {
        // STP (pre-index): opc=10, 101, 0, 011, 0, imm7=-2 (i.e. -16/8), Rt2=30, Rn=31(SP), Rt=28
        let imm7 = ((-2i16 as u16) & 0x7F) as u32;
        0xA9800000 | (imm7 << 15) | (30 << 10) | (31 << 5) | 28
    };
    page.write_instruction(off, STP_X28_X30_PUSH);
    off += 4;

    // Step 2: Load buf_pre base address into X28.
    emit_load_imm64(page, &mut off, 28, buf_pre_ptr as u64);

    // Step 3: Dump X0–X27 to buf_pre.gprs[0..28] via STP pairs through X28.
    // STP Xi, Xi+1, [X28, #GPRS_OFFSET + i*8]
    for i in (0..28).step_by(2) {
        let pair_offset = (GPRS_OFFSET + i * 8) as i16;
        page.write_instruction(off, encode_stp_x(i as u8, (i + 1) as u8, 28, pair_offset));
        off += 4;
    }

    // Step 4: Recover original X28 and X30 from the stack into X9, X10.
    //   LDP X9, X10, [SP], #16   (post-index: load then SP += 16)
    //   Encoding: post-index LDP = 10 101 0 001 1 <imm7=2> <Rt2=10> <Rn=31(SP)> <Rt=9>
    const LDP_X9_X10_POP: u32 = {
        let imm7 = 2u32; // 2 * 8 = 16
        0xA8C00000 | (imm7 << 15) | (10 << 10) | (31 << 5) | 9
    };
    page.write_instruction(off, LDP_X9_X10_POP);
    off += 4;

    // Step 5: Store original X28 (now in X9) into buf_pre.gprs[28].
    let x28_offset = (GPRS_OFFSET + 28 * 8) as u16;
    page.write_instruction(off, encode_str_x_uoff(9, 28, x28_offset));
    off += 4;

    // Step 6: Store original X30/LR (now in X10) into buf_pre.gprs[30].
    let x30_offset = (GPRS_OFFSET + 30 * 8) as u16;
    page.write_instruction(off, encode_str_x_uoff(10, 28, x30_offset));
    off += 4;

    // Step 7: Store X29/FP into buf_pre.gprs[29].
    let x29_offset = (GPRS_OFFSET + 29 * 8) as u16;
    page.write_instruction(off, encode_str_x_uoff(29, 28, x29_offset));
    off += 4;

    // Step 8: Load seed values into X0–X27.
    // X28 keeps the buf_pre pointer — we DON'T seed it.
    for reg in 0..28u8 {
        emit_load_imm64(page, &mut off, reg, seed_value(reg));
    }

    // ── Step 9: Conditionally enable AMX/SME ──
    // Only emit SMSTART for streaming-mode sweeps. For standard ALU/HINT sweeps
    // we leave the CPU in normal mode — unconditional SMSTART would cause massive
    // false-positive SIGILLs because most standard instructions are illegal in
    // streaming SVE mode.
    if streaming {
        page.write_instruction(off, SMSTART);
        off += 4;
    }

    off
}

/// Emit the **postlude** sequence into the JIT page starting at `start_offset`.
///
/// The postlude:
/// 1. Dumps X0–X27 into `buf_post.gprs[]` via X28 (still holding buf_post ptr).
/// 2. Optionally captures AMX tile state and emits `SMSTOP` (when `streaming == true`).
/// 3. Restores X29, X30 from `buf_pre` (saved by prelude).
/// 4. Executes RET.
///
/// Returns the byte offset of the next free instruction slot (past the RET).
///
/// # Arguments
/// - `page`: The JIT page (must be in writable mode).
/// - `start_offset`: Byte offset where the postlude begins.
/// - `buf_post_ptr`: Pointer to the `SnapshotBuffer` for the post-probe dump.
/// - `buf_pre_ptr`: Pointer to the pre-probe buffer (to restore X29, X30).
/// - `streaming`: Must match the value passed to [`emit_prelude`]. When `true`,
///   the postlude stores AMX tiles T0–T7 to `buf_post.amx` and then emits
///   `SMSTOP` to return to normal mode before the STP restore sequence.
pub fn emit_postlude(
    page: &JitPage,
    start_offset: usize,
    buf_post_ptr: *mut u8,
    buf_pre_ptr: *mut u8,
    streaming: bool,
) -> usize {
    let mut off = start_offset;

    // ── Step 1: Save X0–X27 to buf_post.gprs[] ──
    // But X28 still points to buf_pre! We need to reload it with buf_post.
    //
    // Problem: reloading X28 clobbers the seed/probe value of X28.
    // But X28 was our scratch register and never seeded, so this is fine.
    //
    // However, we first need to save X0/X1 before we clobber them with the
    // pointer load. Use STP to push them to stack temporarily.

    // Push X0, X1 to stack.
    const STP_X0_X1_PUSH: u32 = {
        let imm7 = ((-2i16 as u16) & 0x7F) as u32;
        0xA9800000 | (imm7 << 15) | (1 << 10) | (31 << 5) | 0
    };
    page.write_instruction(off, STP_X0_X1_PUSH);
    off += 4;

    // Load buf_post base into X28.
    emit_load_imm64(page, &mut off, 28, buf_post_ptr as u64);

    // Recover X0, X1 from stack.
    const LDP_X0_X1_POP: u32 = {
        let imm7 = 2u32;
        0xA8C00000 | (imm7 << 15) | (1 << 10) | (31 << 5) | 0
    };
    page.write_instruction(off, LDP_X0_X1_POP);
    off += 4;

    // Now dump X0–X27 via STP pairs.
    for i in (0..28).step_by(2) {
        let pair_offset = (GPRS_OFFSET + i * 8) as i16;
        page.write_instruction(off, encode_stp_x(i as u8, (i + 1) as u8, 28, pair_offset));
        off += 4;
    }

    // ── Step 2: Conditionally capture AMX state and disable streaming mode ──
    // Only emit the AMX tile stores + SMSTOP when `streaming == true`.
    // In non-streaming mode the CPU has no ZA state and these instructions
    // would be undefined. The `streaming` flag must match what was passed to
    // `emit_prelude` so the SMSTART/SMSTOP pair is balanced.
    if streaming {
        // AMX tiles T0-T7 are stored to buf_post.amx[].
        // Each tile is 64x64 = 4KB. X28 currently holds buf_post_ptr.
        for tile in 0..8u8 {
            // Tile `tile` lives at AMX_OFFSET + tile*4096 within the buffer.
            let tile_offset = AMX_OFFSET + (tile as usize * 4096);

            // Load the tile destination address into X0 (already saved above).
            // X0 = buf_post_ptr + tile_offset
            emit_load_imm64(page, &mut off, 0, tile_offset as u64);

            // ADD X0, X0, X28  →  X0 = X28 (buf_post_ptr) + tile_offset
            // Encoding: 0x8B1C_0000
            page.write_instruction(off, 0x8B1C_0000);
            off += 4;

            page.write_instruction(off, encode_amx_store_tile(tile, 0));
            off += 4;
        }

        // SMSTOP — return CPU to non-streaming mode before the STP restore.
        page.write_instruction(off, SMSTOP);
        off += 4;
    }

    // ── Step 3: Restore X29, X30 from buf_pre ──
    // Load buf_pre base into X28 (scratch).
    emit_load_imm64(page, &mut off, 28, buf_pre_ptr as u64);

    // LDR X29, [X28, #gprs+29*8]
    let x29_offset = (GPRS_OFFSET + 29 * 8) as u16;
    page.write_instruction(off, encode_ldr_x_uoff(29, 28, x29_offset));
    off += 4;

    // LDR X30, [X28, #gprs+30*8]
    let x30_offset = (GPRS_OFFSET + 30 * 8) as u16;
    page.write_instruction(off, encode_ldr_x_uoff(30, 28, x30_offset));
    off += 4;

    // ── Step 3: RET ──
    page.write_instruction(off, RET);
    off += 4;

    off
}

/// Total estimated instruction count for the prelude + postlude.
///
/// Used to verify the JIT page is large enough.
/// Prelude: ~1 (push) + 4 (load ptr) + 14 (STP pairs) + 1 (pop) + 3 (STR) + 28*2 (seeds) ≈ 79
/// Postlude: ~1 (push) + 4 (load ptr) + 1 (pop) + 14 (STP) + 4 (load ptr) + 2 (LDR) + 1 (RET) ≈ 27
/// Total: ~106 instructions × 4 bytes = ~424 bytes, well within 4096.
#[allow(dead_code)]
pub const ESTIMATED_OVERHEAD_BYTES: usize = 512; // conservative estimate

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stp_encoding() {
        // STP X0, X1, [X28, #0]
        let enc = encode_stp_x(0, 1, 28, 0);
        // Expected: 10 101 0 010 0 0000000 00001 11100 00000
        // = 0xA900_0380 + rt2(1)<<10 = let's just verify it round-trips.
        assert_eq!(enc & 0xFFC0_0000, 0xA900_0000, "STP base encoding");
    }

    #[test]
    fn ldp_encoding() {
        // LDP X0, X1, [X28, #16]
        let enc = encode_ldp_x(0, 1, 28, 16);
        assert_eq!(enc & 0xFFC0_0000, 0xA940_0000, "LDP base encoding");
    }

    #[test]
    fn movz_encoding() {
        // MOVZ X5, #42, LSL #0
        let enc = encode_movz_x(5, 42, 0);
        assert_eq!(enc, 0xD280_0545, "MOVZ X5, #42");
    }

    #[test]
    fn movk_encoding() {
        // MOVK X5, #0xDEAD, LSL #48
        let enc = encode_movk_x(5, 0xDEAD, 48);
        assert_eq!(enc & 0xFFE0_0000, 0xF2E0_0000, "MOVK hw=3 base");
    }

    #[test]
    fn str_ldr_encoding() {
        // STR X9, [X28, #232]  (28*8 + 8 = 232)
        let enc = encode_str_x_uoff(9, 28, 232);
        assert_eq!(enc & 0xFFC0_0000, 0xF900_0000, "STR base encoding");

        // LDR X29, [X28, #240]
        let enc = encode_ldr_x_uoff(29, 28, 240);
        assert_eq!(enc & 0xFFC0_0000, 0xF940_0000, "LDR base encoding");
    }

    #[test]
    fn prelude_fits_in_page_non_streaming() {
        let page = JitPage::alloc(4096).expect("alloc");
        let mut buf = crate::cpu_state::SnapshotBuffer::new();
        page.make_writable();
        let end = emit_prelude(&page, buf.as_mut_ptr(), false);
        assert!(
            end < 4096 - 256,
            "prelude (non-streaming) used {end} bytes, not enough room for postlude"
        );
    }

    #[test]
    fn prelude_fits_in_page_streaming() {
        let page = JitPage::alloc(4096).expect("alloc");
        let mut buf = crate::cpu_state::SnapshotBuffer::new();
        page.make_writable();
        // Streaming adds 1 extra instruction (SMSTART).
        let end = emit_prelude(&page, buf.as_mut_ptr(), true);
        assert!(
            end < 4096 - 256,
            "prelude (streaming) used {end} bytes, not enough room for postlude"
        );
    }
}
