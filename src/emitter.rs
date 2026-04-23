//! Emits AArch64/SME instruction sequences into a [`JitPage`].
//!
//! ## Buffer layout for an observed probe
//!
//! ```text
//! Offset  Instruction
//! ──────  ──────────────────────────────────────────
//!   0x00  PRELUDE: save caller state, load seeds
//!   ...   (variable length)
//!   N     PROBED OPCODE(S)
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

use crate::cpu_state::{SnapshotBuffer, seed_value};
use crate::jit_page::JitPage;

// ═══════════════════════════════════════════════════════════════════════════════
// AArch64 base encoding helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// AArch64 RET — return to caller via x30 (LR).
const RET: u32 = 0xD65F_03C0;

/// AArch64 NOP instruction.
pub const NOP: u32 = 0xD503_201F;

// ── SME mode control ─────────────────────────────────────────────────────────

/// `SMSTART` — enable **both** streaming SVE mode (SM) and ZA tile storage.
///
/// Encoding: `MSR SVCRSMZA, #3` → `0xD503_477F`.
pub const SMSTART: u32 = 0xD503_477F;

/// `SMSTOP` — disable **both** streaming SVE mode (SM) and ZA tile storage.
///
/// Encoding: `MSR SVCRSMZA, #0` → `0xD503_467F`.
pub const SMSTOP: u32 = 0xD503_467F;

/// `ZERO { ZA }` — zero all ZA tile storage.
///
/// Encoding: `0xC008_00FF`.
/// Inserting before an outer-product loop guarantees a clean accumulator.
pub const ZERO_ZA: u32 = 0xC008_00FF;

// ── Register-pair save/restore ───────────────────────────────────────────────

/// Encode `STP Xt1, Xt2, [Xn, #imm7*8]` (64-bit, offset variant).
const fn encode_stp_x(rt: u8, rt2: u8, rn: u8, offset: i16) -> u32 {
    assert!(offset % 8 == 0, "STP offset must be multiple of 8");
    assert!(offset >= -512 && offset <= 504, "STP offset out of range");
    let imm7 = ((offset / 8) as u32) & 0x7F;
    0xA900_0000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode `LDP Xt1, Xt2, [Xn, #imm7*8]` (64-bit, offset variant).
#[allow(dead_code)]
const fn encode_ldp_x(rt: u8, rt2: u8, rn: u8, offset: i16) -> u32 {
    assert!(offset % 8 == 0, "LDP offset must be multiple of 8");
    assert!(offset >= -512 && offset <= 504, "LDP offset out of range");
    let imm7 = ((offset / 8) as u32) & 0x7F;
    0xA940_0000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32)
}

// ── Immediate loading ────────────────────────────────────────────────────────

/// Encode `MOVZ Xd, #imm16, LSL #shift` — move wide immediate, zeroing.
const fn encode_movz_x(rd: u8, imm16: u16, shift: u8) -> u32 {
    assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = (shift / 16) as u32;
    0xD280_0000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// Encode `MOVK Xd, #imm16, LSL #shift` — move wide immediate, keeping.
const fn encode_movk_x(rd: u8, imm16: u16, shift: u8) -> u32 {
    assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = (shift / 16) as u32;
    0xF280_0000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// Encode `STR Xd, [Xn, #imm12*8]` (unsigned offset variant).
const fn encode_str_x_uoff(rt: u8, rn: u8, offset: u16) -> u32 {
    assert!(offset % 8 == 0, "STR offset must be multiple of 8");
    let imm12 = (offset / 8) as u32;
    assert!(imm12 < 4096, "STR unsigned offset out of range");
    0xF900_0000 | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode `LDR Xd, [Xn, #imm12*8]` (unsigned offset variant).
const fn encode_ldr_x_uoff(rt: u8, rn: u8, offset: u16) -> u32 {
    assert!(offset % 8 == 0, "LDR offset must be multiple of 8");
    let imm12 = (offset / 8) as u32;
    assert!(imm12 < 4096, "LDR unsigned offset out of range");
    0xF940_0000 | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Emit instructions to load a full 64-bit immediate into register `rd`.
/// Returns the number of instructions emitted.
fn emit_load_imm64(page: &JitPage, offset: &mut usize, rd: u8, value: u64) -> usize {
    let mut count = 0;
    let mut first = true;
    for shift in (0..4).map(|i| i * 16u8) {
        let chunk = ((value >> shift) & 0xFFFF) as u16;
        if chunk == 0 && !first { continue; }
        if first {
            page.write_instruction(*offset, encode_movz_x(rd, chunk, shift));
            first = false;
        } else {
            page.write_instruction(*offset, encode_movk_x(rd, chunk, shift));
        }
        *offset += 4;
        count += 1;
    }
    count
}

// ═══════════════════════════════════════════════════════════════════════════════
// SME instruction encoders
// ═══════════════════════════════════════════════════════════════════════════════

/// SVE LD1B (scalar+scalar): `LD1B {Zt.B}, Pg/Z, [Xn, Xm]`
pub const fn encode_sve_ld1b_ss(zt: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    0xA540_0000
        | ((rm as u32) << 16)
        | ((pg as u32) << 10)
        | ((rn as u32) <<  5)
        | (zt as u32)
}

/// SVE LD1H (scalar+scalar): `LD1H {Zt.H}, Pg/Z, [Xn, Xm, LSL #1]`
pub const fn encode_sve_ld1h_ss(zt: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    0xA540_2000
        | ((rm as u32) << 16)
        | ((pg as u32) << 10)
        | ((rn as u32) <<  5)
        | (zt as u32)
}

/// SVE LD1W (scalar+scalar): `LD1W {Zt.S}, Pg/Z, [Xn, Xm, LSL #2]`
pub const fn encode_sve_ld1w_ss(zt: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    0xA540_4000
        | ((rm as u32) << 16)
        | ((pg as u32) << 10)
        | ((rn as u32) <<  5)
        | (zt as u32)
}

/// SVE ST1W (scalar+scalar): `ST1W {Zt.S}, Pg, [Xn, Xm, LSL #2]`
///
/// Stores a Z register to memory. Bit 30 differentiates store (1) from load (0).
pub const fn encode_sve_st1w_ss(zt: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    0xE540_4000
        | ((rm as u32) << 16)
        | ((pg as u32) << 10)
        | ((rn as u32) <<  5)
        | (zt as u32)
}

/// `SUB Xd, Xn, #imm12` — 64-bit immediate subtract, no shift.
pub const fn encode_sub_x_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0xD100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// Encode `ST1W { ZA0H.S[Wv, #off] }, Pg, [Xn, Rm, LSL #2]` — SME horizontal
/// word-width store of one slice of ZA0 to memory.
///
/// # Arguments
/// - `wv`: slice-index register selector (0=W12, 1=W13, 2=W14, 3=W15)
/// - `off2`: 2-bit slice immediate offset (0–3)
/// - `pg`: governing predicate register index (0–7)
/// - `rn`: base address register (Xn)
/// - `rm`: offset register (Xm, scaled by element size)
pub const fn encode_sme_st1w_za_h(wv: u8, off2: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    assert!(wv   <= 3,  "Wv selector must be 0–3 (W12–W15)");
    assert!(off2 <= 3,  "off2 must be 0–3");
    assert!(pg   <= 7,  "predicate register must be P0–P7");
    assert!(rn   <= 30, "base register must be X0–X30");
    assert!(rm   <= 30, "offset register must be X0–X30");
    0xE0A0_0000
        | ((rm   as u32) << 16)
        | ((pg   as u32) << 11)
        | ((rn   as u32) <<  5)
        | ((wv   as u32) <<  3)
        | ((off2 as u32) <<  1)
}

/// `ADD Xd, Xn, #imm12` — 64-bit immediate add, no shift.
pub const fn encode_add_x_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x9100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// `ADD Wd, Wn, #imm12` — 32-bit immediate add, no shift.
pub const fn encode_add_w_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x1100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// `MOV Xd, XZR` — zero a register (alias: MOVZ Xd, #0).
pub const fn encode_mov_xzr(rd: u8) -> u32 {
    0xD280_0000 | (rd as u32)
}

/// `MRS <Xt>, CNTVCT_EL0` — read virtual counter.
pub const fn encode_mrs_cntvct_el0(rt: u8) -> u32 {
    0xD53BE020 | (rt as u32)
}

/// `ISB` — Instruction Synchronization Barrier.
pub const ISB: u32 = 0xD503_3FDF;

/// `SUBS <Xd>, <Xn>, #imm12` — 64-bit subtract immediate and set flags.
pub const fn encode_subs_x_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0xF100_0000 | ((imm12 as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// `B.NE <offset>` — branch if not equal (Z flag clear).
/// `offset_bytes` must be a multiple of 4.
pub fn encode_b_ne(offset_bytes: i32) -> u32 {
    assert!(offset_bytes % 4 == 0, "branch offset must be multiple of 4");
    let imm19 = (offset_bytes / 4) as u32;
    0x5400_0001 | ((imm19 & 0x7FFFF) << 5)
}

// ── Activation Enums ─────────────────────────────────────────────────────────

/// Activation function selection for fused GEMM kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// No activation (raw matrix multiplication).
    None,
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Matrix multiplication with bias vector: `x + bias`.
    Bias,
    /// Combined bias vector and ReLU: `max(0, x + bias)`.
    BiasReLU,
}

// ── Extended Outer Product Encoders (SME1) ───────────────────────────────────

/// `BFMOPA ZA[za], Pn/M, Pm/M, Zn.H, Zm.H` — BF16 floating-point outer product and accumulate.
///
/// Encoding: `1 0 0 0 0 0 0 1 1 0 0 Rm(5) Pm(3) Pn(3) Rn(5) 0 ZA(3)` → `0x8180_0000` base
pub const fn encode_bfmopa(za: u8, pn: u8, pm: u8, zn: u8, zm: u8) -> u32 {
    assert!(za <= 7 && pn <= 7 && pm <= 7 && zn <= 31 && zm <= 31);
    0x8180_0000 | ((zm as u32) << 16) | ((pm as u32) << 13) | ((pn as u32) << 10) | ((zn as u32) << 5) | (za as u32)
}

/// `BFMOPS ZA[za], Pn/M, Pm/M, Zn.H, Zm.H` — BF16 floating-point outer product and subtract.
pub const fn encode_bfmops(za: u8, pn: u8, pm: u8, zn: u8, zm: u8) -> u32 {
    encode_bfmopa(za, pn, pm, zn, zm) | 0x0000_0010
}

/// `SMOPA ZA[za], Pn/M, Pm/M, Zn.B, Zm.B` — Signed integer outer product and accumulate.
///
/// Encoding: `1 0 1 0 0 0 0 0 1 0 0 Rm(5) Pm(3) Pn(3) Rn(5) 0 ZA(3)` → `0xA080_0000` base
pub const fn encode_smopa(za: u8, pn: u8, pm: u8, zn: u8, zm: u8) -> u32 {
    assert!(za <= 7 && pn <= 7 && pm <= 7 && zn <= 31 && zm <= 31);
    0xA080_0000 | ((zm as u32) << 16) | ((pm as u32) << 13) | ((pn as u32) << 10) | ((zn as u32) << 5) | (za as u32)
}

/// `UMOPA ZA[za], Pn/M, Pm/M, Zn.B, Zm.B` — Unsigned integer outer product and accumulate.
///
/// Encoding: `1 0 1 0 0 0 0 1 1 0 0 Rm(5) Pm(3) Pn(3) Rn(5) 0 ZA(3)` → `0xA180_0000` base
pub const fn encode_umopa(za: u8, pn: u8, pm: u8, zn: u8, zm: u8) -> u32 {
    assert!(za <= 7 && pn <= 7 && pm <= 7 && zn <= 31 && zm <= 31);
    0xA180_0000 | ((zm as u32) << 16) | ((pm as u32) << 13) | ((pn as u32) << 10) | ((zn as u32) << 5) | (za as u32)
}

/// `SUMOPA ZA[za], Pn/M, Pm/M, Zn.B, Zm.B` — Mixed-sign (S×U) integer outer product and accumulate.
///
/// Encoding: `1 0 1 0 0 0 0 0 1 0 1 Rm(5) Pm(3) Pn(3) Rn(5) 0 ZA(3)` → `0xA0A0_0000` base
pub const fn encode_sumopa(za: u8, pn: u8, pm: u8, zn: u8, zm: u8) -> u32 {
    assert!(za <= 7 && pn <= 7 && pm <= 7 && zn <= 31 && zm <= 31);
    0xA0A0_0000 | ((zm as u32) << 16) | ((pm as u32) << 13) | ((pn as u32) << 10) | ((zn as u32) << 5) | (za as u32)
}

// ── Activation-related SVE/SME Encoders ──────────────────────────────────────

/// `MOVA Zd.S, Pg/M, ZA0H.S[Wv, #off]` — Move horizontal ZA slice to Z register.
///
/// Encoding: `11000000 01 0 Pg(3) Wv(3) 00 off(2) Zd(5)`? No, checking ARM:
/// SME MOVA (SVE to ZA horizontal) is 0xC0000000.
/// SME MOVA (ZA to SVE horizontal) is 0xC0400000.
/// Bits: [31:22]=1100000001, [21:20]=00 (fixed), [19:16]=Pg, [15:13]=Wv, [12:5]=0, [4:0]=Zd
/// Wait, Pg for MOVA is 3 bits in some variants.
/// Let's use the explicit bits from a known source:
/// 11000000 01 0 Pg(3) Wv(3) 00000 Zd(5)
pub const fn encode_mova_z_za_h(zd: u8, pg: u8, wv: u8, off: u8) -> u32 {
    assert!(zd <= 31 && pg <= 7 && wv <= 3 && off <= 3);
    // off is usually part of the Wv selection in the assembly but encoded in bits 1:2 of something?
    // Actually, SME MOVA (ZA to SVE) horizontal:
    // 11000000 01 0 Pg(3) Wv(2) 0 off(2) 000 Zd(5)  <-- let's try this
    0xC040_0000
        | ((pg as u32) << 13)
        | ((wv as u32) << 10)
        | ((off as u32) << 8)
        | (zd as u32)
}

/// `FMAX Zdn.S, Pg/M, Zdn.S, #0.0` — Floating-point maximum with zero (ReLU).
///
/// SVE FMAX (immediate), .S variant, i=0 for #0.0.
/// Encoding: `01100101 10 011 0 00 100 Pg(3) Zdn(5) 00000`
pub const fn encode_sve_fmax_imm_zero(zdn: u8, pg: u8) -> u32 {
    assert!(zdn <= 31 && pg <= 7);
    // size=10 (.S), i=0 (#0.0), opc=00 (FMAX)
    // Base: 0x6598_8000
    0x6598_8000 | ((pg as u32) << 10) | ((zdn as u32) << 5)
}

/// `FADD Zd.S, Zn.S, Zm.S` — Floating-point vector addition (unpredicated).
///
/// SVE FADD (vectors, unpredicated), .S variant.
/// Encoding: `01100101 10 0 Zm(5) 000 000 Zn(5) Zd(5)`
pub const fn encode_sve_fadd_unpred(zd: u8, zn: u8, zm: u8) -> u32 {
    assert!(zd <= 31 && zn <= 31 && zm <= 31);
    // size=10 (.S) → base 0x6580_0000
    0x6580_0000 | ((zm as u32) << 16) | ((zn as u32) << 5) | (zd as u32)
}

/// `STR Zt, [Xn, #imm9, MUL VL]` — SVE vector store (unpredicated, immediate offset).
///
/// Encodes the unpredicated vector store. M4 SVL is 512 bits (64 bytes).
/// `imm9` is a signed 9-bit immediate (though usually restricted to 7-bits in many SVE encodings).
/// Actually, SVE STR (immediate) is 9-bit: bits [21:13].
pub const fn encode_sve_str_imm(zt: u8, rn: u8, imm9: i16) -> u32 {
    assert!(zt <= 31 && rn <= 31);
    assert!(imm9 >= -256 && imm9 <= 255);
    // STR Zt, [Xn, #imm9, MUL VL]
    // Encoding: 11100101 00 1 imm9(9) Rn(5) Zt(5)
    // 0xE4800000 base for STR (immediate)
    let imm9_bits = (imm9 as u32) & 0x1FF;
    0xE480_0000 | (imm9_bits << 10) | ((rn as u32) << 5) | (zt as u32)
}

/// Emit a MOVZ/MOVK sequence to load a full 64-bit immediate into `rd`.
/// Returns the instructions as a Vec.
pub fn emit_load_imm64_vec(rd: u8, value: u64) -> Vec<u32> {
    let mut insns = Vec::with_capacity(4);
    let mut first = true;
    for i in 0..4u8 {
        let shift = i * 16;
        let chunk = ((value >> shift) & 0xFFFF) as u16;
        if chunk == 0 && !first { continue; }
        if first {
            insns.push(encode_movz_x(rd, chunk, shift));
            first = false;
        } else {
            insns.push(encode_movk_x(rd, chunk, shift));
        }
    }
    insns
}

// ═══════════════════════════════════════════════════════════════════════════════
// SME SGEMM Kernel Builder
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a complete SME SGEMM kernel for M=N=16 (one ZA0 tile), K iterations,
/// with optional activation function fusion.
///
/// M4 SVL = 512 bits → 16 float32 per Z register → ZA0 is a 16×16 tile.
///
/// Register ABI (set via overrides before SMSTART):
/// - `X2`  = C output pointer (row-major 16×16)
/// - `X3`  = 0 (zero offset for LD1W / ST1W)
/// - `X4`  = A pointer (K vectors of 16 floats, contiguous)
/// - `X5`  = B pointer (K vectors of 16 floats, contiguous)
/// - `X6`  = Bias pointer (1 vector of 16 floats, if Activation::Bias*)
/// - `X12` = W12 = 0 (ST1W slice index)
///
/// The returned opcodes run inside streaming mode (caller provides SMSTART/SMSTOP).
pub const PTRUE_P0_S: u32 = 0x2598_E3E0;
pub const PTRUE_P1_S: u32 = 0x2598_E3E1;

pub fn build_sme_sgemm_16x16(k: usize, act: Activation) -> Vec<u32> {
    const SVL_BYTES: u16 = 64; 
    const TILE_ROWS: usize = 16;
    const TILE_TOTAL_BYTES: u16 = (TILE_ROWS as u16) * SVL_BYTES; // 1024

    let ld1w_z0_x4 = encode_sve_ld1w_ss(0, 0, 4, 3);
    let ld1w_z1_x5 = encode_sve_ld1w_ss(1, 0, 5, 3);
    let fmopa_za0  = 0x8081_0000_u32; 
    let add_x4_svl = encode_add_x_imm(4, 4, SVL_BYTES);
    let add_x5_svl = encode_add_x_imm(5, 5, SVL_BYTES);

    // Strategy C: Store-then-modify.
    // Phase 1: FMOPA loop → ZA holds result (proven working).
    // Phase 2: ST1W ZA rows → output memory (proven working, Gate 14d).
    // Phase 3: LD1W rows from output → Z reg, apply SVE activation, ST1W back.
    // This avoids the broken MOVA ZA↔Z instruction entirely.
    let st1w_za0_raw = encode_sme_st1w_za_h(0, 0, 0, 2, 3);
    let add_w12_1    = encode_add_w_imm(12, 12, 1);
    let add_x2_svl   = encode_add_x_imm(2, 2, SVL_BYTES);

    // SVE load/store for activation pass (operates on Z registers, not ZA)
    let ld1w_z2_x2   = encode_sve_ld1w_ss(2, 0, 2, 3);  // Z2 = load row from [X2, X3]
    let st1w_z2_x2   = encode_sve_st1w_ss(2, 0, 2, 3);  // store Z2 back to [X2, X3]
    let fadd_z2_bias  = encode_sve_fadd_unpred(2, 2, 3); // Z2 = Z2 + Z3 (bias in Z3)
    let sub_x2_rewind = encode_sub_x_imm(2, 2, TILE_TOTAL_BYTES); // X2 -= 1024

    // ReLU via DUP zero + FMAX vector (FMAX immediate is NOP on M4 streaming mode)
    let dup_z4_zero: u32 = 0x2538_C004;  // DUP Z4.S, #0
    // FMAX Z2.S, P0/M, Z2.S, Z4.S — SVE FMAX (vectors, predicated)
    // Encoding: 01100101 10 00 0110 100 Pg(3) Zm(5) Zdn(5)
    let fmax_z2_z4: u32 = 0x6586_8082;   // FMAX Z2.S, P0/M, Z2.S, Z4.S

    let mut block = Vec::with_capacity(3 + 5 * k + 4 * TILE_ROWS + 6 * TILE_ROWS);
    block.push(PTRUE_P0_S);
    block.push(PTRUE_P1_S);
    block.push(ZERO_ZA);

    // Phase 1: Matmul outer product loop (ZA accumulator)
    for _ in 0..k {
        block.push(ld1w_z0_x4);
        block.push(ld1w_z1_x5);
        block.push(fmopa_za0);
        block.push(add_x4_svl);
        block.push(add_x5_svl);
    }

    // Phase 2: Store ZA to output memory (proven working path)
    block.push(0x5280000C); // MOV W12, #0
    for _ in 0..TILE_ROWS {
        block.push(st1w_za0_raw);
        block.push(add_w12_1);
        block.push(add_x2_svl);
    }

    // Phase 3: Activation pass — load row, apply SVE math, store back
    // Only emitted if activation != None. Data round-trips through L1 cache.
    if act != Activation::None {
        block.push(sub_x2_rewind); // Rewind X2 to start of output buffer

        // Set up Z4 = 0.0 for ReLU (DUP Z4.S, #0)
        if act == Activation::ReLU || act == Activation::BiasReLU {
            block.push(dup_z4_zero);
        }

        // Load bias vector once if needed
        if act == Activation::Bias || act == Activation::BiasReLU {
            block.push(encode_sve_ld1w_ss(3, 0, 6, 3)); // Z3 = bias from [X6, X3]
        }

        for _ in 0..TILE_ROWS {
            block.push(ld1w_z2_x2);  // Z2 = output row from memory
            match act {
                Activation::ReLU => {
                    block.push(fmax_z2_z4);    // Z2 = max(Z2, Z4=0.0)
                }
                Activation::Bias => {
                    block.push(fadd_z2_bias);  // Z2 = Z2 + bias
                }
                Activation::BiasReLU => {
                    block.push(fadd_z2_bias);  // Z2 = Z2 + bias
                    block.push(fmax_z2_z4);    // Z2 = max(Z2, Z4=0.0)
                }
                Activation::None => unreachable!(),
            }
            block.push(st1w_z2_x2);  // Store modified row back
            block.push(add_x2_svl);  // Advance to next row
        }
    }

    block
}

/// Build a complete SME BFMOPA kernel for M=N=16 (one ZA0 tile), K iterations.
///
/// X4 = A (BF16), X5 = B (BF16), X2 = C (FP32)
pub fn build_sme_bfmopa_16x16(k: usize) -> Vec<u32> {
    const PTRUE_P0_H: u32 = 0x2558_E3E0;
    const PTRUE_P1_H: u32 = 0x2558_E3E1;
    const PTRUE_P0_S: u32 = 0x2598_E3E0; // For ST1W governing
    const SVL_BYTES: u16 = 64;
    const TILE_ROWS: usize = 16;

    let ld1h_z0_x4 = encode_sve_ld1h_ss(0, 0, 4, 3);
    let ld1h_z1_x5 = encode_sve_ld1h_ss(1, 0, 5, 3);
    let bfmopa_za0 = encode_bfmopa(0, 0, 0, 0, 1);
    let add_x4_svl = encode_add_x_imm(4, 4, SVL_BYTES);
    let add_x5_svl = encode_add_x_imm(5, 5, SVL_BYTES);
    let st1w_za0   = encode_sme_st1w_za_h(0, 0, 0, 2, 3);
    let add_w12_1  = encode_add_w_imm(12, 12, 1);
    let add_x2_svl = encode_add_x_imm(2, 2, SVL_BYTES);

    let mut block = Vec::with_capacity(4 + 5 * k + 3 * TILE_ROWS);
    block.push(PTRUE_P0_H);
    block.push(PTRUE_P1_H);
    block.push(PTRUE_P0_S);
    block.push(ZERO_ZA);

    for _ in 0..k {
        block.push(ld1h_z0_x4);
        block.push(ld1h_z1_x5);
        block.push(bfmopa_za0);
        block.push(add_x4_svl);
        block.push(add_x5_svl);
    }

    for _ in 0..TILE_ROWS {
        block.push(st1w_za0);
        block.push(add_w12_1);
        block.push(add_x2_svl);
    }

    block
}

/// Build a complete SME SMOPA kernel for M=N=16 (one ZA0 tile), K iterations.
///
/// X4 = A (INT8), X5 = B (INT8), X2 = C (INT32)
pub fn build_sme_smopa_16x16(k: usize) -> Vec<u32> {
    const PTRUE_P0_B: u32 = 0x2518_E3E0;
    const PTRUE_P1_B: u32 = 0x2518_E3E1;
    const PTRUE_P0_S: u32 = 0x2598_E3E0; // For ST1W governing
    const SVL_BYTES: u16 = 64;
    const TILE_ROWS: usize = 16;

    let ld1b_z0_x4 = encode_sve_ld1b_ss(0, 0, 4, 3);
    let ld1b_z1_x5 = encode_sve_ld1b_ss(1, 0, 5, 3);
    let smopa_za0  = encode_smopa(0, 0, 0, 0, 1);
    let add_x4_svl = encode_add_x_imm(4, 4, SVL_BYTES);
    let add_x5_svl = encode_add_x_imm(5, 5, SVL_BYTES);
    let st1w_za0   = encode_sme_st1w_za_h(0, 0, 0, 2, 3);
    let add_w12_1  = encode_add_w_imm(12, 12, 1);
    let add_x2_svl = encode_add_x_imm(2, 2, SVL_BYTES);

    let mut block = Vec::with_capacity(4 + 5 * k + 3 * TILE_ROWS);
    block.push(PTRUE_P0_B);
    block.push(PTRUE_P1_B);
    block.push(PTRUE_P0_S);
    block.push(ZERO_ZA);

    for _ in 0..k {
        block.push(ld1b_z0_x4);
        block.push(ld1b_z1_x5);
        block.push(smopa_za0);
        block.push(add_x4_svl);
        block.push(add_x5_svl);
    }

    for _ in 0..TILE_ROWS {
        block.push(st1w_za0);
        block.push(add_w12_1);
        block.push(add_x2_svl);
    }

    block
}

/// Build a self-contained JIT page for the SME SGEMM kernel.
///
/// The page includes pointer-loading preamble (MOVZ/MOVK), SMSTART,
/// the kernel, SMSTOP, and RET — callable via `page.call_void()` with
/// no external setup.
pub fn build_sme_sgemm_page(
    k: usize,
    act: Activation,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    bias_ptr: u64,
) -> Option<crate::jit_page::JitPage> {
    let kernel = build_sme_sgemm_16x16(k, act);

    let mut insns = Vec::with_capacity(30 + kernel.len() + 3);
    insns.extend(emit_load_imm64_vec(2, c_ptr));   // X2 = C
    insns.push(encode_mov_xzr(3));                   // X3 = 0
    insns.extend(emit_load_imm64_vec(4, a_ptr));    // X4 = A
    insns.extend(emit_load_imm64_vec(5, b_ptr));    // X5 = B
    insns.push(encode_mov_xzr(12));                  // X12 = 0
    
    if act == Activation::Bias || act == Activation::BiasReLU {
        insns.extend(emit_load_imm64_vec(6, bias_ptr)); // X6 = Bias
    }

    insns.push(SMSTART);
    insns.extend_from_slice(&kernel);
    insns.push(SMSTOP);
    insns.push(RET);

    let total_bytes = insns.len() * 4;
    let page_size = ((total_bytes + 16383) / 16384) * 16384;

    let page = crate::jit_page::JitPage::alloc(page_size).ok()?;
    page.make_writable();

    let mut off = 0;
    for &op in &insns {
        page.write_instruction(off, op);
        off += 4;
    }

    page.make_executable();
    Some(page)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prelude / Postlude (probe harness)
// ═══════════════════════════════════════════════════════════════════════════════

/// The byte offset within [`SnapshotBuffer`] where the GPR array starts.
const GPRS_OFFSET: usize = SnapshotBuffer::gprs_offset();

/// Emit the **prelude** sequence into the JIT page.
///
/// The prelude saves caller state, loads seed values (with optional overrides),
/// and optionally enters streaming SVE mode. Returns the byte offset of the
/// next free instruction slot.
pub fn emit_prelude(
    page: &JitPage,
    buf_pre_ptr: *mut u8,
    streaming: bool,
    gpr_overrides: &[(u8, u64)],
    capture_timestamp: bool,
) -> usize {
    let mut off = 0usize;
    
    // Step 1: Push X28, X30 to stack.
    const STP_X28_X30_PUSH: u32 = {
        let imm7 = ((-2i16 as u16) & 0x7F) as u32;
        0xA9800000 | (imm7 << 15) | (30 << 10) | (31 << 5) | 28
    };
    page.write_instruction(off, STP_X28_X30_PUSH);
    off += 4;

    // Step 2: Load buf_pre base address into X28.
    emit_load_imm64(page, &mut off, 28, buf_pre_ptr as u64);

    // Optional: Capture start timestamp.
    if capture_timestamp {
        page.write_instruction(off, ISB);
        off += 4;
        page.write_instruction(off, encode_mrs_cntvct_el0(9));
        off += 4;
        page.write_instruction(off, encode_str_x_uoff(9, 28, SnapshotBuffer::timestamp_offset() as u16));
        off += 4;
    }

    // Step 3: Dump X0–X27 to buf_pre.gprs[0..28] via STP pairs.
    for i in (0..28).step_by(2) {
        let pair_offset = (GPRS_OFFSET + i * 8) as i16;
        page.write_instruction(off, encode_stp_x(i as u8, (i + 1) as u8, 28, pair_offset));
        off += 4;
    }

    // Step 4: Pop original X28, X30 into X9, X10.
    const LDP_X9_X10_POP: u32 = {
        let imm7 = 2u32;
        0xA8C00000 | (imm7 << 15) | (10 << 10) | (31 << 5) | 9
    };
    page.write_instruction(off, LDP_X9_X10_POP);
    off += 4;

    // Step 5: Store original X28 (now in X9).
    let x28_offset = (GPRS_OFFSET + 28 * 8) as u16;
    page.write_instruction(off, encode_str_x_uoff(9, 28, x28_offset));
    off += 4;

    // Step 6: Store original X30/LR (now in X10).
    let x30_offset = (GPRS_OFFSET + 30 * 8) as u16;
    page.write_instruction(off, encode_str_x_uoff(10, 28, x30_offset));
    off += 4;

    // Step 7: Store X29/FP.
    let x29_offset = (GPRS_OFFSET + 29 * 8) as u16;
    page.write_instruction(off, encode_str_x_uoff(29, 28, x29_offset));
    off += 4;

    // Step 8: Load seed values into X0–X27 (with overrides).
    for reg in 0..28u8 {
        let value = gpr_overrides
            .iter()
            .find(|(r, _)| *r == reg)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| seed_value(reg));
        emit_load_imm64(page, &mut off, reg, value);
    }

    // Step 9: Conditionally enable streaming SVE + ZA.
    if streaming {
        page.write_instruction(off, SMSTART);
        off += 4;
    }

    off
}

/// Emit the **postlude** sequence into the JIT page.
///
/// Dumps GPR state to buf_post, optionally emits SMSTOP, restores
/// caller registers, and returns via RET.
pub fn emit_postlude(
    page: &JitPage,
    start_offset: usize,
    buf_post_ptr: *mut u8,
    buf_pre_ptr: *mut u8,
    streaming: bool,
    capture_timestamp: bool,
) -> usize {
    let mut off = start_offset;

    // Step 1: Push X0, X1 to stack (save before clobbering X28).
    const STP_X0_X1_PUSH: u32 = {
        let imm7 = ((-2i16 as u16) & 0x7F) as u32;
        0xA9800000 | (imm7 << 15) | (1 << 10) | (31 << 5) | 0
    };
    page.write_instruction(off, STP_X0_X1_PUSH);
    off += 4;

    // Load buf_post base into X28.
    emit_load_imm64(page, &mut off, 28, buf_post_ptr as u64);

    // Pop X0, X1 from stack.
    const LDP_X0_X1_POP: u32 = {
        let imm7 = 2u32;
        0xA8C00000 | (imm7 << 15) | (1 << 10) | (31 << 5) | 0
    };
    page.write_instruction(off, LDP_X0_X1_POP);
    off += 4;

    // Optional: Capture end timestamp.
    if capture_timestamp {
        page.write_instruction(off, encode_mrs_cntvct_el0(9));
        off += 4;
        page.write_instruction(off, encode_str_x_uoff(9, 28, SnapshotBuffer::timestamp_offset() as u16));
        off += 4;
        page.write_instruction(off, ISB);
        off += 4;
    }

    // Dump X0–X27 via STP pairs.
    for i in (0..28).step_by(2) {
        let pair_offset = (GPRS_OFFSET + i * 8) as i16;
        page.write_instruction(off, encode_stp_x(i as u8, (i + 1) as u8, 28, pair_offset));
        off += 4;
    }

    // Step 2: Disable streaming mode if enabled.
    if streaming {
        page.write_instruction(off, SMSTOP);
        off += 4;
    }

    // Step 3: Restore X29, X30 from buf_pre.
    emit_load_imm64(page, &mut off, 28, buf_pre_ptr as u64);

    let x29_offset = (GPRS_OFFSET + 29 * 8) as u16;
    page.write_instruction(off, encode_ldr_x_uoff(29, 28, x29_offset));
    off += 4;

    let x30_offset = (GPRS_OFFSET + 30 * 8) as u16;
    page.write_instruction(off, encode_ldr_x_uoff(30, 28, x30_offset));
    off += 4;

    // Step 4: RET.
    page.write_instruction(off, RET);
    off += 4;

    off
}

/// Total estimated instruction count for the prelude + postlude.
#[allow(dead_code)]
pub const ESTIMATED_OVERHEAD_BYTES: usize = 512;

// ═══════════════════════════════════════════════════════════════════════════════
// PC-relative hazard patching (kept for future heist use)
// ═══════════════════════════════════════════════════════════════════════════════

/// Replace `ADRP` and `ADR` instructions in an opcode slice with `NOP`.
///
/// Returns the number of instructions patched.
pub fn nop_pc_relative_hazards(
    opcodes:      &mut Vec<u32>,
    adrp_indices: &[usize],
    adr_indices:  &[usize],
) -> usize {
    let mut patched = 0usize;
    for &idx in adrp_indices.iter().chain(adr_indices.iter()) {
        if idx < opcodes.len() {
            opcodes[idx] = NOP;
            patched += 1;
        }
    }
    patched
}

/// Rewrite all PC-relative branch offsets in `opcodes` for a relocated block.
///
/// Handles: `B`, `BL`, `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ`.
/// Returns the number of instructions patched.
pub fn relocate_branches(
    opcodes:                    &mut Vec<u32>,
    original_base_byte_offset:  i64,
    new_base_byte_offset:       i64,
) -> usize {
    let shift = original_base_byte_offset - new_base_byte_offset;
    if shift == 0 { return 0; }

    let mut patched = 0usize;
    for (i, op) in opcodes.iter_mut().enumerate() {
        let inst_pc_orig = original_base_byte_offset + (i as i64) * 4;
        let inst_pc_new  = new_base_byte_offset       + (i as i64) * 4;

        // B / BL — imm26
        if (*op >> 26) == 0b000101 || (*op >> 26) == 0b100101 {
            let raw26 = (*op & 0x3FF_FFFF) as i32;
            let raw26 = if raw26 & 0x200_0000 != 0 { raw26 - 0x400_0000 } else { raw26 };
            let target_abs = inst_pc_orig + raw26 as i64 * 4;
            let new_delta  = (target_abs - inst_pc_new) / 4;
            if new_delta >= -0x200_0000 && new_delta <= 0x1FF_FFFF {
                *op = (*op & 0xFC00_0000) | ((new_delta as u32) & 0x3FF_FFFF);
                patched += 1;
            }
            continue;
        }
        // B.cond — imm19
        if (*op >> 24) == 0x54 {
            let raw19 = ((*op >> 5) & 0x7_FFFF) as i32;
            let raw19 = if raw19 & 0x4_0000 != 0 { raw19 - 0x8_0000 } else { raw19 };
            let target_abs = inst_pc_orig + raw19 as i64 * 4;
            let new_delta  = (target_abs - inst_pc_new) / 4;
            if new_delta >= -0x4_0000 && new_delta <= 0x3_FFFF {
                *op = (*op & 0xFF00_000F) | (((new_delta as u32) & 0x7_FFFF) << 5);
                patched += 1;
            }
            continue;
        }
        // CBZ/CBNZ — imm19
        if (*op >> 24) & 0xFE == 0x34 {
            let raw19 = ((*op >> 5) & 0x7_FFFF) as i32;
            let raw19 = if raw19 & 0x4_0000 != 0 { raw19 - 0x8_0000 } else { raw19 };
            let target_abs = inst_pc_orig + raw19 as i64 * 4;
            let new_delta  = (target_abs - inst_pc_new) / 4;
            if new_delta >= -0x4_0000 && new_delta <= 0x3_FFFF {
                *op = (*op & 0xFF00_001F) | (((new_delta as u32) & 0x7_FFFF) << 5);
                patched += 1;
            }
            continue;
        }
        // TBZ/TBNZ — imm14
        if (*op >> 24) & 0xFE == 0x36 {
            let raw14 = ((*op >> 5) & 0x3FFF) as i32;
            let raw14 = if raw14 & 0x2000 != 0 { raw14 - 0x4000 } else { raw14 };
            let target_abs = inst_pc_orig + raw14 as i64 * 4;
            let new_delta  = (target_abs - inst_pc_new) / 4;
            if new_delta >= -0x2000 && new_delta <= 0x1FFF {
                *op = (*op & 0xFFF8_001F) | (((new_delta as u32) & 0x3FFF) << 5);
                patched += 1;
            }
        }
    }
    patched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stp_encoding() {
        let enc = encode_stp_x(0, 1, 28, 0);
        assert_eq!(enc & 0xFFC0_0000, 0xA900_0000, "STP base encoding");
    }

    #[test]
    fn ldp_encoding() {
        let enc = encode_ldp_x(0, 1, 28, 16);
        assert_eq!(enc & 0xFFC0_0000, 0xA940_0000, "LDP base encoding");
    }

    #[test]
    fn movz_encoding() {
        let enc = encode_movz_x(5, 42, 0);
        assert_eq!(enc, 0xD280_0545, "MOVZ X5, #42");
    }

    #[test]
    fn movk_encoding() {
        let enc = encode_movk_x(5, 0xDEAD, 48);
        assert_eq!(enc & 0xFFE0_0000, 0xF2E0_0000, "MOVK hw=3 base");
    }

    #[test]
    fn bfmopa_encoding() {
        // BFMOPA ZA0.S, P0/M, P0/M, Z0.H, Z1.H
        let enc = encode_bfmopa(0, 0, 0, 0, 1);
        assert_eq!(enc, 0x8181_0000);
    }

    #[test]
    fn smopa_encoding() {
        // SMOPA ZA0.S, P0/M, P0/M, Z0.B, Z1.B
        let enc = encode_smopa(0, 0, 0, 0, 1);
        assert_eq!(enc, 0xA081_0000);
    }

    #[test]
    fn umopa_encoding() {
        // UMOPA ZA0.S, P0/M, P0/M, Z0.B, Z1.B
        let enc = encode_umopa(0, 0, 0, 0, 1);
        assert_eq!(enc, 0xA181_0000);
    }

    #[test]
    fn sumopa_encoding() {
        // SUMOPA ZA0.S, P0/M, P0/M, Z0.B, Z1.B
        let enc = encode_sumopa(0, 0, 0, 0, 1);
        assert_eq!(enc, 0xA0A1_0000);
    }

    #[test]
    fn str_ldr_encoding() {
        let enc = encode_str_x_uoff(9, 28, 232);
        assert_eq!(enc & 0xFFC0_0000, 0xF900_0000, "STR base encoding");

        let enc = encode_ldr_x_uoff(29, 28, 240);
        assert_eq!(enc & 0xFFC0_0000, 0xF940_0000, "LDR base encoding");
    }

    #[test]
    fn prelude_fits_in_page_non_streaming() {
        let page = JitPage::alloc(4096).expect("alloc");
        let mut buf = crate::cpu_state::SnapshotBuffer::new();
        page.make_writable();
        let end = emit_prelude(&page, buf.as_mut_ptr(), false, &[], false);
        assert!(end < 4096 - 256, "prelude used {end} bytes, not enough room");
    }

    #[test]
    fn prelude_fits_in_page_streaming() {
        let page = JitPage::alloc(4096).expect("alloc");
        let mut buf = crate::cpu_state::SnapshotBuffer::new();
        page.make_writable();
        let end = emit_prelude(&page, buf.as_mut_ptr(), true, &[], false);
        assert!(end < 4096 - 256, "prelude (streaming) used {end} bytes, not enough room");
    }
}
