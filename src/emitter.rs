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
    // Pg field is at bits 12–10 (empirically confirmed via probe sweep; Gate 27).
    // Silent for pg=0 (zero anywhere = zero), wrong for pg≥1 at bits 13–11.
    0xE0A0_0000
        | ((rm   as u32) << 16)
        | ((pg   as u32) << 10)
        | ((rn   as u32) <<  5)
        | ((wv   as u32) <<  3)
        | ((off2 as u32) <<  1)
}

/// `FMOPA ZAda.S, Pn/M, Pm/M, Zn.S, Zm.S` — FP32 outer-product accumulate.
///
/// ZAda ∈ 0..3 (ZA tile index), Pn/Pm ∈ 0..7 (row/col predicate).
/// Only ZA entries where both the row-predicate (Pn) and col-predicate (Pm) lane
/// are active get updated. Accumulates into the existing ZAda contents.
///
/// Field layout (base 0x8080_0000):
/// - bits 20..16 = Zm
/// - bits 15..13 = Pm
/// - bits 12..10 = Pn
/// - bits  9..5  = Zn
/// - bits   1..0 = ZAda
pub const fn encode_sme_fmopa(zada: u8, zn: u8, zm: u8, pn: u8, pm: u8) -> u32 {
    assert!(zada <= 3,  "ZAda must be 0–3 for FP32");
    assert!(zn   <= 31, "Zn must be 0–31");
    assert!(zm   <= 31, "Zm must be 0–31");
    assert!(pn   <= 7,  "Pn must be 0–7");
    assert!(pm   <= 7,  "Pm must be 0–7");
    0x8080_0000
        | ((zm   as u32) << 16)
        | ((pm   as u32) << 13)
        | ((pn   as u32) << 10)
        | ((zn   as u32) <<  5)
        | (zada  as u32)
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

/// `ADD Xd, Xn, Xm` — 64-bit register-register add (no shift).
pub const fn encode_add_x_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x8B00_0000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
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

/// `WHILELT Pd.S, Xn, Xm` — SVE generate predicate while less than (64-bit, signed).
///
/// ARM SVE WHILELT (predicate-as-counter) encoding:
/// bits [31:24] = 0010_0101 (SVE prefix)
/// bits [23:22] = 10 (size = .S)
/// bit  [21]    = 1 (fixed)
/// bits [20:16] = Rm
/// bits [15:13] = 000
/// bit  [12]    = sf = 1 (64-bit Xn/Xm)
/// bit  [11]    = U = 0 (signed)
/// bit  [10]    = lt = 1
/// bits [9:5]   = Rn
/// bit  [4]     = eq = 0 (LT, not LE)
/// bits [3:0]   = Pd
///
/// Verified against clang: `whilelt p0.s, x3, x2` → `0x25a2_1460`.
pub const fn encode_sve_whilelt_s(pd: u8, rn: u8, rm: u8) -> u32 {
    assert!(pd <= 7 && rn <= 31 && rm <= 31);
    0x25a0_1400 | ((rm as u32) << 16) | ((rn as u32) << 5) | (pd as u32)
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

// ── Activation-related SVE/SME Encoders ──────────────────────────────────────

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
    let fmopa_za0  = encode_sme_fmopa(0, 0, 1, 0, 0);
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

/// Encode `MOV Xd, Xn` (alias of ORR Xd, XZR, Xn).
pub const fn encode_mov_x(rd: u8, rn: u8) -> u32 {
    // ORR Xd, XZR, Xn → 0xAA000000 | (Xn << 16) | (XZR << 5) | Xd
    0xAA00_0000 | ((rn as u32) << 16) | (31 << 5) | (rd as u32)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gate 21: Tiled SGEMM Kernel Builder
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a tiled SME SGEMM kernel for arbitrary M×N×K (Gate 21).
///
/// Tiles the computation into 16×16 ZA tile blocks with a branched K-loop
/// for each tile. Supports fused activation functions.
///
/// **Data layout:**
/// - A: column-major M×K (`A[k*M + i]` = element at row i, column k)
/// - B: row-major K×N (`B[k*N + j]` = element at row k, column j)
/// - C: row-major M×N (output)
/// - Bias: N floats (one per output column)
///
/// **Constraints:**
/// - M, N must be positive multiples of 16
/// - M, N ≤ 128
/// - K must be 1..65535
///
/// **Register ABI** (designed for cached page calling convention):
/// - `X0`  = A pointer (passed at call time)
/// - `X1`  = C pointer (passed at call time)
/// - `X5`  = B pointer (baked into instruction stream)
/// - `X6`  = Bias pointer (baked, optional)
///
/// **Internal register usage:**
/// - `X2`  = C tile pointer (current output)
/// - `X3`  = 0 (zero offset for LD1W/ST1W scalar+scalar)
/// - `X4`  = A tile pointer (advances by M*4 per K iteration)
/// - `X7`  = B tile pointer (advances by N*4 per K iteration)
/// - `X8`  = K loop counter
/// - `X9`  = scratch (C tile save, offset computation)
/// - `X10` = A base pointer (preserved across tiles)
/// - `X11` = C base pointer (preserved across tiles)
/// - `X12` = W12 = ZA slice counter for ST1W
/// - `X13` = scratch for bias tile pointer
pub fn build_sme_tiled_sgemm(m: usize, n: usize, k: usize, act: Activation) -> Vec<u32> {
    assert!(m % 16 == 0 && m > 0, "M must be a positive multiple of 16, got {}", m);
    assert!(n % 16 == 0 && n > 0, "N must be a positive multiple of 16, got {}", n);
    assert!(k > 0 && k <= 65535, "K must be 1..65535, got {}", k);
    assert!(m <= 128 && n <= 128, "M,N must be ≤ 128, got {}×{}", m, n);

    const TILE: usize = 16;
    let tiles_m = m / TILE;
    let tiles_n = n / TILE;
    let a_k_stride = (m * 4) as u16;    // A: bytes between K-iterations (column stride)
    let b_k_stride = (n * 4) as u16;    // B: bytes between K-iterations (row stride)
    let c_row_stride = (n * 4) as u16;  // C: bytes between output rows

    // ── Pre-encode reusable instructions ──
    let ld1w_z0_x4  = encode_sve_ld1w_ss(0, 0, 4, 3);    // Z0 ← [X4, X3, LSL #2]
    let ld1w_z1_x7  = encode_sve_ld1w_ss(1, 0, 7, 3);    // Z1 ← [X7, X3, LSL #2]
    let fmopa_za0   = encode_sme_fmopa(0, 0, 1, 0, 0);
    let add_x4_a    = encode_add_x_imm(4, 4, a_k_stride); // X4 += M*4
    let add_x7_b    = encode_add_x_imm(7, 7, b_k_stride); // X7 += N*4
    let subs_x8_1   = encode_subs_x_imm(8, 8, 1);         // X8 -= 1, set flags
    let bne_kloop   = encode_b_ne(-6 * 4);                 // B.NE back 6 insns to LD1W Z0

    let st1w_za0    = encode_sme_st1w_za_h(0, 0, 0, 2, 3);
    let add_w12_1   = encode_add_w_imm(12, 12, 1);
    let add_x2_c    = encode_add_x_imm(2, 2, c_row_stride);

    let ld1w_z2_x2  = encode_sve_ld1w_ss(2, 0, 2, 3);
    let st1w_z2_x2  = encode_sve_st1w_ss(2, 0, 2, 3);
    let fadd_z2_bias = encode_sve_fadd_unpred(2, 2, 3);
    let dup_z4_zero = 0x2538_C004u32;    // DUP Z4.S, #0
    let fmax_z2_z4  = 0x6586_8082u32;    // FMAX Z2.S, P0/M, Z2.S, Z4.S

    let mut block = Vec::with_capacity(1024);

    // ── Preamble: save base pointers, set up constants ──
    block.push(encode_mov_x(10, 0));   // X10 = A_base (from X0)
    block.push(encode_mov_x(11, 1));   // X11 = C_base (from X1)
    block.push(encode_mov_xzr(3));     // X3 = 0 (zero offset)
    block.push(PTRUE_P0_S);           // P0 = all-true (.S)

    if act == Activation::ReLU || act == Activation::BiasReLU {
        block.push(dup_z4_zero);       // Z4 = 0.0 (ReLU zero vector, once)
    }

    // ── Tile loop (fully unrolled at JIT time) ──
    for ti in 0..tiles_m {
        for tj in 0..tiles_n {
            let a_tile_off = (ti * TILE * 4) as u16;  // max 7*64 = 448
            let b_tile_off = (tj * TILE * 4) as u16;  // max 7*64 = 448
            let c_tile_off = ti * TILE * n * 4 + tj * TILE * 4;

            // ── A tile pointer: X4 = A_base + ti*64 ──
            if a_tile_off > 0 {
                block.push(encode_add_x_imm(4, 10, a_tile_off));
            } else {
                block.push(encode_mov_x(4, 10));
            }

            // ── B tile pointer: X7 = B_base + tj*64 ──
            if b_tile_off > 0 {
                block.push(encode_add_x_imm(7, 5, b_tile_off));
            } else {
                block.push(encode_mov_x(7, 5));
            }

            // ── C tile pointer: X2 = C_base + c_tile_off ──
            if c_tile_off == 0 {
                block.push(encode_mov_x(2, 11));
            } else if c_tile_off <= 4095 {
                block.push(encode_add_x_imm(2, 11, c_tile_off as u16));
            } else {
                // Large offset: load into X9 then register-add
                block.extend(emit_load_imm64_vec(9, c_tile_off as u64));
                block.push(encode_add_x_reg(2, 11, 9));
            }

            // ── Load K counter ──
            block.push(encode_movz_x(8, k as u16, 0));

            // ── ZERO ZA ──
            block.push(ZERO_ZA);

            // ── K-loop (7 instructions, branch-based) ──
            // [0] LD1W Z0 ← A column slice
            // [1] LD1W Z1 ← B row slice
            // [2] FMOPA ZA0 += Z0 ⊗ Z1
            // [3] ADD X4 += M*4 (next A column)
            // [4] ADD X7 += N*4 (next B row)
            // [5] SUBS X8 -= 1
            // [6] B.NE → [0]
            block.push(ld1w_z0_x4);
            block.push(ld1w_z1_x7);
            block.push(fmopa_za0);
            block.push(add_x4_a);
            block.push(add_x7_b);
            block.push(subs_x8_1);
            block.push(bne_kloop);

            // ── Store ZA rows to C ──
            block.push(encode_mov_x(9, 2));     // X9 = save C_tile_start
            block.push(0x5280_000Cu32);         // MOV W12, #0

            for _row in 0..TILE {
                block.push(st1w_za0);
                block.push(add_w12_1);
                block.push(add_x2_c);
            }

            // ── Activation pass (if any) ──
            if act != Activation::None {
                block.push(encode_mov_x(2, 9));  // Restore X2 to C_tile_start

                // Load bias slice for this tile's columns
                if act == Activation::Bias || act == Activation::BiasReLU {
                    if b_tile_off > 0 {
                        block.push(encode_add_x_imm(13, 6, b_tile_off));
                        block.push(encode_sve_ld1w_ss(3, 0, 13, 3));
                    } else {
                        block.push(encode_sve_ld1w_ss(3, 0, 6, 3));
                    }
                }

                for _row in 0..TILE {
                    block.push(ld1w_z2_x2);
                    match act {
                        Activation::ReLU => {
                            block.push(fmax_z2_z4);
                        }
                        Activation::Bias => {
                            block.push(fadd_z2_bias);
                        }
                        Activation::BiasReLU => {
                            block.push(fadd_z2_bias);
                            block.push(fmax_z2_z4);
                        }
                        Activation::None => unreachable!(),
                    }
                    block.push(st1w_z2_x2);
                    block.push(add_x2_c);
                }
            }
        }
    }

    block
}

/// Build a **cached** JIT page for tiled SME SGEMM (Gate 21).
///
/// Immutable pointers (B weights, bias) are baked into the instruction stream.
/// Mutable pointers (A input, C output) are passed at call time via X0, X1.
///
/// Call via `page.call_with_args(a_ptr as u64, c_ptr as u64)`.
///
/// **Data layout:**
/// - A (X0): column-major M×K
/// - C (X1): row-major M×N (output)
/// - B (baked): row-major K×N
/// - Bias (baked): N floats
pub fn build_sme_tiled_sgemm_page_cached(
    m: usize,
    n: usize,
    k: usize,
    act: Activation,
    b_ptr: u64,
    bias_ptr: u64,
) -> Option<crate::jit_page::JitPage> {
    let kernel = build_sme_tiled_sgemm(m, n, k, act);

    let mut insns = Vec::with_capacity(20 + kernel.len() + 3);

    // Bake immutable pointers before SMSTART
    insns.extend(emit_load_imm64_vec(5, b_ptr));    // X5 = B (weights)
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
// Gate 23: Monolithic Fused Inference Kernel
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a monolithic inference kernel that chains multiple GEMM+activation layers
/// into a single JitPage with one SMSTART/SMSTOP pair.
///
/// Between intermediate layers, stores are emitted in **column-major** format
/// so the next layer's LD1W can read the data directly — no Rust transpose needed.
///
/// **Calling convention:**
/// - `X0` = input A pointer (column-major [K1 × 16])
/// - `X1` = final output C pointer (row-major [16 × N_last])
///
/// **Baked pointers:**
/// - Layer weights (W1, W2, W3) and biases (B1, B2, B3) are baked into the instruction stream.
/// - Intermediate buffers (buf1, buf2) are baked into the instruction stream.
///
/// **Register usage per-layer:**
/// - `X10` = A_base, `X11` = C_base
/// - `X4/X7` = A/B tile pointers, `X5` = B weights, `X6` = bias
/// - `X2` = C tile pointer, `X3` = 0, `X8` = K counter, `X9/X12/X13` = scratch
/// - `X14` = intermediate buf1, `X15` = intermediate buf2
pub struct MonolithicLayerConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub act: Activation,
    pub w_ptr: u64,   // B (weights) pointer, row-major [K×N]
    pub b_ptr: u64,   // Bias pointer, N floats
}

/// Build the inner kernel for one layer of a monolithic inference pipeline.
///
/// If `col_major_output` is true, the store phase writes in column-major format
/// [N×16] instead of row-major [16×N]. This allows the next layer to read A
/// directly without a transpose.
#[allow(dead_code)]
fn build_layer_kernel(
    m: usize, n: usize, k: usize,
    act: Activation,
    col_major_output: bool,
) -> Vec<u32> {
    assert!(m % 16 == 0 && m > 0);
    assert!(n % 16 == 0 && n > 0);
    assert!(k > 0 && k <= 65535);

    const TILE: usize = 16;
    let tiles_m = m / TILE;
    let tiles_n = n / TILE;
    let a_k_stride = (m * 4) as u16;    // A: column-major stride between K iterations
    let b_k_stride = (n * 4) as u16;    // B: row-major stride between K iterations

    // Output stride depends on layout:
    // Row-major: row stride = N*4 bytes (advance by N floats to next row)
    // Column-major: row stride = 4 bytes (each row = next float in same column)
    //   But for column-major [N×16], each ZA row i should go to offset i*4 within a column,
    //   and tile column tj starts at tj*16*4 = tj*64 bytes * TILE in the output.
    //   Actually: col-major [N×16] means element [i][j] is at offset j*N*4 + i*4.
    //   Wait no — we want the output to be read as A for the next layer in col-major [N×16].
    //   That means: A[col][row] at offset col*16*4 + row*4.
    //   So for ZA row `r` of tile (ti, tj):
    //     row in output = ti*16 + r
    //     col_start = tj*16
    //     For row-major: addr = base + (ti*16+r)*N*4 + tj*16*4
    //     For col-major [N×16]: addr = base + col*16*4 + row*4
    //       = base + (tj*16+c)*16*4 + (ti*16+r)*4
    //   But ST1W stores a whole ZA row (16 floats for columns tj*16..(tj+1)*16).
    //   In row-major these 16 floats are contiguous at offset (row*N + tj*16)*4.
    //   In column-major [N×16] they're at stride 16*4=64 bytes apart (one per column).
    //   → We can't use a single ST1W for non-contiguous column-major stores.

    // REVISED APPROACH: Column-major via ZA vertical slices.
    // Actually, we can just store row-major and then transpose.
    // OR: we store row-major intermediate, then do an SVE-based transpose.

    // SIMPLEST APPROACH that still eliminates Rust transposes:
    // Store intermediate results row-major [16×N], then emit an SVE transpose
    // sequence: for each column c, LD1W from row-major col c (stride N*4),
    // ST1W to col-major position c (contiguous 16 floats).
    //
    // But that's N SVE load-store pairs = 2*N instructions.
    // For N=48 that's 96 extra instructions. Not ideal but still inside the kernel.
    //
    // EVEN SIMPLER: Just store row-major, let the next layer handle the
    // column-major layout by adjusting its LD1W stride.
    //
    // ACTUALLY the cleanest approach: store row-major [16×N] to buf.
    // Next layer reads A from buf. A must be column-major [K×16] = [N×16].
    // A[k][m] at buf + k*16*4 + m*4 (column-major).
    // But we stored: C[m][n] at buf + m*N*4 + n*4 (row-major).
    // So A[k=n][m] = C[m][n]. Column-major A[n][m] needs to be at n*16*4 + m*4 = n*64 + m*4.
    // Row-major C[m][n] is at m*N*4 + n*4.
    // These are different unless N=16.
    //
    // For N=16: row-major [16×16] = column-major [16×16] transposed.
    //   C[m][n] at m*64 + n*4. A[n][m] at n*64 + m*4. Not the same.
    //
    // SO: we DO need to transpose. The question is whether to do it in SVE
    // instructions (inside streaming mode) or in Rust (outside).
    //
    // SVE transpose for [16×N] → [N×16]:
    //   For each column c in 0..N:
    //     Load 16 values from rows: addr = buf + c*4, stride = N*4
    //     Store 16 values contiguously: addr = buf2 + c*64
    //   That's a gather-load (non-contiguous) then a contiguous store.
    //   SVE LD1W with scalar+scalar won't do strided access.
    //   We'd need LD1W scalar+immediate with repeated ADD, or use gather loads.
    //
    // ALTERNATIVE: Use the ZA tile itself!
    //   After FMOPA, ZA[r][c] holds result[r][c].
    //   ST1W horizontal stores row r: 16 contiguous values = one row of C.
    //   ST1W vertical would store column c: 16 contiguous values = one row of A!
    //   → If we use ST1W *vertical* slices, we get column-major output for free!

    // YES! ST1W ZA vertical slice:
    //   `ST1W { ZA0V.S[Wv, #off] }, Pg, [Xn, Xm, LSL #2]`
    //   Stores column `off` of ZA as 16 contiguous float32s.
    //   If we store columns 0..15 contiguously (stride = 64 bytes between columns),
    //   we get the output in column-major [16×16] = exactly what LD1W reads as A!
    //
    // For multi-tile (N>16): tile (ti, tj) produces ZA[0..15][0..15].
    //   Row-major output tile address: base + ti*16*N*4 + tj*16*4 (row stride = N*4).
    //   Col-major output tile address: base + tj*16*16*4 + ti*16*4 (col stride = 16*4).
    //     → Store vertical slices to base + (tj*16+c)*16*4 + ti*16*4 + r*4
    //     → Base for tile (ti,tj) col-major: base + tj*16*64 + ti*64
    //     → Within tile: column c stored at offset c*64 (16 floats contiguous)
    //
    // THIS IS THE KEY INSIGHT: ST1W vertical slices give us column-major for free!

    let c_row_stride = if col_major_output {
        // Column-major [N×16]: stride between "rows" in vertical store = column stride
        // Actually for vertical ST1W, we advance the base by 16*4=64 per column
        (TILE * 4) as u16  // 64 bytes — stride between vertical slices
    } else {
        (n * 4) as u16     // Row-major: stride between rows = N*4
    };

    // Pre-encode reusable instructions
    let ld1w_z0_x4  = encode_sve_ld1w_ss(0, 0, 4, 3);    // Z0 ← A[X4]
    let ld1w_z1_x7  = encode_sve_ld1w_ss(1, 0, 7, 3);    // Z1 ← B[X7]
    let fmopa_za0   = encode_sme_fmopa(0, 0, 1, 0, 0);
    let add_x4_a    = encode_add_x_imm(4, 4, a_k_stride);
    let add_x7_b    = encode_add_x_imm(7, 7, b_k_stride);
    let subs_x8_1   = encode_subs_x_imm(8, 8, 1);
    let bne_kloop   = encode_b_ne(-6 * 4);

    // Store instruction: horizontal (row-major) or vertical (col-major)
    let st1w_za0 = if col_major_output {
        // ST1W { ZA0V.S[W12, #0] }, P0, [X2, X3, LSL #2]
        // Vertical variant: bit pattern differs from horizontal
        encode_sme_st1w_za_v(0, 0, 0, 2, 3)
    } else {
        encode_sme_st1w_za_h(0, 0, 0, 2, 3)
    };
    let add_w12_1   = encode_add_w_imm(12, 12, 1);
    let add_x2_c    = encode_add_x_imm(2, 2, c_row_stride);

    // Activation instructions
    let ld1w_z2_x2  = encode_sve_ld1w_ss(2, 0, 2, 3);
    let st1w_z2_x2  = encode_sve_st1w_ss(2, 0, 2, 3);
    let fadd_z2_bias = encode_sve_fadd_unpred(2, 2, 3);
    let dup_z4_zero = 0x2538_C004u32;
    let fmax_z2_z4  = 0x6586_8082u32;

    let mut block = Vec::with_capacity(2048);

    // Preamble
    block.push(encode_mov_x(10, 0));   // X10 = A_base
    block.push(encode_mov_x(11, 1));   // X11 = C_base
    block.push(encode_mov_xzr(3));     // X3 = 0
    block.push(PTRUE_P0_S);

    if act == Activation::ReLU || act == Activation::BiasReLU {
        block.push(dup_z4_zero);
    }

    // Tile loop (fully unrolled)
    for ti in 0..tiles_m {
        for tj in 0..tiles_n {
            let a_tile_off = (ti * TILE * 4) as u16;
            let b_tile_off = (tj * TILE * 4) as u16;

            let c_tile_off = if col_major_output {
                // Col-major [N×16]: tile (ti, tj) starts at tj*16*16*4 + ti*16*4
                tj * TILE * TILE * 4 + ti * TILE * 4
            } else {
                // Row-major [16×N]: tile (ti, tj) starts at ti*16*N*4 + tj*16*4
                ti * TILE * n * 4 + tj * TILE * 4
            };

            // A tile pointer
            if a_tile_off > 0 {
                block.push(encode_add_x_imm(4, 10, a_tile_off));
            } else {
                block.push(encode_mov_x(4, 10));
            }

            // B tile pointer
            if b_tile_off > 0 {
                block.push(encode_add_x_imm(7, 5, b_tile_off));
            } else {
                block.push(encode_mov_x(7, 5));
            }

            // C tile pointer
            if c_tile_off == 0 {
                block.push(encode_mov_x(2, 11));
            } else if c_tile_off <= 4095 {
                block.push(encode_add_x_imm(2, 11, c_tile_off as u16));
            } else {
                block.extend(emit_load_imm64_vec(9, c_tile_off as u64));
                block.push(encode_add_x_reg(2, 11, 9));
            }

            // K counter
            block.push(encode_movz_x(8, k as u16, 0));
            block.push(ZERO_ZA);

            // K-loop
            block.push(ld1w_z0_x4);
            block.push(ld1w_z1_x7);
            block.push(fmopa_za0);
            block.push(add_x4_a);
            block.push(add_x7_b);
            block.push(subs_x8_1);
            block.push(bne_kloop);

            // Store ZA rows/columns
            block.push(encode_mov_x(9, 2));
            block.push(0x5280_000Cu32);  // MOV W12, #0

            for _slice in 0..TILE {
                block.push(st1w_za0);
                block.push(add_w12_1);
                block.push(add_x2_c);
            }

            // Activation pass
            if act != Activation::None {
                block.push(encode_mov_x(2, 9));  // Restore C tile start

                if act == Activation::Bias || act == Activation::BiasReLU {
                    if b_tile_off > 0 {
                        block.push(encode_add_x_imm(13, 6, b_tile_off));
                    } else {
                        block.push(encode_mov_x(13, 6));
                    }
                }

                // For col-major output, activation operates on vertical slices
                // (16 contiguous floats per column), same stride as store
                if col_major_output && (act == Activation::Bias || act == Activation::BiasReLU) {
                    // Need to broadcast each bias element for the column
                    // Actually bias[tj*16 + c] applies to column c.
                    // In col-major store, each 16-float chunk IS one column.
                    // So LD1W Z2 loads column c (16 values), and we add bias[c] as scalar.
                    // But our FADD is vector: Z2 = Z2 + Z3 where Z3 = bias vector.
                    // For col-major: each LD1W loads 16 values from the SAME column c,
                    // so they all need bias[c] added (a scalar broadcast).
                    //
                    // Hmm, the existing approach adds the bias vector to each row.
                    // For col-major, each stored chunk is a column, not a row.
                    // All elements in that chunk belong to the same output column c,
                    // so they all get bias[c].
                    //
                    // We need: DUP Z3.S, bias[c] for each column c.
                    // Or: load bias into Z3 once, then use INDEX to extract element c.
                    //
                    // SIMPLEST: load bias vector into Z3, then for each column c:
                    //   - LD1W Z2 from col-major chunk c
                    //   - DUP Z5.S, Z3.S[c]  -- broadcast element c
                    //   - FADD Z2, Z2, Z5
                    //   - (optional FMAX for ReLU)
                    //   - ST1W Z2
                    //
                    // DUP Z5.S, Z3.S[c] is: 0x0520_0063 | (c << ?) ... complex.
                    // Alternative: just LD1RW (broadcast load) from bias + c*4.
                    // LD1RW Z5.S, P0/Z, [X13, #c*4] — but that's an immediate offset form
                    // that may not exist for arbitrary c.
                    //
                    // SIMPLEST WORKING APPROACH: Use row-major store for activation,
                    // then transpose to col-major. This is what the existing code does.
                    //
                    // ACTUALLY — let's just load the bias once into Z3 and apply it
                    // BEFORE the store, while the data is still in ZA row format.
                    // The activation happens AFTER ST1W, loading back from memory.
                    // In the col-major path, we could restructure to:
                    //   1. ST1W horizontal (row-major) to a temp
                    //   2. Load rows, apply activation, store col-major
                    // But that defeats the purpose.
                    //
                    // Let's take the pragmatic approach: for intermediate layers with
                    // col-major output + activation, we apply bias/relu on row-major
                    // data (LD1W row, FADD bias, FMAX, ST1W row) and then do a
                    // separate col-major rewrite pass. Actually that's worse.
                    //
                    // BEST APPROACH: Apply activation to ZA row slices BEFORE storing.
                    // But MOVA is unreliable on M4.
                    //
                    // PRAGMATIC FINAL APPROACH:
                    // Store row-major, apply activation row-major (existing proven logic),
                    // then SVE-transpose in-kernel from row-major to col-major.
                    // The transpose is: for each row r of output [16×N]:
                    //   for each col c (in 16-element tiles):
                    //     read: buf[r*N + c] → this is a scalar at stride N*4
                    //     write: buf_out[c*16 + r] → contiguous within column
                    // Can't easily vectorize the read (non-contiguous).
                    //
                    // OR: Just keep the Rust transposes but eliminate SMSTART/SMSTOP
                    // overhead. That's still a win.

                    // DECISION: For Gate 23, go with the pragmatic hybrid:
                    // - Single SMSTART/SMSTOP (big win: ~300-600ns)
                    // - Keep row-major stores for intermediate layers
                    // - Do SVE-accelerated transpose inside streaming mode
                    // This will be cleaner. Let me restructure.

                    // Fall through to standard row-major activation
                    block.push(encode_sve_ld1w_ss(3, 0, 13, 3));
                } else if act == Activation::Bias || act == Activation::BiasReLU {
                    block.push(encode_sve_ld1w_ss(3, 0, 13, 3));
                }

                for _row in 0..TILE {
                    block.push(ld1w_z2_x2);
                    match act {
                        Activation::ReLU => block.push(fmax_z2_z4),
                        Activation::Bias => block.push(fadd_z2_bias),
                        Activation::BiasReLU => {
                            block.push(fadd_z2_bias);
                            block.push(fmax_z2_z4);
                        }
                        Activation::None => unreachable!(),
                    }
                    block.push(st1w_z2_x2);
                    block.push(add_x2_c);
                }
            }
        }
    }

    block
}

/// Encode `ST1W { ZA0V.S[Wv, #off] }, Pg, [Xn, Xm, LSL #2]` — SME **vertical**
/// word-width store of one slice of ZA0 to memory.
///
/// The vertical variant stores a column of ZA as 16 contiguous floats.
/// Encoding differs from horizontal by bit 15: horizontal=0, vertical=1.
pub const fn encode_sme_st1w_za_v(wv: u8, off2: u8, pg: u8, rn: u8, rm: u8) -> u32 {
    // Same as horizontal but with bit 15 set for vertical
    encode_sme_st1w_za_h(wv, off2, pg, rn, rm) | (1 << 15)
}

/// Build a monolithic inference JitPage for a multi-layer MLP.
///
/// All layers are emitted into a single page with one SMSTART/SMSTOP pair.
/// Intermediate transposes are done with SVE loads/stores inside streaming mode.
///
/// **Calling convention:** `page.call_with_args(input_ptr, output_ptr)`
/// - X0 = input (column-major [K1 × 16])
/// - X1 = output (row-major [16 × N_last])
///
/// Intermediate buffers (buf1, buf2) must be pre-allocated and valid for the
/// lifetime of the page.
pub fn build_monolithic_inference_page(
    layers: &[MonolithicLayerConfig],
    buf1_ptr: u64,
    buf2_ptr: u64,
) -> Option<crate::jit_page::JitPage> {
    assert!(!layers.is_empty() && layers.len() <= 4);

    // ═══════════════════════════════════════════════════════════════
    // Gate 23 Strategy: Vertical ST1W + LD1RW — Zero Transposes
    //
    // Key insight: ST1W *vertical* slices store a ZA column as 16
    // contiguous floats → column-major output. The next layer reads
    // this directly as column-major A via LD1W. No transpose needed.
    //
    // For activation on column-major data: LD1RW broadcasts one bias
    // element to all lanes, then FADD applies it to the 16-row column.
    // Both LD1RW and ST1W vertical are available in streaming mode.
    //
    // Buffer strategy:
    //   Layer 0: A = X0 (input), C = buf1 (col-major via vertical ST1W)
    //   Layer 1: A = buf1 (col-major), C = buf2 (col-major via vertical ST1W)
    //   Layer 2 (last): A = buf2 (col-major), C = X1 (row-major via horizontal ST1W)
    // ═══════════════════════════════════════════════════════════════

    const TILE: usize = 16;

    let mut insns = Vec::with_capacity(8192);

    // Save output pointer before we clobber X1
    insns.push(encode_mov_x(16, 1));  // X16 = final output ptr

    // SMSTART — enter streaming mode once for all layers
    insns.push(SMSTART);

    let num_layers = layers.len();

    for (layer_idx, layer) in layers.iter().enumerate() {
        let is_last = layer_idx == num_layers - 1;
        let is_intermediate = !is_last;

        // ── Set up A (X0) and C (X1) pointers for this layer ──
        if layer_idx == 0 {
            // X0 already has input pointer from caller
            insns.extend(emit_load_imm64_vec(1, buf1_ptr)); // C = buf1
        } else if layer_idx == 1 {
            insns.extend(emit_load_imm64_vec(0, buf1_ptr)); // A = buf1
            if is_last {
                insns.push(encode_mov_x(1, 16)); // C = final output
            } else {
                insns.extend(emit_load_imm64_vec(1, buf2_ptr)); // C = buf2
            }
        } else {
            insns.extend(emit_load_imm64_vec(0, buf2_ptr)); // A = buf2
            insns.push(encode_mov_x(1, 16)); // C = final output (must be last)
        }

        // Load weight and bias pointers
        insns.extend(emit_load_imm64_vec(5, layer.w_ptr));
        if layer.act == Activation::Bias || layer.act == Activation::BiasReLU {
            insns.extend(emit_load_imm64_vec(6, layer.b_ptr));
        }

        // ── Emit tiled GEMM + store + activation ──
        let m = layer.m;
        let n = layer.n;
        let k = layer.k;
        let act = layer.act;
        assert!(m % TILE == 0 && n % TILE == 0);

        let tiles_m = m / TILE;
        let tiles_n = n / TILE;
        let a_k_stride = (m * 4) as u16;
        let b_k_stride = (n * 4) as u16;

        // For intermediate layers: vertical ST1W → column-major [N×16]
        // Output stride between slices = 64 bytes (16 floats)
        // For last layer: horizontal ST1W → row-major [16×N]
        // Output stride between rows = N*4 bytes
        let c_stride = if is_intermediate { 64u16 } else { (n * 4) as u16 };

        let ld1w_z0_x4 = encode_sve_ld1w_ss(0, 0, 4, 3);
        let ld1w_z1_x7 = encode_sve_ld1w_ss(1, 0, 7, 3);
        let fmopa_za0 = encode_sme_fmopa(0, 0, 1, 0, 0);
        let add_x4_a = encode_add_x_imm(4, 4, a_k_stride);
        let add_x7_b = encode_add_x_imm(7, 7, b_k_stride);
        let subs_x8_1 = encode_subs_x_imm(8, 8, 1);
        let bne_kloop = encode_b_ne(-6 * 4);

        let st1w_za0 = if is_intermediate {
            encode_sme_st1w_za_v(0, 0, 0, 2, 3) // Vertical
        } else {
            encode_sme_st1w_za_h(0, 0, 0, 2, 3) // Horizontal
        };
        let add_w12_1 = encode_add_w_imm(12, 12, 1);
        let add_x2_c = encode_add_x_imm(2, 2, c_stride);

        // Activation
        let dup_z4_zero = 0x2538_C004u32;
        let fmax_z2_z4 = 0x6586_8082u32;
        let ld1w_z2_x2 = encode_sve_ld1w_ss(2, 0, 2, 3);
        let st1w_z2_x2 = encode_sve_st1w_ss(2, 0, 2, 3);

        // Preamble for this layer
        insns.push(encode_mov_x(10, 0));   // X10 = A_base
        insns.push(encode_mov_x(11, 1));   // X11 = C_base
        insns.push(encode_mov_xzr(3));     // X3 = 0
        insns.push(PTRUE_P0_S);

        if act == Activation::ReLU || act == Activation::BiasReLU {
            insns.push(dup_z4_zero);
        }

        for ti in 0..tiles_m {
            for tj in 0..tiles_n {
                let a_tile_off = (ti * TILE * 4) as u16;
                let b_tile_off = (tj * TILE * 4) as u16;

                let c_tile_off = if is_intermediate {
                    // Column-major [N×16]: tile (ti,tj) at tj*16*64 + ti*16*4
                    tj * TILE * TILE * 4 + ti * TILE * 4
                } else {
                    // Row-major [16×N]: tile (ti,tj) at ti*16*N*4 + tj*16*4
                    ti * TILE * n * 4 + tj * TILE * 4
                };

                // A tile pointer
                if a_tile_off > 0 {
                    insns.push(encode_add_x_imm(4, 10, a_tile_off));
                } else {
                    insns.push(encode_mov_x(4, 10));
                }
                // B tile pointer
                if b_tile_off > 0 {
                    insns.push(encode_add_x_imm(7, 5, b_tile_off));
                } else {
                    insns.push(encode_mov_x(7, 5));
                }
                // C tile pointer
                if c_tile_off == 0 {
                    insns.push(encode_mov_x(2, 11));
                } else if c_tile_off <= 4095 {
                    insns.push(encode_add_x_imm(2, 11, c_tile_off as u16));
                } else {
                    insns.extend(emit_load_imm64_vec(9, c_tile_off as u64));
                    insns.push(encode_add_x_reg(2, 11, 9));
                }

                // K-loop
                insns.push(encode_movz_x(8, k as u16, 0));
                insns.push(ZERO_ZA);
                insns.push(ld1w_z0_x4);
                insns.push(ld1w_z1_x7);
                insns.push(fmopa_za0);
                insns.push(add_x4_a);
                insns.push(add_x7_b);
                insns.push(subs_x8_1);
                insns.push(bne_kloop);

                // Store ZA slices
                insns.push(encode_mov_x(9, 2)); // save C tile start
                insns.push(0x5280_000Cu32); // MOV W12, #0
                for _ in 0..TILE {
                    insns.push(st1w_za0);
                    insns.push(add_w12_1);
                    insns.push(add_x2_c);
                }

                // ── Activation pass ──
                if act != Activation::None {
                    insns.push(encode_mov_x(2, 9)); // restore C tile start

                    if is_intermediate {
                        // Column-major: each LD1W loads 16 rows of ONE column.
                        // Bias: need bias[tj*16 + c] broadcast for each column c.
                        // Use LD1RW to broadcast-load one bias float.
                        // LD1RW {Z3.S}, P0/Z, [X13, #0]
                        // X13 = &bias[tj*16], advance by 4 per column.
                        if act == Activation::Bias || act == Activation::BiasReLU {
                            if b_tile_off > 0 {
                                insns.push(encode_add_x_imm(13, 6, b_tile_off));
                            } else {
                                insns.push(encode_mov_x(13, 6));
                            }
                        }

                        for _col in 0..TILE {
                            insns.push(ld1w_z2_x2); // Z2 = 16 rows of this column

                            if act == Activation::Bias || act == Activation::BiasReLU {
                                // LD1RW Z3.S, P0/Z, [X13, #0] — broadcast bias
                                insns.push(encode_ld1rw(3, 0, 13, 0));
                                insns.push(encode_sve_fadd_unpred(2, 2, 3));
                                insns.push(encode_add_x_imm(13, 13, 4)); // next bias element
                            }
                            if act == Activation::ReLU || act == Activation::BiasReLU {
                                insns.push(fmax_z2_z4);
                            }

                            insns.push(st1w_z2_x2); // store back
                            insns.push(add_x2_c);   // next column (stride=64)
                        }
                    } else {
                        // Row-major (last layer): existing row-based activation
                        if act == Activation::Bias || act == Activation::BiasReLU {
                            if b_tile_off > 0 {
                                insns.push(encode_add_x_imm(13, 6, b_tile_off));
                                insns.push(encode_sve_ld1w_ss(3, 0, 13, 3));
                            } else {
                                insns.push(encode_sve_ld1w_ss(3, 0, 6, 3));
                            }
                        }
                        for _row in 0..TILE {
                            insns.push(ld1w_z2_x2);
                            match act {
                                Activation::ReLU => insns.push(fmax_z2_z4),
                                Activation::Bias => insns.push(encode_sve_fadd_unpred(2, 2, 3)),
                                Activation::BiasReLU => {
                                    insns.push(encode_sve_fadd_unpred(2, 2, 3));
                                    insns.push(fmax_z2_z4);
                                }
                                Activation::None => unreachable!(),
                            }
                            insns.push(st1w_z2_x2);
                            insns.push(add_x2_c);
                        }
                    }
                }
            }
        }
    }

    // SMSTOP + RET
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

/// Build a predicated array copy kernel (Gate 26).
///
/// Logic:
/// 1. Initialize X3 = 0 (index)
/// 2. WHILELT P0.S, X3, X2 (generate predicate for index X3 vs limit X2)
/// 3. LD1W Z0.S, P0/Z, [X0, X3, LSL #2]
/// 4. ST1W Z0.S, P0, [X1, X3, LSL #2]
/// 5. ADD X3, X3, #16 (next vector chunk)
/// 6. WHILELT P0.S, X3, X2
/// 7. B.ANY loop (if any lanes are active)
///
/// Actually, the standard SVE loop pattern is:
///   WHILELT P0.S, X3, X2
///   B.NONE end
/// loop:
///   LD1W ...
///   ST1W ...
///   ADD X3, X3, #16
///   WHILELT P0.S, X3, X2
///   B.ANY loop
/// end:
///   RET
pub fn build_sve_predicated_copy(_limit: usize) -> Vec<u32> {
    let mut block = Vec::with_capacity(16);

    // X0 = src, X1 = dst, X2 = limit (already set via args)
    block.push(encode_mov_xzr(3)); // X3 = 0 (index)

    // Initial predicate
    block.push(encode_sve_whilelt_s(0, 3, 2));
    
    // If no lanes are active (limit=0), skip
    block.push(0x5400_0080); // B.EQ (none) -> end (skip 5 insns)

    // Loop start
    // Instruction order:
    // [0] LD1W
    // [1] ST1W
    // [2] ADD
    // [3] WHILELT
    // [4] B.NE -4 (back to [0])
    let ld1w = encode_sve_ld1w_ss(0, 0, 0, 3);
    let st1w = encode_sve_st1w_ss(0, 0, 1, 3);
    let add_x3_16 = encode_add_x_imm(3, 3, 16);
    let whilelt = encode_sve_whilelt_s(0, 3, 2);
    let b_any = encode_b_ne(-4 * 4);

    block.push(ld1w);
    block.push(st1w);
    block.push(add_x3_16);
    block.push(whilelt);
    block.push(b_any);

    block
}

/// Build a JIT page for the predicated copy test (Gate 26).
pub fn build_gate26_page(limit: usize) -> Option<crate::jit_page::JitPage> {
    let mut insns = Vec::with_capacity(32);

    // X0, X1 passed at call time. X2 = limit.
    insns.push(encode_movz_x(2, limit as u16, 0));

    insns.push(SMSTART);
    insns.extend(build_sve_predicated_copy(limit));
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

/// Build a JIT page for Gate 27: predicated FMOPA dot product.
///
/// Computes `c[0] = Σ a[i]·b[i]` for `i in 0..k` using FMOPA with P1 = lane-0-only
/// predicate. `c[1..15]` remain 0.0, proving `FMOPA Pn/M, Pm/M` only updates ZA
/// entries where both row-predicate and col-predicate lanes are active.
///
/// All three pointers are baked into the instruction stream; call with `call_void()`.
///
/// # Key M4 quirk
/// `FMOPA P1/M` modifies P1 as a side effect on M4 (undocumented). The workaround is
/// to re-emit `WHILELT P1.S, X14, X15` at the top of each loop iteration.
pub fn build_gate27_page(k: usize, a_ptr: u64, b_ptr: u64, c_ptr: u64)
    -> Option<crate::jit_page::JitPage>
{
    assert!(k >= 1 && k <= 65535, "k must be 1..=65535");

    let mut insns: Vec<u32> = Vec::with_capacity(64);

    // Bake all three pointers: X0=a, X1=b, X2=c
    insns.extend(emit_load_imm64_vec(0, a_ptr));
    insns.extend(emit_load_imm64_vec(1, b_ptr));
    insns.extend(emit_load_imm64_vec(2, c_ptr));
    // X3=0 (element index), X8=k (counter), X14=0, X15=1
    insns.push(encode_mov_xzr(3));
    insns.push(encode_movz_x(8, k as u16, 0));
    insns.push(encode_movz_x(14, 0, 0));
    insns.push(encode_movz_x(15, 1, 0));

    insns.push(SMSTART);
    insns.push(PTRUE_P0_S);  // P0 = all-true; SMSTART resets predicates to 0
    insns.push(ZERO_ZA);

    // ── Loop body: 7 instructions, BNE at [6] → [0] = −24 bytes ──
    // [0] Re-issue WHILELT each iter: FMOPA P1/M corrupts P1 on M4 (undocumented).
    insns.push(encode_sve_whilelt_s(1, 14, 15));        // [0] P1 = {T,F,F,...,F}
    insns.push(encode_sve_ld1w_ss(0, 1, 0, 3));         // [1] Z0 ← a[X3]  (P1/Z)
    insns.push(encode_sve_ld1w_ss(1, 1, 1, 3));         // [2] Z1 ← b[X3]  (P1/Z)
    insns.push(encode_sme_fmopa(0, 0, 1, 1, 1));        // [3] ZA0[0][0] += Z0[0]*Z1[0]
    insns.push(encode_add_x_imm(3, 3, 1));              // [4] X3 += 1
    insns.push(encode_subs_x_imm(8, 8, 1));             // [5] X8 -= 1, set flags
    insns.push(encode_b_ne(-6 * 4));                    // [6] B.NE [0]

    // ── Epilogue: extract row 0 via P0 (all-true), write 16 floats ──
    // Use P0 for ZA store: predicated ZA stores misbehave after ≥2 FMOPAs on M4.
    insns.push(encode_movz_x(12, 0, 0));                // W12 = 0 → ZA0H.S[W12,#0] = row 0
    insns.push(encode_mov_xzr(3));                      // X3 = 0 (store offset)
    insns.push(encode_sme_st1w_za_h(0, 0, 0, 2, 3));   // ST1W ZA0H[W12,#0], P0, [X2,X3,LSL#2]
    insns.push(SMSTOP);
    insns.push(RET);

    let total_bytes = insns.len() * 4;
    let page_size = ((total_bytes + 16383) / 16384) * 16384;
    let page = crate::jit_page::JitPage::alloc(page_size).ok()?;
    page.make_writable();
    for (i, &op) in insns.iter().enumerate() {
        page.write_instruction(i * 4, op);
    }
    page.make_executable();
    Some(page)
}

/// Gate 27.5 probe A — row-edge FMOPA with **separate** row/col predicates.
///
/// P1 = row-0-only (WHILELT, set **once** before the loop — not refreshed per iter).
/// P0 = all-true (PTRUE).
/// FMOPA: `ZA0.S, P1/M, P0/M` — P1 governs rows, P0 governs cols.
///
/// Probes: does FMOPA corrupt P1 when P1 is the ROW predicate only (not col)?
/// If `c[j] == Σ a[i]*b[i*16+j]` for all j, P1 survived unchanged → no corruption.
pub fn build_gate27p5_row_edge(k: usize, a_ptr: u64, b_ptr: u64, c_ptr: u64)
    -> Option<crate::jit_page::JitPage>
{
    assert!(k >= 1 && k <= 65535);
    let mut insns: Vec<u32> = Vec::with_capacity(80);

    insns.extend(emit_load_imm64_vec(0, a_ptr));   // X0 = a_ptr (K scalars)
    insns.extend(emit_load_imm64_vec(1, b_ptr));   // X1 = b_ptr (K × 16 floats)
    insns.extend(emit_load_imm64_vec(2, c_ptr));   // X2 = c_ptr (16-float output)
    insns.push(encode_mov_xzr(3));                 // X3 = 0 (A scalar index)
    insns.push(encode_mov_xzr(6));                 // X6 = 0 (B vector index)
    insns.push(encode_movz_x(8, k as u16, 0));     // X8 = K
    insns.push(encode_movz_x(14, 0, 0));           // X14 = 0
    insns.push(encode_movz_x(15, 1, 0));           // X15 = 1

    insns.push(SMSTART);
    insns.push(PTRUE_P0_S);                                   // P0 = all-true
    insns.push(ZERO_ZA);
    insns.push(encode_sve_whilelt_s(1, 14, 15));              // P1 = {T,F,…,F} — set ONCE

    // 7-instruction loop: BNE at [6] → [0] = −24 bytes
    insns.push(encode_sve_ld1w_ss(0, 1, 0, 3));              // [0] Z0[0] = a[X3]
    insns.push(encode_sve_ld1w_ss(1, 0, 1, 6));              // [1] Z1[0..15] = b[X6..X6+15]
    insns.push(encode_sme_fmopa(0, 0, 1, 1, 0));             // [2] ZA0 P1/M(row) P0/M(col)
    insns.push(encode_add_x_imm(3, 3, 1));                   // [3] X3++
    insns.push(encode_add_x_imm(6, 6, 16));                  // [4] X6 += 16
    insns.push(encode_subs_x_imm(8, 8, 1));                  // [5] X8--
    insns.push(encode_b_ne(-6 * 4));                         // [6] B.NE [0]

    insns.push(encode_movz_x(12, 0, 0));                     // W12 = 0 (ZA row 0)
    insns.push(encode_mov_xzr(3));                           // X3 = 0
    insns.push(encode_sme_st1w_za_h(0, 0, 0, 2, 3));        // ST1W ZA0H[W12,#0], P0, [X2]
    insns.push(SMSTOP);
    insns.push(RET);

    let total_bytes = insns.len() * 4;
    let page_size = ((total_bytes + 16383) / 16384) * 16384;
    let page = crate::jit_page::JitPage::alloc(page_size).ok()?;
    page.make_writable();
    for (i, &op) in insns.iter().enumerate() { page.write_instruction(i * 4, op); }
    page.make_executable();
    Some(page)
}

/// Gate 27.5 probe B — col-edge FMOPA with **separate** row/col predicates.
///
/// P0 = all-true (PTRUE). P1 = col-0-only (WHILELT, set **once** — not refreshed).
/// FMOPA: `ZA0.S, P0/M, P1/M` — P0 governs rows, P1 governs cols.
///
/// Two output stores:
/// - `c_ptr` (16 floats): ZA row 0 via **P0** — ground truth; c[0]=Σ a[i*16]*b[i], c[1..15]=0.0
/// - `d_ptr` (16 floats, pre-filled with sentinel): ZA row 0 via **P1** — tests predicated ZA store;
///   if working: d[0]=c[0], d[1..15]=sentinel; if broken: d[1..15] get overwritten.
pub fn build_gate27p5_col_edge(k: usize, a_ptr: u64, b_ptr: u64, c_ptr: u64, d_ptr: u64)
    -> Option<crate::jit_page::JitPage>
{
    assert!(k >= 1 && k <= 65535);
    let mut insns: Vec<u32> = Vec::with_capacity(80);

    insns.extend(emit_load_imm64_vec(0, a_ptr));   // X0 = a_ptr (K × 16 floats)
    insns.extend(emit_load_imm64_vec(1, b_ptr));   // X1 = b_ptr (K scalars)
    insns.extend(emit_load_imm64_vec(2, c_ptr));   // X2 = c_ptr (P0-store output)
    insns.extend(emit_load_imm64_vec(5, d_ptr));   // X5 = d_ptr (P1-store output)
    insns.push(encode_mov_xzr(3));                 // X3 = 0 (A vector index, +16/iter)
    insns.push(encode_mov_xzr(4));                 // X4 = 0 (B scalar index, +1/iter)
    insns.push(encode_movz_x(8, k as u16, 0));     // X8 = K
    insns.push(encode_movz_x(14, 0, 0));           // X14 = 0
    insns.push(encode_movz_x(15, 1, 0));           // X15 = 1

    insns.push(SMSTART);
    insns.push(PTRUE_P0_S);                                   // P0 = all-true
    insns.push(ZERO_ZA);
    insns.push(encode_sve_whilelt_s(1, 14, 15));              // P1 = {T,F,…,F} — set ONCE

    // 7-instruction loop: BNE at [6] → [0] = −24 bytes
    insns.push(encode_sve_ld1w_ss(0, 0, 0, 3));              // [0] Z0[0..15] = a[X3..X3+15]
    insns.push(encode_sve_ld1w_ss(1, 1, 1, 4));              // [1] Z1[0] = b[X4]
    insns.push(encode_sme_fmopa(0, 0, 1, 0, 1));             // [2] ZA0 P0/M(row) P1/M(col)
    insns.push(encode_add_x_imm(3, 3, 16));                  // [3] X3 += 16
    insns.push(encode_add_x_imm(4, 4, 1));                   // [4] X4++
    insns.push(encode_subs_x_imm(8, 8, 1));                  // [5] X8--
    insns.push(encode_b_ne(-6 * 4));                         // [6] B.NE [0]

    insns.push(encode_movz_x(12, 0, 0));                     // W12 = 0 (ZA row 0)
    insns.push(encode_mov_xzr(3));                           // X3 = 0
    insns.push(encode_sme_st1w_za_h(0, 0, 0, 2, 3));        // P0 store → c_ptr
    insns.push(encode_sme_st1w_za_h(0, 0, 1, 5, 3));        // P1 store → d_ptr (probes ZA store)
    insns.push(SMSTOP);
    insns.push(RET);

    let total_bytes = insns.len() * 4;
    let page_size = ((total_bytes + 16383) / 16384) * 16384;
    let page = crate::jit_page::JitPage::alloc(page_size).ok()?;
    page.make_writable();
    for (i, &op) in insns.iter().enumerate() { page.write_instruction(i * 4, op); }
    page.make_executable();
    Some(page)
}

/// Encode `LD1RW {Zt.S}, Pg/Z, [Xn, #imm]` — SVE load-and-replicate word.
///
/// Loads a single 32-bit value from memory and broadcasts it to all S-element
/// lanes of the destination Z register. Available in streaming SVE mode.
///
/// `imm` is a byte offset, must be a multiple of 4, range 0..252.
pub const fn encode_ld1rw(zt: u8, pg: u8, rn: u8, imm: u16) -> u32 {
    assert!(zt <= 31 && pg <= 7 && rn <= 30);
    assert!(imm % 4 == 0 && imm <= 252);
    let imm6 = (imm / 4) as u32;
    // LD1RW {Zt.S}, Pg/Z, [Xn, #imm]
    // Encoding: 1000 0101 01 imm6(6) 110 Pg(3) Rn(5) Zt(5)
    0x8540_C000 | (imm6 << 16) | ((pg as u32) << 10) | ((rn as u32) << 5) | (zt as u32)
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
    fn whilelt_s_encoding() {
        // Reference values from clang `aarch64-apple-darwin -march=armv9-a+sme`:
        //   whilelt p0.s, x3, x2  → 0x25a2_1460
        //   whilelt p0.s, xzr, x2 → 0x25a2_17e0  (xzr = reg 31 → 31<<5 = 0x3e0)
        //   whilelt p0.s, x0, x1  → 0x25a1_1400
        assert_eq!(encode_sve_whilelt_s(0, 3, 2),  0x25a2_1460);
        assert_eq!(encode_sve_whilelt_s(0, 31, 2), 0x25a2_17e0);
        assert_eq!(encode_sve_whilelt_s(0, 0, 1),  0x25a1_1400);
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

    #[test]
    fn fmopa_encoding() {
        // Verified against known M4 instruction stream:
        // - baseline: FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S (zm=1, all predicates P0)
        assert_eq!(encode_sme_fmopa(0, 0, 1, 0, 0), 0x8081_0000);
        // - P1/M row+col predicates: pn=1 (<<10 = 0x400), pm=1 (<<13 = 0x2000)
        assert_eq!(encode_sme_fmopa(0, 0, 1, 1, 1), 0x8081_2400);
        // - arbitrary fields: zada=1, zn=4, zm=5, pn=2, pm=3
        assert_eq!(encode_sme_fmopa(1, 4, 5, 2, 3), 0x8085_6881);
    }

    #[test]
    fn st1w_za_h_pg_field() {
        // pg=1 must set bit 10, not bit 11 (Pg field is at bits 12–10)
        let enc = encode_sme_st1w_za_h(0, 0, 1, 2, 3);
        assert_eq!(enc & (1 << 10), 1 << 10, "Pg bit 10 should be set for pg=1");
        assert_eq!(enc & (1 << 11), 0,       "Bit 11 must be clear for pg=1");
    }
}
