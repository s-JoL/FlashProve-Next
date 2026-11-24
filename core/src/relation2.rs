// SPDX-License-Identifier: Apache-2.0
//! Relation 2 (`LLM_INFERENCE_V2`, `specs/11-wire-format.md` §6):
//! Standalone, zero-dependency, panic-free Rust verifier for fixed-size `FPN2` containers backed by a
//! canonical public `CircuitDesc` (pinned by `circuit_root = BLAKE2b-256("FPN2-CDESC-v2", desc)` and
//! bound to caller-pinned `stmt_digest = BLAKE2b-256("FPN2-STMT-v2", canonical_statement)`).
//!
//! Security hardening (closing all v1.0 security audit findings):
//!   * Independent left/right denominator blinders (`m = m_mask - 1` holds `(0, d_L)` on left,
//!     `(0, d_R)` on right) so layer-0 child denominator ratio `dl0 / dr0` is uniformly random in `E*`.
//!   * Minimum query floor `MIN_STANDARD_QUERIES = 72` enforced on `STANDARD`-tier profiles.
//!   * Pin-before-decode: `circuit_root` and `stmt_digest` are verified BEFORE parsing `raw_desc`;
//!     all `Vec::with_capacity` allocations are capped by remaining input bytes; every index, width,
//!     shift, alignment, extent, and non-overlap constraint is validated before evaluation (panic-free).
//!   * Only linear GKR numerator tags `0` (const 1) and `1` (linear AST) are admitted (`>= 2` rejected).

use crate::blake2b::blake2b;
use crate::field::{eq_pt, lagrange_eval, Fp, Fp3};
use crate::folded_gkr::eq1;
use crate::multi_gkr::{verify_multi_gkr, MultiGkrLayout};
use crate::multi_pcs::{tail_len_for, verify_multi_open, MultiPcsLayout};
use crate::profile::{lookup as lookup_profile, Relation, SecurityTier, HEADER_LEN};
use crate::transcript::{Fpn2Transcript, LABEL_V2};
use crate::verify::{Policy, RejectCode};
use crate::wire::Fpn2Header;

pub const DESC_MAGIC: &[u8; 8] = b"FPN2DESC";
pub const DESC_VERSION: u16 = 1;
pub const PERSONAL_CDESC: &[u8; 16] = b"FPN2-CDESC-v2\0\0\0";
pub const PERSONAL_STMT: &[u8; 16] = b"FPN2-STMT-v2\0\0\0\0";
/// Minimum rate-1/8 Basefold queries required under `SecurityTier::Standard` (matches `fpn2-folded-air-v1`
/// and `fpn2-folded-vec-n22-v1`).
pub const MIN_STANDARD_QUERIES: usize = 72;

// ----------------------------------------------------------------------------- CircuitDesc AST & types
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecPlace {
    pub group_idx: usize,
    pub offset: usize,
    pub nbits: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewRef {
    pub place: VecPlace,
    pub shift: i32,
    pub varmap: Vec<Option<usize>>,
    pub varmap_const: Vec<u8>,
    pub fixed: Vec<(usize, u8)>,
}

/// Compact symbolic expression AST over sumcheck/GKR cube variables `r[0..nv]` and view claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AstNode {
    Const(Fp3),
    ViewAtom(usize),
    EqZc(Vec<usize>),
    SelLt {
        vars: Vec<usize>,
        limit: u64,
        fixed: Vec<(usize, u8)>,
    },
    IndexP {
        vars: Vec<usize>,
        scale: Fp3,
        fixed: Vec<(usize, u8)>,
    },
    Add(Box<AstNode>, Box<AstNode>),
    Sub(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Scale(Fp3, Box<AstNode>),
}

/// One aligned lookup block inside a GKR batch (strictly linear numerator: const 1 or linear AST).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrBlockDesc {
    pub offset: usize,
    pub nbits: usize,
    pub tid: u64,
    pub col_exprs: Vec<AstNode>,
    pub num_expr: Option<AstNode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrBatchDesc {
    pub s: usize,
    pub m_mask: usize,
    pub aux_place: VecPlace,
    pub view_refs: Vec<ViewRef>,
    pub blocks: Vec<GkrBlockDesc>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckDesc {
    pub nv: usize,
    pub deg: usize,
    pub n_constraints: usize,
    pub mask_place: VecPlace,
    pub view_refs: Vec<ViewRef>,
    pub constraints: Vec<AstNode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitDesc {
    pub n_max: usize,
    pub queries: usize,
    pub group_ns: Vec<usize>,
    pub gkr_batches: Vec<GkrBatchDesc>,
    pub sumchecks: Vec<SumcheckDesc>,
}

pub fn circuit_desc_root(raw_desc: &[u8]) -> [u8; 32] {
    let d = blake2b(32, PERSONAL_CDESC, raw_desc);
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

pub fn stmt_digest_bytes(raw_stmt: &[u8]) -> [u8; 32] {
    let d = blake2b(32, PERSONAL_STMT, raw_stmt);
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

// ----------------------------------------------------------------------------- Bounded binary decoder
struct Cur<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Cur<'a> {
    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.off)
    }
    fn check_count(&self, count: usize, min_elem_bytes: usize) -> Result<usize, RejectCode> {
        let need = count.checked_mul(min_elem_bytes).ok_or(RejectCode::Length)?;
        if need > self.remaining() {
            return Err(RejectCode::Length);
        }
        Ok(count)
    }
    fn u8_count(&mut self, min_elem_bytes: usize) -> Result<usize, RejectCode> {
        let n = self.u8()? as usize;
        self.check_count(n, min_elem_bytes)
    }
    fn u16_count(&mut self, min_elem_bytes: usize) -> Result<usize, RejectCode> {
        let n = self.u16()? as usize;
        self.check_count(n, min_elem_bytes)
    }
    fn u32_count(&mut self, min_elem_bytes: usize) -> Result<usize, RejectCode> {
        let n = self.u32()? as usize;
        self.check_count(n, min_elem_bytes)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], RejectCode> {
        let end = self.off.checked_add(n).ok_or(RejectCode::Length)?;
        if end > self.b.len() {
            return Err(RejectCode::Length);
        }
        let s = &self.b[self.off..end];
        self.off = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, RejectCode> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, RejectCode> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, RejectCode> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, RejectCode> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, RejectCode> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn fp3(&mut self) -> Result<Fp3, RejectCode> {
        Fp3::decode(self.take(24)?).map_err(|_| RejectCode::NonCanonical)
    }
}

fn validate_place(
    place: &VecPlace,
    group_ns: &[usize],
    queries: usize,
) -> Result<(), RejectCode> {
    if place.group_idx >= group_ns.len() {
        return Err(RejectCode::NonCanonical);
    }
    let ng = group_ns[place.group_idx];
    if place.nbits >= ng {
        return Err(RejectCode::NonCanonical);
    }
    let sz = 1usize << place.nbits;
    let cap = (1usize << ng).saturating_sub(tail_len_for(queries));
    if !place.offset.is_multiple_of(sz) {
        return Err(RejectCode::NonCanonical);
    }
    let end = place.offset.checked_add(sz).ok_or(RejectCode::NonCanonical)?;
    if end > cap {
        return Err(RejectCode::NonCanonical);
    }
    Ok(())
}

fn decode_place(
    c: &mut Cur<'_>,
    group_ns: &[usize],
    queries: usize,
) -> Result<VecPlace, RejectCode> {
    let group_idx = c.u16()? as usize;
    let offset = c.u32()? as usize;
    let nbits = c.u8()? as usize;
    let pl = VecPlace {
        group_idx,
        offset,
        nbits,
    };
    validate_place(&pl, group_ns, queries)?;
    Ok(pl)
}

fn decode_view_ref(
    c: &mut Cur<'_>,
    group_ns: &[usize],
    queries: usize,
    cube_nv: usize,
) -> Result<ViewRef, RejectCode> {
    let place = decode_place(c, group_ns, queries)?;
    let shift = c.i32()?;
    if shift.unsigned_abs() as u64 >= (1u64 << place.nbits) {
        return Err(RejectCode::NonCanonical);
    }
    let vlen = c.u8()? as usize;
    if vlen != place.nbits {
        return Err(RejectCode::NonCanonical);
    }
    c.check_count(vlen, 2)?;
    let mut varmap = Vec::with_capacity(vlen);
    let mut varmap_const = Vec::with_capacity(vlen);
    for _ in 0..vlen {
        let tag = c.u8()?;
        let val = c.u8()?;
        if tag == 0 {
            let vidx = val as usize;
            if vidx >= cube_nv {
                return Err(RejectCode::NonCanonical);
            }
            varmap.push(Some(vidx));
            varmap_const.push(0);
        } else if tag == 1 && val <= 1 {
            varmap.push(None);
            varmap_const.push(val);
        } else {
            return Err(RejectCode::NonCanonical);
        }
    }
    let flen = c.u8()? as usize;
    c.check_count(flen, 2)?;
    let mut fixed = Vec::with_capacity(flen);
    for _ in 0..flen {
        let var = c.u8()? as usize;
        let bit = c.u8()?;
        if var >= cube_nv || bit > 1 {
            return Err(RejectCode::NonCanonical);
        }
        fixed.push((var, bit));
    }
    Ok(ViewRef {
        place,
        shift,
        varmap,
        varmap_const,
        fixed,
    })
}

fn decode_ast(
    c: &mut Cur<'_>,
    depth: usize,
    cube_nv: usize,
    n_views: usize,
    allow_eqzc: bool,
) -> Result<AstNode, RejectCode> {
    if depth > 64 {
        return Err(RejectCode::Length);
    }
    match c.u8()? {
        0 => Ok(AstNode::Const(c.fp3()?)),
        1 => {
            let idx = c.u32()? as usize;
            if idx >= n_views {
                return Err(RejectCode::NonCanonical);
            }
            Ok(AstNode::ViewAtom(idx))
        }
        2 => {
            if !allow_eqzc {
                return Err(RejectCode::NonCanonical);
            }
            let n = c.u8()? as usize;
            c.check_count(n, 1)?;
            let mut vars = Vec::with_capacity(n);
            for _ in 0..n {
                let v = c.u8()? as usize;
                if v >= cube_nv {
                    return Err(RejectCode::NonCanonical);
                }
                vars.push(v);
            }
            Ok(AstNode::EqZc(vars))
        }
        3 => {
            let n = c.u8()? as usize;
            if n > 63 {
                return Err(RejectCode::NonCanonical);
            }
            c.check_count(n, 1)?;
            let mut vars = Vec::with_capacity(n);
            for _ in 0..n {
                let v = c.u8()? as usize;
                if v >= cube_nv {
                    return Err(RejectCode::NonCanonical);
                }
                vars.push(v);
            }
            let limit = c.u64()?;
            let fn_ = c.u8()? as usize;
            c.check_count(fn_, 2)?;
            let mut fixed = Vec::with_capacity(fn_);
            for _ in 0..fn_ {
                let v = c.u8()? as usize;
                let b = c.u8()?;
                if v >= cube_nv || b > 1 {
                    return Err(RejectCode::NonCanonical);
                }
                fixed.push((v, b));
            }
            Ok(AstNode::SelLt { vars, limit, fixed })
        }
        4 => {
            let n = c.u8()? as usize;
            if n > 63 {
                return Err(RejectCode::NonCanonical);
            }
            c.check_count(n, 1)?;
            let mut vars = Vec::with_capacity(n);
            for _ in 0..n {
                let v = c.u8()? as usize;
                if v >= cube_nv {
                    return Err(RejectCode::NonCanonical);
                }
                vars.push(v);
            }
            let scale = c.fp3()?;
            let fn_ = c.u8()? as usize;
            c.check_count(fn_, 2)?;
            let mut fixed = Vec::with_capacity(fn_);
            for _ in 0..fn_ {
                let v = c.u8()? as usize;
                let b = c.u8()?;
                if v >= cube_nv || b > 1 {
                    return Err(RejectCode::NonCanonical);
                }
                fixed.push((v, b));
            }
            Ok(AstNode::IndexP { vars, scale, fixed })
        }
        5 => Ok(AstNode::Add(
            Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
            Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
        )),
        6 => Ok(AstNode::Sub(
            Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
            Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
        )),
        7 => Ok(AstNode::Mul(
            Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
            Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
        )),
        8 => {
            let s = c.fp3()?;
            Ok(AstNode::Scale(
                s,
                Box::new(decode_ast(c, depth + 1, cube_nv, n_views, allow_eqzc)?),
            ))
        }
        _ => Err(RejectCode::NonCanonical),
    }
}

pub fn decode_circuit_desc(raw: &[u8]) -> Result<CircuitDesc, RejectCode> {
    let mut c = Cur { b: raw, off: 0 };
    if c.take(8)? != DESC_MAGIC {
        return Err(RejectCode::Magic);
    }
    if c.u16()? != DESC_VERSION {
        return Err(RejectCode::Version);
    }
    let n_max = c.u8()? as usize;
    let queries = c.u16()? as usize;
    let n_groups = c.u16_count(1)?;
    let mut group_ns = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        group_ns.push(c.u8()? as usize);
    }
    let _ = MultiPcsLayout::new(n_max, queries, group_ns.clone())?;

    let n_gkr = c.u16_count(18)?;
    let mut gkr_batches = Vec::with_capacity(n_gkr);
    for _ in 0..n_gkr {
        let s = c.u8()? as usize;
        let m_mask = c.u16()? as usize;
        let gl = MultiGkrLayout::new(s, m_mask)?;
        let aux_place = decode_place(&mut c, &group_ns, queries)?;
        if (1usize << aux_place.nbits) < gl.libra_coeff_count() + 2 * gl.m_mask {
            return Err(RejectCode::NonCanonical);
        }
        let n_views = c.u32_count(13)?;
        let mut view_refs = Vec::with_capacity(n_views);
        for _ in 0..n_views {
            view_refs.push(decode_view_ref(&mut c, &group_ns, queries, s)?);
        }
        let n_blocks = c.u32_count(15)?;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut intervals: Vec<(usize, usize)> = Vec::with_capacity(n_blocks);
        let half = gl.half();
        let n_leaves = gl.n_leaves();
        let lim_l = half - gl.m_mask;
        let lim_r = n_leaves - gl.m_mask;
        for _ in 0..n_blocks {
            let offset = c.u32()? as usize;
            let nbits = c.u8()? as usize;
            if nbits >= s {
                return Err(RejectCode::NonCanonical);
            }
            let sz = 1usize << nbits;
            let end = offset.checked_add(sz).ok_or(RejectCode::NonCanonical)?;
            if !offset.is_multiple_of(sz) {
                return Err(RejectCode::NonCanonical);
            }
            // Must fit entirely inside left real region [0, lim_l) or right real region [half, lim_r)
            let in_left = end <= lim_l;
            let in_right = offset >= half && end <= lim_r;
            if !in_left && !in_right {
                return Err(RejectCode::NonCanonical);
            }
            intervals.push((offset, end));

            let tid = c.u64()?;
            let n_cols = c.u8_count(1)?;
            let mut col_exprs = Vec::with_capacity(n_cols);
            for _ in 0..n_cols {
                col_exprs.push(decode_ast(&mut c, 0, nbits, n_views, false)?);
            }
            let num_kind = c.u8()?;
            let num_expr = match num_kind {
                0 => None,
                1 => Some(decode_ast(&mut c, 0, nbits, n_views, false)?),
                // Reject unbound tag 2 and any other unknown tag fail-closed!
                _ => return Err(RejectCode::NonCanonical),
            };
            blocks.push(GkrBlockDesc {
                offset,
                nbits,
                tid,
                col_exprs,
                num_expr,
            });
        }
        // Verify GKR blocks are pairwise disjoint
        intervals.sort_unstable();
        for pair in intervals.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(RejectCode::NonCanonical);
            }
        }
        gkr_batches.push(GkrBatchDesc {
            s,
            m_mask,
            aux_place,
            view_refs,
            blocks,
        });
    }

    let n_sc = c.u16_count(17)?;
    let mut sumchecks = Vec::with_capacity(n_sc);
    for _ in 0..n_sc {
        let nv = c.u8()? as usize;
        let deg = c.u8()? as usize;
        if !(1..=28).contains(&nv) || !(1..=16).contains(&deg) {
            return Err(RejectCode::NonCanonical);
        }
        let n_constraints = c.u32_count(1)?;
        let mask_place = decode_place(&mut c, &group_ns, queries)?;
        if (1usize << mask_place.nbits) < nv * (deg + 1) {
            return Err(RejectCode::NonCanonical);
        }
        let n_views = c.u32_count(13)?;
        let mut view_refs = Vec::with_capacity(n_views);
        for _ in 0..n_views {
            view_refs.push(decode_view_ref(&mut c, &group_ns, queries, nv)?);
        }
        let mut constraints = Vec::with_capacity(n_constraints);
        for _ in 0..n_constraints {
            constraints.push(decode_ast(&mut c, 0, nv, n_views, true)?);
        }
        sumchecks.push(SumcheckDesc {
            nv,
            deg,
            n_constraints,
            mask_place,
            view_refs,
            constraints,
        });
    }

    if c.off != raw.len() {
        return Err(RejectCode::Length);
    }
    Ok(CircuitDesc {
        n_max,
        queries,
        group_ns,
        gkr_batches,
        sumchecks,
    })
}

// ----------------------------------------------------------------------------- MLE & AST evaluation helpers
fn ind_fixed(fixed: &[(usize, u8)], r: &[Fp3]) -> Fp3 {
    let mut acc = Fp3::ONE;
    for &(v, b) in fixed {
        acc = acc.mul(if b == 1 { r[v] } else { Fp3::ONE.sub(r[v]) });
    }
    acc
}

fn mle_lt(z: &[Fp3], n: u64) -> Fp3 {
    let bits = z.len();
    if bits == 0 {
        return if n > 0 { Fp3::ONE } else { Fp3::ZERO };
    }
    if n == 0 {
        return Fp3::ZERO;
    }
    if n >= (1u64 << bits) {
        return Fp3::ONE;
    }
    let mut acc = Fp3::ZERO;
    let mut pref = Fp3::ONE;
    for i in (0..bits).rev() {
        let zi = z[i];
        if (n >> i) & 1 == 1 {
            acc = acc.add(pref.mul(Fp3::ONE.sub(zi)));
            pref = pref.mul(zi);
        } else {
            pref = pref.mul(Fp3::ONE.sub(zi));
        }
    }
    acc
}

fn mle_shift(r: &[Fp3], alpha: &[Fp3], delta: u64) -> Fp3 {
    let n = r.len();
    if delta >= (1u64 << n) {
        return Fp3::ZERO;
    }
    let mut f = [Fp3::ONE, Fp3::ZERO];
    for i in 0..n {
        let d = ((delta >> i) & 1) as usize;
        let mut g = [Fp3::ZERO, Fp3::ZERO];
        for c in 0..2 {
            if f[c] == Fp3::ZERO {
                continue;
            }
            for xb in 0..2 {
                let s = xb + d + c;
                let yb = s & 1;
                let c2 = s >> 1;
                let rx = if xb == 1 { r[i] } else { Fp3::ONE.sub(r[i]) };
                let ay = if yb == 1 {
                    alpha[i]
                } else {
                    Fp3::ONE.sub(alpha[i])
                };
                g[c2] = g[c2].add(f[c].mul(rx.mul(ay)));
            }
        }
        f = g;
    }
    f[0]
}

fn mle_shift_signed(r: &[Fp3], alpha: &[Fp3], delta: i32) -> Fp3 {
    if delta >= 0 {
        mle_shift(r, alpha, delta as u64)
    } else {
        mle_shift(alpha, r, delta.unsigned_abs() as u64)
    }
}

fn eval_ast(node: &AstNode, r: &[Fp3], r_zc: &[Fp3], view_vals: &[Fp3]) -> Fp3 {
    match node {
        AstNode::Const(c) => *c,
        AstNode::ViewAtom(idx) => view_vals[*idx],
        AstNode::EqZc(vars) => {
            let mut acc = Fp3::ONE;
            for &v in vars {
                acc = acc.mul(eq1(r_zc[v], r[v]));
            }
            acc
        }
        AstNode::SelLt { vars, limit, fixed } => {
            let z: Vec<Fp3> = vars.iter().map(|&v| r[v]).collect();
            ind_fixed(fixed, r).mul(mle_lt(&z, *limit))
        }
        AstNode::IndexP { vars, scale, fixed } => {
            let mut sum = Fp3::ZERO;
            for (k, &v) in vars.iter().enumerate() {
                sum = sum.add(r[v].scale(Fp::new(1u64 << k)));
            }
            ind_fixed(fixed, r).mul(scale.mul(sum))
        }
        AstNode::Add(a, b) => {
            eval_ast(a, r, r_zc, view_vals).add(eval_ast(b, r, r_zc, view_vals))
        }
        AstNode::Sub(a, b) => {
            eval_ast(a, r, r_zc, view_vals).sub(eval_ast(b, r, r_zc, view_vals))
        }
        AstNode::Mul(a, b) => {
            eval_ast(a, r, r_zc, view_vals).mul(eval_ast(b, r, r_zc, view_vals))
        }
        AstNode::Scale(s, a) => s.mul(eval_ast(a, r, r_zc, view_vals)),
    }
}

fn block_ind_at(offset: usize, nbits: usize, r_full: &[Fp3]) -> Fp3 {
    let mut acc = Fp3::ONE;
    for (b, &rb) in r_full.iter().enumerate().skip(nbits) {
        acc = acc.mul(if (offset >> b) & 1 == 1 {
            rb
        } else {
            Fp3::ONE.sub(rb)
        });
    }
    acc
}

fn local_point_of(vr: &ViewRef, r: &[Fp3]) -> Vec<Fp3> {
    vr.varmap
        .iter()
        .zip(vr.varmap_const.iter())
        .map(|(v, &cb)| match v {
            Some(idx) => r[*idx],
            None => Fp3::from_u64(cb as u64),
        })
        .collect()
}

// ----------------------------------------------------------------------------- PCS claim accumulator
#[derive(Clone, Debug)]
enum ClaimSpec {
    Pt(Vec<Fp3>),
    Sh(Vec<Fp3>, i32),
    Pw(Vec<Fp3>, usize),
    LinearIdx(Vec<(usize, Fp3)>),
}

struct GroupClaim {
    place: VecPlace,
    spec: ClaimSpec,
    val: Fp3,
}

fn eval_claim_weight_mle(cl: &GroupClaim, r_sub: &[Fp3]) -> Fp3 {
    let nbits = cl.place.nbits;
    match &cl.spec {
        ClaimSpec::Pt(r) => eq_pt(r, r_sub),
        ClaimSpec::Sh(r, delta) => mle_shift_signed(r, r_sub, *delta),
        ClaimSpec::Pw(r, d) => {
            let mut acc = Fp3::ZERO;
            for (i, &ri) in r.iter().enumerate() {
                let mut pw = Fp3::ONE;
                for k in 0..=*d {
                    let idx = i * (*d + 1) + k;
                    let mut eq_v = Fp3::ONE;
                    for (b, &rb) in r_sub.iter().enumerate().take(nbits) {
                        eq_v = eq_v.mul(if (idx >> b) & 1 == 1 {
                            rb
                        } else {
                            Fp3::ONE.sub(rb)
                        });
                    }
                    acc = acc.add(pw.mul(eq_v));
                    pw = pw.mul(ri);
                }
            }
            acc
        }
        ClaimSpec::LinearIdx(entries) => {
            let mut acc = Fp3::ZERO;
            for &(idx, coeff) in entries {
                let mut eq_v = Fp3::ONE;
                for (b, &rb) in r_sub.iter().enumerate().take(nbits) {
                    eq_v = eq_v.mul(if (idx >> b) & 1 == 1 {
                        rb
                    } else {
                        Fp3::ONE.sub(rb)
                    });
                }
                acc = acc.add(coeff.mul(eq_v));
            }
            acc
        }
    }
}

// ----------------------------------------------------------------------------- Container verification
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation2Claim {
    pub profile_id: [u8; 32],
    pub circuit_root: [u8; 32],
    pub stmt_digest: [u8; 32],
    pub body_used: usize,
    pub c_p: usize,
}

pub fn body_used(desc: &CircuitDesc) -> Result<usize, RejectCode> {
    let g_cnt = desc.group_ns.len();
    let pcs_lay = MultiPcsLayout::new(desc.n_max, desc.queries, desc.group_ns.clone())?;
    let mut n = 64usize
        .checked_add(32usize.checked_mul(g_cnt).ok_or(RejectCode::Length)?)
        .ok_or(RejectCode::Length)?;
    for b in &desc.gkr_batches {
        let gl = MultiGkrLayout::new(b.s, b.m_mask)?;
        let gc = (b.view_refs.len() + 2)
            .checked_mul(24)
            .ok_or(RejectCode::Length)?;
        n = n
            .checked_add(gl.proof_bytes())
            .and_then(|v| v.checked_add(gc))
            .ok_or(RejectCode::Length)?;
    }
    for sc in &desc.sumchecks {
        let sc_b = 24usize
            .checked_add(sc.nv * (sc.deg + 1) * 24)
            .and_then(|v| v.checked_add((sc.view_refs.len() + 1) * 24))
            .ok_or(RejectCode::Length)?;
        n = n.checked_add(sc_b).ok_or(RejectCode::Length)?;
    }
    n = n
        .checked_add(pcs_lay.proof_bytes())
        .ok_or(RejectCode::Length)?;
    Ok(n)
}

/// Normative standalone verifier for `FPN2` Relation 2 (`LLM_INFERENCE_V2`).
pub fn verify_relation2(
    raw: &[u8],
    raw_desc: &[u8],
    policy: &Policy,
) -> Result<Relation2Claim, RejectCode> {
    if raw.len() < HEADER_LEN {
        return Err(RejectCode::Length);
    }
    let hdr = Fpn2Header::parse(&raw[..HEADER_LEN])?;
    let prof = lookup_profile(&hdr.profile_id).ok_or(RejectCode::ProfileId)?;
    if let Some(exp) = policy.expected_profile_id {
        if hdr.profile_id != exp {
            return Err(RejectCode::ProfilePinned);
        }
    }
    if prof.relation != Relation::LlmInferenceV2 {
        return Err(RejectCode::ProfileId);
    }
    if prof.security_tier == SecurityTier::Reduced && !policy.allow_reduced {
        return Err(RejectCode::ReducedSecurity);
    }
    if raw.len() != prof.c_p as usize {
        return Err(RejectCode::Length);
    }
    if policy.expected_x_root.is_some() {
        return Err(RejectCode::XRootPinned);
    }

    // Pin-before-decode: authenticate caller's circuit pin AND statement pin BEFORE decoding `raw_desc`!
    let expected_pin = policy
        .expected_circuit_root
        .ok_or(RejectCode::CircuitUnpinned)?;
    let desc_root = circuit_desc_root(raw_desc);
    if desc_root != expected_pin {
        return Err(RejectCode::CircuitPinned);
    }
    let expected_stmt = policy
        .expected_stmt_digest
        .ok_or(RejectCode::StmtUnpinned)?;
    if raw.len() < HEADER_LEN + 64 {
        return Err(RejectCode::Length);
    }
    if raw[HEADER_LEN..HEADER_LEN + 32] != desc_root {
        return Err(RejectCode::CircuitPinned);
    }
    if raw[HEADER_LEN + 32..HEADER_LEN + 64] != expected_stmt {
        return Err(RejectCode::StmtPinned);
    }

    let desc = decode_circuit_desc(raw_desc)?;

    // Enforce minimum query floor under `STANDARD` security tier (Finding #2)
    if prof.security_tier == SecurityTier::Standard && desc.queries < MIN_STANDARD_QUERIES {
        return Err(RejectCode::ReducedSecurity);
    }

    let used = body_used(&desc)?;
    let body_end = HEADER_LEN.checked_add(used).ok_or(RejectCode::Length)?;
    if body_end > raw.len() {
        return Err(RejectCode::Length);
    }
    if raw[body_end..].iter().any(|&b| b != 0) {
        return Err(RejectCode::Padding);
    }

    let body = &raw[HEADER_LEN..body_end];
    let mut cur = Cur { b: body, off: 0 };

    let mut circuit_root = [0u8; 32];
    circuit_root.copy_from_slice(cur.take(32)?);
    let mut stmt_digest = [0u8; 32];
    stmt_digest.copy_from_slice(cur.take(32)?);

    let g_cnt = desc.group_ns.len();
    let mut group_roots = Vec::with_capacity(g_cnt);
    for _ in 0..g_cnt {
        let mut r = [0u8; 32];
        r.copy_from_slice(cur.take(32)?);
        group_roots.push(r);
    }

    let mut tr = Fpn2Transcript::new(LABEL_V2);
    tr.absorb_bytes(&raw[..HEADER_LEN]);
    tr.absorb_bytes(&circuit_root);
    tr.absorb_bytes(&stmt_digest);
    for r in &group_roots {
        tr.absorb_bytes(r);
    }

    let mut claims: Vec<Vec<GroupClaim>> = (0..g_cnt).map(|_| Vec::new()).collect();

    // ---- 1. Verify all GKR batches (0 Phase-2 commitments)
    if !desc.gkr_batches.is_empty() {
        let gamma_lu = tr.squeeze_fp3();
        let beta_lu = tr.squeeze_fp3();
        for b in &desc.gkr_batches {
            let gl = MultiGkrLayout::new(b.s, b.m_mask)?;
            let gkr_pf = cur.take(gl.proof_bytes())?;
            let ver = verify_multi_gkr(&gl, gkr_pf, &mut tr)?;
            let r_gkr = &ver.r_gkr;

            let gc_start = cur.off;
            let mut view_vals = Vec::with_capacity(b.view_refs.len());
            for vr in &b.view_refs {
                let val = cur.fp3()?;
                let loc = local_point_of(vr, r_gkr);
                let spec = if vr.shift != 0 {
                    ClaimSpec::Sh(loc, vr.shift)
                } else {
                    ClaimSpec::Pt(loc)
                };
                claims[vr.place.group_idx].push(GroupClaim {
                    place: vr.place.clone(),
                    spec,
                    val,
                });
                view_vals.push(val.mul(ind_fixed(&vr.fixed, r_gkr)));
            }

            let mut acc_num = Fp3::ZERO;
            let mut acc_den = Fp3::ZERO;
            let mut sum_blk_ind = Fp3::ZERO;
            for blk in &b.blocks {
                let ind_blk = block_ind_at(blk.offset, blk.nbits, r_gkr);
                sum_blk_ind = sum_blk_ind.add(ind_blk);
                let r_loc = &r_gkr[..blk.nbits];

                let mut fp_eval = Fp3::from_u64(blk.tid);
                let mut bp = beta_lu;
                for ce in &blk.col_exprs {
                    let cv = eval_ast(ce, r_loc, &[], &view_vals);
                    fp_eval = fp_eval.add(bp.mul(cv));
                    bp = bp.mul(beta_lu);
                }
                let den_blk = gamma_lu.sub(fp_eval);
                acc_den = acc_den.add(ind_blk.mul(den_blk));

                let n_blk = match &blk.num_expr {
                    Some(ne) => eval_ast(ne, r_loc, &[], &view_vals),
                    None => Fp3::ONE,
                };
                acc_num = acc_num.add(ind_blk.mul(n_blk));
            }

            let val_a = cur.fp3()?;
            let val_d = cur.fp3()?;
            let gc_raw = &body[gc_start..cur.off];

            // Mask contributions:
            //   m < m_mask - 1 : cancelling pairs (+a_m, d_m) / (-a_m, d_m)
            //   m == m_mask - 1: independent zero-numerator denominator blinders (0, d_L) / (0, d_R)
            let half = gl.half();
            let n_l = gl.libra_coeff_count();
            let last = gl.m_mask - 1;
            let mut a_entries = Vec::with_capacity(last);
            let mut d_entries = Vec::with_capacity(gl.m_mask + 1);
            let mut sum_mask_ind = Fp3::ZERO;
            for m in 0..gl.m_mask {
                let il = half - gl.m_mask + m;
                let ir = gl.n_leaves() - gl.m_mask + m;
                let mut el = Fp3::ONE;
                let mut er = Fp3::ONE;
                for (bit, &rb) in r_gkr.iter().enumerate() {
                    el = el.mul(if (il >> bit) & 1 == 1 {
                        rb
                    } else {
                        Fp3::ONE.sub(rb)
                    });
                    er = er.mul(if (ir >> bit) & 1 == 1 {
                        rb
                    } else {
                        Fp3::ONE.sub(rb)
                    });
                }
                if m < last {
                    a_entries.push((n_l + m, el.sub(er)));
                    d_entries.push((n_l + gl.m_mask + m, el.add(er)));
                } else {
                    d_entries.push((n_l + gl.m_mask + m, el));
                    d_entries.push((n_l + m, er));
                }
                sum_mask_ind = sum_mask_ind.add(el.add(er));
            }
            claims[b.aux_place.group_idx].push(GroupClaim {
                place: b.aux_place.clone(),
                spec: ClaimSpec::LinearIdx(a_entries),
                val: val_a,
            });
            claims[b.aux_place.group_idx].push(GroupClaim {
                place: b.aux_place.clone(),
                spec: ClaimSpec::LinearIdx(d_entries),
                val: val_d,
            });

            let unused_ind = Fp3::ONE.sub(sum_blk_ind.add(sum_mask_ind));
            let expect_num = acc_num.add(val_a);
            let expect_den = acc_den.add(unused_ind).add(val_d);
            if expect_num != ver.num_eval || expect_den != ver.den_eval {
                return Err(RejectCode::Gkr);
            }

            for (idx_k, (r_k, &g_r)) in ver
                .layer_points
                .iter()
                .zip(ver.layer_libra_evals.iter())
                .enumerate()
            {
                let k = idx_k + 1;
                let off_l = gl.layer_libra_offset(k);
                let mut entries_k = Vec::with_capacity(4 * k);
                for (i, &ri) in r_k.iter().enumerate() {
                    let r2 = ri.mul(ri);
                    let r3 = r2.mul(ri);
                    entries_k.push((off_l + 4 * i, Fp3::ONE));
                    entries_k.push((off_l + 4 * i + 1, ri));
                    entries_k.push((off_l + 4 * i + 2, r2));
                    entries_k.push((off_l + 4 * i + 3, r3));
                }
                claims[b.aux_place.group_idx].push(GroupClaim {
                    place: b.aux_place.clone(),
                    spec: ClaimSpec::LinearIdx(entries_k),
                    val: g_r,
                });
            }
            tr.absorb_bytes(gc_raw);
        }
    }

    // ---- 2. Verify all Libra-masked Sumchecks
    for sc in &desc.sumchecks {
        let mut r_zc = Vec::with_capacity(sc.nv);
        for _ in 0..sc.nv {
            r_zc.push(tr.squeeze_fp3());
        }
        let mut lambdas = Vec::with_capacity(sc.n_constraints);
        for _ in 0..sc.n_constraints {
            lambdas.push(tr.squeeze_fp3());
        }

        let g_start = cur.off;
        let g_sum = cur.fp3()?;
        let mut g_hdr = Vec::with_capacity(28);
        g_hdr.extend_from_slice(&(sc.nv as u32).to_le_bytes());
        g_hdr.extend_from_slice(&body[g_start..cur.off]);
        tr.absorb_bytes(&g_hdr);
        let rho = tr.squeeze_fp3();

        let mut cl = rho.mul(g_sum);
        let mut r = Vec::with_capacity(sc.nv);
        for _ in 0..sc.nv {
            let rd_start = cur.off;
            let mut vals = Vec::with_capacity(sc.deg + 1);
            for _ in 0..=sc.deg {
                vals.push(cur.fp3()?);
            }
            if vals[0].add(vals[1]) != cl {
                return Err(RejectCode::Sumcheck);
            }
            tr.absorb_bytes(&body[rd_start..cur.off]);
            let a = tr.squeeze_fp3();
            r.push(a);
            cl = lagrange_eval(&vals, a);
        }

        let c_start = cur.off;
        let mut view_vals = Vec::with_capacity(sc.view_refs.len());
        for vr in &sc.view_refs {
            let val = cur.fp3()?;
            let loc = local_point_of(vr, &r);
            let spec = if vr.shift != 0 {
                ClaimSpec::Sh(loc, vr.shift)
            } else {
                ClaimSpec::Pt(loc)
            };
            claims[vr.place.group_idx].push(GroupClaim {
                place: vr.place.clone(),
                spec,
                val,
            });
            view_vals.push(val.mul(ind_fixed(&vr.fixed, &r)));
        }
        let gval = cur.fp3()?;
        claims[sc.mask_place.group_idx].push(GroupClaim {
            place: sc.mask_place.clone(),
            spec: ClaimSpec::Pw(r.clone(), sc.deg),
            val: gval,
        });
        tr.absorb_bytes(&body[c_start..cur.off]);

        let mut lhs = rho.mul(gval);
        for (ast, &lam) in sc.constraints.iter().zip(lambdas.iter()) {
            lhs = lhs.add(lam.mul(eval_ast(ast, &r, &r_zc, &view_vals)));
        }
        if lhs != cl {
            return Err(RejectCode::Sumcheck);
        }
    }

    // ---- 3. Verify batched multi-size Coset-4 Folded ZK-Basefold opening
    let pcs_lay = MultiPcsLayout::new(desc.n_max, desc.queries, desc.group_ns.clone())?;
    let mut group_mus: Vec<Vec<Fp3>> = Vec::with_capacity(g_cnt);
    let mut group_claims: Vec<Fp3> = Vec::with_capacity(g_cnt);
    for gi in 0..g_cnt {
        let gcl = &claims[gi];
        tr.absorb_bytes(&(gcl.len() as u32).to_le_bytes());
        let mut mus = Vec::with_capacity(gcl.len());
        let mut y_g = Fp3::ZERO;
        for cl in gcl {
            let mu = tr.squeeze_fp3();
            y_g = y_g.add(mu.mul(cl.val));
            mus.push(mu);
        }
        group_mus.push(mus);
        group_claims.push(y_g);
    }

    let pcs_pf = cur.take(pcs_lay.proof_bytes())?;
    debug_assert_eq!(cur.off, body.len());

    verify_multi_open(
        &pcs_lay,
        &group_roots,
        &group_claims,
        |gi, r_ip_g| {
            let gcl = &claims[gi];
            let mus = &group_mus[gi];
            let mut acc = Fp3::ZERO;
            for (cl, &mu) in gcl.iter().zip(mus.iter()) {
                let ind = block_ind_at(cl.place.offset, cl.place.nbits, r_ip_g);
                let loc = eval_claim_weight_mle(cl, &r_ip_g[..cl.place.nbits]);
                acc = acc.add(mu.mul(ind.mul(loc)));
            }
            acc
        },
        pcs_pf,
        &mut tr,
    )?;

    Ok(Relation2Claim {
        profile_id: hdr.profile_id,
        circuit_root,
        stmt_digest,
        body_used: used,
        c_p: prof.c_p as usize,
    })
}
