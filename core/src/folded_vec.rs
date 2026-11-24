// SPDX-License-Identifier: Apache-2.0
//! Relation 5 (`FOLDED_VEC_MATMUL`, `specs/11-wire-format.md` §5.6): `B` public full-width `32 × K` GEMM
//! tiles `W` (committed in the public vector `C`, root `weight_root`, pinned by the caller) times private
//! activations `X` with private outputs `Y` (both in the witness vector `F`, root `root_0`).
//!
//! ONE Libra-masked inner-product sumcheck (zero-claim form, degree 3) reduces every `Y[b][i] = Σ_j W[b][i][j]
//! X[b][j]` to linear claims on `F` plus the public evaluation `e_W = W̃(s_b, r_i, s_j)`, all discharged by ONE
//! batched folded ZK-Basefold opening of `(F, C)` (relation 4's two batching rounds + coset-4 fold chain).
#![forbid(unsafe_code)]

use crate::field::{lagrange_eval, Fp, Fp3, P};
use crate::folded_air::{dec1, decv, vals, weights_eval, Block, Fixed, Reader, MASK_POINTS};
use crate::folded_gkr::{eq, fold4, merkle_ok};
use crate::profile::Profile;
use crate::transcript::{Fpn2Transcript, LABEL_V2};
use crate::verify::RejectCode;

pub const RATE: u64 = 8;
pub const ROWS: usize = 32;
pub const ROW_BITS: usize = 5;
pub const IP_DEGREE: usize = 3;

/// Relation-5 geometry, a pure function of `(n_vars, n_queries, max_hidden)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VecLayout {
    pub n: usize,
    pub queries: usize,
    /// `log2 K` (`K = max_hidden`, the tile width).
    pub k: usize,
}

impl VecLayout {
    pub fn of(p: &Profile) -> Option<VecLayout> {
        let n = p.n_vars as usize;
        let kd = p.max_hidden as usize;
        if p.sc_degree != 3
            || n < 12
            || !n.is_multiple_of(2)
            || n > 28
            || !(1..=4096).contains(&p.n_queries)
            || kd < ROWS
            || !kd.is_power_of_two()
        {
            return None;
        }
        let k = kd.trailing_zeros() as usize;
        if n < 1 + ROW_BITS + k {
            return None;
        }
        Some(VecLayout {
            n,
            queries: p.n_queries as usize,
            k,
        })
    }
    /// `log2 B` (number of tiles).
    pub fn beta(&self) -> usize {
        self.n - 1 - ROW_BITS - self.k
    }
    pub fn tiles(&self) -> u64 {
        1 << self.beta()
    }
    pub fn kdim(&self) -> u64 {
        1 << self.k
    }
    /// Inner-product sumcheck variables (`k` column bits, then `beta` tile bits).
    pub fn d(&self) -> usize {
        self.k + self.beta()
    }
    pub fn folds(&self) -> usize {
        self.n / 2
    }
    pub fn depth(&self, t: usize) -> usize {
        self.n + 1 - 2 * t
    }
    pub fn ip_len(&self) -> usize {
        24 + 24 * (IP_DEGREE + 1) * self.d() + 24
    }
    pub fn pcs_len(&self) -> usize {
        24 + 72 * (self.n + 2) + 32 * (self.folds() - 1) + 24
    }
    pub fn query_len(&self) -> usize {
        (96 + 32 * self.depth(0))
            + (0..self.folds())
                .map(|t| 96 + 32 * self.depth(t))
                .sum::<usize>()
    }
    pub fn body_used(&self) -> usize {
        96 + self.ip_len() + self.pcs_len() + self.queries * self.query_len()
    }
    fn free_bits(&self) -> Vec<usize> {
        let mut v = vec![0];
        v.extend(2..self.n);
        v
    }
    pub(crate) fn block(&self, off: u64, kk: usize) -> (Fixed, Vec<usize>) {
        let fb = self.free_bits();
        let mut fixed = vec![(1usize, 1u64)];
        for i in 0..fb.len() - kk {
            fixed.push((fb[kk + i], (off >> (kk + i)) & 1));
        }
        (fixed, fb[..kk].to_vec())
    }
    pub(crate) fn libra_slot(&self, l: usize, e: usize) -> u64 {
        let v = (4 * l + e) as u64;
        let mut idx = 0u64;
        for (j, bit) in self.free_bits().iter().enumerate() {
            idx |= ((v >> j) & 1) << bit;
        }
        idx
    }
    pub(crate) fn root_after(&self, i: usize) -> bool {
        i >= 3 && i % 2 == 1 && (i - 3) / 2 < self.folds() - 1
    }
}

/// Normative check 12 for relation 5: canonical decoding of every `Fp3` field in wire order.
pub fn decode(body: &[u8], lay: &VecLayout) -> Result<(), RejectCode> {
    if body.len() != lay.body_used() {
        return Err(RejectCode::Length);
    }
    let mut rd = Reader { b: body, off: 0 };
    rd.take(96);
    vals(rd.take(lay.ip_len()), lay.ip_len() / 24)?;
    vals(rd.take(24), 1)?;
    for i in 0..lay.n + 2 {
        vals(rd.take(72), 3)?;
        if lay.root_after(i) {
            rd.take(32);
        }
    }
    vals(rd.take(24), 1)?;
    for _ in 0..lay.queries {
        for t in 0..lay.folds() {
            vals(rd.take(96), 4)?;
            rd.take(32 * lay.depth(t));
            if t == 0 {
                vals(rd.take(96), 4)?;
                rd.take(32 * lay.depth(0));
            }
        }
    }
    Ok(())
}

/// `(F-blocks, F-sparse, C-blocks)` of `target + μ e_W + μ² z_blind` (§5.6.4).
fn terms(
    lay: &VecLayout,
    r_i: &[Fp3],
    r_b: &[Fp3],
    s: &[Fp3],
    lam: Fp3,
    e_w: Fp3,
    mu: Fp3,
) -> (Vec<Block>, Vec<(u64, Fp3)>, Vec<Block>) {
    let (k, beta) = (lay.k, lay.beta());
    let (s_j, s_b) = (&s[..k], &s[k..]);
    let eqrb = eq(r_b, s_b);
    let mut e_s = eq(r_i, &s_j[..ROW_BITS]);
    for t in &s_j[ROW_BITS..] {
        e_s = e_s.mul(Fp3::ONE.sub(*t));
    }
    let blk = |off: u64, kk: usize, c: Fp3, z: Vec<Fp3>| -> Block {
        let (fixed, free) = lay.block(off, kk);
        (fixed, free, c, z)
    };
    let y_point: Vec<Fp3> = s_j[..ROW_BITS].iter().chain(s_b.iter()).copied().collect();
    let mut fb = vec![
        blk(0, lay.d(), eqrb.mul(e_w), s.to_vec()),
        blk(
            lay.tiles() * lay.kdim(),
            ROW_BITS + beta,
            eqrb.mul(e_s).neg(),
            y_point,
        ),
    ];
    let mut sparse = Vec::with_capacity(lay.d() * (IP_DEGREE + 1));
    for (l, r) in s.iter().enumerate() {
        let mut pw = lam;
        for e in 0..=IP_DEGREE {
            sparse.push((lay.libra_slot(l, e), pw));
            pw = pw.mul(*r);
        }
    }
    let coef = mu.mul(mu);
    let free: Vec<usize> = (0..lay.n).filter(|b| *b != 1).collect();
    for zv in MASK_POINTS {
        let z = Fp3::from_u64(zv);
        fb.push((
            vec![(1, 0)],
            free.clone(),
            coef.mul(Fp3::ONE.sub(z)),
            vec![z; free.len()],
        ));
    }
    let w_point: Vec<Fp3> = s_j
        .iter()
        .chain(r_i.iter())
        .chain(s_b.iter())
        .copied()
        .collect();
    let cb = vec![blk(0, lay.n - 1, mu, w_point)];
    (fb, sparse, cb)
}

/// Verifies a relation-5 body (checks 12–17). `weight_root` is the caller's pinned commitment.
pub fn verify(
    header: &[u8],
    body: &[u8],
    lay: &VecLayout,
    weight_root: Option<&[u8; 32]>,
) -> Result<(), RejectCode> {
    decode(body, lay)?; // 12
    let mut rd = Reader { b: body, off: 0 };
    let stmt = rd.take(96);
    let pin = weight_root.ok_or(RejectCode::CircuitUnpinned)?; // 13
    if &stmt[32..64] != pin.as_slice() {
        return Err(RejectCode::CircuitPinned);
    }
    let mut tr = Fpn2Transcript::new(LABEL_V2);
    tr.absorb_bytes(header);
    tr.absorb_bytes(stmt);
    let r_i: Vec<Fp3> = (0..ROW_BITS).map(|_| tr.squeeze_fp3()).collect();
    let r_b: Vec<Fp3> = (0..lay.beta()).map(|_| tr.squeeze_fp3()).collect();
    // 14: inner-product sumcheck
    let gs_raw = rd.take(24);
    tr.absorb_bytes(gs_raw);
    let lam = tr.squeeze_fp3();
    let mut target = lam.mul(dec1(gs_raw));
    let mut s = Vec::with_capacity(lay.d());
    for _ in 0..lay.d() {
        let raw = rd.take(24 * (IP_DEGREE + 1));
        tr.absorb_bytes(raw);
        let sv = decv(raw, IP_DEGREE + 1);
        if sv[0].add(sv[1]) != target {
            return Err(RejectCode::Zerocheck);
        }
        let r = tr.squeeze_fp3();
        target = lagrange_eval(&sv, r);
        s.push(r);
    }
    let ew_raw = rd.take(24);
    tr.absorb_bytes(ew_raw);
    let e_w = dec1(ew_raw);
    let zb_raw = rd.take(24);
    tr.absorb_bytes(zb_raw);
    let mu = tr.squeeze_fp3();
    let (fb, sparse, cb) = terms(lay, &r_i, &r_b, &s, lam, e_w, mu);
    let mut y = target
        .add(mu.mul(e_w))
        .add(mu.mul(mu).mul(dec1(zb_raw)));
    // 15: batched Basefold sumcheck
    let mut rs: Vec<Fp3> = Vec::with_capacity(lay.n + 2);
    let mut roots: Vec<&[u8]> = vec![&stmt[..32]];
    for i in 0..lay.n + 2 {
        let raw = rd.take(72);
        tr.absorb_bytes(raw);
        let h = decv(raw, 3);
        if h[0].add(h[1]) != y {
            return Err(RejectCode::Sumcheck);
        }
        let r = tr.squeeze_fp3();
        y = lagrange_eval(&h, r);
        rs.push(r);
        if lay.root_after(i) {
            let root_t = rd.take(32);
            tr.absorb_bytes(root_t);
            roots.push(root_t);
        }
    }
    let final_raw = rd.take(24);
    tr.absorb_bytes(final_raw);
    let final_ = dec1(final_raw);
    let (theta, r1) = (rs[0], rs[1]);
    let rest = &rs[2..];
    let a_r = weights_eval(lay.n, &fb, &sparse, rest);
    let c_r = weights_eval(lay.n, &cb, &[], rest);
    let w_r = Fp3::ONE.sub(r1).mul(
        Fp3::ONE
            .sub(theta)
            .mul(a_r)
            .add(theta.add(theta).sub(Fp3::ONE).mul(c_r)),
    );
    if y != final_.mul(w_r) {
        return Err(RejectCode::Final); // 16
    }
    // 17: queries
    let length: u64 = RATE << lay.n;
    let omega = Fp::new(7).pow((P - 1) / length);
    let iota_inv = Fp::new(7).pow((P - 1) / 4).inv().ok_or(RejectCode::Final)?;
    let root_w = &stmt[32..64];
    for _ in 0..lay.queries {
        let raw8 = tr.squeeze_bytes(8);
        let mut b8 = [0u8; 8];
        b8.copy_from_slice(&raw8);
        let mut pos = u64::from_le_bytes(b8) % (length / 4);
        let mut prev = Fp3::ZERO;
        for t in 0..lay.folds() {
            let quarter = length >> (2 * t + 2);
            let (j, slot) = (pos % quarter, (pos / quarter) as usize);
            let vraw = rd.take(96);
            let sibs: Vec<&[u8]> = (0..lay.depth(t)).map(|_| rd.take(32)).collect();
            if !merkle_ok(roots[t], vraw, j, &sibs) {
                return Err(RejectCode::Merkle);
            }
            let mut v = decv(vraw, 4);
            if t == 0 {
                let craw = rd.take(96);
                let csibs: Vec<&[u8]> = (0..lay.depth(0)).map(|_| rd.take(32)).collect();
                if !merkle_ok(root_w, craw, j, &csibs) {
                    return Err(RejectCode::Merkle);
                }
                let cv = decv(craw, 4);
                for e in 0..4 {
                    v[e] = v[e].add(theta.mul(cv[e]));
                }
            } else if v[slot] != prev {
                return Err(RejectCode::Fold);
            }
            let omega_t = omega.pow(1u64 << (2 * t));
            let x_inv = omega_t.pow((P - 1 - j) % (P - 1));
            prev = fold4(
                &[v[0], v[1], v[2], v[3]],
                x_inv,
                iota_inv,
                rest[2 * t],
                rest[2 * t + 1],
            );
            pos = j;
        }
        if prev != final_ {
            return Err(RejectCode::Fold);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1};

    #[test]
    fn layout_of_registered_profiles() {
        let lay = VecLayout::of(&FOLDED_VEC_N16_V1).unwrap();
        assert_eq!((lay.k, lay.beta(), lay.tiles(), lay.d()), (10, 0, 1, 10));
        assert_eq!(lay.tiles() * ROWS as u64 * lay.kdim(), 1 << 15);
        assert_eq!(lay.body_used(), 288_368);
        assert!(64 + lay.body_used() <= FOLDED_VEC_N16_V1.c_p as usize);
        let lay = VecLayout::of(&FOLDED_VEC_N22_V1).unwrap();
        assert_eq!((lay.k, lay.beta(), lay.tiles(), lay.d()), (10, 6, 64, 16));
        assert_eq!(lay.tiles() * ROWS as u64 * lay.kdim(), 2_097_152);
        assert_eq!(lay.body_used(), 469_184);
        assert!(64 + lay.body_used() <= FOLDED_VEC_N22_V1.c_p as usize);
    }
}
