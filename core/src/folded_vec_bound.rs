// SPDX-License-Identifier: Apache-2.0
//! Relation 6 (`FOLDED_VEC_BOUND`, `specs/11-wire-format.md` §5.7): relation 5 with the activations `X` moved out
//! of the witness vector `F` (root `root_0`) into a SEPARATE pre-committed, ZK-blinded vector `X_com` (root
//! `x_root`, pinned by the caller and anchored into `C_pre` before the challenge). Closes the relation-5 case-(c)
//! gap: the proof is about the `X` committed under the pinned `x_root`, not about a privately substituted one.
//!
//! One Libra-masked inner-product sumcheck (identical to relation 5) reduces every `Y[b][i] = Σ_j W[b][i][j]
//! X[b][j]` to linear claims on `F` (Y, Libra, mask), `X_com` (X) and the public `C` (W), discharged by ONE batched
//! folded ZK-Basefold opening of the THREE vectors: batching slots `[F, F + C, F + X_com, F + C + X_com]` folded by
//! the first two sumcheck challenges `(θ, r_1)` give `H = F + θ C + r_1 X_com`; zero extra fold layers.
#![forbid(unsafe_code)]

use crate::field::{lagrange_eval, Fp, Fp3, P};
use crate::folded_air::{dec1, decv, vals, weights_eval, Block, Reader, MASK_POINTS};
use crate::folded_gkr::{eq, fold4, merkle_ok};
use crate::folded_vec::{VecLayout, IP_DEGREE, RATE, ROW_BITS};
use crate::profile::Profile;
use crate::transcript::{Fpn2Transcript, LABEL_V2};
use crate::verify::RejectCode;

/// `root_0 ‖ weight_root ‖ x_root ‖ N_s`.
pub const STMT: usize = 128;

/// Relation-6 geometry: relation 5's `(n, Q, k)` plus the `X_com` layer-0 leaf in every query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundLayout(pub VecLayout);

impl BoundLayout {
    pub fn of(p: &Profile) -> Option<BoundLayout> {
        VecLayout::of(p).map(BoundLayout)
    }
    pub fn query_len(&self) -> usize {
        let l = &self.0;
        2 * (96 + 32 * l.depth(0)) + (0..l.folds()).map(|t| 96 + 32 * l.depth(t)).sum::<usize>()
    }
    pub fn body_used(&self) -> usize {
        let l = &self.0;
        STMT + l.ip_len() + l.pcs_len() + l.queries * self.query_len()
    }
}

/// Normative check 12 for relation 6: canonical decoding of every `Fp3` field in wire order.
pub fn decode(body: &[u8], lay: &BoundLayout) -> Result<(), RejectCode> {
    let l = &lay.0;
    if body.len() != lay.body_used() {
        return Err(RejectCode::Length);
    }
    let mut rd = Reader { b: body, off: 0 };
    rd.take(STMT);
    vals(rd.take(l.ip_len()), l.ip_len() / 24)?;
    vals(rd.take(24), 1)?;
    for i in 0..l.n + 2 {
        vals(rd.take(72), 3)?;
        if l.root_after(i) {
            rd.take(32);
        }
    }
    vals(rd.take(24), 1)?;
    for _ in 0..l.queries {
        for t in 0..l.folds() {
            vals(rd.take(96), 4)?;
            rd.take(32 * l.depth(t));
            if t == 0 {
                vals(rd.take(96), 4)?;
                rd.take(32 * l.depth(0));
                vals(rd.take(96), 4)?;
                rd.take(32 * l.depth(0));
            }
        }
    }
    Ok(())
}

type Terms = (Vec<Block>, Vec<(u64, Fp3)>, Vec<Block>, Vec<Block>);

/// `(F-blocks, F-sparse, X-blocks, C-blocks)` of `target + μ e_W + μ² z_blind` (§5.7.4).
fn terms(
    lay: &VecLayout,
    r_i: &[Fp3],
    r_b: &[Fp3],
    s: &[Fp3],
    lam: Fp3,
    e_w: Fp3,
    mu: Fp3,
) -> Terms {
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
    let mut fb = vec![blk(
        lay.tiles() * lay.kdim(),
        ROW_BITS + beta,
        eqrb.mul(e_s).neg(),
        y_point,
    )];
    let xb = vec![blk(0, lay.d(), eqrb.mul(e_w), s.to_vec())];
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
    (fb, sparse, xb, cb)
}

/// Verifies a relation-6 body (checks 12–17). `weight_root` and `x_root` are the caller's pinned commitments
/// (both fail closed).
pub fn verify(
    header: &[u8],
    body: &[u8],
    lay: &BoundLayout,
    weight_root: Option<&[u8; 32]>,
    x_root: Option<&[u8; 32]>,
) -> Result<(), RejectCode> {
    decode(body, lay)?; // 12
    let l = &lay.0;
    let mut rd = Reader { b: body, off: 0 };
    let stmt = rd.take(STMT);
    let pin_w = weight_root.ok_or(RejectCode::CircuitUnpinned)?; // 13
    let pin_x = x_root.ok_or(RejectCode::XRootUnpinned)?;
    if &stmt[32..64] != pin_w.as_slice() {
        return Err(RejectCode::CircuitPinned);
    }
    if &stmt[64..96] != pin_x.as_slice() {
        return Err(RejectCode::XRootPinned);
    }
    let mut tr = Fpn2Transcript::new(LABEL_V2);
    tr.absorb_bytes(header);
    tr.absorb_bytes(stmt);
    let r_i: Vec<Fp3> = (0..ROW_BITS).map(|_| tr.squeeze_fp3()).collect();
    let r_b: Vec<Fp3> = (0..l.beta()).map(|_| tr.squeeze_fp3()).collect();
    // 14: inner-product sumcheck
    let gs_raw = rd.take(24);
    tr.absorb_bytes(gs_raw);
    let lam = tr.squeeze_fp3();
    let mut target = lam.mul(dec1(gs_raw));
    let mut s = Vec::with_capacity(l.d());
    for _ in 0..l.d() {
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
    let (fb, sparse, xb, cb) = terms(l, &r_i, &r_b, &s, lam, e_w, mu);
    let mut y = target.add(mu.mul(e_w)).add(mu.mul(mu).mul(dec1(zb_raw)));
    // 15: batched Basefold sumcheck
    let mut rs: Vec<Fp3> = Vec::with_capacity(l.n + 2);
    let mut roots: Vec<&[u8]> = vec![&stmt[..32]];
    for i in 0..l.n + 2 {
        let raw = rd.take(72);
        tr.absorb_bytes(raw);
        let h = decv(raw, 3);
        if h[0].add(h[1]) != y {
            return Err(RejectCode::Sumcheck);
        }
        let r = tr.squeeze_fp3();
        y = lagrange_eval(&h, r);
        rs.push(r);
        if l.root_after(i) {
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
    let a_r = weights_eval(l.n, &fb, &sparse, rest);
    let x_r = weights_eval(l.n, &xb, &[], rest);
    let c_r = weights_eval(l.n, &cb, &[], rest);
    let (om_t, om_r) = (Fp3::ONE.sub(theta), Fp3::ONE.sub(r1));
    let w_r = om_r
        .mul(om_t)
        .mul(a_r)
        .add(om_r.mul(theta.add(theta).sub(Fp3::ONE)).mul(c_r))
        .add(om_t.mul(r1.add(r1).sub(Fp3::ONE)).mul(x_r));
    if y != final_.mul(w_r) {
        return Err(RejectCode::Final); // 16
    }
    // 17: queries
    let length: u64 = RATE << l.n;
    let omega = Fp::new(7).pow((P - 1) / length);
    let iota_inv = Fp::new(7).pow((P - 1) / 4).inv().ok_or(RejectCode::Final)?;
    let (root_w, root_x) = (&stmt[32..64], &stmt[64..96]);
    for _ in 0..l.queries {
        let raw8 = tr.squeeze_bytes(8);
        let mut b8 = [0u8; 8];
        b8.copy_from_slice(&raw8);
        let mut pos = u64::from_le_bytes(b8) % (length / 4);
        let mut prev = Fp3::ZERO;
        for t in 0..l.folds() {
            let quarter = length >> (2 * t + 2);
            let (j, slot) = (pos % quarter, (pos / quarter) as usize);
            let vraw = rd.take(96);
            let sibs: Vec<&[u8]> = (0..l.depth(t)).map(|_| rd.take(32)).collect();
            if !merkle_ok(roots[t], vraw, j, &sibs) {
                return Err(RejectCode::Merkle);
            }
            let mut v = decv(vraw, 4);
            if t == 0 {
                let craw = rd.take(96);
                let csibs: Vec<&[u8]> = (0..l.depth(0)).map(|_| rd.take(32)).collect();
                if !merkle_ok(root_w, craw, j, &csibs) {
                    return Err(RejectCode::Merkle);
                }
                let xraw = rd.take(96);
                let xsibs: Vec<&[u8]> = (0..l.depth(0)).map(|_| rd.take(32)).collect();
                if !merkle_ok(root_x, xraw, j, &xsibs) {
                    return Err(RejectCode::Merkle);
                }
                let (cv, xv) = (decv(craw, 4), decv(xraw, 4));
                for e in 0..4 {
                    v[e] = v[e].add(theta.mul(cv[e])).add(r1.mul(xv[e]));
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
    use crate::profile::{FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1};

    #[test]
    fn layout_of_registered_profiles() {
        let lay = BoundLayout::of(&FOLDED_VEC_BOUND_N16_V1).unwrap();
        assert_eq!(lay.0.tiles(), 1);
        assert_eq!(lay.query_len(), 4608);
        assert_eq!(lay.body_used(), 334_480);
        assert!(64 + lay.body_used() <= FOLDED_VEC_BOUND_N16_V1.c_p as usize);
        let lay = BoundLayout::of(&FOLDED_VEC_BOUND_N22_V1).unwrap();
        assert_eq!((lay.0.tiles(), lay.0.d()), (64, 16));
        assert_eq!(lay.body_used(), 529_120);
        assert!(64 + lay.body_used() <= FOLDED_VEC_BOUND_N22_V1.c_p as usize);
        // n = 22 does not fit the relation-5 container size (512 KiB): C_p = 1 MiB.
        assert!(64 + lay.body_used() > 524_288);
    }
}
