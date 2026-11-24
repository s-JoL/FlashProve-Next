// SPDX-License-Identifier: Apache-2.0
//! Generalized multi-table / weighted ZK-masked FracLogUp-GKR verifier (`multi_gkr`),
//! bit-exact with `prover/fpn_prover/backend/multi_gkr.py`.

use crate::field::{lagrange_eval, Fp3};
use crate::folded_gkr::{eq, mix};
use crate::transcript::Fpn2Transcript;
use crate::verify::RejectCode;

/// Layout of one batched FracLogUp-GKR binary fraction tree of height `s` (2^s leaves).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiGkrLayout {
    pub s: usize,
    pub m_mask: usize,
}

impl MultiGkrLayout {
    pub fn new(s: usize, m_mask: usize) -> Result<Self, RejectCode> {
        if !(3..=28).contains(&s) || m_mask < 2 || m_mask > (1usize << (s - 2)) {
            return Err(RejectCode::Length);
        }
        Ok(Self { s, m_mask })
    }

    pub fn n_leaves(&self) -> usize {
        1usize << self.s
    }

    pub fn half(&self) -> usize {
        1usize << (self.s - 1)
    }

    pub fn libra_coeff_count(&self) -> usize {
        2 * (self.s - 1) * self.s
    }

    pub fn layer_libra_offset(&self, k: usize) -> usize {
        2 * (k - 1) * k
    }

    pub fn proof_bytes(&self) -> usize {
        let layers: usize = (1..self.s).map(|k| 24 + 96 * k + 96 + 24).sum();
        48 + 96 + layers
    }
}

/// Verified claims output by `verify_multi_gkr`, to be discharged by `multi_pcs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedGkrClaims {
    pub r_gkr: Vec<Fp3>,
    pub num_eval: Fp3,
    pub den_eval: Fp3,
    pub layer_points: Vec<Vec<Fp3>>,
    pub layer_libra_evals: Vec<Fp3>,
}

fn combine_frac(nl: Fp3, dl: Fp3, nr: Fp3, dr: Fp3) -> (Fp3, Fp3) {
    (nl.mul(dr).add(nr.mul(dl)), dl.mul(dr))
}

fn read_fp3(buf: &[u8], off: &mut usize) -> Result<Fp3, RejectCode> {
    if *off + 24 > buf.len() {
        return Err(RejectCode::Length);
    }
    let val = Fp3::decode(&buf[*off..*off + 24]).map_err(|_| RejectCode::NonCanonical)?;
    *off += 24;
    Ok(val)
}

/// Verify a `multi_gkr` proof slice, advancing `tr` in normative raw-byte order.
pub fn verify_multi_gkr(
    layout: &MultiGkrLayout,
    proof: &[u8],
    tr: &mut Fpn2Transcript,
) -> Result<VerifiedGkrClaims, RejectCode> {
    if proof.len() != layout.proof_bytes() {
        return Err(RejectCode::Length);
    }
    let mut off = 0usize;

    let r_start = off;
    let root_num = read_fp3(proof, &mut off)?;
    let root_den = read_fp3(proof, &mut off)?;
    if root_num != Fp3::ZERO || root_den == Fp3::ZERO {
        return Err(RejectCode::GkrRoot);
    }
    tr.absorb_bytes(&proof[r_start..off]);

    // Layer 0
    let c0_start = off;
    let nl0 = read_fp3(proof, &mut off)?;
    let dl0 = read_fp3(proof, &mut off)?;
    let nr0 = read_fp3(proof, &mut off)?;
    let dr0 = read_fp3(proof, &mut off)?;
    if combine_frac(nl0, dl0, nr0, dr0) != (root_num, root_den) {
        return Err(RejectCode::Gkr);
    }
    tr.absorb_bytes(&proof[c0_start..off]);

    let br0 = tr.squeeze_fp3();
    let mut pt = vec![br0];
    let mut claim_num = mix(nl0, nr0, br0);
    let mut claim_den = mix(dl0, dr0, br0);

    let mut layer_points = Vec::with_capacity(layout.s - 1);
    let mut layer_libra_evals = Vec::with_capacity(layout.s - 1);

    for k in 1..layout.s {
        let alpha = tr.squeeze_fp3();
        let gs_start = off;
        let g_sum = read_fp3(proof, &mut off)?;
        tr.absorb_bytes(&proof[gs_start..off]);
        let lam = tr.squeeze_fp3();

        let mut target = claim_num.add(alpha.mul(claim_den)).add(lam.mul(g_sum));
        let mut r_k = Vec::with_capacity(k);
        for _ in 0..k {
            let rd_start = off;
            let v0 = read_fp3(proof, &mut off)?;
            let v1 = read_fp3(proof, &mut off)?;
            let v2 = read_fp3(proof, &mut off)?;
            let v3 = read_fp3(proof, &mut off)?;
            if v0.add(v1) != target {
                return Err(RejectCode::Gkr);
            }
            tr.absorb_bytes(&proof[rd_start..off]);
            let ri = tr.squeeze_fp3();
            r_k.push(ri);
            target = lagrange_eval(&[v0, v1, v2, v3], ri);
        }

        let fin_start = off;
        let nl = read_fp3(proof, &mut off)?;
        let dl = read_fp3(proof, &mut off)?;
        let nr = read_fp3(proof, &mut off)?;
        let dr = read_fp3(proof, &mut off)?;
        let g_r = read_fp3(proof, &mut off)?;
        tr.absorb_bytes(&proof[fin_start..off]);

        let (c_num, c_den) = combine_frac(nl, dl, nr, dr);
        let eq_val = eq(&pt, &r_k);
        let expect = eq_val
            .mul(c_num.add(alpha.mul(c_den)))
            .add(lam.mul(g_r));
        if expect != target {
            return Err(RejectCode::Gkr);
        }

        layer_points.push(r_k.clone());
        layer_libra_evals.push(g_r);

        let br_k = tr.squeeze_fp3();
        let mut next_pt = Vec::with_capacity(k + 1);
        next_pt.push(br_k);
        next_pt.extend_from_slice(&r_k);
        pt = next_pt;
        claim_num = mix(nl, nr, br_k);
        claim_den = mix(dl, dr, br_k);
    }

    debug_assert_eq!(off, proof.len());
    Ok(VerifiedGkrClaims {
        r_gkr: pt,
        num_eval: claim_num,
        den_eval: claim_den,
        layer_points,
        layer_libra_evals,
    })
}
