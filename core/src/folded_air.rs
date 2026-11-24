// SPDX-License-Identifier: Apache-2.0
//! Relation 4 (`FOLDED_GKR_AIR`, `specs/11-wire-format.md` §5.5): a Libra-masked Plonkish zerocheck
//! (gate `q_M a b + q_L a + q_R b + q_O c + q_C = 0`) plus ONE ZK-masked FracLogUp-GKR tree for the
//! copy constraints (wiring, challenge `gamma`) and the `[0, 128)` limb range lookup, all reduced to
//! linear claims on a private witness vector `F` (`root_0`) and a public circuit vector `C`
//! (`root_circ`, pinned by the caller) that are discharged by ONE batched folded ZK-Basefold opening
//! (two batching rounds, then the fused coset-4 fold chain of `H = F + r_0 C`).
#![forbid(unsafe_code)]

use crate::field::{lagrange_eval, Fp, Fp3, P};
use crate::folded_gkr::{bit3, eq, fold4, merkle_ok, mix};
use crate::profile::Profile;
use crate::transcript::{Fpn2Transcript, LABEL_V2};
use crate::verify::RejectCode;

pub const RATE: u64 = 8;
pub const TABLE_BITS: usize = 7;
pub const ZC_DEGREE: usize = 4;
pub const N_FINALS: usize = 9;
pub(crate) const MASK_POINTS: [u64; 2] = [2, 3];

/// Relation-4 geometry, a pure function of `(n_vars, n_queries)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AirLayout {
    pub n: usize,
    pub queries: usize,
}

pub(crate) type Fixed = Vec<(usize, u64)>;
pub(crate) type Block = (Fixed, Vec<usize>, Fp3, Vec<Fp3>);

impl AirLayout {
    pub fn of(p: &Profile) -> Option<AirLayout> {
        let n = p.n_vars as usize;
        if p.sc_degree != 3
            || n < 12
            || !n.is_multiple_of(2)
            || n > 28
            || !(1..=4096).contains(&p.n_queries)
        {
            return None;
        }
        Some(AirLayout {
            n,
            queries: p.n_queries as usize,
        })
    }
    pub fn g(&self) -> usize {
        self.n - 5
    }
    pub fn k(&self) -> u64 {
        1 << self.g()
    }
    pub fn s(&self) -> usize {
        self.g() + 3
    }
    pub fn folds(&self) -> usize {
        self.n / 2
    }
    pub fn depth(&self, t: usize) -> usize {
        self.n + 1 - 2 * t
    }
    pub fn layer_len(k: usize) -> usize {
        24 + 96 * k + 120
    }
    pub fn gkr_len(&self) -> usize {
        48 + 96 + (1..=self.s()).map(AirLayout::layer_len).sum::<usize>()
    }
    pub fn zc_len(&self) -> usize {
        24 + self.g() * 24 * (ZC_DEGREE + 1) + 24 * N_FINALS
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
        96 + self.gkr_len() + self.zc_len() + self.pcs_len() + self.queries * self.query_len()
    }
    fn free_bits(&self) -> Vec<usize> {
        let mut v = vec![0];
        v.extend(2..self.n);
        v
    }
    fn block(&self, off: u64, k: usize) -> (Fixed, Vec<usize>) {
        let fb = self.free_bits();
        let mut fixed = vec![(1usize, 1u64)];
        for i in 0..fb.len() - k {
            fixed.push((fb[k + i], (off >> (k + i)) & 1));
        }
        (fixed, fb[..k].to_vec())
    }
    fn deposit(&self, v: u64) -> u64 {
        // Libra half {bit1 = 0}.
        let mut idx = 0u64;
        for (j, bit) in self.free_bits().iter().enumerate() {
            idx |= ((v >> j) & 1) << bit;
        }
        idx
    }
    fn libra_slot(&self, k: usize, i: usize, e: usize) -> u64 {
        self.deposit((2 * k * (k - 1) + 4 * i + e) as u64)
    }
    fn zc_slot(&self, i: usize, e: usize) -> u64 {
        let s = self.s();
        self.deposit((2 * s * (s + 1) + (ZC_DEGREE + 1) * i + e) as u64)
    }
    /// Wire order of PCS round `i` (0-based over `n + 2` rounds): is it followed by a fold root?
    fn root_after(&self, i: usize) -> bool {
        i >= 3 && i % 2 == 1 && (i - 3) / 2 < self.folds() - 1
    }
}

pub(crate) struct Reader<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) off: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        s
    }
}

pub(crate) fn vals(raw: &[u8], count: usize) -> Result<Vec<Fp3>, RejectCode> {
    (0..count)
        .map(|j| Fp3::decode(&raw[24 * j..24 * j + 24]).map_err(|_| RejectCode::NonCanonical))
        .collect()
}

/// Normative check 12 for relation 4: canonical decoding of every `Fp3` field in wire order.
pub fn decode(body: &[u8], lay: &AirLayout) -> Result<(), RejectCode> {
    if body.len() != lay.body_used() {
        return Err(RejectCode::Length);
    }
    let mut rd = Reader { b: body, off: 0 };
    rd.take(96);
    vals(rd.take(48), 2)?;
    vals(rd.take(96), 4)?;
    for k in 1..=lay.s() {
        let len = AirLayout::layer_len(k);
        vals(rd.take(len), len / 24)?;
    }
    vals(rd.take(lay.zc_len()), lay.zc_len() / 24)?;
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

fn int_ext(point: &[Fp3]) -> Fp3 {
    point.iter().enumerate().fold(Fp3::ZERO, |acc, (i, r)| {
        acc.add(r.scale(Fp::new(1u64 << i)))
    })
}

fn bits(v: u64, count: usize) -> Vec<Fp3> {
    (0..count).map(|i| bit3(v, i)).collect()
}

pub(crate) fn weights_eval(n: usize, blocks: &[Block], sparse: &[(u64, Fp3)], r: &[Fp3]) -> Fp3 {
    let mut acc = Fp3::ZERO;
    for (fixed, free, c, z) in blocks {
        let rf: Vec<Fp3> = free.iter().map(|b| r[*b]).collect();
        let mut term = c.mul(eq(z, &rf));
        for (bit, val) in fixed {
            term = term.mul(if *val == 1 {
                r[*bit]
            } else {
                Fp3::ONE.sub(r[*bit])
            });
        }
        acc = acc.add(term);
    }
    for (idx, c) in sparse {
        acc = acc.add(c.mul(eq(&bits(*idx, n), r)));
    }
    acc
}

struct Terms {
    num_pub: Fp3,
    den_pub: Fp3,
    fb: Vec<Block>,
    sparse: Vec<(u64, Fp3)>,
    cb: Vec<Block>,
    coefs: Vec<Fp3>,
    zcoef: Fp3,
}

/// Batched linear claim (§5.5.5).
fn terms(
    lay: &AirLayout,
    point: &[Fp3],
    beta: Fp3,
    gamma: Fp3,
    mu: Fp3,
    libra: &[(usize, Vec<Fp3>)],
    zc_point: &[Fp3],
) -> Terms {
    let (g, s, k) = (lay.g(), lay.s(), lay.k());
    let rho0 = point[0];
    let rho = &point[1..];
    let low = rho[..g].to_vec();
    let top = &rho[g..g + 3];
    let ra = rho[..s - 1].to_vec();
    let eqb: Vec<Fp3> = (0..8u64).map(|b| eq(top, &bits(b, 3))).collect();
    let real = Fp3::ONE.sub(rho0);
    let num_pub = real.mul(eqb[0].add(eqb[1]).add(eqb[2]));
    let s05 = eqb[..6].iter().fold(Fp3::ZERO, |a, b| a.add(*b));
    let den_pub = real.mul(
        beta.mul(s05)
            .sub(gamma.mul(eqb[3]).mul(Fp3::ONE.add(int_ext(&low))))
            .sub(eqb[5].mul(int_ext(&low[..TABLE_BITS]))),
    );
    let blk = |off: u64, kk: usize, c: Fp3, z: &[Fp3]| -> Block {
        let (fixed, free) = lay.block(off, kk);
        (fixed, free, c, z.to_vec())
    };
    let mut fb = vec![
        blk(4 * k, g, real.mul(eqb[5]).neg(), &low),
        blk(5 * k, g, real.mul(eqb[6].sub(eqb[7])), &low),
        blk(
            8 * k,
            s - 1,
            rho0.mul(Fp3::ONE.sub(rho[s - 1].add(rho[s - 1]))),
            &ra,
        ),
    ];
    for c in 0..3u64 {
        fb.push(blk(c * k, g, mu.mul(real.mul(eqb[c as usize]).neg()), &low));
    }
    fb.push(blk(
        3 * k,
        g,
        mu.mul(real.mul(eqb[3].add(eqb[4])).neg()),
        &low,
    ));
    fb.push(blk(6 * k, g, mu.mul(real.mul(eqb[6].add(eqb[7]))), &low));
    fb.push(blk(12 * k, s - 1, mu.mul(rho0), &ra));
    let mut cb = vec![
        blk(8 * k, g, real.mul(eqb[3]).neg(), &low),
        blk(9 * k, g, real.mul(eqb[4]), &low),
    ];
    for c in 0..3u64 {
        cb.push(blk(
            (5 + c) * k,
            g,
            mu.mul(real.mul(gamma).mul(eqb[c as usize]).neg()),
            &low,
        ));
    }
    let mut sparse = Vec::new();
    let mut coefs = Vec::new();
    let mut coef = mu.mul(mu);
    for (kk, rs) in libra {
        for (i, r) in rs.iter().enumerate() {
            let mut pw = Fp3::ONE;
            for e in 0..4 {
                sparse.push((lay.libra_slot(*kk, i, e), coef.mul(pw)));
                pw = pw.mul(*r);
            }
        }
        coefs.push(coef);
        coef = coef.mul(mu);
    }
    for j in 0..N_FINALS {
        coefs.push(coef);
        if j < 3 {
            fb.push(blk(j as u64 * k, g, coef, zc_point));
        } else if j < 8 {
            cb.push(blk((j as u64 - 3) * k, g, coef, zc_point));
        } else {
            for (i, r) in zc_point.iter().enumerate() {
                let mut pw = Fp3::ONE;
                for e in 0..=ZC_DEGREE {
                    sparse.push((lay.zc_slot(i, e), coef.mul(pw)));
                    pw = pw.mul(*r);
                }
            }
        }
        coef = coef.mul(mu);
    }
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
    Terms {
        num_pub,
        den_pub,
        fb,
        sparse,
        cb,
        coefs,
        zcoef: coef,
    }
}

pub(crate) fn dec1(raw: &[u8]) -> Fp3 {
    // Only called on slices already validated by `decode`.
    Fp3::decode(raw).unwrap_or(Fp3::ZERO)
}

pub(crate) fn decv(raw: &[u8], count: usize) -> Vec<Fp3> {
    (0..count)
        .map(|j| dec1(&raw[24 * j..24 * j + 24]))
        .collect()
}

/// Verifies a relation-4 body (checks 12–19). `circuit_root` is the caller's pinned `root_circ`.
pub fn verify(
    header: &[u8],
    body: &[u8],
    lay: &AirLayout,
    circuit_root: Option<&[u8; 32]>,
) -> Result<(), RejectCode> {
    decode(body, lay)?; // 12
    let mut rd = Reader { b: body, off: 0 };
    let stmt = rd.take(96);
    let pin = circuit_root.ok_or(RejectCode::CircuitUnpinned)?; // 13
    if &stmt[32..64] != pin.as_slice() {
        return Err(RejectCode::CircuitPinned);
    }
    let mut tr = Fpn2Transcript::new(LABEL_V2);
    tr.absorb_bytes(header);
    tr.absorb_bytes(stmt);
    let beta = tr.squeeze_fp3();
    let gamma = tr.squeeze_fp3();
    let root_raw = rd.take(48);
    tr.absorb_bytes(root_raw);
    let root = decv(root_raw, 2);
    if root[0] != Fp3::ZERO || root[1] == Fp3::ZERO {
        return Err(RejectCode::GkrRoot); // 14
    }
    let mut point: Vec<Fp3> = Vec::new();
    let mut claim = (root[0], root[1]);
    let mut libra: Vec<(usize, Vec<Fp3>)> = Vec::new();
    let mut g_rs: Vec<Fp3> = Vec::new();
    for k in 0..=lay.s() {
        // 15
        let alpha = tr.squeeze_fp3();
        let mut target = claim.0.add(alpha.mul(claim.1));
        let mut lam = Fp3::ZERO;
        if k > 0 {
            let g_raw = rd.take(24);
            tr.absorb_bytes(g_raw);
            lam = tr.squeeze_fp3();
            target = target.add(lam.mul(dec1(g_raw)));
        }
        let mut rs = Vec::with_capacity(k);
        for _ in 0..k {
            let raw = rd.take(96);
            tr.absorb_bytes(raw);
            let gv = decv(raw, 4);
            if gv[0].add(gv[1]) != target {
                return Err(RejectCode::Gkr);
            }
            let r = tr.squeeze_fp3();
            target = lagrange_eval(&gv, r);
            rs.push(r);
        }
        let tail = rd.take(if k > 0 { 120 } else { 96 });
        tr.absorb_bytes(tail);
        let tv = decv(tail, if k > 0 { 5 } else { 4 });
        let (nl, dl, nr, dr) = (tv[0], tv[1], tv[2], tv[3]);
        let num = nl.mul(dr).add(nr.mul(dl));
        let den = dl.mul(dr);
        let mut expect = eq(&point, &rs).mul(num.add(alpha.mul(den)));
        if k > 0 {
            expect = expect.add(lam.mul(tv[4]));
            libra.push((k, rs.clone()));
            g_rs.push(tv[4]);
        }
        if target != expect {
            return Err(RejectCode::Gkr);
        }
        let branch = tr.squeeze_fp3();
        point = std::iter::once(branch).chain(rs).collect();
        claim = (mix(nl, nr, branch), mix(dl, dr, branch));
    }
    // 16: zerocheck
    let z: Vec<Fp3> = (0..lay.g()).map(|_| tr.squeeze_fp3()).collect();
    let zs_raw = rd.take(24);
    tr.absorb_bytes(zs_raw);
    let lam_z = tr.squeeze_fp3();
    let mut target = lam_z.mul(dec1(zs_raw));
    let mut zr = Vec::with_capacity(lay.g());
    for _ in 0..lay.g() {
        let raw = rd.take(24 * (ZC_DEGREE + 1));
        tr.absorb_bytes(raw);
        let sv = decv(raw, ZC_DEGREE + 1);
        if sv[0].add(sv[1]) != target {
            return Err(RejectCode::Zerocheck);
        }
        let r = tr.squeeze_fp3();
        target = lagrange_eval(&sv, r);
        zr.push(r);
    }
    let fin_raw = rd.take(24 * N_FINALS);
    tr.absorb_bytes(fin_raw);
    let f = decv(fin_raw, N_FINALS);
    let (ea, eb, ec) = (f[0], f[1], f[2]);
    let (qm, ql, qr, qo, qc) = (f[3], f[4], f[5], f[6], f[7]);
    let gate = qm
        .mul(ea.mul(eb))
        .add(ql.mul(ea))
        .add(qr.mul(eb).add(qo.mul(ec)))
        .add(qc);
    if target != eq(&z, &zr).mul(gate).add(lam_z.mul(f[8])) {
        return Err(RejectCode::Zerocheck);
    }
    let zb_raw = rd.take(24);
    tr.absorb_bytes(zb_raw);
    let mu = tr.squeeze_fp3();
    let tm = terms(lay, &point, beta, gamma, mu, &libra, &zr);
    let mut y = claim.0.sub(tm.num_pub).add(mu.mul(claim.1.sub(tm.den_pub)));
    let claimed: Vec<Fp3> = g_rs.iter().copied().chain(f.iter().copied()).collect();
    for (c, v) in tm.coefs.iter().zip(claimed.iter()) {
        y = y.add(c.mul(*v));
    }
    y = y.add(tm.zcoef.mul(dec1(zb_raw)));
    // 17: batched Basefold sumcheck
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
    let a_r = weights_eval(lay.n, &tm.fb, &tm.sparse, rest);
    let c_r = weights_eval(lay.n, &tm.cb, &[], rest);
    let w_r = Fp3::ONE.sub(r1).mul(
        Fp3::ONE
            .sub(theta)
            .mul(a_r)
            .add(theta.add(theta).sub(Fp3::ONE).mul(c_r)),
    );
    if y != final_.mul(w_r) {
        return Err(RejectCode::Final); // 18
    }
    // 19: queries
    let length: u64 = RATE << lay.n;
    let omega = Fp::new(7).pow((P - 1) / length);
    let iota_inv = Fp::new(7).pow((P - 1) / 4).inv().ok_or(RejectCode::Final)?;
    let root_circ = &stmt[32..64];
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
                if !merkle_ok(root_circ, craw, j, &csibs) {
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
    use crate::profile::{FOLDED_AIR_N12, FOLDED_AIR_N20_V1, FOLDED_AIR_V1};

    #[test]
    fn layout_of_registered_profiles() {
        let lay = AirLayout::of(&FOLDED_AIR_V1).unwrap();
        assert_eq!((lay.g(), lay.k(), lay.s(), lay.folds()), (9, 512, 12, 7));
        assert_eq!(lay.body_used(), 247_176);
        assert!(64 + lay.body_used() <= FOLDED_AIR_V1.c_p as usize);
        let lay = AirLayout::of(&FOLDED_AIR_N12).unwrap();
        assert_eq!(lay.body_used(), 14_504);
        assert!(64 + lay.body_used() <= FOLDED_AIR_N12.c_p as usize);
        let lay = AirLayout::of(&FOLDED_AIR_N20_V1).unwrap();
        assert_eq!((lay.g(), lay.k(), lay.s(), lay.folds()), (15, 32_768, 18, 10));
        assert_eq!(lay.body_used(), 424_104);
        assert!(64 + lay.body_used() <= FOLDED_AIR_N20_V1.c_p as usize);
    }
}
