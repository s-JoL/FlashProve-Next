// SPDX-License-Identifier: Apache-2.0
//! Relation 3 (`FOLDED_GKR_LOOKUP`, `specs/11-wire-format.md` §5.4): ZK-masked FracLogUp-GKR whose
//! leaf claims and Libra-mask evaluations are discharged by one fused two-round (coset-4) folded
//! ZK-Basefold opening with BLAKE2b coset Merkle trees. Every transcript absorption takes the raw
//! wire slice of the container.

use crate::blake2b::blake2b;
use crate::field::{lagrange_eval, Fp, Fp3, P};
use crate::profile::Profile;
use crate::transcript::{Fpn2Transcript, LABEL_V2};
use crate::verify::RejectCode;

pub const PERSONAL_MERKLE: &[u8; 16] = b"FPN2-MERKLE-v1\0\0";
pub const RATE: u64 = 8;
const DATA_FIXED: [(usize, u64); 2] = [(1, 1), (3, 1)];
const LIBRA_FIXED: [(usize, u64); 2] = [(1, 0), (3, 0)];
const MASK_POINTS: [u64; 2] = [2, 3];

/// Fixed-size relation-3 layout, a pure function of `(n_vars, n_queries)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub n: usize,
    pub queries: usize,
}

impl Layout {
    pub fn of(p: &Profile) -> Option<Layout> {
        let n = p.n_vars as usize;
        if p.sc_degree != 3
            || n < 8
            || !n.is_multiple_of(2)
            || n > 28
            || !(1..=4096).contains(&p.n_queries)
        {
            return None;
        }
        Some(Layout {
            n,
            queries: p.n_queries as usize,
        })
    }
    pub fn s(&self) -> usize {
        self.n - 3
    }
    pub fn quarter(&self) -> u64 {
        1 << (self.s() - 2)
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
        48 + 96 + (1..=self.s()).map(Layout::layer_len).sum::<usize>()
    }
    pub fn pcs_len(&self) -> usize {
        24 + 72 * self.n + 32 * (self.folds() - 1) + 24
    }
    pub fn query_len(&self) -> usize {
        (0..self.folds()).map(|t| 96 + 32 * self.depth(t)).sum()
    }
    pub fn body_used(&self) -> usize {
        64 + self.gkr_len() + self.pcs_len() + self.queries * self.query_len()
    }
    fn free_bits(&self) -> Vec<usize> {
        let mut v = vec![0, 2];
        v.extend(4..self.n);
        v
    }
    /// `(fixed bits, free bits)` of the data-coordinate block `[off, off + 2^k)`.
    fn data_block(&self, off: u64, k: usize) -> (Vec<(usize, u64)>, Vec<usize>) {
        let fb = self.free_bits();
        let mut fixed = DATA_FIXED.to_vec();
        for i in 0..fb.len() - k {
            fixed.push((fb[k + i], (off >> (k + i)) & 1));
        }
        (fixed, fb[..k].to_vec())
    }
    fn libra_slot(&self, k: usize, i: usize, e: usize) -> u64 {
        let v = (2 * k * (k - 1) + 4 * i + e) as u64;
        let mut idx = 0u64;
        for (bit, val) in LIBRA_FIXED {
            idx |= val << bit;
        }
        for (j, bit) in self.free_bits().iter().enumerate() {
            idx |= ((v >> j) & 1) << bit;
        }
        idx
    }
}

// ----------------------------------------------------------------------------- decoding (check 12)
/// One GKR layer `k`: `(g_sum, rounds, children, g_r)` (`g_sum`/`g_r` absent for `k = 0`).
pub type GkrLayerWire = (Option<Fp3>, Vec<[Fp3; 4]>, [Fp3; 4], Option<Fp3>);
/// One query: per folded layer, the coset values and Merkle siblings.
pub type QueryWire = Vec<([Fp3; 4], Vec<[u8; 32]>)>;

/// Decoded relation-3 body (values only; the verifier re-reads the raw wire slices).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedBody {
    pub stmt: [u8; 64],
    pub root: (Fp3, Fp3),
    pub layers: Vec<GkrLayerWire>,
    pub z_blind: Fp3,
    pub pcs_rounds: Vec<[Fp3; 3]>,
    pub pcs_roots: Vec<[u8; 32]>,
    pub final_: Fp3,
    pub queries: Vec<QueryWire>,
}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        s
    }
    fn fp3(&mut self) -> Result<Fp3, RejectCode> {
        Fp3::decode(self.take(24)).map_err(|_| RejectCode::NonCanonical)
    }
    fn fp3s<const N: usize>(&mut self) -> Result<[Fp3; N], RejectCode> {
        let mut out = [Fp3::ZERO; N];
        for v in out.iter_mut() {
            *v = self.fp3()?;
        }
        Ok(out)
    }
    fn digest(&mut self) -> [u8; 32] {
        let mut d = [0u8; 32];
        d.copy_from_slice(self.take(32));
        d
    }
}

fn dec(raw: &[u8], j: usize) -> Fp3 {
    // Only called on slices already validated by `decode`.
    Fp3::decode(&raw[24 * j..24 * j + 24]).unwrap_or(Fp3::ZERO)
}

/// Normative check 12 for relation 3: canonical decoding of every `Fp3` field in wire order.
pub fn decode(body: &[u8], lay: &Layout) -> Result<FoldedBody, RejectCode> {
    if body.len() != lay.body_used() {
        return Err(RejectCode::Length);
    }
    let mut rd = Reader { b: body, off: 0 };
    let mut stmt = [0u8; 64];
    stmt.copy_from_slice(rd.take(64));
    let root = (rd.fp3()?, rd.fp3()?);
    let mut layers = vec![(None, Vec::new(), rd.fp3s::<4>()?, None)];
    for k in 1..=lay.s() {
        let g_sum = rd.fp3()?;
        let mut rounds = Vec::with_capacity(k);
        for _ in 0..k {
            rounds.push(rd.fp3s::<4>()?);
        }
        let children = rd.fp3s::<4>()?;
        let g_r = rd.fp3()?;
        layers.push((Some(g_sum), rounds, children, Some(g_r)));
    }
    let z_blind = rd.fp3()?;
    let mut pcs_rounds = Vec::new();
    let mut pcs_roots = Vec::new();
    for t in 0..lay.folds() {
        pcs_rounds.push(rd.fp3s::<3>()?);
        pcs_rounds.push(rd.fp3s::<3>()?);
        if t + 1 < lay.folds() {
            pcs_roots.push(rd.digest());
        }
    }
    let final_ = rd.fp3()?;
    let mut queries = Vec::with_capacity(lay.queries);
    for _ in 0..lay.queries {
        let mut q = Vec::with_capacity(lay.folds());
        for t in 0..lay.folds() {
            let vals = rd.fp3s::<4>()?;
            q.push((vals, (0..lay.depth(t)).map(|_| rd.digest()).collect()));
        }
        queries.push(q);
    }
    Ok(FoldedBody {
        stmt,
        root,
        layers,
        z_blind,
        pcs_rounds,
        pcs_roots,
        final_,
        queries,
    })
}

/// Canonical serializer of a decoded body (inverse of [`decode`]).
pub fn encode(b: &FoldedBody) -> Vec<u8> {
    let mut out = b.stmt.to_vec();
    let put = |v: &Fp3, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    put(&b.root.0, &mut out);
    put(&b.root.1, &mut out);
    for (g_sum, rounds, children, g_r) in &b.layers {
        if let Some(g) = g_sum {
            put(g, &mut out);
        }
        for r in rounds {
            r.iter().for_each(|v| put(v, &mut out));
        }
        children.iter().for_each(|v| put(v, &mut out));
        if let Some(g) = g_r {
            put(g, &mut out);
        }
    }
    put(&b.z_blind, &mut out);
    for (i, h) in b.pcs_rounds.iter().enumerate() {
        h.iter().for_each(|v| put(v, &mut out));
        if i % 2 == 1 && i / 2 < b.pcs_roots.len() {
            out.extend_from_slice(&b.pcs_roots[i / 2]);
        }
    }
    put(&b.final_, &mut out);
    for q in &b.queries {
        for (vals, sibs) in q {
            vals.iter().for_each(|v| put(v, &mut out));
            sibs.iter().for_each(|d| out.extend_from_slice(d));
        }
    }
    out
}

// ----------------------------------------------------------------------------- helpers
pub(crate) fn eq1(a: Fp3, b: Fp3) -> Fp3 {
    a.mul(b).add(Fp3::ONE.sub(a).mul(Fp3::ONE.sub(b)))
}

pub(crate) fn eq(a: &[Fp3], b: &[Fp3]) -> Fp3 {
    a.iter()
        .zip(b.iter())
        .fold(Fp3::ONE, |acc, (x, y)| acc.mul(eq1(*x, *y)))
}

pub(crate) fn bit3(v: u64, i: usize) -> Fp3 {
    if (v >> i) & 1 == 1 {
        Fp3::ONE
    } else {
        Fp3::ZERO
    }
}

pub(crate) fn mix(a: Fp3, b: Fp3, r: Fp3) -> Fp3 {
    a.add(r.mul(b.sub(a)))
}

fn leaf_hash(raw: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(97);
    m.push(0u8);
    m.extend_from_slice(raw);
    blake2b(32, PERSONAL_MERKLE, &m)
}

fn node_hash(l: &[u8], r: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(65);
    m.push(1u8);
    m.extend_from_slice(l);
    m.extend_from_slice(r);
    blake2b(32, PERSONAL_MERKLE, &m)
}

pub(crate) fn merkle_ok(root: &[u8], raw_leaf: &[u8], mut idx: u64, sibs: &[&[u8]]) -> bool {
    let mut node = leaf_hash(raw_leaf);
    for s in sibs {
        node = if idx & 1 == 1 {
            node_hash(s, &node)
        } else {
            node_hash(&node, s)
        };
        idx >>= 1;
    }
    node == root
}

/// Fused coset-4 fold: `f_e = x^-e / 4 * sum_k iota^-ek v_k`, out `= f0 + ra f1 + rb f2 + ra rb f3`.
pub fn fold4(v: &[Fp3; 4], x_inv: Fp, iota_inv: Fp, ra: Fp3, rb: Fp3) -> Fp3 {
    let inv4 = Fp::new(4).inv().unwrap_or(Fp::ZERO);
    let a = v[0].sub(v[2]);
    let tb = v[1].sub(v[3]).scale(iota_inv);
    let s = [
        v[0].add(v[1]).add(v[2].add(v[3])),
        a.add(tb),
        v[0].add(v[2]).sub(v[1].add(v[3])),
        a.sub(tb),
    ];
    let mut f = [Fp3::ZERO; 4];
    let mut scale = inv4;
    for e in 0..4 {
        f[e] = s[e].scale(scale);
        scale = scale.mul(x_inv);
    }
    f[0].add(ra.mul(f[1]))
        .add(rb.mul(f[2]))
        .add(ra.mul(rb).mul(f[3]))
}

type Block = (Vec<(usize, u64)>, Vec<usize>, Fp3, Vec<Fp3>);

/// Blocks/sparse cells of the batched claim (§5.4.4); returns `(num_pub, den_pub, blocks, sparse, mask_coef)`.
fn terms(
    lay: &Layout,
    point: &[Fp3],
    beta: Fp3,
    libra: &[(usize, Vec<Fp3>, Fp3)],
    mu: Fp3,
) -> (Fp3, Fp3, Vec<Block>, Vec<(u64, Fp3)>, Fp3) {
    let s = lay.s();
    let rho0 = point[0];
    let rho = &point[1..];
    let low = rho[..s - 2].to_vec();
    let lowp = rho[..s - 3].to_vec();
    let ra = rho[..s - 1].to_vec();
    let pair = Fp3::ONE.sub(rho[s - 3].add(rho[s - 3]));
    let top = [rho[s - 2], rho[s - 1]];
    let eqq: Vec<Fp3> = (0..4u64)
        .map(|qd| eq(&top, &[bit3(qd, 0), bit3(qd, 1)]))
        .collect();
    let real = Fp3::ONE.sub(rho0);
    let mut t_ext = Fp3::ZERO;
    for (i, r) in low.iter().enumerate() {
        t_ext = t_ext.add(r.scale(Fp::new(1u64 << i)));
    }
    let num_pub = real.mul(eqq[0]);
    let den_pub = real.mul(eqq[0].mul(beta).add(eqq[2].mul(beta.sub(t_ext))));
    let q4 = lay.quarter();
    let (off_m, off_e1, off_e3) = (q4, 2 * q4, 3 * q4);
    let off_a = 1u64 << s;
    let off_d = off_a + (1u64 << (s - 1));
    let mut blocks: Vec<Block> = Vec::new();
    let mut push = |off: u64, k: usize, c: Fp3, z: &Vec<Fp3>| {
        let (fixed, free) = lay.data_block(off, k);
        blocks.push((fixed, free, c, z.clone()));
    };
    let h = q4 / 2;
    // y_num: multiplicities, paired dummy numerators (+N / -N across the top query bit), mask A.
    push(off_m, s - 2, real.mul(eqq[2]).neg(), &low);
    push(off_e1, s - 3, real.mul(eqq[1]).mul(pair), &lowp);
    push(off_e3, s - 3, real.mul(eqq[3]).mul(pair), &lowp);
    push(
        off_a,
        s - 1,
        rho0.mul(Fp3::ONE.sub(rho[s - 1].add(rho[s - 1]))),
        &ra,
    );
    // mu * y_den: queries, dummy denominators, mask D.
    push(0, s - 2, mu.mul(real.mul(eqq[0]).neg()), &low);
    push(off_e1 + h, s - 3, mu.mul(real.mul(eqq[1])), &lowp);
    push(off_e3 + h, s - 3, mu.mul(real.mul(eqq[3])), &lowp);
    push(off_d, s - 1, mu.mul(rho0), &ra);
    let mut sparse = Vec::new();
    let mut coef = mu.mul(mu);
    for (k, rs, _) in libra {
        for (i, r) in rs.iter().enumerate() {
            let mut pw = Fp3::ONE;
            for e in 0..4 {
                sparse.push((lay.libra_slot(*k, i, e), coef.mul(pw)));
                pw = pw.mul(*r);
            }
        }
        coef = coef.mul(mu);
    }
    let subcubes: [&[(usize, u64)]; 2] = [&[(1, 0)], &[(1, 1), (3, 0)]];
    for zv in MASK_POINTS {
        let z = Fp3::from_u64(zv);
        for fixed in subcubes {
            let mut c = coef;
            for (_, val) in fixed {
                c = c.mul(if *val == 1 { z } else { Fp3::ONE.sub(z) });
            }
            let free: Vec<usize> = (0..lay.n)
                .filter(|b| !fixed.iter().any(|(fb, _)| fb == b))
                .collect();
            let zs = vec![z; free.len()];
            blocks.push((fixed.to_vec(), free, c, zs));
        }
    }
    (num_pub, den_pub, blocks, sparse, coef)
}

fn weights_eval(lay: &Layout, blocks: &[Block], sparse: &[(u64, Fp3)], r: &[Fp3]) -> Fp3 {
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
        let bits: Vec<Fp3> = (0..lay.n).map(|i| bit3(*idx, i)).collect();
        acc = acc.add(c.mul(eq(&bits, r)));
    }
    acc
}

// ----------------------------------------------------------------------------- verifier (checks 13-17)
/// Verifies a relation-3 body (`body.len() == lay.body_used()`, already canonically decoded).
pub fn verify(header: &[u8], body: &[u8], lay: &Layout) -> Result<(), RejectCode> {
    decode(body, lay)?; // 12 (idempotent; also guards direct callers)
    let mut rd = Reader { b: body, off: 0 };
    let mut tr = Fpn2Transcript::new(LABEL_V2);
    tr.absorb_bytes(header);
    let stmt = rd.take(64);
    tr.absorb_bytes(stmt);
    let beta = tr.squeeze_fp3();
    let root_raw = rd.take(48);
    tr.absorb_bytes(root_raw);
    let root = (dec(root_raw, 0), dec(root_raw, 1));
    if root.0 != Fp3::ZERO || root.1 == Fp3::ZERO {
        return Err(RejectCode::GkrRoot); // 13
    }
    let mut point: Vec<Fp3> = Vec::new();
    let mut claim = root;
    let mut libra: Vec<(usize, Vec<Fp3>, Fp3)> = Vec::new();
    for k in 0..=lay.s() {
        // 14
        let alpha = tr.squeeze_fp3();
        let mut target = claim.0.add(alpha.mul(claim.1));
        let mut lam = Fp3::ZERO;
        if k > 0 {
            let g_raw = rd.take(24);
            tr.absorb_bytes(g_raw);
            lam = tr.squeeze_fp3();
            target = target.add(lam.mul(dec(g_raw, 0)));
        }
        let mut rs = Vec::with_capacity(k);
        for _ in 0..k {
            let raw = rd.take(96);
            tr.absorb_bytes(raw);
            let g: Vec<Fp3> = (0..4).map(|j| dec(raw, j)).collect();
            if g[0].add(g[1]) != target {
                return Err(RejectCode::Gkr);
            }
            let r = tr.squeeze_fp3();
            target = lagrange_eval(&g, r);
            rs.push(r);
        }
        let tail = rd.take(if k > 0 { 120 } else { 96 });
        tr.absorb_bytes(tail);
        let (nl, dl, nr, dr) = (dec(tail, 0), dec(tail, 1), dec(tail, 2), dec(tail, 3));
        let num = nl.mul(dr).add(nr.mul(dl));
        let den = dl.mul(dr);
        let mut expect = eq(&point, &rs).mul(num.add(alpha.mul(den)));
        if k > 0 {
            let g_r = dec(tail, 4);
            expect = expect.add(lam.mul(g_r));
            libra.push((k, rs.clone(), g_r));
        }
        if target != expect {
            return Err(RejectCode::Gkr);
        }
        let branch = tr.squeeze_fp3();
        point = std::iter::once(branch).chain(rs).collect();
        claim = (mix(nl, nr, branch), mix(dl, dr, branch));
    }
    let z_raw = rd.take(24);
    tr.absorb_bytes(z_raw);
    let mu = tr.squeeze_fp3();
    let (num_pub, den_pub, blocks, sparse, coef) = terms(lay, &point, beta, &libra, mu);
    let mut y = claim.0.sub(num_pub).add(mu.mul(claim.1.sub(den_pub)));
    let mut c = mu.mul(mu);
    for (_, _, g_r) in &libra {
        y = y.add(c.mul(*g_r));
        c = c.mul(mu);
    }
    y = y.add(coef.mul(dec(z_raw, 0)));
    let mut rs: Vec<Fp3> = Vec::with_capacity(lay.n);
    let mut roots: Vec<&[u8]> = vec![&stmt[..32]];
    for t in 0..lay.folds() {
        // 15
        for _ in 0..2 {
            let raw = rd.take(72);
            tr.absorb_bytes(raw);
            let h: Vec<Fp3> = (0..3).map(|j| dec(raw, j)).collect();
            if h[0].add(h[1]) != y {
                return Err(RejectCode::Sumcheck);
            }
            let r = tr.squeeze_fp3();
            y = lagrange_eval(&h, r);
            rs.push(r);
        }
        if t + 1 < lay.folds() {
            let root_t = rd.take(32);
            tr.absorb_bytes(root_t);
            roots.push(root_t);
        }
    }
    let final_raw = rd.take(24);
    tr.absorb_bytes(final_raw);
    let final_ = dec(final_raw, 0);
    if y != final_.mul(weights_eval(lay, &blocks, &sparse, &rs)) {
        return Err(RejectCode::Final); // 16
    }
    let length: u64 = RATE << lay.n;
    let omega = Fp::new(7).pow((P - 1) / length);
    let iota_inv = Fp::new(7).pow((P - 1) / 4).inv().ok_or(RejectCode::Final)?;
    for _ in 0..lay.queries {
        // 17
        let raw = tr.squeeze_bytes(8);
        let mut b8 = [0u8; 8];
        b8.copy_from_slice(&raw);
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
            let vals = [dec(vraw, 0), dec(vraw, 1), dec(vraw, 2), dec(vraw, 3)];
            if t > 0 && vals[slot] != prev {
                return Err(RejectCode::Fold);
            }
            let omega_t = omega.pow(1u64 << (2 * t));
            let x_inv = omega_t.pow((P - 1 - j) % (P - 1));
            prev = fold4(&vals, x_inv, iota_inv, rs[2 * t], rs[2 * t + 1]);
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
    use crate::profile::FOLDED_GKR_V1;

    #[test]
    fn layout_of_registered_profile() {
        let lay = Layout::of(&FOLDED_GKR_V1).unwrap();
        assert_eq!((lay.s(), lay.quarter(), lay.folds()), (9, 128, 6));
        assert_eq!(lay.body_used(), 158_960);
        assert!(64 + lay.body_used() <= FOLDED_GKR_V1.c_p as usize);
    }

    #[test]
    fn fold4_is_two_binary_folds() {
        // f(X) = sum c_i X^i, deg < 8; folding (ra, rb) must give f0 + ra f1 + rb f2 + ra rb f3 at x^4.
        let c: Vec<Fp3> = (1..=8u64).map(|i| Fp3::from_u64(i * 7919)).collect();
        let f = |x: Fp| {
            c.iter()
                .rev()
                .fold(Fp3::ZERO, |acc, ci| acc.scale(x).add(*ci))
        };
        let iota = Fp::new(7).pow((P - 1) / 4);
        let x = Fp::new(123_456_789);
        let vals = [
            f(x),
            f(x.mul(iota)),
            f(x.mul(iota.square())),
            f(x.mul(iota.square().mul(iota))),
        ];
        let ra = Fp3([Fp::new(3), Fp::new(5), Fp::new(7)]);
        let rb = Fp3([Fp::new(11), Fp::new(13), Fp::new(17)]);
        let got = fold4(&vals, x.inv().unwrap(), iota.inv().unwrap(), ra, rb);
        let y = x.square().square();
        let comp = |e: usize| c[e].add(c[e + 4].scale(y));
        let want = comp(0)
            .add(ra.mul(comp(1)))
            .add(rb.mul(comp(2)))
            .add(ra.mul(rb).mul(comp(3)));
        assert_eq!(got, want);
    }
}
