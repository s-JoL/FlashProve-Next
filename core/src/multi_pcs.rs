// SPDX-License-Identifier: Apache-2.0
//! Multi-group, multi-size batched rate-1/8 coset-4 folded ZK-Basefold verifier (`multi_pcs`),
//! bit-exact with `prover/fpn_prover/backend/multi_pcs.py`.

use crate::field::{lagrange_eval, Fp, Fp3, P};
use crate::folded_gkr::{eq1, fold4, merkle_ok, RATE};
use crate::transcript::Fpn2Transcript;
use crate::verify::RejectCode;

pub fn tail_len_for(queries: usize) -> usize {
    ((4 * queries + 16) + 15) & !15
}

pub fn libra_group_n(n_max: usize, queries: usize) -> usize {
    let need = 3 * n_max + tail_len_for(queries);
    let mut n = 8usize;
    while (1usize << n) < need {
        n += 2;
    }
    n
}

/// Public layout of a multi-group, multi-size batched rate-1/8 coset-4 Basefold opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPcsLayout {
    pub n: usize,
    pub queries: usize,
    pub group_ns: Vec<usize>,
}

impl MultiPcsLayout {
    pub fn new(n: usize, queries: usize, group_ns: Vec<usize>) -> Result<Self, RejectCode> {
        if !(8..=28).contains(&n) || !n.is_multiple_of(2) {
            return Err(RejectCode::Length);
        }
        if !(1..=4096).contains(&queries) || group_ns.is_empty() || group_ns.len() > 128 {
            return Err(RejectCode::Length);
        }
        let tl = tail_len_for(queries);
        for &ng in &group_ns {
            if !(8..=n).contains(&ng) || !ng.is_multiple_of(2) || tl >= (1usize << ng) {
                return Err(RejectCode::Length);
            }
        }
        Ok(Self {
            n,
            queries,
            group_ns,
        })
    }

    pub fn n_groups(&self) -> usize {
        self.group_ns.len()
    }

    pub fn code_len(&self) -> u64 {
        RATE << self.n
    }

    pub fn folds(&self) -> usize {
        self.n / 2
    }

    pub fn n_lib(&self) -> usize {
        libra_group_n(self.n, self.queries)
    }

    pub fn inject_layer(&self, g: usize) -> usize {
        (self.n - self.group_ns[g]) / 2
    }

    pub fn depth(&self, t: usize) -> usize {
        self.n + 1 - 2 * t
    }

    pub fn group_depth(&self, g: usize) -> usize {
        self.group_ns[g] + 1
    }

    pub fn step_a_bytes(&self) -> usize {
        24 + self.n * 72 + (self.n_groups() + 2) * 24
    }

    pub fn step_b_header_bytes(&self) -> usize {
        self.n * 72 + (self.folds() - 1) * 32 + 24
    }

    pub fn main_query_bytes(&self) -> usize {
        let b0 = 96 + self.depth(0) * 32;
        let gs: usize = (0..self.n_groups())
            .map(|g| 96 + self.group_depth(g) * 32)
            .sum();
        let rest: usize = (1..self.folds())
            .map(|t| 96 + self.depth(t) * 32)
            .sum();
        b0 + gs + rest
    }

    pub fn libra_open_bytes(&self) -> usize {
        let nl = self.n_lib();
        let fl = nl / 2;
        let hdr = nl * 72 + (fl - 1) * 32 + 24;
        let qb: usize = (0..fl).map(|t| 96 + (nl + 1 - 2 * t) * 32).sum();
        hdr + self.queries * qb
    }

    pub fn proof_bytes(&self) -> usize {
        64 + self.step_a_bytes()
            + self.libra_open_bytes()
            + self.step_b_header_bytes()
            + self.queries * self.main_query_bytes()
    }
}

fn root_of_unity(length: u64) -> Fp {
    Fp::new(7).pow((P - 1) / length)
}

fn read_fp3(buf: &[u8], off: &mut usize) -> Result<Fp3, RejectCode> {
    if *off + 24 > buf.len() {
        return Err(RejectCode::Length);
    }
    let v = Fp3::decode(&buf[*off..*off + 24]).map_err(|_| RejectCode::NonCanonical)?;
    *off += 24;
    Ok(v)
}

fn read_bytes<'a>(buf: &'a [u8], off: &mut usize, count: usize) -> Result<&'a [u8], RejectCode> {
    if *off + count > buf.len() {
        return Err(RejectCode::Length);
    }
    let s = &buf[*off..*off + count];
    *off += count;
    Ok(s)
}

type CosetOpening<'a> = ([Fp3; 4], &'a [u8], Vec<&'a [u8]>);

fn read_coset_and_path<'a>(
    buf: &'a [u8],
    off: &mut usize,
    depth: usize,
) -> Result<CosetOpening<'a>, RejectCode> {
    let raw_leaf = read_bytes(buf, off, 96)?;
    let mut vals = [Fp3::ZERO; 4];
    for k in 0..4 {
        vals[k] =
            Fp3::decode(&raw_leaf[24 * k..24 * k + 24]).map_err(|_| RejectCode::NonCanonical)?;
    }
    let mut sibs = Vec::with_capacity(depth);
    for _ in 0..depth {
        sibs.push(read_bytes(buf, off, 32)?);
    }
    Ok((vals, raw_leaf, sibs))
}

fn libra_weight_mle_at0(r_ip: &[Fp3], r_lib: &[Fp3]) -> Fp3 {
    let n_lib = r_lib.len();
    let mut acc = Fp3::ZERO;
    let mut k = 0usize;
    for &r in r_ip {
        let ws = [Fp3::ONE, r, r.mul(r)];
        for &w in &ws {
            let mut eq_val = Fp3::ONE;
            for b in 0..n_lib {
                let rb = r_lib[b];
                eq_val = eq_val.mul(if (k >> b) & 1 == 1 {
                    rb
                } else {
                    Fp3::ONE.sub(rb)
                });
            }
            acc = acc.add(w.mul(eq_val));
            k += 1;
        }
    }
    acc
}

struct SingleOpenSpec<'a> {
    n: usize,
    root: &'a [u8],
    claim: Fp3,
    queries: usize,
}

fn verify_single_open(
    spec: SingleOpenSpec<'_>,
    w_mle_fn: impl Fn(&[Fp3]) -> Fp3,
    buf: &[u8],
    off: &mut usize,
    tr: &mut Fpn2Transcript,
) -> Result<(), RejectCode> {
    let n = spec.n;
    let root = spec.root;
    let claim = spec.claim;
    let queries = spec.queries;
    let folds = n / 2;
    let code_len = RATE << n;
    let mut target = claim;
    let mut r_bf = Vec::with_capacity(n);
    let mut fold_roots: Vec<&[u8]> = Vec::with_capacity(folds.saturating_sub(1));

    for i in 0..n {
        let rs = *off;
        let p0 = read_fp3(buf, off)?;
        let p1 = read_fp3(buf, off)?;
        let p2 = read_fp3(buf, off)?;
        if p0.add(p1) != target {
            return Err(RejectCode::Sumcheck);
        }
        tr.absorb_bytes(&buf[rs..*off]);
        let ri = tr.squeeze_fp3();
        r_bf.push(ri);
        target = lagrange_eval(&[p0, p1, p2], ri);
        if i % 2 == 1 && (i / 2) + 1 < folds {
            let rt = read_bytes(buf, off, 32)?;
            tr.absorb_bytes(rt);
            fold_roots.push(rt);
        }
    }

    let fs = *off;
    let final_val = read_fp3(buf, off)?;
    tr.absorb_bytes(&buf[fs..*off]);
    if final_val.mul(w_mle_fn(&r_bf)) != target {
        return Err(RejectCode::Final);
    }

    let q_mask = (code_len / 4) - 1;
    let mut positions = Vec::with_capacity(queries);
    for _ in 0..queries {
        let raw = tr.squeeze_bytes(8);
        let p = u64::from_le_bytes(raw[..8].try_into().unwrap()) & q_mask;
        positions.push(p);
    }

    let omega = root_of_unity(code_len);
    let inv_iota = root_of_unity(4).inv().unwrap_or(Fp::ZERO);
    let mut omega_t = omega;
    let mut omegas = Vec::with_capacity(folds);
    for _ in 0..folds {
        omegas.push(omega_t);
        omega_t = omega_t.square().square();
    }

    for p0 in positions {
        let mut cur_p = p0;
        let mut prev = Fp3::ZERO;
        for t in 0..folds {
            let dt = n + 1 - 2 * t;
            let (coset, raw_leaf, sibs) = read_coset_and_path(buf, off, dt)?;
            if t == 0 {
                if !merkle_ok(root, raw_leaf, cur_p, &sibs) {
                    return Err(RejectCode::Merkle);
                }
            } else {
                let q_t = (code_len >> (2 * t)) / 4;
                let slot = (cur_p / q_t) as usize;
                cur_p %= q_t;
                if !merkle_ok(fold_roots[t - 1], raw_leaf, cur_p, &sibs) {
                    return Err(RejectCode::Merkle);
                }
                if coset[slot] != prev {
                    return Err(RejectCode::Fold);
                }
            }
            let x_inv = omegas[t].pow(cur_p).inv().unwrap_or(Fp::ZERO);
            prev = fold4(&coset, x_inv, inv_iota, r_bf[2 * t], r_bf[2 * t + 1]);
        }
        if prev != final_val {
            return Err(RejectCode::Fold);
        }
    }
    Ok(())
}

/// Verify a `multi_pcs` opening proof slice, advancing `tr` in normative raw-byte order.
pub fn verify_multi_open<F>(
    layout: &MultiPcsLayout,
    group_roots: &[[u8; 32]],
    claims: &[Fp3],
    w_eval_fn: F,
    proof: &[u8],
    tr: &mut Fpn2Transcript,
) -> Result<(Vec<Fp3>, Vec<Fp3>, Fp3), RejectCode>
where
    F: Fn(usize, &[Fp3]) -> Fp3,
{
    let g_cnt = layout.n_groups();
    let n = layout.n;
    if group_roots.len() != g_cnt || claims.len() != g_cnt {
        return Err(RejectCode::Length);
    }
    if proof.len() != layout.proof_bytes() {
        return Err(RejectCode::Length);
    }

    let mut off = 0usize;
    let root_b = read_bytes(proof, &mut off, 32)?;
    let root_l = read_bytes(proof, &mut off, 32)?;
    let g_sum = read_fp3(proof, &mut off)?;

    for r in group_roots {
        tr.absorb_bytes(r);
    }
    tr.absorb_bytes(root_b);
    tr.absorb_bytes(root_l);
    let mut claims_raw = Vec::with_capacity(g_cnt * 24);
    for c in claims {
        claims_raw.extend_from_slice(&c.to_le_bytes());
    }
    tr.absorb_bytes(&claims_raw);
    tr.absorb_bytes(&g_sum.to_le_bytes());

    let mu = tr.squeeze_fp3();
    let lam = tr.squeeze_fp3();

    let mut mu_pows = Vec::with_capacity(g_cnt);
    let mut mp = Fp3::ONE;
    for _ in 0..g_cnt {
        mu_pows.push(mp);
        mp = mp.mul(mu);
    }

    // ---- Step A verification
    let mut target = lam.mul(g_sum);
    for g in 0..g_cnt {
        target = target.add(mu_pows[g].mul(claims[g]));
    }

    let mut r_ip = Vec::with_capacity(n);
    for _ in 0..n {
        let rs = off;
        let p0 = read_fp3(proof, &mut off)?;
        let p1 = read_fp3(proof, &mut off)?;
        let p2 = read_fp3(proof, &mut off)?;
        if p0.add(p1) != target {
            return Err(RejectCode::Sumcheck);
        }
        tr.absorb_bytes(&proof[rs..off]);
        let ri = tr.squeeze_fp3();
        r_ip.push(ri);
        target = lagrange_eval(&[p0, p1, p2], ri);
    }

    let finals_start = off;
    let mut v_groups = Vec::with_capacity(g_cnt);
    for _ in 0..g_cnt {
        v_groups.push(read_fp3(proof, &mut off)?);
    }
    let v_b = read_fp3(proof, &mut off)?;
    let e_lib = read_fp3(proof, &mut off)?;
    tr.absorb_bytes(&proof[finals_start..off]);

    let mut expect_a = lam.mul(e_lib);
    for g in 0..g_cnt {
        let s_g = n - layout.group_ns[g];
        let mut z_lo_g = Fp3::ONE;
        for &rj in &r_ip[..s_g] {
            z_lo_g = z_lo_g.mul(Fp3::ONE.sub(rj));
        }
        let wg_eval = w_eval_fn(g, &r_ip[s_g..]);
        expect_a = expect_a.add(mu_pows[g].mul(z_lo_g).mul(v_groups[g].mul(wg_eval)));
    }
    if expect_a != target {
        return Err(RejectCode::Sumcheck);
    }

    // ---- Tiny Libra opening for L
    verify_single_open(
        SingleOpenSpec {
            n: layout.n_lib(),
            root: root_l,
            claim: e_lib,
            queries: layout.queries,
        },
        |r_lib| libra_weight_mle_at0(&r_ip, r_lib),
        proof,
        &mut off,
        tr,
    )?;

    // ---- Step B verification (single chain with multi-size layer injection)
    let theta = tr.squeeze_fp3();
    let mut theta_pows = Vec::with_capacity(g_cnt);
    let mut tp = theta;
    for _ in 0..g_cnt {
        theta_pows.push(tp);
        tp = tp.mul(theta);
    }

    let folds = layout.folds();
    let mut by_layer: Vec<Vec<usize>> = vec![Vec::new(); folds];
    for g in 0..g_cnt {
        by_layer[layout.inject_layer(g)].push(g);
    }

    let mut target_b = v_b;
    let mut r_bf = Vec::with_capacity(n);
    let mut fold_roots: Vec<&[u8]> = Vec::with_capacity(folds.saturating_sub(1));
    let mut c_lo = Fp3::ONE;

    for i in 0..n {
        if i % 2 == 0 {
            let t = i / 2;
            for &g in &by_layer[t] {
                target_b = target_b.add(theta_pows[g].mul(c_lo.mul(v_groups[g])));
            }
        }
        let rs = off;
        let p0 = read_fp3(proof, &mut off)?;
        let p1 = read_fp3(proof, &mut off)?;
        let p2 = read_fp3(proof, &mut off)?;
        if p0.add(p1) != target_b {
            return Err(RejectCode::Sumcheck);
        }
        tr.absorb_bytes(&proof[rs..off]);
        let ri = tr.squeeze_fp3();
        r_bf.push(ri);
        c_lo = c_lo.mul(eq1(r_ip[i], ri));
        target_b = lagrange_eval(&[p0, p1, p2], ri);
        if i % 2 == 1 && (i / 2) + 1 < folds {
            let rt = read_bytes(proof, &mut off, 32)?;
            tr.absorb_bytes(rt);
            fold_roots.push(rt);
        }
    }

    let fs = off;
    let final_val = read_fp3(proof, &mut off)?;
    tr.absorb_bytes(&proof[fs..off]);
    if final_val.mul(c_lo) != target_b {
        return Err(RejectCode::Final);
    }

    // ---- Query verification
    let code_len = layout.code_len();
    let q_mask = (code_len / 4) - 1;
    let mut positions = Vec::with_capacity(layout.queries);
    for _ in 0..layout.queries {
        let raw = tr.squeeze_bytes(8);
        let p = u64::from_le_bytes(raw[..8].try_into().unwrap()) & q_mask;
        positions.push(p);
    }

    let omega = root_of_unity(code_len);
    let inv_iota = root_of_unity(4).inv().unwrap_or(Fp::ZERO);
    let mut omega_t = omega;
    let mut omegas = Vec::with_capacity(folds);
    for _ in 0..folds {
        omegas.push(omega_t);
        omega_t = omega_t.square().square();
    }

    for p0 in positions {
        let mut cur_p = p0;
        let mut prev = Fp3::ZERO;
        for t in 0..folds {
            let dt = layout.depth(t);
            let (base_coset, base_raw, base_sibs) = read_coset_and_path(proof, &mut off, dt)?;
            if t == 0 {
                if !merkle_ok(root_b, base_raw, cur_p, &base_sibs) {
                    return Err(RejectCode::Merkle);
                }
            } else {
                let q_t = (code_len >> (2 * t)) / 4;
                let slot = (cur_p / q_t) as usize;
                cur_p %= q_t;
                if !merkle_ok(fold_roots[t - 1], base_raw, cur_p, &base_sibs) {
                    return Err(RejectCode::Merkle);
                }
                if base_coset[slot] != prev {
                    return Err(RejectCode::Fold);
                }
            }

            let mut comb = base_coset;
            for &g in &by_layer[t] {
                let (g_coset, g_raw, g_sibs) = read_coset_and_path(proof, &mut off, dt)?;
                if !merkle_ok(&group_roots[g], g_raw, cur_p, &g_sibs) {
                    return Err(RejectCode::Merkle);
                }
                let tg = theta_pows[g];
                for k in 0..4 {
                    comb[k] = comb[k].add(tg.mul(g_coset[k]));
                }
            }

            let x_inv = omegas[t].pow(cur_p).inv().unwrap_or(Fp::ZERO);
            prev = fold4(&comb, x_inv, inv_iota, r_bf[2 * t], r_bf[2 * t + 1]);
        }
        if prev != final_val {
            return Err(RejectCode::Fold);
        }
    }

    debug_assert_eq!(off, proof.len());
    Ok((r_ip, r_bf, final_val))
}
