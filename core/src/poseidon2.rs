// SPDX-License-Identifier: Apache-2.0
//! Poseidon2 over Goldilocks, width 12, S-box `x^7`, `R_F = 8`, `R_P = 22` (Plonky3 instance),
//! with the 2-to-1 compression, both sponges and Merkle paths of `specs/11-wire-format.md` §3.

use crate::field::Fp;
pub use crate::poseidon2_constants::{DIAG, RC_FINAL, RC_INITIAL, RC_INTERNAL};

pub const WIDTH: usize = 12;
pub const RATE: usize = 8;
pub const TAG_NODE: u64 = 1;
pub const TAG_SPONGE: u64 = 2;
pub const DOM_LEAF: u64 = 64;

pub type Digest = [Fp; 4];
pub type State = [Fp; WIDTH];

const M4: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];

fn sbox(x: Fp) -> Fp {
    let x2 = x.square();
    let x3 = x2.mul(x);
    x3.square().mul(x)
}

/// External layer `M_E = circ(2 M4, M4, M4)`.
fn external(s: &State) -> State {
    let mut out = [Fp::ZERO; WIDTH];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc: u128 = 0;
        for j in 0..WIDTH {
            let c = M4[i % 4][j % 4] * if i / 4 == j / 4 { 2 } else { 1 };
            acc += c as u128 * s[j].value() as u128;
        }
        *o = Fp::from_u128(acc);
    }
    out
}

/// Internal layer `M_I = J + diag(DIAG)`.
fn internal(s: &State) -> State {
    let mut tot: u128 = 0;
    for x in s.iter() {
        tot += x.value() as u128;
    }
    let tot = Fp::from_u128(tot);
    let mut out = [Fp::ZERO; WIDTH];
    for i in 0..WIDTH {
        out[i] = tot.add(Fp::new(DIAG[i]).mul(s[i]));
    }
    out
}

/// The Poseidon2 permutation.
pub fn permute(state: &State) -> State {
    let mut s = external(state);
    let full = |s: &State, rc: &[u64; 12]| -> State {
        let mut t = [Fp::ZERO; WIDTH];
        for i in 0..WIDTH {
            t[i] = sbox(s[i].add(Fp::new(rc[i])));
        }
        external(&t)
    };
    for rc in RC_INITIAL.iter() {
        s = full(&s, rc);
    }
    for &c in RC_INTERNAL.iter() {
        s[0] = sbox(s[0].add(Fp::new(c)));
        s = internal(&s);
    }
    for rc in RC_FINAL.iter() {
        s = full(&s, rc);
    }
    s
}

/// 2-to-1 compression (lane 11 = `TAG_NODE`).
pub fn compress(l: &Digest, r: &Digest) -> Digest {
    let mut st = [Fp::ZERO; WIDTH];
    st[..4].copy_from_slice(l);
    st[4..8].copy_from_slice(r);
    st[11] = Fp::new(TAG_NODE);
    let o = permute(&st);
    [o[0], o[1], o[2], o[3]]
}

/// `poseidon2.py::sponge` over whole 8-element blocks (lane 10 = domain, lane 11 = `TAG_SPONGE`).
pub fn sponge_blocks(blocks: &[[Fp; RATE]], domain: u64) -> Digest {
    let mut st = [Fp::ZERO; WIDTH];
    st[10] = Fp::new(domain);
    st[11] = Fp::new(TAG_SPONGE);
    for b in blocks {
        for i in 0..RATE {
            st[i] = st[i].add(b[i]);
        }
        st = permute(&st);
    }
    [st[0], st[1], st[2], st[3]]
}

/// Length-in-capacity sponge (`state_zk.py::sponge_trace`): lane 8 = `len`, zero padding.
pub fn sponge_msg(msg: &[Fp], domain: u64) -> Digest {
    let nb = core::cmp::max(1, msg.len().div_ceil(RATE));
    let mut st = [Fp::ZERO; WIDTH];
    st[8] = Fp::new(msg.len() as u64);
    st[10] = Fp::new(domain);
    st[11] = Fp::new(TAG_SPONGE);
    for b in 0..nb {
        for i in 0..RATE {
            if let Some(x) = msg.get(RATE * b + i) {
                st[i] = st[i].add(*x);
            }
        }
        st = permute(&st);
    }
    [st[0], st[1], st[2], st[3]]
}

/// Leaf digest of the harness relation: `sponge_msg([v], DOM_LEAF)`.
pub fn leaf_digest(v: Fp) -> Digest {
    sponge_msg(&[v], DOM_LEAF)
}

/// Merkle tree over `2^n` leaves; returns all levels (level 0 = leaves, last = `[root]`).
pub fn merkle_levels(leaves: &[Digest]) -> Option<Vec<Vec<Digest>>> {
    if leaves.is_empty() || !leaves.len().is_power_of_two() {
        return None;
    }
    let mut levels = vec![leaves.to_vec()];
    while levels.last().map(|l| l.len()).unwrap_or(0) > 1 {
        let a = levels.last()?;
        let next: Vec<Digest> = a.chunks(2).map(|p| compress(&p[0], &p[1])).collect();
        levels.push(next);
    }
    Some(levels)
}

pub fn merkle_root(leaves: &[Digest]) -> Option<Digest> {
    merkle_levels(leaves).and_then(|l| l.last().and_then(|x| x.first().copied()))
}

/// Siblings from the leaf level upward.
pub fn merkle_path(levels: &[Vec<Digest>], mut idx: usize) -> Vec<Digest> {
    let mut out = Vec::new();
    for lvl in &levels[..levels.len().saturating_sub(1)] {
        out.push(lvl[idx ^ 1]);
        idx >>= 1;
    }
    out
}

/// Checks an authentication path; `idx` must be `< 2^path.len()`.
pub fn verify_path(root: &Digest, leaf: &Digest, mut idx: u64, path: &[Digest]) -> bool {
    let mut cur = *leaf;
    for sib in path {
        cur = if idx & 1 == 1 {
            compress(sib, &cur)
        } else {
            compress(&cur, sib)
        };
        idx >>= 1;
    }
    idx == 0 && cur == *root
}
