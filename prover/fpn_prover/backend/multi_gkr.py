# SPDX-License-Identifier: Apache-2.0
"""Generalized multi-table / weighted ZK-masked FracLogUp-GKR (`multi_gkr`).

Eliminates ALL Phase-2 LogUp inverse commitments: arbitrarily many lookup tables (single-column,
multi-column fingerprinted, boolean-selected, or Fp3-weighted such as HCO) are packed into ONE binary
fraction tree of height `s` (2^s leaves). Each leaf holds `(num, den) in E x E`:

    query leaf  : (weight_k,  beta - (table_id_k + sum_j gamma^{j+1} col_{k,j}))
    table leaf  : (-mult_k,   beta - (table_id_k + sum_j gamma^{j+1} tab_{k,j}))
    pad leaf    : (0, 1)
    mask pairs  : M_mask cancelling pairs placed symmetrically in the two halves:
                    left  slot  (2^{s-1} - M_mask + m) : (+a_m, d_m)
                    right slot  (2^s     - M_mask + m) : (-a_m, d_m)
                  for 0 <= m < M_mask (a_m, d_m in E, d_m != 0, committed BEFORE GKR starts).

Because every mask pair sums to `a_m/d_m + (-a_m)/d_m = 0` with nonzero denominator `d_m^2`, the root
satisfies `root_num == 0` and `root_den != 0` iff the sum of all real lookup fractions vanishes identically.

Indexing convention (LSB-first, matching `cuda_engine.gkr_round` and `multi_pcs`):
  * `tree[0]` holds the 2^s leaves; `tree[s]` holds the 1 root.
  * Parent `x` in `tree[d]` combines children `2x` (left, bit 0 = 0) and `2x + 1` (right, bit 0 = 1)
    in `tree[d - 1]`.
  * Layer 0 reduces root `tree[s]` -> `tree[s - 1]` (2 nodes), producing 1-bit `pt = [br_0]`.
  * Masked layers `k = 1 .. s - 1` reduce `tree[s - k]` (2^k nodes) -> `tree[s - k - 1]` (2^{k+1} nodes)
    via a k-round degree-3 Libra-masked sumcheck; the next point is `pt_next = [br_k] + r_k` (bit 0 = br_k).
  * After layer `s - 1`, `r_gkr = pt` has length `s` and `(num_eval, den_eval)` are the MLE evaluations of
    the leaf numerator and denominator vectors at `r_gkr`, discharged together with the Libra mask
    evaluations `g_k(r_k)` by `multi_pcs`.

Supports both pure-NumPy CPU and GPU (`CudaEngine`) with bit-identical transcripts and proofs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flashprove_next.field import ONE, P, ZERO, eq1, hadd, hf, hmul, hsub, lagrange_eval

from . import cuda_engine as CE
from .fpn2 import Fpn2Transcript, dec3, enc3, enc3s
from .multi_pcs import _os_rand_fp3

GKR_ROUND_DEG = 3


class MultiGkrError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# ----------------------------------------------------------------------------- layout & helpers
@dataclass(frozen=True)
class MultiGkrLayout:
    s: int              # log2(total leaves), >= 3
    m_mask: int = 8     # mask slots per half (>= 2: m_mask-1 cancelling pairs + 1 independent (0, d_L)/(0, d_R) pair)

    def __post_init__(self):
        if type(self.s) is not int or not 3 <= self.s <= 28:
            raise ValueError("MultiGkrLayout requires 3 <= s <= 28")
        if type(self.m_mask) is not int or not 2 <= self.m_mask <= (1 << (self.s - 2)):
            raise ValueError("m_mask must be in [2, 2^(s-2)] for tree height s")

    @property
    def n_leaves(self):
        return 1 << self.s

    @property
    def half(self):
        return 1 << (self.s - 1)

    @property
    def real_capacity_left(self):
        return self.half - self.m_mask

    @property
    def real_capacity_right(self):
        return self.half - self.m_mask

    @property
    def total_real_capacity(self):
        return self.n_leaves - 2 * self.m_mask

    @property
    def libra_coeff_count(self):
        # Masked layers k = 1 .. s - 1: each has k variables * 4 coeffs (degree 3)
        return 2 * (self.s - 1) * self.s

    def layer_libra_offset(self, k):
        # 1 <= k <= s - 1
        return 2 * (k - 1) * k

    @property
    def mask_leaf_indices(self):
        h = self.half
        return (
            list(range(h - self.m_mask, h)),
            list(range(self.n_leaves - self.m_mask, self.n_leaves)),
        )

    @property
    def proof_bytes(self):
        # root (num, den): 48 B; layer 0 children (nl, dl, nr, dr): 96 B
        # masked layers k = 1 .. s - 1: G_sum (24) + k * 4 Fp3 (96*k) + children (96) + g_r (24)
        layers_b = sum(24 + k * 96 + 96 + 24 for k in range(1, self.s))
        return 48 + 96 + layers_b


def combine_frac(left, right):
    n, d = left
    m, e = right
    return hadd(hmul(n, e), hmul(m, d)), hmul(d, e)


def _libra3_sum(coeffs, k):
    scale = pow(2, k - 1, P)
    acc = ZERO
    for i in range(k):
        c0, c1, c2, c3 = (tuple(int(v) for v in coeffs[4 * i + j]) for j in range(4))
        s = hadd(hadd(c0, c0), hadd(hadd(c1, c2), c3))
        acc = hadd(acc, s)
    return hmul(hf(scale), acc)


def _libra3_round_vals(coeffs, k, i, prefix_sum):
    rem = k - 1 - i
    cnt = pow(2, rem, P)
    half_cnt = pow(2, rem - 1, P) if rem > 0 else 0
    tail_sum = ZERO
    if rem > 0:
        acc = ZERO
        for j in range(i + 1, k):
            c0, c1, c2, c3 = (tuple(int(v) for v in coeffs[4 * j + t]) for t in range(4))
            acc = hadd(acc, hadd(hadd(c0, c0), hadd(hadd(c1, c2), c3)))
        tail_sum = hmul(hf(half_cnt), acc)
    c0, c1, c2, c3 = (tuple(int(v) for v in coeffs[4 * i + t]) for t in range(4))
    out = []
    for t in (0, 1, 2, 3):
        tf = hf(t)
        gi_t = hadd(c0, hmul(tf, hadd(c1, hmul(tf, hadd(c2, hmul(tf, c3))))))
        out.append(hadd(hmul(hf(cnt), hadd(prefix_sum, gi_t)), tail_sum))
    return out


def _libra3_eval_var(coeffs, i, r_i):
    c0, c1, c2, c3 = (tuple(int(v) for v in coeffs[4 * i + t]) for t in range(4))
    return hadd(c0, hmul(r_i, hadd(c1, hmul(r_i, hadd(c2, hmul(r_i, c3))))))


def eq_eval_pt(a, b):
    acc = ONE
    for x, y in zip(a, b):
        acc = hmul(acc, eq1(x, y))
    return acc


def mask_leaf_contributions(layout: MultiGkrLayout, r_gkr, a_vals, d_vals):
    """Contribution of the M_mask mask slots to (N_leaf~(r_gkr), D_leaf~(r_gkr)).

    Slots 0..m_mask-2 are cancelling pairs (+a_m, d_m) / (-a_m, d_m).
    Slot m_mask-1 is the independent zero-numerator denominator blinder pair:
      left:  (0, d_L = d_vals[m_mask-1])
      right: (0, d_R = a_vals[m_mask-1])
    so dl0 / dr0 = (D_L / D_R) * (d_L / d_R) is uniformly random in E* (zero quotient leakage).
    """
    left_idx, right_idx = layout.mask_leaf_indices
    s = layout.s
    num_c, den_c = ZERO, ZERO
    last = layout.m_mask - 1
    for m in range(layout.m_mask):
        am = a_vals[m]
        dm = d_vals[m]
        el, er = ONE, ONE
        il, ir = left_idx[m], right_idx[m]
        for b in range(s):
            rb = r_gkr[b]
            el = hmul(el, rb if (il >> b) & 1 else hsub(ONE, rb))
            er = hmul(er, rb if (ir >> b) & 1 else hsub(ONE, rb))
        if m < last:
            num_c = hadd(num_c, hmul(am, hsub(el, er)))
            den_c = hadd(den_c, hmul(dm, hadd(el, er)))
        else:
            # Independent zero-numerator denominator blinders: left (0, dm), right (0, am)
            den_c = hadd(den_c, hadd(hmul(dm, el), hmul(am, er)))
    return num_c, den_c


def pack_real_leaves(layout: MultiGkrLayout, real_num, real_den, mask_a, mask_d):
    """Place real leaves (<= total_real_capacity) + mask pairs into (2^s, 6) uint64."""
    real_num = np.asarray(real_num, dtype=np.uint64).reshape(-1, 3)
    real_den = np.asarray(real_den, dtype=np.uint64).reshape(-1, 3)
    count = len(real_num)
    if count != len(real_den) or count > layout.total_real_capacity:
        raise ValueError(f"real leaves count {count} exceeds capacity {layout.total_real_capacity}")

    out = np.zeros((layout.n_leaves, 6), dtype=np.uint64)
    out[:, 3] = 1  # (0, 1) identity fraction

    n_left = min(count, layout.real_capacity_left)
    if n_left > 0:
        out[:n_left, :3] = real_num[:n_left]
        out[:n_left, 3:] = real_den[:n_left]
    n_right = count - n_left
    if n_right > 0:
        h = layout.half
        out[h:h + n_right, :3] = real_num[n_left:count]
        out[h:h + n_right, 3:] = real_den[n_left:count]

    left_idx, right_idx = layout.mask_leaf_indices
    last = layout.m_mask - 1
    for m in range(layout.m_mask):
        am = tuple(int(v) for v in mask_a[m])
        dm = tuple(int(v) for v in mask_d[m])
        if dm == ZERO:
            raise ValueError("mask leaf denominator must be nonzero")
        if m < last:
            neg_am = hsub(ZERO, am)
            out[left_idx[m], :3] = am
            out[left_idx[m], 3:] = dm
            out[right_idx[m], :3] = neg_am
            out[right_idx[m], 3:] = dm
        else:
            if am == ZERO:
                raise ValueError("right independent denominator blinder must be nonzero")
            out[left_idx[m], :3] = ZERO
            out[left_idx[m], 3:] = dm
            out[right_idx[m], :3] = ZERO
            out[right_idx[m], 3:] = am
    return out


# ----------------------------------------------------------------------------- CPU / GPU GKR round ops
def _eq_table_host(pt):
    from flashprove_next.field import CPU
    if not pt:
        return np.array([[1, 0, 0]], dtype=np.uint64)
    return np.ascontiguousarray(CPU.eq_table(list(pt)).T, dtype=np.uint64)


def _gkr_round_host(cols, alpha):
    from flashprove_next.field import CPU

    def emul(a, b):
        a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
        b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
        t0 = CPU.add(CPU.mul(a1, b2), CPU.mul(a2, b1))
        c0 = CPU.add(CPU.mul(a0, b0), CPU.add(t0, t0))
        t1 = CPU.mul(a2, b2)
        c1 = CPU.add(CPU.add(CPU.mul(a0, b1), CPU.mul(a1, b0)), CPU.add(t1, t1))
        c2 = CPU.add(CPU.add(CPU.mul(a0, b2), CPU.mul(a1, b1)), CPU.mul(a2, b0))
        out = np.empty_like(a)
        out[:, 0], out[:, 1], out[:, 2] = c0, c1, c2
        return out

    def eadd(a, b):
        out = np.empty_like(a)
        for j in range(3):
            out[:, j] = CPU.add(a[:, j], b[:, j])
        return out

    def esub(a, b):
        out = np.empty_like(a)
        for j in range(3):
            out[:, j] = CPU.sub(a[:, j], b[:, j])
        return out

    def esum(a):
        return (int(CPU.bsum(a[:, 0])), int(CPU.bsum(a[:, 1])), int(CPU.bsum(a[:, 2])))

    alpha_arr = np.broadcast_to(np.array(alpha, dtype=np.uint64), (len(cols[0]) // 2, 3))
    c0 = [c[0::2] for c in cols]
    c1 = [c[1::2] for c in cols]
    diff = [esub(b, a) for a, b in zip(c0, c1)]

    res = []
    cur = list(c0)
    for t in range(4):
        if t == 1:
            cur = list(c1)
        elif t >= 2:
            cur = [eadd(x, d) for x, d in zip(cur, diff)]
        w, nl, dl, nr, dr = cur
        num = eadd(emul(nl, dr), emul(nr, dl))
        den = emul(dl, dr)
        comb = eadd(num, emul(alpha_arr, den))
        res.append(esum(emul(w, comb)))
    return res


def _halve_col(engine, col, r):
    if engine is not None and hasattr(engine, "cp"):
        return engine.halve(col, r)
    from .multi_pcs import _halve_vec
    return _halve_vec(None, col, r)


# ----------------------------------------------------------------------------- Prover
@dataclass
class MultiGkrProofResult:
    proof: bytes
    r_gkr: list[tuple[int, int, int]]                # length s
    num_eval: tuple[int, int, int]
    den_eval: tuple[int, int, int]
    layer_points: list[list[tuple[int, int, int]]]   # layer k = 1..s-1 challenges r_k
    layer_libra_evals: list[tuple[int, int, int]]    # g_k(r_k) for k = 1..s-1
    libra_coeffs: np.ndarray                         # (libra_coeff_count, 3) uint64
    mask_a: np.ndarray                               # (m_mask, 3) uint64
    mask_d: np.ndarray                               # (m_mask, 3) uint64


def sample_gkr_masks(layout: MultiGkrLayout, *, rng=None):
    """Sample fresh OS-CSPRNG (libra_coeffs, mask_a, mask_d) with every d_m != 0 and right blinder mask_a[-1] != 0."""
    libra_coeffs = _os_rand_fp3(layout.libra_coeff_count, rng=rng)
    mask_a = _os_rand_fp3(layout.m_mask, rng=rng)
    mask_d = _os_rand_fp3(layout.m_mask, rng=rng)
    for m in range(layout.m_mask):
        while int(mask_d[m, 0]) == 0 and int(mask_d[m, 1]) == 0 and int(mask_d[m, 2]) == 0:
            mask_d[m] = _os_rand_fp3(1, rng=rng)[0]
    last = layout.m_mask - 1
    while int(mask_a[last, 0]) == 0 and int(mask_a[last, 1]) == 0 and int(mask_a[last, 2]) == 0:
        mask_a[last] = _os_rand_fp3(1, rng=rng)[0]
    return libra_coeffs, mask_a, mask_d


def prove_multi_gkr(
    layout: MultiGkrLayout,
    real_num: np.ndarray,
    real_den: np.ndarray,
    transcript: Fpn2Transcript,
    *,
    libra_coeffs: np.ndarray,
    mask_a: np.ndarray,
    mask_d: np.ndarray,
    engine=None,
) -> MultiGkrProofResult:
    """Prove that the binary fraction tree over `(real_num, real_den)` + cancelling `(mask_a, mask_d)`
    has `root_num == 0` and `root_den != 0`.

    Caller MUST commit `(libra_coeffs, mask_a, mask_d)` in the PCS BEFORE calling this function!
    """
    s = layout.s
    leaves = pack_real_leaves(layout, real_num, real_den, mask_a, mask_d)
    use_dev = engine is not None and hasattr(engine, "cp")

    if use_dev:
        tree = CE.gkr_tree_dev(engine, leaves)
        root_row = CE._host(engine, tree[-1])[0]
    elif engine is not None:
        tree = CE.gkr_tree(engine, leaves)
        root_row = tree[-1][0]
    else:
        cur = leaves
        tree = [cur]
        while len(cur) > 1:
            pairs = len(cur) // 2
            nxt = np.empty((pairs, 6), dtype=np.uint64)
            for j in range(pairs):
                nl = (int(cur[2 * j, 0]), int(cur[2 * j, 1]), int(cur[2 * j, 2]))
                dl = (int(cur[2 * j, 3]), int(cur[2 * j, 4]), int(cur[2 * j, 5]))
                nr = (int(cur[2 * j + 1, 0]), int(cur[2 * j + 1, 1]), int(cur[2 * j + 1, 2]))
                dr = (int(cur[2 * j + 1, 3]), int(cur[2 * j + 1, 4]), int(cur[2 * j + 1, 5]))
                n_, d_ = combine_frac((nl, dl), (nr, dr))
                nxt[j, :3], nxt[j, 3:] = n_, d_
            cur = nxt
            tree.append(cur)
        root_row = tree[-1][0]

    root_num = (int(root_row[0]), int(root_row[1]), int(root_row[2]))
    root_den = (int(root_row[3]), int(root_row[4]), int(root_row[5]))

    out = bytearray()
    root_bytes = enc3(root_num) + enc3(root_den)
    out += root_bytes
    transcript.absorb_bytes(root_bytes)

    # Layer 0: root combines tree[s - 1][0] and tree[s - 1][1]
    l1 = CE._host(engine, tree[s - 1]) if use_dev else tree[s - 1]
    nl0 = (int(l1[0, 0]), int(l1[0, 1]), int(l1[0, 2]))
    dl0 = (int(l1[0, 3]), int(l1[0, 4]), int(l1[0, 5]))
    nr0 = (int(l1[1, 0]), int(l1[1, 1]), int(l1[1, 2]))
    dr0 = (int(l1[1, 3]), int(l1[1, 4]), int(l1[1, 5]))
    ch0_bytes = enc3s([nl0, dl0, nr0, dr0])
    out += ch0_bytes
    transcript.absorb_bytes(ch0_bytes)

    br0 = transcript.squeeze_fp3()
    pt = [br0]
    claim_num = hadd(nl0, hmul(br0, hsub(nr0, nl0)))
    claim_den = hadd(dl0, hmul(br0, hsub(dr0, dl0)))

    layer_points = []
    layer_libra_evals = []

    # Masked layers k = 1 .. s - 1 (reducing tree[s - k] -> tree[s - k - 1])
    for k in range(1, s):
        alpha = transcript.squeeze_fp3()
        off_l = layout.layer_libra_offset(k)
        l_coeffs = libra_coeffs[off_l:off_l + 4 * k]
        g_sum = _libra3_sum(l_coeffs, k)
        g_sum_bytes = enc3(g_sum)
        out += g_sum_bytes
        transcript.absorb_bytes(g_sum_bytes)
        lam = transcript.squeeze_fp3()

        child_layer = tree[s - k - 1]
        eq_host = _eq_table_host(pt)
        if use_dev:
            cp = engine.cp
            w_col = engine.dev(eq_host)
            nl_col = cp.ascontiguousarray(child_layer[0::2, :3])
            dl_col = cp.ascontiguousarray(child_layer[0::2, 3:])
            nr_col = cp.ascontiguousarray(child_layer[1::2, :3])
            dr_col = cp.ascontiguousarray(child_layer[1::2, 3:])
        else:
            w_col = eq_host
            nl_col = np.ascontiguousarray(child_layer[0::2, :3])
            dl_col = np.ascontiguousarray(child_layer[0::2, 3:])
            nr_col = np.ascontiguousarray(child_layer[1::2, :3])
            dr_col = np.ascontiguousarray(child_layer[1::2, 3:])

        cols = [w_col, nl_col, dl_col, nr_col, dr_col]
        r_k = []
        libra_prefix = ZERO
        for i in range(k):
            if use_dev:
                raw_arr = engine.gkr_round(cols, alpha)
                base_vals = [tuple(int(v) for v in raw_arr[t]) for t in range(4)]
            else:
                base_vals = _gkr_round_host(cols, alpha)
            l_vals = _libra3_round_vals(l_coeffs, k, i, libra_prefix)
            vals = [hadd(base_vals[t], hmul(lam, l_vals[t])) for t in range(4)]
            msg = enc3s(vals)
            out += msg
            transcript.absorb_bytes(msg)
            ri = transcript.squeeze_fp3()
            r_k.append(ri)
            libra_prefix = hadd(libra_prefix, _libra3_eval_var(l_coeffs, i, ri))
            cols = [_halve_col(engine if use_dev else None, c, ri) for c in cols]

        # Extract final child evaluations (nl, dl, nr, dr) at r_k
        finals = []
        for idx_c in (1, 2, 3, 4):
            row = CE._host(engine, cols[idx_c])[0] if use_dev else cols[idx_c][0]
            finals.append((int(row[0]), int(row[1]), int(row[2])))
        nl, dl, nr, dr = finals
        g_r = libra_prefix
        fin_bytes = enc3s([nl, dl, nr, dr, g_r])
        out += fin_bytes
        transcript.absorb_bytes(fin_bytes)

        layer_points.append(r_k)
        layer_libra_evals.append(g_r)

        br_k = transcript.squeeze_fp3()
        pt = [br_k] + r_k
        claim_num = hadd(nl, hmul(br_k, hsub(nr, nl)))
        claim_den = hadd(dl, hmul(br_k, hsub(dr, dl)))

    assert len(out) == layout.proof_bytes, (len(out), layout.proof_bytes)
    return MultiGkrProofResult(
        proof=bytes(out),
        r_gkr=pt,
        num_eval=claim_num,
        den_eval=claim_den,
        layer_points=layer_points,
        layer_libra_evals=layer_libra_evals,
        libra_coeffs=libra_coeffs,
        mask_a=mask_a,
        mask_d=mask_d,
    )


# ----------------------------------------------------------------------------- Verifier
def _read_fp3(buf, off):
    if off + 24 > len(buf):
        raise MultiGkrError("E_LENGTH", "truncated Fp3 in GKR proof")
    try:
        return dec3(buf[off:off + 24]), off + 24
    except ValueError as e:
        raise MultiGkrError("E_NONCANONICAL", str(e)) from None


@dataclass(frozen=True)
class VerifiedGkrClaims:
    r_gkr: list[tuple[int, int, int]]
    num_eval: tuple[int, int, int]
    den_eval: tuple[int, int, int]
    layer_points: list[list[tuple[int, int, int]]]
    layer_libra_evals: list[tuple[int, int, int]]


def verify_multi_gkr(
    layout: MultiGkrLayout,
    proof: bytes,
    transcript: Fpn2Transcript,
) -> VerifiedGkrClaims:
    """Verify the GKR reduction from `(root_num == 0, root_den != 0)` down to leaf & Libra claims.

    Caller MUST subsequently verify via `multi_pcs` that:
      1. the committed leaf MLEs at `r_gkr` (including the committed cancelling mask pairs) equal
         `(num_eval, den_eval)`, and
      2. the committed Libra mask polynomials evaluated at `layer_points[k-1]` equal `layer_libra_evals[k-1]`.
    """
    s = layout.s
    if len(proof) != layout.proof_bytes:
        raise MultiGkrError("E_LENGTH", f"got {len(proof)} B, expected {layout.proof_bytes} B")

    off = 0
    r_start = off
    root_num, off = _read_fp3(proof, off)
    root_den, off = _read_fp3(proof, off)
    if root_num != ZERO or root_den == ZERO:
        raise MultiGkrError("E_GKR_ROOT", "root_num != 0 or root_den == 0")
    transcript.absorb_bytes(proof[r_start:off])

    # Layer 0
    c0_start = off
    nl0, off = _read_fp3(proof, off)
    dl0, off = _read_fp3(proof, off)
    nr0, off = _read_fp3(proof, off)
    dr0, off = _read_fp3(proof, off)
    if combine_frac((nl0, dl0), (nr0, dr0)) != (root_num, root_den):
        raise MultiGkrError("E_GKR", "layer 0 combine mismatch")
    transcript.absorb_bytes(proof[c0_start:off])

    br0 = transcript.squeeze_fp3()
    pt = [br0]
    claim_num = hadd(nl0, hmul(br0, hsub(nr0, nl0)))
    claim_den = hadd(dl0, hmul(br0, hsub(dr0, dl0)))

    layer_points = []
    layer_libra_evals = []

    for k in range(1, s):
        alpha = transcript.squeeze_fp3()
        gs_start = off
        g_sum, off = _read_fp3(proof, off)
        transcript.absorb_bytes(proof[gs_start:off])
        lam = transcript.squeeze_fp3()

        target = hadd(hadd(claim_num, hmul(alpha, claim_den)), hmul(lam, g_sum))
        r_k = []
        for _ in range(k):
            rd_start = off
            v0, off = _read_fp3(proof, off)
            v1, off = _read_fp3(proof, off)
            v2, off = _read_fp3(proof, off)
            v3, off = _read_fp3(proof, off)
            if hadd(v0, v1) != target:
                raise MultiGkrError("E_GKR", f"layer {k} sumcheck round sum mismatch")
            transcript.absorb_bytes(proof[rd_start:off])
            ri = transcript.squeeze_fp3()
            r_k.append(ri)
            target = lagrange_eval([v0, v1, v2, v3], ri)

        fin_start = off
        nl, off = _read_fp3(proof, off)
        dl, off = _read_fp3(proof, off)
        nr, off = _read_fp3(proof, off)
        dr, off = _read_fp3(proof, off)
        g_r, off = _read_fp3(proof, off)
        transcript.absorb_bytes(proof[fin_start:off])

        c_num, c_den = combine_frac((nl, dl), (nr, dr))
        eq_val = eq_eval_pt(pt, r_k)
        expect = hadd(hmul(eq_val, hadd(c_num, hmul(alpha, c_den))), hmul(lam, g_r))
        if expect != target:
            raise MultiGkrError("E_GKR", f"layer {k} final equation mismatch")

        layer_points.append(r_k)
        layer_libra_evals.append(g_r)

        br_k = transcript.squeeze_fp3()
        pt = [br_k] + r_k
        claim_num = hadd(nl, hmul(br_k, hsub(nr, nl)))
        claim_den = hadd(dl, hmul(br_k, hsub(dr, dl)))

    assert off == len(proof)
    return VerifiedGkrClaims(
        r_gkr=pt,
        num_eval=claim_num,
        den_eval=claim_den,
        layer_points=layer_points,
        layer_libra_evals=layer_libra_evals,
    )
