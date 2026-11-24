# SPDX-License-Identifier: Apache-2.0
"""Relation 6 (`FOLDED_VEC_BOUND`): relation 5's Libra-masked inner-product sumcheck with the activations `X`
moved out of the witness vector into a SEPARATE pre-committed, ZK-blinded vector `X_com` (root `x_root`, pinned by
the caller), discharged by ONE batched folded ZK-Basefold opening of the three vectors `(F, C, X_com)`.

Independent re-implementation of specs/11-wire-format.md §5.7 (never imports the prover).
Every transcript absorption takes the raw wire slice of the container.
"""
from dataclasses import dataclass

from .field import NonCanonical, ONE3, P, ZERO3, e_add, e_decode, e_mul, e_sub, lagrange_eval
from .folded_air import MASK_POINTS, _Reader, _vals, _weights_eval
from .folded_gkr import _eq, _leaf, _merkle_ok, fold4
from .folded_vec import IP_DEGREE, RATE, ROW_BITS, ROWS, VecLayout, _root_after
from .folded_vec import layout_of as _vec_layout_of
from .transcript import LABEL_V2, Fpn2Transcript
from .wire import Reject

STMT = 128      # root_0 || weight_root || x_root || N_s


@dataclass(frozen=True)
class BoundLayout(VecLayout):
    """Relation-6 geometry: relation 5's `(n, Q, k)` plus the `X_com` layer-0 leaf in every query."""

    @property
    def query_len(self):
        return 2 * (96 + 32 * self.depth(0)) + sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return STMT + self.ip_len + self.pcs_len + self.queries * self.query_len


def layout_of(prof):
    lay = _vec_layout_of(prof)
    return None if lay is None else BoundLayout(lay.n, lay.queries, lay.k)


def decode_bound(body, lay):
    """Check 12 for relation 6: every Fp3 field canonical, in wire order."""
    rd = _Reader(body)
    try:
        rd.take(STMT)
        _vals(rd.take(lay.ip_len), lay.ip_len // 24)
        _vals(rd.take(24), 1)
        for i in range(lay.n + 2):
            _vals(rd.take(72), 3)
            if _root_after(i, lay):
                rd.take(32)
        _vals(rd.take(24), 1)
        for _ in range(lay.queries):
            for t in range(lay.folds):
                _vals(rd.take(96), 4)
                rd.take(32 * lay.depth(t))
                if t == 0:                       # weight leaf, then X_com leaf, at the same coset
                    _vals(rd.take(96), 4)
                    rd.take(32 * lay.depth(0))
                    _vals(rd.take(96), 4)
                    rd.take(32 * lay.depth(0))
    except NonCanonical as e:
        raise Reject("E_NONCANONICAL", str(e))
    assert rd.off == len(body)


def _terms(lay, r_i, r_b, s, lam, e_w, mu):
    """(F-blocks, F-sparse, X-blocks, C-blocks) of target + mu e_W + mu^2 z_blind (§5.7.4)."""
    k = lay.k
    s_j, s_b = list(s[:k]), list(s[k:])
    eqrb = _eq(r_b, s_b)
    e_s = _eq(r_i, s_j[:ROW_BITS])
    for t in s_j[ROW_BITS:]:
        e_s = e_mul(e_s, e_sub(ONE3, t))
    fb = [(lay.block(lay.B * lay.K, ROW_BITS + lay.beta), e_sub(ZERO3, e_mul(eqrb, e_s)),
           tuple(s_j[:ROW_BITS]) + tuple(s_b))]
    xb = [(lay.block(0, lay.d), e_mul(eqrb, e_w), tuple(s))]
    sparse = []
    for l, r in enumerate(s):
        pw = lam
        for e in range(IP_DEGREE + 1):
            sparse.append((lay.libra_slot(l, e), pw))
            pw = e_mul(pw, r)
    coef = e_mul(mu, mu)
    free = tuple(b for b in range(lay.n) if b != 1)
    for z in MASK_POINTS:
        fb.append(((((1, 0),), free), e_mul(coef, e_sub(ONE3, z)), tuple(z for _ in free)))
    cb = [(lay.block(0, lay.n - 1), mu, tuple(s_j) + tuple(r_i) + tuple(s_b))]
    return fb, sparse, xb, cb


def verify_bound(header, body, lay, expected_weight_root, expected_x_root):
    """Checks 12-17 of relation 6 on an already length/padding-checked container."""
    decode_bound(body, lay)                                                          # 12
    rd = _Reader(body)
    stmt = rd.take(STMT)
    if expected_weight_root is None:                                                 # 13
        raise Reject("E_CIRCUIT_UNPINNED")
    if expected_x_root is None:
        raise Reject("E_X_ROOT_UNPINNED")
    if bytes(expected_weight_root) != stmt[32:64]:
        raise Reject("E_CIRCUIT_PINNED")
    if bytes(expected_x_root) != stmt[64:96]:
        raise Reject("E_X_ROOT_PINNED")
    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(header)
    tr.absorb_bytes(stmt)
    r_i = [tr.squeeze_fp3() for _ in range(ROW_BITS)]
    r_b = [tr.squeeze_fp3() for _ in range(lay.beta)]
    gs_raw = rd.take(24)                                                             # 14
    tr.absorb_bytes(gs_raw)
    lam = tr.squeeze_fp3()
    target = e_mul(lam, e_decode(gs_raw))
    s = []
    for l in range(lay.d):
        raw = rd.take(24 * (IP_DEGREE + 1))
        tr.absorb_bytes(raw)
        sv = _vals(raw, IP_DEGREE + 1)
        if e_add(sv[0], sv[1]) != target:
            raise Reject("E_ZEROCHECK", f"inner-product round {l}")
        r = tr.squeeze_fp3()
        s.append(r)
        target = lagrange_eval(sv, r)
    ew_raw = rd.take(24)
    tr.absorb_bytes(ew_raw)
    e_w = e_decode(ew_raw)
    zb_raw = rd.take(24)
    tr.absorb_bytes(zb_raw)
    mu = tr.squeeze_fp3()
    fb, sparse, xb, cb = _terms(lay, r_i, r_b, s, lam, e_w, mu)
    y = e_add(e_add(target, e_mul(mu, e_w)), e_mul(e_mul(mu, mu), e_decode(zb_raw)))
    rs, roots = [], [stmt[:32]]
    for i in range(lay.n + 2):                                                       # 15
        raw = rd.take(72)
        tr.absorb_bytes(raw)
        h = _vals(raw, 3)
        if e_add(h[0], h[1]) != y:
            raise Reject("E_SUMCHECK", f"pcs round {i}")
        r = tr.squeeze_fp3()
        rs.append(r)
        y = lagrange_eval(h, r)
        if _root_after(i, lay):
            root_t = rd.take(32)
            tr.absorb_bytes(root_t)
            roots.append(root_t)
    final_raw = rd.take(24)
    tr.absorb_bytes(final_raw)
    final = e_decode(final_raw)
    theta, r1, rest = rs[0], rs[1], rs[2:]
    a_r = _weights_eval(lay.n, fb, sparse, rest)
    x_r = _weights_eval(lay.n, xb, [], rest)
    c_r = _weights_eval(lay.n, cb, [], rest)
    om_t, om_r = e_sub(ONE3, theta), e_sub(ONE3, r1)
    w_r = e_add(e_add(e_mul(e_mul(om_r, om_t), a_r),
                      e_mul(e_mul(om_r, e_sub(e_add(theta, theta), ONE3)), c_r)),
                e_mul(e_mul(om_t, e_sub(e_add(r1, r1), ONE3)), x_r))
    if y != e_mul(final, w_r):
        raise Reject("E_FINAL")                                                      # 16
    length = RATE << lay.n
    omega = pow(7, (P - 1) // length, P)
    iota_inv = pow(pow(7, (P - 1) // 4, P), P - 2, P)
    root_w, root_x = stmt[32:64], stmt[64:96]
    for qi in range(lay.queries):                                                    # 17
        pos = int.from_bytes(tr.squeeze_bytes(8), "little") % (length // 4)
        prev = None
        for t in range(lay.folds):
            quarter = length >> (2 * t + 2)
            j, slot = pos % quarter, pos // quarter
            raw = rd.take(96)
            sibs = [rd.take(32) for _ in range(lay.depth(t))]
            if not _merkle_ok(roots[t], _leaf(raw), j, sibs):
                raise Reject("E_MERKLE", f"query {qi} layer {t}")
            vals = _vals(raw, 4)
            if t == 0:
                craw = rd.take(96)
                csibs = [rd.take(32) for _ in range(lay.depth(0))]
                if not _merkle_ok(root_w, _leaf(craw), j, csibs):
                    raise Reject("E_MERKLE", f"query {qi} weight leaf")
                xraw = rd.take(96)
                xsibs = [rd.take(32) for _ in range(lay.depth(0))]
                if not _merkle_ok(root_x, _leaf(xraw), j, xsibs):
                    raise Reject("E_MERKLE", f"query {qi} x leaf")
                vals = [e_add(e_add(f, e_mul(theta, c)), e_mul(r1, xv))
                        for f, c, xv in zip(vals, _vals(craw, 4), _vals(xraw, 4))]
            elif vals[slot] != prev:
                raise Reject("E_FOLD", f"query {qi} layer {t}")
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(omega_t, (P - 1 - j) % (P - 1), P)
            prev = fold4(vals, x_inv, iota_inv, rest[2 * t], rest[2 * t + 1])
            pos = j
        if prev != final:
            raise Reject("E_FOLD", f"query {qi} terminal")
