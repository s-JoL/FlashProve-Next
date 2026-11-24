# SPDX-License-Identifier: Apache-2.0
"""Relation 5 (`FOLDED_VEC_MATMUL`): Libra-masked inner-product sumcheck for B public 32 x K GEMM tiles + one
batched folded ZK-Basefold opening of (F, C).

Independent re-implementation of specs/11-wire-format.md §5.6 (never imports the prover).
Every transcript absorption takes the raw wire slice of the container.
"""
from dataclasses import dataclass

from .field import NonCanonical, ONE3, P, ZERO3, e_add, e_decode, e_mul, e_sub, lagrange_eval
from .folded_air import MASK_POINTS, _Reader, _vals, _weights_eval, deposit, free_bits
from .folded_gkr import _eq, _leaf, _merkle_ok, fold4
from .transcript import LABEL_V2, Fpn2Transcript
from .wire import Reject

RATE = 8
ROWS = 32
ROW_BITS = 5
IP_DEGREE = 3


@dataclass(frozen=True)
class VecLayout:
    n: int
    queries: int
    k: int

    @property
    def beta(self): return self.n - 1 - ROW_BITS - self.k

    @property
    def K(self): return 1 << self.k

    @property
    def B(self): return 1 << self.beta

    @property
    def d(self): return self.k + self.beta

    @property
    def folds(self): return self.n // 2

    def depth(self, t): return self.n + 1 - 2 * t

    @property
    def ip_len(self): return 24 + 24 * (IP_DEGREE + 1) * self.d + 24

    @property
    def pcs_len(self): return 24 + 72 * (self.n + 2) + 32 * (self.folds - 1) + 24

    @property
    def query_len(self):
        return (96 + 32 * self.depth(0)) + sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return 96 + self.ip_len + self.pcs_len + self.queries * self.query_len

    def block(self, off, kk):
        fb = free_bits(self.n)
        fixed = ((1, 1),) + tuple((fb[kk + i], (off >> (kk + i)) & 1) for i in range(len(fb) - kk))
        return fixed, tuple(fb[:kk])

    def libra_slot(self, l, e):
        return deposit(4 * l + e, ((1, 0),), free_bits(self.n))


def layout_of(prof):
    n, kdim = prof.n_vars, prof.max_hidden
    if prof.sc_degree != IP_DEGREE or n < 12 or n % 2 or n > 28 or not 1 <= prof.n_queries <= 4096:
        return None
    if kdim < ROWS or kdim & (kdim - 1):
        return None
    k = kdim.bit_length() - 1
    if n - 1 - ROW_BITS - k < 0:
        return None
    return VecLayout(n, prof.n_queries, k)


def _root_after(i, lay):
    return i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1


def decode_vec(body, lay):
    """Check 12 for relation 5: every Fp3 field canonical, in wire order."""
    rd = _Reader(body)
    try:
        rd.take(96)
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
                if t == 0:                       # weight-commitment leaf at the same coset
                    _vals(rd.take(96), 4)
                    rd.take(32 * lay.depth(0))
    except NonCanonical as e:
        raise Reject("E_NONCANONICAL", str(e))
    assert rd.off == len(body)


def _terms(lay, r_i, r_b, s, lam, e_w, mu):
    """F-blocks, F-sparse and C-blocks of target + mu e_W + mu^2 z_blind (§5.6.4)."""
    k = lay.k
    s_j, s_b = list(s[:k]), list(s[k:])
    eqrb = _eq(r_b, s_b)
    e_s = _eq(r_i, s_j[:ROW_BITS])
    for t in s_j[ROW_BITS:]:
        e_s = e_mul(e_s, e_sub(ONE3, t))
    fb = [(lay.block(0, lay.d), e_mul(eqrb, e_w), tuple(s)),
          (lay.block(lay.B * lay.K, ROW_BITS + lay.beta), e_sub(ZERO3, e_mul(eqrb, e_s)),
           tuple(s_j[:ROW_BITS]) + tuple(s_b))]
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
    return fb, sparse, cb


def verify_vec(header, body, lay, expected_weight_root):
    """Checks 12-17 of relation 5 on an already length/padding-checked container."""
    decode_vec(body, lay)                                                            # 12
    rd = _Reader(body)
    stmt = rd.take(96)
    if expected_weight_root is None:
        raise Reject("E_CIRCUIT_UNPINNED")                                           # 13
    if bytes(expected_weight_root) != stmt[32:64]:
        raise Reject("E_CIRCUIT_PINNED")
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
    fb, sparse, cb = _terms(lay, r_i, r_b, s, lam, e_w, mu)
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
    c_r = _weights_eval(lay.n, cb, [], rest)
    w_r = e_mul(e_sub(ONE3, r1), e_add(e_mul(e_sub(ONE3, theta), a_r), e_mul(e_sub(e_add(theta, theta), ONE3), c_r)))
    if y != e_mul(final, w_r):
        raise Reject("E_FINAL")                                                      # 16
    length = RATE << lay.n
    omega = pow(7, (P - 1) // length, P)
    iota_inv = pow(pow(7, (P - 1) // 4, P), P - 2, P)
    root_w = stmt[32:64]
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
                vals = [e_add(f, e_mul(theta, c)) for f, c in zip(vals, _vals(craw, 4))]
            elif vals[slot] != prev:
                raise Reject("E_FOLD", f"query {qi} layer {t}")
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(omega_t, (P - 1 - j) % (P - 1), P)
            prev = fold4(vals, x_inv, iota_inv, rest[2 * t], rest[2 * t + 1])
            pos = j
        if prev != final:
            raise Reject("E_FOLD", f"query {qi} terminal")
