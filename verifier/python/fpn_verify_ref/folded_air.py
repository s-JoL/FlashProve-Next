# SPDX-License-Identifier: Apache-2.0
"""Relation 4 (`FOLDED_GKR_AIR`): Plonkish zerocheck + wiring/range FracLogUp-GKR + batched folded ZK-Basefold.

Independent re-implementation of specs/11-wire-format.md §5.5 (never imports the prover).
Every transcript absorption takes the raw wire slice of the container.
"""
from dataclasses import dataclass
from typing import List

from .field import NonCanonical, ONE3, P, ZERO3, e_add, e_decode, e_mul, e_sub, lagrange_eval
from .folded_gkr import _bits, _combine, _eq, _leaf, _merkle_ok, _mix, _scale, fold4
from .transcript import LABEL_V2, Fpn2Transcript
from .wire import Reject

RATE = 8
TABLE_BITS = 7
ZC_DEGREE = 4
N_FINALS = 9
MASK_POINTS = ((2, 0, 0), (3, 0, 0))


def free_bits(n):
    return [0] + list(range(2, n))


def deposit(value, fixed, free):
    idx = 0
    for bit, val in fixed:
        idx |= val << bit
    for i, bit in enumerate(free):
        idx |= ((value >> i) & 1) << bit
    return idx


# ----------------------------------------------------------------------------- layout (§5.5.2, §5.5.7)
@dataclass(frozen=True)
class AirLayout:
    n: int
    queries: int

    @property
    def g(self): return self.n - 5

    @property
    def K(self): return 1 << self.g

    @property
    def S(self): return self.g + 3

    @property
    def folds(self): return self.n // 2

    def depth(self, t): return self.n + 1 - 2 * t

    def layer_len(self, k): return 24 + 96 * k + 120

    @property
    def gkr_len(self): return 48 + 96 + sum(self.layer_len(k) for k in range(1, self.S + 1))

    @property
    def zc_len(self): return 24 + self.g * 24 * (ZC_DEGREE + 1) + 24 * N_FINALS

    @property
    def pcs_len(self): return 24 + 72 * (self.n + 2) + 32 * (self.folds - 1) + 24

    @property
    def query_len(self):
        return (96 + 32 * self.depth(0)) + sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return 96 + self.gkr_len + self.zc_len + self.pcs_len + self.queries * self.query_len

    def block(self, off, k):
        """(fixed, free) subcube of the data-coordinate block [off, off + 2^k) of the half {bit1 = 1}."""
        fb = free_bits(self.n)
        fixed = ((1, 1),) + tuple((fb[k + i], (off >> (k + i)) & 1) for i in range(len(fb) - k))
        return fixed, tuple(fb[:k])

    def libra_slot(self, k, i, e):
        return deposit(2 * k * (k - 1) + 4 * i + e, ((1, 0),), free_bits(self.n))

    def zc_slot(self, i, e):
        return deposit(2 * self.S * (self.S + 1) + (ZC_DEGREE + 1) * i + e, ((1, 0),), free_bits(self.n))


def layout_of(prof):
    n = prof.n_vars
    if prof.sc_degree != 3 or n < 12 or n % 2 or n > 28 or not 1 <= prof.n_queries <= 4096:
        return None
    return AirLayout(n, prof.n_queries)


# ----------------------------------------------------------------------------- canonical decoding (§5.5.7)
class _Reader:
    def __init__(self, b):
        self.b, self.off = b, 0

    def take(self, k):
        out = self.b[self.off:self.off + k]
        self.off += k
        return out


def _vals(raw, count):
    return [e_decode(raw[24 * j:24 * j + 24]) for j in range(count)]


@dataclass
class AirBody:
    stmt: bytes
    sections: List[bytes]        # every absorbed slice in wire order (decoded lazily by the verifier)


def decode_air(body, lay):
    """Check 12 for relation 4: every Fp3 field canonical, in wire order."""
    rd = _Reader(body)
    try:
        rd.take(96)
        _vals(rd.take(48), 2)
        _vals(rd.take(96), 4)
        for k in range(1, lay.S + 1):
            _vals(rd.take(lay.layer_len(k)), (lay.layer_len(k)) // 24)
        _vals(rd.take(lay.zc_len), lay.zc_len // 24)
        _vals(rd.take(24), 1)
        for i in range(lay.n + 2):
            _vals(rd.take(72), 3)
            if i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1:
                rd.take(32)
        _vals(rd.take(24), 1)
        for _ in range(lay.queries):
            for t in range(lay.folds):
                _vals(rd.take(96), 4)
                rd.take(32 * lay.depth(t))
                if t == 0:                       # circuit-commitment leaf at the same coset
                    _vals(rd.take(96), 4)
                    rd.take(32 * lay.depth(0))
    except NonCanonical as e:
        raise Reject("E_NONCANONICAL", str(e))
    assert rd.off == len(body)


# ----------------------------------------------------------------------------- helpers
def _int_ext(point):
    acc = ZERO3
    for i, r in enumerate(point):
        acc = e_add(acc, _scale(1 << i, r))
    return acc


def _upoly(coeffs, x):
    acc = ZERO3
    for c in reversed(coeffs):
        acc = e_add(c, e_mul(x, acc))
    return acc


def _weights_eval(n, blocks, sparse, r):
    acc = ZERO3
    for (fixed, free), c, z in blocks:
        term = e_mul(c, _eq(z, [r[b] for b in free]))
        for bit, val in fixed:
            term = e_mul(term, r[bit] if val else e_sub(ONE3, r[bit]))
        acc = e_add(acc, term)
    for idx, c in sparse:
        acc = e_add(acc, e_mul(c, _eq(_bits(idx, n), r)))
    return acc


def _neg(x):
    return e_sub(ZERO3, x)


# ----------------------------------------------------------------------------- batched linear claim (§5.5.5)
def _terms(lay, point, beta, gamma, mu, libra, zc_point):
    """Returns (num_pub, den_pub, F-blocks, F-sparse, C-blocks, coefficient list, z_blind coefficient)."""
    g, S, K = lay.g, lay.S, lay.K
    rho0, rho = point[0], point[1:]
    low, top, ra = rho[:g], rho[g:g + 3], rho[:S - 1]
    eqb = [_eq(top, _bits(k, 3)) for k in range(8)]
    real = e_sub(ONE3, rho0)
    num_pub = e_mul(real, e_add(e_add(eqb[0], eqb[1]), eqb[2]))
    s05 = ZERO3
    for k in range(6):
        s05 = e_add(s05, eqb[k])
    den_pub = e_mul(real, e_sub(e_sub(e_mul(beta, s05), e_mul(e_mul(gamma, eqb[3]), e_add(ONE3, _int_ext(low)))),
                                e_mul(eqb[5], _int_ext(low[:TABLE_BITS]))))
    fb = [(lay.block(4 * K, g), _neg(e_mul(real, eqb[5])), low),
          (lay.block(5 * K, g), e_mul(real, e_sub(eqb[6], eqb[7])), low),
          (lay.block(8 * K, S - 1), e_mul(rho0, e_sub(ONE3, e_add(rho[S - 1], rho[S - 1]))), ra)]
    for c in range(3):
        fb.append((lay.block(c * K, g), e_mul(mu, _neg(e_mul(real, eqb[c]))), low))
    fb += [(lay.block(3 * K, g), e_mul(mu, _neg(e_mul(real, e_add(eqb[3], eqb[4])))), low),
           (lay.block(6 * K, g), e_mul(mu, e_mul(real, e_add(eqb[6], eqb[7]))), low),
           (lay.block(12 * K, S - 1), e_mul(mu, rho0), ra)]
    cb = [(lay.block(8 * K, g), _neg(e_mul(real, eqb[3])), low),
          (lay.block(9 * K, g), e_mul(real, eqb[4]), low)]
    for c in range(3):
        cb.append((lay.block((5 + c) * K, g), e_mul(mu, _neg(e_mul(e_mul(real, gamma), eqb[c]))), low))
    sparse, coefs = [], []
    coef = e_mul(mu, mu)
    for k, rs in libra:
        for i, r in enumerate(rs):
            pw = ONE3
            for e in range(4):
                sparse.append((lay.libra_slot(k, i, e), e_mul(coef, pw)))
                pw = e_mul(pw, r)
        coefs.append(coef)
        coef = e_mul(coef, mu)
    for j in range(N_FINALS):
        coefs.append(coef)
        if j < 3:
            fb.append((lay.block(j * K, g), coef, tuple(zc_point)))
        elif j < 8:
            cb.append((lay.block((j - 3) * K, g), coef, tuple(zc_point)))
        else:
            for i, r in enumerate(zc_point):
                pw = ONE3
                for e in range(ZC_DEGREE + 1):
                    sparse.append((lay.zc_slot(i, e), e_mul(coef, pw)))
                    pw = e_mul(pw, r)
        coef = e_mul(coef, mu)
    free = tuple(b for b in range(lay.n) if b != 1)
    for z in MASK_POINTS:
        fb.append(((((1, 0),), free), e_mul(coef, e_sub(ONE3, z)), tuple(z for _ in free)))
    return num_pub, den_pub, fb, sparse, cb, coefs, coef


# ----------------------------------------------------------------------------- verifier (§5.5.8)
def verify_air(header, body, lay, expected_circuit_root):
    """Checks 12-19 of relation 4 on an already length/padding-checked container."""
    decode_air(body, lay)                                                            # 12
    rd = _Reader(body)
    stmt = rd.take(96)
    if expected_circuit_root is None:
        raise Reject("E_CIRCUIT_UNPINNED")                                           # 13
    if bytes(expected_circuit_root) != stmt[32:64]:
        raise Reject("E_CIRCUIT_PINNED")
    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(header)
    tr.absorb_bytes(stmt)
    beta = tr.squeeze_fp3()
    gamma = tr.squeeze_fp3()
    root_raw = rd.take(48)
    tr.absorb_bytes(root_raw)
    root = _vals(root_raw, 2)
    if root[0] != ZERO3 or root[1] == ZERO3:
        raise Reject("E_GKR_ROOT")                                                   # 14
    point, claim, libra, g_rs = [], tuple(root), [], []
    for k in range(lay.S + 1):                                                       # 15
        alpha = tr.squeeze_fp3()
        target = e_add(claim[0], e_mul(alpha, claim[1]))
        lam = None
        if k:
            g_raw = rd.take(24)
            tr.absorb_bytes(g_raw)
            lam = tr.squeeze_fp3()
            target = e_add(target, e_mul(lam, e_decode(g_raw)))
        rs = []
        for _ in range(k):
            raw = rd.take(96)
            tr.absorb_bytes(raw)
            gv = _vals(raw, 4)
            if e_add(gv[0], gv[1]) != target:
                raise Reject("E_GKR", f"layer {k} round sum")
            r = tr.squeeze_fp3()
            rs.append(r)
            target = lagrange_eval(gv, r)
        tail = rd.take(120 if k else 96)
        tr.absorb_bytes(tail)
        tv = _vals(tail, 5 if k else 4)
        nl, dl, nr, dr = tv[:4]
        num, den = _combine(nl, dl, nr, dr)
        expect = e_mul(_eq(point, rs), e_add(num, e_mul(alpha, den)))
        if k:
            expect = e_add(expect, e_mul(lam, tv[4]))
            libra.append((k, rs))
            g_rs.append(tv[4])
        if target != expect:
            raise Reject("E_GKR", f"layer {k} final")
        branch = tr.squeeze_fp3()
        point = [branch] + rs
        claim = (_mix(nl, nr, branch), _mix(dl, dr, branch))
    z = [tr.squeeze_fp3() for _ in range(lay.g)]                                     # 16
    zs_raw = rd.take(24)
    tr.absorb_bytes(zs_raw)
    lam_z = tr.squeeze_fp3()
    target = e_mul(lam_z, e_decode(zs_raw))
    zr = []
    for i in range(lay.g):
        raw = rd.take(24 * (ZC_DEGREE + 1))
        tr.absorb_bytes(raw)
        sv = _vals(raw, ZC_DEGREE + 1)
        if e_add(sv[0], sv[1]) != target:
            raise Reject("E_ZEROCHECK", f"round {i}")
        r = tr.squeeze_fp3()
        zr.append(r)
        target = lagrange_eval(sv, r)
    fin_raw = rd.take(24 * N_FINALS)
    tr.absorb_bytes(fin_raw)
    ea, eb, ec, qm, ql, qr, qo, qc, zg = _vals(fin_raw, N_FINALS)
    gate = e_add(e_add(e_add(e_mul(qm, e_mul(ea, eb)), e_mul(ql, ea)), e_add(e_mul(qr, eb), e_mul(qo, ec))), qc)
    if target != e_add(e_mul(_eq(z, zr), gate), e_mul(lam_z, zg)):
        raise Reject("E_ZEROCHECK", "final")
    zb_raw = rd.take(24)
    tr.absorb_bytes(zb_raw)
    mu = tr.squeeze_fp3()
    num_pub, den_pub, fb, sparse, cb, coefs, zcoef = _terms(lay, point, beta, gamma, mu, libra, zr)
    y = e_add(e_sub(claim[0], num_pub), e_mul(mu, e_sub(claim[1], den_pub)))
    for c, v in zip(coefs, g_rs + [ea, eb, ec, qm, ql, qr, qo, qc, zg]):
        y = e_add(y, e_mul(c, v))
    y = e_add(y, e_mul(zcoef, e_decode(zb_raw)))
    rs, roots = [], [stmt[:32]]
    for i in range(lay.n + 2):                                                       # 17
        raw = rd.take(72)
        tr.absorb_bytes(raw)
        h = _vals(raw, 3)
        if e_add(h[0], h[1]) != y:
            raise Reject("E_SUMCHECK", f"pcs round {i}")
        r = tr.squeeze_fp3()
        rs.append(r)
        y = lagrange_eval(h, r)
        if i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1:
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
        raise Reject("E_FINAL")                                                      # 18
    length = RATE << lay.n
    omega = pow(7, (P - 1) // length, P)
    iota_inv = pow(pow(7, (P - 1) // 4, P), P - 2, P)
    root_circ = stmt[32:64]
    for qi in range(lay.queries):                                                    # 19
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
                if not _merkle_ok(root_circ, _leaf(craw), j, csibs):
                    raise Reject("E_MERKLE", f"query {qi} circuit leaf")
                vals = [e_add(f, e_mul(theta, c)) for f, c in zip(vals, _vals(craw, 4))]
            elif vals[slot] != prev:
                raise Reject("E_FOLD", f"query {qi} layer {t}")
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(omega_t, (P - 1 - j) % (P - 1), P)
            prev = fold4(vals, x_inv, iota_inv, rest[2 * t], rest[2 * t + 1])
            pos = j
        if prev != final:
            raise Reject("E_FOLD", f"query {qi} terminal")
