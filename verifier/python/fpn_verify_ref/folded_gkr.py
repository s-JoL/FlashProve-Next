# SPDX-License-Identifier: Apache-2.0
"""Relation 3 (`FOLDED_GKR_LOOKUP`): ZK-masked FracLogUp-GKR + fused two-round folded ZK-Basefold.

Independent re-implementation of specs/11-wire-format.md §5.4 (never imports the prover).
Every transcript absorption takes the raw wire slice of the container.
"""
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

from .field import (NonCanonical, ONE3, P, ZERO3, e_add, e_decode, e_mul, e_sub, lagrange_eval)
from .transcript import LABEL_V2, Fpn2Transcript
from .wire import Reject

PERSONAL_MERKLE = b"FPN2-MERKLE-v1\x00\x00"
RATE = 8
INV4 = pow(4, P - 2, P)
MASK_POINTS = ((2, 0, 0), (3, 0, 0))
DATA_FIXED = ((1, 1), (3, 1))
LIBRA_FIXED = ((1, 0), (3, 0))
MASK_SUBCUBES = (((1, 0),), ((1, 1), (3, 0)))


FoldedReject = Reject


def _scale(c, a):
    return (c * a[0] % P, c * a[1] % P, c * a[2] % P)


def _eq1(a, b):
    return e_add(e_mul(a, b), e_mul(e_sub(ONE3, a), e_sub(ONE3, b)))


def _eq(a, b):
    acc = ONE3
    for x, y in zip(a, b):
        acc = e_mul(acc, _eq1(x, y))
    return acc


def _bits(value, count):
    return [ONE3 if value >> i & 1 else ZERO3 for i in range(count)]


def free_bits(n):
    return [0, 2] + list(range(4, n))


def deposit(value, fixed, free):
    idx = 0
    for bit, val in fixed:
        idx |= val << bit
    for i, bit in enumerate(free):
        idx |= ((value >> i) & 1) << bit
    return idx


# ----------------------------------------------------------------------------- layout (§5.4.2, §5.4.7)
@dataclass(frozen=True)
class FoldedLayout:
    n: int
    queries: int

    @property
    def s(self): return self.n - 3

    @property
    def quarter(self): return 1 << (self.s - 2)

    @property
    def folds(self): return self.n // 2

    def depth(self, t): return self.n + 1 - 2 * t

    def layer_len(self, k): return 24 + 96 * k + 120

    @property
    def gkr_len(self): return 48 + 96 + sum(self.layer_len(k) for k in range(1, self.s + 1))

    @property
    def pcs_len(self): return 24 + 72 * self.n + 32 * (self.folds - 1) + 24

    @property
    def query_len(self): return sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self): return 64 + self.gkr_len + self.pcs_len + self.queries * self.query_len

    def data_block(self, off, k):
        fb = free_bits(self.n)
        fixed = DATA_FIXED + tuple((fb[k + i], (off >> (k + i)) & 1) for i in range(len(fb) - k))
        return fixed, tuple(fb[:k])

    def libra_slot(self, k, i, e):
        return deposit(2 * k * (k - 1) + 4 * i + e, LIBRA_FIXED, free_bits(self.n))


def layout_of(prof):
    n = prof.n_vars
    if prof.sc_degree != 3 or n < 8 or n % 2 or n > 28 or not 1 <= prof.n_queries <= 4096:
        return None
    return FoldedLayout(n, prof.n_queries)


# ----------------------------------------------------------------------------- canonical decoding (§5.4.7)
@dataclass
class GkrLayer:
    g_sum: object
    g_sum_raw: bytes
    rounds: List[list]
    rounds_raw: List[bytes]
    children: list
    tail_raw: bytes          # children (|| g_r) as absorbed
    g_r: object


@dataclass
class FoldedBody:
    stmt: bytes
    root_raw: bytes
    root: Tuple
    layers: List[GkrLayer]
    z_blind: Tuple
    z_raw: bytes
    pcs_rounds: List[list]
    pcs_rounds_raw: List[bytes]
    pcs_roots: List[bytes]
    final: Tuple
    final_raw: bytes
    queries: List[list]      # per query: list of (values[4], siblings[])


class _Reader:
    def __init__(self, b):
        self.b, self.off = b, 0

    def take(self, k):
        out = self.b[self.off:self.off + k]
        self.off += k
        return out

    def fp3(self):
        return e_decode(self.take(24))


def decode_folded(body, lay):
    """Check 12 for relation 3: every Fp3 field canonical, in wire order (raises Reject codes)."""
    rd = _Reader(body)
    try:
        stmt = rd.take(64)
        root_raw = rd.take(48)
        root = (e_decode(root_raw[:24]), e_decode(root_raw[24:]))
        layers = []
        raw = rd.take(96)
        layers.append(GkrLayer(None, b"", [], [], [e_decode(raw[24 * i:24 * i + 24]) for i in range(4)], raw, None))
        for k in range(1, lay.s + 1):
            g_raw = rd.take(24)
            g_sum = e_decode(g_raw)
            rounds_raw = [rd.take(96) for _ in range(k)]
            rounds = [[e_decode(r[24 * j:24 * j + 24]) for j in range(4)] for r in rounds_raw]
            tail = rd.take(120)
            vals = [e_decode(tail[24 * j:24 * j + 24]) for j in range(5)]
            layers.append(GkrLayer(g_sum, g_raw, rounds, rounds_raw, vals[:4], tail, vals[4]))
        z_raw = rd.take(24)
        z_blind = e_decode(z_raw)
        pcs_rounds_raw, pcs_roots = [], []
        for t in range(lay.folds):
            pcs_rounds_raw += [rd.take(72), rd.take(72)]
            if t < lay.folds - 1:
                pcs_roots.append(rd.take(32))
        pcs_rounds = [[e_decode(r[24 * j:24 * j + 24]) for j in range(3)] for r in pcs_rounds_raw]
        final_raw = rd.take(24)
        final = e_decode(final_raw)
        queries = []
        for _ in range(lay.queries):
            q = []
            for t in range(lay.folds):
                vb = rd.take(96)
                vals = [e_decode(vb[24 * j:24 * j + 24]) for j in range(4)]
                sibs = [rd.take(32) for _ in range(lay.depth(t))]
                q.append((vals, vb, sibs))
            queries.append(q)
    except NonCanonical as e:
        raise FoldedReject("E_NONCANONICAL", str(e))
    return FoldedBody(stmt, root_raw, root, layers, z_blind, z_raw, pcs_rounds, pcs_rounds_raw, pcs_roots,
                      final, final_raw, queries)


# ----------------------------------------------------------------------------- fraction GKR helpers
def _combine(nl, dl, nr, dr):
    return e_add(e_mul(nl, dr), e_mul(nr, dl)), e_mul(dl, dr)


def _mix(a, b, r):
    return e_add(a, e_mul(r, e_sub(b, a)))


# ----------------------------------------------------------------------------- batched linear claim (§5.4.4)
def _terms(lay, point, beta, libra, mu):
    s = lay.s
    rho0, rho = point[0], point[1:]
    low, ra, lowp = rho[:s - 2], rho[:s - 1], rho[:s - 3]
    top = (rho[s - 2], rho[s - 1])
    pair = e_sub(ONE3, e_add(rho[s - 3], rho[s - 3]))
    eqq = [_eq(top, _bits(qd, 2)) for qd in range(4)]
    real = e_sub(ONE3, rho0)
    t_ext = ZERO3
    for i, r in enumerate(low):
        t_ext = e_add(t_ext, _scale(1 << i, r))
    num_pub = e_mul(real, eqq[0])
    den_pub = e_mul(real, e_add(e_mul(eqq[0], beta), e_mul(eqq[2], e_sub(beta, t_ext))))
    q4 = lay.quarter
    off_m, off_e1, off_e3 = q4, 2 * q4, 3 * q4
    off_a, off_d = 1 << s, (1 << s) + (1 << (s - 1))
    neg = lambda x: e_sub(ZERO3, x)  # noqa: E731
    h = q4 // 2
    blocks = [(lay.data_block(off_m, s - 2), neg(e_mul(real, eqq[2])), low),
              (lay.data_block(off_e1, s - 3), e_mul(e_mul(real, eqq[1]), pair), lowp),
              (lay.data_block(off_e3, s - 3), e_mul(e_mul(real, eqq[3]), pair), lowp),
              (lay.data_block(off_a, s - 1), e_mul(rho0, e_sub(ONE3, e_add(rho[s - 1], rho[s - 1]))), ra),
              (lay.data_block(0, s - 2), e_mul(mu, neg(e_mul(real, eqq[0]))), low),
              (lay.data_block(off_e1 + h, s - 3), e_mul(mu, e_mul(real, eqq[1])), lowp),
              (lay.data_block(off_e3 + h, s - 3), e_mul(mu, e_mul(real, eqq[3])), lowp),
              (lay.data_block(off_d, s - 1), e_mul(mu, rho0), ra)]
    sparse = []
    coef = e_mul(mu, mu)
    for k, rs, _ in libra:
        for i, r in enumerate(rs):
            pw = ONE3
            for e in range(4):
                sparse.append((lay.libra_slot(k, i, e), e_mul(coef, pw)))
                pw = e_mul(pw, r)
        coef = e_mul(coef, mu)
    for z in MASK_POINTS:
        for fixed in MASK_SUBCUBES:
            c = coef
            for _, val in fixed:
                c = e_mul(c, z if val else e_sub(ONE3, z))
            fb = {b for b, _ in fixed}
            free = tuple(b for b in range(lay.n) if b not in fb)
            blocks.append(((tuple(fixed), free), c, tuple(z for _ in free)))
    return num_pub, den_pub, blocks, sparse, coef


def _weights_eval(lay, blocks, sparse, r):
    acc = ZERO3
    for (fixed, free), c, z in blocks:
        term = e_mul(c, _eq(z, [r[b] for b in free]))
        for bit, val in fixed:
            term = e_mul(term, r[bit] if val else e_sub(ONE3, r[bit]))
        acc = e_add(acc, term)
    for idx, c in sparse:
        acc = e_add(acc, e_mul(c, _eq(_bits(idx, lay.n), r)))
    return acc


# ----------------------------------------------------------------------------- Merkle + fold (§5.4.5)
def _leaf(raw_values):
    return hashlib.blake2b(b"\x00" + raw_values, digest_size=32, person=PERSONAL_MERKLE).digest()


def _node(a, b):
    return hashlib.blake2b(b"\x01" + a + b, digest_size=32, person=PERSONAL_MERKLE).digest()


def _merkle_ok(root, leaf, idx, sibs):
    node = leaf
    for sib in sibs:
        node = _node(sib, node) if idx & 1 else _node(node, sib)
        idx >>= 1
    return node == root


def fold4(v, x_inv, iota_inv, ra, rb):
    a, b = e_sub(v[0], v[2]), e_sub(v[1], v[3])
    tb = _scale(iota_inv, b)
    s = (e_add(e_add(v[0], v[1]), e_add(v[2], v[3])), e_add(a, tb),
         e_sub(e_add(v[0], v[2]), e_add(v[1], v[3])), e_sub(a, tb))
    f, scale = [], INV4
    for e in range(4):
        f.append(_scale(scale, s[e]))
        scale = scale * x_inv % P
    return e_add(e_add(f[0], e_mul(ra, f[1])), e_add(e_mul(rb, f[2]), e_mul(e_mul(ra, rb), f[3])))


# ----------------------------------------------------------------------------- verifier (§5.4.8)
def verify_folded(header, body, lay):
    """Checks 13-17 of relation 3 on an already length/padding-checked container."""
    fb = decode_folded(body, lay)                                                      # 12
    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(header)
    tr.absorb_bytes(fb.stmt)
    beta = tr.squeeze_fp3()
    tr.absorb_bytes(fb.root_raw)
    if fb.root[0] != ZERO3 or fb.root[1] == ZERO3:
        raise FoldedReject("E_GKR_ROOT")                                               # 13
    point, claim, libra = [], fb.root, []
    for k, layer in enumerate(fb.layers):                                              # 14
        alpha = tr.squeeze_fp3()
        target = e_add(claim[0], e_mul(alpha, claim[1]))
        lam = None
        if k:
            tr.absorb_bytes(layer.g_sum_raw)
            lam = tr.squeeze_fp3()
            target = e_add(target, e_mul(lam, layer.g_sum))
        rs = []
        for g, raw in zip(layer.rounds, layer.rounds_raw):
            tr.absorb_bytes(raw)
            if e_add(g[0], g[1]) != target:
                raise FoldedReject("E_GKR", f"layer {k} round sum")
            r = tr.squeeze_fp3()
            rs.append(r)
            target = lagrange_eval(g, r)
        tr.absorb_bytes(layer.tail_raw)
        nl, dl, nr, dr = layer.children
        num, den = _combine(nl, dl, nr, dr)
        expect = e_mul(_eq(point, rs), e_add(num, e_mul(alpha, den)))
        if k:
            expect = e_add(expect, e_mul(lam, layer.g_r))
            libra.append((k, rs, layer.g_r))
        if target != expect:
            raise FoldedReject("E_GKR", f"layer {k} final")
        branch = tr.squeeze_fp3()
        point = [branch] + rs
        claim = (_mix(nl, nr, branch), _mix(dl, dr, branch))
    tr.absorb_bytes(fb.z_raw)
    mu = tr.squeeze_fp3()
    num_pub, den_pub, blocks, sparse, coef = _terms(lay, point, beta, libra, mu)
    y = e_add(e_sub(claim[0], num_pub), e_mul(mu, e_sub(claim[1], den_pub)))
    c = e_mul(mu, mu)
    for _, _, g_r in libra:
        y = e_add(y, e_mul(c, g_r))
        c = e_mul(c, mu)
    y = e_add(y, e_mul(coef, fb.z_blind))
    rs = []
    for t in range(lay.folds):                                                         # 15
        for half in range(2):
            i = 2 * t + half
            h = fb.pcs_rounds[i]
            tr.absorb_bytes(fb.pcs_rounds_raw[i])
            if e_add(h[0], h[1]) != y:
                raise FoldedReject("E_SUMCHECK", f"pcs round {i}")
            r = tr.squeeze_fp3()
            rs.append(r)
            y = lagrange_eval(h, r)
        if t < lay.folds - 1:
            tr.absorb_bytes(fb.pcs_roots[t])
    tr.absorb_bytes(fb.final_raw)
    if y != e_mul(fb.final, _weights_eval(lay, blocks, sparse, rs)):
        raise FoldedReject("E_FINAL")                                                  # 16
    roots = [fb.stmt[:32]] + fb.pcs_roots
    length = RATE << lay.n
    omega = pow(7, (P - 1) // length, P)
    iota_inv = pow(pow(7, (P - 1) // 4, P), P - 2, P)
    for qi, q in enumerate(fb.queries):                                                # 17
        pos = int.from_bytes(tr.squeeze_bytes(8), "little") % (length // 4)
        prev = None
        for t, (vals, raw, sibs) in enumerate(q):
            quarter = length >> (2 * t + 2)
            j, slot = pos % quarter, pos // quarter
            if not _merkle_ok(roots[t], _leaf(raw), j, sibs):
                raise FoldedReject("E_MERKLE", f"query {qi} layer {t}")
            if t and vals[slot] != prev:
                raise FoldedReject("E_FOLD", f"query {qi} layer {t}")
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(omega_t, (P - 1 - j) % (P - 1), P)
            prev = fold4(vals, x_inv, iota_inv, rs[2 * t], rs[2 * t + 1])
            pos = j
        if prev != fb.final:
            raise FoldedReject("E_FOLD", f"query {qi} terminal")
    return fb
