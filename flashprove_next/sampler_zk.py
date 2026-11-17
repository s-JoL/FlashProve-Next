# SPDX-License-Identifier: Apache-2.0
"""Zero-knowledge circuit for the verifier-side sampling check S_t (specs/01 §5.4, specs/06 §3.2) of rule
``fpn-samp-bk-v3`` (``flashprove_next.sampling_v3``), inside the flat argument (``flashprove_next.zk_flat``).

Relation (per checked generation step s < K; everything hidden unless marked public):
    inputs from the HEAD gadget interface (``certified.blk_head``): reference logit centres Y_v and certified radii
        R_v at the hidden per-step exponent e_y (z_v = Y_v 2^e_y, r_v = R_v 2^e_y), v < V; allowed mask a_v
        (constrained-decoding interface);
    K_prf (hidden) bound to the committed config by kcom = Poseidon2-sponge(DOM_KCOM; salt, K_prf)   [public kcom]
    ctx encoding (public), role tag d_s and step index t_s (public in the standalone statement);
    the emitted token u_s (public in the standalone statement, hidden otherwise)
  and the circuit proves that u_s is accepted by the relaxed predicate Samp^bk of specs/06 §3.2 / sampling_v3.acceptable_set:
    m = max_{a_v} z_v (one-hot argmax + d_v >= 0 for every allowed v), j_v from (m - z_v) c, global band Delta from
    R = max_{a_v} r_v, w_lo = E[min(j + Delta, J)], w_hi = E[max(j - Delta, 0)], tight nucleus bounds (N^lo, N^hi) via
    hidden boundary tokens + interval classification, bucket masses, Gumbel bucket keys from the in-circuit Poseidon2
    PRF (one call per 16 buckets, K_prf in the capacity), LN keys, bucket condition, in-bucket inverse-CDF condition.
  Nothing about the logits leaves the circuit: every per-entry / per-bucket / per-step quantity is a committed
  (T1-blinded) column; the only public values are the statement items above.

Integer form of step 1 (the one place where the circuit differs from sampling_v3's float64 arithmetic):
    q_v  = floor(d_v 2^(e_y + Fq)),  d_v = M - Y_v      (hidden shift kq = -e_y - Fq in [0, 32]; q-unit 2^-Fq logits)
    j_v  = min(floor((2 q_v + 1) c 2^-(Fq+1)), JC)       (public table TJ on q < 2^QL; q >= 2^QL -> j_v >= BIG)
    Delta = ceil(2 R c + c 2^-(Fq+1)) + 2               (R rounded up at 2^-16 q-units; c rounded up to 25 bits)
  sampling_v3 uses j = floor(fl64(fl64(m - z) c)) and Delta = ceil(2 R c) + 2.  The circuit's j differs from the exact
  real j by less than c 2^-(Fq+1) table steps (midpoint rounding), which is added to the band, so completeness
  (theorem 06-6.8) holds with the same proof; the accepted set is a superset of sampling_v3's on the same (z, R)
  whenever Delta matches, and Delta is one step wider in a fraction ~ c 2^-(Fq+1) of steps (measured, see tests).
  Everything after j (exp table, top-p threshold form, nucleus bounds, buckets, LN, Gumbel, PRF, inverse CDF) is
  bit-exact with sampling_v3.

Domain (completeness) conditions, all public profile conditions: kq = -e_y - Fq in [0, 32]; logit spread
< 2^(QL + 16 - Fq) logits; Delta <= DMAX = 2^15 (R <= ~2^14 / c logits; also R < 2^(RQ - Fq) logits); |Y| < 2^YB with YB <= 60.

Layout (K steps, s-axis MSB; vocabulary in at most two power-of-two pieces of 32-token buckets):
    VS_p (s, b, k)    per vocabulary slot of piece p       KS (s, k5)   in-bucket one-hots / prefix sums
    BS_p (s, b)       per bucket of piece p                SS (s, z)    per-step scalars (>= 32 rows)
    GS   (s, g13)     all bucket slots (one Gumbel lookup part: the 2^24-row table is shared by every step)
    PS   (s, c)       Poseidon2 PRF rows (call c: BKT c < nb, INV c = nb)    AB (r3): kcom + ctx absorption rows
    TS   (s, i4)      per-step interval table of the nucleus classification (weighted LogUp, commit round)

usage:
  python -m flashprove_next.sampler_zk census --V 151936 --K 2,4,128 [--top-p 0.95 --T 0.6]     (cells, no proving)
  python -m flashprove_next.sampler_zk prove  --V 151936 --K 2 --backend cupy --out FILE        (synthetic steps)
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import sampling_v3 as S3
from . import poseidon2 as PS
from . import certified as CZ
from . import gadgets as G
from . import zk_flat as Z
from .zk_flat import Ex, View, SelLt, IndexP
from .field import P, CPU
from .certified import X

VERSION = "fpn-sampzk-1"
LABEL = b"fpn-sampzk-v1"
QL = 20                      # TJ table: q < 2^QL
DMAX = 1 << 15               # band cap (domain): Delta <= DMAX  (2 R c <= ~2^15 table steps)
BIG = 1 << 18                # j offset of entries with q >= 2^QL (and of masked entries): weight 0
JB = 19                      # j < BIG + JC < 2^JB
KQMAX = 32                   # r has two hidden-width 16-bit limbs (BatchedHead MB=20 + efloor keeps kq in [0, 32])
RQ = 15                      # domain: Rmax < 2^(kq + RQ)  (R < 2^(RQ - Fq) logits)
TROWS = 16                   # interval-table rows per step (<= 13 used)
DOM_KCOM = S3._dom(b"fpn/kprf/commit/v1")
M24 = (1 << 24) - 1
G_TAIL_BITS = 16             # exact Gumbel tail lookup on idx < 2^16 and idx >= 2^24 - 2^16 (131,072 rows)
G_BLK_BITS = 8               # 256 indices per interior piecewise-linear block (65,536 blocks)
G_ERR = 16                   # proven max |G[idx] - ghat(idx)| <= 16 units of 2^-24 (9.54e-7 nats) over all 2^24 indices


# ============================================================================ parameters
@dataclass
class Prm:
    V: int
    rule: S3.Rule
    K: int = 2                       # steps in the proof (power of two; pad by repeating steps)
    YB: int = 60                     # |Y| < 2^YB (HEAD centre bits)
    piecewise_gumbel: bool = True    # 65k-block linear + 131k exact tail (~100x smaller than 2^24 Gumbel table)

    def __post_init__(self):
        assert self.K >= 1 and self.K & (self.K - 1) == 0, "K must be a power of two >= 1 (pad by repeating steps)"
        assert self.YB <= 60
        self.g_err = G_ERR if self.piecewise_gumbel else 0
        r = self.rule
        assert r.beta == 32, "the circuit is laid out for beta = 32"
        self.sb = self.K.bit_length() - 1
        self.B = r.n_buckets(self.V)
        self.nb = r.n_bkt_calls(self.V)            # BKT calls; INV call index = nb
        self.NC = self.nb + 1
        self.cb = max(1, (self.NC - 1).bit_length())
        self.gb = self.cb + 4                       # bucket slots in GS
        # vocabulary pieces: largest power of two <= B, then the next power of two >= the rest
        pb = 1 << (self.B.bit_length() - 1)
        self.pieces = [(0, pb.bit_length() - 1)]
        rest = self.B - pb
        if rest > 0:
            self.pieces.append((pb, max(0, (rest - 1).bit_length())))
        assert all(off % (1 << bb) == 0 for off, bb in self.pieces)
        assert self.pieces[-1][0] + (1 << self.pieces[-1][1]) <= (1 << self.gb)
        self.NS = sum(1 << (bb + 5) for _, bb in self.pieces)          # vocabulary slots
        self.c = r.c
        self.cF = Fraction(r.c)
        self.J = S3.exp_J()
        self.JC = self.J + DMAX
        # Fq: largest F with floor((2^(QL+1) - 1) c / 2^(F+1)) >= JC  (every j <= JC reachable below 2^QL)
        F = 40
        while math.floor((2 ** (QL + 1) - 1) * self.cF / 2 ** (F + 1)) < self.JC:
            F -= 1
        self.Fq = F
        self.EQ = self.cF / 2 ** (F + 1)                               # half q-unit in table steps
        lc = math.floor(math.log2(r.c))
        self.SCc = 24 - lc                                             # Cup = ceil(c 2^SCc) ~ 25 bits
        self.Cup = math.ceil(self.cF * 2 ** self.SCc)
        self.SD = self.Fq + 16 + self.SCc                              # Delta = ceil(N / 2^SD) + 2
        self.EQn = math.ceil(self.EQ * 2 ** self.SD)
        self.MAXJ = BIG + self.JC
        assert self.MAXJ < (1 << JB) and BIG - DMAX > self.JC + DMAX
        self.top = r.p16 >= (1 << S3.P_BITS)                           # top-p = 1: no nucleus machinery

    def as_dict(self):
        return {"V": self.V, "K": self.K, "rule": self.rule.as_dict(), "B": self.B, "nb": self.nb,
                "pieces": self.pieces, "slots": self.NS, "Fq": self.Fq, "QL": QL, "DMAX": DMAX, "BIG": BIG,
                "EQ_steps": float(self.EQ), "YB": self.YB, "J": self.J, "JC": self.JC,
                "piecewise_gumbel": self.piecewise_gumbel, "g_err": self.g_err}


# ============================================================================ public tables
_T = {}


def _tab(key, fn):
    if key not in _T:
        _T[key] = fn()
    return _T[key]


def te_table(prm):
    """TE = {(x, E'(x))}, E'(x) = E[clip(x, 0, J)], x in [-DMAX-1, JC+DMAX] u [BIG-DMAX, BIG+JC+DMAX]; the row at
    -DMAX-1 is a sentinel (E[0] + 1) used only by the plateau-start check of the scalar logic."""
    def mk():
        E = S3.exp_table()
        J = prm.J
        x1 = np.arange(-DMAX, prm.JC + DMAX + 1, dtype=np.int64)
        x2 = np.arange(BIG - DMAX, BIG + prm.JC + DMAX + 1, dtype=np.int64)
        x = np.concatenate([[-DMAX - 1], x1, x2])
        e = E[np.clip(x, 0, J)].astype(np.int64)
        e[0] = int(E[0]) + 1
        return [x, e]
    return _tab(("TE", prm.JC), mk)


def te_val(prm, x):
    E = S3.exp_table()
    return E[np.clip(np.asarray(x, dtype=np.int64), 0, prm.J)].astype(np.int64)


def tj_table(prm):
    """TJ = {(q, min(floor((2q+1) c / 2^(Fq+1)), JC)) : q < 2^QL} (exact rational arithmetic)."""
    def mk():
        n, d = prm.cF.numerator, prm.cF.denominator * (1 << (prm.Fq + 1))
        q = np.arange(1 << QL, dtype=np.int64)
        num = (2 * q + 1).astype(object) * n
        j = np.array([int(v) // d for v in num], dtype=np.int64)
        return [q, np.minimum(j, prm.JC)]
    return _tab(("TJ", prm.rule.c, prm.Fq), mk)


def rw_table(flag=False):
    """RW = {(w, x) : 0 <= w <= 16, x < 2^w}  (flag: third column [x > 0])."""
    def mk():
        ws, xs = [], []
        for w in range(17):
            ws.append(np.full(1 << w, w, dtype=np.int64)); xs.append(np.arange(1 << w, dtype=np.int64))
        w, x = np.concatenate(ws), np.concatenate(xs)
        return [w, x, (x > 0).astype(np.int64)] if flag else [w, x]
    return _tab(("RW", flag), mk)


def r19_table():
    return _tab("R19", lambda: [np.arange(1 << JB, dtype=np.int64)])


def kq_table():
    """(kq, 2^kq, w0, w1, v0, v1, v2, A16, B16) rows kq in [0, KQMAX]: r widths, 2^kq, Rmax limb widths
    (kq + RQ split 16/16/16) and (A16, B16) = (2^max(16-kq,0), 2^max(kq-16,0))."""
    def mk():
        k = np.arange(KQMAX + 1, dtype=np.int64)
        w0 = np.minimum(k, 16); w1 = k - w0
        t = k + RQ
        v0 = np.minimum(t, 16); v1 = np.minimum(t - v0, 16); v2 = t - v0 - v1
        A16 = np.left_shift(1, np.maximum(16 - k, 0)); B16 = np.left_shift(1, np.maximum(k - 16, 0))
        return [k, np.left_shift(1, k), w0, w1, v0, v1, v2, A16, B16]
    return _tab("KQ", mk)


def wq_of(prm, kq):
    # Cap kq + QL + wq <= 61 so r + 2^kq * (qlo + 2^QL * qhi) < 2^61 + 2^40 < p/2 (zero modular wrap in dec.{p})
    return np.minimum(16, np.maximum(0, min(prm.YB + 1, 61) - np.asarray(kq) - QL))


LN_EMAX = 46


def lne_table():
    """(e, A', B', w0, w1): W 2^max(15-e,0) = m 2^max(e-15,0) + rem, rem < 2^max(e-15,0) split 16 + rest."""
    def mk():
        e = np.arange(LN_EMAX + 1, dtype=np.int64)
        sh = np.maximum(e - 15, 0)
        A = np.left_shift(1, np.maximum(15 - e, 0)); B = np.left_shift(1, sh)
        w0 = np.minimum(sh, 16); w1 = sh - w0
        return [e, A, B, w0, w1]
    return _tab("LNE", mk)


def lnt_table():
    LNT, LN2 = S3.ln_table()
    return _tab("LNT", lambda: [np.arange(1 << 15, 1 << 16, dtype=np.int64), np.asarray(LNT, dtype=np.int64)])


def g_table():
    return _tab("G", lambda: [np.arange(1 << S3.G_BITS, dtype=np.int64), S3.gumbel_table().astype(np.int64)])


def gtail_table():
    """Exact Gumbel tail rows: idx in [0, 2^16) u [2^24 - 2^16, 2^24) -> (idx, G[idx]) (131,072 rows)."""
    def mk():
        G = S3.gumbel_table().astype(np.int64)
        n = 1 << S3.G_BITS
        tl = 1 << G_TAIL_BITS
        idx = np.concatenate([np.arange(tl, dtype=np.int64), np.arange(n - tl, n, dtype=np.int64)])
        return [idx, G[idx]]
    return _tab("GTAIL", mk)


def gmid_table():
    """65,536-row block table (blk, base, slope16, istail) for idx = 256 * blk + off:
    on interior blocks [256, 65280), ghat(idx) = base + ((slope16 * off) >> 16) approximates G[idx] within +-16."""
    def mk():
        G = S3.gumbel_table().astype(np.int64)
        B = 1 << G_BLK_BITS
        nb = (1 << S3.G_BITS) // B
        tb = (1 << G_TAIL_BITS) // B
        blks = np.arange(nb, dtype=np.int64)
        bases = np.zeros(nb, dtype=np.int64)
        slopes = np.zeros(nb, dtype=np.int64)
        istail = np.ones(nb, dtype=np.int64)
        offs = np.arange(B, dtype=np.int64)
        for b in range(tb, nb - tb):
            seg = G[b * B:(b + 1) * B]
            s16 = int(round((int(seg[-1]) - int(seg[0])) * 65536.0 / (B - 1)))
            pred0 = (s16 * offs) >> 16
            res = seg - pred0
            bases[b] = (int(res.max()) + int(res.min())) // 2
            slopes[b] = s16
            istail[b] = 0
        return [blks, bases, slopes, istail]
    return _tab("GMID", mk)


def gumbel_approx(idx):
    """Evaluate the circuit's piecewise-linear + exact-tail Gumbel table at integer indices idx in [0, 2^24)."""
    idx = np.asarray(idx, dtype=np.int64)
    G = S3.gumbel_table().astype(np.int64)
    _, bases, slopes, istail = gmid_table()
    blk = idx >> G_BLK_BITS
    off = idx & ((1 << G_BLK_BITS) - 1)
    tl = istail[blk].astype(bool)
    lin = bases[blk] + ((slopes[blk] * off) >> 16)
    return np.where(tl, G[idx], lin)


def _head_rmax_from_A(em, Amax, ey, head_d, tag="sz.hd"):
    """Scalar certified fp32 LM-head radius upper bound from Amax = max_{a_v} (W_head^r Z)_v and ey on SS:
    since |Y_v| <= A_v <= Amax for every v, R1_tilde(Amax) >= R1_v for every allowed v (within < 1/d ~ 0.1%)."""
    R0g = CZ.mul_up(em, Amax, CZ.C0.gamma(int(head_d)), tag + ".g")
    eu0 = CZ.eta_units(em, ey, tag + ".e0")
    R0 = CZ.capped(em, em.lin([(1, R0g), (1, eu0)], bits=61), 56, tag + ".R0")
    st = CZ.shr_pub(em, em.lin([(1, Amax), (1, R0)], bits=61), 24, tag + ".st", up_=True)
    eu1 = CZ.eta_units(em, ey, tag + ".e1")
    return CZ.capped(em, em.lin([(1, R0), (1, st), (1, eu1)], bits=58), 59, tag + ".R1")


def head_rmax_val(Amax, ey, head_d):
    """Exact integer evaluator of _head_rmax_from_A on a scalar (Amax, ey)."""
    e0 = CZ.Em(emit=False, prove=True)
    sc = e0.space("sc", [])
    ax = e0.col(sc, "A", np.full(sc.n, int(Amax), dtype=np.int64), 60)
    ex = e0.col(sc, "e", np.full(sc.n, int(ey), dtype=np.int64), 10)
    r = _head_rmax_from_A(e0, ax, ex, head_d)
    return int(r.v[0]), dict(e0.fails)


# ============================================================================ reference predicate (independent of the circuit code)
def step_j(prm, Y, R, ey, a, head_d=None):
    """Integer step 1.  When head_d is given, R holds per-slot A_v = (W_head^r Z)_v and Rmax = head_rmax_val(max_a A, ey, head_d).
    Returns dict(M, kq, d, q, j, jc, Rmax, Amax, rho, Delta) or raises ValueError on a domain violation."""
    Y = np.asarray(Y, dtype=np.int64); R = np.asarray(R, dtype=np.int64); a = np.asarray(a, dtype=bool)
    if not a.any():
        if Z.FORCE:
            a = a.copy(); a[0] = True
        else:
            raise S3.SamplingError("empty allowed set")
    M = int(Y[a].max())
    kq = -int(ey) - prm.Fq
    if not (0 <= kq <= KQMAX):
        if Z.FORCE:
            kq = int(np.clip(kq, 0, KQMAX))
        else:
            raise ValueError(f"domain: kq = {kq} outside [0, {KQMAX}] (e_y = {ey})")
    d = np.where(a, np.maximum(0, M - Y) if Z.FORCE else (M - Y), 0)
    q = d >> kq
    qlo = q & ((1 << QL) - 1)
    qhi = q >> QL
    wq = int(wq_of(prm, kq))
    if np.any(qhi >= (1 << wq)):
        if Z.FORCE:
            qhi = np.minimum(qhi, max(0, (1 << wq) - 1))
        else:
            raise ValueError("domain: logit spread exceeds 2^(QL + wq - Fq) logits")
    TJ = tj_table(prm)[1]
    jlo = np.where(a, TJ[qlo], 0)
    f = np.where(a, (qhi > 0), True).astype(np.int64)
    j = jlo + BIG * f
    jc = np.where(f == 1, prm.JC, jlo)
    Amax = int(R[a].max())
    if head_d is not None:
        Rmax, _ = head_rmax_val(max(0, Amax) if Z.FORCE else Amax, ey, head_d)
    else:
        Rmax = Amax
    if Rmax >= (1 << (kq + RQ)) or Rmax < 0 or (head_d is not None and (Amax < 0 or Amax >= (1 << 60))):
        if Z.FORCE:
            Amax = int(np.clip(Amax, 0, (1 << 60) - 1))
            Rmax = int(np.clip(Rmax, 0, (1 << (kq + RQ)) - 1))
        else:
            raise ValueError("domain: radius too large")
    A16 = 1 << max(16 - kq, 0); B16 = 1 << max(kq - 16, 0)
    rho = (Rmax * A16 + B16 - 1) // B16
    N = 2 * rho * prm.Cup + prm.EQn
    D0 = (N + (1 << prm.SD) - 1) >> prm.SD
    Delta = D0 + 2
    if Delta > DMAX:
        if Z.FORCE:
            Delta = DMAX; D0 = DMAX - 2
        else:
            raise ValueError(f"domain: Delta = {Delta} > DMAX")
    return dict(M=M, kq=kq, d=d, q=q, qlo=qlo, qhi=qhi, wq=wq, jlo=jlo, f=f, j=j, jc=jc, Rmax=Rmax, Amax=Amax,
                rho=rho, N=N, D0=D0, Delta=Delta, A16=A16, B16=B16)


def weights(prm, j, Delta):
    return te_val(prm, np.asarray(j) + Delta), te_val(prm, np.asarray(j) - Delta)


def accept_from_weights(w_lo, w_hi, outs, rule, V, a, piecewise_gumbel=False):
    """specs/06 §3.2 relaxed acceptance from (w_lo, w_hi) and the step's PRF outputs (same code as
    sampling_v3.acceptable_set after its weight computation; piecewise_gumbel adds 2 * G_ERR = 32 units of 2^-24
    to cover the +-16 Gumbel table approximation).  Returns (accept (V,) bool, info)."""
    if rule.p16 >= (1 << S3.P_BITS):
        in_lo, in_hi = w_lo > 0, w_hi > 0
    else:
        in_lo, in_hi = S3.nucleus_bounds_tight(w_lo, w_hi, rule.p16)
    n_lo = np.where(in_lo & (w_lo > 0), w_lo, 0)
    n_hi = np.where(in_hi & (w_hi > 0), w_hi, 0)
    blo, Wlo = S3.bucket_masses(n_lo, rule.beta)
    bhi, Whi = S3.bucket_masses(n_hi, rule.beta)
    idx, u48 = S3.noise_indices(outs, rule.n_buckets(V))
    u48 = int(u48)
    gv = gumbel_approx(idx) if piecewise_gumbel else S3.gumbel_table().astype(np.int64)[idx]
    g_slack = 2 * G_ERR if piecewise_gumbel else 0
    lo_keys = np.where(Wlo > 0, S3.ln_fx(np.maximum(Wlo, 1)) + gv, np.iinfo(np.int64).min)
    hi_keys = np.where(Whi > 0, S3.ln_fx(np.maximum(Whi, 1)) + gv, np.iinfo(np.int64).min)
    rhs = lo_keys.max()
    acc = np.zeros(rule.n_buckets(V) * rule.beta, dtype=bool)
    passing = [int(b) for b in np.nonzero((Whi > 0) & (hi_keys + g_slack >= rhs))[0]]
    for b in passing:
        Shi = np.cumsum(bhi[b])
        Slo_prev = np.concatenate([[0], np.cumsum(blo[b])[:-1]])
        WH, WL = int(Whi[b]), int(Wlo[b])
        for k in range(rule.beta):
            if bhi[b, k] <= 0:
                continue
            acc[b * rule.beta + k] = (int(Slo_prev[k]) << S3.U_BITS <= u48 * WH) and (u48 * WL < int(Shi[k]) << S3.U_BITS)
    acc = acc[:V] & np.asarray(a, dtype=bool)[:V]
    return acc, dict(in_lo=in_lo, in_hi=in_hi, n_lo=n_lo, n_hi=n_hi, Wlo=Wlo, Whi=Whi, idx=idx, u48=u48,
                     passing=passing)


def ref_accept(prm, Y, R, ey, a, outs, head_d=None):
    """The circuit's relation on one step, evaluated directly: accepted-token mask over the V real tokens."""
    st = step_j(prm, Y[:prm.V], R[:prm.V], ey, a[:prm.V], head_d=head_d)
    wlo, whi = weights(prm, st["jc"], st["Delta"])
    wlo = np.where(a[:prm.V], wlo, 0); whi = np.where(a[:prm.V], whi, 0)
    acc, info = accept_from_weights(wlo, whi, outs, prm.rule, prm.V, a[:prm.V], piecewise_gumbel=prm.piecewise_gumbel)
    info.update(st)
    return acc, info


def v3_accept(prm, Y, R, ey, a, prefix, d, t):
    """sampling_v3.acceptable_set on z = Y 2^e_y (float64), r = max_a R 2^e_y."""
    z = np.ldexp(np.asarray(Y[:prm.V], dtype=np.float64), int(ey))
    r = float(np.ldexp(float(np.asarray(R[:prm.V])[np.asarray(a[:prm.V], bool)].max()), int(ey)))
    rel = S3.acceptable_set(prefix, t, z, r, prm.rule, allowed=np.asarray(a[:prm.V], bool), d_t=d)
    return rel


# ============================================================================ per-step witness
def kcom_of(kprf, salt):
    blk = np.zeros((8, 1), dtype=np.uint64)
    blk[:4, 0] = [int(x) % P for x in salt]; blk[4:, 0] = S3.kprf_elems(kprf)
    return [int(x) for x in PS.sponge(blk[None], domain=DOM_KCOM)[:, 0]]


def prf_inputs(prm, prefix, d, t):
    """(12, 2^cb) inputs of the step's PRF rows (BKT c < nb, INV c = nb, unused rows: zero input)."""
    n = 1 << prm.cb
    st = np.zeros((12, n), dtype=np.uint64)
    c = np.arange(prm.NC)
    st[:, :prm.NC] = np.asarray(prefix, dtype=np.uint64)[:, None]
    add = [np.full(prm.NC, int(d) % P), np.full(prm.NC, int(t) % P),
           np.where(c == prm.nb, S3.LBL_INV, S3.LBL_BKT), np.where(c == prm.nb, 0, c)]
    for i in range(4):
        st[i, :prm.NC] = S3._addp(st[i, :prm.NC], np.asarray(add[i], dtype=np.uint64))
    return st


def pad_slots(prm, x, fill=0, dtype=np.int64):
    """real vocabulary vector (V,) -> slot vector (NS,) in piece order (slot = global vocabulary index)."""
    out = np.full(prm.NS, fill, dtype=dtype)
    x = np.asarray(x)
    n = min(len(x), prm.NS)
    out[:n] = x[:n]
    return out


def _inv_field(x):
    """field inverses of a uint64 array (0 -> 0)."""
    x = np.asarray(x, dtype=np.uint64)
    if x.size == 0:
        return x
    return np.ascontiguousarray(CPU.einv(CPU.ext(x))[0])


def _ln_parts(W):
    """LN decomposition of W >= 0 (int64 array): (e, A, B, w0, w1, m, rem0, rem1, LNv, z) with dummies for W = 0."""
    W = np.asarray(W, dtype=np.int64)
    z = (W > 0).astype(np.int64)
    e = np.where(z == 1, S3.bitlen_floor_log2(np.maximum(W, 1)), 0)
    sh = np.maximum(e - 15, 0)
    A = np.left_shift(1, np.maximum(15 - e, 0)); B = np.left_shift(1, sh)
    WA = W * A
    m = np.where(z == 1, WA >> sh, 1 << 15)
    rem = np.where(z == 1, WA - (m << sh), 0)
    w0 = np.minimum(sh, 16); w1 = sh - w0
    LNT, LN2 = S3.ln_table()
    LNv = np.asarray(LNT, dtype=np.int64)[m - (1 << 15)]
    return dict(e=e, A=A, B=B, w0=w0, w1=w1, m=m, rem0=rem & 0xFFFF, rem1=rem >> 16, LNv=LNv, z=z,
                LN=e * int(LN2) + LNv)


def _mulshift_parts(u0, u1, W):
    """tau = floor((u0 + 2^24 u1) W / 2^48) with the limb witness of the circuit."""
    W0, W1 = W & M24, W >> 24
    L0, L1, L2 = u0 * W0, u0 * W1 + u1 * W0, u1 * W1
    L1l, L1h = L1 & M24, L1 >> 24
    s = L0 + (L1l << 24)
    cb, rho = s >> 48, s & ((1 << 48) - 1)
    tau = L2 + L1h + cb
    assert tau == ((u0 + (u1 << 24)) * W) >> 48
    return dict(W0=W0, W1=W1, L0=L0, L1=L1, L2=L2, L1l=L1l, L1h=L1h, cb=cb, rho=rho, tau=tau)


def _wide_parts(q, P16, A, R):
    """beta = [P16 R - q A >= 1] with the two-limb witness (A, R < 2^60)."""
    A0, A1 = A & ((1 << 30) - 1), A >> 30
    R0, R1 = R & ((1 << 30) - 1), R >> 30
    Dh, Dl = P16 * R1 - q * A1, P16 * R0 - q * A0
    D = Dh * (1 << 30) + Dl
    beta = int(D >= 1)
    sg = 2 * beta - 1
    Eh, El = sg * Dh, sg * Dl - beta
    cc, e0 = El >> 30, El & ((1 << 30) - 1)
    assert Eh + cc >= 0
    return dict(A0=A0, A1=A1, R0=R0, R1=R1, beta=beta, cc=cc, e0=e0, top=Eh + cc)


def wit_step(prm, Y, R, ey, a, prefix, d, t, tok, head_d=None):
    """Every value the circuit commits for one step (prover side).  Y, R, a: slot vectors (NS,); when head_d is given,
    R holds per-slot A_v = (W_head^r Z)_v.  Raises ValueError / SamplingError on a domain violation; returns (w, ok)."""
    V, rule = prm.V, prm.rule
    a = np.asarray(a, dtype=bool).copy()
    a[V:] = False
    if Z.FORCE and not a[:V].any():
        a[0] = True
    Y = np.asarray(Y, dtype=np.int64); R = np.asarray(R, dtype=np.int64)
    st = step_j(prm, Y, R, ey, a, head_d=head_d)
    jc = st["jc"]
    wlo, whi = weights(prm, jc, st["Delta"])
    wlo = np.where(a, wlo, 0); whi = np.where(a, whi, 0)
    outs = PS.permute(prf_inputs(prm, prefix, d, t))                 # (12, 2^cb)
    outs_r = outs[:8].T                                                # (rows, 8)
    acc, info = accept_from_weights(wlo[:V], whi[:V], outs_r[:prm.NC][None, ...][0], rule, V, a[:V],
                                    piecewise_gumbel=prm.piecewise_gumbel)
    w = dict(st)
    w.update(a=a.astype(np.int64), Y=Y, R=R, ey=int(ey), wlo=wlo, whi=whi, prf_in=prf_inputs(prm, prefix, d, t),
             prf_out=outs)
    # ---- one-hots: argmax of Y, argmax of R / A (over allowed), output token
    yh = np.flatnonzero(a & (Y == st["M"]))
    rh = np.flatnonzero(a & (R == st["Amax"]))
    vY = int(yh[0]) if len(yh) else 0
    vR = int(rh[0]) if len(rh) else 0
    tok_c = int(np.clip(tok, 0, V - 1)) if Z.FORCE else int(tok)
    w.update(vY=vY, vR=vR, tok=tok_c)
    # ---- r / q decomposition of allowed entries
    kq = st["kq"]
    r = np.where(a, st["d"] & ((1 << kq) - 1), 0)
    w.update(r0=r & 0xFFFF, r1=r >> 16, qlo=np.where(a, st["qlo"], 0), qhi=np.where(a, st["qhi"], 0),
             jlo=st["jlo"], f=st["f"], j=st["j"])
    KQ = kq_table()
    w.update(B=int(KQ[1][kq]), w0=int(KQ[2][kq]), w1=int(KQ[3][kq]), v0=int(KQ[4][kq]), v1=int(KQ[5][kq]),
             v2=int(KQ[6][kq]))
    Rm = st["Rmax"]
    w.update(Rm0=Rm & 0xFFFF, Rm1=(Rm >> 16) & 0xFFFF, Rm2=Rm >> 32)
    rem16 = max(0, Rm * st["A16"] + st["B16"] - 1 - st["rho"] * st["B16"])
    remS = max(0, st["N"] + (1 << prm.SD) - 1 - (st["D0"] << prm.SD))
    w.update(rem16=rem16, remS=remS)
    # ---- nucleus classification
    j = st["j"]
    if prm.top:
        sv = (wlo > 0).astype(np.int64); pv = (whi > 0).astype(np.int64)
    else:
        pad = lambda m: np.concatenate([m, np.zeros(prm.NS - V, dtype=bool)])
        sv = pad(info["in_lo"]).astype(np.int64) & a; pv = pad(info["in_hi"]).astype(np.int64) & a
        nw = _nucleus(prm, j, wlo, whi, a, sv, pv, st["Delta"])
        w.update(nw)
        sv = nw["nuc"]["Sb"][nw["row"]].astype(np.int64)            # per-slot value = the slot's table row bit
        pv = nw["nuc"]["Pb"][nw["row"]].astype(np.int64)
        if not Z.FORCE:
            assert np.array_equal(sv * a, sv) and np.array_equal(pv * a, pv)
    w.update(sv=sv, pv=pv)
    nlo, nhi = sv * wlo, pv * whi
    w.update(nlo=nlo, nhi=nhi)
    # ---- buckets (slot order = global bucket order)
    Wlo = nlo.reshape(-1, 32).sum(axis=1); Whi = nhi.reshape(-1, 32).sum(axis=1)
    w.update(Wlo=Wlo, Whi=Whi, ln=_ln_parts(Wlo))
    # Gumbel indices / values for all 2^gb bucket slots
    e = outs[:8].astype(np.uint64)                                     # (8, rows)
    lo24 = (e & np.uint64(M24)).astype(np.int64); hi24 = ((e >> np.uint64(24)) & np.uint64(M24)).astype(np.int64)
    idx = np.stack([lo24.T, hi24.T], axis=-1).reshape(-1)              # slot g = 16 c + 2 i + h
    Gt = S3.gumbel_table().astype(np.int64)
    if prm.piecewise_gumbel:
        _, bases, slopes, istail = gmid_table()
        gblk = idx >> G_BLK_BITS
        goff = idx & ((1 << G_BLK_BITS) - 1)
        gtail = istail[gblk]
        gbase = bases[gblk]
        gslope = slopes[gblk]
        gprod = gslope * goff
        gquo = gprod >> 16
        grem = gprod & 0xFFFF
        gtail_val = np.where(gtail == 1, Gt[idx], 0)
        gval = np.where(gtail == 1, gtail_val, gbase + gquo)
        w.update(gblk=gblk, goff=goff, gtail=gtail, gbase=gbase, gslope=gslope,
                 gquo=gquo, grem=grem, gtail_val=gtail_val)
    else:
        gval = Gt[idx]
    w.update(gidx=idx, gval=gval, lo24=lo24, hi24=hi24)
    nbs = prm.NS // 32
    assert np.array_equal(idx[:prm.B], info["idx"][:prm.B])
    # ---- output token
    bt, kt = tok_c // 32, tok_c % 32
    Xlo, Xhi = nlo[bt * 32:(bt + 1) * 32], nhi[bt * 32:(bt + 1) * 32]
    Plo, Phi = np.cumsum(Xlo), np.cumsum(Xhi)
    Slo_prev, Shi_u, Xhi_u = int(Plo[kt] - Xlo[kt]), int(Phi[kt]), int(Xhi[kt])
    Wlo_s, Whi_s = int(Wlo[bt]), int(Whi[bt])
    lnh = _ln_parts(np.array([Whi_s]))
    Gs_val = int(gval[bt]) if bt < len(gval) else 0
    Kstar = int(lnh["LN"][0]) + Gs_val
    u0 = int(lo24[0, prm.nb]); u1 = int(hi24[0, prm.nb])
    w.update(bt=bt, kt=kt, Xlo=Xlo, Xhi=Xhi, Plo=Plo, Phi=Phi, Slo_prev=Slo_prev, Shi_u=Shi_u, Xhi_u=Xhi_u,
             Wlo_s=Wlo_s, Whi_s=Whi_s, lnh=lnh, Gs=Gs_val, Kstar=Kstar, u0=u0, u1=u1,
             tau_hi=_mulshift_parts(u0, u1, Whi_s), tau_lo=_mulshift_parts(u0, u1, Wlo_s))
    assert u0 + (u1 << 24) == info["u48"]
    keylo = w["ln"]["LN"] + w["gval"][:nbs]
    w["keylo"] = keylo
    ok = bool(0 <= tok < V and acc[int(tok)])
    # circuit-side recomputation of the verdict (must equal the reference)
    ok2 = (Xhi_u >= 1 and Slo_prev <= w["tau_hi"]["tau"] and w["tau_lo"]["tau"] < Shi_u and a[tok_c]
           and bool(np.all((w["ln"]["z"] == 0) | (keylo <= Kstar + 2 * prm.g_err))))
    if not Z.FORCE:
        assert ok == ok2 or not (0 <= tok < V), (ok, ok2)
    w["acc"] = acc
    return w, ok


def _nucleus(prm, j, wlo, whi, a, sv, pv, Delta):
    """Boundary candidates, breakpoints, per-entry interval assignment, per-row table and scalar witnesses."""
    J = prm.J
    al = a.astype(bool)
    jj = j.astype(np.int64)
    sure_j = jj[al & (sv == 1)]; nons_j = jj[al & (sv == 0)]
    poss_j = jj[al & (pv == 1)]; nonp_j = jj[al & (pv == 0)]
    # prefix property (Lemma 6.7'')
    if len(sure_j) and len(nons_j):
        assert sure_j.max() < nons_j.min(), "sure set is not a j-prefix"
    if len(poss_j) and len(nonp_j):
        assert poss_j.max() < nonp_j.min(), "possible set is not a j-prefix"
    es, fs = int(len(sure_j) == 0), int(len(nons_j) == 0)
    ep, fp = int(len(poss_j) == 0), int(len(nonp_j) == 0)
    ls = -1 if es else int(sure_j.max()); lsp = BIG if fs else int(nons_j.min())
    lp = -1 if ep else int(poss_j.max()); lpp = BIG if fp else int(nonp_j.min())
    cands = {}
    for nm, ell, typ in (("s", ls, "sure"), ("sp", lsp, "sure"), ("p", lp, "poss"), ("pp", lpp, "poss")):
        xlo, xhi = ell + Delta, ell - Delta
        wl, wh = int(te_val(prm, xlo)), int(te_val(prm, xhi))
        x = xlo if typ == "sure" else xhi
        ps = _ps_exact(prm, x)
        L = ps + Delta if typ == "sure" else ps - Delta
        cands[nm] = dict(ell=ell, wl=wl, wh=wh, x=x, ps=ps, L=L, typ=typ,
                         psm1=int(te_table(prm)[1][_te_row(prm, ps - 1)]), Eps=int(te_table(prm)[1][_te_row(prm, ps)]))
    bps = [ls, ls + 1, lsp, lsp + 1, lp, lp + 1, lpp, lpp + 1,
           cands["s"]["L"], cands["sp"]["L"], cands["p"]["L"], cands["pp"]["L"]]
    cuts = sorted(set(b for b in bps if 0 < b <= prm.MAXJ))
    LO = [0] + cuts
    HI = [c - 1 for c in cuts] + [prm.MAXJ]
    assert len(LO) <= TROWS
    while len(LO) < TROWS:
        LO.append(prm.MAXJ + 1); HI.append(prm.MAXJ)
    LO, HI = np.array(LO, dtype=np.int64), np.array(HI, dtype=np.int64)
    # bits[i, x] = [HI_i < X_x] (with LO_i >= X_x otherwise)
    Xs = np.array(bps, dtype=np.int64)
    bits = (HI[:, None] < Xs[None, :]).astype(np.int64)
    for i in range(TROWS):
        for x in range(len(Xs)):
            assert bits[i, x] or LO[i] >= Xs[x] or LO[i] > HI[i], "row straddles a breakpoint"
    Sb, Pb = bits[:, 1], bits[:, 5]                  # [HI < ls + 1], [HI < lp + 1]
    # per-entry row assignment
    row = np.searchsorted(HI[:len(cuts) + 1], jj, side="left")
    assert np.all(LO[row] <= jj) and np.all(jj <= HI[row])
    eLO, eHI = LO[row], HI[row]
    assert np.array_equal(Sb[row] * a, sv) and np.array_equal(Pb[row] * a, pv), "classification mismatch"
    m = np.bincount(row, minlength=TROWS).astype(np.int64)
    Ma = np.bincount(row, weights=a, minlength=TROWS).astype(np.int64)
    Mlo = np.zeros(TROWS, dtype=object); Mhi = np.zeros(TROWS, dtype=object)
    for i in range(TROWS):
        Mlo[i] = int(wlo[row == i].sum()); Mhi[i] = int(whi[row == i].sum())
    Mlo = Mlo.astype(np.int64); Mhi = Mhi.astype(np.int64)
    # scalar checks per candidate
    q16, p16 = (1 << S3.P_BITS) - prm.rule.p16, prm.rule.p16
    xi = {"s": 8, "sp": 9, "p": 10, "pp": 11}
    for nm, cd in cands.items():
        cL = bits[:, xi[nm]]
        lt = int(cd["ell"] < cd["L"])
        if cd["typ"] == "sure":
            SA = int((cL * Mhi).sum()); SR = int(((1 - cL) * Mlo).sum())
            Aq = SA - lt * cd["wh"]; Rq = cd["wl"] + SR - (1 - lt) * cd["wl"]
            zt = int(cd["wl"] > 0)
            wz, wzinv = cd["wl"], cd["wl"]
        else:
            SA = int((cL * Mlo).sum()); SR = int(((1 - cL) * Mhi).sum())
            Aq, Rq = SA, SR
            zt = int(cd["wh"] > 0)
            wz = cd["wh"]
        cd.update(SA=SA, SR=SR, lt=lt, Aq=Aq, Rq=Rq, zt=zt, wz=wz, wide=_wide_parts(q16, p16, Aq, Rq))
        cd["pred"] = cd["zt"] * cd["wide"]["beta"]
    # consistency with nucleus_bounds_tight (non-virtual candidates)
    if not Z.FORCE:
        assert es or cands["s"]["pred"] == 1
        assert fs or cands["sp"]["pred"] == 0
        assert ep or cands["p"]["pred"] == 1
        assert fp or cands["pp"]["pred"] == 0
    att = {}
    for nm, X0, X1 in (("s", 0, 1), ("sp", 2, 3), ("p", 4, 5), ("pp", 6, 7)):
        att[nm] = int(((bits[:, X1] - bits[:, X0]) * Ma).sum())
    if not Z.FORCE:
        assert (es or att["s"] >= 1) and (fs or att["sp"] >= 1) and (ep or att["p"] >= 1) and (fp or att["pp"] >= 1)
    gap_s = int(((1 - bits[:, 1]) * bits[:, 2] * Ma).sum()); gap_p = int(((1 - bits[:, 5]) * bits[:, 6] * Ma).sum())
    if not Z.FORCE:
        assert gap_s == 0 and gap_p == 0
    return dict(nuc=dict(es=es, fs=fs, ep=ep, fp=fp, cands=cands, Xs=Xs, LO=LO, HI=HI, bits=bits, Sb=Sb, Pb=Pb,
                         m=m, Ma=Ma, Mlo=Mlo, Mhi=Mhi, att=att), eLO=eLO, eHI=eHI, row=row)


def _te_row(prm, x):
    """row index of x in te_table."""
    x = int(x)
    if x == -DMAX - 1:
        return 0
    if -DMAX <= x <= prm.JC + DMAX:
        return 1 + x + DMAX
    off = 1 + prm.JC + 2 * DMAX + 1
    assert BIG - DMAX <= x <= BIG + prm.JC + DMAX, x
    return off + x - (BIG - DMAX)


def _ps_exact(prm, x):
    """plateau start in the TE domain: min{y in domain : E'(y) = E'(x)} (y = -DMAX at the top plateau)."""
    xs, es = te_table(prm)
    i = _te_row(prm, x)
    v = es[i]
    # domain rows are sorted by x and E' is non-increasing along them (sentinel row 0 is larger)
    k = i
    lo, hi = 1, i                      # first row in [1, i] with value v
    while lo < hi:
        mid = (lo + hi) // 2
        if es[mid] == v:
            hi = mid
        else:
            lo = mid + 1
    return int(xs[lo])


# ============================================================================ statement / witness assembly
def make_statement(prm, kcom, ctx, ds, ts=None, toks=None):
    return {"v": VERSION, "prm": prm.as_dict(),
            "kcom": None if kcom is None else [int(x) for x in kcom],
            "ctx": [int(x) for x in ctx.encode()],
            "d": [int(x) for x in ds],
            "t": None if ts is None else [int(x) for x in ts],
            "tok": None if toks is None else [int(x) for x in toks]}


# ============================================================================ circuit helpers
def _sp(em, name, axes, minbits=5):
    """padded space with at least 2^minbits rows (unique 'zz.<name>' replication axis) for small per-step spaces."""
    bits = sum(b for _, b in axes)
    if bits < minbits:
        axes = list(axes) + [(f"zz.{name}", minbits - bits)]
    return em.space(name, [(a, b) for a, b in axes if b > 0])


def _rep(sp, vals_per_s, K):
    """per-step values (K, ...) -> rows of space sp whose leading axis is s (replicated over trailing 'z')."""
    v = np.asarray(vals_per_s)
    rows = sp.n
    per = rows // K
    if v.ndim == 1:
        return np.repeat(v.astype(np.int64), per)
    v = v.reshape(K, -1)
    reps = per // v.shape[1]
    return np.repeat(v, reps, axis=1).reshape(-1).astype(np.int64)


def _ax(em, sp, ax, off=0):
    """integer index of axis ax (+ off) on space sp as an X (constant off when ax was 0-bit)."""
    if not sp.has(ax):
        return em.const(sp, off)
    v = (sp.grid(ax) + off) if em.pv else None
    ex = sp.S.idx(ax) + Ex.const(off) if em.emit else None
    return X(sp, v, max(1, sp.axes[[a for a, _ in sp.axes].index(ax)][1] + 1 + int(off).bit_length()), ex=ex)


def _reduce(v, src, dst):
    """prover: sum v (rows of src) over the axes dst lacks, replicated over dst's own extra axes (e.g. zz)."""
    shared = [(a, b) for a, b in dst.axes if src.has(a)]
    gs = np.zeros(src.n, dtype=np.int64); gd = np.zeros(dst.n, dtype=np.int64)
    for a, b in shared:
        gs = (gs << b) | src.grid(a); gd = (gd << b) | dst.grid(a)
    tot = np.zeros(1 << sum(b for _, b in shared), dtype=np.int64)
    np.add.at(tot, gs, np.asarray(v, dtype=np.int64))
    return tot[gd]


def _sum_to(em, terms, dst, tag, bits):
    """committed S on dst = sum over the axes dst lacks of prod(terms) (zero-sum constraint)."""
    src = terms[0].sp
    zax = next((a for a, _ in src.axes if a.startswith("zz") and not dst.has(a)), None)
    if zax is not None:
        zv = (src.grid(zax) == 0).astype(np.int64) if em.pv else None
        zex = src.S.ind(**{zax: 0}) if em.emit else None
        terms = list(terms) + [X(src, zv, 1, ex=zex)]
    v = None
    if em.pv:
        pr = terms[0].v.astype(np.int64)
        for t_ in terms[1:]:
            pr = pr * t_.v
        v = _reduce(pr, src, dst)
    Sx = em.col(dst, tag, v, bits)
    CZ.dot(em, src, terms, Sx, [a for a, _ in dst.axes], tag)
    return Sx


def _lk(em, tname, sp, xs, sel=None):
    G.wire_lookup(em.ctx, tname, sp.S, [x.ex for x in xs], [x.v for x in xs] if em.pv else None,
                  sel=sel.ex if sel is not None else None, sel_vals=sel.v if (sel is not None and em.pv) else None)


def _rng_sel(em, x, bits, tag, sel):
    vals = None
    if em.pv:
        vals = x.v
        if Z.FORCE:
            vals = np.where((x.v >= 0) & (x.v < (1 << bits)), x.v, 0)
        em.fail("rng:" + tag, (((x.v < 0) | (x.v >= (1 << bits))) & (sel.v != 0)).sum())
    G.range_check(x.sp.S, x.ex, vals, bits, em.nm("r." + tag), sel=sel.ex, sel_vals=sel.v if em.pv else None)


def _ucol(em, sp, name, vals):
    """committed column holding raw field elements (uint64; inverses)."""
    c = sp.S.col(em.nm(name), np.asarray(vals, dtype=np.uint64) if em.pv else None)
    return X(sp, None, 1, col=c, nm=name)


def _cmp_bit(em, sp, c, lo_or_x, hi, Xe, tag, bits=JB + 2):
    """non-straddle comparison on rows: c = 1 => hi <= X - 1, c = 0 => lo >= X (range check of the selected
    difference, bits wide)."""
    v = None
    if em.pv:
        v = c.v * (Xe.v - 1 - hi.v) + (1 - c.v) * (lo_or_x.v - Xe.v)
    ex = (c.ex * (Xe.ex - Ex.const(1) - hi.ex) + (Ex.const(1) - c.ex) * (lo_or_x.ex - Xe.ex)) if em.emit else None
    em.rng(X(sp, v, bits, ex=ex), bits, tag)


def _one_hot_pair(em, BSs, KS, SS, eb_vals, ek_vals, tag):
    """one-hot over (bucket of any piece) x (in-bucket index): booleans, sum over buckets = 1, sum over k = 1."""
    ebs = []
    tot = None
    for p, BSp in enumerate(BSs):
        eb = em.col(BSp, f"{tag}.eb{p}", eb_vals[p] if em.pv else None, 1)
        em.boolean(eb, f"{tag}.eb{p}")
        sm = _sum_to(em, [eb], SS, f"{tag}.ebs{p}", 1)
        tot = sm if tot is None else em.lin([(1, tot), (1, sm)])
        ebs.append(eb)
    em.zero(SS, lambda: tot.ex - Ex.const(1), f"{tag}.eb1")
    ek = em.col(KS, f"{tag}.ek", ek_vals if em.pv else None, 1)
    em.boolean(ek, f"{tag}.ek")
    sk = _sum_to(em, [ek], SS, f"{tag}.eks", 1)
    em.zero(SS, lambda: sk.ex - Ex.const(1), f"{tag}.ek1")
    return ebs, ek


def _sum_pieces(em, VSs, termfn, SS, tag, bits=62):
    """sum over all slots of all pieces of prod(termfn(p)) -> committed per-step scalar (bits: declared bound of
    the TOTAL; callers guarantee it, e.g. one-hot sums)."""
    tot = None
    for p, VSp in enumerate(VSs):
        terms = list(termfn(p))
        zax = next((a for a, _ in VSp.axes if a.startswith("zz") and not SS.has(a)), None)
        if zax is not None:
            zv = (VSp.grid(zax) == 0).astype(np.int64) if em.pv else None
            zex = VSp.S.ind(**{zax: 0}) if em.emit else None
            terms.append(X(VSp, zv, 1, ex=zex))
        v = None
        if em.pv:
            pr = terms[0].v.astype(object) if bits > 62 else terms[0].v
            for t_ in terms[1:]:
                pr = pr * t_.v
            v = _reduce(np.asarray(pr, dtype=np.int64), VSp, SS)
        S_ = em.col(SS, f"{tag}.p{p}", v, bits)
        CZ.dot(em, VSp, terms, S_, [a for a, _ in SS.axes], f"{tag}.p{p}")
        tot = S_ if tot is None else em.lin([(1, tot), (1, S_)], bits=bits)
    return tot


# ---- LN gadget: LN(W) = e LN2 + LNT[m], W A' = m B' + rem (gated by z), on any space
def _ln_gadget(em, sp, W, z, parts, tag):
    LNT, LN2 = S3.ln_table()
    pv = em.pv
    g = lambda k: parts[k] if pv else None
    e = em.col(sp, tag + ".e", g("e"), 7)
    A = em.col(sp, tag + ".A", g("A"), 16)
    Bc = em.col(sp, tag + ".B", g("B"), 32)
    w0 = em.col(sp, tag + ".w0", g("w0"), 5)
    w1 = em.col(sp, tag + ".w1", g("w1"), 5)
    _lk(em, "sz.lne", sp, [e, A, Bc, w0, w1])
    m = em.col(sp, tag + ".m", g("m"), 17)
    LNv = em.col(sp, tag + ".lnv", g("LNv"), 25)
    _lk(em, "sz.lnt", sp, [m, LNv])
    r0 = em.col(sp, tag + ".r0", g("rem0"), 16)
    r1 = em.col(sp, tag + ".r1", g("rem1"), 16)
    _lk(em, "sz.rw", sp, [w0, r0])
    _lk(em, "sz.rw", sp, [w1, r1])
    em.zero(sp, lambda: z.ex * (W.ex * A.ex - m.ex * Bc.ex - r0.ex - Ex.const(1 << 16) * r1.ex), tag + ".dec")
    return em.lin([(int(LN2), e), (1, LNv)], bits=32)


def _mulshift_gadget(em, SS, u0, u1, W, parts, tag):
    """tau = floor((u0 + 2^24 u1) W / 2^48) for W < 2^46 (all on SS)."""
    g = lambda k: parts[k] if em.pv else None
    W0 = em.col(SS, tag + ".W0", g("W0"), 24); em.rng(W0, 24, tag + ".W0")
    W1 = em.col(SS, tag + ".W1", g("W1"), 22); em.rng(W1, 22, tag + ".W1")
    em.zero(SS, lambda: W.ex - W0.ex - Ex.const(1 << 24) * W1.ex, tag + ".Wdec")
    L0 = CZ.mul(em, u0, W0, tag + ".L0", bits=48)
    L1 = em.col(SS, tag + ".L1", g("L1"), 49)
    em.zero(SS, lambda: L1.ex - u0.ex * W1.ex - u1.ex * W0.ex, tag + ".L1d")
    L2 = CZ.mul(em, u1, W1, tag + ".L2", bits=46)
    L1l = em.col(SS, tag + ".L1l", g("L1l"), 24); em.rng(L1l, 24, tag + ".L1l")
    L1h = em.col(SS, tag + ".L1h", g("L1h"), 25); em.rng(L1h, 25, tag + ".L1h")
    em.zero(SS, lambda: L1.ex - L1l.ex - Ex.const(1 << 24) * L1h.ex, tag + ".L1s")
    cb = em.col(SS, tag + ".cb", g("cb"), 1); em.boolean(cb, tag + ".cb")
    rho = em.col(SS, tag + ".rho", g("rho"), 48); em.rng(rho, 48, tag + ".rho")
    em.zero(SS, lambda: L0.ex + Ex.const(1 << 24) * L1l.ex - Ex.const(1 << 48) * cb.ex - rho.ex, tag + ".carry")
    return em.lin([(1, L2), (1, L1h), (1, cb)], bits=48)


def _wide_gadget(em, SS, q16, p16, Aq, Rq, parts, tag):
    """beta = [p16 Rq - q16 Aq >= 1] for 0 <= Aq, Rq < 2^60 (two 30-bit limbs each)."""
    g = lambda k: parts[k] if em.pv else None
    A0 = em.col(SS, tag + ".A0", g("A0"), 30); em.rng(A0, 30, tag + ".A0")
    A1 = em.col(SS, tag + ".A1", g("A1"), 30); em.rng(A1, 30, tag + ".A1")
    R0 = em.col(SS, tag + ".R0", g("R0"), 30); em.rng(R0, 30, tag + ".R0")
    R1 = em.col(SS, tag + ".R1", g("R1"), 30); em.rng(R1, 30, tag + ".R1")
    em.zero(SS, lambda: Aq.ex - A0.ex - Ex.const(1 << 30) * A1.ex, tag + ".Ad")
    em.zero(SS, lambda: Rq.ex - R0.ex - Ex.const(1 << 30) * R1.ex, tag + ".Rd")
    beta = em.col(SS, tag + ".beta", g("beta"), 1); em.boolean(beta, tag + ".beta")
    cc = em.col(SS, tag + ".cc", g("cc"), 20)
    e0 = em.col(SS, tag + ".e0", g("e0"), 30); em.rng(e0, 30, tag + ".e0")
    em.srng(cc, 19, tag + ".cc")
    # E_lo = (2 beta - 1) D_lo - beta = cc 2^30 + e0;  E_hi + cc >= 0, E_hi = (2 beta - 1) D_hi
    sg = lambda: Ex.const(2) * beta.ex - Ex.const(1)
    Dl = lambda: Ex.const(p16) * R0.ex - Ex.const(q16) * A0.ex
    Dh = lambda: Ex.const(p16) * R1.ex - Ex.const(q16) * A1.ex
    em.zero(SS, lambda: sg() * Dl() - beta.ex - Ex.const(1 << 30) * cc.ex - e0.ex, tag + ".lo")
    top = em.col(SS, tag + ".top", g("top"), 48)
    em.zero(SS, lambda: top.ex - sg() * Dh() - cc.ex, tag + ".hi")
    em.rng(top, 48, tag + ".top")
    return beta


# ============================================================================ the circuit
def build(prm, stmt, wit=None, heads=None):
    """Build the sampler circuit.  wit (prover): dict(kprf, salt, steps=[wit_step dicts]).
    heads (HEAD-gadget composition): callable em -> list over steps of dicts(Y, R, ey, OS) -- the reference X objects
    left by ``certified.blk_head`` in ``em.head_ref`` -- built on the SAME Em before the sampler; the sampler's
    per-slot Y / R and per-step e_y are then tied to them by copy constraints (real slots v < V)."""
    pv = wit is not None
    em = CZ.Em(emit=True, prove=pv)
    head_refs = heads(em) if heads is not None else None
    build_on(em, prm, stmt, wit, head_refs=head_refs)
    return em


def build_on(em, prm, stmt, wit=None, head_refs=None, commit_tok=False, hidden_t=False, all_allowed=False,
             head_d=None, kprf_on_ab=None):
    """The sampler circuit on an EXISTING emitting Em (composition: ``flashprove_next.gen_zk`` / ``flashprove_next.reply_zk`` builds
    norm + head on ``em`` first and passes ``head_refs``).
      * commit_tok: commit the emitted token on SS and return it in ``out["tok"]``;
      * hidden_t: step indices ``t`` are hidden committed values on SS (prover: ``wit["t"]`` or per-step ``w["t"]``);
      * all_allowed: constrain ``a_v = 1`` on every real vocabulary slot ``v < V``;
      * head_d: when given, ``Rs`` holds per-slot ``A_v = (W_head^r Z)_v`` and scalar ``Rmax`` is derived in-circuit
        on ``SS`` via ``_head_rmax_from_A(em, Amax, ey, head_d)`` (eliminates per-vocab ``rv_store`` / ``absx``);
      * kprf_on_ab: optional callable ``sp -> [ex0, ex1, ex2, ex3]`` binding ``K_prf`` directly in-circuit on ``AB``
        row 0 (e.g. from ``cfg_ext.vol_col`` indices 3..6 in ``reply_zk``) instead of public ``stmt["kcom"]``."""
    pv = wit is not None
    assert em.emit and em.pv == pv, "build_on needs an emitting Em in the matching prover / verifier mode"
    ctx = em.ctx
    K, sb = prm.K, prm.sb
    W = wit["steps"] if pv else None
    st = lambda k: np.stack([np.asarray(w[k]) for w in W]) if pv else None      # (K, ...)
    sc = lambda k: np.array([w[k] for w in W], dtype=np.int64) if pv else None   # (K,)
    sa = [("s", sb)] if sb else []
    # ---------------------------------------------------------------- spaces
    SS = _sp(em, "sz.SS", sa)
    KS = _sp(em, "sz.KS", sa + [("k", 5)])
    VSs = [_sp(em, f"sz.VS{p}", sa + [("b", bb), ("k", 5)]) for p, (off, bb) in enumerate(prm.pieces)]
    BSs = [_sp(em, f"sz.BS{p}", sa + [("b", bb)]) for p, (off, bb) in enumerate(prm.pieces)]
    GS = _sp(em, "sz.GS", sa + [("g", prm.gb)])
    PSp = _sp(em, "sz.PS", sa + [("c", prm.cb)])
    AB = em.space("sz.AB", [("r", 3)])
    TS = _sp(em, "sz.TS", sa + [("i", 4)]) if not prm.top else None

    def ss(vals):                       # per-step scalar values -> SS rows
        return _rep(SS, vals, K) if pv else None

    def per_slot(p, vals_ns):           # (K, NS) slot values -> rows of VS_p
        if not pv:
            return None
        off, bb = prm.pieces[p]
        a = np.asarray(vals_ns).reshape(K, prm.NS)[:, off * 32:(off + (1 << bb)) * 32]
        return _rep(VSs[p], a, K)

    def per_bkt(p, vals_nb, sp=None):   # (K, NS/32) bucket values -> rows of BS_p
        if not pv:
            return None
        off, bb = prm.pieces[p]
        a = np.asarray(vals_nb).reshape(K, -1)[:, off:off + (1 << bb)]
        return _rep(sp or BSs[p], a, K)

    def ks(vals):                       # (K, 32) -> KS rows
        return _rep(KS, np.asarray(vals).reshape(K, 32), K) if pv else None

    # ---------------------------------------------------------------- tables
    em.table("sz.te", te_table(prm))
    em.table("sz.tj", tj_table(prm))
    em.table("sz.rw", rw_table())
    em.table("sz.rwf", rw_table(True))
    em.table("sz.kq", kq_table())
    em.table("sz.lne", lne_table())
    em.table("sz.lnt", lnt_table())
    if prm.piecewise_gumbel:
        em.table("sz.gm", gmid_table())
        em.table("sz.gt", gtail_table())
    else:
        em.table("sz.g", g_table())
    if not prm.top:
        em.table("sz.r19", r19_table())
    # ---------------------------------------------------------------- per-step public / hidden data
    d_pub = em.pub(SS, "d", _rep(SS, np.asarray(stmt["d"], dtype=np.int64), K), 33)
    if hidden_t:
        t_vals = wit.get("t") if pv else None
        t_pub = em.col(SS, "t", _rep(SS, np.asarray(t_vals, dtype=np.int64), K) if pv else None, 33)
    else:
        t_pub = em.pub(SS, "t", _rep(SS, np.asarray(stmt["t"]), K), 33)
    # ---------------------------------------------------------------- HEAD-interface inputs
    Ys = [em.col(VSs[p], f"Y{p}", per_slot(p, st("Y")), prm.YB) for p in range(len(VSs))]
    Rs = [em.col(VSs[p], f"R{p}", per_slot(p, st("R")), prm.YB) for p in range(len(VSs))]
    ey = em.col(SS, "ey", ss(sc("ey")), 10)
    if head_refs is not None:
        _link_heads(em, prm, head_refs, VSs, SS, Ys, Rs, ey)
    As = []
    for p, (off, bb) in enumerate(prm.pieces):
        a = em.col(VSs[p], f"a{p}", per_slot(p, st("a")), 1)
        em.boolean(a, f"a{p}")
        nreal = prm.V - off * 32
        if nreal < (1 << (bb + 5)):
            S_ = VSs[p].S
            vs = S_.vars("k") + (S_.vars("b") if VSs[p].has("b") else [])
            em.zero(VSs[p], lambda a=a, vs=vs, nreal=nreal: a.ex * (Ex.const(1) - Ex.atom(SelLt(vs, max(nreal, 0)))),
                    f"a{p}.pad")
        if all_allowed:
            S_ = VSs[p].S
            vs = S_.vars("k") + (S_.vars("b") if VSs[p].has("b") else [])
            em.zero(VSs[p], lambda a=a, vs=vs, nreal=nreal: (Ex.const(1) - a.ex) * Ex.atom(SelLt(vs, max(nreal, 0)))
                    if nreal < (1 << (bb + 5)) else Ex.const(1) - a.ex, f"a{p}.all")
        As.append(a)
    # ---------------------------------------------------------------- step-1 scalars
    kq = em.lin([(-1, ey)], -prm.Fq, bits=11)
    Bq = em.col(SS, "B", ss(sc("B")), 33)
    w0 = em.col(SS, "w0", ss(sc("w0")), 5); w1 = em.col(SS, "w1", ss(sc("w1")), 5)
    v0 = em.col(SS, "v0", ss(sc("v0")), 5); v1 = em.col(SS, "v1", ss(sc("v1")), 5); v2 = em.col(SS, "v2", ss(sc("v2")), 5)
    A16 = em.col(SS, "A16", ss(sc("A16")), 17); B16 = em.col(SS, "B16", ss(sc("B16")), 17)
    _lk(em, "sz.kq", SS, [kq, Bq, w0, w1, v0, v1, v2, A16, B16])
    wq_tab = {int(k): int(wq_of(prm, k)) for k in range(KQMAX + 1)}
    if "sz.wq" not in ctx.tables:
        em.table("sz.wq", [np.arange(KQMAX + 1, dtype=np.int64), np.array([wq_tab[k] for k in range(KQMAX + 1)], dtype=np.int64)])
    wq = em.col(SS, "wq", ss(sc("wq")), 5)
    _lk(em, "sz.wq", SS, [kq, wq])
    M = em.col(SS, "M", ss(sc("M")), prm.YB)
    Amax = em.col(SS, "Amax", ss(sc("Amax")), prm.YB)
    # one-hot argmax of Y and of R/A over allowed slots + upper bound Amax >= Rs_v on allowed slots
    def onehot_vals(key):
        if not pv:
            return None, None
        ebv, ekv = [], np.zeros((K, 32), dtype=np.int64)
        for p, (off, bb) in enumerate(prm.pieces):
            e = np.zeros((K, prm.NS // 32), dtype=np.int64)
            for s_, w in enumerate(W):
                e[s_, w[key] // 32] = 1
            ebv.append(per_bkt(p, e))
        for s_, w in enumerate(W):
            ekv[s_, w[key] % 32] = 1
        return ebv, ks(ekv)
    ebY, ekY = _one_hot_pair(em, BSs, KS, SS, *onehot_vals("vY"), "eY")
    ebR, ekR = _one_hot_pair(em, BSs, KS, SS, *onehot_vals("vR"), "eR")
    Mv = [em.view(M, VSs[p]) for p in range(len(VSs))]
    Av = [em.view(Amax, VSs[p]) for p in range(len(VSs))]
    for nm, ebs, ek, centre, vals in (("Y", ebY, ekY, Mv, Ys), ("R", ebR, ekR, Av, Rs)):
        nota = _sum_pieces(em, VSs, lambda p, ebs=ebs, ek=ek: [em.view(ebs[p], VSs[p]), em.view(ek, VSs[p]),
                                                              em.lin([(-1, As[p])], 1)], SS, f"e{nm}.na", bits=1)
        em.zero(SS, lambda nota=nota: nota.ex, f"e{nm}.allowed")
        dif = _sum_pieces(em, VSs, lambda p, ebs=ebs, ek=ek, centre=centre, vals=vals: [
            em.view(ebs[p], VSs[p]), em.view(ek, VSs[p]), em.lin([(1, centre[p]), (-1, vals[p])], bits=prm.YB + 1)],
            SS, f"e{nm}.diff", bits=prm.YB + 1)
        em.zero(SS, lambda dif=dif: dif.ex, f"e{nm}.attained")
    # soundness: enforce Amax >= Rs_v on EVERY allowed slot v
    for p, VSp in enumerate(VSs):
        adif = em.lin([(1, Av[p]), (-1, Rs[p])], bits=prm.YB + 1)
        _rng_sel(em, adif, 60, f"Amax.ge.{p}", As[p])
    if head_d is not None:
        em.rng(Amax, 60, "Amax.nn")
        Rmax = _head_rmax_from_A(em, Amax, ey, head_d, "sz.hd")
    else:
        Rmax = Amax
    # Rmax < 2^(kq + RQ) (hidden widths) and rho = ceil(Rmax 2^16 / 2^kq)
    Rm0 = em.col(SS, "Rm0", ss(sc("Rm0")), 16); Rm1 = em.col(SS, "Rm1", ss(sc("Rm1")), 16)
    Rm2 = em.col(SS, "Rm2", ss(sc("Rm2")), 16)
    for vv, cc in ((v0, Rm0), (v1, Rm1), (v2, Rm2)):
        _lk(em, "sz.rw", SS, [vv, cc])
    em.zero(SS, lambda: Rmax.ex - Rm0.ex - Ex.const(1 << 16) * Rm1.ex - Ex.const(1 << 32) * Rm2.ex, "Rmax.dec")
    rho = em.col(SS, "rho", ss(sc("rho")), 33); em.rng(rho, 33, "rho")
    rem16 = em.col(SS, "rem16", ss(sc("rem16")), 16); em.rng(rem16, 16, "rem16")
    em.rng(em.lin([(1, B16), (-1, rem16)], -1, bits=17), 16, "rem16.hi")
    em.zero(SS, lambda: Rmax.ex * A16.ex + B16.ex - Ex.const(1) - rho.ex * B16.ex - rem16.ex, "rho.def")
    D0 = em.col(SS, "D0", ss(sc("D0")), 17)
    remS = em.col(SS, "remS", ss(sc("remS")), prm.SD); em.rng(remS, prm.SD, "remS")
    em.zero(SS, lambda: Ex.const(2 * prm.Cup) * rho.ex + Ex.const(prm.EQn + (1 << prm.SD) - 1)
            - D0.ex * Ex.const(1 << prm.SD) - remS.ex, "delta.def")
    Delta = em.lin([(1, D0)], 2, bits=17)
    em.rng(em.lin([(-1, Delta)], DMAX, bits=17), 16, "delta.cap")
    # ---------------------------------------------------------------- per-entry step 1 + weights
    js, wlos, whis = [], [], []
    for p, VSp in enumerate(VSs):
        a = As[p]
        vw = lambda x: em.view(x, VSp)
        r0 = em.col(VSp, f"r0.{p}", per_slot(p, st("r0")), 16)
        r1 = em.col(VSp, f"r1.{p}", per_slot(p, st("r1")), 16)
        qlo = em.col(VSp, f"qlo.{p}", per_slot(p, st("qlo")), QL)
        jlo = em.col(VSp, f"jlo.{p}", per_slot(p, st("jlo")), 18)
        qhi = em.col(VSp, f"qhi.{p}", per_slot(p, st("qhi")), 16)
        f = em.col(VSp, f"f.{p}", per_slot(p, st("f")), 1)
        _lk(em, "sz.rw", VSp, [vw(w0), r0], sel=a)
        _lk(em, "sz.rw", VSp, [vw(w1), r1], sel=a)
        _lk(em, "sz.rwf", VSp, [vw(wq), qhi, f], sel=a)
        _lk(em, "sz.tj", VSp, [qlo, jlo], sel=a)
        Bv, Mvp = vw(Bq), Mv[p]
        Y = Ys[p]
        em.zero(VSp, lambda a=a, Bv=Bv, Mvp=Mvp, Y=Y, r0=r0, r1=r1, qlo=qlo, qhi=qhi: a.ex * (
            Mvp.ex - Y.ex - r0.ex - Ex.const(1 << 16) * r1.ex - Bv.ex * (qlo.ex + Ex.const(1 << QL) * qhi.ex)), f"dec.{p}")
        em.zero(VSp, lambda a=a, f=f: (Ex.const(1) - a.ex) * (f.ex - Ex.const(1)), f"masked.{p}")
        j = em.lin([(1, jlo), (BIG, f)], bits=JB)
        Dv = vw(Delta)
        wlo = em.col(VSp, f"wlo.{p}", per_slot(p, st("wlo")), 41)
        whi = em.col(VSp, f"whi.{p}", per_slot(p, st("whi")), 41)
        _lk(em, "sz.te", VSp, [em.lin([(1, j), (1, Dv)], bits=JB + 1), wlo])
        _lk(em, "sz.te", VSp, [em.lin([(1, j), (-1, Dv)], bits=JB + 1), whi])
        js.append(j); wlos.append(wlo); whis.append(whi)
    # ---------------------------------------------------------------- nucleus classification (top-p < 1)
    if prm.top:
        nlos, nhis = [[w_] for w_ in wlos], [[w_] for w_ in whis]
    else:
        nlos, nhis = [], []
        LOs, HIs, svs, pvs = [], [], [], []
        for p, VSp in enumerate(VSs):
            LO = em.col(VSp, f"LO.{p}", per_slot(p, st("eLO")), JB + 1)
            HI = em.col(VSp, f"HI.{p}", per_slot(p, st("eHI")), JB + 1)
            sv = em.col(VSp, f"sv.{p}", per_slot(p, st("sv")), 1)
            pv_ = em.col(VSp, f"pv.{p}", per_slot(p, st("pv")), 1)
            _lk(em, "sz.r19", VSp, [em.lin([(1, js[p]), (-1, LO)], bits=JB + 1)])
            _lk(em, "sz.r19", VSp, [em.lin([(1, HI), (-1, js[p])], bits=JB + 1)])
            LOs.append(LO); HIs.append(HI); svs.append(sv); pvs.append(pv_)
            nlos.append([sv, wlos[p]])             # products stay inside the sumchecks (no committed cell)
            nhis.append([pv_, whis[p]])
        _nucleus_circuit(em, prm, stmt, W, SS, TS, VSs, As, js, wlos, whis, LOs, HIs, svs, pvs, Delta, ss, st, sc)
    # ---------------------------------------------------------------- buckets
    Wlo_b, Whi_b, keyconds = [], [], []
    for p, BSp in enumerate(BSs):
        Wl = _sum_to(em, nlos[p], BSp, f"Wlo.{p}", 46)
        Wh = _sum_to(em, nhis[p], BSp, f"Whi.{p}", 46)
        Wlo_b.append(Wl); Whi_b.append(Wh)
    # ---------------------------------------------------------------- PRF: kcom / vol binding + ctx absorption rows (AB) and call rows (PS)
    ab_in = None
    if pv:
        kv = S3.kprf_elems(wit["kprf"])
        rows = np.zeros((12, 8), dtype=np.uint64)
        rows[:4, 0] = [int(x) % P for x in wit.get("salt", (0, 0, 0, 0))]; rows[4:8, 0] = kv
        rows[10, 0] = DOM_KCOM; rows[11, 0] = PS.TAG_SPONGE
        enc = np.asarray(stmt["ctx"] + [0] * (S3.CTX_BLOCKS * 8 - len(stmt["ctx"])), dtype=np.uint64)
        state = np.zeros(12, dtype=np.uint64); state[8:] = kv
        for b in range(S3.CTX_BLOCKS):
            inp = state.copy()
            inp[:8] = S3._addp(inp[:8], enc[8 * b:8 * b + 8])
            rows[:, 1 + b] = inp
            state = PS.permute(inp[:, None])[:, 0]
        ab_in = rows
    ab = G.PoseRegion(AB.S, ab_in)
    abI = lambda k: ab.I(k).ex
    abO = lambda k: ab.O(k).ex
    r0sel = AB.S.ind(r=0)
    if kprf_on_ab is not None:
        kex_list = kprf_on_ab(AB.S)
        assert len(kex_list) == 4
        for k in range(4):
            AB.S.zero(abI(4 + k) - kex_list[k], sel=r0sel, tag=f"kprf.vol.{k}")
    else:
        for k in range(8, 10):
            AB.S.zero(abI(k), sel=r0sel, tag="kcom.zero")
        AB.S.zero(abI(10) - Ex.const(DOM_KCOM), sel=r0sel, tag="kcom.dom")
        AB.S.zero(abI(11) - Ex.const(PS.TAG_SPONGE), sel=r0sel, tag="kcom.tag")
        for k in range(4):
            AB.S.zero(abO(k) - Ex.const(int(stmt["kcom"][k]) % P), sel=r0sel, tag="kcom.out")
    enc = list(stmt["ctx"]) + [0] * (S3.CTX_BLOCKS * 8 - len(stmt["ctx"]))
    for b in range(S3.CTX_BLOCKS):
        sel = AB.S.ind(r=1 + b)
        for k in range(12):
            blk = enc[8 * b + k] if k < 8 else 0
            if b == 0:
                prev = Ex.const(0) if k < 8 else ab.I(4 + (k - 8)).shifted(-1)
            else:
                prev = ab.O(k).shifted(-1)
            AB.S.zero(abI(k) - prev - Ex.const(blk % P), sel=sel, tag="ctx.absorb")
    # PRF call rows
    ps_in = None
    outv = None
    if pv:
        ps_raw = np.concatenate([w["prf_in"] for w in W], axis=1)                # (12, K 2^cb) uint64
        out_raw = np.concatenate([w["prf_out"][:8] for w in W], axis=1)          # (8, K 2^cb) uint64
        ps_reps = PSp.n // ps_raw.shape[1]
        ps_in = np.repeat(ps_raw, ps_reps, axis=1)
        outv = np.repeat(out_raw, ps_reps, axis=1)
    reg = G.PoseRegion(PSp.S, ps_in)
    pre = [ab.O(k).on(PSp.S, {"r": ("c", S3.CTX_BLOCKS)}) for k in range(12)]
    used = PSp.S.lt("c", prm.NC)
    is_inv = PSp.S.ind(c=prm.nb)
    cidx = PSp.S.idx("c")
    dv = em.view(d_pub, PSp); tv = em.view(t_pub, PSp)
    adds = [dv.ex, tv.ex, Ex.const(S3.LBL_BKT) + Ex.const(S3.LBL_INV - S3.LBL_BKT) * is_inv,
            cidx * (Ex.const(1) - is_inv)]
    for k in range(12):
        e = reg.I(k).ex - pre[k]
        if k < 4:
            e = e - adds[k]
        PSp.S.zero(e, sel=used, tag="prf.in")
    # canonical limb split of the 8 output elements: l0(16) l1(8) | h0(8) h1(16) | t(16)
    lo24s, hi24s = [], []
    for i in range(8):
        ev = outv[i].astype(np.uint64) if pv else None
        gv = (lambda sh, msk: ((ev >> np.uint64(sh)) & np.uint64(msk)).astype(np.int64)) if pv else (lambda sh, msk: None)
        l0 = em.col(PSp, f"l0.{i}", gv(0, 0xFFFF), 16); l1 = em.col(PSp, f"l1.{i}", gv(16, 0xFF), 8)
        h0 = em.col(PSp, f"h0.{i}", gv(24, 0xFF), 8); h1 = em.col(PSp, f"h1.{i}", gv(32, 0xFFFF), 16)
        tt = em.col(PSp, f"t.{i}", gv(48, 0xFFFF), 16)
        for c_, wd in ((l0, 16), (l1, 8), (h0, 8), (h1, 16), (tt, 16)):
            _lk(em, "sz.rw", PSp, [em.const(PSp, wd), c_])
        o = reg.O(i).ex
        PSp.S.zero(o - l0.ex - Ex.const(1 << 16) * l1.ex - Ex.const(1 << 24) * h0.ex - Ex.const(1 << 32) * h1.ex
                   - Ex.const(1 << 48) * tt.ex, tag="prf.split")
        Xv = None; mu = None; fz = None
        if pv:
            Xv = (h1.v + (tt.v << 16)) - ((1 << 32) - 1)
            fz = (Xv == 0).astype(np.int64)
            mu = _inv_field(np.array([int(x) % P for x in Xv], dtype=np.uint64))
        fzc = em.col(PSp, f"fz.{i}", fz, 1)
        muc = _ucol(em, PSp, f"mu.{i}", mu)
        Xe = h1.ex + Ex.const(1 << 16) * tt.ex - Ex.const((1 << 32) - 1)
        PSp.S.zero(fzc.ex - Ex.const(1) + Xe * muc.ex, tag="prf.canon1")
        PSp.S.zero(Xe * fzc.ex, tag="prf.canon2")
        PSp.S.zero(fzc.ex * (l0.ex + l1.ex + h0.ex), tag="prf.canon3")
        lo24s.append((l0, l1)); hi24s.append((h0, h1))
    # Gumbel slots: g = 16 c + 2 i + h ;  idx = (1-h) lo24_i + h hi24_i
    GSs = GS.S
    gidx = em.col(GS, "gidx", _rep(GS, st("gidx"), K) if pv else None, 24)
    gval = em.col(GS, "gval", _rep(GS, st("gval"), K) if pv else None, 30)
    gvars = GSs.vars("g")
    cmap = {"c": [("g", 4 + t) for t in range(prm.cb)]}
    hbit = Ex.atom(IndexP([gvars[0]], 1))
    acc = Ex.const(0)
    for i in range(8):
        indi = Ex.atom(SelLt([], 1, fixed=tuple((gvars[1 + t], (i >> t) & 1) for t in range(3))))
        lo = em.view(lo24s[i][0], GS, cmap).ex + Ex.const(1 << 16) * em.view(lo24s[i][1], GS, cmap).ex
        hi = em.view(hi24s[i][0], GS, cmap).ex + Ex.const(1 << 8) * em.view(hi24s[i][1], GS, cmap).ex
        acc = acc + indi * ((Ex.const(1) - hbit) * lo + hbit * hi)
    GSs.zero(gidx.ex - acc, sel=GSs.lt("g", prm.NC * 16), tag="gumbel.idx")
    if prm.piecewise_gumbel:
        gblk = em.col(GS, "gblk", _rep(GS, st("gblk"), K) if pv else None, 16)
        goff = em.col(GS, "goff", _rep(GS, st("goff"), K) if pv else None, G_BLK_BITS)
        gtail = em.col(GS, "gtail", _rep(GS, st("gtail"), K) if pv else None, 1)
        gbase = em.col(GS, "gbase", _rep(GS, st("gbase"), K) if pv else None, 30)
        gslope = em.col(GS, "gslope", _rep(GS, st("gslope"), K) if pv else None, 25)
        gquo = em.col(GS, "gquo", _rep(GS, st("gquo"), K) if pv else None, 25)
        grem = em.col(GS, "grem", _rep(GS, st("grem"), K) if pv else None, 16)
        gtv = em.col(GS, "gtv", _rep(GS, st("gtail_val"), K) if pv else None, 30)
        _lk(em, "sz.rw", GS, [em.const(GS, G_BLK_BITS), goff])
        _lk(em, "sz.rw", GS, [em.const(GS, 16), grem])
        em.rng(gquo, 25, "gumbel.quo")
        GSs.zero(gidx.ex - goff.ex - Ex.const(1 << G_BLK_BITS) * gblk.ex, tag="gumbel.split")
        _lk(em, "sz.gm", GS, [gblk, gbase, gslope, gtail])
        GSs.zero(gslope.ex * goff.ex - Ex.const(1 << 16) * gquo.ex - grem.ex, tag="gumbel.lin")
        _lk(em, "sz.gt", GS, [gidx, gtv], sel=gtail)
        GSs.zero(gval.ex - gtail.ex * gtv.ex - (Ex.const(1) - gtail.ex) * (gbase.ex + gquo.ex), tag="gumbel.val")
    else:
        _lk(em, "sz.g", GS, [gidx, gval])
    # u48 of the INV call: e_0 of row c = nb
    u0 = em.col(SS, "u0", ss(sc("u0")), 24); u1 = em.col(SS, "u1", ss(sc("u1")), 24)
    imap = {"c": ("c", prm.nb)}
    SS.S.zero(u0.ex - em.view(lo24s[0][0], SS, imap).ex - Ex.const(1 << 16) * em.view(lo24s[0][1], SS, imap).ex, tag="u0")
    SS.S.zero(u1.ex - em.view(hi24s[0][0], SS, imap).ex - Ex.const(1 << 8) * em.view(hi24s[0][1], SS, imap).ex, tag="u1")
    # ---------------------------------------------------------------- output token: one-hots, extraction, conditions
    ebO, ekO = _one_hot_pair(em, BSs, KS, SS, *onehot_vals("tok"), "eO")
    tok_parts = []
    for p, (off, bb) in enumerate(prm.pieces):
        tok_parts.append(_sum_to(em, [ebO[p], _ax(em, BSs[p], "b", off)], SS, f"tokb.{p}", prm.gb + 1))
    tk = _sum_to(em, [ekO, _ax(em, KS, "k")], SS, "tokk", 6)
    tok = em.lin([(32, t_) for t_ in tok_parts] + [(1, tk)], bits=prm.gb + 7)
    if stmt.get("tok") is not None:
        tpub = em.pub(SS, "tokpub", _rep(SS, np.asarray(stmt["tok"]), K), prm.gb + 7)
        em.zero(SS, lambda: tok.ex - tpub.ex, "tok.public")
    if commit_tok:
        tok = em.commit(tok, "tok.c")
    # chosen-bucket extraction onto KS: X(s,k) = sum_b ob(s,b) n(s,b,k)
    def extract(ns, tag):
        tot = None
        for p, VSp in enumerate(VSs):
            S_ = _sum_to(em, [em.view(ebO[p], VSp)] + ns[p], KS, f"{tag}.{p}", 46)
            tot = S_ if tot is None else em.lin([(1, tot), (1, S_)], bits=46)
        return em.commit(tot, tag)
    Xlo, Xhi = extract(nlos, "Xlo"), extract(nhis, "Xhi")
    Plo = em.col(KS, "Plo", ks(st("Plo")), 46); Phi = em.col(KS, "Phi", ks(st("Phi")), 46)
    k0 = KS.S.ind(k=0)
    for Pc, Xc, nm in ((Plo, Xlo, "Plo"), (Phi, Xhi, "Phi")):
        KS.S.zero(Pc.ex - Xc.ex - (Ex.const(1) - k0) * Pc.col.shifted(-1), tag=nm + ".prefix")
    SloP = _sum_to(em, [ekO, em.lin([(1, Plo), (-1, Xlo)], bits=47)], SS, "Slo_prev", 46)
    ShiU = _sum_to(em, [ekO, Phi], SS, "Shi_u", 46)
    XhiU = _sum_to(em, [ekO, Xhi], SS, "Xhi_u", 41)
    em.rng(em.lin([(1, XhiU)], -1, bits=42), 41, "tok.inNhi")
    WloS = _sum_pieces_b(em, BSs, ebO, Wlo_b, SS, "Wlo_s")
    WhiS = _sum_pieces_b(em, BSs, ebO, Whi_b, SS, "Whi_s")
    gvb = [em.view(gval, BSs[p], {"g": ([("b", t) for t in range(bb)] if bb else []) +
                                  [("c", (off >> t) & 1) for t in range(bb, prm.gb)]}) for p, (off, bb) in enumerate(prm.pieces)]
    Gs = _sum_pieces_b(em, BSs, ebO, gvb, SS, "G_s", bits=30)
    tau_hi = _mulshift_gadget(em, SS, u0, u1, WhiS, _scal_parts(W, "tau_hi", SS, K) if pv else None, "tauhi")
    tau_lo = _mulshift_gadget(em, SS, u0, u1, WloS, _scal_parts(W, "tau_lo", SS, K) if pv else None, "taulo")
    em.rng(em.lin([(1, tau_hi), (-1, SloP)], bits=48), 47, "inv.c1")
    em.rng(em.lin([(1, ShiU), (-1, tau_lo)], -1, bits=48), 47, "inv.c2")
    # bucket keys: LN(W_hi of the chosen bucket) + G_s + 2*g_err >= LN(W_lo_b) + G_b for every bucket with W_lo_b > 0
    zone = em.const(SS, 1)
    LNh = _ln_gadget(em, SS, WhiS, zone, _scal_parts(W, "lnh", SS, K) if pv else None, "lnh")
    Kstar = em.lin([(1, LNh), (1, Gs)], bits=33)
    for p, BSp in enumerate(BSs):
        off, bb = prm.pieces[p]
        lp = None
        if pv:
            lp = {k_: per_bkt(p, np.stack([w["ln"][k_] for w in W])) for k_ in ("e", "A", "B", "w0", "w1", "m", "rem0",
                                                                               "rem1", "LNv", "z")}
        z = em.col(BSp, f"z.{p}", lp["z"] if pv else None, 1)
        mu = None
        if pv:
            mu = _inv_field(np.asarray(Wlo_b[p].v, dtype=np.int64).astype(np.uint64))
        muc = _ucol(em, BSp, f"zmu.{p}", mu)
        em.zero(BSp, lambda z=z, muc=muc, Wl=Wlo_b[p]: Wl.ex * muc.ex - z.ex, f"z.def.{p}")
        em.zero(BSp, lambda z=z, Wl=Wlo_b[p]: Wl.ex * (Ex.const(1) - z.ex), f"z.zero.{p}")
        LNl = _ln_gadget(em, BSp, Wlo_b[p], z, lp, f"lnl.{p}")
        diff = em.lin([(1, em.view(Kstar, BSp)), (-1, LNl), (-1, gvb[p])], 2 * prm.g_err, bits=34)
        _rng_sel(em, diff, 33, f"bkt.cond.{p}", z)
    return dict(tok=tok, t=t_pub, d=d_pub, SS=SS, Ys=Ys, Rs=Rs, As=As, ey=ey, VSs=VSs)


def _link_heads(em, prm, refs, VSs, SS, Ys, Rs, ey):
    """copy constraints Y_p(s, slot) = HEAD_s.Y(slot), R likewise (slots v < V), e_y(s) = HEAD_s.e_y."""
    assert len(refs) == prm.K
    for s_, ref in enumerate(refs):
        OS = ref["OS"]
        ob = OS.bits
        for p, (off, bb) in enumerate(prm.pieces):
            VSp = VSs[p]
            vbits = [("k", t) for t in range(5)] + ([("b", t) for t in range(bb)] if bb else [])
            hi = [("c", ((off * 32) >> t) & 1) for t in range(bb + 5, ob)]
            amap = {"o": (vbits + hi)[:ob]}
            if (off * 32) >= (1 << ob):
                continue
            S_ = VSp.S
            vs = S_.vars("k") + (S_.vars("b") if VSp.has("b") else [])
            sel = Ex.atom(SelLt(vs, max(0, prm.V - off * 32)))
            if prm.sb:
                sel = sel * S_.ind(s=s_)
            for mine, theirs in ((Ys[p], ref["Y"]), (Rs[p], ref["R"])):
                tv = em.view(theirs, VSp, amap)
                S_.zero(mine.ex - tv.ex, sel=sel, tag="head.link")
        sel = SS.S.ind(s=s_) if prm.sb else Ex.const(1)
        SS.S.zero(ey.ex - em.view(ref["ey"], SS).ex, sel=sel, tag="head.ey")


def _scal_parts(W, key, SS, K):
    """per-step dict of scalars (W[s][key]) -> dict of SS row vectors."""
    keys = W[0][key].keys()
    out = {}
    for k_ in keys:
        vals = [w[key][k_] for w in W]
        vals = [int(np.asarray(v).ravel()[0]) for v in vals]
        out[k_] = _rep(SS, np.array(vals, dtype=np.int64), K)
    return out


def _sum_pieces_b(em, BSs, ebs, vals, SS, tag, bits=46):
    tot = None
    for p, BSp in enumerate(BSs):
        S_ = _sum_to(em, [ebs[p], vals[p]], SS, f"{tag}.{p}", bits)
        tot = S_ if tot is None else em.lin([(1, tot), (1, S_)], bits=bits)
    return em.commit(tot, tag)


# ---------------------------------------------------------------- nucleus: interval table, boundary candidates, weighted LogUp
NX = 12          # breakpoints per step: ls, ls+1, lsp, lsp+1, lp, lp+1, lpp, lpp+1, Ls, Lsp, Lp, Lpp


def _nucleus_circuit(em, prm, stmt, W, SS, TS, VSs, As, js, wlos, whis, LOs, HIs, svs, pvs, Delta, ss, st, sc):
    pv = em.pv
    K = prm.K
    q16, p16 = (1 << S3.P_BITS) - prm.rule.p16, prm.rule.p16
    nuc = (lambda k: np.stack([np.asarray(w["nuc"][k]) for w in W])) if pv else (lambda k: None)
    ts = lambda vals: _rep(TS, np.asarray(vals).reshape(K, TROWS), K) if pv else None
    flags = {}
    for nm in ("es", "fs", "ep", "fp"):
        c = em.col(SS, nm, ss([w["nuc"][nm] for w in W]) if pv else None, 1)
        em.boolean(c, nm)
        flags[nm] = c
    cd = lambda nm, k: ss([w["nuc"]["cands"][nm][k] for w in W]) if pv else None
    ell = {}
    for nm in ("s", "sp", "p", "pp"):
        ell[nm] = em.col(SS, f"ell.{nm}", cd(nm, "ell"), JB + 1)
    # virtual candidates: es -> ls = -1, fs -> lsp = BIG (same for the possible side)
    em.zero(SS, lambda: flags["es"].ex * (ell["s"].ex + Ex.const(1)), "virt.s")
    em.zero(SS, lambda: flags["fs"].ex * (ell["sp"].ex - Ex.const(BIG)), "virt.sp")
    em.zero(SS, lambda: flags["ep"].ex * (ell["p"].ex + Ex.const(1)), "virt.p")
    em.zero(SS, lambda: flags["fp"].ex * (ell["pp"].ex - Ex.const(BIG)), "virt.pp")
    for nm in ("s", "sp", "p", "pp"):
        em.rng(em.lin([(1, ell[nm])], 1, bits=JB + 1), JB + 1, f"ell.{nm}")
    # per candidate: wl = E'(ell + D), wh = E'(ell - D), ps = plateau start of x (x = ell + D sure / ell - D poss)
    C = {}
    for nm, typ in (("s", "sure"), ("sp", "sure"), ("p", "poss"), ("pp", "poss")):
        wl = em.col(SS, f"wl.{nm}", cd(nm, "wl"), 41)
        wh = em.col(SS, f"wh.{nm}", cd(nm, "wh"), 41)
        _lk(em, "sz.te", SS, [em.lin([(1, ell[nm]), (1, Delta)], bits=JB + 2), wl])
        _lk(em, "sz.te", SS, [em.lin([(1, ell[nm]), (-1, Delta)], bits=JB + 2), wh])
        x = em.lin([(1, ell[nm]), (1 if typ == "sure" else -1, Delta)], bits=JB + 2)
        ps = em.col(SS, f"ps.{nm}", cd(nm, "ps"), JB + 2)
        Eps = em.col(SS, f"Eps.{nm}", cd(nm, "Eps"), 42)
        Epm = em.col(SS, f"Epm.{nm}", cd(nm, "psm1"), 42)
        _lk(em, "sz.te", SS, [ps, Eps])
        _lk(em, "sz.te", SS, [em.lin([(1, ps)], -1, bits=JB + 2), Epm])
        Ex_ = wl if typ == "sure" else wh                       # E'(x)
        em.zero(SS, lambda Eps=Eps, Ex_=Ex_: Eps.ex - Ex_.ex, f"ps.eq.{nm}")
        em.rng(em.lin([(1, Epm), (-1, Eps)], -1, bits=43), 42, f"ps.strict.{nm}")
        em.rng(em.lin([(1, x), (-1, ps)], bits=JB + 3), JB + 2, f"ps.le.{nm}")
        L = em.lin([(1, ps), (1 if typ == "sure" else -1, Delta)], bits=JB + 3)
        C[nm] = dict(wl=wl, wh=wh, L=L, typ=typ)
    Xs = [ell["s"], em.lin([(1, ell["s"])], 1), ell["sp"], em.lin([(1, ell["sp"])], 1),
          ell["p"], em.lin([(1, ell["p"])], 1), ell["pp"], em.lin([(1, ell["pp"])], 1),
          C["s"]["L"], C["sp"]["L"], C["p"]["L"], C["pp"]["L"]]
    # ---- interval table rows (TS): contiguous partition of [0, MAXJ]
    LO = em.col(TS, "tLO", ts(nuc("LO")), JB + 1)
    HI = em.col(TS, "tHI", ts(nuc("HI")), JB + 1)
    i0 = TS.S.ind(i=0)
    iL = TS.S.ind(i=TROWS - 1)
    TS.S.zero(LO.ex, sel=i0, tag="t.lo0")
    TS.S.zero(HI.ex - Ex.const(prm.MAXJ), sel=iL, tag="t.hiN")
    TS.S.zero((Ex.const(1) - i0) * (LO.ex - HI.col.shifted(-1) - Ex.const(1)), tag="t.contig")
    em.rng(em.lin([(1, HI), (-1, LO)], 1, bits=JB + 2), JB + 1, "t.order")
    bitc = []
    for x in range(NX):
        b = em.col(TS, f"tb{x}", ts(nuc("bits")[..., x]) if pv else None, 1)
        em.boolean(b, f"tb{x}")
        _cmp_bit(em, TS, b, LO, HI, em.view(Xs[x], TS), f"t.cmp{x}")
        bitc.append(b)
    Sb, Pb = bitc[1], bitc[5]
    m = em.col(TS, "tm", ts(nuc("m")), JB); em.rng(m, JB, "tm")
    Ma = em.col(TS, "tMa", ts(nuc("Ma")), JB); em.rng(Ma, JB, "tMa")
    Mlo = em.col(TS, "tMlo", ts(nuc("Mlo")), 60); em.rng(Mlo, 60, "tMlo")
    Mhi = em.col(TS, "tMhi", ts(nuc("Mhi")), 60); em.rng(Mhi, 60, "tMhi")
    # gap emptiness and attainment
    for nm, xa, xb in (("s", 1, 2), ("p", 5, 6)):
        gp = _sum_to(em, [em.lin([(-1, bitc[xa])], 1), bitc[xb], Ma], SS, f"gap.{nm}", JB)
        em.zero(SS, lambda gp=gp: gp.ex, f"gap.{nm}")
    for nm, x0, x1, fl in (("s", 0, 1, "es"), ("sp", 2, 3, "fs"), ("p", 4, 5, "ep"), ("pp", 6, 7, "fp")):
        at = _sum_to(em, [em.lin([(1, bitc[x1]), (-1, bitc[x0])], bits=2), Ma], SS, f"att.{nm}", JB)
        em.rng(X(SS, (at.v - 1) * (1 - flags[fl].v) if pv else None, JB + 1,
                 ex=(at.ex - Ex.const(1)) * (Ex.const(1) - flags[fl].ex)), JB + 1, f"att.{nm}")
    # candidate predicates
    xi = {"s": 8, "sp": 9, "p": 10, "pp": 11}
    want = {"s": (1, "es"), "sp": (0, "fs"), "p": (1, "ep"), "pp": (0, "fp")}
    for nm in ("s", "sp", "p", "pp"):
        c_ = C[nm]
        cL = bitc[xi[nm]]
        g = lambda k: cd(nm, k)
        if c_["typ"] == "sure":
            SA = _sum_to(em, [cL, Mhi], SS, f"SA.{nm}", 60)
            SR = _sum_to(em, [em.lin([(-1, cL)], 1), Mlo], SS, f"SR.{nm}", 60)
        else:
            SA = _sum_to(em, [cL, Mlo], SS, f"SA.{nm}", 60)
            SR = _sum_to(em, [em.lin([(-1, cL)], 1), Mhi], SS, f"SR.{nm}", 60)
        lt = em.col(SS, f"lt.{nm}", g("lt"), 1)
        em.boolean(lt, f"lt.{nm}")
        # lt = [ell < L]
        em.rng(X(SS, (lt.v * (c_["L"].v - 1 - ell[nm].v) + (1 - lt.v) * (ell[nm].v - c_["L"].v)) if pv else None, JB + 4,
                 ex=lt.ex * (c_["L"].ex - Ex.const(1) - ell[nm].ex) + (Ex.const(1) - lt.ex) * (ell[nm].ex - c_["L"].ex)),
               JB + 4, f"lt.{nm}")
        if c_["typ"] == "sure":
            Aq = em.col(SS, f"Aq.{nm}", g("Aq"), 60)
            em.zero(SS, lambda Aq=Aq, SA=SA, lt=lt, c_=c_: Aq.ex - SA.ex + lt.ex * c_["wh"].ex, f"Aq.{nm}")
            Rq = em.col(SS, f"Rq.{nm}", g("Rq"), 60)
            em.zero(SS, lambda Rq=Rq, SR=SR, lt=lt, c_=c_: Rq.ex - c_["wl"].ex - SR.ex + (Ex.const(1) - lt.ex) * c_["wl"].ex,
                    f"Rq.{nm}")
            wz = c_["wl"]
        else:
            Aq, Rq, wz = SA, SR, c_["wh"]
        zt = em.col(SS, f"zt.{nm}", g("zt"), 1)
        mu = _ucol(em, SS, f"ztmu.{nm}", _inv_field(np.asarray(wz.v, dtype=np.int64).astype(np.uint64)) if pv else None)
        em.zero(SS, lambda zt=zt, mu=mu, wz=wz: wz.ex * mu.ex - zt.ex, f"zt.def.{nm}")
        em.zero(SS, lambda zt=zt, wz=wz: wz.ex * (Ex.const(1) - zt.ex), f"zt.zero.{nm}")
        wp = None
        if pv:
            wp = {k_: _rep(SS, np.array([w["nuc"]["cands"][nm]["wide"][k_] for w in W], dtype=np.int64), K)
                  for k_ in ("A0", "A1", "R0", "R1", "beta", "cc", "e0", "top")}
        beta = _wide_gadget(em, SS, q16, p16, Aq, Rq, wp, f"wide.{nm}")
        val, fl = want[nm]
        em.zero(SS, lambda zt=zt, beta=beta, fl=flags[fl], val=val: (Ex.const(1) - fl.ex) * (zt.ex * beta.ex - Ex.const(val)),
                f"pred.{nm}")
    # ---- weighted LogUp (commit round): every slot's (s, LO, HI, sv, pv) is a table row; per-row multiplicities
    #      m (all slots), Ma (allowed), Mlo / Mhi (weight sums) are the coefficients of 1, delta, delta^2, delta^3
    cir = em.ctx.cir
    SIG = em.space("sz.SIG", [("z", 5)])

    def rnd(tr):
        gam, bet, dlt = tr.ext(), tr.ext(), tr.ext()
        bp = [CPU.ext(CPU.fromi(np.array([1])))]  # placeholder
        b1 = bet; b2 = _emul(bet, bet); b3 = _emul(b2, bet); b4 = _emul(b3, bet)
        d1 = dlt; d2 = _emul(dlt, dlt); d3 = _emul(d2, dlt)
        sums = []
        Hs = []

        def comp_ex(s_ex, lo, hi, sv_, pv__):
            return s_ex + lo * Ex.const(b1) + hi * Ex.const(b2) + sv_ * Ex.const(b3) + pv__ * Ex.const(b4)

        def comp_vals(s_v, lo, hi, sv_, pv__):
            acc = CPU.ext(CPU.fromi(np.asarray(s_v, dtype=np.int64)))
            for co, v in ((b1, lo), (b2, hi), (b3, sv_), (b4, pv__)):
                acc = CPU.eadd(acc, CPU.escale(CPU.ext(CPU.fromi(np.asarray(v, dtype=np.int64))), co))
            return acc

        def wsum_vals(c0, c1, c2, c3):
            acc = CPU.ext(CPU.fromi(np.asarray(c0, dtype=np.int64)))
            for co, v in ((d1, c1), (d2, c2), (d3, c3)):
                acc = CPU.eadd(acc, CPU.escale(CPU.ext(CPU.fromi(np.asarray(v, dtype=np.int64))), co))
            return acc

        sig_cols = []
        for p, VSp in enumerate(VSs):
            S_ = VSp.S
            s_ex = S_.idx("s") if prm.sb else Ex.const(0)
            zsel_v = _zsel(VSp)
            Hv = None
            sv_tot = None
            if pv:
                s_v = VSp.grid("s") if prm.sb else np.zeros(VSp.n, dtype=np.int64)
                lo_v, hi_v, sv_v, pv_v = LOs[p].v, HIs[p].v, svs[p].v, pvs[p].v
                a_v, wl_v, wh_v = As[p].v, wlos[p].v, whis[p].v
                zax = next((a for a, _ in VSp.axes if a.startswith("zz")), None)
                z_msk = (VSp.grid(zax) == 0).astype(np.uint64) if zax is not None else None
                Hv = np.empty((3, VSp.n), dtype=np.uint64)
                sv_tot = (0, 0, 0)
                from .field import hadd
                B_ = getattr(em, "B", CPU)
                gpu = getattr(B_, "name", "") == "cupy"
                CH = 1 << 20
                for c0 in range(0, VSp.n, CH):
                    c1 = min(VSp.n, c0 + CH)
                    m_ = c1 - c0
                    den_c = CPU.esub(np.stack([np.full(m_, gam[l], dtype=np.uint64) for l in (0, 1, 2)]),
                                     comp_vals(s_v[c0:c1], lo_v[c0:c1], hi_v[c0:c1], sv_v[c0:c1], pv_v[c0:c1]))
                    num_c = wsum_vals(np.ones(m_, dtype=np.int64), a_v[c0:c1], wl_v[c0:c1], wh_v[c0:c1])
                    if z_msk is not None:
                        num_c = num_c * z_msk[None, c0:c1]
                    if gpu:
                        hc = B_.tonumpy(B_.emul(B_.asarray(num_c), B_.einv(B_.asarray(den_c))))
                    else:
                        hc = CPU.emul(num_c, CPU.einv(den_c))
                    Hv[:, c0:c1] = hc
                    sv_tot = hadd(sv_tot, CPU.esum(hc))
                del lo_v, hi_v, sv_v, pv_v, a_v, wl_v, wh_v
            from .mc_zk import ecol, view as mview
            H = ecol(S_, f"H{p}", Hv)
            sig_v = np.tile(np.array([[sv_tot[0]], [sv_tot[1]], [sv_tot[2]]], dtype=np.uint64), (1, 32)) if pv else None
            sig = ecol(SIG.S, f"sig{p}", sig_v)
            SIG.S.zero(sig.ex - sig.shifted(1), sel=SIG.S.lt("z", 31), tag=f"wlu.sigrep.{p}")
            sig_cols.append(sig)
            S_.zero(H.ex * (Ex.const(gam) - comp_ex(s_ex, LOs[p].ex, HIs[p].ex, svs[p].ex, pvs[p].ex))
                    - zsel_v * (Ex.const(1) + As[p].ex * Ex.const(d1) + wlos[p].ex * Ex.const(d2) + whis[p].ex * Ex.const(d3)),
                    tag=f"wlu.val.{p}")
            ind0 = S_.ind(**{a: 0 for a, _ in VSp.axes})
            zv = [(ax, t) for ax, b in VSp.axes for t in range(b)][:5]
            S_.sumzero(H.ex - ind0 * mview(sig, S_, {"z": zv}), tag=f"wlu.sum.{p}")
        T_ = TS.S
        s_ex = T_.idx("s") if prm.sb else Ex.const(0)
        HTv = None
        if pv:
            s_v = TS.grid("s") if prm.sb else np.zeros(TS.n, dtype=np.int64)
            den = CPU.esub(np.stack([np.full(TS.n, gam[l], dtype=np.uint64) for l in (0, 1, 2)]),
                           comp_vals(s_v, LO.v, HI.v, Sb.v, Pb.v))
            num = wsum_vals(m.v, Ma.v, Mlo.v, Mhi.v)
            zax_t = next((a for a, _ in TS.axes if a.startswith("zz")), None)
            if zax_t is not None:
                num = num * (TS.grid(zax_t) == 0).astype(np.uint64)[None, :]
            HTv = CPU.esub(CPU.zeros_e(TS.n), CPU.emul(num, CPU.einv(den)))
        from .mc_zk import ecol, view as mview
        HT = ecol(T_, "HT", HTv)
        zsel = _zsel(TS)
        T_.zero(HT.ex * (Ex.const(gam) - comp_ex(s_ex, LO.ex, HI.ex, Sb.ex, Pb.ex))
                + zsel * (m.ex + Ma.ex * Ex.const(d1) + Mlo.ex * Ex.const(d2) + Mhi.ex * Ex.const(d3)), tag="wlu.tab")
        ind0 = T_.ind(**{a: 0 for a, _ in TS.axes})
        zt = [(ax, t) for ax, b in TS.axes for t in range(b)][:5]
        tot = Ex.const(0)
        for sig in sig_cols:
            tot = tot + mview(sig, T_, {"z": zt})
        T_.sumzero(HT.ex + ind0 * tot, tag="wlu.sum.T")

    cir.rounds.append(rnd)


def _zsel(sp):
    """indicator of zz = 0 on a space with a replication axis zz* (1 if none)."""
    zax = next((a for a, _ in sp.axes if a.startswith("zz")), None)
    if zax is not None:
        return sp.S.ind(**{zax: 0})
    return Ex.const(1)


def _emul(a, b):
    from .field import hmul
    return hmul(a, b)


# ============================================================================ witness assembly, prove / verify, census
def make_wit(prm, kprf, salt, ctx, steps, head_d=None, hidden_kcom=False, hidden_t=False, hidden_tok=False):
    """steps: list of dict(Y, R, ey, a, d, t, tok) (slot vectors or real-V vectors; R holds A_v when head_d is given).
    Returns (stmt, wit, oks)."""
    pre = S3.prefix_state(kprf, ctx)
    W, oks = [], []
    for s_ in steps:
        Y = pad_slots(prm, s_["Y"]); R = pad_slots(prm, s_["R"]); a = pad_slots(prm, s_["a"], 0).astype(bool)
        w, ok = wit_step(prm, Y, R, s_["ey"], a, pre, s_["d"], s_["t"], s_["tok"], head_d=head_d)
        W.append(w); oks.append(ok)
    stmt = make_statement(prm, None if hidden_kcom else kcom_of(kprf, salt), ctx,
                          [s_["d"] for s_ in steps],
                          None if hidden_t else [s_["t"] for s_ in steps],
                          None if hidden_tok else [s_["tok"] for s_ in steps])
    return stmt, dict(kprf=kprf, salt=salt, steps=W, t=[int(s_["t"]) for s_ in steps]), oks


def _grp(name):
    """space group of a vector name: sz.VS0 -> VS (per slot), sz.BS -> BS, ...; tables / LogUp helpers by table."""
    if name.startswith("H:") or name.startswith("mult:"):
        return name.split(":")[0] + ":" + name.split(":")[1].split("#")[0]
    if name.startswith("T:"):
        return "T:" + name.split(".")[0][2:]
    parts = name.split(".")
    if parts[0] == "sz" and len(parts) > 1:
        return "".join(ch for ch in parts[1] if not ch.isdigit())
    return parts[0]


def cells(cir, lookups_done=False):
    """Committed cells (Vec.N, incl. the blinding half / tail; an E-valued vector counts once, as packed in its
    commitment group -- the convention of mc_zk.cells) and data cells (Vec.content; for a LogUp helper H:
    one cell per looked-up content row + table rows), by group.  H vectors that zk_flat creates at proving time
    are sized here from the declared lookups unless lookups_done."""
    out = {}
    for v in cir.vecs:
        if v.name.startswith("H:") and not lookups_done:
            continue
        e = out.setdefault(_grp(v.name), [0, 0])
        e[0] += v.N
        e[1] += v.content
    if not lookups_done:
        for name, lk in cir.lookups.items():
            if lk.table is None:
                continue
            nvt = lk.table[0]
            blocks = [nvt] + [p_[0] for p_ in lk.parts]
            off = 0
            for nb_ in sorted(blocks, reverse=True):
                sz = 1 << nb_
                off = (off + sz - 1) // sz * sz + sz
            nvm = max((off + Z.TB - 1).bit_length(), max(blocks))
            if off > (1 << nvm) - Z.TB:
                nvm += 1
            data = lk.mult.content + sum(1 << (p_[0] - 1) for p_ in lk.parts)
            e = out.setdefault("H:" + name.split("#")[0], [0, 0])
            e[0] += 1 << nvm
            e[1] += data
    tot = [sum(x[0] for x in out.values()), sum(x[1] for x in out.values())]
    return {"total_committed": tot[0], "total_data": tot[1], "by_group": out}


def prove(prm, stmt, wit, B=CPU):
    em = build(prm, stmt, wit)
    t0 = time.time()
    pf = Z.prove(em.ctx.cir, stmt, B, label=LABEL)
    return pf, em, time.time() - t0


def verify(prm, stmt, pf):
    em = build(prm, stmt, None)
    Z.verify(em.ctx.cir, stmt, pf, label=LABEL)
    return True


def try_verify(prm, stmt, pf):
    try:
        verify(prm, stmt, pf)
        return "accepted"
    except Z.ProofError as e:
        return f"rejected: {e}"


def synth_steps(prm, n, seed=0, scale=3.0, ey=-40, rad=2 ** 22, mask_frac=0.0, peaked=False, kprf=None, ctx=None,
                cheat=None):
    """synthetic HEAD outputs (Y ~ N(0, scale) logits at 2^ey, radii ~ rad) with honest engine tokens: the engine
    samples with sampling_v3 on fp32 logits within the certified radius of the centre."""
    rng = np.random.default_rng(seed)
    pre = S3.prefix_state(kprf, ctx)
    out = []
    for i in range(n):
        z = rng.standard_normal(prm.V) * scale
        if peaked and i % 2 == 0:
            z[rng.integers(0, prm.V, 3)] += 15
        Y = np.round(np.ldexp(z, -ey)).astype(np.int64)
        R = (rad * (1 + rng.random(prm.V))).astype(np.int64)
        a = rng.random(prm.V) >= mask_frac
        a[int(np.argmax(z))] = True
        zr = np.ldexp(Y.astype(np.float64), ey)
        zh = (zr + np.ldexp(R.astype(np.float64), ey) * rng.uniform(-0.95, 0.95, prm.V)).astype(np.float32)
        d, t = 0, 1000 + i
        tok = S3.sample(pre, t, zh, prm.rule, allowed=a, d_t=d)
        out.append(dict(Y=Y, R=R, ey=ey, a=a, d=d, t=t, tok=int(tok), zh=zh))
    return out


def census(prm):
    """cells of the circuit from a verifier-mode build (shapes only, no witness); commit-round vectors included."""
    ctx = S3.SampCtx(b"\x01" * 16, b"\x02" * 32)
    stmt = make_statement(prm, [0, 0, 0, 0], ctx, [0] * prm.K, list(range(prm.K)), [0] * prm.K)
    em = build(prm, stmt, None)
    cir = em.ctx.cir
    from .transcript import Transcript
    tr = Transcript(b"census")
    for k, fn in enumerate(cir.rounds):
        cir.cur_phase = 10 + k
        fn(tr)
    cir.cur_phase = 1
    return cells(cir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "prove"])
    ap.add_argument("--V", type=int, default=151936)
    ap.add_argument("--K", default="2")
    ap.add_argument("--T", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rule = S3.Rule(a.T, a.top_p, 32)
    res = []
    for K in [int(k) for k in a.K.split(",")]:
        prm = Prm(a.V, rule, K)
        if a.cmd == "census":
            t0 = time.time()
            c = census(prm)
            r = dict(V=a.V, K=K, rule=rule.as_dict(), census_s=round(time.time() - t0, 1), **c)
        else:
            from .field import backend as get_backend
            B = get_backend(a.backend)
            kprf = [11, 22, 33, 44]; salt = [5, 6, 7, 8]
            ctx = S3.SampCtx(b"\x01" * 16, b"\x02" * 32)
            steps = synth_steps(prm, K, kprf=kprf, ctx=ctx)
            stmt, wit, oks = make_wit(prm, kprf, salt, ctx, steps)
            assert all(oks)
            t0 = time.time()
            em = build(prm, stmt, wit)
            tb = time.time() - t0
            with Z.PoolPeak() as pk:
                t1 = time.time()
                pf = Z.prove(em.ctx.cir, stmt, B, label=LABEL)
                tp = time.time() - t1
            from .transcript import dumps
            r = dict(V=a.V, K=K, rule=rule.as_dict(), build_s=round(tb, 1), prove_s=round(tp, 1),
                     gpu_pool_peak_gib=round(pk.peak / 2 ** 30, 2), proof_bytes=len(dumps(pf)), **cells(em.ctx.cir, True))
        print(json.dumps({k: v for k, v in r.items() if k != "by_group"}), flush=True)
        res.append(r)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
