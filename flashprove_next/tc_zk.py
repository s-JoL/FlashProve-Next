# SPDX-License-Identifier: Apache-2.0
"""Standalone trace-commitment slot proof (specs/10): one sampled attention slot read from a per-request history
matrix H in ZERO KNOWLEDGE.

H (specs/10 §2.1): the request's KV cache lifted to integers (Lift_K, Lift_V, tc_ref), rows = chunks of TC tokens,
columns = (token in chunk, layer, port, group, dim); every row RS-encoded at rate 1/2; the columns hashed with an
unsalted Poseidon2 sponge (domain DOM_HCOL) into a Merkle tree with root rt_H; R = H(DOM_ROOT; s_R, rt_obj, rt_H, T_pad)
is the public trace root (s_R hidden salt, rt_obj = root of the point-object tree, supplied
as an input and not constructed by this module).

Statement (public): tier constants, R, and -- standalone only -- slot_digest, a salted Poseidon2 digest of the hidden
slot data (layer l, position t, scaled query q, engine output o, producer position tau_p and its K/V output).  The
digest stands in for the outer proof that produces these values; in a reply proof they come from the certified QKV
producer (flashprove_next.attn_zk, flashprove_next.reply_zk).

Relation proven (FS order: statement -> phase 1 -> round A -> round B -> round C -> round D -> LogUp / Libra -> openings):
  phase 1  slot vector, causal mask, per-key exp witness (idx, E, Ehi, rem limbs), head / component quantities
           (m, Z, N, M2hi, Q2, the tolerance witness), the hidden H-coordinates (bits of l and tau_p), mask polynomials
  round A  points of the four attention products are drawn; four EXPLICIT masked sumchecks are run (messages public,
           Libra mask rho g with the hidden sum S' = S + rho G revealed); their final evaluations become hidden scalars:
             SC1 score    sum_{tau,h} eq mask z = sum_{tau,g,i} [eq mask](tau) [eq(g) qq](g,i) K(tau,g,i)  -> K claim c1
             SC2 output   N~(g',rho',i')        = sum eq(g',i') E'(tau,g) V(tau,g,i)                   -> V claim c2
             SC3 Jensen   M2hi~(g'',rho'',i'')  = sum eq(g'',i'') Ehi'(tau,g) V^2                     -> V claim c3
             SC4 C-S      Q2~(g''',i''')        = sum eq(g''',i''') mask(tau) V^2  (a4 = mask~(tau*))  -> V claim c4
           plus two producer point reads c5 = k_p~(g_p, i_p), c6 = v_p~(g_p, i_p) (hidden row AND layer coordinates)
  round B  gamma: hidden-selector sumcheck (specs/10 §1.2) of A(x) H(x), A = sum_j gamma^j eq(P_j, x), row
           variables first; ends at the public point r = (r_col, r_row); commits u_H = eq(r_row)^T H, u_px = s^T H
           (proximity combination, s_c = sigma^c) and the chain of A(r) = sum_j gamma^j prod_b eq(P_j[b], r_b)
           computed from the hidden coordinates; in-circuit A(r) u_H~(r_col) + rho g(r) = v_fin
  round C  m_q public columns: hidden columns of M = RS(H) -> Poseidon2 leaves -> Merkle paths -> rt_H -> R (in-circuit)
           and the slot-digest sponge
  round D  beta, mu: batched column check  <u_H + mu u_px, g> = sum_q beta_q sum_c (eq(r_row, c) + mu sigma^c) M[c, j_q]
Tolerance per output component (tc_ref): |o_eng - N/Z| <= A sqrt(Var) + B |N/Z| + Cq SQn + C0 (specs/06 §2.2 row 7,
with the D-9 Jensen / Cauchy-Schwarz bounds and the reference-error terms).

usage:  python -m flashprove_next.tc_zk qwen --data DIR --T 4096 --layer 13 --out FILE [--tamper]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from . import poseidon2 as PS
from . import modelcommit as MC
from . import mc_zk as MZ
from . import tc_ref as R
from .zk_flat import Ex, EqP, TB
from .field import P, CPU, ZERO, ONE, hadd, hsub, hmul, lagrange_eval, rand_field
from .hwinfo import describe as describe_machine
from .transcript import Transcript, canon, dumps

VERSION = "fpn-tc1"
LABEL = b"fpn-tc1-v1"
DOM_HCOL, DOM_ROOT, DOM_SLOT = 25, 26, 27
J_MAX = 8                                   # claim slots of the hidden-selector sumcheck (6 used, 2 zero claims)
SC_DEG = (2, 3, 4, 3)                       # per-variable degree of SC1..SC4
HS_DEG = 2
SCB = 5                                     # replicated-scalar block bits


def _lg(n):
    return max(0, int(n - 1).bit_length())


def _e(t):
    return (int(t[0]) % P, int(t[1]) % P, int(t[2]) % P)


def fe(a):
    """signed integer array -> canonical field elements (uint64)."""
    a = np.asarray(a)
    if a.dtype == object:
        return np.array([int(x) % P for x in a.ravel()], dtype=np.uint64).reshape(a.shape)
    return CPU.fromi(a.astype(np.int64).ravel()).reshape(a.shape)


def ext(a):
    """base field (n,) -> E (3, n)."""
    a = np.asarray(a, dtype=np.uint64).ravel()
    z = np.zeros_like(a)
    return np.stack([a, z, z])


def eq_t(pt):
    return CPU.eq_table([_e(x) for x in pt])


def eqv(a, b):
    out = ONE
    for x, y in zip(a, b):
        x, y = _e(x), _e(y)
        out = hmul(out, hadd(hmul(x, y), hmul(hsub(ONE, x), hsub(ONE, y))))
    return out


class Shape:
    """Public layout constants (functions of the tier only)."""

    def __init__(self, tier):
        t = tier
        self.t = t
        self.KB = t.nc.bit_length() - 1
        self.IB = t.nr.bit_length() - 1
        self.QB = _lg(t.MQ)
        assert self.QB + self.IB <= self.KB, "column block must embed into the u cube"
        assert t.nr % 8 == 0
        self.A = t.nr // 8                                   # absorb rows per column leaf
        self.depth = t.depth
        self.CB = _lg(self.A + self.depth + 1)
        self.INIT = (1 << self.CB) - 1
        self.ROOT = self.A + self.depth - 1
        self.RR = t.MQ << self.CB                            # R preimage rows RR, RR + 1
        self.NB = (4 << t.WB) // 8                           # slot-digest absorb rows (slot vector 4 2^WB entries)
        nbb = self.NB.bit_length() - 1
        self.X0 = ((self.RR + 2 + (2 << nbb) - 1) >> (nbb + 1)) << (nbb + 1)
        self.CLI = self.X0 + self.NB                         # slot-digest init row (bit nbb set)
        self.PSB = (self.CLI + TB).bit_length()
        assert self.CLI < (1 << self.PSB) - TB
        self.nH = t.nH
        assert self.nH <= 32
        self.n_sc = [t.nt + t.GB + t.DB] * 4                 # variables of SC1..SC4 (tau, g, i)

    def xbit(self, p):
        return ("b", p) if p < self.CB else ("q", p - self.CB)


# ============================================================================ the history-matrix commitment
def commit_H(tier, H, B=CPU, log=lambda *a: None):
    """H: (n_r, n_c) int32 host array.  Rows RS-encoded (rate 1/2) on the device, columns absorbed by an unsalted
    Poseidon2 sponge [j, n_r, 0.., DOM_HCOL, TAG] + rows, Merkle tree.  Returns dict(root, levels (host), stats).
    The codeword is not stored: opened columns are re-encoded on demand (columns())."""
    t0 = time.time()
    xp = B.xp
    n, nr = tier.n, tier.nr
    st = xp.zeros((PS.T, n), dtype=np.uint64)
    st[0] = B.asarray(np.arange(n, dtype=np.uint64)); st[1] = np.uint64(nr)
    st[10] = np.uint64(DOM_HCOL); st[11] = np.uint64(PS.TAG_SPONGE)
    st = PS.permute(st, B)
    t_ntt = t_sp = 0.0
    for g0 in range(0, nr, 8):
        t1 = time.time()
        X = _encode_rows(tier, H[g0:g0 + 8], B)
        MC._sync(B); t_ntt += time.time() - t1
        t1 = time.time()
        st[0:8] = B.add(st[0:8], X)
        del X
        st = PS.permute(st, B)
        MC._sync(B); t_sp += time.time() - t1
    t1 = time.time()
    lv = [st[0:4].copy()]
    del st
    while lv[-1].shape[1] > 1:
        a = lv[-1]
        lv.append(PS.compress(xp.ascontiguousarray(a[:, 0::2]), xp.ascontiguousarray(a[:, 1::2]), B))
    levels = [B.tonumpy(x) for x in lv]
    del lv
    MC._sync(B)
    stats = {"ntt_s": round(t_ntt, 3), "sponge_s": round(t_sp, 3), "merkle_s": round(time.time() - t1, 3),
             "total_s": round(time.time() - t0, 3),
             "permutations": int(n * (1 + nr // 8) + (n - 1))}
    return {"root": tuple(int(v) for v in levels[-1][:, 0]), "levels": levels, "stats": stats}


def _encode_rows(tier, rows, B):
    """rows (k, n_c) int -> (k, n) codeword rows on the device."""
    xp = B.xp
    k = rows.shape[0]
    X = xp.zeros((k, tier.n), dtype=np.uint64)
    for r in range(k):
        f = B.asarray(CPU.fromi(np.asarray(rows[r], dtype=np.int64)))
        if MC._is_gpu(B):
            B.ntt(f, tier.n, out=X[r])
        else:
            X[r] = CPU.ntt(f, tier.n)
    return X


def columns(tier, H, Q, B=CPU):
    """M[:, j] for j in Q (re-encoding every row once): (n_r, m_q) uint64 host."""
    out = np.zeros((tier.nr, len(Q)), dtype=np.uint64)
    qd = B.asarray(np.asarray(Q, dtype=np.int64)) if MC._is_gpu(B) else np.asarray(Q, dtype=np.int64)
    for g0 in range(0, tier.nr, 8):
        X = _encode_rows(tier, H[g0:g0 + 8], B)
        out[g0:g0 + X.shape[0]] = B.tonumpy(X[:, qd])
        del X
    return out


def path(levels, j, depth):
    out = []
    for d in range(depth):
        out.append(tuple(int(v) for v in levels[d][:, j ^ 1])); j >>= 1
    return out


def trace_root(s_R, rt_obj, rt_H, T_pad):
    blocks = np.array([list(s_R) + list(rt_obj), list(rt_H) + [T_pad, 0, 0, 0]], dtype=np.uint64).reshape(2, 8, 1)
    return tuple(int(v) for v in PS.sponge(blocks, CPU, domain=DOM_ROOT)[:, 0])


# ============================================================================ slot vector and digest
def slot_vector(tier, s):
    """[seg 0: q_int (h, i) | seg 1: o_int (h, i) | seg 2: k_p (g, i), v_p (g, i) | seg 3: l, t, tau_p, 0 ..]."""
    W = 1 << tier.WB
    v = np.zeros(4 * W, dtype=object)
    v[0:W] = s.q_int.ravel()
    v[W:2 * W] = s.o_int.ravel()
    gd = tier.G * tier.DH
    v[2 * W:2 * W + gd] = s.kp.ravel(); v[2 * W + gd:2 * W + 2 * gd] = s.vp.ravel()
    v[3 * W] = s.layer; v[3 * W + 1] = s.t; v[3 * W + 2] = s.tau_p
    return fe(v)


def slot_digest(tier, sv, salt):
    blocks = np.concatenate([np.array(list(salt) + [0, 0, 0, 0], dtype=np.uint64), sv]).reshape(-1, 8, 1)
    return tuple(int(v) for v in PS.sponge(blocks, CPU, domain=DOM_SLOT)[:, 0])


# ============================================================================ explicit masked sumcheck
def _mask_part(coef, i, n, X, prefix):
    """sum over x_{>i} of g(r_<i, X, x_>i), g(x) = sum_b g_b(x_b), g_b(X) = sum_k coef[b][k] X^k."""
    gi = ZERO
    xp_ = ONE
    for c in coef[i]:
        gi = hadd(gi, hmul(c, xp_)); xp_ = hmul(xp_, (X, 0, 0))
    rest = ZERO
    for b in range(i + 1, n):
        s01 = hadd(coef[b][0], ZERO)
        for c in coef[b]:
            s01 = hadd(s01, c)                               # g_b(0) + g_b(1) = 2 c0 + c1 + ...
        rest = hadd(rest, s01)
    m = n - i - 1
    out = hmul(hadd(prefix, gi), (pow(2, m, P), 0, 0))
    if m >= 1:
        out = hadd(out, hmul(rest, (pow(2, m - 1, P), 0, 0)))
    return out


def mask_sum(coef, n):
    s = ZERO
    for b in range(n):
        s = hadd(s, hadd(coef[b][0], ZERO))
        for c in coef[b]:
            s = hadd(s, c)
    return hmul(s, (pow(2, n - 1, P), 0, 0))


def mask_eval(coef, r):
    s = ZERO
    for b, rb in enumerate(r):
        x = ONE
        for c in coef[b]:
            s = hadd(s, hmul(c, x)); x = hmul(x, rb)
    return s


def esc_prove(B, tables, deg, coef, rho, tr, terms=None):
    """Prover of sum_x [sum_t prod(tables_t)](x) + rho g(x): tables = list of factor lists (one product term each; a
    plain list = one term).  Returns (S', msgs, r, true sum).  Variables are bound LSB first."""
    if terms is None:
        terms = [tables]
    n = int(terms[0][0].shape[1]).bit_length() - 1
    S = ZERO
    for fs in terms:
        prod = fs[0]
        for f in fs[1:]:
            prod = B.emul(prod, f)
        S = hadd(S, B.esum(prod))
    Sp = hadd(S, hmul(rho, mask_sum(coef, n)))
    tr.absorb(canon([list(Sp)]))
    msgs, r = [], []
    prefix = ZERO
    for i in range(n):
        vals = []
        for X in range(deg + 1):
            acc = ZERO
            for fs in terms:
                prod = None
                for f in fs:
                    lo, hi = f[:, 0::2], f[:, 1::2]
                    v = lo if X == 0 else (hi if X == 1 else B.eadd(lo, B.escale(B.esub(hi, lo), (X, 0, 0))))
                    prod = v if prod is None else B.emul(prod, v)
                acc = hadd(acc, B.esum(prod))
            vals.append(hadd(acc, hmul(rho, _mask_part(coef, i, n, X, prefix))))
        msgs.append([list(v) for v in vals])
        tr.absorb(canon(msgs[-1]))
        ri = tr.ext()
        r.append(ri)
        terms = [[B.fold(f, ri) for f in fs] for fs in terms]
        x = ONE; gi = ZERO
        for c in coef[i]:
            gi = hadd(gi, hmul(c, x)); x = hmul(x, ri)
        prefix = hadd(prefix, gi)
    return Sp, msgs, r, S


def esc_verify(Sp, msgs, n, deg, tr, tag):
    """Checks the public chain; returns (r, final value)."""
    if not isinstance(msgs, list) or len(msgs) != n or any(len(m) != deg + 1 for m in msgs):
        raise Z.ProofError(f"{tag}: malformed sumcheck")
    Sp = _e(Sp)
    tr.absorb(canon([list(Sp)]))
    prev = Sp
    r = []
    for i in range(n):
        vals = [_e(v) for v in msgs[i]]
        if hadd(vals[0], vals[1]) != prev:
            raise Z.ProofError(f"{tag}: round {i} sum mismatch")
        tr.absorb(canon([list(v) for v in vals]))
        ri = tr.ext()
        r.append(ri)
        prev = lagrange_eval(vals, ri)
    return r, prev


# ============================================================================ circuit helpers
def var_ref(sp, v):
    for ax, vs in sp._vars.items():
        if ax != "__pad__" and v in vs:
            return (ax, vs.index(v))
    raise KeyError(v)


class Scalars:
    """Hidden E-valued scalars, each a replicated block of 2^SCB entries on its own padded space (claims at random
    points only, so a scalar can be used in many instances without exhausting its blinding rows)."""

    def __init__(self, ctx):
        self.ctx, self.c = ctx, {}

    def add(self, name, val=None):
        sp = self.ctx.space("tcs:" + name, [("x", SCB)])
        vals = None
        if self.ctx.prove:
            v = _e(val)
            vals = np.array([[v[0]] * (1 << SCB), [v[1]] * (1 << SCB), [v[2]] * (1 << SCB)], dtype=np.uint64)
        col = MZ.ecol(sp, "v", vals)
        sp.zero(col.ex - col.shifted(1), sel=sp.lt("x", (1 << SCB) - 1), tag="tc.scalar.rep")
        self.c[name] = col
        return col

    def on(self, name, sp):
        col = self.c[name]
        nb = min(SCB, sp.bits)
        lst = [var_ref(sp, k) for k in range(nb)] + [("c", 0)] * (SCB - nb)
        return MZ.view(col, sp, {"x": lst})


def limbs(sp, name, vals, bits_list, ctx):
    """Committed 16-bit limbs with range lookups (no combined column); returns the list of Cols."""
    G.range_table(ctx)
    out = []
    for k, w in enumerate(bits_list):
        lv = None
        if ctx.prove:
            lv = np.asarray(vals[k], dtype=np.int64)
            if np.any((lv < 0) | (lv >= (1 << w))):
                ctx.violation(f"limb {name}.{k} out of range")
                lv = np.clip(lv, 0, (1 << w) - 1)
        c = sp.col(f"{name}.l{k}", lv)
        wv = np.full(sp.n, w, dtype=np.int64) if ctx.prove else None
        ctx.cir.lookup("rng", sp.nv, [Ex.const(w), c.ex], sel=sp.sel(), vals=[wv, lv] if ctx.prove else None,
                       sel_vals=np.ones(sp.n, dtype=np.int64) if ctx.prove else None)
        out.append(c)
    return out


def rc(sp, ex, vals, bits, name):
    """range check 0 <= ex < 2^bits (gadgets.range_check; vals: object / int64 values on content rows)."""
    v = None
    if sp.ctx.prove:
        v = np.asarray(vals, dtype=object).ravel()
        if len(v) < sp.n:
            v = np.concatenate([v, np.zeros(sp.n - len(v), dtype=object)])
    return G.range_check(sp, ex, v, bits, name)


def col_obj(sp, name, vals):
    """committed base-field column from signed Python / numpy integers (content rows)."""
    v = None
    if sp.ctx.prove:
        a = np.asarray(vals, dtype=object).ravel()
        v = np.zeros(sp.n, dtype=np.uint64)
        v[:len(a)] = fe(a)
    return sp.col(name, v)


# ============================================================================ circuit
def build(tier, stmt, wit=None, B=CPU, info=None):
    sh = Shape(tier)
    t = tier
    pv = wit is not None
    ctx = G.Ctx(pv)
    cir = ctx.cir
    info = {} if info is None else info
    nt, GB, DB, HB, RB, WB = t.nt, t.GB, t.DB, t.HB, t.RB, t.WB
    Hn = t.G * t.R
    S = {}
    s = wit["slot"] if pv else None
    w = s.w if pv else None
    SC = Scalars(ctx)
    rcI = R.radius_constants(t)["ints"]
    Wc = rcI["W"]

    def clk(tag, t0):
        info[tag] = round(info.get(tag, 0.0) + time.time() - t0, 4)

    # ------------------------------------------------------------------ phase 1
    t0 = time.time()
    SL = ctx.space("tcS", [("seg", 2), ("w", WB)])
    slot = SL.col("slot", wit["sv"] if pv else None)

    def slot_on(sp, seg, wmap):
        return MZ.view(slot, sp, {"seg": ("c", seg), "w": wmap})

    TS = ctx.space("tcT", [("tau", nt)])
    mask = TS.col("mask", w["mask"] if pv else None)
    KS = ctx.space("tcK", [("tau", nt), ("h", HB)])
    ex_bits = t.EXP_BITS
    ET, EHT = R.expt(ex_bits)
    G.table(ctx, "exp", [np.arange(1 << ex_bits, dtype=np.int64), ET, EHT])
    idx = KS.col("idx", w["idx"].ravel() if pv else None)
    E = KS.col("E", w["E"].ravel() if pv else None)
    Ehi = KS.col("Ehi", w["Ehi"].ravel() if pv else None)
    ctx.cir.lookup("exp", KS.nv, [idx.ex, E.ex, Ehi.ex], sel=KS.sel(),
                   vals=[w["idx"].ravel(), w["E"].ravel(), w["Ehi"].ravel()] if pv else None,
                   sel_vals=np.ones(KS.n, dtype=np.int64) if pv else None)
    SHb = t.SH
    lw = [16] * (SHb // 16) + ([SHb % 16] if SHb % 16 else [])
    rl = limbs(KS, "rem", [((w["rem"].ravel() >> (16 * k)) & ((1 << wd) - 1)) for k, wd in enumerate(lw)] if pv else None,
               lw, ctx)
    rem_ex = Ex.const(0)
    for k, c in enumerate(rl):
        rem_ex = rem_ex + c.ex * Ex.const(1 << (16 * k))
    D_ex = idx.ex * Ex.const(1 << SHb) + rem_ex - Ex.const(1 << (SHb - 1))
    HSp = ctx.space("tcH", [("h", HB)])
    m_c = col_obj(HSp, "m", w["m"] if pv else None)
    Z_c = col_obj(HSp, "Z", w["Z"] if pv else None)
    Zc_c = col_obj(HSp, "Zc", w["Zc"] if pv else None)
    rz_c = col_obj(HSp, "rz", w["rz"] if pv else None)
    CS = ctx.space("tcC", [("h", HB), ("i", DB)])
    GS = ctx.space("tcG", [("g", GB), ("i", DB)])
    names_cs = ["N", "M2", "oh", "rd", "aoh", "W2", "rW", "SJ", "SQ", "rS"]
    cs = {}
    for nm in names_cs:
        cs[nm] = col_obj(CS, nm, (w["N"] if nm == "N" else w["M2"] if nm == "M2" else w[nm]).ravel() if pv else None)
    Q2_c = col_obj(GS, "Q2", w["Q2"].ravel() if pv else None)
    SQ0_c = col_obj(GS, "SQ0", w["SQ0"].ravel() if pv else None)
    SB = ctx.space("tcB", [("b", 5)])
    hidpos = np.zeros(32, dtype=np.int64)
    hidpos[t.B_L:t.B_L + t.LB] = 1
    hidpos[t.B_TAU:t.B_TAU + nt] = 1
    sbv = None
    if pv:
        sbv = np.zeros(32, dtype=np.int64)
        for k in range(t.LB):
            sbv[t.B_L + k] = (s.layer >> k) & 1
        for k in range(nt):
            sbv[t.B_TAU + k] = (s.tau_p >> k) & 1
    sbits = SB.col("srcbits", sbv)
    MK = ctx.space("tcM", [("sc", 3), ("b", 5), ("k", 3)])
    coefs = {}
    mk_v = None
    nsc = sh.n_sc + [sh.nH]
    degs = list(SC_DEG) + [HS_DEG]
    if pv:
        mk_v = np.zeros((3, 1 << 11), dtype=np.uint64)
        for k in range(5):
            cf = []
            for b in range(nsc[k]):
                row = []
                for kk in range(degs[k] + 1):
                    rf = rand_field(3)
                    v = (int(rf[0]), int(rf[1]), int(rf[2]))
                    row.append(v)
                    pos = (k << 8) | (b << 3) | kk
                    mk_v[0, pos], mk_v[1, pos], mk_v[2, pos] = v
                cf.append(row)
            coefs[k] = cf
    mkc = MZ.ecol(MK, "g", mk_v)
    clk("phase1_build_s", t0)

    # ---- phase-1 constraints that need no challenge
    # slot: q range (no wrap-around of the scores), header used through views
    qv = s.q_int.ravel() if pv else None
    QS = ctx.space("tcQ", [("h", HB), ("i", DB)])
    q_q = slot_on(QS, 0, [("i", k) for k in range(DB)] + [("h", k) for k in range(HB)])
    rc(QS, q_q + Ex.const(1 << 27), (qv.astype(object) + (1 << 27)) if pv else None, 28, "q.rng")
    # mask: boolean, mask(0) = 1, monotone non-increasing, sum = t + 1  ==> mask = [tau <= t]
    TS.zero(mask.ex * (mask.ex - Ex.const(1)), tag="tc.mask.bool")
    TS.zero(mask.ex - Ex.const(1), sel=TS.ind(tau=0), tag="tc.mask.first")
    dlt = mask.ex - mask.shifted(1)
    TS.zero(dlt * (dlt - Ex.const(1)), sel=TS.lt("tau", t.T_pad - 1), tag="tc.mask.mono")
    t_hdr = slot_on(TS, 3, ("c", 1))
    SC.add("t_sc", (s.t, 0, 0) if pv else None)
    TS.zero((SC.on("t_sc", TS) - t_hdr) * TS.ind(tau=0), tag="tc.t_sc")
    TS.sumzero(mask.ex - (t_hdr + Ex.const(1)) * TS.ind(tau=0), tag="tc.mask.sum")
    mask_ks = MZ.view(mask, KS, {"tau": "tau"})
    KS.zero((Ex.const(1) - mask_ks) * E.ex, tag="tc.E.masked")
    # Z_h = sum_tau E;  Z >= 2^EB;  Z = Zc 2^CB + rz
    Z_ks = MZ.view(Z_c, KS, {"h": "h"})
    KS.sumzero(E.ex - Z_ks * KS.ind(tau=0), eq_axes=("h",), tag="tc.Z")
    rc(HSp, Z_c.ex - Ex.const(1 << R.EB), (w["Z"] - (1 << R.EB)) if pv else None, 38, "Z.lo")
    HSp.zero(Z_c.ex - Zc_c.ex * Ex.const(1 << R.CB) - rz_c.ex, tag="tc.Zc")
    rc(HSp, rz_c.ex, w["rz"] if pv else None, R.CB, "rz")
    rc(HSp, Zc_c.ex, w["Zc"] if pv else None, 27, "Zc")
    # tolerance per component
    Zcs = MZ.view(Z_c, CS, {"h": "h"}); Zccs = MZ.view(Zc_c, CS, {"h": "h"})
    gmap = [("h", RB + k) for k in range(GB)]
    Q2cs = MZ.view(Q2_c, CS, {"g": gmap, "i": "i"}); SQ0cs = MZ.view(SQ0_c, CS, {"g": gmap, "i": "i"})
    oi = slot_on(CS, 1, [("i", k) for k in range(DB)] + [("h", k) for k in range(HB)])
    c_ = {k: v.ex for k, v in cs.items()}
    CS.zero(c_["N"] * Ex.const(64) - c_["oh"] * Zcs - c_["rd"], tag="tc.oh")
    if pv:
        Zr = np.repeat(w["Z"].astype(object), t.DH); Zcr = np.repeat(w["Zc"].astype(object), t.DH)
        W_ = {k: (w[k].ravel().astype(object)) for k in ("oh", "rd", "aoh", "W2", "rW", "SJ", "SQ", "rS", "X", "sp", "sm")}
    rc(CS, c_["rd"], W_["rd"] if pv else None, 38, "rd.lo")
    rc(CS, Zcs - Ex.const(1) - c_["rd"], (Zr - 1 - W_["rd"]) if pv else None, 38, "rd.hi")
    CS.zero((c_["aoh"] - c_["oh"]) * (c_["aoh"] + c_["oh"]), tag="tc.aoh")
    rc(CS, c_["aoh"], W_["aoh"] if pv else None, 26, "aoh")
    rc(CS, c_["M2"], w["M2"].ravel() if pv else None, 60, "M2")
    CS.zero(c_["W2"] * Zccs - c_["M2"] - c_["rW"], tag="tc.W2")
    rc(CS, c_["rW"], W_["rW"] if pv else None, 27, "rW.lo")
    rc(CS, Zccs - Ex.const(1) - c_["rW"], (Zcr - 1 - W_["rW"]) if pv else None, 27, "rW.hi")
    rc(CS, c_["W2"], W_["W2"] if pv else None, 36, "W2")
    X_ex = Ex.const(rcI["KH"] << 12) * c_["W2"] - Ex.const(rcI["KL"]) * (c_["aoh"] * c_["aoh"] - Ex.const(2) * c_["aoh"]) + Ex.const(1 << 12)
    rc(CS, c_["SJ"] * c_["SJ"] * Ex.const(1 << 12) - X_ex,
       (W_["SJ"] * W_["SJ"] * 4096 - W_["X"]) if pv else None, 40, "SJ.up")
    rc(CS, X_ex - (c_["SJ"] - Ex.const(1)) * (c_["SJ"] - Ex.const(1)) * Ex.const(1 << 12) - Ex.const(1),
       (W_["X"] - (W_["SJ"] - 1) * (W_["SJ"] - 1) * 4096 - 1) if pv else None, 40, "SJ.tight")
    rc(CS, c_["SJ"], W_["SJ"] if pv else None, 26, "SJ")
    # SQ0 = ceil(sqrt((t + 1) Q2 + 1)), Q2 < 2^(62 - nt) over visible keys only => (t + 1) Q2 + 1 < 2^62 + 1 < p/2 (no wrap)
    rc(GS, Q2_c.ex, w["Q2"].ravel() if pv else None, 62 - nt, "Q2")
    GQv = None
    if pv:
        GQv = (int(np.asarray(w["mask"]).sum()) * w["Q2"].ravel().astype(object) + 1)
        sq0 = w["SQ0"].ravel().astype(object)
    gq_ex = Q2_c.ex * (SC.on("t_sc", GS) + Ex.const(1)) + Ex.const(1)
    rc(GS, SQ0_c.ex * SQ0_c.ex - gq_ex, (sq0 * sq0 - GQv) if pv else None, 34, "SQ0.up")
    rc(GS, gq_ex - (SQ0_c.ex - Ex.const(1)) * (SQ0_c.ex - Ex.const(1)) - Ex.const(1),
       (GQv - (sq0 - 1) * (sq0 - 1) - 1) if pv else None, 34, "SQ0.tight")
    rc(GS, SQ0_c.ex, sq0 if pv else None, 31, "SQ0")
    CS.zero(c_["SQ"] * Zcs - SQ0cs * Ex.const(32) - c_["rS"], tag="tc.SQ")
    rc(CS, c_["rS"], W_["rS"] if pv else None, 38, "rS.lo")
    rc(CS, Zcs - Ex.const(1) - c_["rS"], (Zr - 1 - W_["rS"]) if pv else None, 38, "rS.hi")
    rc(CS, c_["SQ"], W_["SQ"] if pv else None, 20, "SQ")
    rhs = (Ex.const(rcI["A"]) * c_["SJ"] + Ex.const(rcI["B"]) * c_["aoh"] + Ex.const(rcI["Cq"]) * c_["SQ"]
           + Ex.const(rcI["C0"]))
    dd = (oi - c_["oh"]) * Ex.const(1 << Wc)
    if pv and (any(int(x) < 0 for x in W_["sp"]) or any(int(x) < 0 for x in W_["sm"])):
        ctx.violation("attention output outside the certified radius")
    rc(CS, rhs - dd, W_["sp"] if pv else None, 48, "tol.p")
    rc(CS, rhs + dd, W_["sm"] if pv else None, 48, "tol.m")
    # hidden H coordinates: bits of l and tau_p
    SB.zero(sbits.ex * (sbits.ex - Ex.const(1)), tag="tc.sb.bool")
    hp = SB.pub("hidpos", hidpos).ex
    SB.zero((Ex.const(1) - hp) * sbits.ex, tag="tc.sb.zero")
    wl = np.zeros(32, dtype=np.int64); wl[t.B_L:t.B_L + t.LB] = [1 << k for k in range(t.LB)]
    wtp = np.zeros(32, dtype=np.int64); wtp[t.B_TAU:t.B_TAU + nt] = [1 << k for k in range(nt)]
    SB.sumzero(SB.pub("wl", wl).ex * sbits.ex - slot_on(SB, 3, ("c", 0)) * SB.ind(b=0), tag="tc.sb.layer")
    SB.sumzero(SB.pub("wtp", wtp).ex * sbits.ex - slot_on(SB, 3, ("c", 2)) * SB.ind(b=0), tag="tc.sb.taup")
    S.update(slot=slot, mask=mask, E=E, Ehi=Ehi, D_ex=D_ex, m_c=m_c, cs=cs, Q2=Q2_c, sbits=sbits, mkc=mkc)

    # ------------------------------------------------------------------ round A: attention sumchecks
    def round_a(tr):
        t0 = time.time()
        pf = cir.pf
        if pv:
            pf.setdefault("tc", {})["sc"] = []
        pts = {}
        # points: SC1 (tau_r, g_r, rho_r); SC2 (g', rho', i'); SC3 (g'', rho'', i''); SC4 (g''', i'''); producer (g_p, i_p)
        pts[1] = dict(tau=tr.exts(nt), g=tr.exts(GB), r=tr.exts(RB))
        pts[2] = dict(g=tr.exts(GB), r=tr.exts(RB), i=tr.exts(DB))
        pts[3] = dict(g=tr.exts(GB), r=tr.exts(RB), i=tr.exts(DB))
        pts[4] = dict(g=tr.exts(GB), i=tr.exts(DB))
        pts[5] = dict(g=tr.exts(GB), i=tr.exts(DB))
        rhos = tr.exts(4)
        fin = {}
        for k in range(1, 5):
            n = sh.n_sc[k - 1]
            if pv:
                tabs = _sc_tables(t, k, pts[k], s, B)
                Sp, msgs, r, Strue = esc_prove(B, tabs, SC_DEG[k - 1], coefs[k - 1], rhos[k - 1], tr)
                pf["tc"]["sc"].append({"S": list(Sp), "msgs": msgs})
                vfin = lagrange_eval([_e(v) for v in msgs[-1]], r[-1])
                del tabs
            else:
                ent = pf.get("tc", {}).get("sc", [])
                if len(ent) != 4:
                    raise Z.ProofError("missing attention sumchecks")
                Sp = _e(ent[k - 1]["S"])
                r, vfin = esc_verify(Sp, ent[k - 1]["msgs"], n, SC_DEG[k - 1], tr, f"SC{k}")
            fin[k] = dict(Sp=_e(Sp), r=r, v=vfin)
        S["pts"], S["fin"], S["rhos"] = pts, fin, rhos
        # final points split as (i, g, tau) LSB first
        def split(r):
            return r[:DB], r[DB:DB + GB], r[DB + GB:]
        # ---- hidden scalars and their defining constraints
        vals = _round_a_values(t, s, pts, fin, coefs, split) if pv else {}
        for nm in ("S1", "S2", "S3", "S4", "G1", "G2", "G3", "G4", "gr1", "gr2", "gr3", "gr4",
                   "a1", "b1", "c1", "b2", "c2", "b3", "c3", "a4", "c4", "c5", "c6"):
            SC.add(nm, vals.get(nm))
        SCX = ctx.space("tcX", [("x", SCB)])
        X_ = lambda nm: SC.on(nm, SCX)
        # S_k' = S_k + rho_k G_k
        for k in range(1, 5):
            SCX.zero(X_(f"S{k}") + Ex.const(rhos[k - 1]) * X_(f"G{k}") - Ex.const(fin[k]["Sp"]), tag=f"tc.S{k}'")
        # mask sums / evaluations on MK
        for k in range(1, 5):
            gw, rw = _mask_weights(k - 1, nsc[k - 1], degs[k - 1], fin[k]["r"])
            MK.sumzero(Ex.atom(MZ.EVec(f"Gw{k}", gw, MK.vars("k") + MK.vars("b") + MK.vars("sc"))) * mkc.ex
                       - SC.on(f"G{k}", MK) * MK.ind(sc=0, b=0, k=0), tag=f"tc.G{k}")
            MK.sumzero(Ex.atom(MZ.EVec(f"rw{k}", rw, MK.vars("k") + MK.vars("b") + MK.vars("sc"))) * mkc.ex
                       - SC.on(f"gr{k}", MK) * MK.ind(sc=0, b=0, k=0), tag=f"tc.gr{k}")
        kv = KS.vars("h") + KS.vars("tau")
        # SC1: S1 = sum_{tau,h} eq(pt1) mask (m - D);  a1 = [eq(tau_r) mask]~(tau*);  b1 = [eq(g_r) qq]~(g*, i*)
        p1 = list(pts[1]["r"]) + list(pts[1]["g"]) + list(pts[1]["tau"])
        m_ks = MZ.view(m_c, KS, {"h": "h"})
        KS.sumzero(Ex.atom(EqP(kv, p1)) * mask_ks * (m_ks - D_ex) - SC.on("S1", KS) * KS.ind(tau=0, h=0), tag="tc.S1")
        i1, g1, ta1 = split(fin[1]["r"])
        tv = TS.vars("tau")
        TS.sumzero(Ex.atom(EqP(tv, ta1)) * Ex.atom(EqP(tv, pts[1]["tau"])) * mask.ex - SC.on("a1", TS) * TS.ind(tau=0),
                   tag="tc.a1")
        wt1 = _qq_weights(t, pts[1], g1, i1)
        sv_ = SL.vars("w")
        SL.sumzero(Ex.atom(MZ.EVec("wt1", wt1, sv_, fixed=tuple((v, 0) for v in SL.vars("seg")))) * slot.ex
                   - SC.on("b1", SL) * SL.ind(seg=0, w=0), tag="tc.b1")
        SCX.zero(X_("a1") * X_("b1") * X_("c1") + Ex.const(rhos[0]) * X_("gr1") - Ex.const(fin[1]["v"]), tag="tc.SC1.fin")
        # SC2: S2 = N~(g', rho', i');  b2 = E~(tau*, g*, rho');  final eq(g', g*) eq(i', i*) b2 c2
        cv = CS.vars("i") + CS.vars("h")
        for k, Mc, Ec in ((2, cs["N"], E), (3, cs["M2"], Ehi)):
            pk = pts[k]
            CS.sumzero(Ex.atom(EqP(cv, list(pk["i"]) + list(pk["r"]) + list(pk["g"]))) * (Mc.ex - SC.on(f"S{k}", CS)),
                       tag=f"tc.S{k}")
            ik, gk_, tk = split(fin[k]["r"])
            KS.sumzero(Ex.atom(EqP(kv, list(pk["r"]) + list(gk_) + list(tk))) * (Ec.ex - SC.on(f"b{k}", KS)), tag=f"tc.b{k}")
            cpub = hmul(eqv(pk["g"], gk_), eqv(pk["i"], ik))
            cl = X_(f"c{k}") if k == 2 else X_(f"c{k}") * X_(f"c{k}")
            SCX.zero(Ex.const(cpub) * X_(f"b{k}") * cl + Ex.const(rhos[k - 1]) * X_(f"gr{k}") - Ex.const(fin[k]["v"]),
                     tag=f"tc.SC{k}.fin")
        # SC4: S4 = Q2~(g''', i''') = sum_{tau,g,i} eq(g''',i''') mask(tau) V^2;  a4 = mask~(tau*);  final eq eq a4 c4^2
        gv = GS.vars("i") + GS.vars("g")
        GS.sumzero(Ex.atom(EqP(gv, list(pts[4]["i"]) + list(pts[4]["g"]))) * (Q2_c.ex - SC.on("S4", GS)), tag="tc.S4")
        i4, g4, t4 = split(fin[4]["r"])
        TS.sumzero(Ex.atom(EqP(tv, t4)) * mask.ex - SC.on("a4", TS) * TS.ind(tau=0), tag="tc.a4")
        cpub = hmul(eqv(pts[4]["g"], g4), eqv(pts[4]["i"], i4))
        SCX.zero(Ex.const(cpub) * X_("a4") * X_("c4") * X_("c4") + Ex.const(rhos[3]) * X_("gr4") - Ex.const(fin[4]["v"]),
                 tag="tc.SC4.fin")
        # producer reads: c5 = k_p~(g_p, i_p), c6 = v_p~(g_p, i_p) on slot segment 2
        ep = eq_t(list(pts[5]["i"]) + list(pts[5]["g"]))                      # (3, G DH)
        for nm, port in (("c5", 0), ("c6", 1)):
            wv_ = np.zeros((3, 1 << WB), dtype=np.uint64)
            off = port * t.G * t.DH
            wv_[:, off:off + t.G * t.DH] = ep
            SL.sumzero(Ex.atom(MZ.EVec("wp" + nm, wv_, sv_, fixed=((SL.vars("seg")[0], 0), (SL.vars("seg")[1], 1)))) * slot.ex
                       - SC.on(nm, SL) * SL.ind(seg=2, w=0), tag="tc." + nm)
        # claim points for the hidden-selector sumcheck
        S["claims"] = _claim_points(t, fin, pts, split)
        clk("round_a_s", t0)

    # ------------------------------------------------------------------ round B: hidden-selector sumcheck, u_H, u_px, A(r)
    def round_b(tr):
        t0 = time.time()
        pf = cir.pf
        gam = tr.ext(); rho_hs = tr.ext()
        Gam = MZ.epowers(gam, J_MAX)
        claims = S["claims"]
        n = sh.nH
        if pv:
            Sp, msgs, r, A_r, uH, Strue = _hs_prove(t, wit, claims, Gam, coefs[4], rho_hs, tr, B, info)
            pf["tc"]["hs"] = {"S": list(Sp), "msgs": msgs}
            vfin = lagrange_eval([_e(v) for v in msgs[-1]], r[-1])
            wit["_uH"] = uH
        else:
            ent = pf.get("tc", {}).get("hs")
            if not isinstance(ent, dict):
                raise Z.ProofError("missing hidden-selector sumcheck")
            Sp = _e(ent["S"])
            r, vfin = esc_verify(Sp, ent["msgs"], n, HS_DEG, tr, "HS")
        IB, KB = sh.IB, sh.KB
        r_row, r_col = r[:IB], r[IB:]
        rH = list(r_col) + list(r_row)                     # H-index bit order
        sig = tr.ext()
        S.update(r_row=r_row, r_col=r_col, rH=rH, sig=sig, Gam=Gam, rho_hs=rho_hs, Sp_hs=_e(Sp), v_hs=vfin)
        US = ctx.space("tcU", [("k", KB)])
        uH_v = upx_v = None
        if pv:
            uH_v = wit["_uH"]
            upx_v = _combine(t, wit, MZ.epowers(sig, t.nr), B)
        uHc = MZ.ecol(US, "uH", uH_v); upc = MZ.ecol(US, "upx", upx_v)
        # A(r) chain
        AR = ctx.space("tcA", [("j", 3), ("b", 5)])
        C1, C0 = _chain_consts(t, claims, rH)
        pp_v = None
        if pv:
            pp_v = _chain_values(t, claims, rH, wit["slot"])
        ppc = MZ.ecol(AR, "pp", pp_v)
        av = AR.vars("b") + AR.vars("j")
        e_ex = Ex.atom(MZ.EVec("C1", C1, av)) * MZ.view(sbits, AR, {"b": "b"}) + Ex.atom(MZ.EVec("C0", C0, av))
        AR.zero(ppc.ex - e_ex, sel=AR.ind(b=0), tag="tc.chain0")
        AR.zero(ppc.ex - ppc.shifted(-1) * e_ex, sel=AR.rng("b", 1, 32), tag="tc.chain")
        gj = np.zeros((3, 8), dtype=np.uint64); gj[:, :J_MAX] = Gam
        vals = {}
        if pv:
            vals = _round_b_values(t, wit, claims, Gam, coefs[4], r, rH, uH_v, pp_v)
        for nm in ("Ghs", "grhs", "Ar", "uHr"):
            SC.add(nm, vals.get(nm))
        AR.sumzero(Ex.atom(MZ.EVec("gamj", gj, AR.vars("j"), fixed=tuple((v, 1) for v in AR.vars("b")))) * ppc.ex
                   - SC.on("Ar", AR) * AR.ind(j=0, b=0), tag="tc.Ar")
        US.sumzero(Ex.atom(EqP(US.vars("k"), r_col)) * (uHc.ex - SC.on("uHr", US)), tag="tc.uHr")
        gw, rw = _mask_weights(4, n, HS_DEG, r)
        MK.sumzero(Ex.atom(MZ.EVec("Gwhs", gw, MK.vars("k") + MK.vars("b") + MK.vars("sc"))) * mkc.ex
                   - SC.on("Ghs", MK) * MK.ind(sc=0, b=0, k=0), tag="tc.Ghs")
        MK.sumzero(Ex.atom(MZ.EVec("rwhs", rw, MK.vars("k") + MK.vars("b") + MK.vars("sc"))) * mkc.ex
                   - SC.on("grhs", MK) * MK.ind(sc=0, b=0, k=0), tag="tc.grhs")
        SCX = ctx.space("tcY", [("x", SCB)])
        Y = lambda nm: SC.on(nm, SCX)
        cl = Ex.const(0)
        for j, nm in enumerate(("c1", "c2", "c3", "c4", "c5", "c6")):
            cl = cl + Ex.const(_e(Gam[:, j])) * Y(nm)
        SCX.zero(cl + Ex.const(rho_hs) * Y("Ghs") - Ex.const(_e(Sp)), tag="tc.HS.sum")
        SCX.zero(Y("Ar") * Y("uHr") + Ex.const(rho_hs) * Y("grhs") - Ex.const(vfin), tag="tc.HS.fin")
        S.update(US=US, uHc=uHc, upc=upc)
        clk("round_b_s", t0)

    # ------------------------------------------------------------------ round C: columns, Poseidon2, R, slot digest
    def round_c(tr):
        t0 = time.time()
        Q = tr.indices(t.MQ, t.n)
        S["Q"] = Q
        QB, IB = sh.QB, sh.IB
        MS = ctx.space("tcMc", [("q", QB), ("c", IB)])
        M_v = inp = None
        if pv:
            t1 = time.time()
            cols = columns(t, wit["H"], Q, B)                    # (n_r, m_q)
            if wit.get("flip") is not None:
                cols = cols.copy(); cols[wit["flip"], 0] ^= np.uint64(1)
            clk("columns_s", t1)
            paths = [path(wit["levels"], j, t.depth) for j in Q]
            M_v = np.zeros((1 << QB, 1 << IB), dtype=np.uint64)
            M_v[:t.MQ] = cols.T
            M_v = M_v.reshape(-1)
            t1 = time.time()
            inp, roots, Rv, dig = ps_inputs(sh, Q, cols, paths, wit)
            clk("ps_inputs_s", t1)
            if any(tuple(int(x) for x in roots[:, q]) != tuple(wit["rt_H"]) for q in range(t.MQ)):
                ctx.violation("opened column does not authenticate to rt_H")
            if Rv != tuple(int(x) for x in stmt["R"]):
                ctx.violation("trace root mismatch")
            if dig != tuple(int(x) for x in stmt["slot_digest"]):
                ctx.violation("slot digest mismatch")
        M = MS.col("M", M_v)
        PSp = ctx.space("tcPS", [("q", sh.PSB - sh.CB), ("b", sh.CB)], padded=False)
        t1 = time.time()
        reg = G.PoseRegion(PSp, inp, B=B if pv else None)
        clk("poseidon_trace_s", t1)
        I = lambda k: reg.I(k).ex
        O = lambda k: reg.O(k).ex
        Osh = lambda k, d: reg.O(k).shifted(d)
        Ish = lambda k, d: reg.I(k).shifted(d)
        cst = lambda c: Ex.const(c % P)
        nv = PSp.nv
        allv = list(range(nv))

        def pt(x):
            return Ex.atom(Z.SelLt([], 1, fixed=tuple((v, (x >> v) & 1) for v in allv)))

        def frng(lo, hi):
            return Ex.atom(Z.SelLt(allv, hi)) - Ex.atom(Z.SelLt(allv, lo))

        jq = np.zeros(1 << nv, dtype=np.int64); pb = np.zeros(1 << nv, dtype=np.int64)
        for q, j in enumerate(Q):
            jq[(q << sh.CB) + sh.INIT] = j
            for d in range(sh.depth):
                pb[(q << sh.CB) + sh.A + d] = (j >> d) & 1
        jq_ex = PSp.pub("jq", jq).ex; pb_ex = PSp.pub("pbit", pb).ex
        colsel = PSp.lt("q", t.MQ)
        s_init = colsel * PSp.ind(b=sh.INIT)
        PSp.zero(I(0) - jq_ex, sel=s_init, tag="tc.leaf.idx")
        PSp.zero(I(1) - cst(t.nr), sel=s_init, tag="tc.leaf.nr")
        for k in range(2, 10):
            PSp.zero(I(k), sel=s_init, tag="tc.leaf.zero")
        PSp.zero(I(10) - cst(DOM_HCOL), sel=s_init, tag="tc.leaf.dom")
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=s_init, tag="tc.leaf.tag")
        first = PSp.ind(b=0); absorb = PSp.lt("b", sh.A); rest = absorb - first
        s_abs = colsel * absorb
        qmap = [("q", k) for k in range(QB)]
        for k in range(12):
            prev = first * Osh(k, sh.INIT) + rest * Osh(k, -1)
            e = I(k) - prev
            if k < 8:
                imap = [("c", (k >> b) & 1) for b in range(3)] + [("b", b) for b in range(IB - 3)]
                e = e - MZ.view(M, PSp, {"q": qmap, "c": imap}, pad_to=("b", IB - 3))
            PSp.zero(e, sel=s_abs, tag="tc.leaf.absorb")
        s_path = colsel * PSp.rng("b", sh.A, sh.A + sh.depth)
        for k in range(4):
            cur = Osh(k, -1)
            PSp.zero((Ex.const(1) - pb_ex) * (I(k) - cur) + pb_ex * (I(4 + k) - cur), sel=s_path, tag="tc.path")
        for k in (8, 9, 10):
            PSp.zero(I(k), sel=s_path, tag="tc.path.pad")
        PSp.zero(I(11) - cst(PS.TAG_NODE), sel=s_path, tag="tc.path.tag")
        xlast = ((t.MQ - 1) << sh.CB) + sh.ROOT
        for k in range(4):
            if t.MQ > 1:
                PSp.zero(Osh(k, 1 << sh.CB) - O(k), sel=PSp.lt("q", t.MQ - 1) * PSp.ind(b=sh.ROOT), tag="tc.root.chain")
        # R = H(DOM_ROOT; s_R, rt_obj | rt_H, T_pad): rows RR, RR + 1
        RR = sh.RR
        for k in (8, 9):
            PSp.zero(I(k), sel=pt(RR), tag="tc.R.zero")
        PSp.zero(I(10) - cst(DOM_ROOT), sel=pt(RR), tag="tc.R.dom")
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=pt(RR), tag="tc.R.tag")
        # rt_H (the output of the last column's root row xlast) enters the R preimage at row RR + 1.  The constraint
        # is anchored at row xlast with POSITIVE shifts (RR + 1 - xlast, RR - xlast in [1, 2^CB]): a shift of
        # xlast - (RR + 1) <= -TB would leave no random tail row of O in its support (zk_flat.zk_budget_check; T=1024
        # had -30).  Every other shift of this region is -1 or positive and < N - TB (asserted below).
        assert 1 <= RR - xlast <= (1 << sh.CB) and RR + 1 + TB <= (1 << PSp.nv), (RR, xlast)
        for k in range(4):
            PSp.zero(Ish(k, RR + 1 - xlast) - Osh(k, RR - xlast) - O(k), sel=pt(xlast), tag="tc.R.rtH")
        PSp.zero(I(4) - Osh(4, -1) - cst(t.T_pad), sel=pt(RR + 1), tag="tc.R.T")
        for k in range(5, 12):
            PSp.zero(I(k) - Osh(k, -1), sel=pt(RR + 1), tag="tc.R.carry")
        for k in range(4):
            PSp.zero(O(k) - cst(int(stmt["R"][k])), sel=pt(RR + 1), tag="tc.R.out")
        # slot digest: init row CLI, absorb rows X0 .. X0 + NB - 1 (the whole slot vector, 8 entries per row)
        for k in range(4, 10):
            PSp.zero(I(k), sel=pt(sh.CLI), tag="tc.sd.zero")
        PSp.zero(I(10) - cst(DOM_SLOT), sel=pt(sh.CLI), tag="tc.sd.dom")
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=pt(sh.CLI), tag="tc.sd.tag")
        c_first = pt(sh.X0); c_abs = frng(sh.X0, sh.X0 + sh.NB); c_rest = c_abs - c_first
        nbb = sh.NB.bit_length() - 1
        xb = [sh.xbit(p) for p in range(nbb + 1)]
        for k in range(12):
            prev = c_first * Osh(k, sh.CLI - sh.X0) + c_rest * Osh(k, -1)
            e = I(k) - prev
            if k < 8:
                full = [("c", (k >> b) & 1) for b in range(3)] + xb[:nbb]          # slot index bits (WB + 2)
                e = e - MZ.view(slot, PSp, {"w": full[:WB], "seg": full[WB:WB + 2]}, pad_to=xb[nbb])
            PSp.zero(e, sel=c_abs, tag="tc.sd.absorb")
        for k in range(4):
            PSp.zero(O(k) - cst(int(stmt["slot_digest"][k])), sel=pt(sh.X0 + sh.NB - 1), tag="tc.sd.out")
        S.update(M=M, MS=MS)
        clk("round_c_s", t0)

    # ------------------------------------------------------------------ round D: batched column check
    def round_d(tr):
        beta = tr.exts(t.MQ); mu = tr.ext()
        US = S["US"]
        QB, IB, KB = sh.QB, sh.IB, sh.KB
        kv = US.vars("k")
        bq = np.zeros((3, 1 << QB), dtype=np.uint64)
        for q, b in enumerate(beta):
            bq[0, q], bq[1, q], bq[2, q] = b
        bq_us = Ex.atom(MZ.EVec("beta", bq, kv[IB:IB + QB], fixed=tuple((v, 0) for v in kv[IB + QB:])))
        prm = _gq_params(t)
        gq = Ex.atom(MZ.GQ(prm, S["Q"], beta, kv))
        ac = CPU.eadd(eq_t(S["r_row"]), CPU.escale(MZ.epowers(S["sig"], t.nr), mu))
        a_us = Ex.atom(MZ.EVec("acoef", ac, kv[:IB]))
        m_us = MZ.view(S["M"], US, {"q": [("k", IB + k) for k in range(QB)], "c": [("k", k) for k in range(IB)]})
        US.sumzero((S["uHc"].ex + Ex.const(mu) * S["upc"].ex) * gq - a_us * bq_us * m_us, tag="tc.colcheck")

    cir.rounds = [round_a, round_b, round_c, round_d]
    return ctx


def _gq_params(t):
    return MC.Params(TO=t.nc, TI=1, R_MAX=t.nr, RATE_LOG=1, MQ=t.MQ, GROUP=8)


# ============================================================================ prover-side helpers
def _sc_tables(t, k, p, s, B):
    """factor tables of SC k over the cube (tau, g, i) (i LSB): device E arrays (3, T_pad G DH)."""
    T, G_, D = t.T_pad, t.G, t.DH
    w = s.w
    up = (lambda a: B.asarray(np.ascontiguousarray(a))) if MC._is_gpu(B) else (lambda a: np.ascontiguousarray(a))
    if k == 1:
        a = CPU.emul(eq_t(p["tau"]), ext(fe(w["mask"])))                   # (3, T)
        ta = np.repeat(a, G_ * D, axis=1)
        qq = np.zeros((3, G_ * D), dtype=np.uint64)
        er = eq_t(p["r"]); eg = eq_t(p["g"])
        qf = fe(s.q_int.reshape(G_, t.R, D))
        for g in range(G_):
            acc = np.zeros((3, D), dtype=np.uint64)
            for rr in range(t.R):
                acc = CPU.eadd(acc, CPU.emul(np.repeat(er[:, rr:rr + 1], D, axis=1), ext(qf[g, rr])))
            qq[:, g * D:(g + 1) * D] = CPU.emul(acc, np.repeat(eg[:, g:g + 1], D, axis=1))
        tb = np.tile(qq, (1, T))
        tc = ext(fe(s.K.reshape(-1)))
        return [up(ta), up(tb), up(tc)]
    if k in (2, 3):
        egi = CPU.emul(np.repeat(eq_t(p["g"]), D, axis=1), np.tile(eq_t(p["i"]), (1, G_)))   # (3, G D)
        ta = np.tile(egi, (1, T))
        Ew = w["E"] if k == 2 else w["Ehi"]                                  # (T, G R)
        er = eq_t(p["r"])
        Ef = fe(Ew).reshape(T, G_, t.R)
        ep = np.zeros((3, T * G_), dtype=np.uint64)
        for rr in range(t.R):
            ep = CPU.eadd(ep, CPU.emul(np.repeat(er[:, rr:rr + 1], T * G_, axis=1), ext(Ef[:, :, rr].ravel())))
        tb = np.repeat(ep, D, axis=1)
        tv = ext(fe(s.V.reshape(-1)))
        return [up(ta), up(tb), up(tv)] + ([up(tv)] if k == 3 else [])
    # SC4: eq(g, i) mask(tau) V^2 (the causal mask keeps rows above t and padding rows out of Q2)
    egi = CPU.emul(np.repeat(eq_t(p["g"]), D, axis=1), np.tile(eq_t(p["i"]), (1, G_)))
    ta = CPU.emul(np.tile(egi, (1, T)), np.repeat(ext(fe(w["mask"])), G_ * D, axis=1))
    tv = ext(fe(s.V.reshape(-1)))
    return [up(ta), up(tv), up(tv)]


def _mle(vals_e, pt):
    return CPU.edot_dev(eq_t(pt), vals_e)


def _qq_weights(t, p1, g1, i1):
    """wt(h, i) = eq(g*, g) eq(i*, i) eq(g_r, g) eq(rho_r, rho) on the slot's (h, i) index (w bits)."""
    eg = CPU.emul(eq_t(g1), eq_t(p1["g"]))           # (3, G)
    er = eq_t(p1["r"]); ei = eq_t(i1)
    out = np.zeros((3, 1 << t.WB), dtype=np.uint64)
    for g in range(t.G):
        for rr in range(t.R):
            h = g * t.R + rr
            c = hmul(_e(eg[:, g]), _e(er[:, rr]))
            out[:, h * t.DH:(h + 1) * t.DH] = CPU.escale(ei, c)
    return out


def _mask_weights(k, n, deg, r):
    gw = np.zeros((3, 1 << 11), dtype=np.uint64); rw = np.zeros((3, 1 << 11), dtype=np.uint64)
    half = pow(2, n - 1, P)
    for b in range(n):
        x = ONE
        for kk in range(deg + 1):
            pos = (k << 8) | (b << 3) | kk
            gw[0, pos] = (2 if kk == 0 else 1) * half % P
            rw[0, pos], rw[1, pos], rw[2, pos] = x
            x = hmul(x, _e(r[b]))
    return gw, rw


def _round_a_values(t, s, pts, fin, coefs, split):
    w = s.w
    v = {}
    gk = R.ihead(t)
    T = t.T_pad
    for k in range(1, 5):
        v[f"G{k}"] = mask_sum(coefs[k - 1], len(fin[k]["r"]))
        v[f"gr{k}"] = mask_eval(coefs[k - 1], fin[k]["r"])
    # hidden sums from the witness
    p1 = pts[1]
    kpt = list(p1["r"]) + list(p1["g"]) + list(p1["tau"])
    zm = (w["mask"][:, None] * w["z"]).astype(object)                      # = mask (m - D) in-circuit
    v["S1"] = _mle(ext(fe(zm.ravel())), kpt)
    v["S2"] = _mle(ext(fe(w["N"].ravel())), list(pts[2]["i"]) + list(pts[2]["r"]) + list(pts[2]["g"]))
    v["S3"] = _mle(ext(fe(w["M2"].ravel())), list(pts[3]["i"]) + list(pts[3]["r"]) + list(pts[3]["g"]))
    v["S4"] = _mle(ext(fe(w["Q2"].ravel())), list(pts[4]["i"]) + list(pts[4]["g"]))
    i1, g1, ta1 = split(fin[1]["r"])
    v["a1"] = _mle(CPU.emul(eq_t(p1["tau"]), ext(fe(w["mask"]))), ta1)
    v["b1"] = CPU.edot_dev(_qq_weights(t, p1, g1, i1)[:, :t.G * t.R * t.DH], ext(fe(s.q_int.ravel())))
    v["c1"] = _mle(ext(fe(s.K.reshape(-1))), list(i1) + list(g1) + list(ta1))
    for k, Ew in ((2, w["E"]), (3, w["Ehi"])):
        ik, gk_, tk = split(fin[k]["r"])
        v[f"b{k}"] = _mle(ext(fe(Ew.ravel())), list(pts[k]["r"]) + list(gk_) + list(tk))
        v[f"c{k}"] = _mle(ext(fe(s.V.reshape(-1))), list(ik) + list(gk_) + list(tk))
    i4, g4, t4 = split(fin[4]["r"])
    v["c4"] = _mle(ext(fe(s.V.reshape(-1))), list(i4) + list(g4) + list(t4))
    v["a4"] = _mle(ext(fe(w["mask"])), t4)
    pp = list(pts[5]["i"]) + list(pts[5]["g"])
    v["c5"] = _mle(ext(fe(s.kp.reshape(-1))), pp)
    v["c6"] = _mle(ext(fe(s.vp.reshape(-1))), pp)
    return v


def _claim_points(t, fin, pts, split):
    """Per claim slot j: list over H bits b of ('pub', E value) or ('hid', None); unused slots point to a zero entry
    (layer slot L_MAX - 1, which is empty by the profile)."""
    nH = t.nH
    out = []

    def mk(i, g, port, tau, hid_tau):
        P_ = [None] * nH
        for b in range(t.DB):
            P_[b] = ("pub", _e(i[b]))
        for b in range(t.GB):
            P_[t.B_G + b] = ("pub", _e(g[b]))
        P_[t.B_PORT] = ("pub", (port, 0, 0))
        for b in range(t.LB):
            P_[t.B_L + b] = ("hid", None)
        for b in range(t.nt):
            P_[t.B_TAU + b] = ("hid", None) if hid_tau else ("pub", _e(tau[b]))
        return P_
    for k, port in ((1, 0), (2, 1), (3, 1), (4, 1)):
        ik, gk_, tk = split(fin[k]["r"])
        out.append(mk(ik, gk_, port, tk, False))
    out.append(mk(pts[5]["i"], pts[5]["g"], 0, None, True))
    out.append(mk(pts[5]["i"], pts[5]["g"], 1, None, True))
    zx = (t.L_MAX - 1) << t.B_L
    zero = [("pub", ((zx >> b) & 1, 0, 0)) for b in range(nH)]
    while len(out) < J_MAX:
        out.append(zero)
    return out


def _resolve(t, claims, slot):
    """Full claim points (prover): hidden bits filled from the slot's layer / tau_p."""
    out = []
    for P_ in claims:
        q = []
        for b, (kind, v) in enumerate(P_):
            if kind == "pub":
                q.append(v)
            elif t.B_L <= b < t.B_L + t.LB:
                q.append(((slot.layer >> (b - t.B_L)) & 1, 0, 0))
            else:
                q.append(((slot.tau_p >> (b - t.B_TAU)) & 1, 0, 0))
        out.append(q)
    return out


def _hs_prove(t, wit, claims, Gam, coef, rho, tr, B, info):
    """Hidden-selector sumcheck prover (row variables first).  Returns (S', msgs, r (row then col), A_r, u_H, sum)."""
    t0 = time.time()
    Pfull = _resolve(t, claims, wit["slot"])
    sh = Shape(t)
    KB, IB = sh.KB, sh.IB
    J = len(Pfull)
    Hd = wit["H_dev"]
    gpu = MC._is_gpu(B)
    # column folds hc_j(row) = sum_col eq(P_j,col, col) H(row, col)
    Ecol = np.stack([eq_t(Pj[:KB]) for Pj in Pfull], axis=1)                 # (3, J, n_c)
    if gpu:
        hc = MC.tiles_matvec(B, Hd, [0] * J, B.asarray(Ecol))                  # (3, J, n_r)
    else:
        hc = MC.tiles_matvec(B, Hd, [0] * J, Ecol)
    del Ecol
    terms = []
    for j in range(J):
        a = CPU.escale(eq_t(Pfull[j][KB:]), _e(Gam[:, j]))
        aj = B.asarray(a) if gpu else a
        hj = hc[:, j, :]
        hj = B.xp.ascontiguousarray(hj) if gpu else np.ascontiguousarray(hj)
        terms.append([aj, hj])
    S = ZERO
    for fs in terms:
        S = hadd(S, B.esum(B.emul(fs[0], fs[1])))
    n = sh.nH
    Sp = hadd(S, hmul(rho, mask_sum(coef, n)))
    tr.absorb(canon([list(Sp)]))
    msgs, r = [], []
    prefix = ZERO
    gi_at = lambda i, x: _gsum(coef[i], x)

    def round_(terms_, i):
        vals = []
        for X in range(HS_DEG + 1):
            acc = ZERO
            for fs in terms_:
                prod = None
                for f in fs:
                    lo, hi = f[:, 0::2], f[:, 1::2]
                    v = lo if X == 0 else (hi if X == 1 else B.eadd(lo, B.escale(B.esub(hi, lo), (X, 0, 0))))
                    prod = v if prod is None else B.emul(prod, v)
                acc = hadd(acc, B.esum(prod))
            vals.append(hadd(acc, hmul(rho, _mask_part(coef, i, n, X, prefix))))
        return vals
    for i in range(IB):
        vals = round_(terms, i)
        msgs.append([list(v) for v in vals]); tr.absorb(canon(msgs[-1]))
        ri = tr.ext(); r.append(ri)
        terms = [[B.fold(f, ri) for f in fs] for fs in terms]
        prefix = hadd(prefix, gi_at(i, ri))
    alpha = [B.tonumpy(fs[0])[:, 0] for fs in terms]                       # gamma^j eq(P_j,row, r_row)
    erow = eq_t(r[:IB])
    if gpu:
        uH = B.tonumpy(MC.tiles_vecmat(B, Hd, [0], B.asarray(erow.reshape(3, 1, -1)))[:, 0])
    else:
        uH = MC.tiles_vecmat(B, Hd, [0], erow.reshape(3, 1, -1))[:, 0]
    Ar = np.zeros((3, t.nc), dtype=np.uint64)
    for j in range(J):
        Ar = CPU.eadd(Ar, CPU.escale(eq_t(Pfull[j][:KB]), _e(alpha[j])))
    terms = [[B.asarray(Ar) if gpu else Ar, B.asarray(uH) if gpu else uH]]
    for i in range(IB, n):
        vals = round_(terms, i)
        msgs.append([list(v) for v in vals]); tr.absorb(canon(msgs[-1]))
        ri = tr.ext(); r.append(ri)
        terms = [[B.fold(f, ri) for f in fs] for fs in terms]
        prefix = hadd(prefix, gi_at(i, ri))
    info["hs_sumcheck_s"] = round(time.time() - t0, 3)
    return Sp, msgs, r, Ar, uH, S


def _gsum(cf, x):
    s = ZERO; p = ONE
    for c in cf:
        s = hadd(s, hmul(c, p)); p = hmul(p, x)
    return s


def _combine(t, wit, coef_e, B):
    """s^T H for an E-valued row vector (3, n_r)."""
    if MC._is_gpu(B):
        return B.tonumpy(MC.tiles_vecmat(B, wit["H_dev"], [0], B.asarray(coef_e.reshape(3, 1, -1)))[:, 0])
    return MC.tiles_vecmat(B, wit["H_dev"], [0], coef_e.reshape(3, 1, -1))[:, 0]


def _chain_consts(t, claims, rH):
    """e(j, b) = C1(j, b) srcbits(b) + C0(j, b) = eq(P_j[b], r_b) (padding coordinates b >= nH: e = 1)."""
    C1 = np.zeros((3, 256), dtype=np.uint64); C0 = np.zeros((3, 256), dtype=np.uint64)
    for j in range(8):
        for b in range(32):
            pos = j * 32 + b
            if j >= len(claims) or b >= len(rH):
                C0[0, pos] = 1
                continue
            rb = _e(rH[b])
            two_r1 = hsub(hadd(rb, rb), ONE)
            kind, v = claims[j][b]
            if kind == "hid":
                C1[:, pos] = two_r1
                C0[:, pos] = hsub(ONE, rb)
            else:
                C0[:, pos] = hadd(hmul(v, two_r1), hsub(ONE, rb))
    return C1, C0


def _chain_values(t, claims, rH, slot):
    Pfull = _resolve(t, claims, slot)
    pp = np.zeros((3, 256), dtype=np.uint64)
    for j in range(8):
        acc = ONE
        for b in range(32):
            if j < len(Pfull) and b < len(rH):
                acc = hmul(acc, eqv([Pfull[j][b]], [rH[b]]))
            pp[0, j * 32 + b], pp[1, j * 32 + b], pp[2, j * 32 + b] = acc
    return pp


def _round_b_values(t, wit, claims, Gam, coef, r, rH, uH, pp):
    n = len(r)
    v = {"Ghs": mask_sum(coef, n), "grhs": mask_eval(coef, r)}
    Ar = ZERO
    for j in range(J_MAX):
        Ar = hadd(Ar, hmul(_e(Gam[:, j]), _e(pp[:, j * 32 + 31])))
    v["Ar"] = Ar
    sh = Shape(t)
    v["uHr"] = _mle(uH, rH[:sh.KB])
    return v


def ps_inputs(sh, Q, cols, paths, wit):
    """Poseidon2 inputs of every tcPS row; returns (inputs, roots (4, m_q), R, slot digest)."""
    t = sh.t
    n = (1 << sh.PSB) - TB
    inp = np.zeros((12, n), dtype=np.uint64)
    mq = t.MQ
    Qa = np.asarray(Q, dtype=np.uint64)
    base = np.arange(mq, dtype=np.int64) << sh.CB
    st = np.zeros((12, mq), dtype=np.uint64)
    st[0] = Qa; st[1] = t.nr; st[10] = DOM_HCOL; st[11] = PS.TAG_SPONGE
    inp[:, base + sh.INIT] = st
    st = PS.permute(st, CPU)
    for b in range(sh.A):
        st = st.copy()
        st[0:8] = CPU.add(st[0:8], np.ascontiguousarray(cols[8 * b:8 * b + 8]))
        inp[:, base + b] = st
        st = PS.permute(st, CPU)
    cur = st[0:4].copy()
    sib = np.asarray(paths, dtype=np.uint64)
    for d in range(sh.depth):
        bit = ((Qa >> np.uint64(d)) & np.uint64(1)).astype(bool)
        s_ = np.ascontiguousarray(sib[:, d, :].T)
        st = np.zeros((12, mq), dtype=np.uint64)
        st[0:4] = np.where(bit, s_, cur); st[4:8] = np.where(bit, cur, s_); st[11] = PS.TAG_NODE
        inp[:, base + sh.A + d] = st
        cur = PS.permute(st, CPU)[0:4]
    roots = cur
    rt = roots[:, mq - 1]
    stA = np.zeros((12, 1), dtype=np.uint64)
    stA[0:4, 0] = wit["s_R"]; stA[4:8, 0] = wit["rt_obj"]; stA[10] = DOM_ROOT; stA[11] = PS.TAG_SPONGE
    inp[:, sh.RR] = stA[:, 0]
    stB = PS.permute(stA, CPU).copy()
    stB[0:4, 0] = CPU.add(stB[0:4, 0], rt)
    stB[4, 0] = CPU.add(stB[4:5, 0], np.array([t.T_pad], dtype=np.uint64))[0]
    inp[:, sh.RR + 1] = stB[:, 0]
    Rv = tuple(int(v) for v in PS.permute(stB, CPU)[0:4, 0])
    st = np.zeros((12, 1), dtype=np.uint64)
    st[0:4, 0] = wit["salt_sd"]; st[10] = DOM_SLOT; st[11] = PS.TAG_SPONGE
    inp[:, sh.CLI] = st[:, 0]
    st = PS.permute(st, CPU)
    blocks = np.asarray(wit["sv"], dtype=np.uint64).reshape(sh.NB, 8)
    for rho in range(sh.NB):
        st = st.copy()
        st[0:8, 0] = CPU.add(st[0:8, 0], blocks[rho])
        inp[:, sh.X0 + rho] = st[:, 0]
        st = PS.permute(st, CPU)
    dig = tuple(int(v) for v in st[0:4, 0])
    return inp, roots, Rv, dig


# ============================================================================ prove / verify
def statement(tier, Rv, sd):
    return {"v": VERSION, "tier": asdict(tier), "R": [int(x) for x in Rv], "slot_digest": [int(x) for x in sd]}


def cells(cir):
    out = {}
    for v in cir.vecs:
        sp = v.name.split(".")[0]
        sp = "tcs:*" if sp.startswith("tcs:") else sp
        out[sp] = out.get(sp, 0) + v.N
    return out


def content_cells(cir):
    out = {}
    for v in cir.vecs:
        sp = v.name.split(".")[0]
        sp = "tcs:*" if sp.startswith("tcs:") else sp
        out[sp] = out.get(sp, 0) + v.content
    return out


def prove(tier, stmt, wit, B=CPU):
    info = {}
    t0 = time.time()
    ctx = build(tier, stmt, wit, B, info)
    ctx.cir.pf = None
    pf = _prove(ctx.cir, stmt, B)
    info["prove_s"] = round(time.time() - t0, 3)
    info["cells_by_space"] = cells(ctx.cir)
    info["content_by_space"] = content_cells(ctx.cir)
    info["cells"] = int(sum(info["cells_by_space"].values()))
    info["vectors"] = len(ctx.cir.vecs)
    info["zk_budget"] = dict(Z.ZK_REPORT)
    info["violations"] = list(ctx.notes)
    return pf, info


def _prove(cir, stmt, B):
    return Z.prove(cir, stmt, B, label=LABEL)


def verify(tier, stmt, pf):
    ctx = build(tier, stmt, None)
    return Z.verify(ctx.cir, stmt, pf, label=LABEL)


def try_verify(tier, stmt, pf):
    try:
        verify(tier, stmt, pf)
        return "accepted"
    except Z.ProofError as e:
        return "rejected: " + str(e)
    except (KeyError, ValueError, IndexError, TypeError) as e:
        return f"rejected: malformed proof ({type(e).__name__}: {e})"


# ============================================================================ witness assembly
def make_witness(tier, H, com, s, B=CPU, salts=None):
    """H (n_r, n_c) int32 committed matrix (com = commit_H(...)), slot s (tc_ref.Slot with witness computed)."""
    rnd = lambda: tuple(int(x) for x in rand_field(4))
    s_R, rt_obj, salt_sd = (salts or (rnd(), rnd(), rnd()))
    Rv = trace_root(s_R, rt_obj, com["root"], tier.T_pad)
    sv = slot_vector(tier, s)
    sd = slot_digest(tier, sv, salt_sd)
    if MC._is_gpu(B):
        Hd = B.xp.asarray(np.ascontiguousarray(H, dtype=np.int32).reshape(1, tier.nr, tier.nc))
    else:
        Hd = np.ascontiguousarray(H, dtype=np.int32).reshape(1, tier.nr, tier.nc)
    wit = {"slot": s, "sv": sv, "H": H, "H_dev": Hd, "levels": com["levels"], "rt_H": com["root"], "s_R": s_R,
           "rt_obj": rt_obj, "salt_sd": salt_sd}
    return statement(tier, Rv, sd), wit


def shape_sig(x, _q=False):
    return MZ.shape_sig(x, _q)


# ============================================================================ Qwen3-0.6B driver (GPU)
def _rss_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def _exact_attn(K, V, q_u16, l, keys, scaling, kl=None):
    """exact-softmax attention of layer l's query over `keys` of layer kl (bf16 inputs), rounded to bf16 codes:
    the output of a simulated cheating engine (truncated window / wrong layer)."""
    kl = l if kl is None else kl
    q = R.bf16(q_u16).astype(np.float64)
    n_q, dh = q.shape
    grp = n_q // K.shape[1]
    out = np.zeros((n_q, dh))
    for h in range(n_q):
        g = h // grp
        z = scaling * (K[kl, g, keys].astype(np.float64) @ q[h])
        p = np.exp(z - z.max()); p /= p.sum()
        out[h] = p @ V[kl, g, keys].astype(np.float64)
    u = out.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + ((u >> 16) & 1) + 0x7FFF) >> 16).astype(np.uint16)


def _one(tier, H, com, slot, B, reps=1, force=False, label="honest", mutate=None, keep=False):
    import cupy as cp
    R.witness(slot, check=not force)
    stmt, wit = make_witness(tier, H, com, slot, B)
    if mutate:
        mutate(stmt, wit)
    runs = []
    pf = None
    for rep in range(reps):
        cp.get_default_memory_pool().free_all_blocks()
        Z.FORCE = force
        try:
            with Z.PoolPeak() as pk:
                pf, info = prove(tier, stmt, wit, B)
        finally:
            Z.FORCE = False
        info["gpu_pool_peak_gib"] = round(pk.peak / (1 << 30), 3)
        blob = dumps(pf)
        info["proof_bytes"] = len(blob)
        t1 = time.time()
        info["verify"] = try_verify(tier, stmt, pf)
        info["verify_cpu_s"] = round(time.time() - t1, 3)
        info["rep"] = rep
        info["max_ratio"] = float(slot.w["ratio"].max())
        runs.append(info)
        print(json.dumps({"case": label, **{k: v for k, v in info.items() if k not in ("cells_by_space", "content_by_space")}}),
              flush=True)
        if not keep:
            pf = None
    del wit
    cp.get_default_memory_pool().free_all_blocks()
    return runs, pf


def cmd_qwen(a):
    from .field import gpu_backend
    import cupy as cp
    B = gpu_backend()
    d = np.load(os.path.join(a.data, f"kv_T{a.T}.npz"))
    meta = json.loads(str(d["meta"]))
    tier = R.TIERS[a.T]
    sh = Shape(tier)
    out = {"model": meta["model"], "machine": a.machine or describe_machine(), "version": VERSION, "T": a.T, "layer": a.layer,
           "engine": meta["engine"], "tier": asdict(tier),
           "geometry": {"n_r": tier.nr, "n_c": tier.nc, "N_H": tier.NH, "n": tier.n, "depth": tier.depth,
                        "variables_H": tier.nH, "column_block_rows": 1 << sh.CB, "poseidon_rows": 1 << sh.PSB}}
    t0 = time.time()
    K = R.bf16(d["K"]); V = R.bf16(d["V"])
    H = R.build_H(tier, K, V)
    out["build_H_s"] = round(time.time() - t0, 2)
    cp.get_default_memory_pool().free_all_blocks()
    with Z.PoolPeak() as pk:
        com = commit_H(tier, H, B)
    out["commit"] = dict(com["stats"], gpu_pool_peak_gib=round(pk.peak / (1 << 30), 3), root=list(com["root"]))
    print(json.dumps({"commit": out["commit"]}), flush=True)
    rcon = R.radius_constants(tier)
    out["radius"] = {k: v for k, v in rcon.items() if k != "ints"}
    out["radius_ints"] = rcon["ints"]
    scaling = meta["scaling"]
    tau_p = a.T // 3
    mk = lambda Hm=H, oo=None, l=a.layer: R.slot_from_capture(tier, Hm, l, d["q"][l], d["o"][l] if oo is None else oo,
                                                             a.T - 1, scaling, tau_p)
    runs, _ = _one(tier, H, com, mk(), B, reps=a.reps)
    out["runs"] = runs
    out["cells"] = runs[0]["cells"]
    out["cells_by_space"] = runs[0]["cells_by_space"]
    out["content_by_space"] = runs[0]["content_by_space"]
    if a.tamper:
        tam = {}
        L = a.layer
        Hv = H.reshape(tier.T_pad, tier.L_MAX, 2, tier.G, tier.DH)
        # zeroed KV rows (keys 1..63 of the slot's layer): H commits zeros, attention witness consistent with H
        Hz = Hv.copy(); Hz[1:64, L] = 0; Hz = Hz.reshape(tier.nr, tier.nc)
        comz = commit_H(tier, Hz, B)
        r_, _ = _one(tier, Hz, comz, mk(Hm=Hz), B, force=True, label="zeroed_kv_rows")
        tam["zeroed_kv_rows"] = {k: r_[0][k] for k in ("verify", "violations", "max_ratio", "prove_s")}
        del Hz, comz
        # producer read of a zeroed row: engine output + attention consistent with the zeroed H, producer honest
        Hp = Hv.copy(); Hp[tau_p, L] = 0; Hp = Hp.reshape(tier.nr, tier.nc)
        comp = commit_H(tier, Hp, B)
        Kp = K.copy(); Vp = V.copy(); Kp[L, :, tau_p] = 0; Vp[L, :, tau_p] = 0
        op = _exact_attn(Kp, Vp, d["q"][L], L, np.arange(a.T), scaling)
        del Kp, Vp
        sp = mk(Hm=Hp, oo=op)
        sp.kp = Hv[tau_p, L, 0].astype(np.int64); sp.vp = Hv[tau_p, L, 1].astype(np.int64)
        r_, _ = _one(tier, Hp, comp, sp, B, force=True, label="zeroed_row_producer_true")
        tam["zeroed_row_producer_true"] = {k: r_[0][k] for k in ("verify", "violations", "max_ratio", "prove_s")}
        del Hp, comp
        # truncated window: the engine dropped keys 0..63
        ot = _exact_attn(K, V, d["q"][L], L, np.arange(64, a.T), scaling)
        r_, _ = _one(tier, H, com, mk(oo=ot), B, force=True, label="truncated_window")
        tam["truncated_window"] = {k: r_[0][k] for k in ("verify", "violations", "max_ratio", "prove_s")}
        # wrong layer's KV: engine output and attention witness from layer L + 1, the slot says L
        ow = _exact_attn(K, V, d["q"][L], L, np.arange(a.T), scaling, kl=L + 1)
        sw = R.slot_from_capture(tier, H, L + 1, d["q"][L], ow, a.T - 1, scaling, tau_p)
        sw.layer = L
        r_, _ = _one(tier, H, com, sw, B, force=True, label="wrong_layer_kv")
        tam["wrong_layer_kv"] = {k: r_[0][k] for k in ("verify", "violations", "max_ratio", "prove_s")}
        # one bit flipped in one opened column entry
        r_, _ = _one(tier, H, com, mk(), B, force=True, label="flipped_column_bit",
                     mutate=lambda st, w: w.__setitem__("flip", 5))
        tam["flipped_column_bit"] = {k: r_[0][k] for k in ("verify", "violations", "max_ratio", "prove_s")}
        out["tamper"] = tam
    out["peak_rss_gib"] = round(_rss_gib(), 2)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("runs", "cells_by_space", "content_by_space")}, default=str),
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qwen")
    q.add_argument("--data", required=True); q.add_argument("--T", type=int, default=4096)
    q.add_argument("--layer", type=int, default=13); q.add_argument("--reps", type=int, default=2)
    q.add_argument("--tamper", action="store_true"); q.add_argument("--out", default=None)
    q.add_argument("--machine", default=None, help="host description (default: detected)")
    a = ap.parse_args(argv)
    {"qwen": cmd_qwen}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
