# SPDX-License-Identifier: Apache-2.0
"""In-circuit certified-tolerance checks for per-GEMM blocks of a dense Qwen3-style layer (specs/06 §1-§2, specs/05 §1.3).

A fixed relative tolerance per layer is not complete for native floating point (specs/04 §3), so every gadget

  * recomputes the reference of its block from the OBSERVED (committed) inputs and the committed weights,
  * propagates the certified box radius of specs/06 §2.2 from the declared rounding contract (R32 tier: fp32 GEMM
    reductions; SiLU floor 2^-120; attention row 7 with P rounding; LSE observation),
  * checks |observed - reference| <= radius componentwise,

with exact integer arithmetic in the Goldilocks field: every relation is an integer identity whose two sides are
statically bounded by 2^62 < p/2 (``bits`` tracking, asserted at build time), every rounding is a floor with a
range-checked remainder, and every approximation (canonical weight decode, truncated activations, rsqrt / exp / ln /
sigmoid tables, rescaling) is enclosed by an explicit error term that is ADDED to the radius (specs/06 §2.1: the
reference's own deviation joins the box part).  Non-finite codes are not in the decode tables, and values out of
the representable range make the relation unsatisfiable: both are "bad".

Representation.  A reference quantity is a pair (center Y, radius R) of integers at a common power-of-two scale
2^e (per vector, per head, or per element; e is a hidden exponent column).  Meaning: the engine value lies in
[(Y - R) 2^e, (Y + R) 2^e].  Observed bf16 / fp32 values are decoded by table lookup into (sign, significand,
exponent) and aligned to the center's scale by a hidden shift (floor; the < 1 ulp floor error is charged to the
check, ``obs_cmp``).  GEMM inputs are aligned to one fixed-point scale per vector (``align``); elements below the
grid get a truncation flag t that enters the radius through Z = |X| + kappa t.

Weights (``WeightProvider``).  A GEMM y = W x is checked through two inner-product relations on the matrix cube:
Y = W^cd X and A = W^r Z, where W^cd is the specs/09 canonical decode (floor(w 2^G), |W^cd| < 2^22) and
W^r = |W^cd| + kappa_n [w 2^G not an integer] is the *radius matrix* (one more committed matrix per tensor in the
specs/09 path; it encloses both the Dot-contract term gamma |W||x| and the canonical-decode truncation, see ``gemm``).
The provider supplies the two matrix atoms on the matrix cube: ``PublicWeights`` (clear oracle: verifier evaluates
the MLE itself; used for standalone soundness tests and cost measurement) or ``CommittedWeights`` (hidden witness
matrices; the specs/09 HCO path replaces them by its hidden claims W~(xi, zeta) -- the circuit only ever needs these two
evaluations per tensor).

Per-slot quantities that depend on the (hidden) sampled position / layer are hidden committed columns in the
zero-knowledge reply circuit, never public constants chosen by the caller: the RoPE rows (``rope_hidden``: hidden
position scalar, LogUp into one public table), the q/k score exponents (``qk_exp_hidden``: exactly qk_exponent of the
hidden gain vector; ``blk_att`` takes them as hidden scalars inside a public profile window), the tensor exponents G
(``flashprove_next.wsum_zk``, Tt table).  The int / public-row paths below stay for the clear-oracle providers.

One code path computes the witness (numpy, exact int64) and, when ``Em.emit``, declares the constraints; with
``emit = False`` it is a fast integer evaluator used to score every captured block (``verdict``).
"""
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from .zk_flat import Ex

SCAL = "z"                   # axis of scalar spaces
LIM = 62                     # every integer identity keeps both sides below 2^62 < p/2
P2LO, P2HI = -61, 61         # pow2 table range (2^61 < 2^LIM)


# ============================================================================ contract (specs/06 §1.1 / §1.2, R32 tier)
@dataclass
class Contract:
    """Rounding contract (specs/06 §1.1 templates), parameters of the specs/06 §1.2 example C_max, R32 tier (fp32 partial sums).
    Includes the attention row 7 (P rounding), LSE and fp32-logit entries."""
    u_store: Fraction = Fraction(1, 1 << 8)          # Store_bf16
    u_f32: Fraction = Fraction(1, 1 << 24)           # Store_fp32 (logits, specs/06 §1.2)
    eta_log2: int = -126                             # eta for every store / dot (FTZ-safe)
    u_acc: Fraction = Fraction(1, 1 << 24)           # fp32 accumulation
    h_div: int = 16
    h_add: int = 64
    u_elem: Fraction = Fraction(1, 1 << 21)          # silu / rsqrt / exp / log, 4 ulp fp32
    eta_silu_log2: int = -120                        # torch silu overflow-to-zero path (specs/06 §1.1)
    u_P: Fraction = Fraction(1, 1 << 8)              # P stored bf16 before PV
    B_c: int = 64                                    # FlashAttention key tile
    split_steps: int = 8                             # S_kv split / combine steps
    silu_L: Fraction = Fraction(10998, 10000)        # sup |silu'|

    def h(self, n):
        return -(-n // self.h_div) + self.h_add

    def gamma(self, n):
        hu = self.h(n) * self.u_acc
        return hu / (1 - hu)


C0 = Contract()


def up(c, k=30):
    """Upper dyadic bound ceil(c 2^k) for a nonnegative rational / float c (returns the integer numerator)."""
    c = Fraction(c)
    assert c >= 0
    return -((-c.numerator << k) // c.denominator) if k >= 0 else -(-c.numerator // (c.denominator << -k))


def bitlen(x):
    return int(x).bit_length()


# ============================================================================ spaces and values
def _idx(dst_axes, src_axes, amap=None):
    """Flat source row index for every destination row.  amap: src axis -> dst axis name | ('c', value) |
    list of per-bit (dst axis, bit) / ('c', bit), LSB first (the flashprove_next.gadgets.Col.on convention)."""
    amap = amap or {}
    nb = sum(b for _, b in dst_axes)
    r = np.arange(1 << nb, dtype=np.int64)
    pos, p = {}, nb
    for nm, b in dst_axes:
        p -= b
        pos[nm] = (p, b)
    out = np.zeros(1 << nb, dtype=np.int64)
    q = sum(b for _, b in src_axes)
    for nm, b in src_axes:
        q -= b
        t = amap.get(nm, nm)
        if isinstance(t, list):
            assert len(t) == b, (nm, t)
            for k, tt in enumerate(t):
                if tt[0] == "c":
                    out |= np.int64(tt[1] & 1) << (q + k)
                else:
                    p0, bb = pos[tt[0]]
                    out |= ((r >> (p0 + tt[1])) & 1) << (q + k)
        elif isinstance(t, tuple):
            assert t[0] == "c"
            out |= np.int64(t[1]) << q
        else:
            p0, bb = pos[t]
            assert bb == b, (nm, t, bb, b)
            out |= ((r >> p0) & ((1 << b) - 1)) << q
    return out


class Sp:
    """A hypercube with named axes (MSB first).  S: the flashprove_next.gadgets.Space (padded: the upper half of every vector
    is uniform blinding) when emitting constraints."""

    def __init__(self, em, name, axes):
        self.em, self.name = em, name
        self.axes = [(a, int(b)) for a, b in axes]
        self.bits = sum(b for _, b in self.axes)
        self.n = 1 << self.bits
        self.shape = tuple(1 << b for _, b in self.axes)
        self.S = em.ctx.space(name, self.axes) if em.emit else None

    @property
    def in_bits(self):
        return self.bits - self.em.bbits

    @property
    def in_n(self):
        return 1 << self.in_bits

    def has(self, ax):
        return any(a == ax for a, _ in self.axes)

    def grid(self, ax):
        """index of axis ax for every row (flat)."""
        p = self.bits
        for nm, b in self.axes:
            p -= b
            if nm == ax:
                return (np.arange(self.n, dtype=np.int64) >> p) & ((1 << b) - 1)
        raise KeyError(ax)


class X:
    """An integer vector on a space: committed column (col), or a linear expression (ex) of committed columns.
    v: int64 values (prover / evaluator), bits: static bound |v| < 2^bits."""
    __slots__ = ("sp", "col", "_ex", "_v", "bits", "nm", "_views")

    def __init__(self, sp, v, bits, col=None, ex=None, nm=""):
        self.sp, self.bits, self.col, self._ex, self.nm = sp, int(bits), col, ex, nm
        # When backed by an emitted Col, delegate .v to col.v (compact Vec.vals) to avoid an 8 B/cell duplicate
        self._v = None if (col is not None and sp.em.emit) else v
        self._views = {}
        assert self.bits <= LIM + 1, (nm, bits)

    @property
    def v(self):
        return self.col.v if (self._v is None and self.col is not None) else self._v

    @v.setter
    def v(self, val):
        self._v = val

    @property
    def ex(self):
        return self.col.ex if self.col is not None else self._ex


def _i64(v):
    a = np.asarray(v)
    if a.dtype == object:
        a = np.array([int(t) for t in a.ravel()], dtype=np.int64)
    return a.astype(np.int64, copy=False)


class Em:
    """Builder: evaluator (emit = False) or constraint emitter (emit = True, prove = True for the prover).
    baxes: optional leading batch axes (e.g. [('s', sb)]) prepended to every space declared on this Em."""

    def __init__(self, emit=False, prove=True, baxes=(), ctx=None):
        self.emit, self.pv = emit, prove
        self.ctx = (G.Ctx(prove) if ctx is None else ctx) if emit else None
        self.baxes = [(a, int(b)) for a, b in baxes]
        self.bbits = sum(b for _, b in self.baxes)
        self.fails = {}
        self.stats = {}
        self._n = 0

    def sub_em(self, baxes=()):
        """Child Em sharing the SAME Ctx, fails, stats, and counter, with its own leading batch axes `baxes`."""
        ch = type(self).__new__(type(self))
        ch.emit, ch.pv, ch.ctx = self.emit, self.pv, self.ctx
        ch.baxes = [(a, int(b)) for a, b in baxes]
        ch.bbits = sum(b for _, b in ch.baxes)
        ch.fails, ch.stats = self.fails, self.stats
        ch._parent = getattr(self, "_parent", self)
        for attr in ("cfg_ext", "vec_ext", "B"):
            if hasattr(self, attr):
                setattr(ch, attr, getattr(self, attr))
        return ch

    # ---------------------------------------------------------------- bookkeeping
    def nm(self, base):
        root = getattr(self, "_parent", self)
        root._n += 1
        return f"{base}.{root._n}"

    def fail(self, tag, cnt):
        cnt = int(cnt)
        if cnt:
            self.fails[tag] = self.fails.get(tag, 0) + cnt

    @property
    def ok(self):
        return not self.fails

    def space(self, name, axes):
        """axes == []: a scalar space (32 replicated rows on axis 'z' when unbatched; baxes when batched)."""
        if self.baxes:
            return Sp(self, self.nm(name), self.baxes + list(axes))
        if not axes:
            axes = [(SCAL, 5)]
        return Sp(self, self.nm(name), axes)

    # ---------------------------------------------------------------- columns and expressions
    def col(self, sp, name, v, bits):
        nm = self.nm(name)
        v = _i64(v) if (self.pv and v is not None) else None
        if v is not None:
            assert v.shape == (sp.n,), (nm, v.shape, sp.n)
        c = sp.S.col(nm, v) if self.emit else None
        return X(sp, v, bits, col=c, nm=nm)

    def const(self, sp, value):
        """public constant on a space (an Ex, no committed cells)."""
        v = np.full(sp.n, int(value), dtype=np.int64) if self.pv else None
        return X(sp, v, max(1, bitlen(abs(int(value))) + 1), ex=Ex.const(int(value)) if self.emit else None)

    def pub(self, sp, name, v, bits):
        v = _i64(v)
        c = sp.S.pub(self.nm(name), v) if self.emit else None
        return X(sp, v if self.pv else None, bits, col=c)

    def lin(self, terms, const=0, sp=None, bits=None):
        """sum c_k x_k + const (no new column).  terms: list of (int coefficient, X) on one space."""
        sp = sp or terms[0][1].sp
        b = abs(int(const))
        ex = Ex.const(int(const)) if self.emit else None
        v = np.full(sp.n, int(const), dtype=np.int64) if self.pv else None
        for c, x in terms:
            c = int(c)
            if x.sp is not sp:                   # broadcast (group / scalar values onto a finer space)
                x = self.view(x, sp)
            b += abs(c) << x.bits
            if self.emit:
                ex = ex + x.ex * Ex.const(c)
            if self.pv:
                v = v + c * x.v
        return X(sp, v, bits if bits is not None else max(1, bitlen(b)), ex=ex)

    def commit(self, x, name="cp"):
        """committed copy of an expression."""
        if x.col is not None:
            return x
        c = self.col(x.sp, name, x.v, x.bits)
        if self.emit:
            x.sp.S.zero(c.ex - x.ex, tag=c.nm)
        return c

    def view(self, x, sp, amap=None):
        """x viewed on space sp (broadcast along axes x does not have; by-name axis map, or amap)."""
        if x.sp is sp and not amap:
            return x
        amap = dict(amap or {})
        for a, b in x.sp.axes:                   # own axes missing on the target: constant 0 (scalars, groups)
            if a not in amap and not sp.has(a):
                amap[a] = ("c", 0)
        key = (id(sp), repr(sorted(amap.items())))
        if key in x._views:
            return x._views[key]
        v = x.v[_idx(sp.axes, x.sp.axes, amap)] if self.pv else None
        ex = None
        if self.emit:
            if x.col is None:
                xc = self.commit(x)
                x.col = xc.col
            ex = x.col.on(sp.S, amap)
        y = X(sp, v, x.bits, ex=ex, nm=x.nm + "@" + sp.name)
        x._views[key] = y
        return y

    def zero(self, sp, ex, tag):
        """constraint ex = 0 on every row (ex: an Ex or a zero-argument callable building it, deferred so the
        evaluator never touches expressions)."""
        if self.emit:
            sp.S.zero(ex() if callable(ex) else ex, tag=tag)

    # ---------------------------------------------------------------- checks
    def rng(self, x, bits, tag):
        """0 <= x < 2^bits on every row (16-bit limb range check)."""
        assert 1 <= bits <= 63, (tag, bits)
        if self.pv:
            bad = (x.v < 0) | (x.v >= (1 << bits)) if bits < 63 else (x.v < 0)
            self.fail("rng:" + tag, bad.sum())
        if self.emit:
            vals = None
            if self.pv:
                vals = np.where((x.v >= 0) & ((x.v < (1 << bits)) if bits < 63 else True), x.v, 0) if Z.FORCE else x.v
            G.range_check(x.sp.S, x.ex, vals, bits, self.nm("r." + tag))

    def srng(self, x, bits, tag):
        """|x| < 2^bits."""
        self.rng(self.lin([(1, x)], 1 << bits, bits=bits + 1), bits + 1, tag)

    def boolean(self, x, tag):
        if self.pv:
            self.fail("bool:" + tag, ((x.v != 0) & (x.v != 1)).sum())
        self.zero(x.sp, lambda: x.ex * (x.ex - Ex.const(1)), "bool:" + tag)

    def lookup(self, tname, sp, xs, tag, table_check=None):
        """every row of (xs...) is a row of public table tname."""
        if self.pv and table_check is not None:
            self.fail("lk:" + tag, (~table_check(*[x.v for x in xs])).sum())
        if self.emit:
            G.wire_lookup(self.ctx, tname, sp.S, [x.ex for x in xs], [x.v for x in xs] if self.pv else None)

    # ---------------------------------------------------------------- tables
    def table(self, name, cols):
        if self.emit:
            G.table(self.ctx, name, cols)


# ============================================================================ public tables
_TAB = {}


def _bf16_table():
    if "bf16" not in _TAB:
        c = np.arange(1 << 16, dtype=np.int64)
        s = (c >> 15) & 1; e = (c >> 7) & 0xFF; m = c & 0x7F
        a = np.where(e == 0, m, 128 + m)
        E = np.where(e == 0, -133, e - 134)
        fin = e != 0xFF
        _TAB["bf16"] = [c[fin], s[fin], a[fin], E[fin]]
        _TAB["bf16_fin"] = fin
        _TAB["bf16_dec"] = (s, a, E)
    return _TAB["bf16"]


def _f32hi_table():
    if "f32hi" not in _TAB:
        c = np.arange(1 << 16, dtype=np.int64)
        s = (c >> 15) & 1; e = (c >> 7) & 0xFF; m = c & 0x7F
        ahi = np.where(e == 0, m, 128 + m)
        E = np.where(e == 0, -149, e - 150)
        fin = e != 0xFF
        _TAB["f32hi"] = [c[fin], s[fin], ahi[fin], E[fin]]
        _TAB["f32hi_fin"] = fin
        _TAB["f32hi_dec"] = (s, ahi, E)
    return _TAB["f32hi"]


def _pow2_table():
    if "pow2" not in _TAB:
        k = np.arange(P2LO, P2HI + 1, dtype=np.int64)
        A = np.array([1 << max(int(t), 0) for t in k], dtype=np.int64)
        B = np.array([1 << max(-int(t), 0) for t in k], dtype=np.int64)
        f = (k < 0).astype(np.int64)
        _TAB["pow2"] = [k, A, B, f]
    return _TAB["pow2"]


SIG_F = 12                    # sigmoid table input step 2^-12
SIG_CLAMP = 16 << SIG_F       # |g| <= 16
SIG_OUT = 30                  # output fixed point 2^-30


def _dec_round_half_up(x_f64, exact_fn, window=1e-5):
    """Platform-independent floor(exact_fn(idx) + 0.5): recomputes entries near .5 in 60-digit Decimal."""
    from decimal import Decimal, ROUND_FLOOR, localcontext
    r = np.floor(x_f64 + 0.5)
    frac = x_f64 - np.floor(x_f64)
    near = np.nonzero(np.abs(frac - 0.5) < window)[0]
    if len(near):
        with localcontext() as ctx:
            ctx.prec = 60
            half = Decimal("0.5")
            for idx in near:
                r[int(idx)] = float(int((exact_fn(int(idx)) + half).to_integral_value(rounding=ROUND_FLOOR)))
    return r.astype(np.int64)


def _dec_floor(x_f64, exact_fn, window=1e-5):
    """Platform-independent floor(exact_fn(idx)): recomputes entries near an integer in 60-digit Decimal."""
    from decimal import ROUND_FLOOR, localcontext
    r = np.floor(x_f64)
    frac = x_f64 - r
    near = np.nonzero((frac < window) | (frac > 1.0 - window))[0]
    if len(near):
        with localcontext() as ctx:
            ctx.prec = 60
            for idx in near:
                r[int(idx)] = float(int(exact_fn(int(idx)).to_integral_value(rounding=ROUND_FLOOR)))
    return r.astype(np.int64)


def _sig_table():
    if "sig" not in _TAB:
        from decimal import Decimal
        k = np.arange(-SIG_CLAMP, SIG_CLAMP + 1, dtype=np.int64)
        x = k.astype(np.float64) / (1 << SIG_F)
        approx = 1.0 / (1.0 + np.exp(-x)) * (1 << SIG_OUT)
        scale = Decimal(1 << SIG_OUT)
        den = Decimal(1 << SIG_F)
        v = _dec_round_half_up(approx, lambda idx: scale / (Decimal(1) + (Decimal(-int(k[idx])) / den).exp()))
        _TAB["sig"] = [k, v]
    return _TAB["sig"]


EXP_F = 18                    # softmax exponent argument step 2^-18
EXP_HI_SH = 12                # dq = dh 2^12 + dl
EXP_DMAX = 32                 # e^-d for d >= 32 is < 2^-46
EXP_OUT = 30


def _exp_tables():
    if "e1" not in _TAB:
        from decimal import Decimal
        nh = (EXP_DMAX << EXP_F) >> EXP_HI_SH
        dh = np.arange(nh + 1, dtype=np.int64)
        dl = np.arange(1 << EXP_HI_SH, dtype=np.int64)
        scale = Decimal(1 << EXP_OUT)
        den_f = Decimal(1 << EXP_F)
        e1_f64 = np.exp(-dh.astype(np.float64) * (1 << EXP_HI_SH) / (1 << EXP_F)) * (1 << EXP_OUT)
        e2_f64 = np.exp(-dl.astype(np.float64) / (1 << EXP_F)) * (1 << EXP_OUT)
        e1 = _dec_round_half_up(e1_f64, lambda idx: scale * (Decimal(-int(dh[idx]) * (1 << EXP_HI_SH)) / den_f).exp())
        e2 = _dec_round_half_up(e2_f64, lambda idx: scale * (Decimal(-int(dl[idx])) / den_f).exp())
        _TAB["e1"] = [dh, e1]
        _TAB["e2"] = [dl, e2]
    return _TAB["e1"], _TAB["e2"]


LN_BITS = 18                  # ln table on 18-bit normalized mantissas [2^17, 2^18)
LN_OUT = 40


def _ln_table():
    if "ln" not in _TAB:
        from decimal import Decimal
        base = 1 << (LN_BITS - 1)
        i = np.arange(base, 1 << LN_BITS, dtype=np.int64)
        scale = Decimal(1 << LN_OUT)
        den = Decimal(base)
        v_f64 = np.log(i.astype(np.float64) / base) * (1 << LN_OUT)
        v = _dec_floor(v_f64, lambda idx: scale * (Decimal(int(i[idx])) / den).ln())
        _TAB["ln"] = [i, v]
    return _TAB["ln"]


def _member(tab_cols):
    keys = {}
    T = np.stack(tab_cols, axis=1)
    s = set(map(tuple, T.tolist()))

    def chk(*vals):
        V = np.stack([np.asarray(v, dtype=np.int64) for v in vals], axis=1)
        return np.array([tuple(r) in s for r in V.tolist()], dtype=bool)
    return chk


def _dense_check(lo, n):
    def chk(*vals):
        x = vals[0]
        return (x >= lo) & (x < lo + n)
    return chk


# ============================================================================ primitive gadgets
def pow2(em, k, tag):
    """k (X, values in [-62, 62]) -> (A = 2^max(k,0), B = 2^max(-k,0), f = [k < 0]) committed on k's space."""
    sp = k.sp
    tab = _pow2_table()
    em.table("cz.pow2", tab)
    A = B = f = None
    if em.pv:
        kc = np.clip(k.v, P2LO, P2HI)
        em.fail("pow2:" + tag, ((k.v < P2LO) | (k.v > P2HI)).sum())
        A = np.left_shift(np.int64(1), np.maximum(kc, 0)); B = np.left_shift(np.int64(1), np.maximum(-kc, 0))
        f = (kc < 0).astype(np.int64)
    Ac = em.col(sp, tag + ".A", A, P2HI + 1)
    Bc = em.col(sp, tag + ".B", B, P2HI + 1)
    fc = em.col(sp, tag + ".f", f, 1)
    if em.emit:
        G.wire_lookup(em.ctx, "cz.pow2", sp.S, [k.ex, Ac.ex, Bc.ex, fc.ex],
                      [k.v, A, B, f] if em.pv else None)
    return Ac, Bc, fc


def clamp(em, k, lo, hi, tag):
    """kk = min(max(k, lo), hi) (k an X with |k| < 2^k.bits)."""
    sp = k.sp
    kv = c1 = c2 = None
    if em.pv:
        kv = np.clip(k.v, lo, hi); c1 = np.maximum(lo - k.v, 0); c2 = np.maximum(k.v - hi, 0)
    kk = em.col(sp, tag + ".kk", kv, max(bitlen(abs(lo)), bitlen(abs(hi))) + 1)
    C1 = em.col(sp, tag + ".c1", c1, k.bits + 1)
    C2 = em.col(sp, tag + ".c2", c2, k.bits + 1)
    em.zero(sp, lambda: kk.ex - k.ex - C1.ex + C2.ex, tag + ".def")
    em.rng(C1, k.bits + 2, tag + ".c1")
    em.rng(C2, k.bits + 2, tag + ".c2")
    em.zero(sp, lambda: C1.ex * (kk.ex - Ex.const(lo)), tag + ".lo")
    em.zero(sp, lambda: C2.ex * (kk.ex - Ex.const(hi)), tag + ".hi")
    em.rng(em.lin([(1, kk)], -lo), bitlen(hi - lo), tag + ".rng")
    return kk


def shr_pub(em, x, k, tag, up_=False):
    """q = floor(x / 2^k) (public k >= 0); with up_=True returns floor + 1 (>= ceil for any x)."""
    if k == 0:
        return em.lin([(1, x)], 1) if up_ else x
    assert x.bits <= LIM - 1, (tag, x.bits)
    q = r = None
    if em.pv:
        q = x.v >> k; r = x.v - (q << k)
    qb = max(1, x.bits - k + 1)
    Q = em.col(x.sp, tag + ".q", q, qb)
    R = em.col(x.sp, tag + ".r", r, k)
    em.zero(x.sp, lambda: x.ex - Q.ex * Ex.const(1 << k) - R.ex, tag + ".shr")
    em.rng(R, k, tag + ".rem")
    em.srng(Q, qb, tag + ".q")
    return em.lin([(1, Q)], 1) if up_ else Q


def prodx(em, a, b):
    """uncommitted product expression a b (same space)."""
    bb = b.bits if a.bits == 1 else a.bits + b.bits          # a boolean: |a b| <= |b|
    assert a.sp is b.sp and bb <= LIM + 1
    return X(a.sp, a.v * b.v if em.pv else None, bb, ex=(a.ex * b.ex) if em.emit else None)


def shr_hid(em, x, A, B, kmax_r, tag, ybits, up_=False, f=None):
    """y = floor(x A / B) with (A, B) = (2^max(-k,0), 2^max(k,0)) from pow2 of -k... i.e. y = floor(x 2^-k):
    x A = y B + rem, 0 <= rem < B <= 2^kmax_r, |y| < 2^ybits (range-checked).  A, B: X on x's space (views).
    Static: x.bits + log2(max A) must stay below LIM (caller passes A's static bound via A.bits)."""
    assert x.bits + A.bits <= LIM, (tag, x.bits, A.bits)
    y = r = None
    if em.pv:
        num = x.v * A.v
        y = np.floor_divide(num, B.v); r = num - y * B.v
    Y = em.col(x.sp, tag + ".y", y, ybits)
    R = em.col(x.sp, tag + ".r", r, kmax_r)
    em.zero(x.sp, lambda: x.ex * A.ex - Y.ex * B.ex - R.ex, tag + ".shr")
    em.rng(R, max(1, kmax_r), tag + ".rlo")
    em.rng(em.lin([(1, B), (-1, R)], -1, bits=kmax_r + 1), max(1, kmax_r), tag + ".rhi")
    em.srng(Y, ybits, tag + ".y")
    if ybits + kmax_r > LIM:
        # |y B + rem| < 2^(ybits + kmax_r) would not stay below p/2.  One of A, B is 1: when B > 1 (f = 1, right
        # shift) |y| <= |x|, so range-checking f y below 2^x.bits keeps |y B + rem| < 2^(x.bits + kmax_r) + B.
        assert f is not None and x.bits + kmax_r <= LIM, (tag, ybits, kmax_r, x.bits)
        em.srng(prodx(em, f, Y), x.bits, tag + ".yf")
    return em.lin([(1, Y)], 1) if up_ else Y


def absx(em, x, tag):
    """(A, s) with x = A (1 - 2 s), s boolean, 0 <= A < 2^bits."""
    a = s = None
    if em.pv:
        s = (x.v < 0).astype(np.int64); a = np.abs(x.v)
    A = em.col(x.sp, tag + ".abs", a, x.bits)
    S = em.col(x.sp, tag + ".sgn", s, 1)
    em.boolean(S, tag + ".sgn")
    em.zero(x.sp, lambda: x.ex - A.ex + Ex.const(2) * S.ex * A.ex, tag + ".absdef")
    em.rng(A, x.bits, tag + ".abs")
    return A, S


def _gidx_ex(em, sp, gsp):
    """group index of every row of sp (the flat row index of gsp), as an Ex on sp."""
    acc = Ex.const(0)
    for ax, b in gsp.axes:
        acc = acc + Ex.atom(Z.IndexP(sp.S.vars(ax), 1 << gsp.S.vars(ax)[0]))
    return acc


def gmax(em, x, gsp, tag, neg=False):
    """per-group max (min if neg) of x over the axes of x.sp that gsp lacks: m on gsp with m - x >= 0 (x - m >= 0)
    on every row and (group, m) a row of the wire table {(group(i), x_i)}."""
    sp = x.sp
    single = not any(sp.has(a) for a, _ in gsp.axes)
    m = None
    if em.pv:
        if single:
            m = np.full(gsp.n, x.v.min() if neg else x.v.max(), dtype=np.int64)
        else:
            gidx = np.zeros(sp.n, dtype=np.int64)
            for ax, b in gsp.axes:
                gidx = (gidx << b) | sp.grid(ax)
            m = np.full(gsp.n, np.iinfo(np.int64).max if neg else np.iinfo(np.int64).min, dtype=np.int64)
            (np.minimum if neg else np.maximum).at(m, gidx, x.v)
    M = em.col(gsp, tag + ".m", m, x.bits)
    Mv = em.view(M, sp)
    d = em.lin([(1, x), (-1, Mv)]) if neg else em.lin([(1, Mv), (-1, x)])
    em.rng(d, x.bits + 1, tag + ".ge")
    if em.emit:
        xc = em.commit(x)
        tname = em.nm("cz.max:" + tag)
        if not any(sp.has(a) for a, _ in gsp.axes):
            kex_sp, kex_g = Ex.const(0), Ex.const(0)
            kv_sp = np.zeros(sp.n, dtype=np.int64) if em.pv else None
            kv_g = np.zeros(gsp.n, dtype=np.int64) if em.pv else None
        else:
            kex_sp = _gidx_ex(em, sp, gsp)
            kex_g = Ex.atom(Z.IndexP(list(range(gsp.bits))))
            kv_sp = None
            if em.pv:
                kv_sp = np.zeros(sp.n, dtype=np.int64)
                for ax, b in gsp.axes:
                    kv_sp = (kv_sp << b) | sp.grid(ax)
            kv_g = np.arange(gsp.n, dtype=np.int64) if em.pv else None
        G.wire_table(em.ctx, tname, sp.S, [kex_sp], [xc.col], [kv_sp] if em.pv else None)
        G.wire_lookup(em.ctx, tname, gsp.S, [kex_g, M.ex], [kv_g, m] if em.pv else None)
    return M


def dot(em, ms, terms, out, out_axes, tag):
    """For every assignment of out_axes: sum over the other axes of ms of prod(terms) = out (out on a space whose
    axes are out_axes).  terms: list of lists of X viewed on ms (their product is summed)."""
    if not em.emit:
        return
    e = Ex.const(1)
    for t in terms:
        e = e * t.ex
    out_axes = [a for a in out_axes if ms.has(a)]
    red = [a for a, _ in ms.axes if a not in out_axes]
    ind = ms.S.ind(**{a: 0 for a in red})
    ov = em.view(out, ms)
    ms.S.sumzero(e - ov.ex * ind, eq_axes=list(out_axes), tag=tag)


def reduce_vals(em, v, src, dst):
    """prover: sum of v (rows of src) into the rows of dst (dst axes subset of src axes; a scalar dst gets the
    total replicated)."""
    if not any(src.has(a) for a, _ in dst.axes):
        return np.full(dst.n, int(v.sum()), dtype=np.int64)
    gidx = np.zeros(src.n, dtype=np.int64)
    for ax, b in dst.axes:
        gidx = (gidx << b) | src.grid(ax)
    out = np.zeros(dst.n, dtype=np.int64)
    np.add.at(out, gidx, v)
    return out


def sum_to(em, terms, dst, tag, bits):
    """S = sum over the axes missing in dst of prod(terms) (terms on one space): committed on dst + zero-sum."""
    src = terms[0].sp
    v = None
    if em.pv:
        p = terms[0].v
        for t in terms[1:]:
            p = p * t.v
        v = reduce_vals(em, p, src, dst)
    assert sum(t.bits for t in terms) + (src.bits - dst.bits) <= LIM, (tag, [t.bits for t in terms])
    Sx = em.col(dst, tag, v, bits)
    dot(em, src, terms, Sx, [a for a, _ in dst.axes], tag)
    return Sx


def mul(em, a, b, tag, bits=None, p_half_ok=False):
    """committed product a b (same space).  p_half_ok: the caller has shown |a b| < p/2 by a finer argument than
    the static bit sum (then bits may be 63)."""
    assert a.sp is b.sp
    bb = a.bits + b.bits if bits is None else bits
    assert a.bits + b.bits <= LIM or p_half_ok, (tag, a.bits, b.bits)
    c = em.col(a.sp, tag, a.v * b.v if em.pv else None, bb)
    em.zero(a.sp, lambda: c.ex - a.ex * b.ex, tag + ".mul")
    return c


def mul_up(em, x, c, tag, cbits=24):
    """upper bound >= c x for x >= 0 and a nonnegative rational constant c.  With C = ceil(c 2^k) (cbits bits) and
    xs = x (or floor(x / 2^s) + 1 >= x / 2^s when x is too wide): c x <= xs C 2^(s - k), result floor(.) + 1.
    Relative slack ~ 2^-(cbits - 1)."""
    c = Fraction(c)
    if c == 0:
        return em.const(x.sp, 0)
    lc = math.floor(math.log2(c.numerator) - math.log2(c.denominator))
    k = cbits - 1 - lc
    Cn = up(c, k)
    s = max(0, x.bits + bitlen(Cn) - (LIM - 3))
    xs = x if s == 0 else shr_pub(em, x, s, tag + ".pre", up_=True)
    sh = k - s
    if sh <= 0:
        return em.lin([(Cn << (-sh), xs)], bits=xs.bits + bitlen(Cn) - sh)
    prod = em.lin([(Cn, xs)], bits=xs.bits + bitlen(Cn))
    return shr_pub(em, prod, sh, tag, up_=True)


def minmax2(em, a, b, tag, take_max):
    """m = max(a, b) (or min): m - a, m - b >= 0 (<= 0) and (m - a)(m - b) = 0."""
    m = np.maximum(a.v, b.v) if (em.pv and take_max) else (np.minimum(a.v, b.v) if em.pv else None)
    M = em.col(a.sp, tag, m, max(a.bits, b.bits))
    sg = 1 if take_max else -1
    em.rng(em.lin([(sg, M), (-sg, a)]), max(a.bits, b.bits) + 1, tag + ".a")
    em.rng(em.lin([(sg, M), (-sg, b)]), max(a.bits, b.bits) + 1, tag + ".b")
    em.zero(a.sp, lambda: (M.ex - a.ex) * (M.ex - b.ex), tag + ".one")
    return M


# ============================================================================ decoding and alignment
class Dec:
    """decoded floats: value = (1 - 2 s) a 2^E (a < 2^abits)."""

    def __init__(self, s, a, E, abits, code=None):
        self.s, self.a, self.E, self.abits, self.code = s, a, E, abits, code


def decode_bf16(em, sp, codes, tag):
    """codes: uint16 array (len sp.n) -> Dec with a <= 255, E in [-133, 120]; non-finite codes are not in the table."""
    tab = _bf16_table()
    em.table("cz.bf16", tab)
    s = a = E = cv = None
    if em.pv:
        cv = np.asarray(codes, dtype=np.int64).ravel()
        assert cv.shape == (sp.n,)
        S_, A_, E_ = _TAB["bf16_dec"]
        em.fail("nonfinite:" + tag, (~_TAB["bf16_fin"][cv]).sum())
        s, a, E = S_[cv], A_[cv], E_[cv]
    C = em.col(sp, tag + ".code", cv, 16)
    Sx = em.col(sp, tag + ".s", s, 1)
    Ax = em.col(sp, tag + ".a", a, 8)
    Exx = em.col(sp, tag + ".E", E, 9)
    if em.emit:
        G.wire_lookup(em.ctx, "cz.bf16", sp.S, [C.ex, Sx.ex, Ax.ex, Exx.ex], [cv, s, a, E] if em.pv else None)
    return Dec(Sx, Ax, Exx, 8, C)


def decode_f32(em, sp, codes, tag):
    """codes: uint32 array -> Dec with a < 2^24, E in [-149, 104]; hi16 by table, lo16 range-checked."""
    tab = _f32hi_table()
    em.table("cz.f32hi", tab)
    hi = lo = s = ahi = E = a = None
    if em.pv:
        cv = np.asarray(codes, dtype=np.int64).ravel()
        hi, lo = cv >> 16, cv & 0xFFFF
        S_, A_, E_ = _TAB["f32hi_dec"]
        em.fail("nonfinite:" + tag, (~_TAB["f32hi_fin"][hi]).sum())
        s, ahi, E = S_[hi], A_[hi], E_[hi]
        a = (ahi << 16) | lo
    H = em.col(sp, tag + ".hi", hi, 16)
    L = em.col(sp, tag + ".lo", lo, 16)
    em.rng(L, 16, tag + ".lo")
    Sx = em.col(sp, tag + ".s", s, 1)
    Ah = em.col(sp, tag + ".ahi", ahi, 8)
    Exx = em.col(sp, tag + ".E", E, 9)
    if em.emit:
        G.wire_lookup(em.ctx, "cz.f32hi", sp.S, [H.ex, Sx.ex, Ah.ex, Exx.ex], [hi, s, ahi, E] if em.pv else None)
    A = em.lin([(1 << 16, Ah), (1, L)], bits=24)
    return Dec(Sx, A, Exx, 24)


def scale_to(em, dec, e, tag, dhi, sign=True):
    """M = sign * floor(a 2^(E - e)) for a per-row target exponent e (X on dec's space).  Shifts below -abits are
    clamped (the floor is then 0 exactly, flag t = 1); shifts above dhi make the relation unsatisfiable (caller's
    range argument).  Returns (M signed, m = |M| >= 0, t = [E < e] truncation flag)."""
    sp = dec.a.sp
    d = em.lin([(1, dec.E), (-1, e)], bits=max(dec.E.bits, e.bits) + 1)
    dd = clamp(em, d, -dec.abits, dhi, tag + ".d")
    A, B, f = pow2(em, dd, tag + ".p")
    A.bits = max(dhi, 0)                     # A <= 2^dhi: a A < 2^(abits + dhi)
    m = shr_hid(em, dec.a, A, B, dec.abits, tag + ".m", dec.abits + max(dhi, 0), f=f)
    m.bits = dec.abits + max(dhi, 0)
    if not sign:
        return m, m, f
    M = em.col(sp, tag + ".M", (m.v * (1 - 2 * dec.s.v)) if em.pv else None, m.bits)
    em.zero(sp, lambda: M.ex - m.ex + Ex.const(2) * dec.s.ex * m.ex, tag + ".sg")
    return M, m, f


EFX = -100                   # exponent floor of aligned vectors (values below 2^(EFX) are truncated, flagged)
EFN = -50                    # exponent floor targeted by normalize (bounded by its maximal shift)


def align(em, dec, gsp, MB, tag, efloor=EFX):
    """fixed point per group: e_g = max(max_group(E) - (MB - abits), efloor); X = sign floor(a 2^(E - e_g)),
    |X| < 2^MB.  Returns (X, |X|, t, e on gsp)."""
    Emax = gmax(em, dec.E, gsp, tag + ".Emax")
    fl = em.const(gsp, efloor) if isinstance(efloor, int) else efloor
    e = minmax2(em, em.lin([(1, Emax)], -(MB - dec.abits), bits=10), fl, tag + ".ef", True)
    ev = em.view(e, dec.a.sp)
    M, m, t = scale_to(em, dec, ev, tag, MB - dec.abits)
    return M, m, t, e


# ============================================================================ radius helpers
ETA_CAP = 40                 # eta in ulps is capped at 2^40 (exact whenever the scale exponent is >= log2eta - 40;
                             # below that only all-zero / sub-2^-126 vectors, whose engine values are exact zeros)


def eta_units(em, e, tag, log2eta=None):
    """ceil(2^log2eta / 2^e) = 2^min(max(log2eta - e, 0), ETA_CAP) on e's space (eta of every store / dot)."""
    l2 = C0.eta_log2 if log2eta is None else log2eta
    k = em.lin([(-1, e)], l2, bits=e.bits + 1)
    kk = clamp(em, k, 0, ETA_CAP, tag + ".eta")
    A, B, f = pow2(em, kk, tag + ".etap")
    A.bits = ETA_CAP + 1
    return A


def store(em, Y, R, absY, etaU, tag, ulog2=8):
    """06 Store template on (center, radius): R' = R + ceil(2^-ulog2 (|Y| + R)) + eta."""
    t = shr_pub(em, em.lin([(1, absY), (1, R)]), ulog2, tag + ".st", up_=True)
    return em.lin([(1, R), (1, t), (1, etaU)])


def obs_cmp(em, dec, Y, R, e, tag):
    """check |observed - Y 2^e| <= R 2^e (componentwise): Yo = sign floor(a 2^(E - e)) and |Yo - Y| + 1 <= R
    (the +1 pays the floor of Yo).  Observed values that would need a shift above dhi are certainly outside the box
    (|obs| >= 2^(abits-1+dhi+1) 2^e > (|Y| + R) 2^e statically) and make the relation unsatisfiable."""
    bY = max(Y.bits, R.bits)
    assert bY <= 60, (tag, Y.bits, R.bits)
    dhi = bY + 1 - (dec.abits - 1)
    Yo, _, _ = scale_to(em, dec, e, tag + ".obs", dhi)
    diff = em.lin([(1, Yo), (-1, Y)], bits=max(Yo.bits, Y.bits) + 1)
    if em.pv:
        err = np.abs(diff.v) + 1
        ratio = np.where(R.v > 0, err / np.maximum(R.v, 1), np.inf)
        st = em.stats.setdefault(tag, {})
        st["max_ratio"] = max(st.get("max_ratio", 0.0), float(ratio.max()))
        st["n"] = st.get("n", 0) + int(Y.v.size)
        em.fail("box:" + tag, (err > R.v).sum())
    em.rng(em.lin([(1, R), (-1, diff)], -1, bits=62), 62, tag + ".le")
    em.rng(em.lin([(1, R), (1, diff)], -1, bits=62), 62, tag + ".ge")
    return Yo


def normalize(em, x, gsp, tb, tag, nonneg=False, e=None, efloor=EFN):
    """per-group rescale y = floor(x / 2^k), |y| < 2^(tb-1), with the deterministic shift
        k = max(kb, kf),  kb = max(0, bitlen(max|x|) - (tb - 1)),  kf = min(max(efloor - e, 0), kmax)
    (kf keeps the exponent e + k of tiny / zero vectors from drifting down, so products stay representable).
    Constraints: |y| < 2^tb (range), max|x| < 2^(tb-1+k), k >= kf, and k > kf => max|x| >= 2^(tb-2+k).
    Returns (y, K, Bk = 2^K, f = [K > 0], M = max|x|) on gsp."""
    sp = x.sp
    if nonneg:
        A = x
        em.rng(x, x.bits, tag + ".nn")
    else:
        A, _ = absx(em, x, tag + ".a")
    M = A if gsp is sp else gmax(em, A, gsp, tag + ".max")
    kmax = max(0, x.bits - tb + 1)
    if e is not None:
        e = em.view(e, gsp) if e.sp is not gsp else e
        kf = clamp(em, em.lin([(-1, e)], efloor, bits=e.bits + 1), 0, kmax, tag + ".kf")
    else:
        kf = em.const(gsp, 0)
    k = g = None
    if em.pv:
        kb = np.maximum(0, np.array([bitlen(int(t)) for t in M.v], dtype=np.int64) - (tb - 1))
        k = np.maximum(kb, kf.v); g = (k > kf.v).astype(np.int64)
    K = em.col(gsp, tag + ".k", k, bitlen(kmax) + 1)
    em.rng(em.lin([(1, K), (-1, kf)], bits=bitlen(kmax) + 1), bitlen(kmax) + 1, tag + ".k0")
    em.rng(em.lin([(-1, K)], kmax), bitlen(kmax) + 1, tag + ".k1")
    Gb = em.col(gsp, tag + ".g", g, 1)
    em.boolean(Gb, tag + ".g")
    em.zero(gsp, lambda: (K.ex - kf.ex) * (Ex.const(1) - Gb.ex), tag + ".gdef")
    One, Bk, f = pow2(em, em.lin([(-1, K)]), tag + ".p")
    One.bits = 1
    yM = shr_hid(em, M, One, Bk, kmax, tag + ".ym", tb, f=f)
    em.rng(em.lin([(-1, yM)], (1 << (tb - 1)) - 1), tb, tag + ".ub")
    em.rng(prodx(em, Gb, em.lin([(1, yM)], -(1 << (tb - 2)))), tb, tag + ".min")
    if gsp is sp:
        y = shr_hid(em, x, One, Bk, kmax, tag + ".y", tb, f=f)
    else:
        y = shr_hid(em, x, em.view(One, sp), em.view(Bk, sp), kmax, tag + ".y", tb, f=em.view(f, sp))
    y.bits = tb
    return y, K, Bk, f, M


def rshift_up(em, R, Bk, f, kmax, tag, qb, extra=1):
    """R' = floor(R / 2^k) + 1 + extra >= R / 2^k + extra for R >= 0, with the range condition floor(R/2^k) < 2^qb
    (specs/06 §2.1 Rng_b; keeps the floor identity exact: qb + kmax <= 62)."""
    One = em.const(R.sp, 1)
    One.bits = 1
    assert qb + kmax <= LIM, (tag, qb, kmax)
    q = shr_hid(em, R, One, Bk, kmax, tag, qb, f=f)
    return em.lin([(1, q)], 1 + extra, bits=qb + 1)


def capped(em, R, bits, tag):
    """range certificate on a radius (specs/06 §2.1 Rng_b): 0 <= R < 2^bits, else the block is bad ("range")."""
    R2 = em.commit(R, tag)
    em.rng(R2, bits, "cap." + tag)
    R2.bits = bits
    return R2


# ============================================================================ weights
@dataclass
class Mat:
    """One weight tensor in canonical-decode form (specs/09 CD): w ~ W 2^-G, |W| < 2^22; Wr = |W| + kappa [inexact]."""
    W: np.ndarray            # (out, in) int32
    G: int
    Wr: np.ndarray           # (out, in) int32
    n_in: int


def cd_matrix(w_bf16_codes, contract=C0):
    """bf16 codes (out, in) -> Mat with the specs/09 canonical decode (flashprove_next.modelcommit.cd_exponent / cd_decode)."""
    from .modelcommit import cd_exponent, cd_decode, _bf_tables
    codes = np.asarray(w_bf16_codes, dtype=np.uint16)
    Gx = cd_exponent(codes)
    W = cd_decode(codes, Gx)
    sig_t, E_t, _ = _bf_tables()
    k = E_t[codes.astype(np.int64)] + Gx
    inexact = k < 0
    if inexact.any():
        sig = sig_t[codes.astype(np.int64)]
        sh = np.clip(-k, 0, 62)
        inexact = (sig != 0) & inexact & (((np.abs(sig) & ((np.int64(1) << sh) - 1)) != 0) | (sh > 20))
    n = codes.shape[1]
    kap = kappa(n, contract)
    Wr = (np.abs(W.astype(np.int64)) + kap * inexact.astype(np.int64))
    assert Wr.max() < (1 << 23)
    return Mat(W.astype(np.int32), int(Gx), Wr.astype(np.int32), n)


def kappa(n, contract=C0):
    g = contract.gamma(n)
    return int(-(-(1 + g).numerator * g.denominator // (g.numerator * (1 + g).denominator)))   # ceil((1+g)/g)


class PublicWeights:
    """Clear weight oracle: W^cd and W^r are public vectors on the matrix cube (the verifier evaluates their MLE).
    mats: name -> Mat; vecs: name -> bf16 codes (1-D: norm gains, RoPE rows)."""
    hidden = False

    def __init__(self, mats=None, vecs=None, eps_bits=None, rope=None, qk=None, scale=None, pos_x=None, t_pad=None):
        self.mats, self.vecs = dict(mats or {}), dict(vecs or {})
        self.eps_bits = eps_bits                # fp32 bits of the RMSNorm epsilon
        self.rope = rope                        # callable pos -> (cos, sin) float64 arrays of the bf16 table row
        self.qk = qk or {}                      # layer -> (eq, ek) public score exponents
        self.scale = scale                      # attention scale (the engine's fp32 constant)
        self.pos_x = pos_x                      # hidden position scalar (X on a scalar space) of the hidden RoPE path
        self.t_pad = t_pad                      # public RoPE table height (profile constant) of the hidden path
        self.require_hidden_rope = False        # True: the public-row path of _rope is refused (hidden providers)
        self.qk_window = None                   # public profile window [lo, hi] of eq + ek (hidden-exponent ATT)

    def rope_rows(self, pos):
        return self.rope(pos)

    def qk_exp(self, layer):
        return self.qk[layer]

    def rope_wit(self, pos):
        """prover: the (cos, sin) rows it commits for position pos (a cheating prover overrides this)."""
        return self.rope(pos)

    def on_qk_gain(self, em, gname, dg, dh, tag):
        """hook called by _qk_norm with the decoded per-head gain vector (hidden providers derive their score
        exponents in-circuit from it)."""
        return None

    def scale_fx(self):
        f = float(np.float32(self.scale))
        m, e = math.frexp(f)
        Cs = int(m * (1 << 24))
        assert Cs == m * (1 << 24) and (1 << 23) <= Cs < (1 << 24)
        return Cs, e - 24

    def _atom(self, em, ms, name, arr, bits, tag):
        return em.pub(ms, tag, arr, bits)

    def matvec(self, em, name, Xv, Zv, OS, tag):
        """Y = W^cd X, A = W^r Z on the output space OS (X, Z on the input space)."""
        M = self.mats[name]
        XS = Xv.sp
        ms = em.space(tag + ".MS", OS.axes + XS.axes)
        on, inn = OS.n, XS.n
        Wp = Wrp = None
        if em.pv or em.emit:
            Wp = np.zeros((on, inn), dtype=np.int64); Wp[:M.W.shape[0], :M.W.shape[1]] = M.W
            Wrp = np.zeros((on, inn), dtype=np.int64); Wrp[:M.Wr.shape[0], :M.Wr.shape[1]] = M.Wr
        Y = A = None
        nb = bitlen(inn - 1) if inn > 1 else 0
        if em.pv:
            Y = Wp @ Xv.v; A = Wrp @ Zv.v
        yb, ab = 22 + Xv.bits + nb, 23 + Zv.bits + nb
        assert yb <= LIM and ab <= LIM, (tag, yb, ab)
        Yx = em.col(OS, tag + ".Y", Y, yb)
        Ax = em.col(OS, tag + ".A", A, ab)
        if em.emit:
            W = self._atom(em, ms, name, Wp.ravel(), 22, tag + ".W")
            Wr = self._atom(em, ms, name + ".r", Wrp.ravel(), 23, tag + ".Wr")
            dot(em, ms, [W, em.view(Xv, ms)], Yx, [a for a, _ in OS.axes], tag + ".mvY")
            dot(em, ms, [Wr, em.view(Zv, ms)], Ax, [a for a, _ in OS.axes], tag + ".mvA")
        return Yx, Ax, M.G, M.n_in

    def vec_dec(self, em, sp, name, tag):
        """1-D bf16 weight vector (norm gains) decoded on sp: public (s, a, E) columns."""
        codes = np.zeros(sp.n, dtype=np.int64)
        c = np.asarray(self.vecs[name], dtype=np.int64).ravel()
        codes[:len(c)] = c
        s_, a_, E_ = (t[codes] for t in _TAB["bf16_dec"]) if "bf16_dec" in _TAB else (None, None, None)
        if s_ is None:
            _bf16_table(); s_, a_, E_ = (t[codes] for t in _TAB["bf16_dec"])
        return Dec(em.pub(sp, tag + ".s", s_, 1), em.pub(sp, tag + ".a", a_, 8), em.pub(sp, tag + ".E", E_, 9), 8)


class CommittedWeights(PublicWeights):
    """Hidden witness matrices (the specs/09 path supplies the same two evaluations per tensor as hidden claims;
    binding the witness to model_id is done by that path, not by a standalone block proof)."""
    hidden = True

    def _atom(self, em, ms, name, arr, bits, tag):
        return em.col(ms, tag, arr, bits)

    def vec_dec(self, em, sp, name, tag):
        return decode_bf16(em, sp, _pad_sp(self.vecs[name], sp), tag)


class TiledWeights(PublicWeights):
    """specs/09 path (flashprove_next.wsum_zk): no matrix cube at all.  matvec commits Y = W^cd X and A = W^r Z on the output space
    (as before) and registers the GEMM; the weight side is proven once for all registered GEMMs by wsum_zk (folded
    rows at the shared Fiat-Shamir tile point -> hidden tile claims -> hidden column opening -> model_id).
    mats: name -> Mat (prover: the matrices it computes with; verifier: Mat(W=None, G, Wr=None, n_in) -- G and the
    shapes are public profile constants); tensors: local name -> committed tensor name (prover-only tile table key).
    Norm-gain vectors are hidden witnesses (as CommittedWeights) and NOT bound to model_id by THIS class; the RoPE rows,
    the q/k score exponents and the gain vectors of the zero-knowledge reply circuit are the hidden, bound quantities of
    ``flashprove_next.wsum_zk.CfgTiledWeights`` (hidden pos scalar + RoPE table LogUp, ``qk_exp_hidden``, ``VecExt``)."""
    hidden = True

    def __init__(self, mats=None, vecs=None, tensors=None, **kw):
        super().__init__(mats, vecs, **kw)
        self.tensors = dict(tensors or {})
        self.gemms = []

    def matvec(self, em, name, Xv, Zv, OS, tag):
        M = self.mats[name]
        XS = Xv.sp
        on, inn = OS.n, XS.n
        nb = bitlen(inn - 1) if inn > 1 else 0
        yb, ab = 22 + Xv.bits + nb, 23 + Zv.bits + nb
        assert yb <= LIM and ab <= LIM, (tag, yb, ab)
        Y = A = None
        if em.pv:
            o, i = M.W.shape
            Y = np.zeros(on, dtype=np.int64); A = np.zeros(on, dtype=np.int64)
            Y[:o] = M.W.astype(np.int64) @ Xv.v[:i]
            A[:o] = M.Wr.astype(np.int64) @ Zv.v[:i]
        Xc = em.commit(Xv, tag + ".Xc")                 # phase-1 copies: the wsum rounds only view committed columns
        Zc = em.commit(Zv, tag + ".Zc")
        Yx = em.col(OS, tag + ".Y", Y, yb)
        Ax = em.col(OS, tag + ".A", A, ab)
        self.gemms.append(dict(name=name, tensor=self.tensors.get(name, name), X=Xc, Z=Zc, Y=Yx, A=Ax, OS=OS, XS=XS,
                               mat=M, tag=tag))
        return Yx, Ax, M.G, M.n_in

    def vec_dec(self, em, sp, name, tag):
        return CommittedWeights.vec_dec(self, em, sp, name, tag)


# ============================================================================ GEMM with the certified radius
def gemm_mb(prov, n_in_bits, cap=26):
    """fixed-point width MB of a GEMM input (align): the radius product A = W^r Z has 23 + extra + (MB + 1) + n_in_bits
    bits (extra = the provider's per-tile alignment headroom, wsum_zk.CfgTiledWeights.s_bits) and must stay <= LIM - 1
    for mul_up -> MB <= 37 - n_in_bits - extra.  Every GEMM input uses this one rule (Qwen3-0.6B, s_bits = 1: QKV / GU
    26, OPROJ 25, DOWN 24 bits)."""
    return min(cap, 37 - int(n_in_bits) - int(getattr(prov, "extra_bits", 0)))


def gemm(em, prov, name, X, mX, t, ex, OS, tag):
    """Engine pre-store output of y = W x (Dot contract, fp32 accumulation) as (Y, R0) at exponent ey = ex - G:
        |y_eng - sum w x_obs| <= gamma_n sum |w||x_obs| + eta                             (contract)
        |sum w x_obs - Y 2^ey| <= sum m |x_obs| 2^-G + sum |W| t 2^ey                    (W^cd and X truncation)
    with |w| <= (|W| + m) 2^-G, |x_obs| <= (|X| + t) 2^ex, m = [w 2^G not integer], t = [x truncated]:
        R0 = ceil(gamma_n A) + eta,   A = W^r (|X| + kappa t),  W^r = |W| + kappa m,  kappa = ceil((1+gamma)/gamma)
    covers both (gamma W^r Z >= gamma |W||X| + (1+gamma)(|W| t + m |X| + m t))."""
    M = prov.mats[name]
    kap = kappa(M.n_in)
    Z = em.lin([(1, mX), (kap, t)], bits=mX.bits + 1)
    assert kap < (1 << mX.bits)
    Zc = em.commit(Z, tag + ".Z")
    Y, A, Gx, n_in = prov.matvec(em, name, X, Zc, OS, tag)
    ey = em.lin([(1, ex)], -Gx, bits=10)
    R0 = mul_up(em, A, C0.gamma(n_in), tag + ".g")
    eu = eta_units(em, ey, tag + ".e0")
    return Y, em.lin([(1, R0), (1, em.view(eu, OS))]), ey


def rv_store(em, Y, R, ey, tag, ulog2=8, absY=None):
    """bf16 (ulog2 = 8) or fp32 (24) store of (Y, R) at exponent ey (X on a space Y.sp views)."""
    if absY is None:
        absY, _ = absx(em, Y, tag + ".ay")
    eu = em.view(eta_units(em, ey, tag + ".e"), Y.sp)
    return store(em, Y, R, absY, eu, tag, ulog2), absY


# ============================================================================ blocks
def _pad(codes, n):
    c = np.zeros(n, dtype=np.int64)
    a = np.asarray(codes, dtype=np.int64).ravel()
    c[:len(a)] = a
    return c


def _pad_sp(codes, sp):
    if sp.em.bbits == 0:
        return _pad(codes, sp.n)
    K = 1 << sp.em.bbits
    inn = sp.in_n
    a = np.asarray(codes, dtype=np.int64)
    if a.ndim == 1:
        return np.tile(_pad(a, inn), K)
    a2 = a.reshape(K, -1)
    out = np.zeros((K, inn), dtype=np.int64)
    out[:, :a2.shape[1]] = a2
    return out.reshape(-1)


def _lg(n):
    return max(0, (int(n) - 1).bit_length())


EPS_LOG2_FIX = 48            # rsqrt witness scale: T^2 Ph <= d 2^(48 + par)
RSQ_W = 8                    # enclosure width: t* in [T, T + 8) 2^et


def add_pos(em, a, b, tag, mb=20):
    """[al, ah] 2^ea + [bl, bh] 2^eb (all >= 0, mantissas <= 2^mb) -> ([cl, ch], ec), ec = max(ea, eb):
    cl = floor-shifted sum, ch = floor-shifted sum + 2 (>= the exact upper sum); both <= 2^(mb+1) + 2."""
    (al, ah, ea), (bl, bh, eb) = a, b
    sp = al.sp
    ec = None
    if em.pv:
        ec = np.maximum(ea.v, eb.v)
    Ec = em.col(sp, tag + ".ec", ec, 11)
    da = em.lin([(1, Ec), (-1, ea)], bits=12)
    db = em.lin([(1, Ec), (-1, eb)], bits=12)
    em.rng(da, 11, tag + ".da")
    em.rng(db, 11, tag + ".db")
    em.zero(sp, lambda: da.ex * db.ex, tag + ".one")
    out = []
    for (lo, hi, d, nm) in ((al, ah, da, "a"), (bl, bh, db, "b")):
        dd = clamp(em, d, 0, mb + 1, tag + ".c" + nm)
        One, Bd, f = pow2(em, em.lin([(-1, dd)]), tag + ".p" + nm)
        One.bits = 1
        l2 = shr_hid(em, lo, One, Bd, mb + 1, tag + ".l" + nm, mb + 1, f=f)
        h2 = shr_hid(em, hi, One, Bd, mb + 1, tag + ".h" + nm, mb + 1, f=f)
        out.append((l2, h2))
    cl = em.lin([(1, out[0][0]), (1, out[1][0])], bits=mb + 2)
    ch = em.lin([(1, out[0][1]), (1, out[1][1])], 2, bits=mb + 2)
    return cl, ch, Ec


def rsqrt_enc(em, Pl, Ph, eP, d, tag):
    """t* = sqrt(d / P), P in [Pl, Ph] 2^eP, 2^19 <= Pl <= Ph <= 2^21 + 2: integers T, et with
    T 2^et <= t* < (T + 8) 2^et.  Witness par in {0,1} with 2 et + eP + 48 + par = 0 and
        T^2 Ph <= d 2^(48+par) < (T + 8)^2 Pl.
    Field safety: T < 2^21, T^2 < 2^41 (range), so both products stay below 2^41 (2^21 + 2) < 2^62.1 < p/2."""
    sp = Pl.sp
    assert d <= (1 << 10)
    em.rng(em.lin([(1, Pl)], -(1 << 19)), 22, tag + ".plo")          # Rng: P mantissa normalised
    T = et = par = None
    if em.pv:
        T = np.zeros(sp.n, dtype=np.int64); par = np.zeros(sp.n, dtype=np.int64); et = np.zeros(sp.n, dtype=np.int64)
        for i in range(sp.n):
            eP_ = int(eP.v[i]); par_ = (eP_ + EPS_LOG2_FIX) & 1
            K = EPS_LOG2_FIX + par_
            ph = max(1, int(Ph.v[i]))
            T_ = math.isqrt((d << K) // ph)
            while T_ * T_ * ph > (d << K):
                T_ -= 1
            T[i], par[i], et[i] = T_, par_, -(eP_ + EPS_LOG2_FIX + par_) // 2
    Tx = em.col(sp, tag + ".T", T, 21)
    em.rng(Tx, 21, tag + ".T")
    Px = em.col(sp, tag + ".par", par, 1)
    em.boolean(Px, tag + ".par")
    Et = em.col(sp, tag + ".et", et, 10)
    em.srng(Et, 10, tag + ".et")
    em.zero(sp, lambda: Ex.const(2) * Et.ex + eP.ex + Ex.const(EPS_LOG2_FIX) + Px.ex, tag + ".etdef")
    rhs = em.lin([((d << EPS_LOG2_FIX), Px)], d << EPS_LOG2_FIX, bits=60)     # d 2^48 (1 + par) <= 2^59
    T2 = mul(em, Tx, Tx, tag + ".T2", bits=41)
    em.rng(T2, 41, tag + ".T2")
    lhs = mul(em, T2, Ph, tag + ".lhs", bits=63, p_half_ok=True)
    em.rng(em.lin([(1, rhs), (-1, lhs)], bits=63), 62, tag + ".lo")
    Tq = em.commit(em.lin([(1, T2), (2 * RSQ_W, Tx)], RSQ_W * RSQ_W, bits=42), tag + ".Tq")   # (T + 8)^2
    upv = mul(em, Tq, Pl, tag + ".up", bits=63, p_half_ok=True)
    em.rng(em.lin([(1, upv), (-1, rhs)], -1, bits=63), 63, tag + ".hi")
    return Tx, Et


def blk_N(em, prov, gname, h, x, d, eps_bits, tag="N"):
    """RMSNorm + bf16 cast (HF recipe: fp32 internals, x * rsqrt stored bf16, gain * that stored bf16), exact input:
        y = h g / s,  s = sqrt(mean(h^2) + eps);  r = C_N |y| + (2 + u) eta,
        C_N = (1 + u)^2 (1 + rho) - 1,  rho = (1 + gamma_d + 2 u_acc)(1 + u_elem)(1 + u_acc) - 1   (reference RMSNorm)
    Per-element exponents: y_i = sign (a_h a_g) T 2^(E_h + E_g + et) with T from the rsqrt enclosure; the enclosure
    width (8 a_h a_g units) joins the radius."""
    c = C0
    XS = em.space(tag, [("i", _lg(d))])
    SC = em.space(tag + ".sc", [])
    dh = decode_bf16(em, XS, _pad_sp(h, XS), tag + ".h")
    dx = decode_bf16(em, XS, _pad_sp(x, XS), tag + ".x")
    dg = prov.vec_dec(em, XS, gname, tag + ".g")
    # sum of squares, enclosed: S in [Slo, Shi] 2^(2 eH)
    H, mH, tH, eH = align(em, dh, SC, 25, tag + ".al")
    Slo = sum_to(em, [mH, mH], SC, tag + ".Slo", 61)
    Sex = sum_to(em, [tH, em.commit(em.lin([(2, mH)], 1), tag + ".2m1")], SC, tag + ".Sex", 37)
    Shi = em.lin([(1, Slo), (1, Sex)], bits=61)
    Sn, K, Bk, f, _ = normalize(em, Shi, SC, 21, tag + ".sn", nonneg=True)
    One = em.const(SC, 1); One.bits = 1
    Sl = shr_hid(em, Slo, One, Bk, Shi.bits - 21 + 1, tag + ".sl", 21, f=f)
    Sh = em.lin([(1, Sn)], 1, bits=21)                          # Sn < 2^20 => Sh <= 2^20
    eS = em.lin([(2, eH), (1, K)], bits=11)
    # d * eps (eps: fp32 bits of the engine constant), normalised to 20 bits (public)
    ev = np.array([eps_bits], dtype=np.uint32).view(np.float32)[0]
    ef = Fraction(float(ev)) * d
    xe = math.floor(math.log2(ef.numerator) - math.log2(ef.denominator)) - 19     # Dl in [2^19, 2^20)
    Dl = math.floor(ef / Fraction(2) ** xe); Dh = Dl + (0 if Fraction(Dl) * Fraction(2) ** xe == ef else 1)
    Pl, Ph, eP = add_pos(em, (Sl, Sh, eS), (em.const(SC, Dl), em.const(SC, Dh), em.const(SC, xe)), tag + ".P", 20)
    Pl, Ph = em.commit(Pl, tag + ".Pl"), em.commit(Ph, tag + ".Ph")
    Pl.bits = Ph.bits = 22
    T, et = rsqrt_enc(em, Pl, Ph, eP, d, tag + ".rs")
    # per element
    ahg = mul(em, dh.a, dg.a, tag + ".ahg")                     # a_h a_g < 2^16
    Tv = em.view(T, XS)
    Y0 = mul(em, ahg, Tv, tag + ".Y0", bits=37)
    sg = em.lin([(1, dh.s), (1, dg.s)], bits=2)                 # sign bit = s_h xor s_g
    Ys = em.col(XS, tag + ".Y", (Y0.v * (1 - 2 * ((dh.s.v + dg.s.v) & 1))) if em.pv else None, 37)
    # Y = Y0 (1 - 2 s_h)(1 - 2 s_g)
    em.zero(XS, lambda: Ys.ex - Y0.ex * (Ex.const(1) - Ex.const(2) * dh.s.ex) * (Ex.const(1) - Ex.const(2) * dg.s.ex), tag + ".Ysg")
    rho = (1 + c.gamma(d) + 2 * c.u_acc) * (1 + c.u_elem) * (1 + c.u_acc) - 1
    CN = (1 + c.u_store) ** 2 * (1 + rho) - 1
    Ymax = em.lin([(1, Y0), (RSQ_W, ahg)], bits=38)               # >= |y| in units
    R = em.lin([(1, mul_up(em, Ymax, CN, tag + ".cn")), (RSQ_W, ahg)], bits=38)
    eps_i = em.lin([(1, dh.E), (1, dg.E), (1, em.view(et, XS))], bits=11)
    eu = eta_units(em, eps_i, tag + ".eta", log2eta=c.eta_log2 + 2)
    R = capped(em, em.lin([(1, R), (1, eu)], bits=62), 60, tag + ".R")
    obs_cmp(em, dx, Ys, R, eps_i, tag)
    return em


def blk_gemm_res(em, prov, name, xin, hres, hout, n_in, n_out, tag):
    """OPROJ / DOWN: y = W x (fp32 acc) -> bf16 store -> + residual (bf16) -> bf16 store (reference: store(add(store(
    gemm(W, x)), h)))."""
    XS = em.space(tag + ".x", [("i", _lg(n_in))])
    OS = em.space(tag + ".o", [("o", _lg(n_out))])
    SC = em.space(tag + ".sc", [])
    dxi = decode_bf16(em, XS, _pad_sp(xin, XS), tag + ".xi")
    dhr = decode_bf16(em, OS, _pad_sp(hres, OS), tag + ".hr")
    dho = decode_bf16(em, OS, _pad_sp(hout, OS), tag + ".ho")
    MBx = gemm_mb(prov, XS.in_bits)                       # W^r Z = 23 + extra + (MBx + 1) + log2 n_in bits <= 61
    # the output scale must hold the residual (massive activations): e_x >= max_E(h) - 51 + G, so that
    # h 2^-e_y < 2^(8 + 51) and the sum stays below 2^60 (o-proj precision is only lost relative to |h|)
    Gx = prov.mats[name].G
    Emh = gmax(em, dhr.E, SC, tag + ".Emh")
    Xs, mX, t, ex = align(em, dxi, SC, MBx, tag + ".al",
                          efloor=minmax2(em, em.lin([(1, Emh)], Gx - 51, bits=10), em.const(SC, EFX), tag + ".fl", True))
    Y, R0, ey = gemm(em, prov, name, Xs, mX, t, ex, OS, tag + ".mm")
    R0 = capped(em, R0, 56, tag + ".R0")
    R1, aY = rv_store(em, Y, R0, ey, tag + ".s1")
    eyv = em.view(ey, OS)
    Hm, mHm, th = scale_to(em, dhr, eyv, tag + ".res", 51)[0:3]   # d = E_h - e_y <= 51 by the floor above
    S = em.commit(em.lin([(1, Y), (1, Hm)], bits=60), tag + ".S")
    R2 = em.lin([(1, R1), (1, th)])
    R3, _ = rv_store(em, S, R2, ey, tag + ".s2")
    R3 = capped(em, R3, 60, tag + ".R3")
    obs_cmp(em, dho, S, R3, eyv, tag)
    return em


def blk_head(em, prov, name, xin, logits_f32, n_in, n_out, tag="HEAD"):
    """LM head with fp32 logits (specs/06 §1.2): y = W x, fp32 accumulate, fp32 store.  logits_f32 = None: no observed
    logits (generation steps, specs/01 §5.4) -- only the reference (Y, R1, e_y) is built and left in em.head_ref."""
    XS = em.space(tag + ".x", [("i", _lg(n_in))])
    OS = em.space(tag + ".o", [("o", _lg(n_out))])
    SC = em.space(tag + ".sc", [])
    dxi = decode_bf16(em, XS, _pad_sp(xin, XS), tag + ".xi")
    dlo = decode_f32(em, OS, _pad_sp(logits_f32, OS), tag + ".lg") if logits_f32 is not None else None
    Xs, mX, t, ex = align(em, dxi, SC, gemm_mb(prov, XS.in_bits), tag + ".al")
    Y, R0, ey = gemm(em, prov, name, Xs, mX, t, ex, OS, tag + ".mm")
    R0 = capped(em, R0, 56, tag + ".R0")
    R1, _ = rv_store(em, Y, R0, ey, tag + ".s1", ulog2=24)
    R1 = capped(em, R1, 59, tag + ".R1")
    em.head_ref = dict(Y=Y, R=R1, ey=ey, OS=OS)          # interface for flashprove_next.sampler_zk (reference logits + radius)
    if logits_f32 is not None:
        obs_cmp(em, dlo, Y, R1, em.view(ey, OS), tag)
    return em


def _norm_rv(em, Y, R, ey, gsp, tb, rbits, tag):
    """renormalise a center to tb bits per group and carry its radius: Y' = floor(Y / 2^k), R' = ceil(R / 2^k) + 1
    (the +1 pays the floor of Y')."""
    Yn, K, Bk, f, M = normalize(em, Y, gsp, tb, tag, e=ey)
    sp = Y.sp
    kmax = max(0, Y.bits - tb + 1)
    Rn = rshift_up(em, R, em.view(Bk, sp), em.view(f, sp), kmax, tag + ".R", min(LIM - kmax, rbits))
    Rn = capped(em, Rn, rbits, tag + ".Rc")
    en = em.lin([(1, em.view(ey, gsp) if ey.sp is not gsp else ey), (1, K)], bits=11)
    return Yn, Rn, en, Bk, f


def blk_gu(em, prov, xin, aout, d, dff, gname="gate", uname="up", tag="GU"):
    """a = store(store(silu(store(W_g x))) * store(W_u x))   (reference GU)."""
    c = C0
    XS = em.space(tag + ".x", [("i", _lg(d))])
    OS = em.space(tag + ".o", [("o", _lg(dff))])
    SC = em.space(tag + ".sc", [])
    dxi = decode_bf16(em, XS, _pad_sp(xin, XS), tag + ".xi")
    dao = decode_bf16(em, OS, _pad_sp(aout, OS), tag + ".ao")
    Xs, mX, t, ex = align(em, dxi, SC, gemm_mb(prov, XS.in_bits), tag + ".al")
    out = {}
    for nm in (gname, uname):
        Y, R0, ey = gemm(em, prov, nm, Xs, mX, t, ex, OS, tag + "." + nm)
        R0 = capped(em, R0, 56, tag + nm + ".R0")
        R1, _ = rv_store(em, Y, R0, ey, tag + "." + nm + ".s")
        Yn, Rn, en, _, _ = _norm_rv(em, Y, R1, ey, SC, 29, 25, tag + "." + nm + ".n")
        out[nm] = (Yn, Rn, en)
    Yg, Rg, eg = out[gname]
    Yu, Ru, eu_ = out[uname]
    # sigmoid of g = Yg 2^eg on the 2^-12 grid, |g| <= 16 (clamped)
    sh = em.lin([(1, eg)], SIG_F, bits=11)                                 # g 2^12 = Yg 2^(eg + 12)
    shc = clamp(em, sh, -30, 20, tag + ".sgsh")                         # right shifts >= 30 give floor 0 / -1 anyway
    A, B, f = pow2(em, shc, tag + ".sgp")
    A.bits = 20
    gq = shr_hid(em, Yg, em.view(A, OS), em.view(B, OS), 30, tag + ".gq", 50, f=em.view(f, OS))
    gqc = clamp(em, gq, -SIG_CLAMP, SIG_CLAMP, tag + ".gqc")
    stab = _sig_table()
    em.table("cz.sig", stab)
    sig = None
    if em.pv:
        sig = stab[1][gqc.v + SIG_CLAMP]
    Sg = em.col(OS, tag + ".sig", sig, SIG_OUT + 1)
    em.lookup("cz.sig", OS, [gqc, Sg], tag + ".sig", _dense_check(-SIG_CLAMP, 2 * SIG_CLAMP + 1))
    # silu center Sv = floor(Yg sig / 2^30); |silu(g) - Sv| <= |Yg| delta_sig + 1 (units 2^eg)
    #   delta_sig: |sigma(g) - sig 2^-30| <= 2^-12 / 4 (grid, L = 1/4) + 2^-31 (rounding) + e^-16 (clamp) <= 2^-13.9
    prod = mul(em, Yg, Sg, tag + ".sp", bits=61)
    Sv = shr_pub(em, prod, SIG_OUT, tag + ".sv")
    Sv.bits = 30
    aYg, _ = absx(em, Yg, tag + ".ayg")
    dsig = Fraction(1, 1 << 14) + Fraction(1, 1 << 31) + Fraction(1, 1 << 23)
    Ds = em.lin([(1, mul_up(em, aYg, dsig, tag + ".ds"))], 1, bits=20)
    # silu rule: r_s = L r_g + u_e (|silu| + L r_g) + eta_silu   (+ Ds, the center's own error)
    LRg = mul_up(em, Rg, c.silu_L, tag + ".lrg")
    aSv, _ = absx(em, Sv, tag + ".asv")
    t2 = mul_up(em, em.lin([(1, aSv), (1, Ds), (1, LRg)]), c.u_elem, tag + ".ue")
    esil = em.view(eta_units(em, eg, tag + ".esil", log2eta=c.eta_silu_log2), OS)
    Rs = em.lin([(1, LRg), (1, t2), (1, esil), (1, Ds)])
    Rs, _ = rv_store(em, Sv, Rs, eg, tag + ".ss", absY=aSv)
    Rs = capped(em, Rs, 27, tag + ".Rs")
    # product a = s u: center P = Sv Yu, radius (|s| + r_s) r_u + r_s |u|
    P = mul(em, Sv, Yu, tag + ".P", bits=60)
    aYu, _ = absx(em, Yu, tag + ".ayu")
    Rp = em.lin([(1, mul(em, em.commit(em.lin([(1, aSv), (1, Rs)], bits=31), tag + ".asr"), Ru, tag + ".rp1", bits=56)),
                 (1, mul(em, Rs, aYu, tag + ".rp2", bits=56))], bits=57)
    ep = em.lin([(1, eg), (1, eu_)], bits=11)
    Ra, _ = rv_store(em, P, Rp, ep, tag + ".sa")
    Ra = capped(em, Ra, 60, tag + ".Ra")
    obs_cmp(em, dao, P, Ra, em.view(ep, OS), tag)
    return em


def isqrt_up(em, W, tag, wbits):
    """q = ceil(sqrt(W)) for W >= 0 (W < 2^wbits): q^2 >= W > (q - 1)^2."""
    qv = None
    if em.pv:
        qv = np.array([math.isqrt(int(w) - 1) + 1 if w > 0 else 0 for w in W.v], dtype=np.int64)
    qb = (wbits + 1) // 2 + 1
    Q = em.col(W.sp, tag + ".q", qv, qb)
    em.rng(Q, qb, tag + ".q")
    Q2 = mul(em, Q, Q, tag + ".q2")
    em.rng(em.lin([(1, Q2), (-1, W)], bits=2 * qb + 1), 2 * qb + 1, tag + ".ge")
    # W - (q-1)^2 - 1 >= 0  <=>  W - q^2 + 2q - 2 >= 0  (for q >= 1; q = 0 iff W = 0 handled by the first check)
    Qm = em.lin([(1, W), (-1, Q2), (2, Q)], -2, bits=2 * qb + 1)
    nz = em.col(W.sp, tag + ".nz", (qv > 0).astype(np.int64) if em.pv else None, 1)
    em.boolean(nz, tag + ".nz")
    em.zero(W.sp, lambda: Q.ex * (Ex.const(1) - nz.ex), tag + ".nzdef")
    em.rng(prodx(em, nz, em.lin([(1, Qm)], 1 << (2 * qb), bits=2 * qb + 2)), 2 * qb + 2, tag + ".lt")
    em.rng(prodx(em, nz, Qm), 2 * qb + 1, tag + ".lt2")
    return Q


def _qk_norm(em, prov, gname, Y, R, ey, OS, HS, dh, tag):
    """per-head RMSNorm with an input radius (reference RMSNorm with x.b = 0, l2 branch of ds) on (Y, R) at the
    vector exponent ey; returns (center, radius, per-head exponent) of the bf16-stored, gain-scaled output.
        s = sqrt(mean v^2 + eps), y = v g / s, ds = ||r|| / sqrt(d), smin = s - ds,
        r_y = |g| (r / smin + |v| ds / (s smin)) <= |g| t_hi (r + |v| rho) / (1 - rho),  rho = ds t_hi,
        r_y += rel32 (|y| + r_y);  two bf16 stores.
    (specs/06 §2.2 row 5; the box branch of ds is not used: the l2 branch is ~2x smaller whenever r ~ |v|.)"""
    c = C0
    m = _lg(dh)
    Yn, Rn, en, _, _ = _norm_rv(em, Y, R, ey, HS, 25, 21, tag + ".n0")    # per head, |Yn| <= 2^24
    # mean of squares (exact), enclosed as [Sl, Sh] 2^(2 en + k)
    S = sum_to(em, [Yn, Yn], HS, tag + ".S", 2 * 25 + m)
    Sn, K, Bk, f, _ = normalize(em, S, HS, 21, tag + ".sn", nonneg=True)
    eS = em.lin([(2, en), (1, K)], bits=11)
    eps_bits = prov.eps_bits
    ev = np.array([eps_bits], dtype=np.uint32).view(np.float32)[0]
    ef = Fraction(float(ev)) * dh
    xe = math.floor(math.log2(ef.numerator) - math.log2(ef.denominator)) - 19
    Dl = math.floor(ef / Fraction(2) ** xe); Dh = Dl + (0 if Fraction(Dl) * Fraction(2) ** xe == ef else 1)
    Pl, Ph, eP = add_pos(em, (Sn, em.lin([(1, Sn)], 1, bits=21), eS),
                         (em.const(HS, Dl), em.const(HS, Dh), em.const(HS, xe)), tag + ".P", 20)
    Pl, Ph = em.commit(Pl, tag + ".Pl"), em.commit(Ph, tag + ".Ph")
    Pl.bits = Ph.bits = 22
    T, et = rsqrt_enc(em, Pl, Ph, eP, dh, tag + ".rs")                     # t* in [T, T+8) 2^et
    # ds_l2 = ||r|| / sqrt(dh) <= q2 2^(en - ceil(m/2)),  q2 = ceil(sqrt(W2 2^(m mod 2))),  W2 = sum Rn^2
    W2 = sum_to(em, [Rn, Rn], HS, tag + ".W2", 2 * 21 + m)
    W2s = em.lin([(1 << (m & 1), W2)], bits=2 * 21 + m + 1)
    q2 = isqrt_up(em, em.commit(W2s, tag + ".W2s"), tag + ".q2", 2 * 21 + m + 1)
    # rho = ds t_hi <= q2 (T + 8) 2^(en + et - ceil(m/2)) -> 2^-30 fixed point rho30 (ceil), require rho <= 1/2
    Pq = mul(em, q2, em.commit(em.lin([(1, T)], RSQ_W, bits=22), tag + ".T8"), tag + ".Pq", bits=47)
    Pqn, kq, _, _, _ = normalize(em, Pq, HS, 30, tag + ".pqn", nonneg=True)
    sh = em.lin([(1, en), (1, et), (1, kq)], 30 - (m + 1) // 2, bits=12)
    shc = clamp(em, sh, -31, 1, tag + ".rsh")                                # right shifts >= 31 floor to 0 (Pqn < 2^29)
    em.rng(em.lin([(-1, sh)], 1, bits=13), 12, tag + ".rshhi")             # left shift > 1 => rho > 1: bad (Rng)
    A, B, ff = pow2(em, shc, tag + ".rp")
    A.bits = 1
    rho30 = shr_hid(em, Pqn, A, B, 31, tag + ".rho", 31, up_=True, f=ff)
    em.rng(em.lin([(-1, rho30)], 1 << 28, bits=30), 29, tag + ".rho")         # rho <= 1/4 (Rng)
    rho30 = em.commit(rho30, tag + ".rho30"); rho30.bits = 29
    # 1 / (1 - rho) <= 1 + rho + 2 rho^2   -> F30 >= 2^30 / (1 - rho)
    r2 = shr_pub(em, mul(em, rho30, rho30, tag + ".r2"), 29, tag + ".r2s", up_=True)
    F30 = em.commit(em.lin([(1, rho30), (1, r2)], 1 << 30, bits=31), tag + ".F30")   # <= 2^30 + 2^28 + 2^27
    # gains: per-element bf16 g aligned to one 16-bit fixed point (|g - G' 2^eg| <= tg 2^eg)
    SC = em.space(tag + ".gsc", [])
    GD = em.space(tag + ".gd", [a for a in OS.axes if a[0] in ("hf", "dj")])   # gains depend on the head dim only
    dg = prov.vec_dec(em, GD, gname, tag + ".g")
    prov.on_qk_gain(em, gname, dg, dh, tag + ".g")
    Gq, mG, tg, eg = align(em, dg, SC, 16, tag + ".gal")
    Gq, mG, tg = em.view(Gq, OS), em.view(mG, OS), em.view(tg, OS)
    Hv = lambda x: em.view(x, OS)
    # center: y = v g t,  Q1 = Yn G' (exact) -> normalise -> Q1n; err1 = |Yn| tg (units 2^(en+eg))
    Q1 = mul(em, Yn, Gq, tag + ".Q1", bits=41)
    aYn, _ = absx(em, Yn, tag + ".ayn")
    Q1n, k1, Bk1, f1, _ = normalize(em, Q1, HS, 29, tag + ".q1n")
    e1 = mul(em, aYn, tg, tag + ".e1", bits=25)
    kmax1 = 41 - 29 + 1
    err1 = rshift_up(em, e1, Hv(Bk1), Hv(f1), kmax1, tag + ".e1s", 25, extra=1)   # units of Q1n
    # Q2 = Q1n T; |vgt* - Q2| <= (|Q1n| + err1) 8 + err1 T
    Tv = Hv(T)
    Q2 = mul(em, Q1n, Tv, tag + ".Q2", bits=50)
    aQ1n, _ = absx(em, Q1n, tag + ".aq1")
    err2 = em.lin([(RSQ_W, aQ1n), (RSQ_W, err1), (1, mul(em, em.commit(err1, tag + ".err1"), Tv, tag + ".e1T", bits=48))],
                  bits=50)
    Yq, Ryq, eq_, Bk2, f2 = _norm_rv(em, Q2, err2, em.lin([(1, en), (1, eg), (1, k1), (1, et)], bits=11), HS, 28, 24,
                                     tag + ".q2n")
    # engine radius r_y = |g| t_hi F (r + |v| rho) in the same chain of units
    B1 = em.lin([(1, Rn), (1, shr_pub(em, mul(em, aYn, Hv(rho30), tag + ".ar", bits=55), 30, tag + ".ars", up_=True))],
                bits=26)
    gB = mul(em, em.commit(em.lin([(1, mG), (1, tg)], bits=17), tag + ".gt"), em.commit(B1, tag + ".B1"), tag + ".gB",
             bits=43)
    gBs = rshift_up(em, gB, Hv(Bk1), Hv(f1), kmax1, tag + ".gBs", 31, extra=0)  # units of Q1n
    T8 = Hv(em.commit(em.lin([(1, T)], RSQ_W, bits=22), tag + ".T8b"))
    gBT = mul(em, em.commit(gBs, tag + ".gBs2"), T8, tag + ".gBT", bits=54)     # units of Q2
    # shift to the final scale (the normalisation of Q2) and apply F30 / 2^30
    kmax2 = 50 - 28 + 1
    gBTs = rshift_up(em, gBT, Hv(Bk2), Hv(f2), kmax2, tag + ".gBTs", 29, extra=0)
    ry = shr_pub(em, mul(em, em.commit(gBTs, tag + ".gBTs2"), Hv(F30), tag + ".ryF", bits=61), 30, tag + ".ry", up_=True)
    Rtot = em.lin([(1, ry), (1, Ryq)], bits=32)
    # rel32 (fp32 internals) and two bf16 stores
    rho32 = (1 + c.gamma(dh) + 2 * c.u_acc) * (1 + c.u_elem) * (1 + c.u_acc) - 1
    aYq, _ = absx(em, Yq, tag + ".ayq")
    Rtot = em.lin([(1, Rtot), (1, mul_up(em, em.lin([(1, aYq), (1, Rtot)], bits=33), rho32, tag + ".r32"))], bits=33)
    R1, _ = rv_store(em, Yq, Rtot, eq_, tag + ".st1", absY=aYq)
    R2, _ = rv_store(em, Yq, R1, eq_, tag + ".st2", absY=aYq)
    R2 = capped(em, R2, 26, tag + ".R2")
    return Yq, R2, eq_


def rope_fx(a):
    """RoPE table entry on the 2^-30 grid (floor; |error| < 2^-30, charged by _rope)."""
    return np.floor(np.asarray(a, dtype=np.float64) * (1 << 30)).astype(np.int64)


ROPE_TAB = "cz.rope"


def rope_table_cols(rope, t_pad, half):
    """Public RoPE table (profile: model rope function, T_pad, dh): one row per (position, j), columns
        (pos * half + j, cl, ch, sl, sh, |cl| + 1, |ch| + 1, |sl| + 1, |sh| + 1)
    with cl = floor(cos[j] 2^30), ch = floor(cos[half + j] 2^30), sl / sh likewise: bit-identical to the entries the
    public-row path of _rope puts on the circuit for the same position."""
    cols = [[] for _ in range(9)]
    for p in range(t_pad):
        cos, sin = rope(p)
        cl, ch = rope_fx(cos[:half]), rope_fx(cos[half:2 * half])
        sl, sh = rope_fx(sin[:half]), rope_fx(sin[half:2 * half])
        for k, v in enumerate((p * half + np.arange(half, dtype=np.int64), cl, ch, sl, sh,
                               np.abs(cl) + 1, np.abs(ch) + 1, np.abs(sl) + 1, np.abs(sh) + 1)):
            cols[k].append(np.asarray(v, dtype=np.int64))
    return [np.concatenate(c) for c in cols]


def rope_hidden(em, prov, pos_x, JS, dh, tag):
    """Hidden-position RoPE rows (specs/09 §3.2, zero knowledge: the position is a hidden committed scalar).  The eight
    rows cl, ch, sl, sh, |.| + 1 are COMMITTED columns on the head-free space (dj) and constrained by one LogUp into
    the public table ROPE_TAB with rows (pos, j, ...) for pos < T_pad (profile constants T_pad, dh): a prover that
    commits the rows of another position (or any other value) leaves the table.  Viewed on JS (broadcast over the
    head axes).  Returns the eight X on JS in the order (cl, ch, sl, sh, acl, ach, asl, ash)."""
    half = dh // 2
    hb = _lg(half)
    assert (1 << hb) == half and prov.t_pad, ("RoPE table needs a power-of-two half dimension and t_pad", dh, prov.t_pad)
    cache = prov.__dict__.setdefault("_rope_hid", {})
    key = (id(pos_x), id(em))
    if key not in cache:
        tab = rope_table_cols(prov.rope, int(prov.t_pad), half)
        em.table(ROPE_TAB, tab)
        RS = em.space("rope", [("dj", hb)])
        names = ("cl", "ch", "sl", "sh", "acl", "ach", "asl", "ash")
        vals = [None] * 8
        pv = em.view(pos_x, RS)
        if em.pv:
            K_ = 1 << em.bbits
            p_per_s = pv.v.reshape(K_, half)[:, 0]
            v_arr = [np.zeros((K_, half), dtype=np.int64) for _ in range(8)]
            for up in np.unique(p_per_s):
                cos, sin = prov.rope_wit(int(up))
                cl, ch = rope_fx(cos[:half]), rope_fx(cos[half:2 * half])
                sl, sh = rope_fx(sin[:half]), rope_fx(sin[half:2 * half])
                row8 = (cl, ch, sl, sh, np.abs(cl) + 1, np.abs(ch) + 1, np.abs(sl) + 1, np.abs(sh) + 1)
                msk = (p_per_s == up)
                for ii in range(8):
                    v_arr[ii][msk] = row8[ii]
            vals = [va.reshape(-1) for va in v_arr]
        cols = [em.col(RS, "rope." + n, v, 31) for n, v in zip(names, vals)]
        kx = X(RS, (pv.v * half + RS.grid("dj")) if em.pv else None, 40,
               ex=(pv.ex * Ex.const(half) + RS.S.idx("dj")) if em.emit else None)
        em.lookup(ROPE_TAB, RS, [kx] + cols, "rope", _member(tab) if em.pv else None)
        cache[key] = cols
    cols = cache[key]
    return [em.view(c, JS, {"dj": "dj"}) for c in cols]


def _rope(em, prov, Y, R, e, OS, JS, dobs_lo, dobs_hi, rope_key, tag, t_pad=None):
    """RoPE on (Y, R) at per-head exponent e (on a group space of OS), then bf16 store, compared
    with the observed post-RoPE halves.  cos / sin: the committed bf16 table row, floored to 2^-30 (|error| < 2^-30
    each, charged).  out_lo = x_lo c_lo - x_hi s_lo, out_hi = x_hi c_hi + x_lo s_hi.
    Position: an int (public-row path: the rows are public constants of the circuit, clear oracle / unit tests) or a
    hidden scalar X (``rope_key`` or ``prov.pos_x``): the rows are then hidden committed columns bound to one public
    table by LogUp (``rope_hidden``)."""
    c = C0
    pos_x = rope_key if isinstance(rope_key, X) else getattr(prov, "pos_x", None)
    lo = lambda x: em.view(x, JS, {"hf": ("c", 0)})
    hi = lambda x: em.view(x, JS, {"hf": ("c", 1)})
    if pos_x is not None:
        if t_pad is not None and not prov.t_pad:
            prov.t_pad = t_pad
        dh = 2 * (1 << dict(JS.axes)["dj"])
        Cl, Ch, Sl, Sh, aCl, aCh, aSl, aSh = rope_hidden(em, prov, pos_x, JS, dh, tag)
    else:
        assert not prov.require_hidden_rope, "hidden provider: the RoPE position must be a hidden scalar"
        cos, sin = prov.rope_rows(rope_key)                 # float64 arrays (dh,) of exact bf16 values
        half = len(cos) // 2
        fx = rope_fx
        jidx = JS.grid("dj")
        Cl = em.pub(JS, tag + ".cl", fx(cos[:half])[jidx], 31); Ch = em.pub(JS, tag + ".ch", fx(cos[half:])[jidx], 31)
        Sl = em.pub(JS, tag + ".sl", fx(sin[:half])[jidx], 31); Sh = em.pub(JS, tag + ".sh", fx(sin[half:])[jidx], 31)
        aCl = em.pub(JS, tag + ".acl", np.abs(fx(cos[:half])[jidx]) + 1, 31)
        aCh = em.pub(JS, tag + ".ach", np.abs(fx(cos[half:])[jidx]) + 1, 31)
        aSl = em.pub(JS, tag + ".asl", np.abs(fx(sin[:half])[jidx]) + 1, 31)
        aSh = em.pub(JS, tag + ".ash", np.abs(fx(sin[half:])[jidx]) + 1, 31)
    aY, _ = absx(em, Y, tag + ".ay")
    Xl, Xh, Rl, Rh, Al, Ah = lo(Y), hi(Y), lo(R), hi(R), lo(aY), hi(aY)
    out = []
    for (Xa, Xb, Ra, Rb, Aa, Ab, Ca, Sa, aCa, aSa, sg, nm, dob) in (
            (Xl, Xh, Rl, Rh, Al, Ah, Cl, Sl, aCl, aSl, -1, "lo", dobs_lo),
            (Xh, Xl, Rh, Rl, Ah, Al, Ch, Sh, aCh, aSh, 1, "hi", dobs_hi)):
        V = em.lin([(1, prodx(em, Xa, Ca)), (sg, prodx(em, Xb, Sa))], bits=60)
        Vc = em.commit(V, tag + ".v" + nm)
        # rp = |c| r_a + |s| r_b; A = (|x_a| + r_a)|c|; B = (|x_b| + r_b)|s|; r = rp + u (A + B) + 2 eta + table err
        rp = em.lin([(1, prodx(em, Ra, aCa)), (1, prodx(em, Rb, aSa))], bits=57)
        XA = em.commit(em.lin([(1, Aa), (1, Ra)], bits=29), tag + ".xa" + nm)
        XB = em.commit(em.lin([(1, Ab), (1, Rb)], bits=29), tag + ".xb" + nm)
        AB = em.lin([(1, prodx(em, XA, aCa)), (1, prodx(em, XB, aSa))], bits=61)
        rr = em.lin([(1, rp), (1, mul_up(em, em.commit(AB, tag + ".ab" + nm), c.u_store, tag + ".u" + nm)),
                     (1, XA), (1, XB), (2, em.view(eta_units(em, em.lin([(1, e)], -30, bits=12), tag + ".e" + nm), JS))],
                    bits=58)
        eo = em.lin([(1, e)], -30, bits=12)
        Rr, _ = rv_store(em, Vc, rr, eo, tag + ".st" + nm)
        Rr = capped(em, Rr, 59, tag + ".R" + nm)
        obs_cmp(em, dob, Vc, Rr, em.view(eo, JS), tag + "." + nm)
        out.append((Vc, Rr))
    return out


def _blk_qkv_paths(em, prov, xin, paths, pos, prm, tag):
    d, dh = prm["d"], prm["dh"]
    hb = _lg(dh) - 1
    K_ = 1 << em.bbits
    XS = em.space(tag + ".x", [("i", _lg(d))])
    SC = em.space(tag + ".sc", [])
    dxi = decode_bf16(em, XS, _pad_sp(xin, XS), tag + ".xi")
    Xs, mX, t, ex = align(em, dxi, SC, gemm_mb(prov, XS.in_bits), tag + ".al")
    for nm, haxes, obs, gname in paths:
        OS = em.space(tag + "." + nm, haxes + [("hf", 1), ("dj", hb)])
        HS = em.space(tag + "." + nm + "h", haxes)
        Y, R0, ey = gemm(em, prov, nm, Xs, mX, t, ex, OS, tag + "." + nm)
        R0 = capped(em, R0, 56, tag + nm + ".R0")
        R1, _ = rv_store(em, Y, R0, ey, tag + "." + nm + ".s")
        R1 = capped(em, R1, 59, tag + nm + ".R1")
        obs_2d = np.asarray(obs, dtype=np.int64).reshape(K_, -1)
        if gname is None:
            dob = decode_bf16(em, OS, _pad_sp(obs_2d, OS), tag + ".vo")
            obs_cmp(em, dob, Y, R1, em.view(ey, OS), tag + ".v")
            continue
        Yq, Rq, eq_ = _qk_norm(em, prov, gname, Y, R1, ey, OS, HS, dh, tag + "." + nm)
        JS = em.space(tag + "." + nm + "j", haxes + [("dj", hb)])
        c3 = obs_2d.reshape(K_, -1, 2, dh // 2)
        dlo = decode_bf16(em, JS, _pad_sp(c3[:, :, 0, :].reshape(K_, -1), JS), tag + "." + nm + ".olo")
        dhi = decode_bf16(em, JS, _pad_sp(c3[:, :, 1, :].reshape(K_, -1), JS), tag + "." + nm + ".ohi")
        _rope(em, prov, Yq, Rq, eq_, OS, JS, dlo, dhi, pos, tag + "." + nm + ".rope",
              t_pad=1 << prm["tb"] if "tb" in prm else None)
    return em


def blk_qkv(em, prov, xin, q_obs, k_obs, v_obs, pos, prm, tag="QKV"):
    """x -> q, k, v (three GEMMs, bf16 stores) -> q/k per-head RMSNorm (bf16) -> RoPE (bf16); v compared after its
    store (reference QKV block: qk_path, v = store(gemm(Wv, x)))."""
    nq, nkv = prm["n_q"], prm["n_kv"]
    gb, rb = _lg(nkv), _lg(nq // nkv)
    paths = (("q", [("g", gb), ("r", rb)], q_obs, "q_norm"),
             ("k", [("g", gb)], k_obs, "k_norm"),
             ("v", [("g", gb)], v_obs, None))
    return _blk_qkv_paths(em, prov, xin, paths, pos, prm, tag)


def blk_kv(em, prov, xin, k_obs, v_obs, pos, prm, tag="KV"):
    """Key-producer KV projection + k RMSNorm + k RoPE + v store (identical to blk_qkv's k and v paths without q)."""
    gb = _lg(prm["n_kv"])
    paths = (("k", [("g", gb)], k_obs, "k_norm"),
             ("v", [("g", gb)], v_obs, None))
    return _blk_qkv_paths(em, prov, xin, paths, pos, prm, tag)


def att_consts(T_pad, c=C0):
    """public FlashAttention contract constants at the profile bound T_pad (specs/06 §2.2 row 7; LSE derivation)."""
    n_t = -(-T_pad // c.B_c) + c.split_steps
    hB = c.h(c.B_c)
    dr = Fraction(1, 1 << EXP_F)                                          # exp-table weight error (multiplicative)
    K1 = (1 + c.u_elem) ** (n_t + 1) * (1 + dr)
    emu = (1 + c.u_P) * (1 + c.u_acc) ** (2 * n_t + hB) - 1
    emu2 = (1 + c.u_acc) ** (2 * n_t + hB) - 1
    ea = (1 + c.u_elem) ** (n_t + 1) * (1 + c.u_acc) ** (2 * n_t + hB) - 1
    return dict(n_t=n_t, K1=K1, emu=emu, emu2=emu2, ea=ea)


def qk_exponent(gain_codes, dh):
    """public per-layer score exponent: |q_i| <= sqrt(2) max|g| sqrt(dh) (1 + 2^-6) after q/k-norm and RoPE (RMSNorm
    cap, pair-norm preservation); returns e with cap < 2^(e + 25)."""
    g = np.abs((np.asarray(gain_codes, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)).max()
    cap = math.sqrt(2.0) * g * math.sqrt(dh) * (1 + 2.0 ** -6)
    return int(math.ceil(math.log2(cap))) - 25


def qk_a_table(dh):
    """Table {(a, c(a))}: c(a) = ceil(log2(sqrt(2 dh) (1 + 2^-6) a)) for a in [1, 255], computed by calling ``qk_exponent``
    itself on a bf16 code with significand a (E = 0 for a >= 128, the subnormal E = -133 below).  For every finite
    nonzero bf16 magnitude g = a 2^E:  qk_exponent([g], dh) = E + c(a) - 25  (power-of-two scaling is exact in float64;
    ``flashprove_next.fp_ref`` checks the identity exhaustively for several dh)."""
    key = ("qka", int(dh))
    if key not in _TAB:
        a_ = np.arange(1, 256, dtype=np.int64)
        c_ = []
        for a in a_:
            code, E = (((134 << 7) | (int(a) - 128)), 0) if a >= 128 else (int(a), -133)
            c_.append(qk_exponent(np.array([code], dtype=np.uint16), dh) + 25 - E)
        _TAB[key] = [a_, np.array(c_, dtype=np.int64)]
    return _TAB[key]


def qk_exp_hidden(em, dg, dh, tag, delta=0):
    """Hidden committed score exponent e = qk_exponent(gain vector) of the decoded hidden gain vector ``dg`` (a Dec on
    the per-head space, ``dg.code`` / ``dg.s`` committed).  EXACT in-circuit equality, no soundness slack:
      * mag_i = code_i - 2^15 s_i is the magnitude code of gain i; bf16 magnitude order is code order, so
        M = max_i mag_i (``gmax``) is the code of max |g| (padding entries are zero codes);
      * (M, 0, a, E) is a row of the finite-bf16 decode table (a significand, E exponent: max |g| = a 2^E), (a, c) a row of
        ``qk_a_table`` (a >= 1: an all-zero gain vector is not provable), and e = E + c - 25 = qk_exponent(g).
    ``delta`` is the cheating prover's offset on the committed e (test hook): any e' != e violates e' = E + c - 25 and
    any other (a, E, c) leaves one of the two tables."""
    tab = _bf16_table()
    em.table("cz.bf16", tab)
    sp = dg.code.sp
    SC = em.space(tag + ".qsc", [])
    mag = em.lin([(1, dg.code), (-(1 << 15), dg.s)], bits=15)
    M = gmax(em, mag, SC, tag + ".mag")
    qa = qk_a_table(dh)
    qname = f"cz.qka.{int(dh)}"
    em.table(qname, qa)
    a = E = cc = None
    if em.pv:
        mv = M.v
        _, A_, E_ = _TAB["bf16_dec"]
        em.fail("qk:zero_gain", int((A_[mv] == 0).sum()))
        a, E = A_[mv], E_[mv]
        lut = np.zeros(256, dtype=np.int64)
        lut[qa[0]] = qa[1]
        cc = lut[a]
    ax = em.col(SC, tag + ".a", a, 8)
    Ex_ = em.col(SC, tag + ".E", E, 9)
    cx = em.col(SC, tag + ".c", cc, 8)
    em.lookup("cz.bf16", SC, [M, em.const(SC, 0), ax, Ex_], tag + ".dec", _member(tab) if em.pv else None)
    em.lookup(qname, SC, [ax, cx], tag + ".ac", _member(qa) if em.pv else None)
    e_v = None if not em.pv else (E + cc - 25 + int(delta))
    ex_ = em.col(SC, tag + ".e", e_v, 9)
    if em.pv:
        em.fail("qk:e", int((ex_.v != E + cc - 25).sum()))
    em.zero(SC, lambda: ex_.ex - Ex_.ex - cx.ex + Ex.const(25), tag + ".edef")
    return ex_


def shr_window(em, x, s_pub, hs, W, tag, up_=False, f_ok=True):
    """floor(x / 2^(s_pub + hs)) (+ 1 with up_) for a PUBLIC s_pub >= 0 and a HIDDEN scalar shift hs in [0, W] (X on a scalar
    space; both ranges are enforced by the caller): public stage (floor), then the hidden stage (two floors compose
    exactly) -- the profile window keeps the hidden remainder small (x.bits + W <= LIM)."""
    xs = shr_pub(em, x, s_pub, tag + ".pub") if s_pub > 0 else x
    A, B, f = pow2(em, em.lin([(-1, hs)], bits=7), tag + ".hp")           # B = 2^hs, f = [hs > 0]
    A.bits = 1
    Bv, fv = em.view(B, xs.sp), em.view(f, xs.sp)
    one = em.const(xs.sp, 1)
    one.bits = 1
    return shr_hid(em, xs, one, Bv, max(1, W), tag + ".hid", max(1, xs.bits), up_=up_, f=fv)


def blk_att(em, prov, layer, q_obs, K_obs, V_obs, o_obs, lse_obs, T, prm, tag="ATT"):
    """FlashAttention contract (specs/06 §2.2 row 7; P stored bf16 before PV; split-KV), observed LSE (M1),
    output cap (hull expanded), for one query position with T visible keys (T <= 2^tb, hidden; keys >= T masked).
    q, K, V: post-RoPE observed bf16 codes; o: bf16; lse: fp32 (natural log, FA2)."""
    c = C0
    nq, nkv, dh, tb = prm["n_q"], prm["n_kv"], prm["dh"], prm["tb"]
    Tp = 1 << tb
    cst = att_consts(Tp)
    gb, rb, db = _lg(nkv), _lg(nq // nkv), _lg(dh)
    MBv = 29 - tb                               # E (31 b) x |V - O| (MBv + 2 b) x 2^tb keys < 2^62
    eqc, ekc = prov.qk_exp(layer)               # caps: |q| < 2^(eqc+26), |k| < 2^(ekc+26); ints (public) or hidden X
    hid_e = isinstance(eqc, X)                  # hidden committed exponents (CfgTiledWeights, qk_exp_hidden)
    Cs, Es = prov.scale_fx()                    # scale = Cs 2^Es, Cs in [2^23, 2^24)
    EW = 10                                     # per-head exponents live in [cap - 10, cap]
    if hid_e:
        # eq + ek lies in the PUBLIC profile window [u_lo, u_hi] (Lt score-exponent window, specs/09 §3.2); the two exponent
        # shifts below are  K - (eq + ek)  =  (K - u_hi) public + (u_hi - eq - ek) hidden in [0, W]
        u_lo, u_hi = prov.qk_window
        Wwin = u_hi - u_lo
        se_pub = -EXP_F - (u_hi + 24 + Es)      # = se_lo at the top of the window
        s3_pub = -((u_hi + 24 + Es) + 30)
        assert Wwin >= 0 and Wwin <= 20 and se_pub >= 0 and s3_pub >= 1, (u_lo, u_hi, Es)
        SCe = eqc.sp
        Usum = em.commit(em.lin([(1, eqc), (1, ekc)], bits=11, sp=SCe), tag + ".U")
        em.rng(em.lin([(1, Usum)], -u_lo, bits=11), max(1, bitlen(Wwin)), tag + ".Ulo")
        em.rng(em.lin([(-1, Usum)], u_hi, bits=11), max(1, bitlen(Wwin)), tag + ".Uhi")
        hs = em.commit(em.lin([(-1, Usum)], u_hi, bits=6), tag + ".hs")       # hidden remainder shift in [0, W]
    else:
        ezmax = eqc + ekc + 24 + Es
        se_lo = -EXP_F - ezmax                  # dZ (units 2^ez) -> 2^-EXP_F grid: right shift se_lo + s1, s1 in [0, 2EW]
        s3_lo = -(ezmax + 30)                   # zmax -> 2^-30 grid
        assert se_lo >= 0 and s3_lo >= 1, (eqc, ekc, Es)
    QS = em.space(tag + ".q", [("g", gb), ("r", rb), ("d", db)])
    KS = em.space(tag + ".k", [("g", gb), ("j", tb), ("d", db)])
    ZS = em.space(tag + ".z", [("g", gb), ("r", rb), ("j", tb)])
    PS = em.space(tag + ".p", [("g", gb), ("r", rb), ("j", tb), ("d", db)])
    HS = em.space(tag + ".h", [("g", gb), ("r", rb)])
    VG = em.space(tag + ".vg", [("g", gb), ("d", db)])
    JS = em.space(tag + ".j", [("j", tb)])
    SC = em.space(tag + ".sc", [])
    # ---- key mask: mk_j = [j < T], hidden T
    mkv = (np.arange(Tp) < T).astype(np.int64) if em.pv else None
    mk = em.col(JS, tag + ".mk", mkv, 1)
    em.boolean(mk, tag + ".mk")
    if em.emit:                                 # mk_0 = 1, mk_(j+1) <= mk_j
        JS.S.zero((mk.ex - Ex.const(1)) * JS.S.ind(j=0), tag=tag + ".mk0")
        JS.S.zero(mk.col.shifted(1) * (Ex.const(1) - mk.ex), sel=JS.S.lt("j", Tp - 1), tag=tag + ".mkmono")
    Tm = sum_to(em, [mk], SC, tag + ".T", tb + 1)
    # ---- observed operands
    def kcodes(A):
        a = np.zeros((nkv, Tp, dh), dtype=np.int64)
        A = np.asarray(A, dtype=np.int64)
        a[:, :A.shape[1], :] = A[:, :Tp, :]
        if em.pv:
            a[:, T:, :] = 0
        return a.reshape(-1)
    dq = decode_bf16(em, QS, _pad(q_obs, QS.n), tag + ".qd")
    dK = decode_bf16(em, KS, kcodes(K_obs), tag + ".Kd")
    dV = decode_bf16(em, KS, kcodes(V_obs), tag + ".Vd")
    do_ = decode_bf16(em, QS, _pad(o_obs, QS.n), tag + ".od")
    dl = decode_f32(em, HS, _pad(lse_obs, HS.n), tag + ".ld")
    mkK = em.view(mk, KS)
    em.zero(KS, lambda: (Ex.const(1) - mkK.ex) * dK.code.ex, tag + ".Kmask")
    em.zero(KS, lambda: (Ex.const(1) - mkK.ex) * dV.code.ex, tag + ".Vmask")
    # ---- scores: Q per head, K per kv head, 26-bit fixed point, exponent in [cap - EW, cap] (Rng: |q|, |k| under the
    #      public caps; heads far below the cap are truncated at the floor, flagged and charged)
    KG = em.space(tag + ".kg", [("g", gb)])
    if hid_e:
        Q, mQ, tQ, eQ = align(em, dq, HS, 26, tag + ".Q", efloor=em.lin([(1, eqc)], -EW, bits=11, sp=HS))
        K, mK, tK, eK = align(em, dK, KG, 26, tag + ".K", efloor=em.lin([(1, ekc)], -EW, bits=11, sp=KG))
        em.rng(em.lin([(-1, eQ), (1, eqc)], bits=11, sp=HS), 4, tag + ".eq")
        em.rng(em.lin([(-1, eK), (1, ekc)], bits=11, sp=KG), 4, tag + ".ek")
    else:
        Q, mQ, tQ, eQ = align(em, dq, HS, 26, tag + ".Q", efloor=eqc - EW)
        K, mK, tK, eK = align(em, dK, KG, 26, tag + ".K", efloor=ekc - EW)
        em.rng(em.lin([(-1, eQ)], eqc, bits=11), 4, tag + ".eq")                # eQ <= eqc (and >= eqc - EW by the floor)
        em.rng(em.lin([(-1, eK)], ekc, bits=11), 4, tag + ".ek")
    P_ = lambda x: em.view(x, PS)
    S = sum_to(em, [P_(Q), P_(K)], ZS, tag + ".S", 26 + 26 + db)
    Sab = sum_to(em, [P_(mQ), P_(mK)], ZS, tag + ".Sab", 26 + 26 + db)
    D1 = sum_to(em, [P_(mQ), P_(tK)], ZS, tag + ".D1", 26 + db)
    D2 = sum_to(em, [P_(tQ), P_(mK)], ZS, tag + ".D2", 26 + db)
    D3 = sum_to(em, [P_(tQ), P_(tK)], ZS, tag + ".D3", 1 + db)
    DS = em.lin([(1, D1), (1, D2), (1, D3)], bits=26 + db + 2)
    Sp_ = shr_pub(em, S, 24, tag + ".Sp")                                 # |S'| < 2^36
    Zc = em.commit(em.lin([(Cs, Sp_)], bits=Sp_.bits + 24), tag + ".Zc")   # z = Zc 2^ez, ez = eQ + eK + 24 + Es
    Dz = em.lin([(Cs, shr_pub(em, DS, 24, tag + ".Dzs"))], 2 * Cs, bits=12 + 24)
    Dz = em.commit(Dz, tag + ".Dz")
    if hid_e:
        s1 = em.lin([(-1, eQ), (-1, em.view(eK, HS)), (1, eqc), (1, ekc)], bits=6, sp=HS)   # ezmax - ez in [0, 2 EW]
    else:
        s1 = em.lin([(-1, eQ), (-1, em.view(eK, HS))], eqc + ekc, bits=6)  # ezmax - ez in [0, 2 EW]
    # engine score radius Rz = scale gamma_dh sum|q||k| + 4 u_acc (|z| + |zmax|)  (+ Dz: center error)
    Ab = em.lin([(1, Sab), (1, DS)], bits=26 + 26 + db + 1)
    Ra = mul_up(em, shr_pub(em, Ab, 24, tag + ".Abs", up_=True), c.gamma(dh) * Cs, tag + ".Ra")
    aZ, _ = absx(em, Zc, tag + ".az")
    aZD = em.commit(em.lin([(1, aZ), (1, Dz)], bits=Zc.bits + 1), tag + ".azd")
    Maz = gmax(em, aZD, HS, tag + ".maz")
    Rb = em.lin([(1, mul_up(em, aZD, 4 * c.u_acc, tag + ".Rb1")), (1, em.view(mul_up(em, Maz, 4 * c.u_acc, tag + ".Rb2"), ZS))])
    Rz = em.commit(em.lin([(1, Ra), (1, Rb), (1, Dz)], bits=Zc.bits), tag + ".Rz")
    Rzm = gmax(em, Rz, HS, tag + ".rzm")
    # x = 2 rbar in 2^-30 fixed point: 2 Rzm 2^(ez + 30) = 2 Rzm / 2^(s3_lo + s1); x <= 1 (Rng, via the range of y)
    One_h = em.const(HS, 1); One_h.bits = 0
    _, B1, f1 = pow2(em, em.lin([(-1, s1)], bits=6), tag + ".p1")         # B1 = 2^s1
    if hid_e:
        R2p = shr_window(em, em.lin([(2, Rzm)], bits=Rzm.bits + 1), s3_pub, hs, Wwin, tag + ".x30p", up_=True)
    else:
        R2p = shr_pub(em, em.lin([(2, Rzm)], bits=Rzm.bits + 1), s3_lo, tag + ".x30p", up_=True)
    X30 = shr_hid(em, R2p, One_h, B1, 2 * EW, tag + ".x30", 31, up_=True, f=f1)
    X30 = capped(em, X30, 31, tag + ".X30")
    em.rng(em.lin([(-1, X30)], 1 << 30, bits=32), 31, tag + ".xle1")
    # ---- max over visible keys, exponent argument on the 2^-18 grid, two-level exp table
    mkZ = em.view(mk, ZS)
    Zm = em.commit(X(ZS, (Zc.v - ((1 - mkZ.v) << 60)) if em.pv else None, 61,
                     ex=(Zc.ex - Ex.const(1 << 60) * (Ex.const(1) - mkZ.ex)) if em.emit else None), tag + ".Zm")
    Zmax = gmax(em, Zm, HS, tag + ".zmax")
    em.rng(em.lin([(1, Zmax)], 1 << 60, bits=62), 62, tag + ".zmaxlo")    # a visible key exists: Zmax >= -2^60
    Zmax.bits = 61
    dZ = em.lin([(1, em.view(Zmax, ZS)), (-1, Zc)], bits=61)            # [0, 2^61) visible, |.| <= 2^60 masked
    # d 2^ez on the 2^-EXP_F grid: public shift se_lo, then min with 2^(23 + s1) and hidden shift s1 <= 2 EW
    if hid_e:
        dZ16 = shr_window(em, em.commit(dZ, tag + ".dZ"), se_pub, hs, Wwin, tag + ".dq")
    else:
        dZ16 = shr_pub(em, em.commit(dZ, tag + ".dZ"), se_lo, tag + ".dq")
    CAPd, _, _ = pow2(em, em.lin([(1, s1)], 23, bits=6), tag + ".cap")
    CAPd.bits = 23 + 2 * EW + 1
    dmin = minmax2(em, dZ16, em.view(CAPd, ZS), tag + ".dmin", False)
    dmin.bits = 23 + 2 * EW + 1
    dqr = shr_hid(em, dmin, em.const(ZS, 1), em.view(B1, ZS), 2 * EW, tag + ".dqh", 25, f=em.view(f1, ZS))
    dqc = clamp(em, dqr, 0, EXP_DMAX << EXP_F, tag + ".dqc")
    e1t, e2t = _exp_tables()
    em.table("cz.e1", e1t); em.table("cz.e2", e2t)
    dh_ = dl_ = None
    if em.pv:
        dh_ = dqc.v >> EXP_HI_SH; dl_ = dqc.v & ((1 << EXP_HI_SH) - 1)
    DH = em.col(ZS, tag + ".dh", dh_, 12)
    DL = em.col(ZS, tag + ".dl", dl_, EXP_HI_SH)
    em.zero(ZS, lambda: dqc.ex - DH.ex * Ex.const(1 << EXP_HI_SH) - DL.ex, tag + ".dsplit")
    em.rng(DL, EXP_HI_SH, tag + ".dl")
    T1 = em.col(ZS, tag + ".e1", e1t[1][dh_] if em.pv else None, 31)
    T2 = em.col(ZS, tag + ".e2", e2t[1][dl_] if em.pv else None, 31)
    em.lookup("cz.e1", ZS, [DH, T1], tag + ".e1", _dense_check(0, len(e1t[0])))
    em.lookup("cz.e2", ZS, [DL, T2], tag + ".e2", _dense_check(0, len(e2t[0])))
    E = shr_pub(em, mul(em, T1, T2, tag + ".E12", bits=61), EXP_OUT, tag + ".E")    # table entries <= 2^30
    Em_ = mul(em, em.commit(E, tag + ".Ec"), mkZ, tag + ".Em", bits=31)     # masked weights, 0 <= E <= 2^30
    em.rng(Em_, 31, tag + ".Erng"); Em_.bits = 31
    # ---- weighted values; V aligned per (g, d) over keys
    Vx, mV, tV, ev = align(em, dV, VG, MBv, tag + ".V")
    Nn = sum_to(em, [P_(Em_), P_(Vx)], QS, tag + ".N", 31 + MBv + tb)
    Dd = sum_to(em, [Em_], HS, tag + ".D", 31 + tb)
    Ddq = em.view(Dd, QS)
    o_ = r_ = None
    if em.pv:
        o_ = np.floor_divide(Nn.v, Ddq.v); r_ = Nn.v - o_ * Ddq.v
    O = em.col(QS, tag + ".O", o_, MBv + 1)
    Or = em.col(QS, tag + ".Or", r_, 31 + tb)
    em.zero(QS, lambda: Nn.ex - O.ex * Ddq.ex - Or.ex, tag + ".div")
    em.rng(Or, 31 + tb, tag + ".orlo")
    em.rng(em.lin([(1, Ddq), (-1, Or)], -1, bits=32 + tb), 31 + tb, tag + ".orhi")
    em.srng(O, MBv + 1, tag + ".O")
    # sum p~ |v - o'| <= M1 / D + 3  (|v - o'| <= |V - O| + tV + 2)
    Ad, _ = absx(em, em.lin([(1, P_(Vx)), (-1, P_(O))], bits=MBv + 2), tag + ".dev")
    M1 = sum_to(em, [P_(Em_), Ad], QS, tag + ".M1", 31 + MBv + 2 + tb)
    m1q = m1r = None
    if em.pv:
        m1q = np.floor_divide(M1.v, Ddq.v); m1r = M1.v - m1q * Ddq.v
    M1q = em.col(QS, tag + ".m1q", m1q, MBv + 3)
    M1r = em.col(QS, tag + ".m1r", m1r, 31 + tb)
    em.zero(QS, lambda: M1.ex - M1q.ex * Ddq.ex - M1r.ex, tag + ".m1div")
    em.rng(M1r, 31 + tb, tag + ".m1lo")
    em.rng(em.lin([(1, Ddq), (-1, M1r)], -1, bits=32 + tb), 31 + tb, tag + ".m1hi")
    em.rng(M1q, MBv + 3, tag + ".m1q")
    Wd = em.lin([(1, M1q)], 4, bits=MBv + 4)
    # ---- per-head contract scalars (2^-30 fixed point, upper bounds)
    Xsq = shr_pub(em, mul(em, X30, X30, tag + ".x2", bits=61), 30, tag + ".x2s", up_=True)   # X30 <= 2^30
    Ex30 = em.commit(em.lin([(1, X30), (1, Xsq)], 1 << 30, bits=33), tag + ".ex30")
    One = capped(em, mul_up(em, Ex30, cst["K1"], tag + ".k1"), 33, tag + ".one")
    eN = capped(em, em.lin([(1, mul_up(em, One, 1 + cst["emu"], tag + ".en"))], -(1 << 30), bits=34), 29, tag + ".eN")
    eD = capped(em, em.lin([(1, mul_up(em, One, 1 + cst["emu2"], tag + ".ed"))], -(1 << 30), bits=34), 28, tag + ".eD")
    # eps_D <= 1/4 (Rng): 1/(1 - eps_D) <= 1 + eps_D + 2 eps_D^2 <= 1.375  -> Inv < 2^31
    Inv = em.commit(em.lin([(1, eD), (1, shr_pub(em, mul(em, eD, eD, tag + ".ed2"), 29, tag + ".ed2s", up_=True))],
                           1 << 30, bits=31), tag + ".inv")
    C1 = capped(em, shr_pub(em, mul(em, eN, Inv, tag + ".c1m", bits=60), 30, tag + ".c1", up_=True), 30, tag + ".C1")
    om = capped(em, mul_up(em, One, cst["emu"] + cst["emu2"], tag + ".om"), 30, tag + ".omc")
    C2 = capped(em, shr_pub(em, mul(em, om, Inv, tag + ".c2m", bits=61), 30, tag + ".c2", up_=True), 30, tag + ".C2")
    Qv = lambda x: em.view(x, QS)
    aO, _ = absx(em, O, tag + ".ao")
    # r_main = [eps_N sum p~|v-o'| + (1+eps_s)(eps_mu+eps_mu') |o'|] / (1 - eps_D)
    rm = em.lin([(1, prodx(em, Qv(C1), Wd)), (1, prodx(em, Qv(C2), em.lin([(1, aO)], 2)))], bits=MBv + 36)
    Rmain = shr_pub(em, em.commit(rm, tag + ".rm"), 30, tag + ".rmain", up_=True)
    # hull per (g, d) over visible keys, Vmax, additive weight error r_a = T Vmax 2^-24
    mkK2 = em.view(mk, KS)
    big = 1 << (MBv + 2)
    Vlo = em.commit(X(KS, (Vx.v + (1 - mkK2.v) * big) if em.pv else None, MBv + 3,
                      ex=(Vx.ex + Ex.const(big) * (Ex.const(1) - mkK2.ex)) if em.emit else None), tag + ".vlo")
    Vhi = em.commit(X(KS, (Vx.v - (1 - mkK2.v) * big) if em.pv else None, MBv + 3,
                      ex=(Vx.ex - Ex.const(big) * (Ex.const(1) - mkK2.ex)) if em.emit else None), tag + ".vhi")
    mn = gmax(em, Vlo, VG, tag + ".vmin", neg=True)
    mx = gmax(em, Vhi, VG, tag + ".vmax")
    Vmax = em.lin([(1, minmax2(em, mx, em.lin([(-1, mn)]), tag + ".vabs", True))], 1, bits=MBv + 2)
    Vmc = em.commit(Vmax, tag + ".vmx")
    TV = mul(em, em.view(Tm, VG), Vmc, tag + ".TV", bits=tb + MBv + 3)
    ra = shr_pub(em, TV, 24, tag + ".ra", up_=True)
    Rpre = em.lin([(1, Rmain), (1, Qv(ra))], 2, bits=MBv + 8)
    Rdiv = em.lin([(1, Rpre), (1, mul_up(em, em.lin([(1, aO), (1, Rpre)], 2, bits=MBv + 9), c.u_acc, tag + ".dv"))],
                  bits=MBv + 9)
    # cap: [mn - 1 - x Vmax, mx + 1 + x Vmax] (x = C2 / 2^30), then division rounding
    xv = em.lin([(1, shr_pub(em, mul(em, em.view(C2, VG), Vmc, tag + ".xvm", bits=MBv + 33), 30, tag + ".xv", up_=True))],
                1, bits=MBv + 3)
    lo = em.commit(em.lin([(1, mn), (-1, xv)], bits=MBv + 4), tag + ".lo")
    hi = em.commit(em.lin([(1, mx), (1, xv)], bits=MBv + 4), tag + ".hi")
    alo, _ = absx(em, lo, tag + ".alo")
    ahi, _ = absx(em, hi, tag + ".ahi")
    lo2 = em.commit(em.lin([(1, lo), (-1, mul_up(em, alo, c.u_acc, tag + ".ulo"))], bits=MBv + 5), tag + ".lo2")
    hi2 = em.commit(em.lin([(1, hi), (1, mul_up(em, ahi, c.u_acc, tag + ".uhi"))], bits=MBv + 5), tag + ".hi2")
    rc = minmax2(em, em.lin([(1, O), (-1, Qv(lo2))], bits=MBv + 6), em.lin([(1, Qv(hi2)), (-1, O)], bits=MBv + 6),
                 tag + ".rc", True)
    Rcl = minmax2(em, em.commit(Rdiv, tag + ".rdiv"), rc, tag + ".rcl", False)
    evq = em.view(ev, QS)
    evo = em.lin([(1, evq)], bits=11)
    Rst, _ = rv_store(em, O, Rcl, evq, tag + ".st", absY=aO)
    Rst = capped(em, Rst, MBv + 10, tag + ".Rst")
    Yo = obs_cmp(em, do_, O, Rst, evq, tag)
    # cap membership of the observed output (specs/06 §2.1 Cap_b), after the bf16 store
    eu = em.view(eta_units(em, evq, tag + ".ceta"), QS)
    alo2, _ = absx(em, lo2, tag + ".alo2")
    ahi2, _ = absx(em, hi2, tag + ".ahi2")
    cl = em.lin([(1, Qv(lo2)), (-1, Qv(shr_pub(em, alo2, 8, tag + ".cls", up_=True))), (-1, eu)], bits=MBv + 7)
    ch = em.lin([(1, Qv(hi2)), (1, Qv(shr_pub(em, ahi2, 8, tag + ".chs", up_=True))), (1, eu)], bits=MBv + 7)
    if em.pv:
        em.fail("cap:" + tag, ((Yo.v < cl.v) | (Yo.v > ch.v)).sum())
    em.rng(em.lin([(1, Yo), (-1, cl)], bits=62), 62, tag + ".capl")
    em.rng(em.lin([(1, ch), (-1, Yo)], bits=62), 62, tag + ".caph")
    # ---- LSE (natural log, fp32): lse_c = zmax_c + ln sum e^{-delta}
    Dn, kD, _, _, _ = normalize(em, Dd, HS, LN_BITS + 1, tag + ".dn", nonneg=True)
    lnt = _ln_table()
    em.table("cz.ln", lnt)
    LNv = em.col(HS, tag + ".ln", lnt[1][Dn.v - (1 << (LN_BITS - 1))] if em.pv else None, LN_OUT)
    em.lookup("cz.ln", HS, [Dn, LNv], tag + ".ln", _dense_check(1 << (LN_BITS - 1), 1 << (LN_BITS - 1)))
    L2lo = math.floor(math.log(2) * (1 << LN_OUT))
    kdm = em.lin([(1, kD)], -(30 - (LN_BITS - 1)), bits=8)                # ln(D 2^-30) = ln(Dn/2^17) + (kD - 13) ln 2
    L30 = em.lin([(1, shr_pub(em, LNv, LN_OUT - 30, tag + ".lns")),
                  (1, shr_pub(em, em.lin([(L2lo, kdm)], bits=48), LN_OUT - 30, tag + ".l2s"))], bits=42)
    # zmax on the 2^-30 grid: public shift s3_lo then hidden s1; Rng: |zmax| < 2^10
    Zs3 = shr_window(em, Zmax, s3_pub, hs, Wwin, tag + ".z30p") if hid_e else shr_pub(em, Zmax, s3_lo, tag + ".z30p")
    Z30 = shr_hid(em, Zs3, One_h, B1, 2 * EW, tag + ".z30", 41, f=f1)
    LSE = em.commit(em.lin([(1, Z30), (1, L30)], bits=50), tag + ".lse")
    DL_ = em.lin([(4, Tm)], 6 + (1 << 14), bits=16)                       # floors + ln step + weight approx
    aZ30, _ = absx(em, em.commit(Z30, tag + ".z30c"), tag + ".az30")
    aL, _ = absx(em, LSE, tag + ".alse")
    X30h = em.view(X30, HS)
    rl = em.lin([(1, shr_pub(em, X30h, 1, tag + ".rbar", up_=True)),
                 (1, mul_up(em, em.lin([(1, aZ30), (1, X30h)], 2, bits=51), c.u_acc, tag + ".t3")),
                 (1, mul_up(em, X30h, 2 * c.u_elem, tag + ".t4")),
                 (1, mul_up(em, em.lin([(1, aL), (1, em.view(DL_, HS)), (1, X30h)], 1 << 30, bits=51), c.u_acc, tag + ".t5")),
                 (1, em.view(DL_, HS))],
                up(cst["ea"] / (1 - cst["ea"]), 30) + up(2 * c.u_elem * (Fraction(math.ceil(math.log(Tp) * 1e6), 10 ** 6) + 1), 30) + 1,
                bits=40)
    Rl = capped(em, rl, 40, tag + ".Rl")
    obs_cmp(em, dl, LSE, Rl, em.const(HS, -30), tag + ".lse")
    return em
