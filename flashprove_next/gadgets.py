# SPDX-License-Identifier: Apache-2.0
"""Circuit-building layer on top of flashprove_next.zk_flat.

* ``Space``: a hypercube with named axes (MSB first).  A *padded* space has one extra top variable whose upper half
  holds uniform blinding values (T1 with b_blind = half the vector); constraints live on the lower half.
  A *quarter* space (sampled-block spaces of a profile with K1 <= 3/4 2^KBITS, KBITS >= 7) has no extra variable: its
  leading ``blk`` axis' top quarter (both top bits 1, blocks >= 3/4 2^KBITS, all unused by the K1 draws) is the
  aligned subcube of uniform blinding rows (>= 32 rows per inner coordinate), constraints live on the other 3/4.
  A non-padded space keeps the last ``TB`` rows as blinding rows.
* ``Col``: a committed (or public) vector on a space, with its integer values on the prover side.
* Gadgets: limb range checks, a power-of-two table, BFP rescaling, Poseidon2 permutation rows, sponge / Merkle
  wiring, hidden-index (wire) lookups.  Every gadget computes its witness (prover) and declares its constraints
  (both sides) in one code path, so prover and verifier build the identical circuit from public data only.
"""
import numpy as np
from . import zk_flat as Z
from .zk_flat import Ex, View, SelLt, IndexP, EqZC, TB
from .field import P, ONE
from . import poseidon2 as PS


def _obj(a):
    return np.asarray(a, dtype=object)


def fmod(a):
    """Integer array -> canonical field values (object dtype; vectorised for int64 input)."""
    a = np.asarray(a)
    if a.dtype.kind in "iu" and a.dtype.itemsize <= 8:
        a = a.ravel()
        if a.dtype.kind == "u":
            v = a.astype(np.uint64)
            v = np.where(v >= np.uint64(P), v - np.uint64(P), v)
        else:
            a = a.astype(np.int64)
            v = np.where(a >= 0, a.astype(np.uint64), np.uint64(P) - (-a).astype(np.uint64))
        return v.astype(object)
    return np.array([int(x) % P for x in a.ravel()], dtype=object)


def fsig64(a):
    """canonical field elements (uint64) -> their signed int64 representatives x (x if x < 2^63 else x - p), for
    lookup value lists (LogUp compresses int64 values as signed integers)."""
    w = np.asarray(a, dtype=np.uint64).view(np.int64)
    return w + np.where(w < 0, np.int64((1 << 32) - 1), np.int64(0))


class Ctx:
    def __init__(self, prove):
        self.cir = Z.Circuit(prove)
        self.prove = prove
        self.tables = {}
        self.notes = []          # what a forced (cheating) witness violated
        self.colreg = {}         # vec id -> Col
        self.force_copy = set()  # vec ids always served through fresh copies inside lookups (whatever their tail)
        self.cir.fresh_hook = self.fresh_ex
        self.quarter = None      # KBITS of the sampled-block axis when blk spaces use quarter-cube tails (see Space)

    def fresh_ex(self, ex):
        """Serve views of small committed columns inside a lookup through fresh copies (T1 budget)."""
        terms = []
        changed = False
        for c, f in ex.terms:
            nf = []
            for l, p in f:
                items = {}
                # canonical order (committed vector id, then key): copy numbering never depends on dict history
                order = sorted(l.items.values(), key=lambda t: (t[0].vec.id if isinstance(t[0], View) and isinstance(t[0].vec, Z.Vec) else -1, repr(t[0].key)))
                for a, co in order:
                    if isinstance(a, View) and isinstance(a.vec, Z.Vec) and (a.vec.tail < Col.COPY_BELOW or a.vec.id in self.force_copy) \
                            and a.vec.id in self.colreg:
                        col = self.colreg[a.vec.id]
                        cp = col.space.col(a.vec.name.split(".", 1)[-1] + ".lk%d" % len(self.cir.vecs), col.v if self.prove else None)
                        col.space.zero(cp.ex - col.ex, tag="copy")
                        a = View(cp.vec, a.varmap, a.fixed, a.shift)
                        changed = True
                    items[a.kid] = (a, co)
                nf.append((Z.Lin(items, l.const), p))
            terms.append((c, tuple(nf)))
        return Ex(terms) if changed else ex

    def violation(self, msg):
        if not Z.FORCE:
            raise Z.ProofError(msg)
        self.notes.append(msg)

    def space(self, name, axes, padded=True, content=None):
        axes = list(axes)
        if padded and self.quarter and axes and axes[0] == ("blk", self.quarter):
            return Space(self, name, axes, padded=False, quarter=True)
        return Space(self, name, axes, padded, content)


def quarter_ok(kb, k1):
    """Quarter-cube blinding for the sampled-block axis: the top quarter of 2^kb block rows is unused by the K1 draws
    and holds >= TB uniform rows per inner coordinate (a pure function of the public profile)."""
    return kb >= 7 and (1 << (kb - 2)) >= TB and k1 <= 3 * (1 << (kb - 2))


class Space:
    def __init__(self, ctx, name, axes, padded=True, content=None, quarter=False):
        self.ctx, self.name, self.axes, self.padded = ctx, name, list(axes), padded
        self.quarter = quarter
        bits = sum(b for _, b in self.axes)
        self.bits = bits
        self.nv = bits + (1 if padded else 0)
        self.nrows = 1 << bits
        if padded:
            self.content = self.nrows
            self.tail = self.nrows
        elif quarter:
            assert not padded and self.axes[0][0] == "blk" and self.axes[0][1] >= 2, name
            self.tail = self.nrows >> 2                  # blk rows >= 3/4 2^kb (top two blk bits 1)
            self.content = self.nrows - self.tail
        else:
            self.content = self.nrows - TB if content is None else content
            self.tail = None
            assert self.content <= (1 << self.nv) - TB, (name, self.content)
        self._vars = {}
        pos = 0
        for nm, b in reversed(self.axes):
            self._vars[nm] = list(range(pos, pos + b)); pos += b
        if padded:
            self._vars["__pad__"] = [bits]
        self.shape = tuple(1 << b for _, b in self.axes)
        self._cols = 0

    # ---------------------------------------------------------------- variables & structured helpers
    def vars(self, axis):
        return self._vars[axis]

    def bitsof(self, axis):
        return len(self._vars[axis])

    def sel(self):
        if self.padded:
            return Ex.atom(SelLt([], 1, fixed=((self.nv - 1, 0),)))
        return Ex.atom(SelLt(list(range(self.nv)), self.content))

    def idx(self, axis, scale=1):
        return Ex.atom(IndexP(self.vars(axis), scale))

    def lt(self, axis, n):
        """[axis index < n]."""
        return Ex.atom(SelLt(self.vars(axis), n))

    def rng(self, axis, lo, hi):
        """[lo <= axis index < hi]."""
        return self.lt(axis, hi) - self.lt(axis, lo)

    def ind(self, **fixed):
        """indicator of axis == value for each given axis."""
        fx = []
        for ax, val in fixed.items():
            for k, v in enumerate(self.vars(ax)):
                fx.append((v, (val >> k) & 1))
        return Ex.atom(SelLt([], 1, fixed=tuple(fx)))

    def grid(self, axis):
        """prover helper: axis index of every content row (flat, length nrows)."""
        n = self.nrows if self.padded else self.content
        r = np.arange(n, dtype=np.int64)
        lo = self._vars[axis][0]; b = len(self._vars[axis])
        return (r >> lo) & ((1 << b) - 1)

    @property
    def n(self):
        return self.nrows if self.padded else self.content

    def eqzc(self, axes):
        vs = []
        for a in axes:
            vs += self.vars(a)
        return Ex.atom(EqZC(vs))

    # ---------------------------------------------------------------- columns
    def col(self, name, vals=None, phase=1):
        cir = self.ctx.cir
        nm = f"{self.name}.{name}"
        if self.padded or self.quarter:
            v = cir.vec(nm, self.nv, self.content, phase=phase, tail=self.tail)
        else:
            v = cir.vec(nm, self.nv, self.content, phase=phase)
        c = Col(self, v, None)
        if self.ctx.prove:
            c.setv(vals)
        self.ctx.colreg[v.id] = c
        return c

    def pub(self, name, vals):
        nm = f"{self.name}.{name}"
        full = np.zeros(1 << self.nv, dtype=np.uint64)
        a = np.asarray(fmod(vals)).astype(np.uint64)
        full[:len(a)] = a
        pv = self.ctx.cir.pubvec(nm, full, self.nv)
        c = Col(self, pv, None)
        c.v = _obj(np.asarray(vals))
        return c

    def zero(self, ex, sel=None, tag=""):
        s = self.sel() if sel is None else self.sel() * sel
        self.ctx.cir.zero(self.nv, ex, sel=s, tag=tag or self.name)

    def sumzero(self, ex, eq_axes=(), sel=None, tag=""):
        """For every assignment of eq_axes: sum over the remaining content rows of ex = 0."""
        s = self.sel() if sel is None else self.sel() * sel
        e = ex * s
        if eq_axes:
            e = e * self.eqzc(eq_axes)
        self.ctx.cir.sumzero(self.nv, e, eq_vars=(), tag=tag or self.name)


class Col:
    def __init__(self, space, vec, v):
        self.space, self.vec, self._v = space, vec, v

    @property
    def v(self):
        if self._v is not None:
            return self._v
        vv = getattr(self.vec, "vals", None)
        if vv is not None and not getattr(self.vec, "is_ext", False):
            a = vv if vv.dtype in (np.uint64, object) else vv.astype(np.int64)
            n = self.space.n
            return a[:n] if len(a) >= n else np.concatenate([a, np.zeros(n - len(a), dtype=a.dtype)])
        grp = getattr(self.vec, "grp", None)
        if grp is not None and getattr(grp, "built", False) and not getattr(self.vec, "is_ext", False):
            u = grp.vec_range_e(self.vec, 0, self.space.n)[0]
            return np.where(u >= np.uint64(1 << 63), (u.astype(object) - P).astype(np.int64), u.astype(np.int64))
        return vv

    @v.setter
    def v(self, val):
        self._v = val

    def setv(self, vals):
        sp = self.space
        if vals is None:
            vals = np.zeros(sp.n, dtype=np.int64)
        a = np.asarray(vals)
        if a.dtype == np.uint64:
            pass                              # field elements, stored as is (no Python-int object arrays)
        elif a.dtype.kind == "f":
            a = a.astype(np.int64)
        elif a.dtype != object:
            a = a.astype(np.int64)
        a = a.ravel()
        assert len(a) == sp.n, (self.vec.name, len(a), sp.n)
        self._v = None                        # single compact copy lives in self.vec.vals (no 8 B/cell duplicate)
        self.space.ctx.cir.set(self.vec, a)

    def atom(self):
        a = getattr(self, "_atom", None)
        if a is None:
            a = self._atom = View(self.vec)
        return a

    def vec_kid(self):
        return self.atom().kid

    @property
    def ex(self):
        return Ex.atom(View(self.vec))

    def shifted(self, delta):
        return Ex.atom(View(self.vec, shift=delta))

    def _bits(self, sp2, amap):
        """per own non-pad variable: (target axis, bit) or ('c', b)."""
        amap = dict(amap or {})
        own = self.space
        spec = {}
        for ax, b in own.axes:
            tgt = amap.get(ax, ax)
            if isinstance(tgt, list):
                lst = tgt
            elif isinstance(tgt, tuple) and tgt[0] == "c":
                lst = [("c", (tgt[1] >> k) & 1) for k in range(b)]
            else:
                name, lo = (tgt, 0) if isinstance(tgt, str) else tgt
                lst = [(name, lo + k) for k in range(b)]
            assert len(lst) == b, (self.vec.name, ax, lst)
            for k, v in enumerate(own.vars(ax)):
                spec[v] = lst[k]
        return spec

    COPY_BELOW = 64

    def _tail(self):
        sp = self.space
        return sp.tail if sp.tail is not None else TB

    def on(self, sp2, amap=None, shift=0, pad_to=None):
        """View on another space. amap: own axis -> target axis | (target axis, lo bit) | ('c', const) |
        list of per-bit (target axis, bit) / ('c', bit).
        A small committed column (fewer than COPY_BELOW blinding rows) viewed from a different cube size is
        served through a fresh committed copy (equality enforced on its own cube) so that every column keeps
        at most one revealed evaluation per blinding row (T1 budget).
        pad_to = (axis, bit): a padded committed column viewed on a quarter space reads its blinding half where that
        target bit is 1 (the caller restricts the constraint to rows with that bit 0)."""
        own = self.space
        spec = self._bits(sp2, amap)
        if isinstance(self.vec, Z.Vec) and self._tail() < self.COPY_BELOW and sp2.nv != own.nv:
            if not hasattr(self, "_copies"):
                self._copies = {}
            key = (sp2.nv, repr(sorted(spec.items())), shift)
            if key not in self._copies:
                c = own.col(self.vec.name.split(".", 1)[-1] + ".cp%d" % len(self._copies), self.v if own.ctx.prove else None)
                own.zero(c.ex - self.ex, tag="copy")
                self._copies[key] = c
            return self._copies[key]._on(sp2, spec, shift, pad_to)
        return self._on(sp2, spec, shift, pad_to)

    def _on(self, sp2, spec, shift, pad_to=None):
        own = self.space
        varmap = [None] * own.nv
        for v, t in spec.items():
            varmap[v] = t if t[0] == "c" else sp2.vars(t[0])[t[1]]
        public = isinstance(self.vec, Z.PubVec)
        if own.padded:
            pv_ = own.vars("__pad__")[0]
            if sp2.padded:
                varmap[pv_] = sp2.vars("__pad__")[0]
            elif public:
                varmap[pv_] = ("c", 0)
            elif pad_to is not None:
                varmap[pv_] = sp2.vars(pad_to[0])[pad_to[1]]
            else:
                raise AssertionError(f"padded column {self.vec.name} viewed on non-padded space {sp2.name} without pad_to")
        elif own.quarter and not public:
            # T1 / value safety: the source's tail subcube (top two blk bits) must land on the target's blinding rows and
            # its content rows on the target's content rows: blk -> blk (quarter target) or (pad, const 1) (padded target)
            kb = own.axes[0][1]
            top, sec = own.vars("blk")[kb - 1], own.vars("blk")[kb - 2]
            if sp2.quarter:
                ok = varmap[top] == sp2.vars("blk")[sp2.axes[0][1] - 1] and varmap[sec] == sp2.vars("blk")[sp2.axes[0][1] - 2]
            else:
                ok = sp2.padded and varmap[top] == sp2.vars("__pad__")[0] and varmap[sec] == ("c", 1)
            if not ok:
                raise AssertionError(f"quarter column {self.vec.name} viewed on {sp2.name} with an unsafe variable map")
        return Ex.atom(View(self.vec, varmap, shift=shift))

    def val_on(self, sp2, amap=None, shift=0):
        """Prover: values of .on(sp2, amap) on sp2's content rows."""
        spec = self._bits(sp2, amap)
        y = np.zeros(sp2.n, dtype=np.int64)
        grids = {}
        for v, t in spec.items():
            if t[0] == "c":
                if t[1]:
                    y |= np.int64(1) << v
            else:
                if t[0] not in grids:
                    grids[t[0]] = sp2.grid(t[0])
                y |= ((grids[t[0]] >> t[1]) & 1) << v
        y = y + shift
        ok = (y >= 0) & (y < len(self.v))
        out = np.zeros(sp2.n, dtype=self.v.dtype)
        out[ok] = self.v[y[ok]]
        return out


# ============================================================================ fixed tables
RANGE_W = 16


def _table_space(ctx, name, nrows):
    b = max(1, int(nrows - 1).bit_length())
    return ctx.space("T:" + name, [("row", b)])


def table(ctx, name, cols):
    """Register a public lookup table (cols: list of int arrays of equal length)."""
    if name in ctx.tables:
        return ctx.tables[name]
    n = len(cols[0])
    sp = _table_space(ctx, name, n)
    def padded(c):
        c = np.asarray(c)
        if c.dtype.kind in "iu":
            return np.concatenate([c.astype(np.int64), np.full(sp.n - n, int(c[0]), dtype=np.int64)])
        return list(c) + [c[0]] * (sp.n - n)
    pcs = [sp.pub(f"c{i}", padded(c)) for i, c in enumerate(cols)]
    ctx.cir.lookup_table(name, sp.nv, [p.ex for p in pcs], n, vals=[np.asarray(c) for c in cols] if ctx.prove else None)
    ctx.tables[name] = (sp, pcs)
    return ctx.tables[name]


def range_table(ctx):
    ws, vs = [], []
    for w in range(1, RANGE_W + 1):
        ws.append(np.full(1 << w, w, dtype=np.int64)); vs.append(np.arange(1 << w, dtype=np.int64))
    return table(ctx, "rng", [np.concatenate(ws), np.concatenate(vs)])


CODE16_MAX = 65534      # max valid 16-bit tile code (0xFF7F = 65407 for finite bf16; 65534 * sum 2^(16s) < 2^64 - 2^48 < P)


def code16_table(ctx):
    """[0 .. 65534]: makes 4x16-bit code packing sum_{s=0..3} c_s 2^(16s) strictly < P and injective on every slot."""
    return table(ctx, "u16c", [np.arange(CODE16_MAX + 1, dtype=np.int64)])


POW_LO, POW_HI = -62, 62


def pow2_table(ctx):
    """rows (k, 2^max(k,0), 2^max(-k,0), 2^(62-max(-k,0)), [k > 0], [k >= 38])."""
    ks = np.arange(POW_LO, POW_HI + 1)
    a = [1 << max(int(k), 0) for k in ks]; b = [1 << max(-int(k), 0) for k in ks]
    c = [1 << (62 - max(-int(k), 0)) for k in ks]
    nk = [1 if int(k) > 0 else 0 for k in ks]
    bk = [1 if int(k) >= 38 else 0 for k in ks]
    return table(ctx, "pow2", [ks, _obj(a), _obj(b), _obj(c), nk, bk])


# ============================================================================ range checks
def range_check(sp, ex, vals, bits, name, sel=None, sel_vals=None):
    """Prove 0 <= value < 2^bits on every (selected) content row of sp, for value = ex (affine Ex).
    Commits ceil(bits/16) limb columns."""
    ctx = sp.ctx
    range_table(ctx)
    nl = (bits + RANGE_W - 1) // RANGE_W
    widths = [RANGE_W] * (nl - 1) + [bits - RANGE_W * (nl - 1)]
    limbs = []
    if ctx.prove:
        from .zk_flat import as_i64
        v64 = as_i64(vals, 63) if bits <= 62 else None
        if v64 is not None:                   # vectorised int64 path (|v| < 2^63, bits <= 62)
            v64 = v64.ravel()
            okr = (v64 >= 0) & (v64 < (1 << bits))
            sv_ = np.ones(len(v64), dtype=bool) if sel_vals is None else (np.asarray(sel_vals).astype(np.int64) != 0)
            badm = sv_ & ~okr
            if badm.any():
                i0 = int(np.flatnonzero(badm)[0])
                ctx.violation(f"range {name}: {int(badm.sum())} values outside [0, 2^{bits}) e.g. row {i0} = {int(v64[i0])}")
            vv = np.where(v64 >= 0, v64 & ((1 << bits) - 1), 0)
        else:
            v = np.asarray(vals, dtype=object)
            bad = [i for i in range(len(v)) if (sel_vals is None or sel_vals[i]) and not (0 <= int(v[i]) < (1 << bits))]
            if bad:
                ctx.violation(f"range {name}: {len(bad)} values outside [0, 2^{bits}) e.g. row {bad[0]} = {v[bad[0]]}")
            vv = np.array([int(x) % (1 << bits) if 0 <= int(x) else 0 for x in v], dtype=object)
    acc = Ex.const(0)
    for k, w in enumerate(widths):
        lv = None
        if ctx.prove:
            if vv.dtype != object:
                lv = ((vv >> (RANGE_W * k)) & ((1 << w) - 1)).astype(np.int64)
            else:
                lv = np.array([(int(x) >> (RANGE_W * k)) & ((1 << w) - 1) for x in vv], dtype=np.int64)
        c = sp.col(f"{name}.l{k}", lv)
        limbs.append(c)
        acc = acc + c.ex * Ex.const(1 << (RANGE_W * k))
        lsel = sp.sel() if sel is None else sp.sel() * sel
        wv = np.full(sp.n, w, dtype=np.int64) if ctx.prove else None
        sv = None
        if ctx.prove:
            sv = np.ones(sp.n, dtype=np.int64) if sel_vals is None else np.asarray(sel_vals, dtype=np.int64)
        ctx.cir.lookup("rng", sp.nv, [Ex.const(w), c.ex], sel=lsel, vals=[wv, lv] if ctx.prove else None, sel_vals=sv)
    sp.zero(ex - acc, sel=sel, tag=f"range:{name}")
    return limbs


def nonneg(sp, ex, vals, name, bits=62, sel=None, sel_vals=None):
    return range_check(sp, ex, vals, bits, name, sel, sel_vals)


def signed_range(sp, ex, vals, bits, name, sel=None, sel_vals=None):
    """|value| < 2^(bits-1)  i.e. value + 2^(bits-1) in [0, 2^bits)."""
    off = 1 << (bits - 1)
    v2 = None
    if vals is not None:
        from .zk_flat import as_i64
        a64 = as_i64(vals, 61)
        v2 = (a64 + off) if (a64 is not None and bits <= 61) else _obj(vals) + off
    return range_check(sp, ex + Ex.const(off), v2, bits, name, sel, sel_vals)


def boolean(sp, c, sel=None):
    sp.zero(c.ex * (c.ex - Ex.const(1)), sel=sel, tag="bool")


# ============================================================================ wire (hidden-index) lookups
def wire_table(ctx, name, sp, key_exs, cols, key_vals=None):
    """Declare table `name` whose rows are (keys..., col values...) over the content rows of sp."""
    vecs = key_exs + [c.ex for c in cols]
    vals = None
    if ctx.prove:
        vals = list(key_vals) + [c.v for c in cols]
    lk = ctx.cir.lookups.setdefault(name, Z.Lookup(name))
    lk.table = (sp.nv, [e for e in vecs])
    lk.table_vals = vals
    lk.mult = ctx.cir.vec(f"mult:{name}", sp.nv, sp.n, tail=sp.tail)
    # rows of the table outside the content (blinding half) must not count: the mult selector handles it
    return lk


def wire_lookup(ctx, name, sp, exs, vals, sel=None, sel_vals=None):
    s = sp.sel() if sel is None else sp.sel() * sel
    sv = None
    if ctx.prove:
        sv = np.ones(sp.n, dtype=np.int64) if sel_vals is None else np.asarray(sel_vals, dtype=np.int64)
        vals = [np.asarray(v) for v in vals]
    ctx.cir.lookup(name, sp.nv, exs, sel=s, vals=vals, sel_vals=sv)


# ============================================================================ Poseidon2 permutation rows
POS_COLS = ["I%d" % i for i in range(12)] + ["S%d_%d" % (k, i) for k in (1, 2, 3, 4) for i in range(12)] + \
    ["x%d" % j for j in range(22)] + ["S%d_%d" % (k, i) for k in (26, 27, 28, 29, 30) for i in range(12)]
_XS, _S26 = PS.partial_round_linear_forms()


class PoseRegion:
    """One Poseidon2 permutation per content row of a non-padded space.  inputs: (12, n) ints (prover)."""

    def __init__(self, sp, inputs=None, B=None):
        self.sp = sp
        ctx = sp.ctx
        vals = {}
        if ctx.prove:
            from .field import CPU
            Bk = B or CPU
            st = Bk.asarray(np.asarray(inputs, dtype=np.uint64))
            out, cols = PS.permute(st, Bk, trace=True)
            cols = {k: Bk.tonumpy(v) for k, v in cols.items()}
            for i in range(12):
                vals["I%d" % i] = cols["I"][i]
            for k in (1, 2, 3, 4, 26, 27, 28, 29, 30):
                for i in range(12):
                    vals["S%d_%d" % (k, i)] = cols["S%d" % k][i]
            for j in range(22):
                vals["x%d" % j] = cols["x"][j]
        self.c = {}
        for nm in POS_COLS:
            v = None
            if ctx.prove:
                v = np.ascontiguousarray(vals[nm], dtype=np.uint64)      # field elements (uint64, no objects)
            self.c[nm] = sp.col("p" + nm, v)
        self._constrain()

    def I(self, i): return self.c["I%d" % i]
    def O(self, i): return self.c["S30_%d" % i]

    def _constrain(self):
        """Round constraints, built directly as term lists (same polynomials as the operator form, ~10x faster)."""
        sp = self.sp; c = self.c
        cons = []
        L = {nm: Z.Lin({col.vec_kid(): (col.atom(), ONE)}, _own=True) for nm, col in c.items()}

        def lin_of(items, const):
            it = {}
            for nm, co in items:
                if co % P == 0:
                    continue
                a = c[nm].atom(); k = a.kid
                e = it.get(k)
                it[k] = (a, (co % P, 0, 0)) if e is None else (a, ((e[1][0] + co) % P, 0, 0))
            return Z.Lin(it, (const % P, 0, 0), _own=True)

        def full_round(src, dst, k, src_is_input=False):
            # dst = ME * sbox(pre + RC[k]),  pre = ME * src if src is the raw input else src
            if src_is_input:
                ins = [lin_of([("I%d" % j, PS.ME[i][j]) for j in range(12)], PS.RC[k][i]) for i in range(12)]
            else:
                ins = [lin_of([(src + "_%d" % i, 1)], PS.RC[k][i]) for i in range(12)]
            for o in range(12):
                terms = [((P - 1, 0, 0), ((L[dst + "_%d" % o], 1),))]
                terms += [((PS.ME[o][i] % P, 0, 0), ((ins[i], 7),)) for i in range(12)]
                cons.append(Ex(terms))
        full_round(None, "S1", 0, True)
        full_round("S1", "S2", 1); full_round("S2", "S3", 2); full_round("S3", "S4", 3)

        def form_minus(target, f):
            # target - form(f): linear part (target - S4 combination - const) and -coef * x_j^7 terms
            items = [(target, 1)] + [("S4_%d" % key[1], -co) for key, co in f.items() if key != 1 and key[0] == "S"]
            terms = [(ONE, ((lin_of(items, -f.get(1, 0)), 1),))]
            for key, co in f.items():
                if key != 1 and key[0] != "S":
                    terms.append((((-co) % P, 0, 0), ((L["x%d" % key[1]], 7),)))
            return Ex(terms)
        for j in range(22):
            cons.append(form_minus("x%d" % j, _XS[j]))
        for i in range(12):
            cons.append(form_minus("S26_%d" % i, _S26[i]))
        full_round("S26", "S27", 26); full_round("S27", "S28", 27); full_round("S28", "S29", 28); full_round("S29", "S30", 29)
        for e in cons:
            sp.zero(e, tag="poseidon2")


# ============================================================================ helpers
def floor_div(a, b):
    return a // b


def rescale_vals(acc, k):
    """floor(acc * 2^-k) with remainder for k >= 0, exact multiply for k < 0."""
    acc = _obj(acc); k = _obj(k)
    m = np.empty(len(acc), dtype=object); r = np.empty(len(acc), dtype=object)
    for i in range(len(acc)):
        a = int(acc[i]); kk = int(k[i])
        if kk >= 0:
            m[i] = a >> kk; r[i] = a - (m[i] << kk)
        else:
            m[i] = a << (-kk); r[i] = 0
    return m, r
