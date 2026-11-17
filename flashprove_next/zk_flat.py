# SPDX-License-Identifier: Apache-2.0
"""Flat (single-layer, recursion-free) zero-knowledge argument over Goldilocks^3 (E = F_p[u]/(u^3 - 2)).

One proof = two committed phases + one Libra-masked sumcheck per cube size + two ZK-Basefold openings.

  * Committed vectors (``Vec``) are base-field witness columns with a tail of ``TB`` uniform E rows  [T1].
    All vectors of a phase are packed (aligned) into one multilinear polynomial ``Big``; one Merkle tree with
    256-bit salted leaves per phase.
  * Constraints are polynomial identities over hypercubes built from *atoms*: views of committed vectors (with
    variable maps, Boolean embeddings and constant shifts), views of public vectors, and structured public
    polynomials (eq, ranges, indicators, indices).  Two kinds:
        zerocheck:  sum_x eq(r, x) sel(x) C(x) = 0          (C vanishes on the selected rows)
        zero-sum:   sum_x eq(r_S, x_S) C(x) = 0              (linear / inner-product / LogUp relations)
    All constraints of one cube size are batched with random coefficients into ONE sumcheck that is masked by a
    committed Libra polynomial g(x) = sum_i g_i(x_i)                                                  [T2].
    Inner (matrix-vector) relations are zero-sum constraints inside the same masked sumcheck, and every value
    the verifier sees is an MLE evaluation of a T1-blinded vector at a random point                    [T3].
  * Lookups use a merged-domain zero-sum LogUp: value parts and the table part live on one hypercube inside a
    single committed helper H; only "sum H = 0" is ever proven, no partial harmonic sum is revealed     [F2].
  * Each phase is opened once by a generalized Basefold proving <Big, W> = c for a verifier-computable
    weight table W (any batch of point / shifted-point / power claims), with a uniform full-length mask
    h_bf committed in the same tree: F' = Big + eps h_bf masks every folded codeword and the constant [T4].

Nothing here depends on the model: the circuit (flashprove_next.gadgets and the modules built on it) is built from
public bucket constants only, identically by prover and verifier.
"""
import os
import numpy as np
from .field import (P, EDIM, ZERO, ONE, CPU, hadd, hsub, hmul, hneg, hinv, hpow, eq1, eq_pt, lagrange_eval, INV2,
                    urandom_u64, rand_field, seeded_rng)
from .transcript import Transcript, canon

TB = 32              # T1: uniform E rows at the tail of every committed vector
FILLER_BITS = 12     # one 4096-entry uniform vector per phase (masks codeword queries)
RATE = 4             # Reed-Solomon rate 1/4
ARITY_BITS = 3
T_QUERIES = 168      # Basefold query count (0.645^168 <= 2^-106; ProtocolBridge.basefold_qry_esc_le)
SALT_WORDS = 4       # 256-bit leaf salt
GROUP_BITS = 25      # protocol constant: max 2^25 cells per commitment group (one Merkle root each)
SHARD_BITS = 24      # protocol constant: a fixed-table LogUp helper H never exceeds 2^24 cells (extra parts -> table copy)
LEAN_BITS = int(os.environ.get("FPN_LEAN_BITS", "28"))     # prover-only memory knob: groups >= 2^LEAN_BITS always spill h_bf / salts


class ProofError(Exception):
    pass


_T0 = [None]
_WIN = [0]           # CuPy pool peak since the previous _rss checkpoint (filled by PoolPeak)


def _trim():
    """Return freed heap memory to the OS (glibc keeps freed blocks below its dynamic mmap threshold, up to 32 MiB,
    in the heap; after releasing GBs of witness columns this is the difference between RSS and live data)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _rss(tag):
    """FPN_LOG=1: print the elapsed time and resident set size at a prover checkpoint (Linux /proc)."""
    if not os.environ.get("FPN_LOG"):
        return
    import time
    if _T0[0] is None:
        _T0[0] = time.time()
    try:
        with open("/proc/self/status") as f:
            kb = [int(l.split()[1]) for l in f if l.startswith("VmRSS")][0]
        gpu = ""
        try:
            import sys
            if "cupy" in sys.modules:
                gpu = (f"  pool {sys.modules['cupy'].get_default_memory_pool().total_bytes() / (1 << 30):.2f} GiB"
                       f" (window peak {_WIN[0] / (1 << 30):.2f})")
                _WIN[0] = 0
        except Exception:  # noqa: BLE001
            pass
        print(f"[mem] {time.time() - _T0[0]:7.1f}s {tag}: RSS {kb / 1048576:.2f} GB{gpu}", flush=True)
    except (OSError, IndexError):
        pass



MEM_REPORT = []      # (tag, breakdown dict) filled by _mem_report when FPN_LOG is set


def _mem_report(cir, pcs, tag):
    """FPN_LOG=1: itemised host memory of the prover state (bytes): compact witness, E-valued vectors, lookup row
    matrices, group random material in RAM, and what is spilled to disk."""
    if not os.environ.get("FPN_LOG"):
        return
    cells = sum(v.N for v in cir.vecs)
    wit_b = ext_ram = ext_disk = 0
    for v in cir.vecs:
        x = v.vals
        if isinstance(x, _Spilled):
            ext_disk += x.nbytes
        elif isinstance(x, np.ndarray):
            if v.is_ext:
                ext_ram += x.nbytes
            else:
                wit_b += x.nbytes
    lk_b = 0
    for lk in cir.lookups.values():
        for part in lk.parts:
            V = part[3]
            if isinstance(V, np.ndarray):
                lk_b += V.nbytes
            elif isinstance(V, (list, tuple)):
                lk_b += sum(getattr(c, "nbytes", 0) for c in V)
    grp_ram = sum(pc.ram_bytes() for pc in pcs if pc.built)
    grp_disk = sum(pc.disk_bytes() for pc in pcs if pc.built)
    d = {"cells": cells, "witness_compact": wit_b, "ext_vectors_ram": ext_ram, "lookup_rows": lk_b,
         "group_random_ram": grp_ram, "spilled_disk": grp_disk + ext_disk}
    MEM_REPORT.append((tag, d))
    per = {k: round(v / cells, 2) for k, v in d.items() if k != "cells"}
    print(f"[mem-breakdown] {tag}: cells {cells / 1e6:.1f} M; bytes/cell {per}", flush=True)


class PoolPeak:
    """Measure the peak of the CuPy default memory pool (bytes held by the pool, used + cached) during a block by
    sampling it every `dt` seconds on a background thread (prover diagnostics only; no-op without CuPy)."""

    def __init__(self, dt=0.002):
        self.dt, self.peak, self.peak_used, self._run, self._th = dt, 0, 0, False, None

    def __enter__(self):
        try:
            import cupy as cp
            pool = cp.get_default_memory_pool()
        except ImportError:
            return self
        import threading
        self._run = True

        def loop():
            while self._run:
                t, u = pool.total_bytes(), pool.used_bytes()
                self.peak = max(self.peak, t); self.peak_used = max(self.peak_used, u)
                _WIN[0] = max(_WIN[0], t)
                time.sleep(self.dt)
        import time
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._run = False
        if self._th is not None:
            self._th.join()


FORCE = False        # adversarial test mode: keep going when the witness violates a constraint
VIOLATIONS = []      # what a forced (cheating) prover had to fake


def _to_e(x):
    if isinstance(x, tuple):
        if len(x) != 3:
            raise ValueError(f"expected 3-tuple for E, got {x!r}")
        return (x[0] % P, x[1] % P, x[2] % P)
    return (int(x) % P, 0, 0)


# ============================================================================ compact witness values
_I8, _I16, _I32 = (np.iinfo(t) for t in (np.int8, np.int16, np.int32))


def as_i64(a, bound_bits=62):
    """int64 view of an integer array if every |x| < 2^bound_bits, else None."""
    a = np.asarray(a)
    if a.dtype.kind in "iu" and a.dtype.itemsize <= 8 and not (a.dtype == np.uint64 and len(a) and int(a.max()) >= (1 << 63)):
        v = a.astype(np.int64)
    else:
        try:
            v = np.array(a.ravel(), dtype=np.int64).reshape(a.shape)
        except (OverflowError, TypeError, ValueError):
            return None
    if v.size and int(np.abs(v).max()) >= (1 << bound_bits):
        return None
    return v


def compact_vals(a):
    """Integer witness values -> an owned array in the smallest dtype that holds them exactly: signed int8/16/32/64
    (integers, field value x mod p) or uint64 (canonical field elements, e.g. Poseidon2 trace columns).  Always a
    fresh copy (the builder's arrays may be views it later reuses)."""
    a = np.asarray(a).ravel()
    if a.dtype == object:
        a64 = as_i64(a, 63)
        a = a64 if a64 is not None else CPU.fromi(a)
    if a.dtype == np.bool_:
        return a.astype(np.int8)
    if a.dtype == np.uint64:
        if len(a) and a.max() >= np.uint64(P):
            a = np.where(a >= np.uint64(P), a - np.uint64(P), a)
        if len(a) and a.max() <= np.uint64(_I32.max):
            return a.astype(np.int32)
        return np.array(a, dtype=np.uint64, copy=True)
    if a.dtype.kind == "u":
        a = a.astype(np.int64)
    elif a.dtype.kind != "i":
        a = a.astype(np.int64)
    if len(a) == 0:
        return np.zeros(0, dtype=np.int8)
    lo, hi = int(a.min()), int(a.max())
    for t in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        ii = np.iinfo(t)
        if ii.min <= lo and hi <= ii.max:
            return a.astype(t)
    return np.array(a, dtype=np.int64, copy=True)


def _compact_part(vals, sel_vals):
    """lookup part values -> (list of compact columns, compact selector or None); callables stay lazy."""
    if vals is None or callable(vals):
        return vals, sel_vals
    if isinstance(vals, np.ndarray) and vals.ndim == 2:
        cols = [compact_vals(vals[:, j]) for j in range(vals.shape[1])]
    else:
        cols = [compact_vals(c) for c in vals]
    if sel_vals is not None:
        sv = compact_vals(sel_vals)
        sel_vals = None if (len(sv) == len(cols[0]) and len(sv) and sv.min() == 1 and sv.max() == 1) else sv
    return cols, sel_vals


def field_u64(a):
    """compact values (any integer dtype / uint64 field elements) -> canonical uint64 field elements (host)."""
    a = np.asarray(a)
    if a.dtype == np.uint64:
        return a
    return CPU.fromi(a.astype(np.int64, copy=False))


def wit(v):
    """content values of a base vector (compact), regenerated by ``v.gen`` when the prover keeps none."""
    if v.vals is not None:
        return v.vals
    if v.gen is not None:
        return compact_vals(v.gen())
    return None


_EXT_CACHE = [None, None, None, None]   # 1-vector host LRU: [vec_id, a0, a1, (3, a1-a0) host uint64]


def _ext_rows(v, a0, a1):
    """rows [a0, a1) of an E-valued vector's (3, N) values (in RAM, spilled to disk, or regenerated on demand)."""
    x = v.vals
    if isinstance(x, _Spilled):
        return x.read(a0, a1)
    if x is None and v.ext_gen is not None:
        if _EXT_CACHE[0] == v.id and _EXT_CACHE[1] == a0 and _EXT_CACHE[2] == a1:
            return _EXT_CACHE[3]
        arr = v.ext_gen(a0, a1, False)
        _EXT_CACHE[:] = [v.id, a0, a1, arr]
        return arr
    if x is None and v.gen is not None:
        return np.asarray(v.gen())[:, a0:a1]
    return x[:, a0:a1]


# ============================================================================ vectors
class Vec:
    """A committed vector of length 2^nbits whose last TB entries are uniform blinding rows."""
    __slots__ = ("id", "name", "nbits", "content", "phase", "vals", "is_ext", "full", "tail", "grp", "gen",
                 "ext_gen", "on_release")

    def __init__(self, vid, name, nbits, content, phase, tail=None):
        tail = TB if tail is None else tail
        if content > (1 << nbits) - tail:
            raise ValueError(f"vector {name}: content {content} leaves no room for {tail} blinding rows in 2^{nbits}")
        self.id, self.name, self.nbits, self.content, self.phase = vid, name, nbits, content, phase
        self.vals = None; self.is_ext = False; self.full = None; self.tail = tail; self.grp = None; self.gen = None
        self.ext_gen = None; self.on_release = None

    @property
    def N(self): return 1 << self.nbits


class PubVec:
    """A public vector (both parties hold its values as field elements)."""
    __slots__ = ("name", "nbits", "vals", "_mle")

    def __init__(self, name, vals, nbits=None):
        v = np.asarray(vals, dtype=np.uint64)
        nb = max(0, int(len(v) - 1).bit_length()) if nbits is None else nbits
        full = np.zeros(1 << nb, dtype=np.uint64); full[:len(v)] = v
        self.name, self.nbits, self.vals, self._mle = name, nb, full, {}

    def mle(self, z):
        key = tuple(z)
        if key not in self._mle:
            e = CPU.eq_table(list(z))
            self._mle[key] = hadd(_dot_base(self.vals, e), ZERO)
        return self._mle[key]


def _dot_base(vals, e):
    """sum_x vals[x] * e[x] with vals base (uint64), e (3, n) on CPU."""
    from .field import native, ptr
    lib = native()
    vals = np.asarray(vals, dtype=np.uint64); e = np.asarray(e, dtype=np.uint64)
    if lib is not None and vals.ndim == 1 and e.ndim == 2 and e.shape[0] == 3 and len(vals) == e.shape[1]:
        v = np.ascontiguousarray(vals, dtype=np.uint64); w = np.ascontiguousarray(e, dtype=np.uint64)
        out = np.zeros(3, dtype=np.uint64)
        lib.base_ext_dot(ptr(v), ptr(w), len(v), ptr(out))
        return (int(out[0]), int(out[1]), int(out[2]))
    a = CPU.mul(e[0], vals); b = CPU.mul(e[1], vals); c = CPU.mul(e[2], vals)
    return (CPU.bsum(a), CPU.bsum(b), CPU.bsum(c))


# ============================================================================ atoms
_IMAP = {}      # (backend, varmap, fixed, block) -> (local index map, indicator); cleared after each proof
PUB_DIRECT_BITS = 20   # structured public atoms with larger local tables are evaluated directly on their block


class Atom:
    """Value on a cube point x: ind(x_fixed) * local(map(x)).  varmap[j] = cube variable feeding local var j."""
    kind = "?"

    def __init__(self, varmap, fixed):
        self.varmap = tuple(varmap); self.fixed = tuple(sorted(fixed))

    def ind_mle(self, r):
        out = ONE
        for v, b in self.fixed:
            out = hmul(out, r[v] if b else hsub(ONE, r[v]))
        return out

    def local_point(self, r):
        return [((v[1], 0, 0) if isinstance(v, tuple) else r[v]) for v in self.varmap]

    def block(self, nv):
        """(off, m): the atom vanishes outside the aligned block [off, off + 2^m) (top fixed vars)."""
        fx = dict(self.fixed)
        m = nv
        while m > 0 and (m - 1) in fx:
            m -= 1
        off = sum(fx[v] << v for v in range(m, nv))
        return off, m

    def index_map(self, nv, xp, blk=None):
        off, m = blk if blk is not None else (0, nv)
        if m > 16:                       # large blocks: compute directly on the device, never cached
            return self._index_map(nv, xp, off, m)
        ck = (xp.__name__, self.varmap, self.fixed, off, m)
        hit = _IMAP.get(ck)
        if hit is not None:
            return hit[0], hit[1].copy()
        y, ok = self._index_map(nv, np, off, m)
        if xp is not np:
            y, ok = xp.asarray(y), xp.asarray(ok)
        if len(_IMAP) < 8192:
            _IMAP[ck] = (y, ok)
        return y, ok.copy()

    def _index_map(self, nv, xp, off, m):
        x = xp.arange(1 << m, dtype=np.int64) + off
        y = xp.zeros(1 << m, dtype=np.int64)
        for j, v in enumerate(self.varmap):
            if isinstance(v, tuple):
                if v[1]:
                    y |= 1 << j
            else:
                y |= ((x >> v) & 1) << j
        ok = xp.ones(1 << m, dtype=bool)
        for v, b in self.fixed:
            ok &= ((x >> v) & 1) == b
        return y, ok

    def embed(self, remap, extra_fixed):
        """Return the same atom on a bigger cube: cube var v -> remap[v]; add Boolean fixes."""
        raise NotImplementedError


class View(Atom):
    """View of a committed or public vector."""
    kind = "view"

    def __init__(self, vec, varmap=None, fixed=(), shift=0):
        varmap = tuple(range(vec.nbits)) if varmap is None else tuple(varmap)
        assert len(varmap) == vec.nbits, (vec.name, varmap)
        super().__init__(varmap, fixed)
        self.vec, self.shift = vec, shift
        tag = ("C", vec.id) if isinstance(vec, Vec) else ("U", vec.name)
        self.key = ("V", tag, self.varmap, self.fixed, shift)

    @property
    def public(self): return isinstance(self.vec, PubVec)

    def embed(self, remap, extra_fixed):
        return View(self.vec, [v if isinstance(v, tuple) else remap[v] for v in self.varmap],
                    tuple((remap[v], b) for v, b in self.fixed) + tuple(extra_fixed), self.shift)

    def analyze_block(self, off, m):
        """Pure-Python range analysis of y + shift on the aligned cube block [off, off + 2^m) (zero GPU syncs):
        returns ('empty', 0, 0), ('contig', a0, a1) when y + shift == a0 + (x - off) on all 2^m rows, or ('range', lo, hi)."""
        for v, b in self.fixed:
            if v >= m and ((off >> v) & 1) != b:
                return ("empty", 0, 0)
        N = 1 << self.vec.nbits
        nb = len(self.varmap)
        if (not any(v < m for v, _ in self.fixed) and nb >= m
                and all(self.varmap[j] == j for j in range(m))
                and all(isinstance(self.varmap[j], tuple) or self.varmap[j] >= m for j in range(m, nb))):
            base = self.shift
            for j in range(m, nb):
                vj = self.varmap[j]
                bit = vj[1] if isinstance(vj, tuple) else ((off >> vj) & 1)
                base += bit << j
            if 0 <= base and base + (1 << m) <= N:
                return ("contig", base, base + (1 << m))
        fx = dict(self.fixed)
        ylo = yhi = self.shift
        for j, vj in enumerate(self.varmap):
            if isinstance(vj, tuple):
                b = vj[1]; ylo += b << j; yhi += b << j
            elif vj >= m:
                b = (off >> vj) & 1; ylo += b << j; yhi += b << j
            elif vj in fx:
                b = fx[vj]; ylo += b << j; yhi += b << j
            else:
                yhi += 1 << j
        if yhi < 0 or ylo >= N:
            return ("empty", 0, 0)
        return ("range", max(0, ylo), min(N, yhi + 1))

    def table(self, B, nv, vals_e, blk=None):
        off, m = blk if blk is not None else (0, nv)
        kind, a0, a1 = self.analyze_block(off, m)
        if kind == "empty":
            return B.zeros_e(1 << m)
        if kind == "contig":
            return B.xp.ascontiguousarray(vals_e[:, a0:a1])
        xp = B.xp
        y, ok = self.index_map(nv, xp, blk)
        if self.shift:
            y = y + self.shift
            ok &= (y < (1 << self.vec.nbits)) & (y >= 0)
            y = xp.where(ok, y, 0)
        out = vals_e[:, y]
        return out * ok.astype(np.uint64)[None, :]


class Pub(Atom):
    kind = "pub"

    def local_table(self, B):
        raise NotImplementedError

    def local_mle(self, z):
        raise NotImplementedError

    def table(self, B, nv, blk=None):
        xp = B.xp
        if len(self.varmap) > PUB_DIRECT_BITS:          # never materialize a huge local table (eq over a big cube)
            d = self._direct(B, nv, blk)
            if d is not None:
                return d
        loc = self.local_table(B)
        y, ok = self.index_map(nv, xp, blk)
        out = loc[:, y]
        return out * ok.astype(np.uint64)[None, :]

    def _direct(self, B, nv, blk):
        return None

    def _eq_direct(self, B, nv, blk):
        """eq(point, local(x)) on the block directly: product of the constant bits times an eq table over the cube
        variables of the block that feed local variables (same values as the gathered local table)."""
        xp = B.xp
        off, m = blk if blk is not None else (0, nv)
        x = None
        ok = None
        for v, b in self.fixed:
            if v >= m:
                if ((off >> v) & 1) != b:
                    return B.zeros_e(1 << m)
            else:
                x = xp.arange(1 << m, dtype=np.int64) + off if x is None else x
                o = ((x >> v) & 1) == b
                ok = o if ok is None else (ok & o)
        scal = ONE; vary = []
        for j, v in enumerate(self.varmap):
            if isinstance(v, tuple):
                bit = v[1]
            elif v >= m:
                bit = (off >> v) & 1
            else:
                vary.append((v, self.point[j])); continue
            scal = hmul(scal, self.point[j] if bit else hsub(ONE, self.point[j]))
        vary.sort(key=lambda t: t[0])
        t = B.eq_table([p_ for _, p_ in vary])
        if not (len(vary) == m and all(v == i for i, (v, _) in enumerate(vary))):
            xx = xp.arange(1 << m, dtype=np.int64)
            idx = xp.zeros(1 << m, dtype=np.int64)
            for k, (v, _) in enumerate(vary):
                idx |= ((xx >> v) & 1) << k
            t = t[:, idx]
        if scal != ONE:
            t = B.escale(t, scal)
        if ok is not None:
            t = t * ok.astype(np.uint64)[None, :]
        return t

    def mle(self, r):
        return hmul(self.ind_mle(r), self.local_mle(self.local_point(r)))

    def embed(self, remap, extra_fixed):
        c = object.__new__(type(self)); c.__dict__.update(self.__dict__); c.__dict__.pop("_kidv", None)
        c.varmap = tuple(remap[v] for v in self.varmap)
        c.fixed = tuple(sorted(tuple((remap[v], b) for v, b in self.fixed) + tuple(extra_fixed)))
        c.key = (self.key[0], self.key[1], c.varmap, c.fixed)
        return c


class EqP(Pub):
    """eq(point, local x)."""

    def __init__(self, varmap, point, fixed=()):
        super().__init__(varmap, fixed); self.point = [tuple(p) for p in point]
        self.key = ("eq", tuple(self.point), self.varmap, self.fixed)

    def local_table(self, B): return B.eq_table(self.point)
    def local_mle(self, z): return eq_pt(self.point, z)
    def _direct(self, B, nv, blk): return self._eq_direct(B, nv, blk)


class EqZC(Pub):
    """eq(r_zc|varmap, local x) where r_zc is the instance's random point (bound when the instance is assembled)."""
    late = True

    def __init__(self, varmap, fixed=()):
        super().__init__(varmap, fixed); self.point = None
        self.key = ("eqzc", 0, self.varmap, self.fixed)

    def bind(self, r_zc):
        self.point = [r_zc[v] for v in self.varmap]

    def local_table(self, B): return B.eq_table(self.point)
    def local_mle(self, z): return eq_pt(self.point, z)
    def _direct(self, B, nv, blk): return self._eq_direct(B, nv, blk)


class EqFold(Pub):
    """Column fold of a public matrix M (R x C): local(x) = sum_i eq(rho, i) M[i, x], rho = r_zc|rowvars."""
    late = True

    def __init__(self, name, M, rowvars, varmap, fixed=()):
        super().__init__(varmap, fixed)
        self.name, self.M, self.rowvars = name, M, tuple(rowvars)
        assert M.shape[1] <= (1 << len(self.varmap)) and M.shape[0] <= (1 << len(self.rowvars))
        self.point = None; self._cache = {}
        self.key = ("eqfold", name + repr(self.rowvars), self.varmap, self.fixed)

    def bind(self, r_zc):
        self.point = [r_zc[v] for v in self.rowvars]

    def _folded(self):
        k = tuple(self.point)
        if k not in self._cache:
            e = CPU.eq_table(self.point)[:, :self.M.shape[0]]
            a0 = np.zeros(1 << len(self.varmap), dtype=np.uint64)
            a1 = np.zeros_like(a0); a2 = np.zeros_like(a0)
            for i in range(self.M.shape[0]):
                row = self.M[i]
                a0[:len(row)] = CPU.add(a0[:len(row)], CPU.mul(row, e[0, i]))
                a1[:len(row)] = CPU.add(a1[:len(row)], CPU.mul(row, e[1, i]))
                a2[:len(row)] = CPU.add(a2[:len(row)], CPU.mul(row, e[2, i]))
            self._cache[k] = np.stack([a0, a1, a2])
        return self._cache[k]

    def local_table(self, B): return B.asarray(self._folded())

    def local_mle(self, z):
        return CPU.esum(CPU.emul(self._folded(), CPU.eq_table(list(z))))


class SelLt(Pub):
    """[local x < n] (local integer index, var 0 = LSB)."""

    def __init__(self, varmap, n, fixed=()):
        super().__init__(varmap, fixed); self.n = int(n)
        self.key = ("lt", self.n, self.varmap, self.fixed)

    def local_table(self, B):
        xp = B.xp
        v = (xp.arange(1 << len(self.varmap), dtype=np.int64) < self.n).astype(np.uint64)
        return B.ext(v)

    def _direct(self, B, nv, blk):
        y, ok = self.index_map(nv, B.xp, blk)
        v = ((y < self.n) & ok).astype(np.uint64)
        return B.ext(v)

    def local_mle(self, z):
        return mle_lt(z, self.n)


class IndexP(Pub):
    """local integer index sum_j 2^j x_j (times scale)."""

    def __init__(self, varmap, scale=1, fixed=()):
        super().__init__(varmap, fixed); self.scale = scale % P
        self.key = ("idx", self.scale, self.varmap, self.fixed)

    def local_table(self, B):
        xp = B.xp
        v = B.mul(xp.arange(1 << len(self.varmap), dtype=np.uint64), np.uint64(self.scale))
        return B.ext(v)

    def _direct(self, B, nv, blk):
        xp = B.xp
        y, ok = self.index_map(nv, xp, blk)
        v = B.mul(y.astype(np.uint64), np.uint64(self.scale)) * ok.astype(np.uint64)
        return B.ext(v)

    def local_mle(self, z):
        acc = ZERO
        for j, zj in enumerate(z):
            acc = hadd(acc, hmul((pow(2, j, P), 0, 0), zj))
        return hmul(acc, (self.scale, 0, 0))


def mle_lt(z, n):
    """sum_{x < n} eq(z, x) over len(z) variables."""
    m = len(z)
    if n >= (1 << m):
        return ONE
    if n <= 0:
        return ZERO
    # walk from MSB: x < n iff at the first differing bit (from MSB) x has 0 and n has 1
    acc = ZERO; pref = ONE
    for j in range(m - 1, -1, -1):
        bit = (n >> j) & 1
        if bit:
            low = ONE  # sum over lower bits of eq is 1
            acc = hadd(acc, hmul(pref, hsub(ONE, z[j])))
            pref = hmul(pref, z[j])
        else:
            pref = hmul(pref, hsub(ONE, z[j]))
    return acc


def mle_shift(r, alpha, delta):
    """sum_{x, x+delta < 2^n} eq(r, x) eq(alpha, x + delta)."""
    n = len(r)
    if delta >= (1 << n):
        return ZERO
    f = [ONE, ZERO]
    for i in range(n):
        d = (delta >> i) & 1; g = [ZERO, ZERO]
        for c in (0, 1):
            if f[c] == ZERO:
                continue
            for xb in (0, 1):
                s = xb + d + c; yb = s & 1; c2 = s >> 1
                w = hmul(r[i] if xb else hsub(ONE, r[i]), alpha[i] if yb else hsub(ONE, alpha[i]))
                g[c2] = hadd(g[c2], hmul(f[c], w))
        f = g
    return f[0]


def shift_table(B, r, delta, nbits):
    """W[y] = eq(r, y - delta) (prover side): <vec, W> = sum_x eq(r, x) vec[x + delta]."""
    e = B.eq_table(list(r))
    N = 1 << nbits
    out = B.zeros_e(N)
    if 0 <= delta < N:
        out[:, delta:] = e[:, :N - delta]
    elif -N < delta < 0:
        out[:, :N + delta] = e[:, -delta:]
    return out


def mle_shift_signed(z, alpha, delta):
    return mle_shift(z, alpha, delta) if delta >= 0 else mle_shift(alpha, z, -delta)


# ============================================================================ linear forms and expressions
# Atoms are identified by an interned integer id (``kid``) derived from their structural ``key``; linear forms are
# keyed by frozensets of (kid, coefficient) so hashing never walks deep tuples or calls repr() (avoids an O(C^2) repr
# cost).  The ORDER of terms / factors never enters the transcript: only the set of view atoms
# (sorted by repr(key) in _view_atoms) and the polynomial's values do, so prover and verifier stay identical.
_KID = {}


def _intern(key):
    k = _KID.get(key)
    if k is None:
        k = _KID[key] = len(_KID)
    return k


def _akid(a):
    k = a.__dict__.get("_kidv")
    if k is None:
        k = a.__dict__["_kidv"] = _intern(a.key)
    return k


Atom.kid = property(_akid)


class Lin:
    """sum_k c_k atom_k + const.  items: {kid: (atom, coef)}."""
    __slots__ = ("items", "const", "_key", "_pub")

    def __init__(self, items=None, const=ZERO, _own=False):
        self.items = items if (_own and items is not None) else dict(items or {})
        self.const = _to_e(const); self._key = None; self._pub = None

    @property
    def key(self):
        if self._key is None:
            self._key = (frozenset((k, c) for k, (a, c) in self.items.items()), self.const)
        return self._key

    def is_pub(self):
        """exactly one public structured atom, no constant (mergeable factor)."""
        if self._pub is None:
            self._pub = (self.const == ZERO and len(self.items) == 1
                         and isinstance(next(iter(self.items.values()))[0], Pub))
        return self._pub

    def scaled(self, c):
        c = _to_e(c)
        if c == ONE:
            return self
        return Lin({k: (a, hmul(v, c)) for k, (a, v) in self.items.items()}, hmul(self.const, c), _own=True)

    def plus(self, other):
        acc = _LinAcc(); acc.add(self, ONE); acc.add(other, ONE)
        return acc.lin()

    def atoms(self):
        return [a for a, _ in self.items.values()]

    def embed(self, remap, extra_fixed):
        it = {}
        for a, c in self.items.values():
            b = a.embed(remap, extra_fixed); it[b.kid] = (b, c)
        return Lin(it, self.const, _own=True)


class _LinAcc:
    """Mutable accumulator of scaled linear forms (one dict, no per-addition copies)."""
    __slots__ = ("it", "c0", "c1", "c2")

    def __init__(self):
        self.it = {}; self.c0 = 0; self.c1 = 0; self.c2 = 0

    def add(self, l, c):
        it = self.it
        if c == ONE:
            for k, (a, v) in l.items.items():
                e = it.get(k)
                it[k] = (a, v) if e is None else (a, ((e[1][0] + v[0]) % P, (e[1][1] + v[1]) % P, (e[1][2] + v[2]) % P))
            self.c0 += l.const[0]; self.c1 += l.const[1]; self.c2 += l.const[2]
        else:
            for k, (a, v) in l.items.items():
                v = hmul(v, c)
                e = it.get(k)
                it[k] = (a, v) if e is None else (a, ((e[1][0] + v[0]) % P, (e[1][1] + v[1]) % P, (e[1][2] + v[2]) % P))
            cc = hmul(l.const, c); self.c0 += cc[0]; self.c1 += cc[1]; self.c2 += cc[2]

    def addc(self, c):
        self.c0 += c[0]; self.c1 += c[1]; self.c2 += c[2]

    def lin(self):
        return Lin({k: v for k, v in self.it.items() if v[1] != ZERO}, (self.c0 % P, self.c1 % P, self.c2 % P), _own=True)


def _unit(l, p):
    """single-atom, constant-free Lin with coefficient c  ->  (unit Lin, c^p)."""
    if l.const != ZERO or len(l.items) != 1:
        return l, ONE
    k, (a, c) = next(iter(l.items.items()))
    if c == ONE:
        return l, ONE
    return Lin({k: (a, ONE)}, _own=True), hpow(c, p)


class Ex:
    """Sum of products of Lin factors: list of (coef, ((Lin, pow), ...))."""
    __slots__ = ("terms",)

    def __init__(self, terms=None):
        self.terms = list(terms or [])

    @staticmethod
    def const(c):
        return Ex([(_to_e(c), ())])

    @staticmethod
    def atom(a, c=ONE):
        return Ex([(ONE, ((Lin({a.kid: (a, _to_e(c))}, _own=True), 1),))])

    @staticmethod
    def lin(l):
        return Ex([(ONE, ((l, 1),))])

    @staticmethod
    def sum_all(exs, factor=False):
        """Sum of many expressions with ONE normalisation pass (no quadratic re-normalising)."""
        terms = []
        for e in exs:
            terms.extend(e.terms)
        return Ex(terms).norm(factor=factor)

    def _coerce(self, o):
        if isinstance(o, Ex):
            return o
        return Ex.const(o)

    def __add__(self, o): return Ex(self.terms + self._coerce(o).terms).norm()
    __radd__ = __add__
    def __neg__(self): return Ex([(hneg(c), f) for c, f in self.terms])
    def __sub__(self, o): return self + (-self._coerce(o))
    def __rsub__(self, o): return self._coerce(o) - self

    def scaled(self, c):
        c = _to_e(c)
        return Ex([(hmul(a, c), f) for a, f in self.terms])

    def __mul__(self, o):
        o = self._coerce(o)
        if len(o.terms) == 1 and o.terms[0][1] == ():
            return self.scaled(o.terms[0][0])
        if len(self.terms) == 1 and self.terms[0][1] == ():
            return o.scaled(self.terms[0][0])
        out = []
        for c1, f1 in self.terms:
            for c2, f2 in o.terms:
                out.append((hmul(c1, c2), _merge_factors(f1, f2)))
        return Ex(out).norm()
    __rmul__ = __mul__

    def __pow__(self, k):
        if len(self.terms) == 1 and len(self.terms[0][1]) == 1 and self.terms[0][0] == ONE:
            l, p = self.terms[0][1][0]
            return Ex([(ONE, ((l, p * k),))])
        lin = self.as_lin()
        if lin is not None:
            return Ex([(ONE, ((lin, k),))])
        out = Ex.const(1)
        for _ in range(k):
            out = out * self
        return out

    def as_lin(self):
        """If the expression is affine, return it as one Lin."""
        acc = _LinAcc()
        for c, f in self.terms:
            if f == ():
                acc.addc(c)
            elif len(f) == 1 and f[0][1] == 1:
                acc.add(f[0][0], c)
            else:
                return None
        return acc.lin()

    def norm(self, factor=False):
        lin = _LinAcc(); others = {}; merged = {}
        for c, f in self.terms:
            if c == ZERO:
                continue
            if f == ():
                lin.addc(c)
            elif len(f) == 1 and f[0][1] == 1:
                lin.add(f[0][0], c)
            else:
                m = _mergeable(f)
                if m is not None:
                    pubs, l = m
                    k = frozenset((x.key, p) for x, p in pubs)
                    e = merged.get(k)
                    if e is None:
                        e = merged[k] = (pubs, _LinAcc())
                    e[1].add(l, c)
                else:
                    k = frozenset((l.key, p) for l, p in f)
                    e = others.get(k)
                    others[k] = (c, f) if e is None else (hadd(e[0], c), f)
        terms = [(c, f) for c, f in others.values() if c != ZERO]
        for pubs, acc in merged.values():
            l = acc.lin()
            if l.items or l.const != ZERO:
                terms.append((ONE, tuple(pubs) + ((l, 1),)))
        if factor:
            terms = _factor2(terms)
        l = lin.lin()
        if l.items:
            terms.append((ONE, ((l, 1),)))
        elif l.const != ZERO:
            terms.append((l.const, ()))
        return Ex(terms)

    def degree(self):
        return max((sum(p for _, p in f) for _, f in self.terms), default=0)

    def embed(self, remap, extra_fixed):
        return Ex([(c, tuple((l.embed(remap, extra_fixed), p) for l, p in f)) for c, f in self.terms])


def _factor2(terms):
    """Common-factor extraction:  sum_i c_i K Y_i  ->  K (sum_i c_i Y_i)  where K is a product of factors shared by
    several terms and Y_i is each term's remaining pow-1 factor.  Single-atom factors are first normalised to
    coefficient 1 (gamma*H and H become the same factor).  This turns the per-part LogUp terms
    eq * H * (gamma ind_i - a_i) of a merged lookup domain into ONE product (2^21 cube: ~1000 terms -> a handful)."""
    cand = []; rest = []
    for c, f in terms:
        if 2 <= len(f) <= 5:
            nf = []
            for l, p in f:
                l2, cc = _unit(l, p)
                if cc != ONE:
                    c = hmul(c, cc)
                nf.append((l2, p))
            cand.append((c, nf))
        else:
            rest.append((c, f))
    if len(cand) < 2:
        return terms
    cnt = {}
    opts = []
    for c, f in cand:
        ks = [(l.key, p) for l, p in f]
        o = []
        for j, (l, p) in enumerate(f):
            if p != 1:
                continue
            key = frozenset(ks[:j] + ks[j + 1:])
            if len(key) != len(f) - 1:
                continue
            cnt[key] = cnt.get(key, 0) + 1
            o.append((key, j))
        opts.append(o)
    groups = {}
    out = rest
    for (c, f), o in zip(cand, opts):
        if not o:
            out.append((c, tuple(f))); continue
        key, j = max(o, key=lambda kj: cnt[kj[0]])
        if cnt[key] < 2:
            out.append((c, tuple(f))); continue
        e = groups.get(key)
        if e is None:
            e = groups[key] = (tuple(x for i, x in enumerate(f) if i != j), _LinAcc())
        e[1].add(f[j][0], c)
    for kf, acc in groups.values():
        y = acc.lin()
        if y.items or y.const != ZERO:
            out.append((ONE, _merge_factors(kf, ((y, 1),))))
    return out


def _mergeable(f):
    """Factors = (pub-only factors) x (exactly one pow-1 general Lin)  ->  (pubs, that Lin)."""
    pubs = []; gen = None
    for l, p in f:
        if l.is_pub():
            pubs.append((l, p))
        else:
            if gen is not None or p != 1:
                return None
            gen = l
    if gen is not None and pubs:
        return pubs, gen
    return None


def _merge_factors(f1, f2):
    d = {}
    for l, p in f1:
        k = l.key
        e = d.get(k)
        d[k] = (l, p) if e is None else (l, e[1] + p)
    for l, p in f2:
        k = l.key
        e = d.get(k)
        d[k] = (l, p) if e is None else (l, e[1] + p)
    return tuple(d.values())


# ============================================================================ circuit container
class Lookup:
    def __init__(self, name):
        self.name = name; self.table = None; self.parts = []; self.mult = None
        self.table_vals = None; self.H = None


class Circuit:
    """Holds committed / public vectors and constraints.  Built identically by prover (with values) and verifier."""

    def __init__(self, prove):
        self.prove = prove
        self.vecs = []; self.pubs = {}
        self.zeros = {}      # nv -> list of (Ex, tag)
        self.sums = {}       # nv -> list of (Ex, eq_vars, tag)
        self.lookups = {}
        self.finalized = False
        self.rounds = []     # extra commit rounds after phase 1: callables f(tr) (see _commit_rounds)
        self.cur_phase = 1   # vectors declared with the default phase 1 inside round k get phase 10 + k

    # -- vectors
    def vec(self, name, nbits, content=None, vals=None, phase=1, tail=None):
        if phase == 1:
            phase = self.cur_phase
        t = TB if tail is None else tail
        content = (1 << nbits) - t if content is None else content
        v = Vec(len(self.vecs), name, nbits, content, phase, t)
        if self.prove and vals is not None:
            self.set(v, vals)
        self.vecs.append(v)
        return v

    def set(self, v, vals):
        """vals: integer array-like (int64 or object) of length <= content.  Stored compactly (``compact_vals``):
        content rows only (no zero tail), smallest integer dtype; field conversion happens when the group is
        assembled (``PhaseCommit``)."""
        a = np.asarray(vals)
        if len(a) > v.content:
            raise ValueError(f"{v.name}: {len(a)} values exceed content {v.content}")
        v.vals = compact_vals(a)

    def pubvec(self, name, vals, nbits=None):
        if name in self.pubs:
            return self.pubs[name]
        pv = PubVec(name, vals, nbits); self.pubs[name] = pv
        return pv

    # -- constraints
    def zero(self, nv, ex, sel=None, tag=""):
        """For every x on the cube with sel(x) != 0:  ex(x) = 0.  sel is an Ex (default: all rows)."""
        e = ex if sel is None else ex * sel
        if e.terms:
            self.zeros.setdefault(nv, []).append((e, tag))

    def sumzero(self, nv, ex, eq_vars=(), tag=""):
        """sum over all cube variables not in eq_vars of ex(x) = 0, for every assignment of eq_vars."""
        if ex.terms:
            self.sums.setdefault(nv, []).append((ex, tuple(eq_vars), tag))

    # -- lookups
    def lookup_table(self, name, nv, cols, mult_content, vals=None):
        """Declare table `name` on an nv-cube: cols = list of affine Ex (tuple components); vals = int arrays (prover)."""
        lk = self.lookups.setdefault(name, Lookup(name))
        assert lk.table is None
        lk.table = (nv, cols); lk.table_vals = vals
        lk.mult = self.vec(f"mult:{name}", nv, mult_content)
        if not hasattr(self, "_shard"):
            self._shard = {}
        self._shard[name] = {"spec": (nv, cols, mult_content, vals), "cur": name, "k": 0, "used": 1 << nv}
        return lk

    def _route(self, name, nv):
        """Fixed tables are sharded deterministically (declaration order, identical for prover and verifier): once the
        parts of a shard would push its LogUp helper H beyond 2^SHARD_BITS cells, further parts go to a fresh copy of
        the table (``name#k``), so no H vector forces a commitment group above GROUP_BITS."""
        sh = getattr(self, "_shard", {}).get(name)
        if sh is None:
            return name
        sz = 1 << nv
        if sh["used"] + sz > (1 << SHARD_BITS) - TB and sh["used"] > (1 << sh["spec"][0]):
            sh["k"] += 1
            tnv, tcols, mc, tv = sh["spec"]
            nm = f"{name}#{sh['k']}"
            lk = self.lookups.setdefault(nm, Lookup(nm))
            lk.table = (tnv, tcols); lk.table_vals = tv
            lk.mult = self.vec(f"mult:{nm}", tnv, mc)
            sh["cur"] = nm; sh["used"] = 1 << tnv
        sh["used"] += sz
        return sh["cur"]

    def lookup(self, name, nv, cols, sel=None, vals=None, sel_vals=None):
        """Every selected row of cols (tuple of affine Ex on an nv-cube) must be a row of table `name`."""
        name = self._route(name, nv)
        lk = self.lookups.setdefault(name, Lookup(name))
        hook = getattr(self, "fresh_hook", None)
        if hook is not None:
            cols = [hook(c) for c in cols]
            sel = hook(sel) if sel is not None else None
        vals, sel_vals = _compact_part(vals, sel_vals)       # prover: compact copies, builder temporaries released
        lk.parts.append((nv, cols, sel, vals, sel_vals))


# ============================================================================ phase commitments
def _spill_root():
    """Spill files must live on a real disk (/tmp is often a RAM-backed tmpfs): $FPN_SPILL_DIR, else
    ~/.cache/fpn-spill."""
    root = os.environ.get("FPN_SPILL_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "fpn-spill")
    os.makedirs(root, exist_ok=True)
    return root


def _spill_dir():
    import tempfile, shutil
    root = _spill_root()
    for d in os.listdir(root):                 # stale spills of killed provers (pid no longer alive)
        parts = d.split("-")
        if d.startswith("fpn-spill-") and len(parts) >= 4 and parts[2].isdigit():
            try:
                os.kill(int(parts[2]), 0)
            except ProcessLookupError:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            except OSError:
                pass
    return tempfile.mkdtemp(prefix=f"fpn-spill-{os.getpid()}-", dir=root)


def _drop_cache(path):
    """Write a spilled file through to disk and evict its (now clean) pages from the page cache (Linux)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            getattr(os, "fdatasync", os.fsync)(fd)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _write_npy(path, arr, sync=True):
    """plain buffered writes (msync of a writable memmap can be ~10x slower); sync: + fdatasync + page-cache eviction
    (giant groups); otherwise the kernel writes the file back in the background and its pages stay reclaimable cache,
    not process memory."""
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(arr.dtype), "fortran_order": False,
                                                 "shape": arr.shape})
        flat = arr.reshape(-1).view(np.uint8)
        for a0 in range(0, flat.size, 64 << 20):
            f.write(flat[a0:a0 + (64 << 20)])
    if sync:
        _drop_cache(path)


SPILL_INFLIGHT = float(os.environ.get("FPN_SPILL_INFLIGHT_GB", "1")) * (1 << 30)


class _SpillWriter:
    """One background thread writes spilled arrays to disk (overlapping the prover's compute); at most
    SPILL_INFLIGHT bytes wait in RAM, beyond that ``submit`` blocks (bounded host memory on slow disks)."""

    def __init__(self):
        import threading, queue
        self.q = queue.Queue(); self.cv = threading.Condition(); self.inflight = 0; self.err = None
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            path, arr, done, sync = job
            try:
                _write_npy(path, arr, sync)
            except BaseException as e:      # noqa: BLE001  (re-raised in the prover thread)
                self.err = e
            with self.cv:
                self.inflight -= arr.nbytes
                self.cv.notify_all()
            done.set()

    def drain(self):
        with self.cv:
            while self.inflight > 0:
                self.cv.wait()

    def submit(self, path, arr, sync=True):
        import threading
        if self.err is not None:
            raise ProofError(f"spill write failed: {self.err}")
        with self.cv:
            while self.inflight > 0 and self.inflight + arr.nbytes > SPILL_INFLIGHT:
                self.cv.wait()
            self.inflight += arr.nbytes
        done = threading.Event()
        self.q.put((path, arr, done, sync))
        return done


_WRITER = None


class _Spilled:
    """A host array being written to disk in the background; ``get()`` returns a read-only memmap once written."""

    def __init__(self, dirname, name, arr, sync=True):
        global _WRITER
        if _WRITER is None:
            _WRITER = _SpillWriter()
        self.path = os.path.join(dirname, name + ".npy")
        self.shape = arr.shape; self.nbytes = arr.nbytes
        self.done = _WRITER.submit(self.path, np.ascontiguousarray(arr), sync)
        self._mm = None

    def load(self):
        """the whole array read sequentially into RAM (fast from the page cache; memmap page faults are not)."""
        self.done.wait()
        if _WRITER.err is not None:
            raise ProofError(f"spill write failed: {_WRITER.err}")
        return np.load(self.path)

    def get(self):
        if self._mm is None:
            self.done.wait()
            if _WRITER.err is not None:
                raise ProofError(f"spill write failed: {_WRITER.err}")
            self._mm = np.load(self.path, mmap_mode="r")
        return self._mm

    def read(self, a0, a1):
        """columns [a0, a1) along the last axis (rows of a 1-D array) as an in-RAM copy; the file mapping is dropped
        right after, so the pages count as page cache, not as process memory."""
        mm = self.get()
        out = np.array(mm[..., a0:a1])
        self._mm = None
        del mm
        return out

    def __getitem__(self, idx):
        return self.get()[idx]

    def __len__(self):
        return self.shape[0]


HOST_BUDGET = float(os.environ.get("FPN_HOST_SPILL_GB", "4"))   # random group material kept in RAM up to this many GB per phase
OOC = os.environ.get("FPN_OOC", "") not in ("", "0")               # out-of-core: every group's random material on disk


def _is_ooc():
    return OOC or os.environ.get("FPN_OOC", "") not in ("", "0")


_RND_C64 = np.uint64((1 << 32) - 1)
_RND_C128 = np.uint64(P - (1 << 32))
_RND_PU = np.uint64(P)


def _aes_ctr_bytes(key, blk0, nbytes):
    """Deterministic AES-256-CTR stream starting at 128-bit big-endian block counter `blk0` (`nbytes` bytes)."""
    out = np.empty(nbytes, dtype=np.uint8)
    if nbytes == 0:
        return out
    mv = memoryview(out)
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        if nbytes >= (4 << 20):
            from . import field as _F
            if _F._URAND_POOL is None:
                from concurrent.futures import ThreadPoolExecutor
                _F._URAND_POOL = ThreadPoolExecutor(max_workers=min(16, max(2, (os.cpu_count() or 4))))
            chunk = 1 << 20
            z = bytes(chunk)

            def fill(off):
                k = min(chunk, nbytes - off)
                iv = (blk0 + (off >> 4)).to_bytes(16, "big")
                mv[off:off + k] = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor().update(z[:k])
            list(_F._URAND_POOL.map(fill, range(0, nbytes, chunk)))
            return out
        enc = Cipher(algorithms.AES(key), modes.CTR(blk0.to_bytes(16, "big"))).encryptor()
        step = 32 << 20
        z = bytes(min(step, nbytes))
        for off in range(0, nbytes, step):
            k = min(step, nbytes - off)
            mv[off:off + k] = enc.update(z[:k])
        return out
    except ImportError:
        import hashlib
        assert nbytes % 16 == 0
        for b in range(nbytes >> 4):
            mv[16 * b:16 * (b + 1)] = hashlib.blake2b(key + (blk0 + b).to_bytes(16, "big"), digest_size=16).digest()
        return out


def _prg_rnd_cols(subkeys, a0, a1, limbs=(0, 1, 2)):
    """Random-access T1 blinding columns [a0, a1) from per-limb AES-256-CTR keys (192 bits mod P; P | 2^192 - 1 ->
    statistical distance < 2^-192 per coordinate, 0 bytes resident in RAM or disk)."""
    m = a1 - a0
    out = np.empty((len(limbs), m), dtype=np.uint64)
    if m <= 0:
        return out
    for k, l in enumerate(limbs):
        raw = _aes_ctr_bytes(subkeys[l], 2 * a0, 32 * m).view(np.uint64).reshape(m, 4)
        r0 = np.where(raw[:, 0] >= _RND_PU, raw[:, 0] - _RND_PU, raw[:, 0])
        r1 = np.where(raw[:, 1] >= _RND_PU, raw[:, 1] - _RND_PU, raw[:, 1])
        r2 = np.where(raw[:, 2] >= _RND_PU, raw[:, 2] - _RND_PU, raw[:, 2])
        out[k] = CPU.add(r0, CPU.add(CPU.mul(r1, _RND_C64), CPU.mul(r2, _RND_C128)))
    return out


class PhaseCommit:
    """One commitment group: Big (all vectors aligned + uniform filler), the T4 mask h_bf, 256-bit leaf salts and the
    round-0 Merkle tree (``CommitTree``).  In-RAM groups draw h_bf, the salts and every blinding row straight from
    os.urandom (statistical ZK).  Lean groups (phase over HOST_BUDGET, groups >= 2^LEAN_BITS, or FPN_OOC=1; outside
    FPN_TEST_RNG_SEED) draw a fresh per-group 256-bit secret key from os.urandom, derive domain-separated BLAKE2b
    subkeys (``rnd0..2``, ``h0..2``, ``salt``), regenerate T1 tails / h_bf / salts via AES-256-CTR on demand, and erase
    the key in ``release()`` (computational ZK under A-PRG of specs/07, 0 bytes resident in RAM or disk)."""

    def __init__(self, B, vecs, nb_min=3):
        self.B = B
        self.vecs = sorted(vecs, key=lambda v: (-v.nbits, v.id))
        off = 0; self.offset = {}; self.toff = {}; to = 0
        for v in self.vecs:
            assert off % v.N == 0
            self.offset[v.id] = off; off += v.N
            self.toff[v.id] = to; to += v.tail
        self.ntail = to
        off = (off + (1 << FILLER_BITS) - 1) >> FILLER_BITS << FILLER_BITS
        self.filler_off = off; off += 1 << FILLER_BITS
        nb = max(nb_min, (off - 1).bit_length())
        self.nb = nb; self.N = 1 << nb
        self.vec_by_id = {v.id: v for v in self.vecs}
        self.tree = None; self._h = None; self._salt = None; self._spill = None
        self._rnd = None; self._fill = None; self.built = False; self._big = None
        self._rng_key = None; self._rnd_keys = None

    force_lean = False

    @property
    def lean(self):
        return self.force_lean or _is_ooc() or self.nb >= LEAN_BITS

    def ram_bytes(self):
        """group material currently held in RAM (assembled Big of in-RAM groups, tails, h_bf, salts)."""
        n = 0
        for x in (self._big, self._rnd, self._h, getattr(self.tree, "salt", None), getattr(self, "_hfull", None)):
            if isinstance(x, np.ndarray):
                n += x.nbytes
        return n

    def disk_bytes(self):
        n = 0
        for x in (self._rnd, self._h, getattr(self.tree, "salt", None)):
            if isinstance(x, _Spilled):
                n += x.nbytes
        return n

    # ---------------------------------------------------------------- assembly of Big from the group store
    def _rnd_cols(self, a0, a1, limbs=(0, 1, 2)):
        if self._rnd is None and self._rnd_keys is not None:
            return _prg_rnd_cols(self._rnd_keys, a0, a1, limbs)
        r = self._rnd
        sub = r.read(a0, a1) if isinstance(r, _Spilled) else r[:, a0:a1]
        return sub if limbs == (0, 1, 2) else sub[list(limbs)]

    def _vec_rows(self, v, a0, a1, limbs=(0, 1, 2)):
        """(len(limbs), a1 - a0) host values of rows [a0, a1) of vector v (witness / E values, then T1 tail)."""
        if self._big is not None:
            o = self.offset[v.id]
            return np.array(self._big[list(limbs), o + a0:o + a1])
        n = a1 - a0
        out = np.zeros((len(limbs), n), dtype=np.uint64)
        t0 = v.N - v.tail
        if v.is_ext:
            cend = v.content if v.ext_gen is not None else t0
            e = _ext_rows(v, a0, min(a1, cend)) if a0 < cend else None
            if e is not None:
                for k, l in enumerate(limbs):
                    out[k, :e.shape[1]] = e[l]
        elif 0 in limbs and a0 < t0:
            w = wit(v)
            if w is None:
                raise ProofError(f"vector {v.name} has no values")
            c1 = min(a1, len(w), t0)
            if c1 > a0:
                out[limbs.index(0), :c1 - a0] = field_u64(w[a0:c1])
        if a1 > t0:
            s0 = max(a0, t0)
            ro = self.toff[v.id]
            r = self._rnd_cols(ro + s0 - t0, ro + a1 - t0, limbs)
            out[:, s0 - a0:] = r
        return out

    def vec_e(self, v):
        return self._vec_rows(v, 0, v.N)

    def vec_range_e(self, v, a0, a1):
        return self._vec_rows(v, a0, a1)

    def limb(self, l):
        """dense host limb l (N,) of Big."""
        if self._big is not None:
            return self._big[l]
        out = np.zeros(self.N, dtype=np.uint64)
        rnd_l = _prg_rnd_cols(self._rnd_keys, 0, self.ntail, (l,))[0] if (self._rnd is None and self._rnd_keys is not None and self.ntail > 0) else None
        for v in self.vecs:
            if l != 0 and not v.is_ext:          # base vector: only its tail rows have limbs 1, 2
                o = self.offset[v.id] + v.N - v.tail; ro = self.toff[v.id]
                out[o:o + v.tail] = rnd_l[ro:ro + v.tail] if rnd_l is not None else self._rnd_cols(ro, ro + v.tail, (l,))[0]
                continue
            o = self.offset[v.id]
            if rnd_l is not None:
                t0 = v.N - v.tail
                if v.is_ext:
                    cend = v.content if v.ext_gen is not None else t0
                    if cend > 0:
                        out[o:o + cend] = _ext_rows(v, 0, cend)[l]
                elif l == 0 and t0 > 0:
                    w = wit(v)
                    if w is None:
                        raise ProofError(f"vector {v.name} has no values")
                    c1 = min(len(w), t0)
                    if c1 > 0:
                        out[o:o + c1] = field_u64(w[:c1])
                if v.tail > 0:
                    ro = self.toff[v.id]
                    out[o + t0:o + v.N] = rnd_l[ro:ro + v.tail]
            else:
                out[o:o + v.N] = self._vec_rows(v, 0, v.N, (l,))[0]
        out[self.filler_off:self.filler_off + (1 << FILLER_BITS)] = self._fill[l]
        return out

    @property
    def big(self):
        """dense host (3, N) Big (compatibility / small groups; large groups use ``limb`` one limb at a time)."""
        return np.stack([self.limb(0), self.limb(1), self.limb(2)])

    def h_limb(self, l):
        if self._h is None and self._rng_key is not None:
            with seeded_rng(self._subkey(b"h%d" % l)):
                return rand_field(self.N)
        return np.ascontiguousarray(self.h[l])

    def src(self, fi, l):
        """CommitTree source: limb l of Big (fi = 0) or of h_bf (fi = 1), as a host array."""
        if fi == 0:
            return self.limb(l)
        return self.h_limb(l)

    def _subkey(self, tag):
        import hashlib
        return hashlib.blake2b(b"fpn-pc:" + tag + b":" + self._rng_key, digest_size=32).digest()

    def _regen_salt(self):
        m = (RATE * self.N) >> ARITY_BITS
        with seeded_rng(self._subkey(b"salt")):
            return urandom_u64(m * SALT_WORDS).reshape(m, SALT_WORDS)

    # ---------------------------------------------------------------- commitment
    def build(self):
        B = self.B
        for v in self.vecs:
            if not v.is_ext and wit(v) is None:
                raise ProofError(f"vector {v.name} has no values")
            if v.is_ext and v.vals is None and v.gen is None and v.ext_gen is None:
                raise ProofError(f"vector {v.name} has no values")
        use_prg = self.lean and not os.environ.get("FPN_TEST_RNG_SEED")
        if use_prg:
            self._rng_key = bytes(urandom_u64(4))
            self._rnd_keys = tuple(self._subkey(b"rnd%d" % l) for l in (0, 1, 2))
            self._rnd = None
        else:
            self._rnd = CPU.rand_e(self.ntail)                           # one CSPRNG draw for every T1 tail
        self._fill = CPU.rand_e(1 << FILLER_BITS)
        for v in self.vecs:
            v.grp = self
        if not self.lean:
            # in-RAM group (the phase fits the HOST_BUDGET): keep the assembled Big as before (fastest)
            big = np.empty((3, self.N), dtype=np.uint64)
            big[0] = self.limb(0); big[1] = self.limb(1); big[2] = self.limb(2)
            self._big = big
            self._rnd = None
            if all(v.phase != 1 for v in self.vecs):
                for v in self.vecs:
                    v.vals = None
        m = (RATE * self.N) >> ARITY_BITS
        if use_prg:
            h = None
            salt = self._regen_salt()
        else:
            h = CPU.rand_e(self.N)                                          # T4 mask h_bf: uniform, os.urandom
            salt = urandom_u64(m * SALT_WORDS).reshape(m, SALT_WORDS)      # 256-bit leaf salts, os.urandom
        self._h = h
        self.tree = CommitTree(B, self.src, h, salt, N=self.N)
        self.root = self.tree.root
        self._h = None
        _EXT_CACHE[:] = [None, None, None, None]
        if use_prg:
            self.tree.salt = self._regen_salt
            for v in self.vecs:
                if v.is_ext and isinstance(v.vals, np.ndarray) and v.vals.nbytes >= (512 << 20):
                    if self._spill is None:
                        self._spill = _spill_dir()
                    sync = self.nb >= LEAN_BITS or _is_ooc()
                    v.vals = _Spilled(self._spill, f"ext{v.id}", v.vals, sync)
        elif self.lean:
            self._spill = _spill_dir()      # on a real disk ($FPN_SPILL_DIR or ~/.cache), never tmpfs
            sync = self.nb >= LEAN_BITS or _is_ooc()      # write through to disk and evict; budget spills: page cache
            self._h = _Spilled(self._spill, "h", h, sync)         # written in the background, RAM freed after
            self.tree.salt = _Spilled(self._spill, "salt", salt, sync)
            self._rnd = _Spilled(self._spill, "rnd", self._rnd, sync)
            for v in self.vecs:
                if v.is_ext and isinstance(v.vals, np.ndarray) and v.vals.nbytes >= (8 << 20):
                    v.vals = _Spilled(self._spill, f"ext{v.id}", v.vals, sync)
        else:
            self._h = h; self.tree.salt = salt
        self.tree.h = None                          # the tree must not pin the in-RAM copy of a spilled mask
        self.built = True
        del h, salt
        _free(B)
        return self.root

    @property
    def h(self):
        if self._h is None and self._rng_key is not None:
            if getattr(self, "_hfull", None) is None:
                self._hfull = np.stack([self.h_limb(0), self.h_limb(1), self.h_limb(2)])
            return self._hfull
        if isinstance(self._h, _Spilled):
            # opening: read the spilled mask once and keep it until release (it is read three times per opening)
            if getattr(self, "_hfull", None) is None:
                self._hfull = self._h.load()
            return self._hfull
        return self._h

    def ensure_tree(self):
        if self.tree is None:
            raise ProofError("commitment group already released")
        return self.tree

    def release(self):
        """Drop the opening material and delete the spill files (idempotent; called from a finally block)."""
        for x in (self._h, getattr(self.tree, "salt", None), self._rnd):
            if isinstance(x, _Spilled):
                x.done.wait()               # never delete a file the writer thread still holds
        for v in self.vecs:
            if isinstance(v.vals, _Spilled):
                v.vals.done.wait()
        self.tree = None; self._h = None; self._hfull = None; self._rnd = None; self._fill = None; self._big = None
        self._rng_key = None; self._rnd_keys = None
        _EXT_CACHE[:] = [None, None, None, None]
        _dev_big_drop(self)
        for v in self.vecs:                 # an opened group's vectors are never read again: free them right away
            if v.on_release is not None:
                v.on_release(); v.on_release = None
            if v.grp is self:
                v.full = None; v.grp = None; v.vals = None; v.ext_gen = None
        sp = self._spill
        if sp:
            import shutil
            shutil.rmtree(sp, ignore_errors=True); self._spill = None


KEEP_LEVEL_BITS = 16      # Merkle levels with <= 2^16 nodes are kept; lower levels are rebuilt per query at opening


RESIDUE_BITS = int(os.environ.get("FPN_RESIDUE_BITS", "26"))   # groups >= 2^this: residue-class leaf hashing


def _src_of(big, h):
    """CommitTree source callback for host arrays big, h (2, N)."""
    def src(fi, l):
        return np.ascontiguousarray((big if fi == 0 else h)[l])
    return src


class CommitTree:
    """Round-0 Merkle tree of a group: leaf j = cosets {j + s m} (s < 8) of code(Big) and code(h_bf), then salt_j
    (same leaf layout as ``Tree``).  The rate-1/4 codewords are encoded on the device, hashed and freed: only the
    top levels (<= 2^KEEP_LEVEL_BITS nodes) are kept.  ``open_many`` re-encodes Big and h_bf limb by limb on the
    device, gathers just the subtrees of the queried leaves, rebuilds them and checks each against the kept level
    (no host codeword, no full-tree rebuild).

    ``src(fi, l)`` returns limb l of Big (fi = 0) or h_bf (fi = 1) as a host array (or pass the (2, N) arrays).
    Groups >= 2^RESIDUE_BITS on the GPU never form a whole codeword: all eight entries of leaf j lie in the residue
    class j mod 8 of the codeword, and C[8q + r] = NTT_{n0/8}(G_r)[q] with G_r[k] = sum_t F[k + t n0/8] w^(r (k + t n0/8))
    (a twisted fold of the coefficients), so the leaves are absorbed one residue class at a time from N/2-point
    transforms (same digests as the NumPy path)."""
    CHUNK = 1 << 19

    def __init__(self, B, src, h, salt, N=None):
        xp = B.xp
        if not callable(src):
            N = src.shape[1]; src = _src_of(src, h)
        self.B = B; self.N = N; self.n0 = RATE * self.N
        self.m = self.n0 >> ARITY_BITS
        self.depth = self.m.bit_length() - 1
        self.sub = max(0, self.depth - KEEP_LEVEL_BITS)
        self.src, self.h, self.salt = src, h, salt
        m = self.m
        self.residue = (hasattr(B, "leaf_absorb") and ARITY_BITS == 3 and N >= (1 << RESIDUE_BITS) and m % 8 == 0)
        self._limbs = None
        if self.residue:
            lv0 = self._residue_leaves(salt)
        elif hasattr(B, "leaf_absorb") and ARITY_BITS == 3:
            # VRAM-lean streaming: the 52-word leaf message is [Big_0 | Big_1] (16w), [Big_2 | h_0] (16w), [h_1 | h_2] (16w),
            # then [salt] (4w), i.e. three 128-byte BLAKE2b blocks + final 32-byte block (t = 416), so at most two codeword
            # limbs (2 x 8 n0 bytes) plus the per-leaf states (8 n0 bytes) are ever resident on the device.
            H = xp.empty((m, 8), dtype=np.uint64)
            for blk_i, ((f0, l0), (f1, l1)) in enumerate((((0, 0), (0, 1)), ((0, 2), (1, 0)), ((1, 1), (1, 2)))):
                L0 = B.ntt_host(src(f0, l0), self.n0); L1 = B.ntt_host(src(f1, l1), self.n0)
                B.leaf_absorb(L0, L1, m, H, blk_i == 0, 128 * (blk_i + 1))
                del L0, L1
                _free(B)
            lv0 = B.leaf_final(H, salt, 8 * (2 * (3 << ARITY_BITS) + SALT_WORDS))
            del H
        else:
            limbs = [self._ntt(fi, l) for fi in (0, 1) for l in (0, 1, 2)]
            lv0 = xp.empty((m, 4), dtype=np.uint64)
            for a0 in range(0, m, self.CHUNK):
                a1 = min(m, a0 + self.CHUNK)
                rows = xp.empty((a1 - a0, 2 * (3 << ARITY_BITS) + SALT_WORDS), dtype=np.uint64)
                for k, L in enumerate(limbs):
                    for s_ in range(1 << ARITY_BITS):
                        rows[:, 8 * k + s_] = L[a0 + s_ * m:a1 + s_ * m]
                rows[:, 48:] = B.asarray(salt[a0:a1])
                lv0[a0:a1] = B.blake_rows(rows)
                del rows
            # CPU back end (small unit-test circuits): keep the host codewords; device back ends re-encode at opening
            self._limbs = limbs if B.name == "numpy" else None
            del limbs
        _free(B)
        self.levels = [None] * (self.depth + 1)
        top = lv0
        for d in range(self.depth + 1):
            if d >= self.sub:
                self.levels[d] = B.tonumpy(top)
            if d < self.depth:
                top = B.blake_rows(top.reshape(-1, 8))
        del lv0, top
        self.root = self.levels[-1][0].tobytes()

    def _ntt(self, fi, l):
        B = self.B
        row = self.src(fi, l)
        if hasattr(B, "ntt_host"):
            return B.ntt_host(row, self.n0)
        return B.ntt(np.asarray(row), self.n0)

    # ---------------------------------------------------------------- residue classes (large groups, GPU)
    def _coset_ntt(self, F, r):
        """D_r[q] = C[8q + r] for q < m (C = rate-RATE codeword of the device coefficient vector F of length N)."""
        B = self.B
        m, N = self.m, self.N
        if r == 0:
            G = B.add(F[:m], F[m:])
        else:
            w = pow(_root_of_unity(self.n0), r, P)
            tw = B.pow_dev(w, m)
            wm = pow(w, m, P)
            t0 = B.mul(F[:m], tw)
            t1 = B.mul(F[m:], B.mul(tw, wm))
            del tw
            G = B.add(t0, t1)
            del t0, t1
        return B.ntt(G, m)

    def _residue_leaves(self, salt):
        B, xp, m = self.B, self.B.xp, self.m
        if self.N >= (1 << 27) and hasattr(B, "clear_caches"):
            B.clear_caches(); _free(B)
        q = m >> 3
        H_r = [xp.empty((q, 8), dtype=np.uint64) for _ in range(8)]
        for blk_i, ((f0, l0), (f1, l1)) in enumerate((((0, 0), (0, 1)), ((0, 2), (1, 0)), ((1, 1), (1, 2)))):
            F0 = B.asarray(self.src(f0, l0))
            F1 = B.asarray(self.src(f1, l1))
            for r in range(8):
                D0 = self._coset_ntt(F0, r); D1 = self._coset_ntt(F1, r)
                B.leaf_absorb(D0, D1, q, H_r[r], blk_i == 0, 128 * (blk_i + 1))
                del D0, D1
            del F0, F1
            _free(B)
        sh = salt() if callable(salt) else (salt.get() if isinstance(salt, _Spilled) else salt)
        lv0 = xp.empty((m, 4), dtype=np.uint64)
        for r in range(8):
            lv0[r::8] = B.leaf_final(H_r[r], np.ascontiguousarray(sh[r::8]), 8 * (2 * (3 << ARITY_BITS) + SALT_WORDS))
            H_r[r] = None
        del H_r
        _free(B)
        return lv0

    def _rows(self, js):
        """Leaf rows (len(js), 52) for host leaf indices js: six limb NTTs on the device, one at a time."""
        B = self.B
        js = np.asarray(js, dtype=np.int64)
        m = self.m
        rows = np.empty((len(js), 2 * (3 << ARITY_BITS) + SALT_WORDS), dtype=np.uint64)
        if self.residue:
            res = js & 7; qq = js >> 3; q8 = m >> 3
            for k, (fi, l) in enumerate((fi, l) for fi in (0, 1) for l in (0, 1, 2)):
                F = B.asarray(self.src(fi, l))
                for r in range(8):
                    sel = np.flatnonzero(res == r)
                    if not len(sel):
                        continue
                    D = self._coset_ntt(F, r)
                    pos = (qq[sel][None, :] + (np.arange(8, dtype=np.int64) * q8)[:, None])
                    g = B.tonumpy(D[B.asarray(pos.ravel(), dtype=np.int64)]).reshape(8, len(sel))
                    del D
                    for s_ in range(8):
                        rows[sel, 8 * k + s_] = g[s_]
                del F
                _free(B)
        else:
            pos = (js[None, :] + (np.arange(1 << ARITY_BITS, dtype=np.int64) * m)[:, None])     # (8, n)
            pdev = B.asarray(pos.ravel(), dtype=np.int64)
            for k, (fi, l) in enumerate((fi, l) for fi in (0, 1) for l in (0, 1, 2)):
                L = self._limbs[k] if self._limbs is not None else self._ntt(fi, l)
                g = B.tonumpy(L[pdev]).reshape(1 << ARITY_BITS, len(js))
                del L
                _free(B)
                for s_ in range(1 << ARITY_BITS):
                    rows[:, 8 * k + s_] = g[s_]
        if callable(self.salt):
            salt = self.salt()
        elif isinstance(self.salt, _Spilled):
            salt = self.salt.get()
        else:
            salt = self.salt
        rows[:, 48:] = np.asarray(salt[js])
        return rows

    def open_many(self, js):
        B = self.B
        js = [int(j) for j in js]
        sub = self.sub
        qs = sorted({j >> sub for j in js})
        span = 1 << sub
        leaves = np.concatenate([np.arange(q << sub, (q + 1) << sub, dtype=np.int64) for q in qs])
        rows = self._rows(leaves)
        # rebuild every queried subtree (levels 0 .. sub) and pin it to the kept level `sub`
        sublv = [B.tonumpy(B.blake_rows(B.asarray(rows)))]
        for d in range(sub):
            sublv.append(B.tonumpy(B.blake_rows(B.asarray(sublv[-1].reshape(-1, 8)))))
        qpos = {q: i for i, q in enumerate(qs)}
        for i, q in enumerate(qs):
            if sublv[sub][i].tobytes() != self.levels[sub][q].tobytes():
                raise ProofError("re-encoded commitment does not match its Merkle tree")
        out = []
        for j in js:
            i = qpos[j >> sub]
            data = rows[i * span + (j - ((j >> sub) << sub))]
            path, q = [], j
            for d in range(self.depth):
                if d < sub:
                    loc = (q ^ 1) - ((j >> sub) << (sub - d))
                    path.append(sublv[d][i * (span >> d) + loc].tobytes())
                else:
                    path.append(self.levels[d][q ^ 1].tobytes())
                q >>= 1
            out.append((data, path))
        return out


_DEV_BIG = {}             # PhaseCommit -> device copy of its Big (GPU only), LRU-bounded
DEV_BIG_BUDGET = float(os.environ.get("FPN_DEV_BIG_GB", "4")) * (1 << 30)
VCACHE_BYTES = float(os.environ.get("FPN_VCACHE_GB", "3")) * (1 << 30)   # sumcheck device vector cache bound
BIG_INSTANCE_BITS = int(os.environ.get("FPN_BIG_INSTANCE_BITS", "64"))      # sumcheck cubes >= 2^this run cache-free
BIG_INSTANCE_BYTES = float(os.environ.get("FPN_BIG_INSTANCE_GB", "1.5")) * (1 << 30)   # ... or tables above this
_NO_DEVBIG = [False]


def _dev_big(B, pc):
    nbytes = 24 * pc.N
    if B.name != "cupy" or nbytes > DEV_BIG_BUDGET / 2 or _NO_DEVBIG[0] or pc.lean:
        return None
    d = _DEV_BIG.pop(pc, None)
    if d is None:
        while _DEV_BIG and sum(x.nbytes for x in _DEV_BIG.values()) + nbytes > DEV_BIG_BUDGET:
            _DEV_BIG.pop(next(iter(_DEV_BIG)))
        d = B.zeros_e(pc.N)
        for l in (0, 1, 2):
            d[l] = B.asarray(pc.limb(l))
    _DEV_BIG[pc] = d          # most recently used last
    return d


def _dev_big_drop(pc=None):
    if pc is None:
        _DEV_BIG.clear()
    else:
        _DEV_BIG.pop(pc, None)


def partition(vecs, max_bits=None):
    """Deterministic split of one phase's vectors into commitment groups of at most 2^max_bits cells (by descending
    size, so every vector stays aligned inside its group).  Both parties compute it from the public circuit."""
    max_bits = GROUP_BITS if max_bits is None else max_bits
    cap = (1 << max_bits) - (1 << FILLER_BITS)
    groups, cur, used = [], [], 0
    for v in sorted(vecs, key=lambda v: (-v.nbits, v.id)):
        if cur and used + v.N > cap:
            groups.append(cur); cur, used = [], 0
        cur.append(v); used += v.N
    if cur:
        groups.append(cur)
    return groups


def _free(B):
    if B.name == "cupy":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


class Tree:
    """BLAKE2b Merkle tree; leaf j = cosets {j + s n/8} of each codeword + 256-bit salt.
    Leaf hashes are computed in chunks on the device; codewords are then kept on the host for openings."""
    CHUNK = 1 << 19

    def __init__(self, B, codes, salt=None):
        xp = B.xp
        n = codes[0].shape[1]; m = n >> ARITY_BITS
        self.m = m
        self.salt = salt if salt is not None else urandom_u64(m * SALT_WORDS).reshape(m, SALT_WORDS)
        hs = []
        for a0 in range(0, m, self.CHUNK):
            a1 = min(m, a0 + self.CHUNK)
            parts = []
            for C in codes:
                blk = xp.stack([B.asarray(C[:, a0 + s * m:a1 + s * m]) for s in range(1 << ARITY_BITS)], axis=2)   # (3, w, 8)
                parts.append(blk.transpose(1, 0, 2).reshape(a1 - a0, -1))
            parts.append(B.asarray(self.salt[a0:a1]))
            hs.append(B.tonumpy(B.blake_rows(xp.ascontiguousarray(xp.concatenate(parts, axis=1)))))
            del parts
        lv = [np.concatenate(hs, axis=0) if len(hs) > 1 else hs[0]]
        top = B.asarray(lv[0])
        while top.shape[0] > 1:
            top = B.blake_rows(top.reshape(-1, 8)); lv.append(B.tonumpy(top))
        self.levels = lv
        self.root = self.levels[-1][0].tobytes()
        self.codes = [C if isinstance(C, np.ndarray) else B.tonumpy(C) for C in codes]
        self.B = B

    def open_many(self, js):
        js = np.asarray(js, dtype=np.int64)
        m = self.m
        parts = []
        for C in self.codes:
            blk = np.stack([C[:, js + s * m] for s in range(1 << ARITY_BITS)], axis=2)
            parts.append(blk.transpose(1, 0, 2).reshape(len(js), -1))
        parts.append(self.salt[js])
        data = np.concatenate(parts, axis=1)
        out = []
        for k, j in enumerate(js):
            path, q = [], int(j)
            for lv in self.levels[:-1]:
                path.append(lv[q ^ 1].tobytes()); q >>= 1
            out.append((data[k], path))
        return out


def _check_path(root, leaf_u64, j, path, m):
    import hashlib
    if not isinstance(root, (bytes, bytearray)) or len(root) != 32:
        return False
    if not isinstance(j, int) or isinstance(j, bool) or j < 0 or j >= m:
        return False
    depth = m.bit_length() - 1
    if m <= 0 or (1 << depth) != m or not isinstance(path, (list, tuple)) or len(path) != depth:
        return False
    h = hashlib.blake2b(np.ascontiguousarray(leaf_u64, dtype=np.uint64).tobytes(), digest_size=32).digest()
    q = j
    for s in path:
        if not isinstance(s, (bytes, bytearray)) or len(s) != 32:
            return False
        h = hashlib.blake2b(bytes(s) + h if q & 1 else h + bytes(s), digest_size=32).digest(); q >>= 1
    return h == bytes(root)


def _leaf_val(leaf, poly_i, slot):
    st = 1 << ARITY_BITS
    k = poly_i * (3 * st) + slot
    return (int(leaf[k]), int(leaf[k + st]), int(leaf[k + 2 * st]))


# ============================================================================ ZK-Basefold (generalized weights, T4 mask)
def _root_of_unity(n): return pow(7, (P - 1) // n, P)


_INV2X = {}


def _fold_codeword(B, C, alpha, inplace=False):
    xp = B.xp
    n = C.shape[1]; h = n // 2
    if hasattr(B, "fold_code"):
        # device: inv2x(n)[i] = inv2x(n')[i n'/n] for any cached n' >= n (w_n = w_n'^(n'/n)): keep only the largest table
        big = [nn for (bn, nn) in _INV2X if bn == B.name and nn >= n]
        if big:
            nn = max(big)
            return B.fold_code(C, alpha, _INV2X[(B.name, nn)], INV2, nn // n, inplace)
        _INV2X[(B.name, n)] = B.inv2x(n)
        return B.fold_code(C, alpha, _INV2X[(B.name, n)], INV2, 1, inplace)
    key = (B.name, n)
    if key not in _INV2X:
        if hasattr(B, "inv2x"):
            _INV2X[key] = B.inv2x(n)                     # computed on the device (no host pow_table / transfer)
        else:
            from .field import pow_table
            w = _root_of_unity(n); wi = pow(w, P - 2, P)
            _INV2X[key] = B.asarray(CPU.mul(pow_table(wi, h), np.uint64(INV2)))
    inv2x = _INV2X[key]
    if hasattr(B, "fold_code"):
        return B.fold_code(C, alpha, inv2x, INV2)
    a, b = C[:, :h], C[:, h:]
    pe = B.escale(B.eadd(a, b), (INV2, 0, 0))
    d = B.esub(a, b)
    po = xp.stack([B.mul(d[0], inv2x), B.mul(d[1], inv2x), B.mul(d[2], inv2x)])
    return B.eadd(pe, B.escale(B.esub(po, pe), alpha))


def _tree_after(k, n, size):
    return k % ARITY_BITS == 0 and k < n and size >= 16


def basefold_prove(pc, W, claim, tr):
    """Prove <Big, W> = claim for phase commitment pc and weight table W (2, N) on the prover's device.

    VRAM: hw is a fused dot product, F = Big + eps h_bf is formed in place, the round sums are four fused dot products
    over strided halves (no temporaries), and the codeword is never formed at full length: folding commutes with the
    encoding (fold(code(F), a) = code(fold(F, a)) exactly, see _fold_codeword), so the first codeword built is
    code(F) after the rounds that precede the first folded tree (rate 1/4 of N/8 for the usual arity 8)."""
    B = pc.B
    n = pc.nb
    pc.ensure_tree()
    if pc.N >= (1 << 27) and hasattr(B, "clear_caches"):
        B.clear_caches(); _free(B)
    if pc._h is None and pc._rng_key is not None:
        hdev = B.zeros_e(pc.N)
        for l in (0, 1, 2):
            hdev[l] = B.asarray(pc.h_limb(l))
    else:
        hdev = B.asarray(pc.h)
    hw = B.fetch_pairs([B.edot_dev(hdev, W)])[0] if hasattr(B, "edot_dev") else B.esum(B.emul(hdev, W))
    tr.absorb(b"bf", pc.root, canon(claim), canon(hw))
    eps = tr.ext()
    bd = _dev_big(B, pc)
    if bd is not None:
        F = bd.copy()
        B.eaxpy_(F, hdev, eps)
        del hdev, bd
    else:
        F = B.escale(hdev, eps)
        del hdev
        _free(B)
        for l in (0, 1, 2):
            bl = B.asarray(pc.limb(l))
            F[l] = B.add(F[l], bl)
            del bl
    _dev_big_drop(pc); _free(B)
    Wt = W
    msgs, roots, trees = [], [], []
    final_round = None; final_code = None
    C = None
    for i in range(n):
        f0, f1, e0, e1 = F[:, 0::2], F[:, 1::2], Wt[:, 0::2], Wt[:, 1::2]
        if hasattr(B, "edot_dev"):
            # <2 f1 - f0, 2 e1 - e0> = 4 <f1,e1> - 2 <f1,e0> - 2 <f0,e1> + <f0,e0>
            d00, d11, d10, d01 = B.fetch_pairs([B.edot_dev(f0, e0), B.edot_dev(f1, e1), B.edot_dev(f1, e0),
                                                B.edot_dev(f0, e1)])
            v2 = hadd(hsub(hmul((4, 0, 0), d11), hmul((2, 0, 0), hadd(d10, d01))), d00)
            vals = [d00, d11, v2]
        else:
            df, de = B.esub(f1, f0), B.esub(e1, e0)
            vals = [B.esum(B.emul(f0, e0)), B.esum(B.emul(f1, e1)),
                    B.esum(B.emul(B.eadd(f1, df), B.eadd(e1, de)))]
            del df, de
        tr.absorb(canon(vals)); msgs.append(vals)
        a = tr.ext()
        del f0, f1, e0, e1
        F = B.fold(F, a); Wt = B.fold(Wt, a)
        if final_round is not None:
            continue
        if C is not None:
            C = _fold_codeword(B, C, a, inplace=True)    # trees / final code copy C to the host before the next fold
        size = RATE * F.shape[1]
        need_tree = _tree_after(i + 1, n, size)
        if C is None and (need_tree or (i + 1) % ARITY_BITS == 0 or i + 1 == n):
            C = B.encode(F, RATE)                         # == the full codeword folded i + 1 times
            _free(B)
        if need_tree:
            t = Tree(B, [C]); trees.append(t); roots.append(t.root); tr.absorb(t.root)
        elif (i + 1) % ARITY_BITS == 0 or i + 1 == n:
            final_round = i + 1
            fc = B.tonumpy(C); final_code = fc
            tr.absorb(fc)
            C = None
    c = (int(B.tonumpy(F[0, :1])[0]), int(B.tonumpy(F[1, :1])[0]), int(B.tonumpy(F[2, :1])[0]))
    tr.absorb(canon(c))
    m0 = pc.tree.m
    Q = tr.indices(T_QUERIES, m0)
    js = sorted(set(int(q) for q in Q))
    q0 = [(j, d, p) for j, (d, p) in zip(js, pc.tree.open_many(js))]
    pos = list(Q); per_tree = []
    for t in trees:
        mt = t.m
        pos = [p_ % mt for p_ in pos]
        per_tree.append((pos, t.open_many(pos)))
    qs = [[(jj[qi], op[qi][0], op[qi][1]) for jj, op in per_tree] for qi in range(len(Q))]
    pc.release()
    return {"hw": list(hw), "msgs": msgs, "roots": roots, "final_round": final_round, "final_code": final_code,
            "c": list(c), "q0": q0, "qs": qs}


def _fold_coset(vals, j, n, alphas):
    from .field import native, ptr
    lib = native()
    if (lib is not None and len(vals) == (1 << ARITY_BITS) and all(len(v) == 3 for v in vals)
            and len(alphas) <= ARITY_BITS and all(len(a) == 3 for a in alphas)
            and j >= 0 and n >= (1 << ARITY_BITS) and (n & (n - 1)) == 0):
        buf = np.array([int(x) % P for v in vals for x in v], dtype=np.uint64)
        al = np.array([int(x) % P for a in alphas for x in a], dtype=np.uint64) if alphas else np.zeros(3, dtype=np.uint64)
        lib.fold_coset(ptr(buf), ARITY_BITS, j, n, _root_of_unity(n), ptr(al), len(alphas))
        k = len(vals) >> len(alphas)
        return [(int(buf[3 * s_]), int(buf[3 * s_ + 1]), int(buf[3 * s_ + 2])) for s_ in range(k)]
    cur, size, step = list(vals), n, n >> ARITY_BITS
    for a in alphas:
        k = len(cur) // 2; w = _root_of_unity(size); nxt = []
        for s in range(k):
            x = pow(w, j + s * step, P)
            A, Bv = cur[s], cur[s + k]
            pe = hmul(hadd(A, Bv), (INV2, 0, 0))
            po = hmul(hsub(A, Bv), (pow(2 * x % P, P - 2, P), 0, 0))
            nxt.append(hadd(pe, hmul(a, hsub(po, pe))))
        cur, size = nxt, size // 2
    return cur


def _fold_full(code, alphas, size):
    from .field import native, ptr
    lib = native()
    if (lib is not None and alphas and len(code) == size and all(len(v) == 3 for v in code)
            and all(len(a) == 3 for a in alphas) and size >= (1 << len(alphas)) and (size & (size - 1)) == 0):
        C = np.array([[int(a) % P for a, _, _ in code], [int(b) % P for _, b, _ in code],
                      [int(c) % P for _, _, c in code]], dtype=np.uint64)
        for a in alphas:
            out = np.empty((3, size // 2), dtype=np.uint64)
            lib.fold_full(ptr(C), size, _root_of_unity(size), int(a[0]) % P, int(a[1]) % P, int(a[2]) % P, ptr(out))
            C, size = out, size // 2
        return [(int(x), int(y), int(z)) for x, y, z in zip(C[0], C[1], C[2])]
    for a in alphas:
        h = size // 2; w = _root_of_unity(size); nxt = []
        wi = pow(w, P - 2, P); xi = INV2
        for i in range(h):
            A, Bv = code[i], code[i + h]
            pe = hmul(hadd(A, Bv), (INV2, 0, 0)); po = hmul(hsub(A, Bv), (xi, 0, 0))
            nxt.append(hadd(pe, hmul(a, hsub(po, pe))))
            xi = xi * wi % P
        code, size = nxt, h
    return code


def basefold_verify(root, nb, claim, pr, tr, w_mle):
    """Returns True iff the opening is valid; w_mle(alpha) computes the weight table's MLE."""
    n = nb; N = 1 << nb
    if not isinstance(pr["hw"], (list, tuple)) or len(pr["hw"]) != 3:
        raise ProofError("basefold: bad hw")
    hw = tuple(int(x) % P for x in pr["hw"])
    tr.absorb(b"bf", root, canon(claim), canon(list(hw)))
    eps = tr.ext()
    cl = hadd(tuple(claim), hmul(eps, hw))
    if len(pr["msgs"]) != n:
        raise ProofError("basefold: wrong number of rounds")
    alphas, nroots, final_round = [], 0, None
    size = RATE * N
    for i, vals in enumerate(pr["msgs"]):
        vals = [tuple(int(t) % P for t in v) for v in vals]
        if len(vals) != 3 or any(len(v) != 3 for v in vals) or hadd(vals[0], vals[1]) != cl:
            raise ProofError("basefold: sumcheck round mismatch")
        tr.absorb(canon(vals))
        a = tr.ext(); alphas.append(a)
        cl = lagrange_eval(vals, a)
        size //= 2
        if final_round is None and _tree_after(i + 1, n, size):
            if nroots >= len(pr["roots"]):
                raise ProofError("basefold: missing root")
            tr.absorb(pr["roots"][nroots]); nroots += 1
        elif final_round is None and ((i + 1) % ARITY_BITS == 0 or i + 1 == n):
            final_round = i + 1
            if pr["final_round"] != final_round:
                raise ProofError("basefold: final round mismatch")
            fcode = np.asarray(pr["final_code"], dtype=np.uint64)
            tr.absorb(fcode)
    if nroots != len(pr["roots"]) or final_round is None:
        raise ProofError("basefold: tree count")
    if not isinstance(pr["c"], (list, tuple)) or len(pr["c"]) != 3:
        raise ProofError("basefold: bad c")
    c = tuple(int(x) % P for x in pr["c"]); tr.absorb(canon(list(c)))
    if hmul(c, w_mle(alphas)) != cl:
        raise ProofError("basefold: final evaluation mismatch")
    fsize = RATE * N >> final_round
    fc_arr = np.asarray(pr["final_code"], dtype=np.uint64)
    if fc_arr.shape != (3, fsize) or np.any(fc_arr >= np.uint64(P)):
        raise ProofError("basefold: bad final codeword shape or range")
    fc = [(int(a), int(b), int(c_)) for a, b, c_ in zip(fc_arr[0], fc_arr[1], fc_arr[2])]
    if any(v != c for v in _fold_full(fc, alphas[final_round:], fsize)):
        raise ProofError("basefold: final codeword not constant")
    n0 = RATE * N; m0 = n0 >> ARITY_BITS
    Q = tr.indices(T_QUERIES, m0)
    q0 = {}
    for j, d, p in pr["q0"]:
        if isinstance(j, bool) or not isinstance(j, int):
            raise ProofError("basefold: non-integer query index")
        d = np.asarray(d, dtype=np.uint64)
        nf = 2 * (3 << ARITY_BITS)
        if d.ndim != 1 or len(d) != nf + SALT_WORDS or np.any(d[:nf] >= np.uint64(P)) or not _check_path(root, d, j, p, m0):
            raise ProofError("basefold: bad leaf / path in round-0 tree")
        q0[j] = d
    if len(pr["qs"]) != len(Q):
        raise ProofError("basefold: query count")
    G = len(pr["roots"])
    for qi, j in enumerate(Q):
        if j not in q0:
            raise ProofError("basefold: query not opened")
        d = q0[j]
        cur = [hadd(_leaf_val(d, 0, s), hmul(eps, _leaf_val(d, 1, s))) for s in range(1 << ARITY_BITS)]
        steps = pr["qs"][qi]
        if len(steps) != G:
            raise ProofError("basefold: query depth")
        size, pos, ai = n0, j, 0
        for g in range(G + 1):
            k = min(ARITY_BITS, final_round - ai) if g == G else ARITY_BITS
            vals = _fold_coset(cur, pos, size, alphas[ai:ai + k])
            step = size >> ARITY_BITS
            ai += k; size >>= k
            if g < G:
                jj, leafd, path = steps[g]
                if isinstance(jj, bool) or not isinstance(jj, int):
                    raise ProofError("basefold: non-integer folded index")
                leafd = np.asarray(leafd, dtype=np.uint64)
                mt = size >> ARITY_BITS
                nf1 = 3 << ARITY_BITS
                if (jj != pos % mt or leafd.ndim != 1 or len(leafd) != nf1 + SALT_WORDS
                        or np.any(leafd[:nf1] >= np.uint64(P)) or not _check_path(pr["roots"][g], leafd, jj, path, mt)):
                    raise ProofError("basefold: bad folded leaf")
                if _leaf_val(leafd, 0, pos // mt) != vals[0]:
                    raise ProofError("basefold: fold inconsistency")
                cur = [_leaf_val(leafd, 0, s) for s in range(1 << ARITY_BITS)]; pos = int(jj)
            else:
                for s, v in enumerate(vals):
                    if fc[(pos + s * step) % fsize] != v:
                        raise ProofError("basefold: final fold inconsistency")
    return True


# ============================================================================ the flat argument
def _bind_late(ex, r_zc):
    for c, f in ex.terms:
        for l, p in f:
            for a in l.atoms():
                if getattr(a, "late", False):
                    a.bind(r_zc)


def _instance_terms(cir, nv, r_zc, lambdas):
    """Batched polynomial for cube size nv (an Ex).  All zerochecks share the factor eq(r_zc, x), so they are first
    combined into one expression and multiplied by eq once; everything is normalised in a single pass."""
    for ex, tag in cir.zeros.get(nv, []):
        _bind_late(ex, r_zc)
    for ex, ev, tag in cir.sums.get(nv, []):
        _bind_late(ex, r_zc)
    full = list(range(nv))
    li = 0
    zs = []
    for ex, tag in cir.zeros.get(nv, []):
        zs.append(ex.scaled(lambdas[li])); li += 1
    parts = []
    if zs:
        zsum = Ex.sum_all(zs, factor=True)
        parts.append(zsum * Ex.atom(EqP(full, r_zc)) if nv else zsum)
    for ex, eq_vars, tag in cir.sums.get(nv, []):
        if eq_vars:
            e = Ex.atom(EqP(list(eq_vars), [r_zc[v] for v in eq_vars]))
            parts.append(ex.scaled(lambdas[li]) * e)
        else:
            parts.append(ex.scaled(lambdas[li]))
        li += 1
    return Ex.sum_all(parts, factor=True)


def _n_constraints(cir, nv):
    return len(cir.zeros.get(nv, [])) + len(cir.sums.get(nv, []))


def _degree(cir, nv):
    d = 0
    for ex, _ in cir.zeros.get(nv, []):
        d = max(d, ex.degree() + 1)
    for ex, ev, _ in cir.sums.get(nv, []):
        d = max(d, ex.degree() + (1 if ev else 0))
    return max(d, 2)


def _finalize_lookups(cir, tr, B):
    """Draw (beta, gamma) per table, create helper vectors H (phase 2), add LogUp constraints."""
    for name in sorted(cir.lookups):
        lk = cir.lookups[name]
        if lk.table is None or not lk.parts:
            raise ProofError(f"lookup {name}: table or parts missing")
        beta, gamma = tr.ext(), tr.ext()
        lk.beta, lk.gamma = beta, gamma
        nvt, tcols = lk.table
        blocks = [("T", nvt, None)] + [("V%d" % i, p[0], p) for i, p in enumerate(lk.parts)]
        order = sorted(range(len(blocks)), key=lambda i: -blocks[i][1])
        off = 0; offs = {}
        for i in order:
            sz = 1 << blocks[i][1]
            off = (off + sz - 1) // sz * sz
            offs[i] = off; off += sz
        nvm = max((off + TB - 1).bit_length(), max(b[1] for b in blocks))
        if off > (1 << nvm) - TB:
            nvm += 1
        H = cir.vec(f"H:{name}", nvm, off, phase=2)
        H.is_ext = True
        lk.H = H
        hv = Ex.atom(View(H))

        def emb(ex, nv_from, o):
            remap = list(range(nv_from)) + [None] * 0
            fixed = tuple((v, (o >> v) & 1) for v in range(nv_from, nvm))
            return ex.embed(remap, fixed)

        def comp(cols):
            acc = Ex.const(0); co = ONE
            for c in cols:
                acc = acc + c * Ex.const(co); co = hmul(co, beta)
            return acc

        # table part:  H (gamma - t) + m = 0
        o = offs[0]
        ind = Ex.atom(SelLt(list(range(nvt)), 1 << nvt, fixed=tuple((v, (o >> v) & 1) for v in range(nvt, nvm))))
        t_ex = _embconst(emb(comp(tcols), nvt, o), ind)
        m_ex = emb(Ex.atom(View(lk.mult)) * Ex.atom(SelLt(list(range(nvt)), lk.mult.content)), nvt, o)
        cir.sumzero(nvm, hv * (ind * Ex.const(gamma) - t_ex) + m_ex, eq_vars=tuple(range(nvm)), tag=f"logup-table:{name}")
        for i in range(1, len(blocks)):
            nvp, cols, sel, vals, sel_vals = blocks[i][2]
            o = offs[i]
            ind = Ex.atom(SelLt(list(range(nvp)), 1 << nvp, fixed=tuple((v, (o >> v) & 1) for v in range(nvp, nvm))))
            a_ex = emb(comp(cols), nvp, o)
            s_ex = emb(sel, nvp, o) if sel is not None else ind
            cir.sumzero(nvm, hv * (ind * Ex.const(gamma) - _embconst(a_ex, ind)) - s_ex, eq_vars=tuple(range(nvm)), tag=f"logup-value:{name}:{i}")
        cir.sumzero(nvm, hv * Ex.atom(SelLt(list(range(nvm)), off)), eq_vars=(), tag=f"logup-sum:{name}")
        lk.offs = offs; lk.nvm = nvm; lk.blocks = blocks
        if cir.prove:
            _logup_witness(cir, lk, B)


def _embconst(ex, ind):
    """Replace the constant term c of an affine Ex by c * ind (keeps the factor block-local)."""
    lin = ex.as_lin()
    if lin is None or lin.const == ZERO:
        return ex
    c = lin.const
    return Ex.lin(Lin(lin.items, ZERO)) + ind * Ex.const(c)


def _comp_vals(cols_vals, beta):
    """Host-side compression of integer tuple columns: returns (3, n) uint64 on CPU."""
    n = len(cols_vals[0])
    acc = np.zeros((3, n), dtype=np.uint64); co = ONE
    for v in cols_vals:
        fv = CPU.fromi(np.asarray(v))
        acc = CPU.eadd(acc, CPU.escale(CPU.ext(fv), co)); co = hmul(co, beta)
    return acc


def _logup_mult(cir, lk):
    nvt, tcols = lk.table
    if lk.table_vals is None and getattr(lk, "table_rows", None) is not None:   # already compacted (idempotent)
        T = lk.table_rows
    else:
        T = np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.int64) for c in lk.table_vals], axis=1))
    nt, ncol = T.shape
    lk.table_rows = T
    mult = np.zeros(nt, dtype=np.int64)
    # Fast direct-indexing path for single-column dense integer tables [lo .. lo + nt - 1] (e.g. u16c, q16, exp, silu, gumbel)
    dense_lo = int(T[0, 0]) if (ncol == 1 and nt >= 1 and T[-1, 0] - T[0, 0] == nt - 1
                                and np.array_equal(T[:, 0], np.arange(T[0, 0], T[0, 0] + nt, dtype=np.int64))) else None
    if dense_lo is None:
        vt = T.view(np.dtype((np.void, ncol * 8))).ravel()
        order = np.argsort(vt, kind="stable"); vs = vt[order]
    compact = []
    for (nvp, cols, sel, vals, sel_vals) in lk.parts:
        if callable(vals):                    # lazily generated part: (rows matrix, sel values) regenerated on demand
            V, sel_vals = vals()
        else:
            V = vals
        n = V.shape[0] if isinstance(V, np.ndarray) and V.ndim == 2 else len(V[0])
        if sel is None and n != (1 << nvp):
            raise ProofError(f"lookup {lk.name}: unselected part must fill its cube")
        sv = np.ones(n, dtype=np.int64) if sel_vals is None else np.asarray(sel_vals, dtype=np.int64)
        V = np.ascontiguousarray(V, dtype=np.int64) if isinstance(V, np.ndarray) and V.ndim == 2 else \
            np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.int64) for c in V], axis=1))
        if dense_lo is not None:
            idx = V[:, 0] - dense_lo
            hit = (idx >= 0) & (idx < nt)
            bad = np.nonzero((sv != 0) & ~hit)[0]
            if len(bad):
                j = int(bad[0])
                msg = f"lookup {lk.name}: value {tuple(int(x) for x in V[j])} not in table (row {j}, {len(bad)} rows)"
                if not FORCE:
                    raise ProofError(msg)
                VIOLATIONS.append(msg)
            good = (sv != 0) & hit
            mult += np.bincount(idx[good], weights=sv[good], minlength=nt).astype(np.int64)
        else:
            vv = V.view(np.dtype((np.void, V.shape[1] * 8))).ravel()
            pos = np.searchsorted(vs, vv)
            pos = np.minimum(pos, nt - 1)
            hit = vs[pos] == vv
            bad = np.nonzero((sv != 0) & ~hit)[0]
            if len(bad):
                j = int(bad[0])
                msg = f"lookup {lk.name}: value {tuple(int(x) for x in V[j])} not in table (row {j}, {len(bad)} rows)"
                if not FORCE:
                    raise ProofError(msg)
                VIOLATIONS.append(msg)
            good = (sv != 0) & hit
            np.add.at(mult, order[pos[good]], sv[good])
        compact.append((nvp, cols, sel, [compact_vals(V[:, j]) for j in range(V.shape[1])] if not callable(vals) else vals,
                        None if sel_vals is None else compact_vals(sv)))
        del V
    # keep only compact row matrices (the per-column value lists, often Python-object arrays, are released)
    lk.parts = compact; lk.table_vals = None
    cir.set(lk.mult, mult)
    lk.mult_vals = mult


def _part_rows(part):
    """(int64 row matrix V, sv) of a lookup part: compact columns stacked, lazy parts regenerated."""
    nvp, cols, sel, V, sv = part
    if callable(V):
        V, sv = V()
    V = np.ascontiguousarray(V, dtype=np.int64) if isinstance(V, np.ndarray) and V.ndim == 2 else \
        np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.int64) for c in V], axis=1))
    n = V.shape[0]
    sv = np.ones(n, dtype=np.int64) if sv is None else np.asarray(sv, dtype=np.int64)
    return V, sv


_PROOF_SPILL = [None]


def _proof_spill_dir():
    if _PROOF_SPILL[0] is None:
        _PROOF_SPILL[0] = _spill_dir()
    return _PROOF_SPILL[0]


def _logup_slice(lk, B, a0, a1, dev=False):
    """Evaluate rows [a0, a1) of lk.H on demand from compact table/part witnesses (no disk spill)."""
    n_out = a1 - a0
    use_gpu = getattr(B, "name", "") == "cupy"
    out = B.zeros_e(n_out) if (dev and use_gpu) else np.zeros((3, n_out), dtype=np.uint64)
    if n_out <= 0 or lk.table_rows is None:
        return out
    T = lk.table_rows; nt = T.shape[0]; mult = lk.mult_vals
    g0, g1, g2 = np.uint64(lk.gamma[0]), np.uint64(lk.gamma[1]), np.uint64(lk.gamma[2])
    STEP = 1 << 22

    def eval_seg(cols_seg, num_seg, dst_off):
        m = len(cols_seg[0])
        if use_gpu:
            xp = B.xp
            for c0 in range(0, m, STEP):
                c1 = min(m, c0 + STEP); cm = c1 - c0
                acc = B.zeros_e(cm); co = ONE
                for col in cols_seg:
                    B.eaxpy_(acc, B.ext(B.fromi(col[c0:c1])), co); co = hmul(co, lk.beta)
                gam = xp.stack([xp.full(cm, g0, dtype=xp.uint64), xp.full(cm, g1, dtype=xp.uint64),
                                xp.full(cm, g2, dtype=xp.uint64)])
                inv = B.einv(B.esub(gam, acc))
                del acc, gam
                if num_seg is not None:
                    inv = B.emul(inv, B.ext(B.fromi(num_seg[c0:c1])))
                out[:, dst_off + c0:dst_off + c1] = inv if dev else B.tonumpy(inv)
                del inv
        else:
            gam = np.stack([np.full(m, g0, dtype=np.uint64), np.full(m, g1, dtype=np.uint64),
                            np.full(m, g2, dtype=np.uint64)])
            inv = CPU.einv(CPU.esub(gam, _comp_vals(cols_seg, lk.beta)))
            if num_seg is not None:
                inv = CPU.emul(inv, CPU.ext(CPU.fromi(num_seg)))
            out[:, dst_off:dst_off + m] = inv

    o0 = lk.offs[0]
    if o0 < a1 and o0 + nt > a0:
        s0 = max(a0, o0) - o0; s1 = min(a1, o0 + nt) - o0
        eval_seg([T[s0:s1, j] for j in range(T.shape[1])], -mult[s0:s1], o0 + s0 - a0)
    for i, part in enumerate(lk.parts, start=1):
        nvp, cols, sel, V, sv = part
        o = lk.offs[i]
        if callable(V):
            if o >= a1 or o + (1 << nvp) <= a0:
                continue
            Vm, sva = _part_rows(part)
            n = Vm.shape[0]
            if o >= a1 or o + n <= a0:
                continue
            s0 = max(a0, o) - o; s1 = min(a1, o + n) - o
            eval_seg([Vm[s0:s1, j] for j in range(Vm.shape[1])], sva[s0:s1], o + s0 - a0)
        else:
            n = len(V[0])
            if o >= a1 or o + n <= a0:
                continue
            s0 = max(a0, o) - o; s1 = min(a1, o + n) - o
            eval_seg([c[s0:s1] for c in V], None if sv is None else sv[s0:s1], o + s0 - a0)
    if use_gpu and not dev:
        _free(B)
    return out


def _logup_witness(cir, lk, B):
    if (_is_ooc() and lk.H.N >= (1 << 16)) or lk.H.N >= (1 << 20):
        lk.H.ext_gen = lambda a0, a1, dev=False, _lk=lk, _B=B: _logup_slice(_lk, _B, a0, a1, dev)
        def _rel(_lk=lk):
            _lk.parts = [(p[0], p[1], p[2], None, None) for p in _lk.parts]
            _lk.table_rows = None; _lk.mult_vals = None
        lk.H.on_release = _rel
        return
    _logup_witness_vals(cir, lk, B)


def _logup_witness_vals(cir, lk, B):
    nvt, tcols = lk.table
    T = lk.table_rows
    nt = T.shape[0]
    mult = lk.mult_vals
    if getattr(B, "name", "") == "cupy":
        xp = B.xp
        Hdev = B.zeros_e(lk.H.N)
        g0, g1, g2 = np.uint64(lk.gamma[0]), np.uint64(lk.gamma[1]), np.uint64(lk.gamma[2])
        gam_dev = lambda m: xp.stack([xp.full(m, g0, dtype=xp.uint64), xp.full(m, g1, dtype=xp.uint64),
                                      xp.full(m, g2, dtype=xp.uint64)])
        def comp_dev(M):
            acc = B.zeros_e(M.shape[0])
            co = ONE
            for j in range(M.shape[1]):
                B.eaxpy_(acc, B.ext(B.fromi(M[:, j])), co)
                co = hmul(co, lk.beta)
            return acc
        t = comp_dev(T)
        den = B.esub(gam_dev(nt), t)
        del t
        o = lk.offs[0]
        Hdev[:, o:o + nt] = B.emul(B.einv(den), B.ext(B.fromi(-mult)))
        del den
        for i, part in enumerate(lk.parts, start=1):
            V, sv = _part_rows(part)
            n = V.shape[0]; o = lk.offs[i]
            a = comp_dev(V)
            den = B.esub(gam_dev(n), a)
            del a
            Hdev[:, o:o + n] = B.emul(B.einv(den), B.ext(B.fromi(sv)))
            del den
        lk.H.vals = B.tonumpy(Hdev)
        del Hdev
        _free(B)
        lk.parts = [(p[0], p[1], p[2], None, None) for p in lk.parts]; lk.table_rows = None
        return
    H = np.zeros((3, lk.H.N), dtype=np.uint64)
    g = lk.gamma
    gam_np = lambda m: np.stack([np.full(m, g[0], dtype=np.uint64), np.full(m, g[1], dtype=np.uint64),
                                 np.full(m, g[2], dtype=np.uint64)])
    # table part
    t = _comp_vals([T[:, j] for j in range(T.shape[1])], lk.beta)
    den = CPU.esub(gam_np(nt), t)
    h = CPU.emul(CPU.einv(den), CPU.ext(CPU.fromi(-mult)))
    o = lk.offs[0]; H[:, o:o + nt] = h
    for i, part in enumerate(lk.parts, start=1):
        V, sv = _part_rows(part)
        n = V.shape[0]; o = lk.offs[i]
        a = _comp_vals([V[:, j] for j in range(V.shape[1])], lk.beta)
        den = CPU.esub(gam_np(n), a)
        del a
        H[:, o:o + n] = CPU.emul(CPU.einv(den), CPU.ext(CPU.fromi(sv)))
    lk.H.vals = H
    lk.parts = [(p[0], p[1], p[2], None, None) for p in lk.parts]; lk.table_rows = None


def _masks(cir, sizes, degs):
    """Libra mask vectors (phase 2), one per cube size: coefficients a[i, k] in E = F_{p^3}, i < nv, k <= deg.

    The coefficients MUST be uniform in E (not in F_p): the round messages M_i(X) = H_i(X) + rho g_i-part(X) live in E,
    and with base-field coefficients the u and u^2 components of M_i(X) / rho would be left unmasked."""
    out = {}
    for nv in sizes:
        D = degs[nv]
        cnt = max(1, nv * (D + 1))
        nb = (cnt + TB - 1).bit_length()
        v = cir.vec(f"libra:{nv}", nb, cnt, phase=2)
        v.is_ext = True
        if cir.prove:
            v.vals = np.zeros((3, v.N), dtype=np.uint64)
            v.vals[:, :cnt] = CPU.rand_e(cnt)
        out[nv] = v
    return out


def _mask_coef(v, cnt):
    """E coefficients (list of (a0, a1, a2)) of a Libra mask vector (its values; the committed copy is identical)."""
    src = v.vals if v.vals is not None else v.full
    if src is None and v.grp is not None:
        src = v.grp.vec_range_e(v, 0, cnt)
    elif isinstance(src, _Spilled):
        src = src.read(0, cnt)
    return [(int(src[0, k]), int(src[1, k]), int(src[2, k])) for k in range(cnt)]


def _mask_sum(coef, nv, D):
    """sum over the cube of g(x) = sum_i sum_k a[i,k] x_i^k  (an element of E)."""
    tot = ZERO
    h = (pow(2, nv - 1, P), 0, 0)
    for i in range(nv):
        gi0 = coef[i * (D + 1)]
        gi1 = ZERO
        for k in range(D + 1):
            gi1 = hadd(gi1, coef[i * (D + 1) + k])
        tot = hadd(tot, hmul(hadd(gi0, gi1), h))
    return tot


def _gi(coef, i, D, x):
    acc = ZERO; xp_ = ONE
    for k in range(D + 1):
        acc = hadd(acc, hmul(coef[i * (D + 1) + k], xp_)); xp_ = hmul(xp_, x)
    return acc


def _gi_vals(coef, i, D):
    return [_gi(coef, i, D, (X, 0, 0)) for X in range(D + 1)]


class _Claims:
    """Evaluation claims on committed vectors: ("pt", vec_id, z) | ("sh", vec_id, r, delta) | ("pw", vec_id, r, D)."""

    def __init__(self):
        self.items = []; self.vals = []

    def add(self, spec, val):
        self.items.append(spec); self.vals.append(val)


def _atom_value_verifier(a, r, claimvals):
    if isinstance(a, View):
        ind = a.ind_mle(r)
        if a.public:
            if a.shift:
                z = a.local_point(r)
                w = shift_table(CPU, z, a.shift, a.vec.nbits)
                return hmul(ind, _dot_base(a.vec.vals, w))
            return hmul(ind, a.vec.mle(a.local_point(r)))
        return hmul(ind, claimvals[a.kid])
    return a.mle(r)


def _eval_terms(total, r, claimvals):
    """total(r) given the claimed view values; atom and Lin values are computed once each (cached by id / key)."""
    acache = {}; lcache = {}

    def lin_value(l):
        k = l.key
        v = lcache.get(k)
        if v is None:
            a0, a1, a2 = l.const
            for kid, (a, c) in l.items.items():
                av = acache.get(kid)
                if av is None:
                    av = acache[kid] = _atom_value_verifier(a, r, claimvals)
                a0 += c[0] * av[0] + 2 * (c[1] * av[2] + c[2] * av[1])
                a1 += c[0] * av[1] + c[1] * av[0] + 2 * c[2] * av[2]
                a2 += c[0] * av[2] + c[1] * av[1] + c[2] * av[0]
            v = lcache[k] = (a0 % P, a1 % P, a2 % P)
        return v
    acc = ZERO
    for c, f in total.terms:
        t = c
        for l, p in f:
            t = hmul(t, hpow(lin_value(l), p))
        acc = hadd(acc, t)
    return acc


def _view_atoms(total):
    seen = {}
    for c, f in total.terms:
        for l, p in f:
            for kid, (a, _) in l.items.items():
                if kid not in seen and isinstance(a, View) and not a.public:
                    seen[kid] = a
    return sorted(seen.values(), key=lambda a: repr(a.key))


# ---------------------------------------------------------------------------- prover side tables
def _dev_rows(B, v, a0, a1):
    """device (3, a1 - a0) E values of rows [a0, a1) of a committed vector: compact witness uploaded and converted on
    the device, T1 tail rows (and E-valued vectors) from the group store."""
    pc = v.grp
    n = a1 - a0
    t0 = v.N - v.tail
    if pc._big is not None and B.name == "cupy":     # in-RAM group: three contiguous row copies of the assembled Big
        o = pc.offset[v.id]
        out = B.zeros_e(n)
        for l in (0, 1, 2):
            out[l] = B.asarray(pc._big[l, o + a0:o + a1])
        return out
    if B.name == "cupy" and v.is_ext and v.vals is None and v.ext_gen is not None:
        out = B.zeros_e(n)
        cend = min(v.content, t0)
        if a0 < cend:
            c1 = min(a1, cend)
            out[:, :c1 - a0] = v.ext_gen(a0, c1, True)
        if a1 > t0:
            s0 = max(a0, t0); ro = pc.toff[v.id]
            out[:, s0 - a0:] = B.asarray(pc._rnd_cols(ro + s0 - t0, ro + a1 - t0))
        return out
    if B.name != "cupy" or v.is_ext:
        return B.asarray(pc.vec_range_e(v, a0, a1))
    out = B.zeros_e(n)
    if a0 < t0:
        w = wit(v)
        c1 = min(a1, len(w), t0)
        if c1 > a0:
            seg = w[a0:c1]
            if seg.dtype == np.uint64:
                out[0, :c1 - a0] = B.asarray(seg)
            else:
                out[0, :c1 - a0] = B.fromi(B.asarray(seg, dtype=seg.dtype))
    if a1 > t0:
        s0 = max(a0, t0); ro = pc.toff[v.id]
        r = pc._rnd_cols(ro + s0 - t0, ro + a1 - t0)
        out[:, s0 - a0:] = B.asarray(r)
    return out


def _vec_e(B, v):
    if v.full is not None:                       # check_witness: explicit (2, N) values
        return B.asarray(v.full)
    if v.grp is not None and v.grp.built:
        d = _dev_big(B, v.grp)
        if d is not None:
            o = v.grp.offset[v.id]
            return d[:, o:o + v.N]
        return _dev_rows(B, v, 0, v.N)
    if v.is_ext:
        if v.vals is None and v.ext_gen is not None:
            cend = min(v.content, v.N - v.tail)
            if B.name == "cupy":
                full = B.zeros_e(v.N)
                if cend > 0:
                    full[:, :cend] = v.ext_gen(0, cend, True)
                return full
            full = np.zeros((3, v.N), dtype=np.uint64)
            if cend > 0:
                full[:, :cend] = v.ext_gen(0, cend, False)
            return B.asarray(full)
        arr = v.vals if not isinstance(v.vals, _Spilled) else v.vals.load()
        if arr.shape[1] < v.N:
            full = np.zeros((3, v.N), dtype=np.uint64)
            full[:, :arr.shape[1]] = arr
            arr = full
        return B.asarray(arr)
    full = np.zeros(v.N, dtype=np.uint64)
    w = wit(v)
    full[:len(w)] = field_u64(w)
    return B.ext(B.asarray(full))


def _lin_block(l, nv):
    if l.const != ZERO or not l.items:
        return 0, nv
    offs = [a.block(nv) for a in l.atoms()]
    m = max(mm for _, mm in offs)
    while len({o >> m for o, _ in offs}) > 1:
        m += 1
    return (offs[0][0] >> m) << m, m


_VCACHE_USED = [0]


def _vcache_put(vcache, key, arr):
    nb = getattr(arr, "nbytes", 0)
    if key in vcache:
        _VCACHE_USED[0] -= getattr(vcache.pop(key), "nbytes", 0)
    while vcache and _VCACHE_USED[0] + nb > VCACHE_BYTES:
        _, old = next(iter(vcache.items()))
        vcache.pop(next(iter(vcache)))
        _VCACHE_USED[0] -= getattr(old, "nbytes", 0)
    vcache[key] = arr
    _VCACHE_USED[0] += nb
    return arr


def _vcache_clear(vcache):
    vcache.clear()
    _VCACHE_USED[0] = 0


def _atom_src(B, a, vcache):
    if a.public:
        hit = vcache.get(a.vec.name)
        if hit is not None:
            return hit
        return _vcache_put(vcache, a.vec.name, B.ext(B.asarray(a.vec.vals)))
    hit = vcache.get(a.vec.id)
    if hit is not None:
        return hit
    return _vcache_put(vcache, a.vec.id, _vec_e(B, a.vec))


def _materialize(B, total, nv, vcache):
    """key -> [off, arr]: Lin factor tables restricted to the aligned block outside which they vanish.  Each atom is
    evaluated only on its OWN block and accumulated into its slice (a merged Lin over many disjoint blocks costs the
    sum of the block sizes, not #atoms x block hull)."""
    tables = {}
    for c, f in total.terms:
        for l, p in f:
            if l.key in tables:
                continue
            off, m = _lin_block(l, nv)
            items = list(l.items.values())
            if len(items) == 1 and l.const == ZERO:
                a, co = items[0]
                t = a.table(B, nv, _atom_src(B, a, vcache), (off, m)) if isinstance(a, View) else a.table(B, nv, (off, m))
                tables[l.key] = [off, B.escale(t, co) if co != ONE else t]
                continue
            acc = B.zeros_e(1 << m)
            for a, co in items:
                ao, am = a.block(nv) if (l.const == ZERO) else (off, m)
                if am > m or (ao >> m) != (off >> m):
                    ao, am = off, m
                t = a.table(B, nv, _atom_src(B, a, vcache), (ao, am)) if isinstance(a, View) else a.table(B, nv, (ao, am))
                B.eaxpy_(acc[:, ao - off:ao - off + (1 << am)], t, co)
            if l.const != ZERO:
                acc = B.eaddc(acc, l.const)
            tables[l.key] = [off, acc]
    return tables


def _expand2(B, ent):
    """make a block cover at least one aligned pair."""
    off, arr = ent
    if arr.shape[1] >= 2:
        return ent
    z = B.zeros_e(2)
    z[:, off & 1] = arr[:, 0]
    return [off & ~1, z]


def _fold_block(B, ent, a):
    off, arr = ent
    if arr.shape[1] >= 2:
        return [off >> 1, B.fold(arr, a)]
    b = off & 1
    return [off >> 1, B.escale(arr, a if b else hsub(ONE, a))]


def _term_support(ents):
    """intersection of aligned blocks (None if disjoint)."""
    lo, hi = 0, None
    for off, arr in ents:
        n = arr.shape[1]
        if hi is None:
            lo, hi = off, off + n
        else:
            lo2, hi2 = max(lo, off), min(hi, off + n)
            if lo2 >= hi2:
                return None
            lo, hi = lo2, hi2
    return lo, hi


def hadd_list(a, b):
    return [hadd(x, y) for x, y in zip(a, b)]


def _mask_round(mask_coef, i, nv, D, pre, rho):
    """rho * (Libra contribution to round i at X = 0..D): sum over the remaining cube of g with x_<i bound (pre)."""
    m = nv - i - 1
    rest = ZERO
    for j in range(i + 1, nv):
        rest = hadd(rest, mask_coef[j * (D + 1)])
        for k in range(D + 1):
            rest = hadd(rest, mask_coef[j * (D + 1) + k])
    gvals = _gi_vals(mask_coef, i, D)
    out = []
    for X in range(D + 1):
        gx = hmul((pow(2, m, P), 0, 0), hadd(pre, gvals[X]))
        if m >= 1:
            gx = hadd(gx, hmul((pow(2, m - 1, P), 0, 0), rest))
        out.append(hmul(rho, gx))
    return out


def _sc_terms(total):
    terms = [(c, [(l.key, p) for l, p in f]) for c, f in total.terms]
    return terms


def _sc_fused(B, D, terms):
    return hasattr(B, "sc_round") and D <= 15 and all(len(f) <= 8 and all(p <= 15 for _, p in f) for _, f in terms)


def _sc_round_vals(B, terms, tables, D, fused, h, with_const=True):
    """Round values (X = 0..D) of sum over the pairs of the given (block-sparse) tables; terms with a factor missing
    from `tables` contribute nothing (streaming chunks hold only the tables meeting the chunk)."""
    vals = [ZERO] * (D + 1)
    jobs = []
    for c, f in terms:
        if not f:
            if with_const:
                for X in range(D + 1):
                    vals[X] = hadd(vals[X], hmul(c, (h % P, 0, 0)))
            continue
        ents = []
        for k, _ in f:
            e = tables.get(k)
            if e is None:
                break
            ents.append(e)
        if len(ents) != len(f):
            continue
        sup = _term_support(ents)
        if sup is None:
            continue
        lo, hi = sup
        if fused:
            jobs.append((c, [(tables[k][1], lo - tables[k][0], p) for k, p in f], (hi - lo) // 2))
            continue
        los, dds = [], []
        for k, p in f:
            off, arr = tables[k]
            sl = arr[:, lo - off:hi - off]
            l0 = sl[:, 0::2]
            los.append(l0); dds.append(B.esub(sl[:, 1::2], l0))
        cur = list(los)
        for X in range(D + 1):
            prod = None
            for (k, p), t in zip(f, cur):
                t2 = t if p == 1 else B.epow(t, p)
                prod = t2 if prod is None else B.emul(prod, t2)
            vals[X] = hadd(vals[X], hmul(c, B.esum(prod)))
            if X < D:
                cur = [B.eadd(t, d) for t, d in zip(cur, dds)]
    if jobs:
        vals = hadd_list(vals, B.sc_round(jobs, D))
    return vals


def _sumcheck_prove(B, total, nv, D, tables, mask_coef, rho, tr, start=0, r=None, msgs=None, pre=ZERO):
    """Sumcheck of sum_x [total(x) + rho g(x)] = rho G with block-sparse factor tables.  Returns (msgs, point).
    On the GPU every term of a round is evaluated by one fused kernel launch and the whole round is reduced on the
    device with a single host transfer.  start > 0: continue after `start` rounds done by _sumcheck_stream (tables are
    then already folded `start` times)."""
    msgs = [] if msgs is None else msgs
    r = [] if r is None else r
    terms = _sc_terms(total)
    fused = _sc_fused(B, D, terms)
    for i in range(start, nv):
        tables = {k: _expand2(B, e) for k, e in tables.items()}
        h = 1 << (nv - i - 1)
        vals = _sc_round_vals(B, terms, tables, D, fused, h)
        vals = hadd_list(vals, _mask_round(mask_coef, i, nv, D, pre, rho))
        tr.absorb(canon(vals)); msgs.append(vals)
        a = tr.ext(); r.append(a)
        pre = hadd(pre, _gi(mask_coef, i, D, a))
        for k in list(tables.keys()):
            tables[k] = _fold_block(B, tables.pop(k), a)
        if i == start:
            _free(B)
    return msgs, r


# ---------------------------------------------------------------------------- streaming (out-of-core) sumcheck
STREAM_BYTES = float(os.environ.get("FPN_STREAM_GB", "3.5")) * (1 << 30)   # device budget of one instance's tables
STREAM_ANY = os.environ.get("FPN_STREAM_ANY", "") not in ("", "0")         # tests: stream on the NumPy back end too


def _instance_lins(total, nv):
    lins = {}
    for c, f in total.terms:
        for l, p in f:
            if l.key not in lins:
                off, m = _lin_block(l, nv)
                lins[l.key] = (l, off, m)
    return lins


def _table_bytes(total, nv):
    return sum(24 << m for (_, _, m) in _instance_lins(total, nv).values())


class _StreamSrc:
    """Per-instance source access for chunked materialization: small vectors are cached whole on the device (LRU,
    byte-bounded), large ones are read as the row range a chunk's view touches."""
    SMALL = 1 << 16

    def __init__(self, B, budget=3.0 * (1 << 30)):
        self.B, self.budget, self.cache, self.used = B, budget, {}, 0
        self.fetched = 0

    def full(self, vec):
        key = ("C", vec.id) if isinstance(vec, Vec) else ("U", vec.name)
        hit = self.cache.pop(key, None)
        if hit is None:
            B = self.B
            hit = _vec_e(B, vec) if isinstance(vec, Vec) else B.ext(B.asarray(vec.vals))
            self.fetched += 24 * hit.shape[1]
            nb = hit.nbytes
            while self.cache and self.used + nb > self.budget:
                self.used -= self.cache.pop(next(iter(self.cache))).nbytes
            self.used += nb
        self.cache[key] = hit
        return hit

    def rows(self, vec, a0, a1):
        self.fetched += 24 * (a1 - a0)
        return _dev_rows(self.B, vec, a0, a1)


def _view_table_stream(B, a, nv, blk, src):
    off, m = blk
    kind, ylo, yhi = a.analyze_block(off, m)
    if kind == "empty":
        return B.zeros_e(1 << m)
    if kind == "contig":
        if a.public:
            return B.xp.ascontiguousarray(src.full(a.vec)[:, ylo:yhi])
        return B.xp.ascontiguousarray(src.rows(a.vec, ylo, yhi))
    xp = B.xp
    y, ok = a.index_map(nv, xp, blk)
    N = 1 << a.vec.nbits
    if a.shift:
        y = y + a.shift
        ok &= (y < N) & (y >= 0)
        y = xp.where(ok, y, 0)
    okm = ok.astype(np.uint64)[None, :]
    if a.public or a.vec.N <= max(src.SMALL, 2 * len(y)):
        return src.full(a.vec)[:, y] * okm
    if yhi - ylo <= 4 * len(y) + 4096:
        rows = src.rows(a.vec, ylo, yhi)
        return rows[:, xp.clip(y - ylo, 0, yhi - ylo - 1)] * okm
    return src.full(a.vec)[:, y] * okm


def _atom_table_stream(B, a, nv, blk, src):
    if isinstance(a, View):
        return _view_table_stream(B, a, nv, blk, src)
    return a.table(B, nv, blk)


def _lin_table_stream(B, l, nv, lin_blk, blk, src):
    """Table of Lin l restricted to the aligned sub-block blk of its block lin_blk (same values as _materialize)."""
    off, m = lin_blk
    b0, bm = blk
    items = list(l.items.values())
    if len(items) == 1 and l.const == ZERO:
        a, co = items[0]
        t = _atom_table_stream(B, a, nv, blk, src)
        return B.escale(t, co) if co != ONE else t
    acc = B.zeros_e(1 << bm)
    for a, co in items:
        ao, am = a.block(nv) if (l.const == ZERO) else (off, m)
        if am > m or (ao >> m) != (off >> m):
            ao, am = off, m
        if am >= bm:
            if (b0 >> am) != (ao >> am):
                continue
            t = _atom_table_stream(B, a, nv, blk, src)
            B.eaxpy_(acc, t, co)
        else:
            if (ao >> bm) != (b0 >> bm):
                continue
            t = _atom_table_stream(B, a, nv, (ao, am), src)
            B.eaxpy_(acc[:, ao - b0:ao - b0 + (1 << am)], t, co)
        del t
    if l.const != ZERO:
        acc = B.eaddc(acc, l.const)
    return acc


def _chunk_tables(B, lins, nv, c0, c, src):
    out = {}
    for key, (l, off, m) in lins.items():
        if m <= c:
            if (off >> c) != (c0 >> c):
                continue
            out[key] = [off, _lin_table_stream(B, l, nv, (off, m), (off, m), src)]
        else:
            if (c0 >> m) != (off >> m):
                continue
            out[key] = [c0, _lin_table_stream(B, l, nv, (off, m), (c0, c), src)]
    return out


def _fold_k(B, tabs, rs):
    for a in rs:
        tabs = {k: _fold_block(B, _expand2(B, e), a) for k, e in tabs.items()}
    return tabs


STREAM_STATS = []


def _sumcheck_stream(B, total, nv, D, mask_coef, rho, tr):
    """Out-of-core sumcheck for an instance whose tables exceed STREAM_BYTES of device memory.  Variables are bound
    LSB first, so every pair of round i < k lies inside one aligned chunk of 2^c cube points: rounds 0..k-1 stream the
    cube chunk by chunk (each chunk's tables are materialized from the group store, folded with the challenges drawn
    so far, and reduced), then one more pass keeps the k-times folded tables (2^-k of the size) on the device and the
    in-core prover finishes.  Same round polynomials as _sumcheck_prove (exact field arithmetic), so the same proof."""
    import time
    t0 = time.time()
    terms = _sc_terms(total)
    fused = _sc_fused(B, D, terms)
    lins = _instance_lins(total, nv)
    tot = sum(24 << m for (_, _, m) in lins.values())

    def chunk_bytes(cb):
        span_b = 0
        bkts = {}
        for _, off, m in lins.values():
            if m > cb:
                span_b += 24 << cb
            else:
                q = off >> cb
                bkts[q] = bkts.get(q, 0) + (24 << m)
        return span_b + max(bkts.values(), default=0)
    c = nv
    while c > 2 and chunk_bytes(c) > STREAM_BYTES / 3:
        c -= 1
    k = 1
    while k < c - 1 and sum(24 * max(2, 1 << max(0, m - k)) for (_, _, m) in lins.values()) > STREAM_BYTES / 2:
        k += 1
    # Pre-index active Lins and active terms per chunk idx = c0 >> c to avoid O(|lins| + |terms|) scans per chunk
    nchunks = 1 << (nv - c)
    lins_by_chunk = [[] for _ in range(nchunks)]
    for key, (l, off, m) in lins.items():
        if m <= c:
            lins_by_chunk[off >> c].append((key, l, off, m, False))
        else:
            lo_q = (off >> c)
            hi_q = ((off + (1 << m)) >> c)
            for q in range(lo_q, min(hi_q, nchunks)):
                lins_by_chunk[q].append((key, l, off, m, True))
    terms_by_chunk = [[] for _ in range(nchunks)]
    for cc, f in terms:
        if not f:
            continue
        q_set = None
        for lk_key, _ in f:
            _, off, m = lins[lk_key]
            qs = {off >> c} if m <= c else set(range(off >> c, min((off + (1 << m)) >> c, nchunks)))
            q_set = qs if q_set is None else (q_set & qs)
            if not q_set:
                break
        for q in (q_set or ()):
            terms_by_chunk[q].append((cc, f))

    def build_chunk(q):
        c0 = q << c
        out = {}
        for key, l, off, m, is_span in lins_by_chunk[q]:
            blk = (c0, c) if is_span else (off, m)
            out[key] = [blk[0], _lin_table_stream(B, l, nv, (off, m), blk, src)]
        return out

    src = _StreamSrc(B)
    msgs, r, pre = [], [], ZERO
    for i in range(k):
        h = 1 << (nv - i - 1)
        vals = _sc_round_vals(B, [(cc, f) for cc, f in terms if not f], {}, D, fused, h)
        for q in range(nchunks):
            if not terms_by_chunk[q]:
                continue
            tabs = build_chunk(q)
            if not tabs:
                continue
            tabs = _fold_k(B, tabs, r)
            tabs = {kk: _expand2(B, e) for kk, e in tabs.items()}
            vals = hadd_list(vals, _sc_round_vals(B, terms_by_chunk[q], tabs, D, fused, h, with_const=False))
            del tabs
        vals = hadd_list(vals, _mask_round(mask_coef, i, nv, D, pre, rho))
        tr.absorb(canon(vals)); msgs.append(vals)
        a = tr.ext(); r.append(a)
        pre = hadd(pre, _gi(mask_coef, i, D, a))
        _free(B)
    pieces = {}
    for q in range(nchunks):
        if not lins_by_chunk[q]:
            continue
        tabs = build_chunk(q)
        for kk, e in _fold_k(B, tabs, r).items():
            pieces.setdefault(kk, []).append(e)
        del tabs
    src.cache.clear()
    tables = {}
    xp = B.xp
    for kk, lst in pieces.items():
        if len(lst) == 1:
            tables[kk] = lst[0]
        else:
            lst.sort(key=lambda e: e[0])
            tables[kk] = [lst[0][0], xp.concatenate([e[1] for e in lst], axis=1)]
    del pieces
    _free(B)
    msgs, r = _sumcheck_prove(B, total, nv, D, tables, mask_coef, rho, tr, start=k, r=r, msgs=msgs, pre=pre)
    del tables
    STREAM_STATS.append({"nv": nv, "tables": len(lins), "table_gib": round(tot / (1 << 30), 2), "chunk_bits": c,
                         "stream_rounds": k, "fetched_gib": round(src.fetched / (1 << 30), 2),
                         "s": round(time.time() - t0, 2)})
    if os.environ.get("FPN_LOG"):
        print(f"[zk] streamed instance 2^{nv}: {STREAM_STATS[-1]}", flush=True)
    return msgs, r


CLAIM_CHUNK_BITS = int(os.environ.get("FPN_CLAIM_CHUNK_BITS", "22"))   # larger vectors: chunked claim evaluation


def _eq_rows(B, z, a0, cb, shift=0):
    """W[y] = eq(z, y - shift) for y in [a0, a0 + 2^cb) (zero where y - shift falls outside the cube), a0 aligned."""
    nb = len(z); L = 1 << cb
    T = B.eq_table(list(z[:cb]))

    def blockw(b):
        if b < 0 or b >= (1 << (nb - cb)):
            return B.zeros_e(L)
        sc = ONE
        for j in range(cb, nb):
            sc = hmul(sc, z[j] if (b >> (j - cb)) & 1 else hsub(ONE, z[j]))
        return B.escale(T, sc)
    x0 = a0 - shift
    b1 = x0 // L if x0 >= 0 else -((-x0 + L - 1) // L)
    o = x0 - b1 * L
    if o == 0:
        return blockw(b1)
    both = B.xp.concatenate([blockw(b1), blockw(b1 + 1)], axis=1)
    return B.xp.ascontiguousarray(both[:, o:o + L])


def _claim_values(B, atoms, r, vcache):
    """MLE values of the view atoms at their local points: one eq / shift table per distinct (nbits, shift, point),
    device-side dot products, one host transfer.  Vectors above 2^CLAIM_CHUNK_BITS are evaluated in row chunks
    straight from the group store (eq table of a chunk = small eq table x constant-bit factor)."""
    groups = {}
    for i, a in enumerate(atoms):
        z = tuple(a.local_point(r))
        groups.setdefault((a.vec.nbits, a.shift, z), []).append(i)
    dev = [None] * len(atoms)
    for (nb, sh, z), idx in groups.items():
        if nb > CLAIM_CHUNK_BITS and B.name == "cupy":
            cb = CLAIM_CHUNK_BITS
            for i in idx:
                v = atoms[i].vec
                acc = None
                for a0 in range(0, 1 << nb, 1 << cb):
                    W = _eq_rows(B, list(z), a0, cb, sh)
                    src = _dev_rows(B, v, a0, a0 + (1 << cb))
                    d = B.edot_dev(src, W)
                    acc = d if acc is None else B.eadd(acc[:, None], d[:, None])[:, 0]
                    del W, src
                dev[i] = acc
            _free(B)
            continue
        W = shift_table(B, list(z), sh, nb) if sh else B.eq_table(list(z))
        for i in idx:
            dev[i] = B.edot_dev(_atom_src(B, atoms[i], vcache), W)
        del W
    return B.fetch_pairs(dev)


def _claim_value_prover(B, a, r, vcache):
    v = a.vec
    src = vcache.get(v.id)
    if src is None:
        src = vcache[v.id] = _vec_e(B, v)
    if a.shift:
        W = shift_table(B, a.local_point(r), a.shift, v.nbits)
    else:
        W = B.eq_table(a.local_point(r))
    return B.esum(B.emul(src, W))


def prove(cir, statement, B=CPU, label=b"fpn-flat-v1", phase1_only=False):
    """cir: Circuit in prove mode with all phase-1 values set; lookups declared.  Returns the proof dict.
    phase1_only: stop right after the phase-1 commitments and return {"root1": [...]} (used by flashprove_next.mc_setup, whose
    row-block proofs must all be committed before a shared challenge; the second pass reproduces the same roots under
    field.seeded_rng)."""
    cir.phase1_only = phase1_only
    tr = Transcript(label)
    tr.absorb(statement)
    pf = {}
    import gc
    _rss("prove start")
    for name in sorted(cir.lookups):
        _logup_mult(cir, cir.lookups[name])
    gc.collect()
    _rss("after lookup multiplicities")
    _mem_report(cir, [], "after lookup multiplicities")
    pcs = []
    try:
        return _prove_body(cir, tr, pf, B, pcs)
    finally:
        for pc in pcs:                      # delete spill files and opening material even on exceptions
            pc.release()
        _dev_big_drop(); _INV2X.clear(); _IMAP.clear()
        _EXT_CACHE[:] = [None, None, None, None]
        if _PROOF_SPILL[0] is not None:
            import shutil
            if _WRITER is not None:
                _WRITER.drain()
            shutil.rmtree(_PROOF_SPILL[0], ignore_errors=True); _PROOF_SPILL[0] = None


def _spill_policy(pcs_phase):
    """lean (disk-spilled tails + h_bf + salts) iff the phase's random material would exceed HOST_BUDGET GB of RAM."""
    need = sum(40 * pc.N + 24 * pc.ntail for pc in pcs_phase)
    for pc in pcs_phase:
        pc.force_lean = need > HOST_BUDGET * (1 << 30)


def _prove_body(cir, tr, pf, B, pcs_all):
    import gc
    ph1 = [v for v in cir.vecs if v.phase == 1]
    pcs1 = [PhaseCommit(B, g) for g in partition(ph1)]
    pcs_all.extend(pcs1)
    _spill_policy(pcs1)
    pf["root1"] = []
    for pc in pcs1:
        pf["root1"].append(pc.build()); tr.absorb(pf["root1"][-1])
        gc.collect(); _trim()
        _rss(f"after commit group 2^{pc.nb}")
    _mem_report(cir, pcs_all, "after phase-1 commit")
    if os.environ.get("FPN_LOG"):
        print(f"[zk] phase1: {len(ph1)} vectors, groups Big = {[f'2^{pc.nb}' for pc in pcs1]}", flush=True)
    if getattr(cir, "phase1_only", False):
        return {"root1": list(pf["root1"])}
    cir.root1 = list(pf["root1"])          # readable by commit rounds (flashprove_next.mc_setup derives a shared seed from it)
    pcsx = _commit_rounds(cir, tr, pf, B, pcs_all)
    for pc in pcs1 + pcsx:
        if not pc.lean:
            for v in pc.vecs:
                v.vals = None
    pcs1 = pcs1 + pcsx
    _finalize_lookups(cir, tr, B)
    gc.collect(); _trim()
    _rss("after LogUp witness")
    sizes = sorted(set(cir.zeros) | set(cir.sums))
    degs = {nv: _degree(cir, nv) for nv in sizes}
    masks = _masks(cir, sizes, degs)
    ph2 = [v for v in cir.vecs if v.phase == 2]
    pcs2 = [PhaseCommit(B, g) for g in partition(ph2)]
    pcs_all.extend(pcs2)
    _spill_policy(pcs2)
    pf["root2"] = []
    for pc in pcs2:
        pf["root2"].append(pc.build()); tr.absorb(pf["root2"][-1])
        gc.collect(); _trim()
    if os.environ.get("FPN_LOG"):
        print(f"[zk] phase2: {len(ph2)} vectors, groups Big = {[f'2^{pc.nb}' for pc in pcs2]}", flush=True)
    _rss("after phase-2 commit")
    _mem_report(cir, pcs_all, "after phase-2 commit")
    pcs = pcs1 + pcs2
    gid = {v.id: gi for gi, pc in enumerate(pcs) for v in pc.vecs}
    claims = {gi: _Claims() for gi in range(len(pcs))}
    vcache = {}
    pf["sc"] = []
    for nv in sizes:
        D = degs[nv]
        r_zc = tr.exts(nv)
        lambdas = tr.exts(_n_constraints(cir, nv))
        total = _instance_terms(cir, nv, r_zc, lambdas)
        mv = masks[nv]; cnt = nv * (D + 1)
        coef = _mask_coef(mv, cnt)
        G = _mask_sum(coef, nv, D)
        tr.absorb(canon([nv, list(G)]))
        rho = tr.ext()
        if os.environ.get("FPN_LOG"):
            print(f"[zk] instance 2^{nv}: terms={len(total.terms)} tables={len({l.key for c, f in total.terms for l, p in f})} deg={D}", flush=True)
        ntab = len({l.key for c, f in total.terms for l, p in f})
        big_inst = nv >= BIG_INSTANCE_BITS or ntab * (24 << nv) > BIG_INSTANCE_BYTES
        stream = (B.name == "cupy" or STREAM_ANY) and _table_bytes(total, nv) > STREAM_BYTES
        if big_inst or stream:             # large instance: only its own tables (+ one source at a time) on the device
            vcache.clear(); _dev_big_drop(); _free(B); _NO_DEVBIG[0] = True
        if stream:
            try:
                msgs, r = _sumcheck_stream(B, total, nv, D, coef, rho, tr)
            finally:
                _NO_DEVBIG[0] = False
        else:
            try:
                tables = _materialize(B, total, nv, vcache)
            finally:
                _NO_DEVBIG[0] = False
            if big_inst:                   # sources are re-uploaded for the claim values (cheap next to the tables)
                _vcache_clear(vcache); _dev_big_drop(); _free(B)
            msgs, r = _sumcheck_prove(B, total, nv, D, tables, coef, rho, tr)
            del tables
        if big_inst or stream:
            _free(B)
        atoms = _view_atoms(total)
        cvals = []
        for a, val in zip(atoms, _claim_values(B, atoms, r, vcache)):
            cvals.append(list(val))
            spec = ("sh", a.vec.id, tuple(a.local_point(r)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r)))
            claims[gid[a.vec.id]].add(spec, val)
        gval = ZERO
        for i in range(nv):
            gval = hadd(gval, _gi(coef, i, D, r[i]))
        claims[gid[mv.id]].add(("pw", mv.id, tuple(r), D), gval)
        if _VCACHE_USED[0] > VCACHE_BYTES:
            _vcache_clear(vcache); _free(B)
        if nv >= 20:
            _rss(f"after sumcheck 2^{nv}")
        tr.absorb(canon(cvals), canon(list(gval)))
        pf["sc"].append({"nv": nv, "G": list(G), "msgs": msgs, "claims": cvals, "g": list(gval)})
    _rss("after all sumchecks")
    zk_budget_check(cir, claims)
    _vcache_clear(vcache); _free(B)
    pf["open"] = []
    for gi, pc in enumerate(pcs):
        for other in [g for g in _DEV_BIG if g is not pc]:     # only the group being opened may keep a device Big
            _DEV_BIG.pop(other)
        _free(B)
        _rss(f"before opening group {gi} (2^{pc.nb})")
        pc.ensure_tree()
        if hasattr(B, "clear_caches"):
            B.clear_caches()
        cl = claims[gi]
        tr.absorb(canon(len(cl.items)))
        mus = tr.exts(len(cl.items))
        W = B.zeros_e(pc.N); tot = ZERO
        groups = {}
        for mu, spec, val in zip(mus, cl.items, cl.vals):
            v = pc.vec_by_id[spec[1]]
            gk = (spec[0], v.nbits, tuple(spec[2])) + tuple(spec[3:])
            groups.setdefault(gk, []).append((mu, v))
            tot = hadd(tot, hmul(mu, val))
        for gk, members in groups.items():
            kind, nb, pt = gk[0], gk[1], list(gk[2])
            if kind == "pt":
                w = B.eq_table(pt)
            elif kind == "sh":
                w = shift_table(B, pt, gk[3], nb)
            else:
                w = B.asarray(_pow_weights(pt, gk[3], nb))
            for mu, v in members:
                o = pc.offset[v.id]
                B.eaxpy_(W[:, o:o + v.N], w, mu)
            del w
        if os.environ.get("FPN_DEBUG"):
            for mu, spec, val in zip(mus, cl.items, cl.vals):
                v = pc.vec_by_id[spec[1]]; o = pc.offset[v.id]
                if spec[0] == "pt":
                    w = B.eq_table(list(spec[2]))
                elif spec[0] == "sh":
                    w = shift_table(B, spec[2], spec[3], v.nbits)
                else:
                    w = B.asarray(_pow_weights(spec[2], spec[3], v.nbits))
                got = B.esum(B.emul(pc.big[:, o:o + v.N], w))
                if got != tuple(val):
                    print("CLAIM MISMATCH", ph, spec[0], v.name)
        pf["open"].append(basefold_prove(pc, W, tot, tr))
        del W; _INV2X.clear(); _free(B)
    _rss("after all openings")
    _INV2X.clear(); _IMAP.clear()
    if hasattr(B, "clear_caches"):
        B.clear_caches()
    return pf


def _commit_rounds(cir, tr, pf, B, pcs_all=None, prover=True):
    """Extra interactive commit rounds between phase 1 and the LogUp / Libra phase (used by flashprove_next.mc_zk: a vector
    must be committed before the challenge that depends on it, e.g. the combined row u before the column positions).
    Round k: cir.rounds[k](tr) draws its challenges from the transcript and declares (prover: fills) the round's
    vectors and constraints; the vectors (phase 10 + k) are committed as their own groups and their roots absorbed.
    Proofs of circuits without rounds are unchanged (no "rootx" key).  Returns the round groups in order."""
    out = []
    if not cir.rounds:
        return out
    cir.pf = pf                              # rounds may read / write proof-side data (flashprove_next.tc_zk: explicit sumchecks)
    if prover:
        pf["rootx"] = []
    elif not isinstance(pf.get("rootx"), list) or len(pf["rootx"]) != len(cir.rounds):
        raise ProofError("wrong number of commit rounds")
    for k, fn in enumerate(cir.rounds):
        ph = 10 + k
        cir.cur_phase = ph
        try:
            fn(tr)
        finally:
            cir.cur_phase = 1
        grp = [PhaseCommit(B, g) for g in partition([v for v in cir.vecs if v.phase == ph])]
        if prover:
            pcs_all.extend(grp)
            _spill_policy(grp)
            roots = []
            for pc in grp:
                roots.append(pc.build()); tr.absorb(roots[-1])
            pf["rootx"].append(roots)
            _rss(f"after commit round {k}")
        else:
            if not isinstance(pf["rootx"][k], list) or len(pf["rootx"][k]) != len(grp):
                raise ProofError(f"wrong number of round-{k} commitments")
            for r_ in pf["rootx"][k]:
                tr.absorb(r_)
        out.extend(grp)
    return out


def _pow_weights(r, D, nbits):
    w = np.zeros((3, 1 << nbits), dtype=np.uint64)
    for i, ri in enumerate(r):
        x = ONE
        for k in range(D + 1):
            w[0, i * (D + 1) + k] = x[0]; w[1, i * (D + 1) + k] = x[1]; w[2, i * (D + 1) + k] = x[2]
            x = hmul(x, ri)
    return w


def verify(cir, statement, pf, label=b"fpn-flat-v1"):
    """cir: Circuit in verify mode (same structure).  Raises ProofError on rejection."""
    tr = Transcript(label)
    tr.absorb(statement)
    ph1 = [v for v in cir.vecs if v.phase == 1]
    pcs1 = [PhaseCommit(CPU, g) for g in partition(ph1)]
    if not isinstance(pf["root1"], list) or len(pf["root1"]) != len(pcs1):
        raise ProofError("wrong number of phase-1 commitments")
    for r_ in pf["root1"]:
        tr.absorb(r_)
    cir.root1 = list(pf["root1"])
    pcsx = _commit_rounds(cir, tr, pf, CPU, prover=False)
    _finalize_lookups(cir, tr, CPU)
    sizes = sorted(set(cir.zeros) | set(cir.sums))
    degs = {nv: _degree(cir, nv) for nv in sizes}
    masks = _masks(cir, sizes, degs)
    ph2 = [v for v in cir.vecs if v.phase == 2]
    pcs2 = [PhaseCommit(CPU, g) for g in partition(ph2)]
    if not isinstance(pf["root2"], list) or len(pf["root2"]) != len(pcs2):
        raise ProofError("wrong number of phase-2 commitments")
    for r_ in pf["root2"]:
        tr.absorb(r_)
    pcs = pcs1 + pcsx + pcs2
    roots = list(pf["root1"]) + [r_ for rr in pf.get("rootx", []) for r_ in rr] + list(pf["root2"])
    gid = {v.id: gi for gi, pc in enumerate(pcs) for v in pc.vecs}
    claims = {gi: _Claims() for gi in range(len(pcs))}
    if len(pf["sc"]) != len(sizes):
        raise ProofError("wrong number of sumcheck instances")
    for nv, scp in zip(sizes, pf["sc"]):
        D = degs[nv]
        if scp["nv"] != nv:
            raise ProofError("instance size mismatch")
        r_zc = tr.exts(nv)
        lambdas = tr.exts(_n_constraints(cir, nv))
        total = _instance_terms(cir, nv, r_zc, lambdas)
        if not isinstance(scp["G"], (list, tuple)) or len(scp["G"]) != 3:
            raise ProofError("Libra mask sum must be an element of E")
        G = (int(scp["G"][0]) % P, int(scp["G"][1]) % P, int(scp["G"][2]) % P)
        tr.absorb(canon([nv, list(G)]))
        rho = tr.ext()
        cl = hmul(rho, G)
        if len(scp["msgs"]) != nv:
            raise ProofError("sumcheck length")
        r = []
        for vals in scp["msgs"]:
            vals = [tuple(int(t) % P for t in v) for v in vals]
            if len(vals) != D + 1 or any(len(v) != 3 for v in vals) or hadd(vals[0], vals[1]) != cl:
                raise ProofError(f"sumcheck round mismatch (cube 2^{nv})")
            tr.absorb(canon(vals))
            a = tr.ext(); r.append(a)
            cl = lagrange_eval(vals, a)
        atoms = _view_atoms(total)
        if len(scp["claims"]) != len(atoms) or any(len(val) != 3 for val in scp["claims"]):
            raise ProofError("claim count")
        if not isinstance(scp["g"], (list, tuple)) or len(scp["g"]) != 3:
            raise ProofError("bad Libra evaluation")
        cv = {}
        scp["claims"] = [[int(t) % P for t in val] for val in scp["claims"]]
        scp["g"] = [int(t) % P for t in scp["g"]]
        for a, val in zip(atoms, scp["claims"]):
            val = tuple(val)
            cv[a.kid] = val
            spec = ("sh", a.vec.id, tuple(a.local_point(r)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r)))
            claims[gid[a.vec.id]].add(spec, val)
        gval = tuple(int(t) % P for t in scp["g"])
        claims[gid[masks[nv].id]].add(("pw", masks[nv].id, tuple(r), D), gval)
        tr.absorb(canon(scp["claims"]), canon(list(gval)))
        lhs = hadd(_eval_terms(total, r, cv), hmul(rho, gval))
        if lhs != cl:
            raise ProofError(f"sumcheck final evaluation mismatch (cube 2^{nv})")
    if len(pf["open"]) != len(pcs):
        raise ProofError("missing openings")
    for gi, (pc, op, root) in enumerate(zip(pcs, pf["open"], roots)):
        if not isinstance(root, (bytes, bytearray)) or len(root) != 32:
            raise ProofError("bad commitment root")
        cl = claims[gi]
        tr.absorb(canon(len(cl.items)))
        mus = tr.exts(len(cl.items))
        tot = ZERO
        for mu, val in zip(mus, cl.vals):
            tot = hadd(tot, hmul(mu, val))

        def w_mle(alpha, cl=cl, pc=pc, mus=mus):
            """MLE of the batched weight table at alpha: each distinct local weight (point / shift / power, nbits) and each
            distinct high-bit selector (offset, nbits) is evaluated once; claims only add mu-weighted products."""
            lw = {}; eh = {}; per = {}
            for mu, spec in zip(mus, cl.items):
                v = pc.vec_by_id[spec[1]]; o = pc.offset[v.id]; nb = v.nbits
                gk = (spec[0], nb) + tuple(spec[2:])
                hk = (o >> nb, nb)
                e = per.get((gk, hk))
                per[(gk, hk)] = mu if e is None else hadd(e, mu)
            acc = ZERO
            for (gk, hk), m in per.items():
                kind, nb = gk[0], gk[1]
                wl = lw.get(gk)
                if wl is None:
                    lo = alpha[:nb]
                    if kind == "pt":
                        wl = eq_pt(list(gk[2]), lo)
                    elif kind == "sh":
                        wl = mle_shift_signed(list(gk[2]), lo, gk[3])
                    else:
                        w = _pow_weights(gk[2], gk[3], nb)
                        wl = CPU.esum(CPU.emul(w, CPU.eq_table(list(lo))))
                    lw[gk] = wl
                ehi = eh.get(hk)
                if ehi is None:
                    ehi = ONE; hb = hk[0]
                    for k, a in enumerate(alpha[nb:]):
                        ehi = hmul(ehi, a if (hb >> k) & 1 else hsub(ONE, a))
                    eh[hk] = ehi
                acc = hadd(acc, hmul(m, hmul(ehi, wl)))
            return acc
        basefold_verify(root, pc.nb, tot, op, tr, w_mle)
    return True


ZK_REPORT = {}


def _next_match(lo, mask, base, n):
    """Smallest y with lo <= y < 2^n and (y & mask) == base, or None (bits outside `mask` are free)."""
    if lo >= (1 << n):
        return None
    best = None
    prefix = 0
    for j in range(n - 1, -1, -1):
        bl = (lo >> j) & 1
        low_fixed = base & ((1 << j) - 1)
        if (mask >> j) & 1:
            bj = (base >> j) & 1
            if bj > bl:
                return prefix | (1 << j) | low_fixed          # y > lo from here: lowest completion
            if bj < bl:
                return best
            prefix |= bj << j
        elif bl == 0:
            best = prefix | (1 << j) | low_fixed               # branch y > lo (deeper branches are smaller)
        else:
            prefix |= 1 << j
    return prefix                                              # y == lo matches


def _tail_assign(lo, nbits, pats):
    """Greedy first-fit: each entry (eff_lo, eff_hi, pat_dict) takes the first unused row y in [max(lo, eff_lo), eff_hi)
    agreeing with pat_dict.  Returns the rows, or None if some claim finds no free random row in its support."""
    used = set(); rows = []
    for eff_lo, eff_hi, p in pats:
        mask = 0; base = 0
        for j, bb in p.items():
            mask |= 1 << j; base |= bb << j
        start = max(lo, eff_lo)
        y = _next_match(start, mask, base, nbits) if start < eff_hi else None
        while y is not None and y < eff_hi and y in used:
            y = _next_match(y + 1, mask, base, nbits)
        if y is None or y >= eff_hi:
            return None
        used.add(y); rows.append(y)
    return rows


def zk_budget_check(cir, claims):
    """T1 accounting: every revealed evaluation of a committed vector must be masked by its own random rows.
    Each claim (most constrained first) takes the FIRST unused random tail row y in [N - tail, N) within the claim's
    non-zero weight support whose bits agree with the claim's Boolean coordinates; raises if none exists.
    For shifted claims ('sh', delta), W[y] = eq(r, y - delta) is supported only on y in [max(0, delta), N + min(0, delta)),
    and Boolean coordinates below the lowest 1-bit of |delta| translate directly to y."""
    byv = {}
    for gi in claims:
        for spec in claims[gi].items:
            byv.setdefault(spec[1], []).append(spec)
    worst = 0
    for vid, specs in byv.items():
        v = cir.vecs[vid]
        if v.phase == 2 and v.name.startswith("libra:"):
            # Libra sumcheck mask vectors are purely random helper polynomials (not secret witnesses)
            continue
        def pattern(spec):
            kind = spec[0]
            if kind == "pt":
                out = {}
                for j, z in enumerate(spec[2]):
                    if z[1] == 0 and z[2] == 0 and z[0] in (0, 1):
                        out[j] = z[0]
                return (0, v.N, out)
            if kind == "sh":
                delta = int(spec[3])
                lo_s = max(0, delta); hi_s = v.N + min(0, delta)
                tz = (abs(delta) & -abs(delta)).bit_length() - 1 if delta != 0 else v.nbits
                out = {}
                for j, z in enumerate(spec[2]):
                    if j < tz and z[1] == 0 and z[2] == 0 and z[0] in (0, 1):
                        out[j] = z[0]
                return (lo_s, hi_s, out)
            raise ProofError(f"ZK budget: unsupported claim kind {kind!r} on vector {v.name}")
        pats = sorted((pattern(s) for s in specs), key=lambda p: (-(len(p[2])), p[1] - p[0]))
        if _tail_assign(v.N - v.tail, v.nbits, pats) is None:
            raise ProofError(f"ZK budget: vector {v.name} has {len(specs)} claims, random rows {v.tail}: "
                             + str([(s_[0], len(s_[2]), [z for z in s_[2] if z[1] == 0 and z[2] == 0 and z[0] < 2]) for s_ in specs]))
        worst = max(worst, len(specs) / max(1, v.tail))
    assert EDIM == 3
    ZK_REPORT["max_claims_per_random_row"] = worst
    ZK_REPORT["vectors_checked"] = len(byv)
    ZK_REPORT["edim"] = EDIM


def check_witness(cir, B=CPU, max_report=20):
    """Debug helper (prover side): evaluate every constraint on the witness (tails included) and report failures.
    Lookups are checked when multiplicities are computed.  Returns a list of failing tags."""
    import random
    bad = []
    vcache = {}
    for name in sorted(cir.lookups):
        _logup_mult(cir, cir.lookups[name])
    for v in cir.vecs:
        if v.phase == 1 and v.full is None and wit(v) is not None and not v.is_ext:
            w = wit(v)
            full = np.zeros((3, v.N), dtype=np.uint64); full[0, :len(w)] = field_u64(w)
            full[:, v.N - v.tail:] = CPU.rand_e(v.tail)
            v.full = full
    for nv in sorted(set(cir.zeros) | set(cir.sums)):
        r_zc = [(random.randrange(P), random.randrange(P), random.randrange(P)) for _ in range(nv)]
        for ex, tag in cir.zeros.get(nv, []):
            _bind_late(ex, r_zc)
            tabs = _materialize(B, ex, nv, vcache)
            val = np.zeros((3, 1 << nv), dtype=np.uint64)
            for c, f in ex.terms:
                if not f:
                    val = CPU.eaddc(val, c); continue
                sup = _term_support([tabs[l.key] for l, p in f])
                if sup is None:
                    continue
                lo, hi = sup
                prod = None
                for l, p in f:
                    off, arr = tabs[l.key]
                    t = B.tonumpy(arr[:, lo - off:hi - off])
                    t = t if p == 1 else CPU.epow(t, p)
                    prod = t if prod is None else CPU.emul(prod, t)
                val[:, lo:hi] = CPU.eadd(val[:, lo:hi], CPU.escale(prod, c))
            nz = np.nonzero(val[0] | val[1] | val[2])[0]
            if len(nz):
                bad.append((tag, nv, int(nz[0]), len(nz)))
        for ex, ev, tag in cir.sums.get(nv, []):
            _bind_late(ex, r_zc)
            tabs = _materialize(B, ex, nv, vcache)
            acc = ZERO
            for c, f in ex.terms:
                if not f:
                    acc = hadd(acc, hmul(c, ((1 << nv) % P, 0, 0))); continue
                sup = _term_support([tabs[l.key] for l, p in f])
                if sup is None:
                    continue
                lo, hi = sup
                prod = None
                for l, p in f:
                    off, arr = tabs[l.key]
                    t = arr[:, lo - off:hi - off]
                    t = t if p == 1 else B.epow(t, p)
                    prod = t if prod is None else B.emul(prod, t)
                acc = hadd(acc, hmul(c, B.esum(prod)))
            if acc != ZERO:
                bad.append((tag, nv, "sum", 1))
    for v in cir.vecs:
        if v.phase == 1:
            v.full = None
    for b in bad[:max_report]:
        print("CONSTRAINT FAILS:", b)
    return bad


def static_claims(cir):
    """Count, per committed vector, the distinct view atoms (= revealed evaluations) over all instances.
    Returns {vec_id: [(nv, atom key), ...]} (call after lookups are finalized)."""
    out = {}
    for nv in sorted(set(cir.zeros) | set(cir.sums)):
        seen = set()
        exs = [e for e, _ in cir.zeros.get(nv, [])] + [e for e, _, _ in cir.sums.get(nv, [])]
        for ex in exs:
            for c, f in ex.terms:
                for l, p in f:
                    for a in l.atoms():
                        if isinstance(a, View) and not a.public and a.key not in seen:
                            seen.add(a.key)
                            out.setdefault(a.vec.id, []).append((nv, a.varmap))
    return out
