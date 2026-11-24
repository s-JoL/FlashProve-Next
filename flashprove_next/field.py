# SPDX-License-Identifier: Apache-2.0
"""Goldilocks field p = 2^64 - 2^32 + 1 and its cubic extension E = F_p[u]/(u^3 - 2).

Two array back ends with the same API:
  * ``CPU`` -- pure NumPy (used by the verifier and by small CPU proofs); no GPU dependency.
  * ``gpu_backend()`` -- CuPy kernels (imported lazily so the verifier never touches CuPy).

Conventions: a base-field vector is a uint64 array with values in [0, p); an extension vector is a (3, n) uint64
array (limb 0 = constant term, limb 1 = coefficient of u, limb 2 = coefficient of u^2).  Extension scalars on the
host are Python int 3-tuples.
"""
import os
import numpy as np

P = (1 << 64) - (1 << 32) + 1
EDIM = 3
NR = 2
INV2 = pow(2, P - 2, P)

# ----------------------------------------------------------------------------- host scalars (Python ints)
ZERO, ONE = (0, 0, 0), (1, 0, 0)


def hadd(a, b): return ((a[0] + b[0]) % P, (a[1] + b[1]) % P, (a[2] + b[2]) % P)
def hsub(a, b): return ((a[0] - b[0]) % P, (a[1] - b[1]) % P, (a[2] - b[2]) % P)
def hneg(a): return ((-a[0]) % P, (-a[1]) % P, (-a[2]) % P)
def hmul(a, b):
    return ((a[0] * b[0] + 2 * (a[1] * b[2] + a[2] * b[1])) % P,
            (a[0] * b[1] + a[1] * b[0] + 2 * a[2] * b[2]) % P,
            (a[0] * b[2] + a[1] * b[1] + a[2] * b[0]) % P)
def hscal(c, a): return ((c * a[0]) % P, (c * a[1]) % P, (c * a[2]) % P)
def hpow(a, e):
    r = ONE
    while e:
        if e & 1: r = hmul(r, a)
        a = hmul(a, a); e >>= 1
    return r
def hinv(a):
    t0 = (a[0] * a[0] - 2 * a[1] * a[2]) % P
    t1 = (2 * a[2] * a[2] - a[0] * a[1]) % P
    t2 = (a[1] * a[1] - a[0] * a[2]) % P
    n = (a[0] * t0 + 2 * (a[2] * t1 + a[1] * t2)) % P
    if n == 0: raise ZeroDivisionError("inverse of zero in E")
    ni = pow(n, P - 2, P)
    return (t0 * ni % P, t1 * ni % P, t2 * ni % P)
def hf(x):
    """Signed Python int -> base-field element embedded in E."""
    return (int(x) % P, 0, 0)


def lagrange_eval(vals, r):
    """Univariate polynomial with g(0..d) = vals, evaluated at r (E)."""
    d = len(vals) - 1; out = ZERO
    for i in range(d + 1):
        num, den = ONE, 1
        for j in range(d + 1):
            if j != i:
                num = hmul(num, hsub(r, (j, 0, 0))); den = den * (i - j) % P
        out = hadd(out, hmul(hmul(vals[i], num), (pow(den, P - 2, P), 0, 0)))
    return out


def eq1(a, b):
    """eq(a, b) for one variable: a b + (1-a)(1-b)."""
    return hadd(hmul(a, b), hmul(hsub(ONE, a), hsub(ONE, b)))


def eq_pt(a, b):
    n = min(len(a), len(b))
    if n >= 6:
        lib = native()
        if lib is not None:
            pa = _pts(a[:n]); pb = _pts(b[:n]); out = np.zeros(3, dtype=np.uint64)
            if len(pa) == 3 * n and len(pb) == 3 * n:
                lib.eq_pt(ptr(pa), ptr(pb), n, ptr(out))
                return (int(out[0]), int(out[1]), int(out[2]))
    out = ONE
    for x, y in zip(a, b):
        out = hmul(out, eq1(x, y))
    return out


def rand_ext():
    d = rand_field(3)
    return (int(d[0]), int(d[1]), int(d[2]))


# ----------------------------------------------------------------------------- CSPRNG (os.urandom only, no PRG)
_URAND_CHUNK = 16 << 20          # bytes per getrandom() call / worker task
_URAND_POOL = None


_URAND_FD = None


def _urand_fill(mv):
    """Fill a writable byte memoryview from the kernel CSPRNG: readv() on /dev/urandom straight into the buffer (the
    same CRNG as os.urandom / getrandom(); no intermediate bytes objects), os.urandom where /dev/urandom is absent."""
    global _URAND_FD
    if _URAND_FD is None:
        try:
            _URAND_FD = os.open("/dev/urandom", os.O_RDONLY)
        except OSError:
            _URAND_FD = -1
    n = 0
    while n < len(mv):
        if _URAND_FD >= 0:
            n += os.readv(_URAND_FD, [mv[n:n + (32 << 20) - 1]])
        else:
            k = min(len(mv) - n, 32 << 20)
            mv[n:n + k] = os.urandom(k); n += k


_SEEDED = []                     # stack of [key, call counter] (seeded_rng); empty = OS CSPRNG


class seeded_rng:
    """Context manager: inside it, urandom_u64 / rand_field / rand_e draw from a keyed CSPRNG instead of the OS:
    AES-256-CTR keystream (the `cryptography` package; SHAKE-256 counter mode as fallback), key = a secret 256-bit
    value, one fresh 2^64-block counter range per call.  Used only where a prover must REPRODUCE a commitment
    bit-for-bit later (flashprove_next.mc_setup: every row-block's phase-1 commitment is computed twice, once before and once
    after the shared column challenge).  The key comes from os.urandom and never leaves the prover, so the stream is
    computationally indistinguishable from the OS one (the T1/T4 blinding becomes computational instead of
    statistical)."""

    def __init__(self, key):
        assert isinstance(key, (bytes, bytearray)) and len(key) == 32
        self.key = bytes(key)

    def __enter__(self):
        _SEEDED.append([self.key, 0])
        return self

    def __exit__(self, *a):
        _SEEDED.pop()


def _seeded_fill(mv):
    st = _SEEDED[-1]
    key, ctr = st[0], st[1]
    st[1] += 1
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        n = len(mv)
        if n >= (32 << 20):
            global _URAND_POOL
            if _URAND_POOL is None:
                from concurrent.futures import ThreadPoolExecutor
                _URAND_POOL = ThreadPoolExecutor(max_workers=min(16, max(2, (os.cpu_count() or 4))))
            chunk = 16 << 20
            z = bytes(chunk)

            def fill(a0):
                k = min(chunk, n - a0)
                iv = ((ctr << 64) + (a0 >> 4)).to_bytes(16, "big")
                mv[a0:a0 + k] = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor().update(z[:k])
            list(_URAND_POOL.map(fill, range(0, n, chunk)))
            return
        enc = Cipher(algorithms.AES(key), modes.CTR((ctr << 64).to_bytes(16, "big"))).encryptor()
        step = 32 << 20
        zero = bytes(min(step, n))
        for a in range(0, n, step):
            k = min(step, n - a)
            mv[a:a + k] = enc.update(zero[:k])
    except ImportError:                                   # pragma: no cover (slow path)
        import hashlib
        step = 1 << 20
        for c, a in enumerate(range(0, len(mv), step)):
            k = min(step, len(mv) - a)
            mv[a:a + k] = hashlib.shake_256(key + ctr.to_bytes(8, "little") + c.to_bytes(8, "little")).digest(k)


_TEST_RNG = [None, None]   # (seed string, numpy Generator): FPN_TEST_RNG_SEED, regression tests only


def _test_rng():
    """FPN_TEST_RNG_SEED (tests only, never in production): replace the kernel CSPRNG by a seeded PCG64 stream so
    two prover builds can be compared byte for byte on the same circuit.  Re-seeded whenever the variable changes."""
    s = os.environ.get("FPN_TEST_RNG_SEED")
    if not s:
        return None
    if _TEST_RNG[0] != s:
        _TEST_RNG[0], _TEST_RNG[1] = s, np.random.default_rng(int(s))
    return _TEST_RNG[1]


def urandom_u64(n, out=None):
    """n uniform 64-bit words straight from the OS CSPRNG (os.urandom's kernel CRNG), filled in parallel threads
    (the syscall releases the GIL and the kernel CRNG is per-CPU: 8-16 workers reach several GB/s).  No user-space PRG
    (except inside a `seeded_rng` context, see there)."""
    global _URAND_POOL
    out = np.empty(int(n), dtype=np.uint64) if out is None else out
    nb = out.size * 8
    if nb == 0:
        return out
    if _SEEDED:
        _seeded_fill(memoryview(out.reshape(-1).view(np.uint8)))
        return out
    g = _test_rng()
    if g is not None:
        out.reshape(-1)[:] = g.integers(0, 1 << 64, size=out.size, dtype=np.uint64, endpoint=False)
        return out
    mv = memoryview(out.reshape(-1).view(np.uint8))
    if nb <= 2 * _URAND_CHUNK:
        _urand_fill(mv)
        return out
    if _URAND_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _URAND_POOL = ThreadPoolExecutor(max_workers=min(16, max(2, (os.cpu_count() or 4))))

    def fill(a0):
        _urand_fill(mv[a0:min(nb, a0 + _URAND_CHUNK)])
    list(_URAND_POOL.map(fill, range(0, nb, _URAND_CHUNK)))
    return out


def rand_field(n, out=None):
    """n uniform elements of F_p by rejection sampling of os.urandom words (a word is rejected with probability
    (2^32 - 1) / 2^64; rejected words are replaced by fresh draws, so the output is exactly uniform)."""
    out = urandom_u64(n, out)
    flat = out.reshape(-1)
    bad = np.flatnonzero(flat >= _PU_)
    while len(bad):
        fresh = urandom_u64(len(bad))
        flat[bad] = fresh
        bad = bad[fresh >= _PU_]
    return out


_PU_ = np.uint64(P)


# ----------------------------------------------------------------------------- NumPy back end
_PU = np.uint64(P); _M32 = np.uint64(0xFFFFFFFF); _S32 = np.uint64(32)


def _np_red(lo, hi):
    hh = hi >> _S32; hl = hi & _M32
    t = lo - hh; t = np.where(lo < hh, t - _M32, t)
    m = hl * _M32
    r = t + m; r = np.where(r < t, r + _M32, r)
    return np.where(r >= _PU, r - _PU, r)


class _NumpyBackend:
    name = "numpy"
    xp = np

    # -- base field
    @staticmethod
    def mul(a, b):
        a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
        if a.size >= 4096:
            lib = native()
            if lib is not None:
                a2, b2 = np.broadcast_arrays(a, b)
                a2 = _c(a2); b2 = _c(b2); out = np.empty(a2.shape, dtype=np.uint64)
                lib.gl_mul(ptr(a2), ptr(b2), ptr(out), a2.size); return out
        al, ah = a & _M32, a >> _S32; bl, bh = b & _M32, b >> _S32
        ll = al * bl; lh = al * bh; hl = ah * bl; hh = ah * bh
        mid = lh + hl; c1 = (mid < lh).astype(np.uint64)
        lo = ll + (mid << _S32); c2 = (lo < ll).astype(np.uint64)
        hi = hh + (mid >> _S32) + (c1 << _S32) + c2
        return _np_red(lo, hi)

    @staticmethod
    def add(a, b):
        a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
        if a.size >= 4096 and a.shape == b.shape:
            lib = native()
            if lib is not None:
                a2 = _c(a); b2 = _c(b); out = np.empty(a2.shape, dtype=np.uint64)
                lib.gl_add(ptr(a2), ptr(b2), ptr(out), a2.size); return out
        r = a + b
        return np.where((r < a) | (r >= _PU), r - _PU, r)

    @staticmethod
    def sub(a, b):
        a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
        if a.size >= 4096 and a.shape == b.shape:
            lib = native()
            if lib is not None:
                a2 = _c(a); b2 = _c(b); out = np.empty(a2.shape, dtype=np.uint64)
                lib.gl_sub(ptr(a2), ptr(b2), ptr(out), a2.size); return out
        return np.where(a >= b, a - b, a + (_PU - b))

    @staticmethod
    def fromi(v):
        """int64 (or Python ints / object) array -> field elements."""
        v = np.asarray(v)
        if v.dtype == object:
            return np.array([int(x) % P for x in v.ravel()], dtype=np.uint64).reshape(v.shape)
        v = v.astype(np.int64)
        return np.where(v >= 0, v.astype(np.uint64), _PU - (-v).astype(np.uint64))

    @staticmethod
    def bsum(a):
        lib = native()
        a = np.asarray(a, dtype=np.uint64)
        if lib is not None:
            a = _c(a).ravel()
            return int(lib.gl_sum(ptr(a), a.size))
        return int(a.astype(object).sum()) % P

    # -- extension field (3, n)
    @classmethod
    def emul(cls, a, b):
        a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
        lib = native()
        if lib is not None and a.shape == b.shape and a.ndim == 2 and a.shape[0] == 3:
            a = _c(a); b = _c(b); out = np.empty_like(a)
            lib.ext_mul(ptr(a), ptr(b), ptr(out), a.shape[1]); return out
        t0 = cls.add(cls.mul(a[1], b[2]), cls.mul(a[2], b[1]))
        c0 = cls.add(cls.mul(a[0], b[0]), cls.add(t0, t0))
        t1 = cls.mul(a[2], b[2])
        c1 = cls.add(cls.add(cls.mul(a[0], b[1]), cls.mul(a[1], b[0])), cls.add(t1, t1))
        c2 = cls.add(cls.add(cls.mul(a[0], b[2]), cls.mul(a[1], b[1])), cls.mul(a[2], b[0]))
        return np.stack([c0, c1, c2])

    @classmethod
    def eadd(cls, a, b):
        a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
        if a.shape == b.shape and a.ndim == 2 and a.shape[0] == 3:
            return cls.add(a, b)            # whole (3, n) block in one pass (limb-wise, identical values)
        return np.stack([cls.add(a[0], b[0]), cls.add(a[1], b[1]), cls.add(a[2], b[2])])

    @classmethod
    def esub(cls, a, b):
        a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
        if a.shape == b.shape and a.ndim == 2 and a.shape[0] == 3:
            return cls.sub(a, b)
        return np.stack([cls.sub(a[0], b[0]), cls.sub(a[1], b[1]), cls.sub(a[2], b[2])])

    @classmethod
    def efold(cls, col, r):
        """Multilinear halving of a (3, 2m) column: out[i] = col[2i] + r (col[2i+1] - col[2i])."""
        col = np.asarray(col, dtype=np.uint64)
        lib = native()
        if lib is not None and col.ndim == 2 and col.shape[0] == 3 and col.shape[1] >= 2:
            col = _c(col); out = np.empty((3, col.shape[1] // 2), dtype=np.uint64)
            lib.ext_fold(ptr(col), col.shape[1], ptr(_pts([r])), ptr(out)); return out
        lo, hi = col[:, 0::2], col[:, 1::2]
        return cls.eadd(lo, cls.emul(cls.esub(hi, lo), np.repeat(np.array(r, dtype=np.uint64).reshape(3, 1),
                                                                   lo.shape[1], axis=1)))

    @classmethod
    def escale(cls, a, c):
        """(3, n) times host scalar c."""
        c0, c1, c2 = np.uint64(c[0]), np.uint64(c[1]), np.uint64(c[2])
        if c1 == 0 and c2 == 0:
            return np.stack([cls.mul(a[0], c0), cls.mul(a[1], c0), cls.mul(a[2], c0)])
        t0 = cls.add(cls.mul(a[1], c2), cls.mul(a[2], c1))
        r0 = cls.add(cls.mul(a[0], c0), cls.add(t0, t0))
        t1 = cls.mul(a[2], c2)
        r1 = cls.add(cls.add(cls.mul(a[0], c1), cls.mul(a[1], c0)), cls.add(t1, t1))
        r2 = cls.add(cls.add(cls.mul(a[0], c2), cls.mul(a[1], c1)), cls.mul(a[2], c0))
        return np.stack([r0, r1, r2])

    @classmethod
    def eaddc(cls, a, c):
        return np.stack([cls.add(a[0], np.uint64(c[0])), cls.add(a[1], np.uint64(c[1])), cls.add(a[2], np.uint64(c[2]))])

    @staticmethod
    def esum(a):
        a = np.asarray(a, dtype=np.uint64)
        if a.ndim < 1 or a.shape[0] != 3:
            raise ValueError(f"extension array must have leading dimension 3, got {a.shape}")
        lib = native()
        if lib is not None:
            r0 = _c(a[0]); r1 = _c(a[1]); r2 = _c(a[2])
            return (int(lib.gl_sum(ptr(r0), r0.size)), int(lib.gl_sum(ptr(r1), r1.size)), int(lib.gl_sum(ptr(r2), r2.size)))
        return (int(a[0].astype(object).sum()) % P, int(a[1].astype(object).sum()) % P, int(a[2].astype(object).sum()) % P)

    @classmethod
    def eaxpy_(cls, dst, w, c):
        dst[...] = cls.eadd(dst, cls.escale(w, c))

    @classmethod
    def edot_dev(cls, a, w):
        a = np.asarray(a, dtype=np.uint64); w = np.asarray(w, dtype=np.uint64)
        lib = native()
        if lib is not None and a.shape == w.shape and a.ndim == 2 and a.shape[0] == 3:
            a = _c(a); w = _c(w); out = np.zeros(3, dtype=np.uint64)
            lib.ext_dot(ptr(a), ptr(w), a.shape[1], ptr(out)); return (int(out[0]), int(out[1]), int(out[2]))
        return cls.esum(cls.emul(a, w))

    @staticmethod
    def fetch_pairs(lst):
        return [tuple(x) for x in lst]

    @classmethod
    def epow(cls, a, e):
        r = None; b = a
        while e:
            if e & 1: r = b if r is None else cls.emul(r, b)
            e >>= 1
            if e: b = cls.emul(b, b)
        return r

    @classmethod
    def einv(cls, a):
        """Batch inverse in E = F_p[u]/(u^3 - 2) via the norm map N(a) and Montgomery batch inversion of the norms
        (native: one modular exponentiation per call; zero maps to zero)."""
        a1a2 = cls.mul(a[1], a[2])
        t0 = cls.sub(cls.mul(a[0], a[0]), cls.add(a1a2, a1a2))
        a2a2 = cls.mul(a[2], a[2])
        t1 = cls.sub(cls.add(a2a2, a2a2), cls.mul(a[0], a[1]))
        t2 = cls.sub(cls.mul(a[1], a[1]), cls.mul(a[0], a[2]))
        s = cls.add(cls.mul(a[2], t1), cls.mul(a[1], t2))
        nrm = cls.add(cls.mul(a[0], t0), cls.add(s, s))
        lib = native()
        if lib is not None:
            nrm = _c(nrm); inv = np.empty_like(nrm)
            lib.gl_batch_inv(ptr(nrm), ptr(inv), nrm.size)
        else:
            inv = np.array([pow(int(x), P - 2, P) if x else 0 for x in nrm], dtype=np.uint64)
        return np.stack([cls.mul(t0, inv), cls.mul(t1, inv), cls.mul(t2, inv)])

    @staticmethod
    def ext(a):
        a = np.asarray(a, dtype=np.uint64)
        z = np.zeros_like(a)
        return np.stack([a, z, z])

    @staticmethod
    def zeros_e(n): return np.zeros((3, n), dtype=np.uint64)

    @staticmethod
    def rand_e(n):
        return rand_field(3 * n).reshape(3, n)

    @staticmethod
    def asarray(a, dtype=np.uint64): return np.asarray(a, dtype=dtype)

    @staticmethod
    def tonumpy(a): return np.asarray(a)

    @classmethod
    def eq_table(cls, r):
        """eq(r, x) for all x (variable 0 = least significant bit)."""
        lib = native()
        if lib is not None and len(r) <= 30:
            pts = _pts(r)
            if len(pts) == 3 * len(r):
                out = np.empty((3, 1 << len(r)), dtype=np.uint64)
                lib.eq_table(ptr(pts), len(r), ptr(out)); return out
        t = np.array([[1], [0], [0]], dtype=np.uint64)
        for ri in r:
            a = cls.escale(t, hsub(ONE, ri)); b = cls.escale(t, ri)
            t = np.concatenate([a, b], axis=1)
        return t

    @classmethod
    def fold(cls, T, a):
        """T (3, n): out[i] = T[2i] + a (T[2i+1] - T[2i])."""
        lo, hi = T[:, 0::2], T[:, 1::2]
        return cls.eadd(lo, cls.escale(cls.esub(hi, lo), a))

    # -- NTT (radix 2, values at powers of the 2^k-th root of unity 7^((p-1)/n))
    _tw = {}

    @classmethod
    def ntt(cls, a, n):
        x = np.zeros(n, dtype=np.uint64); x[:len(a)] = a
        x = x[_bitrev(n)]
        if n not in cls._tw:
            cls._tw[n] = pow_table(pow(7, (P - 1) // n, P), n // 2)
        tw = cls._tw[n]; half = 1
        while half < n:
            x = x.reshape(-1, 2 * half)
            w = tw[::n // (2 * half)][:half]
            lo = x[:, :half].copy(); hi = cls.mul(x[:, half:], w[None, :])
            x[:, :half] = cls.add(lo, hi); x[:, half:] = cls.sub(lo, hi)
            x = x.reshape(-1); half *= 2
        return x

    @classmethod
    def encode(cls, f, rate):
        """f: (3, N) -> codeword (3, rate N)."""
        n0 = rate * f.shape[1]
        return np.stack([cls.ntt(f[l], n0) if (l == 0 or f[l].any()) else np.zeros(n0, dtype=np.uint64) for l in (0, 1, 2)])

    @classmethod
    def encode_host(cls, f, rate, out=None):
        c = cls.encode(np.asarray(f), rate)
        if out is None:
            return c
        out[...] = c
        return out

    @staticmethod
    def blake_rows(A):
        import hashlib
        A = np.ascontiguousarray(A, dtype=np.uint64)
        out = np.empty((A.shape[0], 4), dtype=np.uint64)
        for i in range(A.shape[0]):
            out[i] = np.frombuffer(hashlib.blake2b(A[i].tobytes(), digest_size=32).digest(), dtype=np.uint64)
        return out


_BR = {}


def pow_table(w, n):
    """[w^0, w^1, ..., w^(n-1)] (uint64), built by doubling with vectorised multiplications."""
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    lib = native()
    if lib is not None:
        out = np.empty(n, dtype=np.uint64)
        lib.gl_pow_table(int(w) % P, ptr(out), n)
        return out
    out = np.ones(1, dtype=np.uint64)
    while len(out) < n:
        step = pow(int(w), len(out), P)
        out = np.concatenate([out, CPU.mul(out, np.uint64(step))])
    return out[:n]


def _bitrev(n):
    """bit-reversal permutation of range(n), n a power of two (vectorised: built by doubling, O(n))."""
    if n not in _BR:
        out = np.zeros(1, dtype=np.int64)
        m = 1
        while m < n:
            out = np.concatenate([out * 2, out * 2 + 1]); m *= 2
        _BR[n] = out
    return _BR[n]


CPU = _NumpyBackend()

# ----------------------------------------------------------------------------- optional native accelerator
_NATIVE = False
_NATIVE_VERSION = 7              # == tg_version() in _fast_cpu.cpp
_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "_fast_cpu.so")


def _try_build():
    """Compile flashprove_next/_fast_cpu.so with the system C++ compiler (silently skipped if impossible)."""
    from .build_ext import compile_native
    try:
        return compile_native()[0]
    except Exception:
        return False


def native():
    """ctypes handle of flashprove_next/_fast_cpu.so, or None (pure NumPy fallback).  FPN_NO_NATIVE=1 (or FPN_NATIVE=0) disables
    it; an unbuilt / stale library is auto-built when a C++17 compiler is available (FPN_NO_BUILD=1 disables)."""
    global _NATIVE
    if _NATIVE is not False:
        return _NATIVE
    _NATIVE = None
    if os.environ.get("FPN_NATIVE", "1") == "0" or os.environ.get("FPN_NO_NATIVE", "") not in ("", "0"):
        return None
    import ctypes
    src = os.path.join(_HERE, "_fast_cpu.cpp")
    can_build = os.environ.get("FPN_NO_BUILD") != "1"
    stale = (not os.path.exists(_SO)) or (os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(_SO))
    if stale and can_build:
        _try_build()

    def load(path):
        if not os.path.exists(path):
            return None
        try:
            h = ctypes.CDLL(path)
            return h if h.tg_version() == _NATIVE_VERSION else None
        except (OSError, AttributeError):
            return None

    lib = load(_SO)
    if lib is None and can_build and not stale:
        # A library built for another platform / ABI (e.g. a macOS .so copied to Linux: "invalid ELF header") or an
        # older version with a newer mtime: rebuild from source once instead of silently using the NumPy fallback.
        # The rebuilt file is loaded through a private copy (the dynamic loader caches an already-mapped path).
        if _try_build():
            import shutil
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".so")
            os.close(fd)
            try:
                shutil.copyfile(_SO, tmp)
                lib = load(tmp)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    if lib is None:
        return None
    u64p = ctypes.POINTER(ctypes.c_uint64); L = ctypes.c_long; U = ctypes.c_uint64
    for nm in ("gl_mul", "gl_add", "gl_sub", "ext_mul"):
        getattr(lib, nm).argtypes = [u64p, u64p, u64p, L]
    lib.gl_sum.argtypes = [u64p, L]; lib.gl_sum.restype = U
    lib.gl_batch_inv.argtypes = [u64p, u64p, L]
    lib.gl_pow_table.argtypes = [U, u64p, L]
    lib.ext_dot.argtypes = [u64p, u64p, L, u64p]
    lib.base_ext_dot.argtypes = [u64p, u64p, L, u64p]
    lib.eq_table.argtypes = [u64p, ctypes.c_int, u64p]
    lib.eq_pt.argtypes = [u64p, u64p, ctypes.c_int, u64p]
    lib.fold_full.argtypes = [u64p, L, U, U, U, U, u64p]
    lib.fold_coset.argtypes = [u64p, ctypes.c_int, L, L, U, u64p, ctypes.c_int]
    lib.p2_init.argtypes = [u64p, u64p]
    lib.p2_permute.argtypes = [u64p, L, u64p]
    lib.fpn_set_threads.argtypes = [ctypes.c_int]; lib.fpn_set_threads.restype = ctypes.c_int
    lib.fpn_set_threads(native_threads())
    lib.ext_fold.argtypes = [u64p, L, u64p, u64p]
    lib.ext_scale.argtypes = [u64p, L, u64p, u64p]
    lib.gkr_round.argtypes = [u64p, u64p, u64p, u64p, u64p, L, u64p, u64p]
    _NATIVE = lib
    return lib


def effective_cpus():
    """CPUs this process may actually use: the scheduler affinity mask, capped by a cgroup CPU quota (containers often
    expose every host core in /proc while throttling to a fraction of them)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as f:
                parts = f.read().split()
            if path.endswith("cpu.max"):
                quota, period = parts[0], parts[1]
            else:
                quota = parts[0]
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    period = f.read().split()[0]
            if quota not in ("max", "-1"):
                n = min(n, max(1, int(quota) // int(period)))
            break
        except (OSError, ValueError, IndexError):
            continue
    return n


def native_threads():
    """OpenMP team size of the native library: $FPN_NATIVE_THREADS, else min(16, effective CPUs)."""
    env = os.environ.get("FPN_NATIVE_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(16, effective_cpus()))


def ptr(a):
    import ctypes
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _c(a):
    return np.ascontiguousarray(a, dtype=np.uint64)


def _pts(r):
    return np.array([int(x) % P for z in r for x in z], dtype=np.uint64)

_GPU = None


def gpu_backend():
    """CuPy back end (imported lazily)."""
    global _GPU
    if _GPU is None:
        from . import field_gpu
        _GPU = field_gpu.GpuBackend()
    return _GPU


def backend(name=None):
    """Back end by name ("cpu" | "numpy" | "cupy"); default $FPN_BACKEND, else "cpu"."""
    name = name or os.environ.get("FPN_BACKEND", "cpu")
    return CPU if name in ("numpy", "cpu") else gpu_backend()
