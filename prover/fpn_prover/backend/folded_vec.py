# SPDX-License-Identifier: Apache-2.0
"""End-to-end prover of `FPN2` relation 5 (`FOLDED_VEC_MATMUL`, specs/11-wire-format.md §5.6).

Statement (public: the weight commitment `weight_root`, pinned by the verifier's caller): for the public
integer-unit weight tiles W[b] (B tiles of R = 32 rows x K columns, F_p values) committed under `weight_root`,
the prover knows activation vectors X[b] in F_p^K and outputs Y[b] in F_p^R (committed under `root_0`) with

        Y[b][i] = sum_j W[b][i][j] X[b][j]   over F_p,   for every tile b < B and row i < R.

Geometry is a pure function of the profile: n = n_vars, K = max_hidden (a power of two), R = 32,
B = 2^(n - 1 - 5 - log2 K) -- the B * R * K weight cells exactly fill the data half of the public vector C.

Protocol (all challenges from the raw-byte transcript, after both roots are absorbed):
  * row point r_row = (r_i in E^5, r_b in E^beta);
  * ONE Libra-masked inner-product sumcheck over v = (j, b) in {0,1}^(k + beta) of
        P(v) = eq(r_b, b) * ( W~(b, r_i, j) X(b, j) - eq(0, j_hi) eq(r_i, j_lo) Y(b, j_lo) ) + lambda G(v)
    with claimed sum lambda * G_sum (the zero-claim form: the verifier never learns Y~(r_row));
    per-variable degree 3 (eq * W * X in the tile bits), 4 samples per round;
  * the prover reveals ONLY e_W = W~(s_b, r_i, s_j) (a function of the public W); X~(s), Y~(s) and G(s) stay
    inside ONE batched linear claim  y = target + mu e_W + mu^2 z_blind = sum F a + sum C c,
  * discharged by ONE batched folded ZK-Basefold opening of (F, C) -- byte-for-byte relation 4's `pair_open`
    (two batching rounds, then the fused coset-4 fold chain of H = F + theta C).

`engine=None` runs the PCS on the pure-Python reference (`folded_pcs.zk_commit` + `folded_air.pair_open`);
`engine=CudaEngine()/HostEngine()` runs commitment, batched sumcheck and fold chain on the kernels.  The witness
sampling, inner-product sumcheck and weight vectors are shared NumPy code, so both paths give identical bytes for
the same rng stream.

Scope: no range check (the statement is over F_p; integer faithfulness needs |sum| < p/2, which the callers check
and report); `max_hidden` is the only architecture-like profile field and is a public scale bound.
"""
import hashlib
import secrets
import time
from dataclasses import dataclass

import numpy as np

from flashprove_next.field import CPU as X
from flashprove_next.field import ONE, P, ZERO, hadd, hf, hinv, hmul, hsub, lagrange_eval, rand_field

from . import folded_air as FA
from . import folded_pcs as pcs
from . import frac_logup_gkr as gkr
from .fpn2 import FOLDED_VEC_N16_V1, Fpn2Transcript, deposit, enc3, enc3s, header, profile_id

ROWS = 32                 # R: rows per GEMM tile (fixed by the relation)
ROW_BITS = 5
IP_DEGREE = 3             # inner-product round degree: eq(r_b, b) * W * X in the tile bits
DATA_FIXED = ((1, 1),)    # data half of F / C: index bit 1 == 1 (as relation 4)
LIBRA_FIXED = ((1, 0),)   # Libra + blinding half


def free_bits(n):
    return [0] + list(range(2, n))


# ----------------------------------------------------------------------------- layout (§5.6.2)
@dataclass(frozen=True)
class VecLayout:
    """Relation-5 geometry, a pure function of (n, Q, k = log2 K).

    Data coordinates (n - 1 bits -> cell bits 0, 2, 3, ..., n-1 of the half {bit1 = 1}):
        F: [0, B K) X  (index b K + j)  ·  [B K, B K + B R) Y  (index B K + b R + i)  ·  rest uniform spare;
           half {bit1 = 0}: Libra coefficients (slot 4 l + e, l < d, e < 4), rest uniform blinding.
        C: [0, B R K) W  (index (b R + i) K + j) -- the whole data half; the other half is 0.
    """
    n: int
    queries: int
    k: int

    def __post_init__(self):
        if self.n < 12 or self.n % 2 or self.n > 28 or not 1 <= self.queries <= 4096:
            raise ValueError("relation 5 needs even n in [12, 28] and 1 <= Q <= 4096")
        if self.k < ROW_BITS or self.beta < 0:
            raise ValueError("relation 5 needs 32 <= K and B = 2^(n - 6 - log2 K) >= 1")

    @property
    def K(self): return 1 << self.k

    @property
    def beta(self): return self.n - 1 - ROW_BITS - self.k

    @property
    def B(self): return 1 << self.beta

    @property
    def d(self): return self.k + self.beta            # inner-product sumcheck variables (j low, then b)

    @property
    def macs(self): return self.B * ROWS * self.K

    @property
    def folds(self): return self.n // 2

    def depth(self, t): return self.n + 1 - 2 * t

    @property
    def off_y(self): return self.B * self.K

    def cell(self, data_index):
        return deposit(data_index, DATA_FIXED, free_bits(self.n))

    def data_block(self, off, kk):
        fb = free_bits(self.n)
        fixed = DATA_FIXED + tuple((fb[kk + i], (off >> (kk + i)) & 1) for i in range(len(fb) - kk))
        return fixed, tuple(fb[:kk])

    def libra_slot(self, l, e):
        return deposit(4 * l + e, LIBRA_FIXED, free_bits(self.n))

    # wire sizes (§5.6.6)
    STMT = 96

    @property
    def ip_len(self): return 24 + 24 * (IP_DEGREE + 1) * self.d + 24

    @property
    def pcs_len(self): return 24 + 72 * (self.n + 2) + 32 * (self.folds - 1) + 24

    @property
    def query_len(self):
        return (96 + 32 * self.depth(0)) + sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return self.STMT + self.ip_len + self.pcs_len + self.queries * self.query_len


def layout_of(profile):
    k = profile["max_hidden"].bit_length() - 1
    if profile["max_hidden"] != 1 << k or profile["sc_degree"] != IP_DEGREE:
        raise ValueError("relation 5 needs a power-of-two max_hidden and sc_degree = 3")
    return VecLayout(profile["n_vars"], profile["n_queries"], k)


def cell_np(data_index):
    return FA.cell_np(data_index)


# ----------------------------------------------------------------------------- field encodings
def to_field(values):
    """Signed integers (int64 or object array) -> canonical F_p uint64 array of the same shape."""
    a = np.asarray(values)
    if a.dtype == object:
        return np.array([int(v) % P for v in a.ravel()], dtype=np.uint64).reshape(a.shape)
    return X.fromi(a.astype(np.int64))


def exact_products(W_int, X_int):
    """Exact integer Y[b][i] = sum_j W[b][i][j] X[b][j] (Python ints, object array (B, R)) and the bit width of
    sum_j |W| |X| (integer faithfulness of the F_p statement needs that bound < log2(p) - 1)."""
    Wo = np.asarray(W_int, dtype=object)
    Xo = np.asarray(X_int, dtype=object)
    Y = np.einsum("bij,bj->bi", Wo, Xo)
    bound = np.einsum("bij,bj->bi", np.abs(Wo), np.abs(Xo))
    return Y, max(int(v) for v in bound.ravel()).bit_length()


def bf16_words_to_ints(words):
    """BF16 bit patterns -> (exact signed integers, shift) with value = int * 2^-shift (one shift per array).

    A finite BF16 value is (-1)^s m 2^(e - 134) with m the 8-bit significand (implicit 1 for normals), so every value
    of the array is an exact integer multiple of 2^-shift with shift = 134 - min(e) over nonzero entries."""
    w = np.asarray(words, dtype=np.uint32).ravel()
    exp = (w >> 7) & 0xFF
    if (exp == 0xFF).any():
        raise ValueError("BF16 inf / nan has no integer encoding")
    man = (w & 0x7F) | np.where(exp > 0, 0x80, 0).astype(np.uint32)
    e = np.where(exp > 0, exp, 1).astype(np.int64)
    nz = man != 0
    emin = int(e[nz].min()) if nz.any() else 134
    shift = np.where(nz, e - emin, 0)
    sign = ((w >> 15) & 1).astype(bool)
    if int(shift.max(initial=0)) <= 54:                     # 8-bit significand << 54 still fits int64
        mag = man.astype(np.int64) << shift
        out = np.where(sign, -mag, mag)
    else:
        out = np.empty(len(w), dtype=object)
        for idx in range(len(w)):
            v = int(man[idx]) << int(shift[idx])
            out[idx] = -v if sign[idx] else v
    return out.reshape(np.shape(words)), 134 - emin


# ----------------------------------------------------------------------------- randomness
def _aes256_ctr_bytes(key32: bytes, nonce16: bytes, nbytes: int) -> bytes:
    """Fast cryptographic stream via OpenSSL AES-256-CTR (AES-NI ~8 GB/s when `cryptography` is installed),
    falling back cleanly to BLAKE2b-512 counter mode (zero external dependency)."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        enc = Cipher(algorithms.AES(key32), modes.CTR(nonce16)).encryptor()
        return enc.update(bytes(nbytes)) + enc.finalize()
    except ImportError:
        import hashlib
        out = bytearray(nbytes)
        pos = 0
        ctr = 0
        while pos < nbytes:
            blk = hashlib.blake2b(nonce16 + ctr.to_bytes(8, "little"), key=key32, digest_size=64).digest()
            take = min(64, nbytes - pos)
            out[pos:pos + take] = blk[:take]
            pos += take
            ctr += 1
        return bytes(out)


def _rand_field_from_seed(seed32: bytes, n: int) -> np.ndarray:
    """Deterministic uniform F_p words derived from a 32-byte ephemeral CSPRNG key via AES-256-CTR
    with rejection sampling (canonical < P)."""
    if len(seed32) != 32:
        raise ValueError("seed32 must be 32 bytes")
    nonce0 = b"FPN2-CSPRNG-0000"
    raw = _aes256_ctr_bytes(seed32, nonce0, 8 * n)
    out = np.frombuffer(raw, dtype="<u8").copy()
    bad = np.flatnonzero(out >= P)
    ctr = 1
    while len(bad):
        nonce_k = b"FPN2-CSPRNG-" + ctr.to_bytes(4, "little")
        refill = np.frombuffer(_aes256_ctr_bytes(seed32, nonce_k, 8 * len(bad)), dtype="<u8")
        out[bad] = refill
        bad = bad[out[bad] >= P]
        ctr += 1
    return out


def _os_rand_field(n):
    """Rejection-sampled uniform F_p words exclusively from OS CSPRNG (never honors FPN_TEST_RNG_SEED).

    Uses a fresh ephemeral 256-bit `secrets.token_bytes(32)` key expanded via OpenSSL AES-256-CTR for large
    draws (`n >= 4096`) when `cryptography` is installed (~20x faster than raw `/dev/urandom` syscalls at
    multi-hundred-MB scale while preserving 256-bit IND-CPA unpredictability); small draws use `secrets.token_bytes`.
    """
    if n >= 4096:
        try:
            import cryptography  # noqa: F401
            return _rand_field_from_seed(secrets.token_bytes(32), n)
        except ImportError:
            pass
    out = np.frombuffer(secrets.token_bytes(8 * n), dtype="<u8").copy()
    bad = np.flatnonzero(out >= P)
    while len(bad):
        out[bad] = np.frombuffer(secrets.token_bytes(8 * len(bad)), dtype="<u8")
        bad = bad[out[bad] >= P]
    return out


class EphemeralSeedRng:
    """Per-chunk 256-bit OS-CSPRNG-keyed AES-256-CTR stream.

    Allows `commit_x_chunks` to commit a chunk, keep only `(seed32, X_c)` (512 KB instead of 100 MB per chunk),
    and deterministically reconstruct the exact same blinded cells ONLY for the single challenged chunk `c*`.
    """

    def __init__(self, seed32: bytes | None = None):
        self.seed32 = secrets.token_bytes(32) if seed32 is None else bytes(seed32)
        if len(self.seed32) != 32:
            raise ValueError("seed32 must be 32 bytes")
        self._call = 0

    def fp3_array(self, count):
        import hashlib
        subkey = hashlib.blake2b(self._call.to_bytes(8, "little"), key=self.seed32, digest_size=32).digest()
        self._call += 1
        return _rand_field_from_seed(subkey, 3 * count).reshape(count, 3)


class SecretRng:
    """OS randomness; `fp3_array` draws uniform F_p3 cells in bulk exclusively via OS CSPRNG."""

    def fp3(self):
        return tuple(secrets.randbelow(P) for _ in range(3))

    def fp3_array(self, count):
        return _os_rand_field(3 * count).reshape(count, 3)


def draw_cells(rng, count):
    """`count` uniform F_p3 cells as a (count, 3) uint64 array: rng.fp3_array if the rng has it, else count
    successive rng.fp3() draws.  This is the ONLY randomness of the relation-5 prover (draw order = parity)."""
    if hasattr(rng, "fp3_array"):
        return np.ascontiguousarray(np.asarray(rng.fp3_array(count), dtype=np.uint64).reshape(count, 3))
    return np.array([rng.fp3() for _ in range(count)], dtype=np.uint64).reshape(count, 3)


# ----------------------------------------------------------------------------- vectors F and C
def _check_tiles(layout, W, Xv, Y):
    B, K = layout.B, layout.K
    if W is not None and np.shape(W) != (B, ROWS, K):
        raise ValueError(f"W must have shape ({B}, {ROWS}, {K})")
    if Xv is not None and np.shape(Xv) != (B, K):
        raise ValueError(f"X must have shape ({B}, {K})")
    if Y is not None and np.shape(Y) != (B, ROWS):
        raise ValueError(f"Y must have shape ({B}, {ROWS})")
    for a in (W, Xv, Y):
        if a is not None and (np.asarray(a).dtype != np.uint64 or (np.asarray(a) >= P).any()):
            raise ValueError("tiles must be canonical F_p uint64 arrays (use to_field)")


def weight_cells_np(layout, W):
    _check_tiles(layout, W, None, None)
    out = np.zeros((1 << layout.n, 3), dtype=np.uint64)
    out[cell_np(np.arange(layout.macs, dtype=np.int64)), 0] = np.asarray(W, dtype=np.uint64).reshape(-1)
    return out


_W_CACHE = {}


def commit_weights(layout, W, engine=None):
    """Deterministic commitment of C (cached per (n, W, path)): `EngineCommitment` with an engine, else the
    pure-Python `folded_pcs.Commitment`.  Both have the same root."""
    key = (layout.n, layout.k, hashlib.blake2b(np.ascontiguousarray(W).tobytes(), digest_size=32).digest(),
           type(engine).__name__, getattr(engine, "device_id", None))
    if key not in _W_CACHE:
        cells = weight_cells_np(layout, W)
        if engine is not None:
            _W_CACHE.clear()
            _W_CACHE[key] = FA.engine_commit(engine, cells)
        else:
            _W_CACHE[key] = pcs.zk_commit([tuple(int(v) for v in row) for row in cells])
    return _W_CACHE[key]


def weight_root(layout, W, engine=None):
    return commit_weights(layout, W, engine).root


def build_witness_np(layout, Xv, Y, rng):
    """F cells (2^n, 3): one bulk uniform draw (blinding / Libra / spare), then the X and Y cells."""
    _check_tiles(layout, None, Xv, Y)
    cells = draw_cells(rng, 1 << layout.n)
    xi = cell_np(np.arange(layout.B * layout.K, dtype=np.int64))
    cells[xi, 0] = np.asarray(Xv, dtype=np.uint64).reshape(-1)
    cells[xi, 1:] = 0
    yi = cell_np(layout.off_y + np.arange(layout.B * ROWS, dtype=np.int64))
    cells[yi, 0] = np.asarray(Y, dtype=np.uint64).reshape(-1)
    cells[yi, 1:] = 0
    return cells


# ----------------------------------------------------------------------------- inner-product sumcheck (§5.6.3)
def _tup(col):
    return tuple(int(v) for v in col)


def ip_columns(layout, Fa, W, r_i, r_b):
    """Columns over v = b K + j: eq(r_b, b), U = W~(b, r_i, j), X(b, j), E = eq(0, j_hi) eq(r_i, j_lo), Y(b, j_lo)."""
    B, K = layout.B, layout.K
    v = np.arange(B * K, dtype=np.int64)
    eq_i = X.eq_table(list(r_i))                                    # (3, 32)
    eq_b = X.eq_table(list(r_b)) if r_b else np.array([[1], [0], [0]], dtype=np.uint64)
    Wt = np.asarray(W, dtype=np.uint64)
    U = X.zeros_e(B * K)
    for i in range(ROWS):
        U = X.eadd(U, X.escale(X.ext(np.ascontiguousarray(Wt[:, i, :]).reshape(-1)), _tup(eq_i[:, i])))
    E = X.zeros_e(K)
    E[:, :ROWS] = eq_i
    cols = [np.ascontiguousarray(np.repeat(eq_b, K, axis=1)), U,
            np.ascontiguousarray(Fa[:, cell_np(v)]),
            np.ascontiguousarray(np.tile(E, (1, B))),
            np.ascontiguousarray(Fa[:, cell_np(layout.off_y + (v // K) * ROWS + (v % K) % ROWS)])]
    return cols


def ip_prove(tr, layout, Fa, W, hooks=None):
    """Libra-masked inner-product sumcheck.  Returns (wire, r_i, r_b, s, lam, e_W, libra coefficients)."""
    hooks = hooks or {}
    r_i = [tr.squeeze_fp3() for _ in range(ROW_BITS)]
    r_b = [tr.squeeze_fp3() for _ in range(layout.beta)]
    g = [tuple(_tup(Fa[:, layout.libra_slot(l, e)]) for e in range(IP_DEGREE + 1)) for l in range(layout.d)]
    g_sum = gkr.libra_sum(g)
    tr.absorb_bytes(enc3(g_sum))
    lam = tr.squeeze_fp3()
    cols = ip_columns(layout, Fa, W, r_i, r_b)
    wire = enc3(g_sum)
    rs = []
    target = hmul(lam, g_sum)                   # only the adaptive forgery below tracks it
    half = hinv(hf(2))
    for l in range(layout.d):
        folds = [FA._small_folds(col, IP_DEGREE + 1) for col in cols]
        samples = []
        for t in range(IP_DEGREE + 1):
            eb, u, x, e, y = (f[t] for f in folds)
            val = X.esum(X.emul(eb, X.esub(X.emul(u, x), X.emul(e, y))))
            samples.append(hadd(val, hmul(lam, gkr.libra_round(g, l, rs, hf(t)))))
        if "ip_round" in hooks:
            samples = hooks["ip_round"](l, samples)
        if hooks.get("ip_adaptive"):           # white-box forgery: shift every round so its sum check passes
            delta = hmul(hsub(target, hadd(samples[0], samples[1])), half)
            samples = [hadd(v, delta) for v in samples]
        tr.absorb_bytes(enc3s(samples))
        wire += enc3s(samples)
        r = tr.squeeze_fp3()
        rs.append(r)
        target = lagrange_eval(samples, r)
        cols = [FA._fold(col, r) for col in cols]
    e_w = _tup(cols[1][:, 0])
    if "e_w" in hooks:
        e_w = hooks["e_w"](e_w)
    tr.absorb_bytes(enc3(e_w))
    wire += enc3(e_w)
    return wire, r_i, r_b, tuple(rs), lam, e_w, g


def batch_terms(layout, r_i, r_b, s, lam, e_w, mu):
    """(F-blocks, F-sparse, C-blocks) of  target + mu e_W + mu^2 z_blind = sum F a + sum C c  (§5.6.4)."""
    k, beta = layout.k, layout.beta
    s_j, s_b = list(s[:k]), list(s[k:])
    eqrb = FA.eq_pt(r_b, s_b)
    e_s = FA.eq_pt(r_i, s_j[:ROW_BITS])
    for t in s_j[ROW_BITS:]:
        e_s = hmul(e_s, hsub(ONE, t))
    fb = [(*layout.data_block(0, layout.d), hmul(eqrb, e_w), tuple(s)),
          (*layout.data_block(layout.off_y, ROW_BITS + beta), hsub(ZERO, hmul(eqrb, e_s)),
           tuple(s_j[:ROW_BITS]) + tuple(s_b))]
    sparse = []
    for l, r in enumerate(s):
        pw = lam
        for e in range(IP_DEGREE + 1):
            sparse.append((layout.libra_slot(l, e), pw))
            pw = hmul(pw, r)
    fb += FA.mask_blocks(layout, hmul(mu, mu))
    cb = [(*layout.data_block(0, layout.n - 1), mu, tuple(s_j) + tuple(r_i) + tuple(s_b))]
    return fb, sparse, cb


# ----------------------------------------------------------------------------- prover
def prove(W, Xv, Y, n_s=b"\x00" * 32, profile=FOLDED_VEC_N16_V1, rng=None, hooks=None, engine=None):
    """Exact C_p-byte FPN2 relation-5 container for weight tiles W (B, 32, K), activations X (B, K) and outputs
    Y (B, 32), all canonical F_p uint64 arrays.  The prover does NOT check Y = W X: a false statement yields bytes
    the verifiers reject.

    hooks (tests / adversarial vectors): stmt_weight_root -> bytes written as weight_root (the proof still uses W);
    ip_round(l, samples) / e_w(value) -> tamper the inner-product messages; ip_adaptive -> shift every round by a
    constant so each round-sum check passes for a FALSE statement (the folded-in final claim must then fail); pcs -> `pair_open` hooks (CPU only);
    record (dict) -> filled with internals; timings (dict) -> per-phase milliseconds."""
    layout = layout_of(profile)
    rng = rng or SecretRng()
    hooks = hooks or {}
    if len(n_s) != 32:
        raise ValueError("N_s must be 32 bytes")
    if engine is not None and "pcs" in hooks:
        raise ValueError("pcs tamper hooks are CPU-only (engine=None)")
    W = np.asarray(W, dtype=np.uint64)
    _check_tiles(layout, W, np.asarray(Xv, dtype=np.uint64), np.asarray(Y, dtype=np.uint64))
    tm = hooks.get("timings", {})
    clock = [time.perf_counter()]

    def lap(name):
        if engine is not None and hasattr(engine, "sync"):
            engine.sync()
        now = time.perf_counter()
        tm[name] = tm.get(name, 0.0) + (now - clock[0]) * 1e3
        clock[0] = now

    comC = commit_weights(layout, W, engine)
    lap("commit_weights")
    F = build_witness_np(layout, np.asarray(Xv, dtype=np.uint64), np.asarray(Y, dtype=np.uint64), rng)
    lap("witness_sample")
    if engine is not None:
        comF = FA.engine_commit(engine, F)
    else:
        comF = pcs.zk_commit([tuple(int(v) for v in row) for row in F])
    lap("commit_witness")
    Fa = np.ascontiguousarray(F.T)
    hdr = header(profile_id(profile))
    stmt = comF.root + hooks.get("stmt_weight_root", comC.root) + bytes(n_s)
    tr = Fpn2Transcript()
    tr.absorb_bytes(hdr)
    tr.absorb_bytes(stmt)
    ip_wire, r_i, r_b, s, lam, e_w, _ = ip_prove(tr, layout, Fa, W, hooks)
    lap("ip_sumcheck")
    zb = FA.z_blind_np(layout, Fa)
    tr.absorb_bytes(enc3(zb))
    mu = tr.squeeze_fp3()
    fb, sparse, cb = batch_terms(layout, r_i, r_b, s, lam, e_w, mu)
    a_vec = FA.weights_vector_np(layout.n, fb, sparse)
    c_vec = FA.weights_vector_np(layout.n, cb)
    lap("batch_weights")
    rec = {}
    if engine is not None:
        pcs_wire, rs, final = FA.pair_open_engine(engine, comF, comC, a_vec, c_vec, tr, layout.queries, rec)
    else:
        pcs_wire, rs, final = FA.pair_open(comF, comC, [_tup(v) for v in a_vec.T], [_tup(v) for v in c_vec.T],
                                           tr, layout.queries, hooks.get("pcs"), rec)
    lap("pair_open")
    if "record" in hooks:
        hooks["record"].update(F=F, a=a_vec, c=c_vec, rs=rs, s=s, r_i=r_i, r_b=r_b, lam=lam, mu=mu, e_w=e_w,
                               z_blind=zb, final=final, positions=rec.get("positions"))
    body = stmt + ip_wire + enc3(zb) + pcs_wire
    if len(body) != layout.body_used:
        raise AssertionError(f"layout mismatch {len(body)} != {layout.body_used}")
    out = hdr + body
    if len(out) > profile["c_p"]:
        raise ValueError("proof exceeds C_p")
    return out + b"\x00" * (profile["c_p"] - len(out))


def byte_budget(n, queries, k=10):
    lay = VecLayout(n, queries, k)
    return {"n_vars": n, "queries": queries, "K": lay.K, "tiles": lay.B, "rows": lay.B * ROWS, "macs": lay.macs,
            "body": lay.body_used, "container_min": 64 + lay.body_used}


# ----------------------------------------------------------------------------- ZK rank condition 
# The honest prover's LINEAR view of relation 5.  W is PUBLIC, so for fixed challenges every plaintext value of the
# container is an F_p3-linear functional of the committed cells F (public C terms dropped):
#   claims : the Libra sum g_sum; ALL 4 d inner-product round samples h_l(t) (P = eq(r_b) (W~ X - E Y) is linear in
#            F once W is fixed, and the Libra term is linear in the Libra slots); z_blind.  (e_W = W~(s) is a function
#            of the public W only and carries no witness information.)
#   pcs    : the 3 (n + 2) batched sumcheck samples h_i(0..2) and H~(r); per query the 4 layer-0 coset values of F's
#            codeword and the 4 coset values of every folded layer of H = F + theta C (`folded_air.pcs_functionals`).
# Coordinates: X(b, j) and Y(b, i) are F_p coordinates; every Libra / blinding / spare cell is a uniform F_p3
# coordinate (3 F_p columns).  `require_joint_mask_vec` checks im(A) <= im(B) over F_p (rank_B == rank_BA).
#
# Witness model.  `witness="honest"` (default) parametrises A by the honest witness manifold Y = W X: one coordinate
# per X(b, j) whose column is X's cell column plus sum_i W[b][i][j] times Y(b, i)'s cell column.  `witness="free"`
# treats X and Y as independent; it MUST fail by exactly one F_p3 observation (3 F_p ranks): the zero-claim
# h_0(0) + h_0(1) - lam g_sum = sum_b eq(r_b, b) ((W X)~ - Y~)(b, r_i) is mask-free, i.e. the proof reveals the
# statement residual, which is 0 for every honest witness and therefore no information about it.
def _sparse_dense(n, sparse):
    return FA.weights_vector_np(n, [], sparse)


def _fp_scale(row, c):
    """(3, m) F_p3 row times a base-field scalar / vector."""
    return np.stack([X.mul(row[i], c) for i in range(3)])


def _libra_round_sparse(layout, l, s, lam, t):
    d = layout.d
    sp = []
    c_hi = hmul(lam, hf(pow(2, d - 1 - l, P)))
    for m in range(l):
        pw = ONE
        for e in range(IP_DEGREE + 1):
            sp.append((layout.libra_slot(m, e), hmul(c_hi, pw)))
            pw = hmul(pw, s[m])
    tt = hf(t)
    pw = ONE
    for e in range(IP_DEGREE + 1):
        sp.append((layout.libra_slot(l, e), hmul(c_hi, pw)))
        pw = hmul(pw, tt)
    if l < d - 1:
        c_lo = hmul(lam, hf(pow(2, d - 2 - l, P)))
        for m in range(l + 1, d):
            sp.append((layout.libra_slot(m, 0), hmul(c_lo, hf(2))))
            for e in range(1, IP_DEGREE + 1):
                sp.append((layout.libra_slot(m, e), c_lo))
    return sp


def ip_sample_rows(layout, rec, W):
    """Yield (label, dense (3, 2^n) row) for every inner-product round sample h_l(t), l < d, t < 4."""
    B, K, d = layout.B, layout.K, layout.d
    r_i, r_b, s, lam = rec["r_i"], rec["r_b"], rec["s"], rec["lam"]
    dummy = X.zeros_e(1 << layout.n)
    cols = ip_columns(layout, dummy, W, r_i, r_b)
    pub = [cols[0], cols[1], cols[3]]                       # eq(r_b, b), U = W~(b, r_i, j), E
    xcells = cell_np(np.arange(B * K, dtype=np.int64))
    ycells = cell_np(layout.off_y + np.arange(B * ROWS, dtype=np.int64))
    for l in range(d):
        eqlow = X.eq_table(list(s[:l])) if l else np.array([[1], [0], [0]], dtype=np.uint64)
        folds = [FA._small_folds(c, IP_DEGREE + 1) for c in pub]
        rest = 1 << (d - l - 1)
        for t in range(IP_DEGREE + 1):
            eb, u, e = (f[t] for f in folds)
            a_x = X.emul(eb, u)
            a_y = X.esub(X.zeros_e(rest), X.emul(eb, e))
            lt = np.array([(1 - t) % P] * (1 << l) + [t % P] * (1 << l), dtype=np.uint64)
            lt = np.tile(lt, rest)
            lowrep = np.tile(eqlow, (1, 2 * rest))
            cx = _fp_scale(X.emul(lowrep, np.repeat(a_x, 2 << l, axis=1)), lt)
            cy = _fp_scale(X.emul(lowrep, np.repeat(a_y, 2 << l, axis=1)), lt)
            row = _sparse_dense(layout.n, _libra_round_sparse(layout, l, s, lam, t))
            row[:, xcells] = X.eadd(row[:, xcells], cx)
            cy = cy.reshape(3, B, K // ROWS, ROWS)
            acc = cy[:, :, 0, :]
            for q in range(1, K // ROWS):
                acc = np.stack([X.add(acc[i], cy[i, :, q, :]) for i in range(3)])
            row[:, ycells] = X.eadd(row[:, ycells], np.ascontiguousarray(acc.reshape(3, B * ROWS)))
            yield f"ip_round_{l}_{t}", row
        pub = [FA._fold(c, s[l]) for c in pub]


def vec_claim_rows(layout, rec, W):
    """[(label, dense row)]: g_sum, the 4 d inner-product samples, z_blind (all linear in F)."""
    d = layout.d
    slot = lambda i, e: layout.libra_slot(i, e)  # noqa: E731
    yield "libra_sum", _sparse_dense(layout.n, FA._sparse_libra_sum(slot, d, IP_DEGREE + 1, pow(2, d - 1, P)))
    yield from ip_sample_rows(layout, rec, W)
    yield "z_blind", FA.weights_vector_np(layout.n, FA.mask_blocks(layout, ONE))


def vec_claim_values(layout, data):
    """The plaintext claim values parsed from a relation-5 container: [g_sum, h_l(t) ..., z_blind]."""
    from .fpn2 import dec3
    off = 64 + layout.STMT
    vals = [dec3(bytes(data[off:off + 24]))]
    off += 24
    for _ in range(layout.d * (IP_DEGREE + 1)):
        vals.append(dec3(bytes(data[off:off + 24])))
        off += 24
    off += 24                                               # e_W (public)
    vals.append(dec3(bytes(data[off:off + 24])))
    return vals


def pcs_rows_np(layout, rec):
    """NumPy port of `folded_air.pcs_functionals` (identical rows; cross-checked in the unit tests)."""
    n, size = layout.n, 1 << layout.n
    a = np.asarray(rec["a"], dtype=np.uint64)
    c = np.asarray(rec["c"], dtype=np.uint64)
    if a.shape[0] != 3:
        a, c = a.T, c.T
    N = n + 2
    wv = X.zeros_e(4 * size)
    wv[:, 0::4] = X.esub(a, c)
    wv[:, 1::4] = c
    rs = list(rec["rs"])

    def to_f(g):
        g = g.reshape(3, size, 4)
        acc = g[:, :, 0]
        for k in range(1, 4):
            acc = X.eadd(acc, np.ascontiguousarray(g[:, :, k]))
        return acc

    b = np.arange(1 << N, dtype=np.int64)
    for i in range(N):
        lo, hi = wv[:, 0::2], wv[:, 1::2]
        step = X.esub(hi, lo)
        high = lo
        for t in range(3):
            low = X.eq_table(list(rs[:i]) + [hf(t)])
            g = X.emul(low[:, b & ((2 << i) - 1)], high[:, b >> (i + 1)])
            yield f"pcs_round_{i}_{t}", to_f(g)
            high = X.eadd(high, step)
        wv = FA._fold(wv, rs[i])
    yield "pcs_final", to_f(X.eq_table(rs))
    rest = rs[2:]
    length = pcs.RATE << n
    omega = pcs.root_of_unity(length)
    seen = set()
    for pos in rec["positions"]:
        for t in range(layout.folds):
            quarter = length >> (2 * t + 2)
            j = pos % quarter
            if (t, j) not in seen:
                seen.add((t, j))
                omega_t = pow(omega, 4 ** t, P)
                for k in range(4):
                    x = pow(omega_t, j + k * quarter, P)
                    pt = list(rest[:2 * t])
                    for _ in range(n - 2 * t):
                        pt.append(hf(x))
                        x = x * x % P
                    yield f"query_t{t}_j{j}_k{k}", X.eq_table(pt)
            pos = j


def _count_queries(layout, positions):
    length = pcs.RATE << layout.n
    seen = set()
    for pos in positions:
        for t in range(layout.folds):
            j = pos % (length >> (2 * t + 2))
            seen.add((t, j))
            pos = j
    return 4 * len(seen)


class VecCoordinates:
    """F_p coordinate model of the honest relation-5 witness vector F."""

    def __init__(self, layout):
        B, K = layout.B, layout.K
        self.layout = layout
        self.x_cells = cell_np(np.arange(B * K, dtype=np.int64))
        self.y_cells = cell_np(layout.off_y + np.arange(B * ROWS, dtype=np.int64))
        self.libra = sorted(layout.libra_slot(l, e) for l in range(layout.d) for e in range(IP_DEGREE + 1))
        lib = set(self.libra)
        idx = np.arange(1 << layout.n, dtype=np.int64)
        self.reserved = [int(i) for i in idx[((idx >> 1) & 1) == 0] if int(i) not in lib]
        self.spare = [int(i) for i in cell_np(np.arange(layout.off_y + B * ROWS, 1 << (layout.n - 1), dtype=np.int64))]


def _project_vec(row, spec, W):
    parts = []
    for kind, payload in spec:
        if kind == "fp":
            parts.append(row[:, np.asarray(payload, dtype=np.int64)])
        elif kind == "fp3":
            parts.append(FA._fp3_columns(row, payload))
        else:                                               # "xw": honest X coordinates (Y = W X folded in)
            xc, yc = payload
            Wt = np.asarray(W, dtype=np.uint64)             # (B, 32, K)
            Bn, _, K = Wt.shape
            ry = row[:, yc].reshape(3, Bn, ROWS)
            acc = row[:, xc].reshape(3, Bn, K).copy()
            for i in range(ROWS):
                for comp in range(3):
                    acc[comp] = X.add(acc[comp], X.mul(ry[comp, :, i:i + 1], Wt[:, i, :]))
            parts.append(acc.reshape(3, Bn * K))
    return np.concatenate(parts, axis=1)


def _spec_width(spec):
    w = 0
    for kind, p in spec:
        w += 3 * len(p) if kind == "fp3" else (len(p[0]) if kind == "xw" else len(p))
    return w


def vec_observations(layout, rec, W, claims=True, pcs_view=True):
    if claims:
        yield from vec_claim_rows(layout, rec, W)
    if pcs_view:
        yield from pcs_rows_np(layout, rec)


def vec_observation_count(layout, rec, claims=True, pcs_view=True):
    cnt = (2 + layout.d * (IP_DEGREE + 1)) if claims else 0
    if pcs_view:
        cnt += 3 * (layout.n + 2) + 1 + _count_queries(layout, rec["positions"])
    return cnt


def require_joint_mask_vec(layout, rec, W, seed=0, witness="honest", drop_libra=False, raise_on_fail=True,
                           oversample=1.5, claims=True, pcs_view=True):
    """im(A) <= im(B) over F_p for the joint linear view of relation 5 (claims + opening).

    A = witness coordinates (see the section comment for `witness`).  B = every Libra slot (unless `drop_libra`, a
    negative control that must FAIL) plus a seeded sample of the blinding half (bit1 = 0 minus Libra) and of the
    spare data cells; im A <= im B_subset implies im A <= im B, so subsampling B (never the observation rows) can
    only make the check fail, never pass wrongly.  `rec` is `prove(..., hooks={"record": rec})`.  Returns the rank
    report; raises ValueError("under-rank mask") when it fails and raise_on_fail."""
    import random as _random
    import time as _time
    t0 = _time.perf_counter()
    coords = VecCoordinates(layout)
    count = vec_observation_count(layout, rec, claims, pcs_view)
    rnd = _random.Random(seed)
    budget = int(oversample * count) + 64
    pool = coords.reserved + coords.spare
    b_spec = [] if drop_libra else [("fp3", coords.libra)]
    b_spec.append(("fp3", sorted(rnd.sample(pool, min(budget, len(pool))))))
    if witness == "honest":
        a_spec = [("xw", (coords.x_cells, coords.y_cells))]
    elif witness == "free":
        a_spec = [("fp", coords.x_cells), ("fp", coords.y_cells)]
    else:
        raise ValueError("witness must be 'honest' or 'free'")
    spec = b_spec + a_spec
    blocks, labels = [], []
    for label, row in vec_observations(layout, rec, W, claims, pcs_view):
        blocks.append(_project_vec(row, spec, W))
        labels.append(label)
    if len(labels) != count:
        raise AssertionError(f"observation count {len(labels)} != {count}")
    matrix = np.concatenate(blocks, axis=0)
    nb = _spec_width(b_spec)
    t1 = _time.perf_counter()
    rank_b, rank_ab = FA.fp_rank_split(matrix, nb)
    report = {"check": "joint_vec", "witness": witness, "drop_libra": drop_libra, "observations": count,
              "claim_observations": (2 + layout.d * (IP_DEGREE + 1)) if claims else 0,
              "fp_rows": int(matrix.shape[0]), "b_columns": nb, "a_columns": _spec_width(a_spec),
              "rank_B": rank_b, "rank_BA": rank_ab, "holds": rank_b == rank_ab,
              "build_s": round(t1 - t0, 2), "rank_s": round(_time.perf_counter() - t1, 2)}
    if raise_on_fail and not report["holds"]:
        raise ValueError(f"under-rank mask: {report}")
    return report


# ----------------------------------------------------------------------------- re-proof leakage (M2-3 style)
def section_map(layout, c_p):
    """[(name, start, end, public)] covering [0, c_p): header | root_0 (private) weight_root (public) N_s (public) |
    inner-product sumcheck + z_blind + PCS rounds (private) | per query: F opening (private), C opening (public: the
    deterministic weight codeword + its Merkle path), folded layers (private) | padding (public)."""
    secs = [("header", 0, 64, True), ("stmt.root_0", 64, 96, False), ("stmt.weight_root", 96, 128, True),
            ("stmt.N_s", 128, 160, True)]
    q_start = 64 + layout.body_used - layout.queries * layout.query_len
    secs.append(("ip_zb_pcs_rounds", 160, q_start, False))
    pos = q_start
    open0 = 96 + 32 * layout.depth(0)
    for q in range(layout.queries):
        secs.append((f"q{q}.F", pos, pos + open0, False))
        secs.append((f"q{q}.C", pos + open0, pos + 2 * open0, True))
        pos += 2 * open0
        for t in range(1, layout.folds):
            ln = 96 + 32 * layout.depth(t)
            secs.append((f"q{q}.L{t}", pos, pos + ln, False))
            pos += ln
    if pos != 64 + layout.body_used:
        raise AssertionError("section map does not sum to body_used")
    secs.append(("padding", pos, c_p, True))
    return secs


def shared_blocks(a, b, layout, c_p, block=64):
    """Re-proof leakage report over 64-byte blocks of two containers of the SAME statement, split public / private
    by `section_map` (aligned blocks + unaligned 64-byte windows at every 8-byte offset of the private bytes)."""
    secs = section_map(layout, c_p)
    priv = [(s, e) for _, s, e, pub in secs if not pub]

    def private(off):
        return all(not pub for _, s, e, pub in secs if s < off + block and off < e)

    def windows(x):
        out = set()
        for s, e in priv:
            for o in range(s, e - block + 1, 8):
                out.add(bytes(x[o:o + block]))
        return out

    offs = range(0, len(a), block)
    priv_bytes = sum(e - s for s, e in priv)
    return {"block_bytes": block, "private_bytes": priv_bytes, "public_bytes": len(a) - priv_bytes,
            "private_aligned_blocks": sum(1 for o in offs if private(o)),
            "private_aligned_shared": sum(1 for o in offs if private(o) and a[o:o + block] == b[o:o + block]),
            "private_window_set_intersection": len(windows(a) & windows(b)),
            "total_aligned_shared_incl_public": sum(1 for o in offs if a[o:o + block] == b[o:o + block])}
