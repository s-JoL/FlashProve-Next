# SPDX-License-Identifier: Apache-2.0
"""pre-challenge ZK-blinded activation CHUNK commitment -> `C_pre` -> relation 6 (`FOLDED_VEC_BOUND`)
with ZERO bytes of plaintext activation rows disclosed.

Relation 6 (`backend/folded_vec_bound.py`) pins ONE 32-byte hiding `x_root` for the B activation rows it proves.
To use it on a live request the prover must fix that `x_root` before it knows WHICH rows will be challenged, so here
it commits every candidate row up front and the challenge picks a chunk:

  1. **Partition** (`partition_rows`): the candidate activation rows -- exact signed integers (|x| <= 2^int_bits,
     default the window 2^23), one K = 1024-wide relation-6 tile input each -- in the order of a PUBLIC row
     schedule (one 4 x uint32 descriptor per row, e.g. (position, layer, slot, column chunk)) are cut into
     deterministic B-row chunks (B = 64 for `fpn2-folded-vec-bound-n22-v1`, B = 1 for `...-n16-v1`); the final chunk
     is zero-padded with `PAD_ROW` descriptors.
  2. **Commit** (`commit_x_chunks`): every chunk c goes through `folded_vec_bound.commit_x(layout, X_c, engine=)`
     (OS-CSPRNG blinding, never `FPN_TEST_RNG_SEED`) -> 32-byte ZK-blinded `x_root_c`.  Only the host cells are kept
     (the device commitment is freed); the challenged chunk is re-committed from them after the challenge and its
     root is asserted equal to the pre-committed `x_root_c`.
  3. **Tree**: leaf_c = H_LEAF(c || x_root_c || meta_c) (meta_c = salted digest of PUBLIC-ised per-row routing
     metadata, e.g. the captured top-8 expert ids of an MoE row, or 32 zero bytes when there is none), padded to a
     power of two with a dedicated PAD leaf, H_NODE internal nodes, and
         x_chunks_root = H_ROOT(n_chunks, B, K, n_rows, int_bits, profile_id, layout_fingerprint, schedule_digest,
                                tree_root)
     with distinct BLAKE2b personalisations for leaf / node / pad / root / meta / schedule.
  4. **C_pre** (`precommit_x_chunks`): `x_chunks_root` is folded into `C_pre` (domain tag `FPN2-XCHUNKS-v1`,
     distinct from relation 6's single-root `FPN2-XROOT-v1` and from a trace payload root) BEFORE the external seed.
  5. **Challenge** (`draw_challenge`): seed = `pipeline.request_seed(C_pre, ...)`; chunk c* = H_DRAW(seed, "chunk")
     mod n_chunks; per row b of c* a 32-byte draw H_DRAW(seed, "tile", b) feeds a PUBLIC tile rule
     (row descriptor, draw, routing row) -> (tensor, r0, c0) of a 32 x K tile of `SetupCertificate`'s
     `weight_commitment_root`.
  6. **Reply** (`prove_rel6_zk_bundle`): `XChunkOpening` (chunk index, n_chunks, B, K, x_root_{c*}, meta_{c*},
     Merkle path: ~300 bytes, zero plaintext activation bytes), the routing opening of c* (MoE only), the drawn tile
     opening (`weight_commit.TileOpening`) and ONE relation-6 container proving Y = W X for the committed X under
     `--weight-root W_root --x-root x_root_{c*}` with N_s = H_NS(C_pre, seed, c*, x_chunks_root, coordinates).
  7. **Verify** (`verify_rel6_zk_bundle`, zero weights, zero plaintext activations): C_pre (`E_CPRE_REQUIRED` /
     `E_CPRE_MISMATCH`) -> re-derived chunk index (`E_X_CHUNK_INDEX`) -> chunk path into `x_chunks_root`
     (`E_X_CHUNK_PATH`) -> routing opening (`E_X_CHUNK_META`) -> drawn tiles under the setup root (`E_TILE_COORDS` /
     `E_TILE_PATH` / ...) and the opened tiles' relation-6 weight root (`E_WEIGHT_ROOT`) -> native Rust
     `fpn-verify --weight-root --x-root` AND the detached `-I` Python reference -> N_s (`E_NS_MISMATCH`).  The
     verifier never receives an activation row: its signature has no X parameter.

Scope (honest): the committed X is whatever the prover put into the chunks -- the commitment binds it before the
challenge, it does NOT prove X equals the served activation (that binding was the plaintext trace opening this
replaces; the relation-4 chain is separate).  Per-chunk `commit_x` cost and host memory scale linearly with the
number of pre-committed chunks.  This module never reads the process environment and never imports `fpn_serve`.
"""
import hashlib
import struct
import threading
from dataclasses import dataclass, field

import numpy as np

from ..backend import folded_vec as FV
from ..backend import folded_vec_bound as FB
from ..backend.fpn2 import FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1, profile_id
from . import weight_commit as WC

P = 0xFFFFFFFF00000001
PAD_ROW = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
DEFAULT_INT_BITS = WC.WINDOW_INT_BITS          # 23: |x| <= 2^23, so sum |W||X| <= 1024 * 2^46 < 2^62
PROFILES = {p["name"]: p for p in (FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1)}

_P_LEAF = b"FPN2-XCHUNK-LEAF"
_P_NODE = b"FPN2-XCHUNK-NODE"
_P_PAD = b"FPN2-XCHUNK-PAD\x00"
_P_ROOT = b"FPN2-XCHUNK-ROOT"
_P_META = b"FPN2-XCHUNK-META"
_P_SCHED = b"FPN2-XCHUNK-SCHD"
_P_DRAW = b"FPN2-XCHUNK-DRAW"
_P_NS = b"FPN2-XCHUNK-NS\x00\x00"
_P_TPR = b"FPN2-XCHUNK-TPR\x00"
_CPRE_TAG = b"FPN2-XCHUNKS-v1"
_OPEN_MAGIC = b"FPN2_XCHUNK\x01"
_ROUTE_MAGIC = b"FPN2_XROUTE\x01"
NO_META = bytes(32)


class XCommitError(ValueError):
    """Typed rejection; `code` is the verdict code (E_...)."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _h(person, *parts):
    h = hashlib.blake2b(digest_size=32, person=person)
    for p in parts:
        h.update(p)
    return h.digest()


def _profile(profile):
    if isinstance(profile, str):
        profile = PROFILES.get(profile)
    if not isinstance(profile, dict) or profile.get("name") not in PROFILES:
        raise XCommitError("E_PROFILE", "x-chunk commitments are defined for the registered relation-6 profiles")
    return PROFILES[profile["name"]]


def _exact_u32(val, name, code="E_INVALID_INPUT"):
    if type(val) is not int:
        raise XCommitError(code, f"{name} must be an exact int, got {type(val).__name__}")
    if not 0 <= val <= 0xFFFFFFFF:
        raise XCommitError(code, f"{name} out of u32 range")
    return val


def _check_tokens(tokens, name):
    try:
        seq = list(tokens)
    except TypeError:
        raise XCommitError("E_INVALID_INPUT", f"{name} must be a sequence of u32 ints") from None
    return [_exact_u32(t, f"{name} element", "E_INVALID_INPUT") for t in seq]


# ----------------------------------------------------------------------------- public row schedule
def _check_schedule(row_desc, n_rows=None):
    try:
        raw_rows = list(row_desc)
    except TypeError:
        raise XCommitError("E_SCHEDULE", "row_desc must be a sequence of 4 x uint32 descriptors") from None
    rows = []
    for d in raw_rows:
        try:
            vals = list(d)
        except TypeError:
            raise XCommitError("E_SCHEDULE", "row descriptors are 4 x uint32") from None
        if len(vals) != 4:
            raise XCommitError("E_SCHEDULE", "row descriptors are 4 x uint32")
        t = tuple(_exact_u32(v, "row descriptor field", "E_SCHEDULE") for v in vals)
        if t == PAD_ROW:
            raise XCommitError("E_SCHEDULE", "PAD_ROW is reserved for zero padding")
        rows.append(t)
    if n_rows is not None and len(rows) != n_rows:
        raise XCommitError("E_SCHEDULE", f"{len(rows)} row descriptors for {n_rows} rows")
    if not rows:
        raise XCommitError("E_SCHEDULE", "empty row schedule")
    if len(set(rows)) != len(rows):
        raise XCommitError("E_SCHEDULE", "duplicate row descriptor")
    return rows


def schedule_digest(row_desc):
    """32-byte digest of the PUBLIC row schedule (row count + every 4 x uint32 descriptor, in order)."""
    rows = _check_schedule(row_desc)
    return _h(_P_SCHED, struct.pack("<Q", len(rows)), b"".join(struct.pack("<4I", *d) for d in rows))


def chunk_rows(row_desc, B, chunk_idx):
    """Descriptors of the B rows of chunk `chunk_idx` (PAD_ROW for the zero padding of the final chunk)."""
    rows = _check_schedule(row_desc)
    n_chunks = -(-len(rows) // B)
    if type(chunk_idx) is not int or not 0 <= chunk_idx < n_chunks:
        raise XCommitError("E_X_CHUNK_INDEX", "chunk index out of range")
    got = rows[B * chunk_idx:B * (chunk_idx + 1)]
    return got + [PAD_ROW] * (B - len(got))


# ----------------------------------------------------------------------------- 1. partition (exact integers)
def partition_rows(X_rows, B, K, int_bits=DEFAULT_INT_BITS):
    """(n_chunks, [(B, K) canonical F_p uint64]) from EXACT signed integer rows (N, K); the final chunk is zero-padded.

    Exact integer validation: `X_rows` must be a numpy integer array (no float / bool / object), shape (N, K) with
    N >= 1, every |x| <= 2^int_bits (so relation 6's integer statement cannot wrap mod p)."""
    if not isinstance(X_rows, np.ndarray) or X_rows.dtype == np.bool_ or not np.issubdtype(X_rows.dtype, np.integer):
        raise XCommitError("E_X_INT", "activation rows must be a numpy integer array (exact integers)")
    if X_rows.ndim != 2 or X_rows.shape[1] != K or X_rows.shape[0] < 1:
        raise XCommitError("E_X_INT", f"activation rows must have shape (N >= 1, {K})")
    if not 1 <= int(int_bits) <= 30:
        raise XCommitError("E_X_INT", "int_bits must be in [1, 30]")
    if X_rows.dtype == np.uint64:
        if (X_rows > np.uint64(1 << int_bits)).any():
            raise XCommitError("E_X_INT", f"|x| exceeds 2^{int_bits}")
        Xi = X_rows.astype(np.int64)
    else:
        Xi = X_rows.astype(np.int64)
        if (np.abs(Xi) > (1 << int_bits)).any():
            raise XCommitError("E_X_INT", f"|x| exceeds 2^{int_bits}")
    n = Xi.shape[0]
    n_chunks = -(-n // B)
    pad = np.zeros((n_chunks * B, K), dtype=np.int64)
    pad[:n] = Xi
    fld = FV.to_field(pad)
    return n_chunks, [np.ascontiguousarray(fld[B * c:B * (c + 1)]) for c in range(n_chunks)]


def field_to_int(Xf):
    """Inverse of `FV.to_field` for |x| < p / 2 (canonical F_p uint64 -> int64)."""
    Xf = np.asarray(Xf, dtype=np.uint64)
    neg = Xf > np.uint64(P // 2)
    out = Xf.astype(np.int64)
    out[neg] = -(np.uint64(P) - Xf[neg]).astype(np.int64)
    return out


# ----------------------------------------------------------------------------- routing metadata (MoE)
def routing_meta(salt, routing_bytes):
    """meta_c = H_META(salt || len || routing bytes): binds a chunk's per-row routing metadata before the
    challenge; the 32-byte OS-CSPRNG salt keeps the low-entropy routing of UNCHALLENGED chunks hidden."""
    if len(salt) != 32:
        raise XCommitError("E_X_CHUNK_META", "salt must be 32 bytes")
    return _h(_P_META, salt, struct.pack("<Q", len(routing_bytes)), bytes(routing_bytes))


@dataclass(frozen=True)
class RoutingOpening:
    """Disclosed routing metadata of the challenged chunk: per row `width` uint8 values (e.g. top-8 expert ids).
    Routing decisions, NOT activation values; reported separately from the x-chunk opening."""
    salt: bytes
    width: int
    data: bytes

    def rows(self, B):
        if len(self.data) != B * self.width:
            raise XCommitError("E_X_CHUNK_META", "routing opening size differs from B x width")
        return [np.frombuffer(self.data, dtype=np.uint8, count=self.width, offset=b * self.width).copy()
                for b in range(B)]

    def encode(self):
        return _ROUTE_MAGIC + self.salt + struct.pack("<IQ", self.width, len(self.data)) + self.data

    @classmethod
    def decode(cls, data):
        data = bytes(data)
        hl = len(_ROUTE_MAGIC) + 32 + 12
        if not data.startswith(_ROUTE_MAGIC) or len(data) < hl:
            raise XCommitError("E_X_CHUNK_META", "bad routing opening")
        width, n = struct.unpack_from("<IQ", data, len(_ROUTE_MAGIC) + 32)
        if len(data) != hl + n:
            raise XCommitError("E_X_CHUNK_META", "routing opening length")
        return cls(data[len(_ROUTE_MAGIC):len(_ROUTE_MAGIC) + 32], width, data[hl:])


# ----------------------------------------------------------------------------- 3. tree
def _leaf(c, x_root, meta):
    return _h(_P_LEAF, struct.pack("<Q", c), x_root, meta)


def _pad_leaf():
    return _h(_P_PAD)


def _levels(leaves):
    m = 1
    while m < len(leaves):
        m <<= 1
    lvl = list(leaves) + [_pad_leaf()] * (m - len(leaves))
    levels = [lvl]
    while len(lvl) > 1:
        lvl = [_h(_P_NODE, lvl[i], lvl[i + 1]) for i in range(0, len(lvl), 2)]
        levels.append(lvl)
    return levels


def _root_from_path(leaf, i, path):
    h = leaf
    for sib in path:
        h = _h(_P_NODE, sib, h) if i & 1 else _h(_P_NODE, h, sib)
        i >>= 1
    return h


def _depth(n):
    d = 0
    while (1 << d) < n:
        d += 1
    return d


def chunks_root(tree_root, *, n_chunks, B, K, n_rows, int_bits, profile, layout_fingerprint, sched_digest):
    """x_chunks_root = H_ROOT(header || tree_root): binds n_chunks, B, K, n_rows, int_bits, profile id, the capture
    layout fingerprint and the public row schedule."""
    if len(layout_fingerprint) != 32 or len(sched_digest) != 32 or len(tree_root) != 32:
        raise XCommitError("E_X_CHUNK_PATH", "fingerprints and roots are 32 bytes")
    hdr = struct.pack("<QIIQI", n_chunks, B, K, n_rows, int_bits) + profile_id(_profile(profile))
    return _h(_P_ROOT, hdr, bytes(layout_fingerprint), bytes(sched_digest), tree_root)


@dataclass(frozen=True)
class XChunkOpening:
    """The ONLY activation-derived bytes the verifier receives: (chunk index, n_chunks, B, K, the chunk's 32-byte
    ZK-blinded x_root, its 32-byte meta digest, the Merkle path into x_chunks_root).  Zero plaintext activations."""
    chunk_idx: int
    n_chunks: int
    B: int
    K: int
    x_root: bytes
    meta: bytes
    path: tuple

    def encode(self):
        return (_OPEN_MAGIC + struct.pack("<QQII", self.chunk_idx, self.n_chunks, self.B, self.K) + self.x_root
                + self.meta + struct.pack("<B", len(self.path)) + b"".join(self.path))

    @classmethod
    def decode(cls, data):
        data = bytes(data)
        o = len(_OPEN_MAGIC)
        if not data.startswith(_OPEN_MAGIC) or len(data) < o + 24 + 64 + 1:
            raise XCommitError("E_X_CHUNK_PATH", "bad x-chunk opening")
        c, n, B, K = struct.unpack_from("<QQII", data, o)
        o += 24
        x_root, meta = data[o:o + 32], data[o + 32:o + 64]
        o += 64
        depth = data[o]
        o += 1
        if len(data) != o + 32 * depth:
            raise XCommitError("E_X_CHUNK_PATH", "x-chunk opening length")
        return cls(c, n, B, K, x_root, meta, tuple(data[o + 32 * i:o + 32 * i + 32] for i in range(depth)))


# ----------------------------------------------------------------------------- 2. commit (prover, private)
@dataclass
class XChunkCommitment:
    """PRIVATE prover state.  Only `root` (x_chunks_root, 32 bytes) leaves the prover before the challenge."""
    profile: dict
    layout: object
    n_rows: int
    int_bits: int
    row_desc: list
    layout_fingerprint: bytes
    x_roots: list
    metas: list
    levels: list
    root: bytes
    cells: list = field(repr=False)          # per chunk (2^n, 3) uint64 host cells (or None when lazily seeded)
    salts: list = field(default=None, repr=False)
    routing: list = field(default=None, repr=False)   # per chunk (B * width) uint8 bytes
    routing_width: int = 0
    engine: object = None
    timings: dict = field(default_factory=dict)
    chunk_seeds: list = field(default=None, repr=False)   # per chunk 32B ephemeral OS-CSPRNG key (200x less RAM)
    chunk_x_vals: list = field(default=None, repr=False)  # per chunk (B, K) uint64 canonical F_p X
    _lock: object = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _chunk_locks: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _issued: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _xcom_used: set = field(default_factory=set, init=False, repr=False, compare=False)

    def _check_chunk_idx(self, c):
        if type(c) is not int or not 0 <= c < self.n_chunks:
            raise XCommitError("E_X_CHUNK_INDEX", f"chunk index {c!r} out of range [0, {self.n_chunks})")
        return c

    def _chunk_lock(self, c):
        c = self._check_chunk_idx(c)
        with self._lock:
            lk = self._chunk_locks.get(c)
            if lk is None:
                lk = threading.Lock()
                self._chunk_locks[c] = lk
            return lk

    @property
    def n_chunks(self):
        return len(self.x_roots)

    @property
    def B(self):
        return self.layout.B

    @property
    def K(self):
        return self.layout.K

    def opening(self, c):
        c = self._check_chunk_idx(c)
        return XChunkOpening(c, self.n_chunks, self.B, self.K, self.x_roots[c], self.metas[c],
                             tuple(self.levels[k][(c >> k) ^ 1] for k in range(len(self.levels) - 1)))

    def routing_opening(self, c):
        c = self._check_chunk_idx(c)
        if self.routing is None:
            return None
        return RoutingOpening(self.salts[c], self.routing_width, self.routing[c])

    def chunk_x(self, c):
        """(B, K) canonical F_p X of chunk c (read back from its committed cells or lazy X buffer)."""
        c = self._check_chunk_idx(c)
        if self.chunk_x_vals is not None and self.chunk_x_vals[c] is not None:
            return np.ascontiguousarray(self.chunk_x_vals[c])
        xi = FV.cell_np(np.arange(self.B * self.K, dtype=np.int64))
        return np.ascontiguousarray(self.cells[c][xi, 0].reshape(self.B, self.K))

    def _build_x_commitment(self, c):
        from ..backend import folded_air as FA
        from ..backend import folded_pcs as pcs
        c = self._check_chunk_idx(c)
        cells = self.cells[c]
        if cells is None:
            cells = FB.build_x_cells_np(self.layout, self.chunk_x_vals[c], FV.EphemeralSeedRng(self.chunk_seeds[c]))
        if self.engine is not None:
            com = FA.engine_commit(self.engine, cells)
        else:
            com = pcs.zk_commit([tuple(int(v) for v in row) for row in cells])
        if com.root != self.x_roots[c]:
            raise AssertionError("re-committed chunk root differs from the pre-committed x_root")
        return FB.XCommitment(self.layout, cells, self.chunk_x(c), com, self.engine)

    def x_commitment(self, c, *, _allow_reuse=False):
        """Re-commit chunk c's stored cells on the SAME engine -> `folded_vec_bound.XCommitment`.

        Enforces single blinding consumption per chunk unless `_allow_reuse=True` (explicit test/forgery hook)."""
        c = self._check_chunk_idx(c)
        with self._chunk_lock(c):
            if not _allow_reuse and c in self._xcom_used:
                raise XCommitError("E_CHUNK_BLINDING_EXHAUSTED",
                                   f"chunk {c} blinding already consumed (single-proof budget)")
            xcom = self._build_x_commitment(c)
            if not _allow_reuse:
                self._xcom_used.add(c)
            return xcom


def commit_x_chunks(profile, X_rows, row_desc, layout_fingerprint, *, int_bits=DEFAULT_INT_BITS, engine=None,
                    routing_rows=None, rng=None, salt_source=None):
    """Commit every candidate activation row BEFORE the challenge.  Returns the private `XChunkCommitment`;
    `.root` is x_chunks_root (fold it into C_pre with `precommit_x_chunks`).

    X_rows: (N, K) exact integers, in the order of the PUBLIC `row_desc` (N x 4 uint32).  routing_rows: optional
    (N, width) uint8 per-row routing metadata (e.g. captured top-8 expert ids) bound by a salted per-chunk digest.
    rng: test-only explicit rng for `commit_x` (default: OS CSPRNG); salt_source: test-only callable(n) -> bytes
    (default `secrets.token_bytes`)."""
    import secrets
    import time
    prof = _profile(profile)
    lay = FB.layout_of(prof)
    rows = _check_schedule(row_desc, None)
    if isinstance(X_rows, np.ndarray) and X_rows.ndim == 2 and X_rows.shape[0] != len(rows):
        raise XCommitError("E_SCHEDULE", f"{X_rows.shape[0]} rows for {len(rows)} descriptors")
    if len(layout_fingerprint) != 32:
        raise XCommitError("E_SCHEDULE", "layout_fingerprint must be 32 bytes")
    n_chunks, chunks = partition_rows(X_rows, lay.B, lay.K, int_bits)
    salts = routing = None
    width = 0
    if routing_rows is not None:
        R = np.asarray(routing_rows)
        if R.dtype != np.uint8 or R.ndim != 2 or R.shape[0] != len(rows) or R.shape[1] < 1:
            raise XCommitError("E_X_CHUNK_META", "routing rows must be (N, width >= 1) uint8")
        width = R.shape[1]
        Rp = np.zeros((n_chunks * lay.B, width), dtype=np.uint8)
        Rp[:len(rows)] = R
        draw = salt_source or secrets.token_bytes
        salts = [bytes(draw(32)) for _ in range(n_chunks)]
        routing = [Rp[lay.B * c:lay.B * (c + 1)].tobytes() for c in range(n_chunks)]
    x_roots, metas, cells, per = [], [], [], []
    lazy_seeds = [] if rng is None else None
    lazy_xvals = [] if rng is None else None
    for c, Xc in enumerate(chunks):
        t = time.perf_counter()
        if rng is None:
            erng = FV.EphemeralSeedRng()
            xc = FB.commit_x(lay, Xc, rng=erng, engine=engine)
            x_roots.append(xc.root)
            cells.append(None)
            lazy_seeds.append(erng.seed32)
            lazy_xvals.append(Xc.copy())
        else:
            xc = FB.commit_x(lay, Xc, rng=rng, engine=engine)
            x_roots.append(xc.root)
            cells.append(xc.cells)
        del xc                                     # frees the device commitment (code + tree); lazy mode drops 100 MB cells too!
        if engine is not None and hasattr(engine, "sync"):
            engine.sync()
        per.append((time.perf_counter() - t) * 1e3)
        metas.append(routing_meta(salts[c], routing[c]) if routing is not None else NO_META)
    t = time.perf_counter()
    levels = _levels([_leaf(c, x_roots[c], metas[c]) for c in range(n_chunks)])
    root = chunks_root(levels[-1][0], n_chunks=n_chunks, B=lay.B, K=lay.K, n_rows=len(rows), int_bits=int_bits,
                       profile=prof, layout_fingerprint=layout_fingerprint, sched_digest=schedule_digest(rows))
    tm = {"commit_x_total_ms": sum(per), "commit_x_per_chunk_ms": per,
          "tree_ms": (time.perf_counter() - t) * 1e3}
    return XChunkCommitment(prof, lay, len(rows), int(int_bits), rows, bytes(layout_fingerprint), x_roots, metas,
                            levels, root, cells, salts, routing, width, engine, tm, lazy_seeds, lazy_xvals)


# ----------------------------------------------------------------------------- 4. C_pre
def precommit_x_chunks(setup_cert_bytes, x_chunks_root, response_tokens, policy_bytes=b""):
    """32-byte C_pre binding the setup, x_chunks_root (domain tag `FPN2-XCHUNKS-v1`), the response and the policy --
    zero plaintext activation bytes.  The trace-payload slot holds H_TPR(x_chunks_root), so x_chunks_root can alias
    neither a trace payload root nor relation 6's single `x_root` pre-commitment."""
    if type(x_chunks_root) is not bytes or len(x_chunks_root) != 32:
        raise XCommitError("E_CPRE_REQUIRED", "x_chunks_root must be 32 bytes")
    from .pipeline import precommit_request
    return precommit_request(setup_cert_bytes, _h(_P_TPR, x_chunks_root), response_tokens,
                             _CPRE_TAG + x_chunks_root + bytes(policy_bytes))


# ----------------------------------------------------------------------------- 5. challenge
def draw_challenge(c_pre, token_ids, response_tokens, external_nonce, n_chunks, B):
    """(seed, chunk index c*, [32-byte per-row tile draws]) from PUBLIC data only."""
    from .pipeline import request_seed
    if n_chunks < 1:
        raise XCommitError("E_X_CHUNK_INDEX", "no chunks")
    seed = request_seed(c_pre, token_ids, response_tokens, external_nonce=external_nonce)
    c = int.from_bytes(_h(_P_DRAW, seed, b"chunk", struct.pack("<Q", n_chunks)), "little") % n_chunks
    draws = [_h(_P_DRAW, seed, b"tile", struct.pack("<QI", c, b)) for b in range(B)]
    return seed, c, draws


def expected_coords(rows, draws, routing, tile_rule):
    """Drawn tile coordinates [(tensor, r0, c0)] of the challenged chunk: tile_rule(row descriptor, 32-byte draw,
    routing row or None) is a PUBLIC deterministic function the prover and the verifier both apply."""
    return [tuple(tile_rule(rows[b], draws[b], None if routing is None else routing[b])) for b in range(len(rows))]


def bundle_n_s(c_pre, seed, chunk_idx, x_chunks_root, coords):
    enc = b"".join(struct.pack("<H", len(n.encode())) + n.encode() + struct.pack("<QQ", r0, c0)
                   for n, r0, c0 in coords)
    return _h(_P_NS, c_pre, seed, struct.pack("<Q", chunk_idx), x_chunks_root, enc)


# ----------------------------------------------------------------------------- 6. prove
@dataclass
class Rel6ZkBundle:
    c_pre: bytes
    x_chunks_root: bytes
    chunk_opening: bytes
    routing_opening: bytes        # b"" when the chunk carries no routing metadata
    tile_opening: bytes
    container: bytes
    report: dict


def prove_rel6_zk_bundle(xcc, *, setup_cert, weight_commitment, checkpoint, token_ids, response_tokens,
                         external_nonce, tile_rule, policy_bytes=b"", bf16_window=DEFAULT_INT_BITS, rng=None,
                         hooks=None):
    """Prover side after the external nonce is revealed (C_pre was published from `xcc.root` before it).

    Enforces single-issuance per committed chunk (thread-safe): an identical retry returns the cached
    `Rel6ZkBundle`; a distinct challenge targeting an already-proved chunk raises
    `XCommitError("E_CHUNK_BLINDING_EXHAUSTED")` so chunk blinding is never consumed twice."""
    import time
    toks = _check_tokens(token_ids, "token_ids")
    resps = _check_tokens(response_tokens, "response_tokens")
    if type(bf16_window) is not int or not 8 <= bf16_window <= 52:
        raise XCommitError("E_INVALID_INPUT", "bf16_window must be an exact int in [8, 52]")
    c_pre = precommit_x_chunks(setup_cert, xcc.root, resps, policy_bytes)
    seed, c, draws = draw_challenge(c_pre, toks, resps, external_nonce, xcc.n_chunks, xcc.B)
    rows = chunk_rows(xcc.row_desc, xcc.B, c)
    rop = xcc.routing_opening(c)
    coords = expected_coords(rows, draws, None if rop is None else rop.rows(xcc.B), tile_rule)
    n_s = bundle_n_s(c_pre, seed, c, xcc.root, coords)
    issue_key = (c_pre, seed, n_s, bytes(setup_cert), bf16_window)

    with xcc._chunk_lock(c):
        if c in xcc._issued:
            prev_key, prev_bundle = xcc._issued[c]
            if prev_key == issue_key and hooks is None:
                return prev_bundle
            raise XCommitError("E_CHUNK_BLINDING_EXHAUSTED",
                               f"chunk {c} already issued a proof for a different challenge")
        if c in xcc._xcom_used:
            raise XCommitError("E_CHUNK_BLINDING_EXHAUSTED",
                               f"chunk {c} blinding was already consumed")
        tm = {}
        t = time.perf_counter()
        opening = WC.open_drawn_tiles(weight_commitment, checkpoint, coords)
        tm["open_tiles_ms"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        W_int = np.stack([w for w, _ in opening.weights_int(bf16_window)]).astype(np.int64)
        X_int = field_to_int(xcc.chunk_x(c))
        ys, bits = [], []
        for b in range(xcc.B):
            yb, bb = FV.exact_products(W_int[b:b + 1], X_int[b:b + 1])
            ys.append(yb[0])
            bits.append(bb)
        Wf, Yf = FV.to_field(W_int), FV.to_field(np.array(ys, dtype=object))
        tm["tiles_and_y_ms"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        xcom = xcc._build_x_commitment(c)
        tm["recommit_chunk_ms"] = (time.perf_counter() - t) * 1e3
        FV._W_CACHE.clear()
        ph = dict(hooks or {})
        ph.setdefault("timings", {})
        t = time.perf_counter()
        container = FB.prove_bound(xcc.profile, Wf, xcom.x_values, Yf, xcom, n_s=n_s, rng=rng, engine=xcc.engine,
                                   hooks=ph)
        tm["prove_bound_ms"] = (time.perf_counter() - t) * 1e3
        tm["prove_bound_phases_ms"] = dict(ph["timings"])
        rep = {"chunk_idx": c, "n_chunks": xcc.n_chunks, "coords": coords, "integer_bits_max": int(max(bits)),
               "integer_faithful": [bool(b < 62) for b in bits], "timings_ms": tm, "seed": seed.hex(),
               "padded_rows": sum(r == PAD_ROW for r in rows)}
        bundle = Rel6ZkBundle(c_pre, xcc.root, xcc.opening(c).encode(), b"" if rop is None else rop.encode(),
                              opening.encode(), container, rep)
        xcc._xcom_used.add(c)
        xcc._issued[c] = (issue_key, bundle)
        return bundle


# ----------------------------------------------------------------------------- 7. verify (no weights, no plaintext X)
def _cli_args(weight_root, x_root):
    return ["--weight-root", weight_root.hex(), "--x-root", x_root.hex()]


def rust_verify_rel6(binary, container, weight_root, x_root):
    """(verdict line, wall ms) of `fpn-verify --weight-root <W> --x-root <x> <file>`."""
    from ..circuit import e2e_prover as E2E
    out, wall, _ = E2E._run_cli([binary] + _cli_args(weight_root, x_root), container)
    return out, wall


def python_verify_rel6(container, weight_root, x_root, python=None):
    """Detached Python reference verifier (`-W error -I`, only verifier/python on sys.path) with both pins."""
    import os
    import sys
    from ..circuit import e2e_prover as E2E
    vdir = os.path.join(E2E.ROOT, "verifier", "python")
    boot = ("import runpy, sys; " f"sys.path.insert(0, {vdir!r}); "
            "sys.argv = ['fpn_verify_ref', *sys.argv[1:]]; runpy.run_module('fpn_verify_ref', run_name='__main__')")
    out, wall, _ = E2E._run_cli([python or sys.executable, "-W", "error", "-I", "-c", boot]
                                + _cli_args(weight_root, x_root), container, cwd=vdir)
    return out, wall


def _code(line):
    return "VERIFIED" if line == "VERIFIED" else line.removeprefix("REJECTED: ")


def verify_rel6_zk_bundle(setup_cert, *, c_pre, token_ids, response_tokens, external_nonce, profile, row_desc,
                          layout_fingerprint, x_chunks_root, chunk_opening, tile_opening, container, tile_rule,
                          routing_opening=b"", policy_bytes=b"", int_bits=DEFAULT_INT_BITS,
                          bf16_window=DEFAULT_INT_BITS, rust_binary=None, check_python=True, python=None,
                          engine=None, rust_runs=1):
    """Offline zero-weight, ZERO-PLAINTEXT verifier.  Inputs: the SetupCertificate, the C_pre published before the
    challenge, public request data, the PUBLIC row schedule (coordinates, not values), and the reply bundle.  No
    activation row is a parameter.  Returns a report whose `verdict` is "VERIFIED" or "REJECTED: <code>"."""
    import time
    rep = {"verdict": "VERIFIED", "plaintext_activation_bytes_received": 0}
    t0 = time.perf_counter()
    try:
        prof = _profile(profile)
        lay = FB.layout_of(prof)
        # 1. C_pre & public input discipline
        if c_pre is None or type(c_pre) is not bytes or len(c_pre) != 32:
            raise XCommitError("E_CPRE_REQUIRED", "the pre-challenge C_pre is required")
        if type(x_chunks_root) is not bytes or len(x_chunks_root) != 32:
            raise XCommitError("E_CPRE_MISMATCH", "x_chunks_root must be 32 bytes")
        if type(external_nonce) is not bytes or len(external_nonce) != 32:
            raise XCommitError("E_INVALID_INPUT", "external_nonce must be 32 bytes")
        if type(int_bits) is not int or not 1 <= int_bits <= 52:
            raise XCommitError("E_INVALID_INPUT", "int_bits must be an exact int in [1, 52]")
        if type(bf16_window) is not int or not 8 <= bf16_window <= 52:
            raise XCommitError("E_INVALID_INPUT", "bf16_window must be an exact int in [8, 52]")
        toks = _check_tokens(token_ids, "token_ids")
        resps = _check_tokens(response_tokens, "response_tokens")
        if precommit_x_chunks(setup_cert, x_chunks_root, resps, policy_bytes) != c_pre:
            raise XCommitError("E_CPRE_MISMATCH", "C_pre does not bind this x_chunks_root / setup / response / policy")
        rows_all = _check_schedule(row_desc)
        n_chunks = -(-len(rows_all) // lay.B)
        # 2. challenge
        seed, c, draws = draw_challenge(c_pre, toks, resps, external_nonce, n_chunks, lay.B)
        op = XChunkOpening.decode(chunk_opening)
        if (op.n_chunks, op.B, op.K) != (n_chunks, lay.B, lay.K):
            raise XCommitError("E_X_CHUNK_INDEX", "opening geometry differs from the public schedule / profile")
        if op.chunk_idx != c:
            raise XCommitError("E_X_CHUNK_INDEX", f"opened chunk {op.chunk_idx} is not the drawn chunk {c}")
        # 3. Merkle path into x_chunks_root
        if len(op.path) != _depth(n_chunks):
            raise XCommitError("E_X_CHUNK_PATH", "path length differs from the tree depth")
        tree_root = _root_from_path(_leaf(c, op.x_root, op.meta), c, op.path)
        if chunks_root(tree_root, n_chunks=n_chunks, B=lay.B, K=lay.K, n_rows=len(rows_all), int_bits=int_bits,
                       profile=prof, layout_fingerprint=layout_fingerprint,
                       sched_digest=schedule_digest(rows_all)) != x_chunks_root:
            raise XCommitError("E_X_CHUNK_PATH", "x_root of the opened chunk is not under x_chunks_root")
        # 3b. routing metadata of the chunk (MoE)
        routing = None
        if routing_opening:
            ro = RoutingOpening.decode(routing_opening)
            if routing_meta(ro.salt, ro.data) != op.meta:
                raise XCommitError("E_X_CHUNK_META", "routing opening does not match the committed meta digest")
            routing = ro.rows(lay.B)
        elif op.meta != NO_META:
            raise XCommitError("E_X_CHUNK_META", "chunk carries routing metadata but no routing opening was sent")
        rows = chunk_rows(rows_all, lay.B, c)
        coords = expected_coords(rows, draws, routing, tile_rule)
        t1 = time.perf_counter()
        # 4. weight tiles against SetupCertificate.weight_commitment_root
        try:
            to = WC.TileOpening.decode(bytes(tile_opening)) if isinstance(tile_opening, (bytes, bytearray)) \
                else tile_opening
            if len(to.tiles) != lay.B or to.K != lay.K:
                raise WC.WeightCommitError("E_TILE_COORDS", f"{prof['name']} proves {lay.B} tiles of K = {lay.K}")
            rep["tiles"] = WC.check_tile_inclusion(WC._setup_root(setup_cert), to, coords)
            FV._W_CACHE.clear()
            w_root = FB.weight_root(lay, to.weights_field(bf16_window), engine)
            FV._W_CACHE.clear()
        except WC.WeightCommitError as exc:
            raise XCommitError(exc.code, str(exc)) from None
        if bytes(container[96:128]) != w_root:
            raise XCommitError("E_WEIGHT_ROOT", "the container's weight_root differs from the opened tiles' root")
        t2 = time.perf_counter()
        # 5. relation 6 under both pins: Rust + detached Python
        rep.update(chunk_idx=c, n_chunks=n_chunks, weight_root=w_root.hex(), x_root=op.x_root.hex(),
                   x_chunk_opening_bytes=len(chunk_opening), routing_opening_bytes=len(routing_opening or b""),
                   seed=seed.hex())
        if rust_binary is not None:
            runs = [rust_verify_rel6(rust_binary, container, w_root, op.x_root) for _ in range(max(1, rust_runs))]
            rep["rust"] = _code(runs[0][0])
            rep["rust_wall_ms"] = [round(r[1], 2) for r in runs]
            if rep["rust"] != "VERIFIED":
                raise XCommitError(rep["rust"], "native fpn-verify rejected")
        if check_python:
            out, wall = python_verify_rel6(container, w_root, op.x_root, python)
            rep["python"], rep["python_wall_ms"] = _code(out), round(wall, 1)
            if rep["python"] != "VERIFIED":
                raise XCommitError(rep["python"], "detached Python reference rejected")
        if rust_binary is None and not check_python:
            raise XCommitError("E_NO_VERIFIER", "neither Rust nor Python verification was requested")
        # 6. N_s binds C_pre, seed, chunk and coordinates (statement offset 160: root_0 | W | x_root | N_s)
        if bytes(container[160:192]) != bundle_n_s(c_pre, seed, c, x_chunks_root, coords):
            raise XCommitError("E_NS_MISMATCH", "N_s does not bind this challenge")
        rep["timings_ms"] = {"cpre_chunk_path_ms": round((t1 - t0) * 1e3, 2),
                             "tiles_weight_root_ms": round((t2 - t1) * 1e3, 2),
                             "relation6_cli_ms": round((time.perf_counter() - t2) * 1e3, 2)}
    except XCommitError as exc:
        rep["verdict"] = f"REJECTED: {exc.code}"
        rep["detail"] = str(exc)
    except (TypeError, ValueError, struct.error) as exc:
        rep["verdict"] = "REJECTED: E_INVALID_INPUT"
        rep["detail"] = str(exc)
    return rep
