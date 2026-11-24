# SPDX-License-Identifier: Apache-2.0
"""Lossless OCP MXFP4 weight commitment and per-request drawn-tile openings .

`gpt-oss-20b` stores its MoE expert matrices as OCP MXFP4 (`Mxfp4GptOssExperts`): per expert, a matrix of shape
(rows, in) is held as `<name>_blocks` (rows, in / 32, 16) uint8 -- 4-bit E2M1 elements, two per byte, the LOW nibble
being the EARLIER element (element 2 i = byte & 0x0F, element 2 i + 1 = byte >> 4; the convention of
`bench/tier2/bench_frontier_calibrate.py::mxfp4_dequant`) -- and `<name>_scales` (rows, in / 32) uint8 E8M0 block
exponents, value = e2m1 * 2^(e - 127), e = 255 reserved for NaN.  This module commits those bytes EXACTLY:

  * the leaf of a 32 x K tile (rows [r0, r0 + 32) x columns [c0, c0 + K), zero-padded at the matrix edge) hashes the
    raw packed bytes `blocks` (32 x K/2, 16 KiB at K = 1024) and `scales` (32 x K/32, 1 KiB): 17,408 B against the
    65,536 B of the same tile in raw BF16 (3.765x smaller);
  * every hash is BLAKE2b-256 domain-separated by a length-prefixed ASCII tag fed as the first message bytes
    (`FPN2-MXFP4-LEAF-v1`, `-NODE-v1`, `-MANI-v1`, `-ROOT-v1`, plus `-PAD-v1` for the Merkle padding leaf): BLAKE2b's
    `person` field is capped at 16 bytes, shorter than these tags, so they are not personalisations.  The manifest
    (K, leaf count, `window_bits`, residual digest, every tensor's name / shape / first leaf) is hashed into the root,
    `root = H_ROOT(H_MANI(manifest) || tree_root)`, which is `SetupCertificate.weight_commitment_root`;
  * relation 5 / 6 need integers.  `materialize_tile` maps the exact dequantised tile to signed integer units on ONE
    power-of-two grid chosen from the tile's max |value| (`weight_commit.window_ints`, round-half-even, public and
    deterministic, applied AFTER opening; the commitment itself stays lossless).  An E2M1 value is exactly
    +-d * 2^-1 * 2^(e - 127) with doubled magnitude d in {0, 1, 2, 3, 4, 6, 8, 12}, so every value is exact in float64
    (no rounding in decode).  Every nonzero doubled magnitude is 1x or 1.5x a power of two, hence the top unit is
    exactly 2^(w - 1) or 3 * 2^(w - 2) and EVERY |unit| < 2^w (strict) for `window_bits` = w in [8, 52]; with w = 23,
    X windowed likewise and K = 1024, sum_j |W||X| <= 1024 * 2^23 * 2^23 = 2^56 < 2^62, so the F_p statement is the
    integer statement.  `integer_faithful` is COMPUTED, never assumed (false for large w).  Entries more than w - 1
    binades below the tile's top binade lose precision: the half-step bound 2^-w * max|W| is exact-checked and
    reported (`quant_window_tail_rel_err`), never hidden;
  * `verify_mxfp4_tiles_against_setup` -- given ONLY the SetupCertificate, the opening, the expected draw coordinates
    and a relation-5 or relation-6 container -- checks tile inclusion under `weight_commitment_root`, the exact
    coordinates, the (optionally pinned) window width, and `FOLDED_VEC*.weight_root` of the windowed opened tiles
    against the container's pinned `weight_root` (`container[96:128]` = STMT[32..64] for both relations).  It binds
    W ONLY: a relation-6 container is X-bound by `fpn-verify --x-root`, which callers MUST still pin.

Scope: this root commits MXFP4 matrices (+ an optional caller-supplied 32-byte residual digest for everything else).
A gpt-oss deployment also holds BF16 tensors (attention, router, embeddings); combining the two roots is NOT done here.

This module never reads the process environment and never imports `fpn_serve`.
"""
import hashlib
import struct
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .weight_commit import ROWS, P, WeightCommitError, merkle_path, tree_depth, window_ints, window_tail

BLOCK = 32                       # MXFP4 block width (elements sharing one E8M0 scale)
DEFAULT_K = 1024
DEFAULT_WINDOW_BITS = 23
MIN_WINDOW_BITS, MAX_WINDOW_BITS = 8, 52
MAX_K = 1 << 20
MAX_DIM = (1 << 32) - 1
MAX_LEAVES = 1 << 40
MAX_DEPTH = 64
MAX_TILES = 1 << 20
FAITHFUL_BITS = 62               # relation-5 integer statement: sum |W||X| < 2^62 < p / 2
E8M0_NAN = 255
LEAF_PAYLOAD_BYTES_BF16 = ROWS * DEFAULT_K * 2

_T_LEAF = b"FPN2-MXFP4-LEAF-v1"
_T_NODE = b"FPN2-MXFP4-NODE-v1"
_T_PAD = b"FPN2-MXFP4-PAD-v1"
_T_MANI = b"FPN2-MXFP4-MANI-v1"
_T_ROOT = b"FPN2-MXFP4-ROOT-v1"
_LAYOUT_TAG = b"E2M1x2/low-nibble-first/E8M0/B32/row-major"
_MANIFEST_MAGIC = b"FPN2_MXFP4_MANIFEST\x01"
_OPENING_MAGIC = b"FPN2_MXFP4_OPEN\x01"

# ----------------------------------------------------------------------------- the exact E2M1 / E8M0 formats
# E2M1: sign, 2 exponent bits (bias 1), 1 mantissa bit.  Doubled magnitude (value = d / 2):
#   e = 0 (subnormal): d = m;   e > 0: d = (2 + m) 2^(e - 1)   ->   0, 1, 2, 3, 4, 6, 8, 12  (= 0, .5, 1, 1.5, 2, 3, 4, 6)
_X2 = tuple(m if e == 0 else (2 + m) << (e - 1) for e in range(4) for m in range(2))
E2M1_TABLE = tuple((s, d) for s in (0, 1) for d in _X2)     # nibble -> (sign, doubled magnitude); nibble 8 is -0
_SIGNED_X2 = np.array([-d if s else d for s, d in E2M1_TABLE], dtype=np.int64)
E2M1_SIGNED_X2 = tuple(int(v) for v in _SIGNED_X2)
if _X2 != (0, 1, 2, 3, 4, 6, 8, 12):
    raise AssertionError("E2M1 table derivation drifted")


def e2m1_nibble_x2(nibble):
    """(sign, doubled magnitude) of one 4-bit E2M1 code; value = (-1)^sign * d * 2^-1 (exact integers)."""
    if type(nibble) is not int or not 0 <= nibble <= 15:
        raise ValueError("an E2M1 code is an int in [0, 15]")
    return E2M1_TABLE[nibble]


def e8m0_exponent(e):
    """Unbiased exponent e - 127 of an E8M0 scale byte (scale = 2^(e - 127)); e = 255 is NaN and is rejected."""
    if isinstance(e, (bool, np.bool_)) or not isinstance(e, (int, np.integer)):
        raise TypeError("an E8M0 scale byte is an int")
    e = int(e)
    if not 0 <= e <= 255:
        raise WeightCommitError("E_TILE_ENCODING", f"E8M0 byte {e} is not a uint8")
    if e == E8M0_NAN:
        raise WeightCommitError("E_TILE_ENCODING", "E8M0 0xFF is NaN and has no integer encoding")
    return e - 127


def e8m0_scale(e):
    """Exact float64 2^(e - 127), e in [0, 254] (ldexp of 1.0: an exact power of two inside float64's range)."""
    return float(np.ldexp(1.0, e8m0_exponent(e)))


def _check_u8(arr, shape, what):
    if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
        raise WeightCommitError("E_TILE_SHAPE", f"{what} must be a numpy uint8 array")
    if tuple(arr.shape) != tuple(shape):
        raise WeightCommitError("E_TILE_SHAPE", f"{what} shape {tuple(arr.shape)} != {tuple(shape)}")
    return np.ascontiguousarray(arr)


def _check_k(K):
    if type(K) is not int or not BLOCK <= K <= MAX_K or K % BLOCK:
        raise WeightCommitError("E_TILE_SHAPE", f"tile width K must be an int multiple of {BLOCK} in [{BLOCK}, {MAX_K}]")
    return K


def _check_window(w):
    if type(w) is not int or not MIN_WINDOW_BITS <= w <= MAX_WINDOW_BITS:
        raise WeightCommitError("E_WINDOW", f"window_bits must be an int in [{MIN_WINDOW_BITS}, {MAX_WINDOW_BITS}]")
    return w


def _check_name(name):
    if type(name) is not str or not name:
        raise WeightCommitError("E_TILE_RANGE", "tensor names are non-empty str")
    try:
        nb = name.encode("utf-8")
    except UnicodeEncodeError:
        raise WeightCommitError("E_TILE_RANGE", "tensor name is not valid UTF-8") from None
    if len(nb) > 0xFFFF:
        raise WeightCommitError("E_TILE_RANGE", "tensor name longer than 65535 bytes")
    return nb


def unpack_nibbles(blocks):
    """(R, C / 2) uint8 -> (R, C) uint8 E2M1 codes; element 2 i is the LOW nibble of byte i, 2 i + 1 the high one."""
    if not isinstance(blocks, np.ndarray) or blocks.dtype != np.uint8 or blocks.ndim != 2:
        raise WeightCommitError("E_TILE_SHAPE", "blocks must be a 2-D numpy uint8 array")
    return np.stack([blocks & 0x0F, blocks >> 4], axis=-1).reshape(blocks.shape[0], 2 * blocks.shape[1])


def scale_exponents(scales):
    """uint8 E8M0 bytes -> int64 exponents (e - 127); any 0xFF (NaN) is refused."""
    if not isinstance(scales, np.ndarray) or scales.dtype != np.uint8:
        raise WeightCommitError("E_TILE_SHAPE", "scales must be a numpy uint8 array")
    if bool((scales == E8M0_NAN).any()):
        raise WeightCommitError("E_TILE_ENCODING", "E8M0 0xFF is NaN and has no integer encoding")
    return scales.astype(np.int64) - 127


def dequant_values(blocks, scales):
    """EXACT float64 values of an MXFP4 matrix (rows, C): signed doubled magnitude * 2^(e - 128) -- an integer of at most
    4 bits times a power of two, so no rounding anywhere."""
    nib = unpack_nibbles(blocks)
    cols = nib.shape[1]
    if cols % BLOCK:
        raise WeightCommitError("E_TILE_SHAPE", f"MXFP4 width {cols} is not a multiple of {BLOCK}")
    s = _check_u8(scales, (nib.shape[0], cols // BLOCK), "scales")
    exps = np.repeat(scale_exponents(s), BLOCK, axis=1) - 1
    return np.ldexp(_SIGNED_X2[nib].astype(np.float64), exps)


def _window_full(blocks, scales, window_bits):
    _check_window(window_bits)
    if not isinstance(blocks, np.ndarray) or blocks.ndim != 2 or blocks.shape[0] != ROWS:
        raise WeightCommitError("E_TILE_SHAPE", f"an MXFP4 tile has {ROWS} rows")
    vals = dequant_values(blocks, scales)
    units, shift = window_ints(vals, window_bits)
    err, mx = window_tail(vals, units, shift)
    return units, shift, {"shift": shift, "max_abs_unit": int(np.abs(units).max(initial=0)), "max_abs_value": mx,
                          "tail_abs": err, "quant_window_tail_rel_err": (err / mx) if mx > 0.0 else 0.0,
                          "half_step_bound_rel": 2.0 ** -window_bits}


def materialize_tile(blocks, scales, window_bits=DEFAULT_WINDOW_BITS):
    """(int64 units (32, K), shift): the relation-5/6 W of one tile on the grid 2^-shift, every |unit| < 2^window_bits."""
    units, shift, _ = _window_full(blocks, scales, window_bits)
    return units, shift


def window_report(blocks, scales, window_bits=DEFAULT_WINDOW_BITS):
    """Exact accounting of what the window did to one tile (all zero for an exactly representable tile): the tail is
    the largest |value - unit 2^-shift|; the analytical bound is half a grid step <= 2^-window_bits * max|value|."""
    return _window_full(blocks, scales, window_bits)[2]


def integer_faithful(K, window_bits, x_window_bits=None, max_w=None, max_x=None):
    """True iff K * max|W| * max|X| < 2^62 (the F_p statement then equals the integer statement).  Defaults to the
    WORST case of the windows: |W| <= 2^window_bits, |X| <= 2^x_window_bits (X's window defaults to W's).  This ASSUMES
    |X| <= max_x: neither this module nor relation 6 proves a bound on the committed activations."""
    _check_k(K)
    xb = window_bits if x_window_bits is None else x_window_bits
    _check_window(window_bits)
    _check_window(xb)
    for v in (max_w, max_x):
        if v is not None and (type(v) is not int or v < 0):
            raise WeightCommitError("E_WINDOW", "max |W| / max |X| bounds are non-negative ints")
    mw = (1 << window_bits) if max_w is None else max_w
    mx = (1 << xb) if max_x is None else max_x
    return K * mw * mx < (1 << FAITHFUL_BITS)


# ----------------------------------------------------------------------------- hashing / Merkle tree
def _h(tag, *parts):
    h = hashlib.blake2b(digest_size=32)
    h.update(struct.pack("<H", len(tag)) + tag)
    for p in parts:
        h.update(p)
    return h.digest()


def leaf_hash(name, r0, c0, K, blocks, scales):
    """Leaf of a 32 x K MXFP4 tile: (name, r0, c0, 32, K, layout tag) + raw packed blocks + raw E8M0 scales."""
    nb = _check_name(name)
    _check_k(K)
    for v in (r0, c0):
        if type(v) is not int or not 0 <= v <= MAX_DIM:
            raise WeightCommitError("E_TILE_RANGE", "tile origin must be an int in [0, 2^32)")
    b = _check_u8(blocks, (ROWS, K // 2), "tile blocks")
    s = _check_u8(scales, (ROWS, K // BLOCK), "tile scales")
    return _h(_T_LEAF, struct.pack("<H", len(nb)), nb, struct.pack("<QQII", r0, c0, ROWS, K), _LAYOUT_TAG,
              b.tobytes(), s.tobytes())


def _node(left, right):
    return _h(_T_NODE, left, right)


def build_levels(leaves):
    """Levels of the Merkle tree, leaves padded to a power of two with the PAD leaf."""
    if not leaves:
        raise WeightCommitError("E_MANIFEST", "no leaves")
    m = 1
    while m < len(leaves):
        m <<= 1
    lvl = list(leaves) + [_h(_T_PAD)] * (m - len(leaves))
    levels = [lvl]
    while len(lvl) > 1:
        lvl = [_node(lvl[i], lvl[i + 1]) for i in range(0, len(lvl), 2)]
        levels.append(lvl)
    return levels


def root_from_path(leaf, i, path):
    h = leaf
    for sib in path:
        h = _node(sib, h) if i & 1 else _node(h, sib)
        i >>= 1
    return h


def global_root(manifest, tree_root):
    return _h(_T_ROOT, _h(_T_MANI, manifest), tree_root)


# ----------------------------------------------------------------------------- manifest
@dataclass(frozen=True)
class Mxfp4Entry:
    name: str
    rows: int
    cols: int
    first_leaf: int
    K: int

    @property
    def row_tiles(self): return -(-self.rows // ROWS)

    @property
    def col_tiles(self): return -(-self.cols // self.K)

    @property
    def leaves(self): return self.row_tiles * self.col_tiles

    def leaf_index(self, r0, c0):
        if (type(r0) is not int or type(c0) is not int or r0 % ROWS or c0 % self.K
                or not (0 <= r0 < self.rows and 0 <= c0 < self.cols)):
            raise WeightCommitError("E_TILE_RANGE", f"{self.name}[{r0}, {c0}] is not a canonical tile origin")
        return self.first_leaf + (r0 // ROWS) * self.col_tiles + c0 // self.K


class _Reader:
    """Bounds-checked sequential reader: every short read is a typed rejection, never a silent short slice."""

    def __init__(self, data, code):
        self.d, self.o, self.code = memoryview(data), 0, code

    def take(self, n):
        if n < 0 or self.o + n > len(self.d):
            raise WeightCommitError(self.code, "truncated input")
        out = self.d[self.o:self.o + n]
        self.o += n
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def done(self):
        if self.o != len(self.d):
            raise WeightCommitError(self.code, "trailing bytes")


def encode_manifest(K, entries, window_bits=DEFAULT_WINDOW_BITS, residual=b"\x00" * 32):
    _check_k(K)
    _check_window(window_bits)
    if type(residual) is not bytes or len(residual) != 32:
        raise WeightCommitError("E_MANIFEST", "residual must be a 32-byte digest")
    for e in entries:
        _check_name(e.name)
        if any(type(v) is not int or not 0 <= v < 1 << 64 for v in (e.rows, e.cols, e.first_leaf)):
            raise WeightCommitError("E_MANIFEST", "entry rows / cols / first_leaf must be unsigned 64-bit ints")
    count = sum(e.leaves for e in entries)
    out = [_MANIFEST_MAGIC, residual, struct.pack("<IIQII", ROWS, K, count, len(entries), window_bits)]
    for e in entries:
        nb = e.name.encode("utf-8")
        out.append(struct.pack("<H", len(nb)) + nb + struct.pack("<QQQ", e.rows, e.cols, e.first_leaf))
    return b"".join(out)


def decode_manifest(data):
    """-> (K, window_bits, leaf count, entries, residual); strict: any malformed / non-canonical byte is E_MANIFEST."""
    if not isinstance(data, (bytes, bytearray)) or not bytes(data).startswith(_MANIFEST_MAGIC):
        raise WeightCommitError("E_MANIFEST", "bad magic")
    rd = _Reader(bytes(data), "E_MANIFEST")
    rd.take(len(_MANIFEST_MAGIC))
    residual = bytes(rd.take(32))
    rows, K, count, n, wbits = rd.unpack("<IIQII")
    if rows != ROWS or K % BLOCK or not BLOCK <= K <= MAX_K or not MIN_WINDOW_BITS <= wbits <= MAX_WINDOW_BITS:
        raise WeightCommitError("E_MANIFEST", "unsupported tile geometry or window width")
    if n < 1 or count < 1 or count > MAX_LEAVES:
        raise WeightCommitError("E_MANIFEST", "empty or oversized commitment")
    entries, nxt, names = [], 0, set()
    for _ in range(n):
        (ln,) = rd.unpack("<H")
        try:
            name = bytes(rd.take(ln)).decode("utf-8")
        except UnicodeDecodeError:
            raise WeightCommitError("E_MANIFEST", "tensor name is not UTF-8") from None
        r, c, first = rd.unpack("<QQQ")
        if (not name or name in names or not 1 <= r <= MAX_DIM or not 1 <= c <= MAX_DIM or c % BLOCK
                or first != nxt):
            raise WeightCommitError("E_MANIFEST", "non-canonical entry (name / shape / leaf layout / duplicate)")
        names.add(name)
        e = Mxfp4Entry(name, r, c, first, K)
        nxt += e.leaves
        if nxt > MAX_LEAVES:
            raise WeightCommitError("E_MANIFEST", "oversized commitment")
        entries.append(e)
    rd.done()
    if nxt != count:
        raise WeightCommitError("E_MANIFEST", "leaf-count mismatch")
    return K, wbits, count, entries, residual


def manifest_window_bits(data):
    """The window width bound in the manifest (full strict decode)."""
    return decode_manifest(data)[1]


# ----------------------------------------------------------------------------- sources
def expand_experts(base, blocks, scales):
    """gpt-oss expert layout -> per-expert matrices: blocks (E, rows, in/32, 16) + scales (E, rows, in/32) uint8 ->
    [(f"{base}[{e}]", blocks[e], scales[e])]."""
    if (not isinstance(blocks, np.ndarray) or blocks.ndim != 4 or not isinstance(scales, np.ndarray)
            or scales.ndim != 3 or blocks.shape[0] != scales.shape[0]):
        raise WeightCommitError("E_TILE_SHAPE", "expected blocks (E, rows, in/32, 16) and scales (E, rows, in/32)")
    return [(f"{base}[{e}]", blocks[e], scales[e]) for e in range(blocks.shape[0])]


def experts_from_checkpoint(checkpoint, include=None):
    """Triples (name, blocks, scales) of every `<base>_blocks` / `<base>_scales` U8 pair of a checkpoint reader
    (`weight_commit.SafetensorsCheckpoint` / `ArrayCheckpoint`).  3-D tensors -> one matrix; 4-D -> one per expert."""
    names = set(checkpoint.names())
    out = []
    for n in sorted(names):
        if not n.endswith("_blocks") or (include is not None and not include(n)):
            continue
        base = n[:-len("_blocks")]
        sn = base + "_scales"
        if sn not in names:
            raise WeightCommitError("E_TILE_SHAPE", f"{n} has no {sn}")
        for nm in (n, sn):
            if checkpoint.dtype(nm) not in ("U8", "uint8"):
                raise WeightCommitError("E_TILE_SHAPE", f"{nm} is not a U8 tensor")
        bs = tuple(checkpoint.shape(n))
        b = np.frombuffer(checkpoint.raw(n), dtype=np.uint8).reshape(bs)
        s = np.frombuffer(checkpoint.raw(sn), dtype=np.uint8).reshape(tuple(checkpoint.shape(sn)))
        out.extend(expand_experts(base, b, s) if b.ndim == 4 else [(base, b, s)])
    return out


def _source_map(source):
    """Mapping {name: (blocks, scales)} or an iterable of (name, blocks, scales) triples -> dict; duplicates refused."""
    items = []
    if isinstance(source, Mapping):
        for name, v in source.items():
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                raise WeightCommitError("E_TILE_SHAPE", "a source value is (blocks, scales)")
            items.append((name, v[0], v[1]))
    else:
        for t in source:
            if not isinstance(t, (tuple, list)) or len(t) != 3:
                raise WeightCommitError("E_TILE_SHAPE", "a source item is (name, blocks, scales)")
            items.append(tuple(t))
    out = {}
    for name, b, s in items:
        _check_name(name)
        if name in out:
            raise WeightCommitError("E_DUPLICATE", f"duplicate tensor name {name!r}")
        out[name] = (b, s)
    return out


def _norm_matrix(name, blocks, scales, scan=True):
    """-> (blocks2d (rows, cols/2), scales (rows, cols/32), rows, cols); strict dtype / shape / NaN checks."""
    if not isinstance(blocks, np.ndarray) or blocks.dtype != np.uint8 or not isinstance(scales, np.ndarray) \
            or scales.dtype != np.uint8:
        raise WeightCommitError("E_TILE_SHAPE", f"{name!r}: blocks and scales must be numpy uint8 arrays")
    if blocks.ndim == 3:
        if blocks.shape[2] != BLOCK // 2:
            raise WeightCommitError("E_TILE_SHAPE", f"{name!r}: blocks[..., {BLOCK // 2}] expected")
        blocks = blocks.reshape(blocks.shape[0], blocks.shape[1] * (BLOCK // 2))
    if blocks.ndim != 2:
        raise WeightCommitError("E_TILE_SHAPE", f"{name!r}: blocks must be (rows, cols/2) or (rows, cols/32, 16)")
    rows, cols = blocks.shape[0], 2 * blocks.shape[1]
    if blocks.shape[1] % (BLOCK // 2) or not 1 <= rows <= MAX_DIM or not 1 <= cols <= MAX_DIM:
        raise WeightCommitError("E_TILE_SHAPE", f"{name!r}: shape ({rows}, {cols}) is empty, oversized or K is not a "
                                                f"multiple of {BLOCK}")
    if tuple(scales.shape) != (rows, cols // BLOCK):
        raise WeightCommitError("E_TILE_SHAPE", f"{name!r}: scales shape {tuple(scales.shape)} != {(rows, cols // BLOCK)}")
    if scan and bool((scales == E8M0_NAN).any()):
        raise WeightCommitError("E_TILE_ENCODING", f"{name!r}: E8M0 0xFF (NaN) scale")
    return np.ascontiguousarray(blocks), np.ascontiguousarray(scales), rows, cols


def _tiles(blocks, scales, K, lo, hi):
    """(blocks (T, 32, K/2), scales (T, 32, K/32)) of row tiles [lo, hi), column tiles fastest, zero-padded."""
    cols = 2 * blocks.shape[1]
    ct = -(-cols // K)
    nt = hi - lo
    bp = np.zeros((nt * ROWS, ct * K // 2), dtype=np.uint8)
    sp = np.zeros((nt * ROWS, ct * K // BLOCK), dtype=np.uint8)
    b = blocks[lo * ROWS:hi * ROWS]
    s = scales[lo * ROWS:hi * ROWS]
    bp[:b.shape[0], :b.shape[1]] = b
    sp[:s.shape[0], :s.shape[1]] = s
    bt = bp.reshape(nt, ROWS, ct, K // 2).transpose(0, 2, 1, 3).reshape(-1, ROWS, K // 2)
    st = sp.reshape(nt, ROWS, ct, K // BLOCK).transpose(0, 2, 1, 3).reshape(-1, ROWS, K // BLOCK)
    return bt, st


# ----------------------------------------------------------------------------- one-time commitment
@dataclass
class Mxfp4Commitment:
    K: int
    window_bits: int
    entries: list
    manifest: bytes
    levels: list
    root: bytes
    stats: dict

    def entry(self, name):
        by = {e.name: e for e in self.entries}
        if name not in by:
            raise WeightCommitError("E_TILE_RANGE", f"tensor {name!r} is not committed")
        return by[name]


def commit_mxfp4_weights(source, K=DEFAULT_K, window_bits=DEFAULT_WINDOW_BITS, residual=b"\x00" * 32, threads=8,
                         chunk_row_tiles=64):
    """One-time commitment of every MXFP4 matrix of `source` ({name: (blocks, scales)} or [(name, blocks, scales)]).
    Rejects duplicate / invalid names, non-uint8 or mis-shaped arrays, widths not divisible by 32, NaN (0xFF) scales,
    a bad K, a bad `window_bits` (bound in the manifest, hence in the root) and an empty source."""
    import time
    t0 = time.perf_counter()
    _check_k(K)
    _check_window(window_bits)
    for v in (threads, chunk_row_tiles):
        if type(v) is not int or not 1 <= v <= 1 << 16:
            raise WeightCommitError("E_TILE_SHAPE", "threads / chunk_row_tiles must be ints in [1, 65536]")
    src = _source_map(source)
    if not src:
        raise WeightCommitError("E_MANIFEST", "nothing to commit")
    mats, entries, first = {}, [], 0
    for name in sorted(src):
        b, s, rows, cols = _norm_matrix(name, *src[name])
        mats[name] = (b, s)
        e = Mxfp4Entry(name, rows, cols, first, K)
        first += e.leaves
        if first > MAX_LEAVES:
            raise WeightCommitError("E_MANIFEST", "oversized commitment")
        entries.append(e)
    leaves = [None] * first
    elements = 0
    with ThreadPoolExecutor(max_workers=min(threads, 256)) as pool:
        for e in entries:
            b, s = mats[e.name]
            for lo in range(0, e.row_tiles, chunk_row_tiles):
                hi = min(e.row_tiles, lo + chunk_row_tiles)
                bt, st = _tiles(b, s, K, lo, hi)
                coords = [((lo + t // e.col_tiles) * ROWS, (t % e.col_tiles) * K) for t in range(len(bt))]
                hs = pool.map(lambda a: leaf_hash(e.name, a[1][0], a[1][1], K, bt[a[0]], st[a[0]]),
                              enumerate(coords))
                base = e.first_leaf + lo * e.col_tiles
                for t, hv in enumerate(hs):
                    leaves[base + t] = hv
            elements += e.rows * e.cols
    levels = build_levels(leaves)
    manifest = encode_manifest(K, entries, window_bits, residual)
    root = global_root(manifest, levels[-1][0])
    tile_payload = ROWS * K // 2 + ROWS * K // BLOCK
    stats = {"tensors": len(entries), "leaves": first, "tree_depth": len(levels) - 1, "elements": elements, "K": K,
             "window_bits": window_bits, "manifest_bytes": len(manifest), "leaf_payload_bytes": tile_payload,
             "bf16_tile_bytes": ROWS * K * 2, "compression_vs_bf16": ROWS * K * 2 / tile_payload,
             "total_s": round(time.perf_counter() - t0, 3)}
    return Mxfp4Commitment(K, window_bits, entries, manifest, levels, root, stats)


# ----------------------------------------------------------------------------- per-request opening
def units_to_field(units):
    """[(32, K) int64 units] -> (B, 32, K) canonical F_p uint64."""
    u = np.stack(units).astype(np.int64)
    neg = np.uint64(P) - np.abs(u).astype(np.uint64)
    return np.where(u < 0, neg, u.astype(np.uint64)).astype(np.uint64)


@dataclass(frozen=True)
class OpenedMxfp4Tile:
    name: str
    r0: int
    c0: int
    blocks: np.ndarray       # (32, K / 2) uint8 packed E2M1
    scales: np.ndarray       # (32, K / 32) uint8 E8M0
    path: tuple              # 32-byte sibling hashes, leaf level first

    def leaf(self, K):
        return leaf_hash(self.name, self.r0, self.c0, K, self.blocks, self.scales)

    def window(self, window_bits):
        return materialize_tile(self.blocks, self.scales, window_bits)


@dataclass(frozen=True)
class Mxfp4TileOpening:
    K: int
    manifest: bytes
    tiles: tuple

    def coords(self):
        return [(t.name, t.r0, t.c0) for t in self.tiles]

    def weights_int(self, window_bits=None):
        w = manifest_window_bits(self.manifest) if window_bits is None else _check_window(window_bits)
        return [t.window(w) for t in self.tiles]

    def weights_field(self, window_bits=None):
        """(B, 32, K) canonical F_p uint64 tiles -- the relation-5/6 W."""
        return units_to_field([w for w, _ in self.weights_int(window_bits)])

    def encode(self):
        _check_k(self.K)
        if not isinstance(self.manifest, bytes) or not isinstance(self.tiles, (tuple, list)):
            raise WeightCommitError("E_OPENING", "manifest must be bytes and tiles a sequence")
        out = [_OPENING_MAGIC, struct.pack("<IIQ", self.K, len(self.tiles), len(self.manifest)), self.manifest]
        for t in self.tiles:
            nb = _check_name(t.name)
            if (not isinstance(t.path, (tuple, list)) or len(t.path) > MAX_DEPTH
                    or any(type(p) is not bytes or len(p) != 32 for p in t.path)):
                raise WeightCommitError("E_OPENING", "path entries are 32-byte hashes")
            if any(type(v) is not int or not 0 <= v <= MAX_DIM for v in (t.r0, t.c0)):
                raise WeightCommitError("E_OPENING", "tile origin must be an int in [0, 2^32)")
            out.append(struct.pack("<H", len(nb)) + nb + struct.pack("<QQI", t.r0, t.c0, len(t.path)))
            out.append(b"".join(t.path))
            out.append(_check_u8(t.blocks, (ROWS, self.K // 2), "tile blocks").tobytes())
            out.append(_check_u8(t.scales, (ROWS, self.K // BLOCK), "tile scales").tobytes())
        return b"".join(out)

    @classmethod
    def decode(cls, data):
        if not isinstance(data, (bytes, bytearray)) or not bytes(data).startswith(_OPENING_MAGIC):
            raise WeightCommitError("E_OPENING", "bad magic")
        rd = _Reader(bytes(data), "E_OPENING")
        rd.take(len(_OPENING_MAGIC))
        K, n, ml = rd.unpack("<IIQ")
        if K % BLOCK or not BLOCK <= K <= MAX_K or n > MAX_TILES:
            raise WeightCommitError("E_OPENING", "unsupported tile width or tile count")
        manifest = bytes(rd.take(ml))
        tiles = []
        for _ in range(n):
            (ln,) = rd.unpack("<H")
            try:
                name = bytes(rd.take(ln)).decode("utf-8")
            except UnicodeDecodeError:
                raise WeightCommitError("E_OPENING", "tensor name is not UTF-8") from None
            r0, c0, depth = rd.unpack("<QQI")
            if depth > MAX_DEPTH:
                raise WeightCommitError("E_OPENING", "path deeper than any admissible tree")
            path = tuple(bytes(rd.take(32)) for _ in range(depth))
            blocks = np.frombuffer(rd.take(ROWS * K // 2), dtype=np.uint8).reshape(ROWS, K // 2).copy()
            scales = np.frombuffer(rd.take(ROWS * K // BLOCK), dtype=np.uint8).reshape(ROWS, K // BLOCK).copy()
            tiles.append(OpenedMxfp4Tile(name, r0, c0, blocks, scales, path))
        rd.done()
        return cls(K, manifest, tuple(tiles))


def open_mxfp4_tiles(commitment, source, coords):
    """Opening of the drawn tiles `coords` = [(tensor name, r0, c0)] (relation-5/6 tile order).  Refuses a `source`
    whose tile does not hash to the committed leaf (wrong checkpoint) or carries a NaN scale."""
    src = _source_map(source)
    K = commitment.K
    tiles = []
    for name, r0, c0 in coords:
        e = commitment.entry(name)
        idx = e.leaf_index(r0, c0)
        if name not in src:
            raise WeightCommitError("E_TILE_RANGE", f"tensor {name!r} is missing from the source")
        b, s, rows, cols = _norm_matrix(name, *src[name], scan=False)
        if (rows, cols) != (e.rows, e.cols):
            raise WeightCommitError("E_TILE_SHAPE", f"{name!r} shape differs from the commitment")
        bt, st = _tiles(b, s, K, r0 // ROWS, r0 // ROWS + 1)
        bb, ss = bt[c0 // K], st[c0 // K]
        if bool((ss == E8M0_NAN).any()):
            raise WeightCommitError("E_TILE_ENCODING", f"{name}[{r0}, {c0}]: E8M0 0xFF (NaN) scale")
        t = OpenedMxfp4Tile(name, r0, c0, bb, ss, tuple(merkle_path(commitment.levels, idx)))
        if t.leaf(K) != commitment.levels[0][idx]:
            raise WeightCommitError("E_TILE_PATH", f"{name}[{r0}, {c0}] of the source is not the committed tile")
        tiles.append(t)
    return Mxfp4TileOpening(K, commitment.manifest, tuple(tiles))


# ----------------------------------------------------------------------------- offline verification
def _setup_root(setup_cert_bytes):
    from .setup_cert import SetupCertificate
    if not isinstance(setup_cert_bytes, (bytes, bytearray)):
        raise WeightCommitError("E_SETUP", "setup certificate must be bytes")
    try:
        return SetupCertificate.decode(bytes(setup_cert_bytes)).weight_commitment_root
    except ValueError as exc:
        raise WeightCommitError("E_SETUP", str(exc)) from None


def _vec_profile(container):
    """(profile, relation, layout) of a registered relation-5 or relation-6 container (header + exact C_p length)."""
    from ..backend import folded_vec as FV
    from ..backend import folded_vec_bound as FB
    from ..backend.fpn2 import (FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1, FOLDED_VEC_N16_V1,
                                FOLDED_VEC_N22_V1, header, profile_id)
    if not isinstance(container, (bytes, bytearray)):
        raise WeightCommitError("E_PROFILE", "container must be bytes")
    for prof in (FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1, FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1):
        hdr = header(profile_id(prof))
        if bytes(container[:len(hdr)]) == hdr and len(container) == prof["c_p"]:
            bound = prof["relation"] == 6
            return prof, prof["relation"], (FB if bound else FV)
    raise WeightCommitError("E_PROFILE", "not a registered relation-5 / relation-6 container")


def _check_opening_structure(opening):
    if (type(opening.K) is not int or not isinstance(opening.manifest, bytes)
            or not isinstance(opening.tiles, (tuple, list))
            or not all(isinstance(t, OpenedMxfp4Tile) and isinstance(t.path, (tuple, list)) for t in opening.tiles)):
        raise WeightCommitError("E_OPENING", "malformed opening object")


def check_tile_inclusion(root, opening, expected_coords):
    """Inclusion of every opened tile under `root` + exact draw coordinates; raises WeightCommitError."""
    _check_opening_structure(opening)
    K, wbits, count, entries, _ = decode_manifest(opening.manifest)
    if K != opening.K:
        raise WeightCommitError("E_MANIFEST", "opening K differs from the manifest")
    by_name = {e.name: e for e in entries}
    depth = tree_depth(count)
    try:
        drawn = [tuple(c) for c in expected_coords]
    except TypeError:
        raise WeightCommitError("E_TILE_COORDS", "expected coordinates must be (name, r0, c0) triples") from None
    if drawn != opening.coords():
        raise WeightCommitError("E_TILE_COORDS", "opened tiles are not the drawn tiles (order included)")
    tree_root = None
    for t in opening.tiles:
        e = by_name.get(t.name)
        if e is None:
            raise WeightCommitError("E_TILE_RANGE", f"tensor {t.name!r} is not committed")
        idx = e.leaf_index(t.r0, t.c0)
        if len(t.path) != depth or any(type(p) is not bytes or len(p) != 32 for p in t.path):
            raise WeightCommitError("E_TILE_PATH", "path length differs from the tree depth")
        cand = root_from_path(t.leaf(K), idx, t.path)
        if tree_root is None:
            tree_root = cand
        if cand != tree_root:
            raise WeightCommitError("E_TILE_PATH", f"tile {t.name}[{t.r0}, {t.c0}] is not in the committed tree")
    if tree_root is None or global_root(opening.manifest, tree_root) != root:
        raise WeightCommitError("E_TILE_PATH", "opened tiles do not hash to SetupCertificate.weight_commitment_root")
    return {"tiles": len(opening.tiles), "tree_depth": depth, "leaves": count, "tensors": len(entries),
            "window_bits": wbits}


def verify_mxfp4_tiles_against_setup(setup_cert_bytes, opening, expected_coords, container, engine=None,
                                     expected_window_bits=None, x_window_bits=None):
    """Offline check with ONLY the SetupCertificate (no model weights): `verdict` is "VERIFIED" or "REJECTED: <code>".
    Order: opening decode, container profile, manifest + coordinates + Merkle inclusion under
    `weight_commitment_root` (E_TILE_COORDS / E_TILE_PATH ...), the verifier-pinned window width
    (`expected_window_bits`, compared with the now-authenticated manifest -> E_WINDOW), tile materialisation (a NaN
    scale -> E_TILE_ENCODING), and `weight_root` of the windowed tiles against `container[96:128]` (E_WEIGHT_ROOT).
    Supports relation 5 and relation 6; for relation 6 `x_root` is reported but NOT checked here -- pin it in
    `fpn-verify --x-root`.  `x_window_bits` (default = the manifest's) only feeds the reported `integer_faithful`."""
    import time
    t0 = time.perf_counter()
    rep = {"verdict": "VERIFIED"}
    try:
        if isinstance(opening, (bytes, bytearray)):
            opening = Mxfp4TileOpening.decode(bytes(opening))
        elif not isinstance(opening, Mxfp4TileOpening):
            raise WeightCommitError("E_OPENING", "opening must be Mxfp4TileOpening or its bytes")
        _check_opening_structure(opening)
        if expected_window_bits is not None:
            _check_window(expected_window_bits)
        prof, relation, M = _vec_profile(container)
        lay = M.layout_of(prof)
        if len(opening.tiles) != lay.B or opening.K != lay.K:
            raise WeightCommitError("E_PROFILE", f"profile {prof['name']} proves {lay.B} tiles of K = {lay.K}")
        inc = check_tile_inclusion(_setup_root(setup_cert_bytes), opening, expected_coords)
        rep.update(inc)
        wbits = inc["window_bits"]
        if expected_window_bits is not None and expected_window_bits != wbits:
            raise WeightCommitError("E_WINDOW", f"manifest window {wbits} != pinned {expected_window_bits}")
        t1 = time.perf_counter()
        full = [_window_full(t.blocks, t.scales, wbits) for t in opening.tiles]
        max_w = max(r["max_abs_unit"] for _, _, r in full)
        xb = wbits if x_window_bits is None else _check_window(x_window_bits)
        W = units_to_field([u for u, _, _ in full])
        from ..backend import folded_vec as FV
        FV._W_CACHE.clear()
        root = M.weight_root(lay, W, engine)
        pinned = bytes(container[96:128])
        rep.update(profile=prof["name"], relation=relation, weight_root=root.hex(), container_weight_root=pinned.hex(),
                   x_root=bytes(container[128:160]).hex() if relation == 6 else None,
                   max_abs_unit=max_w, window_strict=max_w < (1 << wbits),
                   integer_faithful=integer_faithful(lay.K, wbits, xb, max_w=max_w),
                   x_window_bits=xb, quant_window_tail_rel_err=max(r["quant_window_tail_rel_err"] for _, _, r in full),
                   leaf_payload_bytes=ROWS * lay.K // 2 + ROWS * lay.K // BLOCK,
                   inclusion_ms=round((t1 - t0) * 1e3, 2), weight_root_ms=round((time.perf_counter() - t1) * 1e3, 2))
        if root != pinned:
            raise WeightCommitError("E_WEIGHT_ROOT", "container weight_root differs from the opened tiles' root")
    except WeightCommitError as exc:
        rep["verdict"] = f"REJECTED: {exc.code}"
        rep["detail"] = str(exc)
    return rep
