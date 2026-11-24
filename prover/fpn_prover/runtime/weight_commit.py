# SPDX-License-Identifier: Apache-2.0
"""One-time global model-weight commitment and per-request drawn-tile openings .

Closes the gap "weight_root is per request": relation 5 (`FOLDED_VEC_MATMUL`) pins a `weight_root` that
commits only the B drawn tiles, so on its own it says nothing about WHICH model those tiles came from.  Here:

  * `commit_model_weights` (once per deployment) partitions every 2-D BF16 weight matrix of a checkpoint into
    canonical 32 x K tiles (rows [r0, r0 + 32) x columns [c0, c0 + K), zero-padded at the matrix edge), encodes
    each tile EXACTLY as signed integer units with one power-of-two shift per tile (value = unit * 2^-shift, the
    relation-5 encoding `folded_vec.bf16_words_to_ints`), hashes (tensor name, r0, c0, 32, K, shift, units) into a
    BLAKE2b-256 leaf (person `FPN2-WTILE-v1`) and builds ONE Merkle tree over all leaves.  The 32-byte
    `root = H_root(H_manifest(manifest) || tree_root)` is `SetupCertificate.weight_commitment_root`; the manifest
    (K, leaf count, every tensor's name / shape / first leaf) is bound into the root, so a verifier derives each
    tile's leaf index from public data instead of trusting the prover.
    Every tensor that is NOT tiled (1-D norms, non-BF16 tensors) is hashed (name, dtype, shape, raw bytes) into a
    residual digest carried in the manifest, so the root binds the whole checkpoint.
  * `open_drawn_tiles` (per request) emits, for the B drawn tiles of a relation-5 proof, the tiles' integer units
    and their Merkle inclusion paths (32 bytes x tree depth each).
  * `verify_drawn_tiles_against_setup` -- given ONLY the SetupCertificate, the opening, the relation-5 container
    and the expected draw coordinates (re-derived by the verifier from public inputs) -- checks
      1. every opened tile sits at the leaf the manifest assigns to its coordinates under
         `SetupCertificate.weight_commitment_root` (inclusion), and the coordinates are exactly the drawn ones, in
         order (a genuine tile of ANOTHER layer is in the tree, so only this check rejects it); and
      2. `folded_vec.weight_root` of the opened tiles equals the `weight_root` pinned in the container's statement,
    and returns that root so the caller pins it in `fpn-verify --circuit-root`.

-- block-128 FP8 E4M3 matrices and the capped exponent window:

  * a 2-D `F8_E4M3` tensor with a companion `<name>_scale_inv` F32 tensor of shape [ceil(rows/128),
    ceil(cols/128)] (DeepSeek / Qwen3-FP8 block quantisation, value = fp8 * scale_inv[r // 128][c // 128]) is TILED
    LOSSLESSLY: its leaf (person `FPN2-WTILE8-v1`) hashes the raw e4m3 bytes of the 32 x K tile (zero-padded) and
    the K / 128 raw F32 block scales of the tile's row block, so the root binds the checkpoint bytes exactly
    (the scale tensors consumed by FP8 leaves are not re-hashed into the residual digest);
  * relation 5 needs integers: `window_ints` maps the exact dequantised tile (every fp8 * F32 product is exact in
    float64) to signed units on ONE power-of-two grid chosen from the tile's max |value| so that every
    |unit| <= 2^window_int_bits (round-half-even on the grid; values below the grid's half step become 0).  With
    the default `WINDOW_INT_BITS = 23` (= 7 + MAX_SHIFT_SPAN, MAX_SHIFT_SPAN = 16: for a BF16 tile, exactly the
    entries within 16 binades of the tile's top binade are kept exact) and X windowed the same way,
    sum_j |W||X| <= 1024 * 2^23 * 2^23 = 2^56 < 2^62, so the F_p statement is the integer statement.  The window is
    a PUBLIC deterministic function applied AFTER opening (prover and verifier call the same `OpenedTile.window`);
    the commitment itself stays lossless.  For FP8 entries the window width is bound in the v2 manifest; for BF16
    entries the verifier passes it (`verify_drawn_tiles_against_setup(..., window_int_bits=)`), None = the exact
    encoding.  The rounding error of the window is NOT hidden: callers report it per tile.
  * BF16-only checkpoints keep the v1 manifest / leaf / opening bytes, so every root is unchanged.

This module never reads the process environment.  Scope: BF16 and block-128 FP8 E4M3 matrices are TILED
(openable); MXFP4 / F32 / 1-D tensors are bound only through the residual digest (no per-tile opening); a BF16 tile
whose exponent spread exceeds 54 bits has no int64 unit encoding and is refused (none occurs in the checkpoints
measured, see the receipt).
"""
import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

ROWS = 32
DEFAULT_K = 1024
P = 0xFFFFFFFF00000001

_P_LEAF = b"FPN2-WTILE-v1"
_P_NODE = b"FPN2-WNODE-v1"
_P_PAD = b"FPN2-WPAD-v1"
_P_MANIFEST = b"FPN2-WMANI-v1"
_P_ROOT = b"FPN2-WROOT-v1"
_P_RESIDUAL = b"FPN2-WRESID-v1"
_MANIFEST_MAGIC = b"FPN2_WMANIFEST\x01"
_MANIFEST_MAGIC_V2 = b"FPN2_WMANIFEST\x02"
_OPENING_MAGIC = b"FPN2_WOPEN\x01"
_OPENING_MAGIC_V2 = b"FPN2_WOPEN\x02"
_P_LEAF8 = b"FPN2-WTILE8-v1"
_FP8_TAG = b"E4M3/B128/F32"
_MAX_SHIFT_SPREAD = 54        # 8-bit significand << 54 still fits int64
FP8_BLOCK = 128
MAX_SHIFT_SPAN = 16           # BF16: entries within 16 binades of the tile's top binade stay exact
WINDOW_INT_BITS = 7 + MAX_SHIFT_SPAN    # |unit| <= 2^23 (8-bit BF16 significand << 15)
KIND_BF16, KIND_FP8 = 0, 1


class WeightCommitError(ValueError):
    """Opening / binding failure; `code` is the stable rejection code."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _h(person, *parts):
    h = hashlib.blake2b(digest_size=32, person=person)
    for p in parts:
        h.update(p)
    return h.digest()


# ----------------------------------------------------------------------------- checkpoint access
_ST_DTYPES = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8,
              "U8": np.uint8, "I8": np.uint8, "F64": np.uint64, "I64": np.uint64, "I32": np.uint32,
              "U32": np.uint32, "I16": np.uint16, "BOOL": np.uint8}


class SafetensorsCheckpoint:
    """Minimal read-only safetensors reader (one file, or a directory of shards): headers parsed once, tensor
    bytes read on demand with explicit offsets (no memory maps, no file handles kept open)."""

    def __init__(self, path):
        import os
        files = [path] if os.path.isfile(path) else sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors under {path}")
        self.meta = {}
        for f in files:
            fsize = os.path.getsize(f)
            with open(f, "rb") as fh:
                hdr_len_bytes = fh.read(8)
                if len(hdr_len_bytes) != 8:
                    raise ValueError(f"truncated safetensors header length in {f}")
                hl = struct.unpack("<Q", hdr_len_bytes)[0]
                hdr_bytes = fh.read(hl)
                if len(hdr_bytes) != hl:
                    raise ValueError(f"truncated safetensors header in {f}")
                hdr = json.loads(hdr_bytes)
            for name, v in hdr.items():
                if name == "__metadata__":
                    continue
                if name in self.meta:
                    raise ValueError(f"duplicate tensor {name!r} across safetensors shards")
                lo, hi = v["data_offsets"]
                abs_lo, abs_hi = 8 + hl + lo, 8 + hl + hi
                if not (0 <= lo <= hi and abs_hi <= fsize):
                    raise ValueError(f"invalid data_offsets {(lo, hi)} for {name!r} in {f} (size {fsize})")
                self.meta[name] = (f, abs_lo, abs_hi - abs_lo, v["dtype"], tuple(v["shape"]))

    def names(self):
        return sorted(self.meta)

    def dtype(self, name):
        return self.meta[name][3]

    def shape(self, name):
        return self.meta[name][4]

    def raw(self, name):
        """All bytes of a tensor (any dtype / rank), validated against declared byte length."""
        f, off, span, dt, shape = self.meta[name]
        n = np.dtype(_ST_DTYPES[dt]).itemsize
        for d in shape:
            n *= d
        if span != n:
            raise ValueError(f"tensor {name!r} offset span {span} != declared byte size {n}")
        with open(f, "rb") as fh:
            fh.seek(off)
            buf = fh.read(n)
        if len(buf) != n:
            raise ValueError(f"short read for tensor {name!r}: got {len(buf)} of {n} bytes")
        return buf

    def rows(self, name, start, stop):
        """Rows [start, stop) of a 2-D tensor as raw words (uint16 for BF16)."""
        f, off, span, dt, shape = self.meta[name]
        cols = shape[1]
        wd = np.dtype(_ST_DTYPES[dt])
        stop = min(stop, shape[0])
        need = (stop - start) * cols * wd.itemsize
        if start < 0 or stop < start or (stop * cols * wd.itemsize) > span:
            raise ValueError(f"row slice [{start}, {stop}) out of bounds for {name!r}")
        with open(f, "rb") as fh:
            fh.seek(off + start * cols * wd.itemsize)
            buf = fh.read(need)
        if len(buf) != need:
            raise ValueError(f"short row read for {name!r}: got {len(buf)} of {need} bytes")
        return np.frombuffer(buf, dtype=wd.newbyteorder("<")).reshape(stop - start, cols)


class ArrayCheckpoint:
    """In-memory checkpoint {name: 2-D uint16 BF16 words} (tests, live-engine weights).  `dtypes` optionally names
    safetensors dtypes per tensor (e.g. {"w": "F8_E4M3", "w_scale_inv": "F32"}; uint8 / float32 arrays)."""

    def __init__(self, tensors, dtypes=None):
        self.t = {k: np.asarray(v) for k, v in tensors.items()}
        self.dt = dict(dtypes or {})

    def names(self):
        return sorted(self.t)

    def dtype(self, name):
        if name in self.dt:
            return self.dt[name]
        return "BF16" if self.t[name].dtype == np.uint16 else str(self.t[name].dtype)

    def shape(self, name):
        return tuple(self.t[name].shape)

    def rows(self, name, start, stop):
        return self.t[name][start:stop]

    def raw(self, name):
        return np.ascontiguousarray(self.t[name]).tobytes()


def as_checkpoint(source):
    if isinstance(source, (SafetensorsCheckpoint, ArrayCheckpoint)) or all(
            callable(getattr(source, a, None)) for a in ("names", "dtype", "shape", "rows", "raw")):
        return source                     # any checkpoint object with the reader interface (e.g. an overlay)
    if isinstance(source, dict):
        return ArrayCheckpoint(source)
    return SafetensorsCheckpoint(str(source))


# ----------------------------------------------------------------------------- exact tile encoding
def tile_units(words):
    """BF16 words (T, 32, K) -> (int64 units (T, 32, K), shift (T,) int64): value = unit * 2^-shift per tile.

    Identical to `folded_vec.bf16_words_to_ints` applied to each tile separately (shift = 134 - min exponent over
    the tile's nonzero entries; 0 for an all-zero tile)."""
    w = np.asarray(words, dtype=np.uint16).astype(np.uint32)
    exp = (w >> 7) & 0xFF
    if (exp == 0xFF).any():
        raise WeightCommitError("E_TILE_ENCODING", "BF16 inf / nan has no integer encoding")
    man = ((w & 0x7F) | np.where(exp > 0, 0x80, 0)).astype(np.int64)
    e = np.where(exp > 0, exp, 1).astype(np.int64)
    nz = man != 0
    emin = np.where(nz, e, 1 << 20).min(axis=(1, 2))
    emin = np.where(emin == 1 << 20, 134, emin)
    shift = np.where(nz, e - emin[:, None, None], 0)
    if shift.size and int(shift.max()) > _MAX_SHIFT_SPREAD:
        raise WeightCommitError("E_TILE_ENCODING", "tile exponent spread exceeds the int64 unit encoding")
    mag = man << shift
    units = np.where((w >> 15).astype(bool), -mag, mag)
    return units, (134 - emin).astype(np.int64)


# ----------------------------------------------------------------------------- FP8 E4M3 + capped exponent window
def _e4m3_table():
    """float64 value of every e4m3fn byte (NaN for 0x7f / 0xff; no infinities in e4m3fn)."""
    out = np.empty(256, dtype=np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            v = float("nan")
        elif e == 0:
            v = m * 2.0 ** -9
        else:
            v = (8 + m) * 2.0 ** (e - 10)
        out[b] = -v if s else v
    return out


E4M3_TABLE = _e4m3_table()


def fp8_dequant(raw, scales, block=FP8_BLOCK):
    """Exact float64 value of an e4m3 tile (rows, K) uint8 times its per-column-block F32 scales (K / block,): every
    product of a 4-bit significand and a 24-bit F32 significand is exact in float64."""
    raw = np.asarray(raw, dtype=np.uint8)
    sc = np.asarray(scales, dtype="<f4").astype(np.float64)
    if raw.shape[-1] != block * len(sc):
        raise WeightCommitError("E_TILE_ENCODING", "scale count does not match the tile width")
    if (((raw & 0x7F) == 0x7F)).any() or not np.isfinite(sc).all():
        raise WeightCommitError("E_TILE_ENCODING", "FP8 nan / non-finite scale has no integer encoding")
    return E4M3_TABLE[raw] * np.repeat(sc, block)


def window_ints(values, int_bits=WINDOW_INT_BITS):
    """Exact float64 `values` (any shape) -> (int64 units, shift) on ONE grid 2^-shift with every |unit| <= 2^int_bits:
    shift = int_bits - E where max |value| = f 2^E, f in [0.5, 1); units = round-half-even(value * 2^shift) (an exact
    power-of-two scaling, so np.rint is the correctly rounded result).  All-zero input -> (zeros, 0).  Deterministic
    and public: the prover and the verifier call the same function."""
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise WeightCommitError("E_TILE_ENCODING", "non-finite value has no integer encoding")
    if type(int_bits) is not int or not 1 <= int_bits <= 52:
        raise ValueError("int_bits must be an int in [1, 52]")
    m = float(np.abs(v).max()) if v.size else 0.0
    if m == 0.0:
        return np.zeros(v.shape, dtype=np.int64), 0
    shift = int_bits - int(np.frexp(m)[1])
    units = np.rint(np.ldexp(v, shift))
    out = units.astype(np.int64)
    if int(np.abs(out).max()) > 1 << int_bits:
        raise AssertionError("window overflow")          # unreachable: max |v| 2^shift < 2^int_bits
    return out, shift


def window_tail(values, units, shift):
    """(max |value - unit 2^-shift|, max |value|) -- the rounding error the window introduced (0 if exact)."""
    v = np.asarray(values, dtype=np.float64)
    approx = np.ldexp(np.asarray(units, dtype=np.float64), -shift)
    return float(np.abs(v - approx).max(initial=0.0)), float(np.abs(v).max(initial=0.0))


def bf16_values(words):
    """Exact float64 value of BF16 words."""
    w = np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16
    return w.view(np.float32).astype(np.float64)


def leaf_hash8(name, r0, c0, K, raw, scales):
    """Leaf of a block-128 FP8 E4M3 tile: raw e4m3 bytes (32, K) + the K / 128 raw F32 scales of its row block."""
    nb = name.encode()
    r = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8))
    sc = np.ascontiguousarray(np.asarray(scales, dtype="<f4"))
    if r.shape != (ROWS, K) or sc.shape != (K // FP8_BLOCK,) or K % FP8_BLOCK:
        raise WeightCommitError("E_TILE_SHAPE", f"FP8 tile must be {ROWS} x {K} with {K // FP8_BLOCK} block scales")
    return _h(_P_LEAF8, struct.pack("<H", len(nb)), nb, struct.pack("<QQII", r0, c0, ROWS, K), _FP8_TAG,
              r.tobytes(), sc.tobytes())


def leaf_hash(name, r0, c0, K, shift, units):
    nb = name.encode()
    u = np.ascontiguousarray(np.asarray(units, dtype=np.int64)).astype("<i8", copy=False)
    if u.shape != (ROWS, K):
        raise WeightCommitError("E_TILE_SHAPE", f"tile must be {ROWS} x {K}")
    return _h(_P_LEAF, struct.pack("<H", len(nb)), nb, struct.pack("<QQIIq", r0, c0, ROWS, K, int(shift)),
              u.tobytes())


# ----------------------------------------------------------------------------- manifest
@dataclass(frozen=True)
class TensorEntry:
    name: str
    rows: int
    cols: int
    first_leaf: int
    K: int
    kind: int = KIND_BF16

    @property
    def row_tiles(self): return -(-self.rows // ROWS)

    @property
    def col_tiles(self): return -(-self.cols // self.K)

    @property
    def leaves(self): return self.row_tiles * self.col_tiles

    def leaf_index(self, r0, c0):
        if r0 % ROWS or c0 % self.K or not (0 <= r0 < self.rows and 0 <= c0 < self.cols):
            raise WeightCommitError("E_TILE_RANGE", f"{self.name}[{r0}, {c0}] is not a canonical tile origin")
        return self.first_leaf + (r0 // ROWS) * self.col_tiles + c0 // self.K


def encode_manifest(K, entries, residual=b"\x00" * 32, window_int_bits=WINDOW_INT_BITS):
    """`residual` = digest of every tensor NOT tiled (1-D norms, F32 / MXFP4 tensors): name, dtype, shape and raw
    bytes, so the root binds the whole checkpoint, not only the tiled matrices.  BF16-only -> v1 bytes (unchanged);
    any FP8 entry -> v2: (block, window_int_bits) in the header and one kind byte per entry."""
    count = sum(e.leaves for e in entries)
    v2 = any(e.kind != KIND_BF16 for e in entries)
    if not v2:
        out = [_MANIFEST_MAGIC, residual, struct.pack("<IIQI", ROWS, K, count, len(entries))]
    else:
        out = [_MANIFEST_MAGIC_V2, residual, struct.pack("<IIQIII", ROWS, K, count, len(entries), FP8_BLOCK,
                                                         window_int_bits)]
    for e in entries:
        nb = e.name.encode()
        out.append(struct.pack("<H", len(nb)) + nb + struct.pack("<QQQ", e.rows, e.cols, e.first_leaf)
                   + (struct.pack("<B", e.kind) if v2 else b""))
    return b"".join(out)


def manifest_window_int_bits(data):
    """Window width bound in a v2 manifest (None for a v1 / BF16-only manifest)."""
    if data.startswith(_MANIFEST_MAGIC_V2):
        return struct.unpack_from("<I", data, len(_MANIFEST_MAGIC_V2) + 32 + 24)[0]
    return None


def decode_manifest(data):
    v2 = data.startswith(_MANIFEST_MAGIC_V2)
    if not v2 and not data.startswith(_MANIFEST_MAGIC):
        raise WeightCommitError("E_MANIFEST", "bad magic")
    off = len(_MANIFEST_MAGIC) + 32                         # residual digest (opaque to the opening check)
    try:
        if v2:
            rows, K, count, n, block, wbits = struct.unpack_from("<IIQIII", data, off)
            off += 28
            if block != FP8_BLOCK or not 1 <= wbits <= 52 or K % FP8_BLOCK:
                raise WeightCommitError("E_MANIFEST", "unsupported FP8 block / window geometry")
        else:
            rows, K, count, n = struct.unpack_from("<IIQI", data, off)
            off += 20
        if rows != ROWS or K < 1:
            raise WeightCommitError("E_MANIFEST", "unsupported tile geometry")
        entries, nxt = [], 0
        for _ in range(n):
            (ln,) = struct.unpack_from("<H", data, off)
            name = data[off + 2:off + 2 + ln].decode()
            off += 2 + ln
            r, c, first = struct.unpack_from("<QQQ", data, off)
            off += 24
            kind = KIND_BF16
            if v2:
                (kind,) = struct.unpack_from("<B", data, off)
                off += 1
                if kind not in (KIND_BF16, KIND_FP8):
                    raise WeightCommitError("E_MANIFEST", "unknown tile kind")
            e = TensorEntry(name, r, c, first, K, kind)
            if first != nxt or r < 1 or c < 1:
                raise WeightCommitError("E_MANIFEST", "non-canonical leaf layout")
            nxt += e.leaves
            entries.append(e)
    except (struct.error, UnicodeDecodeError) as exc:
        raise WeightCommitError("E_MANIFEST", f"truncated / malformed manifest: {exc}") from None
    if off != len(data) or nxt != count or len({e.name for e in entries}) != n:
        raise WeightCommitError("E_MANIFEST", "trailing bytes, leaf-count mismatch or duplicate tensor")
    if v2 and not any(e.kind == KIND_FP8 for e in entries):
        raise WeightCommitError("E_MANIFEST", "non-canonical v2 manifest without an FP8 entry")
    return K, count, entries


# ----------------------------------------------------------------------------- Merkle tree
def _pad_leaf():
    return _h(_P_PAD)


def build_levels(leaves):
    """[(m, 32) uint8] levels, leaves padded to a power of two with the PAD leaf."""
    m = 1
    while m < len(leaves):
        m <<= 1
    lvl = list(leaves) + [_pad_leaf()] * (m - len(leaves))
    levels = [lvl]
    while len(lvl) > 1:
        lvl = [_h(_P_NODE, lvl[i], lvl[i + 1]) for i in range(0, len(lvl), 2)]
        levels.append(lvl)
    return levels


def merkle_path(levels, i):
    return [levels[k][(i >> k) ^ 1] for k in range(len(levels) - 1)]


def merkle_root_from_path(leaf, i, path):
    h = leaf
    for sib in path:
        h = _h(_P_NODE, sib, h) if i & 1 else _h(_P_NODE, h, sib)
        i >>= 1
    return h


def tree_depth(count):
    d = 0
    while (1 << d) < count:
        d += 1
    return d


def global_root(manifest, tree_root):
    return _h(_P_ROOT, _h(_P_MANIFEST, manifest), tree_root)


# ----------------------------------------------------------------------------- one-time commitment
@dataclass
class WeightCommitment:
    K: int
    entries: list
    manifest: bytes
    levels: list
    root: bytes
    stats: dict

    def entry(self, name):
        idx = self.__dict__.get("_by_name")
        if idx is None or len(idx) != len(self.entries):
            idx = self.__dict__["_by_name"] = {e.name: e for e in self.entries}
        e = idx.get(name)
        if e is None:
            raise WeightCommitError("E_TILE_RANGE", f"tensor {name!r} is not committed")
        return e


def _tensor_tiles(ck, name, K, row_tile_lo, row_tile_hi):
    """BF16 words (T, 32, K) of row tiles [lo, hi) of a 2-D tensor (zero-padded to whole tiles)."""
    rows, cols = ck.shape(name)
    ct = -(-cols // K)
    w = ck.rows(name, row_tile_lo * ROWS, row_tile_hi * ROWS)
    pad = np.zeros(((row_tile_hi - row_tile_lo) * ROWS, ct * K), dtype=np.uint16)
    pad[:w.shape[0], :cols] = w
    return pad.reshape(row_tile_hi - row_tile_lo, ROWS, ct, K).transpose(0, 2, 1, 3).reshape(-1, ROWS, K)


def scale_name(name):
    """Companion block-scale tensor of an FP8 weight (`x.weight` -> `x.weight_scale_inv`)."""
    return name + "_scale_inv"


def _fp8_scale_shape(rows, cols):
    return (-(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK))


def _is_fp8_matrix(ck, name, names):
    if ck.dtype(name) != "F8_E4M3" or len(ck.shape(name)) != 2:
        return False
    sn = scale_name(name)
    if sn not in names:
        return False
    if ck.dtype(sn) != "F32" or tuple(ck.shape(sn)) != _fp8_scale_shape(*ck.shape(name)):
        raise WeightCommitError("E_TILE_SHAPE", f"{sn} is not an F32 block-{FP8_BLOCK} scale of {name}")
    return True


def _tensor_tiles8(ck, name, K, row_tile_lo, row_tile_hi):
    """(raw e4m3 bytes (T, 32, K), F32 block scales (T, K / 128)) of row tiles [lo, hi) (zero-padded)."""
    rows, cols = ck.shape(name)
    ct = -(-cols // K)
    w = np.asarray(ck.rows(name, row_tile_lo * ROWS, row_tile_hi * ROWS), dtype=np.uint8)
    nt = row_tile_hi - row_tile_lo
    pad = np.zeros((nt * ROWS, ct * K), dtype=np.uint8)
    pad[:w.shape[0], :cols] = w
    raw = pad.reshape(nt, ROWS, ct, K).transpose(0, 2, 1, 3).reshape(-1, ROWS, K)
    srows, scols = _fp8_scale_shape(rows, cols)
    rb_lo, rb_hi = (row_tile_lo * ROWS) // FP8_BLOCK, -(-(row_tile_hi * ROWS) // FP8_BLOCK)
    sc = np.asarray(ck.rows(scale_name(name), rb_lo, min(rb_hi, srows)))
    sc = sc.view("<f4") if sc.dtype.kind == "u" and sc.dtype.itemsize == 4 else sc.astype("<f4", copy=False)
    spad = np.zeros((rb_hi - rb_lo, ct * K // FP8_BLOCK), dtype="<f4")
    spad[:sc.shape[0], :scols] = sc
    per_rb = K // FP8_BLOCK
    scales = np.empty((nt * ct, per_rb), dtype="<f4")
    for t in range(nt):
        rb = ((row_tile_lo + t) * ROWS) // FP8_BLOCK - rb_lo
        for c in range(ct):
            scales[t * ct + c] = spad[rb, c * per_rb:(c + 1) * per_rb]
    return raw, scales


def commit_model_weights(source, K=DEFAULT_K, threads=8, chunk_row_tiles=256, include=None,
                         window_int_bits=WINDOW_INT_BITS):
    """One-time global commitment of every 2-D BF16 matrix and every block-128 FP8 E4M3 matrix (+ its F32
    `weight_scale_inv`) of `source` (safetensors file / directory, a dict {name: uint16 words} or a checkpoint
    object).  `include(name) -> bool` optionally narrows the tensor set.  `window_int_bits` is bound in the manifest
    only when an FP8 matrix is committed (it fixes how FP8 tiles map to relation-5 integers)."""
    import time
    t0 = time.perf_counter()
    ck = as_checkpoint(source)
    all_names = set(ck.names())
    entries, skipped, first = [], [], 0
    consumed_scales = set()
    for name in ck.names():
        shape, dt = ck.shape(name), ck.dtype(name)
        if include is not None and not include(name):
            continue
        if len(shape) == 2 and dt == "BF16":
            kind = KIND_BF16
        elif _is_fp8_matrix(ck, name, all_names):
            if K % FP8_BLOCK:
                raise WeightCommitError("E_TILE_SHAPE", f"K = {K} is not a multiple of the FP8 block {FP8_BLOCK}")
            kind = KIND_FP8
            consumed_scales.add(scale_name(name))
        else:
            skipped.append({"name": name, "dtype": dt, "shape": list(shape)})
            continue
        e = TensorEntry(name, shape[0], shape[1], first, K, kind)
        first += e.leaves
        entries.append(e)
    skipped = [sk for sk in skipped if sk["name"] not in consumed_scales]
    leaves = [None] * first
    elements = 0
    max_spread = 0
    fp8_tensors = fp8_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for e in entries:
            for lo in range(0, e.row_tiles, chunk_row_tiles):
                hi = min(e.row_tiles, lo + chunk_row_tiles)
                base = e.first_leaf + lo * e.col_tiles
                coords = [((lo + t // e.col_tiles) * ROWS, (t % e.col_tiles) * K)
                          for t in range((hi - lo) * e.col_tiles)]
                if e.kind == KIND_FP8:
                    raw, scales = _tensor_tiles8(ck, e.name, K, lo, hi)
                    hs = pool.map(lambda a: leaf_hash8(e.name, a[1][0], a[1][1], K, raw[a[0]], scales[a[0]]),
                                  enumerate(coords))
                else:
                    units, shifts = tile_units(_tensor_tiles(ck, e.name, K, lo, hi))
                    hs = pool.map(lambda a: leaf_hash(e.name, a[1][0], a[1][1], K, shifts[a[0]], units[a[0]]),
                                  enumerate(coords))
                    max_spread = max(max_spread, int(shifts.max()) if len(shifts) else 0)
                for t, hv in enumerate(hs):
                    leaves[base + t] = hv
            elements += e.rows * e.cols
            if e.kind == KIND_FP8:
                fp8_tensors += 1
                fp8_bytes += e.rows * e.cols
    t_leaves = time.perf_counter() - t0
    res = hashlib.blake2b(digest_size=32, person=_P_RESIDUAL)
    residual_bytes = 0
    for sk in skipped:
        raw = ck.raw(sk["name"])
        nb = sk["name"].encode()
        meta = json.dumps([sk["dtype"], sk["shape"]]).encode()
        res.update(struct.pack("<HI", len(nb), len(meta)) + nb + meta + struct.pack("<Q", len(raw)))
        res.update(raw)
        residual_bytes += len(raw)
    t_res = time.perf_counter() - t0 - t_leaves
    levels = build_levels(leaves)
    manifest = encode_manifest(K, entries, res.digest(), window_int_bits)
    root = global_root(manifest, levels[-1][0])
    stats = {"tensors": len(entries), "skipped": skipped, "leaves": first, "tree_depth": len(levels) - 1,
             "elements": elements, "K": K, "rows": ROWS, "manifest_bytes": len(manifest),
             "max_tile_shift": max_spread, "leaf_hash_s": round(t_leaves, 3),
             "residual_tensors": len(skipped), "residual_bytes": residual_bytes, "residual_hash_s": round(t_res, 3),
             "fp8_tensors": fp8_tensors, "fp8_bytes": fp8_bytes, "fp8_scale_tensors_in_leaves": len(consumed_scales),
             "manifest_version": 2 if fp8_tensors else 1,
             "window_int_bits_bound": window_int_bits if fp8_tensors else None,
             "total_s": round(time.perf_counter() - t0, 3)}
    return WeightCommitment(K, entries, manifest, levels, root, stats)


# ----------------------------------------------------------------------------- per-request opening
@dataclass(frozen=True)
class OpenedTile:
    name: str
    r0: int
    c0: int
    shift: int
    units: np.ndarray        # (32, K) int64 (BF16 tiles); None for FP8 tiles
    path: tuple              # sibling hashes, leaf level first
    kind: int = KIND_BF16
    raw: object = None       # FP8: (32, K) uint8 e4m3 bytes
    scales: object = None    # FP8: (K / 128,) F32 block scales of the tile's row block

    def leaf(self, K):
        if self.kind == KIND_FP8:
            return leaf_hash8(self.name, self.r0, self.c0, K, self.raw, self.scales)
        return leaf_hash(self.name, self.r0, self.c0, K, self.shift, self.units)

    def values(self):
        """Exact float64 values of the committed tile (BF16 units * 2^-shift are exact: 8-bit significands)."""
        if self.kind == KIND_FP8:
            return fp8_dequant(self.raw, self.scales)
        return np.ldexp(np.asarray(self.units, dtype=np.float64), -int(self.shift))

    def window(self, int_bits=None):
        """(int64 units (32, K), shift) of the relation-5 W: BF16 with int_bits None -> the exact units;
        otherwise `window_ints(values, int_bits)` (FP8 tiles always need a window)."""
        if self.kind == KIND_BF16 and int_bits is None:
            return np.asarray(self.units, dtype=np.int64), int(self.shift)
        if int_bits is None:
            raise WeightCommitError("E_TILE_ENCODING", "an FP8 tile needs a window width")
        return window_ints(self.values(), int_bits)


@dataclass(frozen=True)
class TileOpening:
    K: int
    manifest: bytes
    tiles: tuple

    def coords(self):
        return [(t.name, t.r0, t.c0) for t in self.tiles]

    def weights_int(self, bf16_window=None):
        """[(int64 units (32, K), shift)] per tile: FP8 tiles windowed at the manifest-bound width, BF16 tiles at
        `bf16_window` (None = exact)."""
        w8 = manifest_window_int_bits(self.manifest)
        return [t.window(w8 if t.kind == KIND_FP8 else bf16_window) for t in self.tiles]

    def weights_field(self, bf16_window=None):
        """(B, 32, K) canonical F_p uint64 tiles -- the relation-5 W."""
        u = np.stack([w for w, _ in self.weights_int(bf16_window)]).astype(np.int64)
        neg = np.uint64(P) - np.abs(u).astype(np.uint64)
        return np.where(u < 0, neg, u.astype(np.uint64)).astype(np.uint64)

    def encode(self):
        v2 = any(t.kind != KIND_BF16 for t in self.tiles)
        out = [_OPENING_MAGIC_V2 if v2 else _OPENING_MAGIC,
               struct.pack("<IIQ", self.K, len(self.tiles), len(self.manifest)), self.manifest]
        for t in self.tiles:
            nb = t.name.encode()
            out.append(struct.pack("<H", len(nb)) + nb + struct.pack("<QQqI", t.r0, t.c0, t.shift, len(t.path)))
            if v2:
                out.append(struct.pack("<B", t.kind))
            out.append(b"".join(t.path))
            if t.kind == KIND_FP8:
                out.append(np.ascontiguousarray(t.raw, dtype=np.uint8).tobytes())
                out.append(np.ascontiguousarray(t.scales, dtype="<f4").tobytes())
            else:
                out.append(np.ascontiguousarray(t.units, dtype="<i8").tobytes())
        return b"".join(out)

    @classmethod
    def decode(cls, data):
        v2 = data.startswith(_OPENING_MAGIC_V2)
        if not v2 and not data.startswith(_OPENING_MAGIC):
            raise WeightCommitError("E_OPENING", "bad magic")
        try:
            off = len(_OPENING_MAGIC)
            K, n, ml = struct.unpack_from("<IIQ", data, off)
            off += 16
            manifest = bytes(data[off:off + ml])
            off += ml
            tiles = []
            for _ in range(n):
                (ln,) = struct.unpack_from("<H", data, off)
                name = bytes(data[off + 2:off + 2 + ln]).decode()
                off += 2 + ln
                r0, c0, shift, depth = struct.unpack_from("<QQqI", data, off)
                off += 28
                kind = KIND_BF16
                if v2:
                    (kind,) = struct.unpack_from("<B", data, off)
                    off += 1
                path = tuple(bytes(data[off + 32 * i:off + 32 * i + 32]) for i in range(depth))
                off += 32 * depth
                if kind == KIND_FP8:
                    if K % FP8_BLOCK:
                        raise WeightCommitError("E_OPENING", "FP8 tile width is not block aligned")
                    raw = np.frombuffer(data, dtype=np.uint8, count=ROWS * K, offset=off).reshape(ROWS, K).copy()
                    off += ROWS * K
                    sc = np.frombuffer(data, dtype="<f4", count=K // FP8_BLOCK, offset=off).copy()
                    off += 4 * (K // FP8_BLOCK)
                    tiles.append(OpenedTile(name, r0, c0, shift, None, path, KIND_FP8, raw, sc))
                elif kind == KIND_BF16:
                    units = np.frombuffer(data, dtype="<i8", count=ROWS * K, offset=off).reshape(ROWS, K).astype(
                        np.int64)
                    off += 8 * ROWS * K
                    tiles.append(OpenedTile(name, r0, c0, shift, units, path))
                else:
                    raise WeightCommitError("E_OPENING", "unknown tile kind")
        except (struct.error, ValueError) as exc:
            if isinstance(exc, WeightCommitError):
                raise
            raise WeightCommitError("E_OPENING", f"truncated / malformed opening: {exc}") from None
        if off != len(data):
            raise WeightCommitError("E_OPENING", "trailing bytes")
        return cls(K, manifest, tuple(tiles))


def open_drawn_tiles(commitment, source, coords):
    """Opening of the drawn tiles `coords` = [(tensor name, r0, c0)] (relation-5 tile order)."""
    ck = as_checkpoint(source)
    K = commitment.K
    tiles = []
    for name, r0, c0 in coords:
        e = commitment.entry(name)
        idx = e.leaf_index(r0, c0)
        path = tuple(merkle_path(commitment.levels, idx))
        if e.kind == KIND_FP8:
            raw, scales = _tensor_tiles8(ck, name, K, r0 // ROWS, r0 // ROWS + 1)
            tiles.append(OpenedTile(name, r0, c0, 0, None, path, KIND_FP8, raw[c0 // K], scales[c0 // K]))
        else:
            words = _tensor_tiles(ck, name, K, r0 // ROWS, r0 // ROWS + 1)[c0 // K]
            units, shifts = tile_units(words[None])
            tiles.append(OpenedTile(name, r0, c0, int(shifts[0]), units[0], path))
    return TileOpening(K, commitment.manifest, tuple(tiles))


# ----------------------------------------------------------------------------- offline verification
def _setup_root(setup_cert):
    from .setup_cert import SetupCertificate
    if isinstance(setup_cert, (bytes, bytearray)):
        setup_cert = SetupCertificate.decode(bytes(setup_cert))
    return setup_cert.weight_commitment_root


def _rel5_profile(container):
    from ..backend.fpn2 import FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1, header, profile_id
    for prof in (FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1):
        hdr = header(profile_id(prof))
        if bytes(container[:len(hdr)]) == hdr and len(container) == prof["c_p"]:
            return prof
    raise WeightCommitError("E_PROFILE", "not a registered relation-5 container")


def check_tile_inclusion(root, opening, expected_coords):
    """Checks 1 (inclusion under `root` + exact draw coordinates); raises WeightCommitError."""
    K, count, entries = decode_manifest(opening.manifest)
    if K != opening.K:
        raise WeightCommitError("E_MANIFEST", "opening K differs from the manifest")
    by_name = {e.name: e for e in entries}
    depth = tree_depth(count)
    if [tuple(c) for c in expected_coords] != opening.coords():
        raise WeightCommitError("E_TILE_COORDS", "opened tiles are not the drawn tiles (order included)")
    tree_root = None
    for t in opening.tiles:
        e = by_name.get(t.name)
        if e is None:
            raise WeightCommitError("E_TILE_RANGE", f"tensor {t.name!r} is not committed")
        idx = e.leaf_index(t.r0, t.c0)
        if t.kind != e.kind:
            raise WeightCommitError("E_TILE_KIND", f"tile {t.name}[{t.r0}, {t.c0}] kind differs from the manifest")
        if len(t.path) != depth:
            raise WeightCommitError("E_TILE_PATH", "path length differs from the tree depth")
        cand = merkle_root_from_path(t.leaf(K), idx, t.path)
        if tree_root is None:
            tree_root = cand
        if cand != tree_root:
            raise WeightCommitError("E_TILE_PATH", f"tile {t.name}[{t.r0}, {t.c0}] is not in the committed tree")
    if tree_root is None or global_root(opening.manifest, tree_root) != root:
        raise WeightCommitError("E_TILE_PATH", "opened tiles do not hash to SetupCertificate.weight_commitment_root")
    return {"tiles": len(opening.tiles), "tree_depth": depth, "leaves": count, "tensors": len(entries),
            "fp8_tiles": sum(t.kind == KIND_FP8 for t in opening.tiles),
            "fp8_window_int_bits": manifest_window_int_bits(opening.manifest)}


def verify_drawn_tiles_against_setup(setup_cert, opening, container, expected_coords, engine=None,
                                     window_int_bits=None):
    """Offline check with ONLY the SetupCertificate (no model weights): returns a report whose `verdict` is
    "VERIFIED" or "REJECTED: <code>"; on success `weight_root` is the root to pin in `fpn-verify --circuit-root`.
    `window_int_bits` is the public window applied to BF16 tiles (None = exact units); FP8 tiles use the
    window bound in the manifest."""
    import time
    from ..backend import folded_vec as FV
    t0 = time.perf_counter()
    rep = {"verdict": "VERIFIED"}
    try:
        if isinstance(opening, (bytes, bytearray)):
            opening = TileOpening.decode(bytes(opening))
        prof = _rel5_profile(container)
        lay = FV.layout_of(prof)
        if len(opening.tiles) != lay.B or opening.K != lay.K:
            raise WeightCommitError("E_PROFILE", f"profile {prof['name']} proves {lay.B} tiles of K = {lay.K}")
        rep.update(check_tile_inclusion(_setup_root(setup_cert), opening, expected_coords))
        t1 = time.perf_counter()
        W = opening.weights_field(window_int_bits)
        FV._W_CACHE.clear()
        root = FV.weight_root(lay, W, engine)
        pinned = bytes(container[96:128])
        rep.update(profile=prof["name"], weight_root=root.hex(), container_weight_root=pinned.hex(),
                   bf16_window_int_bits=window_int_bits,
                   inclusion_ms=round((t1 - t0) * 1e3, 2), weight_root_ms=round((time.perf_counter() - t1) * 1e3, 2))
        if root != pinned:
            raise WeightCommitError("E_WEIGHT_ROOT", "relation-5 weight_root differs from the opened tiles' root")
    except WeightCommitError as exc:
        rep["verdict"] = f"REJECTED: {exc.code}"
        rep["detail"] = str(exc)
    return rep
