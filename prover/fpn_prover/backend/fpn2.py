# SPDX-License-Identifier: Apache-2.0
"""Prover-side `FPN2` primitives for relation 3 (`FOLDED_GKR_LOOKUP`, specs/11-wire-format.md §5.4).

Independent of `verifier/python/fpn_verify_ref` (which must never import the prover):
  * `Fpn2Transcript` -- raw-byte Fiat-Shamir over BLAKE2b-512 with 1-byte domain tags (§4);
  * canonical `Fp3` wire codec (24 bytes, every limb < P);
  * BLAKE2b-256 coset Merkle trees (personal ``FPN2-MERKLE-v1``) used by the fused folded PCS;
  * the canonical profile bytes / layout of `fpn2-folded-gkr-v1`.
"""
import hashlib
import struct
from dataclasses import dataclass

from flashprove_next.field import P

# ----------------------------------------------------------------------------- Fp3 codec
def enc3(a):
    if type(a) is not tuple or len(a) != 3:
        raise ValueError("expected Fp3 element")
    out = bytearray()
    for x in a:
        if type(x) is not int or not 0 <= x < P:
            raise ValueError("noncanonical field limb")
        out += x.to_bytes(8, "little")
    return bytes(out)


def enc3s(values):
    return b"".join(enc3(v) for v in values)


def dec3(b):
    if len(b) != 24:
        raise ValueError("Fp3 encoding must be 24 bytes")
    limbs = tuple(int.from_bytes(b[8 * i:8 * i + 8], "little") for i in range(3))
    if max(limbs) >= P:
        raise ValueError("noncanonical field limb")
    return limbs


# ----------------------------------------------------------------------------- transcript (§4)
PERSONAL_FS = b"FPN2-FS-v2" + b"\x00" * 6
LABEL_V2 = b"FPN2/transcript/v2"


def _h_fs(data):
    return hashlib.blake2b(data, digest_size=64, person=PERSONAL_FS).digest()


class Fpn2Transcript:
    """Byte-exact twin of `core/src/transcript.rs` (prover side)."""

    def __init__(self, label=LABEL_V2):
        self.state = _h_fs(b"\x00" + len(label).to_bytes(8, "little") + label)

    def absorb_bytes(self, data):
        data = bytes(data)
        self.state = _h_fs(self.state + b"\x01" + len(data).to_bytes(8, "little") + data)

    def _squeeze(self, tag, n):
        nb = n.to_bytes(8, "little")
        out = b""
        i = 0
        while len(out) < n:
            out += _h_fs(self.state + bytes([tag]) + nb + i.to_bytes(8, "little"))
            i += 1
        self.state = _h_fs(self.state + bytes([0x7F, tag]) + nb)
        return out[:n]

    def squeeze_bytes(self, n):
        return self._squeeze(0x10, n)

    def squeeze_fp3(self):
        o = self._squeeze(0x11, 48)
        return tuple(int.from_bytes(o[16 * k:16 * k + 16], "little") % P for k in range(3))


# ----------------------------------------------------------------------------- BLAKE2b coset Merkle trees
PERSONAL_MERKLE = b"FPN2-MERKLE-v1\x00\x00"


def leaf_hash(values):
    """Leaf of one coset: B256(0x00 || enc(v_0) || ... || enc(v_3))."""
    return hashlib.blake2b(b"\x00" + enc3s(values), digest_size=32, person=PERSONAL_MERKLE).digest()


def node_hash(left, right):
    return hashlib.blake2b(b"\x01" + left + right, digest_size=32, person=PERSONAL_MERKLE).digest()


def merkle_levels(leaves):
    levels = [list(leaves)]
    if len(leaves) & (len(leaves) - 1) or not leaves:
        raise ValueError("Merkle tree needs a power-of-two leaf count")
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([node_hash(cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)])
    return levels


def merkle_path(levels, idx):
    return [levels[k][(idx >> k) ^ 1] for k in range(len(levels) - 1)]


def merkle_verify(root, leaf, idx, path):
    node = leaf
    for sib in path:
        node = node_hash(sib, node) if idx & 1 else node_hash(node, sib)
        idx >>= 1
    return node == root


# ----------------------------------------------------------------------------- profile (§5.2) and layout (§5.4)
REL_FOLDED_GKR_LOOKUP = 3
RATE = 8          # codeword length = RATE * 2^n
ARITY = 4         # fused two-round (coset-4) folding


def canonical_profile_bytes(name, tier, relation, bits, n_vars, sc_degree, n_queries, c_p, max_hidden=0):
    nm = name.encode("ascii")
    return (b"FPN2PROF" + struct.pack("<H", 1) + nm.ljust(32, b"\x00")
            + struct.pack("<BBH", tier, relation, bits) + struct.pack("<5I", 0, max_hidden, 0, 0, 0)
            + struct.pack("<B8I", 0, *([0] * 8))
            + struct.pack("<BBHQ", n_vars, sc_degree, n_queries, c_p) + b"\x00" * 17)


DATA_FIXED = ((1, 1), (3, 1))     # data subcube of F: index bits 1 and 3 are both 1
LIBRA_FIXED = ((1, 0), (3, 0))    # Libra coefficients live in the subcube bits 1 = 3 = 0
MASK_SUBCUBES = (((1, 0),), ((1, 1), (3, 0)))   # complement of the data subcube


def free_bits(n):
    return [0, 2] + list(range(4, n))


def deposit(value, fixed, free):
    idx = 0
    for bit, val in fixed:
        idx |= val << bit
    for i, bit in enumerate(free):
        idx |= ((value >> i) & 1) << bit
    return idx


@dataclass(frozen=True)
class Layout:
    """Fixed-size relation-3 layout as a pure function of (n, Q).

    F has 2^n cells. Data blocks are aligned blocks of the data subcube (bits 1 = 3 = 1, 2^(n-2)
    cells, addressed by "data coordinates" over the free bits 0, 2, 4, 5, ..., n-1); every other
    cell is reserved blinding randomness except the Libra coefficients.  Spreading the blinding
    over all four patterns of every index-bit pair is what lets it reach every monomial class of
    every folded layer (the rank condition of §5.4.6).
    """
    n: int            # committed multilinear vector F has 2^n Fp3 cells (n even)
    queries: int

    def __post_init__(self):
        if self.n < 8 or self.n % 2 or self.n > 28 or not 1 <= self.queries <= 4096:
            raise ValueError("relation 3 needs even n in [8, 28] (rate-1/8 root requires 2^(n+3) | P-1) and 1 <= Q <= 4096")

    # --- lookup geometry: s real-leaf variables, 2^(s-2) private queries in table [0, 2^(s-2))
    @property
    def s(self): return self.n - 3

    @property
    def quarter(self): return 1 << (self.s - 2)

    # --- data blocks, in data coordinates (aligned)
    @property
    def off_q(self): return 0

    @property
    def off_m(self): return self.quarter

    @property
    def off_e1(self): return 2 * self.quarter

    @property
    def off_e3(self): return 3 * self.quarter

    @property
    def off_a(self): return 1 << self.s

    @property
    def off_d(self): return (1 << self.s) + (1 << (self.s - 1))

    def cell(self, data_index):
        """Data coordinate -> index in F."""
        return deposit(data_index, DATA_FIXED, free_bits(self.n))

    def data_block(self, off, k):
        """(fixed, free) subcube of F holding data-coordinate block [off, off + 2^k)."""
        fb = free_bits(self.n)
        fixed = DATA_FIXED + tuple((fb[k + i], (off >> (k + i)) & 1) for i in range(len(fb) - k))
        return fixed, tuple(fb[:k])

    @property
    def libra_len(self): return 2 * self.s * (self.s + 1)

    def libra_slot(self, k, i, e):
        """Cell of coefficient e (0..3) of univariate i (0..k-1) of the Libra mask of layer k."""
        return deposit(2 * k * (k - 1) + 4 * i + e, LIBRA_FIXED, free_bits(self.n))

    def is_data(self, idx):
        return (idx >> 1) & 1 and (idx >> 3) & 1

    @property
    def blind_len(self): return 3 * (1 << (self.n - 2)) - self.libra_len

    # --- PCS geometry
    @property
    def folds(self): return self.n // 2

    def coset_count(self, t):
        """Number of coset leaves (= L_t / 4) in folded layer t."""
        return 1 << (self.n + 1 - 2 * t)

    def depth(self, t):
        return self.n + 1 - 2 * t

    # --- wire sizes
    STMT = 64
    GKR_ROOT = 48
    GKR_L0 = 96

    def gkr_layer_len(self, k):
        return 24 + 96 * k + 120

    @property
    def gkr_len(self):
        return self.GKR_ROOT + self.GKR_L0 + sum(self.gkr_layer_len(k) for k in range(1, self.s + 1))

    @property
    def pcs_len(self):
        return 24 + self.n * 72 + 32 * (self.folds - 1) + 24

    @property
    def query_len(self):
        return sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return self.STMT + self.gkr_len + self.pcs_len + self.queries * self.query_len

    @property
    def observations(self):
        """Upper bound on PCS observations (rows of A|B in the ZK rank condition)."""
        return 4 * self.queries * self.folds + 3 * self.n + 2


# The registered end-to-end profile (n = 12: 256 private u8 queries, Q = 72 queries at rate 1/8).
FOLDED_GKR_V1 = dict(name="fpn2-folded-gkr-v1", tier=1, relation=REL_FOLDED_GKR_LOOKUP, bits=100,
                     n_vars=12, sc_degree=3, n_queries=72, c_p=163840)


# Small REDUCED-tier test profile carrying most negative golden vectors (n = 8, Q = 2).
FOLDED_GKR_N8 = dict(name="fpn2-folded-gkr-n8", tier=0, relation=REL_FOLDED_GKR_LOOKUP, bits=0,
                     n_vars=8, sc_degree=3, n_queries=2, c_p=6144)


def profile_bytes(p=FOLDED_GKR_V1):
    return canonical_profile_bytes(p["name"], p["tier"], p["relation"], p["bits"], p["n_vars"],
                                   p["sc_degree"], p["n_queries"], p["c_p"], p.get("max_hidden", 0))


def profile_id(p=FOLDED_GKR_V1):
    return hashlib.blake2b(profile_bytes(p), digest_size=32).digest()


def header(pid):
    return b"FPN2" + struct.pack("<HH", 2, 0) + pid + b"\x00" * 24


# ----------------------------------------------------------------------------- relation 4 (§5.5)
REL_FOLDED_GKR_AIR = 4

# Registered relation-4 profiles (layout: folded_air.AirLayout).  n = 14 gives K = 512 gate rows /
# variables; Q = 72 queries at rate 1/8 keep the container within 256 KiB.
FOLDED_AIR_V1 = dict(name="fpn2-folded-air-v1", tier=1, relation=REL_FOLDED_GKR_AIR, bits=100,
                     n_vars=14, sc_degree=3, n_queries=72, c_p=262144)

# Small REDUCED-tier relation-4 test profile (n = 12: K = 128 rows / variables, Q = 2).
FOLDED_AIR_N12 = dict(name="fpn2-folded-air-n12", tier=0, relation=REL_FOLDED_GKR_AIR, bits=0,
                      n_vars=12, sc_degree=3, n_queries=2, c_p=16384)

# Production-scale relation-4 profile : n = 20 gives K = 32,768 gate rows / variables (64x the n = 14
# profile); Q = 72 at rate 1/8, body 424,104 B inside C_p = 512 KiB.
FOLDED_AIR_N20_V1 = dict(name="fpn2-folded-air-n20-v1", tier=1, relation=REL_FOLDED_GKR_AIR, bits=100,
                         n_vars=20, sc_degree=3, n_queries=72, c_p=524288)


# ----------------------------------------------------------------------------- relation 5 (§5.6)
REL_FOLDED_VEC_MATMUL = 5

# Registered relation-5 profiles (layout: folded_vec.VecLayout).  max_hidden = K = 1024 is the tile width (the
# only geometry field besides n); R = 32 rows per tile is fixed; B = 2^(n - 16) tiles fill the data half of C.
# n = 16: ONE full-width 32 x 1024 tile (32,768 multiply-adds), body 288,368 B inside C_p = 384 KiB.
FOLDED_VEC_N16_V1 = dict(name="fpn2-folded-vec-n16-v1", tier=1, relation=REL_FOLDED_VEC_MATMUL, bits=100,
                         n_vars=16, sc_degree=3, n_queries=72, c_p=393216, max_hidden=1024)
# n = 22: 64 full-width 32 x 1024 tiles (2,097,152 multiply-adds), body 469,184 B inside C_p = 512 KiB.
FOLDED_VEC_N22_V1 = dict(name="fpn2-folded-vec-n22-v1", tier=1, relation=REL_FOLDED_VEC_MATMUL, bits=100,
                         n_vars=22, sc_degree=3, n_queries=72, c_p=524288, max_hidden=1024)


# ----------------------------------------------------------------------------- relation 6 (§5.7)
REL_FOLDED_VEC_BOUND = 6

# Registered relation-6 profiles (layout: folded_vec_bound.BoundLayout): relation 5's geometry with the activations
# pre-committed under a pinned x_root.  n = 16: body 334,480 B inside C_p = 384 KiB; n = 22: body 529,120 B does
# NOT fit 512 KiB (relation 5's size), so C_p = 1 MiB.
FOLDED_VEC_BOUND_N16_V1 = dict(name="fpn2-folded-vec-bound-n16-v1", tier=1, relation=REL_FOLDED_VEC_BOUND, bits=100,
                               n_vars=16, sc_degree=3, n_queries=72, c_p=393216, max_hidden=1024)
FOLDED_VEC_BOUND_N22_V1 = dict(name="fpn2-folded-vec-bound-n22-v1", tier=1, relation=REL_FOLDED_VEC_BOUND, bits=100,
                               n_vars=22, sc_degree=3, n_queries=72, c_p=1048576, max_hidden=1024)
