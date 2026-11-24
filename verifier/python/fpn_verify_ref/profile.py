# SPDX-License-Identifier: Apache-2.0
"""Canonical FPN2 profile encoding and the pinned registry (specs/11-wire-format.md §5.2)."""
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Tuple

PROFILE_MAGIC = b"FPN2PROF"
PROFILE_FORMAT = 1
PROFILE_LEN = 128
TIER_REDUCED = 0
TIER_STANDARD = 1
REL_HARNESS_SQ_SUMCHECK = 1
REL_LLM_INFERENCE_V2 = 2
REL_FOLDED_GKR_LOOKUP = 3
REL_FOLDED_GKR_AIR = 4
REL_FOLDED_VEC_MATMUL = 5
REL_FOLDED_VEC_BOUND = 6
HEADER_LEN = 64
_NAME_CHARS = set(b"abcdefghijklmnopqrstuvwxyz0123456789.-")


@dataclass(frozen=True)
class Profile:
    name: str
    security_tier: int
    relation_id: int
    security_bits: int
    max_layers: int
    max_hidden: int
    max_seq_len: int
    max_vocab: int
    max_new_tokens: int
    stop_ids: Tuple[int, ...]
    n_vars: int
    sc_degree: int
    n_queries: int
    c_p: int

    def canonical_bytes(self):
        nm = self.name.encode("ascii")
        if not (1 <= len(nm) <= 32) or any(c not in _NAME_CHARS for c in nm):
            raise ValueError("invalid profile name")
        if len(self.stop_ids) > 8:
            raise ValueError("at most 8 stop ids")
        stops = list(self.stop_ids) + [0] * (8 - len(self.stop_ids))
        b = (PROFILE_MAGIC + struct.pack("<H", PROFILE_FORMAT) + nm.ljust(32, b"\x00")
             + struct.pack("<BBH", self.security_tier, self.relation_id, self.security_bits)
             + struct.pack("<5I", self.max_layers, self.max_hidden, self.max_seq_len, self.max_vocab,
                           self.max_new_tokens)
             + struct.pack("<B8I", len(self.stop_ids), *stops)
             + struct.pack("<BBHQ", self.n_vars, self.sc_degree, self.n_queries, self.c_p)
             + b"\x00" * 17)
        assert len(b) == PROFILE_LEN
        return b

    @property
    def profile_id(self):
        return hashlib.blake2b(self.canonical_bytes(), digest_size=32).digest()

    # -- layout of relation 1
    @property
    def stmt_len(self):
        return 120

    @property
    def round_len(self):
        return (self.sc_degree + 1) * 24

    @property
    def opening_len(self):
        return 8 + 32 * self.n_vars

    @property
    def body_used(self):
        if self.relation_id == REL_HARNESS_SQ_SUMCHECK:
            return self.stmt_len + self.n_vars * self.round_len + 24 + self.n_queries * self.opening_len
        if self.relation_id == REL_FOLDED_GKR_LOOKUP:
            from .folded_gkr import layout_of
            lay = layout_of(self)
            return None if lay is None else lay.body_used
        if self.relation_id == REL_FOLDED_GKR_AIR:
            from .folded_air import layout_of as air_layout_of
            lay = air_layout_of(self)
            return None if lay is None else lay.body_used
        if self.relation_id == REL_FOLDED_VEC_MATMUL:
            from .folded_vec import layout_of as vec_layout_of
            lay = vec_layout_of(self)
            return None if lay is None else lay.body_used
        if self.relation_id == REL_FOLDED_VEC_BOUND:
            from .folded_vec_bound import layout_of as bound_layout_of
            lay = bound_layout_of(self)
            return None if lay is None else lay.body_used
        return None


def decode_profile(b):
    """Strict decoder of canonical profile bytes (rejects any non-canonical byte)."""
    if len(b) != PROFILE_LEN or b[0:8] != PROFILE_MAGIC:
        raise ValueError("bad profile magic/length")
    if struct.unpack_from("<H", b, 8)[0] != PROFILE_FORMAT:
        raise ValueError("bad profile format")
    raw = b[10:42]
    nm = raw.rstrip(b"\x00")
    if not nm or b"\x00" in nm or any(c not in _NAME_CHARS for c in nm):
        raise ValueError("bad profile name")
    tier, rel, bits = struct.unpack_from("<BBH", b, 42)
    scale = struct.unpack_from("<5I", b, 46)
    n_stop = b[66]
    stops = struct.unpack_from("<8I", b, 67)
    if n_stop > 8 or any(stops[n_stop:]):
        raise ValueError("bad stop ids")
    n_vars, deg, nq, c_p = struct.unpack_from("<BBHQ", b, 99)
    if any(b[111:128]) or tier not in (TIER_REDUCED, TIER_STANDARD):
        raise ValueError("bad profile reserved/tier")
    p = Profile(nm.decode("ascii"), tier, rel, bits, *scale, tuple(stops[:n_stop]), n_vars, deg, nq, c_p)
    assert p.canonical_bytes() == bytes(b)
    return p


HARNESS_SQ_N4 = Profile(
    name="fpn2-harness-sq-n4", security_tier=TIER_REDUCED, relation_id=REL_HARNESS_SQ_SUMCHECK, security_bits=0,
    max_layers=0, max_hidden=0, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=4, sc_degree=2, n_queries=4, c_p=2048)

LLM_STD_V1 = Profile(
    name="fpn2-llm-std-v1", security_tier=TIER_STANDARD, relation_id=REL_LLM_INFERENCE_V2, security_bits=100,
    max_layers=128, max_hidden=16384, max_seq_len=131072, max_vocab=262144, max_new_tokens=32768,
    stop_ids=(151643, 151645), n_vars=0, sc_degree=0, n_queries=0, c_p=8 * 1024 * 1024)

FOLDED_GKR_V1 = Profile(
    name="fpn2-folded-gkr-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_GKR_LOOKUP, security_bits=100,
    max_layers=0, max_hidden=0, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=12, sc_degree=3, n_queries=72, c_p=163840)

FOLDED_GKR_N8 = Profile(
    name="fpn2-folded-gkr-n8", security_tier=TIER_REDUCED, relation_id=REL_FOLDED_GKR_LOOKUP, security_bits=0,
    max_layers=0, max_hidden=0, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=8, sc_degree=3, n_queries=2, c_p=6144)

FOLDED_AIR_V1 = Profile(
    name="fpn2-folded-air-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_GKR_AIR, security_bits=100,
    max_layers=0, max_hidden=0, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=14, sc_degree=3, n_queries=72, c_p=262144)

FOLDED_AIR_N12 = Profile(
    name="fpn2-folded-air-n12", security_tier=TIER_REDUCED, relation_id=REL_FOLDED_GKR_AIR, security_bits=0,
    max_layers=0, max_hidden=0, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=12, sc_degree=3, n_queries=2, c_p=16384)

FOLDED_AIR_N20_V1 = Profile(
    name="fpn2-folded-air-n20-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_GKR_AIR, security_bits=100,
    max_layers=0, max_hidden=0, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=20, sc_degree=3, n_queries=72, c_p=524288)

FOLDED_VEC_N16_V1 = Profile(
    name="fpn2-folded-vec-n16-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_VEC_MATMUL, security_bits=100,
    max_layers=0, max_hidden=1024, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=16, sc_degree=3, n_queries=72, c_p=393216)

FOLDED_VEC_N22_V1 = Profile(
    name="fpn2-folded-vec-n22-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_VEC_MATMUL, security_bits=100,
    max_layers=0, max_hidden=1024, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=22, sc_degree=3, n_queries=72, c_p=524288)

FOLDED_VEC_BOUND_N16_V1 = Profile(
    name="fpn2-folded-vec-bound-n16-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_VEC_BOUND,
    security_bits=100, max_layers=0, max_hidden=1024, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=16, sc_degree=3, n_queries=72, c_p=393216)

FOLDED_VEC_BOUND_N22_V1 = Profile(
    name="fpn2-folded-vec-bound-n22-v1", security_tier=TIER_STANDARD, relation_id=REL_FOLDED_VEC_BOUND,
    security_bits=100, max_layers=0, max_hidden=1024, max_seq_len=0, max_vocab=0, max_new_tokens=0, stop_ids=(),
    n_vars=22, sc_degree=3, n_queries=72, c_p=1048576)

# Pinned ids: BLAKE2b-256(canonical_bytes).  test_vectors.py asserts recomputation == pin.
PINNED = {
    "fpn2-harness-sq-n4": (HARNESS_SQ_N4, "e24f718cf8c2f4d21417f9e8a3038b9993aa418ca0f7205a5cf6a8c53a3442db"),
    "fpn2-llm-std-v1": (LLM_STD_V1, "588659a6e46c51c021cd65dd287c8968562b637d20afaf0d68f4ad57b16db9a1"),
    "fpn2-folded-gkr-v1": (FOLDED_GKR_V1, "e238db306348de19561c2fb4896ea4a249bd8ce7bd41beb1f715b6fcb305e986"),
    "fpn2-folded-gkr-n8": (FOLDED_GKR_N8, "d7a3d169b65f8c5e328911f4a7edb3750c3244ca0202a9b259a7325d5705c147"),
    "fpn2-folded-air-v1": (FOLDED_AIR_V1, "f6c6b4ca3f4292e6e8917d1285f6105bde168ed7517b71aa65e5c562933d0a97"),
    "fpn2-folded-air-n12": (FOLDED_AIR_N12, "5f661ec4ea1ab8b67de2ed9d13a13d148b4ed639a242d30c81fdb4f2705ced7d"),
    "fpn2-folded-air-n20-v1": (FOLDED_AIR_N20_V1, "966dec06ad5fb40ef7843fe91310684a12378990bbe53483ce7718ef606ba60d"),
    "fpn2-folded-vec-n16-v1": (FOLDED_VEC_N16_V1, "a48dfb0a1b0f17580cbab6b756678238200c1b3028a26f92898430d8da801327"),
    "fpn2-folded-vec-n22-v1": (FOLDED_VEC_N22_V1, "ba72066888c8fc3cf0ca31c86527a5292759eb629c0eed131d812623b3e6b3ef"),
    "fpn2-folded-vec-bound-n16-v1": (FOLDED_VEC_BOUND_N16_V1,
                                     "9492aa70db8c3d79fb242001f178a43b31ba5e610114d2606c87d6350eb0472e"),
    "fpn2-folded-vec-bound-n22-v1": (FOLDED_VEC_BOUND_N22_V1,
                                     "83bab0b60ea45833b70498f5c19dfc2768a5583c0e947cc8b418a66d10591745"),
}

REGISTRY = {bytes.fromhex(pin): prof for prof, pin in PINNED.values()}


def by_name(name):
    return PINNED[name][0]


def lookup(profile_id):
    return REGISTRY.get(bytes(profile_id))
