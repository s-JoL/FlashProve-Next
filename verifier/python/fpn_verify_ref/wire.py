# SPDX-License-Identifier: Apache-2.0
"""FPN2 header and fixed-size container codec (specs/11-wire-format.md §5)."""
import struct
from dataclasses import dataclass
from typing import List, Tuple

from .field import NonCanonical, fp_decode, e_decode, fp_encode, e_encode
from .profile import HEADER_LEN, REL_HARNESS_SQ_SUMCHECK

MAGIC = b"FPN2"
VERSION = 2


class Reject(Exception):
    def __init__(self, code, detail=""):
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code


@dataclass
class Header:
    profile_id: bytes
    version: int = VERSION
    flags: int = 0

    def encode(self):
        return MAGIC + struct.pack("<HH", self.version, self.flags) + bytes(self.profile_id) + b"\x00" * 24


def parse_header(b):
    """Checks 1-5 of the normative order."""
    if len(b) < HEADER_LEN:
        raise Reject("E_LENGTH", "shorter than header")
    if b[0:4] != MAGIC:
        raise Reject("E_MAGIC")
    version, flags = struct.unpack_from("<HH", b, 4)
    if version != VERSION:
        raise Reject("E_VERSION")
    if flags != 0:
        raise Reject("E_FLAGS")
    if any(b[40:64]):
        raise Reject("E_RESERVED")
    return Header(bytes(b[8:40]), version, flags)


def _digest_dec(b):
    return tuple(fp_decode(b[8 * i:8 * i + 8]) for i in range(4))


def _digest_enc(d):
    return b"".join(fp_encode(x) for x in d)


@dataclass
class HarnessBody:
    r_tab: Tuple[int, ...]
    n_s: bytes
    st_new: Tuple[int, ...]
    claim: Tuple[int, int, int]
    rounds: List[List[Tuple[int, int, int]]]
    final: Tuple[int, int, int]
    openings: List[Tuple[int, List[Tuple[int, ...]]]]


@dataclass
class Sections:
    """Raw byte views of a relation-1 container, in wire order."""
    header: bytes
    stmt: bytes
    rounds: List[bytes]
    final: bytes
    openings: List[bytes]
    pad: bytes


def split_sections(b, prof):
    off = HEADER_LEN
    stmt = b[off:off + prof.stmt_len]; off += prof.stmt_len
    rounds = []
    for _ in range(prof.n_vars):
        rounds.append(b[off:off + prof.round_len]); off += prof.round_len
    final = b[off:off + 24]; off += 24
    openings = []
    for _ in range(prof.n_queries):
        openings.append(b[off:off + prof.opening_len]); off += prof.opening_len
    return Sections(bytes(b[:HEADER_LEN]), stmt, rounds, final, openings, b[off:])


def decode_harness(sec, prof):
    """Canonical decoding of every typed field in wire order (raises Reject E_NONCANONICAL)."""
    try:
        s = sec.stmt
        r_tab = _digest_dec(s[0:32]); n_s = bytes(s[32:64]); st_new = _digest_dec(s[64:96]); claim = e_decode(s[96:120])
        rounds = [[e_decode(rb[24 * j:24 * j + 24]) for j in range(prof.sc_degree + 1)] for rb in sec.rounds]
        final = e_decode(sec.final)
        openings = []
        for ob in sec.openings:
            v = fp_decode(ob[0:8])
            sibs = [_digest_dec(ob[8 + 32 * k:8 + 32 * k + 32]) for k in range(prof.n_vars)]
            openings.append((v, sibs))
    except NonCanonical as e:
        raise Reject("E_NONCANONICAL", str(e))
    return HarnessBody(r_tab, n_s, st_new, claim, rounds, final, openings)


def encode_harness(prof, body):
    if prof.relation_id != REL_HARNESS_SQ_SUMCHECK:
        raise ValueError("not a harness profile")
    out = Header(prof.profile_id).encode()
    out += _digest_enc(body.r_tab) + bytes(body.n_s) + _digest_enc(body.st_new) + e_encode(body.claim)
    for rnd in body.rounds:
        out += b"".join(e_encode(g) for g in rnd)
    out += e_encode(body.final)
    for v, sibs in body.openings:
        out += fp_encode(v) + b"".join(_digest_enc(d) for d in sibs)
    assert len(out) == HEADER_LEN + prof.body_used
    return out + b"\x00" * (prof.c_p - len(out))
