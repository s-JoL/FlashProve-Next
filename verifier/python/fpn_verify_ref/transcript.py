# SPDX-License-Identifier: Apache-2.0
"""Fpn2Transcript: raw-byte Fiat-Shamir over BLAKE2b-512 with 1-byte domain tags (specs/11-wire-format.md §4)."""
import hashlib

from .field import P, fp_encode, e_encode

PERSONAL_FS = b"FPN2-FS-v2" + b"\x00" * 6
LABEL_V2 = b"FPN2/transcript/v2"

TAG_INIT = 0x00
TAG_BYTES = 0x01
TAG_FP = 0x02
TAG_FP3 = 0x03
TAG_SQ_BYTES = 0x10
TAG_SQ_FP3 = 0x11
TAG_SQ_U192 = 0x12
TAG_RATCHET = 0x7F


def h_fs(data):
    return hashlib.blake2b(data, digest_size=64, person=PERSONAL_FS).digest()


def _le8(n):
    return int(n).to_bytes(8, "little")


class Fpn2Transcript:
    def __init__(self, label=LABEL_V2):
        label = bytes(label)
        self.state = h_fs(bytes([TAG_INIT]) + _le8(len(label)) + label)

    def _absorb(self, tag, data):
        self.state = h_fs(self.state + bytes([tag]) + _le8(len(data)) + data)

    def absorb_bytes(self, b):
        self._absorb(TAG_BYTES, bytes(b))

    def absorb_fp(self, x):
        self._absorb(TAG_FP, fp_encode(x))

    def absorb_fp3(self, a):
        self._absorb(TAG_FP3, e_encode(a))

    def _squeeze(self, tag, n):
        out = b""
        i = 0
        while len(out) < n:
            out += h_fs(self.state + bytes([tag]) + _le8(n) + _le8(i))
            i += 1
        self.state = h_fs(self.state + bytes([TAG_RATCHET, tag]) + _le8(n))
        return out[:n]

    def squeeze_bytes(self, n):
        return self._squeeze(TAG_SQ_BYTES, n)

    def squeeze_fp3(self):
        o = self._squeeze(TAG_SQ_FP3, 48)
        return tuple(int.from_bytes(o[16 * k:16 * k + 16], "little") % P for k in range(3))

    def squeeze_u192(self):
        return int.from_bytes(self._squeeze(TAG_SQ_U192, 24), "little")
