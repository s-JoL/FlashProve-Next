# SPDX-License-Identifier: Apache-2.0
"""Fixed-width setup record. Authenticate its ID through the setup publisher."""

import hashlib
import struct
from dataclasses import dataclass

_MAGIC = b"FPN2_SETUP\x01"
_FIELDS = struct.Struct(">32s32s32s32sQ")


@dataclass(frozen=True)
class SetupCertificate:
    weight_commitment_root: bytes
    circuit_root: bytes
    rounding_contract_digest: bytes
    model_format_digest: bytes
    update_epoch: int

    def encode(self):
        digests = (self.weight_commitment_root, self.circuit_root,
                   self.rounding_contract_digest, self.model_format_digest)
        if any(type(digest) is not bytes or len(digest) != 32 for digest in digests):
            raise ValueError("setup roots and digests must be 32 bytes each")
        if type(self.update_epoch) is not int or not 0 <= self.update_epoch < 2**64:
            raise ValueError("setup epoch must fit an unsigned 64-bit integer")
        return _MAGIC + _FIELDS.pack(*digests, self.update_epoch)

    @classmethod
    def decode(cls, data):
        if len(data) != len(_MAGIC) + _FIELDS.size or not data.startswith(_MAGIC):
            raise ValueError("invalid FPN2 setup record")
        return cls(*_FIELDS.unpack_from(data, len(_MAGIC)))

    @property
    def id(self):
        return hashlib.blake2b(self.encode(), digest_size=32).hexdigest()
