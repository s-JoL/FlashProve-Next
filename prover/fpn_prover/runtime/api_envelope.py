# SPDX-License-Identifier: Apache-2.0
"""Attach a fixed-width verification extension to a provider response."""

import base64
import math

from ..backend.fpn2 import FOLDED_AIR_V1, FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1, header, profile_id
from ..circuit.protocol import MODE_EXTERNAL_SEED, MODE_NI
from .setup_cert import SetupCertificate

_HEADER = header(profile_id(FOLDED_AIR_V1))
# Fixed-width receipts the envelope may carry: (64-byte header, C_p).  Relation-5 GEMM-tile receipts  use
# their own registered profiles; every receipt of one profile has the same length, whatever the request.
_RECEIPTS = {(_HEADER, FOLDED_AIR_V1["c_p"])} | {
    (header(profile_id(p)), p["c_p"]) for p in (FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1)}


def attach_verification(response, *, receipt, setup_cert, challenge_mode, c_pre):
    """Extend a caller-built public Anthropic or OpenAI response."""
    if response.get("type") != "message" and response.get("object") != "chat.completion":
        raise ValueError("expected a non-streaming messages or chat completion response")
    if type(receipt) is not bytes or (bytes(receipt[:64]), len(receipt)) not in _RECEIPTS:
        raise ValueError("expected a fixed-width FPN2 receipt")
    if type(c_pre) is not bytes or len(c_pre) != 32 or challenge_mode not in (MODE_NI, MODE_EXTERNAL_SEED):
        raise ValueError("invalid precommitment or challenge mode")
    return {
        **response,
        "fpn_receipt_b64": base64.b64encode(receipt).decode("ascii"),
        "setup_cert_id": SetupCertificate.decode(setup_cert).id,
        "challenge_mode": challenge_mode,
        "c_pre_hex": c_pre.hex(),
    }


def release_deadline_s(arrival_s, window_s):
    """Compute the monotonic deadline from arrival and a public fixed window."""
    deadline = arrival_s + window_s
    if not math.isfinite(deadline) or window_s < 0:
        raise ValueError("invalid release deadline")
    return deadline
