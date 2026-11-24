# SPDX-License-Identifier: Apache-2.0
"""FPN2 verifier in the normative order of specs/11-wire-format.md §6."""
from .field import e_add, e_mul, lagrange_eval
from .poseidon2 import leaf_digest, verify_path
from .folded_air import layout_of as air_layout_of, verify_air
from .folded_gkr import layout_of, verify_folded
from .folded_vec import layout_of as vec_layout_of, verify_vec
from .folded_vec_bound import layout_of as bound_layout_of, verify_bound
from .profile import (HEADER_LEN, REL_FOLDED_GKR_AIR, REL_FOLDED_GKR_LOOKUP, REL_FOLDED_VEC_BOUND,
                      REL_FOLDED_VEC_MATMUL, REL_HARNESS_SQ_SUMCHECK, TIER_REDUCED, lookup)
from .transcript import LABEL_V2, Fpn2Transcript
from .wire import Reject, decode_harness, parse_header, split_sections

VERIFIED = "VERIFIED"


def verify_bytes(data, allow_reduced=False, expected_profile_id=None, expected_circuit_root=None,
                 expected_x_root=None):
    """Returns "VERIFIED" or raises Reject(code).

    Relations 4 and 5 fail closed: their statement is relative to a public commitment (relation 4: the circuit
    `root_circ`; relation 5: the weight tiles `weight_root`), so the caller MUST pin `expected_circuit_root`
    (32 bytes); absent -> E_CIRCUIT_UNPINNED, different -> E_CIRCUIT_PINNED.  Relation 6 additionally pins the
    pre-committed activation root `expected_x_root`: absent -> E_X_ROOT_UNPINNED, different -> E_X_ROOT_PINNED
    (order: weight unpinned, x unpinned, weight mismatch, x mismatch)."""
    data = bytes(data)
    hdr = parse_header(data)                                            # 1-5
    prof = lookup(hdr.profile_id)
    if prof is None:
        raise Reject("E_PROFILE_ID")                                    # 6
    if expected_profile_id is not None and bytes(expected_profile_id) != hdr.profile_id:
        raise Reject("E_PROFILE_PINNED")                                # 7
    if prof.security_tier == TIER_REDUCED and not allow_reduced:
        raise Reject("E_REDUCED_SECURITY")                              # 8
    if len(data) != prof.c_p:
        raise Reject("E_LENGTH")                                        # 9
    if prof.relation_id == REL_FOLDED_GKR_LOOKUP and layout_of(prof) is not None:
        used = HEADER_LEN + prof.body_used
        if any(data[used:]):
            raise Reject("E_PADDING")                                   # 11
        if expected_circuit_root is not None:
            raise Reject("E_CIRCUIT_PINNED")
        if expected_x_root is not None:
            raise Reject("E_X_ROOT_PINNED")
        verify_folded(data[:HEADER_LEN], data[HEADER_LEN:used], layout_of(prof))   # 12-17
        return VERIFIED
    if prof.relation_id == REL_FOLDED_GKR_AIR and air_layout_of(prof) is not None:
        lay = air_layout_of(prof)
        used = HEADER_LEN + lay.body_used
        if any(data[used:]):
            raise Reject("E_PADDING")                                   # 11
        if expected_x_root is not None:
            raise Reject("E_X_ROOT_PINNED")
        verify_air(data[:HEADER_LEN], data[HEADER_LEN:used], lay, expected_circuit_root)   # 12-19
        return VERIFIED
    if prof.relation_id == REL_FOLDED_VEC_MATMUL and vec_layout_of(prof) is not None:
        lay = vec_layout_of(prof)
        used = HEADER_LEN + lay.body_used
        if any(data[used:]):
            raise Reject("E_PADDING")                                   # 11
        if expected_x_root is not None:
            raise Reject("E_X_ROOT_PINNED")
        verify_vec(data[:HEADER_LEN], data[HEADER_LEN:used], lay, expected_circuit_root)   # 12-17
        return VERIFIED
    if prof.relation_id == REL_FOLDED_VEC_BOUND and bound_layout_of(prof) is not None:
        lay = bound_layout_of(prof)
        used = HEADER_LEN + lay.body_used
        if any(data[used:]):
            raise Reject("E_PADDING")                                   # 11
        verify_bound(data[:HEADER_LEN], data[HEADER_LEN:used], lay, expected_circuit_root,
                     expected_x_root)                                   # 12-17
        return VERIFIED
    if prof.relation_id != REL_HARNESS_SQ_SUMCHECK:
        raise Reject("E_BACKEND_UNAVAILABLE")                           # 10
    sec = split_sections(data, prof)
    if any(sec.pad):
        raise Reject("E_PADDING")                                       # 11
    if expected_circuit_root is not None:
        raise Reject("E_CIRCUIT_PINNED")
    if expected_x_root is not None:
        raise Reject("E_X_ROOT_PINNED")
    body = decode_harness(sec, prof)                                    # 12
    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(sec.header)
    tr.absorb_bytes(sec.stmt)
    claim = body.claim
    for i, g in enumerate(body.rounds):                                 # 13
        tr.absorb_bytes(sec.rounds[i])
        if e_add(g[0], g[1]) != claim:
            raise Reject("E_SUMCHECK", f"round {i}")
        r = tr.squeeze_fp3()
        claim = lagrange_eval(g, r)
    tr.absorb_bytes(sec.final)
    if claim != e_mul(body.final, body.final):                          # 14
        raise Reject("E_FINAL")
    mask = (1 << prof.n_vars) - 1
    for q, (v, sibs) in enumerate(body.openings):                       # 15
        idx = int.from_bytes(tr.squeeze_bytes(8), "little") & mask
        if not verify_path(body.r_tab, leaf_digest(v), idx, sibs):
            raise Reject("E_MERKLE", f"query {q}")
    return VERIFIED


def verify_result(data, allow_reduced=False, expected_profile_id=None, expected_circuit_root=None,
                  expected_x_root=None):
    """Returns "VERIFIED" or the reject code (never raises Reject)."""
    try:
        return verify_bytes(data, allow_reduced, expected_profile_id, expected_circuit_root, expected_x_root)
    except Reject as e:
        return e.code
