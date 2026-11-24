#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generator of `lean/tests/rel5_multitile.json`, the multi-tile relation-5 fixture that
`check_rel5_vectors.py` runs through the Lean `parseContainerRel5` / `rel5Check`.

This is the ONLY script here that needs the repository: it imports the CPU prover
(`prover/fpn_prover/backend/folded_vec.py`, pure Python `pair_open`, no GPU), the `Fp3` addition of
`flashprove_next/field.py` (for the fold-layer hook) and the independent
reference verifier (`verifier/python/fpn_verify_ref/folded_vec.py::verify_vec`). The checker reads the
committed JSON and imports none of them. Regenerate (needs numpy; about a minute of CPU):

    python3 lean/tests/gen_rel5_multitile.py            # rewrites lean/tests/rel5_multitile.json

The profile is a test-only, UNREGISTERED relation-5 profile, the smallest admissible layout with four
tiles: n = 14, K = 64 (k = 6), beta = 2, B = 4, d = 8, Q = 3, C_p = 12 288 (128 padding bytes). The
statement is Y = W X for four random 32 x 64 tiles. Every expected value written to the JSON comes from
the prover's record (challenges) or from the reference verifier (results); none comes from Lean.

Contents: the accepted container, one container of a FALSE statement (Y wrong in tile 3 only), one
container of the TRUE statement whose committed folded layer 1 is inconsistent with layer 0 under a
valid Merkle root (the `pcs.code` prover hook of `specs/vectors/generate.py::vec_n16_fold_layer_shift`;
a byte patch cannot reach `E_FOLD`, the Merkle path is checked first), the challenges r_i, r_b, lambda,
s, mu of the accepted container, and tamper cases (patches / truncation / pins) with the reference
verifier's reject code.
"""
import json
import os
import random
import sys

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TESTS))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "prover"))
sys.path.insert(0, os.path.join(REPO, "verifier", "python"))

from flashprove_next.field import ONE as HONE, hadd as HADD  # noqa: E402
from fpn_prover.backend import folded_vec as FV  # noqa: E402
from fpn_prover.backend.fpn2 import REL_FOLDED_VEC_MATMUL, profile_id  # noqa: E402
from fpn_verify_ref import folded_vec as RV  # noqa: E402
from fpn_verify_ref.wire import Reject  # noqa: E402

P = 0xFFFFFFFF00000001
PROFILE = dict(name="fpn2-test-vec-n14-b4", tier=0, relation=REL_FOLDED_VEC_MATMUL, bits=0,
               n_vars=14, sc_degree=3, n_queries=3, c_p=12288, max_hidden=64)
HEADER_LEN = 64


class SeededRng:
    """Deterministic stand-in for `SecretRng` (a public test fixture needs no secrecy)."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return tuple(self.r.randrange(P) for _ in range(3))


def ref_verify(data, lay, pin):
    """`fpn_verify_ref.verify.verify_bytes` from check 9 on, for an unregistered profile: length,
    zero padding, then `verify_vec` (checks 12-17, the pin being check 13)."""
    try:
        if len(data) < HEADER_LEN:
            raise Reject("E_LENGTH")
        if len(data) != PROFILE["c_p"]:
            raise Reject("E_LENGTH")
        used = HEADER_LEN + lay.body_used
        if any(data[used:]):
            raise Reject("E_PADDING")
        RV.verify_vec(data[:HEADER_LEN], data[HEADER_LEN:used], lay, pin)
        return "VERIFIED"
    except Reject as e:
        return e.code


def patched(base, patches, truncate=None, append=b""):
    b = bytearray(base)
    for off, by in patches:
        b[off:off + len(by)] = by
    out = bytes(b)
    if truncate is not None:
        out = out[:truncate]
    return out + append


def main():
    lay = FV.layout_of(PROFILE)
    rlay = RV.VecLayout(lay.n, lay.queries, lay.k)
    assert (lay.beta, lay.B, lay.K, lay.d) == (2, 4, 64, 8)
    rng = random.Random(20261124)
    W = np.array([rng.randrange(P) for _ in range(lay.B * 32 * lay.K)], dtype=np.uint64).reshape(lay.B, 32, lay.K)
    Xv = np.array([rng.randrange(P) for _ in range(lay.B * lay.K)], dtype=np.uint64).reshape(lay.B, lay.K)
    Y = np.array([[sum(int(W[b, i, j]) * int(Xv[b, j]) for j in range(lay.K)) % P for i in range(32)]
                  for b in range(lay.B)], dtype=np.uint64)
    n_s = bytes(range(32))
    rec = {}
    good = FV.prove(W, Xv, Y, n_s=n_s, profile=PROFILE, rng=SeededRng(1), hooks={"record": rec})
    wroot = good[HEADER_LEN + 32:HEADER_LEN + 64]
    assert wroot == FV.weight_root(lay, W)
    assert ref_verify(good, rlay, wroot) == "VERIFIED"

    Ybad = Y.copy()
    Ybad[3, 17] = (int(Ybad[3, 17]) + 1) % P                 # tile 3 only: tiles 0..2 still satisfy Y = W X
    false_stmt = FV.prove(W, Xv, Ybad, n_s=n_s, profile=PROFILE, rng=SeededRng(1))
    assert false_stmt[HEADER_LEN + 32:HEADER_LEN + 64] == wroot

    # Layer 1 of the fold chain shifted by one before it is committed: every Merkle path of the proof is
    # valid, the layer-1 value at the queried slot is not the fold of the layer-0 coset.
    fold_shift = FV.prove(W, Xv, Y, n_s=n_s, profile=PROFILE, rng=SeededRng(1),
                          hooks={"pcs": {"code": lambda t, c: [HADD(x, HONE) for x in c] if t == 0 else c}})
    assert fold_shift[HEADER_LEN + 32:HEADER_LEN + 64] == wroot

    used = HEADER_LEN + lay.body_used
    ip0 = HEADER_LEN + 96                                     # G_sum
    rounds0 = ip0 + 24                                        # inner-product round 0
    ew = rounds0 + 96 * lay.d
    zb = ew + 24
    pcs_r0 = zb + 24
    final = zb + lay.pcs_len - 24
    q0 = HEADER_LEN + 96 + lay.ip_len + lay.pcs_len
    f_leaf_len = 96 + 32 * lay.depth(0)

    def bump(off):                                            # +1 on the low byte of a limb (stays canonical here)
        return [(off, bytes([(good[off] + 1) & 0xFF]))]

    cases = [dict(name="mt_valid", base="good", patches=[], truncate=None, append_hex="", pin="root"),
             dict(name="mt_false_statement_tile3", base="false", patches=[], truncate=None, append_hex="", pin="root"),
             dict(name="mt_no_pin", base="good", patches=[], truncate=None, append_hex="", pin=None),
             dict(name="mt_wrong_pin", base="good", patches=[], truncate=None, append_hex="", pin="other"),
             dict(name="mt_truncated", base="good", patches=[], truncate=PROFILE["c_p"] - 1, append_hex="", pin="root"),
             dict(name="mt_extended", base="good", patches=[], truncate=None, append_hex="00", pin="root"),
             dict(name="mt_pad_first", base="good", patches=[(used, b"\x01")], truncate=None, append_hex="", pin="root"),
             dict(name="mt_pad_last", base="good", patches=[(PROFILE["c_p"] - 1, b"\x80")], truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_noncanonical_gsum", base="good", patches=[(ip0, b"\xff" * 8)], truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_gsum", base="good", patches=bump(ip0), truncate=None, append_hex="", pin="root"),
             dict(name="mt_ip_round0", base="good", patches=bump(rounds0), truncate=None, append_hex="", pin="root"),
             dict(name="mt_ip_round_col5", base="good", patches=bump(rounds0 + 96 * 5 + 48), truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_ip_round_tile0", base="good", patches=bump(rounds0 + 96 * 6), truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_ip_round_tile1_s3", base="good", patches=bump(rounds0 + 96 * 7 + 72), truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_e_w", base="good", patches=bump(ew), truncate=None, append_hex="", pin="root"),
             dict(name="mt_z_blind", base="good", patches=bump(zb), truncate=None, append_hex="", pin="root"),
             dict(name="mt_pcs_round0_h2", base="good", patches=bump(pcs_r0 + 48), truncate=None, append_hex="",
                  pin="root"),
             dict(name="mt_final", base="good", patches=bump(final), truncate=None, append_hex="", pin="root"),
             dict(name="mt_stmt_root0", base="good", patches=bump(HEADER_LEN), truncate=None, append_hex="",
                  pin="root"),
             dict(name="mt_stmt_ns", base="good", patches=bump(HEADER_LEN + 64), truncate=None, append_hex="",
                  pin="root"),
             dict(name="mt_stmt_weight_root_repinned", base="good", patches=bump(HEADER_LEN + 32), truncate=None,
                  append_hex="", pin="patched"),
             dict(name="mt_q0_f_leaf", base="good", patches=bump(q0), truncate=None, append_hex="", pin="root"),
             dict(name="mt_q0_weight_leaf", base="good", patches=bump(q0 + f_leaf_len), truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_q0_weight_sibling", base="good", patches=bump(q0 + f_leaf_len + 96), truncate=None,
                  append_hex="", pin="root"),
             dict(name="mt_q2_last_layer", base="good", patches=bump(q0 + 3 * lay.query_len - (96 + 32 * lay.depth(6))),
                  truncate=None, append_hex="", pin="root"),
             dict(name="mt_fold_layer_shift", base="fold", patches=[], truncate=None, append_hex="", pin="root")]
    bases = {"good": good, "false": false_stmt, "fold": fold_shift}
    other = bytes(31) + b"\x01"
    for c in cases:
        data = patched(bases[c["base"]], c["patches"], c["truncate"], bytes.fromhex(c["append_hex"]))
        pin = {None: None, "root": wroot, "other": other, "patched": data[HEADER_LEN + 32:HEADER_LEN + 64]}[c["pin"]]
        c["expect"] = ref_verify(data, rlay, pin)
        c["pin"] = None if pin is None else pin.hex()
        c["patches"] = [[off, by.hex()] for off, by in c["patches"]]
    assert cases[0]["expect"] == "VERIFIED" and all(c["expect"] != "VERIFIED" for c in cases[1:])
    assert cases[-1]["name"] == "mt_fold_layer_shift" and cases[-1]["expect"] == "E_FOLD"

    def limbs(vals):
        return [[int(x) for x in v] for v in vals]

    out = dict(
        note="Generated by lean/tests/gen_rel5_multitile.py (CPU prover + reference verifier). Test-only, "
             "unregistered relation-5 profile with beta = 2 (four tiles).",
        profile=dict(PROFILE, id=profile_id(PROFILE).hex()),
        layout=dict(n=lay.n, k=lay.k, beta=lay.beta, tiles=lay.B, d=lay.d, folds=lay.folds,
                    end=HEADER_LEN + lay.body_used),
        weight_root=wroot.hex(),
        containers={k: dict(prefix_hex=v[:used].hex(), total_len=len(v)) for k, v in bases.items()},
        challenges=dict(r_i=limbs(rec["r_i"]), r_b=limbs(rec["r_b"]), lam=limbs([rec["lam"]])[0],
                        s=limbs(rec["s"]), mu=limbs([rec["mu"]])[0]),
        cases=cases)
    assert len(out["challenges"]["r_b"]) == 2 and len(out["challenges"]["s"]) == 8
    assert not any(good[used:]) and not any(false_stmt[used:]) and not any(fold_shift[used:])
    path = os.path.join(TESTS, "rel5_multitile.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", path, os.path.getsize(path), "bytes")
    for c in cases:
        print(f"  {c['name']:32s} {c['expect']}")


if __name__ == "__main__":
    main()
