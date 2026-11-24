# SPDX-License-Identifier: Apache-2.0
"""Relation 3 (ZK-masked FracLogUp-GKR + fused folded ZK-Basefold) against specs/vectors/fpn2_folded_gkr.json."""
import unittest

from _common import load

from fpn_verify_ref import profile as PR
from fpn_verify_ref.folded_gkr import FoldedLayout, fold4, layout_of
from fpn_verify_ref.field import P, e_add, e_mul
from fpn_verify_ref.verify import verify_result

FOLD = load("fpn2_folded_gkr.json")
BASES = {}


def build(v):
    if "prefix_hex" in v:
        pre = bytes.fromhex(v["prefix_hex"])
        return pre + b"\x00" * (v["total_len"] - len(pre))
    b = bytearray(build(next(x for x in FOLD["vectors"] if x["name"] == v["base"])))
    for off, hx in v["patches"]:
        by = bytes.fromhex(hx)
        b[off:off + len(by)] = by
    out = bytes(b)
    if v["truncate"] is not None:
        out = out[:v["truncate"]]
    return out + bytes.fromhex(v["append_hex"])


def run(v, data=None):
    pin = bytes.fromhex(v["expected_profile_id"]) if v["expected_profile_id"] else None
    return verify_result(build(v) if data is None else data, v["allow_reduced"], pin)


def vec(name):
    return next(x for x in FOLD["vectors"] if x["name"] == name)


class TestFoldedProfiles(unittest.TestCase):
    def test_pins_and_layout(self):
        for name, pid in FOLD["profiles"].items():
            prof, pin = PR.PINNED[name]
            self.assertEqual(pin, pid)
            lay = layout_of(prof)
            self.assertIsNotNone(lay)
            self.assertLessEqual(64 + lay.body_used, prof.c_p)
            self.assertEqual(64 + lay.body_used, FOLD["offsets"][name]["end"])
        v1 = layout_of(PR.FOLDED_GKR_V1)
        self.assertEqual((v1.s, v1.quarter, v1.folds, v1.body_used), (9, 128, 6, 158960))
        self.assertEqual(PR.FOLDED_GKR_V1.security_tier, PR.TIER_STANDARD)

    def test_fold4_matches_two_binary_folds(self):
        coeffs = [(i * 7919 % P, i, 0) for i in range(1, 9)]

        def f(x):
            acc = (0, 0, 0)
            for c in reversed(coeffs):
                acc = e_add((acc[0] * x % P, acc[1] * x % P, acc[2] * x % P), c)
            return acc
        iota = pow(7, (P - 1) // 4, P)
        x = 123456789
        vals = [f(x * pow(iota, k, P) % P) for k in range(4)]
        ra, rb = (3, 5, 7), (11, 13, 17)
        got = fold4(vals, pow(x, P - 2, P), pow(iota, P - 2, P), ra, rb)
        y = pow(x, 4, P)
        comp = [e_add(coeffs[e], tuple(c * y % P for c in coeffs[e + 4])) for e in range(4)]
        want = e_add(e_add(comp[0], e_mul(ra, comp[1])), e_add(e_mul(rb, comp[2]), e_mul(e_mul(ra, rb), comp[3])))
        self.assertEqual(got, want)


class TestFoldedVectors(unittest.TestCase):
    def test_all_vectors(self):
        names, codes = set(), set()
        for v in FOLD["vectors"]:
            names.add(v["name"])
            codes.add(v["expect"])
            with self.subTest(v["name"]):
                self.assertEqual(run(v), v["expect"])
        for req in ("folded_v1_valid_0", "folded_v1_root_den_zero", "folded_v1_root_num_nonzero",
                    "folded_v1_tamper_coset_merkle_path", "folded_v1_flip_pcs_round_0", "folded_v1_flip_gkr_mask_gsum",
                    "folded_n8_adaptive_mask_tamper", "folded_n8_fold_layer_shift", "folded_n8_missing_table_entry",
                    "folded_n8_wrong_multiplicity", "folded_n8_zero_denominator_leaf",
                    "folded_v1_noncanonical_query_value"):
            self.assertIn(req, names)
        for code in ("VERIFIED", "E_GKR_ROOT", "E_GKR", "E_SUMCHECK", "E_FINAL", "E_MERKLE", "E_FOLD",
                     "E_NONCANONICAL", "E_PADDING", "E_LENGTH", "E_REDUCED_SECURITY", "E_PROFILE_PINNED"):
            self.assertIn(code, codes)
        self.assertGreaterEqual(len(FOLD["vectors"]), 45)

    def test_standard_tier_needs_no_flag(self):
        self.assertEqual(verify_result(build(vec("folded_v1_valid_0"))), "VERIFIED")
        self.assertEqual(verify_result(build(vec("folded_n8_valid_0"))), "E_REDUCED_SECURITY")

    def test_bit_flips_reject_n8(self):
        """Flipping a bit in (almost) every byte of a valid reduced container must reject."""
        base = build(vec("folded_n8_valid_0"))
        for i in range(0, len(base), 3):
            for bit in ((i // 3) % 8,):
                b = bytearray(base)
                b[i] ^= 1 << bit
                self.assertNotEqual(verify_result(bytes(b), True), "VERIFIED", f"byte {i} bit {bit}")

    def test_bit_flips_reject_v1_sampled(self):
        base = build(vec("folded_v1_valid_0"))
        end = FOLD["offsets"]["fpn2-folded-gkr-v1"]["end"]
        positions = sorted(set(list(range(0, 400, 7)) + list(range(400, end, 1601)) + [end - 1, end, len(base) - 1]))
        for i in positions:
            b = bytearray(base)
            b[i] ^= 1 << (i % 8)
            self.assertNotEqual(verify_result(bytes(b)), "VERIFIED", f"byte {i}")

    def test_truncations_never_crash(self):
        base = build(vec("folded_n8_valid_0"))
        for n in list(range(0, 200)) + list(range(200, len(base), 97)):
            self.assertEqual(verify_result(base[:n], True), "E_LENGTH")


if __name__ == "__main__":
    unittest.main()
