# SPDX-License-Identifier: Apache-2.0
"""Relation 5 (full-width GEMM tiles: Libra-masked inner-product sumcheck + batched folded ZK-Basefold) against
specs/vectors/fpn2_folded_vec.json."""
import os
import subprocess
import sys
import tempfile
import unittest

from _common import PYROOT, load

from fpn_verify_ref import profile as PR
from fpn_verify_ref.folded_vec import layout_of
from fpn_verify_ref.verify import verify_result

VEC = load("fpn2_folded_vec.json")


def vec(name):
    return next(x for x in VEC["vectors"] if x["name"] == name)


def build(v):
    if "prefix_hex" in v:
        pre = bytes.fromhex(v["prefix_hex"])
        return pre + b"\x00" * (v["total_len"] - len(pre))
    b = bytearray(build(vec(v["base"])))
    for off, hx in v["patches"]:
        by = bytes.fromhex(hx)
        b[off:off + len(by)] = by
    out = bytes(b)
    if v["truncate"] is not None:
        out = out[:v["truncate"]]
    return out + bytes.fromhex(v["append_hex"])


def run(v, data=None):
    pin = bytes.fromhex(v["expected_profile_id"]) if v["expected_profile_id"] else None
    root = bytes.fromhex(v["expected_circuit_root"]) if v["expected_circuit_root"] else None
    return verify_result(build(v) if data is None else data, v["allow_reduced"], pin, root)


class TestVecProfiles(unittest.TestCase):
    def test_pins_and_layout(self):
        for name, pid in VEC["profiles"].items():
            prof, pin = PR.PINNED[name]
            self.assertEqual(pin, pid)
            self.assertEqual(prof.relation_id, PR.REL_FOLDED_VEC_MATMUL)
            self.assertEqual(prof.security_tier, PR.TIER_STANDARD)
            lay = layout_of(prof)
            self.assertLessEqual(64 + lay.body_used, prof.c_p)
            self.assertEqual(lay.B * 32 * lay.K, 1 << (prof.n_vars - 1))      # W fills the data half of C
        n16, n22 = layout_of(PR.FOLDED_VEC_N16_V1), layout_of(PR.FOLDED_VEC_N22_V1)
        self.assertEqual((n16.K, n16.B, n16.d, n16.body_used), (1024, 1, 10, 288368))
        self.assertEqual((n22.K, n22.B, n22.d, n22.body_used), (1024, 64, 16, 469184))
        self.assertEqual(64 + n16.body_used, VEC["offsets"]["fpn2-folded-vec-n16-v1"]["end"])
        self.assertLessEqual(PR.FOLDED_VEC_N16_V1.c_p, 384 * 1024)
        self.assertLessEqual(PR.FOLDED_VEC_N22_V1.c_p, 1024 * 1024)

    def test_geometry_needs_power_of_two_width(self):
        from dataclasses import replace
        self.assertIsNone(layout_of(replace(PR.FOLDED_VEC_N16_V1, max_hidden=1000)))
        self.assertIsNone(layout_of(replace(PR.FOLDED_VEC_N16_V1, max_hidden=16)))
        self.assertIsNone(layout_of(replace(PR.FOLDED_VEC_N16_V1, max_hidden=1 << 11)))


class TestVecVectors(unittest.TestCase):
    def test_every_vector_exact_code(self):
        codes = set()
        for v in VEC["vectors"]:
            with self.subTest(v["name"]):
                got = run(v)
                self.assertEqual(got, v["expect"])
                codes.add(got)
        for c in ("VERIFIED", "E_PROFILE_PINNED", "E_CIRCUIT_UNPINNED", "E_CIRCUIT_PINNED", "E_ZEROCHECK",
                  "E_SUMCHECK", "E_FINAL", "E_MERKLE", "E_FOLD", "E_NONCANONICAL", "E_PADDING", "E_LENGTH"):
            self.assertIn(c, codes)

    def test_sampled_byte_flips_reject(self):
        v = vec("vec_n16_valid_0")
        base = build(v)
        self.assertEqual(run(v, base), "VERIFIED")
        for i in list(range(64, 64 + 96 + 24 + 96 * 10 + 48, 7)) + list(range(0, len(base), 1999)):
            b = bytearray(base)
            b[i] ^= 1 << (i % 8)
            self.assertNotEqual(run(v, bytes(b)), "VERIFIED", i)

    def test_cli_weight_root(self):
        data = build(vec("vec_n16_valid_0"))
        root = VEC["weights"]["tile_n16"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.fpn2")
            with open(path, "wb") as f:
                f.write(data)
            env = dict(os.environ, PYTHONPATH=PYROOT)
            for args, out, code in ((["--circuit-root", root], "VERIFIED", 0),
                                    ([], "REJECTED: E_CIRCUIT_UNPINNED", 1),
                                    (["--circuit-root", VEC["weights"]["tile_n16_tampered"]],
                                     "REJECTED: E_CIRCUIT_PINNED", 1)):
                r = subprocess.run([sys.executable, "-m", "fpn_verify_ref", path, *args], capture_output=True,
                                   text=True, env=env)
                self.assertEqual((r.stdout.strip(), r.returncode), (out, code), args)


if __name__ == "__main__":
    unittest.main()
