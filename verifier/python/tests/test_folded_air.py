# SPDX-License-Identifier: Apache-2.0
"""Relation 4 (Plonkish zerocheck + wiring/range FracLogUp-GKR + batched folded ZK-Basefold) against
specs/vectors/fpn2_folded_air.json."""
import os
import subprocess
import sys
import tempfile
import unittest

from _common import PYROOT, load

from fpn_verify_ref import profile as PR
from fpn_verify_ref.folded_air import layout_of
from fpn_verify_ref.verify import verify_result

AIR = load("fpn2_folded_air.json")


def vec(name):
    return next(x for x in AIR["vectors"] if x["name"] == name)


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


class TestAirProfiles(unittest.TestCase):
    def test_pins_and_layout(self):
        for name, pid in AIR["profiles"].items():
            prof, pin = PR.PINNED[name]
            self.assertEqual(pin, pid)
            lay = layout_of(prof)
            self.assertEqual(prof.relation_id, PR.REL_FOLDED_GKR_AIR)
            self.assertLessEqual(64 + lay.body_used, prof.c_p)
            self.assertEqual(64 + lay.body_used, AIR["offsets"][name]["end"])
        v1 = layout_of(PR.FOLDED_AIR_V1)
        self.assertEqual((v1.g, v1.K, v1.S, v1.folds, v1.body_used), (9, 512, 12, 7, 247176))
        self.assertEqual(PR.FOLDED_AIR_V1.security_tier, PR.TIER_STANDARD)
        self.assertLessEqual(PR.FOLDED_AIR_V1.c_p, 256 * 1024)
        self.assertEqual(PR.FOLDED_AIR_N12.security_tier, PR.TIER_REDUCED)


class TestAirVectors(unittest.TestCase):
    def test_every_vector_exact_code(self):
        codes = set()
        for v in AIR["vectors"]:
            with self.subTest(v["name"]):
                got = run(v)
                self.assertEqual(got, v["expect"])
                codes.add(got)
        for c in ("VERIFIED", "E_REDUCED_SECURITY", "E_CIRCUIT_UNPINNED", "E_CIRCUIT_PINNED", "E_GKR_ROOT",
                  "E_GKR", "E_ZEROCHECK", "E_SUMCHECK", "E_FINAL", "E_MERKLE", "E_FOLD", "E_NONCANONICAL",
                  "E_PADDING", "E_LENGTH"):
            self.assertIn(c, codes)

    def test_sampled_byte_flips_reject(self):
        v = vec("air_n12_valid_0")
        base = build(v)
        self.assertEqual(run(v, base), "VERIFIED")
        for i in range(0, len(base), 53):
            b = bytearray(base)
            b[i] ^= 1 << (i % 8)
            self.assertNotEqual(run(v, bytes(b)), "VERIFIED", i)

    def test_circuit_root_is_the_only_extra_input(self):
        v = vec("air_v1_valid_0")
        data = build(v)
        root = bytes.fromhex(AIR["circuits"]["air_demo_program"])
        self.assertEqual(verify_result(data, False, None, root), "VERIFIED")
        self.assertEqual(verify_result(data, False, None, None), "E_CIRCUIT_UNPINNED")
        other = bytes.fromhex(AIR["circuits"]["air_small_program"])
        self.assertEqual(verify_result(data, False, None, other), "E_CIRCUIT_PINNED")

    def test_cli_circuit_root(self):
        data = build(vec("air_v1_valid_0"))
        root = AIR["circuits"]["air_demo_program"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.fpn2")
            with open(path, "wb") as f:
                f.write(data)
            env = dict(os.environ, PYTHONPATH=PYROOT)
            for args, out, code in ((["--circuit-root", root], "VERIFIED", 0),
                                    ([], "REJECTED: E_CIRCUIT_UNPINNED", 1),
                                    (["--circuit-root", "00" * 32], "REJECTED: E_CIRCUIT_PINNED", 1),
                                    (["--circuit-root", "xyz"], "", 2)):
                r = subprocess.run([sys.executable, "-m", "fpn_verify_ref", path, *args], capture_output=True,
                                   text=True, env=env)
                self.assertEqual((r.stdout.strip(), r.returncode), (out, code), args)


if __name__ == "__main__":
    unittest.main()
