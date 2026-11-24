# SPDX-License-Identifier: Apache-2.0
"""Relation 6 (`FOLDED_VEC_BOUND`: relation 5 with the activations pre-committed under a pinned `x_root`) against
specs/vectors/fpn2_folded_vec_bound.json: exact reject-code parity with the Rust suite on every vector."""
import os
import subprocess
import sys
import tempfile
import unittest

from _common import PYROOT, load

from fpn_verify_ref import profile as PR
from fpn_verify_ref.folded_vec_bound import layout_of
from fpn_verify_ref.verify import verify_result

VEC = load("fpn2_folded_vec_bound.json")
BY_NAME = {x["name"]: x for x in VEC["vectors"]}


def vec(name):
    return BY_NAME[name]


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


def _opt(hx):
    return bytes.fromhex(hx) if hx else None


def run(v, data=None):
    return verify_result(build(v) if data is None else data, v["allow_reduced"], _opt(v["expected_profile_id"]),
                         _opt(v["expected_circuit_root"]), _opt(v["expected_x_root"]))


class TestBoundProfiles(unittest.TestCase):
    def test_pins_and_layout(self):
        for name, pid in VEC["profiles"].items():
            prof, pin = PR.PINNED[name]
            self.assertEqual(pin, pid)
            self.assertEqual(prof.relation_id, PR.REL_FOLDED_VEC_BOUND)
            self.assertEqual(prof.security_tier, PR.TIER_STANDARD)
            lay = layout_of(prof)
            self.assertLessEqual(64 + lay.body_used, prof.c_p)
            self.assertEqual(lay.B * 32 * lay.K, 1 << (prof.n_vars - 1))      # W fills the data half of C
        n16, n22 = layout_of(PR.FOLDED_VEC_BOUND_N16_V1), layout_of(PR.FOLDED_VEC_BOUND_N22_V1)
        self.assertEqual((n16.K, n16.B, n16.d, n16.query_len, n16.body_used), (1024, 1, 10, 4608, 334480))
        self.assertEqual((n22.K, n22.B, n22.d, n22.body_used), (1024, 64, 16, 529120))
        self.assertEqual(64 + n16.body_used, VEC["offsets"]["fpn2-folded-vec-bound-n16-v1"]["end"])
        self.assertLessEqual(PR.FOLDED_VEC_BOUND_N16_V1.c_p, 384 * 1024)
        self.assertGreater(64 + n22.body_used, 512 * 1024)                    # does not fit relation 5's 512 KiB
        self.assertEqual(PR.FOLDED_VEC_BOUND_N22_V1.c_p, 1024 * 1024)
        self.assertNotEqual(PR.FOLDED_VEC_BOUND_N16_V1.profile_id, PR.FOLDED_VEC_N16_V1.profile_id)

    def test_geometry_needs_power_of_two_width(self):
        from dataclasses import replace
        self.assertIsNone(layout_of(replace(PR.FOLDED_VEC_BOUND_N16_V1, max_hidden=1000)))
        self.assertIsNone(layout_of(replace(PR.FOLDED_VEC_BOUND_N16_V1, max_hidden=16)))
        self.assertIsNone(layout_of(replace(PR.FOLDED_VEC_BOUND_N16_V1, max_hidden=1 << 11)))


class TestBoundVectors(unittest.TestCase):
    def test_every_vector_exact_code(self):
        codes = set()
        for v in VEC["vectors"]:
            with self.subTest(v["name"]):
                got = run(v)
                self.assertEqual(got, v["expect"])
                codes.add(got)
        self.assertGreaterEqual(len(VEC["vectors"]), 35)
        for c in ("VERIFIED", "E_PROFILE_PINNED", "E_CIRCUIT_UNPINNED", "E_CIRCUIT_PINNED", "E_X_ROOT_UNPINNED",
                  "E_X_ROOT_PINNED", "E_ZEROCHECK", "E_SUMCHECK", "E_FINAL", "E_MERKLE", "E_FOLD",
                  "E_NONCANONICAL", "E_PADDING", "E_LENGTH"):
            self.assertIn(c, codes)

    def test_case_c_forgeries_rejected(self):
        for name, code in (("vec_bound_n16_case_c_substituted_x_under_honest_x_root", "E_MERKLE"),
                           ("vec_bound_n16_case_c_substituted_x_own_x_root", "E_X_ROOT_PINNED"),
                           ("vec_bound_n16_case_c_forged_e_x", "E_SUMCHECK")):
            self.assertEqual(run(vec(name)), code, name)

    def test_pins_fail_closed_with_fixed_precedence(self):
        v = vec("vec_bound_n16_valid_0")
        data = build(v)
        w, x = bytes.fromhex(v["expected_circuit_root"]), bytes.fromhex(v["expected_x_root"])
        self.assertEqual(verify_result(data, False, None, w, x), "VERIFIED")
        self.assertEqual(verify_result(data, False, None, None, x), "E_CIRCUIT_UNPINNED")
        self.assertEqual(verify_result(data, False, None, w, None), "E_X_ROOT_UNPINNED")
        self.assertEqual(verify_result(data, False, None, None, None), "E_CIRCUIT_UNPINNED")
        self.assertEqual(verify_result(data, False, None, bytes(32), x), "E_CIRCUIT_PINNED")
        self.assertEqual(verify_result(data, False, None, w, bytes(32)), "E_X_ROOT_PINNED")
        self.assertEqual(verify_result(data, False, None, bytes(32), bytes(32)), "E_CIRCUIT_PINNED")
        # the relation-5 weight pin alone never satisfies relation 6
        self.assertEqual(verify_result(data, False, None, w), "E_X_ROOT_UNPINNED")

    def test_sampled_byte_flips_reject(self):
        """Every byte of header .. PCS rounds, then a stride over the queries (the full sweep is opt-in, below)."""
        v = vec("vec_bound_n16_valid_0")
        base = build(v)
        self.assertEqual(run(v, base), "VERIFIED")
        dense = 64 + 128 + 24 + 96 * 10 + 48 + 72 * 18 + 32 * 7 + 24
        for i in list(range(0, dense, 3)) + list(range(dense, len(base), 1999)):
            b = bytearray(base)
            b[i] ^= 1 << (i % 8)
            self.assertNotEqual(run(v, bytes(b)), "VERIFIED", i)

    @unittest.skipUnless(os.environ.get("FPN_EXHAUSTIVE_FLIPS") == "1", "set FPN_EXHAUSTIVE_FLIPS=1 (about 1 h)")
    def test_every_byte_flip_rejects_exhaustive(self):
        """One bit flip (bit = index mod 8) in EVERY one of the 393,216 container bytes, across a process pool."""
        import multiprocessing as mp
        v = vec("vec_bound_n16_valid_0")
        base = build(v)
        n = len(base)
        with mp.Pool() as pool:
            accepted = [r for r in pool.imap_unordered(_flip_chunk, [(a, min(a + 2048, n)) for a in range(0, n, 2048)])
                        for r in r]
        self.assertEqual(accepted, [])

    def test_cli_pins(self):
        data = build(vec("vec_bound_n16_valid_0"))
        root, xr = VEC["weights"]["tile_n16"], VEC["x_roots"]["x_n16"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.fpn2")
            with open(path, "wb") as f:
                f.write(data)
            env = dict(os.environ, PYTHONPATH=PYROOT)
            cases = (
                (["--weight-root", root, "--x-root", xr], "VERIFIED", 0),
                (["--circuit-root", root, "--x-root", xr], "VERIFIED", 0),
                (["--x-root", xr], "REJECTED: E_CIRCUIT_UNPINNED", 1),
                (["--weight-root", root], "REJECTED: E_X_ROOT_UNPINNED", 1),
                ([], "REJECTED: E_CIRCUIT_UNPINNED", 1),
                (["--weight-root", VEC["weights"]["tile_n16_tampered"], "--x-root", xr],
                 "REJECTED: E_CIRCUIT_PINNED", 1),
                (["--weight-root", root, "--x-root", VEC["x_roots"]["x_n16_other"]],
                 "REJECTED: E_X_ROOT_PINNED", 1),
            )
            for args, out, code in cases:
                r = subprocess.run([sys.executable, "-W", "error", "-m", "fpn_verify_ref", path, *args],
                                   capture_output=True, text=True, env=env)
                self.assertEqual((r.stdout.strip(), r.returncode), (out, code), args)
            # Whitespace inside hex root must be rejected (exit 2), matching Rust
            ws_xr = xr[:32] + " " + xr[32:]
            for bad_args in (["--x-root", "zz"], ["--x-root", ws_xr], ["--weight-root", " " + root]):
                r = subprocess.run([sys.executable, "-m", "fpn_verify_ref", path, *bad_args],
                                   capture_output=True, text=True, env=env)
                self.assertEqual(r.returncode, 2, bad_args)

    def test_x_root_and_unsupported_pin_downgrade_rejected(self):
        """Cross-relation pin guard: a valid Relation 5/4/3 proof MUST NOT satisfy an --x-root pin."""
        def _build_from(all_vecs, v):
            if "prefix_hex" in v:
                pre = bytes.fromhex(v["prefix_hex"])
                return pre + b"\x00" * (v["total_len"] - len(pre))
            by_name = {x["name"]: x for x in all_vecs}
            b = bytearray(_build_from(all_vecs, by_name[v["base"]]))
            for off, hx in v["patches"]:
                by = bytes.fromhex(hx)
                b[off:off + len(by)] = by
            return bytes(b)

        r5_all = load("fpn2_folded_vec.json")["vectors"]
        v5 = next(v for v in r5_all if v["name"] == "vec_n16_valid_0")
        b5 = _build_from(r5_all, v5)
        w5 = bytes.fromhex(v5["expected_circuit_root"])
        any_x = b"\x42" * 32
        self.assertEqual(verify_result(b5, False, None, w5, None), "VERIFIED")
        self.assertEqual(verify_result(b5, False, None, w5, any_x), "E_X_ROOT_PINNED")
        with tempfile.TemporaryDirectory() as d:
            p5 = os.path.join(d, "r5.fpn2")
            with open(p5, "wb") as f:
                f.write(b5)
            env = dict(os.environ, PYTHONPATH=PYROOT)
            r = subprocess.run(
                [sys.executable, "-W", "error", "-m", "fpn_verify_ref", p5,
                 "--weight-root", v5["expected_circuit_root"], "--x-root", any_x.hex()],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual((r.stdout.strip(), r.returncode), ("REJECTED: E_X_ROOT_PINNED", 1))
        # Relation 4 with --x-root -> E_X_ROOT_PINNED
        r4_all = load("fpn2_folded_air.json")["vectors"]
        v4 = next(v for v in r4_all if v["name"] == "air_v1_valid_0")
        b4 = _build_from(r4_all, v4)
        self.assertEqual(
            verify_result(b4, False, None, bytes.fromhex(v4["expected_circuit_root"]), any_x),
            "E_X_ROOT_PINNED",
        )
        # Relation 3 with --circuit-root -> E_CIRCUIT_PINNED; with --x-root -> E_X_ROOT_PINNED; both -> E_CIRCUIT_PINNED
        r3_all = load("fpn2_folded_gkr.json")["vectors"]
        v3 = next(v for v in r3_all if v["name"] == "folded_v1_valid_0")
        b3 = _build_from(r3_all, v3)
        self.assertEqual(verify_result(b3, False, None, any_x, None), "E_CIRCUIT_PINNED")
        self.assertEqual(verify_result(b3, False, None, None, any_x), "E_X_ROOT_PINNED")
        self.assertEqual(verify_result(b3, False, None, any_x, any_x), "E_CIRCUIT_PINNED")
        # Relation 1 (harness): unpinned VERIFIED; circuit pin / both -> E_CIRCUIT_PINNED; x pin -> E_X_ROOT_PINNED
        r1_all = load("fpn2_containers.json")["vectors"]
        v1 = next(v for v in r1_all if v["name"] == "valid_harness_0")
        b1 = _build_from(r1_all, v1)
        self.assertEqual(verify_result(b1, True, None, None, None), "VERIFIED")
        self.assertEqual(verify_result(b1, True, None, any_x, None), "E_CIRCUIT_PINNED")
        self.assertEqual(verify_result(b1, True, None, None, any_x), "E_X_ROOT_PINNED")
        self.assertEqual(verify_result(b1, True, None, any_x, any_x), "E_CIRCUIT_PINNED")


def _flip_chunk(span):
    v = vec("vec_bound_n16_valid_0")
    base = build(v)
    out = []
    for i in range(*span):
        b = bytearray(base)
        b[i] ^= 1 << (i % 8)
        if run(v, bytes(b)) == "VERIFIED":
            out.append(i)
    return out


if __name__ == "__main__":
    unittest.main()
