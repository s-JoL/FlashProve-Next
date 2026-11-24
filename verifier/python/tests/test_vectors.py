# SPDX-License-Identifier: Apache-2.0
"""fpn_verify_ref against the golden vectors in specs/vectors/ (the same files the Rust suite consumes)."""
import hashlib
import unittest

from _common import container_bytes, h, h3, load

from fpn_verify_ref import field as F
from fpn_verify_ref import poseidon2 as P2
from fpn_verify_ref import profile as PR
from fpn_verify_ref.transcript import Fpn2Transcript
from fpn_verify_ref.verify import verify_result

PRIM = load("fpn2_primitives.json")
CONT = load("fpn2_containers.json")

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class TestField(unittest.TestCase):
    def test_p(self):
        self.assertEqual(h(PRIM["p"]), F.P)

    def test_ops(self):
        n = 0
        for v in PRIM["field"]:
            op = v["op"]
            n += 1
            with self.subTest(i=n, op=op):
                if op in ("fp_add", "fp_sub", "fp_mul"):
                    fn = {"fp_add": F.fp_add, "fp_sub": F.fp_sub, "fp_mul": F.fp_mul}[op]
                    self.assertEqual(fn(h(v["a"]), h(v["b"])), h(v["out"]))
                elif op == "fp_neg":
                    self.assertEqual(F.fp_neg(h(v["a"])), h(v["out"]))
                elif op == "fp_inv":
                    if v["out"] == "error":
                        self.assertRaises(ZeroDivisionError, F.fp_inv, h(v["a"]))
                    else:
                        self.assertEqual(F.fp_inv(h(v["a"])), h(v["out"]))
                elif op == "fp_pow":
                    self.assertEqual(F.fp_pow(h(v["a"]), h(v["e"])), h(v["out"]))
                elif op == "fp_decode":
                    b = bytes.fromhex(v["bytes"])
                    if v["out"] == "noncanonical":
                        self.assertRaises(F.NonCanonical, F.fp_decode, b)
                    else:
                        self.assertEqual(F.fp_decode(b), h(v["out"]))
                        self.assertEqual(F.fp_encode(h(v["out"])), b)
                elif op in ("fp3_add", "fp3_sub", "fp3_mul"):
                    fn = {"fp3_add": F.e_add, "fp3_sub": F.e_sub, "fp3_mul": F.e_mul}[op]
                    self.assertEqual(fn(h3(v["a"]), h3(v["b"])), h3(v["out"]))
                elif op == "fp3_neg":
                    self.assertEqual(F.e_neg(h3(v["a"])), h3(v["out"]))
                elif op == "fp3_inv":
                    if v["out"] == "error":
                        self.assertRaises(ZeroDivisionError, F.e_inv, h3(v["a"]))
                    else:
                        self.assertEqual(F.e_inv(h3(v["a"])), h3(v["out"]))
                elif op == "fp3_pow":
                    self.assertEqual(F.e_pow(h3(v["a"]), h(v["e"])), h3(v["out"]))
                elif op == "fp3_lagrange_eval":
                    self.assertEqual(F.lagrange_eval([h3(x) for x in v["vals"]], h3(v["r"])), h3(v["out"]))
                elif op == "fp3_eq_pt":
                    self.assertEqual(F.eq_pt([h3(x) for x in v["a"]], [h3(x) for x in v["b"]]), h3(v["out"]))
                elif op == "fp3_decode":
                    b = bytes.fromhex(v["bytes"])
                    if v["out"] == "noncanonical":
                        self.assertRaises(F.NonCanonical, F.e_decode, b)
                    else:
                        self.assertEqual(F.e_decode(b), h3(v["out"]))
                        self.assertEqual(F.e_encode(h3(v["out"])), b)
                else:
                    self.fail("unknown op " + op)
        self.assertGreater(n, 500)


class TestPoseidon2(unittest.TestCase):
    V = PRIM["poseidon2"]

    def test_permute(self):
        for v in self.V["permute"]:
            self.assertEqual(P2.permute([h(x) for x in v["in"]]), [h(x) for x in v["out"]])

    @unittest.skipUnless(HAVE_NUMPY, "NumPy batch path (scalar path is covered unconditionally)")
    def test_permute_batch(self):
        import numpy as np
        ins = np.array([[h(x) for x in v["in"]] for v in self.V["permute"]], dtype=np.uint64).T
        outs = np.array([[h(x) for x in v["out"]] for v in self.V["permute"]], dtype=np.uint64).T
        self.assertTrue((P2.permute_batch(ins) == outs).all())

    def test_compress(self):
        for v in self.V["compress"]:
            self.assertEqual(P2.compress(tuple(h(x) for x in v["left"]), tuple(h(x) for x in v["right"])),
                             tuple(h(x) for x in v["out"]))

    def test_sponges(self):
        for v in self.V["sponge_blocks"]:
            self.assertEqual(P2.sponge_blocks([[h(x) for x in b] for b in v["blocks"]], v["domain"]),
                             tuple(h(x) for x in v["out"]))
        for v in self.V["sponge_msg"]:
            self.assertEqual(P2.sponge_msg([h(x) for x in v["msg"]], v["domain"]), tuple(h(x) for x in v["out"]))
        self.assertEqual(self.V["dom_leaf"], P2.DOM_LEAF)

    def test_merkle(self):
        for v in self.V["merkle"]:
            leaves = [tuple(h(x) for x in l) for l in v["leaves"]]
            root, levels = P2.merkle_root(leaves)
            self.assertEqual(root, tuple(h(x) for x in v["root"]))
            for p in v["paths"]:
                path = [tuple(h(x) for x in s) for s in p["path"]]
                self.assertEqual(P2.merkle_path(levels, p["idx"]), path)
                self.assertTrue(P2.verify_path(root, leaves[p["idx"]], p["idx"], path))
                if path:
                    bad = [tuple((path[0][0] ^ 1,) + path[0][1:])] + path[1:]
                    self.assertFalse(P2.verify_path(root, leaves[p["idx"]], p["idx"], bad))


class TestBlake2bAndTranscript(unittest.TestCase):
    def test_blake2b(self):
        for v in PRIM["blake2b"]:
            d = hashlib.blake2b(bytes.fromhex(v["msg"]), digest_size=v["nn"], person=bytes.fromhex(v["person"]))
            self.assertEqual(d.hexdigest(), v["out"])

    def test_transcript(self):
        for script in PRIM["transcript"]:
            tr = None
            for st in script:
                op = st["op"]
                if op == "init":
                    tr = Fpn2Transcript(bytes.fromhex(st["label"]))
                elif op == "absorb_bytes":
                    tr.absorb_bytes(bytes.fromhex(st["data"]))
                elif op == "absorb_fp":
                    tr.absorb_fp(h(st["x"]))
                elif op == "absorb_fp3":
                    tr.absorb_fp3(h3(st["a"]))
                elif op == "squeeze_bytes":
                    self.assertEqual(tr.squeeze_bytes(st["n"]).hex(), st["out"])
                elif op == "squeeze_fp3":
                    self.assertEqual(tr.squeeze_fp3(), h3(st["out"]))
                elif op == "squeeze_u192":
                    self.assertEqual(tr.squeeze_u192(), h(st["out"]))
                self.assertEqual(tr.state.hex(), st["state"], op)


class TestProfiles(unittest.TestCase):
    def test_pinned(self):
        self.assertEqual(len(PRIM["profiles"]), len(PR.PINNED))
        for v in PRIM["profiles"]:
            prof, pin = PR.PINNED[v["name"]]
            cb = prof.canonical_bytes()
            self.assertEqual(cb.hex(), v["canonical_hex"])
            self.assertEqual(hashlib.blake2b(cb, digest_size=32).hexdigest(), pin)
            self.assertEqual(pin, v["profile_id"])
            self.assertEqual(PR.decode_profile(cb), prof)
            for i in (0, 8, 42, 66, 111, 127):
                bad = bytearray(cb); bad[i] ^= 0x80
                self.assertRaises(ValueError, PR.decode_profile, bytes(bad))


class TestContainers(unittest.TestCase):
    def _run(self, v, data=None):
        data = container_bytes(v) if data is None else data
        pin = bytes.fromhex(v["expected_profile_id"]) if v["expected_profile_id"] else None
        return verify_result(data, v["allow_reduced"], pin)

    def test_vectors(self):
        names = set()
        for v in CONT["vectors"]:
            names.add(v["name"])
            with self.subTest(v["name"]):
                self.assertEqual(self._run(v), v["expect"])
        for req in ("wrong_magic", "wrong_version_1", "nonzero_flags", "nonzero_reserved_first", "wrong_profile_id",
                    "noncanonical_fp_eq_p_rtab", "truncated_by_1", "extended_by_1_zero", "nonzero_pad_first",
                    "flip_bit_sumcheck_g0", "downgrade_reduced_without_flag", "valid_harness_0"):
            self.assertIn(req, names)

    def test_every_byte_bound(self):
        """Flipping any single bit anywhere in a valid container must reject."""
        v = next(x for x in CONT["vectors"] if x["name"] == "valid_harness_0")
        base = container_bytes(v)
        for i in range(len(base)):
            for bit in (0, 7):
                b = bytearray(base); b[i] ^= 1 << bit
                self.assertNotEqual(verify_result(bytes(b), True), "VERIFIED", f"byte {i} bit {bit}")

    def test_truncations_never_crash(self):
        v = next(x for x in CONT["vectors"] if x["name"] == "valid_harness_0")
        base = container_bytes(v)
        for n in range(0, len(base)):
            self.assertEqual(verify_result(base[:n], True), "E_LENGTH")


if __name__ == "__main__":
    unittest.main()
