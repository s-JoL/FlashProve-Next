# SPDX-License-Identifier: Apache-2.0
"""Relation 5 (`FOLDED_VEC_MATMUL`): full-width 32 x 1024 GEMM tile proofs.

* HostEngine (kernel host emulation) vs pure-Python CPU reference: byte-identical containers;
* honest proof VERIFIED by native Rust `fpn-verify` and the detached `-I` Python reference;
* forgeries (tampered X, tampered Y, tampered W under the honest root, wrong pinned root, adaptive e_W)
  REJECTED identically by Rust and Python;
* exact BF16 -> integer encoding and the integer-faithfulness bound.
"""
import os
import random
import struct
import unittest

import numpy as np

from flashprove_next.field import ONE, P, hadd, hmul
from fpn_prover.backend import cuda_engine as CE
from fpn_prover.backend import folded_vec as FV
from fpn_prover.backend.fpn2 import FOLDED_VEC_N16_V1, FOLDED_VEC_N22_V1, profile_id
from fpn_prover.circuit import e2e_prover as E2E

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUST = os.path.join(ROOT, "core", "target", "release", "fpn-verify")


class _Rng:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return (self.r.randrange(P), self.r.randrange(P), self.r.randrange(P))


def _tile(seed, lo=-(1 << 15), hi=1 << 15):
    lay = FV.layout_of(FOLDED_VEC_N16_V1)
    g = np.random.default_rng(seed)
    Wi = g.integers(lo, hi, size=(lay.B, FV.ROWS, lay.K))
    Xi = g.integers(lo, hi, size=(lay.B, lay.K))
    Yi, bits = FV.exact_products(Wi, Xi)
    return lay, Wi, Xi, Yi, bits


def _verdicts(data, root):
    py = E2E.verdict_code(E2E.python_verify_detached(data, root)[0])
    rs = E2E.verdict_code(E2E.rust_verify(RUST, data, root)[0]) if os.path.exists(RUST) else None
    return rs, py


class Relation5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CE.HostEngine()
        cls.lay, cls.Wi, cls.Xi, cls.Yi, cls.bits = _tile(11)
        cls.W, cls.X, cls.Y = FV.to_field(cls.Wi), FV.to_field(cls.Xi), FV.to_field(cls.Yi)
        cls.root = FV.weight_root(cls.lay, cls.W, cls.engine)
        cls.honest = FV.prove(cls.W, cls.X, cls.Y, n_s=b"\x07" * 32, rng=_Rng(3), engine=cls.engine)

    def test_layout_and_profiles(self):
        n16, n22 = FV.layout_of(FOLDED_VEC_N16_V1), FV.layout_of(FOLDED_VEC_N22_V1)
        self.assertEqual((n16.B, n16.K, n16.macs), (1, 1024, 32768))
        self.assertEqual((n22.B, n22.K, n22.macs), (64, 1024, 2_097_152))
        self.assertLessEqual(64 + n16.body_used, FOLDED_VEC_N16_V1["c_p"])
        self.assertLessEqual(64 + n22.body_used, FOLDED_VEC_N22_V1["c_p"])
        self.assertEqual(profile_id(FOLDED_VEC_N16_V1).hex(),
                         "a48dfb0a1b0f17580cbab6b756678238200c1b3028a26f92898430d8da801327")
        self.assertLess(self.bits, 62)

    def test_host_engine_matches_cpu_reference(self):
        self.assertEqual(FV.weight_root(self.lay, self.W), self.root)
        cpu = FV.prove(self.W, self.X, self.Y, n_s=b"\x07" * 32, rng=_Rng(3))
        self.assertEqual(len(cpu), FOLDED_VEC_N16_V1["c_p"])
        self.assertEqual(cpu, self.honest)

    def test_honest_dual_verify(self):
        self.assertEqual(len(self.honest), FOLDED_VEC_N16_V1["c_p"])
        rs, py = _verdicts(self.honest, self.root)
        self.assertEqual(py, "VERIFIED")
        if rs is not None:
            self.assertEqual(rs, "VERIFIED")
        self.assertEqual(_verdicts(self.honest, b"\x00" * 32), (None if rs is None else "E_CIRCUIT_PINNED",
                                                                 "E_CIRCUIT_PINNED"))

    def test_forgeries_rejected_by_rust_and_python(self):
        Xt = self.X.copy()
        Xt[0, 100] = (int(Xt[0, 100]) + 1) % P
        Yt = self.Y.copy()
        Yt[0, 31] = (int(Yt[0, 31]) + P - 1) % P
        Wt = self.Wi.copy()
        Wt[0, 17, 513] += 1
        Wtf = FV.to_field(Wt)
        cases = {
            "tampered_x": (FV.prove(self.W, Xt, self.Y, rng=_Rng(4), engine=self.engine), "E_ZEROCHECK"),
            "tampered_y": (FV.prove(self.W, self.X, Yt, rng=_Rng(5), engine=self.engine), "E_ZEROCHECK"),
            # W' X with W' smuggled under the honest weight_root: the weight-leaf openings fail.
            "tampered_w_honest_root": (FV.prove(Wtf, self.X, FV.to_field(FV.exact_products(Wt, self.Xi)[0]),
                                                rng=_Rng(6), engine=self.engine,
                                                hooks={"stmt_weight_root": self.root}), "E_MERKLE"),
            # the honest outputs claimed for the tampered weights (own root): the zero-claim fails first.
            "tampered_w_own_root_is_unpinned": (FV.prove(Wtf, self.X, self.Y, rng=_Rng(7), engine=self.engine),
                                                "E_CIRCUIT_PINNED"),
            "adaptive_e_w": (FV.prove(self.W, self.X, self.Y, rng=_Rng(8), engine=self.engine,
                                      hooks={"e_w": lambda v: hadd(v, ONE)}), "E_SUMCHECK"),
            # false Y, every inner-product round shifted so its sum check passes: dies at the folded-in final claim.
            "adaptive_ip_rounds_false_y": (FV.prove(self.W, self.X, Yt, rng=_Rng(9), engine=self.engine,
                                                    hooks={"ip_adaptive": True}), "E_SUMCHECK"),
        }
        for name, (data, code) in cases.items():
            with self.subTest(name):
                rs, py = _verdicts(data, self.root)
                self.assertEqual(py, code)
                if rs is not None:
                    self.assertEqual(rs, code)
        own = FV.weight_root(self.lay, Wtf, self.engine)
        rs, py = _verdicts(cases["tampered_w_own_root_is_unpinned"][0], own)
        self.assertEqual(py, "E_ZEROCHECK")

    def test_every_output_and_strided_activation_cell_is_constrained(self):
        """Layer-1 perturbation audit: +1 on ANY single Y cell or (stride 8) X cell must break the round-0
        inner-product sum check -- no witness data cell of relation 5 is free (blinding / Libra cells are)."""
        from fpn_prover.backend.fpn2 import Fpn2Transcript, header
        lay = self.lay

        from fpn_prover.backend import frac_logup_gkr as gkr

        def round0(Xv, Yv):
            F = np.ascontiguousarray(FV.build_witness_np(lay, Xv, Yv, _Rng(1)).T)
            tr = Fpn2Transcript()
            tr.absorb_bytes(b"perturbation audit")
            first = []
            _, _, _, _, lam, _, g = FV.ip_prove(tr, lay, F, self.W,
                                                {"ip_round": lambda l, s: (first.append(s) if l == 0 else None, s)[1]})
            return hadd(first[0][0], first[0][1]) == hmul(lam, gkr.libra_sum(g))

        self.assertTrue(round0(self.X, self.Y))
        for i in range(FV.ROWS):
            Yt = self.Y.copy()
            Yt[0, i] = (int(Yt[0, i]) + 1) % P
            self.assertFalse(round0(self.X, Yt), f"Y[0][{i}] unconstrained")
        for j in range(0, lay.K, 8):
            Xt = self.X.copy()
            Xt[0, j] = (int(Xt[0, j]) + 1) % P
            self.assertFalse(round0(Xt, self.Y), f"X[0][{j}] unconstrained")

    def test_reproof_shares_no_private_blocks(self):
        again = FV.prove(self.W, self.X, self.Y, n_s=b"\x07" * 32, engine=self.engine)
        self.assertEqual(again[96:128], self.honest[96:128])             # weight_root is public
        priv_a, priv_b = self.honest[64:96] + self.honest[160:2000], again[64:96] + again[160:2000]
        blocks = lambda b: {b[i:i + 32] for i in range(0, len(b) - 31, 32)}  # noqa: E731
        self.assertFalse(blocks(priv_a) & blocks(priv_b))

    def test_bf16_integer_encoding_is_exact(self):
        vals = [1.5, -0.0078125, 3.0e-5, 448.0, 0.0, -2.25]
        words = [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in vals]
        ints, shift = FV.bf16_words_to_ints(np.array(words))
        for w, i in zip(words, ints):
            exact = struct.unpack("<f", struct.pack("<I", w << 16))[0]
            self.assertEqual(i, int(exact * 2 ** shift))
            self.assertEqual(float(i) / 2 ** shift, exact)
        with self.assertRaises(ValueError):
            FV.bf16_words_to_ints(np.array([0x7F80]))


if __name__ == "__main__":
    unittest.main()
