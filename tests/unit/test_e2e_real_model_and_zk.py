# SPDX-License-Identifier: Apache-2.0
"""end-to-end relation-4 prover (`circuit/e2e_prover.py`) + strict ZK checks + engine parity.

    PYTHONPATH=prover:serve:verifier/python python3.14 -W error -m unittest \
        tests/unit/test_e2e_real_model_and_zk.py

The samples below are SYNTHETIC stand-ins with the same structure the real-model bench
(`bench/tier2/bench_e2e_real_model.py`) extracts on the GPU host; the witness mapping, lowering, prover and
both verifiers are the same code paths.  Every container is checked by BOTH the Rust
`core/target/release/fpn-verify` (built on demand) and the detached Python `fpn_verify_ref` (fresh
interpreter whose import path holds only `verifier/python`), with the circuit root pinned.
"""
import ast
import hashlib
import os
import re
import sys
import unittest
from fractions import Fraction
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("FPN_NO_NATIVE", "1")
for p in (ROOT, HERE, os.path.join(ROOT, "verifier", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

from prover.fpn_prover.backend import folded_air as FA  # noqa: E402
from prover.fpn_prover.backend import cuda_engine as CE  # noqa: E402
from prover.fpn_prover.backend import fpn2  # noqa: E402
from prover.fpn_prover.backend.fpn2 import FOLDED_AIR_N20_V1, FOLDED_AIR_V1  # noqa: E402
from prover.fpn_prover.circuit import e2e_prover as E  # noqa: E402
from prover.fpn_prover.circuit import lower as LW  # noqa: E402
from test_ir_lowering_and_ntt import SeededRng, build  # noqa: E402
from test_relation4_air_and_wbforge import FPN_VERIFY, rust_verifier  # noqa: E402

bw = E.bf16_word


def synthetic_samples():
    """One sample set per architecture class (values shaped like real BF16 activations / weights)."""
    return [
        E.Samples("dense_gqa_qknorm", "synthetic-Qwen3", (bw(0.73), bw(-1.9)), (bw(1.12), bw(0.88)),
                  bw(-2.375), bw(0.0213), bw(-2.375 * 0.0213), "dense_gqa",
                  dict(normalize=1, query=1.3, key0=-0.7, key1=2.1, value0=0.41, value1=-0.05)),
        E.Samples("dense_gqa", "synthetic-Llama", (bw(0.05), bw(0.31)), (bw(0.4), bw(0.21)),
                  bw(0.9), bw(-0.0071), bw(0.9 * -0.0071), "dense_gqa",
                  dict(normalize=0, query=0.3, key0=-1.7, key1=0.1, value0=-0.02, value1=0.5),
                  epsilon=Fraction(1, 10**5)),
        E.Samples("gdn_hybrid", "synthetic-Qwen3.5", (bw(1.5), bw(-0.25)), (bw(1.01), bw(0.97)),
                  bw(0.125), bw(-0.0332), bw(0.125 * -0.0332), "gdn_lrec",
                  dict(decay=0.93, beta=0.62, key=0.11, kernel=0.37, history=-1.2, value_post=0.08)),
        # gpt-oss-style sliding window + sink attention and a 2-of-N MoE router / expert mixture.
        E.Samples("swa_sink_moe_mxfp4", "synthetic-gpt-oss", (bw(0.6), bw(-1.1)), (bw(0.9), bw(1.3)),
                  bw(1.75), bw(-0.0156), bw(1.75 * -0.0156), "swa_sink",
                  dict(query=0.8, key0=-0.4, key1=1.9, value0=0.27, value1=-0.6)),
        E.Samples("moe_fp8", "synthetic-Qwen3-MoE", (bw(-0.33), bw(2.2)), (bw(1.05), bw(0.95)),
                  bw(0.5), bw(0.043), bw(0.5 * 0.043), "moe_router_expert",
                  dict(scores=(2.31, 1.87), outputs=(0.142, -0.391))),
    ]


class TestE2ERealModelAndZk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rust = rust_verifier()
        if not cls.rust and not os.environ.get("FPN_ALLOW_NO_RUST"):
            raise RuntimeError("Rust fpn-verify missing or stale and cargo unavailable "
                               "(set FPN_ALLOW_NO_RUST=1 only for explicit Python-only runs)")
        cls.samples = synthetic_samples()
        cls.proofs = [E.prove_samples(s, rng=SeededRng(600 + i), record=(i == 0))
                      for i, s in enumerate(cls.samples)]
        cls.root = cls.proofs[0].circuit.root

    def assert_both(self, data, code, root=None):
        root = root or self.root
        py, _, _ = E.python_verify_detached(data, root)
        self.assertEqual(E.verdict_code(py), code)
        if self.rust:
            rs, _, _ = E.rust_verify(FPN_VERIFY, data, root)
            self.assertEqual(E.verdict_code(rs), code)

    # ------------------------------------------------------------------ multi-arch E2E
    def test_all_architectures_share_one_root_and_verify(self):
        test_root = LW.compile_air(build()[0]).root          # the unit-test circuit
        lengths = set()
        for s, pr in zip(self.samples, self.proofs, strict=True):
            with self.subTest(arch=s.arch):
                self.assertEqual(pr.circuit.root, self.root)
                self.assertEqual(pr.circuit.root, test_root)
                self.assertEqual(pr.circuit.layout.n, FOLDED_AIR_V1["n_vars"])
                self.assertTrue(LW.check_binding_air(pr.circuit, pr.container))
                lengths.add(len(pr.container))
                self.assert_both(pr.container, "VERIFIED")
        self.assertEqual(lengths, {FOLDED_AIR_V1["c_p"]})
        self.assertEqual(FOLDED_AIR_V1["c_p"], 262144)
        # The selected opcode is private: the three witnesses select different / differently-normalised branches.
        sel = [tuple(pr.assignment.values[f"slot.{op}"] for op in E.OPCODES) for pr in self.proofs]
        self.assertEqual(sel[0], sel[1])                     # both dense_gqa (normalize differs in a lane)
        self.assertNotEqual(sel[0], sel[2])
        self.assertNotEqual(self.proofs[0].assignment.values["slot.lane0"],
                            self.proofs[1].assignment.values["slot.lane0"])
        # All five opcode classes (incl. swa_sink and moe_router_expert) select distinct one-hot branches.
        self.assertEqual({self.samples[i].opcode for i in range(5)},
                         {"dense_gqa", "gdn_lrec", "swa_sink", "moe_router_expert"})
        self.assertEqual(len({sel[i] for i in (0, 2, 3, 4)}), 4)

    # ------------------------------------------------------------------ new opcodes
    def test_moe_lanes_from_router_logits(self):
        lanes = E.moe_lanes((2.31, 1.87), (0.142, -0.391))
        s0, s1 = E.fixed(2.31), E.fixed(1.87)
        self.assertEqual(lanes[:2], (s0, s1))
        self.assertEqual(lanes[2:6], (1, 1, 1, 0))                   # bitmap all-ones (top_k = experts), argmax 0
        self.assertEqual(lanes[6], 256)                              # exp(0) at scale 256
        self.assertEqual(lanes[10:13], (0, 1, s0))
        self.assertEqual(lanes[13], lanes[6] + lanes[7])
        pr = self.proofs[4]
        self.assertEqual(pr.assignment.values["slot.moe_router_expert"], 1)
        self.assertIn("slot.lane0", pr.provenance.extracted)         # router logit
        self.assertIn("slot.lane8", pr.provenance.extracted)         # expert output
        self.assertIn("slot.lane14", pr.provenance.recomputed)       # mixture
        with self.assertRaises(ValueError):                          # weight exp(-12) * 256 rounds to 0
            E.moe_lanes((12.0, 0.0), (0.1, 0.2))
        with self.assertRaises(ValueError):
            E.slot_lanes("mla_absorbed", {})

    def test_swa_sink_lanes(self):
        pr = self.proofs[3]
        self.assertEqual(pr.assignment.values["slot.swa_sink"], 1)
        self.assertEqual(pr.assignment.values["slot.lane0"], 0)                  # position 0 < extent
        self.assertEqual(pr.assignment.values["slot.lane6"], pr.assignment.values["slot.lane4"])   # out == v0
        self.assertIn("slot.lane0", pr.provenance.recomputed)

    # ------------------------------------------------------------------ engine parity
    def test_host_engine_parity_bit_exact(self):
        """folded_air.prove(engine=HostEngine()) == CPU prove for the same SeededRng stream (dense + swa_sink)."""
        eng = CE.HostEngine()
        for i in (0, 3):
            with self.subTest(arch=self.samples[i].arch):
                cpu = E.prove_samples(self.samples[i], rng=SeededRng(900 + i), record=True)
                eng_pr = E.prove_samples(self.samples[i], rng=SeededRng(900 + i), record=True, engine=eng)
                rec_cpu, rec_eng = cpu.record, eng_pr.record
                self.assertEqual(hashlib.sha256(cpu.container).hexdigest(),
                                 hashlib.sha256(eng_pr.container).hexdigest())
                self.assertEqual(cpu.container, eng_pr.container)
                self.assertEqual(rec_cpu["positions"], rec_eng["positions"])
                self.assertEqual(rec_cpu["a"], rec_eng["a"])
                self.assertEqual(eng_pr.circuit.root, self.root)
                self.assertIn("pair_open", eng_pr.timings_ms["folded_air_prove_phases"])
        self.assertTrue(E.check_public_c_openings(eng_pr.circuit, eng_pr.container, rec_eng["positions"], eng))

    def test_engine_commitment_and_kernels_match_cpu(self):
        eng = CE.HostEngine()
        pr = self.proofs[2]
        lay, table = pr.circuit.layout, pr.circuit.table
        cpu = FA.commit_circuit(lay, table)
        dev = FA.commit_circuit(lay, table, eng)
        self.assertEqual(cpu.root, dev.root)
        self.assertEqual(cpu.values, dev.values)
        self.assertEqual(cpu.code[12345], dev.code_at(12345))
        self.assertEqual([bytes(x) for x in fpn2.merkle_path(cpu.levels, 77)], dev.path(77))
        rng = SeededRng(5)
        V = LW.air_vector(pr.circuit, pr.assignment.values, rng)
        F = FA.build_witness(lay, table, V, SeededRng(6))
        Fn = FA.build_witness_np(lay, table, V, SeededRng(6))
        self.assertEqual([tuple(int(v) for v in row) for row in Fn], F)
        Fa, Ca = Fn.T.copy(), dev.cells
        beta, gamma = (3, 1, 4), (1, 5, 9)
        ref = FA.leaves(lay, F, cpu.values, beta, gamma)
        got = FA.leaves_np(lay, Fa, Ca, beta, gamma)
        self.assertEqual([a + b for a, b in ref], [tuple(int(v) for v in row) for row in got])
        self.assertEqual(FA.z_blind(lay, F), FA.z_blind_np(lay, Fa))

    def test_engine_rejects_cpu_only_hooks(self):
        pr = self.proofs[0]
        eng = CE.HostEngine()
        for hook in ("gkr", "pcs", "zc_round", "zc_finals", "zc_adaptive"):
            with self.subTest(hook=hook), self.assertRaises(ValueError):
                LW.prove_air(pr.circuit, pr.assignment, rng=SeededRng(1), hooks={hook: None}, engine=eng)

    # ------------------------------------------------------------------ n = 20 profile + multi-block
    def test_n20_profile_pinned_consistently(self):
        from fpn_verify_ref import profile as PR
        prof, pin = PR.PINNED["fpn2-folded-air-n20-v1"]
        self.assertEqual(hashlib.blake2b(prof.canonical_bytes(), digest_size=32).hexdigest(), pin)
        self.assertEqual(fpn2.profile_id(FOLDED_AIR_N20_V1).hex(), pin)
        self.assertEqual(fpn2.profile_bytes(FOLDED_AIR_N20_V1), prof.canonical_bytes())
        with open(os.path.join(ROOT, "core", "src", "profile.rs")) as f:
            rs = f.read()
        m = re.search(r"&FOLDED_AIR_N20_V1,\s*\"([0-9a-f]{64})\"", rs)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), pin)
        with open(os.path.join(ROOT, "specs", "11-wire-format.md")) as f:
            self.assertIn(pin[:8], f.read())
        self.assertEqual((prof.n_vars, prof.n_queries, prof.c_p, prof.security_tier, prof.relation_id),
                         (20, 72, 524288, 1, 4))
        lay = FA.layout_of(FOLDED_AIR_N20_V1)
        self.assertEqual((lay.K, lay.body_used), (32768, 424104))
        self.assertEqual(prof.body_used, lay.body_used)
        self.assertLessEqual(64 + lay.body_used, FOLDED_AIR_N20_V1["c_p"])
        # The two existing relation-4 profiles are untouched.
        self.assertEqual(PR.PINNED["fpn2-folded-air-v1"][1][:8], "f6c6b4ca")
        self.assertEqual(PR.PINNED["fpn2-folded-air-n12"][1][:8], "5f661ec4")

    def test_multiblock_witness_and_capacity(self):
        samples = [self.samples[i] for i in (0, 2, 3, 4)]
        program, w, prov = E.build_witness_multi(samples)
        self.assertEqual(program.failures(w), [])
        self.assertEqual(w["b2.slot.swa_sink"], 1)
        self.assertEqual(w["b3.slot.moe_router_expert"], 1)
        self.assertIn("b0.gemm.weight.0.0", prov.extracted)
        with self.assertRaises(ValueError):                          # mixed RMSNorm epsilon
            E.build_witness_multi([self.samples[0], self.samples[1]])
        # 64 blocks fit K = 32,768 (the RTX 6000D GPU receipt proves exactly this circuit)
        one = self.proofs[0].circuit.summary()
        self.assertLessEqual(64 * (one["variables"] - 1 - 3 * FA.ZK_ROWS) + 1 + 3 * FA.ZK_ROWS,
                             FA.layout_of(FOLDED_AIR_N20_V1).K)
        self.assertLessEqual(64 * one["rows"] + FA.ZK_ROWS, FA.layout_of(FOLDED_AIR_N20_V1).K)

    def test_e2e_prover_reads_no_environment(self):
        with open(E.__file__) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ("environ", "getenv", "putenv", "environb"))
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, ("environ", "getenv"))
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                self.assertFalse({a.name for a in node.names} & {"environ", "getenv"})

    def test_provenance_labels_every_wire(self):
        program, names = E.build_program()
        for pr in self.proofs:
            prov = pr.provenance.as_dict()
            labelled = set(prov["extracted"]) | set(prov["recomputed"])
            self.assertLessEqual(set(program.columns), labelled)
            self.assertIn("gemm.weight.0.0", prov["extracted"])
            self.assertIn("gemm.center.0.0", prov["recomputed"])
            self.assertIn("norm.output.0", prov["recomputed"])

    def test_witness_mapping_rejects_inconsistent_samples(self):
        s = synthetic_samples()[0]
        s.gemm_observed = bw(-2.375 * 0.0213 * 1.5)          # far outside the product box tolerance
        with self.assertRaises(LW.LoweringError):
            E.build_witness(s)

    # ------------------------------------------------------------------ ZK: joint rank (security audit BLOCKER regression)
    def test_joint_mask_rank_holds_with_zk_rows(self):
        rep = E.joint_rank(self.proofs[0])
        self.assertTrue(rep["holds"], rep)
        self.assertEqual(rep["rank_B"], rep["rank_BA"])
        self.assertFalse(rep["claims_only"])

    def test_joint_mask_rank_fails_without_zk_rows(self):
        """Removing the ZK_ROWS blinding rows restores the layout (e_a, e_b, e_c share one mask):
        the proof still verifies, but the joint rank check FAILS.  Must stay failing forever."""
        honest = E.joint_rank(self.proofs[0], claims_only=True)
        self.assertTrue(honest["holds"], honest)
        with mock.patch.object(FA, "ZK_ROWS", 0):
            pr = E.prove_samples(self.samples[0], rng=SeededRng(700), record=True)
            self.assertEqual(pr.circuit.summary()["zk_rows"], 0)
            self.assertNotEqual(pr.circuit.root, self.root)
            self.assert_both(pr.container, "VERIFIED", root=pr.circuit.root)
            rep = E.joint_rank(pr, claims_only=True)
        self.assertFalse(rep["holds"], rep)
        self.assertLess(rep["rank_B"], rep["rank_BA"])

    # ------------------------------------------------------------------ ZK: re-proof shares no private block
    def test_reproof_shares_no_private_64_byte_block(self):
        a = E.prove_samples(self.samples[0], record=True)                 # fresh SecretRng randomness
        b = E.prove_samples(self.samples[0], record=True)
        for pr in (a, b):
            self.assertTrue(E.check_public_c_openings(pr.circuit, pr.container, pr.record["positions"]))
        rep = E.shared_blocks(a.container, b.container, a.circuit.layout)
        self.assertEqual(rep["private_aligned_shared"], 0, rep)
        self.assertEqual(rep["private_window_set_intersection"], 0, rep)
        self.assertGreater(rep["private_aligned_blocks"], 3000)
        secs = E.section_map(a.circuit.layout)
        self.assertEqual(secs[0][1], 0)
        self.assertEqual(secs[-1][2], FOLDED_AIR_V1["c_p"])
        self.assertTrue(all(x[2] == y[1] for x, y in zip(secs, secs[1:])))

    # ------------------------------------------------------------------ negative cheats
    def test_cheat_perturbed_weight_rejected_e_zerocheck(self):
        data, info = E.cheat_gemm_weight(self.proofs[0], rng=SeededRng(801))
        self.assertEqual(
            set(info["row_failures"]),
            {"gemm.center_dot.0", "gemm.weight_radius_upper.0.0", "gemm.weight_radius_lower.0.0"},
        )
        self.assertTrue(info["limbs_in_range"])
        self.assert_both(data, "E_ZEROCHECK")

    def test_cheat_multiplicity_rejected_e_gkr_root(self):
        data, _ = E.cheat_multiplicity(self.proofs[2], rng=SeededRng(802))
        self.assert_both(data, "E_GKR_ROOT")


if __name__ == "__main__":
    unittest.main()
