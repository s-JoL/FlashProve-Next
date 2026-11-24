# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the non-GEMM operator calibration bench (``bench/tier2/bench_non_gemm_calibrate.py``).

All four calibrators (GQA softmax, SWA + sinks, Gated DeltaNet recurrence, RMSNorm) and the ``--dry-run`` driver run
without torch.  Every "honest passes" assertion is paired with a negative one: a shrunken radius, a tampered operand
or a cheating kernel must be flagged, and an uncertifiable operating point must raise instead of being clamped.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from fractions import Fraction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(ROOT, "prover"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fpn_prover.circuit import protocol as PR  # noqa: E402

BENCH = os.path.join(ROOT, "bench", "tier2", "bench_non_gemm_calibrate.py")


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_non_gemm_calibrate", BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


NG = load_bench()
SPECS = {m.name: m for m in NG.MODELS}
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def rows(label, count, width, scale):
    return NG._bf16_rows(NG.Stream(label), count, width, scale)


class TestScope(unittest.TestCase):
    def test_constants_are_the_certified_ones(self):
        self.assertEqual(PR.CIRCUIT_NORM_LIFT, Fraction(1, 2**32))
        self.assertEqual(PR.ETA_BF16, Fraction(1, 2**126))
        self.assertEqual((NG.NORM_LIFT, NG.ETAB, NG.ETA32), (2.0**-32, 2.0**-126, 2.0**-126))
        self.assertEqual((NG.U32, NG.UB, NG.UE), (2.0**-24, 2.0**-8, 2.0**-21))
        self.assertEqual(NG.RMS_FLOOR, 1e-3)

    def test_every_family_is_assigned_to_the_required_models(self):
        self.assertEqual(len(SPECS), 5)
        for name in ("Qwen3-4B", "Llama-3.2-1B-Instruct", "Qwen3-30B-A3B-FP8"):
            self.assertIn("gqa_softmax", NG.operators_for(name))
        self.assertEqual(NG.operators_for("gpt-oss-20b"), ("swa_sinks", "rmsnorm"))
        self.assertEqual(NG.operators_for("Qwen3.5-0.8B"), ("lrec", "rmsnorm"))
        for name in SPECS:
            self.assertIn("rmsnorm", NG.operators_for(name))
        self.assertEqual({op for name in SPECS for op in NG.operators_for(name)}, set(NG.OPERATORS))

    def test_importing_and_dry_running_does_not_need_torch(self):
        # A fresh interpreter under -W error: the bench must neither warn nor pull torch in.
        import subprocess
        code = ("import sys, runpy; sys.argv = ['bench', '--dry-run']\n"
                "try:\n    runpy.run_path(%r, run_name='__main__')\nexcept SystemExit as e:\n    rc = e.code\n"
                "assert rc == 0, rc\nassert 'torch' not in sys.modules\n" % BENCH)
        env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "prover")}
        done = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("dry run (synthetic, not calibration evidence): OK", done.stdout)


class TestAttentionRadius(unittest.TestCase):
    def setUp(self):
        self.q = rows("t/attn/q", 1, 16, 2.0)[0]
        self.keys = rows("t/attn/k", 40, 16, 1.0)
        self.values = rows("t/attn/v", 40, 16, 4.0)
        self.scale = 16 ** -0.5

    def test_score_radius_and_lse_agree_with_protocol(self):
        ref = NG.attention_reference(self.q, self.keys, self.values, self.scale)
        rad = NG.attention_radius(ref, 16)
        expect = max(PR.score_radius(Fraction(m), Fraction(abs(z)), Fraction(abs(ref["zmax"])), 16)
                     for z, m in zip(ref["z"], ref["abs_dot"]))
        self.assertAlmostEqual(rad["rbar"], float(expect), delta=float(expect) * 1e-12)
        cert = PR.certified_lse_radius(40, score_radius_max=Fraction(rad["rbar"]),
                                       score_max_abs=Fraction(abs(ref["zmax"])))
        self.assertAlmostEqual(rad["e_den"], float(cert.e_den), delta=float(cert.e_den) * 1e-9)
        r_lse = NG.lse_circuit_radius(rad)
        self.assertGreaterEqual(r_lse, float(cert.circuit))
        self.assertLess(r_lse, float(cert.circuit) * (1 + 1e-9))
        # The uncapped lift and both table steps are part of the circuit radius.
        self.assertGreater(r_lse, float(cert.floating + PR.CIRCUIT_EXP_STEP + PR.CIRCUIT_LOG_STEP))
        self.assertEqual(cert.norm_rel, 40 * PR.CIRCUIT_NORM_LIFT)

    def test_reference_is_a_softmax_and_sink_takes_mass_without_value(self):
        ref = NG.attention_reference(self.q, self.keys, self.values, self.scale)
        self.assertAlmostEqual(sum(ref["p"]), 1.0, places=12)
        sink = NG.attention_reference(self.q, self.keys, self.values, self.scale, sink_logit=ref["zmax"])
        self.assertEqual(len(sink["p"]), 41)
        self.assertAlmostEqual(sum(sink["p"]), 1.0, places=12)
        self.assertEqual(sink["values"][-1], [0.0] * 16)
        share = 1.0 - sink["p"][-1]
        for a, b in zip(sink["o"], ref["o"]):
            self.assertAlmostEqual(a, b * share, delta=1e-9)
        self.assertGreater(sink["lse"], ref["lse"])

    def test_honest_kernel_inside_and_cheats_outside(self):
        ref = NG.attention_reference(self.q, self.keys, self.values, self.scale)
        rad = NG.attention_radius(ref, 16)
        r_lse = NG.lse_circuit_radius(rad)
        out, lse = NG.engine_attention_fp32acc(self.q, self.keys, self.values, self.scale)
        honest = NG.check_head(ref, rad, r_lse, out, lse)
        self.assertFalse(honest["bad"])
        self.assertTrue(0.0 < honest["out_ratio"] < 1.0)
        self.assertTrue(0.0 < honest["lse_ratio"] < 1.0)
        top = max(range(40), key=lambda j: ref["p"][j])
        keep = [j for j in range(40) if j != top]
        out2, lse2 = NG.engine_attention_fp32acc(self.q, [self.keys[j] for j in keep],
                                                 [self.values[j] for j in keep], self.scale)
        self.assertTrue(NG.check_head(ref, rad, r_lse, out2, lse2)["bad"])
        out3, lse3 = NG.engine_attention_fp32acc(self.q, self.keys[20:], self.values[20:], self.scale)
        self.assertTrue(NG.check_head(ref, rad, r_lse, out3, lse3)["bad"])
        # The LSE alone catches a dropped key whose value equals the output (invisible to r_o).
        self.assertGreater(abs(lse2 - ref["lse"]), r_lse)

    def test_radius_fails_closed_instead_of_clamping(self):
        big = [[v * 2.0**30 for v in row] for row in self.keys]
        ref = NG.attention_reference([v * 2.0**30 for v in self.q], big, self.values, self.scale)
        with self.assertRaises(NG.LseDomainError):
            NG.attention_radius(ref, 16)
        with self.assertRaises(NG.LseDomainError):
            NG.lse_circuit_radius({"tokens": 2**32, "rbar": 0.0, "zmax": 0.0})
        with self.assertRaises(NG.LseDomainError):
            NG.lse_circuit_radius({"tokens": 64, "rbar": 0.4, "zmax": 1.0})

    def test_additive_underflow_floor_accepts_the_1025_key_example(self):
        # specs/06 §2.2: z_1 = 0, v_1 = 0 and 1024 keys at z = -104.5 with v = 2^60.  fp32 flushes exp(-104.5),
        # the kernel returns 0, the exact output is about 2^-80.8: only the additive eta term accepts it.
        keys, values = [[0.0]] + [[-104.5]] * 1024, [[0.0]] + [[2.0**60]] * 1024
        ref = NG.attention_reference([1.0], keys, values, 1.0)
        rad = NG.attention_radius(ref, 1)
        out, lse = NG.engine_attention_fp32acc([1.0], keys, values, 1.0)
        self.assertEqual(out, [0.0])
        self.assertTrue(2.0**-82 < ref["o"][0] < 2.0**-80)
        self.assertFalse(NG.check_head(ref, rad, NG.lse_circuit_radius(rad), out, lse)["bad"])
        self.assertGreater(abs(out[0] - ref["o"][0]), 3 * NG.UB * ref["o"][0])      # a relative radius would reject

    def test_group_drivers_count_honest_cheats_and_shrunken_radii(self):
        for name in NG.GQA_MODELS:
            res = NG.dry_run_gqa(SPECS[name])
            self.assertEqual((res["honest_bad_groups"], res["domain_failures"]), (0, 0), name)
            self.assertEqual(res["groups"], len(NG.DRY_POSITIONS))
            self.assertTrue(0.0 < res["max_ratio"] < 1.0 and 0.0 < res["lse_max_ratio"] < 1.0, name)
            self.assertGreater(res["min_den_margin"], 0.99)
            self.assertEqual(res["dropped_top_key_rejected"], res["cheat_groups"])
            self.assertGreater(res["truncated_window_rejected"], 0)
            self.assertEqual(NG.operator_failures("gqa_softmax", res), [])
            tight = NG.dry_run_gqa(SPECS[name], shrink=0.05)
            self.assertGreater(tight["honest_bad_groups"], 0, name)
            self.assertTrue(NG.operator_failures("gqa_softmax", tight))


class TestSlidingWindowSinks(unittest.TestCase):
    def test_gpt_oss_sink_logit_and_retained_token_masks(self):
        spec = SPECS["gpt-oss-20b"]
        res = NG.dry_run_swa(spec)
        logit, token = res["sink_logit"], res["sink_tokens"]
        self.assertEqual(res["window"], 128)
        for b in (logit, token):
            self.assertEqual((b["honest_bad_groups"], b["domain_failures"]), (0, 0))
            self.assertEqual(b["groups"], 3)
            self.assertTrue(0.0 < b["max_ratio"] < 1.0)
            self.assertEqual(b["dropped_top_key_rejected"], b["cheat_groups"])
            self.assertEqual(b["truncated_window_rejected"], b["cheat_groups"])
        self.assertEqual(logit["dropped_sink_rejected"], logit["sink_cheat_groups"])
        self.assertEqual(logit["sink_cheat_groups"], 3)
        self.assertGreater(logit["score_checks"], 0)
        self.assertTrue(0.0 < logit["score_max_ratio"] < 1.0)
        self.assertTrue(token["keeps_sinks"])
        self.assertEqual(token["keys_at_last_position"], spec.window + spec.sink)
        self.assertEqual(NG.operator_failures("swa_sinks", res), [])
        self.assertGreater(NG.dry_run_swa(spec, shrink=0.05)["sink_logit"]["honest_bad_groups"], 0)

    def test_eager_kernel_drops_the_sink_column_and_observed_logits_are_checked(self):
        q, keys, values = rows("t/swa/q", 1, 16, 1.0)[0], rows("t/swa/k", 24, 16, 1.0), rows("t/swa/v", 24, 16, 4.0)
        scale = 0.25
        plain, shifted0, _ = NG.engine_attention_eager(q, keys, values, scale)
        sunk, shifted, lse = NG.engine_attention_eager(q, keys, values, scale, sink_logit=4.0)
        self.assertEqual((len(shifted0), len(shifted)), (24, 25))
        self.assertEqual(max(shifted), 0.0)
        self.assertLess(sum(abs(v) for v in sunk), 0.5 * sum(abs(v) for v in plain))    # the sink holds the mass
        ref = NG.attention_reference(q, keys, values, scale, logits=shifted)
        rad = NG.attention_radius(ref, 16, kernel=NG.KERNELS["eager_observed"], observed=True)
        r_lse = NG.lse_circuit_radius(rad, kernel=NG.KERNELS["eager_observed"])
        self.assertFalse(NG.check_head(ref, rad, r_lse, sunk, lse)["bad"])
        self.assertTrue(NG.check_head(ref, rad, r_lse, plain, lse)["bad"])               # sink dropped
        # Stage A: an observed logit must sit within the bf16 score radius of the exact gap.
        exact = NG.attention_reference(q, keys, values, scale, sink_logit=4.0)
        own = [NG.eager_score_radius(q, k, scale, z) for k, z in zip(keys, exact["z"])]
        for j, key in enumerate(keys):
            gap = exact["z"][j] - exact["zmax"]
            limit = NG.eager_score_radius(q, key, scale, exact["z"][j], a_max=max(own), gap=gap)
            self.assertLessEqual(abs(shifted[j] - gap), limit)
            self.assertGreater(abs(shifted[j] + 1.0 - gap), limit)                       # tampered logit

    def test_sequential_normalizer_widens_the_lse_radius(self):
        rad = {"tokens": 129, "rbar": 0.0, "zmax": 0.0}
        tiled = NG.lse_circuit_radius(rad)
        eager = NG.lse_circuit_radius(rad, kernel=NG.KERNELS["eager_observed"])
        self.assertAlmostEqual(eager - tiled, 129 * NG.U32, delta=1e-12)


class TestGatedDeltaNet(unittest.TestCase):
    def setUp(self):
        st = NG.Stream("t/lrec")
        self.tokens = 24
        self.q = NG._bf16_rows(st, self.tokens, 8, 1.0)
        self.k = NG._bf16_rows(st, self.tokens, 8, 1.0)
        self.v = NG._bf16_rows(st, self.tokens, 4, 2.0)
        self.g = [NG.f32(-0.02 - 0.2 * float(st.unit())) for _ in range(self.tokens)]
        self.b = [NG.f32(0.2 + 0.6 * float(st.unit())) for _ in range(self.tokens)]

    def test_reference_is_the_delta_rule_and_chunks_compose(self):
        ref = NG.lrec_reference(self.q, self.k, self.v, self.g, self.b)
        # One step by hand from the zero state: S = k (beta v)^T, o = (k . q / sqrt(d)) beta v.
        k0, q0 = NG.l2_unit(self.k[0]), NG.l2_unit(self.q[0])
        proj = sum(a * c for a, c in zip(k0, q0)) / math.sqrt(8)
        for j in range(4):
            self.assertAlmostEqual(ref["outputs"][0][j], proj * self.b[0] * self.v[0][j], places=12)
        cut = 10
        head = NG.lrec_reference(self.q[:cut], self.k[:cut], self.v[:cut], self.g[:cut], self.b[:cut])
        tail = NG.lrec_reference(self.q[cut:], self.k[cut:], self.v[cut:], self.g[cut:], self.b[cut:],
                                 state=head["state"])
        for a, c in zip(head["outputs"] + tail["outputs"], ref["outputs"]):
            for x, y in zip(a, c):
                self.assertAlmostEqual(x, y, places=12)
        for row in ref["radii"]:
            self.assertTrue(all(r > 0.0 for r in row))

    def test_honest_inside_cheats_outside(self):
        ref = NG.lrec_reference(self.q, self.k, self.v, self.g, self.b)
        honest = NG.lrec_check(ref, NG.engine_lrec(self.q, self.k, self.v, self.g, self.b))
        self.assertEqual(honest["bad_tokens"], 0)
        self.assertTrue(0.0 < honest["max_ratio"] < 1.0)
        no_decay = NG.lrec_check(ref, NG.engine_lrec(self.q, self.k, self.v, self.g, self.b, decay_cheat=True))
        self.assertGreater(no_decay["bad_tokens"], 0)
        reset = NG.lrec_check(ref, NG.engine_lrec(self.q, self.k, self.v, self.g, self.b, reset_at=12))
        self.assertGreater(reset["bad_tokens"], 0)
        prefix = NG.lrec_check({"outputs": ref["outputs"][:12], "radii": ref["radii"][:12]},
                               NG.engine_lrec(self.q, self.k, self.v, self.g, self.b, reset_at=12)[:12])
        self.assertEqual(prefix["bad_tokens"], 0)                    # the reset only shows after the hand-off

    def test_domain_is_enforced(self):
        with self.assertRaises(ValueError):
            NG.lrec_reference(self.q[:1], self.k[:1], self.v[:1], [0.5], self.b[:1])
        with self.assertRaises(ValueError):
            NG.lrec_reference(self.q[:1], self.k[:1], self.v[:1], self.g[:1], [1.5])

    def test_driver(self):
        res = NG.dry_run_lrec(SPECS["Qwen3.5-0.8B"])
        self.assertEqual(res["enclosure"], "LREC_SEQ_ENCLOSURE_V0")
        self.assertEqual(res["honest_bad_tokens"], 0)
        self.assertEqual(res["tokens"], res["heads"] * NG.DRY_LREC["tokens"])
        self.assertTrue(0.0 < res["max_ratio"] < 1.0)
        self.assertEqual((res["zero_decay_rejected_heads"], res["state_reset_rejected_heads"]),
                         (res["heads"], res["heads"]))
        self.assertEqual(NG.operator_failures("lrec", res), [])
        self.assertGreater(NG.dry_run_lrec(SPECS["Qwen3.5-0.8B"], shrink=0.01)["honest_bad_tokens"], 0)


class TestRmsNorm(unittest.TestCase):
    def test_endpoints_enclose_the_exact_inverse_root(self):
        for squares, width in ((Fraction(48), 48), (Fraction(3, 10**5), 16), (Fraction(10**9), 2560)):
            eps = Fraction(NG.f32(1e-6))
            lo, hi = NG.rms_endpoints(squares, width, eps)
            radicand = squares / width + eps
            self.assertLessEqual(lo * lo * radicand, 1)
            self.assertGreaterEqual(hi * hi * radicand, 1)
            self.assertLess(hi / lo - 1, Fraction(1, 2**14))         # accumulation bound plus rsqrt cells

    def test_templates_match_the_five_checkpoints(self):
        self.assertEqual(NG.RMS_TEMPLATES["LlamaRMSNorm"].stores, 2)
        self.assertEqual(NG.RMS_TEMPLATES["Qwen3RMSNorm"].stores, 2)
        self.assertEqual(NG.RMS_TEMPLATES["Qwen3MoeRMSNorm"].stores, 2)
        self.assertEqual(NG.RMS_TEMPLATES["GptOssRMSNorm"].stores, 1)
        self.assertTrue(NG.RMS_TEMPLATES["Qwen3_5RMSNorm"].one_plus)
        self.assertNotIn("Qwen3_5RMSNormGated", NG.RMS_TEMPLATES)
        self.assertLess(NG.rms_relative(1), NG.rms_relative(2))
        self.assertAlmostEqual(NG.rms_relative(1), 2.0**-8, delta=2.0**-18)

    def test_all_models_honest_floor_and_cheats(self):
        for name, spec in SPECS.items():
            res = NG.dry_run_rmsnorm(spec)
            self.assertEqual(res["honest_bad_rows"], 0, name)
            self.assertTrue(0.0 < res["max_ratio"] < 1.0, name)
            self.assertTrue(res["floor_holds"] and res["min_rms"] >= 1e-3, name)
            self.assertGreater(res["max_outlier_ratio"], 4.0, name)
            self.assertEqual((res["wrong_gain_rejected"], res["skipped_norm_rejected"]), (res["rows"], res["rows"]))
            self.assertEqual(NG.operator_failures("rmsnorm", res), [])
            self.assertGreater(NG.dry_run_rmsnorm(spec, shrink=0.2)["honest_bad_rows"], 0, name)

    def test_two_store_kernel_needs_the_two_store_template(self):
        two, one = NG.RMS_TEMPLATES["LlamaRMSNorm"], NG.RMS_TEMPLATES["GptOssRMSNorm"]
        st = NG.Stream("t/rms/two-store")
        weight = [NG.bf16(1.0 + st.signed(0.5)) for _ in range(48)]
        escaped = 0
        for _ in range(12):
            x = [NG.bf16(st.signed(1.0)) for _ in range(48)]
            out, gain = NG.engine_rmsnorm(x, weight, epsilon=1e-6, template=two)
            self.assertFalse(NG.rms_check_row(x, gain, out, epsilon=1e-6, template=two)["bad"])
            escaped += NG.rms_check_row(x, gain, out, epsilon=1e-6, template=one)["bad"]
        self.assertGreater(escaped, 0)        # the single-store enclosure rejects an honest two-store kernel

    def test_residual_floor_is_a_domain_condition(self):
        template = NG.RMS_TEMPLATES["GptOssRMSNorm"]
        x = [NG.bf16(2.0**-12)] * 16                                  # rms 2.4e-4 < 1e-3
        out, gain = NG.engine_rmsnorm(x, [1.0] * 16, epsilon=1e-6, template=template)
        residual = NG.rms_check_row(x, gain, out, epsilon=1e-6, template=template)
        self.assertFalse(residual["floor_ok"])
        self.assertTrue(residual["bad"])
        self.assertLessEqual(residual["ratio"], 1.0)                  # numerically fine, outside the domain
        self.assertFalse(NG.rms_check_row(x, gain, out, epsilon=1e-6, template=template, residual=False)["bad"])


class TestDriver(unittest.TestCase):
    def test_dry_run_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(NG.main(["--dry-run", "--out", path]), 0)
            with open(path, encoding="utf-8") as fh:
                report = json.load(fh)
        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "dry_run")
        self.assertTrue(report["evidence"].startswith("DRY_RUN_SYNTHETIC"))
        self.assertEqual([m["model"] for m in report["models"]], [m.name for m in NG.MODELS])
        for model in report["models"]:
            self.assertEqual(model["status"], "DRY_RUN_OK")
            self.assertEqual(model["evidence"], "DRY_RUN_SYNTHETIC")
            self.assertEqual(tuple(model["operators"]), NG.operators_for(model["model"]))
        self.assertEqual(report["constants"]["circuit_norm_lift"], "2^-32")
        self.assertIn("not calibration evidence", buf.getvalue())
        self.assertEqual(buf.getvalue().count("DRY_RUN_OK"), 5)

    def test_real_mode_needs_confirmation_and_subsets_work(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(NG.main(["--models", "Qwen3-4B"]), 2)
        self.assertIn("--confirm-gpu", err.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(NG.main(["--dry-run", "--models", "gpt-oss-20b,Qwen3.5-0.8B"]), 0)
        self.assertEqual(buf.getvalue().count("DRY_RUN_OK"), 2)

    def test_failures_are_reported_not_hidden(self):
        bucket = {**NG._attn_bucket(), "groups": 4, "cheat_groups": 4, "dropped_top_key_rejected": 4,
                  "truncated_window_rejected": 1}
        self.assertEqual(NG.attention_failures("gqa", bucket), [])
        self.assertTrue(NG.attention_failures("gqa", {**bucket, "honest_bad_groups": 1}))
        self.assertTrue(NG.attention_failures("gqa", {**bucket, "domain_failures": 2}))
        # Cheats are gated at sigma = 3/10 of the units tried (in-contract no-ops are tolerated, and reported).
        self.assertEqual(NG.SIGMA, 0.3)
        self.assertEqual(NG.attention_failures("gqa", {**bucket, "dropped_top_key_rejected": 3}), [])
        self.assertTrue(NG.attention_failures("gqa", {**bucket, "dropped_top_key_rejected": 1}))
        self.assertTrue(NG.attention_failures("gqa", {**bucket, "truncated_window_rejected": 0}))
        self.assertEqual(NG.cheat_gate("x", "c", 0, 0), [])
        lrec = {**NG.dry_run_lrec(SPECS["Qwen3.5-0.8B"]), "heads": 288}
        self.assertEqual(NG.operator_failures("lrec", {**lrec, "zero_decay_rejected_heads": 287,
                                                       "state_reset_rejected_heads": 288}), [])
        self.assertTrue(NG.operator_failures("lrec", {**lrec, "zero_decay_rejected_heads": 80,
                                                      "state_reset_rejected_heads": 288}))
        self.assertTrue(NG.attention_failures("gqa", {**bucket, "groups": 0}))
        self.assertTrue(NG.operator_failures("lrec", {"status": "UNOBSERVED", "reason": "x"}))
        model = NG.dry_run_model(SPECS["Qwen3-4B"], shrink=0.05)
        self.assertTrue(model["status"].startswith("FAILED: "))


def _lrec_case(tag, tokens, key_dim=8, value_dim=3, *, spread=None, beta=(0.2, 0.6), decay=(0.02, 0.25)):
    """Synthetic bf16 operands of one head; ``spread`` makes every key a small perturbation of one direction."""
    st = NG.Stream(tag)
    q, k = NG._bf16_rows(st, tokens, key_dim, 1.0), NG._bf16_rows(st, tokens, key_dim, 1.0)
    v = NG._bf16_rows(st, tokens, value_dim, 2.0)
    if spread is not None:
        k = [[NG.bf16(a + spread * float(st.signed(1.0))) for a in k[0]] for _ in range(tokens)]
    g = [NG.f32(-decay[0] - decay[1] * float(st.unit())) for _ in range(tokens)]
    b = [NG.f32(beta[0] + beta[1] * float(st.unit())) for _ in range(tokens)]
    return q, k, v, g, b


class TestChunkedWY(unittest.TestCase):
    """``LREC_CHUNK_WY_ENCLOSURE_V1``: the chunk-64 WY kernel of ``torch_chunk_gated_delta_rule`` at T = 512."""

    @classmethod
    def setUpClass(cls):
        cls.case = _lrec_case("t/wy/generic", 512)
        cls.ref = NG.lrec_chunk_reference(*cls.case)
        cls.outs, cls.state = NG.engine_lrec_chunk(*cls.case)

    def test_kernel_table_and_constants(self):
        self.assertEqual(NG.LREC_KERNEL_ENCLOSURE, {"torch_chunk_gated_delta_rule": "LREC_CHUNK_WY_ENCLOSURE_V1",
                                                    "torch_recurrent_gated_delta_rule": "LREC_SEQ_ENCLOSURE_V0"})
        self.assertEqual(NG.LREC_CHUNK, 64)
        # Any-order gamma: 64 terms cost 64 u (the contract's lane depth would be 68, and 72 instead of 128).
        self.assertAlmostEqual(NG.gamma_any(64), 64 * NG.U32, delta=1e-10)
        self.assertGreater(NG.gamma_any(128), NG.gamma(128))
        c = NG.lrec_constants(128, any_order=True)
        # The scalar part of the l2norm error is the square root of a gamma_{n+1} perturbation: about half of it.
        self.assertAlmostEqual(c["c_scale"], NG.gamma_any(129) / 2 + NG.UE, delta=1e-9)
        self.assertLess(c["c_scale"], c["c_key"])
        self.assertEqual(NG.lrec_constants(128)["c_key"], (1 + NG.gamma(128)) * (1 + NG.UE) * (1 + NG.U32) - 1)

    def test_chunked_reference_is_the_sequential_recurrence(self):
        """Exact arithmetic: the WY scan and the token recurrence agree, with padding and with a carried state."""
        q, k, v, g, b = _lrec_case("t/wy/pad", 100)
        seq = NG.lrec_reference(q, k, v, g, b)
        for chunk in (16, 64):                                   # 100 = 6 * 16 + 4 = 64 + 36: both pad
            ref = NG.lrec_chunk_reference(q, k, v, g, b, chunk=chunk)
            self.assertEqual((len(ref["outputs"]), ref["chunks"]), (100, -(-100 // chunk)))
            for a, c in zip(ref["outputs"], seq["outputs"]):
                for x, y in zip(a, c):
                    self.assertAlmostEqual(x, y, places=12)
            for a, c in zip(ref["state"], seq["state"]):
                for x, y in zip(a, c):
                    self.assertAlmostEqual(x, y, places=12)
        head = NG.lrec_chunk_reference(q[:64], k[:64], v[:64], g[:64], b[:64], chunk=16)
        tail = NG.lrec_chunk_reference(q[64:], k[64:], v[64:], g[64:], b[64:], state=head["state"], chunk=16)
        for a, c in zip(tail["outputs"], seq["outputs"][64:]):
            for x, y in zip(a, c):
                self.assertAlmostEqual(x, y, places=12)

    def test_t512_eight_chunks_honest_inside(self):
        self.assertEqual((self.ref["chunks"], len(self.ref["outputs"])), (8, 512))
        honest, held = NG.lrec_check(self.ref, self.outs), NG.lrec_state_check(self.ref, self.state)
        self.assertEqual((honest["bad_tokens"], held["bad_columns"]), (0, 0))
        self.assertTrue(0.0 < honest["max_ratio"] < 1.0)
        self.assertTrue(0.0 < held["max_ratio"] < 1.0)
        self.assertTrue(all(r > 0.0 for row in self.ref["radii"] for r in row))
        # Negative: the same kernel output against shrunken radii, and a state off by one fp32 ulp-scale step.
        tight = {**self.ref, "radii": [[r * 0.5 for r in row] for row in self.ref["radii"]],
                 "state_radius": [r * held["max_ratio"] * 0.5 for r in self.ref["state_radius"]]}
        self.assertGreater(NG.lrec_check(tight, self.outs)["bad_tokens"], 0)
        self.assertGreater(NG.lrec_state_check(tight, self.state)["bad_columns"], 0)
        moved = [[x + 4.0 * max(self.ref["state_radius"]) for x in row] for row in self.state]
        self.assertEqual(NG.lrec_state_check(self.ref, moved)["bad_columns"], 3)

    def test_radius_is_not_vacuous(self):
        """The state radius stays a small fraction of the state it encloses, and of the same order as V0's."""
        seq = NG.lrec_reference(*self.case)
        for j, rho in enumerate(self.ref["state_radius"]):
            sigma = NG._norm2([row[j] for row in self.ref["state"]])
            self.assertLess(rho, 1e-3 * sigma)
            self.assertLess(rho, 20.0 * seq["state_radius"][j])
            self.assertGreater(rho, 0.0)

    def test_wy_conditioning_uses_the_true_inverse(self):
        """Near-duplicate keys, beta ~ 1, g ~ 0: the comparison-matrix bound ``(I - |N|)^-1`` is astronomically
        large while ``|A^-1|`` is not, and the enclosure stays finite, honest-inside and able to reject."""
        case = _lrec_case("t/wy/dup", 512, spread=0.02, beta=(0.95, 0.05), decay=(0.0, 0.002))
        q, k, v, g, b = case
        ref, generic = NG.lrec_chunk_reference(*case), self.ref
        unit = [NG.l2_unit(row) for row in k[:64]]
        cum = [math.fsum(g[:i + 1]) for i in range(64)]
        comparison = [[-abs(b[i] * sum(x * y for x, y in zip(unit[i], unit[j])) * math.exp(cum[i] - cum[j]))
                       if j < i else 0.0 for j in range(64)] for i in range(64)]
        eye = [[1.0 if i == j else 0.0 for j in range(64)] for i in range(64)]
        blown = max(sum(row) for row in NG._unit_lower_solve(comparison, eye))
        self.assertGreater(blown, 1e15)
        self.assertLess(ref["inv_inf_norm"], 64.0)
        self.assertTrue(all(math.isfinite(r) for r in ref["state_radius"]))
        outs, state = NG.engine_lrec_chunk(*case)
        honest, held = NG.lrec_check(ref, outs), NG.lrec_state_check(ref, state)
        self.assertEqual((honest["bad_tokens"], held["bad_columns"]), (0, 0))
        self.assertTrue(0.0 < held["max_ratio"] < 1.0)
        # The correlated chunk costs more relative to its state than the generic one, and more than V0 allows.
        seq = NG.lrec_reference(*case)
        rel = lambda r: max(rho / NG._norm2([row[j] for row in r["state"]])                 # noqa: E731
                            for j, rho in enumerate(r["state_radius"]))
        self.assertGreater(rel(ref), 10.0 * rel(generic))
        self.assertGreater(min(a / c for a, c in zip(ref["state_radius"], seq["state_radius"])), 5.0)
        self.assertLess(rel(ref), 0.5)
        for kwargs in ({"decay_cheat": True}, {"reset_at": 256}):
            cheat_out, cheat_state = NG.engine_lrec_chunk(*case, **kwargs)
            self.assertGreater(NG.lrec_check(ref, cheat_out)["bad_tokens"], 0)
            self.assertGreater(NG.lrec_state_check(ref, cheat_state)["bad_columns"], 0)

    def test_cheats_outside(self):
        no_decay, no_decay_state = NG.engine_lrec_chunk(*self.case, decay_cheat=True)
        self.assertGreater(NG.lrec_check(self.ref, no_decay)["bad_tokens"], 400)
        self.assertEqual(NG.lrec_state_check(self.ref, no_decay_state)["bad_columns"], 3)
        reset, _ = NG.engine_lrec_chunk(*self.case, reset_at=256)
        self.assertGreater(NG.lrec_check(self.ref, reset)["bad_tokens"], 0)
        prefix = {"outputs": self.ref["outputs"][:256], "radii": self.ref["radii"][:256]}
        self.assertEqual(NG.lrec_check(prefix, reset[:256])["bad_tokens"], 0)    # only after the hand-off
        # Slow decay keeps the dropped state visible in the final state too.
        slow = _lrec_case("t/wy/slow", 512, spread=0.02, beta=(0.95, 0.05), decay=(0.0, 0.002))
        ref = NG.lrec_chunk_reference(*slow)
        self.assertEqual(NG.lrec_state_check(ref, NG.engine_lrec_chunk(*slow, reset_at=448)[1])["bad_columns"], 3)

    def test_sequential_kernel_is_not_the_chunked_template(self):
        """The two enclosures are different radii for different kernels; neither is a multiple of the other."""
        seq = NG.lrec_reference(*self.case)
        ratios = [a / c for a, c in zip(self.ref["state_radius"], seq["state_radius"])]
        self.assertGreater(max(ratios) / min(ratios), 1.0)
        self.assertNotAlmostEqual(self.ref["radii"][300][0] / seq["radii"][300][0],
                                  self.ref["radii"][500][1] / seq["radii"][500][1], places=6)

    def test_domain_is_enforced(self):
        q, k, v, g, b = _lrec_case("t/wy/domain", 4)
        with self.assertRaises(ValueError):
            NG.lrec_chunk_reference(q, k, v, [0.5] + g[1:], b, chunk=4)
        with self.assertRaises(ValueError):
            NG.lrec_chunk_reference(q, k, v, g, [1.5] + b[1:], chunk=4)
        with self.assertRaises(ValueError):
            NG.engine_lrec_chunk(q, k, v, g, b, chunk=4, reset_at=2)

    def test_driver_and_gate(self):
        res = NG.dry_run_lrec(SPECS["Qwen3.5-0.8B"])
        wy = res["chunk_wy"]
        self.assertEqual((res["enclosure"], wy["enclosure"]), ("LREC_SEQ_ENCLOSURE_V0", "LREC_CHUNK_WY_ENCLOSURE_V1"))
        self.assertEqual((wy["honest_bad_tokens"], wy["state_bad"]), (0, 0))
        self.assertEqual(wy["tokens"], wy["heads"] * NG.DRY_LREC["tokens"])
        self.assertTrue(0.0 < wy["max_ratio"] < 1.0 and 0.0 < wy["state_max_ratio"] < 1.0)
        self.assertEqual((wy["zero_decay_rejected_heads"], wy["state_reset_rejected_heads"]),
                         (wy["heads"], wy["heads"]))
        self.assertEqual(NG.operator_failures("lrec", res), [])
        for tamper in ({"honest_bad_tokens": 1}, {"state_bad": 1}, {"zero_decay_rejected_heads": 0},
                       {"state_reset_rejected_heads": 0}):
            self.assertTrue(NG.operator_failures("lrec", {**res, "chunk_wy": {**wy, **tamper}}), tamper)
        shrunk = NG.dry_run_lrec(SPECS["Qwen3.5-0.8B"], shrink=0.01)["chunk_wy"]
        self.assertGreater(shrunk["honest_bad_tokens"], 0)
        # The real-mode receipt carries the measured chunk_wy block without the dry run's verdict fields.
        self.assertEqual(NG.operator_failures("lrec", {**res, "chunk_wy": {"calls": 18, "chunk_size": 64}}), [])


@unittest.skipUnless(HAVE_TORCH, "torch is not installed (the real-mode kernels are cross-checked where it is)")
class TestTorchKernelsAgree(unittest.TestCase):
    def test_attention_radius(self):
        q, keys, values = rows("t/torch/q", 3, 16, 2.0), rows("t/torch/k", 70, 16, 1.0), rows("t/torch/v", 70, 16, 4.0)
        scale = 0.25
        refs = [NG.attention_reference(h, keys, values, scale, sink_logit=0.5) for h in q]
        tq, tk = torch.tensor(q, dtype=torch.float64), torch.tensor(keys, dtype=torch.float64)
        tv = torch.tensor(refs[0]["values"], dtype=torch.float64)
        z = torch.cat([scale * tq @ tk.T, torch.full((3, 1), 0.5, dtype=torch.float64)], -1)
        mass = torch.cat([scale * tq.abs() @ tk.abs().T, torch.zeros(3, 1, dtype=torch.float64)], -1)
        p = torch.softmax(z, -1)
        res = NG.t_attention_radius(torch, z, mass, p, p @ tv, tv, 16, kernel=NG.KERNELS["fp32acc"], observed=False)
        for h, ref in enumerate(refs):
            rad = NG.attention_radius(ref, 16)
            for a, b in zip(res[0][h].tolist(), rad["r_o"]):
                self.assertAlmostEqual(a, b, delta=b * 1e-9)
            self.assertAlmostEqual(float(res[1][h]), rad["rbar"], delta=rad["rbar"] * 1e-9)
            self.assertAlmostEqual(float(res[2][h]), rad["e_den"], delta=rad["e_den"] * 1e-9)

    def test_lrec(self):
        st = NG.Stream("t/torch/lrec")
        q, k, v = (NG._bf16_rows(st, 20, d, s) for d, s in ((8, 1.0), (8, 1.0), (4, 2.0)))
        g = [NG.f32(-0.02 - 0.2 * float(st.unit())) for _ in range(20)]
        b = [NG.f32(0.2 + 0.6 * float(st.unit())) for _ in range(20)]
        ref = NG.lrec_reference(q, k, v, g, b)
        as_t = lambda x: torch.tensor(x, dtype=torch.float64)[:, None]       # noqa: E731  (one head)
        out, rad, state, rho = NG.t_lrec(torch, as_t(q), as_t(k), as_t(v),
                                         torch.tensor(g, dtype=torch.float64)[:, None],
                                         torch.tensor(b, dtype=torch.float64)[:, None], None)
        for t in range(20):
            for j in range(4):
                self.assertAlmostEqual(float(out[t, 0, j]), ref["outputs"][t][j], places=11)
                self.assertAlmostEqual(float(rad[t, 0, j]), ref["radii"][t][j], delta=ref["radii"][t][j] * 1e-9)
        for j in range(4):
            self.assertAlmostEqual(float(rho[0, j]), ref["state_radius"][j], delta=ref["state_radius"][j] * 1e-9)

    def test_lrec_chunk_wy(self):
        """The vectorised enclosure is the pure-Python one (padding included), and the fp32 kernel replica stays
        inside it over eight real chunks; the cheats do not."""
        q, k, v, g, b = _lrec_case("t/torch/wy", 100)
        ref = NG.lrec_chunk_reference(q, k, v, g, b, chunk=16)
        as_t = lambda x: torch.tensor(x, dtype=torch.float64)[:, None]       # noqa: E731  (one head)
        col = lambda x: torch.tensor(x, dtype=torch.float64)[:, None]        # noqa: E731
        out, rad, state, rho, info = NG.t_lrec_chunk(torch, as_t(q), as_t(k), as_t(v), col(g), col(b), None, chunk=16)
        self.assertEqual((tuple(out.shape), info["chunks"]), ((100, 1, 3), 7))
        self.assertAlmostEqual(info["inv_inf_norm"], ref["inv_inf_norm"], delta=1e-9)
        for t in range(100):
            for j in range(3):
                self.assertAlmostEqual(float(out[t, 0, j]), ref["outputs"][t][j], places=11)
                self.assertAlmostEqual(float(rad[t, 0, j]), ref["radii"][t][j], delta=ref["radii"][t][j] * 1e-8)
        for j in range(3):
            self.assertAlmostEqual(float(rho[0, j]), ref["state_radius"][j], delta=ref["state_radius"][j] * 1e-8)

    def test_chunk_kernel_replica_inside_v1(self):
        heads, tokens, key_dim, value_dim = 3, 512, 16, 8
        gen = torch.Generator().manual_seed(8)
        mk = lambda *shape: torch.randn(*shape, generator=gen)                                   # noqa: E731
        q, k, v = (mk(tokens, heads, d).to(torch.bfloat16) for d in (key_dim, key_dim, value_dim))
        k[:, 1] = (k[:1, 1].float() + 0.02 * mk(tokens, key_dim)).to(torch.bfloat16)             # correlated head
        g = (-0.3 * torch.rand(tokens, heads, generator=gen)).to(torch.bfloat16)
        g[:, 1] = g[:, 1] * 0.01
        b = torch.rand(tokens, heads, generator=gen).to(torch.bfloat16)
        dbl = [x.double() for x in (q, k, v, g, b)]
        out, rad, state, rho, info = NG.t_lrec_chunk(torch, *dbl, None)
        self.assertEqual(info["chunks"], 8)
        seq_out, _, seq_state, _ = NG.t_lrec(torch, *dbl, None)
        self.assertLess(float((out - seq_out).abs().max()), 1e-11)
        self.assertLess(float((state - seq_state).abs().max()), 1e-11)
        got, last = NG.t_chunk_kernel(torch, q, k, v, g, b)
        ratio = ((got.double() - out).abs() / rad).max()
        held = ((last.double() - state).norm(dim=1) / rho).max()
        self.assertTrue(0.0 < float(ratio) < 1.0 and 0.0 < float(held) < 1.0, (float(ratio), float(held)))
        self.assertGreater(float(((got.double() - out).abs() / (0.5 * rad)).max()), 1.0)         # shrunken radius
        for kwargs in ({"decay_cheat": True}, {"reset_at": 256}):
            cheat, _, cheat_state, _ = NG.t_lrec(torch, *dbl, None, radii=False, **kwargs)
            rejected = ((cheat.to(torch.bfloat16).double() - out).abs() > rad).any(-1).any(0)
            self.assertEqual(int(rejected.sum()), heads, kwargs)
        # The explicit-inverse (export) branch is a different template: it is measured here, not enclosed.
        other, other_state = NG.t_chunk_kernel(torch, q, k, v, g, b, explicit_inverse=True)
        self.assertTrue(math.isfinite(float(((other_state.double() - state).norm(dim=1) / rho).max())))
        self.assertLess(float((other.double() - got.double()).abs().max()), 1e-2)

    def test_replica_is_the_transformers_kernel(self):
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as HF
        except Exception:                                                                        # noqa: BLE001
            self.skipTest("transformers without the Qwen3.5 Gated DeltaNet kernel")
        gen = torch.Generator().manual_seed(9)
        q, k, v = (torch.randn(150, 2, d, generator=gen).to(torch.bfloat16) for d in (16, 16, 8))
        g = (-0.3 * torch.rand(150, 2, generator=gen)).to(torch.bfloat16)
        b = torch.rand(150, 2, generator=gen).to(torch.bfloat16)
        want, last = HF.torch_chunk_gated_delta_rule(q[None], k[None], v[None], g=g[None], beta=b[None],
                                                     output_final_state=True, use_qk_l2norm_in_kernel=True)
        got, state = NG.t_chunk_kernel(torch, q, k, v, g, b)
        self.assertTrue(torch.equal(got, want[0]))
        self.assertTrue(torch.equal(state, last[0]))


if __name__ == "__main__":
    unittest.main()
