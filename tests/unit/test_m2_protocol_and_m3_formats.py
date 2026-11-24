# SPDX-License-Identifier: Apache-2.0
"""Protocol v2 engine (M2-1), production FP8 / MXFP4 contracts (M3-1) and the M2-3 container property."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from math import log2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("FPN_NO_NATIVE", "1")
for _p in (ROOT, os.path.join(ROOT, "prover"), os.path.join(ROOT, "verifier", "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fpn_prover.circuit import protocol as PR  # noqa: E402
from fpn_prover.circuit.blocks import LINEAGE_DEPTH, RESIDUAL_RMS_FLOOR  # noqa: E402
from fpn_prover.ir import format as FM  # noqa: E402
from fpn_prover.ir.format import (  # noqa: E402
    CONTRACTS, KERNEL_ADAPTERS, MODEL_FORMATS, NUM_FORMATS, Calibration, ToleranceClass, UncalibratedContractError,
    exact_dequant_scale_exp_range, gemm_error_bound, weight_commitment,
)

BENCH = os.path.join(ROOT, "bench", "tier2", "bench_frontier_calibrate.py")
SEED = bytes(range(32))
DENSE = PR.SamplerProfile(4, ((64, 96), (64, 40), (96, 64)), q_heads=8, kv_groups=2, head_dim=16, attn_gemms=2)


class TestChallengeSource(unittest.TestCase):
    def test_modes_bind_every_input_and_never_collide(self):
        base = PR.challenge_seed(PR.MODE_EXTERNAL_SEED, b"x", b"r", b"pre", b"c")
        self.assertEqual(base, PR.external_seed(b"x", b"r", b"pre", b"c"))
        for other in ((b"y", b"r", b"pre", b"c"), (b"x", b"s", b"pre", b"c"), (b"x", b"r", b"prf", b"c"),
                      (b"x", b"r", b"pre", b"d"), (b"xr", b"r", b"pre", b"c")):
            self.assertNotEqual(base, PR.external_seed(*other))
        # Length framing: moving a byte across a field boundary changes the seed.
        self.assertNotEqual(PR.external_seed(b"ab", b"c", b"p", b"c"), PR.external_seed(b"a", b"bc", b"p", b"c"))
        ni = PR.challenge_seed(PR.MODE_NI, b"x", b"r", b"pre")
        self.assertNotEqual(ni, base)
        self.assertNotEqual(ni, PR.ni_seed(b"x", b"r", b"other"))
        self.assertEqual(len(ni), 32)

    def test_mode_misuse_is_rejected(self):
        with self.assertRaises(ValueError):
            PR.challenge_seed(PR.MODE_EXTERNAL_SEED, b"x", b"r", b"pre")
        with self.assertRaises(ValueError):
            PR.challenge_seed(PR.MODE_NI, b"x", b"r", b"pre", b"c")
        with self.assertRaises(ValueError):
            PR.challenge_seed("beacon", b"x", b"r", b"pre", b"c")
        with self.assertRaises(ValueError):
            PR.external_seed(b"x", b"", b"pre", b"c")
        with self.assertRaises(ValueError):
            PR.ni_seed(b"x", "r", b"pre")

    def test_retry_cannot_reroll_draws(self):
        ledger = PR.PreCommitLedger()
        seed = ledger.seed(b"x", b"r", b"pre", b"c")
        self.assertEqual(ledger.seed(b"x", b"r", b"pre", b"c"), seed)          # retransmission
        with self.assertRaises(PR.RetryError):
            ledger.commit(b"x", b"r", b"pre2")                                  # fresh C_pre for the same response
        with self.assertRaises(PR.RetryError):
            ledger.seed(b"x", b"r", b"pre", b"c2")                              # second external challenge
        self.assertNotEqual(ledger.seed(b"x", b"r2", b"pre2", b"c"), seed)      # a different response is free


class TestSoundnessBits(unittest.TestCase):
    # specs/02-soundness.md §6, machine-checked in Lean (LifetimeSoundness.table_Q30_e20 .. table_Q64_e40).
    TABLE = {Fraction(1, 2): (51, 71, 85, 105), Fraction(3, 10): (100, 138, 166, 205),
             Fraction(1, 5): (159, 221, 265, 327), Fraction(1, 10): (336, 468, 560, 691)}

    def test_draw_table_is_reproduced_exactly(self):
        for s, row in self.TABLE.items():
            got = tuple(PR.draws_required(s, q, e) for q, e in ((30, -20), (30, -40), (64, -20), (64, -40)))
            self.assertEqual(got, row, s)

    def test_least_sufficient_draw_counts(self):
        for s, q, e, k in ((Fraction(3, 10), 30, -20, 100), (Fraction(1, 2), 30, -20, 51), (Fraction(1, 2), 64, -40, 105)):
            self.assertTrue(PR.sufficient(s, k, q, e))
            self.assertFalse(PR.sufficient(s, k - 1, q, e))
        self.assertEqual(PR.draws_required(1, 64, -40), 1)
        for bad in ((0, 30, -20), (Fraction(3, 2), 30, -20), (Fraction(1, 2), -1, -20), (Fraction(1, 2), 30, 0)):
            with self.assertRaises(ValueError):
                PR.draws_required(*bad)

    def test_regime_bits_are_reported_without_inflation(self):
        reports = PR.assert_regimes()
        tight, light, ni = (reports[n] for n in ("external_tight", "external_light", "ni"))
        self.assertEqual((tight.draws, light.draws, ni.draws), (100, 51, 105))
        # Per-attempt spot-check miss: 0.7^100 < 2^-51, 0.5^51 = 2^-51, 0.5^105 = 2^-105.
        self.assertEqual((tight.per_attempt_bits_floor, light.per_attempt_bits_floor, ni.per_attempt_bits_floor),
                         (51, 51, 105))
        self.assertAlmostEqual(tight.per_attempt_bits, 100 * log2(10 / 7), places=9)
        # Lifetime bits over the attempt budget: the external-seed regimes target 2^-20, NI targets 2^-40.
        self.assertEqual((tight.lifetime_bits_floor, light.lifetime_bits_floor, ni.lifetime_bits_floor), (20, 20, 40))
        self.assertTrue(all(r.meets_target for r in reports.values()))
        # Only the NI regime reaches 100 bits per attempt; no regime reaches 100 lifetime bits.
        self.assertEqual([r.per_attempt_at_least_100 for r in (tight, light, ni)], [False, False, True])
        self.assertTrue(all(r.lifetime_bits_floor < 100 for r in reports.values()))

    def test_reduced_ni_budget(self):
        r = PR.soundness_report("ni", q_log2=PR.Q_NI_REDUCED_LOG2)
        self.assertEqual((r.draws_required, r.lifetime_spot_bits_floor, r.lifetime_bits_floor), (91, 55, 54))
        self.assertTrue(r.meets_target)

    def test_miss_probability_and_hypergeometric(self):
        self.assertEqual(PR.miss_probability(Fraction(1, 2), 51), Fraction(1, 2**51))
        self.assertEqual(PR.miss_bits_floor(Fraction(1, 2), 105), 105)
        self.assertEqual(PR.miss_bits_floor(Fraction(3, 10), 100), 51)
        self.assertEqual(PR.hypergeometric_miss(10, 3, 2), Fraction(7 * 6, 10 * 9))
        self.assertEqual(PR.hypergeometric_miss(10, 3, 8), 0)
        self.assertEqual(PR.hypergeometric_miss(10, 0, 10), 1)
        for units, bad, k in ((1000, 300, 100), (97, 49, 51), (64, 1, 20)):
            self.assertLessEqual(PR.hypergeometric_miss(units, bad, k), PR.miss_probability(Fraction(bad, units), k))
        with self.assertRaises(ValueError):
            PR.hypergeometric_miss(10, 11, 1)

    def test_cost_weighting_matters(self):
        # One expensive bad atom holding 30% of the cost is a 30% target, although it is 1 atom out of 71.
        costs = [30] + [1] * 70
        self.assertEqual(Fraction(costs[0], sum(costs)), Fraction(3, 10))
        self.assertLess(PR.miss_probability(Fraction(3, 10), 100), PR.miss_probability(Fraction(1, 71), 100))


class TestSampler(unittest.TestCase):
    def test_words_are_deterministic_and_pool_separated(self):
        a = PR.draw_words(SEED, PR.STRATUM_FRESH_INPUT, 8)
        self.assertEqual(a, PR.draw_words(SEED, PR.STRATUM_FRESH_INPUT, 8))
        self.assertEqual(a[:4], PR.draw_words(SEED, PR.STRATUM_FRESH_INPUT, 4))
        self.assertNotEqual(a, PR.draw_words(SEED, PR.STRATUM_GENERATION, 8))
        self.assertNotEqual(a, PR.draw_words(bytes(32), PR.STRATUM_FRESH_INPUT, 8))
        self.assertTrue(all(0 <= w < 1 << PR.N_SMP for w in a))
        for bad in ((b"short", 1, 1), (SEED, 0, 1), (SEED, 256, 1), (SEED, 1, -1)):
            with self.assertRaises(ValueError):
                PR.draw_words(*bad)

    def test_unit_index_is_the_floor_map(self):
        top = (1 << PR.N_SMP) - 1
        self.assertEqual((PR.unit_index(0, 7), PR.unit_index(top, 7), PR.unit_index(1 << (PR.N_SMP - 1), 7)), (0, 6, 3))
        with self.assertRaises(ValueError):
            PR.unit_index(0, 0)
        with self.assertRaises(ValueError):
            PR.unit_index(1 << PR.N_SMP, 3)

    def test_gemm_tiles_are_32_row_aligned_and_ragged_tail_is_clipped(self):
        draws = PR.sample_stratum(SEED, DENSE, PR.STRATUM_GENERATION, range(3), 600)
        tiles = [d for d in draws if d.kind == "gemm"]
        self.assertTrue(tiles)
        seen = set()
        for d in tiles:
            inner, rows = DENSE.gemms[d.gemm]
            self.assertEqual(d.tile_start % 32, 0)
            self.assertEqual(d.tile_rows, min(32, rows - d.tile_start))
            self.assertEqual(d.cost, inner * d.tile_rows)
            self.assertTrue(0 <= d.layer < DENSE.layers)
            seen.add((d.gemm, d.tile_start, d.tile_rows))
        self.assertIn((1, 32, 8), seen)            # 40-row GEMM: second tile has 8 rows
        self.assertEqual({t[2] for t in seen if t[0] != 1}, {32})

    def test_attention_draw_covers_every_query_head_of_one_kv_group(self):
        draws = PR.sample_stratum(SEED, DENSE, PR.STRATUM_GENERATION, range(40, 44), 400)
        groups = [d for d in draws if d.kind == "attn"]
        self.assertTrue(groups)
        for d in groups:
            self.assertEqual(len(d.query_heads), DENSE.q_heads // DENSE.kv_groups)
            self.assertEqual(d.query_heads, tuple(range(4 * d.kv_group, 4 * d.kv_group + 4)))
            self.assertEqual(d.cost, 4 * (d.position + 1) * 2 * DENSE.head_dim)
        self.assertEqual({d.kv_group for d in groups}, {0, 1})

    def test_draws_are_cost_weighted(self):
        n = 6000
        draws = PR.sample_stratum(SEED, DENSE, PR.STRATUM_GENERATION, range(8), n)
        total = PR.stratum_cost(DENSE, range(8))
        gemm_share = Fraction(8 * DENSE.layers * DENSE.gemm_cost, total)
        got = Fraction(sum(d.kind == "gemm" for d in draws), n)
        self.assertLess(abs(got - gemm_share), Fraction(3, 100))
        # Longer contexts attract proportionally more attention draws.
        by_pos = Counter(d.position for d in draws if d.kind == "attn")
        self.assertGreater(by_pos[7], by_pos[0])
        # A full tile is hit about four times as often as the 8-row ragged tile of the same GEMM.
        hits = Counter(d.tile_start for d in draws if d.kind == "gemm" and d.gemm == 1)
        self.assertGreater(hits[0], 2 * hits[32])

    def test_fresh_input_stratum_excludes_the_cached_prefix(self):
        out = PR.sample_request(SEED, DENSE, input_tokens=2000, cache_read_input_tokens=1990, output_tokens=5, draws=100)
        fresh, gen, steps = out[PR.STRATUM_FRESH_INPUT], out[PR.STRATUM_GENERATION], out[PR.POOL_TOKEN_SAMPLING]
        self.assertEqual((len(fresh), len(gen), len(steps)), (100, 100, 100))
        # Rule Pi_2: positions 1999 .. 2003 emit the five output tokens; 2004 is never fed forward.
        self.assertTrue(all(1990 <= d.position < 1999 for d in fresh))
        self.assertTrue(all(1999 <= d.position < 2004 for d in gen))
        self.assertEqual({d.position for d in gen}, set(range(1999, 2004)))
        self.assertTrue(all(0 <= s < 5 for s in steps))
        # Ten fresh tokens behind a 1990-token cache still receive the full k draws (no dilution).
        self.assertEqual(sum(not d.empty for d in fresh), 100)
        # Cached keys still enter the attention cost of a fresh position.
        self.assertEqual(DENSE.group_cost(1995), 4 * 1996 * 2 * 16)
        self.assertEqual(list(PR.fresh_input_stratum(5, 3)), [3, 4])
        with self.assertRaises(ValueError):
            PR.fresh_input_stratum(3, 5)

    def test_generation_stratum_follows_rule_pi2(self):
        self.assertEqual(PR.generation_stratum(10, 3), range(9, 12))
        self.assertEqual(PR.generation_stratum(10, 0), range(0))
        self.assertEqual(PR.generation_stratum(0, 2), range(0, 1))
        with self.assertRaises(ValueError):
            PR.generation_stratum(-1, 1)
        # Full cache hit, one output token: the last input position still computes the logits.
        out = PR.sample_request(SEED, DENSE, input_tokens=64, cache_read_input_tokens=64, output_tokens=1, draws=30)
        self.assertTrue(all(d.empty for d in out[PR.STRATUM_FRESH_INPUT]))
        self.assertEqual({d.position for d in out[PR.STRATUM_GENERATION]}, {63})
        self.assertEqual(set(out[PR.POOL_TOKEN_SAMPLING]), {0})
        # Without output the last input position stays ordinary fresh input.
        out = PR.sample_request(SEED, DENSE, input_tokens=8, cache_read_input_tokens=6, output_tokens=0, draws=200)
        self.assertEqual({d.position for d in out[PR.STRATUM_FRESH_INPUT]}, {6, 7})
        # The two compute strata never overlap and together cover every position computed this turn.
        out = PR.sample_request(SEED, DENSE, input_tokens=6, cache_read_input_tokens=2, output_tokens=3, draws=800)
        fresh = {d.position for d in out[PR.STRATUM_FRESH_INPUT]}
        gen = {d.position for d in out[PR.STRATUM_GENERATION]}
        self.assertEqual((fresh, gen), ({2, 3, 4}, {5, 6, 7}))

    def test_attention_draw_resolves_a_key_atom_and_its_lineage(self):
        draws = PR.sample_stratum(SEED, DENSE, PR.STRATUM_GENERATION, range(20, 24), 500)
        attn = [d for d in draws if d.kind == "attn"]
        self.assertTrue(all(0 <= d.key_position <= d.position for d in attn))
        self.assertGreater(len({d.key_position for d in attn}), 15)            # keys are spread over the context
        for d in attn:
            lin = d.lineage()
            self.assertEqual(lin[0], (d.position, d.layer, PR.BLOCK_ATTN))
            self.assertLessEqual(len(lin), PR.lineage_bound())
            if d.layer:
                self.assertIn((d.key_position, d.layer - 1, PR.BLOCK_MLP), lin)   # the unit that produced (k, v)
        gemm = [d for d in draws if d.kind == "gemm"]
        self.assertEqual({d.unit[2] for d in gemm if d.gemm < 2}, {PR.BLOCK_ATTN})
        self.assertEqual({d.unit[2] for d in gemm if d.gemm == 2}, {PR.BLOCK_MLP})
        self.assertTrue(all(d.key_position == -1 and len(d.lineage()) <= 3 for d in gemm))
        # Windowed attention: the key atom is one of the window or sink positions, never an evicted one.
        swa = PR.SamplerProfile(2, ((8, 8),), 4, 2, 8, window=16, sink=2, attn_gemms=1)
        keys = {d.key_position for d in PR.sample_stratum(SEED, swa, 2, range(900, 901), 600) if d.kind == "attn"}
        self.assertLessEqual(keys, {0, 1} | set(range(885, 901)))
        self.assertTrue(keys & {0, 1})
        with self.assertRaises(ValueError):
            PR.Draw(1, 0, True).unit

    def test_empty_strata_keep_the_circuit_shape(self):
        out = PR.sample_request(SEED, DENSE, input_tokens=64, cache_read_input_tokens=64, output_tokens=0, draws=51)
        fresh = out[PR.STRATUM_FRESH_INPUT]
        self.assertEqual(len(fresh), 51)
        self.assertTrue(all(d.empty and d.position == -1 and d.cost == 0 for d in fresh))
        self.assertTrue(all(d.empty for d in out[PR.STRATUM_GENERATION]))
        self.assertEqual(out[PR.POOL_TOKEN_SAMPLING], (-1,) * 51)
        with self.assertRaises(ValueError):
            PR.sample_request(SEED, DENSE, input_tokens=4, cache_read_input_tokens=0, output_tokens=-1, draws=1)

    def test_draw_record_has_no_field_for_cached_positions(self):
        """A draw names one fresh or generated position; nothing in it enumerates the cached prefix."""
        a = PR.sample_request(SEED, DENSE, input_tokens=300, cache_read_input_tokens=0, output_tokens=4, draws=20)
        b = PR.sample_request(SEED, DENSE, input_tokens=300, cache_read_input_tokens=280, output_tokens=4, draws=20)
        self.assertEqual(len(a[1]), len(b[1]))
        self.assertTrue(all(d.position >= 280 for d in b[1]))
        self.assertEqual(a[PR.POOL_TOKEN_SAMPLING], b[PR.POOL_TOKEN_SAMPLING])   # the step pool ignores caching

    def test_sliding_window_and_sinks_bound_the_attention_cost(self):
        swa = PR.SamplerProfile(2, ((8, 8),), 4, 2, 8, window=16, sink=2, attn_gemms=1)
        self.assertEqual((swa.keys(3), swa.keys(15), swa.keys(16), swa.keys(900)), (4, 16, 17, 18))
        dense = PR.SamplerProfile(2, ((8, 8),), 4, 2, 8, attn_gemms=1)
        self.assertEqual(dense.keys(900), 901)
        self.assertLess(swa.position_cost(900), dense.position_cost(900))

    def test_invalid_profiles(self):
        for kw in (dict(q_heads=6, kv_groups=4), dict(layers=0), dict(gemms=()), dict(gemms=((0, 4),)), dict(tile_rows=0),
                   dict(attn_gemms=2)):
            args = dict(layers=1, gemms=((4, 4),), q_heads=4, kv_groups=2, head_dim=8, attn_gemms=1)
            args.update(kw)
            with self.assertRaises(ValueError):
                PR.SamplerProfile(**args)


class TestLineage(unittest.TestCase):
    def test_depth_two_closure(self):
        self.assertEqual(LINEAGE_DEPTH, 2)
        self.assertEqual(PR.lineage((5, 3, PR.BLOCK_MLP)),
                         ((5, 3, PR.BLOCK_MLP), (5, 3, PR.BLOCK_ATTN), (5, 2, PR.BLOCK_MLP)))
        self.assertEqual(PR.lineage((5, 0, PR.BLOCK_ATTN)), ((5, 0, PR.BLOCK_ATTN),))       # stops at the embedding
        self.assertEqual(PR.lineage((5, 3, PR.BLOCK_MLP), depth=0), ((5, 3, PR.BLOCK_MLP),))
        self.assertEqual(len(PR.lineage((5, 9, PR.BLOCK_MLP), depth=3)), 4)

    def test_key_atom_adds_the_kv_producer_closure(self):
        lin = PR.lineage((9, 4, PR.BLOCK_ATTN), key_position=2)
        self.assertEqual(lin[:3], ((9, 4, PR.BLOCK_ATTN), (9, 3, PR.BLOCK_MLP), (9, 3, PR.BLOCK_ATTN)))
        self.assertEqual(lin[3:], ((2, 3, PR.BLOCK_MLP), (2, 3, PR.BLOCK_ATTN), (2, 2, PR.BLOCK_MLP)))
        self.assertLessEqual(len(lin), PR.lineage_bound())
        self.assertEqual(PR.lineage_bound(), 6)
        self.assertEqual(len(PR.lineage((9, 4, PR.BLOCK_ATTN), key_position=9)), 4)         # own position: shared tail
        for bad in (dict(unit=(9, 4, PR.BLOCK_MLP), key_position=2), dict(unit=(9, 4, PR.BLOCK_ATTN), key_position=10),
                    dict(unit=(-1, 0, 0)), dict(unit=(0, 0, 2)), dict(unit=(0, 0, 0), depth=-1)):
            with self.assertRaises(ValueError):
                PR.lineage(**bad)

    def test_lineage_amplifies_skipped_layers(self):
        costs = {(0, layer, block): 1 for layer in range(8) for block in (PR.BLOCK_ATTN, PR.BLOCK_MLP)}
        skip = {u for u in costs if u[1] % 4 >= 2}          # two consecutive layers out of every four
        d = [PR.lineage_defect(costs, skip, depth=h) for h in range(4)]
        self.assertEqual(d[0], Fraction(1, 2))
        self.assertEqual(d[1], Fraction(9, 16))      # each skipped run also taints its first consumer
        self.assertEqual(d[2], Fraction(5, 8))
        self.assertTrue(d[0] < d[1] < d[2] < d[3])
        self.assertEqual(PR.lineage_defect(costs, ()), 0)
        self.assertEqual(PR.lineage_defect({}, skip), 0)
        # A cheat on the cheap blocks only: lineage charges the expensive consumers as well.
        weighted = {u: (9 if u[2] == PR.BLOCK_MLP else 1) for u in costs}
        attn_only = {u for u in costs if u[2] == PR.BLOCK_ATTN}
        self.assertEqual(PR.lineage_defect(weighted, attn_only, depth=0), Fraction(1, 10))
        self.assertEqual(PR.lineage_defect(weighted, attn_only, depth=1), 1)

    def test_residual_floor_is_marked_as_a_calibrated_premise(self):
        self.assertEqual(RESIDUAL_RMS_FLOOR, Fraction(1, 1000))
        premise = PR.COMPLETENESS_PREMISES["residual_rms_floor"]
        self.assertEqual(premise["value"], RESIDUAL_RMS_FLOOR)
        self.assertIn("MODEL_CALIBRATED", premise["deep_layers"])
        self.assertTrue(PR.residual_floor_holds(Fraction(4, 10**6), 4))
        self.assertFalse(PR.residual_floor_holds(Fraction(4, 10**6) - Fraction(1, 10**12), 4))
        self.assertFalse(PR.residual_floor_holds(0, 1024))
        with self.assertRaises(ValueError):
            PR.residual_floor_holds(1, 0)


class TestCertifiedLse(unittest.TestCase):
    def test_constants_match_m0_s1(self):
        with open(os.path.join(ROOT, "benchmarks", "certified", "results", "lse_radius_reverify.json"), encoding="utf-8") as fh:
            params = json.load(fh)["circuit_discretization_params"]
        self.assertEqual(PR.CIRCUIT_EXP_STEP, Fraction(params["exp_step"]))
        self.assertEqual(PR.CIRCUIT_NORM_LIFT, Fraction(params["norm_lift"]))
        self.assertEqual(PR.CIRCUIT_LOG_STEP, Fraction(params["log_step"]))
        self.assertEqual(PR.ETA_BF16, Fraction(params["eta_bf16"]))
        self.assertEqual(PR.ETA_FP16, Fraction(params["eta_fp16"]))
        self.assertTrue(params["norm_rel_uncapped"])
        self.assertEqual((PR.CIRCUIT_EXP_STEP, PR.CIRCUIT_NORM_LIFT, PR.ETA_FP16, PR.ETA_BF16),
                         (Fraction(1, 2**14), Fraction(1, 2**32), Fraction(1, 2**25), Fraction(1, 2**126)))

    def test_floor_radius_is_below_every_measured_m0_s1_median(self):
        """With zero score radius the engine gives the data-free floor of the certified LSE formula, which must sit
        below the measured medians (those add the data-dependent score radius)."""
        with open(os.path.join(ROOT, "benchmarks", "certified", "results", "lse_radius_reverify.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        rows = data["qwen3_4b_sweep"] + data["qwen3_06b_longctx_up_to_64k"]
        self.assertGreaterEqual(len(rows), 10)
        for row in rows:
            precision = "fp16" if "fp16" in row["precision"] else "bf16"
            r = PR.certified_lse_radius(row["T"], precision=precision)
            self.assertLess(log2(r.circuit), row["lse_full_circuit_log2"]["median"], row)
            self.assertLess(log2(r.floating), row["lse_fp_only_log2"]["median"], row)
            self.assertLess(r.floating, r.circuit)

    def test_matches_an_independent_float_evaluation(self):
        import math
        for tokens, rbar, zmax, precision in ((4096, 2.0**-13, 9.5, "bf16"), (32768, 2.0**-12, 30.0, "bf16"),
                                              (16384, 2.0**-12, 4.0, "fp16"), (65536, 0.0, 0.0, "fp16")):
            u_p, eta_p = (2.0**-8, 2.0**-126) if precision == "bf16" else (2.0**-11, 2.0**-25)
            n_t = math.ceil(tokens / 64) + 8
            emu = (1 + u_p) * (1 + 2.0**-24) ** (2 * n_t + 68) - 1
            emu2 = (1 + 2.0**-24) ** (2 * n_t + 64) - 1
            one_s = math.exp(2 * rbar) * (1 + 2.0**-21) ** (n_t + 1)
            e_den = one_s * (1 + emu2) - 1 + tokens * math.exp(rbar) * ((n_t + 1) * (1 + emu) * 2.0**-126 + eta_p)
            fp = rbar - math.log(1 - e_den) + 4 * 2.0**-21
            cir = (fp + 2.0**-14 - math.log(1 - tokens * 2.0**-32) + 2.0**-14
                   + 2.0**-24 * (zmax + math.log(max(tokens, 2))))
            r = PR.certified_lse_radius(tokens, score_radius_max=Fraction(rbar), score_max_abs=Fraction(zmax),
                                        precision=precision)
            self.assertAlmostEqual(float(r.floating) / fp, 1, places=9)
            self.assertAlmostEqual(float(r.circuit) / cir, 1, places=9)
            self.assertGreaterEqual(r.circuit, Fraction(cir) * (1 - Fraction(1, 10**12)))

    def test_monotone_in_context_score_radius_and_score_magnitude(self):
        for precision in ("bf16", "fp16"):
            radii = [PR.certified_lse_radius(t, precision=precision).circuit for t in (64, 256, 4096, 32768, 131072)]
            self.assertEqual(radii, sorted(radii))
            self.assertEqual(len(set(radii)), len(radii))
        base = PR.certified_lse_radius(4096)
        self.assertLess(base.circuit, PR.certified_lse_radius(4096, score_radius_max=Fraction(1, 2**12)).circuit)
        self.assertLess(base.circuit, PR.certified_lse_radius(4096, score_max_abs=100).circuit)
        self.assertEqual(base.floating, PR.certified_lse_radius(4096, score_max_abs=100).floating)

    def test_fp16_is_looser_than_bf16_because_of_the_subnormal_floor(self):
        for tokens in (4096, 16384, 32768, 65536):
            bf, fp = (PR.certified_lse_radius(tokens, precision=p) for p in ("bf16", "fp16"))
            self.assertGreater(fp.circuit, bf.circuit)
            self.assertGreater(fp.e_den - bf.e_den, tokens * PR.ETA_FP16 * (1 - Fraction(1, 2**20)))
        # The gap grows with T: the T * 2^-25 term.
        gap = [PR.certified_lse_radius(t, precision="fp16").circuit - PR.certified_lse_radius(t).circuit
               for t in (4096, 32768)]
        self.assertGreater(gap[1], 7 * gap[0])

    def test_norm_lift_is_uncapped_and_two_sided(self):
        r = PR.certified_lse_radius(1 << 20)
        self.assertEqual(r.norm_rel, Fraction(1, 2**12))
        lift = r.circuit - r.floating - PR.CIRCUIT_EXP_STEP - PR.CIRCUIT_LOG_STEP
        self.assertGreater(lift, r.norm_rel)                       # -log(1 - e) > e: the downward side governs
        self.assertLess(lift, r.norm_rel * (1 + r.norm_rel) + PR.U_FP32 * 14)   # + fp32 store of ln(2^20) < 14
        # The 2^-32 lift at 32k is far below the 2^-24 lift it replaces.
        self.assertEqual(PR.certified_lse_radius(32768).norm_rel, Fraction(1, 2**17))

    def test_fail_closed(self):
        with self.assertRaises(PR.LseDomainError) as lift:
            PR.certified_lse_radius(1 << 32)
        self.assertIn("lift", str(lift.exception))
        with self.assertRaises(PR.LseDomainError) as den:
            PR.certified_lse_radius(4096, score_radius_max=Fraction(1, 2))       # exp(2 rbar) - 1 > 1
        self.assertIn("denominator", str(den.exception))
        with self.assertRaises(PR.LseDomainError):
            PR.certified_lse_radius(1 << 26, precision="fp16")                   # T * 2^-25 >= 1
        PR.certified_lse_radius(1 << 26, precision="bf16")                        # bf16 is still certifiable there
        for bad in (dict(tokens=0), dict(tokens=8, score_radius_max=-1), dict(tokens=8, precision="fp32")):
            with self.assertRaises(ValueError):
                PR.certified_lse_radius(**bad)

    def test_score_radius(self):
        r = PR.score_radius(Fraction(3), 2, 5, 128)
        gamma = Fraction(72, 2**24) / (1 - Fraction(72, 2**24))
        self.assertEqual(r, gamma * 3 + Fraction(28, 2**24))


class TestPaddingAndAuditRamp(unittest.TestCase):
    def test_length_bucket(self):
        self.assertEqual([PR.length_bucket(t) for t in (0, 1, 256, 257, 32768, 32769, 200000)],
                         [256, 256, 256, 512, 32768, 65536, 262144])
        self.assertEqual(PR.length_bucket(262144), 262144)
        with self.assertRaises(ValueError):
            PR.length_bucket(262145)
        with self.assertRaises(ValueError):
            PR.length_bucket(-1)

    def test_budget_depends_only_on_bucket_and_public_caps(self):
        a, b = PR.fpn2_budget(4097, 1000), PR.fpn2_budget(8192, 1000)
        self.assertEqual(a, b)
        self.assertEqual(a.bucket, 8192)
        self.assertEqual(a.total_bytes, a.containers * a.container_bytes)
        self.assertEqual(a.container_bytes % PR.CONTAINER_ALIGN, 0)
        self.assertGreater(PR.fpn2_budget(8193, 1000).total_bytes, a.total_bytes)
        self.assertEqual(a.padding(a.containers), 0)
        with self.assertRaises(ValueError):
            a.padding(a.containers + 1)                # over budget is an error, never a truncation
        with self.assertRaises(ValueError):
            PR.fpn2_budget(100, 0)

    def test_registered_profile_uses_its_exact_container_size(self):
        from fpn_prover.backend.fpn2 import FOLDED_GKR_V1
        one = PR.fpn2_budget(1, 0, fixed_queries=128)
        self.assertTrue(one.registered)
        self.assertEqual((one.n_vars, one.containers, one.container_bytes), (12, 1, FOLDED_GKR_V1["c_p"]))
        big = PR.fpn2_budget(256, 1 << 20)
        self.assertEqual(big.n_vars, 28)
        self.assertEqual(big.containers, (256 << 20) >> 23)
        self.assertFalse(big.registered)

    def test_p_audit_cost_model(self):
        c = PR.audit_cost(2, Fraction(1, 50))
        self.assertEqual((c.overhead, c.passes, c.p_audit_max), (Fraction(1, 25), True, Fraction(1, 40)))
        self.assertFalse(PR.audit_cost(2, Fraction(1, 10)).passes)
        self.assertEqual(PR.audit_cost(Fraction(1, 25), 1).p_audit_max, 1)          # cheap prover: audit everything
        self.assertTrue(PR.audit_cost(Fraction(1, 20), 1).passes)
        with_commit = PR.audit_cost(2, Fraction(1, 100), commit_overhead=Fraction(3, 100))
        self.assertEqual((with_commit.overhead, with_commit.p_audit_max), (Fraction(1, 20), Fraction(1, 100)))
        self.assertEqual(PR.audit_cost(2, 0, commit_overhead=Fraction(6, 100)).p_audit_max, 0)
        self.assertEqual(PR.audit_cost(0, 1).p_audit_max, 1)
        for bad in ((-1, 0), (1, 2), (1, -1)):
            with self.assertRaises(ValueError):
                PR.audit_cost(*bad)


class TestProductionContracts(unittest.TestCase):
    NEW = ("fp8_e4m3_block128_fp32acc", "mxfp4_e2m1_block32_fp32acc")

    def test_contracts_are_registered_and_not_claimed_calibrated(self):
        for name in self.NEW:
            c = CONTRACTS[name]
            self.assertIs(c.calibration, Calibration.SCHEMA_ONLY)
            self.assertEqual(c.calibrated_on, ())
            with self.assertRaises(UncalibratedContractError):
                MODEL_FORMATS[name].require_calibrated()
            self.assertEqual((c.accum_u_log2, c.promote_k, c.store_u_log2, c.eta_log2), (-24, 0, -8, -126))
            self.assertEqual(FM.RoundingContract.from_dict(c.to_dict()), c)
        self.assertIs(CONTRACTS[self.NEW[0]].tol_class, ToleranceClass.BLOCK_FP8)
        self.assertIs(CONTRACTS[self.NEW[1]].tol_class, ToleranceClass.BLOCK_MX)

    def test_existing_entries_are_untouched(self):
        self.assertEqual(CONTRACTS["fp8_e4m3_block128"].promote_k, 128)
        self.assertEqual(CONTRACTS["mxfp4_e2m1"], FM.replace(CONTRACTS["mxfp8_e4m3"], name="mxfp4_e2m1"))
        self.assertEqual(set(CONTRACTS) - set(self.NEW),
                         {"bf16_fp32acc", "fp16_attn_strict", "fp8_e4m3_block128", "mxfp8_e4m3", "mxfp4_e2m1"})

    def test_model_formats_match_the_checkpoint_layouts(self):
        fp8, mx = (MODEL_FORMATS[n] for n in self.NEW)
        self.assertEqual((fp8.weight.scale.block_rows, fp8.weight.scale.block_cols, fp8.weight.scale.scale_format),
                         (128, 128, "fp32"))
        self.assertEqual((fp8.activation.scale.block_rows, fp8.activation.scale.block_cols), (1, 128))
        self.assertEqual((mx.weight.bits, mx.weight.scale.block_elems, mx.weight.scale.scale_format), (4, 32, "e8m0"))
        self.assertEqual(mx.activation.name, "bf16")
        self.assertEqual(mx.weight.storage_bits_per_element, Fraction(17, 4))
        self.assertEqual(fp8.weight.max_finite, 448)
        self.assertEqual(mx.weight.max_finite, 6)

    def test_dot_bound_is_the_fp32_accumulation_template(self):
        r32 = CONTRACTS["bf16_fp32acc"]
        for name in self.NEW:
            for k in (1, 64, 2880, 4096):
                self.assertEqual(CONTRACTS[name].dot_relative_bound(k), r32.dot_relative_bound(k))
        # Tighter than the promote-window schema entry it supersedes for the fp32-accumulate path.
        self.assertLess(CONTRACTS[self.NEW[0]].dot_relative_bound(4096), CONTRACTS["fp8_e4m3_block128"].dot_relative_bound(4096))
        self.assertFalse(CONTRACTS[self.NEW[0]] <= r32)         # tolerance class still above R32
        self.assertFalse(CONTRACTS[self.NEW[1]] <= r32)

    def test_per_gemm_bound_mxfp4_is_exact_weights_plus_arithmetic(self):
        b = gemm_error_bound(MODEL_FORMATS[self.NEW[1]], 2880)
        self.assertTrue(b.weight_exact)
        self.assertEqual((b.operand_store, b.weight_store, b.operand_floor_log2, b.eta_log2), (0, 0, None, -126))
        self.assertEqual(b.relative, b.arithmetic)
        self.assertEqual(b.relative, CONTRACTS[self.NEW[1]].dot_relative_bound(2880))
        self.assertEqual(b.relative, gemm_error_bound(MODEL_FORMATS["bf16"], 2880).relative)

    def test_per_gemm_bound_fp8_charges_scale_product_and_activation_store(self):
        b = gemm_error_bound(MODEL_FORMATS[self.NEW[0]], 2048)
        self.assertFalse(b.weight_exact)                      # 4-bit significand x 24-bit fp32 scale = 28 > 24
        self.assertEqual(MODEL_FORMATS[self.NEW[0]].weight.lift_mantissa_bits, 28)
        self.assertEqual((b.operand_store, b.weight_store, b.operand_floor_log2), (Fraction(1, 16), Fraction(1, 2**24), -10))
        self.assertEqual(b.relative, (1 + b.arithmetic) * (1 + Fraction(1, 16)) * (1 + Fraction(1, 2**24)) - 1)
        self.assertGreater(b.relative, Fraction(1, 16))
        self.assertLess(b.arithmetic, Fraction(1, 200))
        ks = [gemm_error_bound(MODEL_FORMATS[self.NEW[0]], k).relative for k in (16, 256, 4096, 65536)]
        self.assertEqual(ks, sorted(ks))

    def test_exact_dequant_scale_range(self):
        mx = MODEL_FORMATS[self.NEW[1]].weight
        lo, hi = exact_dequant_scale_exp_range(mx, NUM_FORMATS["bf16"])
        self.assertEqual((lo, hi), (-125, 125))
        bf = NUM_FORMATS["bf16"]
        self.assertEqual(Fraction(1, 2) * Fraction(2) ** lo, Fraction(2) ** (1 - bf.bias))    # smallest element = min normal
        self.assertLessEqual(6 * Fraction(2) ** hi, bf.max_finite)
        self.assertGreater(6 * Fraction(2) ** (hi + 1), bf.max_finite)
        self.assertIsNone(exact_dequant_scale_exp_range(MODEL_FORMATS[self.NEW[0]].weight, bf))   # fp32 scale
        self.assertIsNone(exact_dequant_scale_exp_range(bf, bf))

    def test_commitment_size_estimators(self):
        fp8 = weight_commitment(MODEL_FORMATS[self.NEW[0]], 768, 2048)
        self.assertEqual((fp8["elements"], fp8["scale_blocks"], fp8["bytes"]), (768 * 2048, 6 * 16, 768 * 2048 + 96 * 4))
        ragged = weight_commitment(MODEL_FORMATS[self.NEW[0]], 130, 129)
        self.assertEqual((ragged["scale_blocks"], ragged["bytes"]), (4, 130 * 129 + 16))
        mx = weight_commitment(MODEL_FORMATS[self.NEW[1]], 2880, 2880)
        self.assertEqual((mx["scale_blocks"], mx["bytes"]), (2880 * 90, 2880 * 2880 // 2 + 2880 * 90))
        self.assertEqual(mx["cells"], 2880 * 2880 + 2880 * 90)
        self.assertEqual((mx["element_integer_bits"], fp8["element_integer_bits"]), (5, 19))
        bf = weight_commitment(MODEL_FORMATS["bf16"], 2880, 2880)
        self.assertEqual((bf["scale_blocks"], bf["bytes"]), (0, 2 * 2880 * 2880))
        self.assertAlmostEqual(bf["bytes"] / mx["bytes"], 16 / 4.25, places=6)
        self.assertEqual(NUM_FORMATS["bf16"].storage_bytes_2d(3, 5), 30)
        with self.assertRaises(FM.FormatError):
            NUM_FORMATS["bf16"].block_count(-1, 2)

    def test_kernel_adapters_cover_hopper_and_blackwell_without_branching(self):
        for name in self.NEW:
            archs = {a.arch for a in KERNEL_ADAPTERS.values() if a.contract is CONTRACTS[name]}
            self.assertEqual(archs, {"sm_90", "sm_120"})
            adapters = [a for a in KERNEL_ADAPTERS.values() if a.contract is CONTRACTS[name]]
            self.assertTrue(all(a.supports(MODEL_FORMATS[name]) for a in adapters))
            # Same contract object on both architectures: the proof side cannot distinguish them.
            self.assertEqual(len({id(a.contract) for a in adapters}), 1)


class TestFrontierCalibrateDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.tmp.name, "frontier.json")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = os.path.join(ROOT, "prover")
        cls.run_cli = staticmethod(lambda *a: subprocess.run(
            [sys.executable, "-W", "error", BENCH, *a], capture_output=True, text=True, env=env, cwd=cls.tmp.name))
        cls.proc = cls.run_cli("--dry-run", "--out", cls.out)
        if cls.proc.returncode == 0:
            with open(cls.out, encoding="utf-8") as fh:
                cls.report = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dry_run_passes_without_torch(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        self.assertEqual(self.proc.stderr, "")
        self.assertIn("not calibration evidence", self.proc.stdout)
        self.assertEqual(self.report["mode"], "dry_run")
        self.assertTrue(self.report["evidence"].startswith("DRY_RUN_SYNTHETIC"))
        self.assertTrue(self.report["ok"])
        with open(BENCH, encoding="utf-8") as fh:
            head = fh.read().split("def calibrate_real", 1)[0]
        self.assertNotIn("import torch", head)                  # torch is imported inside the real mode only

    def test_all_five_models_and_their_formats(self):
        by = {m["model"]: m for m in self.report["models"]}
        self.assertEqual(set(by), {"Qwen3-30B-A3B-FP8", "gpt-oss-20b", "Qwen3.5-0.8B", "Llama-3.2-1B-Instruct", "Qwen3-4B"})
        self.assertEqual(by["Qwen3-30B-A3B-FP8"]["format"]["model_format"], "fp8_e4m3_block128_fp32acc")
        self.assertEqual(by["gpt-oss-20b"]["format"]["model_format"], "mxfp4_e2m1_block32_fp32acc")
        self.assertEqual(by["gpt-oss-20b"]["format"]["exact_dequant_scale_exp_range_bf16"], [-125, 125])
        for m in by.values():
            self.assertEqual(m["status"], "DRY_RUN_OK")
            self.assertEqual(m["gemm"]["honest_bad_blocks"], 0)
            self.assertEqual(m["gemm"]["honest_bad_blocks_unquantised_radius"], 0)
            self.assertLess(m["gemm"]["honest_max_ratio"], 1)
            self.assertEqual(m["gemm"]["requantised_weight_cheat_bad_blocks"], m["gemm"]["blocks"])
            self.assertTrue(m["rms_floor"]["holds"])
            self.assertEqual(m["sampler"]["cached_positions_drawn"], 0)
            self.assertTrue(m["shapes"].startswith("NOMINAL"))
            self.assertIn("attention kernels", m["not_covered"])
            self.assertIn("DRY_RUN_SYNTHETIC", m["evidence"])

    def test_architecture_specific_checks(self):
        by = {m["model"]: m for m in self.report["models"]}
        for name in ("Qwen3-30B-A3B-FP8", "gpt-oss-20b"):
            router = by[name]["moe_router"]
            self.assertTrue(router["honest_accepted"])
            self.assertFalse(router["dropped_expert_accepted"])
            self.assertFalse(router["skipped_best_accepted"])
        self.assertEqual(by["gpt-oss-20b"]["swa_sinks"]["keys_at_position"], 132)
        self.assertTrue(by["gpt-oss-20b"]["swa_sinks"]["keeps_sinks"])
        self.assertEqual(by["Qwen3-30B-A3B-FP8"]["sampler"]["heads_per_kv_group"], 8)
        self.assertNotIn("moe_router", by["Llama-3.2-1B-Instruct"])

    def test_cost_gate_is_not_claimed_from_a_dry_run(self):
        for m in self.report["models"]:
            cost = m["cost"]
            self.assertIsNone(cost["cost_ratio"])
            self.assertTrue(cost["gate_status"].startswith("UNMEASURED"))
            self.assertTrue(cost["prover_gpu_s_label"].startswith("EXTRAPOLATED"))
            self.assertEqual(cost["p_audit_max_if_cost_ratio"]["1"], 0.05)
            self.assertEqual(cost["p_audit_max_if_cost_ratio"]["1/20"], 1.0)
        self.assertEqual(self.report["soundness"]["external_tight"]["draws"], 100)

    def test_real_mode_refuses_without_gpu_confirmation(self):
        proc = self.run_cli("--models", "Qwen3-4B")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--confirm-gpu", proc.stderr)

    def test_model_subset(self):
        proc = self.run_cli("--dry-run", "--models", "gpt-oss-20b")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.count("DRY_RUN_OK"), 1)


class SeededRng:
    def __init__(self, seed):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def fp3(self):
        from flashprove_next.field import P
        return tuple(int(x) for x in self.rng.integers(0, P, size=3, dtype="uint64"))


def _bf16_word(x):
    word = struct.unpack(">I", struct.pack(">f", x))[0]
    return (word + 0x7fff + ((word >> 16) & 1)) >> 16


# One satisfying lane assignment per microprogram opcode (dense GQA, SWA+sinks, MoE, MLA, Gated DeltaNet).
SLOT_BRANCHES = (
    (0, 256, 256, 0, 512, 0, 512),
    (0, 256, 256, 0, 512, 0, 512),
    (0, 0, 1, 1, 1, 0, 256, 256, 0, 0, 0, 1, 0, 512, 0),
    (256, 0, 256, 512, 0, 0, 0, 512),
    (0, 1, 0, 256, 0, 256, 0, 256, 256, 256, 0),
)


class TestM23ContainerUniformity(unittest.TestCase):
    """Gate M2-3 on the relation-3 containers that exist today: the same public circuit proved for five different
    architectures gives byte-identical lengths, and re-proving shares no 64-byte block outside the public header
    and the zero padding.  This covers the lowering containers only, not a full hidden-architecture reply proof."""

    @classmethod
    def setUpClass(cls):
        try:
            from fpn_prover.backend.fpn2 import FOLDED_GKR_V1
            from fpn_prover.circuit import lower as LW
            from fpn_prover.circuit.blocks import gemm_tile_row_block, rmsnorm_block
            from fpn_prover.circuit.microprogram import install_microprogram
            from fpn_prover.ir.arith import bf16_decode, norm_rsqrt
            from fpn_prover.ir.gadget_ir import Program
        except ImportError as exc:      # numpy / field package unavailable
            raise unittest.SkipTest(f"lowering backend unavailable: {exc}") from None
        cls.LW, cls.c_p = LW, FOLDED_GKR_V1["c_p"]

        def build(opcode):
            p = Program()
            norm = rmsnorm_block(p, "norm", 2)
            contract = FM.get_contract("mxfp4_e2m1_block32_fp32acc")
            gemm = gemm_tile_row_block(p, "gemm", 1, 1, contract)
            slot = install_microprogram(p, "slot", tokens=2, experts=2, top_k=2, window=1, sink=1, conv_width=1, extent=1)
            one = _bf16_word(1)
            x = bf16_decode(one).value
            ref = norm_rsqrt((x * x + x * x) / 2 + Fraction(1, 10**6))
            w = dict(zip(norm, (one, one, one, one, one, one, ref.k, ref.low, ref.high, ref.high - ref.low), strict=True))
            z = 8 * contract.dot_relative_bound(1).numerator
            w.update(zip(gemm, (8, 0, z, 3, 3, 24, 3 * z, 24, 0), strict=True))
            lanes = SLOT_BRANCHES[opcode] + (0,) * (len(slot.lanes) - len(SLOT_BRANCHES[opcode]))
            w.update(zip(slot.selectors + slot.lanes, tuple(int(i == opcode) for i in range(5)) + lanes, strict=True))
            assert not p.failures(w), p.failures(w)
            return LW.lower(p, w)

        cls.lowered = [build(op) for op in range(5)]
        cls.proofs = [LW.prove(c, a, rng=SeededRng(100 + op)) for op, (c, a) in enumerate(cls.lowered)]

    @staticmethod
    def blocks(container, used):
        """Aligned 64-byte blocks of the proof body: after the public header, before the zero padding."""
        return [bytes(container[i:i + 64]) for i in range(64, used, 64)]

    def test_five_architectures_give_byte_identical_lengths_and_public_shape(self):
        digests = {c.digest for c, _ in self.lowered}
        self.assertEqual(len(digests), 1)                           # the public circuit does not name the opcode
        self.assertEqual({len(p) for p in self.proofs}, {self.lowered[0][0].chunks})
        lengths = {len(data) for proof in self.proofs for data in proof}
        self.assertEqual(lengths, {self.c_p})
        headers = {bytes(data[:64]) for proof in self.proofs for data in proof}
        self.assertEqual(len(headers), 1)
        used = 64 + self.lowered[0][0].layout.body_used
        for proof in self.proofs:
            for data in proof:
                self.assertFalse(any(data[used:]))                  # zero padding only, no inner length field

    def test_different_architectures_share_no_body_block(self):
        used = 64 + self.lowered[0][0].layout.body_used
        for i in range(len(self.proofs[0])):
            seen = Counter()
            for proof in self.proofs:
                seen.update(set(self.blocks(proof[i], used)))
            self.assertEqual([b for b, n in seen.items() if n > 1], [])

    def test_reproving_shares_no_64_byte_block(self):
        circuit, assignment = self.lowered[2]
        used = 64 + circuit.layout.body_used
        first = self.proofs[2]
        again = self.LW.prove(circuit, assignment, rng=SeededRng(999))
        fresh = self.LW.prove(circuit, assignment)                  # OS randomness
        for a, b, c in zip(first, again, fresh, strict=True):
            self.assertEqual(len(a), len(b))
            self.assertEqual(bytes(a[:64]), bytes(b[:64]))          # public header is identical by design
            self.assertEqual(bytes(a[96:128]), bytes(b[96:128]))    # public N_s binding is identical by design
            blocks = [set(self.blocks(x, used)) for x in (a, b, c)]
            self.assertEqual(blocks[0] & blocks[1], set())
            self.assertEqual(blocks[0] & blocks[2], set())
            self.assertEqual(blocks[1] & blocks[2], set())
            # Inside one proof blocks do repeat (query openings share upper Merkle path nodes); that is
            # public structure of a single container, not linkage between two proofs.
            self.assertGreater(len(blocks[0]), 1000)

    def test_containers_verify_and_bind_to_the_circuit(self):
        """The compared containers are real proofs: the independent reference verifier accepts each one."""
        try:
            from fpn_verify_ref.verify import verify_result
        except ImportError as exc:
            self.skipTest(f"reference verifier unavailable: {exc}")
        for (circuit, _), proof in zip(self.lowered, self.proofs, strict=True):
            self.assertTrue(self.LW.check_binding(circuit, proof))
            for data in proof:
                self.assertEqual(verify_result(data), "VERIFIED")

    def test_same_randomness_reproduces_the_proof(self):
        """Control for the test above: the difference comes from the blinding, not from nondeterminism."""
        circuit, assignment = self.lowered[0]
        self.assertEqual(self.LW.prove(circuit, assignment, rng=SeededRng(100)), self.proofs[0])


if __name__ == "__main__":
    unittest.main()
