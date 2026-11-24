# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for format/rounding IR, resource budget, capture ports/ring/store/sink, and boundaries.

Run (stdlib only, no numpy / torch / pytest needed)::

    python3 -m unittest tests/unit/test_capture_and_format.py -v          # or:  pytest tests/unit/test_capture_and_format.py

Only the CPU mock backend and the host-emulated CUDA kernels are exercised here; the CUDA ring backend, NVRTC compile,
vLLM plugin and the Tier-1 bench require a CUDA GPU.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib.util
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "prover", ROOT / "serve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fpn_prover.ir import (  # noqa: E402
    CONTRACTS, KERNEL_ADAPTERS, MODEL_FORMATS, NUM_FORMATS, PROFILE_CAPS, Calibration, ContractCapError, FormatError,
    ModelFormat, NumFormat, QuantBlockScale, RoundingContract, ToleranceClass, UncalibratedContractError,
    get_contract,
)
from fpn_prover.runtime import BudgetError, ResourceBudget, validate_spill_dir  # noqa: E402
from fpn_serve.capture import (  # noqa: E402
    ASSUMED_400B_MOE, LLAMA3_70B, QWEN3_0_6B, QWEN3_30B_A3B, QWEN3_4B, CaptureConfig, CaptureError,
    CaptureLayout, CaptureOptions, CaptureSession, MoEDims, CaptureTier, ChunkAddress, ModelDims, PageMissingError,
    PageStore, PageStoreFullError, RequestPoisonedError, RingNotReadyError, RingOverflowError, RingStateError, Segment,
    SinkFaultedError, Span, StagingRing, StepRecord, StripeCoverageError, block_hash_chain, chain_seed, chunk_count,
    chunk_span, merge_row_stripes, reference_full_bytes_per_token, stripe_chunks,
)

# Banned names are assembled from pieces so this file itself stays clean of them.
BANNED = ("teth" + "er", "su" + "b5")

SERVE_DIR = ROOT / "serve"
PROVER_DIR = ROOT / "prover"
OWN_FILES = (
    [p for p in (PROVER_DIR / "fpn_prover").rglob("*.py")]
    + [p for p in SERVE_DIR.rglob("*") if p.suffix in (".py", ".cu")]
    + list((ROOT / "bench" / "tier1").glob("*.py"))
    + [Path(__file__)]
)


# ============================================================================================================== format IR
class FormatTests(unittest.TestCase):
    def test_unit_roundoffs_match_specs_06_table(self) -> None:
        got = {k: v.unit_roundoff_log2 for k, v in NUM_FORMATS.items()}
        self.assertEqual(got["bf16"], -8)
        self.assertEqual(got["fp16"], -11)
        self.assertEqual(got["fp32"], -24)
        self.assertEqual(got["fp8_e4m3"], -4)
        self.assertEqual(got["fp8_e5m2"], -3)
        self.assertIsNone(got["e8m0"])

    def test_exact_storage_widths(self) -> None:
        mf = MODEL_FORMATS
        self.assertEqual(mf["bf16"].weight.storage_bits_per_element, 16)
        self.assertEqual(mf["mxfp4_e2m1"].weight.storage_bits_per_element, Fraction(17, 4))          # 4 + 8/32
        self.assertEqual(mf["mxfp8_e4m3"].weight.storage_bits_per_element, Fraction(33, 4))          # 8 + 8/32
        self.assertEqual(mf["fp8_e4m3_block128"].activation.storage_bits_per_element, Fraction(33, 4))   # 8 + 32/128
        self.assertEqual(mf["fp8_e4m3_block128"].weight.storage_bits_per_element, 8 + Fraction(32, 128 * 128))
        self.assertEqual(mf["mxfp4_e2m1"].weight.storage_bytes(4096), 2048 + 128)                  # 128 e8m0 scales
        self.assertEqual(mf["bf16"].weight.storage_bytes(5), 10)
        with self.assertRaises(FormatError):
            mf["bf16"].weight.storage_bytes(-1)

    def test_mxfp4_commitment_is_about_3_8x_smaller_than_bf16(self) -> None:
        r = MODEL_FORMATS["mxfp4_e2m1"].weight_commit_ratio_vs(MODEL_FORMATS["bf16"])
        self.assertEqual(r, Fraction(64, 17))
        self.assertAlmostEqual(float(r), 3.76, places=2)

    def test_exact_dyadic_lift_widths(self) -> None:
        self.assertEqual(NUM_FORMATS["fp4_e2m1"].element_integer_bits, 5)       # {0..6} in 0.5 steps: |N| <= 12
        self.assertEqual(NUM_FORMATS["fp8_e4m3"].element_integer_bits, 19)      # 448 / 2^-9
        self.assertEqual(NUM_FORMATS["fp8_e4m3"].max_finite, 448)
        self.assertEqual(NUM_FORMATS["fp16"].max_finite, 65504)
        self.assertTrue(all(f.exact_dyadic for f in NUM_FORMATS.values()))
        self.assertTrue(MODEL_FORMATS["mxfp4_e2m1"].weight.scale.pow2)
        self.assertEqual(MODEL_FORMATS["mxfp4_e2m1"].weight.scale.scale_exp_range, (-127, 127))
        self.assertIsNone(MODEL_FORMATS["fp8_e4m3_block128"].weight.scale.scale_exp_range)

    def test_block_scale_validation(self) -> None:
        with self.assertRaises(FormatError):
            QuantBlockScale("fp64", 1, 32)
        with self.assertRaises(FormatError):
            QuantBlockScale("e8m0", 0, 32)
        self.assertEqual(QuantBlockScale("fp32", 128, 128).block_elems, 16384)

    def test_bf16_fp32acc_is_pinned_to_certified_constants(self) -> None:
        src = (ROOT / "flashprove_next" / "certified.py")
        if not src.exists():
            self.skipTest("certified.py not present")
        text = src.read_text(encoding="utf-8")

        def frac_log2(name: str) -> int:
            m = re.search(rf"^\s+{name}: Fraction = Fraction\(1, 1 << (\d+)\)", text, re.M)
            self.assertIsNotNone(m, name)
            return -int(m.group(1))

        def integer(name: str) -> int:
            m = re.search(rf"^\s+{name}: int = (-?\d+)", text, re.M)
            self.assertIsNotNone(m, name)
            return int(m.group(1))

        c = get_contract("bf16_fp32acc")
        self.assertEqual(c.store_u_log2, frac_log2("u_store"))
        self.assertEqual(c.accum_u_log2, frac_log2("u_acc"))
        self.assertEqual(c.elem_u_log2, frac_log2("u_elem"))
        self.assertEqual(c.p_u_log2, frac_log2("u_P"))
        self.assertEqual(c.depth_div, integer("h_div"))
        self.assertEqual(c.depth_add, integer("h_add"))
        self.assertEqual(c.attn_key_tile, integer("B_c"))
        self.assertEqual(c.split_kv_max, integer("split_steps"))
        self.assertEqual(c.eta_log2, integer("eta_log2"))
        self.assertEqual(c.depth(4096), 4096 // 16 + 64)

    def test_builtin_contract_calibration_is_honest(self) -> None:
        self.assertIs(CONTRACTS["bf16_fp32acc"].calibration, Calibration.CALIBRATED)
        CONTRACTS["bf16_fp32acc"].require_calibrated()
        for n in ("fp16_attn_strict", "fp8_e4m3_block128", "mxfp8_e4m3", "mxfp4_e2m1"):
            self.assertIs(CONTRACTS[n].calibration, Calibration.SCHEMA_ONLY, n)
            with self.assertRaises(UncalibratedContractError):
                CONTRACTS[n].require_calibrated()
        with self.assertRaises(UncalibratedContractError):
            MODEL_FORMATS["fp8_e4m3_block128"].require_calibrated()
        with self.assertRaises(FormatError):
            RoundingContract("x", ToleranceClass.R32, Calibration.CALIBRATED)      # calibrated needs evidence

    def test_schema_parameters_are_explicit(self) -> None:
        fp8 = CONTRACTS["fp8_e4m3_block128"]
        self.assertEqual(fp8.promote_k, 128)
        self.assertGreater(fp8.promote_u_log2, fp8.accum_u_log2)                   # TC accumulation is lossier than fp32
        for n in ("bf16_fp32acc", "fp16_attn_strict", "fp8_e4m3_block128", "mxfp8_e4m3", "mxfp4_e2m1"):
            c = CONTRACTS[n]
            self.assertGreaterEqual(c.split_k_max, 1)
            self.assertGreaterEqual(c.tp_reduce_max, 1)

    def test_cap_check_is_a_componentwise_partial_order(self) -> None:
        r32, f16 = CONTRACTS["bf16_fp32acc"], CONTRACTS["fp16_attn_strict"]
        self.assertTrue(r32 <= r32)
        self.assertTrue(f16 <= r32)                       # fp16 P/output is strictly tighter than bf16 P/output
        self.assertFalse(r32 <= f16)
        self.assertTrue(r32 >= f16)
        self.assertFalse(CONTRACTS["fp8_e4m3_block128"] <= r32)
        self.assertFalse(CONTRACTS["mxfp4_e2m1"] <= r32)                           # tolerance class BLOCK_MX > R32
        # incomparable: tighter on one axis, looser on another (a lexicographic order would call one of them <=)
        a = RoundingContract.from_dict({**r32.to_dict(), "name": "a", "calibration": "SCHEMA_ONLY", "calibrated_on": [],
                                        "tp_reduce_max": 4, "tp_reduce_u_log2": -24, "store_u_log2": -11})   # looser fan-in, tighter store
        self.assertFalse(a <= r32)
        self.assertFalse(r32 <= a)
        self.assertFalse(a >= r32 and r32 >= a)
        tighter = RoundingContract.from_dict({**r32.to_dict(), "name": "t", "calibration": "SCHEMA_ONLY", "calibrated_on": [],
                                              "store_u_log2": -11})
        self.assertTrue(tighter <= r32)
        self.assertFalse(r32 <= tighter)

    def test_default_bf16_split_k_violates_the_r32_cap(self) -> None:
        r32 = CONTRACTS["bf16_fp32acc"]
        splitk = RoundingContract.from_dict({
            **r32.to_dict(), "name": "bf16_splitk", "tol_class": "BF16_SPLIT_K", "calibration": "SCHEMA_ONLY",
            "calibrated_on": [], "split_k_max": 8, "split_partial_u_log2": -8})
        bad = splitk.violations(r32)
        self.assertTrue(any("tol_class" in v for v in bad))
        self.assertTrue(any("effective_split_k" in v for v in bad))
        self.assertTrue(any("split_partial_u_log2" in v for v in bad))
        with self.assertRaises(ContractCapError):
            splitk.assert_within(r32)
        # the bound derived from the contract reflects the slack that specs/06 measured (31-39x on real kernels)
        ratio = splitk.dot_relative_bound(3072) / r32.dot_relative_bound(3072)
        self.assertGreater(ratio, 1)

    def test_contract_caps_on_model_formats(self) -> None:
        cap = PROFILE_CAPS["r32_default"]
        MODEL_FORMATS["bf16"].check_within(cap)
        MODEL_FORMATS["bf16_fp16attn"].check_within(cap)
        for n in ("fp8_e4m3_block128", "mxfp8_e4m3", "mxfp4_e2m1"):
            with self.assertRaises(ContractCapError, msg=n):
                MODEL_FORMATS[n].check_within(cap)
        # a profile that explicitly allows the fp8 tier admits it (class upper bound, not a hardware name)
        from fpn_prover.ir import ProfileCap
        fp8cap = ProfileCap("fp8_ok", CONTRACTS["fp8_e4m3_block128"], CONTRACTS["bf16_fp32acc"])
        MODEL_FORMATS["fp8_e4m3_block128"].check_within(fp8cap)
        MODEL_FORMATS["bf16"].check_within(fp8cap)

    def test_dot_bound_ordering_and_exactness(self) -> None:
        b = {n: c.dot_relative_bound(4096) for n, c in CONTRACTS.items()}
        for v in b.values():
            self.assertIsInstance(v, Fraction)
        self.assertLess(b["bf16_fp32acc"], b["fp8_e4m3_block128"])
        # R32 bound ~ gamma_h(h=4096/16+64) + store u: dominated by the bf16 store
        self.assertAlmostEqual(float(b["bf16_fp32acc"]), 2 ** -8 + 320 * 2 ** -24, delta=1e-5)

    def test_tp_reduce_fan_in_enters_the_bound(self) -> None:
        r32 = CONTRACTS["bf16_fp32acc"]
        tp = RoundingContract.from_dict({**r32.to_dict(), "name": "tp8", "calibration": "SCHEMA_ONLY", "calibrated_on": [],
                                         "tp_reduce_max": 8, "tp_reduce_u_log2": -8})
        self.assertGreater(tp.dot_relative_bound(4096), r32.dot_relative_bound(4096))
        self.assertFalse(tp <= r32)

    def test_dict_roundtrip_and_validation(self) -> None:
        for c in CONTRACTS.values():
            self.assertEqual(RoundingContract.from_dict(c.to_dict()), c)
        d = CONTRACTS["mxfp4_e2m1"].to_dict()
        d["surprise"] = 1
        with self.assertRaises(FormatError):
            RoundingContract.from_dict(d)
        with self.assertRaises(FormatError):
            RoundingContract.from_dict({**CONTRACTS["mxfp4_e2m1"].to_dict(), "split_k_max": 0})
        with self.assertRaises(FormatError):
            get_contract("nope")

    def test_kernel_adapters_bind_formats_to_contracts(self) -> None:
        a = KERNEL_ADAPTERS["vllm_cublas_bf16_r32"]
        self.assertTrue(a.supports(MODEL_FORMATS["bf16"]))
        self.assertFalse(a.supports(MODEL_FORMATS["mxfp4_e2m1"]))
        self.assertTrue(a.satisfies(PROFILE_CAPS["r32_default"]))
        self.assertFalse(KERNEL_ADAPTERS["vllm_deepgemm_fp8_block128"].satisfies(PROFILE_CAPS["r32_default"]))
        self.assertTrue(KERNEL_ADAPTERS["vllm_deepgemm_fp8_block128"].supports(MODEL_FORMATS["fp8_e4m3_block128"]))

    def test_frozen(self) -> None:
        with self.assertRaises(Exception):
            CONTRACTS["bf16_fp32acc"].promote_k = 7            # type: ignore[misc]
        with self.assertRaises(Exception):
            NUM_FORMATS["bf16"].bits = 8                        # type: ignore[misc]
        self.assertIsInstance(MODEL_FORMATS["bf16"], ModelFormat)
        self.assertIsInstance(NUM_FORMATS["bf16"], NumFormat)


# ============================================================================================================ ResourceBudget
class BudgetTests(unittest.TestCase):
    def test_rejects_tmp_and_ram_backed_locations(self) -> None:
        for bad in ("/tmp", "/tmp/fpn", "/private/tmp/fpn-spill", "/var/tmp/x", "/dev/shm/spill"):
            with self.assertRaises(BudgetError, msg=bad):
                ResourceBudget(spill_dir=bad)

    def test_rejects_tmpfs_mounts_from_injected_mount_table(self) -> None:
        mounts = "/dev/sdc / ext4 rw 0 0\ntmpfs /data/ram tmpfs rw 0 0\ntmpfs /data/ram/inner/disk ext4 rw 0 0\n"
        with self.assertRaises(BudgetError):
            validate_spill_dir("/data/ram/spill", mounts_text=mounts)
        validate_spill_dir("/data/ram/inner/disk/spill", mounts_text=mounts)            # innermost mount wins
        validate_spill_dir("/data/other", mounts_text=mounts)

    def test_accepts_home_cache_default(self) -> None:
        b = ResourceBudget()
        self.assertTrue(str(b.spill_path).endswith(os.path.join(".cache", "fpn-spill")))

    def test_tmp_symlink_cannot_smuggle_spill_dir(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as d:
            link = Path(d) / "to_tmp"
            try:
                link.symlink_to("/tmp")
            except OSError:
                self.skipTest("cannot symlink")
            with self.assertRaises(BudgetError):
                validate_spill_dir(link / "x")

    def test_presets(self) -> None:
        a, h = ResourceBudget.preset("4070ti_super_16gb"), ResourceBudget.preset("h100_80gb")
        self.assertLess(a.vram_limit_gib, 16)
        self.assertLess(a.host_ram_limit_gib, 25)
        self.assertEqual(a.h2d_chunk_mib, 64)
        self.assertEqual(h.h2d_chunk_mib, 64)
        self.assertEqual(h.vram_limit_gib, 80.0)
        rtx = ResourceBudget.preset("rtx6000d_84gb")
        self.assertEqual(rtx.vram_limit_gib, 83.0)
        self.assertEqual(rtx.host_ram_limit_gib, 896.0)
        for budget in (rtx, h):
            self.assertEqual(budget.spill_path, validate_spill_dir(budget.spill_dir))
        self.assertLessEqual(a.staging_ring_mib, 350)
        with self.assertRaises(BudgetError):
            ResourceBudget.preset("tpu")

    def test_chunk_plan_never_exceeds_64mib(self) -> None:
        b = ResourceBudget()
        plan = b.chunk_plan(200 * (1 << 20) + 5)
        self.assertEqual(sum(n for _, n in plan), 200 * (1 << 20) + 5)
        self.assertTrue(all(n <= 64 * (1 << 20) for _, n in plan))
        self.assertEqual([o for o, _ in plan], [0, 64 << 20, 128 << 20, 192 << 20])
        with self.assertRaises(BudgetError):
            ResourceBudget(h2d_chunk_mib=65)

    def test_ring_sizing_and_validation(self) -> None:
        b = ResourceBudget.preset("4070ti_super_16gb")
        rows = b.ring_rows(2_042_368)
        self.assertEqual(rows, (320 << 20) // 2_042_368)
        self.assertEqual(b.check_ring_fits(2_042_368, capacity_rows=rows), rows * 2_042_368)
        with self.assertRaises(BudgetError):
            b.check_ring_fits(2_042_368, capacity_rows=rows + 1, tier_cap_mib=320)
        with self.assertRaises(BudgetError):
            ResourceBudget(vram_limit_gib=1.0, staging_ring_mib=512)

    def test_from_env_reads_only_the_injected_mapping(self) -> None:
        env = {"FPN_PROFILE": "h100_80gb", "FPN_STAGING_RING_MIB": "100", "FPN_VRAM_LIMIT_GIB": "70.5"}
        before = dict(os.environ)
        b = ResourceBudget.from_env(env)
        self.assertEqual((b.staging_ring_mib, b.vram_limit_gib, b.profile), (100, 70.5, "h100_80gb"))
        self.assertEqual(dict(os.environ), before)
        with self.assertRaises(BudgetError):
            ResourceBudget.from_env({"FPN_SPILL_DIR": "/tmp/spill"})
        with self.assertRaises(BudgetError):
            ResourceBudget.from_env({"FPN_H2D_CHUNK_MIB": "lots"})

    def test_environment_is_read_only_in_budget_py(self) -> None:
        hits = []
        for p in OWN_FILES:
            if p.suffix != ".py" or p == Path(__file__):
                continue
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in ("environ", "getenv", "putenv"):
                    if p.name != "budget.py" and "bench" not in p.parts:
                        hits.append(f"{p.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(hits, [], "only prover/fpn_prover/runtime/budget.py (and the bench driver) may touch os.environ")


# ===================================================================================================== ports / layout math
def _tier_b(dims: ModelDims, h_mid: bool) -> CaptureLayout:
    return CaptureLayout.build(dims, CaptureOptions(CaptureTier.TIER_B, include_h_mid=h_mid))


class PortsTests(unittest.TestCase):
    def test_tier_b_bytes_reproduce_memo_headline_numbers(self) -> None:
        # memo 03 §1 numbers equal (L+1) * d * 2 : one residual per layer + h_L
        for dims, exact in ((QWEN3_4B, 189_440), (QWEN3_30B_A3B, 201_472), (LLAMA3_70B, 1_327_104), (ASSUMED_400B_MOE, 501_856),
                            (QWEN3_0_6B, 59_392)):
            lay = _tier_b(dims, h_mid=False)
            res = (dims.n_layers + 1) * dims.d_model * 2
            ids = dims.n_layers * dims.moe.top_k * 2 if dims.moe else 0
            self.assertEqual(lay.payload_bytes_per_token, res + ids, dims.name)
            self.assertEqual(lay.payload_bytes_per_token, exact, dims.name)
        self.assertEqual(round(_tier_b(QWEN3_4B, False).payload_bytes_per_token / 1000), 189)
        self.assertEqual(round(_tier_b(QWEN3_30B_A3B, False).payload_bytes_per_token / 1000), 201)
        self.assertEqual(round(_tier_b(LLAMA3_70B, False).payload_bytes_per_token / 1e6, 2), 1.33)
        self.assertEqual(round(_tier_b(ASSUMED_400B_MOE, False).payload_bytes_per_token / 1000), 502)

    def test_as_specified_tier_b_with_h_mid_is_2l_plus_1(self) -> None:
        """The memo text lists h_in/h_mid/h_L but its numbers count one residual per layer: both are reproducible."""
        lay = _tier_b(QWEN3_4B, h_mid=True)
        self.assertEqual(lay.payload_bytes_per_token, (2 * 36 + 1) * 2560 * 2)
        self.assertEqual(lay.payload_bytes_per_token, 373_760)
        self.assertEqual(CaptureLayout.build(QWEN3_4B, CaptureOptions(CaptureTier.TIER_B)).options.include_h_mid, True)
        self.assertEqual(set(lay.breakdown()), {"h_in", "h_mid", "h_L"})

    def test_full_no_kv_bytes_per_token_formula(self) -> None:
        for dims in (QWEN3_0_6B, QWEN3_4B):
            D, nq, dh = dims.d_model, dims.n_q_heads, dims.head_dim
            per_layer = 4 * D + 2 * nq * dh + dims.d_ff + 2 * nq     # h_in x_attn h_mid x_mlp | q o | a | lse(fp32)
            want = (dims.n_layers * per_layer + 2 * D) * 2            # + x_f, h_L
            lay = CaptureLayout.build(dims)
            self.assertEqual(lay.payload_bytes_per_token, want, dims.name)
        self.assertEqual(CaptureLayout.build(QWEN3_4B).payload_bytes_per_token, 2_042_368)
        self.assertEqual(CaptureLayout.build(QWEN3_0_6B).payload_bytes_per_token, 636_672)

    def test_no_kv_is_ever_captured_and_savings_are_reported(self) -> None:
        for tier in CaptureTier:
            lay = CaptureLayout.build(QWEN3_4B, CaptureOptions(tier))
            self.assertFalse({"k", "v"} & {s.kind for s in lay.slots}, tier)
        kv = reference_full_bytes_per_token(QWEN3_4B, with_kv=True)
        nokv = reference_full_bytes_per_token(QWEN3_4B, with_kv=False)
        self.assertEqual(kv - nokv, 36 * 2 * 8 * 128 * 2)
        self.assertEqual(nokv, CaptureLayout.build(QWEN3_4B).payload_bytes_per_token)
        for dims in (QWEN3_4B, QWEN3_30B_A3B, LLAMA3_70B, ASSUMED_400B_MOE):
            self.assertGreater(reference_full_bytes_per_token(dims) / _tier_b(dims, False).payload_bytes_per_token, 7.5, dims.name)

    def test_norm_outputs_are_a_flag_not_a_constant(self) -> None:
        on = CaptureLayout.build(QWEN3_4B)
        off = CaptureLayout.build(QWEN3_4B, CaptureOptions(CaptureTier.FULL_NO_KV, capture_norm_outputs=False))
        self.assertFalse({"x_attn", "x_mlp", "x_f"} & {s.kind for s in off.slots})
        self.assertEqual(on.payload_bytes_per_token - off.payload_bytes_per_token, (2 * 36 + 1) * 2560 * 2)
        self.assertNotEqual(on.fingerprint(), off.fingerprint())
        self.assertNotEqual(on.fingerprint(), _tier_b(QWEN3_4B, True).fingerprint())

    def test_payload_vs_stride_and_alignment(self) -> None:
        odd = ModelDims(3, 36, 3, 1, 12, 50)           # widths not multiples of 8 on purpose
        lay = CaptureLayout.build(odd)
        self.assertGreater(lay.stride_bytes, lay.payload_bytes_per_token)
        self.assertEqual(lay.stride_bytes % 16, 0)
        self.assertTrue(all(s.offset % 8 == 0 for s in lay.slots))
        end = 0
        for s in lay.slots:                              # slots are ordered, disjoint, inside the row
            self.assertGreaterEqual(s.offset, end)
            end = s.offset + s.width
        self.assertLessEqual(end, lay.row_u16)
        self.assertEqual(sum(lay.breakdown().values()), lay.payload_bytes_per_token)

    def test_moe_expert_ids_and_kv_reference(self) -> None:
        lay = CaptureLayout.build(QWEN3_30B_A3B, CaptureOptions(CaptureTier.TIER_B, include_h_mid=False))
        self.assertEqual(lay.breakdown()["expert_ids"], 48 * 8 * 2)
        self.assertEqual(lay.kv_ref_bytes_per_token, 2.0)                 # 32 B block hash per 16-token page
        self.assertEqual(CaptureLayout.build(QWEN3_4B).kv_ref_bytes_per_token, 0.0)
        with self.assertRaises(CaptureError):
            ModelDims(2, 8, 1, 1, 8, 8, MoEDims(70000, 2, 8))                  # expert ids are captured as u16

    def test_fused_moe_layout_drops_act_explicitly_and_keeps_default_fingerprints(self) -> None:
        full = CaptureLayout.build(QWEN3_30B_A3B)
        no_act = CaptureLayout.build(QWEN3_30B_A3B, CaptureOptions(capture_act=False))
        self.assertTrue(full.has("a", 0))
        self.assertFalse(any(s.kind == "a" for s in no_act.slots))
        self.assertTrue(all(no_act.has("expert_ids", l) for l in range(48)))
        self.assertEqual(full.payload_bytes_per_token - no_act.payload_bytes_per_token, 48 * 8 * 768 * 2)
        self.assertNotEqual(full.fingerprint(), no_act.fingerprint())
        # the default options are unchanged: same slots and fingerprint as an explicit capture_act=True
        self.assertEqual(full.fingerprint(), CaptureLayout.build(QWEN3_30B_A3B, CaptureOptions(capture_act=True)).fingerprint())
        self.assertEqual(CaptureConfig().options(), CaptureOptions())
        self.assertFalse(CaptureConfig(capture_act=False).options().capture_act)
        with self.assertRaises(CaptureError):                                # LayoutError is a CaptureError
            CaptureLayout.build(QWEN3_30B_A3B, CaptureOptions(CaptureTier.FULL_NO_KV_ACT, capture_act=False))

    def test_from_hf_config(self) -> None:
        d = ModelDims.from_hf_config({"num_hidden_layers": 36, "hidden_size": 2560, "num_attention_heads": 32,
                                      "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9728})
        self.assertEqual((d.n_layers, d.d_model, d.n_q_heads, d.n_kv_heads, d.head_dim, d.d_ff), (36, 2560, 32, 8, 128, 9728))
        moe = ModelDims.from_hf_config({"num_hidden_layers": 48, "hidden_size": 2048, "num_attention_heads": 32,
                                        "num_key_value_heads": 4, "head_dim": 128, "num_experts": 128,
                                        "num_experts_per_tok": 8, "moe_intermediate_size": 768, "intermediate_size": 6144})
        self.assertEqual(moe.moe.top_k, 8)
        self.assertEqual(moe.act_width, 8 * 768)

    def test_chunk_math(self) -> None:
        self.assertEqual(chunk_count(128), 1)
        self.assertEqual(chunk_count(129), 2)
        self.assertEqual(chunk_span(300, 2), (256, 300))
        with self.assertRaises(CaptureError):
            chunk_span(300, 3)
        self.assertEqual(stripe_chunks(300, 100, 260), [(0, 100, 128), (1, 128, 256), (2, 256, 260)])
        with self.assertRaises(CaptureError):
            stripe_chunks(300, 0, 301)

    def test_tp_ep_pp_stripes_merge_to_one_logical_row_without_leaking_topology(self) -> None:
        width = 4096 + 40
        row = bytes((i * 7 + 3) & 0xFF for i in range(width * 2))
        canonical = {c: row[lo * 2:hi * 2] for c, lo, hi in ((c, *chunk_span(width, c)) for c in range(chunk_count(width)))}
        for tp in (1, 2, 3, 4, 8):                        # same logical tensor for every shard count (incl. misaligned splits)
            bounds = [width * i // tp for i in range(tp + 1)]
            pieces = [(bounds[i], row[bounds[i] * 2:bounds[i + 1] * 2]) for i in range(tp)]
            self.assertEqual(merge_row_stripes(width, reversed(pieces)), row, tp)
            seen = {}                                      # every stripe splits at canonical chunk boundaries
            for i in range(tp):
                for c, lo, hi in stripe_chunks(width, bounds[i], bounds[i + 1]):
                    seen.setdefault(c, bytearray())
                    seen[c] += row[lo * 2:hi * 2]
            self.assertEqual({c: bytes(b) for c, b in seen.items()}, canonical, tp)
        with self.assertRaises(StripeCoverageError):       # gap
            merge_row_stripes(8, [(0, bytes(8)), (5, bytes(6))])
        with self.assertRaises(StripeCoverageError):       # overlap
            merge_row_stripes(8, [(0, bytes(10)), (4, bytes(8))])
        with self.assertRaises(StripeCoverageError):       # short
            merge_row_stripes(8, [(0, bytes(8))])

    def test_chunk_addresses_are_a_bijection_and_groups_tile_the_row(self) -> None:
        lay = CaptureLayout.build(ModelDims(4, 160, 2, 1, 40, 200))
        idx = {}
        for page in range(3):
            for layer in range(lay.n_layer_rows):
                for ch in range(lay.layer_chunks(layer)):
                    idx[lay.chunk_linear_index(ChunkAddress(page, layer, ch))] = (page, layer, ch)
        self.assertEqual(sorted(idx), list(range(3 * lay.chunks_per_page)))
        with self.assertRaises(CaptureError):
            lay.chunk_linear_index(ChunkAddress(0, 99, 0))
        for gs in (0, 1, 2, 3):
            groups = lay.layer_groups(gs)
            self.assertEqual(groups[0].col_lo, 0)
            self.assertEqual(groups[-1].col_hi, lay.row_u16)
            for a, b in zip(groups, groups[1:]):
                self.assertEqual(a.col_hi, b.col_lo)
                self.assertEqual(a.layer_hi, b.layer_lo)
        c0, c1 = lay.chunk_columns(1, 0)
        self.assertEqual(c0, lay.layer_base[1])


# =========================================================================================================== staging ring
def _rows(ring: StagingRing, first_id: int, n: int) -> bytes:
    """Row-unique content: every row embeds its global id, so any mix-up / stale read is detected exactly."""
    out = bytearray()
    for i in range(first_id, first_id + n):
        out += struct.pack(f"<{ring.row_u16}H", *[(i * 131 + j * 7 + 1) & 0xFFFF for j in range(ring.row_u16)])
    return bytes(out)


class RingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = StagingRing.cpu(capacity_rows=7, row_u16=6)       # 7 is not a multiple of the step sizes below

    def test_backend_memory_starts_poisoned_not_zero(self) -> None:
        buf = bytes(self.ring.backend.buf)
        self.assertNotIn(b"\x00", buf)
        self.assertEqual(buf[:2], b"\xa5\xa5")

    def test_roundtrip_wraparound_and_exact_contents(self) -> None:
        ring, nid = self.ring, 0
        for step in (3, 3, 3, 2, 3, 3, 3):                           # 20 rows through a 7-row ring: wraps several times
            span = ring.reserve(step)
            self.assertEqual(span.start, nid)
            ring.publish(span, _rows(ring, nid, step))
            self.assertEqual(ring.read(span), _rows(ring, nid, step))
            ring.release(span.end)
            nid += step
        self.assertEqual((ring.write_wp, ring.ready_wp, ring.copied_wp), (20, 20, 20))
        self.assertGreater(ring.write_wp, ring.capacity_rows * 2)

    def test_wrap_with_multiple_unreleased_spans_in_flight(self) -> None:
        ring = self.ring
        a = ring.reserve(3); ring.publish(a, _rows(ring, 0, 3))
        b = ring.reserve(3); ring.publish(b, _rows(ring, 3, 3))
        ring.release(a.end)
        c = ring.reserve(3); ring.publish(c, _rows(ring, 6, 3))        # physically wraps (rows 6,0,1)
        self.assertEqual(ring.read(b), _rows(ring, 3, 3))
        self.assertEqual(ring.read(c), _rows(ring, 6, 3))

    def test_overflow_is_refused_without_changing_anything(self) -> None:
        ring = self.ring
        a = ring.reserve(5); ring.publish(a, _rows(ring, 0, 5))
        snapshot = (ring.write_wp, ring.ready_wp, ring.copied_wp, bytes(ring.backend.buf))
        with self.assertRaises(RingOverflowError):
            ring.reserve(3)                                            # 5 + 3 - 0 > 7
        self.assertEqual((ring.write_wp, ring.ready_wp, ring.copied_wp, bytes(ring.backend.buf)), snapshot)
        self.assertEqual(ring.overflow_events, 1)
        self.assertEqual(ring.read(a), _rows(ring, 0, 5))              # earlier rows untouched
        ring.release(a.end)
        ring.publish(ring.reserve(3), _rows(ring, 5, 3))               # fits again after release

    def test_single_step_larger_than_ring_always_raises(self) -> None:
        with self.assertRaises(RingOverflowError):
            self.ring.reserve(8)
        with self.assertRaises(RingOverflowError):
            self.ring.require_step_capacity(8)
        self.ring.require_step_capacity(7)

    def test_touched_rows_exceeding_committed_count_toward_the_invariant(self) -> None:
        ring = self.ring
        a = ring.reserve(2, touched_rows=5)                           # graph replay launched with the padded size
        self.assertEqual(a.n_rows, 2)
        ring.publish(a, _rows(ring, 0, 2))
        with self.assertRaises(RingOverflowError):
            ring.reserve(1, touched_rows=6)                            # wp=2, touched to 8, copied=0, cap=7
        ring.reserve(1, touched_rows=5)                                # 2 + 5 - 0 = 7 fits exactly
        with self.assertRaises(RingStateError):
            ring.reserve(3, touched_rows=2)

    def test_reads_never_return_unwritten_overwritten_or_released_rows(self) -> None:
        ring = self.ring
        a = ring.reserve(4)
        with self.assertRaises(RingNotReadyError):
            ring.read(a)                                               # reserved but never published: no poison bytes leak
        ring.publish(a, _rows(ring, 0, 4))
        b = ring.reserve(3, poison_on_overflow=True); ring.publish(b, _rows(ring, 4, 3))
        c = ring.reserve(3, poison_on_overflow=True)                   # forced through: overwrites rows 0..2 of `a`
        self.assertEqual(ring.lost_below, 3)
        self.assertEqual(ring.overflow_events, 1)
        ring.publish(c, _rows(ring, 7, 3))
        with self.assertRaises(RingOverflowError):
            ring.read(a)                                               # lost rows: raise, never a partial / stale answer
        with self.assertRaises(RingOverflowError):
            ring.read(Span(2, 2))
        self.assertEqual(ring.read(Span(3, 1)), _rows(ring, 3, 1))     # row 3 survived and is exact
        ring.release(3 + 1)
        with self.assertRaises(RingStateError):
            ring.read(Span(0, 1))                                      # released
        self.assertEqual(ring.read(c), _rows(ring, 7, 3))

    def test_no_read_ever_succeeds_with_foreign_content(self) -> None:
        """Randomised schedule: every successful read equals what was written; every other read raises."""
        import random
        rng = random.Random(1234)
        ring = StagingRing.cpu(capacity_rows=9, row_u16=4)
        written: dict[int, bytes] = {}
        live: list[Span] = []
        nid = 0
        for _ in range(400):
            op = rng.choice("rrrwd")
            if op in "rw":
                n = rng.randint(1, 5)
                try:
                    sp = ring.reserve(n, poison_on_overflow=(op == "w"))
                except RingOverflowError:
                    continue
                data = _rows(ring, nid, n)
                ring.publish(sp, data)
                for i in range(n):
                    written[nid + i] = data[i * ring.row_bytes:(i + 1) * ring.row_bytes]
                live.append(sp)
                nid += n
            elif live:
                if rng.random() < 0.4:
                    ring.release(live.pop(0).end)
                else:
                    sp = rng.choice(live)
                    try:
                        got = ring.read(sp)
                    except (RingOverflowError, RingStateError, RingNotReadyError):
                        continue
                    self.assertEqual(got, b"".join(written[sp.start + i] for i in range(sp.n_rows)))
                    self.assertNotIn(b"\xa5\xa5" * ring.row_u16, got)
            live = [s for s in live if s.start >= ring.copied_wp]

    def test_watermark_protocol_errors(self) -> None:
        ring = self.ring
        sp = ring.reserve(2)
        with self.assertRaises(RingStateError):
            ring.release(2)                                            # not ready yet
        ring.publish(sp, _rows(ring, 0, 2))
        ring.release(2)
        with self.assertRaises(RingStateError):
            ring.release(1)                                            # never backwards
        with self.assertRaises(RingStateError):
            ring.mark_ready(99)
        with self.assertRaises(RingStateError):
            ring.publish(ring.reserve(1), b"short")

    def test_64bit_watermark_is_refused_not_wrapped(self) -> None:
        ring = self.ring
        ring.write_wp = ring.ready_wp = ring.copied_wp = ring._touched_hwm = (1 << 63) - 2
        with self.assertRaises(RingOverflowError):
            ring.reserve(3)
        self.assertEqual(ring.write_wp, (1 << 63) - 2)

    def test_free_rows_and_split(self) -> None:
        ring = self.ring
        self.assertEqual(ring.free_rows, 7)
        ring.reserve(4)
        self.assertEqual(ring.free_rows, 3)
        self.assertEqual(Span(5, 7).split(3), [Span(5, 3), Span(8, 3), Span(11, 1)])
        with self.assertRaises(RingStateError):
            Span(0, 1).split(0)


# ========================================================================================================= page store
def tiny_layout(tier: CaptureTier = CaptureTier.FULL_NO_KV) -> CaptureLayout:
    return CaptureLayout.build(ModelDims(2, 16, 2, 1, 8, 24), CaptureOptions(tier))


def row_for(layout: CaptureLayout, running_hash: bytes) -> bytes:
    rb = layout.stride_bytes
    out = b""
    c = 0
    while len(out) < rb:
        out += hashlib.blake2b(running_hash + bytes([c]), digest_size=32).digest()
        c += 1
    return out[:rb]


def rows_for_conversation(layout: CaptureLayout, conv: list[int], start: int, end: int) -> bytes:
    """Deterministic engine: row(p) depends on the whole prefix conv[:p+1] (causality => prefix-identical rows)."""
    h = b"\x00" * 32
    out = []
    for p, t in enumerate(conv[:end]):
        h = hashlib.blake2b(h + struct.pack("<I", t), digest_size=32).digest()
        if p >= start:
            out.append(row_for(layout, h))
    return b"".join(out)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lay = tiny_layout()
        self.store = PageStore(self.lay)

    def test_hash_chain_is_deterministic_prefix_consistent_and_layout_seeded(self) -> None:
        a = block_hash_chain(self.store.seed, list(range(64)))
        b = block_hash_chain(self.store.seed, list(range(64)) + [9, 9, 9])
        self.assertEqual(a, b)                                          # trailing partial block has no hash
        c = block_hash_chain(self.store.seed, [7] + list(range(1, 64)))
        self.assertNotEqual(a[0], c[0])
        self.assertNotEqual(a[3], c[3])                                 # chained: an early change propagates
        other = PageStore(tiny_layout(CaptureTier.TIER_B))
        self.assertNotEqual(other.seed, self.store.seed)
        self.assertNotEqual(block_hash_chain(other.seed, list(range(64)))[0], a[0])
        self.assertNotEqual(chain_seed(self.lay.fingerprint(), b"tenant-a"), chain_seed(self.lay.fingerprint(), b"tenant-b"))

    def test_write_once_first_write_wins_without_byte_comparison(self) -> None:
        h = block_hash_chain(self.store.seed, list(range(16)))[0]
        first = bytes([1]) * (16 * self.lay.stride_bytes)
        second = bytes([2]) * (16 * self.lay.stride_bytes)           # a 1-ulp-different honest recompute
        self.assertTrue(self.store.put_page(h, None, 16, first))
        self.assertFalse(self.store.put_page(h, None, 16, second))
        self.assertEqual(self.store.get_rows(h)[1], first)
        self.assertEqual(self.store.stats["dedup_hits"], 1)
        self.assertEqual(self.store.stats["pages_written"], 1)
        self.assertEqual(self.store.bytes_stored, len(first))

    def test_missing_pages_raise_never_zero_fill(self) -> None:
        with self.assertRaises(PageMissingError):
            self.store.get_rows(b"\x01" * 32)
        with self.assertRaises(PageMissingError):
            self.store.walk_to_root(b"\x02" * 32)
        with self.assertRaises(PageMissingError):
            self.store.put_page(b"\x03" * 32, b"\x04" * 32, 16, bytes(16 * self.lay.stride_bytes))   # unlinked parent
        with self.assertRaises(PageMissingError):
            self.store.pin(b"\x05" * 32)

    def test_layer_group_parts_roundtrip_exactly(self) -> None:
        for gs in (0, 1, 2):
            store = PageStore(self.lay, layer_group_size=gs)
            rows = rows_for_conversation(self.lay, list(range(16)), 0, 16)
            h = block_hash_chain(store.seed, list(range(16)))[0]
            store.put_page(h, None, 16, rows)
            self.assertEqual(store.get_rows(h), (16, rows))
            self.assertEqual(len(store._pages[h].parts), len(store.groups))
        with self.assertRaises(ValueError):
            self.store.put_page(h, None, 16, b"short")

    def test_lru_eviction_picks_the_least_recent_unpinned_leaf_and_never_an_ancestor(self) -> None:
        page = 16 * self.lay.stride_bytes
        store = PageStore(self.lay, max_bytes=3 * page)
        evicted: list[bytes] = []
        store.on_evict = evicted.append
        chain = block_hash_chain(store.seed, list(range(32)))            # root -> leaf (linked)
        solo = block_hash_chain(store.seed, list(range(500, 516)))[0]    # independent leaf
        store.put_page(chain[0], None, 16, b"\x01" * page)
        store.put_page(chain[1], chain[0], 16, b"\x02" * page)
        store.put_page(solo, None, 16, b"\x03" * page)
        store.get_rows(chain[1])                                          # touch: `solo` becomes the LRU leaf
        store.get_rows(chain[0])
        new = block_hash_chain(store.seed, list(range(900, 916)))[0]
        store.put_page(new, None, 16, b"\x04" * page)                    # over budget -> evict an evictable leaf
        self.assertEqual(evicted, [solo])
        self.assertTrue(store.has(chain[0]) and store.has(chain[1]))     # the ancestor was never a candidate
        self.assertLessEqual(store.bytes_stored, 3 * page)

    def test_store_budget_exhausted_by_pinned_pages_fails_closed(self) -> None:
        page = 16 * self.lay.stride_bytes
        store = PageStore(self.lay, max_bytes=2 * page)
        h1 = block_hash_chain(store.seed, list(range(16)))[0]
        h2 = block_hash_chain(store.seed, list(range(100, 116)))[0]
        h3 = block_hash_chain(store.seed, list(range(200, 216)))[0]
        store.put_page(h1, None, 16, b"\x01" * page)
        store.put_page(h2, None, 16, b"\x02" * page)
        store.pin(h1); store.pin(h2)
        with self.assertRaises(PageStoreFullError):
            store.put_page(h3, None, 16, b"\x03" * page)
        self.assertFalse(store.has(h3))
        store.unpin(h1)
        store.put_page(h3, None, 16, b"\x03" * page)                      # unpinned leaf is LRU-evicted to make room
        self.assertFalse(store.has(h1))
        self.assertTrue(store.has(h2) and store.has(h3))

    def test_kv_eviction_hook_defers_until_the_page_is_a_free_leaf(self) -> None:
        page = 16 * self.lay.stride_bytes
        store = PageStore(self.lay)
        hs = block_hash_chain(store.seed, list(range(48)))
        store.put_page(hs[0], None, 16, b"\x01" * page)
        store.put_page(hs[1], hs[0], 16, b"\x02" * page)
        store.pin(hs[1])
        events = []
        store.on_evict = events.append
        self.assertFalse(store.evict(hs[0]))                              # has a child: deferred
        self.assertFalse(store.evict(hs[1]))                              # pinned: deferred
        self.assertTrue(store.has(hs[0]) and store.has(hs[1]))
        store.unpin(hs[1])                                                # leaf frees -> cascades to the deferred parent
        self.assertEqual(events, [hs[1], hs[0]])
        self.assertEqual(store.n_pages(), 0)
        self.assertEqual(store.bytes_stored, 0)
        self.assertFalse(store.evict(hs[0]))                              # already gone

    def test_request_handle_state_is_independent_of_context_length(self) -> None:
        sizes, refs = [], []
        for pages in (4, 400):
            lay = self.lay
            store = PageStore(lay)
            h = store.open_request(f"r{pages}")
            toks = list(range(16 * pages + 5))
            rows = rows_for_conversation(lay, toks, 0, len(toks))
            for off in range(0, len(toks), 64):                           # stream it in engine-step sized pieces
                h.append(rows[off * lay.stride_bytes:(off + 64) * lay.stride_bytes], toks[off:off + 64])
            ref = h.finish()
            sizes.append(h.state_bytes())
            refs.append(ref)
            self.assertEqual(ref.n_full_pages, pages)
            self.assertEqual(ref.n_tokens, len(toks))
            self.assertEqual(sum(n for _, n, _ in h.iter_pages()), len(toks))
            self.assertEqual(b"".join(r for _, _, r in h.iter_pages()), rows)
            self.assertLessEqual(len(h._tail_tokens), 15)
        self.assertEqual(sizes[0], sizes[1])                               # O(1) private state, no dense max_len x W array
        self.assertLess(sizes[1], 1024)
        self.assertLess(len(pickle_size(refs[1])), 512)                    # the prover-side reference is O(1) too

    def test_partial_tail_pages_are_content_addressed_but_never_a_prefix(self) -> None:
        lay = self.lay
        toks = list(range(16 + 3))
        rows = rows_for_conversation(lay, toks, 0, 19)
        a = self.store.open_request("a"); b = self.store.open_request("b")
        a.append(rows, toks); b.append(rows, toks)
        ra, rb = a.finish(), b.finish()
        self.assertEqual(ra.tip, rb.tip)                                   # the full page is shared (content-addressed)
        self.assertEqual(self.store.stats["dedup_hits"], 2)                # b re-wrote the full page and the tail page
        self.assertEqual(ra.tail, rb.tail)                                 # identical tail content => identical hash
        other = toks[:16] + [7, 7, 7]
        c = self.store.open_request("c")
        c.append(rows_for_conversation(lay, other, 0, 19), other)
        rc = c.finish()
        self.assertEqual(rc.tip, ra.tip)
        self.assertNotEqual(rc.tail, ra.tail)                              # different tokens => different tail page
        d = self.store.open_request("d")
        with self.assertRaises(ValueError):
            d.mount_prefix(toks)                                           # 19 tokens is not whole pages: tails are not prefixes

    def test_mount_prefix_requires_every_page_and_poisons_otherwise(self) -> None:
        lay = self.lay
        toks = list(range(16 * 3))
        w = self.store.open_request("w")
        w.append(rows_for_conversation(lay, toks, 0, 48), toks)
        wref = w.finish()
        r = self.store.open_request("r")
        self.assertEqual(r.mount_prefix(toks[:32]), 2)
        self.assertEqual(r.n_tokens, 32)
        r.append(rows_for_conversation(lay, toks, 32, 48), toks[32:])
        self.assertEqual(r.finish().tip, wref.tip)
        bad = self.store.open_request("bad")
        with self.assertRaises(PageMissingError):
            bad.mount_prefix(toks[:16] + [123] * 16)                       # second page was never captured
        self.assertEqual(bad.state, "poisoned")
        with self.assertRaises(RequestPoisonedError):
            bad.finish()
        with self.assertRaises(RequestPoisonedError):
            bad.append(b"", [])
        with self.assertRaises(RequestPoisonedError):
            list(bad.iter_pages())

    def test_stale_reference_after_eviction_raises(self) -> None:
        lay = self.lay
        toks = list(range(32))
        h = self.store.open_request("x")
        h.append(rows_for_conversation(lay, toks, 0, 32), toks)
        ref = h.finish()
        self.store.close_request("x")                                       # unpins
        self.assertTrue(self.store.evict(ref.tip))
        with self.assertRaises(PageMissingError):
            self.store.walk_to_root(ref.tip)

    def test_pinned_pages_survive_budget_pressure_while_a_request_is_live(self) -> None:
        lay = self.lay
        page = 16 * lay.stride_bytes
        store = PageStore(lay, max_bytes=4 * page)
        live = store.open_request("live")
        toks = list(range(48))
        live.append(rows_for_conversation(lay, toks, 0, 48), toks)
        for i in range(10):                                                 # churn unrelated requests through the cache
            o = store.open_request(f"o{i}")
            t = list(range(1000 + 16 * i, 1016 + 16 * i))
            o.append(rows_for_conversation(lay, t, 0, 16), t)
            o.finish(); store.close_request(f"o{i}")
        self.assertEqual(len(list(live.iter_pages())), 3)                  # tip pinned => whole chain protected
        self.assertLessEqual(store.bytes_stored, 4 * page)

    def test_unpin_is_idempotent_and_never_goes_negative(self) -> None:
        page = 16 * self.lay.stride_bytes
        h = block_hash_chain(self.store.seed, list(range(16)))[0]
        self.store.put_page(h, None, 16, b"\x01" * page)
        self.store.unpin(h); self.store.unpin(h)                         # extra unpins are ignored
        self.store.pin(h)
        self.assertFalse(self.store.evict(h))                            # a single pin must still protect the page
        self.store.unpin(h)
        self.assertFalse(self.store.has(h))                              # deferred eviction fires on the last unpin

    def test_duplicate_open_request_is_rejected(self) -> None:
        self.store.open_request("dup")
        with self.assertRaises(CaptureError):
            self.store.open_request("dup")


def pickle_size(obj) -> bytes:
    import pickle
    return pickle.dumps(obj)


# ============================================================================================ 20-turn agentic prefix dedup
class AgenticDedupTests(unittest.TestCase):
    TURNS = 20
    NEW_PER_TURN = 84                       # 5.x pages of new text per turn -> >= 95% of every later prompt is prefix

    def _conversation(self) -> list[list[int]]:
        convs, cur = [], list(range(1, 1601))                  # 100 pages of system prompt + context
        for t in range(self.TURNS):
            cur = cur + [10_000 + t * 1000 + j for j in range(self.NEW_PER_TURN)]
            convs.append(list(cur))
        return convs

    def _expected_pages(self, convs) -> tuple[int, int]:
        full = len(convs[-1]) // 16
        tails = sum(1 for c in convs if len(c) % 16)
        return full, tails

    def test_prefix_hits_above_95pct_with_engine_prefix_cache_write_zero_duplicates(self) -> None:
        sess = CaptureSession.create(ModelDims(2, 16, 2, 1, 8, 24), CaptureConfig(ring_mib=1, max_batched_tokens=64))
        convs = self._conversation()
        cached = 0                                              # tokens the engine's prefix cache covers
        naive_pages = 0
        written_by_turn, dedup_by_turn, new_pages_by_turn = [], [], []
        for t, conv in enumerate(convs):
            rid = f"turn{t}"
            before = dict(sess.store.stats)
            prefix = conv[:cached]
            sess.register_request(rid, prefix)
            if t > 0:
                self.assertGreaterEqual(cached / len(conv), 0.95, f"turn {t} prefix share")
            start = cached
            for off in range(start, len(conv), 60):
                end = min(off + 60, len(conv))
                ok = sess.step_cpu([(rid, conv[off:end])], rows_for_conversation(sess.layout, conv, off, end))
                self.assertTrue(ok)
                sess.sink.flush()
            sess.finish_requests([rid]); sess.sink.flush()
            ref = sess.sink.finished[rid]
            self.assertEqual(ref.n_tokens, len(conv))
            data = b"".join(r for _, _, r in sess.store.handle(rid).iter_pages())
            self.assertEqual(data, rows_for_conversation(sess.layout, conv, 0, len(conv)))   # full trace is readable & exact
            new_full = len(conv) // 16 - cached // 16
            tail = 1 if len(conv) % 16 else 0
            written_by_turn.append(sess.store.stats["pages_written"] - before["pages_written"])
            dedup_by_turn.append(sess.store.stats["dedup_hits"] - before["dedup_hits"])
            new_pages_by_turn.append(new_full + tail)
            naive_pages += -(-len(conv) // 16)
            sess.store.close_request(rid)
            cached = (len(conv) // 16) * 16                     # engine caches every completed block of this turn
        self.assertEqual(written_by_turn, new_pages_by_turn)    # each turn writes ONLY its new pages
        self.assertEqual(dedup_by_turn, [0] * self.TURNS)       # zero duplicate page writes
        full, tails = self._expected_pages(convs)
        self.assertEqual(sess.store.n_pages(), full + tails)
        self.assertEqual(sess.store.stats["pages_written"], full + tails)
        self.assertEqual(sess.store.bytes_stored, (full * 16 + sum(len(c) % 16 for c in convs)) * sess.layout.stride_bytes)
        self.assertGreater(naive_pages / (full + tails), 8)     # vs re-committing every turn
        self.assertEqual(sess.sink.failed, {})

    def test_engine_without_prefix_cache_refeeds_everything_and_the_store_dedups(self) -> None:
        sess = CaptureSession.create(ModelDims(2, 16, 2, 1, 8, 24), CaptureConfig(ring_mib=1, max_batched_tokens=64))
        convs = self._conversation()
        prev_full = 0
        for t, conv in enumerate(convs):
            rid = f"t{t}"
            sess.register_request(rid)
            before = dict(sess.store.stats)
            for off in range(0, len(conv), 60):
                end = min(off + 60, len(conv))
                sess.step_cpu([(rid, conv[off:end])], rows_for_conversation(sess.layout, conv, off, end))
                sess.sink.flush()
            sess.finish_requests([rid]); sess.sink.flush()
            full_now = len(conv) // 16
            wrote = sess.store.stats["pages_written"] - before["pages_written"]
            self.assertEqual(wrote, full_now - prev_full + (1 if len(conv) % 16 else 0))
            self.assertEqual(sess.store.stats["dedup_hits"] - before["dedup_hits"], prev_full)   # prefix pages deduped
            sess.store.close_request(rid)
            prev_full = full_now
        full, tails = self._expected_pages(convs)
        self.assertEqual(sess.store.n_pages(), full + tails)


# ========================================================================================================== DMA sink
class SinkTests(unittest.TestCase):
    DIMS = ModelDims(2, 16, 2, 1, 8, 24)

    def _session(self, **kw) -> CaptureSession:
        return CaptureSession.create(self.DIMS, CaptureConfig(**{"ring_mib": 1, "max_batched_tokens": 64, **kw}))

    def test_two_requests_in_one_step_are_split_by_segment(self) -> None:
        s = self._session()
        ta, tb = list(range(40)), list(range(500, 530))
        s.register_request("a"); s.register_request("b")
        rows = rows_for_conversation(s.layout, ta, 0, 40) + rows_for_conversation(s.layout, tb, 0, 30)
        self.assertTrue(s.step_cpu([("a", ta), ("b", tb)], rows))
        s.finish_requests(["a", "b"]); s.sink.flush()
        self.assertEqual(b"".join(r for *_, r in s.store.handle("a").iter_pages()), rows_for_conversation(s.layout, ta, 0, 40))
        self.assertEqual(b"".join(r for *_, r in s.store.handle("b").iter_pages()), rows_for_conversation(s.layout, tb, 0, 30))
        self.assertEqual(s.ring.copied_wp, 70)                                   # released only after the data is in the store
        self.assertEqual(set(s.sink.finished), {"a", "b"})

    def test_chunked_dma_matches_unchunked(self) -> None:
        s = self._session()
        s.sink.chunk_rows = 7                                                     # force many <=7-row DMAs incl. wrap
        toks = list(range(50))
        s.register_request("a")
        for i in range(0, 50, 25):
            s.step_cpu([("a", toks[i:i + 25])], rows_for_conversation(s.layout, toks, i, i + 25))
        s.sink.flush()
        s.finish_requests(["a"]); s.sink.flush()
        self.assertEqual(b"".join(r for *_, r in s.store.handle("a").iter_pages()), rows_for_conversation(s.layout, toks, 0, 50))

    def test_ring_overflow_poisons_the_requests_and_nothing_is_zero_filled(self) -> None:
        s = self._session(ring_mib=1, max_batched_tokens=64)
        cap = s.ring.capacity_rows
        s.register_request("big"); s.register_request("ok")
        toks = list(range(cap))
        # fill the ring completely without draining (sink not pumped)
        step = s.begin_step([("ok", toks)])
        self.assertIsNotNone(step)
        s.ring.publish(step.span, rows_for_conversation(s.layout, toks, 0, cap))
        self.assertIsNone(s.begin_step([("big", [1, 2, 3])]))                      # refused -> poisoned, no rows reserved
        self.assertEqual(s.store.handle("big").state, "poisoned")
        self.assertEqual(s.ring.write_wp, cap)
        self.assertEqual(s.store.handle("ok").state, "open")                        # earlier, undrained rows were NOT clobbered
        s.sink.submit(step.record); s.sink.flush()
        s.finish_requests(["big", "ok"]); s.sink.flush()
        self.assertIn("big", s.sink.failed)
        self.assertIn("ok", s.sink.finished)
        self.assertEqual(s.store.handle("big").n_tokens, 0)                         # no page for the lost rows
        self.assertFalse(s.store.n_pages() == 0)
        with self.assertRaises(RequestPoisonedError):
            list(s.store.handle("big").iter_pages())
        self.assertEqual(s.ring.overflow_events, 1)

    def test_rows_lost_to_forced_overwrite_poison_their_requests(self) -> None:
        s = self._session()
        cap = s.ring.capacity_rows
        s.register_request("victim")
        v = s.ring.reserve(cap - 2)
        s.ring.publish(v, _rows(s.ring, 0, cap - 2))
        rec_v = StepRecord(v, (Segment("victim", 0, cap - 2, tuple(range(cap - 2))),))
        s.sink.submit(rec_v)                                                         # queued, not yet drained
        w = s.ring.reserve(5, poison_on_overflow=True)                              # a graph replay clobbers 3 victim rows
        s.ring.publish(w, _rows(s.ring, cap, 5))
        s.sink.pump()
        self.assertEqual(s.store.handle("victim").state, "poisoned")
        self.assertEqual(s.sink.stats["lost_records"], 1)
        self.assertEqual(s.store.n_pages(), 0)

    def test_orphan_segments_and_unknown_requests_are_counted_not_fatal(self) -> None:
        s = self._session()
        s.step_cpu([("ghost", list(range(4)))], rows_for_conversation(s.layout, list(range(4)), 0, 4))
        s.sink.flush()
        self.assertEqual(s.sink.stats["orphan_segments"], 1)
        self.assertEqual(s.ring.copied_wp, 4)

    def test_sink_fault_fails_closed(self) -> None:
        s = self._session()
        s.register_request("a")
        toks = list(range(20))
        s.step_cpu([("a", toks)], rows_for_conversation(s.layout, toks, 0, 20))

        def boom(*_a, **_k):
            raise RuntimeError("disk on fire")
        s.store.put_page = boom                                                       # type: ignore[assignment]
        with self.assertRaises(SinkFaultedError):                                     # unexpected error => whole sink fails closed
            s.sink.pump()
        self.assertEqual(s.store.handle("a").state, "poisoned")                       # ... and the affected request is unprovable
        with self.assertRaises(SinkFaultedError):
            s.sink.submit(StepRecord(Span(0, 0), ()))
        self.assertEqual(s.store.n_pages(), 0)

    def test_background_thread_drains_without_the_producer_blocking(self) -> None:
        s = self._session()
        s.sink.start()
        try:
            toks = list(range(160))
            s.register_request("a")
            for i in range(0, 160, 32):
                self.assertTrue(s.step_cpu([("a", toks[i:i + 32])], rows_for_conversation(s.layout, toks, i, i + 32)))
            s.sink.flush(10.0)
            s.finish_requests(["a"]); s.sink.flush(10.0)
        finally:
            s.sink.stop()
        self.assertEqual(s.sink.finished["a"].n_tokens, 160)
        self.assertEqual(s.sink.stats["rows"], 160)

    def _large_page_session(self) -> CaptureSession:
        from fpn_serve.capture import store

        if store._np is None:
            self.skipTest("parallel page copies require NumPy")
        return CaptureSession.create(ModelDims(8, 512, 8, 2, 64, 2048),
                                     CaptureConfig(ring_mib=24, max_batched_tokens=128))

    def test_parallel_copy_order_and_pool_reuse(self) -> None:
        import threading

        from unittest.mock import patch

        # Exercise worker coordination with a compact fixture.
        self.enterContext(patch("fpn_serve.capture.store.PARALLEL_COPY_MIN_BYTES", 1 << 20))
        s = self._large_page_session()
        self.addCleanup(s.close)
        copy = s.store._copy_into_pooled
        rendezvous = threading.Barrier(4, timeout=5)
        workers = set()

        def concurrent_copy(buf):
            workers.add(threading.get_ident())
            rendezvous.wait()
            return copy(buf)

        s.store._copy_into_pooled = concurrent_copy
        s.register_request("live")
        expected = []
        for step in range(2):
            blob = b"".join(bytes([1 + step * 4 + page]) * (16 * s.ring.row_bytes) for page in range(4))
            expected.append(blob)
            self.assertTrue(s.step_cpu([("live", list(range(step * 64, (step + 1) * 64)))], blob))
        s.finish_requests(["live"])
        s.sink.flush()
        self.assertEqual(len(workers), 4)
        live = s.store.handle("live")
        snapshot = list(live.iter_pages())
        self.assertEqual(b"".join(r for *_, r in snapshot), b"".join(expected))
        self.assertEqual(s.ring.overflow_events, 0)

        s.store._copy_into_pooled = copy
        snapshots = []
        for fill in range(20):
            s.register_request("evict")
            src = bytearray([fill]) * (64 * s.ring.row_bytes)
            self.assertTrue(s.step_cpu([("evict", list(range(1000 + fill, 1064 + fill)))], src))
            s.finish_requests(["evict"])
            s.sink.flush()
            handle = s.store.handle("evict")
            old_pages = list(handle.iter_pages())
            s.forget(["evict"])
            src[:] = bytes(len(src))
            snapshots.append(old_pages)
            self.assertEqual(list(live.iter_pages()), snapshot)
        for fill, pages in enumerate(snapshots):
            self.assertTrue(all(rows == bytes([fill]) * len(rows) for _, _, rows in pages))
        self.assertTrue(s.store._buf_pool)
        self.assertGreaterEqual(s.store.stats["evictions"], 80)

    def test_parallel_copy_fault_waits_for_slab_and_settles_racing_submits(self) -> None:
        import threading

        from unittest.mock import patch

        # Exercise worker coordination with a compact fixture.
        self.enterContext(patch("fpn_serve.capture.store.PARALLEL_COPY_MIN_BYTES", 1 << 20))
        s = self._large_page_session()
        unblock = threading.Event()
        self.addCleanup(s.close)
        self.addCleanup(unblock.set)
        failed_copy = threading.Event()
        slow_copy = threading.Event()
        settled = []
        s.sink.on_settled = settled.append
        copy = s.store._copy_into_pooled

        def injected(buf):
            if buf[0] == 1:
                failed_copy.set()
                raise RuntimeError("copy fault")
            slow_copy.set()
            if not unblock.wait(5):
                raise TimeoutError("test did not unblock copy")
            return copy(buf)

        s.store._copy_into_pooled = injected
        s.sink.token_resolver = lambda *_: None
        s.sink.deferred_timeout_s = 0
        s.register_request("timeout")
        deferred = s.begin_step([("timeout", 1, 0)])
        s.ring.publish(deferred.span, bytes(s.ring.row_bytes))
        s.end_step(deferred)
        s.finish_requests(["timeout"])
        s.register_request("fault")
        src = b"\x01" * (16 * s.ring.row_bytes) + b"\x02" * (16 * s.ring.row_bytes)
        self.assertTrue(s.step_cpu([("fault", list(range(32)))], src))
        s.sink.start()
        self.assertTrue(failed_copy.wait(5))
        self.assertTrue(slow_copy.wait(5))
        self.assertIsNone(s.sink.fault)
        s.sink.submit(StepRecord(Span(33, 0), (), finish=("queued",)))
        unblock.set()
        s.sink._thread.join(5)
        self.assertFalse(s.sink._thread.is_alive())
        with self.assertRaises(SinkFaultedError):
            s.sink.submit(StepRecord(Span(33, 0), (), finish=("late",)))
        self.assertEqual(settled[0], "timeout")
        self.assertCountEqual(settled[1:-1], ["fault", "queued"])
        self.assertEqual(settled[-1], "late")
        self.assertIn("timeout", s.sink.failed)
        self.assertEqual(s.sink.pending, 0)
        with self.assertRaises(RequestPoisonedError):
            list(s.store.handle("fault").iter_pages())

    def test_large_page_precompleted_dma_ingestion(self) -> None:
        from time import perf_counter

        from fpn_serve.capture import store
        from fpn_serve.capture.ring import _ReadyRead
        from fpn_serve.capture.sink import AsyncDmaSink

        if store._np is None:
            self.skipTest("parallel page copies require NumPy")
        for workers in (1, 4):
            with self.subTest(workers=workers):
                s = CaptureSession.create(ModelDims(56, 512, 8, 2, 64, 2048),
                                          CaptureConfig(ring_mib=32, max_batched_tokens=32))
                s.sink.stop()
                s.sink = AsyncDmaSink(s.ring, s.store, copy_workers=workers)
                self.addCleanup(s.close)
                # Model an already-completed DMA slab. The CPU backend's row-wise
                # ring read would add unrelated copies to the timed interval.
                rb = s.ring.row_bytes
                blob = b"\x19" * (16 * rb) + b"\x27" * (16 * rb)
                s.ring.backend.read_async = lambda *_: _ReadyRead(blob)
                elapsed = 0.0
                for i in range(6):
                    s.register_request("r")
                    step = s.begin_step([("r", list(range(32)))])
                    self.assertIsNotNone(step)
                    s.end_step(step)
                    before = perf_counter()
                    self.assertEqual(s.sink.pump(), 1)
                    elapsed += perf_counter() - before
                    if i == 5:
                        self.assertEqual(b"".join(r for *_, r in s.store.handle("r").iter_pages()), blob)
                    s.forget(["r"])
                # Sanity only. The acceptance rate is hardware-specific.
                self.assertGreater(s.sink.stats["bytes"] / elapsed, 0)
                self.assertEqual(s.sink.stats["bytes"], len(blob) * 6)
                self.assertEqual(s.ring.overflow_events, 0)

    def test_begin_step_rejects_steps_that_can_never_fit_at_configuration_time(self) -> None:
        with self.assertRaises(RingOverflowError):
            CaptureSession.create(QWEN3_4B, CaptureConfig(ring_mib=320, max_batched_tokens=2048))     # 164-row ring


# ======================================================================== lifecycle hygiene (RAM, idempotence, headroom)
class LifecycleTests(unittest.TestCase):
    DIMS = ModelDims(2, 16, 2, 1, 8, 24)

    def _session(self, **kw) -> CaptureSession:
        return CaptureSession.create(self.DIMS, CaptureConfig(**{"ring_mib": 1, "max_batched_tokens": 64, **kw}))

    def _run(self, s: CaptureSession, rid: str, toks: list[int]) -> None:
        s.register_request(rid)
        for off in range(0, len(toks), 50):
            s.step_cpu([(rid, toks[off:off + 50])], rows_for_conversation(s.layout, toks, off, min(off + 50, len(toks))))
        s.sink.flush()

    def test_finish_is_idempotent_at_every_level(self) -> None:
        s = self._session()
        toks = list(range(40))
        self._run(s, "a", toks)
        h = s.store.handle("a")
        r1 = h.finish()
        self.assertIs(h.finish(), r1)
        s.finish_requests(["a"]); s.finish_requests(["a"])                          # queued once
        s.sink.submit(StepRecord(Span(s.ring.write_wp, 0), (), None, ("a", "a")))   # and settled once even if marked twice
        s.sink.flush()
        self.assertEqual(s.sink.failed, {})
        self.assertEqual(s.sink.finished["a"].n_tokens, 40)
        self.assertEqual(s.settle()["a"].n_tokens, 40)                              # bench-style settle after the engine's marker
        self.assertEqual(s.sink.failed, {})

    def test_forget_frees_host_pages_but_keeps_pages_shared_with_live_requests(self) -> None:
        s = self._session()
        shared = list(range(64))
        self._run(s, "a", shared + [900, 901, 902])
        self._run(s, "b", shared + [800, 801])
        s.settle()
        before = s.store.n_pages()
        self.assertEqual(s.store.handle("a").drop(), 1)                             # trim reports exactly what it freed
        s.forget(["a"])
        self.assertEqual(s.store.n_pages(), before - 1)                             # only a's private tail page is freed
        self.assertTrue(all(s.store.has(h) for h in s.store.walk_to_root(s.sink.finished["b"].tip)))
        s.forget(["b"])
        self.assertEqual((s.store.n_pages(), s.store.bytes_stored), (0, 0))
        self.assertEqual((s.sink.finished, s.sink.failed, s.store.open_ids()), ({}, {}, []))

    def test_count_only_mode_counts_pages_but_never_returns_bytes(self) -> None:
        s = self._session(retain_payload=False)
        toks = list(range(70))
        self._run(s, "a", toks)
        ref = s.settle()["a"]
        self.assertEqual((ref.n_full_pages, ref.n_tokens), (4, 70))
        self.assertEqual(s.store.bytes_stored, 0)
        self.assertGreater(s.store.stats["bytes_written"], 0)
        with self.assertRaises(PageMissingError):
            list(s.store.handle("a").iter_pages())                                   # fail closed: no silent empty rows

    def test_rows_are_released_as_soon_as_the_dma_bytes_are_taken(self) -> None:
        s = self._session()
        s.register_request("a")
        toks = list(range(20))
        s.step_cpu([("a", toks)], rows_for_conversation(s.layout, toks, 0, 20))

        def boom(*_a, **_k):
            raise RuntimeError("store exploded")
        s.store.put_page = boom                                                      # type: ignore[assignment]
        with self.assertRaises(SinkFaultedError):
            s.sink.pump()
        self.assertEqual(s.ring.copied_wp, 20)                                       # data was on the host: the ring is free again

    def test_two_back_to_back_half_ring_steps_never_overflow_when_the_sink_keeps_up(self) -> None:
        s = self._session()
        half = s.ring.capacity_rows // 2
        s.register_request("a")
        toks = list(range(half * 6))
        for i in range(6):
            sl = toks[i * half:(i + 1) * half]
            self.assertTrue(s.step_cpu([("a", sl)], rows_for_conversation(s.layout, toks, i * half, (i + 1) * half)))
            s.sink.pump()
        s.sink.flush()
        self.assertEqual(s.ring.overflow_events, 0)

    def test_bench_plan_leaves_half_the_ring_as_headroom(self) -> None:
        spec = importlib.util.spec_from_file_location("bench_capture_plan", ROOT / "bench/tier1/bench_capture.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        plan = mod.sizing_plan(QWEN3_4B, 320, ["stream_full_no_kv", "stream_tier_b"])
        for m, p in plan.items():
            self.assertLessEqual(p["max_batched_tokens"] * 2, p["ring_rows"], m)
            self.assertEqual(p["max_batched_tokens"] % 16, 0)
        self.assertEqual(plan["stream_full_no_kv"]["ring_rows"], 164)
        self.assertEqual(plan["stream_full_no_kv"]["max_batched_tokens"], 80)


# ===================================================== plugin runner glue, exercised on the CPU mock ring (no vLLM/torch)
class _FakeReq:
    def __init__(self, prompt, computed=0):
        self.prompt_token_ids, self.output_token_ids, self.num_computed_tokens = list(prompt), [], computed


class _FakeBatch:
    def __init__(self, rids, qsl):
        self.num_reqs, self.req_ids, self.query_start_loc_np = len(rids), rids, qsl


class _FakeSched:
    def __init__(self, finished=()):
        self.finished_req_ids = set(finished)


class _FakeRunner:
    """Minimal stand-in for vLLM's model runner: execute_model calls self.prepare_inputs, then 'runs the model'."""

    def __init__(self, layout, backend):
        self.layout, self.backend, self.requests, self.plan = layout, backend, {}, []

    def prepare_inputs(self, *a, **k):
        rids = [r for r, _ in self.plan]
        qsl = [0]
        for _, n in self.plan:
            qsl.append(qsl[-1] + n)
        return _FakeBatch(rids, qsl)

    def execute_model(self, scheduler_output, dummy_run=False):
        self.prepare_inputs()                                  # the plugin's wrapper reserves ring rows here
        rows = b""
        for (rid, n) in self.plan:
            st = self.requests.get(rid)
            if st is None:
                rows += b"\x07" * (n * self.layout.stride_bytes)
                continue
            conv = st.prompt_token_ids + st.output_token_ids
            start = self.pos[rid]
            rows += rows_for_conversation(self.layout, conv, start, start + n)
            self.pos[rid] = start + n
        n_valid = self.backend.last_ctl[1]                     # the tap kernels read n_valid from the device control block
        self.backend.device_write(rows[: n_valid * self.layout.stride_bytes])
        return "out"


class PluginGlueTests(unittest.TestCase):
    def setUp(self) -> None:
        from fpn_serve.engines.vllm import capture_plugin as cp
        self.cp = cp
        self.sess = CaptureSession.create(ModelDims(2, 16, 2, 1, 8, 24), CaptureConfig(ring_mib=1, max_batched_tokens=64))
        cp._S["session"] = self.sess
        cp._S["token_provider"] = None
        self.backend = self.sess.ring.backend
        self.runner = _FakeRunner(self.sess.layout, self.backend)
        self.runner.pos = {}
        self.worker = type("W", (), {"model_runner": self.runner})()

    def tearDown(self) -> None:
        self.cp.disable_capture()

    def _step(self, plan, finished=()):
        self.runner.plan = plan
        return self.runner.execute_model(_FakeSched(finished))

    def test_runner_wrappers_capture_a_whole_conversation_with_the_sink_running(self) -> None:
        self.cp.install_runner(self.worker)
        self.assertIsNotNone(self.sess.sink._thread, "install_runner must start the drain thread")
        r1, r2 = _FakeReq(range(40)), _FakeReq(range(500, 520))
        self.runner.requests.update({"r1": r1, "r2": r2})
        self.runner.pos.update({"r1": 0, "r2": 0})
        self._step([("r1", 40), ("r2", 20)])
        r1.output_token_ids.append(7); r2.output_token_ids.append(8)
        self._step([("r1", 1), ("r2", 1)])
        self._step([], finished=["r1", "r2"])
        self._step([], finished=["r1"])                                              # a repeated marker must be harmless
        self.sess.sink.flush(10.0)
        self.assertEqual(self.sess.sink.failed, {})
        ref1, ref2 = self.sess.sink.finished["r1"], self.sess.sink.finished["r2"]
        self.assertEqual((ref1.n_tokens, ref2.n_tokens), (41, 21))
        conv1 = list(range(40)) + [7]
        self.assertEqual(b"".join(r for *_, r in self.sess.store.handle("r1").iter_pages()),
                         rows_for_conversation(self.sess.layout, conv1, 0, 41))
        self.assertEqual(self.backend.last_ctl[1], 0)                                 # n_valid zeroed after every step
        self.assertEqual(self.sess.ring.overflow_events, 0)

    def test_prefix_cache_hit_mounts_pages_written_by_an_earlier_request(self) -> None:
        self.cp.install_runner(self.worker)
        base = list(range(64))
        a = _FakeReq(base)
        self.runner.requests["a"] = a
        self.runner.pos["a"] = 0
        self._step([("a", 64)])
        self._step([], finished=["a"])
        self.sess.sink.flush(10.0)
        b = _FakeReq(base + [900, 901, 902], computed=64)                              # engine skipped the 64 cached tokens
        self.runner.requests["b"] = b
        self.runner.pos["b"] = 64
        self._step([("b", 3)])
        self._step([], finished=["b"])
        self.sess.sink.flush(10.0)
        self.assertEqual(self.sess.sink.failed, {})
        self.assertEqual(self.sess.sink.finished["b"].n_tokens, 67)
        self.assertEqual(self.sess.sink.finished["b"].n_full_pages, 4)
        self.assertEqual(self.sess.store.stats["dedup_hits"], 0)                       # nothing was rewritten

    def test_missing_prefix_page_or_unknown_request_poisons_without_crashing_the_engine(self) -> None:
        self.cp.install_runner(self.worker)
        c = _FakeReq(list(range(48)) + [1], computed=48)                               # claims a cached prefix we never captured
        self.runner.requests["c"] = c
        self.runner.pos["c"] = 48
        self._step([("c", 1), ("ghost", 2)])                                           # 'ghost' is unknown to the runner
        self._step([], finished=["c"])
        self.sess.sink.flush(10.0)
        self.assertIn("c", self.sess.sink.failed)
        self.assertEqual(self.sess.store.handle("c").state, "poisoned")

    def test_ring_overflow_in_the_runner_poisons_and_the_step_writes_nothing(self) -> None:
        self.cp.install_runner(self.worker)
        cap = self.sess.ring.capacity_rows
        self.runner.requests["big"] = _FakeReq(range(cap + 5))
        self.runner.pos["big"] = 0
        self._step([("big", cap + 5)])                           # a step larger than the ring: refused, taps told n_valid=0
        self.assertEqual(self.backend.last_ctl[1], 0)
        self.assertEqual(self.sess.ring.write_wp, 0)             # nothing reserved, nothing written, nothing zero-filled
        self.assertEqual(self.sess.store.handle("big").state, "poisoned")
        self._step([], finished=["big"])
        self.sess.sink.flush(10.0)
        self.assertIn("big", self.sess.sink.failed)

    def test_install_requires_a_configured_session(self) -> None:
        self.cp._S["session"] = None
        with self.assertRaises(CaptureError):
            self.cp.install_runner(self.worker)


# ============================================================ tap kernels, host-emulated (logic of the real .cu source)
SHIM = r"""
#include <cstdint>
#include <cstddef>
#define __global__
#define __device__
#define __forceinline__ inline
struct uint4 { unsigned x, y, z, w; };
struct dim3_t { unsigned x = 0, y = 0, z = 0; };
static dim3_t blockIdx, threadIdx, gridDim, blockDim;
#include "tap_kernels.cu"
template <class F> static void emu(unsigned gx, unsigned gy, unsigned bx, F f) {
  gridDim.x = gx; gridDim.y = gy; gridDim.z = 1; blockDim.x = bx; blockDim.y = 1; blockDim.z = 1;
  for (unsigned by = 0; by < gy; by++) for (unsigned b = 0; b < gx; b++) {
    blockIdx.x = b; blockIdx.y = by; blockIdx.z = 0;
    for (unsigned t = 0; t < bx; t++) { threadIdx.x = t; threadIdx.y = 0; threadIdx.z = 0; f(); }
  }
}
extern "C" void emu_single(unsigned gx, unsigned gy, unsigned bx, const u16* src, int ss, u16* ring, int rs, int cap, i64* ctl, int nm, int off, int w) {
  emu(gx, gy, bx, [&] { tap_single_port(src, ss, ring, rs, cap, ctl, nm, off, w); });
}
extern "C" void emu_pair(unsigned gx, unsigned gy, unsigned bx, const u16* r, int rst, int ro, int rw, const u16* n, int nst, int no, int nw,
                         u16* ring, int rs, int cap, i64* ctl, int nm) {
  emu(gx, gy, bx, [&] { tap_residual_and_norm(r, rst, ro, rw, n, nst, no, nw, ring, rs, cap, ctl, nm); });
}
extern "C" void emu_attn(unsigned gx, unsigned gy, unsigned bx, const u16* q, int qs, int qo, int qw, const u16* o, int os_, int oo, int ow,
                         const u32* lse, int ls, int lo, int nq, u16* ring, int rs, int cap, i64* ctl, int nm) {
  emu(gx, gy, bx, [&] { tap_attn_epilogue_no_kv(q, qs, qo, qw, o, os_, oo, ow, lse, ls, lo, nq, ring, rs, cap, ctl, nm); });
}
extern "C" void emu_ids(unsigned gx, unsigned gy, unsigned bx, const i32* ids, int is, int off, int k, u16* ring, int rs, int cap, i64* ctl, int nm) {
  emu(gx, gy, bx, [&] { tap_expert_ids_i32_to_u16(ids, is, off, k, ring, rs, cap, ctl, nm); });
}
extern "C" void emu_advance(i64* ctl) { emu(1, 1, 32, [&] { advance_cursor_u64(ctl); }); }
"""


def _compile_emulator(tmp: Path):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        return None
    (tmp / "shim.cpp").write_text(SHIM, encoding="utf-8")
    lib = tmp / ("emu.dylib" if sys.platform == "darwin" else "emu.so")
    r = subprocess.run([cxx, "-std=c++17", "-O1", "-shared", "-fPIC", "-Wall", "-Wno-unused-function", f"-I{ROOT / 'serve/fpn_serve/capture'}",
                        str(tmp / "shim.cpp"), "-o", str(lib)], capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError("tap_kernels.cu failed to compile under the host shim:\n" + r.stderr)
    return ctypes.CDLL(str(lib))


class TapKernelEmulationTests(unittest.TestCase):
    """Compiles the REAL ``tap_kernels.cu`` for the host (CUDA built-ins shimmed) and runs the launch grid sequentially."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.lib = _compile_emulator(Path(cls._tmp.name))
        if cls.lib is None:
            raise unittest.SkipTest("no C++ compiler available")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- helpers
    @staticmethod
    def u16(n: int, seed: int) -> "ctypes.Array":
        a = (ctypes.c_uint16 * n)()
        for i in range(n):
            a[i] = (i * 2654435761 + seed * 40503) & 0xFFFF
        return a

    @staticmethod
    def at(arr, elem_off: int = 0, elem: int = 2):
        return ctypes.c_void_p(ctypes.addressof(arr) + elem_off * elem)

    def ring(self, cap: int, stride: int):
        r = (ctypes.c_uint16 * (cap * stride))()
        for i in range(cap * stride):
            r[i] = 0xA5A5
        return r

    def ctl(self, cursor: int, n_valid: int):
        c = (ctypes.c_int64 * 4)(cursor, n_valid, 0, 0)
        return c

    def row(self, arr, r: int, stride: int, c0: int, w: int):
        return [arr[r * stride + c0 + i] for i in range(w)]

    def test_source_has_no_kv_and_no_includes_and_no_zero_fill(self) -> None:
        src = (ROOT / "serve/fpn_serve/capture/tap_kernels.cu").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*#\s*include", src, re.M))
        self.assertNotRegex(src, r"\bmemset\b|=\s*0\s*;\s*//\s*fill")
        for name in ("copy_row_u16_aligned", "tap_residual_and_norm", "tap_attn_epilogue_no_kv", "tap_single_port", "advance_cursor_u64"):
            self.assertIn(name, src)
        self.assertNotRegex(src, r"\bk_cache\b|\bkv_cache\b|\bvalue\b")

    def _run_single(self, w: int, src_off: int, misalign: bool) -> None:
        cap, stride, n_max, ss = 7, 64, 10, 48
        src = self.u16(n_max * ss + 8, 1)
        ring = self.ring(cap, stride)
        ctl = self.ctl(cursor=5, n_valid=6)                        # rows 5..10 -> physical 5,6,0,1,2,3 (wraps)
        base = 1 if misalign else 0
        self.lib.emu_single(n_max, 3, 128, self.at(src, base), ss, self.at(ring), stride, cap, ctl, n_max, 8, w)
        for r in range(6):
            phys = (5 + r) % cap
            self.assertEqual(self.row(ring, phys, stride, 8, w), [src[base + r * ss + i] for i in range(w)], (w, r))
            self.assertEqual(self.row(ring, phys, stride, 0, 8), [0xA5A5] * 8)           # neighbours untouched
            self.assertEqual(self.row(ring, phys, stride, 8 + w, 4), [0xA5A5] * 4)
        self.assertEqual(self.row(ring, 4, stride, 0, stride), [0xA5A5] * stride)        # row not owned by this step: untouched

    def test_single_port_copies_every_column_including_the_w_mod_8_tail(self) -> None:
        for w in (1, 7, 8, 9, 16, 37, 40):
            self._run_single(w, 0, misalign=False)

    def test_misaligned_pointers_take_the_scalar_path_and_stay_exact(self) -> None:
        for w in (8, 37, 40):
            self._run_single(w, 0, misalign=True)

    def test_rows_beyond_n_valid_write_nothing_and_zero_valid_is_a_noop(self) -> None:
        cap, stride, n_max = 5, 16, 6
        src = self.u16(n_max * 16, 2)
        ring = self.ring(cap, stride)
        snap = list(ring)
        self.lib.emu_single(n_max, 1, 128, self.at(src), 16, self.at(ring), stride, cap, self.ctl(3, 0), n_max, 0, 16)
        self.assertEqual(list(ring), snap)                                              # n_valid == 0: no write at all
        self.lib.emu_single(n_max, 1, 128, self.at(src), 16, self.at(ring), stride, cap, self.ctl(3, 2), n_max, 0, 16)
        self.assertEqual(self.row(ring, 3, stride, 0, 16), [src[i] for i in range(16)])
        self.assertEqual(self.row(ring, 4, stride, 0, 16), [src[16 + i] for i in range(16)])
        for r in (0, 1, 2):
            self.assertEqual(self.row(ring, r, stride, 0, 16), [0xA5A5] * 16)

    def test_n_valid_above_capture_size_sets_the_violation_flag(self) -> None:
        cap, stride = 5, 16
        src = self.u16(64, 3)
        ring = self.ring(cap, stride)
        ctl = self.ctl(0, 9)
        self.lib.emu_single(4, 1, 128, self.at(src), 16, self.at(ring), stride, cap, ctl, 4, 0, 16)
        self.assertEqual(ctl[2], 1)

    def test_residual_and_norm_fused_and_norm_optional(self) -> None:
        cap, stride, n = 4, 48, 3
        res, nrm = self.u16(n * 20, 4), self.u16(n * 24, 5)
        ring = self.ring(cap, stride)
        self.lib.emu_pair(n, 2, 128, self.at(res), 20, 0, 19, self.at(nrm), 24, 24, 21, self.at(ring), stride, cap, self.ctl(2, n), n)
        for r in range(n):
            phys = (2 + r) % cap
            self.assertEqual(self.row(ring, phys, stride, 0, 19), [res[r * 20 + i] for i in range(19)])
            self.assertEqual(self.row(ring, phys, stride, 24, 21), [nrm[r * 24 + i] for i in range(21)])
        ring2 = self.ring(cap, stride)
        self.lib.emu_pair(n, 1, 128, self.at(res), 20, 0, 19, self.at(res), 20, 24, 0, self.at(ring2), stride, cap, self.ctl(0, n), n)
        self.assertEqual(self.row(ring2, 0, stride, 24, 8), [0xA5A5] * 8)                # w_nrm == 0: norm slot untouched

    def test_attn_epilogue_transposes_lse_and_never_touches_kv(self) -> None:
        cap, stride, n_max, nq = 6, 96, 5, 4
        q, o = self.u16(n_max * 24, 6), self.u16(n_max * 16, 7)
        lse = (ctypes.c_uint32 * (nq * 12))()
        for h in range(nq):
            for t in range(12):
                lse[h * 12 + t] = 0x3F000000 + h * 1000 + t                                # [nq, tokens], head stride 12
        ring = self.ring(cap, stride)
        ctl = self.ctl(cursor=4, n_valid=5)
        self.lib.emu_attn(n_max, 2, 128, self.at(q), 24, 0, 21, self.at(o), 16, 24, 13, self.at(lse, 0, 4), 12, 40, nq,
                          self.at(ring), stride, cap, ctl, n_max)
        for r in range(5):
            phys = (4 + r) % cap
            self.assertEqual(self.row(ring, phys, stride, 0, 21), [q[r * 24 + i] for i in range(21)])
            self.assertEqual(self.row(ring, phys, stride, 24, 13), [o[r * 16 + i] for i in range(13)])
            got = [ring[phys * stride + 40 + 2 * h] | (ring[phys * stride + 41 + 2 * h] << 16) for h in range(nq)]
            self.assertEqual(got, [lse[h * 12 + r] for h in range(nq)], r)               # row-major [row][head] after transpose
            self.assertEqual(self.row(ring, phys, stride, 48, 8), [0xA5A5] * 8)           # nothing beyond the slots was written

    def test_expert_ids_i32_to_u16_and_cursor_advance(self) -> None:
        cap, stride, n, k = 4, 16, 3, 5
        ids = (ctypes.c_int32 * (n * 8))(*[(r * 100 + j) for r in range(n) for j in range(8)])
        ring = self.ring(cap, stride)
        ctl = self.ctl(3, n)
        self.lib.emu_ids(n, 1, 128, self.at(ids, 0, 4), 8, 8, k, self.at(ring), stride, cap, ctl, n)
        for r in range(n):
            self.assertEqual(self.row(ring, (3 + r) % cap, stride, 8, k), [r * 100 + j for j in range(k)])
        self.lib.emu_advance(ctl)
        self.assertEqual(ctl[0], 6)
        ctl2 = (ctypes.c_int64 * 4)((1 << 62) + 5, 3, 0, 0)
        self.lib.emu_advance(ctl2)
        self.assertEqual(ctl2[0], (1 << 62) + 8)                                          # 64-bit cursor, no 32-bit truncation


# =============================================================================================================== boundaries
def _imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append(node.module)
                out += [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.append(node.args[0].value)
            if name in ("find_spec", "spec_from_file_location") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.append(node.args[0].value)
    return out


class BoundaryTests(unittest.TestCase):
    FORBIDDEN_FOR_SERVE = ("fpn_prover", "prover", "flashprove_next", "verifier")

    def _serve_files(self) -> list[Path]:
        return sorted(SERVE_DIR.rglob("*.py"))

    def test_serve_never_imports_prover_or_flashprove_next(self) -> None:
        files = self._serve_files()
        self.assertGreaterEqual(len(files), 10, "AST scan found suspiciously few serve/ files")
        scanned = 0
        for p in files:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            scanned += 1
            for mod in _imports(tree):
                top = mod.split(".")[0]
                self.assertNotIn(top, self.FORBIDDEN_FOR_SERVE, f"{p.relative_to(ROOT)} imports {mod}")
            for node in ast.walk(tree):                         # also catch sys.path games pointing at the prover tree
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotRegex(node.value, r"(^|/)(prover|flashprove_next)(/|$)", f"{p.relative_to(ROOT)}: {node.value!r}")
        self.assertEqual(scanned, len(files))

    def test_the_import_check_actually_detects_violations(self) -> None:
        bad = ast.parse("import flashprove_next\nfrom fpn_prover.ir import x\nimport importlib\nimportlib.import_module('prover.x')\n__import__('flashprove_next.y')")
        tops = {m.split(".")[0] for m in _imports(bad)}
        self.assertTrue({"flashprove_next", "fpn_prover", "prover"} <= tops)

    def test_prover_format_and_budget_do_not_import_serve(self) -> None:
        for p in (PROVER_DIR / "fpn_prover").rglob("*.py"):
            for mod in _imports(ast.parse(p.read_text(encoding="utf-8"))):
                self.assertNotEqual(mod.split(".")[0], "fpn_serve", f"{p}: {mod}")

    def test_no_synchronize_call_sites_anywhere_in_serve(self) -> None:
        hits = []
        for p in self._serve_files():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and node.attr in ("synchronize", "cudaDeviceSynchronize"):
                    hits.append(f"{p.relative_to(ROOT)}:{node.lineno}")
                if isinstance(node, ast.Name) and node.id in ("synchronize",):
                    hits.append(f"{p.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(hits, [])
        for p in SERVE_DIR.rglob("*.cu"):
            self.assertNotIn("synchronize", p.read_text(encoding="utf-8").lower().replace("__syncthreads", ""))

    def test_no_implicit_host_syncs_in_the_step_path_modules(self) -> None:
        """`.item()/.cpu()/.tolist()/.numpy()` on device tensors are hidden syncs; the capture hot path must not use them."""
        for rel in ("capture/kernels.py", "engines/vllm/capture_plugin.py"):
            tree = ast.parse((SERVE_DIR / "fpn_serve" / rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in ("item", "cpu", "tolist", "numpy", "cuda_synchronize"):
                    self.fail(f"{rel}:{node.lineno}: .{node.attr}() is an implicit sync")

    def test_serve_reads_no_environment(self) -> None:
        for p in self._serve_files():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in ("environ", "getenv"):
                    self.fail(f"{p.relative_to(ROOT)}:{node.lineno} reads the environment")

    def test_no_banned_names_anywhere(self) -> None:
        for p in OWN_FILES:
            text = p.read_text(encoding="utf-8").lower()
            for b in BANNED:
                self.assertNotIn(b, text, f"{p.relative_to(ROOT)} contains a banned name")
            self.assertNotIn(b.lower(), str(p.relative_to(ROOT)).lower())

    def test_serve_import_chain_is_torch_free(self) -> None:
        code = ("import sys; sys.path.insert(0, %r)\n"
                "import fpn_serve.capture, fpn_serve.engines.vllm.capture_plugin as cp\n"
                "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, sorted(m for m in sys.modules if m in ('torch','vllm'))\n"
                "print('ok')") % str(SERVE_DIR)
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)

    def test_disabled_capture_allocates_and_imports_nothing(self) -> None:
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from fpn_serve.engines.vllm import capture_plugin as cp\n"
                "cp.configure_capture(cp.CaptureConfig(enabled=False))\n"
                "assert not cp.is_enabled() and cp.get_session() is None\n"
                "assert cp._S['slots'] == [] and cp._S['launcher'] is None and not cp._S['ops'] and not cp._S['loader']\n"
                "assert 'torch' not in sys.modules and 'vllm' not in sys.modules\n"
                "print('ok')") % str(SERVE_DIR)
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)

    def test_prover_packages_are_namespace_safe(self) -> None:
        self.assertFalse((PROVER_DIR / "fpn_prover" / "__init__.py").exists(), "prover/fpn_prover/__init__.py is owned by another task")


class BenchDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("bench_capture", ROOT / "bench/tier1/bench_capture.py")
        cls.mod = importlib.util.module_from_spec(spec)
        sys.modules["bench_capture"] = cls.mod
        spec.loader.exec_module(cls.mod)

    def test_help_and_dry_run_need_no_torch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text('{"num_hidden_layers":36,"hidden_size":2560,"num_attention_heads":32,'
                                                 '"num_key_value_heads":8,"head_dim":128,"intermediate_size":9728}')
            r = subprocess.run([sys.executable, "-I", str(ROOT / "bench/tier1/bench_capture.py"), "--model", d, "--dry-run", "--no-h-mid"],
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        import json
        plan = json.loads(r.stdout)["plan"]
        self.assertEqual(plan["stream_tier_b"]["payload_bytes_per_token"], 189_440)
        self.assertLessEqual(plan["stream_tier_b"]["ring_mib"], 128)
        self.assertLessEqual(plan["stream_full_no_kv"]["ring_mib"], 350)
        self.assertEqual(plan["stream_full_no_kv"]["max_batched_tokens"] % 16, 0)

    def test_gpu_presets_select_cells_and_leave_ring_headroom(self) -> None:
        import contextlib
        import io
        import json

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text('{"num_hidden_layers":36,"hidden_size":2560,"num_attention_heads":32,'
                                                 '"num_key_value_heads":8,"head_dim":128,"intermediate_size":9728}')
            for preset, batch, ctx, step, util in (("b8", "8", "256", 32, 0.72), ("8k", "1", "8192", 64, 0.79)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(self.mod.main(["--model", d, "--preset", preset, "--dry-run"]), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual((plan["batch"], plan["ctx"], plan["max_batched_tokens"]), (batch, ctx, step))
                self.assertEqual(plan["gpu_memory_utilization"], util)

    def test_handoff_checks_full_and_partial_pages_against_transcript(self) -> None:
        from fpn_serve.capture.ports import CaptureError, CaptureLayout, ModelDims

        layout = CaptureLayout.build(ModelDims(2, 16, 2, 1, 8, 32))
        store = PageStore(layout, layer_group_size=1)
        handle = store.open_request("request")
        tokens = list(range(19))
        rows = rows_for_conversation(layout, tokens, 0, len(tokens))
        rb = layout.stride_bytes
        handle.append(memoryview(rows)[:3 * rb], tokens[:3])
        handle.append(memoryview(rows)[3 * rb:], tokens[3:])
        ref = handle.finish()
        self.assertEqual(self.mod.verify_handoff(handle, ref, tokens), 2)
        self.assertEqual(b"".join(data for _, _, data in handle.iter_pages()), rows)
        for idx in (0, 18):
            bad = tokens.copy()
            bad[idx] += 1
            with self.assertRaisesRegex(CaptureError, "hash chain"):
                self.mod.verify_handoff(handle, ref, bad)
        with self.assertRaisesRegex(CaptureError, "length mismatch"):
            self.mod.verify_handoff(handle, ref, tokens[:-1])

    def test_activation_tier_captures_and_hands_off_full_and_partial_pages(self) -> None:
        cfg = CaptureConfig.from_dict({"tier": "full_no_kv_act", "ring_mib": 1, "max_batched_tokens": 32})
        sess = CaptureSession.create(ModelDims(2, 16, 2, 1, 8, 32), cfg)
        lay = sess.layout
        self.assertEqual(set(lay.breakdown()), {"h_in", "h_mid", "q", "o", "lse", "a", "h_L"})
        self.assertEqual(lay.slot("a", 0).width, lay.dims.d_ff)
        self.assertEqual(lay.slot("a", 1).width, lay.dims.d_ff)
        qwen = CaptureLayout.build(QWEN3_4B, cfg.options())
        self.assertEqual(qwen.payload_bytes_per_token, 2_042_368 - 73 * 2560 * 2)
        self.assertNotEqual(qwen.fingerprint(), CaptureLayout.build(QWEN3_4B).fingerprint())
        tokens = list(range(19))
        rows = rows_for_conversation(lay, tokens, 0, len(tokens))
        sess.register_request("request")
        self.assertTrue(sess.step_cpu([("request", tokens)], rows))
        ref = sess.settle(1.0)["request"]
        handle = sess.store.handle("request")
        self.assertEqual(self.mod.verify_handoff(handle, ref, tokens), 2)
        self.assertEqual(b"".join(data for _, _, data in handle.iter_pages()), rows)

    def test_count_only_incremental_tail_never_buffers_payload(self) -> None:
        from fpn_serve.capture.ports import CaptureError, CaptureLayout, ModelDims

        layout = CaptureLayout.build(ModelDims(2, 16, 2, 1, 8, 32))
        store = PageStore(layout, retain_payload=False)
        handle = store.open_request("request")
        for tok in range(19):
            handle.append(bytes(layout.stride_bytes), [tok])
            self.assertIsNone(handle._tail_rows)
        with self.assertRaisesRegex(CaptureError, "empty store"):
            store.retain_payload = True
        ref = handle.finish()
        self.assertEqual(ref.n_tokens, 19)
        self.assertEqual(store.stats["bytes_written"], 19 * layout.stride_bytes)

    def test_prompts_are_deterministic_and_request_unique(self) -> None:
        a, b = self.mod.make_prompt(1, 64), self.mod.make_prompt(2, 64)
        self.assertEqual(a, self.mod.make_prompt(1, 64))
        self.assertNotEqual(a, b)
        self.assertTrue(all(1000 <= t < 150000 for t in a))

    def test_summary_applies_the_gate_to_the_r32_baseline(self) -> None:
        cell = lambda tps, sync=2, ok=True: {"b1_c256": dict(decode_tok_s=tps, e2e_tok_s=tps, sync_calls=sync, capture_complete=ok)}
        res = {
            "stock_default": dict(cells=cell(110.0)),
            "numerics_r32": dict(cells=cell(100.0)),
            "stream_tier_b": dict(ring_mib=120.0, cells=cell(98.0, 2, True)),
            "stream_full_no_kv": dict(ring_mib=330.0, cells=cell(90.0, 5, False)),
        }
        res["stream_tier_b"]["ring_overflow_events"] = 1
        res["stream_tier_b"]["cells"]["b1_c256"].update(ring_overflow_events=0, handoff_verified=True)
        s = self.mod.summarize(res)
        g = s["stream_tier_b"]["b1_c256"]["gate"]
        self.assertTrue(g["overhead_le_3pct"] and g["zero_extra_sync"] and g["ring_within_cap"] and g["capture_complete"])
        self.assertTrue(g["zero_overflows"] and g["handoff_verified"])
        self.assertEqual(s["stream_tier_b"]["b1_c256"]["e2e_tok_s_vs_numerics_r32_pct"], -2.0)
        g2 = s["stream_full_no_kv"]["b1_c256"]["gate"]
        self.assertFalse(g2["overhead_le_3pct"])
        self.assertFalse(g2["zero_extra_sync"])
        self.assertFalse(g2["capture_complete"])
        res["stream_tier_b"]["cells"]["b1_c256"]["capture_complete"] = False          # poisoned runs are faster: never graded
        s2 = self.mod.summarize(res)["stream_tier_b"]["b1_c256"]
        self.assertIsNone(s2["gate"]["overhead_le_3pct"])
        self.assertIn("withheld", " ".join(s2.keys()))
        self.assertEqual(self.mod.decode_rate(4, 101, 1.0, 2.0), 400.0)
        self.assertIsNone(self.mod.decode_rate(4, 1, 1.0, 2.0))


class TestDeferredTokensAndBoundedRetention(unittest.TestCase):
    """Regression for audit finding #8: async-scheduled deferred token resolution + bounded memory cleanup."""

    def test_dma_rows_are_reusable_while_tokens_are_deferred(self) -> None:
        from fpn_serve.capture.ports import ModelDims
        from fpn_serve.capture.session import CaptureConfig, CaptureSession

        dims = ModelDims(2, 16, 2, 1, 8, 32)
        sess = CaptureSession.create(dims, CaptureConfig(max_batched_tokens=16))
        sess.register_request("r")
        toks = [101]
        rows = rows_for_conversation(sess.layout, toks, 0, 1)
        ready = False
        sess.sink.token_resolver = lambda *_: toks if ready else None
        for _ in range(2):
            sess.step_cpu([("r", 1, 0)], rows)
        self.assertEqual(sess.sink.pump(), 0)
        self.assertEqual(sess.ring.copied_wp, 2)
        self.assertEqual(sess.store.handle("r").n_tokens, 0)
        ready = True
        self.assertEqual(sess.sink.pump(), 2)
        sess.settle()
        self.assertEqual(b"".join(r for _, _, r in sess.store.handle("r").iter_pages()), rows * 2)

    def test_on_settled_releases_token_maps_across_50_requests_and_reused_ids(self) -> None:
        from fpn_serve.capture.ports import CaptureTier, ModelDims
        from fpn_serve.capture.session import CaptureConfig, CaptureSession

        dims = ModelDims(n_layers=2, d_model=16, n_q_heads=2, n_kv_heads=1, head_dim=8, d_ff=32)
        sess = CaptureSession.create(dims, CaptureConfig(tier=CaptureTier.TIER_B, ring_mib=8, max_batched_tokens=32))
        rb = sess.layout.stride_bytes
        req_tokens: dict[str, list[int]] = {}
        sess.sink.token_resolver = lambda rid, p, n: (
            tuple(req_tokens[rid][p:p + n]) if rid in req_tokens and len(req_tokens[rid]) >= p + n else None
        )
        sess.sink.on_settled = lambda rid: req_tokens.pop(rid, None)

        # Run 50 sequential requests (reusing IDs every 10 requests) with delayed async token arrival
        for idx in range(50):
            rid = f"req-{idx % 10}"
            req_tokens[rid] = [100 + idx, 200 + idx]
            sess.register_request(rid)
            # Step 0: prefill 2 prompt tokens
            step0 = sess.begin_step([(rid, req_tokens[rid])])
            self.assertIsNotNone(step0)
            sess.ring.publish(step0.span, bytes([idx & 0xFF]) * (2 * rb))
            sess.end_step(step0)
            # Step 1: deferred decode token (pos=2) NOT YET in req_tokens when begin_step runs!
            step1 = sess.begin_step([(rid, 1, 2)])
            self.assertIsNotNone(step1)
            sess.ring.publish(step1.span, bytes([(idx + 1) & 0xFF]) * rb)
            sess.end_step(step1)
            sess.finish_requests([rid])
            # Pump before delayed token arrives: must wait cleanly without poisoning
            sess.sink.pump()
            self.assertIn(rid, req_tokens)
            # Async callback delivers sampled token id -> pump completes & releases req_tokens[rid]
            req_tokens[rid].append(300 + idx)
            while sess.sink.pending:
                sess.sink.pump()
            self.assertNotIn(rid, req_tokens, f"leaked token map for {rid}")
            self.assertIn(rid, sess.sink.finished)
            self.assertEqual(sess.sink.finished[rid].n_tokens, 3)
            sess.forget([rid])

        self.assertEqual(len(req_tokens), 0)
        self.assertEqual(sess.sink.stats["poisoned_requests"], 0)
        self.assertEqual(sess.sink.stats["orphan_segments"], 0)

    def test_missing_deferred_token_times_out_without_blocking_later_requests(self) -> None:
        from fpn_serve.capture.ports import CaptureTier, ModelDims
        from fpn_serve.capture.session import CaptureConfig, CaptureSession

        dims = ModelDims(n_layers=2, d_model=16, n_q_heads=2, n_kv_heads=1, head_dim=8, d_ff=32)
        sess = CaptureSession.create(dims, CaptureConfig(tier=CaptureTier.TIER_B, ring_mib=8, max_batched_tokens=32))
        sess.sink.deferred_timeout_s = 0.02
        rb = sess.layout.stride_bytes
        req_tokens: dict[str, list[int]] = {"cancelled": [1, 2], "healthy": [10, 11]}
        sess.sink.token_resolver = lambda rid, p, n: (
            tuple(req_tokens[rid][p:p + n]) if rid in req_tokens and len(req_tokens[rid]) >= p + n else None
        )
        sess.sink.on_settled = lambda rid: req_tokens.pop(rid, None)

        # Request 1 is cancelled: its deferred token at pos=2 NEVER arrives
        sess.register_request("cancelled")
        s0 = sess.begin_step([("cancelled", [1, 2])])
        sess.ring.publish(s0.span, b"\x01" * (2 * rb))
        sess.end_step(s0)
        s1 = sess.begin_step([("cancelled", 1, 2)])
        sess.ring.publish(s1.span, b"\x02" * rb)
        sess.end_step(s1)
        sess.finish_requests(["cancelled"])

        # Request 2 is healthy and queued right behind Request 1
        sess.register_request("healthy")
        s2 = sess.begin_step([("healthy", [10, 11])])
        sess.ring.publish(s2.span, b"\x03" * (2 * rb))
        sess.end_step(s2)
        sess.finish_requests(["healthy"])

        # Initially pump blocks briefly; after deferred_timeout_s, cancelled is poisoned & cleaned up, healthy succeeds!
        sess.sink.pump()
        import time as _t
        _t.sleep(0.03)
        while sess.sink.pending:
            sess.sink.pump()
        self.assertIn("cancelled", sess.sink.failed)
        self.assertIn("healthy", sess.sink.finished)
        self.assertEqual(sess.sink.finished["healthy"].n_tokens, 2)
        self.assertEqual(len(req_tokens), 0)

    def test_fatal_sink_exception_still_invokes_on_settled_cleanup(self) -> None:
        from fpn_serve.capture.ports import CaptureTier, ModelDims
        from fpn_serve.capture.session import CaptureConfig, CaptureSession
        from fpn_serve.capture.sink import SinkFaultedError

        dims = ModelDims(n_layers=2, d_model=16, n_q_heads=2, n_kv_heads=1, head_dim=8, d_ff=32)
        sess = CaptureSession.create(dims, CaptureConfig(tier=CaptureTier.TIER_B, ring_mib=8, max_batched_tokens=32))
        rb = sess.layout.stride_bytes
        req_tokens: dict[str, list[int]] = {"r_fault": [5, 6]}
        sess.sink.on_settled = lambda rid: req_tokens.pop(rid, None)
        sess.register_request("r_fault")
        s0 = sess.begin_step([("r_fault", [5, 6])])
        sess.ring.publish(s0.span, b"\x04" * (2 * rb))
        sess.end_step(s0)
        # Force an unexpected non-CaptureError inside store.handle
        sess.store.handle = lambda _rid: (_ for _ in ()).throw(RuntimeError("injected fatal fault"))
        with self.assertRaises(SinkFaultedError):
            sess.sink.pump()
        self.assertEqual(len(req_tokens), 0)

        # Any concurrent/post-fault submit must fail atomically under _cv, never enqueue, and invoke on_settled!
        req_tokens["r_late"] = [7, 8]
        from fpn_serve.capture.ring import Span
        from fpn_serve.capture.sink import Segment, StepRecord
        late_rec = StepRecord(
            span=Span(0, 1),
            segments=(Segment("r_late", 0, 1, (7,)),),
            ready_token=None,
            finish=("r_late",),
        )
        with self.assertRaises(SinkFaultedError):
            sess.sink.submit(late_rec)
        self.assertEqual(sess.sink.pending, 0)
        self.assertEqual(len(req_tokens), 0)

    def test_tap_expert_ids_validates_rank_width_device_and_materializes_strided_views(self) -> None:
        """Security audit MAJOR regression: tap_expert_ids rejects bad rank/width/device and makes inner stride 1."""
        import sys
        import types
        from unittest.mock import patch
        from fpn_serve.capture.kernels import TapLauncher
        from fpn_serve.capture.ports import PortSlot

        class _FakeTensor:
            def __init__(self, shape, strides, dtype="i32", is_cuda=True, device="cuda:0"):
                self.shape = tuple(shape)
                self.ndim = len(self.shape)
                self._strides = tuple(strides)
                self.dtype = dtype
                self.is_cuda = is_cuda
                self.device = device
                self.made_contiguous = False

            def stride(self, dim):
                return self._strides[dim]

            def to(self, dtype):
                return _FakeTensor(self.shape, self._strides, dtype=dtype, is_cuda=self.is_cuda, device=self.device)

            def contiguous(self):
                # Mirror PyTorch: a singleton outer dim with stride(0)==0 is already considered contiguous!
                if self.shape[0] == 1 and self._strides[1] == 1:
                    return self
                t = _FakeTensor(self.shape, (self.shape[1], 1), dtype=self.dtype, is_cuda=self.is_cuda, device=self.device)
                t.made_contiguous = True
                return t

            def copy_(self, src):
                self.copied_from = src
                return self

        def _fake_empty(shape, dtype="i32", device="cuda:0"):
            t = _FakeTensor(shape, (shape[1], 1), dtype=dtype, is_cuda=True, device=device)
            t.made_contiguous = True
            return t

        fake_torch = types.SimpleNamespace(int32="i32", empty=_fake_empty)
        launched = []
        ln = object.__new__(TapLauncher)
        ln.buf = _FakeTensor((8, 64), (64, 1))
        ln.stride = 64
        ln.capacity = 8
        ln.ctl = _FakeTensor((4,), (1,))
        ln.ext = types.SimpleNamespace(
            launch_tap_expert_ids_i32_to_u16=lambda iv, st, off, w, *rest: launched.append((iv, st, off, w))
        )
        slot = PortSlot(kind="expert_ids", layer=0, offset=16, width=8)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(ValueError):
                ln.tap_expert_ids(_FakeTensor((4,), (1,)), slot)                      # wrong rank
            with self.assertRaises(ValueError):
                ln.tap_expert_ids(_FakeTensor((4, 4), (4, 1)), slot)                  # wrong width
            with self.assertRaises(ValueError):
                ln.tap_expert_ids(_FakeTensor((4, 8), (8, 1), is_cuda=False), slot)   # non-CUDA
            with self.assertRaises(ValueError):
                ln.tap_expert_ids(_FakeTensor((4, 8), (8, 1), device="cuda:1"), slot) # wrong device
            # Non-contiguous inner stride (e.g. transposed / column-strided view) MUST allocate row-major buffer
            strided = _FakeTensor((4, 8), (1, 4))
            ln.tap_expert_ids(strided, slot)
            # Singleton-row (1, 8) with stride (0, 1) (Security audit ): MUST NOT keep stride(0)==0
            singleton = _FakeTensor((1, 8), (0, 1))
            ln.tap_expert_ids(singleton, slot)
            self.assertEqual(len(launched), 2)
            for iv, st, off, w in launched:
                self.assertTrue(iv.made_contiguous)
                self.assertEqual((iv.stride(0), iv.stride(1), st, off, w), (8, 1, 8, 16, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
