#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 frontier calibration (gate v1): RoundingContract calibration and cost model on real checkpoints.

For each model the real mode loads the checkpoint, runs honest forward passes and counts *bad blocks*: 32-row
GEMM output tiles whose observed output leaves the certified radius of the model's ``RoundingContract`` around
an fp64 recomputation from the same weights and inputs.  It also records the minimum residual RMS per layer
(the ``1e-3`` non-degeneracy premise), MoE router top-k margins, detection of a requantised-weight cheat, and the
serving GPU seconds that enter the ``p_audit`` steady-state cost model.

    python bench/tier2/bench_frontier_calibrate.py --models-dir <WORK_DIR> \\
        --out benchmarks/prover/results/frontier_calibrate.json --confirm-gpu

``--dry-run`` needs neither torch nor a GPU.  It exercises the same contracts, radii, sampler and cost model on
small synthetic matrices with an exact pure-Python emulation of fp32 accumulation and bf16 / e4m3 / e2m1 rounding.
Its output is labelled ``DRY_RUN_SYNTHETIC`` and is NOT calibration evidence: a contract may only be marked
``CALIBRATED`` from a real-mode result file.

What is and is not measured:

* bad blocks, RMS floor, router margins, serving GPU-s: measured in real mode;
* prover GPU-s: EXTRAPOLATED as (reference-IR cell census of the sampled blocks) x (kernel-pipeline ns per cell
  from ``benchmarks/prover/results/tier0_folded_gpu_4070tis.json``).  That figure excludes host-side sumcheck, transcript
  and witness generation, so the cost ratio is a lower bound until an end-to-end prover exists for these models.

The real mode refuses to start without ``--confirm-gpu`` because the calibration host is shared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "prover"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fpn_prover.circuit import protocol as PR  # noqa: E402
from fpn_prover.circuit.census import Profile, census  # noqa: E402
from fpn_prover.circuit.microprogram import swa_indices  # noqa: E402
from fpn_prover.ir.format import (  # noqa: E402
    NUM_FORMATS, exact_dequant_scale_exp_range, gemm_error_bound, get_model_format, weight_commitment,
)
from fpn_prover.ir.gadget_ir import topk_tau  # noqa: E402

KERNEL_NS_PER_CELL = Fraction(845, 100)      # tier0_folded_gpu_4070tis.json: 425.1 ms / 50.3M scalar cells
GATE = Fraction(1, 20)
DRY_K = 64                                   # inner dimension of the synthetic GEMMs (sequential fp32 depth 64)
DRY_ROWS = 64                                # two 32-row tiles
DRY_TOKENS = 6
NOT_COVERED = ("attention kernels (GQA softmax, SWA + sinks, Gated DeltaNet recurrence) are not calibrated here: "
               "only Linear GEMMs, fused MoE expert GEMMs (FP8 block-128 FP8Experts.linear; MXFP4 block-32 triton "
               "matmul_ogs, raw rows re-run without the fused SwiGLU / scatter epilogue), the router logits and "
               "top-k margins, and the residual RMS floor are checked; the SwiGLU epilogue, the expert "
               "scatter-combine and the routing softmax are not")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_format: str
    arch: tuple                 # feature tags exercised by the checks
    hidden: int                 # nominal shapes below are replaced by config.json when the checkpoint is present
    intermediate: int
    layers: int
    heads: int
    kv_groups: int
    head_dim: int
    experts: int = 1
    top_k: int = 1
    window: int = 0
    sink: int = 0


MODELS = (
    ModelSpec("Qwen3-30B-A3B-FP8", "fp8_e4m3_block128_fp32acc", ("moe_router_topk", "gqa", "qk_norm"),
              2048, 768, 48, 32, 4, 128, experts=128, top_k=8),
    ModelSpec("gpt-oss-20b", "mxfp4_e2m1_block32_fp32acc", ("moe_router_topk", "swa_sinks", "gqa"),
              2880, 2880, 24, 64, 8, 64, experts=32, top_k=4, window=128, sink=4),
    ModelSpec("Qwen3.5-0.8B", "bf16", ("gated_deltanet", "gqa"), 1024, 3072, 24, 16, 8, 128),
    ModelSpec("Llama-3.2-1B-Instruct", "bf16", ("gqa",), 2048, 8192, 16, 32, 8, 64),
    ModelSpec("Qwen3-4B", "bf16", ("gqa", "qk_norm"), 2560, 9728, 36, 32, 8, 128),
)


# ====================================================================================== exact rounding emulation
def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x: float) -> float:
    word = struct.unpack(">I", struct.pack(">f", x))[0]
    word = ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) << 16
    return struct.unpack(">f", struct.pack(">I", word))[0]


def grid(fmt) -> list:
    """Every non-negative finite value of a small float format, ascending, as (value, even_code)."""
    out = []
    for e in range(1 << fmt.exp_bits):
        for m in range(1 << fmt.man_bits):
            frac = Fraction(m, 1 << fmt.man_bits)
            value = (frac if e == 0 else 1 + frac) * Fraction(2) ** ((1 if e == 0 else e) - fmt.bias)
            if value <= fmt.max_finite:
                out.append((value, m % 2 == 0))
    return out


def quantize(table: list, x: Fraction) -> Fraction:
    """Round-to-nearest-even onto ``table``, saturating at the largest finite value."""
    sign, mag = (-1 if x < 0 else 1), abs(x)
    lo, hi = 0, len(table) - 1
    if mag >= table[hi][0]:
        return sign * table[hi][0]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if table[mid][0] <= mag else (lo, mid)
    below, above = table[lo], table[hi]
    if mag - below[0] != above[0] - mag:
        return sign * (below[0] if mag - below[0] < above[0] - mag else above[0])
    return sign * (below[0] if below[1] else above[0])


class Stream:
    """Deterministic uniform rationals from BLAKE2b (no ``random``: results are reproducible byte for byte)."""

    def __init__(self, label: str):
        self.label, self.i = label.encode(), 0

    def unit(self) -> Fraction:
        self.i += 1
        word = hashlib.blake2b(self.label + self.i.to_bytes(8, "big"), digest_size=8).digest()
        return Fraction(int.from_bytes(word, "big"), 1 << 64)

    def signed(self, scale: float = 1.0) -> float:
        return float((self.unit() * 2 - 1)) * scale


def pow2_floor(x: Fraction) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e if Fraction(2) ** e <= x else e - 1


def quantize_weights(mf, rows: list) -> tuple:
    """Exactly scaled weights of the model format and the per-row scale (for reporting)."""
    fmt = mf.weight
    if fmt.scale is None:
        return [[Fraction(bf16(v)) for v in row] for row in rows], "none"
    table = grid(NUM_FORMATS["fp4_e2m1" if fmt.bits == 4 else "fp8_e4m3"])
    out = [[Fraction(0)] * len(row) for row in rows]
    br, bc = fmt.scale.block_rows, fmt.scale.block_cols
    for r0 in range(0, len(rows), br):
        for c0 in range(0, len(rows[0]), bc):
            cells = [(r, c) for r in range(r0, min(r0 + br, len(rows))) for c in range(c0, min(c0 + bc, len(rows[0])))]
            peak = max(abs(Fraction(rows[r][c])) for r, c in cells)
            if fmt.scale.pow2:
                scale = Fraction(2) ** (pow2_floor(peak) - pow2_floor(fmt.max_finite)) if peak else Fraction(1)
            else:
                scale = Fraction(f32(float(peak / fmt.max_finite))) if peak else Fraction(1)
            for r, c in cells:
                out[r][c] = quantize(table, Fraction(rows[r][c]) / scale) * scale
    return out, fmt.scale.scale_format


def quantize_activation(mf, x: list) -> tuple:
    """Per-token activation Store step: (operands the kernel multiplies, block scale or None)."""
    fmt = mf.activation
    if fmt.scale is None:
        return [Fraction(v) for v in x], None
    table = grid(NUM_FORMATS["fp8_e4m3"])
    peak = max(abs(Fraction(v)) for v in x)
    scale = Fraction(f32(float(peak / fmt.max_finite))) if peak else Fraction(1)
    return [quantize(table, Fraction(v) / scale) * scale for v in x], scale


def engine_dot(weights: list, operands: list) -> float:
    """fp32 weights and products, sequential fp32 accumulation, bf16 store."""
    acc = 0.0
    for w, a in zip(weights, operands, strict=True):
        acc = f32(acc + f32(f32(float(w)) * float(a)))
    return bf16(acc)


def coarsen(weights: list) -> list:
    """Cheat: every weight re-rounded to one significant bit (a cheaper, lower-precision model)."""
    out = []
    for row in weights:
        new = []
        for w in row:
            if w == 0:
                new.append(w)
                continue
            p = Fraction(2) ** pow2_floor(abs(w))
            new.append((p if abs(w) < p * Fraction(3, 2) else 2 * p) * (1 if w > 0 else -1))
        out.append(new)
    return out


# ====================================================================================== checks shared by both modes
def cost_model(spec: ModelSpec, serving_gpu_s: Optional[float], tokens: int, regime: str = "external_tight") -> dict:
    """Extrapolated prover GPU-s for one fully proved request and the p_audit steady state."""
    profile = Profile(spec.name, spec.hidden, spec.intermediate, spec.heads, spec.kv_groups, spec.head_dim,
                      spec.experts, spec.top_k)
    _, sampled, _ = census(profile, tokens)
    draws = PR.REGIMES[regime].draws
    cells = sum(sampled.values()) * 2 * draws * PR.lineage_bound()
    prover_s = Fraction(cells) * KERNEL_NS_PER_CELL / 10**9
    out: dict[str, Any] = {
        "regime": regime, "draws_per_pool": draws, "lineage_units_per_draw": PR.lineage_bound(),
        "context_tokens": tokens, "sampled_cells_per_draw": sum(sampled.values()), "cells_per_proof": cells,
        "prover_gpu_s": float(prover_s),
        "prover_gpu_s_label": "EXTRAPOLATED: reference-IR census x 8.45 ns/cell kernel pipeline (4070 Ti SUPER); "
                              "excludes host-side sumcheck, transcript and witness generation",
        "gate": float(GATE),
        "p_audit_max_if_cost_ratio": {str(r): float(PR.audit_cost(Fraction(r), 0, gate=GATE).p_audit_max)
                                      for r in ("1/20", "1/2", "1", "5", "50")},
    }
    if serving_gpu_s is None or serving_gpu_s <= 0:
        out.update(serving_gpu_s=None, cost_ratio=None, gate_status="UNMEASURED: serving GPU-s needs a real run")
        return out
    ratio = prover_s / Fraction(serving_gpu_s)
    full = PR.audit_cost(ratio, 1, gate=GATE)
    out.update(serving_gpu_s=serving_gpu_s, cost_ratio=float(ratio), overhead_at_p_audit_1=float(full.overhead),
               p_audit_max=float(full.p_audit_max),
               gate_status=("PASS at p_audit = 1" if full.passes else
                            f"PASS only for p_audit <= {float(full.p_audit_max):.4g} (commit-all / prove-sampled)"))
    return out


def sampler_profile(spec: ModelSpec) -> PR.SamplerProfile:
    h, m, d = spec.hidden, spec.intermediate, spec.head_dim
    q, kv = spec.heads * d, spec.kv_groups * d
    gemms = [(h, q), (h, kv), (h, kv), (q, h)] + [(h, m), (h, m), (m, h)] * spec.top_k
    if spec.experts > 1:
        gemms.append((h, spec.experts))
    return PR.SamplerProfile(spec.layers, tuple(gemms), spec.heads, spec.kv_groups, d,
                             window=spec.window, sink=spec.sink, attn_gemms=4)


def format_report(spec: ModelSpec) -> dict:
    mf = get_model_format(spec.model_format)
    bound = gemm_error_bound(mf, spec.hidden)
    commit = weight_commitment(mf, spec.intermediate, spec.hidden)
    return {
        "model_format": mf.name, "gemm_contract": mf.gemm.name, "contract_calibration": mf.gemm.calibration.value,
        "dot_relative_bound_log2": float(PR.log2(bound.relative)), "eta_log2": bound.eta_log2,
        "activation_store_u": float(bound.operand_store), "weight_exact_in_accumulator": bound.weight_exact,
        "exact_dequant_scale_exp_range_bf16": exact_dequant_scale_exp_range(mf.weight, NUM_FORMATS["bf16"]),
        "weight_commitment_one_mlp_matrix": {k: (float(v) if isinstance(v, Fraction) else v)
                                             for k, v in commit.items()},
    }


# ====================================================================================== dry run
def dry_run_model(spec: ModelSpec) -> dict:
    mf = get_model_format(spec.model_format)
    bound = gemm_error_bound(mf, DRY_K)
    rng = Stream("fpn-dry-run:" + spec.name)
    raw = [[rng.signed(0.05) * (4.0 if c % 37 == 0 else 1.0) for c in range(DRY_K)] for _ in range(DRY_ROWS)]
    weights, scale_kind = quantize_weights(mf, raw)
    cheat = coarsen(weights)
    eta = Fraction(2) ** bound.eta_log2
    arith = (1 + bound.arithmetic) * (1 + bound.weight_store) - 1
    honest_bad = cheat_bad = blocks = 0
    loose_bad = 0
    max_ratio = Fraction(0)
    min_rms_sq = None
    for t in range(DRY_TOKENS):
        x = [bf16(rng.signed(3.0) * (40.0 if c == 5 and t == 2 else 1.0)) for c in range(DRY_K)]
        rms_sq = sum(Fraction(v) ** 2 for v in x) / DRY_K
        min_rms_sq = rms_sq if min_rms_sq is None else min(min_rms_sq, rms_sq)
        operands, scale = quantize_activation(mf, x)
        for tile in range(0, DRY_ROWS, PR.TILE_ROWS):
            blocks += 1
            bad = bad_loose = bad_cheat = False
            for r in range(tile, tile + PR.TILE_ROWS):
                mass = sum(abs(w * a) for w, a in zip(weights[r], operands, strict=True))
                ref = sum(w * a for w, a in zip(weights[r], operands, strict=True))
                radius = arith * mass + eta
                dev = abs(Fraction(engine_dot(weights[r], operands)) - ref)
                max_ratio = max(max_ratio, dev / radius)
                bad |= dev > radius
                bad_cheat |= abs(Fraction(engine_dot(cheat[r], operands)) - ref) > radius
                # Radius against the *unquantised* activations (activation Store folded into the bound).
                ref_u = sum(w * Fraction(a) for w, a in zip(weights[r], x, strict=True))
                mass_u = sum(abs(w * Fraction(a)) for w, a in zip(weights[r], x, strict=True))
                floor = (Fraction(2) ** bound.operand_floor_log2 * scale * sum(map(abs, weights[r]))
                         if scale is not None else 0)
                bad_loose |= abs(Fraction(engine_dot(weights[r], operands)) - ref_u) > bound.relative * mass_u + floor + eta
            honest_bad += bad
            loose_bad += bad_loose
            cheat_bad += bad_cheat
    result: dict[str, Any] = {
        "model": spec.name, "arch": list(spec.arch), "evidence": "DRY_RUN_SYNTHETIC (not a calibration measurement)",
        "shapes": "NOMINAL: built-in shapes, not read from a checkpoint; cost figures inherit this",
        "not_covered": NOT_COVERED,
        "format": format_report(spec), "weight_scale": scale_kind,
        "gemm": {"blocks": blocks, "honest_bad_blocks": honest_bad, "honest_bad_blocks_unquantised_radius": loose_bad,
                 "honest_max_ratio": float(max_ratio), "requantised_weight_cheat_bad_blocks": cheat_bad},
        "rms_floor": {"min_rms": float(min_rms_sq) ** 0.5,
                      "holds": PR.residual_floor_holds(min_rms_sq * DRY_K, DRY_K)},
    }
    checks = [honest_bad == 0, loose_bad == 0, cheat_bad == blocks, result["rms_floor"]["holds"]]
    if spec.experts > 1:
        scores = [int(rng.unit() * 2**20) for _ in range(spec.experts)]
        order = sorted(range(spec.experts), key=lambda i: (-scores[i], i))
        chosen = sorted(order[:spec.top_k])
        bitmap = [int(i in chosen) for i in range(spec.experts)]
        honest = len(chosen) == spec.top_k and topk_tau(scores, bitmap, chosen, 0)
        dropped = chosen[:-1]                                          # cheat: run one expert fewer
        swapped = sorted(order[1:spec.top_k + 1])                      # cheat: skip the best expert
        drop_ok = len(dropped) == spec.top_k and topk_tau(scores, [int(i in dropped) for i in range(spec.experts)],
                                                          dropped, 0)
        swap_ok = topk_tau(scores, [int(i in swapped) for i in range(spec.experts)], swapped, 0)
        result["moe_router"] = {"experts": spec.experts, "top_k": spec.top_k, "honest_accepted": bool(honest),
                                "dropped_expert_accepted": bool(drop_ok), "skipped_best_accepted": bool(swap_ok)}
        checks += [honest, not drop_ok, not swap_ok]
    if spec.window:
        keys = swa_indices(4 * spec.window, spec.window, spec.sink)
        result["swa_sinks"] = {"window": spec.window, "sink": spec.sink, "keys_at_position": len(keys),
                               "keeps_sinks": keys[:spec.sink] == tuple(range(spec.sink))}
        checks += [len(keys) == spec.window + spec.sink, result["swa_sinks"]["keeps_sinks"]]
    profile = sampler_profile(spec)
    seed = hashlib.blake2b(spec.name.encode(), digest_size=32).digest()
    draws = PR.sample_request(seed, profile, input_tokens=32768, cache_read_input_tokens=32000, output_tokens=256,
                              draws=PR.REGIMES["external_tight"].draws)
    fresh = draws[PR.STRATUM_FRESH_INPUT]
    result["sampler"] = {
        "draws_per_pool": len(fresh), "gemm_tile_draws": sum(d.kind == "gemm" for d in fresh),
        "attn_group_draws": sum(d.kind == "attn" for d in fresh),
        "cached_positions_drawn": sum(d.position < 32000 for d in fresh),
        "heads_per_kv_group": profile.heads_per_group,
    }
    checks += [result["sampler"]["cached_positions_drawn"] == 0,
               all(len(d.lineage()) <= PR.lineage_bound() for d in fresh),
               all(d.tile_rows <= PR.TILE_ROWS and d.tile_start % PR.TILE_ROWS == 0 for d in fresh if d.kind == "gemm")]
    result["cost"] = cost_model(spec, None, 4096)
    result["status"] = "DRY_RUN_OK" if all(checks) else "DRY_RUN_FAILED"
    return result


# ====================================================================================== real mode (torch, GPU)
def load_shapes(spec: ModelSpec, path: str) -> ModelSpec:
    with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    def pick(*names, default):
        return next((cfg[n] for n in names if cfg.get(n)), default)

    heads = pick("num_attention_heads", default=spec.heads)
    hidden = pick("hidden_size", default=spec.hidden)
    experts = pick("num_experts", "num_local_experts", default=spec.experts)
    return ModelSpec(
        spec.name, spec.model_format, spec.arch, hidden,
        pick("moe_intermediate_size", "intermediate_size", default=spec.intermediate),
        pick("num_hidden_layers", default=spec.layers), heads,
        pick("num_key_value_heads", default=spec.kv_groups), pick("head_dim", default=hidden // heads),
        experts, pick("num_experts_per_tok", "experts_per_token", default=spec.top_k),
        pick("sliding_window", default=spec.window), spec.sink)


def calibrate_real(spec: ModelSpec, path: str, args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = load_shapes(spec, path)
    mf = get_model_format(spec.model_format)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="cuda").eval()
    stats = {"linears": 0}
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def _dequant_weight_slice(module, sl):
        w = module.weight.detach()
        ws = w[sl].double()
        scale_inv = getattr(module, "weight_scale_inv", None)
        if scale_inv is not None:
            bs = getattr(module, "block_size", (128, 128))
            br, bc = (bs[0], bs[1]) if isinstance(bs, (tuple, list)) else (128, 128)
            r0 = sl.start or 0
            r1 = sl.stop if sl.stop is not None else w.shape[0]
            rows_idx = torch.arange(r0, r1, device=w.device) // br
            cols_idx = torch.arange(w.shape[1], device=w.device) // bc
            s = scale_inv.detach()[rows_idx][:, cols_idx].double()
            ws = ws * s
        return ws

    def check_tile(bucket, x2d, w64, bias64, out_tile, fp8_block, x_dtype):
        """Bad-block check of one 32-row output tile against the contract radius around an fp64 recomputation
        from the same weights and the inputs that actually flowed; plus the requantised-weight cheat."""
        bound = gemm_error_bound(mf, w64.shape[1])
        # For native block-scaled FP8, folding E4M3 activation quantisation (u = 2^-4) into the dot-product box
        # hides 4-bit weight cheating; verifying the deterministic per-block-128 activation quantisation
        # x -> (x_q, s_x) in-circuit tightens the box to the accumulation + store radius.
        if fp8_block and mf.activation.bits == 8 and x2d.shape[-1] % fp8_block == 0:
            xb = x2d.float().reshape(x2d.shape[0], -1, fp8_block)
            sx = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / float(mf.activation.max_finite)
            xq = (xb / sx).clamp(-float(mf.activation.max_finite), float(mf.activation.max_finite)).to(torch.float8_e4m3fn)
            x64 = (xq.double() * sx.double()).reshape_as(x2d)
            rel = float((1 + bound.arithmetic) * (1 + bound.weight_store) - 1)
            floor_log2 = None
        else:
            x64 = x2d.double()
            rel = float(bound.relative)
            floor_log2 = bound.operand_floor_log2
        if bias64 is not None:
            rel = rel * (1 + 2.0 ** -24) + 2.0 ** -24
        ref = x64 @ w64.T
        mass = x64.abs() @ w64.abs().T
        if bias64 is not None:
            ref = ref + bias64
            mass = mass + bias64.abs()
        radius = rel * mass + 2.0 ** bound.eta_log2
        if floor_log2 is not None:
            scale = x64.abs().amax(-1, keepdim=True) / float(mf.activation.max_finite)
            radius = radius + 2.0 ** floor_log2 * scale * w64.abs().sum(-1)
        dev = (out_tile.double() - ref).abs()
        bad = (dev > radius).any(-1)
        bucket["calls"] += 1
        bucket["blocks"] += bad.numel()
        bucket["bad"] += int(bad.sum())
        bucket["max_ratio"] = max(bucket["max_ratio"], float((dev / radius).max()))
        # Requantised-weight cheat (4-bit symmetric per row), evaluated in the activation dtype on the same inputs.
        wf = w64.float()
        peak = wf.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        wq = (torch.round(wf / peak * 7) / 7 * peak).to(x_dtype)
        cheat = (x2d.to(x_dtype) @ wq.T).double()
        if bias64 is not None:
            cheat = cheat + bias64
        cheat_bad = ((cheat - ref).abs() > radius).any(-1)
        bucket["cheat_blocks"] += cheat_bad.numel()
        bucket["cheat_bad"] += int(cheat_bad.sum())

    def tile_slice(rows):
        start = int(torch.randint(0, max(1, -(-rows // PR.TILE_ROWS)), (1,), generator=gen)) * PR.TILE_ROWS
        return slice(start, min(rows, start + PR.TILE_ROWS))

    def hook(module, inputs, output):
        x = inputs[0].detach()
        w = module.weight.detach()
        if x.shape[-1] != w.shape[1] or not torch.is_floating_point(output):
            return
        stats["linears"] += 1
        key_dt = f"{type(module).__name__}:{str(w.dtype).replace('torch.', '')}"
        linear_dtypes[key_dt] = linear_dtypes.get(key_dt, 0) + 1
        rows = w.shape[0]
        sl = tile_slice(rows)
        fp8_block = None
        if getattr(module, "weight_scale_inv", None) is not None:
            bs = getattr(module, "block_size", (128, 128))
            fp8_block = bs[1] if isinstance(bs, (tuple, list)) else 128
        bias64 = module.bias.detach()[sl].double() if module.bias is not None else None
        x2d = x.reshape(-1, x.shape[-1])
        check_tile(linear_bucket, x2d, _dequant_weight_slice(module, sl), bias64,
                   output.detach().reshape(-1, rows)[:, sl], fp8_block, x.dtype)

    # ------------------------------------------------------------------ MoE router (real logits that flowed)
    bf16_mf = get_model_format("bf16")

    def router_check(x2d, weight, bias, logits, top_k):
        x64, w64 = x2d.double(), weight.detach().double()
        ref, mass = x64 @ w64.T, x64.abs() @ w64.abs().T
        bound = gemm_error_bound(bf16_mf, w64.shape[1])
        rel = float(bound.relative)
        lg = logits.detach().reshape(-1, w64.shape[0])
        extra = 0.0
        if bias is not None:
            # F.linear(x, W, b) in bf16 may add the bias as a separate bf16 op (observed on sm_120 for small M):
            # model it as the bf16 GEMM radius plus ONE more bf16 rounding of the stored logit (u = 2^-8 |y|).
            ref, mass = ref + bias.detach().double(), mass + bias.detach().double().abs()
            extra = 2.0 ** -8
        radius = rel * mass + extra * lg.double().abs() + 2.0 ** bound.eta_log2
        dev = (lg.double() - ref).abs()
        r = router
        r["calls"] += 1
        r["tokens"] += lg.shape[0]
        r["bad_rows"] += int((dev > radius).any(-1).sum())
        r["max_ratio"] = max(r["max_ratio"], float((dev / radius).max()))
        top = lg.double().topk(top_k + 1, dim=-1).values
        margin = top[:, top_k - 1] - top[:, top_k]
        tau = radius.max(-1).values          # per-token certified logit error radius
        r["margins"].append(margin.float().cpu())
        r["taus"].append(tau.float().cpu())
        # Expert-substitution cheat: serve expert k+1 instead of the k-th selected expert.  The in-circuit top-k
        # check (topk_tau with slack 2 tau) rejects it iff the honest margin exceeds 2 tau.  Dropping an expert
        # outright changes the index-list cardinality and is rejected unconditionally.
        r["swap_detected"] += int((margin > 2 * tau).sum())

    def qwen_router_hook(module, inputs, output):
        router_check(inputs[0].reshape(-1, inputs[0].shape[-1]), module.weight, getattr(module, "bias", None),
                     output[0], module.top_k)

    def oss_mlp_hook(module, inputs, output):          # MXFP4 mlp_forward returns (routed_out, router_logits)
        hs = inputs[0]
        router_check(hs.reshape(-1, hs.shape[-1]), module.router.weight, module.router.bias, output[1],
                     module.router.top_k)

    # ------------------------------------------------------------------ fused MoE experts (real packed weights)
    def wrap_fp8_experts(module):
        orig = module.linear
        bc = module.block_size[1] if module.block_size else 128
        br = module.block_size[0] if module.block_size else 128

        def linear(inp, weight, weight_scale_inv, activation_scale=None):
            out = orig(inp, weight, weight_scale_inv, activation_scale=activation_scale)
            if weight.element_size() == 1 and capture["on"]:
                proj = "fp8_expert_gate_up" if weight.shape[0] == 2 * module.intermediate_dim else "fp8_expert_down"
                sl = tile_slice(weight.shape[0])
                r0, r1 = sl.start, sl.stop
                s = weight_scale_inv.detach()[torch.arange(r0, r1, device=weight.device) // br][
                    :, torch.arange(weight.shape[1], device=weight.device) // bc].double()
                w64 = weight.detach()[sl].double() * s
                check_tile(expert_buckets[proj], inp.reshape(-1, inp.shape[-1]), w64, None,
                           out.detach().reshape(-1, weight.shape[0])[:, sl], bc, inp.dtype)
            return out
        module.linear = linear

    oss = {}

    def mxfp4_weights(layer, proj):
        key = (layer, proj)
        if key not in oss:
            from safetensors import safe_open
            with open(os.path.join(path, "model.safetensors.index.json"), encoding="utf-8") as fh:
                index = json.load(fh)["weight_map"]
            base = f"model.layers.{layer}.mlp.experts.{proj}"
            out = []
            for suffix in ("_blocks", "_scales"):
                with safe_open(os.path.join(path, index[base + suffix]), "pt", device="cuda") as sf:
                    out.append(sf.get_tensor(base + suffix))
            oss.clear()                                    # one (layer, proj) resident at a time
            oss[key] = tuple(out)
        return oss[key]

    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float64, device="cuda")

    def mxfp4_dequant(blocks, scales):
        """(rows, in/32, 16) uint8 packed e2m1 pairs (low nibble first) x E8M0 scales -> (rows, in) fp64, exact."""
        lo, hi = (blocks & 0x0F).long(), (blocks >> 4).long()
        vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(blocks.shape[0], blocks.shape[1], 32)
        vals = vals * torch.pow(2.0, scales.double() - 127.0).unsqueeze(-1)
        return vals.reshape(blocks.shape[0], -1)

    def wrap_matmul_ogs(hub):
        orig = hub.matmul_ogs.matmul_ogs

        def mm(x, w, bias, routing_data=None, gather_indx=None, scatter_indx=None, precision_config=None,
               gammas=None, fused_activation=None, **kw):
            out = orig(x, w, bias, routing_data, gather_indx=gather_indx, scatter_indx=scatter_indx,
                       precision_config=precision_config, gammas=gammas, fused_activation=fused_activation, **kw)
            if not capture["on"] or routing_data is None or capture.get("layer") is None:
                return out
            # The same Blackwell MXFP4 kernel, re-run without the fused SwiGLU epilogue / scatter-combine, so the
            # raw per-(token, expert) GEMM rows can be checked (the deployed call above is left untouched).
            raw = orig(x, w, bias, routing_data, gather_indx=gather_indx, scatter_indx=None,
                       precision_config=precision_config, gammas=None, fused_activation=None, **kw)
            proj = "gate_up_proj" if gather_indx is not None else "down_proj"
            k = int(routing_data.n_expts_act)
            src = (gather_indx.src_indx.long() // k) if gather_indx is not None else None
            hist = routing_data.expt_hist.long()
            ends = torch.cumsum(hist, 0).tolist()
            blocks, scales = mxfp4_weights(capture["layer"], proj)
            begin = 0
            for e, end in enumerate(ends):
                if end > begin:
                    rows = slice(begin, end)
                    xin = x[src[rows]] if src is not None else x[rows]
                    sl = tile_slice(blocks.shape[1])
                    w64 = mxfp4_dequant(blocks[e, sl], scales[e, sl])
                    b64 = bias[e, sl].double() if bias is not None else None
                    check_tile(expert_buckets["mxfp4_expert_" + proj.replace("_proj", "")], xin, w64, b64,
                               raw[rows][:, sl], None, x.dtype)
                begin = end
            return out
        hub.matmul_ogs.matmul_ogs = mm
        return lambda: setattr(hub.matmul_ogs, "matmul_ogs", orig)

    def oss_layer_pre(index):
        def pre(module, inputs):
            capture["layer"] = index
        return pre

    capture = {"on": True, "layer": None}
    linear_dtypes: dict[str, int] = {}
    linear_bucket = {"calls": 0, "blocks": 0, "bad": 0, "max_ratio": 0.0, "cheat_blocks": 0, "cheat_bad": 0}
    expert_buckets = {k: dict(linear_bucket) for k in ("fp8_expert_gate_up", "fp8_expert_down",
                                                         "mxfp4_expert_gate_up", "mxfp4_expert_down")}
    router = {"calls": 0, "tokens": 0, "bad_rows": 0, "max_ratio": 0.0, "margins": [], "taus": [], "swap_detected": 0}
    restore = []
    handles = []
    expert_modules = []
    for name, m in model.named_modules():
        cls = type(m).__name__
        if cls.endswith("TopKRouter") and hasattr(m, "weight"):
            handles.append(m.register_forward_hook(qwen_router_hook))
        elif hasattr(m, "router") and hasattr(m, "experts") and type(m.experts).__name__.startswith("Mxfp4"):
            handles.append(m.register_forward_hook(oss_mlp_hook))
        if cls == "FP8Experts":
            wrap_fp8_experts(m)
            expert_modules.append(name)
        elif cls.startswith("Mxfp4") and cls.endswith("Experts"):
            layer = int(name.split("layers.")[1].split(".")[0])
            handles.append(m.register_forward_pre_hook(oss_layer_pre(layer)))
            expert_modules.append(name)
    def wrap_fp8_grouped_matmul(fg):
        orig = fg.grouped_matmul

        def gm(inp, weight, weight_scale_inv, offsets=None, tokens_per_expert=None, block_size=None, **kw):
            out = orig(inp, weight, weight_scale_inv, offsets=offsets,
                       tokens_per_expert=tokens_per_expert, block_size=block_size, **kw)
            if not capture["on"] or offsets is None or weight.element_size() != 1:
                return out
            bs = block_size or (128, 128)
            br, bc = (bs[0], bs[1]) if isinstance(bs, (tuple, list)) else (128, 128)
            proj = "fp8_expert_gate_up" if weight.shape[1] == 2 * spec.intermediate else "fp8_expert_down"
            ends = offsets.long().tolist()
            begin = 0
            for e, end in enumerate(ends):
                if end > begin:
                    rows = slice(begin, end)
                    sl = tile_slice(weight.shape[1])
                    r0, r1 = sl.start, sl.stop
                    s = weight_scale_inv.detach()[e][torch.arange(r0, r1, device=weight.device) // br][
                        :, torch.arange(weight.shape[2], device=weight.device) // bc].double()
                    w64 = weight.detach()[e, sl].double() * s
                    check_tile(expert_buckets[proj], inp[rows].reshape(-1, weight.shape[2]), w64, None,
                               out.detach()[rows, sl], bc, inp.dtype)
                begin = end
            return out
        object.__setattr__(fg, "grouped_matmul", gm)
        return lambda: object.__setattr__(fg, "grouped_matmul", orig)

    if any(type(m).__name__ == "FP8Experts" for m in model.modules()):
        import transformers.integrations.finegrained_fp8 as F8
        restore.append(wrap_fp8_grouped_matmul(F8.load_finegrained_fp8_kernel()))

    if any(type(m).__name__.startswith("Mxfp4") for m in model.modules()):
        import transformers.integrations.mxfp4 as MX
        restore.append(wrap_matmul_ogs(MX.triton_kernels_hub))

    handles += [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        text = " ".join(fh.read().split())
    ids = tok(text, return_tensors="pt").input_ids[:, :args.tokens].to("cuda")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    rms = [float(h.float().pow(2).mean(-1).sqrt().min()) for h in out.hidden_states]
    for h in handles:
        h.remove()
    capture["on"] = False
    for undo in restore:
        undo()
    with torch.no_grad():
        model(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            model(ids)
        torch.cuda.synchronize()
        serving = (time.perf_counter() - t0) / args.repeats
    failures = []
    stats.update(blocks=linear_bucket["blocks"], bad=linear_bucket["bad"], max_ratio=linear_bucket["max_ratio"],
                 cheat_blocks=linear_bucket["cheat_blocks"], cheat_bad=linear_bucket["cheat_bad"])
    if stats["linears"] == 0 or stats["blocks"] == 0:
        failures.append("no GEMM was observed (the checkpoint does not expose nn.Linear modules)")
    if stats["bad"]:
        failures.append(f"{stats['bad']} honest bad blocks")
    moe_router = {"experts": spec.experts, "top_k": spec.top_k, "min_topk_margin": None}
    moe_experts = {}
    if spec.experts > 1:
        if router["calls"] == 0:
            failures.append("MoE router hook never fired")
        else:
            margins, taus = torch.cat(router["margins"]), torch.cat(router["taus"])
            q = torch.quantile(margins.double(), torch.tensor([0.0, 0.001, 0.01, 0.5], dtype=torch.float64))
            moe_router.update(
                router_calls=router["calls"], router_tokens=router["tokens"], logit_units="raw router logits",
                min_topk_margin=float(margins.min()), margin_p0001=float(q[1]), margin_p01=float(q[2]),
                margin_median=float(q[3]), zero_margin_tokens=int((margins == 0).sum()),
                tau_max=float(taus.max()), tau_median=float(taus.median()),
                honest_router_bad_rows=router["bad_rows"], honest_router_max_ratio=router["max_ratio"],
                expert_swap_cheat_tokens=router["tokens"], expert_swap_cheat_detected=router["swap_detected"],
                expert_swap_cheat_detection_rate=router["swap_detected"] / router["tokens"],
                expert_drop_cheat="rejected unconditionally (index-list cardinality != top_k)")
            if router["bad_rows"]:
                failures.append(f"{router['bad_rows']} honest router rows outside the bf16 radius")
        used = {k: v for k, v in expert_buckets.items() if v["calls"]}
        if not used:
            failures.append("no fused MoE expert GEMM was observed")
        for kname, b in used.items():
            moe_experts[kname] = {
                "calls": b["calls"], "blocks": b["blocks"], "honest_bad_blocks": b["bad"],
                "honest_max_ratio": b["max_ratio"], "requantised_weight_cheat_blocks": b["cheat_blocks"],
                "requantised_weight_cheat_bad_blocks": b["cheat_bad"],
                "requantised_weight_cheat_detection_rate": b["cheat_bad"] / b["cheat_blocks"]}
            if b["bad"]:
                failures.append(f"{b['bad']} honest bad blocks in {kname}")
        moe_experts["modules_hooked"] = len(expert_modules)
    if min(rms) < float(PR.RESIDUAL_RMS_FLOOR):
        failures.append(f"residual RMS {min(rms):.3g} below the floor")
    sigma = float(PR.REGIMES["external_tight"].sigma)
    detection = stats["cheat_bad"] / stats["cheat_blocks"] if stats["cheat_blocks"] else 0.0
    if detection < sigma:
        failures.append(f"requantised-weight cheat flagged on {detection:.3f} of blocks, below sigma = {sigma}")
    return {
        "model": spec.name, "arch": list(spec.arch), "evidence": "MEASURED", "tokens": int(ids.shape[1]),
        "shapes": "config.json", "not_covered": NOT_COVERED,
        "device": torch.cuda.get_device_name(0), "format": format_report(spec),
        "gemm": {"linear_calls": stats["linears"], "linear_module_dtypes": linear_dtypes,
                 "blocks": stats["blocks"], "honest_bad_blocks": stats["bad"],
                 "honest_max_ratio": stats["max_ratio"], "requantised_weight_cheat_blocks": stats["cheat_blocks"],
                 "requantised_weight_cheat_bad_blocks": stats["cheat_bad"],
                 "requantised_weight_cheat_detection_rate": detection},
        "rms_floor": {"min_rms": min(rms), "min_rms_per_layer": rms, "floor": float(PR.RESIDUAL_RMS_FLOOR),
                      "holds": min(rms) >= float(PR.RESIDUAL_RMS_FLOOR)},
        "moe_router": moe_router,
        "moe_experts": moe_experts,
        "cost": cost_model(spec, serving, int(ids.shape[1])),
        "status": "FAILED: " + "; ".join(failures) if failures else "CALIBRATION_OK",
    }


# ====================================================================================== driver
def git_sha() -> str:
    if os.environ.get("FPN_GIT_SHA"):
        return os.environ["FPN_GIT_SHA"]
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default="<WORK_DIR>")
    ap.add_argument("--models", default=",".join(m.name for m in MODELS), help="comma-separated subset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="synthetic pure-Python run (no torch, no GPU)")
    ap.add_argument("--confirm-gpu", action="store_true", help="required for a real run on the shared GPU host")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    wanted = [m for m in MODELS if m.name in args.models.split(",")]
    if not wanted:
        ap.error("no known model selected")
    if not args.dry_run and not args.confirm_gpu:
        print("refusing a real run without --confirm-gpu (the calibration GPU is shared); use --dry-run otherwise",
              file=sys.stderr)
        return 2
    results = []
    for spec in wanted:
        if args.dry_run:
            results.append(dry_run_model(spec))
            continue
        path = os.path.join(args.models_dir, spec.name)
        try:
            results.append(calibrate_real(spec, path, args))
        except Exception as exc:  # one model failing must not hide the others; the run still fails
            results.append({"model": spec.name, "status": f"FAILED: {type(exc).__name__}: {exc}"})
    ok = all(r["status"] in ("DRY_RUN_OK", "CALIBRATION_OK") for r in results)
    report = {
        "bench": "tier2/bench_frontier_calibrate", "mode": "dry_run" if args.dry_run else "real",
        "evidence": ("DRY_RUN_SYNTHETIC: exercises contracts, radii, sampler and cost model on synthetic data; "
                     "not hardware calibration" if args.dry_run else "MEASURED"),
        "git_sha": git_sha(), "command": " ".join(sys.argv), "machine": platform.platform(),
        "soundness": {name: vars(r) for name, r in PR.assert_regimes().items()},
        "models": results, "ok": ok,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
            fh.write("\n")
    for r in results:
        gemm = r.get("gemm", {})
        print(f"{r['model']:24} {r['status']:16} honest_bad={gemm.get('honest_bad_blocks')}/{gemm.get('blocks')} "
              f"cheat_bad={gemm.get('requantised_weight_cheat_bad_blocks')} "
              f"max_ratio={gemm.get('honest_max_ratio')}")
    print(("dry run (synthetic, not calibration evidence): " if args.dry_run else "calibration: ")
          + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
