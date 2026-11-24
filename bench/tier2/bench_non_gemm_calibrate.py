#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 non-GEMM operator calibration (gate v1, the part ``bench_frontier_calibrate.py`` leaves open).

Four operator families are checked against a reference recomputed from the operands that actually flowed:

1. ``gqa_softmax``   GQA softmax attention under the ``AttnFA`` radius ``r_o`` (specs/06 §2.2 row 7, the M0-S1
                     re-verified formula) and the v2 LSE radius ``r_lse_circuit`` of
                     ``protocol.certified_lse_radius`` (``CIRCUIT_NORM_LIFT = 2^-32`` uncapped, ``ETA_BF16 = 2^-126``).
                     Models: ``Qwen3-4B``, ``Llama-3.2-1B-Instruct``, ``Qwen3-30B-A3B-FP8``.
2. ``swa_sinks``     sliding-window attention with sinks on ``gpt-oss-20b``.  The checkpoint's sink is one learned
                     logit per head appended to the softmax (``GptOssAttention.sinks``), not a set of retained
                     leading tokens; ``microprogram.swa_indices`` models the latter and is exercised separately.
3. ``lrec``          Gated DeltaNet recurrence on ``Qwen3.5-0.8B`` under a column-norm enclosure of the sequential
                     recurrence.  NEW in this file (``LREC_SEQ_ENCLOSURE_V0``): specs/06 row 9 states the bound
                     only in outline and lists the chunked ``Scan`` template as open.
4. ``rmsnorm``       RMSNorm on all five models under the ``norm_rsqrt`` endpoint enclosure and the
                     ``bf16_fp32acc`` contract, plus the ``min_rms >= 1e-3`` premise on residual-stream norms.

Every radius fails closed: a non-positive normalizer denominator or a lift ``T * 2^-32 >= 1`` raises
``LseDomainError`` and is counted as ``domain_failures``; nothing is clamped and no radius is widened to reach
zero bad groups.  A model passes only with zero honest bad groups, zero domain failures and its cheats rejected.

    python bench/tier2/bench_non_gemm_calibrate.py --dry-run
    python bench/tier2/bench_non_gemm_calibrate.py --models-dir <WORK_DIR> \\
        --out benchmarks/prover/results/non_gemm_calibrate.json --confirm-gpu

``--dry-run`` needs neither torch nor a GPU.  It runs the same radii against a pure-Python emulation of the fp32 /
bf16 kernels on small synthetic operands.  Its output is labelled ``DRY_RUN_SYNTHETIC`` and is NOT calibration
evidence.  The real mode refuses to start without ``--confirm-gpu`` (shared GPU host) unless ``--device cpu`` is
given; a CPU run is labelled ``PLUMBING_SMOKE_CPU`` and is not calibration evidence either.

What is NOT covered: RoPE, the SwiGLU epilogue, the MoE scatter-combine, the DeltaNet convolution / gates /
gated output norm (``Qwen3_5RMSNormGated``), the attention output gate, and any kernel the hooks cannot observe
(reported as ``UNOBSERVED``, never as a pass).  The reference is fp64 (relative error below 2^-45 on these
sizes, against radii of at least 2^-24 relative); only ``r_lse_circuit`` and the rsqrt endpoints are rational.

LRec enclosures (this note sits below the first 512 tokens of the file on purpose: the real mode tokenizes this
file's own text, and the rtx6000d-node receipt of ``fece9ba`` must stay reproducible on the same operands).  The
enclosure is selected by the kernel that ran: ``LREC_SEQ_ENCLOSURE_V0`` for ``torch_recurrent_gated_delta_rule``
and ``LREC_CHUNK_WY_ENCLOSURE_V1`` (:func:`lrec_chunk_reference`) for the chunk-64 WY kernel
``torch_chunk_gated_delta_rule``, whose solve / cumsum / 64-term rounding V0 does not bound (rtx6000d-node RTX 6000D,
T = 512: 11 of 36,864 final-state columns at up to 1.268 x the V0 radius).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "prover"), ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_frontier_calibrate as GEMM  # noqa: E402
from fpn_prover.circuit import protocol as PR  # noqa: E402
from fpn_prover.circuit.microprogram import swa_indices  # noqa: E402
from fpn_prover.ir.arith import norm_rsqrt  # noqa: E402
from fpn_prover.ir.format import get_contract  # noqa: E402

f32, bf16, Stream, MODELS, ModelSpec, git_sha = (GEMM.f32, GEMM.bf16, GEMM.Stream, GEMM.MODELS, GEMM.ModelSpec,
                                                 GEMM.git_sha)
LseDomainError = PR.LseDomainError

U32, UB, UE = float(PR.U_FP32), float(PR.U_BF16), float(PR.U_ELEM)
ETA32, ETAB = float(PR.ETA_FP32), float(PR.ETA_BF16)
NORM_LIFT = float(PR.CIRCUIT_NORM_LIFT)
KEY_TILE, SPLIT = PR.ATTN_KEY_TILE, PR.ATTN_SPLIT_STEPS
OUTWARD = 1.0 + 2.0 ** -40          # the radii are evaluated in fp64; this margin dominates their rounding
RMS_FLOOR = float(PR.RESIDUAL_RMS_FLOOR)

GQA_MODELS = ("Qwen3-4B", "Llama-3.2-1B-Instruct", "Qwen3-30B-A3B-FP8")
SWA_MODELS = ("gpt-oss-20b",)
LREC_MODELS = ("Qwen3.5-0.8B",)
OPERATORS = ("gqa_softmax", "swa_sinks", "lrec", "rmsnorm")


def operators_for(name: str) -> tuple:
    out = []
    if name in GQA_MODELS:
        out.append("gqa_softmax")
    if name in SWA_MODELS:
        out.append("swa_sinks")
    if name in LREC_MODELS:
        out.append("lrec")
    return tuple(out) + ("rmsnorm",)


def depth(n: int) -> int:
    """Reduction depth ``h(n) = ceil(n / 16) + 64`` of the ``bf16_fp32acc`` contract."""
    return -(-n // 16) + 64


def gamma(n: int) -> float:
    h = depth(n)
    return h * U32 / (1.0 - h * U32)


# ====================================================================================== attention: radii
@dataclass(frozen=True)
class AttnKernel:
    """How a softmax-attention kernel computes, as far as the radius needs to know.

    ``fp32acc``         SDPA / FlashAttention: scores, exponentials and both accumulators in fp32, key tiles of 64
                        with online rescaling, one bf16 store of the output (the M0-S1 contract).
    ``eager_observed``  ``softmax(logits - max)`` in the activation dtype with fp32 accumulation, probabilities
                        stored in bf16, then a bf16 ``P V`` matmul.  The shifted logits are an observed operand,
                        so the softmax reference starts from them (``observed=True`` in :func:`attention_radius`);
                        the logits themselves are checked against ``Q K^T`` by :func:`eager_score_radius`.
    """
    name: str
    sequential: bool        # the normalizer is one length-T fp32 sum (no key tiles)


KERNELS = {"fp32acc": AttnKernel("fp32acc", False), "eager_observed": AttnKernel("eager_observed", True)}


def softmax_ref(z: list) -> tuple:
    zmax = max(z)
    w = [math.exp(v - zmax) for v in z]
    total = math.fsum(w)
    return [v / total for v in w], zmax, zmax + math.log(total)


def attention_reference(q: list, keys: list, values: list, scale: float, sink_logit: Optional[float] = None,
                        logits: Optional[list] = None) -> dict:
    """fp64 reference of one query head: scores, softmax weights, LSE and output.

    A sink is one more key with the given logit and a zero value row.  With ``logits`` the scores are taken as
    given (observed operands, sink included) instead of being recomputed from ``q`` and ``keys``.
    """
    n = len(values)
    if n < 1 or (logits is None and len(keys) != n):
        raise ValueError("attention needs at least one key and matching value rows")
    width = len(values[0])
    if logits is None:
        z = [scale * math.fsum(a * b for a, b in zip(q, k, strict=True)) for k in keys]
        mass = [scale * math.fsum(abs(a * b) for a, b in zip(q, k, strict=True)) for k in keys]
        if sink_logit is not None:
            z, mass = z + [sink_logit], mass + [0.0]
    else:
        z, mass = list(logits), [0.0] * len(logits)
    rows = list(values) + ([[0.0] * width] if len(z) == n + 1 else [])
    if len(rows) != len(z):
        raise ValueError("logits do not match the value rows")
    p, zmax, lse = softmax_ref(z)
    out = [math.fsum(pj * row[c] for pj, row in zip(p, rows)) for c in range(width)]
    return {"z": z, "abs_dot": mass, "p": p, "zmax": zmax, "lse": lse, "o": out, "values": rows}


def attention_radius(ref: dict, head_dim: int, *, kernel: AttnKernel = KERNELS["fp32acc"], observed: bool = False,
                     u_out: float = UB, eta_out: float = ETAB, u_p: float = UB, eta_p: float = ETAB) -> dict:
    """``AttnFA`` output radius ``r_o`` per coordinate and the score radius ``rbar`` (M0-S1 re-verified formula).

    Raises :class:`LseDomainError` when the normalizer denominator ``1 - e_den`` is not positive or the circuit
    lift ``T * 2^-32`` reaches 1: the caller must refuse, never clamp.
    """
    z, p, o, rows = ref["z"], ref["p"], ref["o"], ref["values"]
    tokens, zmax = len(z), ref["zmax"]
    gam = 0.0 if observed else gamma(head_dim)
    rz = [gam * m + 4.0 * U32 * (abs(v) + abs(zmax)) for v, m in zip(z, ref["abs_dot"])]
    rbar = max(rz)
    tiles = -(-tokens // KEY_TILE) + SPLIT
    pv_depth, den_depth = 2 * tiles + depth(KEY_TILE), 2 * tiles + KEY_TILE
    if kernel.sequential:
        pv_depth, den_depth = max(pv_depth, depth(tokens)), max(den_depth, tokens)
    emu = (1.0 + u_p) * (1.0 + U32) ** pv_depth - 1.0
    emu2 = (1.0 + U32) ** den_depth - 1.0
    if 2.0 * rbar >= math.log(2.0):         # then e_den >= exp(2 rbar) - 1 >= 1 (also keeps exp in range)
        raise LseDomainError(f"score radius {rbar:.3g} makes the normalizer denominator non-positive")
    one_s = math.exp(2.0 * rbar) * (1.0 + UE) ** (tiles + 1)
    e_num, e_den0 = one_s * (1.0 + emu) - 1.0, one_s * (1.0 + emu2) - 1.0
    eta_bar = math.exp(rbar) * ((tiles + 1) * (1.0 + emu) * ETA32 + eta_p)
    e_den = e_den0 + tokens * eta_bar
    den = 1.0 - e_den
    if den <= 0.0:
        raise LseDomainError(f"normalizer denominator 1 - e_den = {den:.3g} <= 0")
    if tokens * NORM_LIFT >= 1.0:
        raise LseDomainError(f"normalizer lift T * 2^-32 = {tokens * NORM_LIFT:.3g} >= 1")
    radius = []
    for c, center in enumerate(o):
        column = [row[c] for row in rows]
        spread = math.fsum(pj * abs(v - center) for pj, v in zip(p, column))
        mass = math.fsum(abs(v) for v in column)
        pre = (e_num * spread + one_s * (emu + emu2) * abs(center) + eta_bar * (mass + tokens * abs(center))) / den
        pre += tiles * ETA32
        pre += U32 * (abs(center) + pre)
        peak = max(abs(v) for v in column)
        hull = (one_s * (emu + emu2) * peak + eta_bar * (mass + tokens * peak)) / den + tiles * ETA32
        lo, hi = min(column) - hull, max(column) + hull
        lo, hi = lo - U32 * abs(lo), hi + U32 * abs(hi)
        pre = min(pre, max(center - lo, hi - center))
        radius.append((pre + u_out * (abs(center) + pre) + eta_out) * OUTWARD)
    return {"r_o": radius, "rbar": rbar, "e_den": e_den, "tokens": tokens, "zmax": zmax}


def lse_circuit_radius(rad: dict, *, kernel: AttnKernel = KERNELS["fp32acc"], precision: str = "bf16") -> float:
    """``r_lse_circuit`` of ``protocol.certified_lse_radius`` (rational, outward rounded), fail closed.

    A sequential normalizer adds ``T * u32`` for the part of the length-T sum the tiled formula does not cover.
    """
    cert = PR.certified_lse_radius(rad["tokens"], score_radius_max=Fraction(rad["rbar"]) * Fraction(OUTWARD),
                                   score_max_abs=Fraction(abs(rad["zmax"])), precision=precision)
    extra = rad["tokens"] * U32 if kernel.sequential else 0.0
    return (float(cert.circuit) + extra) * OUTWARD


def eager_score_radius(q: list, key: list, scale: float, z: float, a_max: Optional[float] = None,
                       gap: float = 0.0) -> float:
    """Radius of one bf16-stored logit of an eager kernel against the exact ``scale * <q, k>``.

    ``a = scale gamma <|q|,|k|> (1 + u)^3 + ((1 + u)^3 - 1) |z| + 2 eta`` covers the fp32-accumulated dot, its bf16
    store, the cast of ``scale`` to bf16 and the bf16 store of the product.  With ``a_max`` the result is the
    radius of the max-shifted logit ``bf16(c_j - c_max)`` against the exact gap: ``(1 + u)(a + a_max) + u |gap|``.
    """
    mass = scale * math.fsum(abs(a * b) for a, b in zip(q, key, strict=True))
    grow = (1.0 + UB) ** 3
    own = (gamma(len(q)) * mass * grow + (grow - 1.0) * abs(z) + 2.0 * ETAB) * OUTWARD
    if a_max is None:
        return own
    return ((1.0 + UB) * (own + a_max) + UB * abs(gap) + ETAB) * OUTWARD


def check_head(ref: dict, rad: dict, r_lse: float, observed_out: list, observed_lse: float) -> dict:
    """One query head: output coordinates against ``r_o`` and the LSE against ``r_lse_circuit``."""
    ratios = [abs(a - b) / r for a, b, r in zip(observed_out, ref["o"], rad["r_o"], strict=True)]
    lse_ratio = abs(observed_lse - ref["lse"]) / r_lse
    return {"out_ratio": max(ratios), "lse_ratio": lse_ratio, "bad": max(ratios) > 1.0 or lse_ratio > 1.0}


# ====================================================================================== attention: engines
def dot32(a: list, b: list) -> float:
    """fp32 dot product with 16 accumulation lanes combined in order: every product and partial sum is rounded to
    fp32 and the reduction depth is ``ceil(n / 16) + 15``, inside the contract's ``h(n) = ceil(n / 16) + 64``."""
    lanes = [0.0] * 16
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        lanes[i % 16] = f32(lanes[i % 16] + f32(x * y))
    acc = 0.0
    for lane in lanes:
        acc = f32(acc + lane)
    return acc


def engine_attention_fp32acc(q: list, keys: list, values: list, scale: float, sink_logit: Optional[float] = None,
                             tile: int = KEY_TILE) -> tuple:
    """FlashAttention-style online softmax in fp32 with key tiles, bf16 output store.  Returns (out, lse)."""
    scale32, width = f32(scale), len(values[0])
    scores = [f32(dot32(q, k) * scale32) for k in keys]
    rows = list(values)
    if sink_logit is not None:
        scores, rows = scores + [f32(sink_logit)], rows + [[0.0] * width]
    best, norm, acc = None, 0.0, [0.0] * width
    for start in range(0, len(scores), tile):
        chunk = range(start, min(len(scores), start + tile))
        new = max(scores[j] for j in chunk) if best is None else max(best, max(scores[j] for j in chunk))
        if best is not None and new != best:
            alpha = f32(math.exp(f32(best - new)))
            norm, acc = f32(norm * alpha), [f32(a * alpha) for a in acc]
        best = new
        for j in chunk:
            w = f32(math.exp(f32(scores[j] - best)))
            norm = f32(norm + w)
            acc = [f32(a + f32(w * v)) for a, v in zip(acc, rows[j])]
    return [bf16(f32(a / norm)) for a in acc], f32(best + f32(math.log(norm)))


def engine_attention_eager(q: list, keys: list, values: list, scale: float,
                           sink_logit: Optional[float] = None) -> tuple:
    """``GptOss`` eager kernel: bf16 logits, ``softmax(logits - max)`` with fp32 accumulation and bf16
    probabilities, bf16 ``P V``.  Returns (out, shifted bf16 logits incl. sink, fp32 LSE of the shifted logits)."""
    scale_b = bf16(scale)
    logits = [bf16(bf16(dot32(q, k)) * scale_b) for k in keys]
    if sink_logit is not None:
        logits = logits + [bf16(sink_logit)]
    top = max(logits)
    shifted = [bf16(c - top) for c in logits]
    weights = [f32(math.exp(c)) for c in shifted]
    total = 0.0
    for w in weights:
        total = f32(total + w)
    probs = [bf16(f32(w / total)) for w in weights][:len(keys)]
    out = [bf16(dot32(probs, [row[c] for row in values])) for c in range(len(values[0]))]
    return out, shifted, f32(math.log(total))


# ====================================================================================== RMSNorm
@dataclass(frozen=True)
class RmsTemplate:
    """Physical template of one RMSNorm implementation.

    ``stores``     bf16 roundings between the fp32 normalisation and the result.  ``GptOssRMSNorm`` and
                   ``Qwen3_5RMSNorm`` multiply by the gain in fp32 and store once; ``LlamaRMSNorm`` /
                   ``Qwen3RMSNorm`` cast the normalised row to bf16 and then multiply by the bf16 gain: two stores.
    ``one_plus``   the gain is ``fl32(1 + w)`` (zero-centred weight) instead of ``w``.
    """
    name: str
    stores: int
    one_plus: bool = False


RMS_TEMPLATES = {
    "LlamaRMSNorm": RmsTemplate("norm_bf16_then_gain", 2), "Qwen3RMSNorm": RmsTemplate("norm_bf16_then_gain", 2),
    "Qwen3MoeRMSNorm": RmsTemplate("norm_bf16_then_gain", 2), "GptOssRMSNorm": RmsTemplate("gain_fp32_one_store", 1),
    "Qwen3_5RMSNorm": RmsTemplate("one_plus_gain_fp32_one_store", 1, one_plus=True),
}
RMS_NOT_COVERED = ("Qwen3_5RMSNormGated",)
_BF16_CONTRACT = get_contract("bf16_fp32acc")


def rms_relative(stores: int) -> float:
    """``(1 + u_elem)^3 (1 + u_store)^stores - 1``: rsqrt, two fp32 multiplications, the bf16 stores."""
    elem = 2.0 ** _BF16_CONTRACT.elem_u_log2
    return (1.0 + elem) ** 3 * (1.0 + float(_BF16_CONTRACT.u_store)) ** stores - 1.0


def rms_endpoints(squares: Fraction, width: int, epsilon: Fraction) -> tuple:
    """Certified enclosure ``[lo, hi]`` of ``1 / sqrt(mean + eps)`` as the fp32 kernel can compute it.

    The mean of squares carries the contract's accumulation bound for a length-``width`` reduction and one
    additive floor; ``norm_rsqrt`` supplies the rational endpoints (the formula of ``blocks.rmsnorm_block``).
    """
    from dataclasses import replace
    acc_contract = replace(_BF16_CONTRACT, store_u_log2=_BF16_CONTRACT.accum_u_log2)
    accumulation = acc_contract.dot_relative_bound(width)
    eta = Fraction(2) ** _BF16_CONTRACT.eta_log2
    mean = squares / width
    low_rad = max(epsilon, mean * (1 - accumulation) + epsilon - eta)
    high_rad = mean * (1 + accumulation) + epsilon + eta
    return norm_rsqrt(high_rad).enclosure[0], norm_rsqrt(low_rad).enclosure[1]


def rms_check_row(x: list, gain: list, observed: list, *, epsilon: float, template: RmsTemplate,
                  residual: bool = True) -> dict:
    """One row against the enclosure.  ``gain`` is the effective gain (``fl32(1 + w)`` already applied).

    ``floor`` reports the ``sum x^2 >= width * (1e-3)^2`` premise; for a residual-stream norm a row below the
    floor is outside the certified domain and counts as bad.
    """
    width = len(x)
    squares = sum(Fraction(v) ** 2 for v in x)
    floor_ok = PR.residual_floor_holds(squares, width)
    lo, hi = (float(v) for v in rms_endpoints(squares, width, Fraction(f32(epsilon))))
    lo, hi = lo / OUTWARD, hi * OUTWARD
    rel, eta = rms_relative(template.stores), template.stores * ETAB
    worst = 0.0
    for a, g, y in zip(x, gain, observed, strict=True):
        low, high = sorted((a * g * lo, a * g * hi))
        radius = (max(abs(low), abs(high)) * rel + eta) * OUTWARD
        excess = max(low - y, y - high, 0.0)
        worst = max(worst, excess / radius)
    rms = math.sqrt(float(squares) / width)
    return {"ratio": worst, "bad": worst > 1.0 or (residual and not floor_ok), "floor_ok": floor_ok, "rms": rms,
            "outlier": max(abs(v) for v in x) / rms if rms else math.inf}


def engine_rmsnorm(x: list, weight: list, *, epsilon: float, template: RmsTemplate) -> tuple:
    """fp32 RMSNorm of a bf16 row as the three Hugging Face implementations compute it.  Returns (out, gain)."""
    inv = f32(1.0 / math.sqrt(f32(f32(dot32(x, x) / len(x)) + f32(epsilon))))
    gain = [f32(1.0 + w) for w in weight] if template.one_plus else list(weight)
    if template.stores == 2:
        return [bf16(bf16(f32(v * inv)) * g) for v, g in zip(x, gain)], gain
    return [bf16(f32(f32(v * inv) * g)) for v, g in zip(x, gain)], gain


# ====================================================================================== LRec (Gated DeltaNet)
def _norm2(v: list) -> float:
    return math.sqrt(math.fsum(a * a for a in v))


L2_EPS = f32(1e-6)                  # ``l2norm(..., eps=1e-6)`` added to an fp32 tensor


def l2_unit(v: list, eps: float = L2_EPS) -> list:
    inv = 1.0 / math.sqrt(math.fsum(a * a for a in v) + eps)
    return [a * inv for a in v]


def lrec_constants(key_dim: int, *, any_order: bool = False) -> dict:
    """Relative perturbations of the kernel's own preprocessing (fp32 ``l2norm``, ``/ sqrt(d)``, ``exp``).
    ``any_order`` replaces the contract's lane depth by the order-free ``gamma_n`` (the chunked template)."""
    g = gamma_any(key_dim) if any_order else gamma(key_dim)
    c_key = (1.0 + g) * (1.0 + UE) * (1.0 + U32) - 1.0
    # The scalar part of the l2norm error alone: k^ = (1 + delta) k (1 + eps) with |eps| <= u and
    # |delta| <= c_scale, the rsqrt of a sum of squares (plus eps) known to gamma_{n+1}; the query also carries
    # the rounded constant n^-1/2.
    gs = gamma_any(key_dim + 1) if any_order else g
    c_scale = (1.0 + UE) / math.sqrt(1.0 - gs) - 1.0
    return {"gamma": g, "c_key": c_key, "c_query": (1.0 + c_key) * (1.0 + U32) ** 2 - 1.0,
            "eps_decay": (1.0 + UE) * (1.0 + U32) - 1.0, "u2": (1.0 + U32) ** 2 - 1.0,
            "c_scale": c_scale, "c_qscale": (1.0 + c_scale) * (1.0 + U32) - 1.0}


def lrec_reference(queries: list, keys: list, values: list, decays: list, betas: list,
                   state: Optional[list] = None) -> dict:
    """fp64 sequential Gated DeltaNet recurrence of one head with its ``LREC_SEQ_ENCLOSURE_V0`` radii.

    State ``S`` has shape ``[key_dim][value_dim]``.  Per token, with unit key ``k`` and ``a = exp(g)``:
    ``S <- a S``, ``d = beta (v - S^T k)``, ``S <- S + k d^T``, ``o = S^T q / sqrt(key_dim)``.

    Enclosure.  ``rho_j`` bounds the 2-norm over the key axis of the error of column ``j``.  Each column evolves by
    ``(I - beta k k^T) a``, an operator of 2-norm at most ``a`` for ``beta ||k||^2 <= 2``, so the error is carried
    with factor ``a`` (specs/06 row 9) and every step adds its local terms: the decay rounding
    ``a eps_a (sigma + rho)``, the key perturbation ``beta ((2 c_k + c_k^2) sigma' + |v_j| c_k)``, the delta
    rounding ``(1 + c_k)(D u2 + beta eps_m (1 + u2))`` with ``eps_m = gamma (sigma' + rho')(1 + c_k)`` and
    ``D = |d_j| + beta (c_k sigma' + (1 + c_k) rho')``, and the two update roundings.  The output radius is
    ``||q|| ((1 + c_q)(1 + gamma) rho + (c_q + gamma (1 + c_q)) sigma)`` followed by the bf16 store.  This is a
    bound for the sequential fp32 template (``torch_recurrent_gated_delta_rule``) only.  The chunked WY kernel
    rounds differently and has its own enclosure, :func:`lrec_chunk_reference`; the real mode still reports the
    chunked calls against these radii, as the diagnostic block ``sequential_v0``.
    """
    key_dim, value_dim = len(keys[0]), len(values[0])
    c = lrec_constants(key_dim)
    g, ck, cq, ea, u2 = c["gamma"], c["c_key"], c["c_query"], c["eps_decay"], c["u2"]
    floor = 4.0 * ETA32 * math.sqrt(key_dim)
    s = [list(row) for row in state] if state is not None else [[0.0] * value_dim for _ in range(key_dim)]
    rho = [0.0] * value_dim
    outputs, radii = [], []
    for q_raw, k_raw, v, gate, beta in zip(queries, keys, values, decays, betas, strict=True):
        k = l2_unit(k_raw)
        q = [a / math.sqrt(key_dim) for a in l2_unit(q_raw)]
        if not 0.0 <= beta <= 1.0 or gate > 0.0:
            raise ValueError("LRec needs beta in [0, 1] and a non-positive log-decay")
        a = math.exp(gate)
        sigma = [_norm2([s[i][j] for i in range(key_dim)]) for j in range(value_dim)]
        rho1 = [a * (1.0 + ea) * r + a * ea * sg for r, sg in zip(rho, sigma)]
        s = [[a * cell for cell in row] for row in s]
        sigma1 = [a * sg for sg in sigma]
        mem = [math.fsum(s[i][j] * k[i] for i in range(key_dim)) for j in range(value_dim)]
        delta = [beta * (vj - mj) for vj, mj in zip(v, mem)]
        s = [[s[i][j] + k[i] * delta[j] for j in range(value_dim)] for i in range(key_dim)]
        sigma2 = [_norm2([s[i][j] for i in range(key_dim)]) for j in range(value_dim)]
        rho = []
        for j in range(value_dim):
            eps_m = g * (sigma1[j] + rho1[j]) * (1.0 + ck)
            big_d = abs(delta[j]) + beta * (ck * sigma1[j] + (1.0 + ck) * rho1[j])
            step = (1.0 + ck) * (big_d * u2 + beta * eps_m * (1.0 + u2))
            pre = (rho1[j] + beta * ((2.0 * ck + ck * ck) * sigma1[j] + abs(v[j]) * ck) + step
                   + U32 * (1.0 + U32) * (1.0 + ck) * (big_d * (1.0 + u2) + beta * eps_m * (1.0 + u2)) + floor)
            rho.append((pre + U32 * (sigma2[j] + pre)) * OUTWARD)
        out = [math.fsum(s[i][j] * q[i] for i in range(key_dim)) for j in range(value_dim)]
        qn = _norm2(q)
        rad = []
        for j in range(value_dim):
            pre = qn * ((1.0 + cq) * (1.0 + g) * rho[j] + (cq + g * (1.0 + cq)) * sigma2[j]) + ETA32
            rad.append((pre + UB * (abs(out[j]) + pre) + ETAB) * OUTWARD)
        outputs.append(out)
        radii.append(rad)
    return {"outputs": outputs, "radii": radii, "state": s, "state_radius": rho}


def engine_lrec(queries: list, keys: list, values: list, decays: list, betas: list, *, decay_cheat: bool = False,
                reset_at: Optional[int] = None) -> list:
    """Sequential fp32 kernel (``torch_recurrent_gated_delta_rule``), bf16 output store.

    ``decay_cheat`` skips the decay multiplication; ``reset_at`` zeroes the carried state before that token
    (a truncated state hand-off at a chunk boundary).
    """
    key_dim, value_dim = len(keys[0]), len(values[0])
    root = f32(math.sqrt(key_dim))
    s = [[0.0] * value_dim for _ in range(key_dim)]
    outputs = []
    for t, (q_raw, k_raw, v, gate, beta) in enumerate(zip(queries, keys, values, decays, betas, strict=True)):
        if reset_at is not None and t == reset_at:
            s = [[0.0] * value_dim for _ in range(key_dim)]
        inv_k = f32(1.0 / math.sqrt(f32(dot32(k_raw, k_raw) + L2_EPS)))
        inv_q = f32(1.0 / math.sqrt(f32(dot32(q_raw, q_raw) + L2_EPS)))
        k = [f32(a * inv_k) for a in k_raw]
        q = [f32(f32(a * inv_q) / root) for a in q_raw]
        if not decay_cheat:
            a = f32(math.exp(f32(gate)))
            s = [[f32(cell * a) for cell in row] for row in s]
        mem = [dot32([s[i][j] for i in range(key_dim)], k) for j in range(value_dim)]
        delta = [f32(f32(vj - mj) * f32(beta)) for vj, mj in zip(v, mem)]
        s = [[f32(s[i][j] + f32(k[i] * delta[j])) for j in range(value_dim)] for i in range(key_dim)]
        outputs.append([bf16(dot32([s[i][j] for i in range(key_dim)], q)) for j in range(value_dim)])
    return outputs


def lrec_check(ref: dict, observed: list) -> dict:
    worst, bad = 0.0, 0
    for out, rad, obs in zip(ref["outputs"], ref["radii"], observed, strict=True):
        ratio = max(abs(a - b) / r for a, b, r in zip(obs, out, rad, strict=True))
        worst, bad = max(worst, ratio), bad + (ratio > 1.0)
    return {"tokens": len(observed), "bad_tokens": bad, "max_ratio": worst}


# ====================================================================================== LRec: chunked WY template
LREC_SEQ, LREC_CHUNK_WY = "LREC_SEQ_ENCLOSURE_V0", "LREC_CHUNK_WY_ENCLOSURE_V1"
LREC_CHUNK = 64                     # ``chunk_size`` default of ``torch_chunk_gated_delta_rule``
LREC_KERNEL_ENCLOSURE = {"torch_chunk_gated_delta_rule": LREC_CHUNK_WY,
                         "torch_recurrent_gated_delta_rule": LREC_SEQ}
REF_SLACK = 1.0 + 2.0 ** -28        # the fp64 reference runs the same chunked algorithm: its error is the same
#                                     bound with u = 2^-53 instead of 2^-24, i.e. below 2^-29 of the radius


def gamma_any(n: int) -> float:
    """``gamma_n = n u / (1 - n u)``: an ``n``-term fp32 sum of products in ANY order (Higham, Lemma 3.1 / §3.1).
    The chunked template does not use the contract's lane depth ``h(n)``: gemm, trsm and cumsum orderings differ
    between CPU and GPU backends, and this constant holds for all of them."""
    return n * U32 / (1.0 - n * U32)


def _mm(a: list, b: list) -> list:
    cols = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in cols] for row in a]


def _mul_err(ax: float, ex: float, ay: float, ey: float) -> float:
    """``|fl(x^ y^) - x y|`` for ``|x^ - x| <= ex``, ``|y^ - y| <= ey``."""
    return (ax * ey + ay * ex + ex * ey) * (1.0 + U32) + U32 * ax * ay + ETA32


def _mm_err(ax: list, ex: list, ay: list, ey: list, gam: float) -> list:
    """Entrywise ``|fl(X^ Y^) - X Y| <= |X| E_Y + E_X (|Y| + E_Y) + gamma (|X| + E_X)(|Y| + E_Y)``."""
    p = len(ay)
    sx = [[a + e for a, e in zip(r, s)] for r, s in zip(ax, ex)]
    sy = [[a + e for a, e in zip(r, s)] for r, s in zip(ay, ey)]
    t1, t2, t3 = _mm(ax, ey), _mm(ex, sy), _mm(sx, sy)
    return [[a + b + gam * c + p * ETA32 for a, b, c in zip(r1, r2, r3)] for r1, r2, r3 in zip(t1, t2, t3)]


def _unit_lower_solve(a: list, b: list) -> list:
    """``X`` with ``A X = B`` for unit lower triangular ``A`` (the strict lower part of ``a`` is read)."""
    x = []
    for i, row in enumerate(b):
        cur = list(row)
        for j in range(i):
            f = a[i][j]
            if f:
                cur = [c - f * y for c, y in zip(cur, x[j])]
        x.append(cur)
    return x


def lrec_chunk_reference(queries: list, keys: list, values: list, decays: list, betas: list,
                         state: Optional[list] = None, *, chunk: int = LREC_CHUNK) -> dict:
    """fp64 chunked Gated DeltaNet of one head with its ``LREC_CHUNK_WY_ENCLOSURE_V1`` radii.

    The kernel (``torch_chunk_gated_delta_rule``).  Per chunk of ``C`` tokens, with unit keys ``k_i``, cumulative
    log-decay ``G_i = g_0 + .. + g_i``, ``Gamma_ij = exp(G_i - G_j)`` (``i >= j``), ``E_i = exp(G_i)`` and
    ``F_i = exp(G_{C-1} - G_i)``::

        N = strict_lower((beta K K^T) o Gamma)      A = I + N            (the WY / UT system, C x C)
        U = A^-1 (beta V)        W = A^-1 (beta K o E)                   (two triangular solves)
        D = U - W S              out = (Q o E) S + lower((Q K^T) o Gamma) D
        S' = E_{C-1} S + (K o F)^T D

    In exact arithmetic ``D`` holds the deltas of the sequential rule, so ``out`` and ``S'`` equal those of
    :func:`lrec_reference`; only the rounding differs, and ``LREC_SEQ_ENCLOSURE_V0`` does not bound it.

    Enclosure.  Three differences are bounded separately and added: ``S^' - S' = [fl(S^) - T'(S^)] + [T'(S^) -
    T(S)]``, where ``T`` is the exact chunk map on unit keys, ``T'`` the exact map on the keys the kernel
    normalised, and ``S^`` the computed state entering the chunk (``rho_j`` bounds the 2-norm over the key axis of
    column ``j`` of ``S^ - S``).

    (a) Key scale, through the exact contraction.  The fp32 ``l2norm`` returns ``k^_i = (1 + delta_i) k_i (1 +
    eps)`` with one scalar ``|delta_i| <= c_s`` per token (``c_scale`` of :func:`lrec_constants`) and ``|eps| <=
    u``.  Running the delta rule on ``k' = (1 + delta) k`` changes one step by ``beta k (delta v - (2 delta +
    delta^2) mem)``, and every exact step has 2-norm at most ``a_i`` (``beta ||k||^2 <= 2``), so with
    ``theta_{-1} = rho`` and ``c_2 = 2 c_s + c_s^2``::

        theta_i = a_i theta_{i-1} (1 + beta_i c_2) + beta_i c_s |v_i| + c_2 |beta_i v_i - D_i|

    bounds the column 2-norm of ``X'_i(S^) - X_i(S)`` after token ``i``; ``theta_{C-1}`` carries ``rho`` across the
    chunk with the factor ``E_{C-1}`` of the exact map (pushing ``rho`` or ``delta`` through the entrywise bounds
    of (b) instead would multiply them by about ``sum beta`` per chunk).  The query scale adds ``c_qs |out|``.

    (b) Rounding of the chunk, evaluated at ``S^`` on the data ``k'``, ``q'``.  Every fp32 intermediate ``X^``
    carries an entrywise bound ``E_X >= |X^ - X'|``, propagated by the standard model: a product adds ``u |x y|``,
    an ``n``-term inner product ``gamma_n |x|^T |y|`` in any order (:func:`gamma_any`), the cumulative sum
    ``gamma_{C-1} |G_i|`` in the log domain (hence a RELATIVE error ``exp(h_ij) (1 + eps_exp) - 1`` on
    ``Gamma_ij`` with ``h_ij = (E_G,i + E_G,j)(1 + u) + u |G_i - G_j|``, which does not shrink when ``G_i - G_j``
    is small).  The terms V0 has no counterpart for:

    1. Triangular solves.  Substitution in any order returns ``X^`` with ``(I + N^ + dN) X^ = B^`` and
       ``|dN| <= gamma_C |N^|`` (Higham, Thm 8.5; the unit diagonal is not read).  Subtracting ``A' X' = B'``:
       ``X^ - X' = A'^-1 (B^ - B' - (N^ - N' + dN) X^)``, so with ``M >= |A'^-1|`` and
       ``Dn = E_N + gamma_C (|N'| + E_N)``::

           E_X <= M (E_B + Dn |X'|) + M Dn E_X     =>     E_X = (I - M Dn)^-1 M (E_B + Dn |X'|)

       ``M Dn`` is non-negative and strictly lower triangular, hence nilpotent, so the inverse is a finite
       non-negative sum and no smallness condition is needed.  ``M`` comes from the inverse of THIS chunk's
       matrix, computed here (``|A^-1|``, not the comparison-matrix bound ``(I - |N|)^-1``, which is exponential
       in ``C`` on correlated keys).  This is the conditioning term: it grows with ``|A^-1| |A|`` of the chunk
       and with ``|U|``, ``|W|``, which exceed the deltas when ``U - W S`` cancels.
    2. ``D^ = fl(U^ - fl(W^ S^))``: ``E_D = (E_U + E_P)(1 + u) + u |D'|`` with
       ``E_P = (E_W + gamma_n (|W'| + E_W)) |S^|`` and ``X |S^| <= X |S| + rownorm_2(X) rho^T``.
    3. The ``C``-term products ``K~^T D^`` and ``lower(..) D^`` (``gamma_C``; ``gamma_64`` for the real kernel).
    4. ``fl(S^ exp(G^_{C-1}))`` with the cumulative-sum error of a whole chunk in the exponent:
       ``n1 = (r_cd (1 + u) + u) E_{C-1} (sigma + rho)``.

    The magnitudes of the primed quantities are majorised from the unit-key ones, again without a smallness
    condition: ``|k'| <= (1 + c_s) |k|``, ``|N'| <= (1 + c_s)^2 |N|``, and with ``R = (I - c_2 |A^-1| |N|)^-1``
    (nilpotent again) ``|A'^-1| <= R |A^-1|``, ``|U'| <= R |U|``, ``|W'| <= R (|W| + c_s |A^-1| |beta K o E|)``,
    ``|D'_i(S^)| <= |D_i| + (1 + c_s) beta_i a_i theta_{i-1} + c_s |beta_i v_i - D_i|``.

    Result: ``rho' = (theta_{C-1} + n1 + n2)(1 + u) + u sigma'`` with ``n2_j`` the 2-norm of column ``j`` of
    ``E_{K~^T D}``; output radius ``(1 + c_qs) ||q_i|| theta_i + c_qs |out_i|`` plus the local terms of ``out``,
    then the bf16 store.  Padding tokens (``beta = 0``, ``g = 0``, zero rows) belong to the last chunk as in the
    kernel.

    Assumptions (stated, not measured): IEEE fp32 with the any-order summation model for gemm / trsm / cumsum
    (no Strassen-like gemm, no explicit inversion of diagonal blocks inside trsm), ``exp`` and ``rsqrt`` within
    ``U_ELEM``, ``exp(0) = 1`` exactly, and the ``solve_triangular`` branch of the kernel (the
    ``is_torchdynamo_exporting`` branch builds ``A^-1`` explicitly and is a different template).
    """
    key_dim, value_dim, tokens = len(keys[0]), len(values[0]), len(keys)
    for gate, beta in zip(decays, betas, strict=True):
        if not 0.0 <= beta <= 1.0 or gate > 0.0:
            raise ValueError("LRec needs beta in [0, 1] and a non-positive log-decay")
    c = lrec_constants(key_dim, any_order=True)
    cs, cqs, ea, u2 = c["c_scale"], c["c_qscale"], c["eps_decay"], c["u2"]
    c2, up, uq = 2.0 * cs + cs * cs, 1.0 + cs, 1.0 + cqs
    u, gn, gc, gs = U32, gamma_any(key_dim), gamma_any(chunk), gamma_any(chunk - 1)
    floor = ETA32 * math.sqrt(key_dim)
    pad = (-tokens) % chunk
    root = math.sqrt(key_dim)
    k_all = [l2_unit(k) for k in keys] + [[0.0] * key_dim] * pad
    q_all = [[a / root for a in l2_unit(q)] for q in queries] + [[0.0] * key_dim] * pad
    v_all = [list(v) for v in values] + [[0.0] * value_dim] * pad
    b_all, g_all = list(betas) + [0.0] * pad, list(decays) + [0.0] * pad
    s = [list(row) for row in state] if state is not None else [[0.0] * value_dim for _ in range(key_dim)]
    rho = [0.0] * value_dim
    outputs, radii, inv_norm = [], [], 0.0
    span = range(chunk)
    for c0 in range(0, tokens + pad, chunk):
        k, q, v, b = (x[c0:c0 + chunk] for x in (k_all, q_all, v_all, b_all))
        cum, acc = [], 0.0
        for gate in g_all[c0:c0 + chunk]:
            acc += gate
            cum.append(acc)
        e_g = [gs * abs(x) for x in cum]

        def pair(i: int, j: int) -> tuple:
            """``Gamma_ij`` and its error bound (exact 1 on the diagonal)."""
            if i == j:
                return 1.0, 0.0
            h = (e_g[i] + e_g[j]) * (1.0 + u) + u * abs(cum[i] - cum[j])
            val = math.exp(cum[i] - cum[j])
            return val, val * (math.expm1(h) * (1.0 + ea) + ea) + ETA32

        gam = [[pair(i, j) if j <= i else (0.0, 0.0) for j in span] for i in span]
        e_cum = [math.exp(x) for x in cum]
        e_e = [x * (math.expm1(h) * (1.0 + ea) + ea) + ETA32 for x, h in zip(e_cum, e_g)]
        f_cum, e_f = [gam[-1][i][0] for i in span], [gam[-1][i][1] for i in span]
        ak = [[up * abs(a) for a in row] for row in k]                      # majorants of the primed data
        ek = [[u * a for a in row] for row in ak]
        aq = [[uq * abs(a) for a in row] for row in q]
        eq = [[u2 * a for a in row] for row in aq]
        akb = [[a * bi for a in row] for row, bi in zip(ak, b)]
        ekb = [[_mul_err(a, e, bi, 0.0) for a, e in zip(ra, re)] for ra, re, bi in zip(ak, ek, b)]
        vb = [[a * bi for a in row] for row, bi in zip(v, b)]
        evb = [[u * abs(a) + ETA32 for a in row] for row in vb]
        kt, akt, ekt = list(zip(*k)), list(zip(*ak)), list(zip(*ek))        # K^T and its bounds, [key][token]
        p1 = _mm([[a * bi for a in row] for row, bi in zip(k, b)], kt)
        ep1 = _mm_err(akb, ekb, akt, ekt, gn)
        p2, ep2 = _mm(q, kt), _mm_err(aq, eq, akt, ekt, gn)
        nmat = [[p1[i][j] * gam[i][j][0] if j < i else 0.0 for j in span] for i in span]
        an = [[up * up * abs(a) for a in row] for row in nmat]
        e_n = [[_mul_err(up * up * abs(p1[i][j]), ep1[i][j], *gam[i][j]) if j < i else 0.0 for j in span]
               for i in span]
        lmat = [[p2[i][j] * gam[i][j][0] if j <= i else 0.0 for j in span] for i in span]
        al = [[up * uq * abs(a) for a in row] for row in lmat]
        e_l = [[_mul_err(up * uq * abs(p2[i][j]), ep2[i][j], *gam[i][j]) if j <= i else 0.0 for j in span]
               for i in span]
        dkb = [[a * bi * ei for a in row] for row, bi, ei in zip(k, b, e_cum)]
        edkb = [[_mul_err(a, e, ei, eei) for a, e in zip(ra, re)] for ra, re, ei, eei in zip(akb, ekb, e_cum, e_e)]
        eye = [[1.0 if i == j else 0.0 for j in span] for i in span]
        inv = _unit_lower_solve(nmat, eye)
        m_abs = [[abs(a) for a in row] for row in inv]
        inv_norm = max(inv_norm, max(sum(row) for row in m_abs))
        u_ref, w_ref = _mm(inv, vb), _mm(inv, dkb)
        # R = (I - c_2 |A^-1| |N|)^-1 majorises the primed inverse and solutions.
        scale_neg = [[-c2 * a for a in row] for row in _mm(m_abs, [[abs(a) for a in row] for row in nmat])]
        m_abs = _unit_lower_solve(scale_neg, m_abs)
        au = _unit_lower_solve(scale_neg, [[abs(a) for a in row] for row in u_ref])
        aw = _unit_lower_solve(scale_neg, [
            [abs(a) + cs * x for a, x in zip(ra, rx)]
            for ra, rx in zip(w_ref, _mm([[abs(a) for a in row] for row in inv],
                                         [[abs(a) for a in row] for row in dkb]))])
        d_n = [[e + gc * (a + e) for a, e in zip(ra, re)] for ra, re in zip(an, e_n)]
        iter_neg = [[-a for a in row] for row in _mm(m_abs, d_n)]            # I - M Dn, strict lower part

        def solve_err(x_abs: list, e_b: list) -> list:
            rhs = [[a + e for a, e in zip(ra, re)] for ra, re in zip(_mm(d_n, x_abs), e_b)]
            return _unit_lower_solve(iter_neg, _mm(m_abs, rhs))

        e_u, e_w = solve_err(au, evb), solve_err(aw, edkb)
        abs_s = [[abs(a) for a in row] for row in s]

        def times_abs_state(x: list) -> list:
            """``X |S^| <= X |S| + rownorm_2(X) rho^T`` for a non-negative ``X``."""
            return [[a + _norm2(row) * r for a, r in zip(prod, rho)] for row, prod in zip(x, _mm(x, abs_s))]

        d_ref = [[a - p for a, p in zip(ra, rp)] for ra, rp in zip(u_ref, _mm(w_ref, s))]
        theta, thetas, d_abs = list(rho), [], []
        for i in span:                                                       # (a): the key scale and rho
            a_i = math.exp(g_all[c0 + i])
            mem = [abs(b[i] * x - d) for x, d in zip(v[i], d_ref[i])]        # beta |k . a S_{i-1}|
            d_abs.append([abs(d) + up * b[i] * a_i * t + cs * mj for d, t, mj in zip(d_ref[i], theta, mem)])
            theta = [a_i * t * (1.0 + b[i] * c2) + b[i] * cs * abs(x) + c2 * mj
                     for t, x, mj in zip(theta, v[i], mem)]
            thetas.append(theta)
        e_p3 = times_abs_state([[e + gn * (a + e) for a, e in zip(ra, re)] for ra, re in zip(aw, e_w)])
        e_d = [[(a + p + key_dim * ETA32) * (1.0 + u) + u * d + ETA32 for a, p, d in zip(ra, rp, rd)]
               for ra, rp, rd in zip(e_u, e_p3, d_abs)]
        qt = [[a * ei for a in row] for row, ei in zip(q, e_cum)]
        aqt = [[uq * abs(a) for a in row] for row in qt]
        eqt = [[_mul_err(a, e, ei, eei) for a, e in zip(ra, re)] for ra, re, ei, eei in zip(aq, eq, e_cum, e_e)]
        out = [[a + t for a, t in zip(ra, rt)] for ra, rt in zip(_mm(qt, s), _mm(lmat, d_ref))]
        e_r = times_abs_state([[e + gn * (a + e) for a, e in zip(ra, re)] for ra, re in zip(aqt, eqt)])
        e_t = _mm_err(al, e_l, d_abs, e_d, gc)
        for i in span:
            if c0 + i >= tokens:
                break
            qn, rad = uq * _norm2(q[i]), []
            for j in range(value_dim):
                carry = qn * thetas[i][j] + cqs * abs(out[i][j])
                pre = (carry + (e_r[i][j] + e_t[i][j] + key_dim * ETA32) * (1.0 + u)
                       + u * (abs(out[i][j]) + carry) + ETA32)
                rad.append((pre + UB * (abs(out[i][j]) + pre) + ETAB) * REF_SLACK)
            outputs.append(out[i])
            radii.append(rad)
        a_c, r_cd = e_cum[-1], e_e[-1] / e_cum[-1]
        ktd = [[a * fi for a, fi in zip(col, f_cum)] for col in kt]          # (K o F)^T, [key][token]
        ektd = [[_mul_err(a, e, fi, efi) for a, e, fi, efi in zip(ca, ce, f_cum, e_f)] for ca, ce in zip(akt, ekt)]
        e_x2 = _mm_err([[up * abs(a) for a in row] for row in ktd], ektd, d_abs, e_d, gc)
        sigma = [_norm2([s[i][j] for i in range(key_dim)]) for j in range(value_dim)]
        s = [[a_c * a + x for a, x in zip(ra, rx)] for ra, rx in zip(s, _mm(ktd, d_ref))]
        new_rho = []
        for j in range(value_dim):
            n1 = (r_cd * (1.0 + u) + u) * a_c * (sigma[j] + rho[j]) + floor
            n2 = _norm2([e_x2[i][j] for i in range(key_dim)])
            pre = theta[j] + n1 + n2
            new_rho.append((pre + u * (_norm2([s[i][j] for i in range(key_dim)]) + pre) + floor) * REF_SLACK)
        rho = new_rho
    return {"outputs": outputs, "radii": radii, "state": s, "state_radius": rho, "inv_inf_norm": inv_norm,
            "chunks": (tokens + pad) // chunk}


def engine_lrec_chunk(queries: list, keys: list, values: list, decays: list, betas: list, *,
                      chunk: int = LREC_CHUNK, decay_cheat: bool = False, reset_at: Optional[int] = None) -> tuple:
    """fp32 emulation of ``torch_chunk_gated_delta_rule`` (forward substitution, :func:`dot32` products, bf16
    output store).  Returns ``(outputs, final_state)``.  ``decay_cheat`` runs the scan with ``g = 0``;
    ``reset_at`` (a chunk boundary) zeroes the state handed to that chunk."""
    key_dim, value_dim, tokens = len(keys[0]), len(values[0]), len(keys)
    if reset_at is not None and reset_at % chunk:
        raise ValueError("the chunked kernel hands its state over at chunk boundaries only")
    scaling, pad = f32(key_dim ** -0.5), (-tokens) % chunk
    k_all, q_all = [], []
    for q_raw, k_raw in zip(queries, keys, strict=True):
        inv_k = f32(1.0 / math.sqrt(f32(dot32(k_raw, k_raw) + L2_EPS)))
        inv_q = f32(1.0 / math.sqrt(f32(dot32(q_raw, q_raw) + L2_EPS)))
        k_all.append([f32(a * inv_k) for a in k_raw])
        q_all.append([f32(f32(a * inv_q) * scaling) for a in q_raw])
    k_all += [[0.0] * key_dim] * pad
    q_all += [[0.0] * key_dim] * pad
    v_all = [list(v) for v in values] + [[0.0] * value_dim] * pad
    b_all = [f32(x) for x in betas] + [0.0] * pad
    g_all = [0.0 if decay_cheat else f32(x) for x in decays] + [0.0] * pad
    s = [[0.0] * value_dim for _ in range(key_dim)]
    outputs, span = [], range(chunk)
    for c0 in range(0, tokens + pad, chunk):
        if reset_at is not None and c0 == reset_at:
            s = [[0.0] * value_dim for _ in range(key_dim)]
        k, q, v, b = (x[c0:c0 + chunk] for x in (k_all, q_all, v_all, b_all))
        cum, acc = [], 0.0
        for gate in g_all[c0:c0 + chunk]:
            acc = f32(acc + gate)
            cum.append(acc)
        gam = [[f32(math.exp(f32(cum[i] - cum[j]))) if j <= i else 0.0 for j in span] for i in span]
        kb = [[f32(a * bi) for a in row] for row, bi in zip(k, b)]
        vb = [[f32(a * bi) for a in row] for row, bi in zip(v, b)]
        ut = [[f32(dot32(kb[i], k[j]) * gam[i][j]) if j < i else 0.0 for j in span] for i in span]
        attn = [[f32(dot32(q[i], k[j]) * gam[i][j]) if j <= i else 0.0 for j in span] for i in span]
        e_cum = [f32(math.exp(x)) for x in cum]
        dkb = [[f32(a * ei) for a in row] for row, ei in zip(kb, e_cum)]

        def solve(rhs: list) -> list:
            x = []
            for i, row in enumerate(rhs):
                cur = list(row)
                for j in range(i):
                    cur = [f32(a - f32(ut[i][j] * y)) for a, y in zip(cur, x[j])]
                x.append(cur)
            return x

        new_values, k_cum = solve(vb), solve(dkb)
        qd = [[f32(a * ei) for a in row] for row, ei in zip(q, e_cum)]
        kd = [[f32(a * gam[-1][i]) for a in row] for i, row in enumerate(k)]
        s_cols = [[s[i][j] for i in range(key_dim)] for j in range(value_dim)]
        v_new = [[f32(new_values[i][j] - dot32(k_cum[i], s_cols[j])) for j in range(value_dim)] for i in span]
        v_cols = list(zip(*v_new))
        for i in span:
            if c0 + i < tokens:
                outputs.append([bf16(f32(dot32(qd[i], s_cols[j]) + dot32(attn[i], v_cols[j])))
                                for j in range(value_dim)])
        kd_cols = list(zip(*kd))
        s = [[f32(f32(s[i][j] * e_cum[-1]) + dot32(kd_cols[i], v_cols[j])) for j in range(value_dim)]
             for i in range(key_dim)]
    return outputs, s


def lrec_state_check(ref: dict, state: list) -> dict:
    """Column 2-norm distance of an observed final state over the reference's ``state_radius``."""
    worst, bad = 0.0, 0
    for j, radius in enumerate(ref["state_radius"]):
        err = _norm2([row[j] - want[j] for row, want in zip(state, ref["state"], strict=True)])
        ratio = err / radius if radius > 0.0 else (0.0 if err == 0.0 else math.inf)
        worst, bad = max(worst, ratio), bad + (ratio > 1.0)
    return {"columns": len(ref["state_radius"]), "bad_columns": bad, "max_ratio": worst}


# ====================================================================================== dry run
DRY_HEAD_DIM = 16
DRY_TOKENS = 96                      # two key tiles for the tiled kernel
DRY_POSITIONS = (0, 63, 95)          # one-key row, tile boundary, last row
DRY_RMS_WIDTH = 48
DRY_RMS_ROWS = 6
DRY_LREC = {"key_dim": 8, "value_dim": 4, "tokens": 40, "heads": 2}
DRY_LREC_CHUNK = 8                  # five chunks over the 40 dry-run tokens (the real kernel uses 64)


def _bf16_rows(stream: Stream, rows: int, width: int, scale: float) -> list:
    return [[bf16(stream.signed(scale)) for _ in range(width)] for _ in range(rows)]


def _summary(counter: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in counter.items()}


def _attn_group(queries: list, keys: list, values: list, scale: float, *, kernel: AttnKernel,
                sink: Optional[list], bucket: dict, shrink: float = 1.0) -> None:
    """All heads of one KV group at one query position: honest check, then the two cheats.

    ``shrink`` scales both radii (1.0 = the certified radii); the unit tests use it to show that the honest
    kernel is not accepted by an arbitrarily small radius.
    """
    head_dim, n = len(queries[0]), len(keys)
    honest_bad, cheats = False, {"truncated_window": False, "dropped_top_key": False, "dropped_sink": False}
    try:
        for h, q in enumerate(queries):
            sink_logit = None if sink is None else sink[h]
            if kernel.name == "eager_observed":
                out, shifted, lse_hat = engine_attention_eager(q, keys, values, scale, sink_logit)
                exact = attention_reference(q, keys, values, scale, sink_logit)
                top = exact["zmax"]
                own = [eager_score_radius(q, k, scale, z) for k, z in zip(keys, exact["z"])]
                own += [0.0] * (len(shifted) - n)
                for j, (obs, z) in enumerate(zip(shifted, exact["z"], strict=True)):
                    key = keys[j] if j < n else [0.0] * head_dim
                    limit = eager_score_radius(q, key, scale, z, a_max=max(own), gap=z - top)
                    if j >= n:
                        limit = ((1.0 + UB) * max(own) + UB * abs(z - top) + ETAB) * OUTWARD
                    bucket["score_checks"] += 1
                    ratio = abs(obs - (z - top)) / limit
                    bucket["score_max_ratio"] = max(bucket["score_max_ratio"], ratio)
                    honest_bad |= ratio > 1.0
                ref = attention_reference(q, keys, values, scale, logits=shifted)
                rad = attention_radius(ref, head_dim, kernel=kernel, observed=True)
            else:
                out, lse_hat = engine_attention_fp32acc(q, keys, values, scale, sink_logit)
                ref = attention_reference(q, keys, values, scale, sink_logit)
                rad = attention_radius(ref, head_dim, kernel=kernel)
            rad = {**rad, "r_o": [r * shrink for r in rad["r_o"]]}
            r_lse = lse_circuit_radius(rad, kernel=kernel) * shrink
            verdict = check_head(ref, rad, r_lse, out, lse_hat)
            honest_bad |= verdict["bad"]
            bucket["max_ratio"] = max(bucket["max_ratio"], verdict["out_ratio"])
            bucket["lse_max_ratio"] = max(bucket["lse_max_ratio"], verdict["lse_ratio"])
            bucket["min_den_margin"] = min(bucket["min_den_margin"], 1.0 - rad["e_den"])
            bucket["r_lse_log2_max"] = max(bucket["r_lse_log2_max"], math.log2(r_lse))

            def tampered(k2, v2, sink2):
                if kernel.name == "eager_observed":
                    o2, _, lse2 = engine_attention_eager(q, k2, v2, scale, sink2)
                    # The cheating kernel's LSE is over its own shifted logits; re-anchor it on the honest maximum.
                    c2 = [bf16(bf16(dot32(q, k)) * bf16(scale)) for k in k2] + ([] if sink2 is None else [bf16(sink2)])
                    honest_c = [bf16(bf16(dot32(q, k)) * bf16(scale)) for k in keys]
                    honest_c += [] if sink_logit is None else [bf16(sink_logit)]
                    lse2 = lse2 + (max(c2) - max(honest_c))
                else:
                    o2, lse2 = engine_attention_fp32acc(q, k2, v2, scale, sink2)
                return check_head(ref, rad, r_lse, o2, lse2)["bad"]

            if n >= 2:
                half = n // 2
                cheats["truncated_window"] |= tampered(keys[half:], values[half:], sink_logit)
                top_key = max(range(n), key=lambda j: ref["p"][j])
                keep = [j for j in range(n) if j != top_key]
                cheats["dropped_top_key"] |= tampered([keys[j] for j in keep], [values[j] for j in keep], sink_logit)
            if sink_logit is not None:
                cheats["dropped_sink"] |= tampered(keys, values, None)
    except LseDomainError:
        bucket["domain_failures"] += 1
        return
    bucket["groups"] += 1
    bucket["honest_bad_groups"] += honest_bad
    if n >= 2:
        bucket["cheat_groups"] += 1
        bucket["truncated_window_rejected"] += cheats["truncated_window"]
        bucket["dropped_top_key_rejected"] += cheats["dropped_top_key"]
    if sink is not None:
        bucket["sink_cheat_groups"] += 1
        bucket["dropped_sink_rejected"] += cheats["dropped_sink"]


def _attn_bucket() -> dict:
    return {"groups": 0, "honest_bad_groups": 0, "domain_failures": 0, "max_ratio": 0.0, "lse_max_ratio": 0.0,
            "min_den_margin": 1.0, "r_lse_log2_max": -math.inf, "cheat_groups": 0, "truncated_window_rejected": 0,
            "dropped_top_key_rejected": 0, "sink_cheat_groups": 0, "dropped_sink_rejected": 0, "score_checks": 0,
            "score_max_ratio": 0.0}


def _synthetic_attention(label: str, tokens: int, heads: int, score_scale: float) -> tuple:
    """bf16 Q / K / V of one KV group.  One planted key per head carries most of the mass (peaked attention)."""
    st = Stream(label)
    keys = _bf16_rows(st, tokens, DRY_HEAD_DIM, 1.0)
    values = _bf16_rows(st, tokens, DRY_HEAD_DIM, 4.0)
    queries = _bf16_rows(st, heads, DRY_HEAD_DIM, score_scale)
    for h in range(heads):
        planted = (7 * h + 3) % tokens
        keys[planted] = [bf16(v * 1.5) for v in queries[h]]
    return queries, keys, values


def dry_run_gqa(spec: ModelSpec, shrink: float = 1.0) -> dict:
    """GQA softmax attention under the tiled fp32 kernel: causal rows of several lengths, all heads of a group."""
    heads = max(2, min(4, spec.heads // spec.kv_groups))
    queries, keys, values = _synthetic_attention(f"non-gemm/gqa/{spec.name}", DRY_TOKENS, heads, 2.0)
    bucket = _attn_bucket()
    for position in DRY_POSITIONS:
        _attn_group(queries, keys[:position + 1], values[:position + 1], DRY_HEAD_DIM ** -0.5,
                    kernel=KERNELS["fp32acc"], sink=None, bucket=bucket, shrink=shrink)
    return {"kernel": "fp32acc", "heads_per_group": heads, "positions": list(DRY_POSITIONS),
            "norm_lift": "2^-32 uncapped", "eta_bf16": "2^-126", **_summary(bucket)}


def dry_run_swa(spec: ModelSpec, shrink: float = 1.0) -> dict:
    """Sliding window with a learned sink logit (the ``gpt-oss`` semantics) under the eager bf16 kernel, and the
    retained-leading-token mask of ``microprogram.swa_indices`` under the tiled fp32 kernel."""
    heads = 2
    window, total = spec.window, 2 * spec.window + 40
    queries, keys, values = _synthetic_attention(f"non-gemm/swa/{spec.name}", total, heads, 1.0)
    sink = [bf16(0.25), bf16(-0.5)]
    logit, token = _attn_bucket(), _attn_bucket()
    positions = (5, window + 7, total - 1)
    for position in positions:
        active = range(max(0, position + 1 - window), position + 1)
        for h in range(heads):                      # keep the planted key inside every window
            keys[active[-1 - h]] = [bf16(v * 1.5) for v in queries[h]]
        _attn_group(queries, [keys[j] for j in active], [values[j] for j in active], DRY_HEAD_DIM ** -0.5,
                    kernel=KERNELS["eager_observed"], sink=sink, bucket=logit, shrink=shrink)
        kept = swa_indices(position, window, spec.sink)
        _attn_group(queries, [keys[j] for j in kept], [values[j] for j in kept], DRY_HEAD_DIM ** -0.5,
                    kernel=KERNELS["fp32acc"], sink=None, bucket=token, shrink=shrink)
    last = swa_indices(total - 1, window, spec.sink)
    return {"window": window, "positions": list(positions),
            "sink_logit": {"kernel": "eager_observed", "semantics": "one learned logit per head, zero value row",
                           **_summary(logit)},
            "sink_tokens": {"kernel": "fp32acc", "sink": spec.sink, "keys_at_last_position": len(last),
                            "keeps_sinks": last[:spec.sink] == tuple(range(spec.sink)), **_summary(token)}}


def dry_run_lrec(spec: ModelSpec, shrink: float = 1.0) -> dict:
    cfg, st = DRY_LREC, Stream(f"non-gemm/lrec/{spec.name}")
    bucket = {"heads": 0, "tokens": 0, "honest_bad_tokens": 0, "max_ratio": 0.0, "zero_decay_rejected_heads": 0,
              "state_reset_rejected_heads": 0, "state_reset_tokens": 0, "state_reset_rejected_tokens": 0}
    chunked = {"heads": 0, "tokens": 0, "honest_bad_tokens": 0, "max_ratio": 0.0, "state_checks": 0,
               "state_bad": 0, "state_max_ratio": 0.0, "inv_inf_norm_max": 0.0, "zero_decay_rejected_heads": 0,
               "state_reset_rejected_heads": 0, "zero_decay_state_rejected_heads": 0,
               "state_reset_state_rejected_heads": 0}
    for _ in range(cfg["heads"]):
        queries = _bf16_rows(st, cfg["tokens"], cfg["key_dim"], 1.0)
        keys = _bf16_rows(st, cfg["tokens"], cfg["key_dim"], 1.0)
        values = _bf16_rows(st, cfg["tokens"], cfg["value_dim"], 2.0)
        decays = [f32(-float(st.unit()) * 0.25 - 0.02) for _ in range(cfg["tokens"])]
        betas = [f32(0.2 + 0.6 * float(st.unit())) for _ in range(cfg["tokens"])]
        ref = lrec_reference(queries, keys, values, decays, betas)
        ref = {**ref, "radii": [[r * shrink for r in row] for row in ref["radii"]]}
        honest = lrec_check(ref, engine_lrec(queries, keys, values, decays, betas))
        bucket["heads"] += 1
        bucket["tokens"] += honest["tokens"]
        bucket["honest_bad_tokens"] += honest["bad_tokens"]
        bucket["max_ratio"] = max(bucket["max_ratio"], honest["max_ratio"])
        skipped = lrec_check(ref, engine_lrec(queries, keys, values, decays, betas, decay_cheat=True))
        bucket["zero_decay_rejected_heads"] += skipped["bad_tokens"] > 0
        cut = cfg["tokens"] // 2
        reset = lrec_check(ref, engine_lrec(queries, keys, values, decays, betas, reset_at=cut))
        bucket["state_reset_rejected_heads"] += reset["bad_tokens"] > 0
        bucket["state_reset_tokens"] += cfg["tokens"] - cut
        bucket["state_reset_rejected_tokens"] += reset["bad_tokens"]
        # The same operands through the chunked WY kernel, against its own enclosure.
        wy = lrec_chunk_reference(queries, keys, values, decays, betas, chunk=DRY_LREC_CHUNK)
        wy = {**wy, "radii": [[r * shrink for r in row] for row in wy["radii"]],
              "state_radius": [r * shrink for r in wy["state_radius"]]}
        outs, final = engine_lrec_chunk(queries, keys, values, decays, betas, chunk=DRY_LREC_CHUNK)
        honest, held = lrec_check(wy, outs), lrec_state_check(wy, final)
        chunked["heads"] += 1
        chunked["tokens"] += honest["tokens"]
        chunked["honest_bad_tokens"] += honest["bad_tokens"]
        chunked["max_ratio"] = max(chunked["max_ratio"], honest["max_ratio"])
        chunked["state_checks"] += held["columns"]
        chunked["state_bad"] += held["bad_columns"]
        chunked["state_max_ratio"] = max(chunked["state_max_ratio"], held["max_ratio"])
        chunked["inv_inf_norm_max"] = max(chunked["inv_inf_norm_max"], wy["inv_inf_norm"])
        for field, kwargs in (("zero_decay", {"decay_cheat": True}),
                              ("state_reset", {"reset_at": (cut // DRY_LREC_CHUNK) * DRY_LREC_CHUNK})):
            outs, final = engine_lrec_chunk(queries, keys, values, decays, betas, chunk=DRY_LREC_CHUNK, **kwargs)
            chunked[f"{field}_rejected_heads"] += lrec_check(wy, outs)["bad_tokens"] > 0
            chunked[f"{field}_state_rejected_heads"] += lrec_state_check(wy, final)["bad_columns"] > 0
    return {"enclosure": LREC_SEQ, "template": "sequential fp32 recurrence, bf16 output store", **cfg,
            **_summary(bucket),
            "chunk_wy": {"enclosure": LREC_CHUNK_WY, "chunk_size": DRY_LREC_CHUNK,
                         "template": "chunked WY scan (unit-triangular solve), bf16 output store",
                         **_summary(chunked)}}


def dry_run_rmsnorm(spec: ModelSpec, shrink: float = 1.0) -> dict:
    """Every RMSNorm template the five checkpoints use, on rows with one large outlier coordinate."""
    templates = {"gpt-oss-20b": ("GptOssRMSNorm",), "Qwen3.5-0.8B": ("Qwen3_5RMSNorm",),
                 "Qwen3-30B-A3B-FP8": ("Qwen3MoeRMSNorm",), "Qwen3-4B": ("Qwen3RMSNorm",),
                 "Llama-3.2-1B-Instruct": ("LlamaRMSNorm",)}[spec.name]
    st = Stream(f"non-gemm/rmsnorm/{spec.name}")
    bucket = {"rows": 0, "honest_bad_rows": 0, "max_ratio": 0.0, "min_rms": math.inf, "max_outlier_ratio": 0.0,
              "floor_violations": 0, "wrong_gain_rejected": 0, "skipped_norm_rejected": 0}
    for class_name in templates:
        template = RMS_TEMPLATES[class_name]
        weight = [bf16(st.signed(0.5) + (0.0 if template.one_plus else 1.0)) for _ in range(DRY_RMS_WIDTH)]
        for row in range(DRY_RMS_ROWS):
            x = [bf16(st.signed(1.0)) for _ in range(DRY_RMS_WIDTH)]
            x[row % DRY_RMS_WIDTH] = bf16(40.0 * (1 + row))             # outlier activation
            out, gain = engine_rmsnorm(x, weight, epsilon=1e-6, template=template)
            verdict = rms_check_row(x, gain, out, epsilon=1e-6, template=template)
            bucket["rows"] += 1
            bucket["honest_bad_rows"] += verdict["ratio"] > shrink or not verdict["floor_ok"]
            bucket["max_ratio"] = max(bucket["max_ratio"], verdict["ratio"])
            bucket["min_rms"] = min(bucket["min_rms"], verdict["rms"])
            bucket["max_outlier_ratio"] = max(bucket["max_outlier_ratio"], verdict["outlier"])
            bucket["floor_violations"] += not verdict["floor_ok"]
            wrong = [bf16(v * (1.0 + 2.0 ** -5)) for v in out]          # gain scaled by 1 + 2^-5
            bucket["wrong_gain_rejected"] += rms_check_row(x, gain, wrong, epsilon=1e-6, template=template)["bad"]
            skipped = [bf16(f32(v * g)) for v, g in zip(x, gain)]       # normalisation skipped
            bucket["skipped_norm_rejected"] += rms_check_row(x, gain, skipped, epsilon=1e-6,
                                                             template=template)["bad"]
    return {"templates": {name: RMS_TEMPLATES[name].name for name in templates}, "floor": RMS_FLOOR,
            "floor_holds": bucket["min_rms"] >= RMS_FLOOR, **_summary(bucket)}


DRY_RUNNERS = {"gqa_softmax": dry_run_gqa, "swa_sinks": dry_run_swa, "lrec": dry_run_lrec, "rmsnorm": dry_run_rmsnorm}


SIGMA = float(PR.REGIMES["external_tight"].sigma)


def cheat_gate(name: str, cheat: str, rejected: int, total: int) -> list:
    """A cheat must be rejected on at least ``sigma`` of the units it was tried on (the gate of the GEMM bench).

    Rejection on every unit is not required: a cheat that changes nothing beyond the radius is in-contract
    (dropping keys of negligible mass, skipping the decay of a head whose gate is zero).  Counts are reported.
    """
    if total and rejected < SIGMA * total:
        return [f"{name}: {cheat} rejected on {rejected} of {total}, below sigma = {SIGMA}"]
    return []


def attention_failures(name: str, b: dict) -> list:
    out = []
    if b["groups"] == 0:
        out.append(f"{name}: no group was checked")
    if b["honest_bad_groups"]:
        out.append(f"{name}: {b['honest_bad_groups']} honest bad groups")
    if b["domain_failures"]:
        out.append(f"{name}: {b['domain_failures']} groups outside the certified LSE domain (fail closed)")
    out += cheat_gate(name, "dropped top-weight key", b["dropped_top_key_rejected"], b["cheat_groups"])
    # Halving the window is often in-contract on peaked attention (the certified key-dropping margin of M0-S1),
    # so it only has to be rejected somewhere; its rate is reported.
    if b["cheat_groups"] and b["truncated_window_rejected"] == 0:
        out.append(f"{name}: truncated KV window never rejected")
    out += cheat_gate(name, "dropped sink", b["dropped_sink_rejected"], b["sink_cheat_groups"])
    return out


def operator_failures(op: str, result: dict) -> list:
    if result.get("status") == "UNOBSERVED":
        return [f"{op}: {result.get('reason', 'kernel not observed')}"]
    if op == "gqa_softmax":
        return attention_failures(op, result)
    if op == "swa_sinks":
        out = attention_failures("swa_sinks(sink logit)", result["sink_logit"])
        if result["sink_logit"]["sink_cheat_groups"] == 0:
            out.append("swa_sinks: no sink was observed")
        if "sink_tokens" in result:
            out += attention_failures("swa_sinks(sink tokens)", result["sink_tokens"])
            if not result["sink_tokens"]["keeps_sinks"]:
                out.append("swa_sinks: the retained-token mask dropped a sink position")
        return out
    if op == "lrec":
        out = []
        if result["tokens"] == 0:
            out.append("lrec: no token was checked")
        if result["honest_bad_tokens"]:
            out.append(f"lrec: {result['honest_bad_tokens']} honest bad tokens")
        out += cheat_gate(op, "zeroed decay", result["zero_decay_rejected_heads"], result["heads"])
        out += cheat_gate(op, "zeroed state", result["state_reset_rejected_heads"], result["heads"])
        wy = result.get("chunk_wy")
        if wy is not None and "honest_bad_tokens" in wy:       # the dry run's separate chunked-template pass
            if wy["honest_bad_tokens"] or wy["state_bad"]:
                out.append(f"lrec(chunk_wy): {wy['honest_bad_tokens']} honest bad tokens, "
                           f"{wy['state_bad']} final states outside the enclosure")
            out += cheat_gate("lrec(chunk_wy)", "zeroed decay", wy["zero_decay_rejected_heads"], wy["heads"])
            out += cheat_gate("lrec(chunk_wy)", "zeroed state", wy["state_reset_rejected_heads"], wy["heads"])
        return out
    out = []
    if result["rows"] == 0:
        out.append("rmsnorm: no row was checked")
    if result["honest_bad_rows"]:
        out.append(f"rmsnorm: {result['honest_bad_rows']} honest bad rows")
    if not result["floor_holds"]:
        out.append(f"rmsnorm: min_rms {result['min_rms']:.3g} below the 1e-3 floor")
    cheat_rows = result.get("cheat_rows", result["rows"])
    out += cheat_gate(op, "wrong gain", result["wrong_gain_rejected"], cheat_rows)
    out += cheat_gate(op, "skipped normalisation", result["skipped_norm_rejected"], cheat_rows)
    return out


def dry_run_model(spec: ModelSpec, shrink: float = 1.0) -> dict:
    operators = {op: DRY_RUNNERS[op](spec, shrink) for op in operators_for(spec.name)}
    failures = [msg for op, res in operators.items() for msg in operator_failures(op, res)]
    return {"model": spec.name, "evidence": "DRY_RUN_SYNTHETIC", "operators": operators,
            "status": "FAILED: " + "; ".join(failures) if failures else "DRY_RUN_OK"}


# ====================================================================================== real mode (torch)
ATTN_SDPA, ATTN_EAGER = "fpn_non_gemm_sdpa", "fpn_non_gemm_eager"
RMS_VEC_MARGIN = 2.0 ** -12     # fp64 rsqrt endpoints are within 2^-22 of the rational ones, i.e. 2^-13 of a radius
QK_NORM_SUFFIXES = ("q_norm", "k_norm")


def t_attention_radius(torch, z, abs_dot, p, o, v, head_dim: int, *, kernel: AttnKernel, observed: bool):
    """Vectorised :func:`attention_radius` over the heads of one KV group (``z, p``: [heads, keys];
    ``o``: [heads, width]; ``v``: [keys, width]).  Returns ``None`` when the domain check fails."""
    tokens = z.shape[-1]
    zmax = z.max(-1, keepdim=True).values
    gam = 0.0 if observed else gamma(head_dim)
    rbar = (gam * abs_dot + 4.0 * U32 * (z.abs() + zmax.abs())).max(-1, keepdim=True).values
    tiles = -(-tokens // KEY_TILE) + SPLIT
    pv_depth, den_depth = 2 * tiles + depth(KEY_TILE), 2 * tiles + KEY_TILE
    if kernel.sequential:
        pv_depth, den_depth = max(pv_depth, depth(tokens)), max(den_depth, tokens)
    emu = (1.0 + UB) * (1.0 + U32) ** pv_depth - 1.0
    emu2 = (1.0 + U32) ** den_depth - 1.0
    one_s = torch.exp(2.0 * rbar) * (1.0 + UE) ** (tiles + 1)
    e_num, e_den0 = one_s * (1.0 + emu) - 1.0, one_s * (1.0 + emu2) - 1.0
    eta_bar = torch.exp(rbar) * ((tiles + 1) * (1.0 + emu) * ETA32 + ETAB)
    e_den = e_den0 + tokens * eta_bar
    den = 1.0 - e_den
    if bool((den <= 0.0).any()) or tokens * NORM_LIFT >= 1.0:
        return None
    spread = (p[:, :, None] * (v[None] - o[:, None, :]).abs()).sum(1)
    mass = v.abs().sum(0, keepdim=True)
    pre = (e_num * spread + one_s * (emu + emu2) * o.abs() + eta_bar * (mass + tokens * o.abs())) / den
    pre = pre + tiles * ETA32
    pre = pre + U32 * (o.abs() + pre)
    peak = v.abs().max(0, keepdim=True).values
    hull = (one_s * (emu + emu2) * peak + eta_bar * (mass + tokens * peak)) / den + tiles * ETA32
    lo, hi = v.min(0, keepdim=True).values - hull, v.max(0, keepdim=True).values + hull
    lo, hi = lo - U32 * lo.abs(), hi + U32 * hi.abs()
    pre = torch.minimum(pre, torch.maximum(o - lo, hi - o))
    r_o = (pre + UB * (o.abs() + pre) + ETAB) * OUTWARD
    return r_o, rbar[:, 0], e_den[:, 0], zmax[:, 0]


def t_lrec(torch, query, key, value, decay, beta, state, *, decay_cheat: bool = False,
           reset_at: Optional[int] = None, radii: bool = True):
    """Vectorised :func:`lrec_reference` over heads (inputs ``[tokens, heads, dim]`` in fp64, ``decay`` and
    ``beta`` ``[tokens, heads]``).  Returns ``(outputs, output_radii, final_state, final_state_radius)``."""
    tokens, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    c = lrec_constants(key_dim)
    g, ck, cq, ea, u2 = c["gamma"], c["c_key"], c["c_query"], c["eps_decay"], c["u2"]
    floor = 4.0 * ETA32 * math.sqrt(key_dim)
    eps = L2_EPS
    k_all = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + eps)
    q_all = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + eps) / math.sqrt(key_dim)
    s = (state.clone() if state is not None
         else torch.zeros(heads, key_dim, value_dim, dtype=torch.float64, device=key.device))
    rho = torch.zeros(heads, value_dim, dtype=torch.float64, device=key.device)
    outs, rads = [], []
    for t in range(tokens):
        if reset_at is not None and t == reset_at:
            s = torch.zeros_like(s)
        k, q, v, b = k_all[t], q_all[t], value[t], beta[t][:, None]
        a = torch.ones_like(b) if decay_cheat else torch.exp(decay[t])[:, None]
        if radii:
            sigma = s.norm(dim=1)
            rho1 = a * (1.0 + ea) * rho + a * ea * sigma
            sigma1 = a * sigma
        s = s * a[:, :, None]
        mem = (s * k[:, :, None]).sum(1)
        delta = b * (v - mem)
        s = s + k[:, :, None] * delta[:, None, :]
        out = (s * q[:, :, None]).sum(1)
        outs.append(out)
        if radii:
            sigma2 = s.norm(dim=1)
            eps_m = g * (sigma1 + rho1) * (1.0 + ck)
            big_d = delta.abs() + b * (ck * sigma1 + (1.0 + ck) * rho1)
            step = (1.0 + ck) * (big_d * u2 + b * eps_m * (1.0 + u2))
            pre = (rho1 + b * ((2.0 * ck + ck * ck) * sigma1 + v.abs() * ck) + step
                   + U32 * (1.0 + U32) * (1.0 + ck) * (big_d * (1.0 + u2) + b * eps_m * (1.0 + u2)) + floor)
            rho = (pre + U32 * (sigma2 + pre)) * OUTWARD
            qn = q.norm(dim=-1, keepdim=True)
            pre = qn * ((1.0 + cq) * (1.0 + g) * rho + (cq + g * (1.0 + cq)) * sigma2) + ETA32
            rads.append((pre + UB * (out.abs() + pre) + ETAB) * OUTWARD)
    return torch.stack(outs), (torch.stack(rads) if radii else None), s, rho


def t_lrec_chunk(torch, query, key, value, decay, beta, state, *, chunk: int = LREC_CHUNK):
    """Vectorised :func:`lrec_chunk_reference` over heads and chunks (same argument layout as :func:`t_lrec`).
    Returns ``(outputs, output_radii, final_state, final_state_radius, info)`` with ``info["inv_inf_norm"]`` the
    largest ``||A^-1||_inf`` and ``info["cond_inf"]`` the largest ``|| |A^-1| |A| ||_inf`` over the chunks."""
    tokens, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    c = lrec_constants(key_dim, any_order=True)
    cs, cqs, ea, u2 = c["c_scale"], c["c_qscale"], c["eps_decay"], c["u2"]
    c2, up, uq = 2.0 * cs + cs * cs, 1.0 + cs, 1.0 + cqs
    u, gn, gc, gs = U32, gamma_any(key_dim), gamma_any(chunk), gamma_any(chunk - 1)
    floor = ETA32 * math.sqrt(key_dim)
    pad = (-tokens) % chunk
    count = (tokens + pad) // chunk
    f64, dev = torch.float64, key.device

    def blocks(x):                                   # [tokens, heads, ...] -> [heads, chunks, chunk, ...]
        x = x.transpose(0, 1)
        if pad:
            x = torch.cat([x, torch.zeros(heads, pad, *x.shape[2:], dtype=f64, device=dev)], 1)
        return x.reshape(heads, count, chunk, *x.shape[2:])

    def mul_err(ax, ex, ay, ey):
        return (ax * ey + ay * ex + ex * ey) * (1.0 + u) + u * ax * ay + ETA32

    def mm_err(ax, ex, ay, ey, gam):
        return ax @ ey + ex @ (ay + ey) + gam * ((ax + ex) @ (ay + ey)) + ay.shape[-2] * ETA32

    k = blocks(key * torch.rsqrt((key * key).sum(-1, keepdim=True) + L2_EPS))
    q = blocks(query * torch.rsqrt((query * query).sum(-1, keepdim=True) + L2_EPS) / math.sqrt(key_dim))
    v, b, gate = blocks(value), blocks(beta)[..., None], blocks(decay)
    cum = gate.cumsum(-1)
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=dev).tril()
    strict, eye = lower.tril(-1), torch.eye(chunk, dtype=f64, device=dev)
    e_g = gs * cum.abs()
    diff = cum[..., :, None] - cum[..., None, :]
    h = (e_g[..., :, None] + e_g[..., None, :]) * (1.0 + u) + u * diff.abs()
    gam = torch.where(lower, diff.clamp(max=0.0).exp(), 0.0)
    e_gam = torch.where(strict, gam * (torch.expm1(h) * (1.0 + ea) + ea) + ETA32, 0.0)      # exp(0) = 1 exactly
    e_cum = cum.exp()[..., None]                                                            # [.., chunk, 1]
    e_e = e_cum * (torch.expm1(e_g)[..., None] * (1.0 + ea) + ea) + ETA32
    f_cum, e_f = gam[..., -1, :, None], e_gam[..., -1, :, None]
    ak, aq = up * k.abs(), uq * q.abs()                                                     # primed majorants
    ek, eq = u * ak, u2 * aq
    akb, ekb = ak * b, mul_err(ak, ek, b, 0.0)
    vb = v * b
    evb = u * vb.abs() + ETA32
    kt, akt, ekt = (x.transpose(-1, -2) for x in (k, ak, ek))
    p1, ep1 = (k * b) @ kt, mm_err(akb, ekb, akt, ekt, gn)
    p2, ep2 = q @ kt, mm_err(aq, eq, akt, ekt, gn)
    nmat, lmat = torch.where(strict, p1 * gam, 0.0), torch.where(lower, p2 * gam, 0.0)
    an, al = up * up * nmat.abs(), up * uq * lmat.abs()
    e_n = torch.where(strict, mul_err(up * up * p1.abs(), ep1, gam, e_gam), 0.0)
    e_l = torch.where(lower, mul_err(up * uq * p2.abs(), ep2, gam, e_gam), 0.0)
    dkb, edkb = k * b * e_cum, mul_err(akb, ekb, e_cum, e_e)
    inv = torch.linalg.solve_triangular(eye + nmat, eye.expand_as(nmat), upper=False, unitriangular=True)
    inv_abs = inv.abs()
    u_ref, w_ref = inv @ vb, inv @ dkb

    def lower_solve(mat, rhs):
        return torch.linalg.solve_triangular(mat, rhs, upper=False, unitriangular=True)

    scale_mat = eye - c2 * (inv_abs @ nmat.abs())                # R^-1: majorises the primed inverse / solutions
    m_abs = lower_solve(scale_mat, inv_abs)
    au, aw = lower_solve(scale_mat, u_ref.abs()), lower_solve(scale_mat, w_ref.abs() + cs * (inv_abs @ dkb.abs()))
    d_n = e_n + gc * (an + e_n)
    iter_mat = eye - m_abs @ d_n                                                            # unit lower triangular
    e_u, e_w = (lower_solve(iter_mat, m_abs @ (e_b + d_n @ x_abs)) for x_abs, e_b in ((au, evb), (aw, edkb)))
    x_p3 = e_w + gn * (aw + e_w)
    qt, eqt = q * e_cum, mul_err(aq, eq, e_cum, e_e)
    x_r = eqt + gn * (uq * qt.abs() + eqt)
    q_norm = uq * q.norm(dim=-1, keepdim=True)
    ktd, ektd = (k * f_cum).transpose(-1, -2), mul_err(ak, ek, f_cum, e_f).transpose(-1, -2)
    a_c, r_cd = e_cum[:, :, -1], (e_e / e_cum)[:, :, -1]                                    # [heads, chunks, 1]
    # (a): theta_i = alpha_i theta_{i-1} + beta_i c_s |v_i| + c_2 |beta_i v_i - D_i|, alpha_i = a_i (1 + beta_i c_2).
    step = gate.exp()[..., None]                                                            # a_i
    log_alpha = (gate + torch.log1p(b[..., 0] * c2)).cumsum(-1)
    carry_mat = torch.where(lower, (log_alpha[..., :, None] - log_alpha[..., None, :]).exp(), 0.0)
    carry_rho = log_alpha.exp()[..., None]
    s = (state.clone() if state is not None else torch.zeros(heads, key_dim, value_dim, dtype=f64, device=dev))
    rho = torch.zeros(heads, value_dim, dtype=f64, device=dev)
    outs, rads = [], []
    for i in range(count):
        abs_s, rho_row = s.abs(), rho[:, None, :]
        d_ref = u_ref[:, i] - w_ref[:, i] @ s
        mem = (vb[:, i] - d_ref).abs()                                                      # beta |k . a S_{i-1}|
        theta = carry_mat[:, i] @ (b[:, i] * cs * v[:, i].abs() + c2 * mem) + carry_rho[:, i] * rho_row
        before = torch.cat([rho_row, theta[:, :-1]], 1)                                     # theta_{i-1}
        d_abs = d_ref.abs() + up * b[:, i] * step[:, i] * before + cs * mem
        e_p3 = x_p3[:, i] @ abs_s + x_p3[:, i].norm(dim=-1, keepdim=True) * rho_row
        e_d = (e_u[:, i] + e_p3 + key_dim * ETA32) * (1.0 + u) + u * d_abs + ETA32
        out = qt[:, i] @ s + lmat[:, i] @ d_ref
        e_r = x_r[:, i] @ abs_s + x_r[:, i].norm(dim=-1, keepdim=True) * rho_row
        e_t = mm_err(al[:, i], e_l[:, i], d_abs, e_d, gc)
        carry = q_norm[:, i] * theta + cqs * out.abs()
        pre = carry + (e_r + e_t + key_dim * ETA32) * (1.0 + u) + u * (out.abs() + carry) + ETA32
        outs.append(out)
        rads.append((pre + UB * (out.abs() + pre) + ETAB) * REF_SLACK)
        e_x2 = mm_err(up * ktd[:, i].abs(), ektd[:, i], d_abs, e_d, gc)
        sigma = s.norm(dim=1)
        s = a_c[:, i, :, None] * s + ktd[:, i] @ d_ref
        n1 = (r_cd[:, i] * (1.0 + u) + u) * a_c[:, i] * (sigma + rho) + floor
        pre = theta[:, -1] + n1 + e_x2.norm(dim=1)
        rho = (pre + u * (s.norm(dim=1) + pre) + floor) * REF_SLACK

    def stack(xs):                                   # [heads, chunks, chunk, value_dim] -> [tokens, heads, ..]
        return torch.stack(xs, 1).reshape(heads, count * chunk, value_dim)[:, :tokens].transpose(0, 1)

    info = {"inv_inf_norm": float(inv_abs.sum(-1).max()),
            "cond_inf": float((inv_abs @ (eye + nmat.abs())).sum(-1).max()), "chunks": count}
    return stack(outs), stack(rads), s, rho, info


def t_chunk_kernel(torch, query, key, value, decay, beta, *, chunk: int = LREC_CHUNK, explicit_inverse: bool = False):
    """fp32 replica of ``torch_chunk_gated_delta_rule`` for one batch element (inputs ``[tokens, heads, dim]`` in
    the activation dtype, l2-normalised in the kernel, zero initial state).  ``explicit_inverse`` takes the
    ``is_torchdynamo_exporting`` branch, which ``LREC_CHUNK_WY_ENCLOSURE_V1`` does not cover.  Returns
    ``(outputs [tokens, heads, value_dim] in the input dtype, final_state [heads, key_dim, value_dim] fp32)``."""
    dtype, tokens = query.dtype, key.shape[0]
    # Same statements as the kernel, on a leading batch dimension of one (layout and contiguity decide which
    # reduction / BLAS path runs, so they are part of the template).
    q, k, v, b, g = (x[None].transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                     for x in (query, key, value, beta, decay))
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    pad = (chunk - tokens % chunk) % chunk
    F = torch.nn.functional
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    b, g = (F.pad(x, (0, pad)) for x in (b, g))
    v_beta, k_beta = v * b.unsqueeze(-1), k * b.unsqueeze(-1)
    q, k, k_beta, v_beta = (x.reshape(x.shape[0], x.shape[1], -1, chunk, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    cum = g.cumsum(dim=3)
    pairwise = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra = (q @ k.transpose(-1, -2)) * pairwise
    decayed = k_beta * cum.exp().unsqueeze(-1)
    if explicit_inverse:
        ut = -ut.tril(-1)
        for i in range(1, chunk):
            row, sub = ut[..., i, :i].clone(), ut[..., :i, :i].clone()
            ut[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut = ut + torch.eye(chunk, dtype=ut.dtype, device=ut.device)
        new_values, k_cum = ut @ v_beta, ut @ decayed
    else:
        new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
        k_cum = torch.linalg.solve_triangular(ut, decayed, upper=False, unitriangular=True)
    s = torch.zeros(1, q.shape[1], k.shape[-1], v.shape[-1], dtype=new_values.dtype, device=q.device)
    out = torch.zeros_like(new_values)
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    for i in range(q.shape[2]):
        v_new = new_values[:, :, i] - k_cum[:, :, i] @ s
        out[:, :, i] = q[:, :, i] @ s + intra[:, :, i] @ v_new
        s = s * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(1, out.shape[1], -1, out.shape[-1])[:, :, :tokens]
    return out.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)[0], s[0]


def calibrate_real(spec: ModelSpec, path: str, args) -> dict:
    import torch
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface, eager_mask, sdpa_mask

    wanted = operators_for(spec.name)
    device = args.device
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    attn = {"bucket": _attn_bucket(), "calls": 0, "kernel": None, "mask_mismatch_rows": 0, "replica_mismatch": 0,
            "sink_heads": 0, "windows": set(), "skipped_batches": 0}
    sink_free = "swa_sinks" not in wanted

    def pick_rows(count: int) -> list:
        extra = torch.randint(0, count, (max(0, args.rows_per_call - 1),), generator=gen).tolist()
        return sorted({count - 1, *extra})

    def analyze_attention(module, query, key, value, mask, out, kw) -> None:
        bucket = attn["bucket"]
        heads, rows, head_dim = query.shape[1], query.shape[2], query.shape[3]
        groups, total = key.shape[1], key.shape[2]
        per = heads // groups
        scale = float(kw.get("scaling", None) or getattr(module, "scaling", head_dim ** -0.5))
        window = kw.get("sliding_window", None) or getattr(module, "sliding_window", None)
        sinks = getattr(module, "sinks", None) if not sink_free else None
        eager = not sink_free
        kernel = KERNELS["eager_observed" if eager else "fp32acc"]
        attn["kernel"] = kernel.name
        attn["windows"].add(window or 0)
        shifted = None
        if eager:
            # Replica of GptOss ``eager_attention_forward``: the shifted bf16 logits are an observed operand.
            logits = torch.matmul(query, key.repeat_interleave(per, dim=1).transpose(2, 3)) * scale
            if mask is not None:
                logits = logits + mask
            if sinks is not None:
                logits = torch.cat([logits, sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, rows, -1)], -1)
            shifted = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.nn.functional.softmax(shifted, dim=-1, dtype=shifted.dtype)
            probs = probs[..., :-1] if sinks is not None else probs
            replica = torch.matmul(probs.to(value.dtype), value.repeat_interleave(per, dim=1)).transpose(1, 2)
            if not torch.equal(replica.contiguous(), out):
                attn["replica_mismatch"] += 1
                return
        offset = total - rows
        for row in pick_rows(rows):
            position = offset + row
            first = max(0, position + 1 - window) if window else 0
            n = position + 1 - first
            if mask is not None:
                allowed = mask[0, 0, row] if mask.dtype == torch.bool else mask[0, 0, row] == 0
                expect = torch.zeros(total, dtype=torch.bool, device=allowed.device)
                expect[first:position + 1] = True
                if not torch.equal(allowed[:total], expect):
                    attn["mask_mismatch_rows"] += 1
                    continue
            for g in range(groups):
                hs = slice(g * per, (g + 1) * per)
                k, v = key[0, g, first:position + 1].double(), value[0, g, first:position + 1].double()
                q, observed_out = query[0, hs, row].double(), out[0, row, hs].double()
                z = scale * (q @ k.T)
                abs_dot = scale * (q.abs() @ k.abs().T)
                sink = sinks[hs].double()[:, None] if sinks is not None else None
                if sink is not None:
                    z, abs_dot = torch.cat([z, sink], -1), torch.cat([abs_dot, torch.zeros_like(sink)], -1)
                    v = torch.cat([v, torch.zeros(1, v.shape[1], dtype=v.dtype, device=v.device)], 0)
                honest_bad = False
                if eager:
                    keep = list(range(first, position + 1)) + ([total] if sinks is not None else [])
                    obs = shifted[0, hs, row][:, keep].double()
                    gap = z - z.max(-1, keepdim=True).values
                    grow = (1.0 + UB) ** 3
                    own = (gamma(head_dim) * abs_dot * grow + (grow - 1.0) * z.abs() + 2.0 * ETAB) * OUTWARD
                    if sink is not None:
                        own[:, -1] = 0.0
                    limit = ((1.0 + UB) * (own + own.max(-1, keepdim=True).values) + UB * gap.abs() + ETAB) * OUTWARD
                    ratio = float(((obs - gap).abs() / limit).max())
                    bucket["score_checks"] += obs.numel()
                    bucket["score_max_ratio"] = max(bucket["score_max_ratio"], ratio)
                    honest_bad |= ratio > 1.0
                    z, abs_dot = obs, torch.zeros_like(obs)
                p = torch.softmax(z, -1)
                o = p @ v
                lse = torch.logsumexp(z, -1)
                res = t_attention_radius(torch, z, abs_dot, p, o, v, head_dim, kernel=kernel, observed=eager)
                if res is None:
                    bucket["domain_failures"] += 1
                    continue
                r_o, rbar, e_den, zmax = res
                try:
                    r_lse = torch.tensor([lse_circuit_radius(
                        {"tokens": z.shape[-1], "rbar": float(rbar[h]), "zmax": float(zmax[h])}, kernel=kernel)
                        for h in range(per)], dtype=torch.float64, device=z.device)
                except LseDomainError:
                    bucket["domain_failures"] += 1
                    continue
                # The kernel does not expose its normalizer: the honest LSE is recomputed in fp32 from the
                # engine's own operands (fp32 scores for SDPA, the observed bf16 logits for the eager kernel).
                z32 = z.float() if eager else torch.cat(
                    [(q.float() @ k.float().T) * scale] + ([sink.float()] if sink is not None else []), -1)
                lse_hat = torch.logsumexp(z32, -1).double()
                out_ratio = float(((observed_out - o).abs() / r_o).max())
                lse_ratio = float(((lse_hat - lse).abs() / r_lse).max())
                honest_bad |= out_ratio > 1.0 or lse_ratio > 1.0
                bucket["groups"] += 1
                bucket["honest_bad_groups"] += honest_bad
                bucket["max_ratio"] = max(bucket["max_ratio"], out_ratio)
                bucket["lse_max_ratio"] = max(bucket["lse_max_ratio"], lse_ratio)
                bucket["min_den_margin"] = min(bucket["min_den_margin"], float(1.0 - e_den.max()))
                bucket["r_lse_log2_max"] = max(bucket["r_lse_log2_max"], math.log2(float(r_lse.max())))

                def rejected(keep_idx) -> bool:
                    """A cheating kernel that attends only ``keep_idx``: fp32 softmax, output-dtype store."""
                    zc = z32[:, keep_idx]
                    oc = (torch.softmax(zc, -1) @ v[keep_idx].float()).to(out.dtype).double()
                    lc = torch.logsumexp(zc, -1).double()
                    return bool((((oc - o).abs() > r_o).any(-1) | ((lc - lse).abs() > r_lse)).any())

                width = z.shape[-1]
                if n >= 2:
                    tail = list(range(n // 2, n)) + ([width - 1] if sink is not None else [])
                    top = int(p[:, :n].max(0).values.argmax())
                    bucket["cheat_groups"] += 1
                    bucket["truncated_window_rejected"] += rejected(tail)
                    bucket["dropped_top_key_rejected"] += rejected([j for j in range(width) if j != top])
                if sink is not None:
                    attn["sink_heads"] += per
                    bucket["sink_cheat_groups"] += 1
                    bucket["dropped_sink_rejected"] += rejected(list(range(n)))

    def attention(kernel_fn):
        def run(module, query, key, value, attention_mask, **kw):
            out, weights = kernel_fn(module, query, key, value, attention_mask, **kw)
            attn["calls"] += 1
            if query.shape[0] != 1:
                attn["skipped_batches"] += 1
            else:
                with torch.no_grad():
                    analyze_attention(module, query.detach(), key.detach(), value.detach(),
                                      None if attention_mask is None else attention_mask.detach(), out.detach(), kw)
            return out, weights
        return run

    def model_eager(module, *a, **kw):
        return sys.modules[type(module).__module__].eager_attention_forward(module, *a, **kw)

    AttentionInterface.register(ATTN_SDPA, attention(sdpa_attention_forward))
    AttentionMaskInterface.register(ATTN_SDPA, sdpa_mask)
    AttentionInterface.register(ATTN_EAGER, attention(model_eager))
    AttentionMaskInterface.register(ATTN_EAGER, eager_mask)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map=device,
                                                 attn_implementation=ATTN_SDPA if sink_free else ATTN_EAGER).eval()

    # ------------------------------------------------------------------ RMSNorm
    rms = {"rows": 0, "honest_bad_rows": 0, "max_ratio": 0.0, "min_rms": math.inf, "min_rms_all_norms": math.inf,
           "max_outlier_ratio": 0.0, "floor_violations": 0, "cheat_rows": 0, "wrong_gain_rejected": 0,
           "skipped_norm_rejected": 0, "exact_rows": 0, "exact_bad_rows": 0, "exact_max_ratio": 0.0,
           "calls": {}, "not_covered": {}}
    acc_cache: dict = {}

    def accumulation(width: int) -> float:
        if width not in acc_cache:
            from dataclasses import replace
            c = replace(_BF16_CONTRACT, store_u_log2=_BF16_CONTRACT.accum_u_log2)
            acc_cache[width] = float(c.dot_relative_bound(width))
        return acc_cache[width]

    def rms_hook(name: str):
        residual = not name.endswith(QK_NORM_SUFFIXES)

        def hook(module, inputs, output):
            cls = type(module).__name__
            template = RMS_TEMPLATES.get(cls)
            if template is None or len(inputs) != 1 or not torch.is_tensor(output):
                rms["not_covered"][cls] = rms["not_covered"].get(cls, 0) + 1
                return
            x, y = inputs[0].detach(), output.detach()
            width = x.shape[-1]
            eps = f32(float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))))
            weight = module.weight.detach()
            gain = (1.0 + weight.float()).double() if template.one_plus else weight.double()
            x2, y2 = x.reshape(-1, width).double(), y.reshape(-1, width).double()
            squares = (x2 * x2).sum(-1, keepdim=True)
            mean = squares / width
            acc = accumulation(width)
            low_rad = torch.clamp(mean * (1.0 - acc) + eps - ETAB, min=eps)
            high_rad = mean * (1.0 + acc) + eps + ETAB
            lo, hi = torch.rsqrt(high_rad) * (1.0 - 2.0 ** -22), torch.rsqrt(low_rad) * (1.0 + 2.0 ** -22)
            a, b = x2 * gain * lo, x2 * gain * hi
            low, high = torch.minimum(a, b), torch.maximum(a, b)
            radius = (torch.maximum(low.abs(), high.abs()) * rms_relative(template.stores)
                      + template.stores * ETAB) * OUTWARD

            def ratio_of(candidate):
                excess = torch.clamp(torch.maximum(low - candidate, candidate - high), min=0.0)
                return (excess / radius).max(-1).values

            ratio = ratio_of(y2)
            row_rms = mean[:, 0].sqrt()
            below = squares[:, 0] < width * RMS_FLOOR ** 2
            bad = ratio > 1.0 - RMS_VEC_MARGIN
            if residual:
                bad = bad | below
                rms["floor_violations"] += int(below.sum())
                rms["min_rms"] = min(rms["min_rms"], float(row_rms.min()))
            rms["min_rms_all_norms"] = min(rms["min_rms_all_norms"], float(row_rms.min()))
            rms["rows"] += ratio.numel()
            rms["honest_bad_rows"] += int(bad.sum())
            rms["max_ratio"] = max(rms["max_ratio"], float(ratio.max()))
            outlier = x2.abs().max(-1).values / row_rms.clamp_min(1e-300)
            rms["max_outlier_ratio"] = max(rms["max_outlier_ratio"], float(outlier.max()))
            key = f"{cls}:{template.name}:{'residual' if residual else 'qk_norm'}"
            rms["calls"][key] = rms["calls"].get(key, 0) + 1
            wrong = (y2 * (1.0 + 2.0 ** -5)).to(y.dtype).double()
            skipped = (x2 * gain).to(y.dtype).double()
            rms["cheat_rows"] += ratio.numel()
            rms["wrong_gain_rejected"] += int((ratio_of(wrong) > 1.0 + RMS_VEC_MARGIN).sum())
            rms["skipped_norm_rejected"] += int((ratio_of(skipped) > 1.0 + RMS_VEC_MARGIN).sum())
            # Exact rational enclosure on the worst row, the largest-outlier row and random rows.
            picks = {int(ratio.argmax()), int(outlier.argmax()),
                     *torch.randint(0, ratio.numel(), (max(0, args.rms_exact_rows - 2),), generator=gen).tolist()}
            gain_list = gain.tolist()
            for r in picks:
                verdict = rms_check_row(x2[r].tolist(), gain_list, y2[r].tolist(), epsilon=eps, template=template,
                                        residual=residual)
                rms["exact_rows"] += 1
                rms["exact_bad_rows"] += verdict["bad"]
                rms["exact_max_ratio"] = max(rms["exact_max_ratio"], verdict["ratio"])
        return hook

    handles = [m.register_forward_hook(rms_hook(name)) for name, m in model.named_modules()
               if "RMSNorm" in type(m).__name__]

    # ------------------------------------------------------------------ Gated DeltaNet recurrence
    lrec = {"calls": 0, "kernels": {}, "heads": 0, "tokens": 0, "honest_bad_tokens": 0, "max_ratio": 0.0,
            "zero_decay_rejected_heads": 0, "state_reset_rejected_heads": 0, "state_checks": 0, "state_bad": 0,
            "state_max_ratio": 0.0, "domain_failures": 0, "unsupported_calls": 0,
            "zero_decay_state_rejected_heads": 0, "state_reset_state_rejected_heads": 0}
    # The chunked-kernel calls measured against the sequential template's radii (diagnostic, never the verdict),
    # and what the chunked enclosure saw.
    seq_v0 = {"tokens": 0, "honest_bad_tokens": 0, "max_ratio": 0.0, "state_checks": 0, "state_bad": 0,
              "state_max_ratio": 0.0}
    wy = {"calls": 0, "chunk_size": 0, "max_chunks": 0, "inv_inf_norm_max": 0.0, "cond_inf_max": 0.0,
          "reference_gap_max": 0.0, "state_radius_over_v0_min": math.inf, "state_radius_over_v0_medians": [],
          "output_radius_over_v0_medians": []}
    restore = []

    def wrap_lrec(owner, fname):
        orig = getattr(owner, fname)

        def run(query, key, value, g, beta, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, **kw):
            result = orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                          output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                          **kw)
            lrec["calls"] += 1
            lrec["kernels"][fname] = lrec["kernels"].get(fname, 0) + 1
            core, last = result
            with torch.no_grad():
                if query.shape[0] != 1 or not use_qk_l2norm_in_kernel:
                    lrec["unsupported_calls"] += 1
                    return result
                qd, kd, vd = (t[0].detach().double() for t in (query, key, value))
                gd, bd = g[0].detach().double(), beta[0].detach().double()
                state = None if initial_state is None else initial_state[0].detach().double()
                observed = core[0].detach().double()
                inputs_ok = (
                    bool(torch.isfinite(qd).all()) and bool(torch.isfinite(kd).all())
                    and bool(torch.isfinite(vd).all()) and bool(torch.isfinite(gd).all())
                    and bool(torch.isfinite(bd).all()) and bool(torch.isfinite(observed).all())
                    and (state is None or bool(torch.isfinite(state).all()))
                    and (last is None or bool(torch.isfinite(last[0].detach().double()).all()))
                )
                if not inputs_ok or bool((gd > 0).any()) or bool((bd < 0).any()) or bool((bd > 1).any()):
                    lrec["domain_failures"] += 1
                    return result
                out, rad, final, rho = t_lrec(torch, qd, kd, vd, gd, bd, state)
                if LREC_KERNEL_ENCLOSURE[fname] == LREC_CHUNK_WY:
                    # V0 stays on record; the verdict below uses the chunked template's own enclosure.
                    ratio0 = ((observed - out).abs() / rad).max(-1).values
                    bad0 = (~torch.isfinite(ratio0)) | (ratio0 > 1.0)
                    seq_v0["tokens"] += ratio0.numel()
                    seq_v0["honest_bad_tokens"] += int(bad0.sum())
                    seq_v0["max_ratio"] = float("inf") if not bool(torch.isfinite(ratio0).all()) else max(seq_v0["max_ratio"], float(ratio0.max()))
                    if last is not None:
                        err0 = (last[0].detach().double() - final).norm(dim=1) / rho.clamp_min(1e-300)
                        bad_s0 = (~torch.isfinite(err0)) | (err0 > 1.0)
                        seq_v0["state_checks"] += err0.numel()
                        seq_v0["state_bad"] += int(bad_s0.sum())
                        seq_v0["state_max_ratio"] = float("inf") if not bool(torch.isfinite(err0).all()) else max(seq_v0["state_max_ratio"], float(err0.max()))
                    size = int(kw.get("chunk_size", LREC_CHUNK))
                    out1, rad1, final1, rho1, info = t_lrec_chunk(torch, qd, kd, vd, gd, bd, state, chunk=size)
                    wy["calls"] += 1
                    wy["chunk_size"], wy["max_chunks"] = size, max(wy["max_chunks"], info["chunks"])
                    wy["inv_inf_norm_max"] = max(wy["inv_inf_norm_max"], info["inv_inf_norm"])
                    wy["cond_inf_max"] = max(wy["cond_inf_max"], info["cond_inf"])
                    wy["reference_gap_max"] = max(wy["reference_gap_max"], float((out1 - out).abs().max()),
                                                  float((final1 - final).abs().max()))
                    over = rho1 / rho.clamp_min(1e-300)
                    wy["state_radius_over_v0_min"] = min(wy["state_radius_over_v0_min"], float(over.min()))
                    wy["state_radius_over_v0_medians"].append(float(over.median()))
                    wy["output_radius_over_v0_medians"].append(float((rad1 / rad).median()))
                    out, rad, final, rho = out1, rad1, final1, rho1
                if not (bool(torch.isfinite(out).all()) and bool(torch.isfinite(rad).all())
                        and bool(torch.isfinite(final).all()) and bool(torch.isfinite(rho).all())):
                    lrec["domain_failures"] += 1
                    return result
                ratio = ((observed - out).abs() / rad).max(-1).values            # [tokens, heads]
                bad_r = (~torch.isfinite(ratio)) | (ratio > 1.0)
                heads = ratio.shape[1]
                lrec["heads"] += heads
                lrec["tokens"] += ratio.numel()
                lrec["honest_bad_tokens"] += int(bad_r.sum())
                lrec["max_ratio"] = float("inf") if not bool(torch.isfinite(ratio).all()) else max(lrec["max_ratio"], float(ratio.max()))
                if last is not None:
                    err = (last[0].detach().double() - final).norm(dim=1) / rho.clamp_min(1e-300)
                    bad_s = (~torch.isfinite(err)) | (err > 1.0)
                    lrec["state_checks"] += err.numel()
                    lrec["state_bad"] += int(bad_s.sum())
                    lrec["state_max_ratio"] = float("inf") if not bool(torch.isfinite(err).all()) else max(lrec["state_max_ratio"], float(err.max()))
                tokens = qd.shape[0]
                for field, kwargs in (("zero_decay_rejected_heads", {"decay_cheat": True}),
                                      ("state_reset_rejected_heads", {"reset_at": tokens // 2})):
                    cheat, _, cheat_state, _ = t_lrec(torch, qd, kd, vd, gd, bd, state, radii=False, **kwargs)
                    cheat = cheat.to(core.dtype).double()
                    lrec[field] += int((((cheat - out).abs() > rad).any(-1)).any(0).sum())
                    moved = (cheat_state.float().double() - final).norm(dim=1) > rho
                    lrec[field.replace("_rejected", "_state_rejected")] += int(moved.any(-1).sum())
            return result

        setattr(owner, fname, run)
        return lambda: setattr(owner, fname, orig)

    if "lrec" in wanted:
        owners = {sys.modules[type(m).__module__] for m in model.modules() if "GatedDeltaNet" in type(m).__name__}
        for owner in owners:
            for fname in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                if hasattr(owner, fname):
                    restore.append(wrap_lrec(owner, fname))

    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        text = " ".join(fh.read().split())
    ids = tok(text, return_tensors="pt").input_ids[:, :args.tokens].to(model.device)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
        for undo in restore:
            undo()

    operators: dict = {}
    b = _summary(attn["bucket"])
    meta = {"kernel": attn["kernel"], "attention_calls": attn["calls"],
            "mask_mismatch_rows": attn["mask_mismatch_rows"], "skipped_batches": attn["skipped_batches"]}
    for op in ("gqa_softmax", "swa_sinks"):
        if op not in wanted:
            continue
        if attn["calls"] == 0 or attn["bucket"]["groups"] + attn["bucket"]["domain_failures"] == 0:
            operators[op] = {"status": "UNOBSERVED", **meta, "eager_replica_mismatch": attn["replica_mismatch"],
                             "reason": "the attention kernel was not observed (no call, a replica mismatch or a "
                                       "mask that disagrees with the declared window)"}
        elif op == "gqa_softmax":
            operators[op] = {**meta, "norm_lift": "2^-32 uncapped", "eta_bf16": "2^-126", **b}
        else:
            operators[op] = {"windows": sorted(attn["windows"]), "eager_replica_mismatch": attn["replica_mismatch"],
                             "sink_logit": {**meta, "sink_heads_checked": attn["sink_heads"],
                                            "semantics": "one learned logit per head, zero value row", **b}}
    if "lrec" in wanted:
        if lrec["calls"] == 0 or lrec["tokens"] == 0:
            operators["lrec"] = {"status": "UNOBSERVED", **lrec,
                                 "reason": "no Gated DeltaNet kernel call was observed (kernelized forward?)"}
        else:
            used = sorted({LREC_KERNEL_ENCLOSURE[name] for name in lrec["kernels"]})
            operators["lrec"] = {"enclosure": used[0] if len(used) == 1 else used,
                                 "enclosure_by_kernel": {name: LREC_KERNEL_ENCLOSURE[name]
                                                         for name in sorted(lrec["kernels"])},
                                 "reference": "fp64: the sequential recurrence (V0 template) / the same chunked "
                                              "algorithm (V1 template); reference_gap_max is their distance",
                                 **_summary(lrec)}
            if wy["calls"]:
                medians = lambda xs: sorted(xs)[len(xs) // 2]                                # noqa: E731
                operators["lrec"]["chunk_wy"] = _summary({
                    **{k: v for k, v in wy.items() if not k.endswith("_medians")},
                    "reference_gap_max": f"{wy['reference_gap_max']:.3e}",
                    "state_radius_over_v0_median": medians(wy["state_radius_over_v0_medians"]),
                    "output_radius_over_v0_median": medians(wy["output_radius_over_v0_medians"]),
                    "terms": "key scale through the exact contraction (theta); cumsum of log-decay "
                             "(gamma_{C-1}); Gram matmul (gamma_n); two unit-triangular solves through |A^-1| "
                             "of each chunk (gamma_C); U - W S; the C-term products K~^T D and L D (gamma_C); "
                             "S exp(G_{C-1}).  Any-order gamma_n = n u / (1 - n u), no lane depth."})
                operators["lrec"]["sequential_v0"] = {
                    "enclosure": LREC_SEQ, "role": "diagnostic: the same chunked-kernel calls against the "
                                                   "sequential template's radii", **_summary(seq_v0)}
    rms_out = _summary({k: v for k, v in rms.items()})
    rms_out.update(floor=RMS_FLOOR, floor_holds=rms["min_rms"] >= RMS_FLOOR,
                   honest_bad_rows=rms["honest_bad_rows"] + rms["exact_bad_rows"])
    operators["rmsnorm"] = rms_out
    failures = [msg for op, res in operators.items() for msg in operator_failures(op, res)]
    if "lrec" in operators and operators["lrec"].get("status") != "UNOBSERVED":
        for field, label in (("state_bad", "final states outside the enclosure"),
                             ("domain_failures", "calls outside the LRec domain"),
                             ("unsupported_calls", "calls the enclosure does not cover")):
            if lrec[field]:
                failures.append(f"lrec: {lrec[field]} {label}")
    if attn["mask_mismatch_rows"]:
        failures.append(f"attention: {attn['mask_mismatch_rows']} rows whose mask is not the declared causal window")
    if attn["replica_mismatch"]:
        failures.append(f"attention: {attn['replica_mismatch']} eager calls the replica did not reproduce bitwise")
    if rms["not_covered"]:
        failures.append(f"rmsnorm: templates not covered {rms['not_covered']}"
                        if set(rms["not_covered"]) - set(RMS_NOT_COVERED) else "")
        failures = [f for f in failures if f]
    on_gpu = str(model.device).startswith("cuda")
    ok_label = "CALIBRATION_OK" if on_gpu else "PLUMBING_SMOKE_CPU_OK"
    return {"model": spec.name, "evidence": "MEASURED" if on_gpu else "PLUMBING_SMOKE_CPU (not calibration evidence)",
            "tokens": int(ids.shape[1]), "device": torch.cuda.get_device_name(0) if on_gpu else "cpu",
            "operators": operators, "status": "FAILED: " + "; ".join(failures) if failures else ok_label}


# ====================================================================================== driver
def summary_line(result: dict) -> str:
    parts = []
    for op, res in result.get("operators", {}).items():
        if res.get("status") == "UNOBSERVED":
            parts.append(f"{op}=UNOBSERVED")
        elif op == "gqa_softmax":
            parts.append(f"gqa={res['honest_bad_groups']}/{res['groups']} r={res['max_ratio']:.3f} "
                         f"topdrop={res['dropped_top_key_rejected']}/{res['cheat_groups']}")
        elif op == "swa_sinks":
            s = res["sink_logit"]
            parts.append(f"swa={s['honest_bad_groups']}/{s['groups']} r={s['max_ratio']:.3f} "
                         f"sinkdrop={s['dropped_sink_rejected']}/{s['sink_cheat_groups']}")
        elif op == "lrec":
            parts.append(f"lrec={res['honest_bad_tokens']}/{res['tokens']} r={res['max_ratio']:.3f} "
                         f"cheats={res['zero_decay_rejected_heads']}+{res['state_reset_rejected_heads']}"
                         f"/{res['heads']}"
                         + (f" state={res['state_bad']}/{res['state_checks']} r={res['state_max_ratio']:.3f}"
                            if "state_checks" in res else "")
                         + (f" wy={res['chunk_wy']['honest_bad_tokens']}/{res['chunk_wy']['tokens']}"
                            if "honest_bad_tokens" in res.get("chunk_wy", {}) else ""))
        else:
            parts.append(f"rms={res['honest_bad_rows']}/{res['rows']} r={res['max_ratio']:.3f} "
                         f"min_rms={res['min_rms']:.3g}")
    return f"{result['model']:24} {result['status'][:60]:22} " + " | ".join(parts)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default="<WORK_DIR>")
    ap.add_argument("--models", default=",".join(m.name for m in MODELS), help="comma-separated subset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="synthetic pure-Python run (no torch, no GPU)")
    ap.add_argument("--confirm-gpu", action="store_true", help="required for a real run on the shared GPU host")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                    help="cpu runs a plumbing smoke test that is not calibration evidence")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--rows-per-call", type=int, default=4, help="query rows analysed per attention call")
    ap.add_argument("--rms-exact-rows", type=int, default=4, help="rows per RMSNorm call re-checked in rationals")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    wanted = [m for m in MODELS if m.name in args.models.split(",")]
    if not wanted:
        ap.error("no known model selected")
    if not args.dry_run and args.device == "cuda" and not args.confirm_gpu:
        print("refusing a real run without --confirm-gpu (the calibration GPU is shared); use --dry-run otherwise",
              file=sys.stderr)
        return 2
    results = []
    for spec in wanted:
        if args.dry_run:
            results.append(dry_run_model(spec))
            continue
        try:
            results.append(calibrate_real(spec, os.path.join(args.models_dir, spec.name), args))
        except Exception as exc:  # one model failing must not hide the others; the run still fails
            results.append({"model": spec.name, "status": f"FAILED: {type(exc).__name__}: {exc}"})
    ok = all(r["status"] in ("DRY_RUN_OK", "CALIBRATION_OK", "PLUMBING_SMOKE_CPU_OK") for r in results)
    evidence = ("DRY_RUN_SYNTHETIC: exercises the radii against a pure-Python kernel emulation; not hardware "
                "calibration" if args.dry_run else
                "MEASURED" if args.device == "cuda" else "PLUMBING_SMOKE_CPU: not calibration evidence")
    report: dict[str, Any] = {
        "bench": "tier2/bench_non_gemm_calibrate", "mode": "dry_run" if args.dry_run else "real",
        "evidence": evidence, "git_sha": os.environ.get("FPN_GIT_SHA") or git_sha(),
        "strict_isfinite_guard": True,
        "command": " ".join(sys.argv), "machine": platform.platform(),
        "constants": {"circuit_norm_lift": "2^-32", "eta_bf16": "2^-126", "attn_key_tile": KEY_TILE,
                      "attn_split_steps": SPLIT, "rms_floor": RMS_FLOOR, "lrec_enclosures": dict(LREC_KERNEL_ENCLOSURE),
                      "lrec_chunk": LREC_CHUNK},
        "operators_by_model": {m.name: list(operators_for(m.name)) for m in wanted},
        "models": results, "ok": ok,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
            fh.write("\n")
    for r in results:
        print(summary_line(r))
    print(("dry run (synthetic, not calibration evidence): " if args.dry_run else
           "calibration: " if args.device == "cuda" else "cpu plumbing smoke (not calibration evidence): ")
          + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
