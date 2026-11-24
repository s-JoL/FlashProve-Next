"""Reference Program cell census under explicit workload assumptions."""

from dataclasses import dataclass
from fractions import Fraction
from functools import cache

from ..ir.format import get_contract
from ..ir.gadget_ir import Program
from .blocks import gemm_tile_row_block, gqa_group_attn_block, rmsnorm_block, swiglu_block
from .microprogram import install_microprogram

LEGACY_STAGE_1_BASELINE = 5_245_800_000


@dataclass(frozen=True)
class Profile:
    name: str
    hidden: int
    intermediate: int
    heads: int
    groups: int
    head_width: int
    experts: int = 1
    active_experts: int = 1
    shared_experts: int = 0


PROFILES = (
    Profile("Qwen3-0.6B", 1024, 3072, 16, 8, 128),
    Profile("Qwen3-4B", 2560, 9728, 32, 8, 128),
    Profile("400B MoE (Maverick-shaped)", 5120, 8192, 40, 8, 128, 128, 1, 1),
)
CONTEXTS = (256, 1024, 4096, 16384, 32768)


@cache
def _count(build, *args, **kwargs):
    p = Program()
    build(p, "block", *args, **kwargs)
    return sum(column.domain_size for column in p.columns.values())


def _repeat(build, amount):
    """Replicate the independent row/token increment of the actual lowering."""
    first, second = build(1), build(2)
    return first + (amount-1)*(second-first)


def census(profile, tokens, *, tile_rows=32):
    gemm = get_contract("bf16_fp32acc")
    h, m, d = profile.hidden, profile.intermediate, profile.head_width
    q, kv = profile.heads*d, profile.groups*d
    shapes = [(h, q), (h, kv), (h, kv), (q, h)]
    active_mlp = profile.active_experts + profile.shared_experts
    shapes += [(h, m), (h, m), (m, h)] * active_mlp
    if profile.experts > 1:
        shapes += [(h, profile.experts)]

    def gemm_cells(inner, rows):
        return _repeat(lambda r: _count(gemm_tile_row_block, inner, r, gemm), rows)

    group = _repeat(lambda t: _count(
        gqa_group_attn_block, profile.heads//profile.groups, t, d, d), tokens)
    norm = 2*_count(rmsnorm_block, h)
    gate = active_mlp*_count(swiglu_block, m, gemm)
    slot = _count(install_microprogram, tokens=tokens, experts=profile.experts,
                  top_k=profile.active_experts, window=128, sink=4, conv_width=4)

    full_cells = sum(gemm_cells(k, n) for k, n in shapes)
    tile_cells = max(gemm_cells(k, min(n, tile_rows)) for k, n in shapes)
    full = {"rmsnorm": norm, "gemm": full_cells, "swiglu": gate,
            "attention": group*profile.groups, "slot": slot}
    # Budget envelope uses the widest tile and one KV group per draw.
    sampled = {**full, "gemm": tile_cells, "attention": group}
    return full, sampled, Fraction(full_cells, sum(k*n for k, n in shapes))


def main():
    print("Legacy Stage 1 baseline (reported, 315 draws)", LEGACY_STAGE_1_BASELINE)
    print("Tier-0 reference IR slopes (scalar cells per additional element)")
    contract = get_contract("bf16_fp32acc")
    print("  RMSNorm per hidden element", _count(rmsnorm_block, 2)-_count(rmsnorm_block, 1))
    print("  SwiGLU per gated element", _count(swiglu_block, 2, contract)
          - _count(swiglu_block, 1, contract))
    print("  attention per added key/value scalar pair", _count(gqa_group_attn_block, 1, 2, 1, 1)
          - _count(gqa_group_attn_block, 1, 1, 1, 1))
    print("Tier-2 reference IR workloads (32-row tile cap; block cells per proof)")
    for tokens in CONTEXTS:
        for profile in PROFILES:
            full, sampled, ratio = census(profile, tokens)
            print(profile.name, f"context={tokens}, full GEMM cells/weight={ratio}")
            for draws, blocks in ((315, full), (300, sampled), (153, sampled)):
                counts = {name: count*draws for name, count in blocks.items()}
                print(f"  {draws:3} draws", counts, "total", sum(counts.values()))


if __name__ == "__main__":
    main()
