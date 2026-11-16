# FlashProve-Next

**Verifiable Inference for Production LLM APIs**

[![Stage](https://img.shields.io/badge/Release-Stage_0_Theory_%26_Formalization-0f766e)](#phased-release-roadmap)
[![Lean 4](https://img.shields.io/badge/Lean_4.29.1-18_modules_·_194_declarations-2563eb)](lean/THEOREM.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-slate)](LICENSE)

English · [简体中文](README.zh-CN.md) · [Project Homepage (`docs/`)](docs/index.html) · [Reviewer Guide](REVIEW.md) · [Technical Report (PDF)](paper/flashprove-next.pdf)

---

## What It Is

When you call a closed-weight language model API, you cannot tell what actually ran behind the endpoint: whether the provider routed your request to a smaller model under load, quantized a `bf16` checkpoint to `int4`, skipped layers, truncated the context window, or shortened a hidden system prompt to cut reasoning tokens.

**FlashProve-Next** lets an API provider attach a publicly verifiable zero-knowledge proof to each response, proving that the reply came from the real model configuration the provider published—under two non-negotiable constraints:

1. **Leave production inference untouched.** No fixed-point surrogate model, no deterministic-integer rewrites, and no restrictions on serving optimizations. The engine runs its normal floating-point kernels (`bf16`/`fp8`), temperature and top-$p$ sampling, FlashAttention, CUDA Graph, continuous batching, prefix caching, tensor/pipeline/expert parallelism, and speculative decoding. Proof generation runs asynchronously in the background and never blocks streaming output.
2. **Reveal nothing beyond the standard API.** Apart from what commercial APIs already return, the proof hides weights, layer count, width, whether the model is a Mixture-of-Experts and how it routes, deployed precision, hidden system prompts, and hidden chain-of-thought (CoT) text.

### How It Works in Three Steps

- **Commit once.** When deploying a model, the provider publishes a single cryptographic commitment `model_id` binding the real deployed weights, per-operator rounding bounds, hidden prompts, and sampling key, accompanied by a one-time setup proof. Weights are Reed–Solomon encoded once and opened inside later proofs without ever exposing static codeword symbols, so the commitment never wears out across the service lifetime.
- **Sample and prove in the background.** During serving, read-only hooks record compact boundary summaries of operator inputs and outputs. After the reply finishes streaming, a background worker draws a cost-weighted random sample of blocks and generation steps (315 total at default security parameters) and proves in zero knowledge that each sampled block matches the committed weights within the rounding error allowed by its operator contract. Proving work scales with the sample count rather than total model depth.
- **Verify on any CPU.** Anyone holding `model_id`, the request, the visible reply, and the proof can verify it on an ordinary laptop CPU in seconds, with zero access to weights or provider hardware.

> **Current Release (`v0.2.0`, Stage 0):** This repository is published by verified stage. Stage 0 contains the bilingual overview ([`docs/`](docs/index.html)), the technical report ([`paper/flashprove-next.pdf`](paper/flashprove-next.pdf)), ten protocol specifications ([`specs/01`–`10`](specs/README.md)), the standalone machine-checked Lean 4 library ([`lean/`](lean/THEOREM.md)), and hardware measurements on real checkpoints ([`benchmarks/certified/`](benchmarks/certified/README.md)). Standalone verifier binaries, example proof bundles, and non-invasive vLLM integration code follow in Stage 1.

---

## Moving Beyond FlashProve

Anthropic's **[FlashProve](https://github.com/anthropics/FlashProve)** showed that trustless zero-knowledge proofs are possible for a closed-weight 400B MoE model without hardware enclaves (TEEs). It also made clear why a direct zkML approach cannot be turned on in production serving:

| Dimension | FlashProve (2026-11) | FlashProve-Next |
|---|---|---|
| **What is committed** | A deterministic 4-bit integer surrogate (0 of 96 native-FP replies matched its exact policy) | The actual deployed checkpoint (`bf16`/`fp8`), rounding contract, hidden prompts, and sampling key |
| **How inference runs** | Greedy or fixed-seed integer execution; vLLM stopped during full-size proving | Unchanged production serving (native FP, random sampling, FlashAttention, CUDA Graph, TP/PP/EP, speculative decoding) |
| **What stays private** | Weights only; depth, width, and MoE layout are public; no hidden prompts, CoT, or multi-turn state | Weights, depth, width, MoE/MLA/linear-attention layout, precision, hidden prompts, hidden CoT, and encrypted multi-turn state |
| **Commitment lifetime** | Wears out after 688,127 batches and must be replaced under an unlinked ID | Never wears out across unbounded proofs (static RS commitment + in-circuit hidden column openings) |
| **Proving cost structure** | Proves every layer of every token (~72–94× per-GPU prefill time on H100) | Proves a fixed cost-weighted sample ($3 \times 105 = 315$ draws for $Q=2^{64}, \varepsilon \le 2^{-40}$), decoupled from layer count |

---

## What Is Proved and Measured Today

We strictly separate what is machine-checked in Lean 4, what is measured on GPU hardware, and what remains on paper or in progress:

### 1. Machine-Checked in Lean 4 (`lean/`, 18 Modules, 194 Declarations, Zero `sorry`)
Verified on Lean `4.29.1` + Mathlib using only Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`):
- **Lifetime soundness against adaptive grinding (`RoundByRound`, `LifetimeSoundness`, `ProtocolBridge`):** Over $Q \le 2^{64}$ adaptive random-oracle queries across the entire service lifetime, no response whose bad-computation cost share reaches $\sigma=0.5$ or whose invalid-sampling share reaches $\rho=0.5$ is accepted with probability greater than $2^{-40}$, using $k=105$ draws per pool ($315$ total across prompt compute, generation compute, and token sampling). Includes the exact $p^4$-denominator straight-line extraction bound, concrete $\mathbb F_{p^3}$ Sumcheck and column-query escape bounds, and the formal `hpath` reduction from protocol specifications.
- **Completeness of bucketed temperature / top-$p$ sampling (`BucketSampling`):** Proves that our sort-free integer bucketed sampler (`fpn-samp-bk-v3`) and its witness-free nucleus envelope accept every honest engine token across all temperatures $T>0$ and $p_{\rm top}\in(0,1]$ (ties and underflow included), and that the boundary-pair circuit check has zero prover slack.
- **Zero-knowledge masking & production equivalences (`MaskJoint`, `Gadgets`, `Speculative`, `LinearAttn`, `ArchGeneral`):** Proves causal multi-round transcript uniformity (`multiround_independent_causal`), Libra sumcheck mask bijections, componentwise MoE Top-$K$ routing bounds, exact equivalence of coupled speculative decoding against any untrusted draft model, and exact ring identities for Multi-head Latent Attention (MLA), chunked linear attention, and pipeline/expert parallelism.

### 2. Measured on Real Models (`benchmarks/certified/`, NVIDIA RTX 4070 Ti SUPER 16 GB)
- **Zero false rejections on honest native-FP inference:** Across **249,384** blocks of `Qwen3-0.6B` and **143,544** blocks of `Llama-3.2-1B-Instruct` (`bf16` with `fp32` cuBLAS reductions and FlashAttention-2), **0 blocks** exceeded their certified error radius. The same test caught a real contract violation in PyTorch's default `bf16` split-K prefill reduction (up to 39× above the `fp32`-accumulation bound), motivating our explicit contract cap.
- **Detection of realistic cost-saving cheats:** Across all **8 tested cheat modes** on `Qwen3-0.6B`—cross-model substitution (`Qwen2.5-0.5B`), `int4` quantization, same-weight `fp8 W8A16`, `fp8 W8A8`, low-rank pruning, naive odd-layer skipping, zero-operand layer skipping, and sliding-window KV truncation—sublayer lineage accounting ($h=3$) flags **97.3%–100%** of computation cost as bad (well above $\sigma=0.5$). By contrast, checking only layer boundaries misses same-weight `fp8` (3.3%) and KV truncation (0%).
- **Explicit bound on undetectable sparse-attention tail dropping:** Dropping ultra-low-probability attention keys whose total effect is smaller than `bf16` rounding cannot be rejected without rejecting honest kernels. An omniscient oracle benchmark on `Qwen3-4B` bounds this saving at **12.4%–55.1%** across 4k–32k context, or **5.9%–36.6%** when FlashAttention's scalar log-sum-exp (LSE) is also recorded. Each `model_id` publishes this bound per context length rather than claiming to prevent sub-rounding drops.

### 3. On Paper / Open for Subsequent Stages
- Basefold unique-decoding proximity soundness (`A-PCS`), mechanical extraction linking concrete circuit-builder code to Lean's `CompiledCS`, multi-proof ROM programming hybrids, and real-valued floating-point propagation proofs are established on paper (`specs/02`–`07`).
- Floating-point error propagation for parallel prefix scans in linear-attention layers (`Scan`) and empirical kernel calibration on Hopper/Blackwell FP8 tensor cores, TPUs, and Trainium remain open.

---

## Phased Release Roadmap

| Stage | Deliverables | Status |
|---|---|---|
| **Stage 0 (`v0.2.0`)** | Technical report, formal protocol specifications (`01`–`10`), standalone Lean 4.29.1 proof library (18 modules / 194 declarations), GPU tolerance & cheat benchmarks, bilingual portal | **Released** |
| **Stage 1** | Standalone CPU verifier CLI, full-model proof bundles (honest `ACCEPT` + 7-class cheat `REJECT`), interactive browser demo, and non-invasive vLLM 0.33 + GPU prover prototype | *In progress* |
| **Stage 2** | Unified multi-architecture support (Dense GQA, Sparse MoE, MLA, Linear Attention), byte-level open spec & test vectors, amortized cluster overhead benchmarks toward $\le 5\%$ | *Planned* |

---

## Repository Layout & Verification

```text
FlashProve-Next/
├── README.md                 # English overview
├── README.zh-CN.md           # Chinese overview (中文导读)
├── REVIEW.md                 # Short reviewer guide, trusted base & paper-to-Lean map
├── CHANGELOG.md              # Release history by stage
├── docs/                     # Bilingual web portal + PDF download
├── paper/                    # Technical report (Markdown source, TeX, compiled PDF)
├── specs/                    # Ten protocol & security specifications (01–10)
├── lean/                     # Standalone Lean 4.29.1 + Mathlib formalization (18 modules)
└── benchmarks/certified/     # Empirical tolerance & cheat-detection tables and raw JSON
```

### Check the Lean 4 Proofs

Requires [`elan`](https://github.com/leanprover/elan) (automatically selects Lean `4.29.1` via `lean/lean-toolchain`):

```bash
cd lean
lake exe cache get
bash check.sh
```

Expected final line: `OK: all 18 modules verified, no sorry, only standard axioms`.

### Rebuild the Technical Report PDF

Requires Python 3 and [`tectonic`](https://tectonic-typesetting.github.io/):

```bash
python3 paper/md2tex.py paper/flashprove-next.md paper/flashprove-next.tex
cd paper && tectonic flashprove-next.tex
```
