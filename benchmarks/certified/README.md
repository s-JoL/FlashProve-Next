# Empirical Rounding-Contract & Cheat-Detection Benchmarks (`benchmarks/certified/`)

Hardware measurements validating the local per-operator rounding contracts, per-GEMM observation granularity, sublayer lineage cost accounting, and long-context sparse-attention bounds specified in [`specs/06-tolerance-and-cheats.md`](../../specs/06-tolerance-and-cheats.md).

This directory provides the experimental methodology, complete summary tables, and raw JSON outputs in [`results/`](results/) (complementing the end-to-end `Qwen3-0.6B` prover and `vLLM 0.33` benchmarks in [`../prover/`](../prover/)). Every measurement was executed on an **NVIDIA RTX 4070 Ti SUPER 16 GB** (Ada Lovelace, `sm_89`) with an Intel i7-14700F CPU under Linux (`torch 2.14.0+cu130`, `transformers 5.20.2`), naming the exact checkpoint in every row.

Evidence labels:
- **Measured:** Executed directly on the named checkpoint and GPU.
- **Oracle upper bound:** Measured on real model activations where the simulated dropper is given exact attention probabilities $p$ to establish a mathematical upper bound on undetectable tail dropping.
- **Unmeasured:** Explicitly noted below so scope boundaries remain transparent.

---

## 1. Executive Summary of Findings

| Suite | Model | Hardware | Metric | Result |
|---|---|---|---|---|
| **Honest Native-FP** | `Qwen3-0.6B` | RTX 4070 Ti SUPER 16 GB | Out-of-radius blocks across 1,152 decode tokens × 9 block types + 108 prefill positions (`fp32` GEMM reductions) | **0 / 249,384** (0 bad across all 15 op types; max $\|e\|/r = 0.994 / 0.990$) |
| **Honest Native-FP** | `Llama-3.2-1B-Instruct` | RTX 4070 Ti SUPER 16 GB | Out-of-radius blocks across 1,152 decode tokens + 108 prefill positions (`fp32` GEMM reductions) | **0 / 143,544** (0 bad across all 13 op types; max $\|e\|/r = 0.990 / 0.986$) |
| **Split-K Falsification** | Both models | RTX 4070 Ti SUPER 16 GB | PyTorch default `allow_bf16_reduced_precision_reduction=True` (`bf16` split-K partial sums on long-K prefill GEMMs) | `Qwen3-0.6B` `o_proj`/`down`: **2,268 / 3,024** prefill rows bad (up to **31×**); `Llama-3.2-1B` up to **39×** (`down` **1,728 / 1,728**) |
| **Granularity** | Both models | RTX 4070 Ti SUPER 16 GB | Rigorous layer-boundary $\ell_2$ ball radius / $\|\Delta_{\text{layer}}\|$ (median, with additive `exp` underflow) | Full layer **$7.2\times 10^{87}$ / $1.4\times 10^{48}$** (vacuous); attention half **$7.3\times 10^{38}$ / $2.4\times 10^{20}$**; MLP half **4.8× / 12.3×** |
| **Cheat Detection** | `Qwen3-0.6B` | RTX 4070 Ti SUPER 16 GB | Bad-cost share $d_C^{(3)}$ under Rule CA ($h=3$ lineage) across 8 cost-saving cheats | `subst` **1.000**, `int4` **1.000**, `fp8w` **1.000**, `fp8wa` **1.000**, `lowrank` **1.000**, `skipA` **0.973**, zero-chain `skipB` **0.987**, `kv_trunc` **1.000** |
| **Attention Tail Bounds** | `Qwen3-4B` (4k–32k) | RTX 4070 Ti SUPER 16 GB | Oracle compute saving $S_{\mathrm{tol}}(T)$ within $0.5\,r_o$ (prose / code) | Output only: **12.4% / 15.2%** (4k), **34.2% / 39.5%** (16k), **52.2% / 55.1%** (32k); with LSE: **5.9% / 8.3%** (4k), **18.9% / 23.7%** (16k), **33.1% / 36.6%** (32k) |

### Architectural Implications

1. **Zero numerical refusals under `fp32` GEMM accumulation (`R32` tier).** Across all **392,928** honest `bf16` blocks of `Qwen3-0.6B` and `Llama-3.2-1B-Instruct` (including the complete `AttnFA` bound with additive `exp` and accumulator underflow), zero blocks and zero individual operators exceeded their certified radius. Peak `ATT` error-to-radius ratios are $0.688\text{–}0.754$ (`Qwen3-0.6B`) and $0.718\text{–}0.723$ (`Llama-3.2-1B`), while the $\approx 0.994$ overall maximum occurs on output-store-dominated blocks (`OPROJ`, `DOWN`, `HEAD`) where IEEE round-to-nearest attains $u|v|$ structurally. Two concrete kernel behaviors shaped the public cap $\mathcal C_{\max}$:
   - PyTorch's default `allow_bf16_reduced_precision_reduction=True` causes cuBLAS to accumulate long-$K$ prefill split-K partial sums in `bf16`, violating `fp32` dot-product bounds by up to $39\times$; $\mathcal C_{\max}$ therefore mandates `fp32` partial sums (`R32`) unless an explicit split-K template is committed.
   - PyTorch evaluates `SiLU(x)` as $x / (1 + e^{-x})$, which overflows $e^{-x}$ to $\infty$ and flushes the quotient to $0$ for $x < -88.7$ while $|\mathrm{silu}(x)| < 2^{-121}$, violating an absolute floor of $2^{-126}$ ($5\times$ ratio) and motivating $\eta_{\mathrm{silu}} = 2^{-120}$.
2. **Per-GEMM observation is mandatory.** Propagating worst-case spectral-norm bounds across an entire layer inflates the radius past $10^{10}\times$ the layer increment—and misses same-weight `fp8 W8A16` ($d_C = 0.033$) and sliding-window KV truncation ($d_C = 0.000$). Observing actual GEMM and attention operands reduces median relative radii to $0.4\%\text{–}2.9\%$.
3. **Sublayer lineage accounting ($h=3$) closes the zero-operand loophole.** Skipping odd layers while publishing zero operands ($x_{\mathrm{attn}} = x_{\mathrm{mlp}} = 0$) satisfies downstream matrix products trivially ($W\cdot 0 = 0$), leaving raw per-block bad cost at $d_C = 0.00011$ ($0.113$ with merged norm tails) despite a $38\%$ FLOP saving. Charging sublayer accounting units along $h=3$ lineages (`Rule CA`, [Spec 01](../../specs/01-relation.md) §4.5) raises the flagged bad-cost share on zero-chain skipping to **$0.987$** and flags **$97.3\%\text{–}100\%$** of compute across all eight tested cheats.
4. **Sub-rounding attention tail savings are bounded and disclosed.** Dropping ultra-low-probability attention keys whose aggregate perturbation stays below `bf16` rounding cannot be rejected without rejecting honest kernels. Recording FlashAttention's scalar log-sum-exp (`LSE`) per head and token cuts the omniscient-oracle upper bound roughly in half (down to $5.9\%\text{–}8.3\%$ at 4k and $33.1\%\text{–}36.6\%$ at 32k on `Qwen3-4B`), and each `model_id` publishes $S_{\mathrm{tol}}(T)$ per context tier.

---

## 2. Complete Measurement Tables

### 2.1 Honest Native-FP Traces (`honest--Qwen3-0.6B.json`, `honest--Llama-3.2-1B-Instruct.json`)

| Model | Phase | Block | Count | Bad | Max $\|e\|/r$ | Comps $>0.5\,r$ | Cap Touch | Median $r/\|v\|$ | Median $r/\mathrm{rms}(v)$ |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen3-0.6B` | decode | `N1` | 32,256 | 0 | 0.983 | 9.59% | 0.000% | 0.78% | 0.29% |
| `Qwen3-0.6B` | decode | `QKV` | 32,256 | 0 | 0.987 | 5.33% | 0.000% | 2.39% | 0.49% |
| `Qwen3-0.6B` | decode | `ATT` | 32,256 | 0 | 0.688 | 0.00% | 0.000% | 1.62% | 0.60% |
| `Qwen3-0.6B` | decode | `OPROJ` | 32,256 | 0 | 0.994 | 16.12% | 0.000% | 0.54% | 0.24% |
| `Qwen3-0.6B` | decode | `N2` | 32,256 | 0 | 0.982 | 9.61% | 0.000% | 0.78% | 0.34% |
| `Qwen3-0.6B` | decode | `GU` | 32,256 | 0 | 0.829 | 0.13% | 0.000% | 2.93% | 0.84% |
| `Qwen3-0.6B` | decode | `DOWN` | 32,256 | 0 | 0.991 | 14.49% | 0.000% | 0.60% | 0.27% |
| `Qwen3-0.6B` | decode | `FN` | 1,152 | 0 | 0.952 | 9.64% | 0.000% | 0.78% | 0.36% |
| `Qwen3-0.6B` | decode | `HEAD` | 1,152 | 0 | 0.989 | 24.02% | 0.000% | 0.40% | 0.33% |
| `Qwen3-0.6B` | prefill | `N1` | 3,024 | 0 | 0.977 | 9.37% | 0.000% | 0.78% | 0.29% |
| `Qwen3-0.6B` | prefill | `QKV` | 3,024 | 0 | 0.986 | 5.34% | 0.000% | 2.39% | 0.47% |
| `Qwen3-0.6B` | prefill | `ATT` | 3,024 | 0 | 0.754 | 0.01% | 2.778% | 1.51% | 0.50% |
| `Qwen3-0.6B` | prefill | `OPROJ` | 3,024 | 0 | 0.989 | 16.47% | 0.000% | 0.52% | 0.23% |
| `Qwen3-0.6B` | prefill | `N2` | 3,024 | 0 | 0.983 | 9.40% | 0.000% | 0.78% | 0.35% |
| `Qwen3-0.6B` | prefill | `GU` | 3,024 | 0 | 0.803 | 0.13% | 0.000% | 2.88% | 0.81% |
| `Qwen3-0.6B` | prefill | `DOWN` | 3,024 | 0 | 0.990 | 14.65% | 0.000% | 0.60% | 0.27% |
| `Qwen3-0.6B` | prefill | `FN` | 108 | 0 | 0.937 | 9.61% | 0.000% | 0.78% | 0.36% |
| `Qwen3-0.6B` | prefill | `HEAD` | 12 | 0 | 0.980 | 22.46% | 0.000% | 0.41% | 0.30% |
| `Llama-3.2-1B-Instruct` | decode | `N1` | 18,432 | 0 | 0.983 | 9.54% | 0.000% | 0.78% | 0.41% |
| `Llama-3.2-1B-Instruct` | decode | `QKV` | 18,432 | 0 | 0.981 | 3.72% | 0.000% | 1.20% | 0.47% |
| `Llama-3.2-1B-Instruct` | decode | `ATT` | 18,432 | 0 | 0.723 | 0.00% | 0.000% | 1.81% | 0.67% |
| `Llama-3.2-1B-Instruct` | decode | `OPROJ` | 18,432 | 0 | 0.990 | 15.22% | 0.000% | 0.58% | 0.34% |
| `Llama-3.2-1B-Instruct` | decode | `N2` | 18,432 | 0 | 0.983 | 9.58% | 0.000% | 0.78% | 0.49% |
| `Llama-3.2-1B-Instruct` | decode | `GU` | 18,432 | 0 | 0.809 | 0.03% | 0.000% | 2.42% | 0.87% |
| `Llama-3.2-1B-Instruct` | decode | `DOWN` | 18,432 | 0 | 0.973 | 7.23% | 0.000% | 0.84% | 0.49% |
| `Llama-3.2-1B-Instruct` | decode | `FN` | 1,152 | 0 | 0.980 | 9.68% | 0.000% | 0.78% | 0.52% |
| `Llama-3.2-1B-Instruct` | decode | `HEAD` | 1,152 | 0 | 0.983 | 18.87% | 0.000% | 0.43% | 0.29% |
| `Llama-3.2-1B-Instruct` | prefill | `N1` | 1,728 | 0 | 0.980 | 9.61% | 0.000% | 0.78% | 0.41% |
| `Llama-3.2-1B-Instruct` | prefill | `QKV` | 1,728 | 0 | 0.968 | 3.68% | 0.000% | 1.20% | 0.48% |
| `Llama-3.2-1B-Instruct` | prefill | `ATT` | 1,728 | 0 | 0.718 | 0.01% | 0.000% | 1.75% | 0.57% |
| `Llama-3.2-1B-Instruct` | prefill | `OPROJ` | 1,728 | 0 | 0.986 | 15.39% | 0.000% | 0.56% | 0.34% |
| `Llama-3.2-1B-Instruct` | prefill | `N2` | 1,728 | 0 | 0.981 | 9.57% | 0.000% | 0.78% | 0.50% |
| `Llama-3.2-1B-Instruct` | prefill | `GU` | 1,728 | 0 | 0.741 | 0.03% | 0.000% | 2.39% | 0.82% |
| `Llama-3.2-1B-Instruct` | prefill | `DOWN` | 1,728 | 0 | 0.958 | 7.16% | 0.000% | 0.83% | 0.49% |
| `Llama-3.2-1B-Instruct` | prefill | `FN` | 108 | 0 | 0.951 | 9.54% | 0.000% | 0.78% | 0.51% |
| `Llama-3.2-1B-Instruct` | prefill | `HEAD` | 12 | 0 | 0.975 | 20.65% | 0.000% | 0.42% | 0.32% |

### 2.2 Cheat Catalog & Lineage Accounting (`cheats--Qwen3-0.6B.json`)

| Model | Cheat Mode | FLOPs Saved $s$ | $d_C$ (Norms Separate) | $d_C$ (Norm Tails Merged) | $d_C^{(3)}$ (Rule CA, $h=3$) | $d_C$ (Layer Boundary) | Median Max $\|e\|/r$ on Bad Blocks |
|---|---|---|---|---|---|---|---|
| `Qwen3-0.6B` | `none` (honest baseline) | 0.00 | 0.000 | 0.000 | **0.000** | 0.000 | — |
| `Qwen3-0.6B` | `subst` (`Qwen2.5-0.5B`) | 0.21 | 0.824 | 0.857 | **1.000** | 1.000 | 1,380–41,962 |
| `Qwen3-0.6B` | `int4` RTN g128 | — | 0.681 | 0.681 | **1.000** | 0.759 | 293–2,126 |
| `Qwen3-0.6B` | `fp8w` W8A16 (same weights) | — | 0.681 | 0.681 | **1.000** | **0.033** | 55–312 |
| `Qwen3-0.6B` | `fp8wa` W8A8 | — | 0.681 | 0.681 | **1.000** | 0.383 | 75–438 |
| `Qwen3-0.6B` | `lowrank` (rank $d/4$) | 0.43 | 0.681 | 0.681 | **1.000** | 0.759 | 806–23,141 |
| `Qwen3-0.6B` | `skipA` (odd layers, naive) | 0.38 | 0.227 | 0.227 | **0.973** | 0.380 | 51–252 |
| `Qwen3-0.6B` | `skipB` (odd layers, zero-chain) | 0.38 | 0.00011 | 0.113 | **0.987** | 0.380 | 128 (`N1`, `N2`) |
| `Qwen3-0.6B` | `kv_trunc` (window $W=16$) | 0.08–0.41 | 0.078 | 0.078 | **1.000** | **0.000** | 13,630 (`ATT`) |

### 2.3 Long-Context Sparse-Attention Oracle Bounds (`longctx--Qwen3-4B.json`)

| Model | Corpus | Context $T$ | Score Std $\sigma_s$ Median (p10–p90) | Attention FLOP Share $a(T)$ | Mass $\le 2^{-8}$ ($f / S_{\mathrm{tol}}$) | Radius Oracle $\beta_{\mathrm{or}}=0.5$ ($f / S_{\mathrm{tol}}$) | LSE Observed (`M1`: Mass $\le 2^{-14} \wedge 0.5\,r_o$) | Per-KV-Group Radius / `M1` |
|---|---|---|---|---|---|---|---|---|
| `Qwen3-4B` | Prose (*Opticks*) | 4,096 | 2.56 (1.92–3.55) | 25.0% | 66.3% / 16.5% | 49.9% / **12.4%** | 23.8% / **5.9%** | 7.4% / 2.6% |
| `Qwen3-4B` | Prose (*Opticks*) | 16,384 | 2.92 (2.13–4.61) | 57.1% | 72.3% / 41.3% | 60.0% / **34.2%** | 33.2% / **18.9%** | 22.1% / 9.6% |
| `Qwen3-4B` | Prose (*Opticks*) | 32,768 | 3.48 (2.40–7.13) | 72.7% | 80.4% / 58.5% | 71.9% / **52.2%** | 45.5% / **33.1%** | 38.6% / 20.1% |
| `Qwen3-4B` | Code (CPython stdlib) | 4,096 | 2.81 (2.13–3.84) | 25.0% | 75.8% / 18.9% | 61.1% / **15.2%** | 33.4% / **8.3%** | 9.8% / 4.1% |
| `Qwen3-4B` | Code (CPython stdlib) | 16,384 | 3.15 (2.36–4.95) | 57.1% | 79.8% / 45.6% | 69.1% / **39.5%** | 41.6% / **23.7%** | 28.2% / 13.1% |
| `Qwen3-4B` | Code (CPython stdlib) | 32,768 | 3.54 (2.55–7.40) | 72.7% | 84.4% / 61.3% | 75.9% / **55.1%** | 50.4% / **36.6%** | 42.4% / 23.5% |

---

## 3. Experimental Protocol & Unmeasured Scope

- **Serving harness:** HuggingFace `transformers` in `bf16` with `DynamicCache` KV-cached decoding and PyTorch SDPA **FlashAttention-2**, evaluated across 12 prompts (9 short chat prompts and 3 long code-comprehension prompts of 1.5k–3.6k tokens) × 96 sampled decode steps plus 108 prefill positions ($T=1.0, p_{\mathrm{top}}=0.99$).
- **Reference scorer:** Exact `float64` GPU evaluator recomputing one-hop local references from the engine's recorded block inputs (`h_in, x_attn, q_rope, K, V, o, h_mid, x_mlp, a, h_out`, final norm, and `fp32` logits) under the example `bf16` cap $\mathcal C_{\max}$ (`u_store = 2^-8`, `fp32` dot accumulation `u_acc = 2^-24` with depth $\lceil n/16\rceil + 64$, elementwise $4\text{ ulp}$ `fp32`, `SiLU` floor $\eta = 2^{-120}$, `AttnFA` `bf16` probability store $u_P = 2^{-8}$, tile $B_c = 64$, IEEE `fp32` `exp` underflow $\eta_{\exp} = 2^{-150}$).
- **Outside this standalone calibration suite:** End-to-end zero-knowledge proving and `vLLM 0.33` CUDA-graph serving benchmarks are reported separately in [`../prover/`](../prover/) and Section 12.7–12.9 of the technical report; native Hopper/Blackwell `fp8` tensor-core instructions (`torch._scaled_mm`; `fp8` cheats above are simulated via quantize-dequantize), trained sparse MoE and linear-attention (`LREC`) checkpoints, non-oracle page-level sparse-attention predictors, and TPU / Trainium accelerators remain open for Stage 2.
