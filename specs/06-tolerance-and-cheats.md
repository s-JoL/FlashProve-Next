# Spec 06: Per-Operator Rounding Contracts, Bucketed Sampling & Cheat Catalog (`fpn-tolerance-1.0`)

**Specification ID:** `fpn-tolerance-1.0` | **Role:** Per-operator rounding-contract templates, forward radius propagation, per-GEMM observation granularity, sort-free bucketed sampling (`fpn-samp-bk-v3`), sublayer lineage accounting (`Rule CA`), and empirical cheat detection

**Hardware Calibration Baseline:** [`benchmarks/certified/`](../benchmarks/certified/README.md) on an **NVIDIA RTX 4070 Ti SUPER 16 GB** (Ada Lovelace, `sm_89`) with an Intel i7-14700F CPU (`torch 2.14.0+cu130`, `transformers 5.20.2`).

---

## Overview and Design Principles

Proving inference for a live closed-weight LLM API cannot rely on replacing deployed floating-point kernels with a fixed-point surrogate model: native `bf16`/`fp8` execution with FlashAttention, tensor cores, and dynamic batching diverges from any fixed integer surrogate, and under strict zero knowledge even a tiny rate of numerical proof failures leaks hidden activations via the "can-it-prove" bit ([Spec 03](03-zero-knowledge.md) §4.6). Conversely, a naive cross-layer fixed tolerance either rejects honest adversarial prompts or becomes loose enough to admit low-bit quantization and skipped layers ([Spec 04](04-completeness.md) Prop 2$'$).

This specification bridges cryptographic spot-checking and unchanged native floating-point serving through four mechanisms:

1. **Local one-hop per-operator rounding contracts (§1–§2).** Each sampled computation block evaluates its exact reference function starting from the **engine's own committed input tensors** and propagates a committed per-operator rounding contract forward across that single block into a box-plus-ball radius $(\mathbf r_b, \varrho_b)$. Because reference values never chain across blocks, floating-point error stays at single-operator scale. With `fp32` cuBLAS reductions (`R32` tier), **0 of 249,384** blocks on `Qwen3-0.6B` and **0 of 143,544** blocks on `Llama-3.2-1B-Instruct` violate their certified radius, whereas PyTorch's default `bf16` split-K prefill path violates `fp32` accumulation by up to $39\times$.
2. **Per-GEMM observation granularity (§2.3).** Propagating worst-case spectral-norm balls across an entire Transformer layer inflates radii past $10^{10}\times$ the layer increment and misses same-weight `fp8 W8A16` ($d_C = 0.033$). Observing actual matrix-multiply (`GEMM`) and attention operands keeps median relative radii at $0.4\%\text{–}2.9\%$ and flags $100\%$ of quantized GEMM blocks.
3. **Sort-free integer bucketed keyed sampling (`fpn-samp-bk-v3`, §3).** Partitioning the vocabulary into contiguous 32-token buckets with keyed Poseidon2 sponge Gumbel noise per bucket and one 48-bit inverse-CDF draw inside the winning bucket eliminates vocabulary-wide sorting and per-token PRF evaluations, shrinking vocabulary circuit cost by roughly an order of magnitude. A witness-free per-token nucleus envelope (`sure` / `poss`, `[Lean: BucketSampling.nucleus_sandwich]`) guarantees $100\%$ honest acceptance across all temperatures $T > 0$ and $p_{\mathrm{top}} \in (0, 1]$ (`[Lean: BucketSampling.bucket_sampling_complete]`).
4. **Fine-grained checks, sublayer billing, and $h=3$ consumer-lineage attribution (`Rule CA`, §5).** An operator whose input is chosen by the prover is trivially satisfied at zero cost when fed zeros ($W\cdot 0 = 0$). Charging sublayer accounting units and propagating defects down $h=3$ causal dependency lineages raises the flagged bad-cost share on zero-operand odd-layer skipping from $0.00011$ to **$0.987$**, and flags **$97.3\%\text{–}100\%$** of compute cost across all eight measured cheat modes. Sub-rounding sparse-attention tail savings are bounded via omniscient-oracle measurements, halved by recording FlashAttention's scalar `LSE`, and published per `model_id` and context tier (§5.2).

---

## 1. Per-Operator Rounding Contracts

### 1.1 Contract Menu `[Def]`

A **rounding contract** $\mathcal C$ maps each `(operator kind, numerical format)` pair to a parameterized error template from a protocol-fixed menu. Circuits interpret templates rather than hardware vendor names. Every template bounds the discrepancy between the physical kernel output $\hat o(\hat a)$ and the exact mathematical operator $o(\hat a)$ evaluated at the **actual engine input** $\hat a \in D_o$; §2 converts these local bounds into bounds around the reference input.

| Template | Contract Statement on Domain $D_o$ | Parameters & Notes |
|---|---|---|
| $\mathsf{Store}_k$ | $\hat v = v(1+\theta) + \eta,\; |\theta| \le u_k,\; |\eta| \le \eta_k$ | Round-to-nearest unit roundoff $u_k$: `bf16` $2^{-8}$, `fp16` $2^{-11}$, `fp32` $2^{-24}$, `e4m3` $2^{-4}$, `e5m2` $2^{-3}$ (doubled under faithful/truncation rounding). Subnormal/FTZ floor $\eta_k$: minimum normal $N_{\min,k}$ when FTZ/DAZ is enabled, else half the subnormal spacing (scaled by block scale $s$ when block-quantized). |
| $\mathsf{Dot}(n; \mathcal T)$ | $\bigl|\widehat{\langle a, b\rangle} - \langle a, b\rangle\bigr| \le \gamma_h \langle |a|, |b|\rangle + \eta_{\mathrm{dot}}$, $\gamma_h = \frac{h u_{\mathrm{acc}}}{1 - h u_{\mathrm{acc}}}$ ($h u_{\mathrm{acc}} < 1$) | $h$ is the maximum leaf-to-root reduction depth of the physical FMA / tree-reduction DAG $\mathcal T$ ($h = n$ for sequential / atomic accumulation; $+1$ at intermediate store precision if products are rounded prior to accumulation). |
| $\mathsf{Dot}^{\mathrm{sk}}(n; S, k_p, k_r)$ | Partition $n$ into $S \le S_{\max}$ contiguous segments accumulated at $u_{\mathrm{acc}}$ (depth $h_s$), stored in format $k_p$, and reduced at depth $h_r$ in format $k_r$ | Collapses to $\mathsf{Dot}$ when $k_p = \text{fp32}$; models cuBLAS split-K when $k_p = \text{bf16}$ (explicit formula below). |
| $\mathsf{Elem}_f$ | $\bigl|\hat f(x) - f(x)\bigr| \le u_f |f(x)| + \eta_f$ | Covers polynomial approximations and special range-reduction paths. On `RTX 4070 Ti SUPER`, PyTorch `SiLU(x) = x/(1 + exp(-x))` flushes to $0$ for $x < -88.7$ where $|\mathrm{silu}(x)| < 2^{-121}$, falsifying $\eta = 2^{-126}$ ($5\times$ ratio) and fixing $\eta_{\mathrm{silu}} = 2^{-120}$ `[Meas]`. |
| $\mathsf{AttnFA}(n; B_c, S_{kv})$ | Composite online-softmax bound (§2.2 Row 7) | Parameterized by probability format $u_P$ before $PV$, $u_{\exp}$, $\eta_{\exp}$, $u_{\mathrm{acc}}$, key tile $B_c$, split-KV factor $S_{kv}$, normalizer depth $h_\ell$, and output format $u_o$. |
| $\mathsf{Reduce}(P; \mathcal T_r, k)$ | Cross-device shard sum stored in format $k$ and reduced over topology $\mathcal T_r$ | Effective depth $h = \lceil\log_2 P\rceil$ for tree reductions; $h = P - 1$ for ring `reduce-scatter` / `all-reduce`. |
| $\mathsf{Scan}(B_c)$ | Chunked parallel prefix scan for linear attention (`LREC`) | `[Open]` — architectures requiring parallel floating-point prefix scans lie outside numerical completeness Corollary 3.1 until a scan DAG error induction is supplied (§2.2, Row 9). |

**Split-K radius composition `[Paper]`.** Let $y_s$ denote the exact partial sum on segment $s$ and $\Sigma_{|y|} := \sum_{s=1}^S |y_s|$. Composing $\mathsf{Dot}$ (intra-segment), $\mathsf{Store}_{k_p}$ (partial-sum store), and $\mathsf{Dot}$ (inter-segment reduction in $k_r$) multiplicatively without first-order truncation gives

$$
\bigl|\hat y - y\bigr| \le \bigl(1 + \gamma_{h_r}^{(k_r)}\bigr)\Bigl[(1 + u_{k_p})\gamma_{h_s}\langle |a|, |b|\rangle + u_{k_p}\Sigma_{|y|} + S\eta_{k_p}\Bigr] + \gamma_{h_r}^{(k_r)}\Sigma_{|y|} + \eta_{\mathrm{dot}}.
$$

Because split count $S$ is selected dynamically by kernel heuristics based on batch geometry, a split-K contract maximizes the right-hand side over the allowed split set $\mathcal S \subseteq \{1, 2, \dots, S_{\max}\}$ (computable in $O(S_{\max})$ extra work from $S_{\max}$ nested base segments; bounding $\Sigma_{|y|} \le \langle |a|, |b|\rangle$ without nested segments loosens radii by $\kappa \approx \frac{2}{\pi}\sqrt{n} \approx 20\text{–}40\times$ at $n = 1024\text{–}4096$ `[Est]`).

---

### 1.2 Publication, Public Cap $\mathcal C_{\max}$, and Reduction Tiers `[Def]`

- **Commitment:** Kernel libraries specify contract parameters per `(hardware, version, op, dtype)`. The provider commits its contract selection inside `model_id` via the per-layer stable configuration table $\mathsf{Lt}$ ([Spec 09](09-model-commitment.md) §3.2).
- **Public profile cap $\mathcal C_{\max}(\mathrm{Prof})$:** Every proof enforces $\mathcal C \preceq \mathcal C_{\max}(\mathrm{Prof})$. Because a malicious provider may commit the loosest contract allowed by the profile, cheat-detection power is always evaluated at $\mathcal C_{\max}$.
- **Mandatory `C_max` entries:**
  1. **Default `R32` GEMM accumulation tier (`k_p = fp32`):** Requires disabling `bf16` split-K partial-sum reductions (e.g., `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`). This preserves tight $0.4\%\text{–}0.8\%$ GEMM radii so same-weight `fp8` quantization cannot hide inside a $5\text{–}40\times$ split-K slack band.
  2. **`fp32` LM-head logits:** Engines compute and sample from `fp32` logits (never written to the trace commitment, adding zero online bandwidth) to keep the sampling manipulation band $\bar\delta(T)$ at $\approx 2.5\times 10^{-3}$ rather than $0.08\text{–}0.24$ (§3.3).
  3. **Bit-exact deterministic sub-computations:** Position-only or schema-only tables (`RoPE` $\cos/\sin$ tables, causal masks, sampling lookup tables $E, \mathsf G, \mathsf{LN}$, and compiled grammar masks) are committed by exact recipe and reproduced bit-for-bit without rounding slack.

---

### 1.3 Empirical Calibration of `H-contract` `[Meas]`

Evaluated on `NVIDIA RTX 4070 Ti SUPER 16 GB` under HuggingFace `bf16` + PyTorch SDPA FlashAttention-2 across 12 prompts (including 1.5k–3.6k code prompts) × 96 decode steps + 108 prefill positions:

| Model | Engine Configuration | Per-GEMM Bad Blocks | Per-Op Bad Ops | Max $\|e\|/r$ (Decode / Prefill) | Output Cap Touch Rate |
|---|---|---|---|---|---|
| `Qwen3-0.6B` | `R32` (`fp32` reductions) | **0 / 249,384** | **0** (15 op types) | 0.994 / 0.990 | `ATT` prefill 2.78% only (1–few keys where value hull is a point); 0% elsewhere |
| `Llama-3.2-1B-Instruct` | `R32` (`fp32` reductions) | **0 / 143,544** | **0** (13 op types) | 0.990 / 0.986 | 0.000% across all blocks |
| `Qwen3-0.6B` | PyTorch default (`bf16` split-K) | `o_proj` / `down`: **2,268 / 3,024** prefill rows bad | — | Up to **31.0×** | Decode: 0 bad |
| `Llama-3.2-1B-Instruct` | PyTorch default (`bf16` split-K) | `k/v_proj` **1,440 / 1,728**, `down` **1,728 / 1,728** bad | — | Up to **38.7×** | Decode: 0 bad |

*(Note: The standalone scorer counts `N1`, `N2`, and `FN` normalizations as separate diagnostic blocks, whereas [Spec 05](05-computation-model.md) §1.3 merges normalization tails into their producing GEMM blocks; zero honest violations hold under both partitionings and per individual operator.)*

---

## 2. Certified Radii, Propagation Rules, and Granularity

### 2.1 Tolerance Predicate $\mathrm{Tol}_b$ `[Def]`

Given the engine's committed block inputs $\hat v_{\mathrm{in}}$, the circuit computes the integer-encoded local reference $v = F_b(\hat v_{\mathrm{in}})$ (incorporating reference-discretization bound $\mathbf r_{\mathrm{ref}}$ into $|v| \leftarrow |v^{\mathrm{ref}}| + \mathbf r_{\mathrm{ref}}$) together with componentwise box radii $\mathbf r \ge 0$ and optional Euclidean ball radius $\varrho \ge 0$:

$$
\mathrm{Tol}_b(\mathrm{In}; \mathrm{Out}) \iff \Biggl[\sum_i \bigl(|\hat v_i - v_i| - r_i\bigr)_+^2 \le \varrho^2\Biggr] \;\wedge\; \hat v \in \mathrm{Cap}_b \;\wedge\; \mathrm{Fin}(\mathrm{In}, \mathrm{Out}) \;\wedge\; \mathrm{In} \in \mathrm{Rng}_b.
$$

- **Box-plus-ball discipline:** $\varrho = 0$ on all standard Transformer tensors; $\varrho > 0$ is permitted exclusively on chunk-boundary recurrent states (`LREC`). Because every block starts fresh from committed engine inputs, $\varrho$ never propagates across blocks.
- **Output range constraints $\mathrm{Cap}_b$ vs. error radii:** An output range bound $\hat v_i \in [\underline C_i, \overline C_i]$ (e.g., attention value convex hull dilated by $(\epsilon_\mu + \epsilon'_\mu)\max_j |v_{ji}|$ plus underflow/store terms, `RMSNorm` $|y_i| \le |g_i|\sqrt{d}(1+u_{\mathrm{rms}})$, `softmax`/`sigmoid` $[0,1]$, `tanh` $[-1,1]$) is enforced as an **independent set membership constraint**, never confused with a symmetric error radius $C_i$ (since $v_i = -C_i, \hat v_i = +C_i$ spans $2C_i$).
- **Finiteness and modular wrap-around guards:** Any `NaN`/`Inf` falsifies $\mathrm{Fin}$, and $\mathrm{Rng}_b$ enforces static range-certificate bounds ($n\cdot 2^{m_a + m_b} < p/2$, `[Lean: BoundedLift.signedLift_unique_of_bounded]`).

---

### 2.2 Forward Propagation Rules `[Paper]`

**Master propagation lemma.** Let $\hat x = x + e$ with $|e| \le \mathbf r_x$. If an operator satisfies $|\hat o(\hat x) - o(\hat x)| \le u|o(\hat x)| + \eta$ on $D_o$ and admits a valid box envelope $|o(\hat x) - o(x)| \le \Lambda_o(x, \mathbf r_x)$ across the entire input box, then by the triangle inequality:

$$
\bigl|\hat o(\hat x) - o(x)\bigr| \le (1 + u)\,\Lambda_o(x, \mathbf r_x) + u\bigl|o(x)\bigr| + \eta.
$$

*(Applying this twice to identity stores yields $(1+u)u + u = 2u + u^2$, matching exact two-step rounding without first-order truncation error.)*

| # | Operator | Propagated Box Radius $\mathbf r_y$ (Prior to Final Output $\mathsf{Store}_k$) |
|---|---|---|
| 1 | **Linear / GEMM** $y = Wx$ ($W$ committed) | $\lvert W\rvert \mathbf r_x + \gamma_h \lvert W\rvert(\lvert x\rvert + \mathbf r_x) + \eta_{\mathrm{dot}}$, evaluated in-circuit via a single product against the committed radius matrix $\mathbf W^r = \lvert\mathbf W^{\mathrm{cd}}\rvert + \kappa_n m$ ([Spec 09](09-model-commitment.md) §2.1), where $\kappa_n = \lceil (1+\gamma_n)/\gamma_n\rceil$ absorbs canonical-decode truncation on the $0.04\%\text{–}0.05\%$ of inexact weights. |
| 2 | **Bilinear Engine Product** $AB$ ($QK^\top$) | $\lvert A\rvert\mathbf r_B + \mathbf r_A\lvert B\rvert + \mathbf r_A\mathbf r_B + \gamma_h(\lvert A\rvert + \mathbf r_A)(\lvert B\rvert + \mathbf r_B) + \eta_{\mathrm{dot}}$. |
| 3 | **Elementwise** $f(x)$ | $\Lambda_f = \bigl(\sup_{|t - x| \le r_x} |f'(t)|\bigr)\mathbf r_x$ (`SiLU` $1.0998$, `GELU` $1.129$, `sigmoid` $1/4$, `tanh` $1$; `exp` uses exact $e^x(e^{r_x} - 1)$). |
| 4 | **Residual Add / Gated Product** $a \odot b$ | $\mathbf r_a + \mathbf r_b$ for addition; $\lvert a\rvert\mathbf r_b + \mathbf r_a\lvert b\rvert + \mathbf r_a\mathbf r_b$ for Hadamard gating. |
| 5 | **RMSNorm** $y = g \odot x / s$ | With $|\hat s - s| \le \Delta_s := \frac{(2|x| + \mathbf r_x)^\top \mathbf r_x}{d\,s} + \text{dot/rsqrt terms}$ and $s_{\min} = \max(\sqrt{\epsilon}, s - \Delta_s)$, $\mathbf r_y = |g|\bigl(\mathbf r_x + |x|\Delta_s / s\bigr) / s_{\min}$ composed with two elementwise multiplications. |
| 6 | **LayerNorm** | Centering radius $\mathbf r_{x-\mu} = \mathbf r_x + \overline{\mathbf r_x} + \gamma_{h(d)}\overline{|x|}$ composed with Row 5 over variance. |
| 7 | **Online-Softmax Attention (`AttnFA`)** | $\mathbf r_o = \dfrac{\epsilon_N \sum_j p_j|v_j - o| + (1+\epsilon_\sigma)(\epsilon_\mu + \epsilon'_\mu)|o| + (1+\epsilon_N)\sum_j p_j\mathbf r_{v_j} + \bar\eta\bigl(\sum_j(|v_j| + \mathbf r_{v_j}) + n|o|\bigr)}{1 - \epsilon_D - n\bar\eta} + n_t\eta_{\mathrm{acc}}$, followed by final division and output store (derivation below). |
| 8 | **RoPE** | Pairwise rotation $|R|\mathbf r_x$ composed with three `bf16` roundings against committed exact $\cos/\sin$ tables. |
| 9 | **Chunked Recurrence (`LREC`)** | Sequential reference bound $\|E_t\|_F \le \alpha_t\|I - \beta_t k_t k_t^\top\|_2 \|E_{t-1}\|_F + \ell_t$; parallel `Scan` template `[Open]`. |
| 10 | **MoE Combine** $\sum_{e \in S} g_e E_e$ | $\sum_{e \in S}\bigl(|g_e|\mathbf r_{E_e} + r_{g_e}|E_e| + r_{g_e}\mathbf r_{E_e}\bigr)$ plus reduction/store; active set $S$ certified via §3.3. |
| 11 | **Row-Split Tensor-Parallel GEMM** | Instantiates $\mathsf{Dot}^{\mathrm{sk}}$ with $S = P$ shards and topology depth $h_r$ from $\mathsf{Reduce}(P; \mathcal T_r, k)$. |

**Derivation of Row 7 (`AttnFA`) and why additive `exp` underflow is essential `[Paper]`.** Let $p_j = e^{z_j - m} / \sum_k e^{z_k - m}$ be the exact softmax weights across $n$ keys processed in $n_t = \lceil n / B_c\rceil + S_{kv}$ tiles. In FlashAttention's online rescaling recurrence, both the numerator accumulator and normalizer $\ell$ multiply by the **identical** rounded tile-rescaling factors $\hat\alpha_{\tau'} = \mathrm{fl}(e^{m_{\mathrm{old}} - m_{\mathrm{new}}})$. Factoring each key's multiplicative perturbation into shared rescaling/score factor $(1+\sigma_j)$ ($|\sigma_j| \le \epsilon_\sigma := e^{2\bar r_z}(1+u_{\exp})^{n_t+1}-1$) and numerator/denominator-specific factors $(1+\mu_j), (1+\mu'_j)$ ($|\mu_j| \le \epsilon_\mu := (1+u_P)(1+u_{\mathrm{acc}})^{2n_t+h_{PV}}-1$, $|\mu'_j| \le \epsilon'_\mu := (1+u_{\mathrm{acc}})^{2n_t+h_\ell}-1$, $\epsilon_N = (1+\epsilon_\sigma)(1+\epsilon_\mu)-1$, $\epsilon_D = (1+\epsilon_\sigma)(1+\epsilon'_\mu)-1$) cancels shared scaling against $\sum_j p_j(v_j - o) = 0$.

*Remark (1025-key exponential underflow example):* A purely multiplicative `exp` error model fails when scores underflow subnormal/FTZ thresholds. Consider $n = 1025$ keys with $z_1 = 0, v_1 = 0$ and $z_j = -104.5, v_j = 2^{60}$ ($j \ge 2$). IEEE `fp32` flushes $e^{-104.5} \approx 2^{-150.8}$ to $0$, so a compliant FlashAttention kernel outputs $\hat o = 0$, whereas the exact mathematical value is $o \approx 1024 \cdot 2^{60} e^{-104.5} \approx 2^{-80.76}$ (a normal `bf16` number), where a purely multiplicative radius $\approx 3u\cdot o \ll o$ would falsely reject the honest kernel. Retaining the additive floor $\eta_{\exp}$ ($2^{-150}$ for IEEE `fp32`, $2^{-126}$ under FTZ) across the key's own exponential, up to $n_t$ rescalings, and probability store floor $\eta_P$ yields per-key additive weight error $\eta_w = (n_t+1)(1+\epsilon_\mu)\eta_{\exp} + \eta_P$, converted to normalizer units via $\bar\eta = e^{\bar r_z}\eta_w \ge \eta_w / \sum_k e^{z_k - \hat m}$, plus accumulator store floor $n_t\eta_{\mathrm{acc}}$. On the extreme 1025-key input this contributes $\ge 2^{-80} > o$ (accepting honest IEEE execution), while on natural inputs ($|v| \le 10, n = 32\text{k}$) it stays below $2^{-104}$—confirmed empirically with **0 / 392,928** bad blocks and tightened `ATT` peak ratios of $0.688\text{–}0.754$ (`Qwen3-0.6B`) and $0.718\text{–}0.723$ (`Llama-3.2-1B`).

---

### 2.3 Observation Granularity `[Paper + Meas]`

Propagating an interval box across a single contraction (`GEMM`) stays within a small constant factor of true worst-case rounding error, whereas propagating across two contractions (`W_up` $\to$ gate $\to$ `W_down` or `QKV` $\to$ `softmax` $\to$ `O_proj`) compounds operator norms multiplicatively. On `Qwen3-0.6B` and `Llama-3.2-1B-Instruct`, rigorous spectral-norm balls at full layer boundaries reach medians of $7.2\times 10^{87}$ and $1.4\times 10^{48}$ times $\|\Delta\|$, and even on the MLP half-layer alone remain $4.8\times$ and $12.3\times$ $\|\Delta\|$.

**Corollary 6.3 (Per-GEMM Observation Rule) `[Def]`.** The trace commitment records the **actual post-quantization input operand** of every matrix multiplication and attention core, every newly produced carried state, and the residual stream; elementwise activations, normalizations, RoPE, and output casts are folded into the producing block as deterministic tails.

---

## 3. Sort-Free Integer Bucketed Keyed Sampling (`fpn-samp-bk-v3`)

### 3.1 Discrete Sampling Specification `[Def]`

Vocabulary-wide Gumbel-max with sorted top-$p$ requires one PRF call per vocabulary token plus two permutation sorts ($\approx 218$ committed cells per vocabulary entry per checked step, or $\approx 3.5\text{ G}$ cells at $V = 151{,}936, k_2 = 105$). Rule `fpn-samp-bk-v3` replaces this with a sort-free integer specification committed in `model_id` via public parameters $(T, P_{16} = \mathrm{round}(p_{\mathrm{top}}2^{16}), \beta_s = 32)$ and three static lookup tables: a non-increasing exponential table $E[j] = \mathrm{round}(2^{40}e^{-j/2^{12}})$ ($E[J] = 0$ for $j \ge J = 116{,}405$), a $2^{24}$-entry discrete Gumbel table $\mathsf G$, and a monotone fixed-point logarithm $\mathsf{LN}$. Given `fp32` logits $\hat z \in \mathbb R^V$ and non-empty allowed token set $A_t$:

1. **Logit-gap quantization:** Compute $\hat m = \max_{v \in A_t} \hat z_v$, $c = \mathrm{fl}_{64}(2^{12}/T)$, $\hat j_v = \lfloor \mathrm{fl}_{64}(\mathrm{fl}_{64}(\hat m - \hat z_v)\cdot c)\rfloor$, and integer weights $\hat w_v = E[\min(\hat j_v, J)]$ for $v \in A_t$ ($\hat w_v = 0$ for $v \notin A_t$). Only these three IEEE `binary64` operations involve floating point; steps 2–5 are exact integer arithmetic.
2. **Threshold top-$p$ nucleus (sort-free):** Let $\hat S = \sum_v \hat w_v$, $\mathrm{tgt}(\hat S) = \lceil \hat S P_{16} / 2^{16}\rceil$, $\theta^\star = \max\{\theta : \sum_{\hat w_v \ge \theta} \hat w_v \ge \mathrm{tgt}(\hat S)\}$, and $\hat N = \{v : \hat w_v \ge \theta^\star\}$ (computable on GPU via binary search + masked reductions, zero sorting).
3. **Vocabulary bucketing:** Partition $[0, V)$ into $B = \lceil V / \beta_s\rceil$ contiguous buckets of size $\beta_s = 32$ with masses $\hat W_b = \sum_{v \in b \cap \hat N} \hat w_v$.
4. **Bucket Gumbel-max:** Select $\hat b = \arg\max_{b : \hat W_b > 0}\bigl(\mathsf{LN}(\hat W_b) + \mathsf G[\mathrm{idx}_b]\bigr)$ (ties broken to smallest index), packing 16 bucket indices (24 bits each) per width-12 Poseidon2 sponge call $\mathsf{PRF}_K(\mathsf{ctx}, d_t, t, \texttt{bkt}, \lfloor b/16\rfloor)$.
5. **In-bucket inverse CDF:** Draw 48-bit uniform $u_{48}$ from $\mathsf{PRF}_K(\mathsf{ctx}, d_t, t, \texttt{inv})$, set $\tau = \lfloor u_{48}\hat W_{\hat b} / 2^{48}\rfloor$, and output the first $v \in \hat b \cap \hat N$ whose in-bucket prefix sum $\hat S_v > \tau$.

**Lemma 6.7a (Per-Token Nucleus Equivalence) `[Lean: BucketSampling.nucleus_iff_per_token]`.** With $q = 2^{16} - P_{16}$, $A(v) = \sum_{u : w_u > w_v} w_u$, and $R(v) = \sum_{u : w_u \le w_v} w_u$, for every $1 \le P_{16} \le 2^{16}$:

$$
v \in N \iff w_v > 0 \;\wedge\; q\,A(v) < P_{16}\,R(v).
$$

Chi-square goodness-of-fit tests ($40{,}000$ draws across $\beta_s \in \{1, 8, 32, V\}$, $p_{\mathrm{top}} \in \{1, 0.95, 0.9\}$, $T \in \{1.0, 0.6\}$) confirm total variation distance $\le 3.5\times 10^{-5}$ ($p = 0.43\text{–}0.95$) relative to continuous truncated softmax `[Meas: Intel i7-14700F]`. In `vLLM 0.33` with CUDA Graphs on `Qwen3-0.6B` (`RTX 4070 Ti SUPER`), `fpn-samp-bk-v3` adds $0.11\text{–}0.31\text{ ms}$ per step ($3.8\%\text{–}7.2\%$ decode overhead across batch sizes 1–32) `[Meas]`.

---

### 3.2 Witness-Free Certified Relaxation and Completeness

Let $z$ and $\mathbf r_z$ denote the reference logits and radii recomputed in-circuit by the `HEAD` block, and let $R = \max_{v \in A_t} r_{z,v}$. Because $|\hat z_v - z_v| \le R$ implies $|(\hat m - \hat z_v) - (m - z_v)| \le 2R$ (the shift $\hat m - m$ does not cancel once quantized to discrete table steps), every compliant engine satisfies $|\hat j_v - j_v| \le \Delta_{\mathrm{b}} := \lceil 2Rc\rceil + 2$, and hence $\hat w_v \in [w_v^{\mathrm{lo}}, w_v^{\mathrm{hi}}]$ for $w_v^{\mathrm{lo}} = E[\min(j_v + \Delta_{\mathrm{b}}, J)]$ and $w_v^{\mathrm{hi}} = E[\max(j_v - \Delta_{\mathrm{b}}, 0)]$ (underflow included cleanly via $E[J] = 0$).

Define the **witness-free tight nucleus envelopes** `[Def]`:

$$
\begin{aligned}
\mathrm{sure}(v) &:\quad w_v^{\mathrm{lo}} > 0 \;\wedge\; q\sum_{u \ne v :\, w_u^{\mathrm{hi}} > w_v^{\mathrm{lo}}} w_u^{\mathrm{hi}} < P_{16}\Bigl(w_v^{\mathrm{lo}} + \sum_{u \ne v :\, w_u^{\mathrm{hi}} \le w_v^{\mathrm{lo}}} w_u^{\mathrm{lo}}\Bigr),\\
\mathrm{poss}(v) &:\quad w_v^{\mathrm{hi}} > 0 \;\wedge\; q\sum_{u :\, w_u^{\mathrm{lo}} > w_v^{\mathrm{hi}}} w_u^{\mathrm{lo}} < P_{16}\sum_{u :\, w_u^{\mathrm{lo}} \le w_v^{\mathrm{hi}}} w_u^{\mathrm{hi}},
\end{aligned}
$$

setting $N^{\mathrm{lo}} = \{v \in A_t : \mathrm{sure}(v)\}$ and $N^{\mathrm{hi}} = \{v \in A_t : \mathrm{poss}(v)\}$.

*(Why scalar threshold witnesses fail: Requiring hidden scalar thresholds $\theta^{\mathrm{lo}}, \theta^{\mathrm{hi}}$ with $\sum_{w^{\mathrm{lo}} \ge \theta^{\mathrm{lo}}} w^{\mathrm{lo}} \ge p_{\mathrm{top}}\sum w^{\mathrm{hi}}$ under multiplicative slack $e^{\pm\bar\delta}$ has **no valid witness** whenever $p_{\mathrm{top}} > e^{-2\bar\delta}$—including $p_{\mathrm{top}} = 1$ for any $\bar\delta > 0$—and drops the boundary token from $N^{\mathrm{lo}}$, inflating average accepted tokens on $V = 4096, p_{\mathrm{top}} = 0.95, r = 0.1$ to $676$ vs. $2.6$ under the per-token sandwich above `[Meas]`.)*

- **Lemma 6.7$'$ (Nucleus Sandwich) `[Lean: BucketSampling.nucleus_sandwich]`:** For every engine weight vector $w^{\mathrm{lo}} \le \hat w \le w^{\mathrm{hi}}$ and $1 \le P_{16} \le 2^{16}$, $N^{\mathrm{lo}} \subseteq \hat N \subseteq N^{\mathrm{hi}}$.
- **Lemma 6.7$''$ ($j$-Prefix Monotonicity & Zero-Slack Circuit Compilation) `[Lean: BucketSampling.sure_mono_j, poss_mono_j, prefix_boundary_exists, boundary_pair_sound_complete]`:** Because $E$ is antitone, both `sure` and `poss` are downward-closed in $j_v$ on $A_t$. Pointing to boundary token pairs $(v_{\mathrm{in}}, v_{\mathrm{out}})$ and classifying vocabulary entries by $j$-order pins the indicator columns of $N^{\mathrm{lo}}$ and $N^{\mathrm{hi}}$ with **zero prover slack**.
- **Theorem 6.8 (Universal Relaxation Completeness) `[Lean: BucketSampling.bucket_sampling_complete]`:** Under `H-contract` and the range-certificate domain bound $2Rc < 2^{52}$ ($\Delta_{\mathrm{b}} \le 2^{15}$), for all $V$, $A_t \ne \varnothing$, $T > 0$, $p_{\mathrm{top}} \in (0, 1]$, and arbitrary logits (plateaus, ties, and underflow included), the honest engine's sampled token $u$ always satisfies all four circuit checks: $u \in A_t \cap N^{\mathrm{hi}}$, $\mathsf{LN}(W_{b(u)}^{\mathrm{hi}}) + \mathsf G[\mathrm{idx}_{b(u)}] \ge \max_{b : W_b^{\mathrm{lo}} > 0}(\mathsf{LN}(W_b^{\mathrm{lo}}) + \mathsf G[\mathrm{idx}_b])$, $S_{u-1}^{\mathrm{lo}}\cdot 2^{48} \le u_{48}W_{b(u)}^{\mathrm{hi}}$, and $u_{48}W_{b(u)}^{\mathrm{lo}} < S_u^{\mathrm{hi}}\cdot 2^{48}$.

Each checked step requires $13\text{–}24$ data cells per vocabulary entry ($26\text{–}49$ with blinding; $\approx 2.1\text{–}3.9\times 10^8$ data cells at $k_2 = 105, V = 151{,}936$) plus $298$ Poseidon2 permutations `[Est]`. With `fp32` logits, the within-band reachability is $(1 + 2\bar\delta)\pi_u + \bar\delta\pi_{b(u)}$ with $\bar\delta = \Delta_{\mathrm{b}}2^{-12} \approx 2R/T + 2^{-11} \approx 2.5\times 10^{-3}$ at $T=1$, published per `model_id` and temperature.

### 3.3 Componentwise Certified MoE Top-$K$ Routing `[Lean: Gadgets.approxTopK_pairwise]`

For router scores $s = W_r x$ with componentwise GEMM box radii $\mathbf r_s$, an active expert set $S$ with $|S| = K$ is accepted iff $\forall e \in S,\, \forall e' \notin S:\; s_e + r_{s,e} \ge s_{e'} - r_{s,e'}$. This pairwise band $r_{s,e} + r_{s,e'}$ is strictly tighter than uniform $2\max_e r_{s,e}$ and accepts every honest Top-$K$ selection (`approxTopK_pairwise`, `pairwiseApproxTopK_tighter`, `pairwiseApproxTopK_sum`).

---

## 4. In-Circuit Gadget Sizing `[Meas + Est]`

On `Qwen3-0.6B` (`RTX 4070 Ti SUPER`), synthesized single-block certified-tolerance gadgets match the `float64` reference scorer with **0 verdict disagreements across 23,724 real blocks** (honest, `int4`, `fp8 W8A16`, `fp8 W8A8`, and skipped layers):
- **GEMM & normalization blocks:** `N` $0.58\text{ M}$, `QKV` $2.27\text{ M}$, `OPROJ` $0.72\text{ M}$, `GU` $3.06\text{ M}$, `DOWN` $0.81\text{ M}$ committed cells ($\approx 60\text{–}70$ cells per activation element including blinding, plus $0.4\text{–}1.9\text{ M}$ shared lookup-table cells per proof).
- **Direct attention gadget (`ATT`):** $17.75\text{ M}$ cells at $T_{\mathrm{pad}} = 128$ and $66.7\text{ M}$ at $T_{\mathrm{pad}} = 512$ ($\approx 1.27\times 10^5$ cells per key, or $\approx 60$ cells per `(head, key, dim)` element due to per-element decoding and $|v_j - o|$ differences). Eliminating this per-key slope on long contexts motivates the per-request KV matrix commitment and sumcheck reduction of [Spec 10](10-trace-commitment.md).

---

## 5. Sublayer Lineage Cost Accounting (`Rule CA`) and Cheat Detection

### 5.1 Closing the Zero-Operand Loophole (`Rule CA`) `[Def + Paper]`

Any homogeneous operator (`GEMM`, linear projection, zero-value attention) satisfies its local check **with zero error and near-zero compute** when fed all-zero inputs ($W\cdot 0 = 0$). On `Qwen3-0.6B`, skipping odd layers ($38\%$ FLOPs saved) while publishing $x_{\mathrm{attn}} = x_{\mathrm{mlp}} = 0$ leaves only the tiny normalization blocks invalid, yielding raw per-block bad cost $d_C = 1.1\times 10^{-4}$ ($0.113$ with merged norm tails).

**`Rule CA` ([Spec 01](01-relation.md) §4.5, [Spec 05](05-computation-model.md) §1.6)** closes local degenerate-input trivializations without altering per-GEMM checks:
1. **Sublayer accounting units $U$:** Costs are grouped into attention sublayers $\mathsf A(p,\ell)$, MLP sublayers $\mathsf M(p,\ell)$, individual routed MoE experts, router+combine, and final-norm+head+sampling.
2. **Operator-attributed check sets:** $\mathrm{Chk}(U) = \{b : \mathrm{ops}(b) \cap U \ne \varnothing\}$ and $V_U = \max_{b \in \mathrm{Chk}(U)} V_b$, so normalization tails hosted at the end of the preceding block (`DOWN` $\to$ `N1`, `OPROJ` $\to$ `N2`) belong to the consuming sublayer's check set.
3. **Consumer key atoms & depth-$h$ lineages:** Attention core FLOPs split into per-key atoms $(p, q, \ell)$ charged to the consuming query $p$. Each atom's lineage $\mathrm{Lin}_h(a)$ comprises $h$ units down its own column residual chain plus (for key atoms) $h$ units down the producing key column $q$, with $V_a = \max_{U' \in \mathrm{Lin}_h(a)} V_{U'}$ (`[Lean: ProtocolBridge.atomBad_of_mem_lineage]`). Default depth is **$h = 3$**.

**Theorem 6.5$'$ & Proposition 6.6 `[Paper]`.** Under the empirical cost-floor hypothesis `H-approx_∞`, saving compute fraction $s$ forces $d_C^{(\infty)} \ge (s - S_{\mathrm{tol}}) / (1 - S_{\mathrm{tol}})$, and at finite depth $h$ satisfies $d_C^{(h)} \ge \frac{s - S_{\mathrm{tol}}}{(1 - S_{\mathrm{tol}})\Lambda_h}$ with $\Lambda_h \le 2\kappa_{\mathrm{layer}}\max(1, L/(h+1))$. This $\Lambda_h$ factor is tight up to constants because zeroing an entire causal prefix above layer $\ell_0$ ("prefix collapse", explicitly scoped in `S4` of [Spec 07](07-assumptions-and-lean.md)) is locally self-consistent above $\ell_0 + h$. At $h = 3$, every realistic serving cheat is caught overwhelmingly:

| Cheat Mode on `Qwen3-0.6B` (`RTX 4070 Ti SUPER`) | Compute Saved $s$ | Raw Block $d_C$ (Sep / Merged) | **`Rule CA` ($h=3$) $d_C^{(3)}$** | Layer-Boundary Norm $d_C$ | Median Max $\|e\|/r$ on Bad Blocks |
|---|---|---|---|---|---|
| Model substitution (`Qwen2.5-0.5B`) | 0.21 | 0.824 / 0.857 | **1.000** | 1.000 | 1,380–41,962 |
| `int4` RTN g128 quantization | — | 0.681 / 0.681 | **1.000** | 0.759 | 293–2,126 |
| Same-weight `fp8 W8A16` | — | 0.681 / 0.681 | **1.000** | **0.033** | 55–312 |
| `fp8 W8A8` | — | 0.681 / 0.681 | **1.000** | 0.383 | 75–438 |
| Low-rank pruning ($r = d/4$) | 0.43 | 0.681 / 0.681 | **1.000** | 0.759 | 806–23,141 |
| Odd-layer skip, naive (`skipA`) | 0.38 | 0.227 / 0.227 | **0.973** | 0.380 | 51–252 |
| Odd-layer skip, zero-chain (`skipB`) | 0.38 | **0.00011** / 0.113 | **0.987** | 0.380 | 128 (`N1`, `N2`) |
| Sliding-window KV eviction ($W=16$) | 0.08–0.41 | 0.078 / 0.078 | **1.000** | **0.000** | 13,630 (`ATT`) |

---

### 5.2 Long-Context Sparse-Attention Tail Bounds (`S_tol(T)`) `[Paper + Meas]`

Dropping a subset $D$ of low-probability attention keys with total mass $m$ and conditional value mean $\bar v_D$ perturbs exact attention output by $o_{\mathrm{drop}} - o = \frac{m}{1 - m}(o - \bar v_D)$. Whenever that perturbation lies strictly inside the `bf16` attention rounding radius, no tolerance check that accepts honest `bf16` kernels can reject the drop.

On `Qwen3-4B` (`RTX 4070 Ti SUPER`, prose & CPython code corpora across 4k, 16k, and 32k tokens), an omniscient oracle that knows exact probabilities and drops the lowest-weight keys within $0.5\,r_o$ saves **12.4%–15.2%** (4k), **34.2%–39.5%** (16k), and **52.2%–55.1%** (32k) of total FLOPs when only attention output $o$ is observed. Recording FlashAttention's per-head scalar log-sum-exp (`LSE`, mode `M1`, bounding dropped probability mass to $\le 2^{-14}$) cuts this upper bound roughly in half to **5.9%–8.3%** (4k), **18.9%–23.7%** (16k), and **33.1%–36.6%** (32k). Observing `LSE` is therefore enabled by default, an optional `fp16`-attention strict tier (`B1`, $u_P = u_o = 2^{-11}$) is defined for tighter long-context bounds, and every `model_id` publishes $S_{\mathrm{tol}}(T)$ per context tier.

### 5.3 Speculative Laundering vs. Coupled Speculative Decoding `[Paper]`

Generating draft tokens on a cheap model and running one parallel verification prefill on the committed target model maps every non-compliant retained token directly into the generation violation fraction $d_G$ (checked against $\rho$). Whenever $100\%$ of retained tokens satisfy `fpn-samp-bk-v3` on the target's verification logits, the execution is mathematically identical to honest coupled speculative decoding (`[Lean: Speculative.specLoop_eq_seqGen]`, $d_G = 0$).

---

## 6. Online Observation Volume & Bandwidth `[Est]`

Under per-GEMM observation, each token and layer records $4d + d_{ff} + 2(n_q + n_{kv})d_h$ values ($6.7\text{–}7.8\times$ layer-boundary size on dense GQA models: $\approx 0.75\text{ MB/tok}$ on `Qwen3-0.6B`, $2.6\text{ MB/tok}$ on `Llama-3.1-8B`, and $12.8\text{ MB/tok}$ on an 80-layer 70B dense model; sparse MoE replaces $d_{ff}$ by $(K_e + n_s)d_e + K_e$). Moving trace hashing off the synchronous GPU serving path via asynchronous host copy and post-reply commitment ([Spec 10](10-trace-commitment.md) §2.2) avoids competing with online tensor-core prefill and decode throughput.
