# Spec 05: Canonical Computation Model, Serving Mechanisms & API Semantics (`fpn-compmodel-1.0`)

**Specification ID:** `fpn-compmodel-1.0` | **Role:** Canonical logical operator graph, declared physical implementation templates, modern LLM architectures (`Dense`, `GQA`, `MoE`, `MLA`, `LREC`, multimodal), distributed serving optimizations, and commercial API semantics

---

## Overview and Architectural Motivation

Production closed-weight LLM serving stacks owe their throughput and latency to aggressive systems optimizations: continuous batching, chunked prefill, `PagedAttention` prefix caching, tensor/pipeline/expert parallelism (`TP`/`PP`/`EP`), prefill–decode (`P/D`) disaggregation, CUDA Graphs, operator fusion, absorbed low-rank Multi-head Latent Attention (`MLA`), chunkwise linear attention (`LREC`), and speculative decoding. Requiring a provider to disable any of these mechanisms—or to execute an integer surrogate model—would immediately disqualify verifiable inference from real-world deployment.

At the same time, physical serving optimizations rearrange floating-point reduction trees (`split-K`, ring/tree `all-reduce`, online softmax rescaling), so physical execution DAGs are **not** graph-isomorphic to a single sequential formula. `fpn-compmodel-1.0` bridges unchanged production serving engines to the cryptographic relation of [Spec 01](01-relation.md) via a two-layer architecture:

1. **Schedule-invariant canonical logical graph.** Every request unfolds a single committed logical operator graph $G_M$ on the turn's logical execution data $\Gamma = (n, \mathsf{seg}, g, N_{\rm patch}, \mathcal R, \Phi)$ into a private slot table $\mathsf{Sl}$ committed **before** spot-check challenges are drawn (§1.5, Lemma 1.1). Batch composition, `TP`/`PP`/`EP` degree, cache hits, `P/D` placement, and kernel tile sizes do not alter the logical block set, object addresses, or cost accounting (Theorem 3.1).
2. **Declared physical implementation templates.** Each logical node $\nu$ declares a family of permitted physical DAGs $\mathfrak T(\nu)$ paired with a proved rounding-closure lemma mapping its reassociated reductions (`Dot^sk`, `AttnFA`, `Reduce`, absorbed vs. expanded `MLA`) into the certified error radii of [Spec 06](06-tolerance-and-cheats.md) (§3.2, Theorem 3.2).
3. **Per-GEMM observation with producer-tail casting and consumer-lineage billing.** Every matrix multiplication and attention kernel observes its exact post-quantization operands, with `Cast`/normalization placed at the producer block's tail (§1.3). Coarse sublayer billing units with operator-ownership check sets and depth-$h$ consumer lineage (`CA`, §1.6) price zero-operand layer-skipping and KV-zeroing cheats at full consumer FLOPs.
4. **Static weight access via Hidden Column Opening (HCO).** Weights are committed once in `model_id` ([Spec 09](09-model-commitment.md)); sampled blocks evaluate weight tiles via shared-challenge multilinear sumcheck claims bound through HCO, with zero per-proof weight Merkle openings or tile-pool copies (§1.1).
5. **Coupled speculative decoding & complete commercial API semantics.** Speculative verification coupled to deterministic keyed bucketed noise produces the exact sequential target trajectory on the committed forward pass (`[Lean: Speculative.specLoop_eq_seqGen]`, §3.5), while structured JSON schemas, byte-level Aho–Corasick stop automata, tool segments, and acyclic multi-turn state anchors compile deterministically into global `Chain` constraints (§4).

---

## 1. Model Encoding, Observation Granularity, Wiring, and Slot Enumeration

### 1.1 Static Model Commitment and In-Proof Weight Access ([Spec 09](09-model-commitment.md))

`[Def]` A model configuration $M = (M_{\rm meta}, W)$ commits its weights once per release ([Spec 09](09-model-commitment.md) §4.1): deployed raw bitstrings $\mathbf W^{\rm raw}$ and canonical integer decodings $\mathbf W^{\rm cd} = \mathrm{CD}(\mathbf W^{\rm raw})$ are interleaved into Reed–Solomon codewords committed under salted $H_{\rm in}$ column Merkle roots $\mathsf C_{\rm raw}, \mathsf C_W$, paired with volatile configuration root $\mathsf C_{\rm cfg}$ inside

$$
\mathsf{model\_id} = H_{\rm in}\bigl(\texttt{dom}_{\rm mid},\;\varsigma_0,\;\mathsf C_W,\;\mathsf C_{\rm raw},\;\mathsf C_{\rm cfg}\bigr)
$$

and accompanied by the one-time setup proof $\pi_{\rm setup}$. Canonical decoding `CD` maps every deployed format (`bf16`, `fp16`, `fp32`, channel/block-scaled `fp8`, `MXFP8`/`MXFP4`) into a Signed Block-Floating-Point (`SBFP`) integer row with a **unique maximal per-row exponent** $G \in [12, 28)$ satisfying $\max |\mathrm{cd}| < 2^{22}$ (`[Lean: ArchGeneral.sbfp_shift_unique]`, preventing scale-deflation attacks; e.g., `Qwen3-0.6B` rows span $G \in [14, 25]$). A companion radius matrix $\mathbf W^r = |\mathbf W^{\rm cd}| + \kappa_n [\text{inexact decode}]$ is committed alongside $\mathbf W^{\rm cd}$ to evaluate $|W| |x|$ error radii in [Spec 06](06-tolerance-and-cheats.md).

`[Def]` **Tile partitioning and HCO instantiation:**
- Every weight tensor—dense projections, routed and shared MoE experts, router matrices, `MLA` down/up projections (plus deployed absorbed matrices, §2.2), causal conv kernels, token embeddings, LM head, multimodal ViT/projector weights, and learnable attention sink scalars—is partitioned into integer tiles of public shape $T_o \times T_i$.
- **Stable metadata in configuration rows ([Spec 09](09-model-commitment.md) §4.7):** The private tile table $\mathsf{Tt}: (\nu, e, \text{tile coords}) \mapsto (\text{tile ID}, G, \kappa)$ alongside the layer table, vector table, graph/port table, cost table, range certificate $\mathsf{RC}_M$, and tokenizer table reside in configuration rows of $\mathbf W$ under $\mathsf C_W$ (with key uniqueness certified by $\pi_{\rm setup}$). Each response proof binds these table columns via HCO evaluation claims at the shared challenge point, while $\mathsf C_{\rm cfg}$ holds only volatile parameters ($K_{\rm prf}$, hidden prompt injections) opened directly via $H_{\rm in}$.
- **Two-stage shared-point tile reduction:** Across all blocks opened in a proof (sampled atoms and their lineages, LM head, and embedding lookups), tile products over $\mathbf W^{\rm cd}$ and $\mathbf W^r$ fold at shared row challenge $\boldsymbol\xi$ and column challenge $\boldsymbol\zeta$ into hidden claims $c_j = \widetilde W_{*,t_j}(\boldsymbol\xi, \boldsymbol\zeta)$ at hidden tile IDs $t_j$, verified against $\mathsf{Tt}$ via lookup `L5'` and bound to $\mathsf C_W$ via linear combination $\mathbf u = \boldsymbol\alpha^\top \mathbf W^{\rm cd}$ and $m_q$ column openings. Signed activation lifts use `[Lean: BoundedLift.signedLift_unique_of_bounded]` and `[Lean: StdAlgOps.intCast_inj_small]`. In MoE layers, only active routed/shared expert tiles receive nonzero $\alpha_i$; under `TP`/`EP`, shards compute partial $\mathbf u$ vectors before reduction ([Spec 09](09-model-commitment.md) §4.3, §4.5).

`[Def]` Each block's single-hop local reference function $(\widehat o_b, \widehat q^{\rm out}_b, \mathbf r_b, \varrho_b) = F^\sharp_{M,b}(a_b, q^{\rm in}_b)$ (`[Lean: StdAlg.trace_exists]`, `[Lean: StdAlg.trace_unique]`) evaluates integer reference arithmetic on the exact inputs committed in $R$ without chaining across blocks.

### 1.2 Primitive Operators and Rounding-Contract Mapping

Deterministic discrete sub-computations (committed `RoPE` $\cos/\sin$ tables, causal masks, sampling PRF variates, and constrained-decoding bitmasks) are reproduced bit-for-bit without tolerance slack. Floating-point primitives map to [Spec 06](06-tolerance-and-cheats.md) §1.1 and §2.2 as follows:

| Logical Primitive | [Spec 06](06-tolerance-and-cheats.md) Contract Template | [Spec 06](06-tolerance-and-cheats.md) §2.2 Row | Verification Status |
|---|---|---|---|
| `GEMM` (incl. `TP` row split) | $\mathsf{Dot}(n; \mathcal T)$, or split-K $\mathsf{Dot}^{\rm sk}(n; S, k_p, k_r)$ + $\mathsf{Store}_k$ | Rows 1, 11 | `[Paper]` |
| `Norm` (RMSNorm / LayerNorm) | Sum-of-squares $\mathsf{Dot}(d)$ + $\mathsf{Elem}_{\rm rsqrt}$ + $2\times$ mul + $\mathsf{Store}$ | Rows 5, 6 | `[Paper]` |
| `RoPE` | Bit-exact committed trig table + $3$ roundings | Row 8 | `[Paper]` |
| `SoftmaxDot` (online softmax, split-KV, sinks) | $\mathsf{AttnFA}(n; B_c, S_{kv})$ with $n_t = \lceil n / B_c \rceil + S_{kv}$ rescalings | Row 7 | `[Paper]` (additive floor included) |
| `RecStep` (chunked linear attention / SSD) | $\mathsf{Scan}(B_c)$ | Row 9 | `[Open]` ([Spec 06](06-tolerance-and-cheats.md) §2.4) |
| `ElemWise` (`SiLU`, `GELU`, gating product, residual add) | $\mathsf{Elem}_f$, product, sum ($\eta_{\mathrm{silu}} = 2^{-120}$) | Rows 3, 4 | `[Paper]`, `[Meas]` |
| `Cast` (activation / KV quantization) | $\mathsf{Store}_k$ (with block/channel scale) | — | `[Def]` (**producer-tail only**) |
| `Reduce` (cross-device `all-reduce`) | $\mathsf{Reduce}(P; \mathcal T_r, k)$ composed via $\mathsf{Dot}^{\rm sk}$ | Row 11 | `[Paper]` |
| `Combine` (MoE weighted expert merge) | Weighted sum with accumulation depth $K_e + n_s$ | Row 10 | `[Paper]` |
| `Route` (MoE Top-$K$ / grouped routing) | Componentwise pairwise band $s_e + r_e \ge s_{e'} - r_{e'}$ | [Spec 06](06-tolerance-and-cheats.md) §3.6 | `[Lean: Gadgets.approxTopK_pairwise]` |
| `Sample` (token decoding) | Certified bucketed keyed sampling $\mathrm{Samp}^{\rm bk}_{\beta_s}$ | [Spec 06](06-tolerance-and-cheats.md) §3.1–§3.2 | `[Lean: BucketSampling.bucket_sampling_complete]` |

### 1.3 Fixed Per-GEMM Observation Granularity and Dense Block Partition

`[Def]` **Observation granularity is fixed to per-GEMM** ([Spec 06](06-tolerance-and-cheats.md) Corollary 6.3). `[Meas]` On `Qwen3-0.6B` and `Llama-3.2-1B-Instruct` (`NVIDIA RTX 4070 Ti SUPER 16 GB`), coarse full-layer worst-case norm balls exceed true sublayer updates by $\ge 5 \times 10^{10}\times$ (and half-layer MLP balls by $4.8\text{--}12.3\times$), providing negligible cheat sensitivity, whereas per-GEMM boxes achieve **`0 / 249,384`** and **`0 / 143,544`** honest bad blocks while catching $100\%$ of `int4`/`fp8` quantized GEMMs.

At sequence position $t$ and layer $\ell$, committed boundary objects are: residual $h(t,\ell)$, pre-attention normalized operand $x^{\rm attn}(t,\ell)$, post-`RoPE`/post-`Cast` $(q, k, v)(t,\ell)$, attention output $o(t,\ell)$, mid-layer residual $h^{\rm mid}(t,\ell)$, pre-MLP normalized operand $x^{\rm mlp}(t,\ell)$, gated hidden state $a(t,\ell)$, and final-layer normalized state $x^{\rm fn}(t)$—totalling $4d + d_{ff} + 2(n_q + n_{kv}) d_h$ scalar values per token per dense layer. Vocabulary logits are **never** materialized as trajectory objects; `HEAD(t)` recomputes reference logits in-circuit.

Each dense transformer layer comprises $5$ blocks (plus `EMB` and generation-only `HEAD`), placing normalizations and `Cast` at the **producer block's tail** so committed tensors equal exact downstream GEMM operands:

| Block Type | Input Objects | Internal Operators | Emitted Boundary Objects |
|---|---|---|---|
| $\mathrm{EMB}(t)$ | Token $u_t$ (global col) or multimodal $e^{\rm mm}(p)$ | Embedding tile lookup; trailing $\mathrm{N1}_0$ + `Cast` | $h(t, 0)$, $x^{\rm attn}(t, 0)$ |
| $\mathrm{QKV}(t, \ell)$ | $x^{\rm attn}(t, \ell)$ | $\mathrm{GEMM}_{qkv}$, `q/k-norm`, `RoPE`, KV `Cast` | $q(t, \ell)$, $k(t, \ell)$, $v(t, \ell)$ |
| $\mathrm{ATT}(t, \ell)$ | $q(t, \ell)$; $(k, v)(\tau, \ell)$ for $\tau \in \mathcal W(t)$ | `SoftmaxDot` (with sinks), `Cast` | $o(t, \ell)$ |
| $\mathrm{OPROJ}(t, \ell)$ | $o(t, \ell)$, $h(t, \ell)$ | $\mathrm{GEMM}_o$ (+`Reduce`), residual add; trailing $\mathrm{N2}_\ell$ + `Cast` | $h^{\rm mid}(t, \ell)$, $x^{\rm mlp}(t, \ell)$ |
| $\mathrm{GU}(t, \ell)$ | $x^{\rm mlp}(t, \ell)$ | $\mathrm{GEMM}_{gu}$, `SiLU`$\odot$, `Cast` | $a(t, \ell)$ |
| $\mathrm{DOWN}(t, \ell)$ | $a(t, \ell)$, $h^{\rm mid}(t, \ell)$ | $\mathrm{GEMM}_d$ (+`Reduce`), residual add; trailing $\mathrm{N1}_{\ell+1}$ (or final norm) + `Cast` | $h(t, \ell+1)$, $x^{\rm attn}(t, \ell+1)$ (or $x^{\rm fn}(t)$ at $\ell = L - 1$) |
| $\mathrm{HEAD}(t)$ (gen cols) | $x^{\rm fn}(t)$ | In-circuit $\mathrm{GEMM}_{\rm lm}$ producing $(z_t, \mathbf r^{\rm lm}_t)$ (engine computes `fp32` logits) | None (consumed directly in-circuit by $S_t$) |

### 1.4 Chunked Leaves, Port Table, and Value-Level Wiring

`[Def]` Building on [Spec 01](01-relation.md) §4.1's object key $\kappa = (\mathsf{seg}, t, s, \mathsf{port}, \mathsf{ver})$ and injective address map $\mathsf{Addr}$:
1. **Fixed-width $\lambda_c$-chunked leaves:** Each object's canonical serialization is sliced into chunks of public constant $\lambda_c$ field elements, with chunk $i$ committed at leaf $\ell_a = H_{\rm in}(\texttt{dom}_{\rm leaf}, a, \mathsf{salt}_a, \mathsf{Enc}(v_\kappa[i]))$ for $a = \mathsf{Addr}(\kappa, i)$. Requiring `TP` shard widths to be multiples of $\lambda_c$ lets every `TP` rank, `PP` stage, and `P/D` node hash its local slice independently without cross-GPU tensor gathering.
2. **Committed port table $\mathsf{Ports}_M(\nu)$:** Specifies for every input port its deterministic producer rule $\mathrm{src}(\nu, \mathsf{port}, t) \mapsto (\nu', \mathsf{port}', t', \mathsf{ver})$ and for every output port its shape, numerical format, and scale layout.
3. **Deterministic in-circuit wiring $\mathsf{Wire}$:** Consumer port $q$ of active slot $s_2$ computes $\mathsf{Addr}(\mathrm{in}_q(s_2)) = \mathsf{Addr}(\kappa_{\mathrm{src}})$ directly from $\mathsf{Sl}$ and $\mathsf{Ports}_M$.

**Lemma 1.3 (Single-Value Wiring Consistency)** `[Paper]`: *Outside of $H_{\rm in}$ sponge collisions bounded by $\Delta_{\rm SL}$ ([Spec 02](02-soundness.md) §2.2), any two openings of the same buffer address $a$ within a proof yield identical values, and unopened addresses extract to a single value or $\bot$ under $\mathcal E_{\rm SL}$, guaranteeing producer–consumer agreement across residual streams, single-write KV entries ($\mathsf{ver}=0$, or $\mathsf{ver}=1$ after an explicit `KVCast` node in §3.4), chunk-versioned recurrent states $S(\ell, m)$, and generation-step `HEAD` inputs.*

### 1.5 Pre-Challenge Complete Slot Table and Unified Lookup Bounds

`[Def]` Over the public slot universe $\mathcal U = \{0, \dots, U_{\rm slot}-1\}$, each row $s$ of the private slot table $\mathsf{Sl}$ (committed in $m_0$ inside $\mathsf{cm}_{\rm glob}$ prior to $\tau_{\rm samp}$) contains: validity bit $\mathsf{val}_s$, segment $\mathsf{seg}_s$, position $t_s$, graph node $\nu_s \in V(G_M)$ (encoding layer, block type, and MoE expert slot $k \in [0, K_e)$), selected expert ID $e_s$, implementation-template bit $\phi_s$, generation-column flag $g_s$, stratum tag $j_s$, attended key count $n^{\rm key}_s$, costs $(c^{\rm self}_s, c^{\rm key}_s, c_s)$, per-stratum prefix sums $P_{j,s}$, and billing-unit ID $\iota_s$. Let $\Gamma = (n, \mathsf{seg}, g, N_{\rm patch}, \mathcal R, \Phi)$ collect the turn's sequence/segment lengths, generation flags, multimodal patch count, committed MoE routing sets $\mathcal R$, and template choices $\Phi$.

`[Def]` Global `Chain` enforces seven deterministic constraints `E1`–`E7` on $\mathsf{Sl}$:
- **E1 (Monotone active prefix):** $\mathsf{val}_s \ge \mathsf{val}_{s+1}$; $N = 0 \iff \mathsf{val}_0 = 0$ (restricted to the three public empty branches of [Spec 01](01-relation.md) §6.3).
- **E2 (Topological graph stepping):** $(t_0, \nu_0)$ is the canonical start node (enumerating multimodal encoder patches first when present). Whenever $\mathsf{val}_{s+1} = 1$, $(t_{s+1}, \nu_{s+1}) = \mathsf{Next}_{G_M}(t_s, \nu_s, g_{t_s})$ steps through the fixed topological order of $G_M$ at $t_s$ (`HEAD` iff $g_{t_s} = 1$; exactly $K_e$ routed + $n_s$ shared expert slots on MoE layers; `RecStep` at chunk boundaries or $n - 1$) before advancing $t_s \mapsto t_s + 1$. When $\mathsf{val}_0 = 1$, the last active slot sits at the terminal node of $t = n - 1$.
- **E3 (Template-invariant positive cost):** $c_s = \mathsf{val}_s\bigl(c^{\rm self}(\nu_s, e_s) + n^{\rm key}_s c^{\rm key}(\nu_s)\bigr) \ge \mathsf{val}_s$ looked up from $\mathsf{Cost}_M$, with $n^{\rm key}_s = \min(t_s + 1, W_{\rm swa}(\ell)) + N_{\rm sink}$ on attention nodes and $0$ otherwise. Costs depend on logical node $(\nu_s, e_s)$, **never** on physical template bit $\phi_s$.
- **E4 (Stratification & prefix sums):** $j_s = \Pi(g_{t_s}, \mathsf{seg}_s)$ (default $\Pi_2$: generation vs. input columns); $P_{j,0} = 0$, $P_{j,s+1} = P_{j,s} + c_s \mathbf 1[j_s = j]$, $C_j = P_{j,U_{\rm slot}}$, and $C = \sum_j C_j \le C_{\max}$.
- **E5 (Unique slot lookup & in-slot atom resolution):** Given stratum sample $J \in [0, C_j)$ ([Spec 02](02-soundness.md) §4.2), lookup `L1` verifies $(s, j, P_{j,s}, c_s) \in \mathsf{Sl}$ with range check $0 \le o := J - P_{j,s} < c_s$. Disjoint coverage of $[0, C_j)$ uniquely determines $s$. If $o < c^{\rm self}_s$, the draw selects $s$'s self atom; otherwise Euclidean division $\tau = \tau_{\min}(t_s) + \lfloor (o - c^{\rm self}_s) / c^{\rm key}_s \rfloor$ resolves the exact key atom $(t_s, \tau, \ell)$ **without materializing an $O(T^2)$ key table**.
- **E6 (Canonical MoE expert ordering):** At each MoE layer and position, the $K_e$ routed expert IDs $e_s$ are strictly increasing ($e_{s_0} < \dots < e_{s_{K_e-1}}$), with score validity checked locally by `ROUTER`'s $V_b$.
- **E7 (Generation-column alignment):** A `HEAD` slot appears at position $t \iff g_t = 1 \iff t$ appears in the generation-step table.

**Lemma 1.1 (Unique Pre-Challenge Graph Enumeration)** `[Paper]`: *For fixed $(G_M, \mathsf{Cost}_M, \Pi, \mathrm{Prof})$, constraints `E1`–`E7` uniquely determine every column of $\mathsf{Sl}$, the active set $B$, all costs $(c_s, C_j, C)$, and the sample-to-atom map as a deterministic function of $\Gamma$ committed prior to $\tau_{\rm samp}$, ruling out adaptive post-challenge slot selection, bad-block deletion, good-block duplication, or cost manipulation.*

`[Def]` **Slot universe and lookup dimension table.** $U_{\rm slot} \ge N_{\rm patch,max} b_{\rm enc} + T_{\rm pad}(1 + \sum_\ell b_\ell) + N_{\max}$, with $b_\ell = 5$ (dense) or $b_\ell = 5 + 2 K_e + 2 n_s$ (MoE: `QKV`, `ATT`, `OPROJ`, `ROUTER`, routed/shared `GU`+`DOWN`, `COMBINE`). All lookup instances satisfy public `profile` bounds:

| Lookup ID | Role | Query Count $n_v$ | Table Rows $n_t$ | Tuple Width $w$ |
|---|---|---|---|---|
| `L1` Slot index | `E5` sample-to-slot | $\sum_j k_{1,j}$ | $U_{\rm slot}$ | $4$ $(s, j, P, c)$ |
| `L1'` Lineage fetch | §1.6 lineage block fetch | $\sum_j k_{1,j} B_h$ | $U_{\rm slot}$ | $4$ $(s, t, \nu, e)$ |
| `L2` Graph step | `E2` topological transition | $U_{\rm slot}$ | $2 N_{\nu,\max}$ | $4$ $(\nu, g, \nu', \Delta t)$ |
| `L3` Cost table | `E3` node/expert weights | $U_{\rm slot}$ | $N_{\nu,\max} E_{\max}$ | $4$ $(\nu, e, c^{\rm self}, c^{\rm key})$ |
| `L4` Routing | `E6` & router consistency | $U_{\rm slot}$ | $U_{\rm slot}$ | $3$ |
| `L5'` Tile table | [Spec 09](09-model-commitment.md) §4.7 HCO binding | $J_{\max} / 2$ | $n_{t,\rm Tt} \le R_{\max} B_{\max}$ | $6$ $(t_j, \nu, e, a \Vert b, G, \kappa)$ |
| `L11` Layer table | Hidden per-layer hyperparams | $\sum_j k_{1,j} B_h$ | $L_{\max}$ | $8$ ($2$ tables) |
| `L12` Vector table | Norm gains & embedding rows | Row-read slots | $\le 2^{10}$ | $4$ |
| `L7` Grammar DFA | §4.2 byte transitions | $N_{\max} b_{\rm tok} + k_2 V_{\max} \bar b$ | $256\,Q_{\max}$ | $3$ |
| `L8` Stop AC DFA | §4.3 byte stop transitions | $N_{\max} b_{\rm tok}$ | $256\,Q_{\rm ac,max}$ | $3$ |
| `L9` Tokenizer | [Spec 01](01-relation.md) §2 merges/bytes | $T_{\rm pad}$ | $V_{\max} + \text{merges}$ | $3\text{--}4$ |
| `L10` Range check | $16$-bit limb checks | Circuit-wide | $2^{16}$ | $1$ |

`[Def]` **Unified `profile` lookup envelope** (referenced by [Specs 02](02-soundness.md), [03](03-zero-knowledge.md), and [04](04-completeness.md)): Every lookup instance satisfies $n_v, n_t \le 2^{48}$ and $w \le 8$ across at most $n_L \le 16$ instances (the $10$ families above plus sampling `exp`/`log`/Gumbel tables), so $\sum (n_v + n_t) \le 2^{53}$ (`[Lean: RoundByRound.slot_table_numerator]`). Over $|\mathbb E| = p^3 > 2^{191.99}$, per-instance soundness escape is $< 9 \cdot 2^{48} / 2^{191.99} < 2^{-140} \ll 2^{-106}$ ([Spec 02](02-soundness.md) Lemma 6.1) and total completeness pole probability is $\varepsilon_{\rm pole} \le 2^{53} / 2^{191.99} < 2^{-138}$ ([Spec 04](04-completeness.md) Theorem 1).

### 1.6 Billing Units, Cost Atoms, and Consumer Accountability (`CA`)

*Motivation (`[Meas]`, `Qwen3-0.6B`, `NVIDIA RTX 4070 Ti SUPER 16 GB`):* Skipping odd layers and writing zero operands saves $s = 0.38$ of FLOPs, yet naive per-GEMM billing records only $d_C = 1.1 \times 10^{-4}$ (if norms are separate blocks) or $0.113$ (if norms sit on producer tails) because `0`-input GEMMs and attention blocks satisfy $W \cdot 0 = 0$ identically! Instantiating [Spec 01](01-relation.md) §4.5's **Rule `CA` (Consumer Accountability)** resolves this across all architectures:

| Architecture | Billing Unit $U$ (Logical Operators) | Operator-Ownership Check Set $\mathrm{Chk}(U)$ | Inter-Unit Lineage Edges |
|---|---|---|---|
| Dense Attn | $\mathsf A(t, \ell)$: $\mathrm{N1}_\ell$, $\mathrm{GEMM}_{qkv}$, `q/k-norm`, `RoPE`, `ATT`, $\mathrm{GEMM}_o$, res-add | $\mathrm{DOWN}(t, \ell-1)$ (or `EMB`), $\mathrm{QKV}, \mathrm{ATT}, \mathrm{OPROJ}(t, \ell)$ | $\mathsf M(t, \ell-1)$; key atom $(t,\tau,\ell)$ adds $\mathsf A(\tau, \ell)$ |
| Dense MLP | $\mathsf M(t, \ell)$: $\mathrm{N2}_\ell$, $\mathrm{GEMM}_{gu}$, `SiLU`$\odot$, $\mathrm{GEMM}_d$, res-add | $\mathrm{OPROJ}, \mathrm{GU}, \mathrm{DOWN}(t, \ell)$ | $\mathsf A(t, \ell)$ |
| MoE Router/Merge | $\mathsf R(t, \ell)$: $\mathrm{N2}_\ell$, router GEMM, Top-$K$, `COMBINE`, res-add | $\mathrm{OPROJ}, \mathrm{ROUTER}, \mathrm{COMBINE}(t, \ell)$ | $\mathsf A(t, \ell)$ and all $\mathsf X(t, \ell, \cdot)$ |
| MoE Expert $k$ | $\mathsf X(t, \ell, k)$: $k$-th routed/shared expert `GU`, `DOWN` | $\mathrm{GU}_k, \mathrm{DOWN}_k(t, \ell)$ | $\mathsf R(t, \ell)$ (`OPROJ` tail + `ROUTER`) |
| `MLA` Attn | $\mathsf A(t, \ell)$: adds $q$/$kv$ low-rank down/up projections $(c^{\rm KV}, k^R)$ & absorbed/expanded ops | Corresponding `MLA` blocks | Key atom adds producer unit of $(c^{\rm KV}, k^R)(\tau, \ell)$ |
| Chunked `LREC` | $\Lambda(\ell, m)$: `INPROJ`, conv, `RecStep`, gated `OUTPROJ` (atoms split per token) | `INPROJ`, `RecStep`, `OUTPROJ` | Lower unit at same column + $\Lambda(\ell, m-1)$ along state chain |
| Multimodal | Per-patch encoder sublayers; projector unit | Encoder/projector blocks | Placeholder `EMB` depends on patch projector unit |
| Output Head | $\mathsf H(t)$: final norm, LM head, sampling | $\mathrm{DOWN}(t, L-1)$, $\mathrm{HEAD}(t)$ | $\mathsf M(t, L-1)$ |

**Proposition 1.2 (Full-Cost Pricing of Zero-Chain and KV-Zeroing at $h = 0$)** `[Paper]` *(evaluated on `Qwen3-0.6B` block defect indicators)*: *Skipping layer $\ell$ via zero operands falsifies $\mathrm{DOWN}(t, \ell-1)$ ($\mathrm{N1}_\ell$ tail) and $\mathrm{OPROJ}(t, \ell)$ ($\mathrm{N2}_\ell$ tail). Because $\mathrm{DOWN}(t, \ell-1) \in \mathrm{Chk}(\mathsf A(t, \ell))$ and $\mathrm{OPROJ}(t, \ell) \in \mathrm{Chk}(\mathsf M(t, \ell))$, **both** skipped sublayers $\mathsf A(t, \ell)$ and $\mathsf M(t, \ell)$ satisfy $V_U = 1$, driving $d^{\rm unit}_C \ge s = 0.38$ even at $h = 0$. Similarly, zeroing $(k, v)(\tau, \ell)$ marks every downstream key atom $(t, \tau, \ell)$ bad at full consumer attention cost.* Default lineage depth $h = 3$ bounds per-sample verification to at most $B_h$ blocks (`Prof` constant, padded uniformly).

### 1.7 Verification of Locality Condition `P-loc`

`[Paper]` Across every block family—`EMB`, `QKV`, `OPROJ`, `GU`, `DOWN`, `ROUTER`, MoE expert blocks, `HEAD`, `ATT`, `RecStep`, `COMBINE`, and generation steps $t$—$V_b$ (and $S_t$) depends exclusively on global columns in $\mathsf{cm}_{\rm glob}$ and local boundary objects/HCO claims in $\mathrm{Ref}(b)$ (§1.3–§1.4). Opening $\bigcup_{U \in \mathrm{Lin}_h(a)} \bigcup_{b \in \mathrm{Chk}(U)} \mathrm{Ref}(b)$ in-circuit satisfies [Spec 01](01-relation.md) §4.5 Condition `(L)` (`P-loc` in [Spec 02](02-soundness.md) §6.1, `[Lean: ProtocolBridge.compiled_cs_forces_all_good]`).

---

## 2. Model Architecture Instantiations

### 2.1 Dense Transformers and Unified Attention `UATT` (`MHA` / `GQA` / `MQA` / `SWA` / Sinks)

`[Def]` Hidden layer parameters $(H_q, G_{kv}, d_h, W_{\rm swa}, N_{\rm sink})$ map query head $h \mapsto g(h) = \lfloor h G_{kv} / H_q \rfloor$ over causal/sliding window $\mathcal W(t) = \{\tau \in [\max(0, t - W_{\rm swa} + 1), t] : \mathsf{mask}(t, \tau) = 1\}$ ($W_{\rm swa} = T_{\rm pad}$ for full causal). Both **learnable sink scalars** $s_{{\rm sink},h,r}$ (accessed via HCO and augmenting the softmax denominator, as in GPT-OSS) and **fixed prefix sink positions** $\mathcal I_{\rm sink} \cup \mathcal W(t)$ are supported:

$$
o^{\rm att}_{t,h} = \frac{\sum_{\tau \in \mathcal W(t)} \exp\!\bigl(\langle q_{t,h}, k_{\tau,g(h)} \rangle / \sqrt{d_h} - m_{t,h}\bigr)\,v_{\tau,g(h)}}{\sum_{\tau \in \mathcal W(t)} \exp\!\bigl(\langle q_{t,h}, k_{\tau,g(h)} \rangle / \sqrt{d_h} - m_{t,h}\bigr) + \sum_{r} \exp(s_{{\rm sink},h,r} - m_{t,h})}.
$$

Logical graphs contain zero padding tokens; every causal row includes at least $\tau = t$, guaranteeing strictly positive denominators.

### 2.2 Multi-Head Latent Attention (`MLA`)

`[Def]` `MLA` (DeepSeek-V2/V3) projects normalized state $\tilde h_t$ into compressed latent vector $c^{\rm KV}_t = \mathrm{Norm}_{kv}(W_{DKV} \tilde h_t) \in \mathbb R^{d_c}$ and decoupled rotary key $k^R_t = \mathrm{RoPE}(W_{KR} \tilde h_t) \in \mathbb R^{d_r}$, making $(c^{\rm KV}_t, k^R_t)$ the **sole** carried-state object. Over any commutative ring, expanded (`prefill`) and matrix-absorbed (`decode`) associations are algebraically identical (`[Lean: ArchGeneral.mla_score_absorbed_eq]`, `[Lean: ArchGeneral.mla_out_absorbed_eq]`):

$$
\bigl\langle q^C_{t,h},\;W_{UK,h} c^{\rm KV}_\tau \bigr\rangle = \bigl\langle W_{UK,h}^\top q^C_{t,h},\;c^{\rm KV}_\tau \bigr\rangle, \qquad \sum_h W_{O,h}\!\left(\sum_\tau p_{t,h,\tau} W_{UV,h} c^{\rm KV}_\tau\right) = \sum_h W^{VO}_{{\rm abs},h}\!\left(\sum_\tau p_{t,h,\tau} c^{\rm KV}_\tau\right).
$$

Because floating-point rounding differs between the two associations, expanded and absorbed executions are modeled as **two declared implementation templates** selected by $\phi_s$ (§3.2) sharing the identical carried state $(c^{\rm KV}_t, k^R_t)$, boundary objects $(q, o)$, and logical cost (`E3`). Whenever a serving engine pre-computes and stores $\widetilde W^{QK}_{\rm abs} = \mathrm{fl}(W_{UQ}^\top W_{UK})$ in `bf16`/`fp8`, that deployed absorbed tensor is committed directly inside `model_id` and proved consistent with $W_{UQ}^\top W_{UK}$ within elementwise rounding once in $\pi_{\rm setup}$.

### 2.3 Sparse Mixture-of-Experts (`MoE`) and Certified Pairwise Routing

`[Def]` Each MoE layer activates $K_e$ of $E$ routed experts plus $n_s$ shared experts per token:
1. **Certified pairwise router (`ROUTER`):** Computes router logits $s$ with radius $\mathbf r_s$ and enforces the componentwise pairwise band $\forall e \in S_{t,\ell}, e' \notin S_{t,\ell} : s_e + r_{s,e} \ge s_{e'} - r_{s,e'}$ with $|S_{t,\ell}| = K_e$ (`[Lean: Gadgets.approxTopK_pairwise]`, `[Lean: Gadgets.pairwiseApproxTopK_tighter]`, generalized hierarchically to grouped top-$K$ routing).
2. **Active-only HCO & expert execution:** Each selected expert runs $\mathrm{GU}_k, \mathrm{DOWN}_k$, merged by `COMBINE` (`[Lean: ArchGeneral.ep_allreduce_eq_moe]`). Executing fewer than $K_e$ experts corrupts the missing expert slot or zeros its input, flagging `COMBINE` and all its atoms under Rule `CA`.
3. **Dropless serving requirement:** Token-dropping capacity overflows violate route consistency and must be disabled (`dropless` MoE execution) for completeness ([Spec 04](04-completeness.md)).

### 2.4 Chunkwise Linear and Gated Recurrent Attention (`LREC`: Gated DeltaNet, Mamba-2)

`[Def]` Matrix-state recurrences preceded by causal depthwise convolution ($w_{\rm conv} \le 4$) include Gated DeltaNet ($S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, $y_t = S_t q_t$) and Mamba-2 SSD ($S_t = \gamma_t S_{t-1} + b_t x_t^\top$, $y_t = S_t c_t + D x_t$). Partitioning into `INPROJ`, chunkwise `RecStep`($\ell, m$) of length $B_c$, and `OUTPROJ` commits state $S(\ell, m)$ **only at chunk boundaries**, cutting committed state bandwidth by $\ge B_c\times$ (`[Lean: LinearAttn.lrec_state_bytes_bound]`). Over exact arithmetic, chunkwise folding equals sequential token-by-token recurrence, boundary states are unique, and any state-truncation cheat leaves at least one bad chunk (`[Lean: LinearAttn.chunkRun_eq_recRun]`, `[Lean: LinearAttn.lrec_state_unique]`, `[Lean: LinearAttn.lrec_exists_bad_of_mismatch]`). *(Scope note: floating-point error propagation for chunkwise WY/UT and SSD block-matrix transforms awaits the closed-form $\mathsf{Scan}(B_c)$ template in [Spec 06](06-tolerance-and-cheats.md) §2.4.)*

### 2.5 Multimodal Encoders and Bit-Exact Preprocessing

`[Def]` Committed preprocessor $\mathsf{Pre}_M$ executes deterministic container decoding, integer/rational resampling, dynamic-resolution patch tiling, and normalization on public request media bytes, fixing $N_{\rm patch}$ in `Chain` and feeding ViT/NaViT + projector blocks billed in the input stratum.

### 2.6 Non-Circular Range Certificates Over Internal Intermediates

Bounding boundary tensors alone (`|x| \le B < F_{\max}`) does not prevent internal overflow (e.g., `fp16` sum-of-squares overflows at $|x| > 256$), and invoking `H-contract` to rule out overflow would be circular because `H-contract` presupposes absence of overflow.

`[Def]` At `model_id` publication, `model_id` commits a static range certificate $\mathsf{RC}_M$ recording order-independent exact bounds $B_\zeta$ for every boundary port and every internal intermediate $\zeta$ (products, partial sums, squares, `exp` arguments, gating products, online-softmax $(m, \ell)$ accumulators, recurrent states) and verifying $\mathsf{infl}(\zeta) \cdot B_\zeta < F_{\max}(\mathsf{dt}(\zeta))$ in $\zeta$'s actual storage format $\mathsf{dt}(\zeta)$ (plus positive lower bounds such as $\sqrt{\epsilon} > 0$).

**Proposition 2.1 (Acyclic Topological Range Induction)** `[Paper]`: *Given $\mathsf{RC}_M$ and declared templates:*
1. *Along any summation DAG of depth $h(\zeta)$, order-independent triangle inequalities ($|W_{i,:}|_1 B_x$ for GEMM, $d B_x^2$ for sum-of-squares) inductively verify $|\hat\zeta| \le (1 + u)^{h(\zeta)} B_\zeta < F_{\max}(\mathsf{dt}(\zeta))$ step-by-step **prior** to invoking output rounding contracts.*
2. *Absence of internal overflow places inputs inside operator domain $D_o$ (with stabilized max-subtracted softmax enforcing `exp` arguments $\le 0$, and `SiLU` overflow-to-zero explicitly absorbed by $\eta_{\mathrm{silu}} = 2^{-120}$).*
3. *`H-contract` and `Cap` then confine block outputs within $B_{\rm obj}$, completing induction across all layers and guaranteeing overflow-free SBFP and Goldilocks integer lifts.*

---

## 3. Production Serving Optimizations: Logical Invariance & Template Closure

### 3.1 Canonical Logical Graph vs. Declared Implementation Templates

`[Def]` An **implementation template** $\mathsf T = (\mathcal D_{\mathsf T}, \Theta_{\mathsf T}, \kappa_{\mathsf T}, D_{\mathsf T}, \mathsf L_{\mathsf T})$ specifies:
- $\mathcal D_{\mathsf T}$: Permitted intra-block physical computation DAGs (e.g., partitioning $K$ into $S \in \mathcal S$ contiguous segments accumulated in `fp32`, stored in format $k_p$, and reduced via tree $\mathcal T_r$);
- $\Theta_{\mathsf T}$: Allowed tuning parameters ($S_{\max}, B_c, S_{kv}$, tile shapes, reduction topologies);
- $\kappa_{\mathsf T}$: Certified error-radius formula from [Spec 06](06-tolerance-and-cheats.md) (evaluated in-circuit at $\max_{\theta \in \Theta_{\mathsf T}}$);
- $D_{\mathsf T}$: Input domain (including §2.6 range bounds and **batch-row independence**: row $i$'s output depends strictly on row $i$'s inputs);
- $\mathsf L_{\mathsf T}$: **Closure lemma** proving that every $(D, \theta) \in \mathcal D_{\mathsf T} \times \Theta_{\mathsf T}$ under `H-contract` lands within $\kappa_{\mathsf T}$'s radius.

Each node $\nu$ commits its allowed template family $\mathfrak T(\nu)$ inside $M.\mathsf{Tol}$ ($\preceq \mathcal C_{\max}$). Fused kernels must materialize committed boundary objects to HBM/host capture buffers rather than eliding observation boundaries.

### 3.2 Schedule Invariance and Mechanism Closure Matrix

**Theorem 3.1 (Schedule Invariance)** `[Paper]`: *Because $\mathsf{Addr}$, $\mathsf{Next}_{G_M}$, $\mathsf{Cost}_M$, and $\Pi$ take zero parameters from batch composition, `TP`/`PP`/`EP` degree, prefix-cache hits, `P/D` disaggregation, or kernel tile choices, the logical unfolding $\Lambda(\text{execution}) = (u, \Gamma, \mathsf{Sl}, \text{object keys}, G)$ is invariant across all physical schedules.*

**Theorem 3.2 (Physical Implementation Completeness)** `[Paper]`: *If every active slot executes a DAG in $\mathfrak T(\nu_s)$ whose closure lemma $\mathsf L_{\mathsf T}$ holds under `H-contract` and $\mathsf{RC}_M$ (Proposition 2.1), then every block satisfies $\mathrm{Tol}_b$, every generation step satisfies $S_t$, and the physical serving run satisfies $\mathcal R_{\mathrm{all}}$.*

| Production Mechanism | Physical Effect | Logical Graph (Thm 3.1) | Covering Template & Closure Status |
|---|---|---|---|
| Continuous batching | Dynamic batch size alters GEMM tiles & split-K | Invariant | $\mathsf{Dot}$ / $\mathsf{Dot}^{\rm sk}$ maximized over $\mathcal S$; row independence (`[Paper]`) |
| Chunked prefill (`UATT`) | Query chunking; arbitrary-order online softmax & split-KV | Invariant | $\mathsf{AttnFA}(n; B_c, S_{kv})$ with $\le \lceil n / B_c \rceil + S_{kv}$ rescalings (`[Paper]`) |
| Chunked prefill (`LREC`) | Unaligned recurrent chunk splits | Invariant when $B_c$-aligned | $\mathsf{Scan}$ composition (`[Open]`) |
| Prefix caching / `PagedAttention` | Non-contiguous physical KV pages reused across turns | Invariant (reuses original salted leaves) | Covered by producer block templates (`[Def]`, §3.4) |
| `TP` column-parallel | Shards output coordinates | Invariant | Per-coordinate $\mathsf{Dot}$ (`[Paper]`) |
| `TP` row-parallel + `all-reduce` | Splits $K$ across $P$ ranks; ring/tree reduction in format $k_p$ | Invariant (observes post-reduce tensor) | $\mathsf{Dot}^{\rm sk}(S = P) + \mathsf{Reduce}(P; \mathcal T_r, k_p)$ (`[Paper]`; `bf16` `all-reduce` must declare $k_p = \mathrm{bf16}$) |
| `PP` pipeline parallelism | Cross-stage activation transfer | Invariant (`[Lean: ArchGeneral.pp_rangeLayers_add]`) | Lossless: none; quantized transfer: producer-tail `Cast` (`[Paper]`) |
| `EP` expert parallelism | `all-to-all` dispatch & cross-rank combine | Invariant (`[Lean: ArchGeneral.ep_allreduce_eq_moe]`) | Expert $\mathsf{Store}$ + `Combine` depth $K_e + n_s$ (`[Paper]`) |
| `P/D` disaggregation | Cross-node KV transfer (lossless or quantized) | Lossless: invariant; lossy: explicit `KVCast` node ($\mathsf{ver}=1$) | Lossless: none; lossy: declared $\mathsf{Store}_k$ node (`[Def]`) |
| Speculative decoding | Parallel verification of $\gamma + 1$ draft tokens | Rejected suffix excluded from $B, G$ (§3.3) | Batch GEMM covered by $\mathsf{Dot}$; coupled sampling (`[Lean: Speculative.specLoop_eq_seqGen]`) |
| `MLA` absorbed vs. expanded | Reassociates low-rank projections | Selected by $\phi_s$; same objects & cost | Dual declared templates + committed absorbed weights (`[Paper]`, §2.2) |
| Kernel fusion & autotuning | Merges intra-block ops; varies $(B_c, S)$ | Invariant (boundary tensors materialized) | Composite block template maximized over $\Theta_{\mathsf T}$ (`[Paper]`) |

### 3.3 Coupled Speculative Decoding

`[Def]` Target verification must use **deterministic coupled verification**: given untrusted draft proposals $\tilde u_p, \dots, \tilde u_{p+\gamma-1}$ from any uncommitted draft model/heuristic, one target forward pass computes engine `fp32` logits at $p - 1, \dots, p + \gamma - 1$ (with committed inputs $x^{\rm fn}$), evaluates $u^{\rm tgt}_{p+i} = \mathsf{Samp}^{\rm bk}_{\beta_s}(\mathsf{logits}(p + i - 1), A, \xi_{\text{step}})$ using step-indexed PRF variates, accepts draft tokens up to the first mismatch $m$, and emits $u^{\rm tgt}_{p+m}$ *(randomized rejection-sampling speculative schemes that inject extra draft-dependent coins lie outside $E_{\mathrm{ctr}}$)*.

**Proposition 3.3 (Coupled Speculative Equivalence and Scope):**
1. **Single-trajectory sequential equivalence** `[Lean: Speculative.specLoop_eq_seqGen]`: *For any deterministic step map $\mathrm{tgt}(\text{prefix}) \mapsto \text{token}$—instantiated here by evaluating $\mathsf{Samp}^{\rm bk}_{\beta_s}$ on the **verification forward pass's** engine logits—multi-round speculative verification outputs the exact same token sequence $u$ as sequential step-by-step selection on those logits.*
2. **Numerical completeness** `[Paper]`: *Whenever the verification forward pass lies in $E_{\mathrm{ctr}}$, every accepted step satisfies $S_t$ against `HEAD`-recomputed reference logits and every accepted block satisfies $\mathrm{Tol}_b$.*
3. **Batch-tile variation vs. sequential decode:** Because batch-($\gamma+1$) GEMM tiling differs slightly from batch-$1$ decode tiling within `H-contract`, near-tied Gumbel scores may select a different valid token within [Spec 06](06-tolerance-and-cheats.md) §3.3's certified band ($\le (1 + 2\bar\delta)\pi_u + \bar\delta \pi_{b(u)}$); exact bitwise equality with unbatched sequential decode is neither required nor claimed.
4. **Discarded draft suffix & recurrent state rollback** `[Def]`: Rejected positions $p + m, \dots, p + \gamma - 1$ are excluded from $u, B, G$, and $R$ (preventing denominator inflation). If a speculative batch crosses an `LREC` chunk boundary $m' B_c$ containing a rejected token, the engine recomputes $S(\ell, m')$ from the accepted prefix before committing.

### 3.4 Prefix Caching: Logical vs. Physical Cost Accounting

`[Def]` Prefix-cached positions remain in $B$ at full logical cost $c_b$ ([Spec 01](01-relation.md) §4.4) and reuse their original `model_id`-scoped salted leaves, ensuring bad cached blocks never gain immunity. Logical cost $C$ governs soundness sampling denominators, whereas physical FLOP savings from legitimate cache hits are tracked separately in overhead benchmarks.

---

## 4. Commercial API Semantics (`v_cap`)

This section maps OpenAI Responses API and Anthropic Messages API semantics into deterministic $(B, G, \mathrm{Chain})$ constraints under frozen capability matrix $v_{\rm cap}$.

### 4.1 Segmented Sequences and Committed Sampling Policies

`[Def]` Logical token sequence $u = (u_0, \dots, u_{n-1})$ ($n \le T_{\rm pad}$) concatenates **input segments** (hidden prompt injections, public messages, replayed $\mathsf{hist}_{i-1}$, in-turn tool returns $\to$ input stratum, excluded from $G$) and **generated segments** (hidden `CoT`, visible text, tool-call arguments, refusals, end/stop tokens $\to$ indexed in $G$, logits in generation stratum). Each `model_id` commits a fixed public policy $\mathsf{Samp}^{\rm bk}_{\beta_s}$ ($\beta_s = 32$); per-request user-overridden temperature/top-$p$ parameters outside $v_{\rm cap}$ are rejected publicly by $\Phi_{\mathrm{pre}}$.

### 4.2 Constrained Decoding: Supported JSON Schema Dialect and Compiler Contract

Unbounded context-sensitive JSON Schema keywords (such as arbitrary-length `uniqueItems`) cannot compile into finite-state automata. `v_cap` fixes a decidable regular/bounded-stack schema dialect $\mathfrak S_v$:

| Schema Construct | `v1` Support Scope | Automaton Compilation |
|---|---|---|
| Primitive types | `object`, `array`, `string`, `number`, `integer`, `boolean`, `null` | Byte-level DFA |
| `object` | `properties`, `required`, `additionalProperties: false` in declared key order (optional keys skippable in order) | Byte-level DFA |
| `array` | Single-schema `items`, bounded `minItems`/`maxItems` ($\le A_{\max}$) | Counter DFA |
| `string` | `enum`, `const`, regular `pattern` (no backreferences/lookaround), `minLength`/`maxLength` ($\le S_{\max}$), regex `format` | Byte-level DFA |
| `number` / `integer` | JSON number grammar, bounded digit-comparison `minimum`/`maximum`, small-modulus integer `multipleOf` | Byte-level DFA |
| Boolean combinators | `anyOf`, `oneOf` with compiled state count $\le Q_{\max}$ | Product/union DFA |
| `$ref` recursion | Non-recursive `$ref` inlined; recursive `$ref` up to stack depth $\le D_{\rm stk}$ | Bounded-stack PDA (stack column + read/write consistency) |
| **Out of `v1` scope** | `uniqueItems`, `if/then/else`, `not`, `dependent*`, `patternProperties`, `contains`, `unevaluated*`, float `multipleOf` | Publicly rejected by $\Phi_{\mathrm{pre}}$ |

`[Def]` **Mask compilation contract $\mathsf{MaskCompile}_M$ (`C1`–`C7`):**
- **C1 (Bounded byte automaton):** Outputs $\mathcal A = (Q, q_0, \delta: Q \times \{0..255\} \rightharpoonup Q, \mathrm{Acc})$ (plus depth-$D_{\rm stk}$ stack) with $|Q| \le Q_{\max}$.
- **C2 (Canonical serialization language):** $L(\mathcal A)$ equals the set of canonical JSON serializations satisfying the schema under $\mathfrak S_v$'s whitespace and key-ordering rules.
- **C3 (Dead-state pruning / prefix extensibility):** All unreachable and non-coaccessible states are pruned at compile time, guaranteeing every reachable state $q \in Q$ extends to $\mathrm{Acc}$.
- **C4 (Token allow-set):** $A(q) = \{v \in \mathcal V_{\rm text} : \delta^*(q, \mathrm{bytes}(v)) \text{ is defined}\} \cup \{\mathrm{EOS} : q \in \mathrm{Acc}\}$ (handling multi-byte/partial UTF-8 tokens cleanly at the byte level while excluding control tokens).
- **C5 (Non-emptiness):** Full byte-fallback coverage implies $A(q) \neq \varnothing$ for every reachable $q$ (otherwise taking [Spec 01](01-relation.md) §5.2's deterministic failure branch).
- **C6 (Bit-exact engine mask `H-mask`):** The serving engine applies the **exact** bitmask $A(q)$ at runtime.
- **C7 (`max_tokens` truncation):** Hitting `max_tokens` halts on a pruned extensible state with `finish_reason = max_tokens` without claiming complete JSON closure.

*(Circuit note `[Est]`: Advancing $q^{\rm fsm}_t$ along the chosen token uses $\le b_{\rm tok}$ lookups in `L7`. Evaluating the full-vocabulary mask indicator $\mathbf 1[v \in A_t]$ across $k_2$ sampled constrained steps naively costs $V \bar b \approx 9 \times 10^5$ byte lookups/step; production circuits amortize this via a single vocabulary-trie traversal or verifier-evaluated precompiled state-mask MLEs.)*

### 4.3 Stop Automaton, Refusals, and Empty Responses

`[Def]` Implements [Spec 01](01-relation.md) §6.2–§6.3 via an Aho–Corasick byte automaton (`L8`, $\le Q_{\rm ac,max}$ states) paired with token-to-byte lookup `L9`—catching intra-token stop substrings (e.g., stop `"END"` inside token `"ENDx"`) missed by naive suffix matching—under the strict priority order: pre-sampling budget guard $\to$ endogenous control token $\to$ first byte stop match $e^\star$. Empty executions ($N = 0 \implies \mathsf{val} \equiv 0, C = 0$) are restricted to `max_tokens = 0`, pre-inference external classifier interception, or pre-first-token disconnect.

### 4.4 Prompt Injections, Client Tools, and In-Turn Server Tools

`[Def]` Hidden system prompts, tool instructions, and reasoning-effort templates render deterministically via $\mathsf{Render}_M$ in `Chain`. Client tool outputs bind directly to next-turn public request bytes; in-turn server tools alternate $G^{(1)} \to z_{\rm tool} \to G^{(2)} \to \dots$, committing $z_{\rm tool}$ in $R$ prior to $\tau_{\rm samp}$, matching echoed bytes in $y$, and billing tool-return prefill in the input stratum.

### 4.5 Multi-Turn Encrypted History and State Anchors

`[Def]` Implements [Spec 01](01-relation.md) §6.4's acyclic pipeline $y^{\rm core}_i \to \mathsf{hist}_i \to \mathsf{St}_i \to \mathsf{blob}_i \to \mathsf{env}_i$, mapping Anthropic `thinking.signature` / `redacted_thinking.data` and OpenAI `reasoning.encrypted_content` / `previous_response_id` into $\mathsf{hist}_i$ and $\mathsf{St}_i$.

**Proposition 4.1 (Multi-Turn Chain Integrity)** `[Paper]`: *Under $H_{\rm in}$ binding and API tenant authentication on $\mathsf{aad}_i$, forging $\widetilde{\mathsf{hist}} \neq \mathsf{hist}_i$ (even by the holder of $K_{\rm wrap}$), splicing cross-tenant anchors, or dropping/reordering `CoT` blocks requires an $H_{\rm in}$ collision, while stateful clients tracking their latest verified $\mathsf{St}_i$ detect any state rollback or fork across turns within the shared lifetime budget $Q$.*

### 4.6 Service Tiers and Explicit `v1` Exclusions

`[Def]` Scheduling-only priority tiers share `model_id`; any tier altering numerical precision, template family $\mathfrak T$ (e.g., `bf16` `all-reduce` vs. `fp32`), or speculative mode publishes a distinct `model_id`. Excluded from `v1` scope (rejected publicly by $\Phi_{\mathrm{pre}}$ or handled outside $(B, G)$): (1) `logprobs`, (2) user-supplied custom sampling temperatures/penalties, and (3) auxiliary outputs generated by separate uncommitted models (reasoning summaries, external safety classifiers, context compaction).

---

## 5. Circuit Complexity, Long-Context Scaling, and Online Hook Overhead

### 5.1 Expected Prover Work vs. Fixed-Shape Circuit Size

`[Def]` Zero knowledge mandates fixed circuit dimensions determined by worst-case block cost $\mathrm{PC}_{\max}$ rather than expected block cost:

$$
\mathbb E[\text{Prover Work}] \approx \sum_{j=1}^{|\Pi|} k_{1,j}\,\mathbb E_{a \sim c / C_j}\!\left[\sum_{U \in \mathrm{Lin}_h(a)} \sum_{b \in \mathrm{Chk}(U)} \mathrm{PC}(b)\right] + k_2\,\mathrm{PC}_{\rm samp} + \mathrm{PC}_{\rm glob},
$$

$$
\text{Fixed Circuit Cells} \approx \sum_{j=1}^{|\Pi|} k_{1,j}\,B_h\,\mathrm{PC}_{\max} + k_2\,\mathrm{PC}_{{\rm samp},\max} + \mathrm{PC}_{\rm glob}.
$$

Under HCO, $\mathrm{PC}_{\rm glob}$ includes global columns $O(U_{\rm slot})$, lookups `L1`–`L12`, HCO (`[Meas]` $7.03\text{ M}$ cells on `Qwen3-0.6B` on `NVIDIA RTX 4070 Ti SUPER 16 GB` + `Intel i7-14700F`, `[Est]` $\approx 12\text{ M}$ with radius matrix $\mathbf W^r$; $\approx 5 \times 10^7$ at `70B` dense; $\approx 10^8$ at `400B` MoE), and volatile $\mathsf C_{\rm cfg}$ opening ($\approx 0.3\text{--}0.4\text{ M}$ cells). Non-attention GEMM blocks require only $O(d_{\rm in} + d_{\rm out})$ activation cells ($\approx 60\text{--}70$ cells/element including blinding), making **long-context attention slots** $\mathrm{PC}(\mathrm{ATT}) = \Theta\bigl(T_{\rm pad}(n_q + n_{kv}) d_h\bigr)$ the dominant driver of $\mathrm{PC}_{\max}$.

#### 5.1.1 Quantifying the Long-Context In-Circuit Attention Bottleneck

For `Qwen3-0.6B` ($n_q = 16, n_{kv} = 8, d_h = 128$, $2{,}048$ `bf16` KV values/token/layer) across two compute strata ($2 \times 105 = 210$ draws) and typed lineage slots ($4$ `ATT` slots/draw at $h = 3$, i.e., **$840$ `ATT` slots/proof**):
- **In-circuit $H_{\rm in}$ KV leaf opening `[Est]`:** Packing $3$ `bf16` values/field element ($684$ elements) takes $\approx 87$ width-12 Poseidon2 permutations plus Merkle path compressions ($142$ cells/permutation) and bit-unpacking, totalling **$1.5\text{--}2.0 \times 10^4$ cells/key**.
- **In-circuit certified attention arithmetic `[Meas]`:** Measured `ATT` gadget cell counts on `Qwen3-0.6B` (`NVIDIA RTX 4070 Ti SUPER 16 GB`) are $17.75\text{ M}$ cells at $T_{\rm pad} = 128$ and $66.7\text{ M}$ cells at $T_{\rm pad} = 512$—an empirical slope of **$1.27 \times 10^5$ cells/key** ($\approx 60$ cells per `(head, key, dim)` element, dominated by elementwise K/V decoding and $\sum_j p_j |v_j - o|$ radius propagation, $6\text{--}8\times$ larger than KV hashing and identical per key on `8B` `GQA` models sharing $8 \times 128$ KV heads).

| Context Bound $T_{\rm pad}$ | In-Circuit KV Openings (`[Est]`) | Certified `ATT` Gadget (`[Meas]` slope extrapolated) | Total Cells / Proof | Naive In-Memory Footprint ($\approx 48\text{ B/cell}$) |
|---|---|---|---|---|
| $64$ (short proxy) | $0.8\text{--}1.1\text{ G}$ | $6.8\text{ G}$ | $7.6\text{--}7.9\text{ G}$ | $0.37\text{--}0.38\text{ TB}$ |
| $1{,}024$ | $13\text{--}17\text{ G}$ | $0.11\text{ T}$ | $0.12\text{--}0.13\text{ T}$ | $5.8\text{--}6.1\text{ TB}$ |
| $4{,}096$ | $52\text{--}69\text{ G}$ | $0.44\text{ T}$ | $0.49\text{--}0.51\text{ T}$ | $24\text{--}25\text{ TB}$ |
| $32{,}768$ (native max) | $0.41\text{--}0.55\text{ T}$ | $3.5\text{ T}$ | $3.9\text{--}4.0\text{ T}$ | $0.19\text{ PB}$ |

Without moving per-dimension KV operations out of the circuit, total proof size scales as $\approx 3\text{ G} + 1.2 \times 10^8 \cdot T_{\rm pad}$ cells.

#### 5.1.2 Architectural Scaling Mitigations (`D1`–`D3`, [Spec 10](10-trace-commitment.md))

1. **`D1` Public-length power-of-two bucket padding:** Because standard completed API responses disclose total input + output token counts in $\mathcal L_{\mathrm{API}}$, setting $T_{\rm pad} = 2^{\lceil \log_2 T \rceil}$ (with per-bucket proof sizes and deadlines) leaks zero new bits while avoiding max-context padding on short requests.
2. **`D2` Per-request polynomial KV commitment with HCO-style matrix–vector opening ([Spec 10](10-trace-commitment.md)):** Commit per-request $\mathbf K, \mathbf V$ as Reed–Solomon matrices (with pre-commitment fixed-point lift) and reduce $\mathbf K_\ell q$, $\mathbf V_\ell^\top p$, and Jensen quadratic radius bounds ($\sqrt{\sum_j p_j v_j^2} + |o|$) to shared-challenge sumchecks and $m_q$ column openings (`[Lean: ProtocolBridge.tc_col_esc_le]`). `[Est]` Moving per-dimension K/V decoding and $PV$ products out of the circuit collapses in-circuit `ATT` cost from $1.27 \times 10^5$ to $\approx 10^2$ cells/key ($\approx 0.34\text{ G}$ cells at $4\text{k}$ and $\approx 2.8\text{ G}$ cells at $32\text{k}$, plus $\sim 10^7$ column-opening cells).
3. **`D3` Split-KV segment boundary observation:** Materializing split-KV partial `(m_i, l_i, acc_i)` states caps each `ATT` check slot to a single segment at the cost of $O(n_{\rm seg} n_q d_h)$ extra boundary values.

### 5.2 Online Boundary Capture, Spool Footprint, and Hook Model

`[Est]` Per-token boundary volume is $V_{\rm tok} = L\bigl(4d + d_{ff} + 2(n_q + n_{kv}) d_h\bigr)$ (`0.75 MB/tok` on `Qwen3-0.6B` $\to$ `24 GB` at `32k`; `2.6 MB/tok` on `Llama-3.1-8B`; `12.8 MB/tok` on `80L` `70B`-class dense $\to$ `52 GB` at `4k`). Packing $3$ `bf16` scalars per Goldilocks element absorbs $24$ scalars per Poseidon2 call ($\approx 1.55 \times 10^4$ calls/tok on `Qwen3-0.6B`, $\approx 5.5 \times 10^4$ calls/tok on `8B`). Against measured GPU batch Poseidon2 throughput of $\approx 3 \times 10^8$ permutations/s on an `NVIDIA RTX 4070 Ti SUPER 16 GB`, synchronous on-GPU hashing during `Qwen3-0.6B` vLLM batch-32 decode ($7{,}574\text{ tok/s}$) would consume $\approx 39\%$ of the GPU (and $> 100\%$ during prefill). Consequently, to preserve zero serving throughput impact and satisfy `H-hook` ([Spec 03](03-zero-knowledge.md) §4.7), online hooks perform **asynchronous device-to-host DMA copy at reserved bandwidth**, deferring commitment construction off the inference critical path.

---

## 6. Cross-Specification Interface Verification

| Upstream Obligation | Section in Spec 05 | Verification Status |
|---|---|---|
| [Spec 01](01-relation.md): Injective $\mathsf{Addr}$, chunked leaves, version-linearized carried state | §1.4 (Lemma 1.3) | `[Def]`, `[Paper]` |
| [Spec 01](01-relation.md) / [Spec 06](06-tolerance-and-cheats.md): Producer-tail `Cast` & per-GEMM observation | §1.3, §1.4 | `[Def]`, `[Meas]` |
| [Spec 01](01-relation.md): Acyclic $y^{\rm core}$ & $\mathsf{St}_i$ multi-turn state chain | §4.5 (Proposition 4.1) | `[Def]`, `[Paper]` |
| [Spec 01](01-relation.md): Segmented rendering, pruned grammar DFA, byte-level stop automaton | §4.1–§4.3 | `[Def]` (compiler `[Open]`) |
| [Spec 01](01-relation.md) / [Spec 02](02-soundness.md): Pre-challenge slot table `E1`–`E7`, unique lookup, $(n_v, n_t, w)$ bounds | §1.5 (Lemma 1.1), `[Lean: RoundByRound.slot_table_numerator]` | `[Paper]`, `[Lean: RoundByRound.slot_table_numerator]` |
| [Spec 02](02-soundness.md): Block & lineage locality `P-loc` | §1.7, `[Lean: ProtocolBridge.compiled_cs_forces_all_good]` | `[Paper]`, `[Lean: ProtocolBridge.compiled_cs_forces_all_good]` |
| [Spec 03](03-zero-knowledge.md): Fixed-count $B_h$ lineage padding & online hook model | §1.6, §5.2 | `[Def]` |
| [Spec 04](04-completeness.md): Acyclic internal range certificate $\mathsf{RC}_M$, schedule/mechanism closure | §2.6 (Proposition 2.1), §3.2 (Theorems 3.1–3.2) | `[Paper]`, `[Lean: ArchGeneral.pp_rangeLayers_add]`, `[Lean: ArchGeneral.ep_allreduce_eq_moe]`, `[Lean: Speculative.specLoop_eq_seqGen]` (`LREC` $\mathsf{Scan}$ `[Open]`) |
| [Spec 09](09-model-commitment.md): Per-row `SBFP` shift, shared-point HCO, stable config rows | §1.1, §2.2, §2.3, `[Lean: ArchGeneral.sbfp_shift_unique]` | `[Def]`, `[Lean: ArchGeneral.sbfp_shift_unique]` |
