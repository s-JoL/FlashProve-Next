# Spec 04: Completeness, Kernel Contracts & Failure State Machine (`fpn-completeness-1.0`)

**Specification ID:** `fpn-completeness-1.0` | **Role:** Algebraic completeness, zero numerical rejections under falsifiable kernel contracts (`H-contract`), and the declarative failure-closure state machine (`R1`–`R13`)

---

## Overview and Architectural Motivation

In verifiable inference for closed-weight production APIs, completeness is simultaneously a usability requirement and a **zero-knowledge requirement**: if certain prompts cause honest GPU executions to fail proof generation due to floating-point cancellation, underflow, or prover overload, observing whether a proof arrives leaks a secret-dependent bit about hidden weights, active MoE routing, or private chain-of-thought (`CoT`) activations ([Spec 03](03-zero-knowledge.md) §4.6).

`fpn-completeness-1.0` decomposes completeness into three orthogonal guarantees so mathematical proof existence is cleanly separated from kernel numerical fidelity and cluster scheduling:

1. **Algebraic completeness ($\varepsilon_{\mathrm{alg}} \le \varepsilon_{\mathrm{pole}}^J$).** Given any execution satisfying the strict relation $\mathcal R_{\mathrm{all}}$ and $J$ proving attempts, the **only** source of algebraic failure is landing on a LogUp denominator pole ($\varepsilon_{\mathrm{pole}} < 2^{-138}$ over $\mathbb F_{p^3}$, Theorem 1). Each retry draws fresh blinding rows for $\mathsf{cm}_{\rm glob}$, refreshing every Fiat–Shamir challenge independently while keeping lifetime honest queries ($\le 2^{60}$) well within $Q = 2^{64}$.
2. **Numerical completeness ($E_{\mathrm{ctr}} \subseteq E_{\mathrm{Tol}}$, i.e., $\varepsilon_{\mathrm{num}} = 0$).** Input-independent fixed tolerances provably fail under floating-point cancellation on adversarial inputs (Proposition 2'). FlashProve-Next instead propagates input-dependent certified error envelopes from a falsifiable per-operator rounding contract `H-contract`, proving $E_{\mathrm{ctr}} \subseteq E_{\mathrm{Tol}}$ across **all** inputs including worst-case cancellation and exponential underflow prompts (Theorem 3, Corollary 3.1). On an `NVIDIA RTX 4070 Ti SUPER 16 GB`, disabling reduced-precision `bf16` split-K reductions yields **zero bad blocks across $392{,}928$ blocks** on `Qwen3-0.6B` and `Llama-3.2-1B-Instruct`, whereas PyTorch's default `bf16` split-K configuration violates the `fp32`-accumulation contract by up to $39\times$—demonstrating that `H-contract` is directly testable and must match engine flags (§4.1).
3. **On-time delivery ($\varepsilon_{\mathrm{del}} \le \phi^J$) and declarative failure closure.** Admission-coupled proving slots, fixed-shape worst-case proving latency, replicated pre-commit persistence (`H-resource`), and a 10-state endpoint state machine (`R1`–`R13`) ensure that correlated infrastructure faults surface solely through a partition-wide public outage log $\mathcal O$, making the joint failure process across arbitrarily many requests a deterministic function of $(\mathcal L, \mathcal O)$ (§5–§7).

---

## 1. Formal Definitions

`[Def]` **Admissible request domain $D_{\mathrm{Prof}}$.** $D_{\mathrm{Prof}}$ consists of all requests passing standard public API validation within the capability matrix $v_{\rm cap}$ ([Spec 01](01-relation.md), [Spec 05](05-computation-model.md) §4), including multi-turn state chains, tool calls, structured JSON schemas, stop sequences, and token budgets, with **zero restriction on prompt semantics or adversarial token patterns**. The profile padding bound satisfies $T_{\mathrm{pad}} \ge \text{max hidden injection} + \text{max API context} + \text{max output}$, eliminating hidden-length overflow rejections.

`[Def]` **Physical execution.** A physical run of serving engine $\mathcal I$ on request $x \in D_{\mathrm{Prof}}$ produces committed boundary objects, discrete routing/sampling selections, and token sequence $u$. We distinguish two sets of executions:
- **Tolerance-compliant executions $E_{\mathrm{Tol}}$:** Executions satisfying [Spec 01](01-relation.md)'s strict relation $\mathcal R_{\mathrm{all}}(X, w)$—every block satisfies $V_b = 0$ (hence every cost atom satisfies $V_a = 0$), every generation step satisfies $S_t$, and $E(X, w) \land \mathrm{Chain}(X, w)$ holds.
- **Contract-compliant executions $E_{\mathrm{ctr}}$:** Executions in which every primitive kernel operator satisfies its committed local rounding template in $\mathcal C$ ([Spec 06](06-tolerance-and-cheats.md) §1.1) on its **actual runtime inputs**, under arbitrary tile sizes, reduction orders, or declared operator fusions ([Spec 05](05-computation-model.md) §3.2). The contract $\mathcal C \preceq \mathcal C_{\max}$ is committed privately inside `model_id` and read in-circuit without public disclosure.

`[Def]` **Three completeness metrics.**
- **Algebraic completeness ($\varepsilon_{\mathrm{alg}}$):** For any $x \in D_{\mathrm{Prof}}$ and execution in $E_{\mathrm{Tol}}$, an honest prover granted up to $J$ attempts produces an accepted proof with probability $\ge 1 - \varepsilon_{\mathrm{alg}}$ over prover coins and oracle answers.
- **Numerical completeness ($\varepsilon_{\mathrm{num}} = 0$):** Exact inclusion $E_{\mathrm{ctr}} \subseteq E_{\mathrm{Tol}}$ for every $x \in D_{\mathrm{Prof}}$.
- **On-time delivery ($\varepsilon_{\mathrm{del}}$):** Outside declared public outage windows $\mathcal O$, an admitted request completes at least one of $J$ proving attempts before its release deadline with probability $\ge 1 - \varepsilon_{\mathrm{del}}$.

Total per-response proof non-delivery outside $\mathcal O$ satisfies $\varepsilon_c \le \varepsilon_{\mathrm{alg}} + \varepsilon_{\mathrm{num}} + \varepsilon_{\mathrm{del}}$ (target $\varepsilon_c \le 2^{-20}$).

`[Def]` **$\mathcal L$-measurable failures.** Every externally observable proof non-delivery must be a deterministic function of public policy $\Phi(\mathcal L_i)$, the public outage log $\mathcal O$, and the client's own actions (§5–§7), strictly separating numerical contract violations from infrastructure faults.

---

## 2. Algebraic Completeness and Range Certificates

**Theorem 1 (Algebraic Completeness)** `[Paper]`: *For every $x \in D_{\mathrm{Prof}}$ and every execution in $E_{\mathrm{Tol}}$ satisfying the model's weight range certificate (Lemma 1.1'), across $J$ retry attempts,*

$$
\varepsilon_{\mathrm{alg}} \;\le\; \varepsilon_{\mathrm{pole}}^{\,J}, \qquad \varepsilon_{\mathrm{pole}} \;\le\; \sum_{\text{LogUp instances}} \frac{n_v + n_t}{|\mathbb E|} \;<\; 2^{-138} \quad \Bigl(|\mathbb E| = p^3 > 2^{191.99},\;\sum (n_v + n_t) \le 2^{53}\Bigr).
$$

| Potential Failure Mode | Resolution | Verification Status |
|---|---|---|
| Missing or non-unique integer reference witness | Stepwise functional determinism guarantees existence and uniqueness | `[Lean: StdAlg.trace_exists]`, `[Lean: StdAlg.trace_unique]`, `[Lean: StdAlgOps.topK_exists]`, `[Lean: Gadgets.sat_existsUnique]`, `[Lean: ArchGeneral.sbfp_shift_unique]`, `[Lean: BoundedLift.signedLift_unique_of_bounded]` |
| Block-float exponent overflow, modular wraparound, or `NaN`/`Inf` | Excluded statically by weight range certificate (Lemma 1.1', [Spec 05](05-computation-model.md) §2.6) | `[Paper]` (`[Lean: BoundedLift.ffn_per_slot_product_lt]`) |
| Discrete selection outside relaxed band | Satisfied by definition in $E_{\mathrm{Tol}}$; guaranteed for $E_{\mathrm{ctr}}$ in §4.2 | `[Lean: BucketSampling.bucket_sampling_complete]`, `[Lean: Gadgets.approxTopK_pairwise]` |
| LogUp denominator pole ($\lambda = -f_\gamma(\cdot)$) | Pole cardinality $\le n_v + n_t$; non-pole completeness exact; full-attempt retry on hit | `[Lean: LogUp.pole_card_le]`, `[Lean: LogUp.logup_complete]` |
| Sumcheck, Basefold, and HCO ([Spec 09](09-model-commitment.md)) | Perfectly complete (zero honest rejection probability) | `[Paper]` |
| Sequence or slot length exceeding `Prof` | Prevented by $T_{\mathrm{pad}}$ and $U_{\mathrm{slot}}$ sizing (§1, [Spec 05](05-computation-model.md) §1.5) | `[Def]` |
| Sampling ticket PoW grinding | Disabled in baseline ($\tau_{\mathrm{pow}} = 1$, [Spec 02](02-soundness.md) §9) | `[Def]` |

`[Paper]` **Retry challenge independence.** On a retry, the prover draws fresh blinding randomness $\omega^\pi$—including fresh blinding rows inside $\mathsf{cm}_{\rm glob}$—changing the initial transcript message $m_0 = (X_0, R, \mathsf{cm}_{\rm glob})$ while keeping $R$ fixed (`R10`). By Fiat–Shamir hash chaining, every subsequent challenge is evaluated on a fresh oracle input prefix independent of earlier attempts, bounding $J$-attempt failure by $\varepsilon_{\mathrm{pole}}^J$ and skewing the released proof's challenge distribution by at most $J \varepsilon_{\mathrm{pole}}$ ([Spec 03](03-zero-knowledge.md) §4.2).

`[Def]` **Accounting honest queries in $Q$.** [Spec 02](02-soundness.md)'s lifetime budget $Q = 2^{64}$ counts **all** first-time challenge queries, including honest proving. At $r_{\mathrm{FS}} \le 2^{10}$ challenge rounds per attempt, $N_\pi \le 2^{48}$ lifetime proofs, and $J \le 4$ attempts, total honest queries satisfy $N_\pi J r_{\mathrm{FS}} \le 2^{60} \ll 2^{64}$.

**Lemma 1.1' (Input-Independent Weight Range Certificate)** `[Paper]` *(closed over internal intermediates in [Spec 05](05-computation-model.md) §2.6)*: *At `model_id` publication, a single input-independent interval propagation over the committed weights $W_*$ bounds every boundary tensor and internal intermediate across all inputs in $D_{\mathrm{Prof}}$:*
- **RMSNorm & projections:** Normalized coordinates satisfy $|y_i| \le |g_i| \sqrt{d}$ (times rounding inflation), sum of squares $\le d \max x^2$, and linear projections $\le \|W_{i,:}\|_1 \max |y|$.
- **Attention:** Pre-softmax scores satisfy $|q \cdot k| / \sqrt{d_h} \le \|q\| \|k\| / \sqrt{d_h}$ from bounded normalized inputs; max-subtracted `exp` arguments are $\le 0$ (never overflowing, with underflow absorbed by additive floor $\eta$); attention outputs lie within the rounding-inflated convex hull of values.
- **Residual stream:** $\|h_\ell\|_\infty \le \max \|\mathrm{emb}\|_\infty + \sum_{\ell' < \ell} C_{\ell'}$, growing linearly with layer count $L$.
- **Gated recurrent states:** Gated DeltaNet updates $S' = \alpha S(I - \beta k k^\top) + \beta v k^\top$ ($\alpha \in (0,1], \beta \in [0,1], \|k\|_2 = 1$) satisfy $\|S'\|_F \le \|S\|_F + \beta \|v\| \|k\|$, bounding $\|S_t\|_F \le \|S_0\|_F + t \max \|v\|$ linearly in $T_{\mathrm{pad}}$ (even orthogonal worst-case steps $\|S'\|_F^2 = \|S\|_F^2 + \|v\|_F^2$ grow at most as $\sqrt{T}$); Mamba-2 SSD states satisfy analogous linear bounds.
- **Certified radii and engine outputs:** Radii are bounded via output caps $\overline C + |\mathrm{ref}|$ ([Spec 06](06-tolerance-and-cheats.md) §2.1), keeping both reference values and contract-compliant engine values strictly inside overflow-free SBFP exponent and Goldilocks modular lift bounds (`[Lean: BoundedLift.signedLift_unique_of_bounded]`).

Domain preconditions ($\epsilon > 0$ in norms, non-empty causal attention rows containing at least the self-key, and `FTZ` absorbed by $\eta$) are verified once per `model_id` at publication; any checkpoint exceeding `Prof` format bounds cannot be published under that `Prof` (a public, per-model decision rather than a per-request failure).

---

## 3. Why Input-Independent Fixed Tolerances Fail Completeness

**Proposition 2' (Incompleteness of Fixed Relative Boxes and Norm Balls)** `[Paper]`: *Consider a sublayer's trailing projection $\Delta = W \tilde x$ where $\tilde x$ is stored in `bf16` ($u_b = 2^{-8}$) with independent input rounding perturbations $\theta_j \in [-u_b, u_b]$. The set of contract-reachable output errors is the zonotope $W\,\mathrm{diag}(\tilde x)\,[-u_b, u_b]^n$:*

1. **Coordinatewise box:** For each output coordinate $i$ individually, choosing $\theta_j = u_b\,\mathrm{sign}(W_{ij} \tilde x_j)$ realizes error $u_b (|W| |\tilde x|)_i$. Thus any coordinatewise box complete for $E_{\mathrm{ctr}}$ must satisfy $r_i \ge u_b (|W| |\tilde x|)_i$. A fixed relative box $r_i = \beta |\mathrm{ref}_i|$ fails on every coordinate where cancellation drives $|\mathrm{ref}_i| < (u_b / \beta)(|W| |\tilde x|)_i$—which occurs routinely on near-zero coordinates in every transformer block.
2. **Euclidean norm ball:** Maximum reachable $\ell_2$ error equals $u_b \|W\,\mathrm{diag}(\tilde x)\|_{\infty \to 2} \ge u_b \|W\,\mathrm{diag}(\tilde x)\|_F$. For any fixed tolerance function $f(\|\mathrm{ref}\|_2, \|\Delta\|_2)$, choosing cancelling $(W, \tilde x)$ keeps $\|W \tilde x\|_2$ bounded while making $\|W\,\mathrm{diag}(\tilde x)\|_F$ arbitrarily large, violating $f$. Hence **no input-independent norm ball is complete across all models and adversarial prompts**.
3. **Natural vs. adversarial activations:** Under benign random-sign cancellation, $\|W\,\mathrm{diag}(\tilde x)\|_{\infty \to 2} \approx 2 u_b \|\Delta\|_2 \approx 0.008 \|\Delta\|_2 \ll 0.0625 \|\Delta\|_2$ (producing empirical max ratios of $0.624$ on `Qwen3-0.6B` and $0.390$ on `Llama-3.2-1B-Instruct` across $\sim 80{,}000$ natural blocks on an `NVIDIA RTX 4070 Ti SUPER 16 GB`, `[Meas]`). Relying on natural-activation cancellation, however, leaves completeness vulnerable to adversarial cancellation prompts and batch-size-induced reduction shifts across co-scheduled tenants.

Certified input-dependent tolerances eliminate this vulnerability by computing $|W| |\tilde x|$ bounds directly inside the proof.

---

## 4. Certified Input-Dependent Tolerances and `H-contract`

### 4.1 Universal Rounding Propagation and Numerical Completeness

`[Def]` Each block's tolerance $\mathrm{Tol}_{M,b}$ propagates per-operator rounding contracts via [Spec 06](06-tolerance-and-cheats.md) §2.1–§2.2 using two structural invariants:
- **Local rounding at actual runtime inputs:** Every operator $o$ uses the envelope

$$
\bigl|\hat o(\hat x) - o(x)\bigr| \;\le\; (1 + u)\,\Lambda_o(x, \mathbf r_x) + u\,\bigl|o(x)\bigr| + \eta,
$$

where $\Lambda_o(x, \mathbf r_x)$ upper-bounds $|o(\hat x) - o(x)|$ across the entire input box $|\hat x - x| \le \mathbf r_x$ (preserving the cross-term $u \Lambda_o$ so two successive `Store` operations correctly compose to $2u + u^2$), and $\eta$ captures subnormal/underflow floors.
- **Output caps vs. error radii:** An output range bound $|\hat v| \le C$ constrains $\mathrm{Out}(b) \in \mathrm{Cap}_b$ and implies the worst-case error bound $C + |\mathrm{ref}|$ (reaching $2C$ when $\mathrm{ref} = -C$ and $\hat v = C$); $\min$ is taken only across valid error bounds.

*Mathematical necessity of additive underflow floors in attention:* Purely multiplicative `exp` error bounds fail completeness on extreme score gaps. Consider $n = 1025$ keys with one `(score = 0, value = 0)` key and $1024$ `(score = -104.5, value = 2^{60})` keys: an IEEE `fp32` kernel flushes $e^{-104.5} < 2^{-150}$ to $0$ and outputs $\hat o = 0$, whereas the exact weighted average is $\approx 2^{-80.76} > 0$, causing any purely multiplicative radius $O(u \cdot |o|)$ to reject the honest IEEE output. Including the additive underflow floor $\bar\eta\bigl(\sum_j (|v_j| + \mathbf r_{v_j}) + n |o|\bigr)$ and accumulator underflow terms in [Spec 06](06-tolerance-and-cheats.md) §2.2 Row 7 makes completeness hold unconditionally across the entire domain.

**Theorem 3 (Certified Numerical Completeness)** `[Paper]` *(sampling completeness mechanized in `[Lean: BucketSampling.bucket_sampling_complete]`; MoE routing in `[Lean: Gadgets.approxTopK_pairwise]`)*: *For every execution in $E_{\mathrm{ctr}}$ and every input in $D_{\mathrm{Prof}}$, $\mathrm{Tol}_{M,b}$ holds on every block covered by [Spec 06](06-tolerance-and-cheats.md) Theorem 6.4, and $S_t$ holds at every generation step.*

**Corollary 3.1 (Zero Honest Numerical Rejections)** `[Paper]`: *Under `H-contract` and Lemma 1.1', $\varepsilon_{\mathrm{num}} = 0$ and $\varepsilon_{\mathrm{alg}} \le \varepsilon_{\mathrm{pole}}^J$ across all of $D_{\mathrm{Prof}}$—including arbitrary adversarial prompts, continuous-batching tile shifts, tensor/expert parallelism, prefix caching, P/D disaggregation, and coupled speculative decoding—forcing $\delta_{\mathrm{num}} = 0$ in [Spec 03](03-zero-knowledge.md) Theorem 4.1.* *(Scope note: hybrid linear-attention layers awaiting the closed-form $\mathsf{Scan}$ error template in [Spec 06](06-tolerance-and-cheats.md) §2.4 remain outside Corollary 3.1 until $\mathsf{Scan}$ is finalized. Forward DAG propagation proves **sufficiency** rather than minimality, since correlated rounding variables such as $a - a = 0$ propagate conservatively as $2r$.)*

`[Meas]` **Empirical calibration and falsifiability of `H-contract` (`NVIDIA RTX 4070 Ti SUPER 16 GB`, HuggingFace `bf16` + SDPA FlashAttention-2, [Spec 06](06-tolerance-and-cheats.md) §1.3):**
- With `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` (`R32` contract), per-GEMM certified boxes achieve **`0 / 249,384` bad blocks** on `Qwen3-0.6B` (max error/radius ratio $0.994$) and **`0 / 143,544` bad blocks** on `Llama-3.2-1B-Instruct` (max ratio $0.990$), with zero operator-level violations.
- Under PyTorch's default `allow_bf16_reduced_precision_reduction = True`, long-$K$ prefill GEMMs store intermediate split-K partial sums in `bf16`, exceeding the `fp32`-accumulation radius by up to **$31\times$** (`Qwen3-0.6B`) and **$39\times$** (`Llama-3.2-1B-Instruct`)—directly falsifying the `R32` contract unless either reduced-precision split-K is disabled or the wider `Dot^sk` split-K contract ([Spec 06](06-tolerance-and-cheats.md) §1.1) is committed.
- Differential testing of `SiLU` overflow-to-zero paths exposed a $5\times$ violation at $\eta = 2^{-126}$, resolved cleanly by setting $\eta_{\mathrm{silu}} = 2^{-120}$.

### 4.2 Certified Discrete Selections (Sampling & MoE Routing)

- **Bucketed keyed sampling (`fpn-samp-bk-v3`):** Formulated over exact integer weights with discrete log-bandwidth $\lceil 2 R c \rceil + 2$ table steps and **witness-free per-token tight nucleus envelopes** $N^{\mathrm{lo}} \subseteq \hat N \subseteq N^{\mathrm{hi}}$ (`[Lean: BucketSampling.nucleus_sandwich]`, `[Lean: BucketSampling.bucket_sampling_complete]`), eliminating scalar threshold witnesses that fail to exist whenever $p_{\mathrm{top}} > e^{-2\bar\delta}$ ([Spec 06](06-tolerance-and-cheats.md) §3.2).
- **MoE Top-$K$ routing:** Checked via tight componentwise pairwise bands $s_e + r_e \ge s_{e'} - r_{e'}$ (`[Lean: Gadgets.approxTopK_pairwise]`, `[Lean: Gadgets.pairwiseApproxTopK_tighter]`, [Spec 06](06-tolerance-and-cheats.md) §3.6), accepting every honest floating-point tie-break.

### 4.3 Cheat Detection Sensitivity Under Certified Tolerances

`[Meas]` On `Qwen3-0.6B` (`NVIDIA RTX 4070 Ti SUPER 16 GB`, [Spec 06](06-tolerance-and-cheats.md) §5), per-GEMM certified boxes flag **$100\%$ of quantized `int4` and `fp8` GEMM blocks** ($d_C = 0.681$, matching their full FLOP share), whereas coarse full-layer norm balls detect only $3.3\%$ of `fp8 W8A16` blocks. Pairing per-GEMM checks with sublayer consumer-lineage billing ([Spec 01](01-relation.md) §4.5) closes zero-operand layer skipping while keeping $V_b = 0$ on all honest runs.

---

## 5. Public Policy Rejections and Declarative Failure Closure

`[Def]` Each `Prof` publishes a deterministic public policy predicate $\Phi = \Phi_{\mathrm{pre}}(x_{\mathrm{pub}}, \text{tenant history}) \lor \Phi_{\mathrm{post}}(\mathcal L_i)$ returning `policy(<code>)` purely as a function of $\mathcal L_i$ (covering requests outside $D_{\mathrm{Prof}}$, unsupported schema keywords in [Spec 05](05-computation-model.md) §4.2, or mismatched prefix claims in §7.3).

**Theorem 4 (Exact Simulation of Correlated Joint Failure Processes)** `[Paper]`: *Decompose request $i$'s proof non-delivery event as*

$$
\mathsf{Fail}_i = \Phi(\mathcal L_i) \;\lor\; \mathsf{Out}\bigl(\mathcal O,\;t^{\mathrm{win}}_i\bigr) \;\lor\; \mathsf{Und}_i,
$$

*where $\mathcal O \in \mathcal L_{\mathrm{pub}}$ is the public partition outage log (§6), $t^{\mathrm{win}}_i$ is request $i$'s proving window determined by $(t^{\mathrm{base}}_i, \mathrm{Prof})$, and $\mathsf{Und}_i$ denotes an undeclared failure. Under **declarative failure closure (`R9`)**—every unrecovered infrastructure fault within $t^{\mathrm{win}}_i$ is covered by publishing a partition-wide outage window in $\mathcal O$—we have $\mathsf{Und}_i \equiv 0$, making the entire multi-request sequence of failures and retry schedules an **exact deterministic function** of $(\mathcal L, \mathcal O)$. Otherwise, simulation error is bounded by $\delta_{\mathrm{flt}} = \Pr[\exists\,\text{adversary-visible } i : \mathsf{Und}_i]$.*

Unlike independent Bernoulli failure models—which leak constant distinguishing advantage whenever a shared worker crash fails two co-scheduled requests simultaneously—declarative failure closure absorbs arbitrary cross-request fault correlations into $\mathcal O$ without exposing global request traffic.

**Proposition 5 (Public Policies Cannot Mask Numerical Incompleteness)** `[Paper]`: *Because $\Phi$ reads only $\mathcal L$ while floating-point error depends on hidden activations, requiring numerical rejection to be a function of $\mathcal L$ across adversary-chosen models $M, M'$ is equivalent to $\varepsilon_{\mathrm{num}} = 0$. Moreover, converting an online request's numerical violation into a public outage in $\mathcal O$ remains a secret-dependent side channel ([Spec 03](03-zero-knowledge.md) §4.6); numerical violations are governed strictly by `R13`.*

---

## 6. On-Time Delivery and Resource Provisioning

`[Assume]` **Premise `H-resource`:**
- **Admission-coupled slot reservation:** Every request admitted by the API reserves one proving slot sized to peak inference concurrency, so proving capacity never saturates independently of native API `429`/`529` backpressure (already in $\mathcal L_{\mathrm{API}}$).
- **Fixed-shape worst-case runtime:** Because circuit dimensions depend solely on $\mathrm{Prof}$ ([Spec 02](02-soundness.md) §4.2), worst-case single-attempt proving latency on reserved hardware satisfies $T^{\max}_{\mathrm{prove}}(\mathrm{Prof}) \le \Delta_{\rm rel}$.
- **Pre-commit replicated persistence:** Trajectory objects and salts are persisted across redundant storage replicas **before** emitting the terminal stream event (`COMMIT` in §7.1), allowing stateless workers to restart attempts on any healthy node.
- **Independent worker retries:** Outside public outage windows $\mathcal O$, single-attempt worker fault probability is $\le \phi$ independently across up to $J$ distinct workers.

**Proposition D (Bounded Delivery Failure)** `[Paper]`: *Under `H-resource`, outside public outage windows $\mathcal O$, $\varepsilon_{\mathrm{del}} \le \phi^J$ within worst-case deadline $t^{\mathrm{base}}_i + J \Delta_{\rm rel}$ (e.g., $\phi = 2^{-5}, J = 4 \implies \varepsilon_{\mathrm{del}} \le 2^{-20}$).*

`[Def]` **Public outage log $\mathcal O$.** An append-only timestamped list of outage intervals published per public serving partition (region or service tier), applying uniformly to every request in that partition and included in $\mathcal L_{\mathrm{pub}}$.

---

## 7. Failure and Delivery State Machine

### 7.1 States and Deterministic Transitions

| State | Entry Condition | Externally Observable Behavior | Outgoing Transitions |
|---|---|---|---|
| `ADMIT` | API admits request; reserves proving slot; draws $256$-bit $N_s$ & `proof_id` | First stream event emits fresh uniform $N_s$ and `proof_id` | $\to$ `STREAM` (native API rejections never enter `ADMIT`) |
| `STREAM` | Token-by-token generation & streaming | Native API stream chunks | Normal stop $\to$ `COMMIT`; client disconnect $\to$ `CUT`; engine crash $\to$ `ABORT` |
| `ABORT` | Crash during `STREAM` or prior to `COMMIT` durability | Native API stream error (identical to unhooked engine) | Terminal: proof endpoint returns `none` |
| `COMMIT` | Stop condition reached | Unobservable; fixed duration $c_{\mathrm{commit}}$ (fixed tree depth) | After replicated persistence, emit terminal stream event carrying $(R, \mathsf{St}_{\rm new}, \mathsf{blob})$ at $t^{\mathrm{end}}$ $\to$ `PROVE` |
| `CUT` | Disconnect detected; generation halted at next step boundary | None | Valid client prefix claim within $t^{\mathrm{cut}} + W_{\mathrm{claim}}$ $\to$ `PROVE` (prefix mode); timeout $\to$ terminal `none` |
| `PROVE` | Background attempt $j \in \{1,\dots,J\}$ with fresh $\omega^\pi$ on stateless worker | Unobservable; proof endpoint returns `pending` | Verified before $t^{\mathrm{base}} + j \Delta_{\rm rel}$ $\to$ `READY`; else if window overlaps $\mathcal O$, retry up to $J$ deadlines, then $\to$ `UNAVAIL` |
| `READY` | Proof passed server-side verifier gate | Unobservable; endpoint continues returning `pending` | At scheduled deadline $t^{\mathrm{rel}} = t^{\mathrm{base}} + j \Delta_{\rm rel}$ $\to$ `RELEASED` |
| `RELEASED` | Scheduled deadline $t^{\mathrm{rel}}$ reached | Endpoint returns fixed-size proof container ($S_{\mathrm{Prof}}$ bytes) | At $t^{\mathrm{base}} + W_{\mathrm{ret}}$ $\to$ `EXPIRED` |
| `UNAVAIL` | All $J$ attempt windows overlapped $\mathcal O$ | Endpoint returns uniform `unavailable` | At $t^{\mathrm{base}} + W_{\mathrm{ret}}$ $\to$ `EXPIRED` |
| `EXPIRED` | Retention window elapsed | Endpoint returns `none`; trajectory & proof erased | Terminal |

Here $t^{\mathrm{base}} = t^{\mathrm{end}}$ in full-response mode and $t^{\mathrm{base}} = t^{\mathrm{claim}}$ in truncated prefix mode. The proof endpoint accepts only `proof_id` and returns strictly one of five canonical responses: `pending`, `proof(π)`, `unavailable`, `policy(<code>)`, or `none`. Unissued, aborted, unclaimed-cut, and expired handles all return the identical `none` response, preventing state-probing via polling.

### 7.2 Operational Invariants (`R1`–`R13`)

`[Def]`
- **R1 (Public-only policy):** $\Phi$ reads strictly $\mathcal L$.
- **R2 (Secret-independent provisioning):** Resource reservation depends solely on $\mathrm{Prof}$ and public counters in $\mathcal L$; circuit geometry is fixed.
- **R3 (Quantized deadline release):** Proofs completed early wait in `READY` until exact grid deadline $t^{\mathrm{base}} + j \Delta_{\rm rel}$.
- **R4 (Unified retry schedule):** All non-$\Phi$ retries follow the identical $t^{\mathrm{base}} + j \Delta_{\rm rel}$ grid and uniform `unavailable` terminal code.
- **R5 (Server-side verification gate):** Every proof passes full local verification before transitioning to `READY`.
- **R6 (Constant-time commit before stream end):** $R$ uses fixed tree depth and is persisted across replicas prior to emitting the terminal stream event.
- **R7 (Cache-boundary alignment):** Trajectory leaf reuse across prefix-cache hits never crosses the native API's tenant/cache isolation boundary.
- **R8 (Metadata scrubbing):** Batch IDs, GPU device IDs, internal step counters, and worker IDs are never exposed.
- **R9 (Declarative resource-failure closure):** Any secret-independent infrastructure fault (worker crash, storage loss, hardware timeout) unrecovered within its deadline window is covered by publishing a partition-wide outage interval in $\mathcal O$; isolated silent single-request drops are prohibited.
- **R10 (Retry mask hygiene):** Retries of a request preserve $(X_0, R, \texttt{proof\_id})$ while drawing fresh CSPRNG masks $\omega^\pi$ (modifying $\mathsf{cm}_{\rm glob}$); VM snapshot rollback or process-fork RNG reuse is prohibited, and $\omega^\pi$ is zeroized immediately after release or abort.
- **R11 (Admission epoch binding):** Each request binds its key/model epoch $e$ at `ADMIT` and proves under $M_e$ even if `model_id` rotates mid-flight; epoch-$e$ secrets are retained until all epoch-$e$ retention windows expire.
- **R12 (Idempotency alignment):** Retransmissions bearing an identical API idempotency key replay the original $(y, N_s, \texttt{proof\_id})$; unkeyed resends enter `ADMIT` as fresh requests with fresh $N_s$.
- **R13 (Kernel-contract violation incident protocol):** Numerical tolerance failure is never an ordinary operational state under `H-contract`. If server self-check (`R5`) ever detects a numerical violation, that request fails proof delivery (incurring a bounded single-request $\delta_{\mathrm{num}}$ event), is logged as a kernel contract defect, and triggers an immediate public `model_id` update retiring or fixing the offending kernel configuration—**never** a partition outage in $\mathcal O$, which would destroy availability without erasing the leaked bit.

### 7.3 Stream Cancellations and Network Interruptions

Because a server cannot distinguish intentional client cancellation from mid-stream packet loss—and standard APIs do not disclose undelivered tokens or hidden `CoT` length on broken streams—truncated streams follow a **client-claimed prefix protocol**:
- **Zero content echo from proof endpoint:** The proof endpoint never returns response text or usage counters.
- **Single-shot client prefix claim:** Within $W_{\mathrm{claim}}$ of disconnect, the client submits $(\texttt{proof\_id}, n_{\mathrm{rcv}}, H_{\rm in}(y_{\mathrm{rcv}}))$, where $n_{\mathrm{rcv}}$ is the count of received stream events. Each `proof_id` accepts at most one claim. If $y_{\mathrm{rcv}}$ matches the server's transmitted prefix, the server enters `PROVE` in prefix mode certifying $y_{\mathrm{rcv}}$ (otherwise returning `policy(mismatch)`, which depends solely on client-held data).
- **Protected hidden prefix length:** Hidden `CoT` tokens preceding $y_{\mathrm{rcv}}$ remain fully constrained by $\mathrm{Chain}$, while their count stays private inside $T_{\mathrm{pad}}$ unless the native API already reported usage counters.
- **Empty prefix ($n_{\mathrm{rcv}} = 0$):** Disconnecting prior to the first token certifies the legal $N = 0$ empty execution branch ([Spec 01](01-relation.md) §6.3). If `COMMIT` already finished before packet loss and $v_{\rm cap}$ provides native response retrieval, the client retrieves the full response and polls `proof_id` normally.

### 7.4 Deterministic Observability Lemma

**Lemma 6 (Deterministic Envelope Observability)** `[Paper]`: *Under `R1`–`R13` and `H-resource`, the joint distribution of all externally observable envelope fields ($N_s$, `proof_id`, $R$, $\mathsf{St}_{\rm new}$, `blob`) and all proof-endpoint responses and timestamps across arbitrarily many concurrent requests is an exact deterministic function of $(\mathcal L, \mathcal O, \text{client actions}, \text{fresh independent randomness})$, up to $\delta_{\mathrm{num}} + \delta_{\mathrm{flt}}$.*

---

## 8. Scope and Future Work

1. **Attention additive-floor empirical scoring:** Re-scoring long-context empirical error/radius ratios under [Spec 06](06-tolerance-and-cheats.md) §2.2 Row 7's additive underflow floor and completing the $\mathsf{Scan}$ error template for hybrid linear-attention layers.
2. **Production engine & precision matrix:** Extending `H-contract` differential testing from HuggingFace `bf16` + FlashAttention-2 on `NVIDIA RTX 4070 Ti SUPER 16 GB` to vLLM custom kernels, `fp8`/MX formats, and multi-GPU `all-reduce` topologies.
3. **Checkpoint range-certificate sweep:** Computing concrete $\mathsf{RC}_M$ bounds ([Spec 05](05-computation-model.md) §2.6) across flagship dense and MoE checkpoints.
4. **Cluster SLA benchmarking & formal state-machine check:** Measuring empirical $(\phi, \Delta_{\rm rel}, T^{\max}_{\mathrm{prove}})$ under multi-daemon proving load and mechanizing Lemma 6's 10-state transition table in TLA+ or Lean 4.
