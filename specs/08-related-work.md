# Spec 08: Positioning Relative to Existing Verifiable Inference Systems (`fpn-related-1.0`)

**Specification ID:** `fpn-related-1.0` | **Role:** Structural comparison across full-trace zkML (including FlashProve), sampled layerwise proofs, activation fingerprints, refereed delegation, incentive sampling, and hardware TEEs

---

## 1. The Production API Quadrant and Evaluation Dimensions

`[Def]` Verifiable inference for commercial frontier language-model APIs must simultaneously satisfy three hard boundary conditions, forming what we call the **production API quadrant**:

1. **Closed weights and hidden topology:** The provider never discloses parameter values, layer count, hidden width, Mixture-of-Experts (MoE) routing topology, or deployed numerical precision (`bf16`, `fp8`, mixed formats) to clients or third-party verifiers.
2. **Private server-side context:** Beyond the public request $x_{\mathrm{pub}}$ and visible response $y$, a single API call incorporates proprietary system prompts, tool and structured-output templates, hidden chain-of-thought (`CoT`) reasoning tokens, and encrypted multi-turn state carried across turns.
3. **Trustless non-interactive public verifiability:** Verification assumes neither an honest provider, nor external replicas holding the weights, nor trusted hardware enclaves (TEEs). Anyone holding the public commitment `model_id`, the request, the visible reply, and a proof can verify an individual response offline on an ordinary CPU.

Operating inside a live serving stack imposes two further systems requirements: **unchanged native floating-point and random-sampling execution** (never replacing deployed GPU/TPU/Trainium kernels with a deterministic fixed-point surrogate), and **bounded amortized proving overhead** (targeting $\le 5\%$ cluster overhead at $100\%$ request coverage in Stage 2).

| Approach & Representative Systems | Secret Weights & Hidden Topology | Private Context (Prompts + CoT + State) | Unchanged Native FP & Sampling | Single-Reply Public CPU Verification | Trust & Security Assumptions | Provider Overhead vs. Native Serving |
|---|---|---|---|---|---|---|
| **Full-trace zkML** (`ZKML` [CWSK24]; `zkLLM` [SLZ24]; `zkGPT` [QSL+25]) | Weights hidden; architecture public | Not modeled in existing systems | **No** (fixed-point arithmetic surrogate) | **Yes** | Standard discrete-log / pairing + ROM assumptions | $10^2\text{–}10^4\times$ inference cost |
| **Full-trace hash IOP** (`FlashProve` [RAC+26]) | Weights hidden; architecture public | No hidden prompts, CoT, or multi-turn state | **No** (4-bit integer surrogate; greedy / fixed-seed decoding) | **Yes** (17–21 s on CPU) | Hash-only ROM (batched ZK not yet formalized in Lean) | $\approx 72\text{–}94\times$ per-GPU H100 vLLM FP8 prefill |
| **Sampled fixed-point proofs** (`SLP` [LY26]) | No weights needed to verify, but **no ZK blinding**; block layout public | Context length and CoT not hidden | **No** (fixed-point canonical model, $84.8\%\text{–}84.9\%$ greedy match vs. `fp32`) | Requires external ordered challenge beacon | KZG (prototype uses local test SRS) + external beacon | Scales with sampled blocks, still requires canonical integer trace |
| **Activation fingerprints & fixed-seed likelihoods** (`TOPLOC` [ODP+25]; `DiFR` [KRR+25]; `FSSL` [RKH+25]) | **No** (verifier holds weights to re-run prefill) | **No** (verifier needs full plaintext context & seed) | **Yes** (tolerates cross-GPU FP nondeterminism) | **No** (verifier needs GPUs and reference weights) | Honest verifier holding true weights; empirical thresholds | Lightweight online logging; verification costs 1 full reference prefill |
| **Learned representation proxies** (`SVIP` [SLZJZ24]) | **No** (trained on open-weight models) | **No** (no hidden CoT or state chain) | **Yes** | **No** (verifiable only by holder of secret tag vector) | Empirical classifier accuracy (no cryptographic soundness; FNR $<5\%$, FPR $<3\%$) | $<0.01\text{ s}$ per request |
| **Optimistic refereed delegation** (`Verde` [AST+25]) | **No** (replicas & referee inspect weights and state) | **No** (all replicas see plaintext inputs) | **Restricted** (`RepOps` fixes reduction order; single-GPU `fp32` only) | Interactive two-phase bisection game | **At least 1-of-$k$ execution nodes honest** + collision-resistant hash | $\ge 2\times$ replication + $58\%\text{–}98\%$ `RepOps` slowdown |
| **Incentive-based sampling** (`PoSP / spML` [ZWT+24]; `VeriLLM` [WZS+25]) | **No** (verifiers hold weights to re-execute) | **No** (verifiers read plaintext prompts) | **No / Restricted** (`spML` requires deterministic fixed point; `VeriLLM` uses $\varepsilon$-ball) | On-chain audit trail, relies on staked verifier committee | Rational staking equilibrium under bounded Byzantine stake | Proportional to audit rate (`VeriLLM` reports $\approx 1\%$ verification cost) |
| **Hardware enclaves** (`GPU TEE` [CSZS25]) | Hidden from host OS; relies on chip-vendor isolation | Confidential inside enclave (software stack must be publicly measurable) | **Yes** | Publicly checkable hardware attestation quote | **Hardware root keys, firmware, and side-channel immunity of chip vendor** | $-2.88\%$ throughput at batch 64, **$-15.38\%$ at batch 1** on H100 |
| **Watermarks & black-box distribution tests** ([KGW+23]; `MET` [GLG25]) | Applies to black-box APIs | Unexposed, unverified | **Yes** | Watermarks test AI authorship only; `MET` tests multi-sample distributions | `MET` needs independent reference samples; blind to low-rate mixing ($\le 20\%$) | Near-zero online cost; hundreds of queries per audit |
| **FlashProve-Next** | **Yes** (weights, depth, width, MoE routing, format hidden) | **Yes** (hidden prompts, hidden CoT, acyclic multi-turn state chain) | **Yes** (unchanged native FP + keyed random sampling + local rounding contracts) | **Yes** (offline CPU verification in seconds) | Hash / ideal-permutation ROM + published empirical cheat catalog | Decoupled from layer count ($O(k)$ spot checks; Stage 2 target $\le 5\%$) |

---

## 2. Cryptographic Proofs of Inference: Full zkML, Hash IOPs, and Sampled Proofs

### 2.1 Full-Trace zkML and the Hollow-Weight Boundary

Full-trace zkML systems prove every arithmetic and non-linear gate across all layers of a neural network:
- **General circuit compilation:** `ZKML` [CWSK24] compiles TensorFlow Lite graphs into Halo2 (KZG/IPA), requiring $3{,}651.7\text{ s}$ on a 128-vCPU, 1 TB RAM host (`r6i.32xlarge`) to prove one forward pass of a distilled 81.3M-parameter GPT-2 ($28\text{ KB}$ proof, $18.7\text{ s}$ verification).
- **LLM-specialized sumcheck and lookup protocols:** `zkLLM` [SLZ24] introduces parallel lookup arguments (`tlookup`) and `zkAttn` over BLS12-381 Hyrax commitments with $2^{16}$ fixed-point scaling, proving LLaMa-2-13B at context length $2{,}048$ in $803\text{ s}$ on an A100-40GB ($188\text{ KB}$ proof, $3.95\text{ s}$ verification). `zkGPT` [QSL+25] combines GKR, Lasso, and Hyrax to prove GPT-2 in $21.8\text{ s}$ on a 32-thread CPU ($101\text{ KB}$ proof, $0.35\text{ s}$ verification).

`[Paper]` While full-trace zkML achieves weight secrecy and public verification, three structural properties prevent direct deployment behind commercial LLM APIs:
1. **Proving cost scales with total model FLOPs**, exceeding native inference by two to four orders of magnitude.
2. **Surrogate divergence:** Replacing floating-point kernels with fixed-point circuits causes honest native `bf16`/`fp8` trajectories to diverge on long contexts or near sampling ties, preventing proof generation.
3. **Hollow weights under public architectures:** `Hollow-LLM` [GLL26] demonstrates that when a zkML system **publishes architecture dimensions** while hiding weights, proving that an output came from *some* committed weights does not guarantee proportional compute: a malicious provider can commit zeroed layers or structured ghost weights embedding a small model inside a large shell. FlashProve-Next addresses this directly by **hiding architecture dimensions** (exposing only coarse public profile bounds) and separating the guarantee into two explicit layers ([Spec 01](01-relation.md), [Spec 06](06-tolerance-and-cheats.md)): the cryptographic layer binds every response to the single `model_id` published at the provider's official endpoint, while the empirical catalog published alongside `model_id` bounds within-tolerance compute savings on that committed checkpoint.

### 2.2 FlashProve and Sampled Layerwise Proofs (SLP)

`FlashProve` [RAC+26] demonstrates post-hoc hash-based zero-knowledge proofs for a 400B-parameter MoE model (Llama 4 Maverick, 17B active parameters, 128 experts, 1,024-token context) on a single NVIDIA H100. Offline commitment takes roughly $39\text{ min}$; online proving batches 24 requests through a fork-join challenge DAG to amortize one weight opening, reaching $\approx 110\text{ prefill tok/s}$ ($223\text{ s}$ for 24 requests, $18.78\text{ MB}$ proof, $17\text{–}21\text{ s}$ CPU verification), with single-request soundness and single-request zero knowledge mechanized in Lean 4.

At the same time, [RAC+26] illustrates the ceiling of **full-layer proving over a deterministic integer surrogate**:
1. **Proving overhead:** Even without activation recomputation ($1\%\text{–}1.5\%$ of prover work), per-GPU proving time is roughly **$72\times$ ($66\text{–}77\times$ typical, up to $94\times$ peak)** the per-GPU prefill time of an $8\times\text{H100}$ vLLM FP8 deployment [RAC+26, §6].
2. **Integer surrogate mismatch and commitment exhaustion:** Weights are converted from 4-bit quantized checkpoints into an exact integer model under greedy or fixed-seed decoding, where native floating-point serving diverges from the exact-match policy; furthermore, static masked leaf tables exhaust after $688{,}127$ batches and must be replaced under an unlinked commitment ID.
3. **Public architecture and missing API semantics:** Layer count, width, and MoE dimensions are public; batched zero knowledge is not mechanized in Lean; and hidden system prompts, hidden CoT, tool calls, and multi-turn state chains are outside scope.

`Sampled Layerwise Proofs` (`SLP`) [LY26] seals polynomial commitments to all $2L+3$ layer boundaries of an $L$-layer Transformer and proves two anchor blocks plus a small sample of internal blocks (proving 5 of 163 blocks of Llama-2-70B in $1{,}259\text{ s}$ on a 2 TB RAM CPU server, $4.34\text{ MiB}$ proof, $46.3\text{ s}$ verification). Relative to production API requirements, `SLP` leaves four gaps: (i) **no zero-knowledge blinding** (unmasked openings leak witness/weight information and expose block structure, using an insecure local test SRS); (ii) **fixed-point canonical model mismatch** ($84.8\%\text{–}84.9\%$ greedy argmax agreement with `fp32` across 334,705 WikiText-2 positions); (iii) **reliance on an external challenge beacon** because local Fiat–Shamir grinding of its boundary manifest takes $12.5\text{ ms}$ and only $s=3$ internal blocks are sampled ($1.86\%$ single-bad-block hit rate); and (iv) **no private API context or random sampling**.

---

## 3. Activation Fingerprints and Fixed-Seed Re-Execution

A second line of work records compact activation sketches or synchronizes sampling seeds so that a **verifier holding the model weights** can re-run a single parallel prefill pass:
- **`TOPLOC`** [ODP+25] encodes top-$k$ ($k=128$) final hidden-state magnitudes and indices via polynomial interpolation into $258\text{ B}$ per 32 generated tokens ($\approx 1000\times$ compression on Llama-3.1-8B), tolerating cross-GPU (`A100` vs. `RTX 4090`) floating-point reordering during reference prefill verification; an engineering blog post (`SYNTHETIC-2` [PI25]) extends this approach to pipeline-parallel serving and fixed-seed Gumbel sampling.
- **`DiFR`** [KRR+25] couples shared per-request sampling seeds (`Token-DiFR`, $>98\%$ honest token match across configurations, detecting 4-bit quantization within 300 tokens at $\text{AUC}>0.999$) or random orthogonal activation projections (`Activation-DiFR`, $\text{AUC}>0.999$ in 2 tokens).
- **`FSSL`** [RKH+25] formalizes steganographic weight exfiltration by a compromised serving stack, characterizes legitimate floating-point and sampling nondeterminism, and bounds exfiltration on `Qwen3-30B-A3B` below $0.5\%$ at $10^{-2}\%$ false-positive rate using a decoupled verifier inside the provider's trust boundary.
- **`SVIP`** [SLZJZ24] trains a lightweight classifier on final hidden representations of open-weight 13B–70B models ($4\text{ KB}$ feature vector, $<0.01\text{ s}$ verification, empirical FNR $<5\%$, FPR $<3\%$).

`[Paper]` While [KRR+25, RKH+25] provide valuable empirical grounding on legitimate floating-point nondeterminism and keyed Gumbel sampling, all four systems require the verifier to **possess the true model weights, GPU compute, and full plaintext context**—an assumption that holds for internal exfiltration defense or open-weight networks, but never for an untrusted closed-weight commercial API facing public clients.

---

## 4. Refereed Delegation and Incentive-Based Sampling

Decentralized compute protocols amortize verification through replication or economic slashing:
- **`Verde`** [AST+25] dispatches each task to $k>1$ nodes and resolves disagreements via a two-level bisection game down to a single operator, backed by `RepOps`, a deterministic CUDA library that fixes floating-point reduction trees (`fp32` single-GPU only, slowing Llama-3.1-1B/8B inference by $+58\%/+98\%$ on A100 on top of $\ge 2\times$ replication).
- **`PoSP / spML`** [ZWT+24] and **`VeriLLM`** [WZS+25] audit a random fraction of tasks via staked verifiers holding the model (`spML` requires deterministic fixed-point execution; `VeriLLM` checks Merkle openings within tolerance $\varepsilon$).

`[Paper]` Neither paradigm transfers to a single closed-weight API provider: `Verde` collapses if one provider controls all replicas (**1-of-$k$ honesty**), staked re-execution exposes weights and private prompts to third-party verifiers, and `RepOps` / fixed-point determinism violate unchanged native `bf16`/`fp8` serving.

---

## 5. Hardware Enclaves, Black-Box Audits, and Checkpoint Backdoors

- **GPU Trusted Execution Environments:** [CSZS25] shows that black-box distribution tests fail when a provider substitutes a cheaper model on $\le 20\%$ of requests, and proposes combining Intel TDX / AMD SEV-SNP with NVIDIA H100 Confidential Computing ($-2.88\%$ throughput at 64-way concurrency, **$-15.38\%$ at concurrency 1** on Llama-3-8B). Enclave attestation shifts trust entirely to **hardware vendor root keys, firmware, and side-channel resistance**, requires a publicly measurable software stack, and does not generalize across non-TEE accelerators such as TPUs or Trainium.
- **Watermarks and black-box distribution testing:** Output watermarks [KGW+23] are keyed by the provider itself and can be applied equally to a cheap substitute model. `Model Equality Testing` (`MET`) [GLG25] applies two-sample kernel MMD tests across dozens of prompts, which requires an external reference oracle, cannot certify an individual API reply, and loses power under low-rate substitution [CSZS25].
- **Optimization-triggered checkpoint backdoors:** [WYL+26] shows that editing $2.44\%$ of rows in one MLP gate projection of Llama-3.2-1B can exploit `SiLU` sign sensitivity near zero so that the same weights behave benignly under eager execution while triggering a backdoor under TorchInductor or CUDA Graphs ($61.7\%\text{–}95.4\%$ attack success). This reinforces two explicit boundaries of [Spec 06](06-tolerance-and-cheats.md): (i) **per-operator rounding tolerance never implies token-for-token output equality across distinct compliant backends**; and (ii) FlashProve-Next commits directly to the **deployed serving configuration** rather than auditing training-time supply-chain poisoning.

---

## 6. Why Hybrid Shortcuts Fail

`[Paper]` Three seemingly natural shortcuts each violate at least one production requirement:
1. **"Sampled FlashProve / fixed-point zkML" (as in `SLP` [LY26]):** Sampling reduces prover time, but as long as the underlying block relation is a fixed-point surrogate ($\approx 15\%$ greedy token mismatch on WikiText-2 in `SLP`), native floating-point serving still diverges on real requests—causing honest proof failures that leak hidden activations ([Spec 03](03-zero-knowledge.md) §4.6).
2. **"Activation fingerprints (`TOPLOC` / `DiFR`) proved inside ZK":** Checking an activation fingerprint requires running a full-model forward prefill pass over the weights. Moving that verifier prefill inside a zero-knowledge circuit to hide closed weights and private CoT immediately restores $O(|B|)$ full-model proving cost.
3. **"GPU TEE inference + ZK input privacy":** Any reliance on enclave signatures re-introduces unconditional trust in chip-vendor hardware keys and microcode while breaking cross-accelerator generality.

Combining **a non-exhausting static weight commitment, full-value boundary commitments with cost-weighted stratified spot-checking, and in-circuit local per-operator rounding contracts** is therefore structurally necessary to occupy the closed-weight, private-context, unchanged-serving quadrant.

---

## References

- **[AST+25]** A. Arun, A. St. Arnaud, A. Titov, B. Wilcox, V. Kolobarić, M. Brinkmann, O. Ersoy, B. Fielding, and J. Bonneau. *Verde: Verification via Refereed Delegation for Machine Learning Programs*. arXiv:2502.19405, 2025.
- **[CSZS25]** W. Cai, T. Shi, X. Zhao, and D. Song. *Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs*. arXiv:2504.04715, 2025.
- **[CWSK24]** B.-J. Chen, S. Waiwitlikhit, I. Stoica, and D. Kang. *ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs*. In EuroSys 2024.
- **[GLG25]** I. Gao, P. Liang, and C. Guestrin. *Model Equality Testing: Which Model Is This API Serving?* In ICLR 2025.
- **[GLL26]** C. Gong, B. Liu, and M. Li. *Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference*. In IEEE S&P 2026.
- **[KGW+23]** J. Kirchenbauer, J. Geiping, Y. Wen, J. Katz, I. Miers, and T. Goldstein. *A Watermark for Large Language Models*. In ICML 2023.
- **[KRR+25]** A. Karvonen, D. Reuter, R. Rinberg, L. Marks, A. Garriga-Alonso, and K. Warr. *DiFR: Inference Verification Despite Nondeterminism*. arXiv:2511.20621, 2025.
- **[LY26]** Y. Lim and S. Yong. *Seal, Then Sample: Sampled Layerwise Proofs for Verifiable LLM Inference from GPT-2 to 70B*. arXiv:2609.27367, 2026.
- **[ODP+25]** J. M. Ong, M. Di Ferrante, A. Pazdera, R. Garner, S. Jaghouar, M. Basra, M. Ryabinin, and J. Hagemann. *TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference*. In ICML 2025.
- **[PI25]** Prime Intellect Team. *SYNTHETIC-2: Planetary-Scale Pipeline Parallel Inference for Verified Reasoning*. Engineering blog post, June 2025 (`https://primeintellect.ai/blog/synthetic-2`).
- **[QSL+25]** W. Qu, Y. Sun, X. Liu, T. Lu, Y. Guo, K. Chen, and J. Zhang. *zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference*. In USENIX Security 2025.
- **[RAC+26]** R. Rinberg, A. Allain, A. Chiesa, M. Christ, R. Dalal, G. Fenzi, B. Mayer, and G. Weissenberg. *FlashProve: an open-source library for provable inference*. Anthropic Frontier Red Team Technical Report, Nov 2026.
- **[RKH+25]** R. Rinberg, A. Karvonen, A. Hoover, D. Reuter, and K. Warr. *Verifying LLM Inference to Detect Model Weight Exfiltration*. arXiv:2511.02620, 2025.
- **[SLZ24]** H. Sun, J. Li, and H. Zhang. *zkLLM: Zero Knowledge Proofs for Large Language Models*. In ACM CCS 2024.
- **[SLZJZ24]** Y. Sun, Y. Li, Y. Zhang, Y. Jin, and H. Zhang. *SVIP: Towards Verifiable Inference of Open-source Large Language Models*. arXiv:2410.22307, 2024.
- **[WYL+26]** Y. Wang, Y. Yang, T. Li, X. Zhang, X. Zhang, and L. Pan. *Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs*. arXiv:2605.20641, 2026.
- **[WZS+25]** K. Wang, Z. Zhao, X. Song, Z. Li, L. Xia, C. Tong, B. Shi, W. Qu, E. Yang, and L. Ai. *VeriLLM: A Lightweight Framework for Publicly Verifiable Decentralized Inference*. arXiv:2509.24257, 2025.
- **[ZWT+24]** Y. Zhang, S. Wang, S. Tan, X. Liu, C. C. Moallemi, and R. A. Popa. *Proof of Sampling: A Nash Equilibrium-Based Verification Protocol for Decentralized Systems*. arXiv:2405.00295, 2024.
