# Protocol Specifications (`specs/`)

English · [简体中文导读](README.zh-CN.md) · [Technical Report (PDF)](../paper/flashprove-next.pdf) · [Lean 4 Theorems](../lean/THEOREM.md)

This directory contains the ten formal specifications (`01`–`10`) of **FlashProve-Next**, defining publicly verifiable, zero-knowledge inference proofs for closed-weight production LLM APIs.

## Reading Order & Module Map

To review the protocol systematically, read in the order **what is proved → why cheating fails and secrets stay hidden → how production serving runs unchanged**:

| Spec | Identifier | Core Question Answered | Contents |
|---|---|---|---|
| [`01-relation.md`](01-relation.md) | `fpn-relation-1.0` | What statement does one API proof establish? | Public inputs, hidden witnesses, injective request canonicalization, immutable tensor wiring, stratified pools ($\Pi_2$), byte-level stops, constrained decoding, and acyclic multi-turn state anchors |
| [`02-soundness.md`](02-soundness.md) | `fpn-soundness-1.0` | Why can an adaptive grinding provider not forge a proof? | Straight-line witness extraction, round-by-round doomed-state analysis across $Q=2^{64}$ queries, $\mathbb F_{p^3}$ degree bounds, and the $k=105$ / $315$ sample-count formula |
| [`03-zero-knowledge.md`](03-zero-knowledge.md) | `fpn-zk-1.0` | Why does the proof leak nothing beyond standard API output? | Explicit leakage $\mathcal L = \mathcal L_{\rm API} \cup \mathcal L_{\rm pub}$, fixed-size containers, scheduled release deadlines, causal multi-round masking uniformity, and key rotation |
| [`04-completeness.md`](04-completeness.md) | `fpn-completeness-1.0` | Why do honest floating-point executions never fail numerically? | Algebraic LogUp pole retries, range certificates, zero numerical refusal under per-operator kernel contracts (`H-contract`, adversarial prompts included), and declarative resource-outage state machine |
| [`05-computation-model.md`](05-computation-model.md) | `fpn-compmodel-1.0` | How does the protocol keep production serving untouched? | Canonical operator graphs and implementation templates across Dense GQA, Sparse MoE, MLA, chunked linear attention, batching, prefix caching, TP/PP/EP, P/D disaggregation, coupled speculative decoding, and commercial APIs |
| [`06-tolerance-and-cheats.md`](06-tolerance-and-cheats.md) | `fpn-tolerance-1.0` | How are rounding bounds propagated and cost-saving cheats caught? | Per-operator rounding templates, per-GEMM granularity, sort-free bucketed sampling (`fpn-samp-bk-v3`), sublayer lineage accounting ($h=3$), and sparse-attention tail bounds |
| [`07-assumptions-and-lean.md`](07-assumptions-and-lean.md) | `fpn-assumptions-1.0` | What assumptions are made, and what is machine-checked in Lean 4? | Complete 30-item assumption inventory, dependency graph showing soundness relies purely on cryptography + protocol obligations, and mapping to all 19 Lean 4 modules |
| [`08-related-work.md`](08-related-work.md) | `fpn-related-1.0` | How does this compare to existing verification systems? | Comparison across full-layer zkML (ZKML, zkLLM, zkGPT, FlashProve), sampled proofs (SLP), activation fingerprints (TOPLOC, DiFR, FSSL, SVIP), refereed delegation (Verde), and GPU TEEs |
| [`09-model-commitment.md`](09-model-commitment.md) | `fpn-modelcom-1.0` | How are weights committed once without wearing out? | Static codeword exhaustion lemma, Reed–Solomon weight & radius matrices with salted column Merkle trees, single-transcript $\pi_{\rm setup}$, in-circuit hidden column & configuration openings |
| [`10-trace-commitment.md`](10-trace-commitment.md) | `fpn-tracecom-1.0` | How does long-context KV history fit in a single-machine circuit? | Pre-commitment fixed-point lift, per-request RS KV matrix commitment, hidden-selection sumcheck reduction, and power-of-two public length tiers |

---

## Evidence Tags & Shared Notation

- **Evidence tags:** Every statement carries an explicit tag: `[Def]` (definition or protocol design), `[Paper]` (proved on paper), `[Lean: Module.decl]` (machine-checked in Lean 4.29.1 + Mathlib), `[Assume]` (explicit assumption), `[Meas]` (hardware measurement), `[Est]` (analytical estimate), or `[Open]` (open direction).
- **Security budget & sampling:** Lifetime first challenge queries $Q = 2^{64}$, total first hash queries $Q_{\rm tot} \le 2^{80}$, target forgery bound $\varepsilon \le 2^{-40}$. At default thresholds $\sigma = \rho = 0.5$ under two-stratum partitioning $\Pi_2$ (prompt compute, generation compute, and token sampling), each pool draws $k = 105$ samples ($315$ total; `[Lean: RoundByRound.three_pool_default]`).
- **Field & hash primitives:** Goldilocks prime $p = 2^{64} - 2^{32} + 1$ with cubic challenge extension $\mathbb E = \mathbb F_{p^3} = \mathbb F_p[u]/(u^3 - 2)$ ($|\mathbb E| > 2^{191.99}$); Goldilocks width-12 Poseidon2 sponge matching the Plonky3 reference constants ($H_{\rm in}$) for in-circuit commitments and keyed sampling PRFs; independent BLAKE3 or SHA-3 ($H_{\rm out}$) for polynomial-commitment Merkle trees and Fiat–Shamir challenges.
