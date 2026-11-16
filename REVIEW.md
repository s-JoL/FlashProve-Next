# Reviewer Guide (`REVIEW.md`)

This short guide shows how to evaluate the claims of **FlashProve-Next** (`v0.2.0`, Stage 0) with minimal reading, how the paper's theorems map to the machine-checked Lean 4 files, and what lies inside vs. outside the formal proof boundary.

---

## 1. Two-Minute Orientation

- **Problem & Goal:** Let a closed-weight LLM API provider prove to any client—on an ordinary CPU in seconds—that an individual API response came from the real model configuration committed in `model_id`, while keeping production inference untouched (native `bf16`/`fp8`, random sampling, FlashAttention, CUDA Graph, batching, TP/PP/EP, speculative decoding) and revealing nothing beyond standard API outputs.
- **Stage 0 Scope:** Publishes the complete protocol design ([`paper/flashprove-next.pdf`](paper/flashprove-next.pdf)), ten formal specifications ([`specs/01`–`10`](specs/README.md)), the standalone Lean 4.29.1 + Mathlib formalization ([`lean/`](lean/THEOREM.md), 18 modules / 194 declarations), and GPU rounding-contract & cheat-detection benchmarks ([`benchmarks/certified/`](benchmarks/certified/README.md)). End-to-end prover/verifier binaries and vLLM hooks follow in Stage 1.

---

## 2. Trusted Base for the Lean 4 Formalization

1. **The Lean 4 kernel**, toolchain `leanprover/lean4:v4.29.1` ([`lean/lean-toolchain`](lean/lean-toolchain)).
2. **Mathlib** at commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6` ([`lean/lake-manifest.json`](lean/lake-manifest.json)).
3. **Standard axioms only:** `propext`, `Classical.choice`, `Quot.sound`. Every exported theorem is audited by `#print axioms` in [`lean/check.sh`](lean/check.sh), which fails the build on any `sorry` or non-standard axiom.

---

## 3. Shortest Formal Reading Path (~1,500 Lines)

You do not need to read all 18 Lean modules to verify that the top-level definitions are non-vacuous and match the paper:

1. **[`lean/FlashProveNext/RoundByRound.lean`](lean/FlashProveNext/RoundByRound.lean) & [`ProtocolBridge.lean`](lean/FlashProveNext/ProtocolBridge.lean)** — Read `ProtocolRBRSpec`, `CompiledCS`, `compiled_cs_doomed_of_bad_sample`, `hpath_of_protocol_spec`, and `fpn_protocol_lifetime_soundness`. Verify that over $Q \le 2^{64}$ lazy-sampled adaptive random-oracle queries with zero economic ("work-paid") premises, any bad-accept event occurs with probability at most $2^{-40}$ at $3 \times 105 = 315$ draws (`three_pool_default`), and check the concrete $\mathbb F_{p^3}$ and column-query escape bounds (`field_p3_bits`, `hco_col_esc_le`, `setup_col_esc_le`, `tc_col_esc_le`, `basefold_qry_esc_le`, `delta_SL_p4_default`).
2. **[`lean/FlashProveNext/BucketSampling.lean`](lean/FlashProveNext/BucketSampling.lean)** — Read `inThresholdNucleus`, `sure`, `poss`, `nucleus_sandwich`, `boundary_pair_sound_complete`, and `bucket_sampling_complete`. Verify that the witness-free integer top-$p$ envelope contains every admissible engine nucleus ($N^{\rm lo} \subseteq \hat N \subseteq N^{\rm hi}$), pins the circuit indicator column with zero prover slack, and accepts every honest bucketed Gumbel-max + inverse-CDF sample across all $T > 0$ and $p_{\rm top} \in (0, 1]$.
3. **[`lean/FlashProveNext/MaskJoint.lean`](lean/FlashProveNext/MaskJoint.lean)** — Read `IsCausal`, `evalTranscript`, `multiround_independent_causal`, and `libra_mask_bijOn_hyperplane`. Verify that prefix-measurable witness shifts induce a unique transcript per mask tuple whose pushforward under uniform masks is the exact uniform distribution, independent of the witness.
4. **[`lean/FlashProveNext/Speculative.lean`](lean/FlashProveNext/Speculative.lean), [`LinearAttn.lean`](lean/FlashProveNext/LinearAttn.lean), [`ArchGeneral.lean`](lean/FlashProveNext/ArchGeneral.lean), [`Gadgets.lean`](lean/FlashProveNext/Gadgets.lean)** — Check coupled speculative decoding (`specLoop_eq_seqGen`), chunked recurrent state uniqueness (`chunkRun_eq_recRun`, `lrec_exists_bad_of_mismatch`), absorbed MLA (`mla_score_absorbed_eq`, `mla_out_absorbed_eq`), PP/EP invariance, and pairwise MoE routing (`approxTopK_pairwise`).

Run the standalone verification in one command:

```bash
cd lean && lake exe cache get && bash check.sh
```

---

## 4. Paper-to-Lean & Evidence Map

| Paper Claim | Section / Spec | Verification Layer | Key Lean Declarations / Artifacts |
|---|---|---|---|
| **Round-by-round lifetime soundness ($\le 2^{-40}$ at $Q \le 2^{64}$)** | Paper §8 · Spec 02 | Lean 4 + Paper (`A-ROM`, `A-AROM`, `A-PCS`) | `RoundByRound.rbr_lifetime_soundness`, `rbr_lifetime_soundness_sum`, `ProtocolBridge.hpath_of_protocol_spec`, `fpn_protocol_lifetime_soundness` |
| **Sample-count formula & 16-cell table ($k=105$, $104$ fails)** | Paper §8.5 · Spec 02 §8 | Lean 4 (exact) | `LifetimeSoundness.kSuff_iff_kReq_le`, `table_Q64_e40`, `RoundByRound.three_pool_default`, `k104_fails_rbr` |
| **Goldilocks $\mathbb F_{p^2}$ insufficiency & $\mathbb F_{p^3}$ / $p^4$ bounds** | Paper §8.2, §8.6 · Spec 02 | Lean 4 (exact) | `RoundByRound.field_p2_iff`, `slot_table_numerator`, `field_p3_bits`, `ProtocolBridge.delta_SL_p4_default` |
| **Multi-column LogUp soundness $(n_t(w-1)+n_v+n_t)/|\mathbb E|$** | Paper §8.4 · Spec 02 §6.3 | Lean 4 (exact) | `LogUp.logup_bad_card_le`, `mergedZeroSum_iff`, `RoundByRound.logup_multicol_card_le` |
| **Sort-free bucketed sampling completeness & zero-slack circuit** | Paper §7 · Spec 06 §3 | Lean 4 (exact) | `BucketSampling.nucleus_iff_per_token`, `nucleus_sandwich`, `sure_mono_j`, `poss_mono_j`, `boundary_pair_sound_complete`, `bucket_sampling_complete` |
| **Single-proof & causal multi-round ZK masking + Libra** | Paper §9 · Spec 03 §4 | Lean 4 + Paper (ROM/PRF hybrids) | `Mask.masked_view_independent`, `MaskJoint.joint_tensor_surjective`, `multiround_independent_causal`, `libra_mask_bijOn_hyperplane` |
| **Coupled speculative decoding, MLA, LREC, PP/EP, MoE routing** | Paper §11 · Spec 05, 06 | Lean 4 (algebraic) | `Speculative.specLoop_eq_seqGen`, `ArchGeneral.mla_score_absorbed_eq`, `mla_out_absorbed_eq`, `pp_rangeLayers_add`, `ep_allreduce_eq_moe`, `LinearAttn.chunkRun_eq_recRun`, `Gadgets.approxTopK_pairwise` |
| **Honest native-FP zero false rejections & split-K falsification** | Paper §12.1–12.2 · Spec 06 | GPU Measurement (`RTX 4070 Ti SUPER`) | `benchmarks/certified/results/honest--Qwen3-0.6B*.json` (0 / 249,384), `honest--Llama-3.2-1B-Instruct*.json` (0 / 143,544) |
| **8-mode cheat detection under sublayer lineage ($h=3$)** | Paper §6.4, §12.4 · Spec 06 §5 | GPU Measurement (`RTX 4070 Ti SUPER`) | `benchmarks/certified/results/cheats--Qwen3-0.6B.json` ($d_C^{(3)} = 0.973\text{–}1.000$) |
| **Long-context sparse-attention tail upper bounds** | Paper §12.5 · Spec 06 §5.4 | GPU Measurement (`RTX 4070 Ti SUPER`) | `benchmarks/certified/results/longctx--Qwen3-4B.json` (4k–32k oracle bounds) |
