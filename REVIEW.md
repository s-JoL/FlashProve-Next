# Reviewer Guide (`REVIEW.md`)

This guide shows how to evaluate **FlashProve-Next** (`v0.3.0`, Stage 1) on an ordinary laptop: how to verify real `Qwen3-0.6B` proofs on CPU without a GPU or model weights, how the paper's theorems map to the machine-checked Lean 4 files and Python prover modules, and what lies inside vs. outside the current formal and empirical boundaries.

---

## 1. Two-Minute Orientation & Laptop CPU Demo

- **Problem & Goal:** Let a closed-weight LLM API provider prove to any client—on an ordinary CPU in seconds—that an individual API response came from the real model configuration committed in `model_id`, while keeping production inference untouched (native `bf16`/`fp8`, random sampling, FlashAttention, CUDA graphs, continuous batching, prefix caching) and revealing nothing beyond standard API outputs.
- **Stage 1 Scope (`v0.3.0`):** Combines the complete protocol design ([`paper/flashprove-next.pdf`](paper/flashprove-next.pdf)) and ten formal specifications ([`specs/01`–`10`](specs/README.md)) with:
  1. **19 machine-checked Lean 4.29.1 modules** ([`lean/`](lean/THEOREM.md), 219 declarations, zero `sorry`, standard axioms only);
  2. **End-to-end $\mathbb F_{p^3}$ prover & standalone CPU verifier** ([`flashprove_next/`](flashprove_next/), [`verifier/fpn_verify.py`](verifier/fpn_verify.py)), demonstrated on full unsliced `Qwen3-0.6B` (`8.67 G`-cell one-time setup proof $\pi_{\rm setup}$ and `5.25 G`-cell, 315-draw full-security response proof at `1.51 B/cell` in `21.0 GiB` host RAM, `12.02 s` CPU verify);
  3. **Non-invasive `vLLM 0.33` integration & laptop CPU demo** ([`integrations/vllm/`](integrations/vllm/), [`demo/`](demo/)), covering concurrent SSE streaming, CUDA graphs, prefix caching, hidden system prompts, hidden `<think>` chain of thought, multi-turn AEAD state anchors, strict JSON Schema constrained decoding, and tool calls.

### Verify Real `Qwen3-0.6B` Proofs on a Laptop CPU (No GPU or Weights Needed)

```bash
pip install -e ".[tokenizers]"
python -m flashprove_next.build_ext   # optional C++ Goldilocks speedup; pure NumPy works too
python demo/run_demo.py
```

`demo/run_demo.py` invokes `verifier/fpn_verify.py` in an isolated CPU-only subprocess (`CUDA_VISIBLE_DEVICES=""`) on `Qwen3-0.6B` proof bundles produced by live `vLLM 0.33`. To keep the repository compact (~10.1 MiB per bundle) and verification fast on any laptop, the bundled demo uses the reduced `qwen3-0.6b-demo` profile (`4` compute + `2` generation draws, `security_tier = "demo_fast"`), which exercises every verifier check but does not carry the $2^{-40}$ lifetime bound of `qwen3-0.6b-sec315` (`210 + 105 = 315` draws, `149.8 MB`, `12.02 s` CPU verify on i7-14700F). By default `fpn-verify` rejects non-production profiles unless `--allow-reduced-security` is passed explicitly; `run_demo.py` verifies 3 honest (`VERIFIED`) + 6 dishonest/tampered (`REJECTED`) checks (including that downgrade protection).

---

## 2. Trusted Base for the Lean 4 Formalization

1. **The Lean 4 kernel**, toolchain `leanprover/lean4:v4.29.1` ([`lean/lean-toolchain`](lean/lean-toolchain)).
2. **Mathlib** at commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6` ([`lean/lake-manifest.json`](lean/lake-manifest.json)).
3. **Standard axioms only:** `propext`, `Classical.choice`, `Quot.sound`. Every exported theorem is audited via `#print axioms` in [`lean/check.sh`](lean/check.sh), which fails on any `sorry` or non-standard axiom.
4. **Explicit modelling hypotheses of top-level theorems:** Every Lean theorem states its premises explicitly in its type signature—for example, `ProtocolBridge.fpn_protocol_lifetime_soundness` takes the round-by-round specification `ProtocolRBRSpec`, per-round escape bounds `hEsc`, and spot-check miss `hm` as hypotheses rather than extracting them mechanically from Python circuit code (see Section 5).

---

## 3. Shortest Formal Reading Path

Open [`lean/FlashProveNext.lean`](lean/FlashProveNext.lean) for the four-layer dependency map, then inspect the theorem **statements** (not the proofs; the five modules total ~1,800 lines including proofs, or ~3,100 lines with `RoundByRound` and architecture modules) of these five entry points:

1. **`SumcheckSound.sumcheck_sound`** ([`lean/FlashProveNext/SumcheckSound.lean`](lean/FlashProveNext/SumcheckSound.lean)) — Full $n$-round Sumcheck soundness $n d / |F|$ against adaptive provers.
2. **`ConstrainedMask.constrained_mask_check_sound`** ([`lean/FlashProveNext/ConstrainedMask.lean`](lean/FlashProveNext/ConstrainedMask.lean)) — One-hot DFA state selection and multilinear Schwartz–Zippel soundness ($\le v_b / |\mathbb E| \le 2^{-186}$ over $\mathbb F_{p^3}$, `constrained_mask_goldilocks`) binding Phase-1 boolean vocabulary masks to compiled JSON-Schema byte DFAs.
3. **`BucketSampling.bucket_sampling_complete`** ([`lean/FlashProveNext/BucketSampling.lean`](lean/FlashProveNext/BucketSampling.lean)) — Witness-free integer top-$p$ sandwich (`nucleus_sandwich`), zero-slack boundary-pair circuit equivalence (`boundary_pair_sound_complete`), and completeness of bucketed Gumbel-max + inverse-CDF sampling across all $T > 0$ and $p_{\rm top} \in (0, 1]$.
4. **`MaskJoint.multiround_independent_causal`** ([`lean/FlashProveNext/MaskJoint.lean`](lean/FlashProveNext/MaskJoint.lean)) — Prefix-measurable witness shifts induce a unique transcript per mask tuple whose pushforward under uniform masks is the exact uniform distribution (`libra_mask_bijOn_hyperplane` covers the Libra outer mask).
5. **`ProtocolBridge.fpn_protocol_lifetime_soundness`** ([`lean/FlashProveNext/ProtocolBridge.lean`](lean/FlashProveNext/ProtocolBridge.lean), importing [`RoundByRound.lean`](lean/FlashProveNext/RoundByRound.lean)) — Composes `ProtocolRBRSpec` (`compiled_cs_doomed_of_bad_sample`, `hpath_of_protocol_spec`) with `RoundByRound.rbr_default_lifetime` (`three_pool_default`, `field_p3_bits`, `hco_col_esc_le`, `setup_col_esc_le`, `tc_col_esc_le`, `basefold_qry_esc_le`, `delta_SL_p4_default`) to establish bad-accept density $\le 2^{-40}$ across $Q \le 2^{64}$ queries.

*(For architecture equivalences, see [`Speculative.lean`](lean/FlashProveNext/Speculative.lean), [`LinearAttn.lean`](lean/FlashProveNext/LinearAttn.lean), [`ArchGeneral.lean`](lean/FlashProveNext/ArchGeneral.lean), and [`Gadgets.lean`](lean/FlashProveNext/Gadgets.lean).)*

Verify all 19 modules in one command:

```bash
cd lean && lake exe cache get && bash check.sh
```

---

## 4. Paper-to-Lean-to-Code & Evidence Map

| Claim | Paper / Spec | Verification Layer | Key Lean Modules / Code / Artifacts |
|---|---|---|---|
| **Round-by-round lifetime soundness ($\le 2^{-40}$ at $Q \le 2^{64}$)** | Paper §8 · Spec 02 | Lean 4 (composition) + Paper (`A-ROM`, `A-AROM`, `A-PCS`, `ProtocolRBRSpec`) | `RoundByRound.rbr_lifetime_soundness`, `ProtocolBridge.fpn_protocol_lifetime_soundness`; `flashprove_next/zk_flat.py`, `reply_zk.py` |
| **Sample-count formula & 16-cell table ($k=105$, $104$ fails)** | Paper §8.5 · Spec 02 §8 | Lean 4 (exact) | `LifetimeSoundness.kSuff_iff_kReq_le`, `table_Q64_e40`, `RoundByRound.three_pool_default`, `k104_fails_rbr` |
| **Goldilocks $\mathbb F_{p^2}$ insufficiency & $\mathbb F_{p^3}$ / $p^4$ bounds** | Paper §8.2, §8.6 · Spec 02 | Lean 4 (exact) | `RoundByRound.field_p2_iff`, `slot_table_numerator`, `field_p3_bits`, `ProtocolBridge.delta_SL_p4_default`; `flashprove_next/field.py` |
| **Multi-column LogUp soundness $(n_t(w-1)+n_v+n_t)/|\mathbb E|$** | Paper §8.4 · Spec 02 §6.3 | Lean 4 (exact) | `LogUp.logup_bad_card_le`, `mergedZeroSum_iff`, `RoundByRound.logup_multicol_card_le`; `flashprove_next/zk_flat.py` |
| **Sort-free bucketed sampling & $\mathbb F_{p^3}$ constrained-mask binding** | Paper §4.5, §7 · Spec 05, 06 | Lean 4 (exact) + GPU E2E | `BucketSampling.nucleus_sandwich`, `boundary_pair_sound_complete`, `bucket_sampling_complete`; `ConstrainedMask`; `flashprove_next/sampling_v3.py`, `sampler_zk.py`, `api_zk.py` |
| **Single-proof & causal multi-round ZK masking + Libra** | Paper §9 · Spec 03 §4 | Lean 4 + Paper (ROM/PRF/PRG) | `Mask.masked_view_independent`, `MaskJoint.multiround_independent_causal`, `libra_mask_bijOn_hyperplane`; `zk_budget_check` in `zk_flat.py` (`0.25 <= 1`) |
| **One-time static RS commitment & 8.67G-cell $\pi_{\rm setup}$** | Paper §5, §12.7 · Spec 09 | GPU E2E (`Qwen3-0.6B`, $\mathbb F_{p^3}$) | `flashprove_next/modelcommit.py`, `mc_setup.py`, `mc_zk.py` (`94.2 s` commit, `3,623.7 s` prove, `59.9 s` CPU verify) |
| **Full 315-draw `Qwen3-0.6B` reply proof (`1.51 B/cell`)** | Paper §12.8 · Spec 10 | GPU E2E (`Qwen3-0.6B`, $\mathbb F_{p^3}$) | `flashprove_next/reply_zk.py`, `attn_zk.py`, `hsel_zk.py`, `wsum_zk.py` (`5.25 G` cells, `21.0 GiB` RSS, `0 B` spill, `3,877.7 s` prove, `12.02 s` CPU verify) |
| **Non-invasive `vLLM 0.33` serving & 33 white-box forgery/tamper checks** | Paper §12.9 · Spec 05 | GPU E2E (`RTX 4070 Ti SUPER`) | `integrations/vllm/`, `tests/test_reply_zk.py` (21), `tests/test_api_zk.py` (12) (`R32` numerics `−1.2%..−2.7%`; sync capture `−53%..−85%`; 9 live scenarios `VERIFIED`; 33/33 checks `REJECTED`) |
| **Honest native-FP zero false rejections & 8-mode cheat catalog** | Paper §12.1–12.5 · Spec 06 | GPU Measurement (`RTX 4070 Ti SUPER`) | `benchmarks/certified/results/` (`0 / 392,928` honest bad blocks; `97.3%–100%` bad cost across 8 cheats; 4k–32k tail bounds) |

---

## 5. Explicit Prototype Scope & Open Boundaries

1. **Current proving & capture overhead:** On a single consumer RTX 4070 Ti SUPER (16 GB VRAM, 25 GB RAM), a full 315-draw `Qwen3-0.6B` proof takes `3,877.7 s` (~`1,009×` single-stream `bf16` inference, with ~`3,000 s` in Basefold openings and $2^{24}\dots 2^{26}$ streaming sumchecks), and the Stage 1 reference `vLLM` harvester uses a synchronous per-step D2H copy (`−53%..−85%` decode throughput vs. `−1.2%..−2.7%` for the numerical contract itself). Double-buffered async CUDA copy streams and opening batching target Stage 2.
2. **Architecture & lineage coverage:** Algebraic equivalences for Sparse MoE, absorbed MLA, chunked linear attention (`LREC`), and `PP`/`EP` are proved in Lean (`Gadgets`, `ArchGeneral`, `LinearAttn`), while the Stage 1 end-to-end circuit implements dense `GQA + QK-Norm` (`Qwen3-0.6B`) with depth-$h=1$ key-producer lineage (`N1 + KV + EMB` + `LSE` + deterministic injection/tool `EMB`). Stage 2 generalizes slots to uniform primitive blocks wired by hidden configuration tables and multi-hop ($h=3$) sublayer lineages under one verifier.
3. **Formalization boundary:** Mechanical extraction linking Python circuit-builder code to Lean's `CompiledCS` / `ProtocolRBRSpec`, Basefold proximity soundness (`A-PCS`), real-valued floating-point rounding propagation, and the parallel scan error template (`Scan`) remain on paper (`specs/02`–`07`).
