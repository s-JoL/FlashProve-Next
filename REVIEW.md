# Reviewer Guide (`REVIEW.md`)

This guide shows how to evaluate **FlashProve-Next** (`v1.0.0`) on an ordinary laptop in minutes: how to verify fixed `8 MiB` `FPN2` containers with our zero-dependency Rust verifier (`~10 ms`) and real `Qwen3-0.6B` API proofs on CPU without a GPU or model weights, how the paper's theorems map to the **28 machine-checked Lean 4 modules (696 declarations)** and prover/verifier code, and what lies inside vs. outside the current formal and empirical boundaries.

---

## 1. Two-Minute Orientation & Two 10-Second Laptop CPU Demos

- **Problem & Goal:** Let a closed-weight LLM API provider prove to any client, on an ordinary CPU, that an API response came from the real model configuration committed in `model_id`, while keeping production inference untouched (native `bf16`/`fp8`/`mxfp4`, random sampling, FlashAttention, CUDA graphs, continuous batching, prefix caching) and revealing nothing beyond standard API outputs.
- **v1.0.0 Scope:** Combines the protocol paper ([`paper/flashprove-next.pdf`](paper/flashprove-next.pdf)) and eleven specifications ([`specs/01`–`11`](specs/README.md)) with:
  1. **28 machine-checked Lean 4.29.1 modules** ([`lean/`](lean/THEOREM.md), **696 declarations**, zero `sorry`, zero `native_decide`, standard axioms only): 19 protocol modules (219 declarations) plus 9 `FPN2` executable-verifier, concrete-soundness, IR gadget, and `Fpn2EndToEnd` modules (477 declarations);
  2. **Zero-dependency `#![forbid(unsafe_code)]` Rust verifier & detached pure-Python reference** ([`core/`](core/), [`verifier/python/`](verifier/python/)): verify fixed-size `FPN2` containers (Relations 1–6, including exact `8,388,608 B` `fpn2-llm-std-v1` in **`~10 ms` Rust / `~110 ms` Python** on Apple M1 CPU) with golden vectors, zero-panic descriptor hardening, and AST-enforced zero prover coupling;
  3. **Unified zero-Phase-2 folded ZK prover (`multi_gkr` + `multi_pcs` + `folded_engine`) & `8 MiB` `FPNZK2` containers** ([`prover/fpn_prover/backend/`](prover/fpn_prover/backend/), [`flashprove_next/`](flashprove_next/)): demonstrated on RTX 6000D across unsliced `Qwen3-0.6B` (`ExtSeed-153`, `Q=128`: `1.47B` cells, `0` Phase-2, `8 MiB` `FPNZK2`, `450.2 s` GPU prove / `569.8 s` total, `10.76 s` isolated CPU verify, plus `Q=100` receipt and a separate `8+4` `14/14` cheating matrix including real 28-layer perturbed and `int4` weight forwards), unsliced 4.02B `Qwen3-4B` (36 layers, dev `4+2` draws, `0` gadget errors), and live `vLLM 0.33` `Qwen3-30B-A3B-FP8` zero-plaintext Relation 6 (`1.49 s` GPU prove, `23.4 ms` Rust verify, `0 B` plaintext activations; receipts in [`benchmarks/prover/results/`](benchmarks/prover/results/));
  4. **Non-invasive `vLLM 0.33` integration & offline laptop CPU demos** ([`integrations/vllm/`](integrations/vllm/), [`serve/`](serve/), [`demo/`](demo/)).

### Demo A (Zero-Dependency Rust `FPN2` Relation-2 Verifier, ~10 ms per `8 MiB` Proof)

```bash
cargo build --release --manifest-path core/Cargo.toml --bin fpn-verify
python demo/run_v1_relation2_demo.py
```

### Demo B (Real `Qwen3-0.6B` `vLLM 0.33` API Proof Bundles on CPU)

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

Open [`lean/FlashProveNext.lean`](lean/FlashProveNext.lean) for the seven-layer dependency map, then inspect the theorem **statements** (not the proofs; the five modules total ~1,800 lines including proofs, or ~3,100 lines with `RoundByRound` and architecture modules) of these five entry points:

1. **`SumcheckSound.sumcheck_sound`** ([`lean/FlashProveNext/SumcheckSound.lean`](lean/FlashProveNext/SumcheckSound.lean)) — Full $n$-round Sumcheck soundness $n d / |F|$ against adaptive provers.
2. **`ConstrainedMask.constrained_mask_check_sound`** ([`lean/FlashProveNext/ConstrainedMask.lean`](lean/FlashProveNext/ConstrainedMask.lean)) — One-hot DFA state selection and multilinear Schwartz–Zippel soundness ($\le v_b / |\mathbb E| \le 2^{-186}$ over $\mathbb F_{p^3}$, `constrained_mask_goldilocks`) binding Phase-1 boolean vocabulary masks to compiled JSON-Schema byte DFAs.
3. **`BucketSampling.bucket_sampling_complete`** ([`lean/FlashProveNext/BucketSampling.lean`](lean/FlashProveNext/BucketSampling.lean)) — Witness-free integer top-$p$ sandwich (`nucleus_sandwich`), zero-slack boundary-pair circuit equivalence (`boundary_pair_sound_complete`), and completeness of bucketed Gumbel-max + inverse-CDF sampling across all $T > 0$ and $p_{\rm top} \in (0, 1]$.
4. **`MaskJoint.multiround_independent_causal`** ([`lean/FlashProveNext/MaskJoint.lean`](lean/FlashProveNext/MaskJoint.lean)) — Prefix-measurable witness shifts induce a unique transcript per mask tuple whose pushforward under uniform masks is the exact uniform distribution (`libra_mask_bijOn_hyperplane` covers the Libra outer mask).
5. **`ProtocolBridge.fpn_protocol_lifetime_soundness`** ([`lean/FlashProveNext/ProtocolBridge.lean`](lean/FlashProveNext/ProtocolBridge.lean), importing [`RoundByRound.lean`](lean/FlashProveNext/RoundByRound.lean)) — Composes `ProtocolRBRSpec` (`compiled_cs_doomed_of_bad_sample`, `hpath_of_protocol_spec`) with `RoundByRound.rbr_default_lifetime` (`three_pool_default`, `field_p3_bits`, `hco_col_esc_le`, `setup_col_esc_le`, `tc_col_esc_le`, `basefold_qry_esc_le`, `delta_SL_p4_default`) to establish bad-accept density $\le 2^{-40}$ across $Q \le 2^{64}$ queries.

*(For architecture equivalences, see [`Speculative.lean`](lean/FlashProveNext/Speculative.lean), [`LinearAttn.lean`](lean/FlashProveNext/LinearAttn.lean), [`ArchGeneral.lean`](lean/FlashProveNext/ArchGeneral.lean), and [`Gadgets.lean`](lean/FlashProveNext/Gadgets.lean).)*

Verify all 28 modules in one command:

```bash
cd lean && lake exe cache get && bash check.sh
```

---

## 4. Paper-to-Lean-to-Code & Evidence Map

### Artifact & Verifier Quick Map (`FPN2` vs `FPNZK2` vs `FPNZK1`)

| Format | Artifact / Relation | Prover | Verifier |
|---|---|---|---|
| **`FPN2`** (binary, zero-padded) | Self-contained Relations 1–6 (Relation 2 64-row demo `8 MiB`, Relation 4 AIR, Relation 5/6 `32×1024` tiles incl. live `vLLM` 30B MoE) | `prover/fpn_prover/backend/{relation2,folded_*}.py` | **Zero-dependency Rust `core/target/release/fpn-verify`** (`10.1 ms` Rel2, `23.4 ms` Rel6) & **detached Python `verifier/python/fpn_verify_ref`** (AST-verified zero prover imports) |
| **`FPNZK2`** (`8 MiB` uncompressed zero-padded) | Full 12-round interactive LLM reply (`Qwen3-0.6B ExtSeed-153`, `Qwen3-4B dev 4+2`, `14/14` cheating matrix) | `flashprove_next/reply_zk.py` → `prover/fpn_prover/backend/folded_engine.py` (`0` Phase-2) | Isolated Python CPU verifier `verifier/fpn_verify.py` / `flashprove_next/reply_bundle.py` (`10.73 s` 0.6B, `7.18 s` 4B; shares Python stack with prover) |
| **`FPNZK1`** (Stage 1 gzip) | Historical `v0.3.0` Stage 1 `NI-315` (`149.8 MB`) & bundled laptop demo (`demo/run_demo.py`) | Legacy `flashprove_next/zk_flat.py` (`--backend legacy`) | Isolated Python CPU verifier `verifier/fpn_verify.py` |

### Claim-to-Evidence Table

| Claim | Paper / Spec | Verification Layer | Key Lean Modules / Code / Public JSON Artifacts |
|---|---|---|---|
| **Round-by-round lifetime soundness (`NI-315` $\le 2^{-40}$ at $Q \le 2^{64}$; `ExtSeed-153` $\le 2^{-20}$ at $Q_{\rm samp} \le 2^{30}$)** | Paper §8, §12.10 · Spec 02 | Lean 4 (composition) + Paper (`A-ROM`, `A-AROM`, `A-PCS`/`A-PCS-fold`, `ProtocolRBRSpec`) | `RoundByRound`, `ProtocolBridge`, `Fpn2EndToEnd` (`CryptoAssumptions`, explicit caveats in `lean/THEOREM.md` §2.4) |
| **Executable `FPN2` byte verifiers (Rel 1, 3–5) & 6 IR gadget families** | Spec 11 · `lean/THEOREM.md` §2.1–2.4 | Lean 4.29.1 (`28` modules, `696` audited decls, `0 sorry`) + Rust/Python parity | `VerifyBytes`, `Fpn2Rel3`–`5` (`131/131` golden vectors + `26/26` multi-tile), `Fpn2Soundness`, `Fpn2Gadgets`, `Fpn2EndToEnd`; `core/`, `verifier/python/` |
| **Sort-free bucketed sampling & $\mathbb F_{p^3}$ constrained-mask binding** | Paper §4.5, §7 · Spec 05, 06 | Lean 4 (exact) + GPU E2E | `BucketSampling`, `ConstrainedMask`; `flashprove_next/{sampling_v3,sampler_zk,api_zk}.py` |
| **v1.0 Folded `Qwen3-0.6B` `ExtSeed-153` (`8 MiB` `FPNZK2`, `0` Phase-2)** | Paper §12.10 · Spec 02 §10, Spec 11 §6 | GPU E2E (`RTX 6000D`) + Isolated Python CPU verify | `prover/fpn_prover/backend/{multi_gkr,multi_pcs,folded_engine}.py`; [`qwen3_06b_extseed153_q128_rtx6000d.json`](benchmarks/prover/results/qwen3_06b_extseed153_q128_rtx6000d.json) (`Q=128` unique-decoding $\le 2^{-106}$: `1.47B` cells, `450.2 s` prove / `569.8 s` total, `99.7 GiB` RSS, `10.76 s` CPU verify) & [`qwen3_06b_extseed153_v1_rtx6000d.json`](benchmarks/prover/results/qwen3_06b_extseed153_v1_rtx6000d.json) (`Q=100`: `461.1 s` / `571.0 s`, `10.73 s` CPU, `n_reason=0`) |
| **v1.0 `Qwen3-0.6B` `14/14` real altered-weight cheating matrix (`8+4`, `Q=100`)** | Paper §12.10 | GPU E2E (`RTX 6000D`) | `benchmarks/prover/bench_v1_cheating_matrix.py`; [`qwen3_06b_cheating_matrix_rtx6000d.json`](benchmarks/prover/results/qwen3_06b_cheating_matrix_rtx6000d.json) (1 honest + 13 negative incl. real 28L perturbed & group-32 `int4` weight forwards) |
| **v1.0 Full unsliced 4.02B 36-layer `Qwen3-4B` (dev `4+2`, `8 MiB` `FPNZK2`)** | Paper §12.10 | GPU E2E (`RTX 6000D`) + Isolated Python CPU verify | Generalized `rsqrt_enc`, `mc_zk` `CC`, sharded `Checkpoint`; [`qwen3_4b_mc4_mg2_v1_rtx6000d.json`](benchmarks/prover/results/qwen3_4b_mc4_mg2_v1_rtx6000d.json) (`259.1M` cells, `160.3 s` prove / `273.0 s` total, `7.18 s` CPU verify, `0` gadget errors) |
| **Live `vLLM 0.33` `Qwen3-30B-A3B-FP8` Relation 6 (zero-plaintext activations)** | Paper §12.10 · Spec 11 §5.7 | GPU E2E (`RTX 6000D`) + Rust `fpn-verify` | `prover/fpn_prover/runtime/x_commit.py`, `core/src/folded_vec_bound.rs`; [`live_vllm_rel6_zk_v1_rtx6000d.json`](benchmarks/prover/results/live_vllm_rel6_zk_v1_rtx6000d.json) (`45.1 s` lazy precommit, `135 MiB` buffers, `1.49 s` prove, `23.4 ms` Rust verify, `0 B` plaintext activations; public `16.8 MB` weight tile openings) |
| **Blackwell 5-model calibration, zero-sync async capture & 20-turn `St_i`** | Paper §12.10 | GPU Measurement (`RTX 6000D`) | [`frontier_calibrate_rtx6000d.json`](benchmarks/certified/results/frontier_calibrate_rtx6000d.json) (`0 / 1,846,272` GEMM+MoE blocks), [`non_gemm_calibrate_rtx6000d.json`](benchmarks/certified/results/non_gemm_calibrate_rtx6000d.json) (`0 / 1.96M` rows), [`async_capture_qwen3_4b_rtx6000d.json`](benchmarks/prover/results/async_capture_qwen3_4b_rtx6000d.json) (`−1.15%..−13.14%` vs `numerics_r32`, `0` extra syncs/overflows), [`agentic_20turn_qwen3_06b_rtx6000d.json`](benchmarks/prover/results/agentic_20turn_qwen3_06b_rtx6000d.json) (`10.5x` dedup, `0.75–0.92 s` turn-20 `St_i`) |
| **Stage 1 historical baseline (`NI-315`, `1.51 B/cell`, $\pi_{\rm setup}$ & `vLLM` 9-case E2E)** | Paper §12.7–12.9 · Spec 09, 10 | GPU E2E (`RTX 4070 Ti SUPER 16 GB`) | [`qwen3_06b_reply_315.json`](benchmarks/prover/results/qwen3_06b_reply_315.json) (`5.25G` cells, `21.0 GiB` RSS, `3,877.7 s`, `12.02 s` CPU, `116` hidden CoT), [`qwen3_06b_mcsetup.json`](benchmarks/prover/results/qwen3_06b_mcsetup.json) (`8.67G` cells), [`e2e_vllm_qwen3_06b.json`](benchmarks/prover/results/e2e_vllm_qwen3_06b.json) |

---

## 5. Explicit v1.0.0 Scope & Open Boundaries

1. **Verifier independence gap on full interactive replies:** Zero-dependency Rust `fpn-verify` (`core/`) and detached Python (`verifier/python/`) verify `FPN2` Relations 1–6 (`~10 ms` on the 64-row Relation-2 demo; `23.4 ms` on live `vLLM` 30B MoE Relation 6). Full 12-round interactive `Qwen3-0.6B`/`4B` `FPNZK2` replies still verify via Python `verifier/fpn_verify.py` (shared stack with `flashprove_next`); lowering those 12 rounds into a self-contained `CircuitDesc` for single-binary Rust verification is the #1 engineering priority for v2.
2. **Proving cost, host memory & capture scaling:** Single-stream `Qwen3-0.6B ExtSeed-153` total prover time (`571.0 s`) is ~`103×` single-stream inference (`5.54 s`), so keeping fleet-wide proving within `<= 100%` of serving relies on probabilistic audit deterrence (`p_audit <= 1%`). The v1.0 folded prover materializes host cell arrays (`100.1 GiB` peak RSS vs Stage 1's `21.0 GiB` streaming), and `Qwen3-4B` full-reply is measured at dev `4+2` draws. Zero-sync async capture (`serve/fpn_serve/`) reaches `−1.15%` single-stream (`b1, c256`) and `−13.14%` at `b8, c8192` vs `numerics_r32` (0 extra syncs, 0 overflows).
3. **Architecture hiding & Relation 6 binding:** v1.0 dense `reply_zk` fixes coarse geometry per profile (cross-architecture hidden-table privacy is v2). Relation 6 (`FOLDED_VEC_BOUND`) eliminates plaintext activation disclosure (`0 B`) on live 30B MoE, but opens `16.8 MB` of public weight tiles and does not bind `X` to the GPU capture or `Y` to the served output; router calibration also discloses `11.5%`/`38.0%` single-token swap detection under zero-margin top-$k$ ties.
4. **Formalization boundary:** Mechanical extraction linking Python circuit-builder code to Lean's `CompiledCS` / `ProtocolRBRSpec`, Basefold proximity soundness (`A-PCS` for Stage 1 unique-decoding rate-1/4, and `A-PCS-fold` for v1.0 rate-1/8 `Q=72/100` multi-size layer injection under provisional Johnson/list-decoding proximity), a full folded-backend ZK simulator, Lean verifiers for Relations 2 & 6, real-valued floating-point rounding propagation, and the parallel scan error template (`Scan`) remain outside Lean (`specs/02`–`07`, `specs/11`).
