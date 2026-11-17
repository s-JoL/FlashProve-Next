# End-to-End Prover & `vLLM 0.33` Serving Benchmarks (`benchmarks/prover/`)

Hardware measurements of the complete **FlashProve-Next** Stage 1 (`v0.3.0`) zero-knowledge prover, standalone CPU verifier, and `vLLM 0.33` serving integration on the full unsliced **`Qwen/Qwen3-0.6B`** checkpoint (`596,049,920` parameters, $L = 28, d = 1024$, $16/8 \times 128$ GQA + QK-Norm, $d_{ff} = 3072$, $V = 151{,}936$, `bf16` weights with `fp32` LM head) over $\mathbb E = \mathbb F_{p^3}$ with audited Plonky3 Poseidon2 constants.

All GPU proving and `vLLM 0.33` runs executed on a single consumer workstation (**NVIDIA RTX 4070 Ti SUPER 16 GB**, Intel i7-14700F, 25 GB RAM); standalone CPU verification ran in isolated processes with `CUDA_VISIBLE_DEVICES=""` on both the Intel i7-14700F and an Apple M1 laptop. Raw JSON outputs are in [`results/`](results/). *(Note: `qwen3_06b_reply_315.json` and `qwen3_06b_mcsetup.json` were recorded at `T_QUERIES = 150` prior to raising `zk_flat.T_QUERIES` to `168`.)*

---

## 1. Summary of Full-Model Measurements (`Qwen3-0.6B`, $\mathbb F_{p^3}$)

| Artifact | Result JSON | Committed Cells | GPU Time | Peak Host RSS / VRAM Pool | Proof / Store Size | Standalone CPU Verify |
|---|---|---|---|---|---|---|
| **One-Time Static Commitment** (`R_max = 2048` with $\mathbf W^r$ & `CFG0`/`CFG1`) | [`qwen3_06b_commit.json`](results/qwen3_06b_commit.json) | — | **94.18 s** (`W`: 31.30 s, `RAW`: 34.74 s) | **12.0 GiB** / < 4 GiB | **7.62 GB** coeff store (`0 B` codeword) | — |
| **Complete Setup Proof $\pi_{\rm setup}$** (items (i)–(vi), 147 blocks + 206-col joint sub-proof) | [`qwen3_06b_mcsetup.json`](results/qwen3_06b_mcsetup.json) | **8.67 G** (`8,673,610,624`) | **3,623.7 s** (`770.8 + 39.9 + 2,810.3 s`) | **12.2 GiB** / **3.54 GiB** (`4.21 GiB` NVML) | **353.0 MB** | **59.91 s** (i7-14700F) / **63.71 s** (Apple M1) |
| **Full-Security 315-Draw Reply Proof** (`M_C = 210, M_G = 105, T = 152, T_pad = 256`) | [`qwen3_06b_reply_315.json`](results/qwen3_06b_reply_315.json) | **5.25 G** (`5,245,774,496`) | **3,877.7 s** (`H`: 3.78 s, build: 122.3 s, prove: 3,751.6 s) | **21.00 GiB** (`1.51 B/cell`, **`0 B` disk spill**) / **10.48 GiB** (`11.29 GiB` NVML) | **149.8 MB** | **12.02 s** (i7-14700F, `2.63 GiB` RSS) |

### Key Systems Breakdown (`qwen3_06b_reply_315.json`)
- **Hidden & public segments in the 315-draw trace (`T = 152, T_pad = 256`):** 11 hidden system-prompt tokens + 15 public request tokens + 116 hidden `<think>` chain-of-thought tokens + 10 visible reply tokens (`3.84 s` single-stream `bf16` reference inference).
- **Streaming witness & cross-slot batching:** Bit-packed integer witnesses (`3.92 GB` after Phase 2), lookup multiplicities (`3.01 GB`), and ephemeral AES-256-CTR blinding regeneration reduce host footprint from `49.7 B/cell` (~260 GB) to **`1.51 B/cell`** with **`0 B` disk spill**. Cross-slot batching collapses circuit vector count from ~1.52M to **`6,887`** (`max_claims_per_random_row = 0.25 <= 1` in `zk_budget_check`), with 210-slot `BatchedAttn` KV sumchecks taking **`5.8 s`** and 16,384-claim `HSel` sumchecks taking **`9.5 s`** on GPU.

---

## 2. `vLLM 0.33` Serving Throughput Impact (`Qwen3-0.6B`, CUDA Graphs Enabled)

Raw JSON: [`results/vllm_serving_overhead_qwen3_06b.json`](results/vllm_serving_overhead_qwen3_06b.json).

| `vLLM 0.33` Configuration | Batch 1 (`tok/s`) | Batch 8 (`tok/s`) | Batch 32 (`tok/s`) |
|---|---|---|---|
| **Stock `vLLM 0.33` default** (`bf16` GEMM reduce, `bf16` head, capture disabled) | 356.1 | 2,379.1 | 8,467.7 |
| **Contract-compliant numerics (`R32`)** (`fp32` GEMM reduce + `fp32` head, capture disabled) | **351.8 (−1.21%)** | **2,325.7 (−2.24%)** | **8,236.0 (−2.74%)** |
| **Stage 1 synchronous D2H full-port capture** (`28` layers $\times$ `10` ports + `fpn-samp-bk-v3`) | 165.9 (−53.41%) | 732.5 (−69.21%) | 1,304.8 (−84.59%) |

*Interpretation:* Enforcing the numerical rounding contract (`allow_bf16_reduced_precision_reduction=False` and `fp32` LM-head logits) costs only **1.2%–2.7%** serving throughput. We attribute most of the larger drop in the Stage 1 full-capture prototype to the synchronous per-step `torch.cuda.synchronize()` copy of `0.82 MB/token` of per-GEMM port buffers (`26.2 MB/step` at batch 32) to pinned host memory; replacing it with double-buffered asynchronous CUDA copy streams is planned for Stage 2.

---

## 3. Live `vLLM 0.33` End-to-End API & Forgery Suite (`Qwen3-0.6B`)

Raw JSON: [`results/e2e_vllm_qwen3_06b.json`](results/e2e_vllm_qwen3_06b.json) (`all_pass = true`, `peak_vram_gib = 8.90`, `peak_host_used_gib = 17.39`, `0` Python callbacks on `FULL` CUDA-graph replay).

Covers 9 live HTTP cases against `fpn_vllm_server.py` + `prover_daemon.py` + standalone `fpn-verify` (`CUDA_VISIBLE_DEVICES=""`):
1. `concurrent_streaming_cudagraph`: 4 concurrent SSE requests (`max_cobatch = 4`, `FULL` CUDA-graph decode) + non-blocking probe (`0.21 s` while background GPU proving runs);
2. `prefix_cache_reuse`: `48` prefix-cached tokens via `PrefixPortCache` first-write-wins locking;
3. `hidden_sys_cot_eos_stop`: hidden system prompt + `63` hidden `<think>` tokens + `stop` termination;
4. `multiturn_state_chain`: Turn-0 $\to$ Turn-1 AEAD state continuation (`St_prev -> St_new`) + tampered `St_prev` rejection;
5. `constrained_json_schema`: strict JSON Schema byte-DFA mask bound over $\mathbb F_{p^3}$ (`{"city": "Paris", "capital": true}`);
6. `tool_calls_and_witness`: client `tool_use` + server-executed private `TOOL_CAP` return witness;
7. `max_tokens_zero`: edge-case `max_tokens = 0` (`n_vis = 0`, `finish_reason = "length"`);
8. `custom_stop_public_refusal`: public pre-inference refusal (`provable = false`) on unsupported custom string `stop`;
9. `downgraded_int4_rejected`: dishonest `int4`-quantized server under the committed `bf16` `model_id` (`HTTP 409` + standalone CPU `REJECTED`).

---

## 4. Reproducing the Full-Model Benchmarks

```bash
# Full-security 315-draw Qwen3-0.6B reply proof (requires ~21 GiB RAM & 12 GiB VRAM)
python benchmarks/prover/bench_qwen_reply.py capture --model /path/to/Qwen3-0.6B --t-pad 256 --out /var/tmp/fpn-qwen315/cap.npz
python benchmarks/prover/bench_qwen_reply.py commit  --model /path/to/Qwen3-0.6B --cap /var/tmp/fpn-qwen315/cap.npz --store /var/tmp/fpn-qwen315/store
python benchmarks/prover/bench_qwen_reply.py prove   --model /path/to/Qwen3-0.6B --cap /var/tmp/fpn-qwen315/cap.npz --store /var/tmp/fpn-qwen315/store --out-dir /var/tmp/fpn-qwen315 --m-c 210 --m-g 105
CUDA_VISIBLE_DEVICES="" python benchmarks/prover/bench_qwen_reply.py verify --out-dir /var/tmp/fpn-qwen315 --json-out benchmarks/prover/results/qwen3_06b_reply_315.json

# vLLM 0.33 serving throughput comparison (stock vs. R32 numerics vs. full capture)
python integrations/vllm/bench_serving_overhead.py \
    --model /path/to/Qwen3-0.6B \
    --out benchmarks/prover/results/vllm_serving_overhead_qwen3_06b.json
```
