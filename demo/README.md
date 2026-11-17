# Standalone CPU Verification Demo (`demo/`)

Verify real zero-knowledge proof bundles on any laptop CPU—**no GPU, no PyTorch, and no model weights required**:

```bash
pip install -e ".[tokenizers]"
python -m flashprove_next.build_ext   # optional C++ Goldilocks speedup; pure NumPy fallback works too
python demo/run_demo.py
```

## Security Tier & Provenance Note

To keep repository clone size compact (~10.1 MiB per bundle) and verification fast (~21 s total on CPU), `demo/bundles/` uses the reduced **`qwen3-0.6b-demo`** profile (`security_tier = "demo_fast"`, `M_C = 4` compute draws + `M_G = 2` generation draws, `T_pad = 256`).
- By default, `fpn-verify` accepts **only** `production` profiles (`qwen3-0.6b-sec315`, `M_C = 210, M_G = 105 = 315` draws for the $2^{-40}$ lifetime soundness bound over $Q = 2^{64}$ queries) so a server can never silently downgrade a client to a demo budget; `demo/run_demo.py` passes `--allow-reduced-security` explicitly and verifies that omitting this flag is `REJECTED`.
- The demo commitment uses fixed public fixture keys (`kprf`, `salt_cfg`, AEAD wrap key, and the illustrative hidden system prompt `"You are a concise, accurate assistant."` in `integrations/vllm/end_to_end_vllm.py`) so anyone with a 16 GB GPU can regenerate an identical `model_id` structure.

## What `demo/run_demo.py` Verifies

`demo/bundles/` contains `FPNZK1` bundles generated from unsliced **`Qwen/Qwen3-0.6B`** (`L = 28, d = 1024, V = 151,936`, native `bf16` + `fp32` LM head) served live on **`vLLM 0.33`** with CUDA Graphs and continuous batching enabled (`manifest.json` records exact hardware, `vLLM` version, and measured CPU verification times):

1. **Honest Turn-0 response with hidden system prompt & hidden `<think>` chain of thought (`VERIFIED`):**
   `verifier/fpn_verify.py` runs in an isolated CPU-only subprocess (`CUDA_VISIBLE_DEVICES=""`), re-tokenizes the client's raw `messages` locally against the public Qwen3 tokenizer (`qwen3_tokenizer.json.gz`, Apache-2.0), checks the public state anchor `St_new` and client nonce, and verifies the zero-knowledge proof without seeing weights, layer count, hidden system prompt tokens, or hidden CoT text.
2. **Honest Turn-1 multi-turn continuation (`VERIFIED`):**
   Verifies a second conversation turn chaining the client's stored state anchor `St_prev -> St_new` across re-injected hidden history.
3. **Honest structured output under strict JSON Schema (`VERIFIED`):**
   Compiles the client's JSON Schema into the canonical minimal co-accessible byte DFA and checks the $\mathbb F_{p^3}$ Schwartz–Zippel full-vocabulary mask binding inside the proof.
4. **Dishonest provider running `int4`-quantized weights (`REJECTED`):**
   A dishonest `vLLM` server quantizes the `bf16` checkpoint to `int4` behind the same `model_id` and forces the prover daemon to emit a proof; the standalone CPU verifier rejects it.
5. **Security-downgrade, replay & tampering checks (`REJECTED`):**
   Verifying a `demo_fast` bundle without `--allow-reduced-security`, modifying a visible output token, replaying under a fresh client nonce, tampering with multi-turn `St_prev`, and injecting an unrequested JSON Schema are all deterministically rejected.

## Regenerating the Bundles on a Single GPU

On a Linux machine with a 16 GB NVIDIA GPU, `vLLM 0.33`, and `CuPy`:

```bash
python demo/generate_qwen_bundles.py \
    --model /path/to/Qwen3-0.6B \
    --work /var/tmp/fpn-demo-build \
    --serve-python /path/to/vllm-venv/bin/python \
    --prove-python /path/to/cupy-venv/bin/python
```
