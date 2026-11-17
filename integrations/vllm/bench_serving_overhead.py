#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serving overhead of the proof hooks: vLLM 0.33 throughput and VRAM on the unsliced Qwen3-0.6B in three modes
(each mode runs in a fresh subprocess):
  (A) stock_default          : unmodified stock vLLM 0.33 (CUDA graphs, bf16 reduction=True, bf16 head, hook disabled)
  (B) numerics_r32_only      : allow_bf16_reduced_precision_reduction=False + head_dtype=float32 + vllm_c RMSNorm,
                               with the capture loader disabled (`enabled=False` -> 0 extra buffer VRAM)
  (C) full_fpn_zk_capture    : (B) + `load_format='flashprove_next'` full 28-layer per-GEMM in-graph port capture,
                               D2H harvest and `FpnZkSampler` (`sampling_v3_torch` GPU sampler)

  python integrations/vllm/bench_serving_overhead.py --model PATH/TO/Qwen3-0.6B --out results/serving_overhead.json
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from flashprove_next.hwinfo import describe as describe_machine  # noqa: E402


def _vllm_version():
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return "unknown"


def _run_mode(model_dir: str, mode: str, batches: list[int], gen_tokens: int, repeats: int) -> dict:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm_capture_zk as VCZ
    from flashprove_next import sampling_v3 as S3

    rule = S3.Rule(0.6, 0.95, 32)
    kprf = [0x0123456789ABCDEF % ((1 << 64) - (1 << 32) + 1), 0x11112222, 0x33334444, 0x55556666]

    if mode == "stock_default":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        VCZ.configure_fpn_loader(enabled=False)
        llm_kw = dict(
            model=model_dir,
            load_format="auto",
            skip_tokenizer_init=True,
            enforce_eager=False,
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            max_model_len=512,
            max_num_seqs=max(batches),
            max_num_batched_tokens=max(1024, max(batches) * 64),
            enable_prefix_caching=False,
            seed=0,
        )
    elif mode in ("numerics_r32_only", "full_fpn_zk_capture"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        cap_on = (mode == "full_fpn_zk_capture")
        mbt = max(1024, max(batches) * 64)
        VCZ.configure_fpn_loader(max_batched_tokens=mbt, enabled=cap_on)
        if cap_on:
            VCZ.configure_sampler(kprf, rule)
        llm_kw = dict(
            model=model_dir,
            load_format="flashprove_next" if cap_on else "auto",
            skip_tokenizer_init=True,
            enforce_eager=False,
            dtype="bfloat16",
            hf_overrides={"head_dtype": "float32"},
            gpu_memory_utilization=0.45,
            max_model_len=512,
            max_num_seqs=max(batches),
            max_num_batched_tokens=mbt,
            enable_prefix_caching=False,
            logits_processors=[VCZ.FpnZkSampler] if cap_on else [],
            kernel_config={
                "ir_op_priority": {
                    "rms_norm": ["vllm_c", "native"],
                    "fused_add_rms_norm": ["vllm_c", "native"],
                }
            },
            compilation_config={"custom_ops": ["all"], "ir_enable_torch_wrap": False},
            seed=0,
        )
    else:
        raise ValueError(mode)

    llm = LLM(**llm_kw)
    cap = None
    if mode == "full_fpn_zk_capture":
        assert VCZ._P["buf"] is not None
        port_buf_mib = round(VCZ._P["buf"].numel() * 2 / (1 << 20), 3)
        bytes_per_tok = int(VCZ._P["layout"]["bytes_per_token"])
        cap = VCZ.ZkPortCapture(prefix_cache=None)
        llm.collective_rpc(cap.install_runner)
    else:
        assert VCZ._P["buf"] is None, "disabled capture loader must allocate 0 bytes of PORT_BUF"
        port_buf_mib = 0.0
        bytes_per_tok = 0

    by_batch = {}
    base_prompt = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198]
    for bsz in batches:
        prompts = [TokensPrompt(prompt_token_ids=base_prompt + [16 + (i % 8)]) for i in range(bsz)]

        def run_once():
            eng = llm.llm_engine
            rids = []
            ncs = {}
            pids_map = {}
            for i, tp in enumerate(prompts):
                rid = f"b{bsz}-{i}-{secrets.token_hex(4)}"
                rids.append(rid)
                pids_map[rid] = tp["prompt_token_ids"]
                if cap is not None:
                    cap.register(rid, tp["prompt_token_ids"], 512)
                    nc = secrets.token_hex(16)
                    ncs[rid] = nc
                    sp = SamplingParams(
                        n=1, temperature=1.0, top_p=1.0, max_tokens=gen_tokens, ignore_eos=True, detokenize=False,
                        extra_args={
                            VCZ.SAMPLER_KEY: {
                                "nonce": nc,
                                "server_nonce": secrets.token_hex(32),
                                "dig": ("11" * 32),
                                "v_rel": 1,
                                "sess": ("22" * 32),
                                "turn": 0,
                                "prompt_len": len(tp["prompt_token_ids"]),
                                "g_offset": 0,
                            }
                        },
                    )
                else:
                    sp = SamplingParams(
                        n=1, temperature=0.6, top_p=0.95, max_tokens=gen_tokens, ignore_eos=True, detokenize=False,
                    )
                eng.add_request(rid, tp, sp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rem = set(rids)
            while rem:
                outs = eng.step()
                for o in outs:
                    brid = o.request_id if o.request_id in rem else o.request_id.rsplit("-", 1)[0]
                    if o.finished and brid in rem:
                        rem.remove(brid)
                        if cap is not None:
                            c_obj, c_meta = cap.finish(
                                brid,
                                pids_map[brid] + list(o.outputs[0].token_ids),
                            )
                            assert c_obj is not None and c_meta["error"] is None, c_meta
                            VCZ.pop_server_nonce(ncs[brid])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            return dt

        # 1 warmup + `repeats` timed runs
        run_once()
        dts = [run_once() for _ in range(repeats)]
        med_s = statistics.median(dts)
        out_toks = bsz * gen_tokens
        by_batch[str(bsz)] = {
            "batch": bsz,
            "gen_tokens_per_req": gen_tokens,
            "median_wall_s": round(med_s, 4),
            "tok_per_s": round(out_toks / med_s, 1),
            "per_req_gpu_ms": round(1000.0 * med_s / bsz, 3),
            "runs_s": [round(x, 4) for x in dts],
        }

    out_d = {
        "mode": mode,
        "port_buf_vram_mib": port_buf_mib,
        "captured_bytes_per_token": bytes_per_tok,
        "batches": by_batch,
    }
    if cap is not None:
        out_d["capture_stats"] = dict(cap.stats)
    print("BENCH_RESULT_JSON:" + json.dumps(out_d), flush=True)
    return out_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Hugging Face snapshot of Qwen3-0.6B")
    ap.add_argument("--mode", choices=["all", "stock_default", "numerics_r32_only", "full_fpn_zk_capture"], default="all")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    batches = [int(x) for x in a.batches.split(",") if x.strip()]

    if a.mode != "all":
        _run_mode(a.model, a.mode, batches, a.gen_tokens, a.repeats)
        return

    modes = ["stock_default", "numerics_r32_only", "full_fpn_zk_capture"]
    res = {
        "model": "Qwen3-0.6B",
        "machine": None,
        "vllm_version": _vllm_version(),
        "gen_tokens_per_req": a.gen_tokens,
        "repeats": a.repeats,
        "modes": {},
    }
    for m in modes:
        print(f"[bench] running {m}...", flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--model", a.model, "--mode", m,
            "--batches", a.batches, "--gen-tokens", str(a.gen_tokens), "--repeats", str(a.repeats),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("BENCH_RESULT_JSON:")][-1]
        res["modes"][m] = json.loads(line.split("BENCH_RESULT_JSON:", 1)[1])
    res["machine"] = describe_machine()

    # Compute overhead summary relative to stock_default
    summary = {}
    for bsz in [str(b) for b in batches]:
        t0 = res["modes"]["stock_default"]["batches"][bsz]["tok_per_s"]
        t1 = res["modes"]["numerics_r32_only"]["batches"][bsz]["tok_per_s"]
        t2 = res["modes"]["full_fpn_zk_capture"]["batches"][bsz]["tok_per_s"]
        summary[bsz] = {
            "stock_tok_s": t0,
            "numerics_r32_tok_s": t1,
            "numerics_r32_delta_pct": round(100.0 * (t1 - t0) / t0, 2),
            "full_fpn_zk_tok_s": t2,
            "full_fpn_zk_delta_pct": round(100.0 * (t2 - t0) / t0, 2),
        }
    res["summary_by_batch"] = summary
    print(json.dumps(summary, indent=2), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
