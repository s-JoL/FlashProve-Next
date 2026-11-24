#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tier-1 online-capture benchmark (acceptance gate v1): throughput cost of streaming capture on vLLM 0.33.

Modes (each one runs in its own fresh subprocess so GPU memory / compile caches never leak across modes)::

    stock_default        stock vLLM defaults (bf16 reduced-precision reduction ON), capture disabled
    numerics_r32         allow_bf16_reduced_precision_reduction=False (the compliant R32 baseline), capture disabled
    stream_full_no_kv    R32 + streaming capture, tier FULL_NO_KV (ring minus K/V)
    stream_full_no_kv_act R32 + streaming capture, tier FULL_NO_KV_ACT
    stream_tier_b        R32 + streaming capture, tier TIER_B (residual boundaries)

The gate compares ``stream_*`` against ``numerics_r32`` (the compliant bf16 deployment): <= 3% throughput loss, zero
``synchronize`` calls inside the timed step loop beyond what the baseline itself does, ring VRAM <= 350 MiB (Tier B <= 128).

Example::

    python bench/tier1/bench_capture.py --model /path/to/models/Qwen3-4B --batch 1,4,8 --ctx 256,8192 \\
        --out bench/tier1/results/qwen3_4b.json

``--dry-run`` prints the sizing plan (bytes/token, ring rows, legal max_num_batched_tokens) using only ``config.json`` --
no torch, no GPU.  Note: a FULL_NO_KV row of a 4B model is ~2 MB, so a 320 MiB ring holds ~160 rows; a step may use at
most half the ring (two consecutive steps must be in flight without overflow), so the legal ``max_num_batched_tokens`` is
~80; the same value is used for every mode so prefill chunking is identical across modes.  A full FULL_NO_KV trace of one
8k-token request is ~17 GB of host pages: cells that exceed the page-store cap run in count-only mode (``trace_retained:
false``) -- throughput/ring behaviour is real, the bytes are not kept (no disk spill yet).  At 8k context the KV cache of a 4B model on a 16 GB card limits real concurrency.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "serve"), os.path.join(ROOT, "prover"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODES = ("stock_default", "numerics_r32", "stream_full_no_kv", "stream_full_no_kv_act", "stream_tier_b")
GATE_OVERHEAD_PCT = 3.0
GATE_RING_MIB = {"stream_full_no_kv": 350, "stream_full_no_kv_act": 350, "stream_tier_b": 128}
STEP_FRACTION_OF_RING = 2        # a step may use at most 1/2 of the ring, so two consecutive steps are in flight without overflow


# ======================================================================================================= pure helpers
def read_model_dims(model_dir: str):
    from fpn_serve.capture.ports import ModelDims

    with open(os.path.join(os.path.expanduser(model_dir), "config.json"), "r", encoding="utf-8") as f:
        return ModelDims.from_hf_config(json.load(f), name=os.path.basename(os.path.normpath(model_dir)))


def capture_config_for(mode: str, ring_mib: int, max_batched_tokens: int, include_h_mid: bool = True):
    from fpn_serve.capture.ports import CaptureTier
    from fpn_serve.capture.session import CaptureConfig

    tier = {
        "stream_full_no_kv": CaptureTier.FULL_NO_KV,
        "stream_full_no_kv_act": CaptureTier.FULL_NO_KV_ACT,
        "stream_tier_b": CaptureTier.TIER_B,
    }[mode]
    return CaptureConfig(tier=tier, include_h_mid=include_h_mid, ring_mib=ring_mib, max_batched_tokens=max_batched_tokens)


def sizing_plan(dims, ring_mib: int, modes: list[str], include_h_mid: bool = True) -> dict[str, Any]:
    """Per stream mode: bytes/token, ring rows, legal ``max_num_batched_tokens`` (multiple of 16, <= rows)."""
    from fpn_serve.capture.ports import CaptureLayout

    plan: dict[str, Any] = {}
    for m in modes:
        if m not in GATE_RING_MIB:
            continue
        cfg = capture_config_for(m, ring_mib, 16, include_h_mid)
        lay = CaptureLayout.build(dims, cfg.options())
        rows = (cfg.ring_cap_mib() << 20) // lay.stride_bytes
        plan[m] = dict(payload_bytes_per_token=lay.payload_bytes_per_token, stride_bytes=lay.stride_bytes,
                       ring_mib=cfg.ring_cap_mib(), ring_rows=rows,
                       max_batched_tokens=rows // STEP_FRACTION_OF_RING // 16 * 16,
                       breakdown=lay.breakdown())
    return plan


def make_prompt(i: int, ctx: int, vocab: int = 150000) -> list[int]:
    """Deterministic, request-unique pseudo-random token ids (no prefix sharing between requests)."""
    x = (0x9E3779B97F4A7C15 * (i + 1)) & ((1 << 64) - 1)
    out = []
    for _ in range(ctx):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(1000 + (x >> 33) % (vocab - 1000))
    return out


def verify_handoff(handle, ref, tokens: list[int]) -> int:
    """Read every page and verify its shape and token-derived address against the engine transcript."""
    from fpn_serve.capture.ports import CaptureError, PAGE_TOKENS
    from fpn_serve.capture.store import next_block_hash, partial_block_hash

    parent = handle.root
    tip = tail = None
    offset = pages = full = 0
    for block_hash, n, rows in handle.iter_pages():
        expected_n = min(PAGE_TOKENS, len(tokens) - offset)
        if not 0 < expected_n == n or len(rows) != n * handle.store.row_bytes:
            raise CaptureError("handoff page length mismatch")
        page_tokens = tokens[offset:offset + n]
        if n == PAGE_TOKENS:
            expected = tip = next_block_hash(parent, page_tokens)
            full += 1
        else:
            expected = tail = partial_block_hash(parent, page_tokens)
        if block_hash != expected:
            raise CaptureError("handoff page hash chain mismatch")
        parent = block_hash
        offset += n
        pages += 1
    if (ref.request_id != handle.request_id or ref.root != handle.root or
            (ref.tip, ref.tail, ref.n_full_pages, ref.n_tokens) != (tip, tail, full, offset) or offset != len(tokens)):
        raise CaptureError("handoff trace reference mismatch")
    return pages


def decode_rate(batch: int, n_new: int, t_prefill_s: float, t_total_s: float) -> Optional[float]:
    dt = t_total_s - t_prefill_s
    return None if dt <= 0 or n_new < 2 else batch * (n_new - 1) / dt


def summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``results[mode]["cells"][key] = {decode_tok_s, e2e_tok_s, sync_calls, ...}`` -> overhead table + gate verdicts."""
    out: dict[str, Any] = {}
    base = results.get("numerics_r32", {}).get("cells", {})
    stock = results.get("stock_default", {}).get("cells", {})
    for mode, r in results.items():
        for key, c in r.get("cells", {}).items():
            row = dict(decode_tok_s=c.get("decode_tok_s"), e2e_tok_s=c.get("e2e_tok_s"), sync_calls=c.get("sync_calls"))
            for name, ref in (("vs_numerics_r32", base), ("vs_stock", stock)):
                for metric in ("decode_tok_s", "e2e_tok_s"):
                    a, b = c.get(metric), ref.get(key, {}).get(metric)
                    row[f"{metric}_{name}_pct"] = None if not a or not b else round(100.0 * (a - b) / b, 2)
            if mode in GATE_RING_MIB:
                b_sync = base.get(key, {}).get("sync_calls")
                ov = row["e2e_tok_s_vs_numerics_r32_pct"]
                if c.get("capture_complete") is not True:
                    ov = None                                  # a poisoned / incomplete run is faster: never grade it
                    row["verdict_withheld"] = "capture incomplete (ring overflow, store budget or sink fault)"
                row["gate"] = dict(
                    overhead_le_3pct=None if ov is None else ov >= -GATE_OVERHEAD_PCT,
                    zero_extra_sync=None if b_sync is None or c.get("sync_calls") is None else c["sync_calls"] <= b_sync,
                    ring_within_cap=r.get("ring_mib") is not None and r["ring_mib"] <= GATE_RING_MIB[mode],
                    capture_complete=c.get("capture_complete"),
                    zero_overflows=c.get("ring_overflow_events", r.get("ring_overflow_events")) == 0,
                    handoff_verified=c.get("handoff_verified") is True,
                )
            out.setdefault(mode, {})[key] = row
    return out


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version())
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / (1 << 30), 2)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        from importlib.metadata import version

        info["vllm"] = version("vllm")
    except Exception:                                   # noqa: BLE001
        info["vllm"] = None
    try:
        info["git_sha"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        info["git_sha"] = None
    return info


# ==================================================================================================== worker (one mode)
def _count_syncs():
    """Patch the explicit CUDA synchronization entry points with counters; returns (counter dict, restore fn)."""
    import torch

    n = {"count": 0}
    orig = (torch.cuda.synchronize, torch.cuda.Stream.synchronize, torch.cuda.Event.synchronize)

    def wrap(f):
        def g(*a, **k):
            n["count"] += 1
            return f(*a, **k)
        return g

    torch.cuda.synchronize = wrap(orig[0])
    torch.cuda.Stream.synchronize = wrap(orig[1])
    torch.cuda.Event.synchronize = wrap(orig[2])

    def restore():
        torch.cuda.synchronize, torch.cuda.Stream.synchronize, torch.cuda.Event.synchronize = orig

    return n, restore


def _worker(a: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    mode = a.mode
    batches = [int(x) for x in a.batch.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    stream = mode in GATE_RING_MIB
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (mode == "stock_default")
    if mode != "stock_default":
        torch.backends.cuda.matmul.allow_tf32 = False

    cp = sess = None
    ring_mib = 0.0
    bytes_per_tok = 0
    if stream:
        from fpn_serve.engines.vllm import capture_plugin as cp

        from fpn_prover.runtime import ResourceBudget   # bench (not serve/) may read the prover-side budget

        budget = ResourceBudget.preset(a.budget_profile)
        store_cap_mib = int(a.store_gib * 1024) if a.store_gib > 0 else int(budget.host_ram_limit_gib * 1024 * 0.35)
        cfg = capture_config_for(mode, a.ring_mib, a.max_batched_tokens, not a.no_h_mid)
        cfg = dataclasses.replace(cfg, page_store_mib=store_cap_mib)
        cp.configure_capture(cfg)
    llm = LLM(
        model=os.path.expanduser(a.model), load_format="fpn_capture" if stream else "auto", skip_tokenizer_init=True,
        dtype="bfloat16", enforce_eager=False, gpu_memory_utilization=a.gpu_mem_util,
        max_model_len=max(ctxs) + a.n_new + 16, max_num_seqs=max(batches), max_num_batched_tokens=a.max_batched_tokens,
        enable_prefix_caching=False, seed=0,
    )
    if stream:
        llm.collective_rpc(cp.install_runner)
        sess = cp.get_session()
        assert sess is not None, "capture loader did not instrument the model"
        ring_mib = sess.ring.capacity_rows * sess.layout.stride_bytes / (1 << 20)
        bytes_per_tok = sess.layout.payload_bytes_per_token
    else:
        try:
            from fpn_serve.engines.vllm import capture_plugin as cp_chk
            assert not cp_chk.is_enabled(), "disabled capture must allocate nothing"
        except ImportError:
            pass

    eng = llm.llm_engine
    cells: dict[str, Any] = {}
    uid = [0]

    def run_once(batch: int, ctx: int, n_new: int):
        from fpn_serve.capture.ports import CaptureError
        failed_before = set(sess.sink.failed) if sess is not None else set()
        overflow_before = sess.ring.overflow_events if sess is not None else 0
        rids = []
        prompts = {}
        outputs = {}
        for i in range(batch):
            uid[0] += 1
            rid = f"b{batch}c{ctx}-{uid[0]}"
            rids.append(rid)
            sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=n_new, ignore_eos=True, detokenize=False)
            prompts[rid] = make_prompt(uid[0], ctx)
            eng.add_request(rid, TokensPrompt(prompt_token_ids=prompts[rid]), sp)
        torch.cuda.synchronize()
        counter, restore = _count_syncs()
        t0 = time.perf_counter()
        rem = set(rids)
        try:
            while rem:
                for o in eng.step():
                    if o.finished:
                        rid = o.request_id if o.request_id in rem else o.request_id.rsplit("-", 1)[0]
                        rem.discard(rid)
                        if stream:
                            outputs[rid] = o
        finally:
            restore()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        complete = True
        verified_requests = verified_pages = 0
        handoff_errors = []
        if sess is not None:
            settled = sess.settle(120.0)                         # end markers + drain (outside the timed region)
            for rid in rids:                                      # the engine may suffix the id it hands the runner
                m = [i for i in settled if i == rid or i.rsplit("-", 1)[0] == rid]
                ok = len(m) == 1 and getattr(settled[m[0]], "n_tokens", None) == ctx + n_new - 1
                complete &= ok
                if not sess.store.retain_payload:
                    continue
                if not ok:
                    handoff_errors.append(f"{rid}: missing or incomplete trace ({m and settled[m[0]]})")
                    continue
                try:
                    output = outputs[rid].outputs[0].token_ids
                    if len(output) != n_new:
                        raise CaptureError("unexpected number of generated tokens")
                    verified_pages += verify_handoff(
                        sess.store.handle(m[0]), settled[m[0]], prompts[rid] + list(output[:-1]))
                    verified_requests += 1
                except (CaptureError, ValueError, KeyError, IndexError) as exc:
                    handoff_errors.append(f"{rid}: {exc}")
            complete &= not (set(sess.sink.failed) - failed_before) and sess.ring.overflow_events == overflow_before
            sess.forget(list(settled))                           # free host pages before the next cell
        return dt, counter["count"], bool(complete and not handoff_errors), verified_requests, verified_pages, handoff_errors

    for ctx in ctxs:
        for batch in batches:
            key = f"b{batch}_c{ctx}"
            overflow_start = sess.ring.overflow_events if sess is not None else None
            if sess is not None:
                est = batch * (ctx + a.n_new) * sess.layout.stride_bytes
                sess.store.retain_payload = a.retention == "keep" or (
                    a.retention == "auto" and est <= sess.store.max_bytes * 0.9)
            _, _, warm_ok, verified_requests, verified_pages, handoff_errors = run_once(batch, ctx, a.n_new)
            t_pre, t_tot, syncs, comp = [], [], [], [warm_ok]
            for _ in range(a.repeats):
                dt1, _, pre_ok, pre_reqs, pre_pages, pre_errors = run_once(batch, ctx, 1)
                dtn, sc, ok, reqs, pages, errors = run_once(batch, ctx, a.n_new)
                verified_requests += pre_reqs + reqs
                verified_pages += pre_pages + pages
                handoff_errors.extend(pre_errors)
                handoff_errors.extend(errors)
                ok &= pre_ok
                t_pre.append(dt1), t_tot.append(dtn), syncs.append(sc), comp.append(ok)
            mp, mt = statistics.median(t_pre), statistics.median(t_tot)
            retained = sess.store.retain_payload if sess is not None else None
            handoff = None if retained is None else (
                verified_requests == batch * (1 + 2 * a.repeats) if retained else "count_only")
            cells[key] = dict(handoff_verified=handoff, handoff_verified_requests=verified_requests,
                              handoff_verified_pages=verified_pages, handoff_errors=handoff_errors,
                              batch=batch, ctx=ctx, n_new=a.n_new, prefill_s=round(mp, 4), total_s=round(mt, 4),
                              decode_tok_s=decode_rate(batch, a.n_new, mp, mt), e2e_tok_s=batch * a.n_new / mt,
                              sync_calls=max(syncs), capture_complete=all(comp) if stream else None,
                              ring_overflow_events=(sess.ring.overflow_events - overflow_start if sess is not None else None),
                              trace_retained=(sess.store.retain_payload if sess is not None else None),
                              runs_total_s=[round(x, 4) for x in t_tot])
    res = dict(mode=mode, ring_mib=round(ring_mib, 3), payload_bytes_per_token=bytes_per_tok, cells=cells)
    if sess is not None:
        res["verify_payload_hash"] = bool(sess.store.verify_payload_hash)
        res["sink_stats"] = dict(sess.sink.stats)
        res["store_stats"] = dict(sess.store.stats)
        res["ring_overflow_events"] = sess.ring.overflow_events
    print("BENCH_RESULT_JSON:" + json.dumps(res), flush=True)
    return res


# ===================================================================================================== parent / dry run
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="local HF snapshot directory (Qwen3-0.6B / Qwen3-4B)")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--preset", choices=("b8", "8k"), help="16 GiB presets: b8 at 256 tokens or b1 at 8192 tokens")
    ap.add_argument("--batch")
    ap.add_argument("--ctx")
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ring-mib", type=int, default=320)
    ap.add_argument("--max-batched-tokens", type=int, default=0, help="0 = largest value legal for every requested stream mode")
    ap.add_argument("--no-h-mid", action="store_true", help="Tier B without h_mid (reproduces the memo's (L+1)*d*2 bytes/token)")
    ap.add_argument("--gpu-mem-util", type=float, default=None)
    ap.add_argument("--budget-profile", default="4070ti_super_16gb", help="ResourceBudget preset (host-RAM cap for the page store)")
    ap.add_argument("--store-gib", type=float, default=0.0, help="page-store cap in GiB (0 = 35%% of the budget's host RAM)")
    ap.add_argument("--retention", choices=("auto", "keep", "count_only"), default="auto",
                    help="auto: keep full traces when they fit the store cap, else count-only (pages addressed + counted, "
                         "bytes dropped; needs disk spill to retain, not implemented)")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)       # internal: run one mode in this process
    a = ap.parse_args(argv)

    preset_batch, preset_ctx = {"b8": ("8", "256"), "8k": ("1", "8192")}.get(a.preset, ("1,4,8", "256,8192"))
    a.batch = a.batch or preset_batch
    a.ctx = a.ctx or preset_ctx
    if a.gpu_mem_util is None:
        a.gpu_mem_util = {"b8": 0.72, "8k": 0.79}.get(a.preset, 0.80)

    modes = [a.mode] if a.mode else [m for m in a.modes.split(",") if m]
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"unknown modes {bad}")
    dims = read_model_dims(a.model)
    plan = sizing_plan(dims, a.ring_mib, modes, not a.no_h_mid)
    legal = min((p["max_batched_tokens"] for p in plan.values()), default=2048)
    if legal < 16:
        ap.error("ring cannot hold two 16-token steps")
    if a.max_batched_tokens <= 0:
        # Leave extra ring headroom for concurrent requests during chunked prefill.
        preset_step = {"b8": 32, "8k": 64}.get(a.preset, 2048)
        a.max_batched_tokens = min(preset_step, legal)
    elif plan and a.max_batched_tokens > legal:
        ap.error(f"--max-batched-tokens {a.max_batched_tokens} exceeds the ring ({legal} rows): a step must fit in the ring")

    if a.mode:
        _worker(a)
        return 0
    header = dict(batch=a.batch, ctx=a.ctx, gpu_memory_utilization=a.gpu_mem_util, model=dims.name, dims=dims.__dict__ | {"moe": None}, max_batched_tokens=a.max_batched_tokens, plan=plan)
    if a.dry_run:
        print(json.dumps(header, indent=1, default=str))
        return 0

    results: dict[str, dict[str, Any]] = {}
    cmds = {}
    for m in modes:
        cmd = [sys.executable, os.path.abspath(__file__), "--model", a.model, "--mode", m, "--batch", a.batch,
               "--ctx", a.ctx, "--n-new", str(a.n_new), "--repeats", str(a.repeats), "--ring-mib", str(a.ring_mib),
               "--max-batched-tokens", str(a.max_batched_tokens), "--gpu-mem-util", str(a.gpu_mem_util),
               "--budget-profile", a.budget_profile, "--store-gib", str(a.store_gib), "--retention", a.retention]
        if a.no_h_mid:
            cmd.append("--no-h-mid")
        cmds[m] = cmd
        print(f"[bench] {m}: {' '.join(cmd)}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("BENCH_RESULT_JSON:")]
        if p.returncode != 0 or not lines:
            sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
            raise SystemExit(f"mode {m} failed (rc={p.returncode})")
        results[m] = json.loads(lines[-1].split("BENCH_RESULT_JSON:", 1)[1])
    report = dict(header, machine=machine_info(), commands=cmds, results=results, summary=summarize(results))
    print(json.dumps(report["summary"], indent=1), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
