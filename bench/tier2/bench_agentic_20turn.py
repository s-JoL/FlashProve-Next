# SPDX-License-Identifier: Apache-2.0
"""Live vLLM 0.33 MoE benchmark: a REAL 20-turn growing agentic session (>= 16k tokens) on live vLLM 0.33 with fpn_capture.

: the per-turn commitment is INCREMENTAL.  `trace_commit.IncrementalTraceCommitter` checks the turn's page
chain against the forwarded token ids, reuses the row leaves of every page committed in an earlier turn and reads /
hashes ONLY the fresh pages (O(fresh tokens) payload hashing), then rebuilds the upper Merkle levels (O(n) 32-byte
nodes); the root is bit-identical to `commit_trace_merkle` (cross-checked at turn 20, which also re-runs the full
re-hash as a same-run baseline).  The fresh pages' leaves feed the `St_i` chain
(`agentic_state.AgenticSession`: `page_hash` of each fresh page = H(first position, count, its row leaves)), and the
per-turn C_pre is `precommit_agentic_turn(setup, St_i record, trace_payload_root, response)`.  After the session
`verify_agentic_chain` re-checks all 20 records WITH the fresh page hashes and the published commitments, plus two
tamper controls (E_CPRE_MISMATCH, E_FRESH_ROOT).  The turn-20 protocol draws / R4 nonces / R5 N_s / independent
verifier use that agentic C_pre (policy bytes = St_20 record).

    FPN_GIT_SHA=$(git rev-parse HEAD) python bench/tier2/bench_agentic_20turn.py \
        --model <MODELS_DIR>/Qwen3-0.6B --fpn-verify core/target/release/fpn-verify \
        --out benchmarks/prover/results/agentic_20turn_rtx6000d.json

One conversation, 20 turns, served by ONE vLLM engine (`load_format="fpn_capture"`, FULL_NO_KV capture,
`enable_prefix_caching=True`, R32 numerics).  Turn k's prompt is turn k-1's prompt + turn k-1's generated tokens
(verbatim, what a chat client sends back) + a fresh tool-result block (a slice of a real source file of this repo,
read as a `read_file` tool response) + a short user instruction, so the context grows to >= 16,384 tokens.

Per turn (all measured):
  * prompt / cached-prefix (vLLM `num_cached_tokens`) / fresh-input / generated tokens and prefix_hit_rate;
  * PageStore write-once accounting: pages and bytes written THIS turn (store stats delta) vs the trace's logical
    pages / bytes (what a naive per-turn re-capture of the whole context would write), retained store bytes, and
    host RSS (current, /proc/self/statm) -- the prefix pages are MOUNTED from earlier turns (hash-chain), never
    re-captured, so the store grows with the session, not with sum-over-turns;
  * C_pre: `commit_trace_merkle` over EVERY captured row of the turn's trace (re-hashed in full each turn: linear in
    context, NOT incremental), `precommit_request`, Protocol v2 `external_seed` + `sample_request` timings.
On turn 20 (full long context): relation 4 (`fpn2-folded-air-v1`, 3 single-block containers under the shared
SetupCertificate circuit) + relation 5 (`fpn2-folded-vec-n22-v1`, 64 drawn 32 x 1024 tiles) on the GPU, native Rust
`fpn-verify` and the `independent_verify` (public inputs + reply bundle only).  The prover / verifier code is
`bench_live_vllm_e2e` UNCHANGED (Qwen3-0.6B geometry).
"""
import argparse
import gc
import glob
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bench_live_vllm_e2e as B  # noqa: E402  (sets sys.path + vLLM env defaults)

import numpy as np  # noqa: E402

NONCE = hashlib.blake2b(b"v1 agentic 20-turn external challenge nonce", digest_size=32).digest()
SYSTEM = ("You are a coding agent working in the FlashProve-Next repository. You call tools (read_file, grep, "
          "run_tests) and reason about their output step by step before answering.")
TASK = ("Audit how the capture page store, the trace commitment and the relation-5 prover fit together. Read the "
        "files I give you one by one and keep a running list of findings.")
INSTRUCTIONS = (
    "Summarise what this file does and note anything that affects soundness.",
    "Which functions here are on the hot path? List them with one line each.",
    "Does this interact with the previous file? Explain the data flow.",
    "Point out one possible edge case and how it is handled.",
    "Update your running list of findings with this file.",
)
PAGE = 16


def git_rev():
    return os.environ.get("FPN_GIT_SHA") or B.git_rev()


def rss_now_mib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def tool_sources():
    """Deterministic list of real repo source files (the 'tool results' of the session)."""
    pats = ("serve/fpn_serve/capture/*.py", "prover/fpn_prover/runtime/trace_commit.py",
            "prover/fpn_prover/runtime/weight_commit.py", "prover/fpn_prover/backend/folded_vec.py",
            "prover/fpn_prover/circuit/protocol.py", "prover/fpn_prover/runtime/pipeline.py",
            "prover/fpn_prover/runtime/setup_cert.py", "prover/fpn_prover/runtime/api_envelope.py",
            "prover/fpn_prover/circuit/e2e_prover.py", "prover/fpn_prover/circuit/lower.py",
            "prover/fpn_prover/backend/cuda_engine.py", "serve/fpn_serve/engines/vllm/capture_plugin.py",
            "bench/tier2/bench_live_vllm_e2e.py", "bench/tier2/bench_frontier_economics.py",
            "verifier/python/*.py")
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(ROOT, p)))
    return [f for f in files if os.path.getsize(f) > 4000]


def expected_gemm_share(profile, positions):
    """Cost share of GEMM atoms over `positions` (PUBLIC geometry: profile + lengths)."""
    g = profile.layers * profile.gemm_cost
    tot = sum(profile.position_cost(p) for p in positions)
    return len(positions) * g / tot if tot else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="<MODELS_DIR>/Qwen3-0.6B")
    ap.add_argument("--fpn-verify", default=os.path.join(ROOT, "core", "target", "release", "fpn-verify"))
    ap.add_argument("--out", default=os.path.join(ROOT, "internal", "results", "m3", "agentic_20turn_rtx6000d.json"))
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--tool-tokens", type=int, default=700, help="tool-result content tokens per turn")
    ap.add_argument("--n-new", type=int, default=160)
    ap.add_argument("--target-ctx", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--per-turn-draws", type=int, default=64)
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--r4-blocks", type=int, default=3)
    ap.add_argument("--draw-margin", type=float, default=1.6)
    ap.add_argument("--ring-mib", type=int, default=1536)
    ap.add_argument("--max-batched-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.35)
    ap.add_argument("--r5-reps", type=int, default=2)
    ap.add_argument("--commit-threads", type=int, default=16)
    ap.add_argument("--inc-threads", type=int, default=16, help="fresh-page hashing workers (incremental commit)")
    ap.add_argument("--v1-turn20-commit-ms", type=float, default=25571.55,
                    help="receipt: commit_trace_merkle wall time on turn 20 (benchmarks/m3)")
    ap.add_argument("--dry-run", action="store_true", help="build the 20 tool turns with the tokenizer only")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    srcs = tool_sources()

    def fresh_block(k):
        f = srcs[(k - 1) % len(srcs)]
        ids = tok.encode(open(f).read(), add_special_tokens=False)[:args.tool_tokens]
        body = tok.decode(ids)
        rel = os.path.relpath(f, ROOT)
        txt = (f"<tool_call>\n{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{rel}\"}}}}\n</tool_call><|im_end|>\n"
               f"<|im_start|>user\n<tool_response>\n{body}\n</tool_response>\n{INSTRUCTIONS[(k - 1) % len(INSTRUCTIONS)]}"
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        if k == 1:
            txt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{TASK}<|im_end|>\n<|im_start|>assistant\n"
                   + txt)
        return list(tok.encode(txt, add_special_tokens=False)), rel

    if args.dry_run:
        n = 0
        for k in range(1, args.turns + 1):
            ids, rel = fresh_block(k)
            n += len(ids) + args.n_new
            print(k, rel, len(ids), n)
        print("projected final context", n, ">= target" if n >= args.target_ctx else "< target")
        return

    from flashprove_next.field import native, native_threads
    if native() is None:
        raise SystemExit("flashprove_next native library unavailable (would time the NumPy fallback)")
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from fpn_prover.circuit import protocol
    from fpn_prover.runtime import agentic_state as AS
    from fpn_prover.runtime import trace_commit as TC
    from fpn_prover.runtime.pipeline import precommit_request
    from fpn_serve.capture.ports import CaptureTier
    from fpn_serve.capture.store import block_hash_chain, partial_block_hash
    from fpn_serve.engines.vllm import capture_plugin as cp

    res = {"bench": "tier2/bench_agentic_20turn", "wave": "13A", "milestone": "v1", "evidence": "MEASURED",
           "git_rev": git_rev(), "command": " ".join(sys.argv),
           "machine": {"host": platform.node(), "python": platform.python_version(), "native_field_lib": True,
                       "native_threads": native_threads()}}
    try:
        res["machine"]["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    part = args.out + ".turns.jsonl"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(part, "w").close()

    cp.configure_capture(cp.CaptureConfig(tier=CaptureTier.FULL_NO_KV, ring_mib=args.ring_mib,
                                          max_batched_tokens=args.max_batched_tokens))
    t0 = time.perf_counter()
    llm = LLM(model=args.model, load_format="fpn_capture", dtype="bfloat16", enforce_eager=False,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len, max_num_seqs=4,
              max_num_batched_tokens=args.max_batched_tokens, enable_prefix_caching=True, seed=0)
    runner = llm.collective_rpc(cp.install_runner)
    t_load = time.perf_counter() - t0
    sess = cp.get_session()
    store = sess.store
    eng = llm.llm_engine
    rss_after_load = rss_now_mib()
    res["serving"] = {"engine": "vLLM " + __import__("vllm").__version__, "model": os.path.basename(args.model.rstrip("/")),
                      "runner": runner, "load_format": "fpn_capture", "capture_tier": "FULL_NO_KV",
                      "enable_prefix_caching": True, "row_bytes": store.row_bytes, "page_tokens": PAGE,
                      "allow_bf16_reduced_precision_reduction":
                          torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                      "max_num_batched_tokens": args.max_batched_tokens, "max_model_len": args.max_model_len,
                      "llm_init_s": round(t_load, 2), "n_new_per_turn": args.n_new,
                      "tool_tokens_per_turn": args.tool_tokens, "host_rss_mib_after_load": round(rss_after_load, 1)}

    def run(rid, ids, n):
        eng.add_request(rid, TokensPrompt(prompt_token_ids=ids),
                        SamplingParams(n=1, temperature=0.0, max_tokens=n, ignore_eos=True, detokenize=False))
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = None
        waited = 0.0
        while out is None:
            # HARNESS back-pressure: the capture path never blocks (it fails closed on overflow), so the bench holds
            # the next engine step until the staging ring has room for two full steps; the wait is recorded
            if sess.ring.free_rows < 2 * args.max_batched_tokens:
                tw = time.perf_counter()
                while sess.ring.free_rows < 2 * args.max_batched_tokens and time.perf_counter() - tw < 30.0:
                    sess.sink.check()
                    time.sleep(0.0005)
                waited += time.perf_counter() - tw
            for o in eng.step():
                if o.finished and o.request_id == rid:
                    out = o
        torch.cuda.synchronize()
        return out, time.perf_counter() - t, waited

    # ONE-TIME setup (before the session): global weight commitment + the shared relation-4 circuit
    from fpn_prover.circuit import e2e_prover as E
    from fpn_prover.circuit import lower as LW
    from fpn_prover.runtime import weight_commit as WC
    from fpn_prover.runtime.setup_cert import SetupCertificate
    t = time.perf_counter()
    wcom = WC.commit_model_weights(args.model, threads=args.commit_threads)
    t_wc = (time.perf_counter() - t) * 1e3
    prog, _ = E.build_program()
    circuit = LW.compile_air(prog)
    setup = SetupCertificate(wcom.root, circuit.root, hashlib.blake2b(b"bf16_fp32acc", digest_size=32).digest(),
                             hashlib.blake2b(b"Qwen3-0.6B/safetensors/bf16", digest_size=32).digest(), 0).encode()
    def chain_hashes(root, toks):
        hs = block_hash_chain(root, toks)
        tail = len(toks) % PAGE
        if tail:
            hs.append(partial_block_hash(hs[-1] if hs else root, toks[len(toks) - tail:]))
        return hs

    layout_fp = sess.layout.fingerprint()
    committer = TC.IncrementalTraceCommitter(store, chain_hashes, threads=args.inc_threads, page_tokens=PAGE)
    ags = AS.AgenticSession(setup, layout_fp, page_tokens=PAGE)
    records, fresh_pages_all, commitments = [], [], []
    turns, prompt, prev_out, prev_key = [], [], [], None
    final = None
    for k in range(1, args.turns + 1):
        fresh, rel = fresh_block(k)
        prompt = prompt + prev_out + fresh
        rid = f"turn{k:02d}"
        st0 = dict(store.stats)
        b0 = store.bytes_stored
        o, t_serve, t_wait = run(rid, prompt, args.n_new)
        out = list(o.outputs[0].token_ids)
        cached = int(o.num_cached_tokens or 0)
        t = time.perf_counter()
        settled = sess.settle(300.0)
        t_settle = (time.perf_counter() - t) * 1e3
        keys = [x for x in settled if x == rid or x.startswith(rid + "-")]
        if len(keys) != 1 or isinstance(settled[keys[0]], str):
            raise SystemExit(f"turn {k}: capture did not settle cleanly: {settled}")
        key, trace = keys[0], settled[keys[0]]
        token_ids = prompt + out[:-1]
        st1 = dict(store.stats)
        logical_pages = trace.n_full_pages + (1 if trace.n_tokens % PAGE else 0)
        written_pages = st1["pages_written"] - st0["pages_written"]
        written_bytes = st1["bytes_written"] - st0["bytes_written"]
        naive_bytes = trace.n_tokens * store.row_bytes
        last = k == args.turns
        # ---- incremental per-turn commitment: fresh pages only -> trace root + St_i + agentic C_pre
        t_inc0 = time.perf_counter()
        # rows are NOT retained at commit time (every turn identical): drawn rows are re-read from the write-once
        # store after the challenge (runs 1-2 retained them on turn 20 only, which roughly doubled that
        # turn's commit wall time through fresh-allocation page faults; disclosed in the receipt)
        inc = committer.commit(trace, key, token_ids)
        t = time.perf_counter()
        prefix_tok = inc.prefix_pages * PAGE
        fph = [AS.page_hash(j * PAGE, min(PAGE, trace.n_tokens - j * PAGE), inc.page_digest_payload(j))
               for j in range(inc.prefix_pages, len(inc.page_leaves))]
        t_pages = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        st_turn = ags.begin_turn(prefix_tok, trace.n_tokens - prefix_tok, fph, out)
        t_st = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        c_pre = ags.precommit(st_turn, inc.tree.root, out)
        t_cpre = (time.perf_counter() - t) * 1e3
        t_inc_total = (time.perf_counter() - t_inc0) * 1e3
        tree = inc.tree
        records.append(st_turn)
        fresh_pages_all.append(fph)
        commitments.append(AS.TurnCommitment(tree.root, tuple(out), b"", c_pre))
        t_commit = None
        full_tree = None
        if last:                      # same-run baselines + bit-identity cross-check (NOT in the per-turn timing)
            t = time.perf_counter()
            full_tree = TC.commit_trace_merkle(store, trace, key, token_ids, retain_rows=True)
            t_commit = (time.perf_counter() - t) * 1e3
            fresh_committer = TC.IncrementalTraceCommitter(store, chain_hashes, threads=args.inc_threads,
                                                           page_tokens=PAGE)
            t = time.perf_counter()
            full_threaded = fresh_committer.commit(trace, key, token_ids)
            t_full_threaded = (time.perf_counter() - t) * 1e3
            fresh_committer.close()
            if full_tree.root != tree.root or full_threaded.tree.root != tree.root:
                raise SystemExit("incremental trace root differs from commit_trace_merkle (turn 20)")
        import struct
        t = time.perf_counter()
        seed = protocol.external_seed(struct.pack(f"<{len(token_ids)}I", *token_ids),
                                      struct.pack(f"<{len(out)}I", *out), c_pre, NONCE)
        t_seed = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        d = protocol.sample_request(seed, B.sampler_profile(), input_tokens=len(prompt), cache_read_input_tokens=cached,
                                    output_tokens=len(out), draws=args.per_turn_draws)
        t_sample = (time.perf_counter() - t) * 1e3
        fresh_draws = [x for x in d[protocol.STRATUM_FRESH_INPUT] if not x.empty]
        assert all(x.position >= cached for x in fresh_draws), "a cached-prefix position received a draw"
        if prev_key is not None:
            sess.forget([prev_key])                       # release the previous turn's handle (shared pages stay)
        row = {"turn": k, "request_id": key, "tool_file": rel, "prompt_tokens": len(prompt),
               "cached_prefix_tokens": cached, "fresh_input_tokens": len(prompt) - cached,
               "new_output_tokens": len(out), "prefix_hit_rate": round(cached / len(prompt), 5),
               "trace_tokens": trace.n_tokens, "serve_s": round(t_serve, 4),
               "serve_s_of_which_ring_backpressure_wait": round(t_wait, 4), "settle_ms": round(t_settle, 2),
               "pages_written_this_turn": written_pages, "total_logical_pages": logical_pages,
               "bytes_written_this_turn": written_bytes, "naive_full_recapture_bytes": naive_bytes,
               "write_once_dedup_ratio": round(naive_bytes / written_bytes, 3) if written_bytes else None,
               "store_dedup_hits_this_turn": st1["dedup_hits"] - st0["dedup_hits"],
               "pagestore_retained_bytes": store.bytes_stored, "pagestore_retained_delta": store.bytes_stored - b0,
               "pagestore_pages": store.n_pages() if callable(getattr(store, "n_pages", None)) else None,
               "host_rss_mib": round(rss_now_mib(), 1),
               "trace_payload_root": tree.root.hex(), "c_pre": c_pre.hex(), "trace_tip": trace.tip.hex(),
               "c_pre_rule": "precommit_agentic_turn(setup, St_i record, trace_payload_root, response)",
               "st_i": {"turn_idx": st_turn.turn_idx, "prefix_tokens": st_turn.prefix_tokens,
                        "fresh_tokens": st_turn.fresh_tokens, "total_tokens": st_turn.total_tokens,
                        "fresh_pages": len(fph), "reused_pages": inc.prefix_pages,
                        "prefix_tokens_minus_vllm_cached": prefix_tok - cached,
                        "state": st_turn.state.hex(), "record_bytes": AS.AgenticTurn.SIZE},
               "incremental_commit": {**{k2: (round(v, 3) if isinstance(v, float) else v)
                                         for k2, v in inc.stats.items()},
                                      "fresh_page_hash_ms": round(t_pages, 3), "st_i_begin_turn_ms": round(t_st, 3),
                                      "precommit_agentic_turn_ms": round(t_cpre, 3),
                                      "total_ms": round(t_inc_total, 2)},
               "timing_ms": {"incremental_commitment_total": round(t_inc_total, 2),
                             "commit_trace_merkle_full_rehash": None if t_commit is None else round(t_commit, 2),
                             "precommit_agentic_turn": round(t_cpre, 3),
                             "external_seed": round(t_seed, 3), "sample_request": round(t_sample, 2)},
               "per_turn_draws": args.per_turn_draws, "fresh_stratum_nonempty_draws": len(fresh_draws)}
        turns.append(row)
        with open(part, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({x: row[x] for x in ("turn", "prompt_tokens", "cached_prefix_tokens", "prefix_hit_rate",
                                              "pages_written_this_turn", "total_logical_pages", "serve_s",
                                              "host_rss_mib", "timing_ms")}), flush=True)
        if last:
            row["turn20_baselines_ms"] = {
                "baseline_full_commit_trace_merkle_ms": args.baseline_turn20_commit_ms,
                "same_run_commit_trace_merkle_single_thread_full_rehash": round(t_commit, 2),
                "same_run_full_rehash_threaded": round(t_full_threaded, 2),
                "incremental_total": round(t_inc_total, 2),
                "speedup_vs_baseline_full_commit": round(args.baseline_turn20_commit_ms / t_inc_total, 1),
                "speedup_vs_same_run_single_thread_full": round(t_commit / t_inc_total, 1),
                "speedup_vs_same_run_threaded_full": round(t_full_threaded / t_inc_total, 2),
                "fresh_token_fraction": round(st_turn.fresh_tokens / st_turn.total_tokens, 5),
                "roots_bit_identical": True}
            print(json.dumps(row["turn20_baselines_ms"]), flush=True)
            final = dict(tree=full_tree, trace=trace, key=key, token_ids=token_ids, prompt=prompt, out=out,
                         cached=cached, st_turn=st_turn)
        prev_out, prev_key = out, key

    agg_prompt = sum(r["prompt_tokens"] for r in turns)
    agg_cached = sum(r["cached_prefix_tokens"] for r in turns)
    agg_written = sum(r["bytes_written_this_turn"] for r in turns)
    agg_naive = sum(r["naive_full_recapture_bytes"] for r in turns)
    res["turns"] = turns
    res["aggregate"] = {
        "turns": len(turns), "final_context_tokens": turns[-1]["trace_tokens"],
        "final_prompt_tokens": turns[-1]["prompt_tokens"],
        "final_context_ge_target": turns[-1]["trace_tokens"] >= args.target_ctx,
        "prefix_hit_rate_turn20": turns[-1]["prefix_hit_rate"],
        "prefix_hit_rate_token_weighted_all_turns": round(agg_cached / agg_prompt, 5),
        "prefix_hit_rate_mean_unweighted_all_turns": round(statistics.mean(r["prefix_hit_rate"] for r in turns), 5),
        "prefix_hit_rate_mean_unweighted_turns_2_to_20": round(statistics.mean(r["prefix_hit_rate"]
                                                                               for r in turns[1:]), 5),
        "sum_pages_written": sum(r["pages_written_this_turn"] for r in turns),
        "sum_logical_pages_over_turns": sum(r["total_logical_pages"] for r in turns),
        "sum_bytes_written": agg_written, "sum_naive_full_recapture_bytes": agg_naive,
        "session_write_once_dedup_ratio": round(agg_naive / agg_written, 3),
        "final_pagestore_retained_bytes": turns[-1]["pagestore_retained_bytes"],
        "final_trace_bytes": turns[-1]["naive_full_recapture_bytes"],
        "retained_over_final_trace": round(turns[-1]["pagestore_retained_bytes"] / turns[-1]["naive_full_recapture_bytes"], 4),
        "host_rss_mib_after_load": round(rss_after_load, 1), "host_rss_mib_final": turns[-1]["host_rss_mib"],
        "sink_stats": {k: v for k, v in dict(sess.sink.stats).items() if isinstance(v, (int, float))},
        "store_stats": dict(store.stats), "ring_overflow_events": sess.ring.overflow_events,
        "serve_s_total": round(sum(r["serve_s"] for r in turns), 3),
        "incremental_commitment_ms_total": round(sum(r["timing_ms"]["incremental_commitment_total"] for r in turns), 1),
        "incremental_commitment_ms_turn20": turns[-1]["timing_ms"]["incremental_commitment_total"],
        "baseline_full_commit_trace_merkle_ms_total": 199222.2,
        "definitions": {
            "prefix_hit_rate": "cached_prefix_tokens / prompt_tokens (vLLM num_cached_tokens)",
            "write_once_dedup_ratio": "naive_full_recapture_bytes (trace_tokens x row_bytes, what re-capturing the "
                                      "whole context every turn would write) / bytes the PageStore actually wrote "
                                      "this turn (store stats delta)",
            "store_dedup_hits": "content-hash collisions on write (re-fed rows); ~0 here because the prefix is not "
                                "re-computed at all: vLLM's prefix cache skips it and the capture MOUNTS the prefix "
                                "pages from the hash chain",
            "pagestore_retained_bytes": "PageStore.bytes_stored after the turn (previous turn's handle released, "
                                        "shared prefix pages stay)",
            "host_rss_mib": "current RSS from /proc/self/statm (not the ru_maxrss peak)"}}
    print(json.dumps(res["aggregate"], indent=1, default=str), flush=True)
    committer.close()

    # =================================================================================== St_i chain verification
    t = time.perf_counter()
    chain_ok = AS.verify_agentic_chain(setup, layout_fp, records, fresh_pages=fresh_pages_all, commitments=commitments)
    t_chain = (time.perf_counter() - t) * 1e3
    bad_c = list(commitments)
    c9 = bad_c[9]
    bad_c[9] = AS.TurnCommitment(c9.trace_payload_root, c9.response_tokens, c9.policy_bytes,
                                 bytes([c9.c_pre[0] ^ 1]) + c9.c_pre[1:])
    tamper_cpre = AS.verify_agentic_chain(setup, layout_fp, records, fresh_pages=fresh_pages_all, commitments=bad_c)
    bad_p = [list(x) for x in fresh_pages_all]
    bad_p[14][0] = bytes([bad_p[14][0][0] ^ 1]) + bad_p[14][0][1:]
    tamper_page = AS.verify_agentic_chain(setup, layout_fp, records, fresh_pages=bad_p, commitments=commitments)
    old_rule = list(commitments)                      # C_pre WITHOUT St_i (the rule) is not accepted
    c0 = old_rule[0]
    old_rule[0] = AS.TurnCommitment(c0.trace_payload_root, c0.response_tokens, b"",
                                    precommit_request(setup, c0.trace_payload_root, list(c0.response_tokens)))
    tamper_rule = AS.verify_agentic_chain(setup, layout_fp, records, fresh_pages=fresh_pages_all,
                                          commitments=old_rule)
    acct = AS.session_accounting(records, store.row_bytes, PAGE)
    res["st_i_chain"] = {
        "verify_agentic_chain_with_fresh_pages_and_commitments": {k2: (v.hex() if isinstance(v, bytes) else v)
                                                                  for k2, v in chain_ok.items()},
        "verify_ms": round(t_chain, 2),
        "tamper_turn10_c_pre": tamper_cpre["verdict"], "tamper_turn10_c_pre_turn": tamper_cpre["turn_idx"],
        "tamper_turn15_first_fresh_page_hash": tamper_page["verdict"],
        "tamper_turn15_turn": tamper_page["turn_idx"],
        "turn1_c_pre_without_st_i": tamper_rule["verdict"],
        "records_bytes_total": AS.AgenticTurn.SIZE * len(records),
        "tip_state": records[-1].state.hex(), "turn20_record_hex": records[-1].encode().hex(),
        "accounting": acct,
        "page_hash_rule": "page_hash(first_position, n_tokens, concat of the page's FPN2-TROW-v1 row leaves): the "
                          "ledger and trace_payload_root are built from the SAME per-row leaves",
        "prefix_rule": "prefix_tokens = leading pages whose row leaves were committed in an earlier turn (page "
                       "aligned, <= previous total); recorded against vLLM num_cached_tokens per turn"}
    print(json.dumps({k2: v for k2, v in res["st_i_chain"].items() if k2 not in ("accounting", "turn20_record_hex")},
                     default=str), flush=True)
    if chain_ok["verdict"] != "VERIFIED" or tamper_cpre["verdict"] != "E_CPRE_MISMATCH" or \
            tamper_page["verdict"] != "E_FRESH_ROOT" or tamper_rule["verdict"] != "E_CPRE_MISMATCH":
        json.dump(res, open(args.out, "w"), indent=1, default=str)
        raise SystemExit("St_i chain verification did not give the expected verdicts")

    # =================================================================================== turn 20: prove + verify
    from fpn_prover.backend import cuda_engine as CE
    from fpn_prover.backend import folded_vec as FV
    from fpn_prover.backend.fpn2 import FOLDED_AIR_V1, FOLDED_VEC_N22_V1
    from fpn_prover.runtime.pipeline import verify_reply_with_setup
    F = final
    payload_root = F["tree"].root
    st20 = AS.AgenticTurn.decode(F["st_turn"].encode())          # the verifier's copy: the public 192-byte record
    policy = AS.agentic_policy_bytes(st20)
    prof = B.sampler_profile()
    gen = protocol.generation_stratum(len(F["prompt"]), len(F["out"]))
    fr = protocol.fresh_input_stratum(len(F["prompt"]), F["cached"])
    fr = range(fr.start, min(fr.stop, gen.start))
    share = (expected_gemm_share(prof, fr) + expected_gemm_share(prof, gen))  # expected GEMM draws per draw-pair
    draws = int(math.ceil(args.draw_margin * args.tiles / share))
    t = time.perf_counter()
    public = (payload_root, F["token_ids"], F["out"], len(F["prompt"]), F["cached"], draws)
    c_pre, seed, flat = B.protocol_draws(setup, *public[:5], draws, nonce=NONCE, policy_bytes=policy)
    if c_pre != commitments[-1].c_pre:
        raise SystemExit("turn-20 protocol C_pre differs from the published agentic C_pre")
    t_draw = (time.perf_counter() - t) * 1e3
    gemm_draws = [d for d in flat if d.kind == "gemm"][:args.tiles]
    proof = {"draws_per_stratum": draws, "draw_rule": f"ceil({args.draw_margin} x {args.tiles} / expected GEMM draws "
             "per (fresh, generation) draw pair), computed from the public sampler geometry and lengths BEFORE the "
             "seed exists", "expected_gemm_draws_per_pair": round(share, 5), "nonempty_draws": len(flat),
             "gemm_draws": sum(d.kind == "gemm" for d in flat), "attn_draws": sum(d.kind == "attn" for d in flat),
             "min_drawn_position": min(d.position for d in flat), "cached_prefix_positions_without_draws": F["cached"],
             "c_pre": c_pre.hex(), "seed": seed.hex(), "trace_payload_root": payload_root.hex(),
             "setup_cert_id": SetupCertificate.decode(setup).id, "weight_commitment_root": wcom.root.hex(),
             "circuit_root": circuit.root.hex(), "global_weight_commit_wall_ms": round(t_wc, 1),
             "timing_ms": {"protocol_draws": round(t_draw, 2)}}
    assert min(d.position for d in flat) >= F["cached"]
    if len(gemm_draws) < args.tiles:
        proof["error"] = f"only {len(gemm_draws)} GEMM draws < {args.tiles}"
        res["turn20_proof"] = proof
        json.dump(res, open(args.out, "w"), indent=1, default=str)
        raise SystemExit(proof["error"])
    t = time.perf_counter()
    bundle = TC.open_trace_rows(F["tree"], {d.position for d in flat})
    trace_openings = bundle.encode()
    proof["timing_ms"]["open_trace_rows"] = round((time.perf_counter() - t) * 1e3, 1)
    proof["trace_row_openings"] = {"positions": len(bundle.openings), "bytes": len(trace_openings)}
    t = time.perf_counter()
    rows = B.read_rows(store, F["tree"].pages, {d.position for d in flat})
    proof["timing_ms"]["rows_for_drawn_positions"] = round((time.perf_counter() - t) * 1e3, 1)
    lw = B.live_weights(llm, sorted({d.layer for d in flat}))
    engine = CE.CudaEngine()
    samples, r4meta = B.r4_samples(flat, rows, lw, args.r4_blocks)
    pw = lambda p, kk, l: rows[p][(kk, l)]  # noqa: E731
    r4_disc = [{"position": m["position"], "layer": m["layer"], "norm_channels": m["norm_channels"],
                "in_col": m["gemm"]["in_col"], "slot_dim": m["slot_dim"]} for m in r4meta]
    r4, r4_receipts = [], []
    for i, s in enumerate(samples):
        nonce_i = B.r4_nonce(c_pre, r4_disc[i], pw)
        t = time.perf_counter()
        pr = E.prove_samples(s, server_nonce=nonce_i, engine=engine)
        wall = (time.perf_counter() - t) * 1e3
        assert pr.circuit.root == circuit.root
        r4_receipts.append(pr.container)
        ver = verify_reply_with_setup(setup, pr.container, expected_c_pre=nonce_i, circuit_digest=pr.circuit.digest,
                                      rust_binary=args.fpn_verify, check_python=True)
        r4.append({**r4meta[i], "prove_wall_ms": round(wall, 1), "container_bytes": len(pr.container), "verify": ver})
        print(json.dumps(r4[-1], default=str), flush=True)
    W_int, X_int = [], []
    for d in gemm_draws:
        wi, _ = FV.bf16_words_to_ints(B.live_tile(lw, d.layer, d.gemm, d.tile_start))
        xi, _ = FV.bf16_words_to_ints(rows[d.position][(B.x_slot(d), d.layer)])
        W_int.append(wi)
        X_int.append(xi)
    W_int = np.array(W_int, dtype=object if any(a.dtype == object for a in W_int) else np.int64)
    X_int = np.array(X_int, dtype=object if any(a.dtype == object for a in X_int) else np.int64)
    lay = FV.layout_of(FOLDED_VEC_N22_V1)
    ys, bits = [], []
    for b in range(lay.B):
        yb, bb = FV.exact_products(W_int[b:b + 1], X_int[b:b + 1])
        ys.append(yb[0])
        bits.append(bb)
    Wf, Xf, Yf = FV.to_field(W_int), FV.to_field(X_int), FV.to_field(np.array(ys, dtype=object))
    n_s = B.rel5_n_s(c_pre, gemm_draws, B.rel5_x_digest(gemm_draws, lambda d: rows[d.position][(B.x_slot(d), d.layer)]))
    r5_runs = []
    for rep in range(args.r5_reps):
        FV._W_CACHE.clear()
        t = time.perf_counter()
        r5 = FV.prove(Wf, Xf, Yf, n_s=n_s, profile=FOLDED_VEC_N22_V1, engine=engine)
        r5_runs.append(round((time.perf_counter() - t) * 1e3, 1))
        print(json.dumps({"r5_rep": rep, "prove_wall_ms": r5_runs[-1]}), flush=True)
    opening = WC.open_drawn_tiles(wcom, args.model, B.tile_coords(gemm_draws)).encode()
    chk = WC.verify_drawn_tiles_against_setup(setup, opening, r5, B.tile_coords(gemm_draws), engine=engine)
    rs5 = E.verdict_code(E.rust_verify(args.fpn_verify, r5, bytes.fromhex(chk["weight_root"]))[0]) \
        if chk["verdict"] == "VERIFIED" else chk["verdict"]
    t = time.perf_counter()
    iv = B.independent_verify(setup, c_pre, payload_root, F["token_ids"], F["out"], len(F["prompt"]), F["cached"], draws,
                              args.tiles, NONCE, sess.layout, trace_openings, opening, r4_receipts, r4_disc, r5,
                              args.fpn_verify, engine=engine, policy_bytes=policy)
    t_iv = (time.perf_counter() - t) * 1e3
    # one forgery control: a tampered drawn row in the opening bundle
    o0 = bundle.openings[0]
    bad_row = bytearray(o0.row)
    bad_row[0] ^= 0x01
    forged = TC.TraceRowOpenings(bundle.n_tokens, bundle.layout_fingerprint, bundle.row_bytes,
                                 (TC.TraceRowOpening(o0.position, bytes(bad_row), o0.path),) + bundle.openings[1:]).encode()
    iv_forged = B.independent_verify(setup, c_pre, payload_root, F["token_ids"], F["out"], len(F["prompt"]), F["cached"],
                                     draws, args.tiles, NONCE, sess.layout, forged, opening, r4_receipts, r4_disc, r5,
                                     args.fpn_verify, engine=engine, policy_bytes=policy)["verdict"]
    # the reply's agentic C_pre checked WITHOUT St_20 (rule) -> must not verify
    iv_no_st = B.independent_verify(setup, c_pre, payload_root, F["token_ids"], F["out"], len(F["prompt"]), F["cached"],
                                    draws, args.tiles, NONCE, sess.layout, trace_openings, opening, r4_receipts,
                                    r4_disc, r5, args.fpn_verify, engine=engine)["verdict"]
    proof.update({
        "relation4": {"profile": FOLDED_AIR_V1["name"], "blocks": r4,
                      "note": "3 single-block fpn2-folded-air-v1 containers (bench_live_vllm_e2e path); NOT the "
                              "64-block fpn2-folded-air-n20-v1 container priced in bench_frontier_economics"},
        "relation5": {"profile": FOLDED_VEC_N22_V1["name"], "tiles": lay.B, "macs": lay.macs,
                      "container_bytes": len(r5), "prove_wall_ms_runs": r5_runs,
                      "integer_bound_bits_max": max(bits), "integer_faithful": max(bits) <= 62,
                      "tiles_by_gemm": {g: sum(B.GEMMS[d.gemm][0] == g for d in gemm_draws) for g, _ in B.GEMMS},
                      "weight_tiles_vs_setup": chk["verdict"], "rust_fpn_verify_pinned": rs5,
                      "weight_opening_bytes": len(opening)},
        "independent_verifier": {"honest": iv, "verify_ms": round(t_iv, 1),
                                 "forgery_tampered_row_in_opening": iv_forged,
                                 "verifier_ignoring_st_20_policy_bytes": iv_no_st,
                                 "c_pre_policy": "agentic_policy_bytes(St_20 record) -- the verifier decodes the "
                                                 "public 192-byte record"},
    })
    res["turn20_proof"] = proof
    print(json.dumps({"independent_verifier": iv["verdict"], "stages": iv["stages"], "forged": iv_forged},
                     default=str), flush=True)
    res["host"] = {"peak_rss_mib_ru_maxrss": round(B.rss_mib(), 1), "rss_mib_now": round(rss_now_mib(), 1),
                   "total_bench_s": round(time.perf_counter() - t_start, 1)}
    res["notes"] = [
        "model: Qwen3-0.6B. The turn-20 prover / verifier path is bench_live_vllm_e2e unchanged, whose relation-5 "
        "tiles (K = 1024 = d_model), sampler profile and relation-4 lane choices are Qwen3-0.6B geometry; Qwen3-4B "
        "(d_model 2560) would need K-chunked tiles + a new profile (not done here) and captures 2.0 MB/token vs "
        "0.66 MB/token (about 35 GB of host PageStore at 17k tokens)",
        "the tool results are slices of real source files of this repository read as read_file responses; the "
        "assistant turns are the model's own greedy outputs (ignore_eos, fixed n_new), fed back verbatim",
        ": the per-turn commitment is incremental (IncrementalTraceCommitter): payload hashing is O(fresh "
        "tokens) (fresh pages only, --inc-threads workers), the upper Merkle levels are rebuilt over all n leaves "
        "(O(n) 32-byte hashes, `tree_levels_ms`), the token chain is re-checked over all pages (O(n) 64-byte "
        "hashes, `chain_check_ms`); the turn-20 root is cross-checked bit-identical to commit_trace_merkle, which is "
        "re-run there as the same-run single-thread baseline, together with a same-thread-count full re-hash "
        "(`same_run_full_rehash_threaded`) so the algorithmic (fresh-only) and the threading gains are separable",
        "per-turn C_pre = precommit_agentic_turn(real SetupCertificate, St_i record, trace_payload_root, response); "
        "St_i chains (prefix, fresh, total, cached-prefix root over the page ledger, fresh-pages root, response "
        "digest) to St_{i-1}; the one-time global weight commitment + circuit compile run BEFORE the session",
        "relation-4 blocks: as bench_live_vllm_e2e (dense_gqa slot lanes filled from captured q / o; the slot "
        "exercises the format, it does not check attention); relation 5 proves exact integer Y = W X over the drawn "
        "tiles, and its hidden witness X is bound to the opened rows only through N_s (scope)",
        "at a 16k context the cost-weighted sampler puts most draws on attention atoms, so 64 GEMM tiles need several "
        "hundred draws per stratum and the trace-row openings carry every drawn row in plaintext (bytes recorded)",
        "serve_s includes a HARNESS back-pressure wait (recorded per turn): before every engine step the bench waits "
        "until the staging ring has >= 2 x max_num_batched_tokens free rows, because the capture path itself never "
        "blocks and fails closed on overflow (two attempts without it, at 512- and 128-token steps, poisoned turn 1 "
        "with ring overflow: page-store ingest of 0.66 MB rows is slower than vLLM's 0.6B prefill steps)",
        "max_num_batched_tokens = 128: a first attempt at 512 (the bench_live_vllm_e2e default) failed CLOSED on turn 1 "
        "(ring overflow -> request poisoned, nothing zero-filled): the FULL_NO_KV ring is capped at 350 MiB = 576 rows "
        "of 0.66 MB, and back-to-back 512-token prefill chunks outran the DMA sink; 128-token chunks keep >= 4 steps "
        "of headroom",
    ]
    json.dump(res, open(args.out, "w"), indent=1, default=str)
    print("WROTE", args.out, flush=True)
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
