# SPDX-License-Identifier: Apache-2.0
"""/ 10A: live vLLM 0.33 serving -> PageStore -> Protocol v2 -> GPU Relation 4 + full-width Relation 5 -> verify.

    python bench/tier2/bench_live_vllm_e2e.py --model <MODELS_DIR>/Qwen3-0.6B \
        --fpn-verify core/target/release/fpn-verify --out benchmarks/prover/results/live_vllm_e2e_rtx6000d.json

1. Serve REAL requests with vLLM 0.33 (`load_format="fpn_capture"`, FULL_NO_KV capture, prefix caching ON,
   `allow_bf16_reduced_precision_reduction=False`): turn 1, then turn 2 (a multi-turn follow-up whose prompt
   extends turn 1 -> prefix-cache hit, mounted capture prefix) batched with an independent request.
2. Settle the PageStore traces and read the captured BF16 rows of the drawn positions through
   `iter_verified_pages` (token-chain checked).
3. Protocol v2 (`external_seed`) samples (pos, layer, tile) draws from the turn-2 trace (fresh-input + generation
   strata; the cached prefix gets no draws).  : C_pre = `precommit_request(setup, commit_trace_payload(...),
   response)` binds the FULL captured activation payload (every full-page and partial-tail row byte, read through
   `iter_verified_pages`) BEFORE the seed is derived; the SetupCertificate carries the ONE-TIME global
   `weight_commit.commit_model_weights` root of the checkpoint (`weight_commitment_root`).
4. GPU (`CudaEngine`) proofs:
   * Relation 4 (`fpn2-folded-air-v1`, shared SetupCertificate circuit): one container per sampled block;
   * Relation 5 (`fpn2-folded-vec-n22-v1`): 64 drawn full-width 32 x 1024 GEMM tiles (2,097,152 MACs) of the
     captured `x_attn` / `x_mlp` rows (the exact q/k/v_proj and gate/up_proj inputs) x the live model's weights.
5. Verify with native Rust `fpn-verify` and the detached `-I` Python reference.  : the 64 drawn tiles are
   opened against SetupCertificate.weight_commitment_root (`open_drawn_tiles`) and the verifying side runs
   `verify_drawn_tiles_against_setup` with ONLY the setup record -> the relation-5 weight_root it pins in fpn-verify
   (the prover took its tiles from vLLM's fused parameters, so a root match also proves live == committed tiles).
   Negative cheat: one captured activation element perturbed in the PageStore row handed to the prover.
6. : `trace_payload_root` is the Merkle root over every captured token row (`trace_commit.commit_trace_merkle`);
   every drawn row is opened against it (`open_trace_rows`) and `independent_verify` -- public inputs + the reply
   bundle only -- re-derives C_pre and the draws, checks the row openings (E_TRACE_ROW_PATH), recomputes the digest
   of the exact X / relation-4 activation words from the OPENED rows (bound into relation-5 N_s and each relation-4
   server nonce), checks the weight-tile openings against the SetupCertificate and runs Rust fpn-verify on R4 + R5.
"""
import argparse
import base64
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import struct
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("prover", "serve", "verifier/python", "."):
    sys.path.insert(0, os.path.join(ROOT, p))

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import numpy as np  # noqa: E402

# Qwen3-0.6B GEMMs with inner dimension K = 1024 (the relation-5 tile width): q, k, v (attention), gate, up (MLP).
GEMMS = (("q_proj", 2048), ("k_proj", 1024), ("v_proj", 1024), ("gate_proj", 3072), ("up_proj", 3072))
ATTN_GEMMS = 3
TURN1 = "Explain in detail how a transformer decoder layer computes attention, step by step, with equations."
TURN2 = "Thanks. Now summarise that in exactly three short sentences for a beginner."
SIDE = "Write a haiku about verifying computations with zero-knowledge proofs."
NONCE = hashlib.blake2b(b"v1 external challenge nonce", digest_size=32).digest()
N_S_DOMAIN = b"FPN2/rel5/N_s/v2"
X_DOMAIN = b"FPN2/rel5/X/v1"
R4_DOMAIN = b"FPN2/rel4/xbind/v1"


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def git_rev():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return open(os.path.join(ROOT, ".rev")).read().strip() if os.path.exists(os.path.join(ROOT, ".rev")) else None


class VramSampler:
    """Peak device memory in use (cudaMemGetInfo every 2 ms) -- sees vLLM, torch and CuPy."""

    def __init__(self):
        import cupy as cp
        self.cp, self.peak, self.stop = cp, 0, threading.Event()
        free, total = cp.cuda.runtime.memGetInfo()
        self.base = total - free
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.peak = max(self.peak, total - free)
            time.sleep(0.002)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.th.join()

    def report(self):
        return {"device_used_peak_mib": round(self.peak / 2**20, 1),
                "device_used_baseline_mib": round(self.base / 2**20, 1),
                "prover_peak_above_baseline_mib": round((self.peak - self.base) / 2**20, 1),
                "cupy_pool_total_mib_after": round(self.cp.get_default_memory_pool().total_bytes() / 2**20, 1)}


def rnd(v, nd=1):
    """Round floats inside nested dicts / lists (timing reports)."""
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return round(v, nd) if isinstance(v, float) else v


def bf16_f(word):
    return struct.unpack("<f", struct.pack("<I", int(word) << 16))[0]


# ===================================================================================== 1. live serving
def serve(args, res):
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from fpn_serve.capture.ports import CaptureTier
    from fpn_serve.engines.vllm import capture_plugin as cp

    cp.configure_capture(cp.CaptureConfig(tier=CaptureTier.FULL_NO_KV, ring_mib=args.ring_mib,
                                          max_batched_tokens=args.max_batched_tokens))
    t0 = time.perf_counter()
    llm = LLM(model=args.model, load_format="fpn_capture", dtype="bfloat16", enforce_eager=False,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=4096, max_num_seqs=8,
              max_num_batched_tokens=args.max_batched_tokens, enable_prefix_caching=True, seed=0)
    runner = llm.collective_rpc(cp.install_runner)
    t_load = time.perf_counter() - t0
    sess = cp.get_session()
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda msgs: list(tok.encode(tok.apply_chat_template(  # noqa: E731
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False))
    eng = llm.llm_engine

    def run(batch):
        outs = {}
        for rid, ids, n in batch:
            eng.add_request(rid, TokensPrompt(prompt_token_ids=ids),
                            SamplingParams(n=1, temperature=0.0, max_tokens=n, ignore_eos=True, detokenize=False))
        torch.cuda.synchronize()
        t = time.perf_counter()
        while len(outs) < len(batch):
            for o in eng.step():
                if o.finished:
                    outs[o.request_id] = o
        torch.cuda.synchronize()
        return outs, time.perf_counter() - t

    def settle():
        t = time.perf_counter()
        st = sess.settle(120.0)
        return st, (time.perf_counter() - t) * 1e3

    m1 = [{"role": "user", "content": TURN1}]
    p1 = enc(m1)
    o1, t1 = run([("turn1", p1, args.n_new)])
    out1 = list(o1["turn1"].outputs[0].token_ids)
    st1, settle1 = settle()
    m2 = m1 + [{"role": "assistant", "content": tok.decode(out1, skip_special_tokens=True)},
               {"role": "user", "content": TURN2}]
    # Turn 2 re-feeds turn 1's prompt + generated tokens verbatim (what a chat client sends back), then the follow-up.
    p2 = p1 + out1 + enc([{"role": "user", "content": TURN2}])
    ps = enc([{"role": "user", "content": SIDE}])
    o2, t2 = run([("turn2", p2, args.n_new), ("side", ps, args.n_new)])
    st2, settle2 = settle()
    out2 = list(o2["turn2"].outputs[0].token_ids)
    cached2 = int(o2["turn2"].num_cached_tokens or 0)
    traces = {**st1, **st2}
    key2 = next(k for k in traces if k == "turn2" or k.startswith("turn2"))
    trace = traces[key2]
    token_ids = p2 + out2[:-1]
    handle = sess.store.handle(key2)
    t = time.perf_counter()
    pages = list(handle.iter_verified_pages(token_ids))
    t_read = (time.perf_counter() - t) * 1e3
    verified = {}
    for k, tr in traces.items():
        ids = {"turn1": p1 + out1[:-1], "side": ps + list(o2["side"].outputs[0].token_ids)[:-1]}.get(
            k.split("-")[0], token_ids)
        verified[k] = sum(c for _, c, _ in sess.store.handle(k).iter_verified_pages(ids))
    res["serving"] = {
        "engine": "vLLM " + __import__("vllm").__version__, "model": os.path.basename(args.model.rstrip("/")),
        "runner": runner, "load_format": "fpn_capture", "capture_tier": "FULL_NO_KV",
        "capture_slots": sorted({s.kind for s in sess.layout.slots}), "row_bytes": sess.store.row_bytes,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "enable_prefix_caching": True, "llm_init_s": round(t_load, 2),
        "turn1": {"prompt_tokens": len(p1), "new_tokens": len(out1), "serve_s": round(t1, 4),
                  "settle_ms": round(settle1, 2), "text": tok.decode(out1, skip_special_tokens=True)[:200]},
        "turn2_batch_with_side": {"turn2_prompt_tokens": len(p2), "turn2_cached_tokens": cached2,
                                  "side_prompt_tokens": len(ps), "new_tokens_each": args.n_new,
                                  "serve_s": round(t2, 4), "settle_ms": round(settle2, 2),
                                  "text": tok.decode(out2, skip_special_tokens=True)[:200]},
        "traces": {k: {"n_tokens": v.n_tokens, "n_full_pages": v.n_full_pages, "tip": v.tip.hex() if v.tip else None,
                       "verified_tokens": verified[k]} for k, v in traces.items()},
        "pagestore_read_turn2_verified_pages_ms": round(t_read, 2),
        "sink_stats": {k: v for k, v in dict(sess.sink.stats).items() if isinstance(v, (int, float))},
        "ring_overflow_events": sess.ring.overflow_events,
    }
    print(json.dumps(res["serving"], indent=1), flush=True)
    return dict(llm=llm, sess=sess, store=sess.store, trace=trace, key=key2, pages=pages, prompt=p2, out=out2,
                token_ids=token_ids, cached=cached2)


def live_weights(llm, layers):
    """Prover side: BF16 words of the live vLLM model's fused parameters (qkv_proj, gate_up_proj, input_layernorm)."""
    def grab(worker):
        import torch
        mr = worker.model_runner
        model = mr.get_model() if hasattr(mr, "get_model") else mr.model
        words = lambda t: t.detach().contiguous().view(torch.int16).cpu().numpy()  # noqa: E731
        return {l: {"qkv": words(model.model.layers[l].self_attn.qkv_proj.weight),
                    "gate_up": words(model.model.layers[l].mlp.gate_up_proj.weight),
                    "ln1": words(model.model.layers[l].input_layernorm.weight)} for l in layers}
    got = llm.collective_rpc(grab)[0]
    return {l: {k: v.view(np.uint16) for k, v in d.items()} for l, d in got.items()}


def live_tile(lw, layer, op, start):
    """32 x 1024 tile of GEMM `op` (rows start..start+32) from vLLM's fused weights."""
    if op < ATTN_GEMMS:
        base = (0, 2048, 3072)[op]
        return lw[layer]["qkv"][base + start: base + start + 32]
    base = (0, 3072)[op - ATTN_GEMMS]
    return lw[layer]["gate_up"][base + start: base + start + 32]


def hf_tile(st, layer, op, start):
    """Verifier side: the same tile straight from the HF safetensors checkpoint (unfused tensor names)."""
    name, _ = GEMMS[op]
    blk = "self_attn" if op < ATTN_GEMMS else "mlp"
    t = st.get_slice(f"model.layers.{layer}.{blk}.{name}.weight")[start:start + 32]
    return t.view(__import__("torch").int16).numpy().view(np.uint16)


# ===================================================================================== 2. rows + protocol
def read_rows(store, pages, positions):
    """{position: {(kind, layer): uint16 words}} for the drawn positions (rows from iter_verified_pages)."""
    want = set(positions)
    out, off = {}, 0
    for _, count, rows in pages:
        for p in range(off, off + count):
            if p in want:
                start = (p - off) * store.row_bytes
                d = {}
                for s in store.layout.slots:
                    if s.kind in ("lse", "h_L", "x_f"):
                        continue
                    d[(s.kind, s.layer)] = np.frombuffer(rows, dtype="<u2", count=s.width,
                                                         offset=start + 2 * s.offset).copy()
                out[p] = d
        off += count
    return out


def sampler_profile():
    from fpn_prover.circuit import protocol
    return protocol.SamplerProfile(28, tuple((1024, rows) for _, rows in GEMMS), q_heads=16, kv_groups=8,
                                   head_dim=128, attn_gemms=ATTN_GEMMS)


def protocol_draws(setup_cert, trace_tip, token_ids, response, prompt_len, cached, draws, nonce=NONCE,
                   policy_bytes=b""):
    """(C_pre, seed, nonempty draws) from PUBLIC inputs only: setup record, trace tip, forwarded token ids, response,
    prompt length, cached-prefix length and the external nonce (prover and verifier both call this).  :
    `policy_bytes` (default empty = unchanged) is folded into C_pre, e.g. `agentic_state.agentic_policy_bytes(St_i)`."""
    from fpn_prover.circuit import protocol
    from fpn_prover.runtime.pipeline import precommit_request
    c_pre = precommit_request(setup_cert, trace_tip, response, policy_bytes)
    prompt_bytes = struct.pack(f"<{len(token_ids)}I", *token_ids)
    resp_bytes = struct.pack(f"<{len(response)}I", *response)
    seed = protocol.external_seed(prompt_bytes, resp_bytes, c_pre, nonce)
    d = protocol.sample_request(seed, sampler_profile(), input_tokens=prompt_len, cache_read_input_tokens=cached,
                                output_tokens=len(response), draws=draws)
    flat = [x for s in (protocol.STRATUM_FRESH_INPUT, protocol.STRATUM_GENERATION) for x in d[s] if not x.empty]
    return c_pre, seed, flat


def rel5_n_s(c_pre, gemm_draws, x_digest):
    """N_s of the relation-5 receipt: C_pre, the PUBLIC draw coordinates and the digest of the exact BF16 X words of
    every tile () -- the verifier recomputes x_digest from rows opened against trace_payload_root."""
    coords = struct.pack(f"<{4 * len(gemm_draws)}I", *(v for d in gemm_draws
                                                        for v in (d.position, d.layer, d.gemm, d.tile_start)))
    return hashlib.blake2b(N_S_DOMAIN + c_pre + coords + x_digest, digest_size=32).digest()


def slot_words(layout, row, kind, layer):
    """uint16 words of one capture slot from a raw row (public capture-layout geometry only)."""
    sl = layout.slot(kind, layer)
    return np.frombuffer(row, dtype="<u2", count=sl.width, offset=2 * sl.offset).copy()


def x_slot(d):
    return "x_attn" if d.gemm < ATTN_GEMMS else "x_mlp"


def rel5_x_digest(gemm_draws, words_of):
    """activation_binding_digest of the X words of every relation-5 tile; words_of(draw) -> uint16 array."""
    from fpn_prover.runtime.trace_commit import activation_binding_digest
    return activation_binding_digest([(x_slot(d), d.position, d.layer, words_of(d).astype("<u2").tobytes())
                                      for d in gemm_draws], X_DOMAIN)


def r4_items(disc, words_of):
    """The activation words one relation-4 block consumes: h_in norm pair, the q_proj input element, slot q / o."""
    p, l, (c0, c1), i_in, j = disc["position"], disc["layer"], disc["norm_channels"], disc["in_col"], disc["slot_dim"]
    h, xa, q, o = (words_of(p, k, l) for k in ("h_in", "x_attn", "q", "o"))
    pick = lambda w, idx: np.array([w[i] for i in idx], dtype="<u2").tobytes()  # noqa: E731
    return [("h_in", p, l, pick(h, (c0, c1))), ("x_attn", p, l, pick(xa, (i_in,))),
            ("q", p, l, pick(q, (j, 128 + j, 256 + j))), ("o", p, l, pick(o, (j, 128 + j)))]


def r4_nonce(c_pre, disc, words_of):
    """Relation-4 server nonce of one block: C_pre + the digest of the exact activation words it consumed."""
    from fpn_prover.runtime.trace_commit import activation_binding_digest
    return hashlib.blake2b(R4_DOMAIN + c_pre + activation_binding_digest(r4_items(disc, words_of), R4_DOMAIN),
                           digest_size=32).digest()


# ===================================================================================== 3a. relation 4
def r4_samples(draws, rows, lw, count):
    """E2E.Samples of `count` drawn blocks from captured rows (see notes for the disclosed lane choices)."""
    import random

    import torch

    from fpn_prover.circuit import e2e_prover as E
    from fpn_prover.circuit import lower as LW
    out, meta = [], []
    for d in draws[:count]:
        r = rows[d.position]
        h, xa, q, o = r[("h_in", d.layer)], r[("x_attn", d.layer)], r[("q", d.layer)], r[("o", d.layer)]
        gain = lw[d.layer]["ln1"]
        rnd = random.Random(f"v1:r4:{d.position}:{d.layer}")
        order = list(range(len(h)))
        rnd.shuffle(order)
        pair = None
        for i in range(0, len(order) - 1, 2):
            c0, c1 = order[i], order[i + 1]
            cand = E.Samples("dense_gqa_qknorm", "Qwen3-0.6B", (int(h[c0]), int(h[c1])), (int(gain[c0]), int(gain[c1])),
                             0x3f80, 0x3f80, 0x3f80, "dense_gqa", dict(normalize=0, query=0, key0=0, key1=0, value0=0,
                                                                         value1=0))
            try:
                E.build_witness(cand)
                pair = (c0, c1)
                break
            except LW.LoweringError:
                continue
        i_in, o_row = rnd.randrange(1024), rnd.randrange(2048)
        while not (int(xa[i_in]) & 0x7fff and int(live_tile(lw, d.layer, 0, o_row - o_row % 32)[o_row % 32][i_in]) & 0x7fff):
            i_in, o_row = rnd.randrange(1024), rnd.randrange(2048)
        w_word = int(live_tile(lw, d.layer, 0, o_row - o_row % 32)[o_row % 32][i_in])
        xt = torch.tensor([bf16_f(xa[i_in])], dtype=torch.bfloat16, device="cuda")
        wt = torch.tensor([bf16_f(w_word)], dtype=torch.bfloat16, device="cuda")
        prod = int(torch.mul(xt, wt).view(torch.int16).item()) & 0xFFFF
        j = int(np.abs(np.array([bf16_f(v) for v in o[:128]])).argmax())
        slot = dict(normalize=1, query=bf16_f(q[j]), key0=bf16_f(q[128 + j]), key1=bf16_f(q[256 + j]),
                    value0=bf16_f(o[j]), value1=bf16_f(o[128 + j]))
        s = E.Samples("dense_gqa_qknorm", "Qwen3-0.6B", (int(h[pair[0]]), int(h[pair[1]])),
                      (int(gain[pair[0]]), int(gain[pair[1]])), int(xa[i_in]), w_word, prod, "dense_gqa", slot)
        out.append(s)
        meta.append({"position": d.position, "layer": d.layer, "draw_kind": d.kind, "stratum": d.stratum,
                     "norm_channels": list(pair), "gemm": {"projection": "q_proj", "out_row": o_row, "in_col": i_in},
                     "slot_dim": j})
    return out, meta


# ===================================================================================== 4. independent verifier
def tile_coords(draws, words=None):
    """Weight-tile coordinates of relation-5 draws (`words(p, kind, layer)`: opened-row words, used by geometries
    whose tensor name depends on captured data, e.g. the routed expert id; unused here)."""
    return [(f"model.layers.{d.layer}.{'self_attn' if d.gemm < ATTN_GEMMS else 'mlp'}.{GEMMS[d.gemm][0]}.weight",
             d.tile_start, 0) for d in draws]


def independent_verify(setup, reply_c_pre, trace_payload_root, token_ids, response, prompt_len, cached, draws, tiles, nonce, layout,
                       trace_openings, weight_openings, r4_receipts, r4_disclosures, r5_receipt, fpn_verify,
                       engine=None, policy_bytes=b"", window_int_bits=None):
    """Offline verifier: PUBLIC inputs + the reply's bundle only (no PageStore, no live weights, no prover rows).

    1. C_pre recomputed from (setup, trace_payload_root, response) and required to EQUAL the reply's C_pre (published
       with the reply BEFORE the external nonce existed; E_CPRE_REQUIRED / E_CPRE_MISMATCH); Protocol v2 seed + draws
       re-derived from (prompt, response, C_pre, external nonce);
    2. every drawn row opens against trace_payload_root (n_tokens = forwarded token count, public layout) and the
       opened set equals the re-derived drawn positions (E_TRACE_*);
    3. X of every relation-5 tile is extracted from the OPENED rows at the re-derived (position, layer, slot) and its
       digest must reproduce the receipt's N_s (E_X_BINDING); every drawn weight tile opens against
       SetupCertificate.weight_commitment_root (E_TILE_* / E_WEIGHT_ROOT) and the recomputed relation-5 weight_root
       is pinned in native Rust fpn-verify;
    4. every relation-4 block sits at a re-derived draw, its consumed activation words are taken from the opened row,
       their digest gives the server nonce the receipt must bind (E_CPRE_MISMATCH otherwise) and Rust + detached
       Python verify it against the SetupCertificate circuit_root."""
    from fpn_prover.circuit import e2e_prover as E
    from fpn_prover.circuit import lower as LW
    from fpn_prover.runtime import trace_commit as TC
    from fpn_prover.runtime import weight_commit as WC
    from fpn_prover.runtime.pipeline import verify_reply_with_setup
    out = {"stages": {}}
    t = time.perf_counter()
    c_pre, seed, flat = protocol_draws(setup, trace_payload_root, token_ids, response, prompt_len, cached, draws, nonce,
                                       policy_bytes)
    gemm = [d for d in flat if d.kind == "gemm"][:tiles]
    out["c_pre"], out["seed"] = c_pre.hex(), seed.hex()
    if type(reply_c_pre) is not bytes or len(reply_c_pre) != 32:
        out["verdict"] = "E_CPRE_REQUIRED"
        return out
    if c_pre != reply_c_pre:
        out["stages"]["c_pre"] = "E_CPRE_MISMATCH"
        out["verdict"] = "E_CPRE_MISMATCH"
        return out
    out["stages"]["c_pre"] = "VERIFIED"
    tv = TC.verify_trace_row_openings(trace_payload_root, len(token_ids), layout.fingerprint(), trace_openings,
                                      row_bytes=layout.stride_bytes, expected_positions={d.position for d in flat})
    out["stages"]["trace_rows"] = {"verdict": tv["verdict"], "detail": tv["detail"], "rows": len(tv["rows"])}
    if tv["verdict"] != "VERIFIED":
        out["verdict"] = tv["verdict"]
        return out
    rows = tv["rows"]
    words = lambda p, k, l: slot_words(layout, rows[p], k, l)  # noqa: E731
    n_s = rel5_n_s(c_pre, gemm, rel5_x_digest(gemm, lambda d: words(d.position, x_slot(d), d.layer)))
    if bytes(r5_receipt[128:160]) != n_s:
        out["stages"]["rel5_x_binding"] = "E_X_BINDING"
        out["verdict"] = "E_X_BINDING"
        return out
    out["stages"]["rel5_x_binding"] = "VERIFIED"
    chk = WC.verify_drawn_tiles_against_setup(setup, weight_openings, r5_receipt, tile_coords(gemm, words),
                                              engine=engine, window_int_bits=window_int_bits)
    out["stages"]["weight_tiles"] = chk["verdict"]
    if chk["verdict"] != "VERIFIED":
        out["verdict"] = chk["verdict"]
        return out
    rs5 = E.verdict_code(E.rust_verify(fpn_verify, r5_receipt, bytes.fromhex(chk["weight_root"]))[0])
    out["stages"]["rel5_rust"] = rs5
    if rs5 != "VERIFIED":
        out["verdict"] = rs5
        return out
    circuit = LW.compile_air(E.build_program()[0])
    r4v = []
    for i, (receipt, disc) in enumerate(zip(r4_receipts, r4_disclosures, strict=True)):
        if (disc["position"], disc["layer"]) != (flat[i].position, flat[i].layer):
            r4v.append({"verdict": "E_DRAW_MISMATCH"})
            continue
        v = verify_reply_with_setup(setup, receipt, expected_c_pre=r4_nonce(c_pre, disc, words),
                                    circuit_digest=circuit.digest, rust_binary=fpn_verify, check_python=True)
        r4v.append({"verdict": "VERIFIED" if v["verified"] else v.get("error") or v.get("rust"), "rust": v["rust"],
                    "python": v["python"]})
    out["stages"]["rel4"] = r4v
    bad = [x["verdict"] for x in r4v if x["verdict"] != "VERIFIED"]
    out["verdict"] = bad[0] if bad else "VERIFIED"
    out["verify_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    return out


# ===================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="<MODELS_DIR>/Qwen3-0.6B")
    ap.add_argument("--fpn-verify", default=os.path.join(ROOT, "core", "target", "release", "fpn-verify"))
    ap.add_argument("--out", default=os.path.join(ROOT, "internal", "results", "m1", "live_vllm_e2e_rtx6000d.json"))
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--r4-blocks", type=int, default=3)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--ring-mib", type=int, default=1536)
    ap.add_argument("--max-batched-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.35)
    ap.add_argument("--verify-runs", type=int, default=5)
    ap.add_argument("--commit-threads", type=int, default=16)
    ap.add_argument("--window-int-bits", type=int, default=23,
                    help="capped exponent window: |W_int|, |X_int| <= 2^bits (23 = 7 + max_shift_span 16); "
                         "0 = the exact encoding")
    args = ap.parse_args(argv)
    WB = args.window_int_bits or None
    t_start = time.perf_counter()
    from flashprove_next.field import native, native_threads
    res = {"bench": "bench_live_vllm_e2e", "wave": "13A", "git_rev": git_rev(),
           "machine": {"host": platform.node(), "python": platform.python_version(),
                       "native_field_lib": native() is not None, "native_threads": native_threads()}}
    if native() is None:
        raise SystemExit("flashprove_next native library unavailable (would time the NumPy fallback)")
    try:
        res["machine"]["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()
    except OSError:
        pass

    live = serve(args, res)
    import torch
    from safetensors import safe_open

    from fpn_prover.backend import cuda_engine as CE
    from fpn_prover.backend import folded_vec as FV
    from fpn_prover.backend.fpn2 import FOLDED_AIR_V1, FOLDED_VEC_N22_V1
    from fpn_prover.circuit import e2e_prover as E
    from fpn_prover.circuit import lower as LW
    from fpn_prover.runtime import weight_commit as WC
    from fpn_prover.runtime.api_envelope import attach_verification
    from fpn_prover.runtime import trace_commit as TC
    from fpn_prover.runtime.pipeline import verify_reply_with_setup
    from fpn_prover.runtime.setup_cert import SetupCertificate
    from fpn_prover.circuit import protocol

    # ---------------------------------------------------------------- setup certificate: ONE-TIME global weight root
    st_path = os.path.join(args.model, "model.safetensors")
    t = time.perf_counter()
    wcom = WC.commit_model_weights(args.model, threads=args.commit_threads)
    t_wcommit = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    prog, _ = E.build_program()
    circuit = LW.compile_air(prog)
    setup = SetupCertificate(wcom.root, circuit.root, hashlib.blake2b(b"bf16_fp32acc", digest_size=32).digest(),
                             hashlib.blake2b(b"Qwen3-0.6B/safetensors/bf16", digest_size=32).digest(), 0).encode()
    t_setup = (time.perf_counter() - t) * 1e3
    res["setup"] = {
        "setup_cert_id": SetupCertificate.decode(setup).id, "weight_commitment_root": wcom.root.hex(),
        "global_weight_commit": {k: v for k, v in wcom.stats.items() if k != "skipped"},
        "global_weight_commit_skipped_tensors": len(wcom.stats["skipped"]),
        "global_weight_commit_wall_ms": round(t_wcommit, 1), "circuit_root": circuit.root.hex(),
        "note": "commit_model_weights runs ONCE per checkpoint (setup); it is not part of any per-request timing"}
    print(json.dumps(res["setup"], default=str), flush=True)

    # ---------------------------------------------------------------- C_pre over the FULL captured payload, then draws
    t = time.perf_counter()
    tree = TC.commit_trace_merkle(live["store"], live["trace"], live["key"], live["token_ids"])
    payload_root = tree.root
    t_payload = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    public = (payload_root, live["token_ids"], live["out"], len(live["prompt"]), live["cached"], args.draws)
    c_pre, seed, flat = protocol_draws(setup, *public)
    c_pre_tip_only = protocol_draws(setup, live["trace"].tip, *public[1:])[0]
    gemm_draws = [d for d in flat if d.kind == "gemm"][:args.tiles]
    if len(gemm_draws) < args.tiles:
        raise SystemExit(f"only {len(gemm_draws)} GEMM draws; raise --draws")
    t_draw = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    trace_bundle = TC.open_trace_rows(tree, {d.position for d in flat})
    trace_openings = trace_bundle.encode()
    t_open_rows = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    rows = read_rows(live["store"], live["pages"], {d.position for d in flat})
    t_rows = (time.perf_counter() - t) * 1e3
    layers = sorted({d.layer for d in flat})
    t = time.perf_counter()
    lw = live_weights(live["llm"], layers)
    t_lw = (time.perf_counter() - t) * 1e3
    res["protocol"] = {
        "challenge_mode": protocol.MODE_EXTERNAL_SEED, "c_pre": c_pre.hex(), "seed": seed.hex(),
        "c_pre_binds": "precommit_request(setup cert id, trace_payload_root = Merkle root over every captured token "
                       "row (commit_trace_merkle), response) -- computed BEFORE external_seed / sample_request",
        "trace_payload_root": payload_root.hex(), "trace_tip": live["trace"].tip.hex(),
        "c_pre_differs_from_tip_only_c_pre": c_pre != c_pre_tip_only,
        "trace_payload_tokens": live["trace"].n_tokens, "trace_merkle_depth": len(tree.levels) - 1,
        "row_bytes": tree.row_bytes,
        "trace_row_openings": {"positions": len(trace_bundle.openings), "bytes": len(trace_openings),
                               "bytes_paths_only": sum(32 * len(o.path) for o in trace_bundle.openings),
                               "note": "each opening discloses the WHOLE captured row of a drawn position"},
        "setup_cert_id": SetupCertificate.decode(setup).id, "draws_per_stratum": args.draws,
        "nonempty_draws": len(flat), "gemm_draws": sum(d.kind == "gemm" for d in flat),
        "attn_draws": sum(d.kind == "attn" for d in flat),
        "cached_prefix_positions_without_draws": live["cached"],
        "min_drawn_position": min(d.position for d in flat), "timing_ms": {
            "setup_cert_circuit": round(t_setup, 1), "commit_trace_merkle": round(t_payload, 2),
            "open_trace_rows": round(t_open_rows, 2),
            "sample_request": round(t_draw, 2),
            "pagestore_rows_for_drawn_positions": round(t_rows, 2), "live_weight_fetch_from_vllm": round(t_lw, 1)}}
    assert min(d.position for d in flat) >= live["cached"], "a cached-prefix position received a draw"

    engine = CE.CudaEngine()

    # ---------------------------------------------------------------- relation 4 (GPU) on sampled blocks
    samples, r4meta = r4_samples(flat, rows, lw, args.r4_blocks)
    prover_words = lambda p, k, l: rows[p][(k, l)]  # noqa: E731
    r4_disc = [{"position": m["position"], "layer": m["layer"], "norm_channels": m["norm_channels"],
                "in_col": m["gemm"]["in_col"], "slot_dim": m["slot_dim"]} for m in r4meta]
    r4, r4_receipts = [], []
    for i, s in enumerate(samples):
        nonce_i = r4_nonce(c_pre, r4_disc[i], prover_words)
        with VramSampler() as vs:
            t = time.perf_counter()
            proof = E.prove_samples(s, server_nonce=nonce_i, engine=engine)
            wall = (time.perf_counter() - t) * 1e3
        if proof.circuit.root != circuit.root:
            raise SystemExit("relation-4 circuit root differs from the setup certificate")
        r4_receipts.append(proof.container)
        ver = verify_reply_with_setup(setup, proof.container, expected_c_pre=nonce_i,
                                      circuit_digest=proof.circuit.digest, rust_binary=args.fpn_verify,
                                      check_python=True)
        rs = [E.rust_verify(args.fpn_verify, proof.container, circuit.root) for _ in range(args.verify_runs)]
        r4.append({**r4meta[i], "prove_wall_ms": round(wall, 1),
                   "prove_phases_ms": rnd(proof.timings_ms),
                   "vram": vs.report(), "container_bytes": len(proof.container), "verify": ver,
                   "rust_cli_wall_ms_median": round(statistics.median(r[1] for r in rs), 2),
                   "rust_child_cpu_ms_median": round(statistics.median(r[2] for r in rs), 2),
                   "sha256": hashlib.sha256(proof.container).hexdigest()})
        print(json.dumps(r4[-1], default=str), flush=True)
        r4_receipt = proof.container
    res["relation4"] = {"profile": FOLDED_AIR_V1["name"], "c_p": FOLDED_AIR_V1["c_p"],
                        "circuit_root": circuit.root.hex(), "blocks": r4}

    # ---------------------------------------------------------------- relation 5 tiles (prover side: live vLLM weights)
    t = time.perf_counter()

    def x_ints(words):
        if WB is None:
            return FV.bf16_words_to_ints(words)
        return WC.window_ints(WC.bf16_values(words), WB)

    W_int, X_int, tiles_meta, x_words = [], [], [], []
    W_exact, X_exact = [], []
    for d in gemm_draws:
        ww = live_tile(lw, d.layer, d.gemm, d.tile_start)
        xw = rows[d.position][("x_attn" if d.gemm < ATTN_GEMMS else "x_mlp", d.layer)]
        we, swe = FV.bf16_words_to_ints(ww)
        xe, sxe = FV.bf16_words_to_ints(xw)
        W_exact.append((we, swe))
        X_exact.append((xe, sxe))
        wi, sw = (we, swe) if WB is None else WC.window_ints(WC.bf16_values(ww), WB)
        xi, sx = x_ints(xw)
        W_int.append(wi)
        X_int.append(xi)
        x_words.append(xw)
        wt_err, wt_max = WC.window_tail(WC.bf16_values(ww), wi, sw)
        xt_err, xt_max = WC.window_tail(WC.bf16_values(xw), xi, sx)
        tiles_meta.append({"position": d.position, "layer": d.layer, "gemm": GEMMS[d.gemm][0],
                           "rows": [d.tile_start, d.tile_start + 32], "w_shift": sw, "x_shift": sx,
                           "w_exact_shift": swe, "x_exact_shift": sxe,
                           "w_window_tail_rel": wt_err / wt_max if wt_max else 0.0,
                           "x_window_tail_rel": xt_err / xt_max if xt_max else 0.0,
                           "w_elems_rounded": int((np.ldexp(np.asarray(wi, dtype=np.float64), -sw)
                                                   != WC.bf16_values(ww)).sum()),
                           "x_elems_rounded": int((np.ldexp(np.asarray(xi, dtype=np.float64), -sx)
                                                   != WC.bf16_values(xw)).sum())})
    lay = FV.layout_of(FOLDED_VEC_N22_V1)
    W_int = np.array(W_int, dtype=object if any(a.dtype == object for a in W_int) else np.int64)
    X_int = np.array(X_int, dtype=object if any(a.dtype == object for a in X_int) else np.int64)
    Y_int = []
    bound_bits = []
    for b in range(lay.B):
        yb, bits = FV.exact_products(W_int[b:b + 1], X_int[b:b + 1])
        Y_int.append(yb[0])
        bound_bits.append(bits)
    Y_int = np.array(Y_int, dtype=object)
    Wf, Xf, Yf = FV.to_field(W_int), FV.to_field(X_int), FV.to_field(Y_int)
    t_tiles = (time.perf_counter() - t) * 1e3
    # : exact (unwindowed) products vs the windowed products the proof is about -- the window tail, exactly
    from fractions import Fraction
    tail_rel, exact_bits = [], []
    for b, m in enumerate(tiles_meta):
        (we, swe), (xe, sxe) = W_exact[b], X_exact[b]
        ye, eb = FV.exact_products(np.asarray(we)[None], np.asarray(xe)[None])
        exact_bits.append(eb)
        ye = [Fraction(int(v), 1 << (swe + sxe)) if swe + sxe >= 0 else Fraction(int(v) << -(swe + sxe))
              for v in ye[0]]
        sh = m["w_shift"] + m["x_shift"]
        yw = [Fraction(int(v), 1 << sh) if sh >= 0 else Fraction(int(v) << -sh) for v in Y_int[b]]
        mx = max(abs(v) for v in ye) or Fraction(1)
        tail_rel.append(float(max(abs(a - c) for a, c in zip(yw, ye)) / mx))
    # exact integer products vs the device BF16 GEMM (fp32 accumulate, bf16 output; reduced precision disabled)
    rel = []
    for b, m in enumerate(tiles_meta):
        Wt = torch.from_numpy(live_tile(lw, m["layer"], [g for g, _ in GEMMS].index(m["gemm"]), m["rows"][0])
                              .view(np.int16).copy()).view(torch.bfloat16).cuda()
        xt = torch.from_numpy(x_words[b].view(np.int16).copy()).view(torch.bfloat16).cuda()
        dev = torch.nn.functional.linear(xt[None], Wt)[0].float().cpu().numpy()
        ex = np.array([float(Fraction(int(v)) / Fraction(2) ** (m["w_shift"] + m["x_shift"])) for v in Y_int[b]])
        scale = np.abs(ex).max() or 1.0
        rel.append(float(np.abs(dev - ex).max() / scale))
    n_s = rel5_n_s(c_pre, gemm_draws, rel5_x_digest(gemm_draws, lambda d: rows[d.position][(x_slot(d), d.layer)]))

    # ---------------------------------------------------------------- engine parity on the first real tile (n = 16)
    from fpn_prover.backend.fpn2 import FOLDED_VEC_N16_V1
    import random as _random

    class _SeedRng:
        def __init__(self, seed):
            self.r = _random.Random(seed)

        def fp3(self):
            return tuple(self.r.randrange(FV.P) for _ in range(3))

    parity = {}
    for name, eng in (("cuda", engine), ("host_emulation", CE.HostEngine()), ("cpu_reference", None)):
        FV._W_CACHE.clear()
        t = time.perf_counter()
        d16 = FV.prove(Wf[:1], Xf[:1], Yf[:1], n_s=n_s, profile=FOLDED_VEC_N16_V1, rng=_SeedRng(8), engine=eng)
        parity[name] = {"sha256": hashlib.sha256(d16).hexdigest(), "prove_ms": round((time.perf_counter() - t) * 1e3, 1)}
    parity["byte_identical"] = len({v["sha256"] for v in parity.values()}) == 1
    parity["rust"] = E.verdict_code(E.rust_verify(args.fpn_verify, d16, bytes(d16[96:128]))[0])
    parity["note"] = ("fpn2-folded-vec-n16-v1 proof of real tile 0 (B = 1) with one seeded rng on CudaEngine, "
                      "HostEngine and the pure-Python CPU reference")
    print(json.dumps(parity), flush=True)

    # ---------------------------------------------------------------- relation 5 n = 22 on the GPU
    r5_runs = []
    for rep in range(3):
        FV._W_CACHE.clear()                           # every run pays the per-request weight commitment
        tm = {}
        with VramSampler() as vs:
            t = time.perf_counter()
            r5 = FV.prove(Wf, Xf, Yf, n_s=n_s, profile=FOLDED_VEC_N22_V1, engine=engine, hooks={"timings": tm})
            wall = (time.perf_counter() - t) * 1e3
        r5_runs.append({"rep": rep, "prove_wall_ms": round(wall, 1), "phases_ms": {k: round(v, 1) for k, v in tm.items()},
                        "vram": vs.report(), "sha256": hashlib.sha256(r5).hexdigest()})
        print(json.dumps(r5_runs[-1]), flush=True)
    prover_root = bytes(r5[96:128])

    # ---------------------------------------------------------------- serving side: open the drawn tiles (global root)
    t = time.perf_counter()
    opening = WC.open_drawn_tiles(wcom, args.model, tile_coords(gemm_draws))
    opening_bytes = opening.encode()
    t_open = (time.perf_counter() - t) * 1e3

    # ---------------------------------------------------------------- verifying side: SetupCertificate ONLY
    t = time.perf_counter()
    c_pre_v, seed_v, flat_v = protocol_draws(setup, *public)
    gemm_v = [d for d in flat_v if d.kind == "gemm"][:args.tiles]
    xdig_v = rel5_x_digest(gemm_v, lambda d: rows[d.position][(x_slot(d), d.layer)])
    n_s_ok = bytes(r5[128:160]) == rel5_n_s(c_pre_v, gemm_v, xdig_v)
    n_s_wrong_c_pre = bytes(r5[128:160]) == rel5_n_s(b"\xff" * 32, gemm_v, xdig_v)
    chk = WC.verify_drawn_tiles_against_setup(setup, opening_bytes, r5, tile_coords(gemm_v), engine=engine,
                                              window_int_bits=WB)
    verifier_root = bytes.fromhex(chk["weight_root"]) if chk["verdict"] == "VERIFIED" else bytes(32)
    t_vroot = (time.perf_counter() - t) * 1e3
    # independent cross-check (outside the protocol): the same tiles straight from the HF safetensors
    with safe_open(st_path, framework="pt") as st:
        Wv = np.array([hf_tile(st, d.layer, d.gemm, d.tile_start) for d in gemm_v])
    tiles_identical = bool(np.array_equal(Wv, np.array([live_tile(lw, d.layer, d.gemm, d.tile_start) for d in gemm_draws])))
    # forgery controls against the setup root (all must be REJECTED)
    forg = {}
    t0 = opening.tiles[0]
    u = t0.units.copy()
    u[0, 0] += 1
    forg["one_weight_element_changed_in_opening"] = WC.verify_drawn_tiles_against_setup(
        setup, WC.TileOpening(opening.K, opening.manifest, (WC.OpenedTile(t0.name, t0.r0, t0.c0, t0.shift, u, t0.path),)
                              + opening.tiles[1:]), r5, tile_coords(gemm_v), engine=engine,
        window_int_bits=WB)["verdict"]
    other = gemm_draws[0]
    swapped_layer = (other.layer + 1) % 28
    sub = WC.open_drawn_tiles(wcom, args.model, [(tile_coords([other])[0][0].replace(
        f"layers.{other.layer}.", f"layers.{swapped_layer}."), other.tile_start, 0)])
    forg["genuine_tile_of_another_layer_substituted"] = WC.verify_drawn_tiles_against_setup(
        setup, WC.TileOpening(opening.K, opening.manifest, sub.tiles + opening.tiles[1:]), r5, tile_coords(gemm_v),
        engine=engine, window_int_bits=WB)["verdict"]
    bad_setup = SetupCertificate(hashlib.blake2b(b"another model", digest_size=32).digest(), circuit.root,
                                 bytes(32), bytes(32), 0).encode()
    forg["setup_certificate_with_another_weight_root"] = WC.verify_drawn_tiles_against_setup(
        bad_setup, opening_bytes, r5, tile_coords(gemm_v), engine=engine, window_int_bits=WB)["verdict"]
    forg["draw_coordinates_of_another_seed"] = WC.verify_drawn_tiles_against_setup(
        setup, opening_bytes, r5, list(reversed(tile_coords(gemm_v))), engine=engine, window_int_bits=WB)["verdict"]
    if WB is not None:                                # the verifier applying NO window to honest windowed tiles
        forg["verifier_window_mismatch_exact_units"] = WC.verify_drawn_tiles_against_setup(
            setup, opening_bytes, r5, tile_coords(gemm_v), engine=engine, window_int_bits=None)["verdict"]
    rs5 = [E.rust_verify(args.fpn_verify, r5, verifier_root) for _ in range(args.verify_runs)]
    py5 = E.python_verify_detached(r5, verifier_root)
    rs5_unpinned = E.verdict_code(subprocess.run([args.fpn_verify, "/dev/stdin"], input=r5, capture_output=True)
                                  .stdout.decode().strip())

    # ---------------------------------------------------------------- negative cheat: perturb one PageStore element
    b0 = 0
    m0 = tiles_meta[b0]
    tampered_words = x_words[b0].copy()
    # the max-|x| element: inside the capped window by construction (its lowest mantissa bit is >= 1 grid unit)
    jbad = int(np.abs(WC.bf16_values(tampered_words)).argmax())
    tampered_words[jbad] ^= 0x0001                    # lowest mantissa bit of one captured x element
    Xt_int = X_int.copy()
    t_ints, t_shift = x_ints(tampered_words)
    assert t_shift == m0["x_shift"]                   # a mantissa-bit flip keeps the tile's exponent range
    Xt_int[b0] = t_ints
    Xtf = FV.to_field(Xt_int)
    cheat = FV.prove(Wf, Xtf, Yf, n_s=n_s, profile=FOLDED_VEC_N22_V1, engine=engine)
    cheat_rs = E.verdict_code(E.rust_verify(args.fpn_verify, cheat, verifier_root)[0])
    cheat_py = E.verdict_code(E.python_verify_detached(cheat, verifier_root)[0])
    Yt_int = np.array([FV.exact_products(W_int[b:b + 1], Xt_int[b:b + 1])[0][0] for b in range(lay.B)], dtype=object)
    recomputed = FV.prove(Wf, Xtf, FV.to_field(Yt_int), n_s=n_s, profile=FOLDED_VEC_N22_V1, engine=engine)
    rec_rs = E.verdict_code(E.rust_verify(args.fpn_verify, recomputed, verifier_root)[0])

    # ---------------------------------------------------------------- : independent offline verifier
    layout = live["sess"].layout
    iv_args = (setup, c_pre, payload_root, live["token_ids"], live["out"], len(live["prompt"]), live["cached"], args.draws,
               args.tiles, NONCE, layout)
    iv = lambda bundle, r5r: independent_verify(*iv_args, bundle, opening_bytes, r4_receipts, r4_disc, r5r,  # noqa: E731
                                                args.fpn_verify, engine=engine, window_int_bits=WB)
    honest = iv(trace_openings, r5)
    print(json.dumps({"independent_verifier_honest": honest["verdict"], "stages": honest["stages"]}, default=str),
          flush=True)
    # forgery: tamper ONE activation element of a drawn row AND recompute a self-consistent Y' = W X'
    d0 = gemm_draws[b0]
    sl0 = layout.slot(x_slot(d0), d0.layer)
    forged_rows = []
    for o in trace_bundle.openings:
        row = o.row
        if o.position == d0.position:
            row = bytearray(row)
            row[2 * (sl0.offset + jbad)] ^= 0x01               # the same mantissa bit as tampered_words[jbad]
            row = bytes(row)
            assert np.array_equal(slot_words(layout, row, x_slot(d0), d0.layer), tampered_words)
        forged_rows.append(TC.TraceRowOpening(o.position, row, o.path))
    forged_bundle = TC.TraceRowOpenings(trace_bundle.n_tokens, trace_bundle.layout_fingerprint, trace_bundle.row_bytes,
                                        tuple(forged_rows)).encode()
    x_forged = lambda d: tampered_words if d is d0 else rows[d.position][(x_slot(d), d.layer)]  # noqa: E731
    n_s_forged = rel5_n_s(c_pre, gemm_draws, rel5_x_digest(gemm_draws, x_forged))
    forged_r5 = FV.prove(Wf, Xtf, FV.to_field(Yt_int), n_s=n_s_forged, profile=FOLDED_VEC_N22_V1, engine=engine)
    forged_r5_rs = E.verdict_code(E.rust_verify(args.fpn_verify, forged_r5, verifier_root)[0])
    trace_forgeries = {
        "a_tampered_row_in_opening_plus_recomputed_Y_and_N_s": iv(forged_bundle, forged_r5),
        "b_honest_row_opened_but_receipt_N_s_from_tampered_X": iv(trace_openings, forged_r5),
        "c_honest_row_and_honest_N_s_but_witness_X_prime_with_Y_prime_eq_W_X_prime": iv(trace_openings, recomputed),
        "forged_r5_alone_rust_with_pinned_weight_root": forged_r5_rs,
    }
    # post-challenge re-commit: rebuild the Merkle root over the tampered payload, open the tampered row against it
    # (self-consistent paths), re-derive C_pre' / draws from it -> must fail against the reply's pre-challenge C_pre
    t_rows = []
    for o in trace_bundle.openings:
        r = bytearray(tree.row(o.position)) if tree.pages else bytearray(o.row)
        if o.position == d0.position:
            r[2 * (sl0.offset + jbad)] ^= 0x01
        t_rows.append((o.position, bytes(r)))
    t_leaves = list(tree.levels[0][:tree.n_tokens])
    for pos, r in t_rows:
        t_leaves[pos] = TC.row_leaf(pos, r)
    t_levels = TC._build_levels(t_leaves)
    t_root = TC.payload_root(t_levels[-1][0], tree.n_tokens, tree.layout_fingerprint)
    t_tree = TC.TraceTree(t_root, tree.n_tokens, tree.layout_fingerprint, tree.row_bytes, t_levels,
                          tuple((pos, 1, r) for pos, r in t_rows))
    t_bundle = TC.open_trace_rows(t_tree, [pos for pos, _ in t_rows]).encode()
    recommit_args = (iv_args[0], c_pre, t_root) + iv_args[3:]
    trace_forgeries["d_root_rebuilt_over_tampered_payload_after_challenge"] = independent_verify(
        *recommit_args, t_bundle, opening_bytes, r4_receipts, r4_disc, forged_r5, args.fpn_verify, engine=engine,
        window_int_bits=WB)
    trace_forgeries["e_reply_without_c_pre"] = independent_verify(
        iv_args[0], None, *iv_args[2:], trace_openings, opening_bytes, r4_receipts, r4_disc, r5, args.fpn_verify,
        engine=engine, window_int_bits=WB)["verdict"]
    for k, v in trace_forgeries.items():
        print(json.dumps({k: v if isinstance(v, str) else v["verdict"]}), flush=True)

    # ---------------------------------------------------------------- zero-leakage envelopes
    envs = []
    for receipt in (r4_receipt, r5):
        env = attach_verification({"object": "chat.completion", "choices": []}, receipt=receipt, setup_cert=setup,
                                  challenge_mode=protocol.MODE_EXTERNAL_SEED, c_pre=c_pre)
        assert base64.b64decode(env["fpn_receipt_b64"]) == receipt
        envs.append(sorted(set(env) - {"object", "choices"}))
    forbidden = ("architecture", "n_layers", "d_model", "n_experts", "update_type", "cache_read", "positions", "layers")

    res["relation5"] = {
        "profile": FOLDED_VEC_N22_V1["name"], "n_vars": lay.n, "tiles": lay.B, "tile_shape": [32, lay.K],
        "macs": lay.macs, "c_p": FOLDED_VEC_N22_V1["c_p"], "body_used": lay.body_used, "container_bytes": len(r5),
        "inputs": "captured x_attn (q/k/v_proj input) or x_mlp (gate/up_proj input) rows x live vLLM fused weights",
        "tiles_by_gemm": {g: sum(m["gemm"] == g for m in tiles_meta) for g, _ in GEMMS},
        "tile_draws": tiles_meta,
        "integer_bound_bits_max": max(bound_bits), "integer_bound_bits_per_tile": bound_bits,
        "integer_faithful": max(bound_bits) <= 62,
        "integer_faithful_tiles": f"{sum(b <= 62 for b in bound_bits)}/{len(bound_bits)}",
        "capped_exponent_window": {
            "window_int_bits": WB, "max_shift_span": None if WB is None else WB - 7,
            "rule": "per tile and operand: shift = bits - E with max|v| = f 2^E (f in [0.5, 1)); unit = "
                    "round-half-even(v 2^shift), so |unit| <= 2^bits and sum_j |W||X| <= 1024 2^(2 bits) = 2^56 "
                    "(bits = 23); BF16 entries within 16 binades of the tile's top binade stay exact. W: the verifier "
                    "applies the same public window to the EXACT units opened against weight_commitment_root "
                    "(verify_drawn_tiles_against_setup(window_int_bits=23)); X: the prover's witness (hidden)",
            "exact_encoding_integer_bound_bits_per_tile": exact_bits,
            "exact_encoding_integer_faithful_tiles": f"{sum(b <= 62 for b in exact_bits)}/{len(exact_bits)}",
            "quant_window_tail_rel_err_max": max(tail_rel), "quant_window_tail_rel_err_per_tile": tail_rel,
            "w_window_tail_rel_max": max(m["w_window_tail_rel"] for m in tiles_meta),
            "x_window_tail_rel_max": max(m["x_window_tail_rel"] for m in tiles_meta),
            "w_elems_rounded_total": sum(m["w_elems_rounded"] for m in tiles_meta),
            "x_elems_rounded_total": sum(m["x_elems_rounded"] for m in tiles_meta),
            "bf16_output_half_ulp_rel": 2.0 ** -9,
            "tail_over_device_bf16_gemm_err": (max(tail_rel) / max(rel)) if max(rel) else None,
            "definitions": "quant_window_tail_rel_err = max_i |Y_window_i - Y_exact_i| / max_i |Y_exact_i| per tile, "
                           "Y_exact = exact product of the unwindowed BF16 operands, Y_window = the windowed integer "
                           "product the proof attests (rescaled), computed with exact rationals; compare with "
                           "exact_vs_device_bf16_gemm_max_rel_err (device BF16 GEMM vs the windowed exact Y) and the "
                           "BF16 output rounding radius 2^-9"},
        "exact_vs_device_bf16_gemm_max_rel_err": max(rel),
        "tile_build_ms": round(t_tiles, 1), "prove_runs": r5_runs,
        "weight_root_prover": prover_root.hex(), "weight_root_verifier": verifier_root.hex(),
        "weight_roots_match": prover_root == verifier_root,
        "global_weight_opening": {
            "weight_commitment_root": wcom.root.hex(), "open_drawn_tiles_ms": round(t_open, 1),
            "opening_bytes": len(opening_bytes), "opening_bytes_paths_only": sum(32 * len(x.path) for x in opening.tiles),
            "verify_drawn_tiles_against_setup": chk, "forgeries": forg,
            "note": "the verifier holds ONLY the SetupCertificate: inclusion of the 64 drawn tiles under "
                    "weight_commitment_root + the exact draw coordinates re-derived from C_pre, then the relation-5 "
                    "weight_root recomputed from the opened tiles is pinned in fpn-verify"},
        "draws_rederived_from_public_inputs_match": seed_v == seed and
        [(d.position, d.layer, d.gemm, d.tile_start) for d in gemm_v] ==
        [(d.position, d.layer, d.gemm, d.tile_start) for d in gemm_draws],
        "n_s_binding": {"receipt_n_s_equals_blake2b(domain, C_pre, draw coords)": n_s_ok,
                        "wrong_c_pre": "E_CPRE_MISMATCH" if not n_s_wrong_c_pre else "ACCEPTED (bug)"},
        "engine_parity_n16_real_tile": parity,
        "vllm_fused_tiles_equal_hf_safetensors_tiles": tiles_identical,
        "verifier_setup_only_check_ms": round(t_vroot, 1),
        "verify": {"rust": {"verdict": E.verdict_code(rs5[0][0]), "all_runs": [E.verdict_code(r[0]) for r in rs5],
                            "cli_wall_ms_median": round(statistics.median(r[1] for r in rs5), 2),
                            "child_cpu_ms_median": round(statistics.median(r[2] for r in rs5), 2)},
                   "python_detached": {"verdict": E.verdict_code(py5[0]), "wall_ms": round(py5[1], 1)},
                   "rust_without_pin": rs5_unpinned},
        "independent_verifier": {
            "honest": honest,
            "forgeries": trace_forgeries,
            "note": "independent_verify() sees ONLY public inputs (setup record, trace_payload_root, forwarded token ids, "
                    "response, prompt / cached lengths, external nonce, public capture layout) and the reply bundle "
                    "(trace-row openings, weight-tile openings, R4 / R5 receipts, R4 lane disclosures) and is anchored to "
                    "the reply's C_pre published before the external nonce ((d) a root rebuilt over the tampered "
                    "payload after the challenge -> E_CPRE_MISMATCH; (e) no C_pre -> E_CPRE_REQUIRED). (a) the tampered "
                    "row cannot open against trace_payload_root in C_pre -> E_TRACE_ROW_PATH; (b) an honest opening with "
                    "a receipt whose N_s binds the tampered X digest -> E_X_BINDING; (c) honest opening + honest N_s but a "
                    "relation-5 WITNESS X' (with Y' = W X'): relation 5 keeps X and Y private under its blinded root_0 "
                    "and its statement is 'there exist X, Y with Y = W X', so this verifies -- binding the hidden "
                    "witness to the opened X needs a public-X relation-5 variant (core/verifier change, out of scope); "
                    "the opened, authenticated X is what the verifier itself holds"},
        "cheat_one_pagestore_element": {
            "tile": b0, "element": jbad, "word_before": int(x_words[b0][jbad]), "word_after": int(tampered_words[jbad]),
            "with_served_outputs": {"rust": cheat_rs, "python": cheat_py},
            "with_outputs_recomputed_from_tampered_row": {"rust": rec_rs},
            "note": "relation-5 receipt alone (no trace openings): the element is perturbed in the row bytes AFTER "
                    "iter_verified_pages returned them; Y fixed at serving time -> rejected; Y recomputed from the "
                    "tampered row -> the bare receipt verifies (gap). closes it at the protocol "
                    "level: see independent_verifier.forgeries (a) / (b)"},
    }
    res["envelope"] = {"extension_keys": envs, "forbidden_keys_absent": all(k not in e for e in envs for k in forbidden),
                       "note": "checks JSON keys only; the relation-5 receipt header names fpn2-folded-vec-n22-v1, whose "
                               "profile bytes pin max_hidden = K = 1024 and (via n_vars) 64 tiles"}
    res["host"] = {"peak_rss_mib": round(rss_mib(), 1), "total_bench_s": round(time.perf_counter() - t_start, 1)}
    res["notes"] = [
        "GEMM inputs are the captured x_attn / x_mlp rows (the exact q/k/v_proj and gate/up_proj inputs); h_in is the "
        "pre-RMSNorm residual and feeds only the relation-4 RMSNorm block",
        "relation 5 proves Y = W X over F_p for integer encodings (value = int * 2^-shift, one shift per tile "
        "and operand); integer_faithful = every tile's sum |W||X| < 2^62 < p/2, so the F_p statement is the integer "
        "statement; no range check on X / Y is part of relation 5",
        "capped exponent window (default 23 bits = max_shift_span 16): W and X are re-gridded per tile so "
        "|unit| <= 2^23 (integer_faithful by construction, 64/64); entries more than 16 binades below a tile's top "
        "binade are rounded to the grid, so the proved Y is the exact product of the WINDOWED operands, which differs "
        "from the exact product of the BF16 operands by quant_window_tail_rel_err (reported per tile, exact "
        "rationals). The weight commitment stays the exact one; the window is a public verifier-side "
        "function of the opened exact units",
        "Y is the exact (infinitely precise) product of the (windowed) operands, not vLLM's rounded bf16 output; "
        "exact_vs_device_bf16_gemm_max_rel_err compares the two outside the proof",
        ": SetupCertificate.weight_commitment_root is the ONE-TIME global commit_model_weights root of the "
        "checkpoint; the relation-5 weight_root (per request, it commits the drawn tiles) is accepted only after the "
        "drawn tiles open against that setup root at the coordinates re-derived from C_pre (setup-only offline check); "
        "C_pre binds commit_trace_payload (every captured activation byte), not just the page-chain tip",
        ": trace_payload_root is a BLAKE2b Merkle root over every captured token row (+ n_tokens + capture "
        "layout fingerprint); every drawn row (relation 4 and relation 5) is opened against it and the exact X / R4 "
        "activation words extracted from the OPENED rows are digested into the relation-5 N_s and each relation-4 "
        "server nonce; payload edits made in the store BEFORE the commitment are committed as-is (capture-time "
        "authentication is serving-side); the capture-layout fingerprint is a public parameter, not pinned by the "
        "SetupCertificate; relation-4 weight-side lanes (gain, weight element, observed product) are not opened",
        "relation-4 blocks: norm channels / q_proj element seeded per draw; FULL_NO_KV does not capture K/V, so the "
        "dense_gqa slot's key / value lanes are filled from captured q / o channels (extent = 1: slot output = value0 "
        "by construction) -- the slot exercises the format, it does not check attention",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
        f.write("\n")
    print("WROTE", args.out, flush=True)
    del live
    gc.collect()
    os._exit(0)                                       # vLLM's engine teardown can hang a single-process run


if __name__ == "__main__":
    main()
