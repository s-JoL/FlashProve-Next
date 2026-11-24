# SPDX-License-Identifier: Apache-2.0
"""Live vLLM 0.33 MoE benchmark: live vLLM 0.33 serving of Qwen3-30B-A3B-FP8 (48 layers, 128-expert top-8 block-FP8 MoE)
-> fpn_capture PageStore -> C_pre -> Protocol v2 draws -> GPU relation 4 + relation 5 -> Rust fpn-verify.

    FPN_GIT_SHA=$(git rev-parse HEAD) python bench/tier2/bench_live_vllm_moe_e2e.py \
        --model <MODELS_DIR>/Qwen3-30B-A3B-FP8 --fpn-verify core/target/release/fpn-verify \
        --out benchmarks/prover/results/live_vllm_moe_30b_rtx6000d.json

Same protocol as `bench_live_vllm_e2e` (), with the MoE-specific parts replaced.  closes the two
gaps ("no FP8 GEMM was proved", "moe_router_expert unused"):

* SetupCertificate.weight_commitment_root = `commit_model_weights` over the WHOLE checkpoint: every block-128 FP8
  E4M3 matrix (attention q/k/v/o and all 128 x 48 x 3 expert projections) is tiled LOSSLESSLY (raw e4m3 bytes + its
  F32 `weight_scale_inv` block scales per leaf), the BF16 embedding / lm_head are tiled exactly, the 48 MoE routers
  are tiled as the LIVE served BF16 matrices (cross-checked = RNE(BF16) of the checkpoint's F32 `mlp.gate.weight`;
  they replace the F32 tensors in the committed set), and every remaining tensor (F32 norms) is in the residual
  digest; tensor / leaf counts and the one-time wall time are reported;
* the relation-5 draw space is the served GEMMs: q/k/v_proj (x_attn, two 1024-column halves), o_proj (o slot, four
  quarters), the ROUTED experts' gate/up_proj (x_mlp halves; the out-row space of 8 x 768 rows is the 8 captured
  top-8 slots, so a draw's expert id is read from the captured `expert_ids` row -- a PUBLIC rule the verifier
  re-applies to the OPENED row) and the BF16 router halves.  64 drawn tiles are opened against the setup root and
  proved in ONE `fpn2-folded-vec-n22-v1` container; W and X use the public capped exponent window (|unit| <= 2^23,
  `weight_commit.window_ints`) so every tile is integer faithful (sum |W||X| <= 2^56), with the window tail and the
  device FP8 GEMM deviation reported per tile;
* relation 4: the shared single-block circuit (circuit_root b2fbf5ff...) proves the dense-lane blocks of AND
  one `moe_router_expert` block from the live capture: norm pair = captured h_mid + live post_attention_layernorm
  gain, GEMM lane = captured x_mlp element x live router element (device BF16 product), slot = router logits of two
  of the captured top-8 experts and the two experts' outputs at one channel (recomputed, see scope_limits);
* independent verifier (public inputs + reply bundle only) + forgery controls.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bench_live_vllm_e2e as B  # noqa: E402  (sets sys.path + vLLM env defaults)

import numpy as np  # noqa: E402

N_LAYERS, D_MODEL, KTILE, N_EXPERTS, TOPK, EXPERT_ROWS = 48, 2048, 1024, 128, 8, 768
SHARED_CIRCUIT_ROOT_PREFIX = "b2fbf5ff"
NONCE = hashlib.blake2b(b"v1 moe external challenge nonce", digest_size=32).digest()
R4_MOE_DOMAIN = b"FPN2/rel4/moe-router-expert/v1"
GATE = "model.layers.{}.mlp.gate.weight"
TURN1 = "Explain how a mixture-of-experts transformer routes each token to its experts, step by step."
TURN2 = "Thanks. Now summarise that in exactly three short sentences for a beginner."
SIDE = "Write a haiku about verifying computations with zero-knowledge proofs."
WINDOW_BITS = 23

# (projection, X slot, 1024-column chunk, out rows in the draw space)
GEMM_TABLE = (
    ("q_proj", "x_attn", 0, 4096), ("q_proj", "x_attn", 1, 4096),
    ("k_proj", "x_attn", 0, 512), ("k_proj", "x_attn", 1, 512),
    ("v_proj", "x_attn", 0, 512), ("v_proj", "x_attn", 1, 512),
    ("o_proj", "o", 0, 2048), ("o_proj", "o", 1, 2048), ("o_proj", "o", 2, 2048), ("o_proj", "o", 3, 2048),
    ("experts.gate_proj", "x_mlp", 0, TOPK * EXPERT_ROWS), ("experts.gate_proj", "x_mlp", 1, TOPK * EXPERT_ROWS),
    ("experts.up_proj", "x_mlp", 0, TOPK * EXPERT_ROWS), ("experts.up_proj", "x_mlp", 1, TOPK * EXPERT_ROWS),
    ("router_gate", "x_mlp", 0, 128), ("router_gate", "x_mlp", 1, 128),
)
ATTN_GEMMS = 10


# ===================================================================================== MoE geometry (patched into B)
def moe_sampler_profile():
    from fpn_prover.circuit import protocol
    return protocol.SamplerProfile(N_LAYERS, tuple((KTILE, rows) for _, _, _, rows in GEMM_TABLE), q_heads=32,
                                   kv_groups=4, head_dim=128, attn_gemms=ATTN_GEMMS)


def gemm_kind(d):
    p = GEMM_TABLE[d.gemm][0]
    return "expert" if p.startswith("experts.") else "router" if p == "router_gate" else "attention"


def moe_x_slot(d):
    return GEMM_TABLE[d.gemm][1]


def x_chunk(words, d):
    c = GEMM_TABLE[d.gemm][2]
    return np.asarray(words)[KTILE * c:KTILE * (c + 1)]


def moe_rel5_x_digest(gemm_draws, words_of):
    """X of a tile = the 1024-column chunk of its captured input slot (x_attn / o / x_mlp)."""
    from fpn_prover.runtime.trace_commit import activation_binding_digest
    return activation_binding_digest(
        [(moe_x_slot(d), d.position, d.layer, x_chunk(words_of(d), d).astype("<u2").tobytes()) for d in gemm_draws],
        B.X_DOMAIN)


def tile_coord(d, words):
    """(tensor name, r0, c0) of a draw; for a routed-expert draw the expert id is the captured top-8 slot
    tile_start // 768 of the (opened) expert_ids row at (position, layer) -- a public rule on authenticated data."""
    proj, _, c, _ = GEMM_TABLE[d.gemm]
    if proj.startswith("experts."):
        if words is None:
            raise ValueError("routed-expert draws need the opened expert_ids row")
        ids = np.asarray(words(d.position, "expert_ids", d.layer))
        e = int(ids[d.tile_start // EXPERT_ROWS])
        if not 0 <= e < N_EXPERTS:
            raise ValueError("captured expert id out of range")
        return (f"model.layers.{d.layer}.mlp.experts.{e}.{proj.split('.')[1]}.weight", d.tile_start % EXPERT_ROWS,
                KTILE * c)
    if proj == "router_gate":
        return (GATE.format(d.layer), d.tile_start, KTILE * c)
    return (f"model.layers.{d.layer}.self_attn.{proj}.weight", d.tile_start, KTILE * c)


def moe_tile_coords(draws, words=None):
    return [tile_coord(d, words) for d in draws]


def moe_r4_items(disc, words_of):
    p, l, (c0, c1), i_in, j = disc["position"], disc["layer"], disc["norm_channels"], disc["in_col"], disc["slot_dim"]
    h, xm, q, o = (words_of(p, k, l) for k in ("h_in", "x_mlp", "q", "o"))
    pick = lambda w, idx: np.array([w[i] for i in idx], dtype="<u2").tobytes()  # noqa: E731
    return [("h_in", p, l, pick(h, (c0, c1))), ("x_mlp", p, l, pick(xm, (i_in,))),
            ("q", p, l, pick(q, (j, 128 + j, 256 + j))), ("o", p, l, pick(o, (j, 128 + j)))]


B.sampler_profile = moe_sampler_profile
B.x_slot = moe_x_slot
B.rel5_x_digest = moe_rel5_x_digest
B.tile_coords = moe_tile_coords
B.r4_items = moe_r4_items
B.GEMMS = tuple((f"{p}[:, {1024 * c}:{1024 * (c + 1)}]", rows) for p, _, c, rows in GEMM_TABLE)
B.ATTN_GEMMS = ATTN_GEMMS


def git_rev():
    return os.environ.get("FPN_GIT_SHA") or B.git_rev()


def rss_now_mib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def expected_gemm_share(profile, positions):
    g = profile.layers * profile.gemm_cost
    tot = sum(profile.position_cost(p) for p in positions)
    return len(positions) * g / tot if tot else 0.0


# ===================================================================================== serving
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
                                          max_batched_tokens=args.max_batched_tokens, capture_act=False))
    t0 = time.perf_counter()
    llm = LLM(model=args.model, load_format="fpn_capture", dtype="auto", enforce_eager=False,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=4096, max_num_seqs=8,
              max_num_batched_tokens=args.max_batched_tokens, enable_prefix_caching=True, seed=0)
    runner = llm.collective_rpc(cp.install_runner)
    t_load = time.perf_counter() - t0
    sess = cp.get_session()
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda msgs: list(tok.encode(tok.apply_chat_template(  # noqa: E731
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False))
    eng = llm.llm_engine

    def info(worker):
        m = worker.model_runner.get_model()
        l0 = m.model.layers[0]
        return {"gate": type(l0.mlp.gate).__name__, "gate_weight_dtype": str(l0.mlp.gate.weight.dtype),
                "experts": type(l0.mlp.experts).__name__, "router": type(l0.mlp.experts.router).__name__,
                "qkv_proj_weight_dtype": str(l0.self_attn.qkv_proj.weight.dtype),
                "input_layernorm_dtype": str(l0.input_layernorm.weight.dtype),
                "quant_method_monolithic": getattr(getattr(l0.mlp.experts, "_quant_method", None), "is_monolithic",
                                                   None)}

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

    p1 = enc([{"role": "user", "content": TURN1}])
    o1, t1 = run([("turn1", p1, args.n_new)])
    out1 = list(o1["turn1"].outputs[0].token_ids)
    st1 = sess.settle(120.0)
    p2 = p1 + out1 + enc([{"role": "user", "content": TURN2}])
    ps = enc([{"role": "user", "content": SIDE}])
    o2, t2 = run([("turn2", p2, args.n_new), ("side", ps, args.n_new)])
    st2 = sess.settle(120.0)
    out2 = list(o2["turn2"].outputs[0].token_ids)
    cached2 = int(o2["turn2"].num_cached_tokens or 0)
    traces = {**st1, **st2}
    bad = {k: v for k, v in traces.items() if isinstance(v, str)}
    if bad:
        raise SystemExit(f"capture failed: {bad}")
    key2 = next(k for k in traces if k.split("-")[0] == "turn2")
    token_ids = p2 + out2[:-1]
    res["serving"] = {
        "engine": "vLLM " + __import__("vllm").__version__, "model": os.path.basename(args.model.rstrip("/")),
        "runner": runner, "load_format": "fpn_capture", "capture_tier": "FULL_NO_KV", "capture_act": False,
        "vllm_use_deep_gemm": os.environ.get("VLLM_USE_DEEP_GEMM"), "live_modules": llm.collective_rpc(info)[0],
        "capture_slots_layer0": [(s.kind, s.width) for s in sess.layout.slots if s.layer == 0],
        "row_bytes": sess.store.row_bytes, "layout_fingerprint": sess.layout.fingerprint().hex(),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "enable_prefix_caching": True, "llm_init_s": round(t_load, 2), "max_num_batched_tokens": args.max_batched_tokens,
        "turn1": {"prompt_tokens": len(p1), "new_tokens": len(out1), "serve_s": round(t1, 4),
                  "text": tok.decode(out1, skip_special_tokens=True)[:200]},
        "turn2_batch_with_side": {"turn2_prompt_tokens": len(p2), "turn2_cached_tokens": cached2,
                                  "side_prompt_tokens": len(ps), "new_tokens_each": args.n_new, "serve_s": round(t2, 4),
                                  "text": tok.decode(out2, skip_special_tokens=True)[:200]},
        "traces": {k: {"n_tokens": v.n_tokens, "n_full_pages": v.n_full_pages} for k, v in traces.items()},
        "sink_stats": {k: v for k, v in dict(sess.sink.stats).items() if isinstance(v, (int, float))},
        "store_stats": dict(sess.store.stats), "ring_overflow_events": sess.ring.overflow_events,
    }
    print(json.dumps(res["serving"], indent=1, default=str), flush=True)
    return dict(llm=llm, sess=sess, store=sess.store, trace=traces[key2], key=key2, prompt=p2, out=out2,
                token_ids=token_ids, cached=cached2)


def live_moe_weights(llm):
    def grab(worker):
        import torch
        m = worker.model_runner.get_model()
        words = lambda t: t.detach().contiguous().view(torch.int16).cpu().numpy()  # noqa: E731
        return {l: {"gate": words(m.model.layers[l].mlp.gate.weight),
                    "ln1": words(m.model.layers[l].input_layernorm.weight),
                    "ln2": words(m.model.layers[l].post_attention_layernorm.weight)}
                for l in range(len(m.model.layers))}
    got = llm.collective_rpc(grab)[0]
    return {l: {k: v.view(np.uint16) for k, v in d.items()} for l, d in got.items()}


class OverlayCheckpoint:
    """Safetensors checkpoint with some tensors REPLACED by in-memory arrays (the live served BF16 routers)."""

    def __init__(self, base, overlay):
        from fpn_prover.runtime import weight_commit as WC
        self.base = base
        self.ov = WC.ArrayCheckpoint(overlay)

    def names(self):
        return sorted(set(self.base.names()) | set(self.ov.names()))

    def _src(self, name):
        return self.ov if name in self.ov.t else self.base

    def dtype(self, name):
        return self._src(name).dtype(name)

    def shape(self, name):
        return self._src(name).shape(name)

    def rows(self, name, start, stop):
        return self._src(name).rows(name, start, stop)

    def raw(self, name):
        return self._src(name).raw(name)


def moe_r4_samples(draws, rows, lw, count, model_name):
    import random

    import torch

    from fpn_prover.circuit import e2e_prover as E
    from fpn_prover.circuit import lower as LW
    out, meta = [], []
    for d in draws[:count]:
        r = rows[d.position]
        h, xm, q, o = r[("h_in", d.layer)], r[("x_mlp", d.layer)], r[("q", d.layer)], r[("o", d.layer)]
        gain = lw[d.layer]["ln1"]
        rnd = random.Random(f"v1:moe:r4:{d.position}:{d.layer}")
        order = list(range(len(h)))
        rnd.shuffle(order)
        pair = None
        for i in range(0, len(order) - 1, 2):
            c0, c1 = order[i], order[i + 1]
            cand = E.Samples("dense_gqa_qknorm", model_name, (int(h[c0]), int(h[c1])), (int(gain[c0]), int(gain[c1])),
                             0x3f80, 0x3f80, 0x3f80, "dense_gqa", dict(normalize=0, query=0, key0=0, key1=0, value0=0,
                                                                         value1=0))
            try:
                E.build_witness(cand)
                pair = (c0, c1)
                break
            except LW.LoweringError:
                continue
        G = lw[d.layer]["gate"]
        i_in, o_row = rnd.randrange(D_MODEL), rnd.randrange(128)
        while not (int(xm[i_in]) & 0x7fff and int(G[o_row][i_in]) & 0x7fff):
            i_in, o_row = rnd.randrange(D_MODEL), rnd.randrange(128)
        w_word = int(G[o_row][i_in])
        xt = torch.tensor([B.bf16_f(xm[i_in])], dtype=torch.bfloat16, device="cuda")
        wt = torch.tensor([B.bf16_f(w_word)], dtype=torch.bfloat16, device="cuda")
        prod = int(torch.mul(xt, wt).view(torch.int16).item()) & 0xFFFF
        j = int(np.abs(np.array([B.bf16_f(v) for v in o[:128]])).argmax())
        slot = dict(normalize=1, query=B.bf16_f(q[j]), key0=B.bf16_f(q[128 + j]), key1=B.bf16_f(q[256 + j]),
                    value0=B.bf16_f(o[j]), value1=B.bf16_f(o[128 + j]))
        out.append(E.Samples("dense_gqa_qknorm", model_name, (int(h[pair[0]]), int(h[pair[1]])),
                             (int(gain[pair[0]]), int(gain[pair[1]])), int(xm[i_in]), w_word, prod, "dense_gqa", slot))
        meta.append({"position": d.position, "layer": d.layer, "draw_kind": d.kind, "stratum": d.stratum,
                     "norm_channels": list(pair), "gemm": {"projection": "mlp.gate (MoE router)", "out_row": o_row,
                                                           "in_col": i_in}, "slot_dim": j})
    return out, meta


def capture_integrity(store, pages, layout, lw):
    """Every slot of every captured row written (no constant / poison rows), expert ids in range and distinct, and
    an outside-the-protocol cross-check of the tapped ids against top-8 of the exact router logits recomputed from
    the captured x_mlp and the live BF16 router weights."""
    rows = b"".join(r for _, _, r in pages)
    rb = store.row_bytes
    A = np.frombuffer(rows, dtype="<u2").reshape(len(rows) // rb, rb // 2)
    const = {}
    for s in layout.slots:
        blk = A[:, s.offset:s.offset + s.width]
        n = int((blk == blk[:, :1]).all(axis=1).sum()) if s.width > 1 else 0
        if n:
            const[f"{s.kind}"] = const.get(s.kind, 0) + n
    ids_ok, mism, tot, margin_mism = True, 0, 0, []
    for l in range(N_LAYERS):
        es = layout.slot("expert_ids", l)
        ids = A[:, es.offset:es.offset + es.width].astype(np.int64)
        ids_ok &= bool((ids < 128).all()) and all(len(set(r.tolist())) == 8 for r in ids)
        xs = layout.slot("x_mlp", l)
        x = (A[:, xs.offset:xs.offset + xs.width].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        g = (lw[l]["gate"].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        logits = x @ g.T
        order = np.argsort(-logits, axis=1)
        for r in range(len(A)):
            tot += 1
            if set(order[r, :8].tolist()) != set(ids[r].tolist()):
                mism += 1
                margin_mism.append(float(logits[r, order[r, 7]] - logits[r, order[r, 8]]))
    return {"rows": int(len(A)), "constant_rows_per_slot_kind": const, "all_slots_written": not const,
            "expert_ids_in_range_and_distinct": ids_ok,
            "expert_ids_vs_fp64_router_top8": {"row_layer_pairs": tot, "set_mismatches": mism,
                                               "max_logit_gap_8th_vs_9th_at_mismatch": max(margin_mism, default=None),
                                               "note": "outside the protocol: vLLM routes on its BF16 router GEMM "
                                                       "output, this recompute uses exact fp64 logits, so near-ties "
                                                       "at the 8th / 9th expert can differ; the tapped ids are the "
                                                       "kernel's own routed topk_ids (BaseRouter.capture_fn)"}}




# ===================================================================================== helpers
def dyadic_ints(values):
    """Exact float64 values -> (Python-int object array, shift): value = int * 2^-shift exactly (one shift)."""
    v = np.asarray(values, dtype=np.float64)
    f, e = np.frexp(v)
    m = np.ldexp(f, 53).astype(np.int64)
    ex = e.astype(np.int64) - 53
    nz = m != 0
    out = np.zeros(v.shape, dtype=object)
    if not nz.any():
        return out, 0
    emin = int(ex[nz].min())
    flat = out.reshape(-1)
    for i in np.flatnonzero(nz.reshape(-1)):
        flat[i] = int(m.reshape(-1)[i]) << int(ex.reshape(-1)[i] - emin)
    return out, -emin


def exact_rel_diff(y_a, s_a, y_b, s_b):
    """max_i |y_a_i 2^-s_a - y_b_i 2^-s_b| / max_i |y_b_i 2^-s_b| with exact integer alignment."""
    S = max(s_a, s_b)
    a = [int(v) << (S - s_a) for v in y_a]
    b = [int(v) << (S - s_b) for v in y_b]
    mx = max(abs(v) for v in b)
    return float(Fraction(max(abs(x - y) for x, y in zip(a, b)), mx)) if mx else 0.0


def live_fp8_and_device(llm, jobs):
    """Prover-side, outside the protocol.  For every job {kind, layer, expert, proj, r0, c0, x (uint16 words of the
    full input row)}: (1) the live vLLM FP8 bytes + block scales of the tile (equality with the committed tile),
    (2) the DEVICE output rows r0..r0+32 of the served path for the full input row: attention -> the live
    qkv_proj / o_proj module (vLLM Fp8LinearMethod, dynamic per-token-group activation quantisation), expert ->
    vLLM's `per_token_group_quant_fp8` + `w8a8_triton_block_scaled_mm` on the live expert's w13 block (the fused-MoE
    kernel itself takes no single-expert input), router -> torch BF16 linear on the live GateLinear weight."""
    def run(worker):
        import torch
        import vllm.model_executor.layers.quantization.utils.fp8_utils as F8
        m = worker.model_runner.get_model()
        out = []
        for j in jobs:
            L = m.model.layers[j["layer"]]
            x = torch.from_numpy(j["x"].view(np.int16).copy()).view(torch.bfloat16).cuda()[None]
            r = {"device_kernel": None, "live_raw": None, "live_scales": None, "device_rows": None, "error": None}
            try:
                r0, c0 = j["r0"], j["c0"]
                if j["kind"] == "attention":
                    mod = L.self_attn.qkv_proj if j["proj"] in ("q_proj", "k_proj", "v_proj") else L.self_attn.o_proj
                    base = {"q_proj": 0, "k_proj": 4096, "v_proj": 4608}.get(j["proj"], 0)
                    w, s = mod.weight, mod.weight_scale_inv
                    y = mod(x)
                    y = y[0] if isinstance(y, tuple) else y
                    r["device_rows"] = y[0, base + r0:base + r0 + 32].float().cpu().numpy()
                    r["device_kernel"] = f"live {type(mod).__name__}.forward ({type(mod.quant_method).__name__})"
                elif j["kind"] == "expert":
                    ex = L.mlp.experts
                    params = dict(ex.named_parameters())
                    w13 = params["routed_experts.w13_weight"][j["expert"]]
                    s13 = params["routed_experts.w13_weight_scale_inv"][j["expert"]]
                    base = 0 if j["proj"] == "gate_proj" else EXPERT_ROWS
                    w, s = w13, s13
                    try:
                        aq, as_ = F8.per_token_group_quant_fp8(x, 128)
                        y = F8.w8a8_triton_block_scaled_mm(aq, w13, as_, s13, [128, 128], output_dtype=torch.bfloat16)
                        r["device_kernel"] = "vllm per_token_group_quant_fp8 + w8a8_triton_block_scaled_mm (live w13)"
                    except Exception as exc:                        # emulate the same W8A8 block semantics
                        r["error"] = f"triton block mm unavailable: {type(exc).__name__}: {exc}"[:300]
                        xg = x.float().view(1, -1, 128)
                        sc = xg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
                        xq = (xg / sc).to(torch.float8_e4m3fn).float() * sc
                        wd = w13.float() * s13.repeat_interleave(128, 0)[:w13.shape[0]].repeat_interleave(128, 1)
                        y = xq.view(1, -1) @ wd.T
                        r["device_kernel"] = "torch emulation of W8A8 block-128 (fallback)"
                    r["device_rows"] = y[0, base + r0:base + r0 + 32].float().cpu().numpy()
                    r0 = base + r0
                else:
                    g = L.mlp.gate.weight
                    y = torch.nn.functional.linear(x, g)
                    r["device_rows"] = y[0, r0:r0 + 32].float().cpu().numpy()
                    r["device_kernel"] = "torch BF16 linear on the live GateLinear weight"
                    w = s = None
                if w is not None:
                    if j["kind"] == "attention":
                        r0 = base + r0
                    r["live_raw"] = w[r0:r0 + 32, c0:c0 + KTILE].contiguous().view(torch.uint8).cpu().numpy()
                    r["live_scales"] = s[r0 // 128, c0 // 128:(c0 + KTILE) // 128].contiguous().cpu().numpy()
            except Exception as exc:
                r["error"] = f"{type(exc).__name__}: {exc}"[:300]
            out.append(r)
        return out
    return llm.collective_rpc(run)[0]


def moe_expert_outputs(ck, layer, experts, x_vals):
    """fp64 reference outputs (2048,) of the given experts on input x (dequantised FP8 checkpoint weights)."""
    from fpn_prover.runtime import weight_commit as WC

    def deq(name):
        rows, cols = ck.shape(name)
        raw = np.asarray(ck.rows(name, 0, rows), dtype=np.uint8)
        sn = WC.scale_name(name)
        sc = np.asarray(ck.rows(sn, 0, ck.shape(sn)[0]))
        sc = sc.view("<f4") if sc.dtype.kind == "u" else sc
        full = np.repeat(np.repeat(sc.astype(np.float64), 128, 0)[:rows], 128, 1)[:, :cols]
        return WC.E4M3_TABLE[raw] * full
    out = {}
    for e in experts:
        pre = f"model.layers.{layer}.mlp.experts.{e}."
        g, u, d = deq(pre + "gate_proj.weight"), deq(pre + "up_proj.weight"), deq(pre + "down_proj.weight")
        a = g @ x_vals
        h = a / (1.0 + np.exp(-a)) * (u @ x_vals)
        out[e] = d @ h
    return out


def build_moe_block(c_pre, p, l, rows, lw, ck, model_name):
    """One relation-4 `moe_router_expert` block from the live capture at (p, l).  Returns (sample, meta, nonce,
    items) or raises if no lowering-valid expert pair exists among the captured top-8."""
    import itertools
    import random

    import torch

    from fpn_prover.circuit import e2e_prover as E
    from fpn_prover.circuit import lower as LW
    from fpn_prover.runtime import weight_commit as WC
    r = rows[p]
    hm, xm, ids = r[("h_mid", l)], r[("x_mlp", l)], [int(v) for v in r[("expert_ids", l)]]
    gain, G = lw[l]["ln2"], lw[l]["gate"]
    xv = WC.bf16_values(xm)
    xt = torch.from_numpy(xm.view(np.int16).copy()).view(torch.bfloat16).cuda()
    Gt = torch.from_numpy(G.view(np.int16).copy()).view(torch.bfloat16).cuda()
    logits = (xt.float()[None] @ Gt.float().T)[0].cpu().numpy().astype(np.float64)     # fp32 router logits
    outs = moe_expert_outputs(ck, l, ids, xv)
    w = np.exp(logits[ids] - logits[ids].max())
    w = w / w.sum()                                                    # norm_topk_prob: renormalised top-8
    mix = sum(wi * outs[e] for wi, e in zip(w, ids))
    rnd = random.Random(f"v1:moe-router-expert:{p}:{l}")
    order = list(range(len(hm)))
    rnd.shuffle(order)
    pair = None
    for i in range(0, len(order) - 1, 2):
        c0, c1 = order[i], order[i + 1]
        cand = E.Samples("moe_qwen3_fp8", model_name, (int(hm[c0]), int(hm[c1])), (int(gain[c0]), int(gain[c1])),
                         0x3f80, 0x3f80, 0x3f80, "dense_gqa", dict(normalize=0, query=0, key0=0, key1=0, value0=0,
                                                                     value1=0))
        try:
            E.build_witness(cand)
            pair = (c0, c1)
            break
        except LW.LoweringError:
            continue
    ch = int(np.abs(mix).argmax())
    i_in = None
    for i in rnd.sample(range(D_MODEL), D_MODEL):
        if int(xm[i]) & 0x7fff and int(G[ids[0]][i]) & 0x7fff:
            i_in = i
            break
    last_err = None
    for a, b in itertools.combinations(range(TOPK), 2):
        ea, eb = ids[a], ids[b]
        w_word = int(G[ea][i_in])
        if not w_word & 0x7fff:
            continue
        xe = torch.tensor([B.bf16_f(xm[i_in])], dtype=torch.bfloat16, device="cuda")
        we = torch.tensor([B.bf16_f(w_word)], dtype=torch.bfloat16, device="cuda")
        prod = int(torch.mul(xe, we).view(torch.int16).item()) & 0xFFFF
        slot = dict(scores=(float(logits[ea]), float(logits[eb])), outputs=(float(outs[ea][ch]), float(outs[eb][ch])))
        s = E.Samples("moe_qwen3_fp8", model_name, (int(hm[pair[0]]), int(hm[pair[1]])),
                      (int(gain[pair[0]]), int(gain[pair[1]])), int(xm[i_in]), w_word, prod, "moe_router_expert", slot)
        try:
            E.build_witness(s)
        except (LW.LoweringError, ValueError) as exc:
            last_err = exc
            continue
        items = moe_items(p, l, pair, i_in, (a, b), lambda pp, k, ll: rows[pp][(k, ll)])
        nonce = moe_nonce(c_pre, items)
        meta = {"position": p, "layer": l, "captured_top8_expert_ids": ids, "slot_pair": [a, b],
                "experts": [ea, eb], "router_logits_fp32": [float(logits[ea]), float(logits[eb])],
                "expert_outputs_channel": ch, "expert_outputs": [float(outs[ea][ch]), float(outs[eb][ch])],
                "norm": {"input": "captured h_mid", "gain": "live post_attention_layernorm", "channels": list(pair)},
                "gemm_lane": {"x": "captured x_mlp", "w": f"live BF16 router row {ea}", "in_col": i_in,
                              "observed": "device BF16 product"},
                "top8_mixture_vs_captured_residual_delta": None}
        return s, meta, nonce, items, mix
    raise RuntimeError(f"no lowering-valid expert pair among the captured top-8: {last_err}")


def moe_items(p, l, pair, i_in, slots, words_of):
    """Activation words the moe_router_expert block consumes (bound into its server nonce)."""
    hm, xm, ids = (words_of(p, k, l) for k in ("h_mid", "x_mlp", "expert_ids"))
    pick = lambda w, idx: np.array([w[i] for i in idx], dtype="<u2").tobytes()  # noqa: E731
    return [("h_mid", p, l, pick(hm, pair)), ("x_mlp", p, l, np.asarray(xm, dtype="<u2").tobytes()),
            ("x_mlp_lane", p, l, pick(xm, (i_in,))), ("expert_ids", p, l, pick(ids, slots))]


def moe_nonce(c_pre, items):
    from fpn_prover.runtime.trace_commit import activation_binding_digest
    return hashlib.blake2b(R4_MOE_DOMAIN + c_pre + activation_binding_digest(items, R4_MOE_DOMAIN),
                           digest_size=32).digest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="<MODELS_DIR>/Qwen3-30B-A3B-FP8")
    ap.add_argument("--fpn-verify", default=os.path.join(ROOT, "core", "target", "release", "fpn-verify"))
    ap.add_argument("--out", default=os.path.join(ROOT, "internal", "results", "m3", "live_vllm_moe_30b_rtx6000d.json"))
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--r4-blocks", type=int, default=3)
    ap.add_argument("--draw-margin", type=float, default=1.6)
    ap.add_argument("--ring-mib", type=int, default=1536)
    ap.add_argument("--max-batched-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--r5-reps", type=int, default=3)
    ap.add_argument("--verify-runs", type=int, default=5)
    ap.add_argument("--commit-threads", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="import + geometry checks only (no GPU)")
    args = ap.parse_args(argv)
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"          # DeepGEMM JIT needs NVCC >= 12.9; rtx6000d-node ships CUDA 12.8
    prof = moe_sampler_profile()
    if args.dry_run:
        print(json.dumps({"profile": str(prof), "gemm_cost": prof.gemm_cost,
                          "share_at_100": expected_gemm_share(prof, range(60, 140))}))
        return
    t_start = time.perf_counter()
    from flashprove_next.field import native, native_threads
    if native() is None:
        raise SystemExit("flashprove_next native library unavailable")
    model_name = os.path.basename(args.model.rstrip("/"))
    res = {"bench": "tier2/bench_live_vllm_moe_e2e", "wave": "13A", "milestone": "v1", "evidence": "MEASURED",
           "git_rev": git_rev(), "command": " ".join(sys.argv),
           "machine": {"host": platform.node(), "python": platform.python_version(), "native_threads": native_threads()}}
    try:
        res["machine"]["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
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
    from fpn_prover.circuit import protocol
    from fpn_prover.runtime import trace_commit as TC
    from fpn_prover.runtime import weight_commit as WC
    from fpn_prover.runtime.pipeline import verify_reply_with_setup
    from fpn_prover.runtime.setup_cert import SetupCertificate

    lw = live_moe_weights(live["llm"])
    handle = live["store"].handle(live["key"])
    pages = list(handle.iter_verified_pages(live["token_ids"]))
    res["capture_integrity"] = capture_integrity(live["store"], pages, live["sess"].layout, lw)
    print(json.dumps(res["capture_integrity"]), flush=True)

    # ---------------------------------------------------------------- setup: WHOLE checkpoint (FP8 lossless + routers)
    gates = {GATE.format(l): lw[l]["gate"] for l in range(N_LAYERS)}
    rne_equal = 0
    idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
    for l in range(N_LAYERS):
        name = GATE.format(l)
        with safe_open(os.path.join(args.model, idx[name]), framework="pt") as st:
            ref = st.get_tensor(name).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        rne_equal += bool(np.array_equal(ref, gates[name]))
    ck = OverlayCheckpoint(WC.SafetensorsCheckpoint(args.model), gates)
    t = time.perf_counter()
    wcom = WC.commit_model_weights(ck, threads=args.commit_threads, window_int_bits=WINDOW_BITS)
    t_wc = (time.perf_counter() - t) * 1e3
    kinds = {}
    for e in wcom.entries:
        k = ("fp8_expert" if ".experts." in e.name else "fp8_attention" if e.kind == WC.KIND_FP8 else
             "bf16_router" if e.name.endswith("mlp.gate.weight") else "bf16_other")
        kinds.setdefault(k, {"tensors": 0, "leaves": 0})
        kinds[k]["tensors"] += 1
        kinds[k]["leaves"] += e.leaves
    prog, _ = E.build_program()
    circuit = LW.compile_air(prog)
    if not circuit.root.hex().startswith(SHARED_CIRCUIT_ROOT_PREFIX):
        raise SystemExit(f"relation-4 circuit root {circuit.root.hex()} is not the shared b2fbf5ff... root")
    setup = SetupCertificate(
        wcom.root, circuit.root, hashlib.blake2b(b"fp8_e4m3_block128+bf16_fp32acc/window23", digest_size=32).digest(),
        hashlib.blake2b(b"Qwen3-30B-A3B-FP8/safetensors/fp8-block128+live-bf16-router", digest_size=32).digest(),
        0).encode()
    st_ = dict(wcom.stats)
    res["setup"] = {"setup_cert_id": SetupCertificate.decode(setup).id, "weight_commitment_root": wcom.root.hex(),
                    "weight_commit_scope": "the WHOLE checkpoint: every block-128 FP8 E4M3 matrix tiled losslessly "
                                           "(raw e4m3 bytes + F32 weight_scale_inv per leaf), BF16 embed / lm_head "
                                           "tiled exactly, the 48 routers as the LIVE served BF16 matrices (replacing "
                                           "the F32 checkpoint tensors), every other tensor in the residual digest",
                    "weight_commit_stats": {k: v for k, v in st_.items() if k != "skipped"},
                    "residual_tensor_names_sample": [s["name"] for s in st_["skipped"][:8]],
                    "committed_by_kind": kinds, "weight_commit_wall_ms": round(t_wc, 1),
                    "manifest_version": st_["manifest_version"], "fp8_window_int_bits_bound": WINDOW_BITS,
                    "live_router_equals_rne_bf16_of_checkpoint_f32": f"{rne_equal}/{N_LAYERS}",
                    "circuit_root": circuit.root.hex(),
                    "circuit_root_is_shared_dense_qwen3_0_6b_root": circuit.root.hex().startswith(
                        SHARED_CIRCUIT_ROOT_PREFIX),
                    "note": "one-time setup (not in any per-request timing)"}
    print(json.dumps({k: v for k, v in res["setup"].items() if k != "residual_tensor_names_sample"}, default=str),
          flush=True)

    # ---------------------------------------------------------------- C_pre -> draws
    t = time.perf_counter()
    tree = TC.commit_trace_merkle(live["store"], live["trace"], live["key"], live["token_ids"])
    t_commit = (time.perf_counter() - t) * 1e3
    payload_root = tree.root
    gen = protocol.generation_stratum(len(live["prompt"]), len(live["out"]))
    fr = protocol.fresh_input_stratum(len(live["prompt"]), live["cached"])
    fr = range(fr.start, min(fr.stop, gen.start))
    share = expected_gemm_share(prof, fr) + expected_gemm_share(prof, gen)
    draws = int(math.ceil(args.draw_margin * args.tiles / share))
    t = time.perf_counter()
    public = (payload_root, live["token_ids"], live["out"], len(live["prompt"]), live["cached"], draws)
    c_pre, seed, flat = B.protocol_draws(setup, *public, nonce=NONCE)
    t_draw = (time.perf_counter() - t) * 1e3
    gemm_draws = [d for d in flat if d.kind == "gemm"][:args.tiles]
    if len(gemm_draws) < args.tiles:
        raise SystemExit(f"only {len(gemm_draws)} GEMM draws")
    assert min(d.position for d in flat) >= live["cached"]
    bundle = TC.open_trace_rows(tree, {d.position for d in flat})
    trace_openings = bundle.encode()
    rows = B.read_rows(live["store"], pages, {d.position for d in flat})
    words_p = lambda p, k, l: rows[p][(k, l)]  # noqa: E731
    res["protocol"] = {"challenge_mode": protocol.MODE_EXTERNAL_SEED, "c_pre": c_pre.hex(), "seed": seed.hex(),
                       "trace_payload_root": payload_root.hex(), "trace_payload_tokens": tree.n_tokens,
                       "sampler_profile": {"layers": prof.layers, "gemms": [list(x) for x in GEMM_TABLE],
                                           "q_heads": prof.q_heads, "kv_groups": prof.kv_groups,
                                           "head_dim": prof.head_dim, "attn_gemms": prof.attn_gemms,
                                           "note": "GEMM draw space = the served GEMMs (FP8 q/k/v/o, FP8 ROUTED "
                                                   "expert gate/up over the 8 captured top-8 slots, BF16 router), "
                                                   "each as 1024-column chunks; down_proj is not in the draw space "
                                                   "(its input, the expert activation, is not captured)"},
                       "draws_per_stratum": draws, "expected_gemm_draws_per_pair": round(share, 5),
                       "nonempty_draws": len(flat), "gemm_draws": sum(d.kind == "gemm" for d in flat),
                       "attn_draws": sum(d.kind == "attn" for d in flat),
                       "cached_prefix_positions_without_draws": live["cached"],
                       "min_drawn_position": min(d.position for d in flat),
                       "trace_row_openings": {"positions": len(bundle.openings), "bytes": len(trace_openings)},
                       "timing_ms": {"commit_trace_merkle": round(t_commit, 2), "protocol_draws": round(t_draw, 2)}}
    engine = CE.CudaEngine()

    # ---------------------------------------------------------------- relation 5: 64 drawn FP8 / BF16 tiles
    t = time.perf_counter()
    coords = moe_tile_coords(gemm_draws, words_p)
    opening = WC.open_drawn_tiles(wcom, ck, coords)
    opening_bytes = opening.encode()
    t_open = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    wint = opening.weights_int(bf16_window=WINDOW_BITS)
    W_int, X_int, tiles_meta, jobs = [], [], [], []
    for b, d in enumerate(gemm_draws):
        tile = opening.tiles[b]
        wi, sw = wint[b]
        slot = moe_x_slot(d)
        xfull = rows[d.position][(slot, d.layer)]
        xw = x_chunk(xfull, d)
        xi, sx = WC.window_ints(WC.bf16_values(xw), WINDOW_BITS)
        W_int.append(wi)
        X_int.append(xi)
        wv, xv = tile.values(), WC.bf16_values(xw)
        wt_err, wt_max = WC.window_tail(wv, wi, sw)
        xt_err, xt_max = WC.window_tail(xv, xi, sx)
        proj, _, chunk, _ = GEMM_TABLE[d.gemm]
        kind = gemm_kind(d)
        expert = int(coords[b][0].split(".experts.")[1].split(".")[0]) if kind == "expert" else None
        tiles_meta.append({"position": d.position, "layer": d.layer, "kind": kind, "tensor": coords[b][0],
                           "rows": [coords[b][1], coords[b][1] + 32], "cols": [coords[b][2], coords[b][2] + KTILE],
                           "x_slot": slot, "expert": expert,
                           "top8_slot": d.tile_start // EXPERT_ROWS if kind == "expert" else None,
                           "dtype": "F8_E4M3/B128" if tile.kind == WC.KIND_FP8 else "BF16",
                           "w_shift": sw, "x_shift": sx,
                           "w_window_tail_rel": wt_err / wt_max if wt_max else 0.0,
                           "x_window_tail_rel": xt_err / xt_max if xt_max else 0.0,
                           "w_elems_rounded": int((np.ldexp(wi.astype(np.float64), -sw) != wv).sum()),
                           "x_elems_rounded": int((np.ldexp(xi.astype(np.float64), -sx) != xv).sum())})
        jobs.append({"kind": kind, "layer": d.layer, "expert": expert,
                     "proj": proj.split(".")[-1], "r0": coords[b][1], "c0": coords[b][2], "x": np.asarray(xfull)})
    W_int, X_int = np.array(W_int, dtype=np.int64), np.array(X_int, dtype=np.int64)
    lay = FV.layout_of(FOLDED_VEC_N22_V1)
    ys, bits = [], []
    for b in range(lay.B):
        yb, bb = FV.exact_products(W_int[b:b + 1], X_int[b:b + 1])
        ys.append(yb[0])
        bits.append(bb)
    Y_int = np.array(ys, dtype=object)
    Wf, Xf, Yf = FV.to_field(W_int), FV.to_field(X_int), FV.to_field(Y_int)
    t_tiles = (time.perf_counter() - t) * 1e3
    # window tail: windowed Y vs the EXACT product of the unwindowed operands (exact rationals)
    t = time.perf_counter()
    for b, d in enumerate(gemm_draws):
        m = tiles_meta[b]
        we, se = dyadic_ints(opening.tiles[b].values())
        xe, sxe = dyadic_ints(WC.bf16_values(x_chunk(rows[d.position][(m["x_slot"], d.layer)], d)))
        ye, _ = FV.exact_products(we[None], xe[None])
        m["quant_window_tail_rel_err"] = exact_rel_diff(Y_int[b], m["w_shift"] + m["x_shift"], ye[0], se + sxe)
    t_tail = (time.perf_counter() - t) * 1e3
    # device: live FP8 bytes == committed tile; served-path output vs exact full-K reference (outside the proof)
    t = time.perf_counter()
    dev = live_fp8_and_device(live["llm"], jobs)
    for b, d in enumerate(gemm_draws):
        m, r, tile = tiles_meta[b], dev[b], opening.tiles[b]
        m["device_kernel"], m["device_error"] = r["device_kernel"], r["error"]
        if tile.kind == WC.KIND_FP8:
            m["live_fp8_bytes_equal_committed"] = (r["live_raw"] is not None
                                                   and bool(np.array_equal(r["live_raw"], tile.raw)))
            m["live_scales_equal_committed"] = (r["live_scales"] is not None and bool(
                np.array_equal(np.asarray(r["live_scales"], dtype="<f4").view("<u4"), tile.scales.view("<u4"))))
        else:
            live_t = lw[d.layer]["gate"][m["rows"][0]:m["rows"][0] + 32, m["cols"][0]:m["cols"][1]]
            m["live_bf16_router_equal_committed"] = bool(np.array_equal(WC.bf16_values(live_t), tile.values()))
        # exact full-K reference rows (fp64 of exact dequantised products) for the served full input row
        name, r0, _ = coords[b]
        K_full = ck.shape(name)[1]
        full = WC.open_drawn_tiles(wcom, ck, [(name, r0, c) for c in range(0, K_full, KTILE)])
        Wrow = np.concatenate([t_.values() for t_ in full.tiles], axis=1)[:, :K_full]
        xrow = WC.bf16_values(rows[d.position][(m["x_slot"], d.layer)])[:K_full]
        ref = Wrow @ xrow
        if r["device_rows"] is not None:
            mx = float(np.abs(ref).max()) or 1.0
            m["exact_vs_device_gemm_rel_err"] = float(np.abs(np.asarray(r["device_rows"], dtype=np.float64)
                                                             - ref).max() / mx)
    t_dev = (time.perf_counter() - t) * 1e3
    n_s = B.rel5_n_s(c_pre, gemm_draws, B.rel5_x_digest(gemm_draws, lambda d: rows[d.position][(moe_x_slot(d),
                                                                                                    d.layer)]))
    r5_runs = []
    for rep in range(args.r5_reps):
        FV._W_CACHE.clear()
        tm = {}
        t = time.perf_counter()
        r5 = FV.prove(Wf, Xf, Yf, n_s=n_s, profile=FOLDED_VEC_N22_V1, engine=engine, hooks={"timings": tm})
        r5_runs.append({"rep": rep, "prove_wall_ms": round((time.perf_counter() - t) * 1e3, 1),
                        "phases_ms": {k: round(v, 1) for k, v in tm.items()}})
        print(json.dumps(r5_runs[-1]), flush=True)
    t = time.perf_counter()
    chk = WC.verify_drawn_tiles_against_setup(setup, opening_bytes, r5, coords, engine=engine,
                                              window_int_bits=WINDOW_BITS)
    t_chk = (time.perf_counter() - t) * 1e3
    vroot = bytes.fromhex(chk["weight_root"]) if chk["verdict"] == "VERIFIED" else bytes(32)
    rs5 = [E.rust_verify(args.fpn_verify, r5, vroot) for _ in range(args.verify_runs)]
    py5 = E.python_verify_detached(r5, vroot)
    print(json.dumps({"r5_setup_check": chk["verdict"], "rust": E.verdict_code(rs5[0][0]),
                      "integer_bits_max": max(bits)}), flush=True)

    # ---------------------------------------------------------------- relation 4: dense-lane blocks (path)
    samples, r4meta = moe_r4_samples(flat, rows, lw, args.r4_blocks, model_name)
    r4_disc = [{"position": m["position"], "layer": m["layer"], "norm_channels": m["norm_channels"],
                "in_col": m["gemm"]["in_col"], "slot_dim": m["slot_dim"]} for m in r4meta]
    r4, r4_receipts = [], []
    for i, s in enumerate(samples):
        nonce_i = B.r4_nonce(c_pre, r4_disc[i], words_p)
        t = time.perf_counter()
        pr = E.prove_samples(s, server_nonce=nonce_i, engine=engine)
        wall = (time.perf_counter() - t) * 1e3
        assert pr.circuit.root == circuit.root
        r4_receipts.append(pr.container)
        ver = verify_reply_with_setup(setup, pr.container, expected_c_pre=nonce_i, circuit_digest=pr.circuit.digest,
                                      rust_binary=args.fpn_verify, check_python=True)
        rs = [E.rust_verify(args.fpn_verify, pr.container, circuit.root) for _ in range(args.verify_runs)]
        r4.append({**r4meta[i], "prove_wall_ms": round(wall, 1), "container_bytes": len(pr.container), "verify": ver,
                   "rust_cli_wall_ms_median": round(statistics.median(r[1] for r in rs), 2),
                   "sha256": hashlib.sha256(pr.container).hexdigest()})
        print(json.dumps(r4[-1], default=str), flush=True)

    # ---------------------------------------------------------------- relation 4: moe_router_expert block (live)
    dmoe = flat[args.r4_blocks]
    ms, mmeta, mnonce, mitems, mix = build_moe_block(c_pre, dmoe.position, dmoe.layer, rows, lw, ck, model_name)
    if dmoe.layer + 1 < N_LAYERS:
        # outside the protocol: the recomputed top-8 mixture vs the captured residual delta h_in[l+1] - h_mid[l]
        hn = WC.bf16_values(rows[dmoe.position][("h_in", dmoe.layer + 1)])
        hmid = WC.bf16_values(rows[dmoe.position][("h_mid", dmoe.layer)])
        delta = hn - hmid
        mmeta["top8_mixture_vs_captured_residual_delta"] = {
            "rel_err_max": float(np.abs(mix - delta).max() / (np.abs(delta).max() or 1.0)),
            "cosine": float(mix @ delta / (np.linalg.norm(mix) * np.linalg.norm(delta) or 1.0)),
            "note": "h_in[l+1] - h_mid[l] is the BF16 residual delta the served fused MoE added; the reference mixes "
                    "fp64 expert outputs of the dequantised FP8 checkpoint with renormalised fp32 router softmax "
                    "(the served kernel quantises activations to FP8 per 128-group and rounds to BF16)"}
    t = time.perf_counter()
    mpr = E.prove_samples(ms, server_nonce=mnonce, engine=engine)
    t_mpr = (time.perf_counter() - t) * 1e3
    mver = verify_reply_with_setup(setup, mpr.container, expected_c_pre=mnonce, circuit_digest=mpr.circuit.digest,
                                   rust_binary=args.fpn_verify, check_python=True)
    mrs = [E.rust_verify(args.fpn_verify, mpr.container, circuit.root) for _ in range(args.verify_runs)]

    def verify_moe_block(receipt, openings, c_pre_v, disc):
        """Verifier side: the block's consumed words come from the OPENED row; its nonce must bind them."""
        tv = TC.verify_trace_row_openings(payload_root, len(live["token_ids"]), live["sess"].layout.fingerprint(),
                                          openings, row_bytes=live["sess"].layout.stride_bytes,
                                          expected_positions={d.position for d in flat})
        if tv["verdict"] != "VERIFIED":
            return tv["verdict"]
        wo = lambda p, k, l: B.slot_words(live["sess"].layout, tv["rows"][p], k, l)  # noqa: E731
        nonce_v = moe_nonce(c_pre_v, moe_items(disc["position"], disc["layer"], tuple(disc["norm_channels"]),
                                               disc["in_col"], tuple(disc["slot_pair"]), wo))
        v = verify_reply_with_setup(setup, receipt, expected_c_pre=nonce_v, circuit_digest=circuit.digest,
                                    rust_binary=args.fpn_verify, check_python=True)
        return "VERIFIED" if v["verified"] else (v.get("error") or v.get("rust") or "REJECTED")

    mdisc = {"position": dmoe.position, "layer": dmoe.layer, "norm_channels": mmeta["norm"]["channels"],
             "in_col": mmeta["gemm_lane"]["in_col"], "slot_pair": mmeta["slot_pair"]}
    moe_iv = verify_moe_block(mpr.container, trace_openings, c_pre, mdisc)
    other_slot = next(s_ for s_ in range(TOPK) if s_ not in mdisc["slot_pair"])
    moe_forged_slots = verify_moe_block(mpr.container, trace_openings, c_pre,
                                        dict(mdisc, slot_pair=[mdisc["slot_pair"][0], other_slot]))
    res["relation4_moe_router_expert"] = {
        **mmeta, "opcode": "moe_router_expert", "prove_wall_ms": round(t_mpr, 1), "container_bytes": len(mpr.container),
        "circuit_root": mpr.circuit.root.hex(), "circuit_root_shared": mpr.circuit.root == circuit.root,
        "selector_moe_router_expert": mpr.assignment.values.get("slot.moe_router_expert"),
        "lanes_extracted": sorted(k for k in mpr.provenance.extracted if k.startswith("slot.")),
        "verify": mver, "rust_all_runs": [E.verdict_code(r[0]) for r in mrs],
        "rust_cli_wall_ms_median": round(statistics.median(r[1] for r in mrs), 2),
        "independent_verify_from_opened_row": moe_iv,
        "forgery_disclosed_other_top8_slot": moe_forged_slots,
        "sha256": hashlib.sha256(mpr.container).hexdigest()}
    print(json.dumps({k: res["relation4_moe_router_expert"][k] for k in
                      ("experts", "router_logits_fp32", "expert_outputs", "verify", "independent_verify_from_opened_row",
                       "forgery_disclosed_other_top8_slot", "top8_mixture_vs_captured_residual_delta")}, default=str),
          flush=True)

    # ---------------------------------------------------------------- independent verifier + forgeries
    layout = live["sess"].layout
    iv_args = (setup, c_pre, payload_root, live["token_ids"], live["out"], len(live["prompt"]), live["cached"], draws,
               args.tiles, NONCE, layout)
    iv = lambda bun, r5r, r4r=r4_receipts, op=opening_bytes: B.independent_verify(  # noqa: E731
        *iv_args, bun, op, r4r, r4_disc, r5r, args.fpn_verify, engine=engine, window_int_bits=WINDOW_BITS)
    t = time.perf_counter()
    honest = iv(trace_openings, r5)
    t_iv = (time.perf_counter() - t) * 1e3
    print(json.dumps({"independent_verifier_honest": honest["verdict"], "stages": honest["stages"]}, default=str),
          flush=True)
    forg = {}
    chkv = lambda op: WC.verify_drawn_tiles_against_setup(setup, op, r5, coords, engine=engine,  # noqa: E731
                                                          window_int_bits=WINDOW_BITS)["verdict"]
    # (a) tampered X element of a drawn row in the opening bundle
    d0 = gemm_draws[0]
    sl = layout.slot(moe_x_slot(d0), d0.layer)
    c0 = GEMM_TABLE[d0.gemm][2]
    xw0 = x_chunk(rows[d0.position][(moe_x_slot(d0), d0.layer)], d0)
    jbad = KTILE * c0 + int(np.abs(WC.bf16_values(xw0)).argmax())
    forged_rows = []
    for o in bundle.openings:
        row = o.row
        if o.position == d0.position:
            row = bytearray(row)
            row[2 * (sl.offset + jbad)] ^= 0x01
            row = bytes(row)
        forged_rows.append(TC.TraceRowOpening(o.position, row, o.path))
    forg["a_tampered_x_in_opened_row"] = iv(TC.TraceRowOpenings(bundle.n_tokens, bundle.layout_fingerprint,
                                                                bundle.row_bytes, tuple(forged_rows)).encode(),
                                            r5)["verdict"]
    # (b) tampered captured expert id in the opened row (claims another routed expert)
    de = next((d for d in gemm_draws if gemm_kind(d) == "expert"), None)
    if de is not None:
        sle = layout.slot("expert_ids", de.layer)
        k = de.tile_start // EXPERT_ROWS
        forged_rows = []
        for o in bundle.openings:
            row = o.row
            if o.position == de.position:
                row = bytearray(row)
                row[2 * (sle.offset + k)] ^= 0x01
                row = bytes(row)
            forged_rows.append(TC.TraceRowOpening(o.position, row, o.path))
        forg["b_tampered_expert_id_in_opened_row"] = iv(
            TC.TraceRowOpenings(bundle.n_tokens, bundle.layout_fingerprint, bundle.row_bytes,
                                tuple(forged_rows)).encode(), r5)["verdict"]
        # (c) genuine FP8 tile of an UNROUTED expert (same layer / rows / cols) substituted in the weight opening
        be = next(b for b, d in enumerate(gemm_draws) if d is de)
        routed = {int(v) for v in rows[de.position][("expert_ids", de.layer)]}
        unrouted = next(e for e in range(N_EXPERTS) if e not in routed)
        name, r0_, c0_ = coords[be]
        uname = name.replace(f".experts.{tiles_meta[be]['expert']}.", f".experts.{unrouted}.")
        usub = WC.open_drawn_tiles(wcom, ck, [(uname, r0_, c0_)]).tiles[0]
        tl = list(opening.tiles)
        tl[be] = usub
        forg["c_genuine_tile_of_unrouted_expert_true_coords"] = chkv(WC.TileOpening(opening.K, opening.manifest,
                                                                                    tuple(tl)))
        tl[be] = WC.OpenedTile(name, r0_, c0_, 0, None, usub.path, WC.KIND_FP8, usub.raw, usub.scales)
        forg["c2_unrouted_expert_tile_relabelled_as_routed"] = chkv(WC.TileOpening(opening.K, opening.manifest,
                                                                                   tuple(tl)))
        forg["c_unrouted_expert"] = unrouted
    # (d) one e4m3 byte / (e) one F32 block scale of an FP8 tile changed in the opening
    bf = next(b for b, t_ in enumerate(opening.tiles) if t_.kind == WC.KIND_FP8)
    t0_ = opening.tiles[bf]
    raw = t0_.raw.copy()
    raw[0, 0] ^= 0x01
    tl = list(opening.tiles)
    tl[bf] = WC.OpenedTile(t0_.name, t0_.r0, t0_.c0, 0, None, t0_.path, WC.KIND_FP8, raw, t0_.scales)
    forg["d_one_fp8_byte_changed_in_opening"] = chkv(WC.TileOpening(opening.K, opening.manifest, tuple(tl)))
    sc = t0_.scales.copy()
    sc[0] = np.float32(sc[0] * 2)
    tl[bf] = WC.OpenedTile(t0_.name, t0_.r0, t0_.c0, 0, None, t0_.path, WC.KIND_FP8, t0_.raw, sc)
    forg["e_one_block_scale_changed_in_opening"] = chkv(WC.TileOpening(opening.K, opening.manifest, tuple(tl)))
    # (f) the manifest's bound window width changed (22) -> root differs
    mm = bytearray(opening.manifest)
    mm[len(b"FPN2_WMANIFEST\x02") + 32 + 24] ^= 0x01
    forg["f_manifest_window_width_changed"] = chkv(WC.TileOpening(opening.K, bytes(mm), opening.tiles))
    # (g) genuine tile of another layer (attention projection) substituted
    ba = next((b for b, d in enumerate(gemm_draws) if gemm_kind(d) == "attention"), None)
    if ba is not None:
        name, r0_, c0_ = coords[ba]
        other_layer = (gemm_draws[ba].layer + 1) % N_LAYERS
        sub = WC.open_drawn_tiles(wcom, ck, [(name.replace(f"layers.{gemm_draws[ba].layer}.",
                                                           f"layers.{other_layer}."), r0_, c0_)]).tiles[0]
        tl = list(opening.tiles)
        tl[ba] = sub
        forg["g_genuine_attention_tile_of_another_layer"] = chkv(WC.TileOpening(opening.K, opening.manifest,
                                                                                tuple(tl)))
    # (h) tampered relation-5 receipt / relation-4 receipt, (i) no C_pre, (j) root rebuilt after the challenge
    bad5 = bytearray(r5)
    bad5[len(bad5) // 2] ^= 0x01
    forg["h_tampered_r5_receipt_byte"] = iv(trace_openings, bytes(bad5))["verdict"]
    forg["h_tampered_r5_receipt_rust_alone"] = E.verdict_code(E.rust_verify(args.fpn_verify, bytes(bad5), vroot)[0])
    bad4 = bytearray(r4_receipts[0])
    bad4[len(bad4) // 2] ^= 0x01
    forg["h_tampered_r4_receipt_byte"] = iv(trace_openings, r5, [bytes(bad4)] + r4_receipts[1:])["verdict"]
    badm = bytearray(mpr.container)
    badm[len(badm) // 2] ^= 0x01
    forg["h_tampered_moe_router_expert_receipt"] = verify_moe_block(bytes(badm), trace_openings, c_pre, mdisc)
    forg["i_reply_without_c_pre"] = B.independent_verify(setup, None, *iv_args[2:], trace_openings, opening_bytes,
                                                         r4_receipts, r4_disc, r5, args.fpn_verify, engine=engine,
                                                         window_int_bits=WINDOW_BITS)["verdict"]
    t_rows = []
    for o in bundle.openings:
        r = bytearray(tree.row(o.position))
        if o.position == d0.position:
            r[2 * (sl.offset + jbad)] ^= 0x01
        t_rows.append((o.position, bytes(r)))
    leaves = list(tree.levels[0][:tree.n_tokens])
    for pos, r in t_rows:
        leaves[pos] = TC.row_leaf(pos, r)
    lv = TC._build_levels(leaves)
    t_root = TC.payload_root(lv[-1][0], tree.n_tokens, tree.layout_fingerprint)
    t_tree = TC.TraceTree(t_root, tree.n_tokens, tree.layout_fingerprint, tree.row_bytes, lv,
                          tuple((pos, 1, r) for pos, r in t_rows))
    forg["j_root_rebuilt_over_tampered_payload_after_challenge"] = B.independent_verify(
        setup, c_pre, t_root, *iv_args[3:], TC.open_trace_rows(t_tree, [p for p, _ in t_rows]).encode(), opening_bytes,
        r4_receipts, r4_disc, r5, args.fpn_verify, engine=engine, window_int_bits=WINDOW_BITS)["verdict"]
    # (k) verifier applying the WRONG BF16 window (exact units) to router tiles, if any were drawn
    if any(m["kind"] == "router" for m in tiles_meta):
        forg["k_verifier_without_bf16_window_on_router_tiles"] = WC.verify_drawn_tiles_against_setup(
            setup, opening_bytes, r5, coords, engine=engine, window_int_bits=None)["verdict"]
    for k, v in forg.items():
        print(json.dumps({k: v}), flush=True)
    rejected = {k: v for k, v in forg.items() if isinstance(v, str)}

    # ---------------------------------------------------------------- receipt
    by_kind = {}
    for m in tiles_meta:
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
    dev_err = [m.get("exact_vs_device_gemm_rel_err") for m in tiles_meta]
    res["relation4"] = {"profile": FOLDED_AIR_V1["name"], "circuit_root": circuit.root.hex(), "blocks": r4,
                        "lanes": "RMSNorm pair: captured h_in + live BF16 input_layernorm gain; GEMM lane: captured "
                                 "x_mlp element x live BF16 router (mlp.gate) weight element, device BF16 product; "
                                 "dense_gqa slot: captured q / o channels (extent 1, slot output = value0 by "
                                 "construction: exercises the format, does not check attention)"}
    res["relation5"] = {
        "profile": FOLDED_VEC_N22_V1["name"], "tiles": lay.B, "tile_shape": [32, lay.K], "macs": lay.macs,
        "container_bytes": len(r5), "tiles_by_kind": by_kind,
        "fp8_tiles": sum(m["dtype"].startswith("F8") for m in tiles_meta),
        "distinct_routed_experts": sorted({(m["layer"], m["expert"]) for m in tiles_meta if m["expert"] is not None}),
        "tiles_by_gemm": {p: sum(m["tensor"].endswith(f"{p.split('.')[-1]}.weight") for m in tiles_meta)
                          for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "mlp.gate")},
        "tile_draws": tiles_meta,
        "integer_bound_bits_max": max(bits), "integer_bound_bits_per_tile": bits,
        "integer_faithful": max(bits) <= 62, "integer_faithful_tiles": f"{sum(x <= 62 for x in bits)}/{len(bits)}",
        "capped_exponent_window": {
            "window_int_bits": WINDOW_BITS, "max_shift_span": WINDOW_BITS - 7,
            "rule": "per tile and operand: shift = 23 - E with max|v| = f 2^E (f in [0.5, 1)); unit = "
                    "round-half-even(v 2^shift) -> |unit| <= 2^23 -> sum_j |W||X| <= 2^56. FP8 W: v = e4m3 x F32 block "
                    "scale (exact in fp64), window bound in the v2 manifest; BF16 router W: verifier window 23; X: "
                    "captured BF16 chunk (prover witness)",
            "quant_window_tail_rel_err_max": max(m["quant_window_tail_rel_err"] for m in tiles_meta),
            "quant_window_tail_rel_err_max_fp8": max((m["quant_window_tail_rel_err"] for m in tiles_meta
                                                      if m["dtype"].startswith("F8")), default=None),
            "w_window_tail_rel_max": max(m["w_window_tail_rel"] for m in tiles_meta),
            "x_window_tail_rel_max": max(m["x_window_tail_rel"] for m in tiles_meta),
            "w_elems_rounded_total": sum(m["w_elems_rounded"] for m in tiles_meta),
            "x_elems_rounded_total": sum(m["x_elems_rounded"] for m in tiles_meta),
            "definitions": "quant_window_tail_rel_err = max_i |Y_window_i - Y_exact_i| / max_i |Y_exact_i| over the "
                           "tile's 32 rows (exact rationals), Y_exact = exact product of the unwindowed operands "
                           "(dequantised FP8 x BF16 X); FP8 W has up to a 4-bit x 24-bit = 28-bit significand per "
                           "element, so the 23-bit window can round FP8 elements (counted in w_elems_rounded_total): "
                           "the tail is disclosed, not hidden"},
        "exact_vs_device_fp8_gemm_max_rel_err": max((e for e, m in zip(dev_err, tiles_meta)
                                                     if e is not None and m["dtype"].startswith("F8")), default=None),
        "exact_vs_device_fp8_gemm_max_rel_err_attention": max((e for e, m in zip(dev_err, tiles_meta)
                                                               if e is not None and m["kind"] == "attention"),
                                                              default=None),
        "exact_vs_device_fp8_gemm_max_rel_err_expert": max((e for e, m in zip(dev_err, tiles_meta)
                                                            if e is not None and m["kind"] == "expert"), default=None),
        "exact_vs_device_gemm_definition": "full-K served-path output rows (device) vs the fp64 product of the exact "
                                           "dequantised FP8 weight rows and the captured BF16 input row; the served "
                                           "W8A8 kernels quantise X to FP8 per 128-group, which dominates this number",
        "live_fp8_bytes_equal_committed": f"{sum(bool(m.get('live_fp8_bytes_equal_committed')) for m in tiles_meta)}/"
                                          f"{sum(m['dtype'].startswith('F8') for m in tiles_meta)}",
        "live_scales_equal_committed": f"{sum(bool(m.get('live_scales_equal_committed')) for m in tiles_meta)}/"
                                       f"{sum(m['dtype'].startswith('F8') for m in tiles_meta)}",
        "tile_build_ms": round(t_tiles, 1), "window_tail_exact_ms": round(t_tail, 1),
        "device_and_live_check_ms": round(t_dev, 1), "open_drawn_tiles_ms": round(t_open, 1),
        "prove_runs": r5_runs, "weight_tiles_vs_setup": chk, "setup_only_check_ms": round(t_chk, 1),
        "weight_opening_bytes": len(opening_bytes),
        "verify": {"rust": [E.verdict_code(r[0]) for r in rs5],
                   "cli_wall_ms_median": round(statistics.median(r[1] for r in rs5), 2),
                   "python_detached": E.verdict_code(py5[0])}}
    res["independent_verifier"] = {"honest": honest, "verify_ms": round(t_iv, 1), "forgeries": forg,
                                   "forgeries_rejected": sum(v not in ("VERIFIED",) for k, v in rejected.items()),
                                   "forgeries_total": len(rejected)}
    res["scope_limits"] = [
        "relation 5 proves Y = W_win X_win exactly for the WINDOWED integer operands (23-bit capped exponent window): "
        "FP8 W = e4m3 x F32 block scale can carry up to 28 significant bits per element; elements whose exact value "
        "is finer than the tile's 2^-shift grid are rounded (half-step <= 2^-23 x max|W|; the measured count is "
        "w_elems_rounded_total); quant_window_tail_rel_err reports the exact deviation of the proved Y from the exact "
        "product of the unwindowed operands per tile",
        "the served W8A8 kernels quantise the activation to FP8 per 128-group (activation_scheme: dynamic); relation 5 "
        "uses the captured BF16 activation, so it attests the BF16-activation x dequantised-FP8-weight product, NOT the "
        "device's FP8-activation product: exact_vs_device_fp8_gemm_max_rel_err measures that gap outside the proof",
        "expert down_proj is committed and openable but NOT in the draw space: its input (the expert activation) is "
        "not captured by FULL_NO_KV with capture_act=False (the fused MoE kernel exposes no boundary)",
        "routed-expert draws use the captured top-8 ids (authenticated through trace_payload_root); no relation "
        "proves that they are the top-8 of the router logits (capture_integrity reports an fp64 recompute outside "
        "the protocol)",
        "moe_router_expert block: the slot's router logits are fp32 recomputes from the captured x_mlp and the live "
        "BF16 router, and its expert outputs are fp64 recomputes from the dequantised FP8 checkpoint experts at one "
        "channel -- NOT values extracted from the served fused-MoE kernel (which exposes no per-expert output); the "
        "shared slot mixes 2 of the 8 routed experts (experts = top_k = 2), at 1/256 fixed point; its lanes are "
        "disclosed, the verifier re-derives only the consumed captured words (h_mid pair, x_mlp row, expert ids) from "
        "the opened row",
        "relation 4 keeps the dense Qwen3-0.6B single-block circuit (same circuit_root); its weight-side lanes are not "
        "opened against the weight root",
        "trace-row openings disclose every drawn row in plaintext (bytes recorded); relation 5's hidden witness X is "
        "bound to the opened rows only through N_s (forgery (c) of applies unchanged)",
        "the 48 routers are committed as their live served BF16 matrices (48/48 = RNE of the F32 checkpoint tensors), "
        "so the F32 router bytes themselves are not in the root",
    ]
    res["host"] = {"peak_rss_mib_ru_maxrss": round(B.rss_mib(), 1), "rss_mib_now": round(rss_now_mib(), 1),
                   "total_bench_s": round(time.perf_counter() - t_start, 1)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, default=str)
    print("WROTE", args.out, flush=True)
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
