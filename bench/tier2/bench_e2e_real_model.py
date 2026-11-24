#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
""": real-model end-to-end relation-4 receipt (end-to-end cost matrix) on one GPU host.

: every relation-4 container is proven by the GPU-accelerated prover (`folded_air.prove(engine=CudaEngine)`,
byte-identical to the CPU reference for the same rng stream -- re-checked here on real witnesses), for FIVE real
checkpoints (adds Qwen3-30B-A3B-FP8 `moe_router_expert` and gpt-oss-20b `swa_sink` + `moe_router_expert`) under ONE
shared circuit root, plus ONE production-scale `fpn2-folded-air-n20-v1` container (K = 32,768) proving 64 real
Qwen3-0.6B sampled blocks.

    source <WORK_DIR>
    python bench/tier2/bench_e2e_real_model.py --models-dir <WORK_DIR> \\
        --fpn-verify core/target/release/fpn-verify --out benchmarks/prover/results/e2e_real_models_rtx6000d.json

For each of Qwen3-0.6B (dense GQA + QK-norm), Llama-3.2-1B-Instruct (dense GQA, no QK-norm) and Qwen3.5-0.8B
(hybrid Gated DeltaNet + gated GQA) it runs a REAL bf16 forward pass on the GPU, hooks real activations,
reads real checkpoint weights, maps them through `prover/fpn_prover/circuit/e2e_prover.py` into the shared
fixed-shape Program (width-2 RMSNorm, 1 x 1 GEMM tile, 5-opcode microprogram slot), lowers it to ONE relation-4
`fpn2-folded-air-v1` container (n_vars = 14) and verifies the bytes with the Rust `fpn-verify` built on this
host and the detached Python `fpn_verify_ref`, both with `--circuit-root` pinned.

Scope (also written into the JSON): the prover is the CPU Python reference `folded_air.prove`; the GPU runs the
model forward pass (so peak VRAM is forward-pass VRAM) and the v1 kernel timings.  The container proves ~214
Plonkish rows of ONE sampled block set, not a full 100-draw request (that projection is the IR census below).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "verifier", "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prover.fpn_prover.backend.fpn2 import FOLDED_AIR_V1  # noqa: E402
from prover.fpn_prover.circuit import e2e_prover as E  # noqa: E402
from prover.fpn_prover.circuit import lower as LW  # noqa: E402
from prover.fpn_prover.circuit.census import Profile, census  # noqa: E402

PROMPT = ("FlashProve-Next proves that a served transformer response was computed by the committed model. "
          "The verifier checks one fixed-size container.")
MODELS = (
    dict(name="Qwen3-0.6B", arch="dense_gqa_qknorm", opcodes=("dense_gqa",), layer=1),
    dict(name="Llama-3.2-1B-Instruct", arch="dense_gqa", opcodes=("dense_gqa",), layer=1),
    dict(name="Qwen3.5-0.8B", arch="gdn_hybrid", opcodes=("gdn_lrec",), layer=0),
    dict(name="Qwen3-30B-A3B-FP8", arch="moe_fp8_128x8_qknorm", opcodes=("moe_router_expert",), layer=1),
    dict(name="gpt-oss-20b", arch="swa_sink_moe_mxfp4_32x4", opcodes=("swa_sink", "moe_router_expert"), layer=0),
)
N20_BLOCKS = 64                     # real Qwen3-0.6B sampled blocks packed into ONE n = 20 container
COST_MODELS = ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B")
COST_T = (256, 2048, 8192)
SEED = 6


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # Linux: KiB


def word(t):
    """BF16 word of a 0-d bf16 tensor."""
    import torch
    return int(t.detach().to(torch.bfloat16).reshape(1).view(torch.int16).item()) & 0xffff


def git_rev():
    if os.environ.get("FPN_SOURCE_COMMIT"):          # archive shipped without .git (git archive | ssh tar -x)
        return os.environ["FPN_SOURCE_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


# ----------------------------------------------------------------------------- model + hooks
def load(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, model


def decoder_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("layers") and hasattr(mod, "__len__") and len(mod) and hasattr(mod[0], "input_layernorm"):
            if "visual" not in name:
                return name, mod
    raise RuntimeError("no decoder layers found")


def capture(model, layer, kind, moe=False):
    """Register hooks on one decoder layer; returns (store, handles)."""
    store = {}
    hs = []

    def keep(key, inputs=False):
        def hook(mod, args, out):
            store[key] = (args[0] if inputs else (out[0] if isinstance(out, tuple) else out)).detach()
            if inputs:
                store[key + ".out"] = out.detach() if not isinstance(out, tuple) else out[0].detach()
            if isinstance(out, tuple) and len(out) > 1 and hasattr(out[1], "detach"):
                store[key + ".out1"] = out[1].detach()
        return hook

    if moe:
        hs.append(layer.mlp.register_forward_hook(keep("mlp_in", inputs=True)))
        if hasattr(layer.mlp, "gate"):                   # Qwen3-MoE: gate returns (logits, top-k probs, indices)
            hs.append(layer.mlp.gate.register_forward_hook(keep("router")))

    hs.append(layer.input_layernorm.register_forward_hook(keep("norm_in", inputs=True)))
    if kind == "gdn":
        la = layer.linear_attn
        hs.append(la.in_proj_qkv.register_forward_hook(keep("qkv_in", inputs=True)))
        hs.append(la.in_proj_b.register_forward_hook(keep("b")))
        hs.append(la.in_proj_a.register_forward_hook(keep("a")))
    else:
        at = layer.self_attn
        hs.append(at.q_proj.register_forward_hook(keep("q_in", inputs=True)))
        hs.append(at.k_proj.register_forward_hook(keep("k")))
        hs.append(at.v_proj.register_forward_hook(keep("v")))
        if hasattr(at, "q_norm"):
            hs.append(at.q_norm.register_forward_hook(keep("qn")))
            hs.append(at.k_norm.register_forward_hook(keep("kn")))
    return store, hs


def effective_gain(norm):
    """Qwen3.5 RMSNorm computes x_hat * (1 + w) in fp32; Qwen3 / Llama x_hat * w."""
    import torch
    w = norm.weight.detach().float()
    if type(norm).__name__.startswith("Qwen3_5"):
        return w + 1.0, "1 + weight (fp32, Qwen3.5 zero-centred gain) rounded to bf16"
    return w.to(torch.bfloat16).float(), "weight (bf16 checkpoint value)"


def expert_outputs(spec, layer_mod, x, experts, models_dir):
    """Real per-expert FFN outputs e_i(x) (hidden,) for the given expert ids, on the GPU.

    Qwen3-30B-A3B-FP8: the model's own `FP8Experts` module (block-scaled FP8 kernels) called with a one-hot routing
    of weight 1.  gpt-oss-20b: the layer's MXFP4 expert blocks dequantised by transformers'
    `convert_moe_packed_tensors` and the reference `GptOssExperts` arithmetic (clamped SwiGLU, alpha / limit, biases)
    -- the served path is a Triton MXFP4 kernel, so `moe_check` compares the top-k mixture against the hooked output."""
    import torch
    if hasattr(layer_mod.mlp, "gate"):
        ex = layer_mod.mlp.experts
        out = {}
        for e in experts:
            idx = torch.tensor([[e]], device=x.device)
            wt = torch.ones((1, 1), dtype=x.dtype, device=x.device)
            out[e] = ex(x.reshape(1, -1), idx, wt).reshape(-1).float()
        return out, "FP8Experts.forward (model's own block-scaled FP8 expert kernels), one-hot routing weight 1"
    import json as _json
    from safetensors import safe_open
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    root = os.path.join(models_dir, spec["name"])
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        wmap = _json.load(f)["weight_map"]
    pre = f"model.layers.{spec['layer']}.mlp.experts."
    t = {}
    for k in ("gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias", "down_proj_blocks",
              "down_proj_scales", "down_proj_bias"):
        with safe_open(os.path.join(root, wmap[pre + k]), framework="pt", device="cuda") as f:
            t[k] = f.get_tensor(pre + k)
    cfg = layer_mod.mlp.experts
    alpha, limit = float(getattr(cfg, "alpha", 1.702)), float(getattr(cfg, "limit", 7.0))
    out = {}
    for e in experts:
        gu = convert_moe_packed_tensors(t["gate_up_proj_blocks"][e:e + 1], t["gate_up_proj_scales"][e:e + 1])[0]
        dn = convert_moe_packed_tensors(t["down_proj_blocks"][e:e + 1], t["down_proj_scales"][e:e + 1])[0]
        h = x.reshape(1, -1).to(torch.bfloat16) @ gu + t["gate_up_proj_bias"][e]
        gate, up = h[..., ::2], h[..., 1::2]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        y = ((up + 1) * glu).to(torch.bfloat16) @ dn + t["down_proj_bias"][e]
        out[e] = y.reshape(-1).float()
    return out, ("MXFP4 blocks dequantised with transformers.integrations.mxfp4.convert_moe_packed_tensors + reference "
                 f"GptOssExperts arithmetic (alpha = {alpha}, limit = {limit}, biases)")


def extract(spec, model, layer_mod, store, cfg, rnd, t=None, models_dir=None):
    """Plain-number Samples + extraction metadata from the hooked tensors (token `t`, default the last one)."""
    import torch
    t_last = store["norm_in"].shape[1] - 1
    t = t_last if t is None else t
    opcode = spec["opcode"]
    x = store["norm_in"][0, t]                           # real RMSNorm input (bf16, hidden)
    gain, gain_note = effective_gain(layer_mod.input_layernorm)
    eps = Fraction(str(getattr(cfg, "rms_norm_eps", 1e-6)))
    hidden = x.shape[0]
    order = list(range(hidden))
    rnd.shuffle(order)
    meta = {"token_index": t, "seq_len": t_last + 1, "layer": spec["layer"], "gain": gain_note,
            "epsilon": str(eps)}
    # --- RMSNorm (width 2): two seeded channels; skip pairs failing the reference predicate (floor etc.).
    skipped = 0
    norm_pair = None
    for i in range(0, hidden - 1, 2):
        c0, c1 = order[i], order[i + 1]
        cand = (word(x[c0]), word(x[c1])), (E.bf16_word(gain[c0].item()), E.bf16_word(gain[c1].item()))
        probe = E.Samples(spec["arch"], spec["name"], cand[0], cand[1], 0x3f80, 0x3f80, 0x3f80, "dense_gqa",
                          dict(normalize=0, query=0, key0=0, key1=0, value0=0, value1=0), epsilon=eps)
        try:
            prog, names = E.build_program(eps)
            E.build_witness(probe, prog, names)
            norm_pair = (c0, c1), cand
            break
        except LW.LoweringError:
            skipped += 1
    if norm_pair is None:
        raise RuntimeError("no RMSNorm channel pair satisfies the reference predicate")
    (c0, c1), (n_in, n_gain) = norm_pair
    real_out = store["norm_in.out"][0, t]
    meta["norm"] = {"channels": [c0, c1], "pairs_skipped": skipped,
                    "model_norm_output_real": [real_out[c0].item(), real_out[c1].item()],
                    "note": "block output = width-2 renormalisation (recomputed), NOT the model's hidden-wide output"}
    # --- GEMM 1 x 1: real activation feeding a real bf16 projection, real weight element, GPU bf16 product.
    if opcode == "gdn_lrec":
        proj, xin, pname = layer_mod.linear_attn.in_proj_qkv.weight, store["qkv_in"], "linear_attn.in_proj_qkv"
    elif opcode == "moe_router_expert":
        router = layer_mod.mlp.gate if hasattr(layer_mod.mlp, "gate") else layer_mod.mlp.router
        proj, xin = router.weight, store["mlp_in"]
        pname = "mlp.gate (router, bf16)" if hasattr(layer_mod.mlp, "gate") else "mlp.router (bf16)"
    else:
        proj, xin, pname = layer_mod.self_attn.q_proj.weight, store["q_in"], "self_attn.q_proj"
    W = proj.detach()
    if W.dtype != torch.bfloat16:
        raise RuntimeError(f"{pname} weight is {W.dtype}, not an exact bf16 checkpoint value")
    i_in, o = rnd.randrange(W.shape[1]), rnd.randrange(W.shape[0])
    for _ in range(64):                                  # need a nonzero activation and weight
        if word(xin[0, t, i_in]) & 0x7fff and word(W[o, i_in]) & 0x7fff:
            break
        i_in, o = rnd.randrange(W.shape[1]), rnd.randrange(W.shape[0])
    xa, wa = xin[0, t, i_in].reshape(1), W[o, i_in].reshape(1)
    prod = torch.mul(xa, wa)                             # device bf16 multiply (CUDA kernel)
    meta["gemm"] = {"projection": pname, "out_row": o, "in_col": i_in,
                    "activation": xa.item(), "weight": wa.item(), "device_product": prod.item(),
                    "note": "1x1 tile: observed = GPU bf16 product of the real pair, NOT a full dot product"}
    # --- microprogram slot (real opcode)
    if opcode in ("dense_gqa", "swa_sink"):
        hd = cfg.head_dim if getattr(cfg, "head_dim", None) else cfg.hidden_size // cfg.num_attention_heads
        qknorm = "qn" in store
        # Selection rule (disclosed): head 0, the head dim with the largest |v| at token 0, so the fixed-point
        # 1/256 lanes are not trivially zero for small real values.
        j = int(store["v"][0, 0, :hd].float().abs().argmax())
        q = (store["qn"][0, t, 0, j] if qknorm else store["q_in.out"][0, t, j])
        k = store["kn"][0, :2, 0, j] if qknorm else store["k"][0, :2, j]
        v = store["v"][0, :2, j]
        slot = dict(query=q.item(), key0=k[0].item(), key1=k[1].item(), value0=v[0].item(), value1=v[1].item())
        meta["slot"] = {"opcode": opcode, "head": 0, "dim": j,
                        "source": "q_norm/k_norm outputs (pre-RoPE)" if qknorm else "q_proj/k_proj outputs (pre-RoPE)",
                        "tokens": [t, 0, 1],
                        "selection": "head 0, dim = argmax_j |v[token 0, j]|", "note": "extent = 1: output == v0 by construction"}
        if opcode == "dense_gqa":
            slot["normalize"] = int(qknorm)
        else:
            lt = getattr(cfg, "layer_types", None)
            meta["slot"].update({
                "layer_type": lt[spec["layer"]] if lt else None, "sliding_window": getattr(cfg, "sliding_window", None),
                "learned_sink_logit_head0": float(layer_mod.self_attn.sinks.detach().float()[0])
                if hasattr(layer_mod.self_attn, "sinks") else None,
                "note": "slot window = 1, sink = 1, extent = 1, position 0: the SWA+sink mask keeps token 0, output == "
                        "v0 by construction; gpt-oss's LEARNED per-head sink logit (s_aux) is NOT modelled by the slot"})
    elif opcode == "moe_router_expert":
        k_model = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "experts_per_token")
        n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
        logits_all = store["router"] if "router" in store else store["mlp_in.out1"]
        logits_all = logits_all.reshape(-1, n_exp)
        attempts = []
        for tt in range(t, -1, -1):                      # disclosed retry: earlier tokens if no exact exp cell
            lg = logits_all[tt].float()
            top = torch.topk(lg, k_model)
            ids = [int(i) for i in top.indices]
            xm = store["mlp_in"][0, tt]
            outs, how = expert_outputs(spec, layer_mod, xm, ids, models_dir)
            j = int(outs[ids[0]].abs().argmax())
            sc = (lg[ids[0]].item(), lg[ids[1]].item())
            oc = (outs[ids[0]][j].item(), outs[ids[1]][j].item())
            try:
                E.moe_lanes(sc, oc)
            except ValueError as err:
                attempts.append({"token": tt, "rejected": str(err)})
                continue
            # consistency of the per-expert outputs with the model's own routed MoE output at this token
            if "router.out1" in store:
                wts = store["router.out1"].reshape(-1, k_model)[tt].float()
            else:
                wts = torch.softmax(top.values, dim=-1)
            mix = sum(w * outs[e] for w, e in zip(wts, ids))
            real = store["mlp_in.out"].reshape(-1, mix.shape[0])[tt].float()
            rel = float((mix - real).norm() / real.norm())
            slot = dict(scores=sc, outputs=oc)
            meta["slot"] = {
                "opcode": opcode, "token": tt, "retry_attempts": attempts, "experts_total": n_exp,
                "experts_per_token_model": k_model, "model_top_k_expert_ids": ids, "selected_expert_ids": ids[:2],
                "router_logits_selected": list(sc), "router_logits_topk": [float(v) for v in top.values],
                "output_channel": j, "expert_outputs_selected_channel": list(oc), "expert_output_source": how,
                "selection": "the model's two highest router logits at the token; channel = argmax_j |e_top1(x)_j|",
                "moe_check_rel_l2_topk_mixture_vs_hooked_mlp_output": rel,
                "note": f"the slot has experts = top_k = 2: it proves the softmax-renormalised 2-expert SUB-mixture of "
                        f"the model's top-{k_model} of {n_exp} experts (top-k selection vacuous), not the full routing"}
            break
        else:
            raise RuntimeError(f"no token with an exact router exp cell: {attempts}")
    else:
        la = layer_mod.linear_attn
        nk, nv = la.num_k_heads, la.num_v_heads
        dk, dv, kd = la.head_k_dim, la.head_v_dim, la.key_dim
        h = rnd.randrange(nv)
        hk = h // (nv // nk)
        pre = store["qkv_in.out"][0, 0].float()          # real in_proj_qkv output at token 0 (pre-conv)
        wconv = la.conv1d.weight.detach().float()[:, 0, -1]          # tap applied to the current token
        post = torch.nn.functional.silu(wconv * pre)     # token 0: earlier taps see zero padding, no bias
        kvec = post[kd + hk * dk: kd + (hk + 1) * dk]
        kvec = kvec / torch.sqrt((kvec * kvec).sum() + 1e-6)          # use_qk_l2norm_in_kernel
        # Selection rule (disclosed): seeded value head h; key dim i = argmax |k_hat|, value dim j = argmax |v|.
        i = int(kvec.abs().argmax())
        j = int(post[2 * kd + h * dv: 2 * kd + (h + 1) * dv].abs().argmax())
        cv = 2 * kd + h * dv + j
        b = store["b"][0, 0, h].float()
        a = store["a"][0, 0, h].float()
        g = -la.A_log.detach().float()[h].exp() * torch.nn.functional.softplus(a + la.dt_bias.detach().float()[h])
        slot = dict(decay=g.exp().item(), beta=torch.sigmoid(b).item(), key=kvec[i].item(),
                    kernel=wconv[cv].item(), history=pre[cv].item(), value_post=post[cv].item(), previous=0.0)
        meta["slot"] = {"opcode": "gdn_lrec", "value_head": h, "key_dim": i, "value_dim": j, "token": 0,
                        "selection": "seeded value head; key dim = argmax |k_hat|, value dim = argmax |v_post|",
                        "real_state_element_beta_k_v": slot["beta"] * slot["key"] * slot["value_post"],
                        "source": "in_proj_qkv output + conv1d weight + in_proj_a/b + A_log/dt_bias (layer "
                                  f"{spec['layer']}, linear_attention)",
                        "note": "position 0 (extent = 1): previous state 0, state lane = round(beta k v) at 1/256"}
    s = E.Samples(spec["arch"], spec["name"], n_in, n_gain, word(xa), word(wa), word(prod), opcode, slot,
                  epsilon=eps, meta=meta)
    return s


def verify_all(binary, data, root, runs=5):
    rs = [E.rust_verify(binary, data, root) for _ in range(runs)]
    py = E.python_verify_detached(data, root)
    return {"rust": {"verdict": rs[0][0], "all_runs": [r[0] for r in rs],
                     "cli_wall_ms_median": statistics.median(r[1] for r in rs),
                     "cli_wall_ms_runs": [round(r[1], 3) for r in rs],
                     "child_cpu_ms_median": statistics.median(r[2] for r in rs)},
            "python_detached": {"verdict": py[0], "wall_ms": py[1], "child_cpu_ms": py[2]}}


class VramSampler:
    """Peak device memory in use (cudaMemGetInfo, every 2 ms) above a baseline -- sees CuPy AND torch."""

    def __init__(self):
        import threading
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
        pool = self.cp.get_default_memory_pool()
        return {"device_used_peak_mib": self.peak / 2**20, "device_used_baseline_mib": self.base / 2**20,
                "prover_peak_above_baseline_mib": (self.peak - self.base) / 2**20,
                "cupy_pool_total_mib_after": pool.total_bytes() / 2**20}


def prove_gpu(fn, *a, **kw):
    """(proof, VRAM report) of one engine-accelerated prove."""
    with VramSampler() as vs:
        pr = fn(*a, **kw)
    return pr, vs.report()


def parity_check(samples, engine, seed):
    """CPU reference vs CudaEngine, same SeededRng stream, on this REAL witness: identical bytes?"""
    sys.path.insert(0, os.path.join(ROOT, "tests", "unit"))
    from test_ir_lowering_and_ntt import SeededRng       # noqa: PLC0415
    t0 = time.perf_counter()
    cpu = E.prove_samples(samples, rng=SeededRng(seed))
    t1 = time.perf_counter()
    gpu = E.prove_samples(samples, rng=SeededRng(seed), engine=engine)
    t2 = time.perf_counter()
    return {"seed": seed, "identical": cpu.container == gpu.container, "sha256_cpu": E.sha256(cpu.container),
            "sha256_cuda": E.sha256(gpu.container), "cpu_folded_air_prove_ms": cpu.timings_ms["folded_air_prove"],
            "cuda_folded_air_prove_ms": gpu.timings_ms["folded_air_prove"],
            "cuda_phases_ms": gpu.timings_ms.get("folded_air_prove_phases"), "wall_s": [t1 - t0, t2 - t1]}


def load_model(spec, args):
    import torch
    path = os.path.join(args.models_dir, spec["name"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok, model = load(path)
    torch.cuda.synchronize()
    return tok, model, time.perf_counter() - t0


def quant_note(model):
    q = getattr(model.config, "quantization_config", None)
    if q is None:
        return "none (bf16 checkpoint)"
    d = q if isinstance(q, dict) else q.to_dict()
    return (f"{d.get('quant_method')}: loaded natively quantised by transformers (NOT dequantised at load); "
            f"attention/router/norm weights used for the gemm tile are bf16 checkpoint tensors")


def run_model(spec, args, results, engine):
    import torch
    tok, model, t_load = load_model(spec, args)
    cfg = getattr(model.config, "text_config", None) or model.config
    _, layers = decoder_layers(model)
    layer_mod = layers[spec["layer"]]
    gdn = "gdn_lrec" in spec["opcodes"]
    if gdn and getattr(layer_mod, "layer_type", cfg.layer_types[spec["layer"]]) != "linear_attention":
        raise RuntimeError("selected Qwen3.5 layer is not a Gated DeltaNet layer")
    store, hs = capture(model, layer_mod, "gdn" if gdn else "dense", moe="moe_router_expert" in spec["opcodes"])
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids)                       # hooked pass (cold: first kernels / cuBLAS handles)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t1
    for h in hs:
        h.remove()
    t1 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids)                             # warm repeat, timing only (hooks removed)
    torch.cuda.synchronize()
    t_fwd_warm = time.perf_counter() - t1
    next_tok = int(out.logits[0, -1].argmax())
    vram_fwd = torch.cuda.max_memory_allocated() / 2**20
    pairs = []
    for opcode in spec["opcodes"]:
        sp = dict(spec, opcode=opcode)
        t2 = time.perf_counter()
        samples = extract(sp, model, layer_mod, store, cfg, random.Random(f"{SEED}:{spec['name']}:{opcode}"),
                          models_dir=args.models_dir)
        torch.cuda.synchronize()
        t_ext = time.perf_counter() - t2
        record = spec["name"] == "Qwen3-0.6B"
        proof, vram = prove_gpu(E.prove_samples, samples, record=record, engine=engine)
        ver = verify_all(args.fpn_verify, proof.container, proof.circuit.root)
        parity = parity_check(samples, engine, seed=7000 + len(results["models"]))
        summ = proof.circuit.summary()
        entry = {
            "model": spec["name"], "arch": spec["arch"], "opcode": opcode,
            "model_class": type(model).__name__, "quantization": quant_note(model),
            "prompt_tokens": int(ids.shape[1]),
            "next_token_argmax": next_tok, "next_token_text": tok.decode([next_tok]),
            "samples_meta": samples.meta,
            "witness_provenance": proof.provenance.as_dict(),
            "container": {"profile": FOLDED_AIR_V1["name"], "n_vars": proof.circuit.layout.n,
                          "bytes": len(proof.container), "sha256": E.sha256(proof.container),
                          "circuit_root": proof.circuit.root.hex(), "rows": summ["rows"],
                          "capacity_K": summ["capacity"], "variables": summ["variables"],
                          "range_checked": summ["range_checked"], "zk_rows": summ["zk_rows"],
                          "ir_gates": summ["ir_gates"],
                          "blocks_proven": ["rmsnorm(width=2)", "gemm_tile_row(1x1)", f"microprogram_slot[{opcode}]"],
                          "binding_ok": LW.check_binding_air(proof.circuit, proof.container)},
            "timing_ms": {"model_load_excluded": t_load * 1e3, "forward": t_fwd * 1e3,
                          "forward_warm_repeat_info": t_fwd_warm * 1e3, "witness_extraction": t_ext * 1e3,
                          **proof.timings_ms,
                          "prover_total_forward_to_container": t_fwd * 1e3 + t_ext * 1e3 + proof.timings_ms[
                              "witness_build"] + proof.timings_ms["lower_air"] + proof.timings_ms["folded_air_prove"]},
            "timing_notes": "forward = the hooked pass actually used; lower_air includes the public circuit "
                            "commitment on the engine (cached in-process after the first container); folded_air_prove "
                            "= CudaEngine-accelerated prover (fresh SecretRng), phases in folded_air_prove_phases",
            "prover_vram": vram,
            "parity_cpu_vs_cuda_seeded": parity,
            "verify": ver,
            "peak_gpu_vram_mib_forward": vram_fwd,
            "host_peak_rss_mib_so_far": rss_mib(),
        }
        results["models"].append(entry)
        print(json.dumps({k: entry[k] for k in ("model", "opcode", "timing_ms", "verify")}, indent=1), flush=True)
        pairs.append((samples, proof))
    del model, out, store
    gc.collect()
    torch.cuda.empty_cache()
    return pairs


def run_multiblock(args, results, engine, blocks=N20_BLOCKS):
    """ONE fpn2-folded-air-n20-v1 container (K = 32,768) proving `blocks` real Qwen3-0.6B sampled blocks."""
    import torch
    from prover.fpn_prover.backend.fpn2 import FOLDED_AIR_N20_V1
    spec = dict(MODELS[0], opcode="dense_gqa")
    tok, model, t_load = load_model(spec, args)
    cfg = model.config
    _, layers = decoder_layers(model)
    caps = [capture(model, lay, "dense") for lay in layers]
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t1
    for _, hs in caps:
        for h in hs:
            h.remove()
    t_last = ids.shape[1] - 1
    t2 = time.perf_counter()
    samples, where = [], []
    for i in range(blocks):
        layer, tok_i = i % len(layers), t_last - i // len(layers)
        sp = dict(spec, layer=layer)
        samples.append(extract(sp, model, layers[layer], caps[layer][0], cfg,
                               random.Random(f"{SEED}:mb:{i}"), t=tok_i))
        where.append([layer, tok_i])
    t_ext = time.perf_counter() - t2
    del model, caps
    gc.collect()
    torch.cuda.empty_cache()
    runs = []
    for rep in range(2):                                 # rep 0 pays the engine circuit commitment (lower_air)
        proof, vram = prove_gpu(E.prove_multi, samples, profile=FOLDED_AIR_N20_V1, engine=engine)
        runs.append({"rep": rep, "timing_ms": proof.timings_ms, "prover_vram": vram,
                     "sha256": E.sha256(proof.container)})
        print(json.dumps(runs[-1], indent=1, default=str), flush=True)
    ver = verify_all(args.fpn_verify, proof.container, proof.circuit.root)
    summ = proof.circuit.summary()
    results["n20_multiblock"] = {
        "model": spec["name"], "blocks": blocks, "block_layer_token": where,
        "profile": FOLDED_AIR_N20_V1["name"], "n_vars": proof.circuit.layout.n, "c_p": FOLDED_AIR_N20_V1["c_p"],
        "container_bytes": len(proof.container), "body_used": proof.circuit.layout.body_used,
        "sha256": E.sha256(proof.container), "circuit_root": proof.circuit.root.hex(), "rows": summ["rows"],
        "capacity_K": summ["capacity"], "variables": summ["variables"], "range_checked": summ["range_checked"],
        "ir_gates": summ["ir_gates"], "binding_ok": LW.check_binding_air(proof.circuit, proof.container),
        "timing_ms": {"model_load_excluded": t_load * 1e3, "forward_all_layers_hooked": t_fwd * 1e3,
                      "witness_extraction_64_blocks": t_ext * 1e3},
        "prove_runs": runs, "verify": ver,
        "notes": ["block i = layer i mod 28 at token t_last - i // 28 (seeded channels / gemm element per block)",
                  "rep 0 includes the one-time engine commitment of the public n = 20 circuit inside lower_air; rep 1 "
                  "hits the in-process circuit cache; folded_air_prove excludes lower_air in both",
                  "the n = 20 circuit (64 blocks) has its OWN circuit_root, distinct from the n = 14 shared root"]}
    return samples, proof


def n20_cpu_parity(samples, engine, seed=7777):
    """CPU reference vs CudaEngine at n = 20 (same SeededRng stream) -- slow (pure-Python CPU prover)."""
    sys.path.insert(0, os.path.join(ROOT, "tests", "unit"))
    from test_ir_lowering_and_ntt import SeededRng       # noqa: PLC0415
    from prover.fpn_prover.backend.fpn2 import FOLDED_AIR_N20_V1
    gpu = E.prove_multi(samples, profile=FOLDED_AIR_N20_V1, rng=SeededRng(seed), engine=engine)
    t = time.perf_counter()
    cpu = E.prove_multi(samples, profile=FOLDED_AIR_N20_V1, rng=SeededRng(seed))
    return {"seed": seed, "identical": cpu.container == gpu.container, "sha256_cpu": E.sha256(cpu.container),
            "sha256_cuda": E.sha256(gpu.container), "cpu_timing_ms": cpu.timings_ms,
            "cuda_timing_ms": gpu.timings_ms, "cpu_wall_s": time.perf_counter() - t}


# ----------------------------------------------------------------------------- v1 ZK + cheats
def zk_and_cheats(samples, proof, args):
    root = proof.circuit.root
    out = {"model": samples.model}
    a = E.prove_samples(samples, record=True)          # fresh SecretRng randomness
    b = E.prove_samples(samples, record=True)
    for pr in (a, b):
        E.check_public_c_openings(pr.circuit, pr.container, pr.record["positions"])
    out["reproof"] = {"sections": [{"name": n, "start": s, "end": e, "public": p}
                                   for n, s, e, p in E.section_map(a.circuit.layout)
                                   if not n.startswith("q") or n.startswith("q0.")],
                      "section_note": "q0.* shown; queries q1..q71 have the same per-query layout",
                      "public_c_openings_checked": True,
                      **E.shared_blocks(a.container, b.container, a.circuit.layout),
                      "both_verify": [verify_all(args.fpn_verify, x.container, root, runs=1)["rust"]["verdict"]
                                      for x in (a, b)]}
    t = time.perf_counter()
    rep = E.joint_rank(proof)
    rep["seconds"] = time.perf_counter() - t
    out["joint_rank_real_qwen3"] = rep
    cheats = {}
    data, info = E.cheat_gemm_weight(proof)
    v = verify_all(args.fpn_verify, data, root, runs=1)
    cheats["perturbed_real_weight"] = {**info, "rust": v["rust"]["verdict"], "python": v["python_detached"]["verdict"]}
    data, info = E.cheat_multiplicity(proof)
    v = verify_all(args.fpn_verify, data, root, runs=1)
    cheats["perturbed_range_multiplicity"] = {**info, "rust": v["rust"]["verdict"],
                                              "python": v["python_detached"]["verdict"]}
    out["cheats"] = cheats
    return out


# ----------------------------------------------------------------------------- v1 cost matrix
def profile_from_config(models_dir, name):
    with open(os.path.join(models_dir, name, "config.json")) as f:
        c = json.load(f)
    c = c.get("text_config", c)
    hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return Profile(name, c["hidden_size"], c["intermediate_size"], c["num_attention_heads"],
                   c["num_key_value_heads"], hd)


def layer_weights(p):
    q, kv = p.heads * p.head_width, p.groups * p.head_width
    return p.hidden * q + 2 * p.hidden * kv + q * p.hidden + 3 * p.hidden * p.intermediate


def gpu_kernel_ms(engine, n, reps=3):
    """Device-resident relation-4-shaped kernel chain at 2^n F_p3 cells: Mobius + rate-1/8 Bailey NTT (3 limbs)
    + Blake2b Merkle tree of the codeword + FracLogUp fraction tree over 2^(n+1) leaves.  Excludes host
    sumcheck / transcript / witness generation."""
    import numpy as np
    cp = engine.cp
    rng = np.random.default_rng(n)
    cells = cp.asarray(rng.integers(0, 2**63, size=(1 << n, 3), dtype=np.uint64))
    leaves = cp.asarray(rng.integers(1, 2**63, size=(1 << (n + 1), 6), dtype=np.uint64))
    times = []
    for rep in range(reps + 1):
        engine.sync()
        t = time.perf_counter()
        coeffs = engine.mobius(cells)
        rows = cp.zeros((3, 8 << n), dtype=cp.uint64)
        rows[:, :1 << n] = coeffs.T
        code = cp.ascontiguousarray(engine.ntt_rows(rows).T)
        level = engine.merkle_leaves(code)
        while len(level) > 1:
            level = engine.merkle_layer(level)
        cur = leaves
        while len(cur) > 1:
            cur = engine.fraction_layer(cur)
        engine.sync()
        if rep:                                          # rep 0 = warm-up (NVRTC compile, allocator)
            times.append((time.perf_counter() - t) * 1e3)
        del coeffs, rows, code, level, cur
    del cells, leaves
    cp.get_default_memory_pool().free_all_blocks()
    return statistics.median(times), times


def lstsq(X, y):
    import numpy as np
    coef, *_ = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)
    return coef


def cost_matrix(args):
    from prover.fpn_prover.backend import cuda_engine as CE
    profiles = [profile_from_config(args.models_dir, m) for m in COST_MODELS]
    cells = []
    for p in profiles:
        for T in COST_T:
            full, sampled, ratio = census(p, T)
            per_draw = sum(sampled.values())
            cells.append({"model": p.name, "T": T, "hidden": p.hidden, "intermediate": p.intermediate,
                          "heads": p.heads, "kv_groups": p.groups, "head_dim": p.head_width,
                          "layer_weights": layer_weights(p), "sampled_per_draw": sampled,
                          "cells_per_draw": per_draw, "cells_full_relation_per_draw": sum(full.values()),
                          "projection_100_draws_cells": 100 * per_draw,
                          "n_vars_per_draw": math.ceil(math.log2(per_draw)),
                          "n_vars_100_draws": math.ceil(math.log2(100 * per_draw))})
    engine = CE.CudaEngine()
    timed = {}
    for c in cells:
        n = c["n_vars_per_draw"]
        if n not in timed:
            timed[n] = gpu_kernel_ms(engine, n)
        c["gpu_kernel_ms_at_n"], c["gpu_kernel_ms_runs"] = timed[n][0], timed[n][1]
    return fit_cost(cells, sorted(timed))


def fit_cost(cells, distinct_n):
    """Leave-one-model-out fits on the measured cells (pure; `--refit` re-runs it on a stored receipt)."""
    import numpy as np
    folds = []
    for held in COST_MODELS:
        train = [c for c in cells if c["model"] != held]
        test = [c for c in cells if c["model"] == held]
        # (1) cell model: cells = a * layer_weights + b * T + c  (architecture-level parametric fit)
        a, b, c0 = lstsq([[c["layer_weights"], c["T"], 1] for c in train], [c["cells_per_draw"] for c in train])
        # (2a) PRE-REGISTERED time model: t = alpha * 2^n + beta on the training models' MEASURED kernel times
        al, be = lstsq([[2 ** c["n_vars_per_draw"], 1] for c in train], [c["gpu_kernel_ms_at_n"] for c in train])
        # (2b) POST-HOC time model (chosen after seeing (2a) miss the gate): log t = u + v * n (power law in cells)
        u, v = lstsq([[1, c["n_vars_per_draw"]] for c in train], [math.log(c["gpu_kernel_ms_at_n"]) for c in train])
        seen_n = {c["n_vars_per_draw"] for c in train}
        rows = []
        for c in test:
            cells_hat = a * c["layer_weights"] + b * c["T"] + c0
            n_hat = math.ceil(math.log2(max(cells_hat, 2)))
            t_lin = al * 2 ** n_hat + be
            t_pow = float(np.exp(u + v * n_hat))
            true = c["gpu_kernel_ms_at_n"]
            rows.append({"T": c["T"], "cells_true": c["cells_per_draw"], "cells_pred": cells_hat,
                         "cells_rel_err": abs(cells_hat - c["cells_per_draw"]) / c["cells_per_draw"],
                         "n_true": c["n_vars_per_draw"], "n_pred": n_hat, "gpu_ms_true": true,
                         "n_pred_timed_in_training": n_hat in seen_n,
                         "gpu_ms_pred_linear": t_lin, "gpu_rel_err_linear": abs(t_lin - true) / true,
                         "gpu_ms_pred_powerlaw": t_pow, "gpu_rel_err_powerlaw": abs(t_pow - true) / true})
        folds.append({"held_out": held, "train": [m for m in COST_MODELS if m != held],
                      "cell_model": {"a_per_layer_weight": a, "b_per_token": b, "c": c0},
                      "time_model_linear_preregistered": {"alpha_ms_per_cell": al, "beta_ms": be},
                      "time_model_powerlaw_posthoc": {"log_ms_intercept": u, "log_ms_per_n_var": v,
                                                      "exponent_in_cells": v / math.log(2)},
                      "predictions": rows,
                      "max_cells_rel_err": max(r["cells_rel_err"] for r in rows),
                      "max_gpu_rel_err_linear": max(r["gpu_rel_err_linear"] for r in rows),
                      "max_gpu_rel_err_powerlaw": max(r["gpu_rel_err_powerlaw"] for r in rows)})
    unseen = [r for f in folds for r in f["predictions"] if not r["n_pred_timed_in_training"]]
    worst = {k: max(f[f"max_{k}"] for f in folds)
             for k in ("cells_rel_err", "gpu_rel_err_linear", "gpu_rel_err_powerlaw")}
    return {"cells": cells, "folds": folds, "distinct_n_vars_timed": distinct_n,
            "worst_held_out_cells_rel_err": worst["cells_rel_err"],
            "gpu_truly_held_out_rows": [{"held_out": f["held_out"], **r} for f in folds for r in f["predictions"]
                                        if not r["n_pred_timed_in_training"]],
            "worst_gpu_rel_err_linear_preregistered_truly_held_out": max((r["gpu_rel_err_linear"] for r in unseen),
                                                                         default=None),
            "worst_gpu_rel_err_powerlaw_posthoc_truly_held_out": max((r["gpu_rel_err_powerlaw"] for r in unseen),
                                                                     default=None),
            "worst_gpu_fit_residual_linear_all_rows": worst["gpu_rel_err_linear"],
            "worst_gpu_fit_residual_powerlaw_all_rows": worst["gpu_rel_err_powerlaw"],
            "gate_20pct": {"cells_held_out_model": bool(worst["cells_rel_err"] <= 0.2),
                           "gpu_linear_preregistered_truly_held_out": bool(unseen) and bool(
                               max(r["gpu_rel_err_linear"] for r in unseen) <= 0.2),
                           "gpu_powerlaw_posthoc_truly_held_out": bool(unseen) and bool(
                               max(r["gpu_rel_err_powerlaw"] for r in unseen) <= 0.2)},
            "notes": [
                "cells = reference-IR census (prover.fpn_prover.circuit.census) of ONE sampled draw "
                "(widest 32-row GEMM tile + 1 KV group + 2 RMSNorm + SwiGLU + slot); the census is closed-form in the "
                "profile, so the cell fit tests the parametric form (layer weights, T), not a measurement",
                "gpu_kernel_ms = MEASURED median of 3 device-resident runs (after 1 warm-up) of the relation-4-shaped "
                "kernel chain at 2^n_vars_per_draw cells on this GPU; several (model, T) cells share an n_vars and "
                "therefore a timing (see distinct_n_vars_timed); time prediction chains cells_pred -> n_pred -> t(n_pred)",
                "GPU timings are keyed by n_vars, so a held-out model whose n_pred was timed for a training model is "
                "predicted from an IDENTICAL measurement: those rows (n_pred_timed_in_training = true) are in-sample FIT "
                "RESIDUALS of the time model, not predictions.  Only rows with n_pred_timed_in_training = false "
                "(gpu_truly_held_out_rows) are genuine held-out GPU predictions, and the GPU gate uses only those",
                "the linear time model t = alpha 2^n + beta was the pre-registered choice; its worst all-rows fit "
                "residual misses 20% (kernel time is super-linear between 2^19 and 2^22 cells); the power-law model was "
                "chosen AFTER seeing that residual and is post-hoc",
                "ACCEPTANCE.md v1 asks for 0.6B/1.7B/4B x T_pad in {256, 4k, 16k} with a held-out error REPORT and no "
                "numeric threshold; this receipt uses the brief's T in {256, 2048, 8192} and its <= 20% gate",
                "n_vars_100_draws > 28 exceeds one relation-4 container (2^23 rows max); the K = 2^(n-5) Plonkish "
                "capacity and IR cells are different units (one IR cell lowers to ~1-3 Plonkish cells)"]}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="<WORK_DIR>")
    ap.add_argument("--fpn-verify", default=os.path.join(ROOT, "core", "target", "release", "fpn-verify"))
    ap.add_argument("--out", default=os.path.join(ROOT, "internal", "results", "m1", "e2e_real_models_rtx6000d.json"))
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--n20-cpu-parity", action="store_true",
                    help="also run the pure-Python CPU reference at n = 20 and compare bytes (slow)")
    ap.add_argument("--refit", help="re-run only the (pure) cost fits on an existing receipt JSON, in place")
    args = ap.parse_args()
    if args.refit:
        with open(args.refit) as f:
            res = json.load(f)
        cm = res["cost_matrix"]
        res["cost_matrix"] = fit_cost(cm["cells"], cm["distinct_n_vars_timed"])
        with open(args.refit, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(json.dumps({k: v for k, v in res["cost_matrix"].items() if k.startswith(("worst", "gate"))}, indent=1))
        return
    import torch
    from prover.fpn_prover.backend import cuda_engine as CE
    t_all = time.perf_counter()
    results = {"wave": "7A", "source_commit": git_rev(), "host": platform.node(),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "fpn_verify_sha256": E.sha256(open(args.fpn_verify, "rb").read()),
               "prompt": PROMPT, "models": []}
    try:
        import cupy
        import transformers
        results["versions"] = {"cupy": cupy.__version__, "transformers": transformers.__version__,
                               "python": platform.python_version()}
    except ImportError:
        pass
    t0 = time.perf_counter()
    engine = CE.CudaEngine()
    E.prove_samples(E.Samples("warmup", "warmup", (0x3f80, 0x3f80), (0x3f80, 0x3f80), 0x3f80, 0x3f80, 0x3f80,
                              "dense_gqa", dict(normalize=0, query=0, key0=0, key1=0, value0=0, value1=0)),
                    engine=engine)
    engine.sync()
    results["engine"] = {"name": engine.name, "cold_init_plus_first_n14_prove_ms": (time.perf_counter() - t0) * 1e3,
                         "note": "NVRTC compile of kernels/goldilocks_ntt.cu + first container (excluded from the "
                                 "per-model timings below, which are warm)"}
    results["scope"] = {
        "statement": "n = 14: ONE fpn2-folded-air-v1 container per (model, opcode) (K = 512 Plonkish rows / "
                     "variables, C_p = 262,144 B) proving the shared fixed-shape Program -- rmsnorm(width 2) + gemm "
                     "tile (1 x 1) + 5-opcode microprogram slot with the model's real opcode selected -- for FIVE "
                     "checkpoints / four opcodes under ONE circuit root.  n = 20: ONE fpn2-folded-air-n20-v1 container "
                     "(K = 32,768, C_p = 524,288 B) proving 64 copies of that block set filled with 64 real "
                     "Qwen3-0.6B samples (its own circuit root).  Every container is produced by the "
                     "CudaEngine-accelerated folded_air.prove and verified by the native Rust fpn-verify AND the "
                     "detached Python fpn_verify_ref (-I, verifier/python only on sys.path), circuit root pinned.",
        "not_claimed": [
            "NOT a full request: the census 100-draw projection (cost_matrix.cells[*].projection_100_draws_cells) "
            "is 10^7-10^8 IR cells vs ~328 Plonkish rows (n = 14) / ~21k rows (n = 20) proven here",
            "LookupSpec reference predicates (rsqrt enclosure, BF16 decoding, gemm box tolerance, branch relations "
            "incl. the MoE exp cells and nearest-rounded mixture) are checked prover-side only (lower.assign_air), "
            "NOT by the verifiers",
            "MoE: the slot has experts = top_k = 2, so a MoE container proves the softmax-renormalised 2-expert "
            "SUB-mixture over the model's two highest router logits (top-k selection is vacuous), not the "
            "128-expert top-8 (Qwen3-30B-A3B) / 32-expert top-4 (gpt-oss) routing",
            "swa_sink: window = 1, sink = 1, extent = 1, position 0 -> output == v0 by construction; gpt-oss's "
            "LEARNED sink logits (s_aux) are not modelled",
            "gpt-oss expert outputs are recomputed from the dequantised MXFP4 checkpoint blocks with the reference "
            "GptOssExperts arithmetic (served path = Triton MXFP4 kernel; see moe_check in samples_meta); Qwen3-30B "
            "expert outputs come from the model's own FP8Experts module",
            "containers are EQUAL-LENGTH by construction (fixed c_p padding), not byte-identical; the "
            "architecture-hiding fact is the shared circuit_root",
            "the width-2 norm output, the 1x1 gemm observed value (GPU bf16 product of the real pair) and the slot "
            "outputs are recomputed from real values (witness_provenance), not the model's own layer outputs",
            "GPU vs host split: the kernels (Mobius + Bailey NTT + BLAKE2b Merkle of F / C, fraction tree, batched "
            "sumcheck rounds + halving, H = F + theta C, coset-4 folds + folded-layer trees) run on the GPU; GKR leaves, "
            "Libra-masked GKR layer sumchecks, zerocheck, z_blind and weight vectors are vectorised host NumPy; "
            "witness sampling, transcript and wire assembly are host Python (folded_air_prove_phases)",
            "the ZK joint-rank check is run at n = 14 only (not at n = 20)"],
    }
    pairs = {}
    for spec in MODELS:
        for (s, p) in run_model(spec, args, results, engine):
            pairs[(spec["name"], s.opcode)] = (s, p)
    ms = results["models"]
    key = lambda m: f"{m['model']}:{m['opcode']}"  # noqa: E731
    results["m2_2_m2_3_shared_root"] = {
        "containers": len(ms), "architectures": sorted({m["arch"] for m in ms}),
        "opcodes": sorted({m["opcode"] for m in ms}),
        "lengths": {key(m): m["container"]["bytes"] for m in ms},
        "circuit_roots": {key(m): m["container"]["circuit_root"] for m in ms},
        "all_equal_length": len({m["container"]["bytes"] for m in ms}) == 1,
        "all_same_root": len({m["container"]["circuit_root"] for m in ms}) == 1,
        "all_verified_rust": all(m["verify"]["rust"]["verdict"] == "VERIFIED" for m in ms),
        "all_verified_python": all(m["verify"]["python_detached"]["verdict"] == "VERIFIED" for m in ms),
        "all_cpu_cuda_parity": all(m["parity_cpu_vs_cuda_seeded"]["identical"] for m in ms)}
    print(json.dumps(results["m2_2_m2_3_shared_root"], indent=1), flush=True)
    mb_samples, _ = run_multiblock(args, results, engine)
    if args.n20_cpu_parity:
        results["n20_multiblock"]["cpu_parity_seeded"] = n20_cpu_parity(mb_samples, engine)
    s, p = pairs[("Qwen3-0.6B", "dense_gqa")]
    results["m2_3_zk_qwen3"] = zk_and_cheats(s, p, args)
    if not args.skip_cost:
        results["cost_matrix"] = cost_matrix(args)
    results["host_peak_rss_mib"] = rss_mib()
    results["wall_seconds"] = time.perf_counter() - t_all
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
