# SPDX-License-Identifier: Apache-2.0
"""Full, unsliced Qwen3-0.6B end-to-end zero-knowledge reply proof benchmark.

Subcommands:
  capture   (torch venv, CUDA): HF bf16 + SDPA FlashAttention-2 + sampling_v3 rollout on real Qwen3-0.6B
            with hidden system prompt, public user request, hidden <think>...</think> CoT, and visible reply.
  commit    (cupy venv, CUDA):  Streaming commitment of full Qwen3-0.6B (W + W^r + CFG0/CFG1 + M^vol) with
            store_coeffs=True, store_cw=False (~5.2 GB for W; 7.62 GB with RAW; queried columns re-encoded on GPU).
  prove     (cupy venv, CUDA):  Full reply_zk proof (default M_C=210, M_G=105 = 315 spot-check draws);
            records build/prove time, RSS (VmHWM), CuPy PoolPeak, nvidia-smi peak VRAM, and ZK budget.
  verify    (CPU-only process, CUDA_VISIBLE_DEVICES=""): Standalone verifier from public profile + statement + proof.

Example (from a source checkout; MODEL is a local Hugging Face snapshot of Qwen3-0.6B):
  python benchmarks/prover/bench_qwen_reply.py capture --model MODEL --out run/cap.npz
  python benchmarks/prover/bench_qwen_reply.py commit  --model MODEL --cap run/cap.npz --store run/store
  python benchmarks/prover/bench_qwen_reply.py prove   --model MODEL --cap run/cap.npz --store run/store --out-dir run
  CUDA_VISIBLE_DEVICES="" python benchmarks/prover/bench_qwen_reply.py verify --out-dir run --json-out run/result.json

Every result records the host it ran on (GPU and CPU model detected at run time).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flashprove_next.hwinfo import describe as describe_machine, nvidia_smi  # noqa: E402

KPRF_DEFAULT = [0x0123456789ABCDEF % ((1 << 64) - (1 << 32) + 1), 0x11112222, 0x33334444, 0x55556666]
NONCE_C = bytes(range(1, 17))
NONCE_S = bytes(range(33, 65))
STATE_R = (3, 5, 7, 11)
THINK_END_ID = 151668   # Qwen3 </think>


def _vm_gib():
    """Return (VmRSS_GiB, VmHWM_GiB) from /proc/self/status on Linux."""
    rss = hwm = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / (1 << 20)
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return round(rss, 3), round(hwm, 3)


def _start_nvml_poller(interval=1.0):
    """Background nvidia-smi memory.used poller (returns stop_fn() -> peak_gib)."""
    stop = threading.Event()
    peak_mib = [None]
    smi = nvidia_smi()
    if smi is None:                       # no nvidia-smi (set FPN_NVIDIA_SMI): report None, not a fake 0
        return lambda: None

    def loop():
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=3,
                ).decode().strip()
                val = float(out.splitlines()[0].strip())
                if peak_mib[0] is None or val > peak_mib[0]:
                    peak_mib[0] = val
            except Exception:
                pass
            stop.wait(interval)

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def finish():
        stop.set()
        th.join(timeout=3)
        return None if peak_mib[0] is None else round(peak_mib[0] / 1024.0, 3)

    return finish


def qwen_rope_fn(dh, rope_theta):
    """Exact platform-independent Qwen3 bf16 RoPE table recipe."""
    from flashprove_next.reply_bundle import qwen_rope_fn as _qwen_rope_fn
    return _qwen_rope_fn(int(dh), float(rope_theta))


def make_qwen_profile(cfg, vec_pad, t_pad, m_c, m_g, sys_cap=32):
    from flashprove_next import modelcommit as MC, reply_zk as RZ, sampling_v3 as S3
    d = int(cfg["hidden_size"])
    dff = int(cfg["intermediate_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    dh = int(cfg.get("head_dim") or d // n_q)
    V = int(cfg["vocab_size"])
    eps_bits = int(np.array([cfg["rms_norm_eps"]], dtype=np.float32).view(np.uint32)[0])
    rope_theta = float(cfg["rope_theta"])
    rule = S3.Rule(0.6, 0.95, 32)
    return RZ.Profile(
        d=d, n_q=n_q, n_kv=n_kv, dh=dh, dff=dff,
        L_MAX=32, T_pad=int(t_pad), M_C=int(m_c), M_G=int(m_g),
        prm=MC.QWEN_PARAMS_R, MQ_H=268, s_bits=1, SYS_CAP=int(sys_cap),
        VEC_PAD=int(vec_pad), QN_CAP=12.0, KN_CAP=1100.0, EXP_BITS=17,
        attention=True, V=V, HIST_CAP=max(16, int(t_pad)), rule=rule,
        rope_fn=qwen_rope_fn(dh, rope_theta), eps_bits=eps_bits,
    )


# ============================================================================ 1. capture (torch venv)
def cmd_capture(a):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask
    from flashprove_next import reply_zk as RZ, sampling_v3 as S3, state_zk as SZ

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    step_rec = {}

    def cap_attn(module, query, key, value, attention_mask, **kw):
        q_len = query.shape[2]
        kw.pop("is_causal", None)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out, w = sdpa_attention_forward(module, query, key, value, None, is_causal=(q_len > 1), **kw)
        rep = query.shape[1] // key.shape[1]
        r = torch.ops.aten._scaled_dot_product_flash_attention(
            query, repeat_kv(key, rep), repeat_kv(value, rep), 0.0, q_len > 1, False, scale=module.scaling
        )
        assert bool(torch.equal(r[0].transpose(1, 2), out)), f"FA2 LSE output mismatch at layer {module.layer_idx}"
        L = module.layer_idx
        D = step_rec.setdefault(L, {})
        D["q"] = query[0].transpose(0, 1).detach()               # (q_len, n_q, dh)
        D["k"] = key[0, :, -q_len:, :].transpose(0, 1).detach()  # (q_len, n_kv, dh)
        D["v"] = value[0, :, -q_len:, :].transpose(0, 1).detach()
        D["o"] = out[0].reshape(q_len, -1).detach()
        D["lse"] = r[1][0].transpose(0, 1).detach()              # (q_len, n_q) fp32
        return out, w

    AttentionInterface.register("fpn_qwen_cap", cap_attn)
    AttentionMaskInterface.register("fpn_qwen_cap", sdpa_mask)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="fpn_qwen_cap"
    ).cuda().eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    cfg = model.config
    L_real = int(cfg.num_hidden_layers)
    d = int(cfg.hidden_size)
    dff = int(cfg.intermediate_size)
    n_q = int(cfg.num_attention_heads)
    n_kv = int(cfg.num_key_value_heads)
    dh = int(getattr(cfg, "head_dim", None) or d // n_q)
    eos_id = int(cfg.eos_token_id)

    handles = []
    g_rec = {}
    for L, layer in enumerate(model.model.layers):
        def mk_io(L_, nm_i, nm_o):
            def fn(mod, inp, out):
                D = step_rec.setdefault(L_, {})
                if nm_i:
                    D[nm_i] = inp[0][0].detach()
                if nm_o:
                    D[nm_o] = out[0].detach()
            return fn

        def mk_lh(L_):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                step_rec.setdefault(L_, {})["h_out"] = t[0].detach()
            return fn

        handles.append(layer.input_layernorm.register_forward_hook(mk_io(L, "h_in", "x_attn")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(mk_io(L, "h_mid", "x_mlp")))
        handles.append(layer.mlp.down_proj.register_forward_hook(mk_io(L, "a", None)))
        handles.append(layer.register_forward_hook(mk_lh(L)))

    handles.append(model.model.norm.register_forward_hook(lambda m, i, o: g_rec.__setitem__("x_fn", o[0].detach())))

    def pe_hook(mod, args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        g_rec["cos"] = cos[0].detach()
        g_rec["sin"] = sin[0].detach()

    handles.append(model.model.layers[0].self_attn.register_forward_pre_hook(pe_hook, with_kwargs=True))

    sys_ids = tok.encode(f"<|im_start|>system\n{a.sys_text}<|im_end|>\n", add_special_tokens=False)
    req_ids = tok.encode(
        f"<|im_start|>user\n{a.user_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
    )
    assert len(sys_ids) <= a.sys_cap, (len(sys_ids), a.sys_cap)
    prompt_ids = sys_ids + req_ids

    # Pure inference timing baseline (no hooks, warm GPU) for the serving-overhead comparison
    plain_model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    rule = S3.Rule(0.6, 0.95, 32)
    sess_t0 = SZ.sess_of(NONCE_C)
    vrel_t0 = SZ.v_rel_of(RZ.VERSION)
    sctx_t0 = RZ.samp_ctx_of({
        "req_tokens": list(req_ids),
        "st": {"N_c": NONCE_C.hex(), "N_s": NONCE_S.hex(), "sess": list(sess_t0), "v_rel": vrel_t0, "turn": 0},
    })
    pre_prf = S3.prefix_state(KPRF_DEFAULT, sctx_t0)
    with torch.no_grad():
        _ = plain_model(input_ids=torch.tensor([prompt_ids], device="cuda"), use_cache=True)
        torch.cuda.synchronize()
        t_inf0 = time.perf_counter()
        out_p = plain_model(input_ids=torch.tensor([prompt_ids], device="cuda"), use_cache=True)
        cache_p = out_p.past_key_values
        cur_ids = list(prompt_ids)
        for g in range(a.max_new_tokens):
            lg_p = out_p.logits[0, -1].float().cpu().numpy()
            nxt_p = int(S3.sample(pre_prf, g, lg_p, rule))
            cur_ids.append(nxt_p)
            if nxt_p == eos_id:
                break
            if len(cur_ids) + 1 >= a.t_pad:
                break
            out_p = plain_model(input_ids=torch.tensor([[nxt_p]], device="cuda"), past_key_values=cache_p, use_cache=True)
            cache_p = out_p.past_key_values
        torch.cuda.synchronize()
        plain_inf_s = time.perf_counter() - t_inf0
    del plain_model, out_p, cache_p
    torch.cuda.empty_cache()

    def u16(t):
        return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)

    def u32(t):
        return t.to(torch.float32).contiguous().view(torch.int32).cpu().numpy().view(np.uint32)

    W_lm = model.lm_head.weight
    all_h_in, all_x_attn, all_q, all_k, all_v, all_o = [], [], [], [], [], []
    all_h_mid, all_x_mlp, all_a, all_h_out, all_lse, all_xfn = [], [], [], [], [], []
    cos_rows, sin_rows = [], []

    def harvest():
        n_tok = g_rec["x_fn"].shape[0]
        all_xfn.append(u16(g_rec["x_fn"]))
        cos_rows.append(u16(g_rec["cos"]))
        sin_rows.append(u16(g_rec["sin"]))
        for arr_l, key, shp in (
            (all_h_in, "h_in", (n_tok, L_real, d)),
            (all_x_attn, "x_attn", (n_tok, L_real, d)),
            (all_q, "q", (n_tok, L_real, n_q * dh)),
            (all_k, "k", (n_tok, L_real, n_kv * dh)),
            (all_v, "v", (n_tok, L_real, n_kv * dh)),
            (all_o, "o", (n_tok, L_real, n_q * dh)),
            (all_h_mid, "h_mid", (n_tok, L_real, d)),
            (all_x_mlp, "x_mlp", (n_tok, L_real, d)),
            (all_a, "a", (n_tok, L_real, dff)),
            (all_h_out, "h_out", (n_tok, L_real, d)),
        ):
            stk = np.stack([u16(step_rec[L][key]).reshape(n_tok, -1) for L in range(L_real)], axis=1)
            assert stk.shape == shp, (key, stk.shape, shp)
            arr_l.append(stk)
        lse_stk = np.stack([u32(step_rec[L]["lse"]).reshape(n_tok, n_q) for L in range(L_real)], axis=1)
        all_lse.append(lse_stk)

    gen_ids = []
    with torch.no_grad():
        torch.cuda.synchronize()
        t_cap0 = time.perf_counter()
        step_rec.clear(); g_rec.clear()
        out = model(input_ids=torch.tensor([prompt_ids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        harvest()
        for g in range(a.max_new_tokens):
            xf_last = g_rec["x_fn"][-1:].to(torch.bfloat16)
            lg = torch.mm(xf_last, W_lm.T, out_dtype=torch.float32)[0]
            lgn = lg.float().cpu().numpy()
            nxt = int(S3.sample(pre_prf, g, lgn, rule))
            gen_ids.append(nxt)
            if len(prompt_ids) + len(gen_ids) >= a.t_pad or nxt == eos_id:
                break
            step_rec.clear(); g_rec.clear()
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            harvest()
        # Note: the final sampled token `gen_ids[-1]` sits at position T - 1 in `tokens = prompt_ids + gen_ids`.
        # If the loop exited right after sampling `gen_ids[-1]`, run one forward step for that last token position so
        # `Capture` has all T = len(prompt_ids) + len(gen_ids) positions populated.
        captured_T = sum(x.shape[0] for x in all_xfn)
        full_tokens = prompt_ids + gen_ids
        if captured_T < len(full_tokens):
            assert captured_T == len(full_tokens) - 1
            step_rec.clear(); g_rec.clear()
            _ = model(input_ids=torch.tensor([[gen_ids[-1]]], device="cuda"), past_key_values=cache, use_cache=True)
            harvest()
        torch.cuda.synchronize()
        cap_s = time.perf_counter() - t_cap0

    for h in handles:
        h.remove()

    # Split gen_ids into hidden CoT (up to and including </think> if present) and visible reply tokens
    if THINK_END_ID in gen_ids:
        cot_end = gen_ids.index(THINK_END_ID) + 1
        # Include trailing newline after </think> inside hidden CoT if present and at least 1 visible token remains
        while cot_end < len(gen_ids) - 1 and tok.decode([gen_ids[cot_end]]) in ("\n", "\n\n"):
            cot_end += 1
        cot_ids = gen_ids[:cot_end]
        vis_ids = gen_ids[cot_end:]
    else:
        # Ensure non-empty visible reply
        cot_ids = []
        vis_ids = gen_ids

    assert len(vis_ids) >= 1, f"expected >= 1 visible token, got gen_ids={gen_ids}"
    assert len(full_tokens) <= a.t_pad, (len(full_tokens), a.t_pad)

    # Verify RoPE recipe matches captured cos/sin bitwise on every position
    cos_all = np.concatenate(cos_rows, axis=0)
    sin_all = np.concatenate(sin_rows, axis=0)
    raw_cfg = json.load(open(os.path.join(a.model, "config.json")))
    rope_theta = float(raw_cfg.get("rope_theta") or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta"))
    rope_fn = qwen_rope_fn(dh, rope_theta)
    f32 = lambda c: (np.asarray(c, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    for pos in range(len(full_tokens)):
        rc, rs = rope_fn(pos)
        assert np.array_equal(rc, f32(cos_all[pos])) and np.array_equal(rs, f32(sin_all[pos])), f"RoPE mismatch at {pos}"

    meta = {
        "model": "Qwen3-0.6B",
        "model_path": os.path.basename(os.path.normpath(a.model)),
        "machine": describe_machine(),
        "sys_text": a.sys_text,
        "user_text": a.user_text,
        "cot_text_preview": tok.decode(cot_ids)[:200],
        "vis_text": tok.decode(vis_ids),
        "sys_tokens": sys_ids,
        "req_tokens": req_ids,
        "cot_tokens": cot_ids,
        "vis_tokens": vis_ids,
        "T": len(full_tokens),
        "t_pad": int(a.t_pad),
        "sys_cap": int(a.sys_cap),
        "k_prf": KPRF_DEFAULT,
        "nonce_c_hex": NONCE_C.hex(),
        "nonce_s_hex": NONCE_S.hex(),
        "rule": rule.as_dict(),
        "plain_inference_gpu_s": round(plain_inf_s, 4),
        "capture_forward_gpu_s": round(cap_s, 4),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(
        a.out,
        tokens=np.asarray(full_tokens, dtype=np.int64),
        h_in=np.concatenate(all_h_in, axis=0),
        x_attn=np.concatenate(all_x_attn, axis=0),
        q=np.concatenate(all_q, axis=0),
        k=np.concatenate(all_k, axis=0),
        v=np.concatenate(all_v, axis=0),
        o=np.concatenate(all_o, axis=0),
        h_mid=np.concatenate(all_h_mid, axis=0),
        x_mlp=np.concatenate(all_x_mlp, axis=0),
        a=np.concatenate(all_a, axis=0),
        h_out=np.concatenate(all_h_out, axis=0),
        x_fn=np.concatenate(all_xfn, axis=0),
        lse=np.concatenate(all_lse, axis=0),
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    print(json.dumps({
        "phase": "capture", "T": meta["T"], "sys_len": len(sys_ids), "n_req": len(req_ids),
        "n_reason": len(cot_ids), "n_vis": len(vis_ids), "vis_text": meta["vis_text"],
        "plain_inference_gpu_s": meta["plain_inference_gpu_s"], "capture_forward_gpu_s": meta["capture_forward_gpu_s"],
    }, ensure_ascii=False), flush=True)


# ============================================================================ 2. commit (cupy venv)
def cmd_commit(a):
    from flashprove_next import mc_zk as MZ, modelcommit as MC, reply_zk as RZ
    from flashprove_next.field import gpu_backend

    B = gpu_backend()
    assert B.name == "cupy", f"expected cupy GPU backend, got {B.name}"
    cap_npz = np.load(a.cap)
    cap_meta = json.loads(bytes(cap_npz["meta"]).decode())
    ck = MC.Checkpoint(a.model)
    lazy_t = RZ.LazyCommitTensors(ck, MC.QWEN_PARAMS_R.TI)
    prm = MC.QWEN_PARAMS_R
    lay = MC.Layout(lazy_t.shapes(), prm, radius=True, cfg_rows=True)
    assert lay.n_vec_rows == 1, f"expected 1 vec row, got {lay.n_vec_rows}"
    vec_pad = 1 << max(1, (lay.vec_len - 1).bit_length())
    assert vec_pad <= prm.nc, (vec_pad, prm.nc)

    os.makedirs(a.store, exist_ok=True)
    commit_json = {
        "W": {
            "n_rows": lay.n_rows,
            "row_ranges": [list(r) for r in lay.row_ranges],
            "radius": True,
            "cfg_rows": True,
        }
    }
    if os.path.exists(os.path.join(a.store, "W_tree.npy")) and os.path.exists(os.path.join(a.store, "W_coeffs.i32")):
        with open(os.path.join(a.store, "commit.json"), "w") as f:
            json.dump(commit_json, f, indent=1)
        st_existing = MC.Store(a.store, prm, tag="W")
        w_root = st_existing.root
        commit_s = 0.0
        cstats = {"reused_store": True, "coeff_bytes": os.path.getsize(os.path.join(a.store, "W_coeffs.i32"))}
    else:
        src = MC.RowSource(lazy_t, lay)
        t0 = time.perf_counter()
        cw = MC.commit_matrix(
            src, lay.n_span, prm, B,
            store_dir=a.store, tag="W",
            store_coeffs=True, store_cw=False,
            row_ranges=lay.row_ranges,
        )
        commit_s = time.perf_counter() - t0
        w_root = cw["root"]
        cstats = {k: v for k, v in cw["stats"].items() if k != "codeword_files"}
        with open(os.path.join(a.store, "commit.json"), "w") as f:
            json.dump(commit_json, f, indent=1)

    rng = np.random.default_rng(7)
    salt_cfg = [int(x) for x in rng.integers(1, 1 << 30, size=4)]
    rule_v3 = S3.Rule(0.6, 0.95, 32)
    vol = MC.encode_volatile_cfg({
        "k_prf": list(cap_meta["k_prf"]),
        "salt_cfg": salt_cfg,
        "sys_prompt": list(cap_meta["sys_tokens"]),
        "L_real": int(ck.config["num_hidden_layers"]),
        "V_real": int(ck.config["vocab_size"]),
        "T_bits": rule_v3.T_bits,
        "p16": rule_v3.p16,
        "beta": rule_v3.beta,
    })
    ccfg = MC.volatile_config_root(vol)
    pre, mid = MZ.model_preimage(w_root, c_cfg=ccfg)

    setup_meta = {
        "model": "Qwen3-0.6B",
        "machine": describe_machine(),
        "prm": asdict(prm),
        "vec_pad": vec_pad,
        "n_rows": lay.n_rows,
        "row_ranges": [list(r) for r in lay.row_ranges],
        "meta_cfg0": lay.meta_cfg0,
        "meta_cfg1": lay.meta_cfg1,
        "mid": [int(x) for x in mid],
        "pre": {k: ([int(x) for x in v] if isinstance(v, (list, tuple)) else int(v)) for k, v in pre.items()},
        "vol": [int(x) for x in vol],
        "salt_cfg": salt_cfg,
        "commit_w_s": round(commit_s, 3),
        "commit_stats": cstats,
    }
    with open(os.path.join(a.store, "setup_meta.json"), "w") as f:
        json.dump(setup_meta, f, indent=1)
    print(json.dumps({
        "phase": "commit", "n_rows": lay.n_rows, "row_ranges": setup_meta["row_ranges"],
        "vec_pad": vec_pad, "commit_w_s": setup_meta["commit_w_s"],
        "coeff_gib": round(cstats.get("coeff_bytes", 0) / (1 << 30), 3),
    }), flush=True)


# ============================================================================ 3. prove (cupy venv)
def cmd_prove(a):
    import cupy as cp
    from flashprove_next import hsel_zk as HZ, modelcommit as MC, reply_zk as RZ, state_zk as SZ, zk_flat as Z
    from flashprove_next.dense_engine import Capture
    from flashprove_next.field import gpu_backend
    from flashprove_next.transcript import dumps

    os.environ.setdefault("FPN_OOC", "1")
    B = gpu_backend()
    assert B.name == "cupy", f"expected cupy GPU backend, got {B.name}"

    stop_nvml = _start_nvml_poller(1.0)
    ck = MC.Checkpoint(a.model)
    lazy_t = RZ.LazyCommitTensors(ck, MC.QWEN_PARAMS_R.TI)
    setup_meta = json.load(open(os.path.join(a.store, "setup_meta.json")))
    cap_d = np.load(a.cap)
    cap_meta = json.loads(bytes(cap_d["meta"]).decode())
    cap = Capture(
        tokens=cap_d["tokens"],
        h_in=cap_d["h_in"], x_attn=cap_d["x_attn"],
        q=cap_d["q"], k=cap_d["k"], v=cap_d["v"], o=cap_d["o"],
        h_mid=cap_d["h_mid"], x_mlp=cap_d["x_mlp"], a=cap_d["a"], h_out=cap_d["h_out"],
        x_fn=cap_d["x_fn"], logits=np.zeros((0, 0), dtype=np.uint32), lse=cap_d["lse"],
    )

    prf = make_qwen_profile(
        ck.config, setup_meta["vec_pad"], cap_meta["t_pad"], a.m_c, a.m_g, sys_cap=cap_meta["sys_cap"]
    )
    lay = MC.Layout(lazy_t.shapes(), prm=MC.QWEN_PARAMS_R, radius=True, cfg_rows=True)
    store = MC.Store(a.store, MC.QWEN_PARAMS_R, tag="W")
    com = {
        "layout": lay,
        "store": store,
        "pre": {k: tuple(v) if isinstance(v, list) else v for k, v in setup_meta["pre"].items()},
        "mid": tuple(setup_meta["mid"]),
        "vol": np.asarray(setup_meta["vol"], dtype=np.uint64),
    }

    t_h0 = time.perf_counter()
    R_root, wit = RZ.make_wit(
        prf, lazy_t, cap, com, tokens=cap.tokens, sys_len=len(cap_meta["sys_tokens"]), B=B
    )
    wit["hs_wit"]["release_h"] = True
    cap_d.close()
    del cap_d
    commit_h_s = time.perf_counter() - t_h0

    api = {
        "req_tokens": list(cap_meta["req_tokens"]),
        "n_reason": len(cap_meta["cot_tokens"]),
        "vis_tokens": list(cap_meta["vis_tokens"]),
    }
    stp, stw, _ = RZ.turn_state(
        prf, com, api, list(cap_meta["cot_tokens"]), NONCE_C, NONCE_S, STATE_R
    )
    wit["st"] = stw
    stmt = RZ.statement(prf, com, R_root, api, stp)

    Z.MEM_REPORT.clear()
    Z.STREAM_STATS.clear()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    with Z.PoolPeak() as pk:
        pf, info = RZ.prove(prf, stmt, wit, B=B)

    blob = dumps(pf)
    rss_gib, hwm_gib = _vm_gib()
    nvml_peak_gib = stop_nvml()
    pool_peak_gib = round(pk.peak / (1 << 30), 3)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "proof.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(a.out_dir, "stmt.json"), "w") as f:
        json.dump({"stmt": stmt, "profile": {
            "vec_pad": setup_meta["vec_pad"], "t_pad": cap_meta["t_pad"],
            "m_c": int(a.m_c), "m_g": int(a.m_g), "sys_cap": cap_meta["sys_cap"],
            "config": ck.config,
        }}, f)

    metrics = {
        "model": "Qwen3-0.6B",
        "machine": describe_machine(),
        "version": RZ.VERSION,
        "M_C": int(a.m_c),
        "M_G": int(a.m_g),
        "total_draws": int(a.m_c) + int(a.m_g),
        "T": int(cap_meta["T"]),
        "T_pad": int(cap_meta["t_pad"]),
        "V": int(ck.config["vocab_size"]),
        "L": int(ck.config["num_hidden_layers"]),
        "sys_len_hidden": len(cap_meta["sys_tokens"]),
        "n_req": len(cap_meta["req_tokens"]),
        "n_reason": len(cap_meta["cot_tokens"]),
        "n_vis": len(cap_meta["vis_tokens"]),
        "vis_text": cap_meta["vis_text"],
        "plain_inference_gpu_s": cap_meta["plain_inference_gpu_s"],
        "capture_forward_gpu_s": cap_meta["capture_forward_gpu_s"],
        "commit_w_s": setup_meta["commit_w_s"],
        "commit_h_s": round(commit_h_s, 3),
        "build_s": info["build_s"],
        "prove_s": info["prove_s"],
        "total_prover_gpu_s": round(commit_h_s + info["build_s"] + info["prove_s"], 3),
        "cells": info["cells"],
        "cells_by_kind": info["cells_by_kind"],
        "slots_used": info["slots_used"],
        "claim_slots_J": prf.claim_slots,
        "hsel_slots_J": prf.J,
        "proof_bytes": len(blob),
        "gadget_fails": info["gadget_fails"],
        "violations": info["violations"],
        "zk_budget": info["zk_budget"],
        "rss_end_gib": rss_gib,
        "max_rss_hwm_gib": hwm_gib,
        "gpu_pool_peak_gib": pool_peak_gib,
        "nvml_peak_gib": nvml_peak_gib,
        "stream_stats": list(Z.STREAM_STATS),
        "mem_report": list(Z.MEM_REPORT),
    }
    with open(os.path.join(a.out_dir, "prove_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("zk_budget", "stream_stats", "mem_report")},
                     ensure_ascii=False), flush=True)


# ============================================================================ 4. verify (CPU-only process)
def cmd_verify(a):
    from flashprove_next import reply_zk as RZ
    from flashprove_next.transcript import loads

    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", \
        "run the verifier as a CPU-only process: CUDA_VISIBLE_DEVICES=''"
    bundle = json.load(open(os.path.join(a.out_dir, "stmt.json")))
    pm = bundle["profile"]
    stmt = bundle["stmt"]
    prf = make_qwen_profile(
        pm["config"], pm["vec_pad"], pm["t_pad"], pm["m_c"], pm["m_g"], sys_cap=pm["sys_cap"]
    )
    with open(os.path.join(a.out_dir, "proof.bin"), "rb") as f:
        blob = f.read()
    pf = loads(blob)
    t0 = time.perf_counter()
    verdict = RZ.try_verify(prf, stmt, pf)
    verify_s = round(time.perf_counter() - t0, 3)
    rss_gib, hwm_gib = _vm_gib()

    metrics = json.load(open(os.path.join(a.out_dir, "prove_metrics.json")))
    metrics.update({
        "verify_verdict": verdict,
        "verify_cpu_s": verify_s,
        "verify_max_rss_gib": hwm_gib,
        "cuda_visible_devices_verifier": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(metrics, f, indent=1, ensure_ascii=False)
    print(json.dumps({
        "phase": "verify", "verdict": verdict, "verify_cpu_s": verify_s,
        "verify_max_rss_gib": hwm_gib, "proof_bytes": len(blob),
    }), flush=True)
    if verdict != "accepted":
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--model", required=True)
    p_cap.add_argument("--out", required=True)
    p_cap.add_argument("--t-pad", type=int, default=128)
    p_cap.add_argument("--sys-cap", type=int, default=32)
    p_cap.add_argument("--max-new-tokens", type=int, default=90)
    p_cap.add_argument("--sys-text", default="You are a concise math assistant.")
    p_cap.add_argument("--user-text", default="Compute 17 * 23.")

    p_com = sub.add_parser("commit")
    p_com.add_argument("--model", required=True)
    p_com.add_argument("--cap", required=True)
    p_com.add_argument("--store", required=True)

    p_prv = sub.add_parser("prove")
    p_prv.add_argument("--model", required=True)
    p_prv.add_argument("--cap", required=True)
    p_prv.add_argument("--store", required=True)
    p_prv.add_argument("--out-dir", required=True)
    p_prv.add_argument("--m-c", type=int, default=210)
    p_prv.add_argument("--m-g", type=int, default=105)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--out-dir", required=True)
    p_ver.add_argument("--json-out", default=None)

    args = ap.parse_args(argv)
    {"capture": cmd_capture, "commit": cmd_commit, "prove": cmd_prove, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
