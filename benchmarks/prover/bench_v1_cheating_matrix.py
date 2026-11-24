#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v1.0 Real-Model End-to-End Cheating Rejection Matrix (`V1-2` Gate).

Runs on a real dense Qwen3 checkpoint (e.g. `Qwen3-0.6B`) with its committed BF16 weight store and
tests the full unified folded ZK reply prover + independent CPU verifier against 1 honest acceptance
plus 13 real-world negative cases (14 total):

  1.  honest_extseed_fpnzk2          -> ACCEPT (exact 8 MiB zero-padded FPNZK2, 0 Phase-2, Q=100)
  2.  real_perturbed_weights_model   -> REJECT (loads real model, perturbs linear projection weights
                                                in-place, runs real 28-layer PyTorch forward capture,
                                                proves against honest BF16 commitment with Z.FORCE=True)
  3.  real_int4_quantized_model      -> REJECT (loads real model, quantizes all 28 layers' linear
                                                weights to symmetric group-32 int4 in-place, runs real
                                                28-layer PyTorch forward capture, proves against honest
                                                BF16 commitment with Z.FORCE=True)
  4.  skipped_layers                 -> REJECT (prover skips transformer layers in capture, Z.FORCE=True)
  5.  truncated_kv_context           -> REJECT (prover zeros early KV rows in capture, Z.FORCE=True)
  6.  tampered_hidden_sys_prompt     -> REJECT (prover alters hidden system prompt tokens, Z.FORCE=True)
  7.  tampered_visible_reply         -> REJECT (adversary alters visible reply token in statement)
  8.  replay_tampered_st_prev        -> REJECT (adversary alters client state anchor St_prev)
  9a. missing_cpre_pin               -> REJECT (E_CPRE_UNPINNED)
  9b. swapped_cpre_pin               -> REJECT (E_CPRE_MISMATCH)
  9c. swapped_cext_pin               -> REJECT (E_CEXT_MISMATCH)
  9d. fpnzk2_nonzero_padding         -> REJECT (E_PADDING)
  9e. downgraded_queries_4           -> REJECT (E_REDUCED_SECURITY)
  9f. fpnzk2_wrong_envelope_size     -> REJECT (E_LENGTH: self-consistent 4 MiB != trusted 8 MiB profile)

Every witness-side attack runs with `Z.FORCE = True` so the malicious prover ACTUALLY generates a
proof and the independent verifier MUST reject it (never relying on honest-prover self-refusal).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.prover.bench_qwen_reply import (  # noqa: E402
    NONCE_C, NONCE_S, STATE_R, _select_mc_params, make_qwen_profile,
)
from flashprove_next.hwinfo import describe as describe_machine  # noqa: E402


def _capture_altered_model_forward(model_path: str, tokens: np.ndarray, *, mode: str):
    """Load the REAL HuggingFace Qwen3 checkpoint, alter its actual linear projection weights in-place
    (`mode='perturb_weights'` or `mode='int4_weights'`), and execute a real full-layer BF16 forward pass
    with FlashAttention-2 + fp32 GEMM reductions to produce an authentic altered-model Capture."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask
    from flashprove_next.dense_engine import Capture

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
        L = module.layer_idx
        D = step_rec.setdefault(L, {})
        D["q"] = query[0].transpose(0, 1).detach()
        D["k"] = key[0, :, -q_len:, :].transpose(0, 1).detach()
        D["v"] = value[0, :, -q_len:, :].transpose(0, 1).detach()
        D["o"] = out[0].reshape(q_len, -1).detach()
        D["lse"] = r[1][0].transpose(0, 1).detach()
        return out, w

    impl_name = f"fpn_cheat_cap_{mode}"
    AttentionInterface.register(impl_name, cap_attn)
    AttentionMaskInterface.register(impl_name, sdpa_mask)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=impl_name
    ).cuda().eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    altered_tensors = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith(".weight") or param.ndim != 2 or "layers." not in name:
                continue
            if mode == "perturb_weights":
                # Substitute model weights with a deterministic 5% scaled + biased perturbation
                param.mul_(1.05).add_(0.02)
                altered_tensors += 1
            elif mode == "int4_weights":
                # Real symmetric per-row group-32 int4 quantization of every Transformer linear weight
                w = param.float()
                orig = w.shape
                g = 32
                pad = (-orig[1]) % g
                if pad:
                    w = torch.nn.functional.pad(w, (0, pad))
                wg = w.reshape(orig[0], -1, g)
                amax = wg.abs().amax(dim=-1, keepdim=True)
                scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
                wq = torch.clamp(torch.round(wg / scale), -7.0, 7.0) * scale
                param.copy_(wq.reshape(orig[0], -1)[:, :orig[1]].to(torch.bfloat16))
                altered_tensors += 1
            else:
                raise ValueError(f"unknown altered-model mode {mode!r}")

    assert altered_tensors > 0, f"expected >0 altered linear weights for mode={mode}"

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

    def u16(t):
        return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)

    def u32(t):
        return t.to(torch.float32).contiguous().view(torch.int32).cpu().numpy().view(np.uint32)

    L_real = int(model.config.num_hidden_layers)
    tok_t = torch.tensor([tokens.tolist()], dtype=torch.long, device="cuda")
    with torch.no_grad():
        _ = model(input_ids=tok_t, use_cache=False)
        torch.cuda.synchronize()

    cap = Capture(
        tokens=tokens.astype(np.int64).copy(),
        h_in=np.stack([u16(step_rec[L]["h_in"]) for L in range(L_real)], axis=1),
        x_attn=np.stack([u16(step_rec[L]["x_attn"]) for L in range(L_real)], axis=1),
        q=np.stack([u16(step_rec[L]["q"]) for L in range(L_real)], axis=1),
        k=np.stack([u16(step_rec[L]["k"]) for L in range(L_real)], axis=1),
        v=np.stack([u16(step_rec[L]["v"]) for L in range(L_real)], axis=1),
        o=np.stack([u16(step_rec[L]["o"]) for L in range(L_real)], axis=1),
        h_mid=np.stack([u16(step_rec[L]["h_mid"]) for L in range(L_real)], axis=1),
        x_mlp=np.stack([u16(step_rec[L]["x_mlp"]) for L in range(L_real)], axis=1),
        a=np.stack([u16(step_rec[L]["a"]) for L in range(L_real)], axis=1),
        h_out=np.stack([u16(step_rec[L]["h_out"]) for L in range(L_real)], axis=1),
        x_fn=u16(g_rec["x_fn"]),
        logits=np.zeros((0, 0), dtype=np.uint32),
        lse=np.stack([u32(step_rec[L]["lse"]) for L in range(L_real)], axis=1),
    )

    for h in handles:
        h.remove()
    del model, step_rec, g_rec, tok_t
    torch.cuda.empty_cache()
    gc.collect()
    return cap, altered_tensors


def _clone_capture(cap):
    from flashprove_next.dense_engine import Capture
    return Capture(
        tokens=cap.tokens.copy(),
        h_in=cap.h_in.copy(), x_attn=cap.x_attn.copy(),
        q=cap.q.copy(), k=cap.k.copy(), v=cap.v.copy(), o=cap.o.copy(),
        h_mid=cap.h_mid.copy(), x_mlp=cap.x_mlp.copy(), a=cap.a.copy(), h_out=cap.h_out.copy(),
        x_fn=cap.x_fn.copy(), logits=cap.logits.copy(), lse=cap.lse.copy(),
    )


def _prove_and_verify_case(prf, lazy_t, cap, com, api, cap_meta, B, *,
                           tokens_override=None, c_ext_bytes=None, force_cheat=False,
                           queries=100, gkr_masks=16):
    import cupy as cp
    from flashprove_next import reply_zk as RZ, zk_flat as Z
    R_root, wit = RZ.make_wit(
        prf, lazy_t, cap, com,
        tokens=cap.tokens if tokens_override is None else tokens_override,
        sys_len=len(cap_meta["sys_tokens"]), B=B,
    )
    wit["hs_wit"]["release_h"] = True
    stp, stw, _ = RZ.turn_state(
        prf, com, api, list(cap_meta["cot_tokens"]), NONCE_C, NONCE_S, STATE_R
    )
    wit["st"] = stw
    stmt = RZ.statement(prf, com, R_root, api, stp, c_ext=c_ext_bytes)
    exp_c_pre = bytes.fromhex(stmt["c_pre"]) if stmt.get("c_pre") else None
    exp_c_ext = bytes.fromhex(stmt["c_ext"]) if stmt.get("c_ext") else None

    old_force = Z.FORCE
    Z.FORCE = bool(force_cheat)
    t0 = time.perf_counter()
    try:
        pf, info = RZ.prove(prf, stmt, wit, B=B, backend="folded", queries=queries, gkr_masks=gkr_masks)
    finally:
        Z.FORCE = old_force
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    prove_s = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    verdict = RZ.try_verify(prf, stmt, pf, expected_c_pre=exp_c_pre, expected_c_ext=exp_c_ext)
    verify_s = round(time.perf_counter() - t1, 3)
    return stmt, pf, info, verdict, prove_s, verify_s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--m-c", type=int, default=8)
    ap.add_argument("--m-g", type=int, default=4)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--gkr-masks", type=int, default=16)
    ap.add_argument("--pad-to-bytes", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args(argv)

    from flashprove_next import modelcommit as MC, reply_bundle as RB, reply_zk as RZ
    from flashprove_next.dense_engine import Capture
    from flashprove_next.field import gpu_backend

    os.environ.setdefault("FPN_OOC", "1")
    B = gpu_backend()
    assert B.name == "cupy", f"expected cupy GPU backend, got {B.name}"

    ck = MC.Checkpoint(a.model)
    mc_prm = _select_mc_params(ck.config)
    lazy_t = RZ.LazyCommitTensors(ck, mc_prm.TI)
    setup_meta = json.load(open(os.path.join(a.store, "setup_meta.json")))
    cap_d = np.load(a.cap)
    cap_meta = json.loads(bytes(cap_d["meta"]).decode())
    cap_honest = Capture(
        tokens=cap_d["tokens"].copy(),
        h_in=cap_d["h_in"].copy(), x_attn=cap_d["x_attn"].copy(),
        q=cap_d["q"].copy(), k=cap_d["k"].copy(), v=cap_d["v"].copy(), o=cap_d["o"].copy(),
        h_mid=cap_d["h_mid"].copy(), x_mlp=cap_d["x_mlp"].copy(), a=cap_d["a"].copy(), h_out=cap_d["h_out"].copy(),
        x_fn=cap_d["x_fn"].copy(), logits=np.zeros((0, 0), dtype=np.uint32), lse=cap_d["lse"].copy(),
    )
    cap_d.close()

    # Generate REAL altered-model forward captures BEFORE cupy allocates heavy GPU pools!
    print("[prep] Running real 28-layer forward pass on weight-perturbed model...", flush=True)
    cap_wrong_w, n_pert = _capture_altered_model_forward(a.model, cap_honest.tokens, mode="perturb_weights")
    print(f"[prep] Weight-perturbed capture done ({n_pert} linear weight tensors altered).", flush=True)

    print("[prep] Running real 28-layer forward pass on group-32 int4-quantized model...", flush=True)
    cap_int4, n_int4 = _capture_altered_model_forward(a.model, cap_honest.tokens, mode="int4_weights")
    print(f"[prep] Int4-quantized capture done ({n_int4} linear weight tensors quantized to int4).", flush=True)

    prf = make_qwen_profile(
        ck.config, setup_meta["vec_pad"], cap_meta["t_pad"], a.m_c, a.m_g, sys_cap=cap_meta["sys_cap"], prm=mc_prm
    )
    lay = MC.Layout(lazy_t.shapes(), prm=mc_prm, radius=True, cfg_rows=True)
    store = MC.Store(a.store, mc_prm, tag="W")
    com = {
        "layout": lay,
        "store": store,
        "pre": {k: tuple(v) if isinstance(v, list) else v for k, v in setup_meta["pre"].items()},
        "mid": tuple(setup_meta["mid"]),
        "vol": np.asarray(setup_meta["vol"], dtype=np.uint64),
    }
    api = {
        "req_tokens": list(cap_meta["req_tokens"]),
        "n_reason": len(cap_meta["cot_tokens"]),
        "vis_tokens": list(cap_meta["vis_tokens"]),
    }
    c_ext = bytes.fromhex("20261124" * 8)
    prof_name = f"{os.path.basename(os.path.normpath(a.model)).lower()}-cheat-matrix"
    spec = RB.profile_spec_from_hf_config(
        ck.config,
        name=prof_name,
        vec_pad=setup_meta["vec_pad"], t_pad=cap_meta["t_pad"],
        m_c=int(a.m_c), m_g=int(a.m_g), sys_cap=cap_meta["sys_cap"],
    )
    if prof_name not in RB.REGISTERED_PROFILES:
        RB.register_profile(spec)

    results = []

    def record(name, expected, verdict, extra=None, *, expect_codes=None):
        if expected == "ACCEPT":
            ok = (verdict == "accepted")
        else:
            ok = (verdict != "accepted") and (
                expect_codes is None or any(c in verdict for c in expect_codes)
            )
        row = {"case": name, "expected": expected, "verdict": verdict, "pass": bool(ok)}
        if expect_codes is not None:
            row["expected_codes"] = list(expect_codes)
        if extra:
            row.update(extra)
        results.append(row)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} expected={expected:6s} got={verdict}", flush=True)
        return ok

    # 1. Honest proof in exact 8 MiB zero-padded FPNZK2 container (verified BOTH in-memory AND via serialized verify_cli)
    stmt_h, pf_h, info_h, v_h, tp_h, tv_h = _prove_and_verify_case(
        prf, lazy_t, cap_honest, com, api, cap_meta, B, c_ext_bytes=c_ext,
        force_cheat=False, queries=a.queries, gkr_masks=a.gkr_masks,
    )
    blob_h = RB.pack_bundle(spec, stmt_h, pf_h, pad_to_bytes=a.pad_to_bytes, format="v2")
    assert len(blob_h) == a.pad_to_bytes
    exp_c_pre_h = bytes.fromhex(stmt_h["c_pre"])
    body_len_h = int.from_bytes(blob_h[15:23], "little")
    mid_b = bytes.fromhex(RB.mid_to_hex(com["mid"]))
    ok_cli_h, why_cli_h, _ = RB.verify_cli(
        mid_b, {"tokens": api["req_tokens"]}, {"tokens": api["vis_tokens"], "n_reason": api["n_reason"]},
        NONCE_C.hex(), blob_h, allowed_profiles=[prof_name], allow_reduced_security=True,
        expected_c_pre=exp_c_pre_h, expected_c_ext=c_ext,
    )
    v_h_full = "accepted" if (v_h == "accepted" and ok_cli_h) else f"in_mem={v_h}; verify_cli={why_cli_h}"
    record("1_honest_extseed_fpnzk2", "ACCEPT", v_h_full, {
        "prove_s": tp_h, "verify_s": tv_h, "cells": info_h["cells"],
        "phase2_cells": info_h["phase2_cells"], "container_bytes": len(blob_h),
        "body_bytes": body_len_h, "verify_cli_accepted": bool(ok_cli_h),
    })

    # 2. Real substituted/perturbed model weights (full 28-layer PyTorch forward pass with altered weights)
    _, _, _, v_ww, tp_ww, tv_ww = _prove_and_verify_case(
        prf, lazy_t, cap_wrong_w, com, api, cap_meta, B, c_ext_bytes=c_ext,
        force_cheat=True, queries=a.queries, gkr_masks=a.gkr_masks,
    )
    record("2_real_perturbed_weights_model", "REJECT", v_ww, {
        "prove_s": tp_ww, "verify_s": tv_ww, "altered_linear_tensors": n_pert,
    }, expect_codes=("E_SUMCHECK", "E_GKR_ROOT", "E_GKR"))

    # 3. Real symmetric group-32 int4-quantized model weights (full 28-layer PyTorch forward pass)
    _, _, _, v_i4, tp_i4, tv_i4 = _prove_and_verify_case(
        prf, lazy_t, cap_int4, com, api, cap_meta, B, c_ext_bytes=c_ext,
        force_cheat=True, queries=a.queries, gkr_masks=a.gkr_masks,
    )
    record("3_real_int4_quantized_model", "REJECT", v_i4, {
        "prove_s": tp_i4, "verify_s": tv_i4, "quantized_linear_tensors": n_int4,
    }, expect_codes=("E_SUMCHECK", "E_GKR_ROOT", "E_GKR"))

    # 4. Skipped transformer layers (copy h_in -> h_out across all layers)
    cap_skip = _clone_capture(cap_honest)
    L_n = cap_skip.h_in.shape[1]
    for l_idx in range(L_n):
        cap_skip.h_out[:, l_idx, :] = cap_skip.h_in[:, l_idx, :]
    _, _, _, v_sk, tp_sk, tv_sk = _prove_and_verify_case(
        prf, lazy_t, cap_skip, com, api, cap_meta, B, c_ext_bytes=c_ext,
        force_cheat=True, queries=a.queries, gkr_masks=a.gkr_masks,
    )
    record("4_skipped_layers", "REJECT", v_sk, {"prove_s": tp_sk, "verify_s": tv_sk},
           expect_codes=("E_SUMCHECK", "E_GKR_ROOT", "E_GKR"))

    # 5. Truncated KV context (zero out early KV rows)
    cap_trunc = _clone_capture(cap_honest)
    half_t = max(1, cap_trunc.k.shape[0] // 2)
    cap_trunc.k[:half_t] = 0
    cap_trunc.v[:half_t] = 0
    _, _, _, v_tr, tp_tr, tv_tr = _prove_and_verify_case(
        prf, lazy_t, cap_trunc, com, api, cap_meta, B, c_ext_bytes=c_ext,
        force_cheat=True, queries=a.queries, gkr_masks=a.gkr_masks,
    )
    record("5_truncated_kv_context", "REJECT", v_tr, {"prove_s": tp_tr, "verify_s": tv_tr},
           expect_codes=("E_SUMCHECK", "E_GKR_ROOT", "E_GKR"))

    # 6. Tampered hidden system prompt
    bad_toks = cap_honest.tokens.copy()
    bad_toks[0] = (int(bad_toks[0]) + 17) % int(ck.config["vocab_size"])
    _, _, _, v_sys, tp_sys, tv_sys = _prove_and_verify_case(
        prf, lazy_t, cap_honest, com, api, cap_meta, B, tokens_override=bad_toks,
        c_ext_bytes=c_ext, force_cheat=True, queries=a.queries, gkr_masks=a.gkr_masks,
    )
    record("6_tampered_hidden_sys_prompt", "REJECT", v_sys, {"prove_s": tp_sys, "verify_s": tv_sys},
           expect_codes=("E_SUMCHECK", "E_GKR_ROOT", "E_GKR"))

    # 7. Tampered visible reply token
    stmt_bad_vis = copy.deepcopy(stmt_h)
    stmt_bad_vis["vis_tokens"][0] = (int(stmt_bad_vis["vis_tokens"][0]) + 1) % int(ck.config["vocab_size"])
    v_vis = RZ.try_verify(prf, stmt_bad_vis, pf_h, expected_c_pre=exp_c_pre_h, expected_c_ext=c_ext)
    record("7_tampered_visible_reply", "REJECT", v_vis, expect_codes=("E_CPRE_MISMATCH",))

    # 8. Replay / tampered St_prev
    stmt_bad_st = copy.deepcopy(stmt_h)
    stmt_bad_st["st"]["St_prev"][0] = int(stmt_bad_st["st"]["St_prev"][0]) ^ 1
    v_st = RZ.try_verify(prf, stmt_bad_st, pf_h, expected_c_pre=exp_c_pre_h, expected_c_ext=c_ext)
    record("8_replay_tampered_st_prev", "REJECT", v_st, expect_codes=("E_CPRE_MISMATCH",))

    # 9a. Missing external-seed precommitment pins -> E_CPRE_UNPINNED
    v_unpin = RZ.try_verify(prf, stmt_h, pf_h)
    record("9a_missing_cpre_pin", "REJECT", v_unpin, expect_codes=("E_CPRE_UNPINNED",))

    # 9b. Swapped C_pre pin
    v_swap_cpre = RZ.try_verify(prf, stmt_h, pf_h, expected_c_pre=bytes([exp_c_pre_h[0] ^ 1]) + exp_c_pre_h[1:],
                                expected_c_ext=c_ext)
    record("9b_swapped_cpre_pin", "REJECT", v_swap_cpre, expect_codes=("E_CPRE_MISMATCH",))

    # 9c. Swapped c_ext pin
    v_swap_cext = RZ.try_verify(prf, stmt_h, pf_h, expected_c_pre=exp_c_pre_h,
                                expected_c_ext=bytes([c_ext[0] ^ 1]) + c_ext[1:])
    record("9c_swapped_cext_pin", "REJECT", v_swap_cext, expect_codes=("E_CEXT_MISMATCH",))

    # 9d. Non-zero FPNZK2 padding byte -> E_PADDING
    bad_pad = bytearray(blob_h)
    bad_pad[-1] ^= 1
    try:
        RB.unpack_bundle(bytes(bad_pad), expected_container_bytes=a.pad_to_bytes)
        v_pad = "accepted"
    except ValueError as e:
        v_pad = f"rejected: {e}"
    record("9d_fpnzk2_nonzero_padding", "REJECT", v_pad, expect_codes=("E_PADDING",))

    # 9e. Downgraded queries=4 proof under default production verifier -> E_REDUCED_SECURITY
    _, pf_weak, _, _, _, _ = _prove_and_verify_case(
        prf, lazy_t, cap_honest, com, api, cap_meta, B, c_ext_bytes=c_ext,
        force_cheat=False, queries=4, gkr_masks=4,
    )
    v_weak = RZ.try_verify(prf, stmt_h, pf_weak, expected_c_pre=exp_c_pre_h, expected_c_ext=c_ext)
    record("9e_downgraded_queries_4", "REJECT", v_weak, expect_codes=("E_REDUCED_SECURITY",))

    # 9f. Self-consistent FPNZK2 envelope with wrong size (4 MiB != 8 MiB trusted profile) -> E_LENGTH in verify_cli
    wrong_sz = max(body_len_h + 4096, 4 * 1024 * 1024)
    assert wrong_sz != a.pad_to_bytes
    blob_wrong_sz = RB.pack_bundle(spec, stmt_h, pf_h, pad_to_bytes=wrong_sz, format="v2")
    ok_sz, why_sz, _ = RB.verify_cli(
        mid_b, {"tokens": api["req_tokens"]}, {"tokens": api["vis_tokens"], "n_reason": api["n_reason"]},
        NONCE_C.hex(), blob_wrong_sz, allowed_profiles=[prof_name], allow_reduced_security=True,
        expected_c_pre=exp_c_pre_h, expected_c_ext=c_ext,
    )
    record("9f_fpnzk2_wrong_envelope_size", "REJECT", "accepted" if ok_sz else f"rejected: {why_sz}",
           expect_codes=("E_LENGTH",))

    all_pass = all(r["pass"] for r in results)
    report = {
        "bench": "v1_cheating_matrix",
        "model": os.path.basename(os.path.normpath(a.model)),
        "machine": describe_machine(),
        "M_C": int(a.m_c),
        "M_G": int(a.m_g),
        "queries": int(a.queries),
        "gkr_masks": int(a.gkr_masks),
        "all_pass": all_pass,
        "n_cases": len(results),
        "n_honest": 1,
        "n_negative": len(results) - 1,
        "n_pass": sum(1 for r in results if r["pass"]),
        "cases": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
    with open(a.json_out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nSUMMARY: {report['n_pass']}/{report['n_cases']} PASSED (1 honest + {report['n_negative']} negative, all_pass={all_pass})", flush=True)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
