# SPDX-License-Identifier: Apache-2.0
"""API-semantics tests with white-box forgeries for `flashprove_next.api_zk` and `flashprove_next.reply_zk`:
  1. Public pre-check Phi_pre (custom string stop refusal, logprobs, sampling overrides, multimodal, bad schema)
  2. finish_reason = 'stop' (EOS) + white-box forgeries (missing EOS, premature EOS hidden inside CoT)
  3. finish_reason = 'length' (max_tokens truncation + max_tokens = 0 empty generation boundary)
  4. Multi-turn state chain (Turn 0 -> AEAD fixed-length blob -> Turn 1 CoT re-injection + server CoT-tamper forgery)
  5. Tool calls (finish_reason = 'tool_use' + multi-segment G^(1) -> committed z_tool witness -> G^(2) + z_tool forgery)
  6. Constrained decoding (strict JSON Schema + mid-JSON C7 truncation + white-box wrong-mask sampler forgery)
  7. Profile pinning and client-side request cross-checks of the proof bundle verifier

CPU by default; `--backend cupy` proves on the GPU.

usage:  python tests/test_api_zk.py [--backend cpu|cupy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flashprove_next import api_zk as APZ  # noqa: E402
from flashprove_next import modelcommit as MC
from flashprove_next import reply_zk as RZ
from flashprove_next import sampling_v3 as S3
from flashprove_next import state_zk as SZ
from flashprove_next import zk_flat as Z
from flashprove_next.dense_engine import TinyDense
from flashprove_next.field import CPU
from flashprove_next.transcript import dumps

SYS = [5, 6]
REQ = [7, 8]
NC0 = b"\x11" * 16
NS0 = b"\x22" * 32
NC1 = b"\x33" * 16
NS1 = b"\x44" * 32
R_ST0 = (3, 5, 7, 11)
R_ST1 = (13, 17, 19, 23)
EOS_ID = 126
TOOL_END_ID = 127
KPRF = [0x1234567, 0x89ABCDEF, 0x0F1E2D3C, 0x4B5A6978]


def _sctx_for(req_tokens, nc, ns, sess, turn=0):
    v_rel = SZ.v_rel_of(RZ.VERSION)
    dig = hashlib.blake2b(
        json.dumps(list(req_tokens), separators=(",", ":")).encode(),
        digest_size=32, person=b"fpn-req-dig",
    ).digest()
    sess_b = b"".join(int(x).to_bytes(8, "little") for x in sess)
    return S3.SampCtx(nc, ns, dig=dig, v_rel=int(v_rel) & 0xFFFFFFFF, epoch=0, sess=sess_b, turn=int(turn))


def _rollout(model, prefix_ids, n_steps, pre, rule, g_offset=0, allowed_fn=None, stop_ids=()):
    seq = list(prefix_ids)
    emitted = []
    for k in range(n_steps):
        cap_k = model.forward(seq)
        lg = np.asarray(cap_k.logits[-1]).view(np.float32).copy()
        allow = allowed_fn(k, emitted) if allowed_fn is not None else None
        tok = int(S3.sample(pre, g_offset + k, lg, rule, allowed=allow))
        seq.append(tok)
        emitted.append(tok)
        if tok in stop_ids:
            break
    return seq, emitted


def _run_case(name, prf, stmt, wit, reject, B):
    Z.FORCE = reject
    try:
        pf, info = RZ.prove(prf, stmt, wit, B=B)
    except Z.ProofError as e:
        Z.FORCE = False
        if reject:
            print(f"PASS {name}: rejected pre-prove ({e})", flush=True)
            return True
        print(f"FAIL {name}: unexpected ProofError ({e})", flush=True)
        return False
    finally:
        Z.FORCE = False
    t0 = time.perf_counter()
    res = RZ.try_verify(prf, stmt, pf)
    tv = time.perf_counter() - t0
    ok = res.startswith("rejected") if reject else (res == "accepted")
    print(f"{'PASS' if ok else 'FAIL'} {name}: {res} (prove {info['prove_s']}s, verify {tv:.2f}s, "
          f"cells {info['cells']}, fails {info['gadget_fails']}, notes {info['violations'][:2]})", flush=True)
    return ok


def test_policy_and_dfa():
    rule = S3.Rule(0.6, 0.95, 32)
    # Custom string stop sequences are refused deterministically (specs/05 §4.6)
    ok, why = APZ.check_public_policy({"messages": [{"role": "user", "content": "hi"}], "stop": ["END"]}, rule)
    assert not ok and "custom_stop_sequence" in why, (ok, why)
    ok, why = APZ.check_public_policy({"messages": [{"role": "user", "content": "hi"}], "stop_sequences": ["\n"]}, rule)
    assert not ok and "custom_stop_sequence" in why
    # logprobs, sampling overrides, multimodal, unsupported schema keywords
    assert not APZ.check_public_policy({"prompt": "hi", "logprobs": True}, rule)[0]
    assert not APZ.check_public_policy({"prompt": "hi", "temperature": 0.1}, rule)[0]
    assert not APZ.check_public_policy({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, rule)[0]
    bad_sch = {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
    assert not APZ.check_public_policy({"response_format": {"type": "json_schema", "json_schema": {"schema": bad_sch, "strict": True}}}, rule)[0]

    # Valid strict JSON Schema & strict tool schema
    good_sch = {
        "type": "object",
        "properties": {"ans": {"type": "integer"}, "ok": {"type": "boolean"}},
        "required": ["ans", "ok"],
        "additionalProperties": False,
    }
    ok, why = APZ.check_public_policy({
        "messages": [{"role": "user", "content": "1+1"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": good_sch, "strict": True}},
    }, rule)
    assert ok and why == "ok", (ok, why)
    dfa = APZ.compile_json_schema_dfa(good_sch)
    assert dfa.accepts_bytes(b'{"ans":42,"ok":true}')
    assert dfa.accepts_bytes(b'{"ans":-7,"ok":false}')
    assert not dfa.accepts_bytes(b'{"ok":true,"ans":42}')        # wrong key order rejected
    assert not dfa.accepts_bytes(b'{"ans":42,"ok":true,"x":1}')  # extra key rejected
    assert dfa.is_extensible_prefix(b'{"ans":4')
    assert not dfa.is_extensible_prefix(b'{"ans":true')
    print(f"PASS policy_and_dfa (states={dfa.n_states})", flush=True)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "cupy"), default="cpu")
    args = ap.parse_args(argv)
    B = CPU
    if args.backend == "cupy":
        from flashprove_next.field import gpu_backend
        B = gpu_backend()

    fails = []
    if not test_policy_and_dfa():
        fails.append("policy_and_dfa")

    V = 128
    # Build deterministic byte vocabulary for constrained decoding tests:
    # tokens 0..95 map to printable ASCII chars chr(32 + i), plus multi-byte JSON tokens, 126 = EOS, 127 = TOOL_END.
    token_bytes: list[bytes | None] = [None] * V
    for i in range(95):
        token_bytes[i] = bytes([32 + i])
    multi = [b'{"', b'":', b',"', b"true", b"false", b"null", b'{"ans":', b',"ok":', b"true}", b"false}"]
    for k, mb in enumerate(multi):
        token_bytes[95 + k] = mb
    for k in range(95 + len(multi), 126):
        token_bytes[k] = f"{k - 105}".encode("ascii")
    token_bytes[EOS_ID] = None
    token_bytes[TOOL_END_ID] = None

    model = TinyDense(d=128, n_q=4, n_kv=2, dh=32, dff=256, L=2, V=V, seed=3)
    prm = MC.Params(TO=64, TI=64, R_MAX=512, RATE_LOG=2, MQ=8, GROUP=8)
    rule = S3.Rule(0.6, 0.95, 32)
    prf = RZ.Profile(
        d=model.d, n_q=model.n_q, n_kv=model.n_kv, dh=model.dh, dff=model.dff,
        L_MAX=4, T_pad=32, M_C=2, M_G=2, V=V, prm=prm, s_bits=2, HIST_CAP=16,
        TOOL_CAP=4, rule=rule, token_bytes=token_bytes,
    )
    com = RZ.commit_model(model.tensors(), prf, L_real=model.L, V_real=V, sys_prompt=SYS, k_prf=KPRF)
    sess = SZ.sess_of(NC0)

    # ------------------------------------------------------------------ 2. finish_reason = 'stop' + EOS forgeries
    sctx0 = _sctx_for(REQ, NC0, NS0, sess, turn=0)
    pre0 = S3.prefix_state(KPRF, sctx0)
    # finish_reason 'stop' needs a genuine sampled EOS: model_eos gives EOS_ID the logit row of token 0, so the keyed
    # sampler emits it within a few steps for some client nonce (searched below).
    model_eos = TinyDense(d=128, n_q=4, n_kv=2, dh=32, dff=256, L=2, V=V, seed=3)
    model_eos._tensors["lm_head.weight"][EOS_ID] = model_eos._tensors["lm_head.weight"][0]
    com_eos = RZ.commit_model(model_eos.tensors(), prf, L_real=model_eos.L, V_real=V, sys_prompt=SYS, k_prf=KPRF)

    # Find a nonce where unconstrained rollout emits >= 2 non-EOS tokens then EOS_ID
    found_eos = None
    for seed_i in range(500):
        nc_try = seed_i.to_bytes(16, "little")
        sess_try = SZ.sess_of(nc_try)
        pre_try = S3.prefix_state(KPRF, _sctx_for(REQ, nc_try, NS0, sess_try, 0))
        seq_try, em_try = _rollout(
            model_eos, SYS + REQ, 8, pre_try, rule,
            stop_ids=(EOS_ID, TOOL_END_ID),
        )
        if len(em_try) >= 2 and em_try[-1] == EOS_ID and TOOL_END_ID not in em_try:
            found_eos = (nc_try, sess_try, seq_try, em_try)
            break
    # If random init of TinyDense has uniform 1/128 per step (~98% chance within 500*8=4000 steps), assert found
    assert found_eos is not None, "expected natural EOS sample within 500 nonces"
    nc_e, sess_e, seq_e, em_e = found_eos
    cot_e, vis_e = em_e[:1], em_e[1:]
    cap_e = model_eos.forward(seq_e)
    api_e = {
        "req_tokens": REQ, "n_reason": len(cot_e), "vis_tokens": vis_e,
        "api_meta": {"finish_reason": "stop", "max_tokens": 12, "eos_token_id": EOS_ID, "tool_end_token_id": TOOL_END_ID},
    }
    R_e, wit_e = RZ.make_wit(prf, model_eos, cap_e, com_eos, sys_len=len(SYS), B=B)
    stp_e, stw_e, _ = RZ.turn_state(prf, com_eos, api_e, cot_e, nc_e, NS0, R_ST0)
    wit_e["st"] = stw_e
    stmt_e = RZ.statement(prf, com_eos, R_e, api_e, stp_e)
    if not _run_case("finish_stop_honest", prf, stmt_e, wit_e, False, B):
        fails.append("finish_stop_honest")

    # White-box forgery 2a: truncate before EOS (vis lacks EOS) while claiming finish_reason = 'stop'
    seq_noeos = seq_e[:-1]
    em_noeos = em_e[:-1]
    cap_ne = model_eos.forward(seq_noeos)
    api_ne = {
        "req_tokens": REQ, "n_reason": 0, "vis_tokens": em_noeos,
        "api_meta": {"finish_reason": "stop", "max_tokens": 12, "eos_token_id": EOS_ID, "tool_end_token_id": TOOL_END_ID},
    }
    R_ne, wit_ne = RZ.make_wit(prf, model_eos, cap_ne, com_eos, sys_len=len(SYS), B=B)
    stp_ne, stw_ne, _ = RZ.turn_state(prf, com_eos, api_ne, [], nc_e, NS0, R_ST0)
    wit_ne["st"] = stw_ne
    stmt_ne = RZ.statement(prf, com_eos, R_ne, api_ne, stp_ne)
    if not _run_case("forged_finish_stop_without_eos", prf, stmt_ne, wit_ne, True, B):
        fails.append("forged_finish_stop_without_eos")

    # White-box forgery 2b: premature EOS hidden inside CoT (model emitted EOS in hidden CoT, prover kept generating)
    # Append one extra sampled token AFTER seq_e's EOS and hide the EOS inside `cot` (`n_reason = len(em_e)`).
    pre_e = S3.prefix_state(KPRF, _sctx_for(REQ, nc_e, NS0, sess_e, 0))
    seq_post, extra_tok = _rollout(model_eos, seq_e, 2, pre_e, rule, g_offset=len(em_e))
    # Force final token to be EOS or claim 'length' so ONLY the premature EOS inside hidden CoT triggers `rz.term.noeos`.
    cot_prem = list(em_e)                                # ends with EOS_ID inside hidden CoT.
    vis_prem = [extra_tok[0]] if extra_tok[0] != EOS_ID else [10]
    seq_prem = SYS + REQ + cot_prem + vis_prem
    cap_prem = model_eos.forward(seq_prem)
    api_prem = {
        "req_tokens": REQ, "n_reason": len(cot_prem), "vis_tokens": vis_prem,
        "api_meta": {"finish_reason": "length", "max_tokens": len(cot_prem) + len(vis_prem),
                     "eos_token_id": EOS_ID, "tool_end_token_id": TOOL_END_ID},
    }
    R_pr, wit_pr = RZ.make_wit(prf, model_eos, cap_prem, com_eos, sys_len=len(SYS), B=B)
    stp_pr, stw_pr, _ = RZ.turn_state(prf, com_eos, api_prem, cot_prem, nc_e, NS0, R_ST0)
    wit_pr["st"] = stw_pr
    stmt_pr = RZ.statement(prf, com_eos, R_pr, api_prem, stp_pr)
    if not _run_case("forged_premature_eos_in_hidden_cot", prf, stmt_pr, wit_pr, True, B):
        fails.append("forged_premature_eos_in_hidden_cot")

    # ------------------------------------------------------------------ 3. finish_reason = 'length' & max_tokens = 0
    seq_len, em_len = _rollout(model, SYS + REQ, 3, pre0, rule)
    assert EOS_ID not in em_len and TOOL_END_ID not in em_len
    cap_len = model.forward(seq_len)
    api_len = {
        "req_tokens": REQ, "n_reason": 1, "vis_tokens": em_len[1:],
        "api_meta": {"finish_reason": "length", "max_tokens": 3, "eos_token_id": EOS_ID, "tool_end_token_id": TOOL_END_ID},
    }
    R_l, wit_l = RZ.make_wit(prf, model, cap_len, com, sys_len=len(SYS), B=B)
    stp_l, stw_l, nxt_turn0 = RZ.turn_state(prf, com, api_len, em_len[:1], NC0, NS0, R_ST0)
    wit_l["st"] = stw_l
    stmt_l = RZ.statement(prf, com, R_l, api_len, stp_l)
    if not _run_case("finish_length_honest", prf, stmt_l, wit_l, False, B):
        fails.append("finish_length_honest")

    # Boundary: max_tokens = 0 (empty generation N_gen = 0, finish_reason = 'length')
    cap_z = model.forward(SYS + REQ)
    api_z = {
        "req_tokens": REQ, "n_reason": 0, "vis_tokens": [],
        "api_meta": {"finish_reason": "length", "max_tokens": 0, "eos_token_id": EOS_ID, "tool_end_token_id": TOOL_END_ID},
    }
    R_z, wit_z = RZ.make_wit(prf, model, cap_z, com, sys_len=len(SYS), B=B)
    stp_z, stw_z, _ = RZ.turn_state(prf, com, api_z, [], NC0, NS0, R_ST0)
    wit_z["st"] = stw_z
    stmt_z = RZ.statement(prf, com, R_z, api_z, stp_z)
    if not _run_case("max_tokens_zero_honest", prf, stmt_z, wit_z, False, B):
        fails.append("max_tokens_zero_honest")

    # ------------------------------------------------------------------ 4. Multi-turn state chain + fixed-length AEAD blob
    wrap_key = b"\x55" * 32
    blob0 = APZ.seal_state_blob(wrap_key, com["mid"], sess, 0, stp_l["St_new"], nxt_turn0, prf.HIST_CAP)
    # Verify fixed-length padding: sealing a 10-token history produces the EXACT same hex length as 1-token history.
    blob_long = APZ.seal_state_blob(wrap_key, com["mid"], sess, 0, stp_l["St_new"],
                                    {**nxt_turn0, "hist": [1] * 10}, prf.HIST_CAP)
    assert len(blob0) == len(blob_long), (len(blob0), len(blob_long))
    opened0 = APZ.open_state_blob(wrap_key, blob0, com["mid"], sess, 0, stp_l["St_new"], prf.HIST_CAP)
    assert opened0["hist"] == em_len[:1]

    req_t1 = REQ + em_len[1:] + [21, 22]
    sctx1 = _sctx_for(req_t1, NC1, NS1, sess, turn=1)
    pre1 = S3.prefix_state(KPRF, sctx1)
    # Turn 1 prefix re-injects opened0['hist'] right after SYS.
    seq_t1, em_t1 = _rollout(model, SYS + opened0["hist"] + req_t1, 3, pre1, rule)
    cap_t1 = model.forward(seq_t1)
    api_t1 = {
        "req_tokens": req_t1, "n_reason": 1, "vis_tokens": em_t1[1:],
        "api_meta": {"finish_reason": "length", "max_tokens": 3, "eos_token_id": EOS_ID},
    }
    R_t1, wit_t1 = RZ.make_wit(prf, model, cap_t1, com, sys_len=len(SYS), B=B)
    stp_t1, stw_t1, _ = RZ.turn_state(
        prf, com, api_t1, em_t1[:1], NC1, NS1, R_ST1,
        turn=1, sess=sess, st_prev=stp_l["St_new"], prev_bundle=opened0,
    )
    wit_t1["st"] = stw_t1
    stmt_t1 = RZ.statement(prf, com, R_t1, api_t1, stp_t1)
    if not _run_case("multiturn_turn1_honest", prf, stmt_t1, wit_t1, False, B):
        fails.append("multiturn_turn1_honest")

    # White-box forgery 4b: server tampers with Turn 0's re-injected hidden CoT in Turn 1 while keeping St_prev = St_1
    forged_hist0 = [(opened0["hist"][0] + 1) % 100]
    seq_t1_bad = SYS + forged_hist0 + req_t1 + em_t1
    cap_t1_bad = model.forward(seq_t1_bad)
    R_t1b, wit_t1b = RZ.make_wit(prf, model, cap_t1_bad, com, sys_len=len(SYS), B=B)
    _, stw_t1b, _ = RZ.turn_state(
        prf, com, api_t1, em_t1[:1], NC1, NS1, R_ST1,
        turn=1, sess=sess, st_prev=stp_l["St_new"], prev_bundle={**opened0, "hist": forged_hist0},
    )
    wit_t1b["st"] = stw_t1b
    stmt_t1b = RZ.statement(prf, com, R_t1b, api_t1, stp_t1)   # public St_prev is STILL the authentic St_1.
    if not _run_case("forged_turn1_tampered_reinjected_cot", prf, stmt_t1b, wit_t1b, True, B):
        fails.append("forged_turn1_tampered_reinjected_cot")

    # ------------------------------------------------------------------ 5. Tool call + committed z_tool witness
    # Find a nonce where step 1 naturally emits TOOL_END_ID (using allowed mask or nonce grind)
    found_tool = None
    for seed_i in range(600):
        nc_t = (seed_i + 1000).to_bytes(16, "little")
        sess_t = SZ.sess_of(nc_t)
        pre_t = S3.prefix_state(KPRF, _sctx_for(REQ, nc_t, NS0, sess_t, 0))
        seq_g1, em_g1 = _rollout(model, SYS + REQ, 6, pre_t, rule, stop_ids=(EOS_ID, TOOL_END_ID))
        if len(em_g1) >= 2 and em_g1[-1] == TOOL_END_ID and EOS_ID not in em_g1:
            found_tool = (nc_t, sess_t, pre_t, seq_g1, em_g1)
            break
    assert found_tool is not None, "expected natural TOOL_END_ID sample"
    nc_t, sess_t, pre_t, seq_g1, em_g1 = found_tool
    z_tool_ret = [40, 41, 42]   # private server-tool return witness (3 tokens <= TOOL_CAP=4).
    # Continue G^(2) after prefilling z_tool_ret (generation step counter continues at len(em_g1))
    seq_g2, em_g2 = _rollout(model, seq_g1 + z_tool_ret, 2, pre_t, rule, g_offset=len(em_g1))
    assert EOS_ID not in em_g2 and TOOL_END_ID not in em_g2
    cap_tool = model.forward(seq_g2)
    cot_seg0, call_seg0 = em_g1[:1], em_g1[1:]
    cot_seg1, vis_seg1 = em_g2[:1], em_g2[1:]
    # Private server tool return (`ret_tokens: None`): committed in rzTok + H before challenge draws.
    all_hist_tool = cot_seg0 + z_tool_ret + cot_seg1
    api_tool = {
        "req_tokens": REQ,
        "n_reason": len(cot_seg1),
        "vis_tokens": vis_seg1,
        "api_meta": {
            "finish_reason": "length",
            "max_tokens": len(em_g1) + len(em_g2),
            "eos_token_id": EOS_ID,
            "tool_end_token_id": TOOL_END_ID,
            "tool_segments": [{
                "n_reason": len(cot_seg0),
                "call_tokens": call_seg0,
                "n_ret": len(z_tool_ret),
                "ret_tokens": None,
            }],
        },
    }
    R_tl, wit_tl = RZ.make_wit(prf, model, cap_tool, com, sys_len=len(SYS), B=B)
    stp_tl, stw_tl, _ = RZ.turn_state(prf, com, api_tool, all_hist_tool, nc_t, NS0, R_ST0)
    wit_tl["st"] = stw_tl
    stmt_tl = RZ.statement(prf, com, R_tl, api_tool, stp_tl)
    if not _run_case("server_tool_committed_witness_honest", prf, stmt_tl, wit_tl, False, B):
        fails.append("server_tool_committed_witness_honest")

    # White-box forgery 5b: H ran on tampered tool return z_tool' = [99, 41, 42] while rzTok commits [40, 41, 42].
    # Caught deterministically (Pr = 1) by `tool.EMB` even if no random compute slot hits the tool-return positions.
    seq_g2_bad = seq_g1 + [99, 41, 42] + em_g2
    cap_tool_bad = model.forward(seq_g2_bad)
    R_tlb, wit_tlb = RZ.make_wit(prf, model, cap_tool_bad, com, tokens=seq_g2, sys_len=len(SYS), B=B)
    stp_tlb, stw_tlb, _ = RZ.turn_state(prf, com, api_tool, all_hist_tool, nc_t, NS0, R_ST0)
    wit_tlb["st"] = stw_tlb
    stmt_tlb = RZ.statement(prf, com, R_tlb, api_tool, stp_tlb)
    if not _run_case("forged_tool_return_unbound_to_H", prf, stmt_tlb, wit_tlb, True, B):
        fails.append("forged_tool_return_unbound_to_H")

    # ------------------------------------------------------------------ 6. Constrained decoding (strict JSON Schema)
    schema_cd = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    dfa_cd = APZ.compile_json_schema_dfa(schema_cd)
    next_q_cd, allow_cd = dfa_cd.compile_token_masks(token_bytes, (EOS_ID,))

    # Step 0: unconstrained CoT token; Steps 1..: constrained by `allow_cd[q]` until EOS_ID.
    q_box = [0]

    def cd_mask_fn(step_k, emitted_so_far):
        if step_k == 0:
            # Avoid EOS / TOOL_END during the 1-token CoT
            m = np.ones(V, dtype=bool)
            m[EOS_ID] = False; m[TOOL_END_ID] = False
            return m
        if step_k > 1:
            q_box[0] = int(next_q_cd[q_box[0], emitted_so_far[-1]])
        return allow_cd[q_box[0], :V]

    # Note: step 0 in our rollout uses unconstrained `ones(V)` so it matches `act = 0` in `reply_zk`.
    nc_cd = (777).to_bytes(16, "little")
    sess_cd = SZ.sess_of(nc_cd)
    pre_cd = S3.prefix_state(KPRF, _sctx_for(REQ, nc_cd, NS0, sess_cd, 0))
    seq_cd = list(SYS + REQ)
    # 1 unconstrained CoT step
    cap_c0 = model.forward(seq_cd)
    cot_tok0 = int(S3.sample(pre_cd, 0, np.asarray(cap_c0.logits[-1]).view(np.float32), rule))
    assert cot_tok0 not in (EOS_ID, TOOL_END_ID)
    seq_cd.append(cot_tok0)
    vis_cd = []
    cur_q = 0
    for step_k in range(1, 20):
        cap_ck = model.forward(seq_cd)
        lg_ck = np.asarray(cap_ck.logits[-1]).view(np.float32)
        tk = int(S3.sample(pre_cd, step_k, lg_ck, rule, allowed=allow_cd[cur_q, :V]))
        seq_cd.append(tk)
        vis_cd.append(tk)
        cur_q = int(next_q_cd[cur_q, tk])
        if tk == EOS_ID:
            break
    assert vis_cd[-1] == EOS_ID and cur_q == dfa_cd.n_states
    rendered_json = b"".join(token_bytes[t] for t in vis_cd[:-1]).decode("utf-8")
    assert json.loads(rendered_json) in ({"ok": True}, {"ok": False}), rendered_json

    cap_cd = model.forward(seq_cd)
    api_cd = {
        "req_tokens": REQ,
        "n_reason": 1,
        "vis_tokens": vis_cd,
        "api_meta": {
            "finish_reason": "stop",
            "max_tokens": 25,
            "eos_token_id": EOS_ID,
            "tool_end_token_id": TOOL_END_ID,
            "schema": schema_cd,
        },
    }
    R_cd, wit_cd = RZ.make_wit(prf, model, cap_cd, com, sys_len=len(SYS), B=B)
    stp_cd, stw_cd, _ = RZ.turn_state(prf, com, api_cd, [cot_tok0], nc_cd, NS0, R_ST0)
    wit_cd["st"] = stw_cd
    stmt_cd = RZ.statement(prf, com, R_cd, api_cd, stp_cd)
    if not _run_case(f"constrained_json_honest ({rendered_json})", prf, stmt_cd, wit_cd, False, B):
        fails.append("constrained_json_honest")

    # White-box forgery 6b: output tokens are still valid JSON `rendered_json`, BUT on a sampled constrained step
    # the cheating prover flips 1 bit in `As` (e.g. sets `a = ones(V)` or flips one disallowed token to allowed).
    for g_salt in range(64):
        R_cdf, wit_cdf = RZ.make_wit(
            prf, model, cap_cd, com, sys_len=len(SYS),
            salts=((g_salt + 301, 0, 0, 0), (7, 0, 0, 0)), B=B,
        )
        stp_cdf, stw_cdf, _ = RZ.turn_state(prf, com, api_cd, [cot_tok0], nc_cd, NS0, R_ST0)
        wit_cdf["st"] = stw_cdf
        stmt_cdf = RZ.statement(prf, com, R_cdf, api_cd, stp_cdf)
        g_drawn = RZ.gen_steps_of(stmt_cdf["u_gen"], 1 + len(vis_cd))
        if any(g >= 1 for g in g_drawn):
            hit_slot = next(i for i, g in enumerate(g_drawn) if g >= 1)
            break
    else:
        raise AssertionError("expected constrained step to be sampled")
    bad_mask = np.ones(V, dtype=bool)    # wrong mask (unconstrained instead of DFA mask)
    wit_cdf["gen_mask_override"] = {hit_slot: bad_mask}
    if not _run_case("forged_wrong_dfa_mask_in_pool3", prf, stmt_cdf, wit_cdf, True, B):
        fails.append("forged_wrong_dfa_mask_in_pool3")

    # ------------------------------------------------------------------ 7. Profile pinning and bundle cross-checks
    from flashprove_next import reply_bundle as RB
    from flashprove_next import mc_zk as MZ

    # 7a. In-circuit M^vol sampling-rule mismatch: model_id committed rule (T=0.9) != profile's rule (T=0.6) -> REJECTED.
    bad_rule = S3.Rule(0.9, 0.95, 32)
    vol_bad_r = MC.encode_volatile_cfg({
        "k_prf": list(com["k_prf"]), "salt_cfg": list(com["salt_cfg"]),
        "sys_prompt": SYS, "L_real": 2, "V_real": V,
        "T_bits": bad_rule.T_bits, "p16": bad_rule.p16, "beta": bad_rule.beta,
    })
    ccfg_br = MC.volatile_config_root(vol_bad_r)
    pre_br, mid_br = MZ.model_preimage(com["store"].root, c_cfg=ccfg_br)
    com_br = {**com, "vol": vol_bad_r, "pre": pre_br, "mid": mid_br}
    R_br, wit_br = RZ.make_wit(prf, model, cap_cd, com_br, sys_len=len(SYS), B=B)
    stp_br, stw_br, _ = RZ.turn_state(prf, com_br, api_cd, [cot_tok0], nc_cd, NS0, R_ST0)
    wit_br["st"] = stw_br
    stmt_br = RZ.statement(prf, com_br, R_br, api_cd, stp_br)
    if not _run_case("forged_mismatched_vol_sampling_rule", prf, stmt_br, wit_br, True, B):
        fails.append("forged_mismatched_vol_sampling_rule")

    # 7b. Statement negative n_ret / unknown api_meta key rejected deterministically by check_statement
    for bad_label, bad_am in (
        ("negative_n_ret", {**api_tool["api_meta"], "tool_segments": [{"n_reason": 0, "call_tokens": call_seg0, "n_ret": -2, "ret_tokens": None}]}),
        ("unknown_api_meta_key", {**api_cd["api_meta"], "regrind_pad": 12345}),
    ):
        stmt_neg = dict(stmt_cd, api_meta=bad_am)
        try:
            RZ.check_statement(prf, stmt_neg)
            print(f"  [FAIL] {bad_label}: accepted by check_statement!", flush=True)
            fails.append(bad_label)
        except Z.ProofError:
            print(f"  [PASS] {bad_label}: rejected by check_statement", flush=True)

    # 7c. Bundle registry & CLI cross-checks: forged weak profile_spec, server-injected schema, fake eos_token_id,
    # and unrequested tool_segments MUST ALL be rejected by RB.verify_cli.
    reg_spec = RB.get_registered_profile("qwen3-0.6b-sec315")
    forged_weak_spec = {**reg_spec, "M_C": 2, "M_G": 2}
    blob_weak = RB.pack_bundle(forged_weak_spec, stmt_cd, {"dummy": 1})
    mid_cd_bytes = bytes.fromhex(RB.mid_to_hex(com["mid"]))
    ok_w, why_w, _ = RB.verify_cli(mid_cd_bytes, {"tokens": REQ, "max_tokens": 25, "schema": schema_cd},
                                   {"tokens": vis_cd, "n_reason": 1, "finish_reason": "stop"}, nc_cd.hex(), blob_weak)
    assert not ok_w and "do not match registered profile" in why_w, (ok_w, why_w)
    print(f"  [PASS] forged_bundle_weak_profile -> REJECTED ({why_w})", flush=True)

    # Register a pinned test profile matching `prf` to test server-injected schema / fake eos_token_id / unrequested tools
    RB.REGISTERED_PROFILES["_test_tiny_api"] = {
        "name": "_test_tiny_api",
        "security_tier": "test",
        "d": prf.d, "n_q": prf.n_q, "n_kv": prf.n_kv, "dh": prf.dh, "dff": prf.dff,
        "L_MAX": prf.L_MAX, "T_pad": prf.T_pad, "M_C": prf.M_C, "M_G": prf.M_G,
        "MQ_H": prf.tier.MQ, "s_bits": prf.s_bits, "SYS_CAP": prf.SYS_CAP,
        "VEC_PAD": prf.VEC_PAD, "HIST_CAP": prf.HIST_CAP, "TOOL_CAP": prf.TOOL_CAP,
        "QN_CAP": prf.atier.QN_CAP, "KN_CAP": prf.atier.KN_CAP, "EXP_BITS": prf.atier.EXP_BITS,
        "V": prf.V, "eps_bits": prf.eps_bits, "rope_theta": 10000.0,
        "eos_token_id": EOS_ID, "tool_end_token_id": TOOL_END_ID, "default_max_tokens": 32,
        "rule": rule.as_dict(), "prm": {"TO": prf.prm.TO, "TI": prf.prm.TI, "R_MAX": prf.prm.R_MAX,
                                        "RATE_LOG": prf.prm.RATE_LOG, "MQ": prf.prm.MQ, "GROUP": prf.prm.GROUP,
                                        "G_LO": prf.prm.G_LO},
    }
    try:
        # Honest proof from 6 (`stmt_cd` has `schema: schema_cd`), verified against a plain client request with NO schema.
        api_cd_no_tool = {**api_cd, "api_meta": {k: v for k, v in api_cd["api_meta"].items() if k != "tool_end_token_id"}}
        stmt_cd_nt = RZ.statement(prf, com, R_cd, api_cd_no_tool, stp_cd)
        pf_cd_nt, _ = RZ.prove(prf, stmt_cd_nt, wit_cd, B=B)
        blob_cd = RB.pack_bundle(RB.get_registered_profile("_test_tiny_api"), stmt_cd_nt, pf_cd_nt)

        ok_no_optin, why_no_optin, _ = RB.verify_cli(
            mid_cd_bytes,
            {"tokens": REQ, "max_tokens": 25, "schema": schema_cd},
            {"tokens": vis_cd, "n_reason": 1, "finish_reason": "stop"},
            nc_cd.hex(), blob_cd,
            allow_reduced_security=False,
        )
        assert not ok_no_optin and "--allow-reduced-security" in why_no_optin, (ok_no_optin, why_no_optin)

        ok_inj, why_inj, _ = RB.verify_cli(
            mid_cd_bytes,
            {"tokens": REQ, "max_tokens": 25},   # Client sent NO schema.
            {"tokens": vis_cd, "n_reason": 1, "finish_reason": "stop"},
            nc_cd.hex(), blob_cd,
            allow_reduced_security=True,
        )
        assert not ok_inj and "schema" in why_inj, (ok_inj, why_inj)
        print(f"  [PASS] forged_server_injected_schema -> REJECTED ({why_inj})", flush=True)

        # Fake eos_token_id chosen by server != pinned profile's EOS_ID
        stmt_fake_eos = dict(stmt_cd_nt, api_meta={**api_cd_no_tool["api_meta"], "eos_token_id": vis_cd[0]})
        blob_feos = RB.pack_bundle(RB.get_registered_profile("_test_tiny_api"), stmt_fake_eos, pf_cd_nt)
        ok_fe, why_fe, _ = RB.verify_cli(
            mid_cd_bytes,
            {"tokens": REQ, "max_tokens": 25, "schema": schema_cd},
            {"tokens": vis_cd, "n_reason": 1, "finish_reason": "stop"},
            nc_cd.hex(), blob_feos,
            allow_reduced_security=True,
        )
        assert not ok_fe and "eos_token_id" in why_fe, (ok_fe, why_fe)
        print(f"  [PASS] forged_fake_eos_token_id -> REJECTED ({why_fe})", flush=True)

        # Unrequested tool_segments on a client request with server_tools=False
        blob_utl = RB.pack_bundle(RB.get_registered_profile("_test_tiny_api"), stmt_tl, pf_cd_nt)
        ok_ut, why_ut, _ = RB.verify_cli(
            mid_cd_bytes,
            {"tokens": REQ, "max_tokens": len(em_g1) + len(em_g2)},   # No server_tools.
            {"tokens": vis_seg1, "n_reason": len(cot_seg1), "finish_reason": "length"},
            nc_t.hex(), blob_utl,
            allow_reduced_security=True,
        )
        assert not ok_ut and "tool_segments" in why_ut, (ok_ut, why_ut)
        print(f"  [PASS] forged_unrequested_tool_segments -> REJECTED ({why_ut})", flush=True)
    finally:
        RB.REGISTERED_PROFILES.pop("_test_tiny_api", None)

    if fails:
        print(f"FAILED: {fails}", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
