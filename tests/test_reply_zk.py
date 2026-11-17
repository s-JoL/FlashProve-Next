# SPDX-License-Identifier: Apache-2.0
"""White-box forgery tests for flashprove_next.reply_zk on the deterministic tiny dense engine.

Each forger satisfies every local relation it can and breaks exactly one binding link; each must be REJECTED:
  h_cheat_slot_honest : H commits a cheating execution (other weights), slots recomputed honestly  -> H binding
  consistent_cheat    : H and slots both from the cheating execution (gadgets violated, FORCE)     -> certified check
  slot_not_draw       : the prover computes slot 0 at a (t, l) other than the draw                  -> draw constraint
  wrong_model_id      : verifier given another model_id                                             -> HCO / model_id
  tok_unbound_to_H_emb: token chain = committed tokens, H = execution on other tokens               -> EMB block
  kv_zeroed_consistent_lift: k, v codes (and their lifts) zeroed at one key row, every later quantity computed on the
                        zeroed cache; ground so that a slot's producer read hits that row       -> producer N1 + KV + EMB
  skip_layer_kv_producer: layer l skipped everywhere except at the drawn slot position, which is computed on the
                        skipped (zero) cache; ground so the producer read hits a skipped row   -> producer N1 + KV + EMB
usage:  python tests/test_reply_zk.py [--quick] [--producer-only] [--gen-only] [--backend cpu|cupy]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flashprove_next import modelcommit as MC  # noqa: E402
from flashprove_next import reply_zk as RZ
from flashprove_next import zk_flat as Z
from flashprove_next.dense_engine import Capture, TinyDense
from flashprove_next import attn_zk as AZ
from flashprove_next.fp_ref import (EPS, W32, bf16, eng_gemm_res, eng_gu, f32_of_bf16, f32codes, rmsnorm_eng, rope_eng,
                             rope_table)
from flashprove_next.field import CPU, P
from flashprove_next.transcript import dumps


SYS, REQ, COT, VIS = [5, 6], [7, 8], [9], [70]
API = {"req_tokens": REQ, "n_reason": len(COT), "vis_tokens": VIS}
NC, NS, RND = b"\x11" * 16, b"\x22" * 32, (3, 5, 7, 11)


def stmt_of(prf, com, R, wit, api=API, hist=COT, **kw):
    """reply statement with the turn-0 state part; sets the prover's state witness in wit."""
    stp, stw, _ = RZ.turn_state(prf, com, api, hist, kw.get("nc", NC), NS, RND)
    wit["st"] = stw
    return RZ.statement(prf, com, R, api, stp)


def setup(M_C=2, seed=0):
    model = TinyDense(d=128, n_q=4, n_kv=2, dh=32, dff=256, L=3, V=128, seed=seed)
    prm = MC.Params(TO=64, TI=64, R_MAX=512, RATE_LOG=2, MQ=8, GROUP=8)
    prf = RZ.Profile(d=model.d, n_q=model.n_q, n_kv=model.n_kv, dh=model.dh, dff=model.dff, L_MAX=4, T_pad=8,
                     M_C=M_C, prm=prm, s_bits=2)
    toks = SYS + REQ + COT + VIS
    cap = model.forward(toks)
    com = RZ.commit_model(model.tensors(), prf, L_real=model.L, sys_prompt=SYS)
    return model, prf, toks, cap, com


def slots(prf, stmt, L):
    """prover-side view of the drawn (t, l) per compute slot (hidden from the verifier)."""
    b, T = len(SYS) + len(REQ), len(SYS + REQ + COT + VIS)
    N = len(COT) + len(VIS)
    return [RZ.slot_of(u, b, T, L, RZ.slot_pool(prf, N, s), prf.C_SELF, prf.C_KEY)[1:3]
            for s, u in enumerate(stmt["u_draws"])]


def forward_hooked(model, tokens, zero_kv=(), skip=None):
    """White-box cheating engine (TinyDense.forward with two hooks):
      zero_kv  set of (tau, l): the k, v codes of (tau, l) are replaced by 0 right after the engine produced them; the key
               stays in the cache (score 0, value 0) and every later quantity is computed from the zeroed cache;
      skip     (l, keep): layer l is skipped (h_out = h_in, every intermediate and k = v = 0) at every position except
               `keep`, where the layer runs on the skipped (zero) cache of the other positions.
    Everything else (downstream layers, residual wiring) is consistent with the hooked cache."""
    W = model.tensors()
    toks = np.asarray(tokens, dtype=np.int64)
    T, L, d, nq, nkv, dh, dff = len(toks), model.L, model.d, model.n_q, model.n_kv, model.dh, model.dff
    rep = nq // nkv
    scale = np.float32(1.0 / np.sqrt(dh))
    z = lambda *sh: np.zeros(sh, dtype=np.uint16)
    h_in, x_attn, q_a, k_a, v_a, o_a = z(T, L, d), z(T, L, d), z(T, L, nq * dh), z(T, L, nkv * dh), z(T, L, nkv * dh), z(T, L, nq * dh)
    h_mid, x_mlp, a_a, h_out = z(T, L, d), z(T, L, d), z(T, L, dff), z(T, L, d)
    lse = np.zeros((T, L, nq), dtype=np.uint32)
    h_in[:, 0, :] = W["model.embed_tokens.weight"][toks]
    for l in range(L):
        p = f"model.layers.{l}."
        for t in range(T):
            if skip is not None and l == skip[0] and t != skip[1]:
                h_out[t, l] = h_in[t, l]
                continue
            xa = rmsnorm_eng(h_in[t, l], W[p + "input_layernorm.weight"], EPS)
            x_attn[t, l] = xa
            xf = f32_of_bf16(xa)
            cos, sin = rope_table(t, dh)
            q_lin = bf16(W32(W[p + "self_attn.q_proj.weight"]) @ xf).reshape(nq, dh)
            k_lin = bf16(W32(W[p + "self_attn.k_proj.weight"]) @ xf).reshape(nkv, dh)
            if model.qk_norm:
                q_lin = rmsnorm_eng(q_lin, W[p + "self_attn.q_norm.weight"], EPS)
                k_lin = rmsnorm_eng(k_lin, W[p + "self_attn.k_norm.weight"], EPS)
            q_a[t, l] = rope_eng(q_lin, cos, sin).reshape(-1)
            k_a[t, l] = rope_eng(k_lin, cos, sin).reshape(-1)
            v_a[t, l] = bf16(W32(W[p + "self_attn.v_proj.weight"]) @ xf).reshape(-1)
            if (t, l) in zero_kv:
                k_a[t, l] = 0; v_a[t, l] = 0
            qf = f32_of_bf16(q_a[t, l].reshape(nq, dh)).astype(np.float64)
            Kf = np.repeat(f32_of_bf16(k_a[:t + 1, l].reshape(t + 1, nkv, dh).transpose(1, 0, 2)).astype(np.float64), rep, axis=0)
            Vf = np.repeat(f32_of_bf16(v_a[:t + 1, l].reshape(t + 1, nkv, dh).transpose(1, 0, 2)).astype(np.float64), rep, axis=0)
            zz = scale * np.einsum("hd,htd->ht", qf, Kf)
            m = zz.max(-1, keepdims=True); w = np.exp(zz - m); ws = w.sum(-1, keepdims=True)
            o_a[t, l] = bf16(np.einsum("ht,htd->hd", w / ws, Vf).astype(np.float32)).reshape(-1)
            lse[t, l] = f32codes((m[:, 0] + np.log(ws[:, 0])).astype(np.float32))
            hm = eng_gemm_res(W[p + "self_attn.o_proj.weight"], o_a[t, l], h_in[t, l])
            h_mid[t, l] = hm
            xm = rmsnorm_eng(hm, W[p + "post_attention_layernorm.weight"], EPS)
            x_mlp[t, l] = xm
            act = eng_gu(W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"], xm)
            a_a[t, l] = act
            h_out[t, l] = eng_gemm_res(W[p + "mlp.down_proj.weight"], act, hm)
        if l + 1 < L:
            h_in[:, l + 1, :] = h_out[:, l, :]
    return Capture(tokens=toks, h_in=h_in, x_attn=x_attn, q=q_a, k=k_a, v=v_a, o=o_a, h_mid=h_mid, x_mlp=x_mlp, a=a_a,
                   h_out=h_out, x_fn=z(T, d), logits=np.zeros((T, model.V), dtype=np.uint32), lse=lse)


def producer_rows(prf, stmt, L):
    """prover-side view: per compute slot (t, l, tau_p) with tau_p = attn_zk.key_draw(u_att, t)."""
    return [(t, l, AZ.key_draw(u, t)) for (t, l), u in zip(slots(prf, stmt, L), stmt["u_att"])]


_BACKEND = [None]


def run(tag, prf, stmt, wit, reject, vstmt=None):
    from flashprove_next.field import CPU
    B = _BACKEND[0] or CPU
    Z.FORCE = reject
    try:
        pf, info = RZ.prove(prf, stmt, wit, B=B)
    finally:
        Z.FORCE = False
    t1 = time.time()
    res = RZ.try_verify(prf, vstmt or stmt, pf)
    tv = time.time() - t1
    ok = res.startswith("rejected") if reject else res == "accepted"
    print(f"{'PASS' if ok else 'FAIL'} {tag}: {res}  (build {info['build_s']} s, prove {info['prove_s']} s, verify "
          f"{tv:.2f} s, {info['cells']} cells {info['cells_by_kind']}, {len(dumps(pf))} B, gadget fails "
          f"{len(info['gadget_fails'])}, notes {info['violations'][:2]})", flush=True)
    return ok, info


def producer_cases(model, prf, toks, cap, com, L, fails):
    """key-producer lineage forgeries (depth 1)."""
    # ---- key-producer lineage (depth 1): the producer read (tau_p, l) of a slot must be a certified N1 + KV + EMB output.
    # The honest hooked engine must reproduce TinyDense bit for bit (otherwise the forgeries test the hook, not the proof)
    hk = forward_hooked(model, toks)
    assert all(np.array_equal(getattr(hk, f), getattr(cap, f)) for f in
               ("h_in", "x_attn", "q", "k", "v", "o", "h_mid", "x_mlp", "a", "h_out")), "hooked engine != TinyDense"
    LZ = 1
    # (a) k, v codes and lifts zeroed at (tau, LZ) (Lift(0) = 0: the lift-only producer check passes); attention and
    #     everything downstream consistent with the zeroed cache.  Grind: some slot of layer LZ reads tau as producer,
    #     no slot computes (tau, LZ) itself.
    found = None
    for g in range(400):
        tau = g % 3
        cz = forward_hooked(model, toks, zero_kv={(tau, LZ)})
        R14, wit14 = RZ.make_wit(prf, model, cz, com, sys_len=len(SYS), salts=((g + 401, 0, 0, 0), (7, 0, 0, 0)))
        stmt14 = stmt_of(prf, com, R14, wit14)
        pr = producer_rows(prf, stmt14, L)
        if any(l == LZ and tp == tau and t != tau for t, l, tp in pr) and not any((t, l) == (tau, LZ) for t, l, _ in pr):
            found = (tau, pr)
            break
    else:
        raise AssertionError("kv_zeroed_consistent_lift: grind found no salt")
    print(f"kv_zeroed_consistent_lift: zeroed (tau, l) = ({found[0]}, {LZ}); slots (t, l, tau_p) = {found[1]}", flush=True)
    ok, _ = run("kv_zeroed_consistent_lift", prf, stmt14, wit14, True)
    fails += [] if ok else ["kv_zeroed_consistent_lift"]

    # (b) layer LZ skipped (h_out = h_in, k = v = 0) everywhere except at `keep`, which runs on the zero cache.  Grind:
    #     every slot of layer LZ computes position keep and at least one of them reads a skipped producer row.
    found = None
    for g in range(400):
        keep = 1 + g % 4
        cs = forward_hooked(model, toks, skip=(LZ, keep))
        R15, wit15 = RZ.make_wit(prf, model, cs, com, sys_len=len(SYS), salts=((g + 901, 0, 0, 0), (7, 0, 0, 0)))
        stmt15 = stmt_of(prf, com, R15, wit15)
        pr = producer_rows(prf, stmt15, L)
        inl = [(t, tp) for t, l, tp in pr if l == LZ]
        if inl and all(t == keep for t, _ in inl) and any(tp != keep for _, tp in inl):
            found = (keep, pr)
            break
    else:
        raise AssertionError("skip_layer_kv_producer: grind found no salt")
    print(f"skip_layer_kv_producer: layer {LZ} skipped except t = {found[0]}; slots (t, l, tau_p) = {found[1]}", flush=True)
    ok, _ = run("skip_layer_kv_producer", prf, stmt15, wit15, True)
    fails += [] if ok else ["skip_layer_kv_producer"]



def gen_cases(fails):
    """Pool-3 generation slot tests (M_G = 2): honest end-to-end sampled generation + 4 white-box forgers."""
    from flashprove_next import sampling_v3 as S3
    from flashprove_next import state_zk as SZ_ST
    kprf = [0x1234567, 0x89ABCDEF, 0x0F1E2D3C, 0x4B5A6978]
    model = TinyDense(d=128, n_q=4, n_kv=2, dh=32, dff=256, L=3, V=128, seed=0)
    prm = MC.Params(TO=64, TI=64, R_MAX=512, RATE_LOG=2, MQ=8, GROUP=8)
    rule = S3.Rule(0.6, 0.95, 32)
    prf = RZ.Profile(d=model.d, n_q=model.n_q, n_kv=model.n_kv, dh=model.dh, dff=model.dff, L_MAX=4, T_pad=8,
                     M_C=2, M_G=2, V=model.V, prm=prm, s_bits=2, rule=rule)
    com = RZ.commit_model(model.tensors(), prf, L_real=model.L, V_real=model.V, sys_prompt=SYS, k_prf=kprf)
    # Autoregressive rollout under rule + K_prf + SampCtx so every generated token (COT + VIS) is genuinely sampled.
    v_rel = SZ_ST.v_rel_of(RZ.VERSION)
    sess = SZ_ST.sess_of(NC)
    dig = __import__("hashlib").blake2b(json.dumps(REQ, separators=(",", ":")).encode(),
                                        digest_size=32, person=b"fpn-req-dig").digest()
    sess_b = b"".join(int(x).to_bytes(8, "little") for x in sess)
    sctx = S3.SampCtx(NC, NS, dig=dig, v_rel=int(v_rel) & 0xFFFFFFFF, epoch=0, sess=sess_b, turn=0)
    pre = S3.prefix_state(kprf, sctx)
    seq = list(SYS + REQ)
    for g_step in range(2):
        cap_step = model.forward(seq)
        lg = np.asarray(cap_step.logits[-1]).view(np.float32)
        seq.append(int(S3.sample(pre, g_step, lg, rule)))
    cot_g, vis_g = [seq[-2]], [seq[-1]]
    api_g = {"req_tokens": REQ, "n_reason": len(cot_g), "vis_tokens": vis_g}
    cap_g = model.forward(seq)

    R0, wit0 = RZ.make_wit(prf, model, cap_g, com, sys_len=len(SYS))
    st0 = stmt_of(prf, com, R0, wit0, api=api_g, hist=cot_g)
    ok, _ = run("gen_honest", prf, st0, wit0, False)
    fails += [] if ok else ["gen_honest"]

    # (1) Unverified draft / forged visible token: replace vis_g[0] with argmax != sampled token, re-run forward so
    #     compute, attention, producer, EMB, and state chain are 100% consistent.  Grind salt so u_gen hits g = 1.
    lg_last = np.asarray(cap_g.logits[-2]).view(np.float32)
    bad_tok = int(np.argmax(lg_last))
    if bad_tok == vis_g[0]:
        bad_tok = int(np.argmin(lg_last))
    seq_bad = list(SYS + REQ + cot_g + [bad_tok])
    cap_bad = model.forward(seq_bad)
    api_bad = {"req_tokens": REQ, "n_reason": len(cot_g), "vis_tokens": [bad_tok]}
    for g in range(64):
        Rb, witb = RZ.make_wit(prf, model, cap_bad, com, sys_len=len(SYS), salts=((g + 2001, 0, 0, 0), (7, 0, 0, 0)))
        stb = stmt_of(prf, com, Rb, witb, api=api_bad, hist=cot_g)
        if 1 in RZ.gen_steps_of(stb["u_gen"], 2):
            break
    ok, _ = run("gen_unverified_draft_tok", prf, stb, witb, True)
    fails += [] if ok else ["gen_unverified_draft_tok"]

    # (2) Wrong K_prf: tokens sampled under kprf2 != committed kprf in vol_col
    kprf2 = [x + 9 for x in kprf]
    pre2 = S3.prefix_state(kprf2, sctx)
    seq2 = list(SYS + REQ)
    for g_step in range(2):
        c_ = model.forward(seq2)
        seq2.append(int(S3.sample(pre2, g_step, np.asarray(c_.logits[-1]).view(np.float32), rule)))
    cap2 = model.forward(seq2)
    api2 = {"req_tokens": REQ, "n_reason": 1, "vis_tokens": [seq2[-1]]}
    Rk, witk = RZ.make_wit(prf, model, cap2, com, sys_len=len(SYS))
    witk["kprf_override"] = kprf2
    stk = stmt_of(prf, com, Rk, witk, api=api2, hist=[seq2[-2]])
    ok, _ = run("gen_wrong_vol_kprf", prf, stk, witk, True)
    fails += [] if ok else ["gen_wrong_vol_kprf"]

    # (3) Wrong generation position override (prover points slot 0 at the other generation position)
    Rp, witp = RZ.make_wit(prf, model, cap_g, com, sys_len=len(SYS))
    stp = stmt_of(prf, com, Rp, witp, api=api_g, hist=cot_g)
    g0 = RZ.gen_steps_of(stp["u_gen"], 2)[0]
    b_pos = len(SYS) + len(REQ)
    witp["gen_pos_override"] = {0: b_pos - 1 + (1 - g0)}
    ok, _ = run("gen_slot_not_draw", prf, stp, witp, True)
    fails += [] if ok else ["gen_slot_not_draw"]

    # (4) Detached h_out(pos_s, L - 1) in H (prover feeds honest cap_g to gen slot while H has altered last-layer h_out)
    cap_dh = Capture(**{f: getattr(cap_g, f).copy() if hasattr(getattr(cap_g, f), "copy") else getattr(cap_g, f)
                        for f in cap_g.__dataclass_fields__})
    g0_pos = b_pos - 1 + RZ.gen_steps_of(st0["u_gen"], 2)[0]
    cap_dh.h_out[g0_pos, model.L - 1, 0] ^= np.uint16(0x0080)
    for g in range(64):
        Rd, witd = RZ.make_wit(prf, model, cap_g, com, H_cap=cap_dh, sys_len=len(SYS),
                               salts=((g + 2501, 0, 0, 0), (7, 0, 0, 0)))
        std = stmt_of(prf, com, Rd, witd, api=api_g, hist=cot_g)
        if b_pos - 1 + RZ.gen_steps_of(std["u_gen"], 2)[0] == g0_pos and (g0_pos, model.L - 1) not in slots(prf, std, model.L):
            break
    ok, _ = run("gen_detached_hout", prf, std, witd, True)
    fails += [] if ok else ["gen_detached_hout"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--producer-only", action="store_true", help="honest + the two key-producer forgeries")
    ap.add_argument("--gen-only", action="store_true", help="Pool-3 generation honest + 4 white-box forgers")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cupy"])
    a = ap.parse_args(argv)
    from flashprove_next.field import backend as get_backend
    _BACKEND[0] = get_backend(a.backend)
    model, prf, toks, cap, com = setup()
    T, L = len(toks), model.L
    print(f"tokens {toks} (sys {SYS} hidden, req {REQ}, cot {COT} hidden, vis {VIS})", flush=True)
    fails = []

    if a.gen_only:
        fails_g = []
        gen_cases(fails_g)
        print("ALL PASS" if not fails_g else f"FAILED: {fails_g}", flush=True)
        return 0 if not fails_g else 1
    R, wit = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS))
    stmt = stmt_of(prf, com, R, wit)
    print(f"slots (prover view; hidden from the verifier) {slots(prf, stmt, L)} (pools {[RZ.slot_pool(prf, len(COT) + len(VIS), k) for k in range(prf.M_C)]})", flush=True)
    ok, info = run("honest", prf, stmt, wit, False)
    fails += [] if ok else ["honest"]
    if a.producer_only:
        producer_cases(model, prf, toks, cap, com, L, fails)
        print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
        return 0 if not fails else 1
    if a.quick:
        return 0 if not fails else 1

    cheat = model.forward(toks, cheat={"exec_seed": 99})
    R2, wit2 = RZ.make_wit(prf, model, cap, com, H_cap=cheat, sys_len=len(SYS))
    stmt2 = stmt_of(prf, com, R2, wit2)
    ok, _ = run("h_cheat_slot_honest", prf, stmt2, wit2, True)
    fails += [] if ok else ["h_cheat_slot_honest"]

    R3, wit3 = RZ.make_wit(prf, model, cheat, com, sys_len=len(SYS))
    stmt3 = stmt_of(prf, com, R3, wit3)
    ok, _ = run("consistent_cheat", prf, stmt3, wit3, True)
    fails += [] if ok else ["consistent_cheat"]

    R4, wit4 = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS))
    stmt4 = stmt_of(prf, com, R4, wit4)
    t0, l0 = slots(prf, stmt4, L)[0]
    wit4["slot_override"] = {0: ((t0 + 1) % T, l0)}
    ok, _ = run("slot_not_draw", prf, stmt4, wit4, True)
    fails += [] if ok else ["slot_not_draw"]

    # White-box modular wrap-around draw shift: add P at limb k = 8 in rzDr to shift J -> J + 1 onto a neighbouring
    # honest slot (t', l'); caught exclusively by the 46-bit carry range check (rem + car*2^16 < 2^62 < P).
    for g in range(64):
        R4w, wit4w = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS), salts=((g + 501, 0, 0, 0), (7, 0, 0, 0)))
        stmt4w = stmt_of(prf, com, R4w, wit4w)
        bw_ = len(SYS) + len(REQ)
        J0, _, _, _ = RZ.slot_of(stmt4w["u_draws"][0], bw_, T, L, 0, prf.C_SELF, prf.C_KEY)
        C0_ = L * ((bw_ - 1) * prf.C_SELF + prf.C_KEY * (bw_ - 1) * bw_ // 2)
        if J0 + 1 < C0_:
            break
    wit4w["draw_wrap_limb"] = {0: 8}
    ok, _ = run("draw_wrap_shift_j_plus_1", prf, stmt4w, wit4w, True)
    fails += [] if ok else ["draw_wrap_shift_j_plus_1"]

    def bad_mid(s):
        s = json.loads(json.dumps(s)); s["model_id"][0] = (s["model_id"][0] + 1) % P
        s["st"]["model_id"] = list(s["model_id"])
        s["u_draws"] = RZ.draw_seeds({k: s[k] for k in RZ.CORE_KEYS}, prf.M_C)
        s["u_att"] = RZ.att_draws(s["u_draws"])    # a well-formed statement: rejection must come from the circuit
        return s
    ok, _ = run("wrong_model_id", prf, stmt, wit, True, vstmt=bad_mid(stmt))
    fails += [] if ok else ["wrong_model_id"]

    # hidden system prompt swapped (the engine ran another sys prompt than the committed one)
    toks_b = [5, 9] + REQ + COT + VIS
    cap_b = model.forward(toks_b)
    R5, wit5 = RZ.make_wit(prf, model, cap_b, com, sys_len=len(SYS))
    stmt5 = stmt_of(prf, com, R5, wit5)
    ok, _ = run("sys_prompt_swapped", prf, stmt5, wit5, True)
    fails += [] if ok else ["sys_prompt_swapped"]

    # visible reply in the statement differs from the token chain the prover executed
    R6, wit6 = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS))
    stmt6 = stmt_of(prf, com, R6, wit6, api=dict(API, vis_tokens=[11]))
    ok, _ = run("visible_reply_mismatch", prf, stmt6, wit6, True)
    fails += [] if ok else ["visible_reply_mismatch"]

    # truncated KV window: the engine attends only to the last 2 keys; H and slots consistent (attention violated)
    tk = model.forward(toks, cheat={"trunc_kv": 2})
    for g in range(64):        # grind the trace-root salt until a drawn slot reads > 2 keys (only attention is violated)
        R7, wit7 = RZ.make_wit(prf, model, tk, com, sys_len=len(SYS), salts=((g + 1, 0, 0, 0), (7, 0, 0, 0)))
        stmt7 = stmt_of(prf, com, R7, wit7)
        if any(t >= 2 for t, _ in slots(prf, stmt7, L)):
            break
    ok, _ = run("trunc_kv_window", prf, stmt7, wit7, True)
    fails += [] if ok else ["trunc_kv_window"]

    # detached residual: every layer l >= 1 reads h_in != h_out(l - 1); everything else consistent with that h_in
    dt = model.forward(toks, cheat={"detach_residual": True})
    for g in range(64):
        R8, wit8 = RZ.make_wit(prf, model, dt, com, sys_len=len(SYS), salts=((g + 101, 0, 0, 0), (7, 0, 0, 0)))
        stmt8 = stmt_of(prf, com, R8, wit8)
        if any(l >= 1 for _, l in slots(prf, stmt8, L)):
            break
    ok, _ = run("detached_residual", prf, stmt8, wit8, True)
    fails += [] if ok else ["detached_residual"]

    # tok column holds the committed sys prompt while H holds an execution on another sys prompt (prompt
    # substitution under a well-formed token chain): caught by the EMB block, which binds emb(t) = E[tok[t]]
    R9, wit9 = RZ.make_wit(prf, model, cap_b, com, sys_len=len(SYS), tokens=toks)
    for g in range(64):        # grind until a drawn slot sits on a swapped position (t = 1)
        R9, wit9 = RZ.make_wit(prf, model, cap_b, com, sys_len=len(SYS), tokens=toks,
                               salts=((g + 201, 0, 0, 0), (7, 0, 0, 0)))
        stmt9 = stmt_of(prf, com, R9, wit9)
        if any(t == 1 for t, _ in slots(prf, stmt9, L)):
            break
    ok, _ = run("tok_unbound_to_H_emb", prf, stmt9, wit9, True)
    fails += [] if ok else ["tok_unbound_to_H_emb"]

    # Pi_2: the generation-layer slot is computed at an input-layer position (the cheater's cheap prefix)
    R10, wit10 = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS))
    stmt10 = stmt_of(prf, com, R10, wit10)
    _, l1 = slots(prf, stmt10, L)[1]
    wit10["slot_override"] = {1: (1, l1)}
    ok, _ = run("gen_pool_slot_in_input_layer", prf, stmt10, wit10, True)
    fails += [] if ok else ["gen_pool_slot_in_input_layer"]

    # H executed on another hidden system prompt while NO compute slot is drawn inside the injection region:
    # only the per-proof EMB check of every injection position can catch it (deterministic class-5 detection)
    for g in range(64):
        R13, wit13 = RZ.make_wit(prf, model, cap_b, com, sys_len=len(SYS), tokens=toks,
                                 salts=((g + 301, 0, 0, 0), (7, 0, 0, 0)))
        stmt13 = stmt_of(prf, com, R13, wit13)
        if all(t >= len(SYS) for t, _ in slots(prf, stmt13, L)):
            break
    ok, _ = run("sys_prompt_executed_unsampled", prf, stmt13, wit13, True)
    fails += [] if ok else ["sys_prompt_executed_unsampled"]

    producer_cases(model, prf, toks, cap, com, L, fails)
    gen_cases(fails)

    # LSE tampered in H at a drawn slot while q, k, v, K, V, o are honest (LSE observation check, specs/06 §5.2)
    cap_lse = Capture(**{f: getattr(cap, f).copy() if hasattr(getattr(cap, f), "copy") else getattr(cap, f)
                         for f in cap.__dataclass_fields__})
    t_lse, l_lse = slots(prf, stmt, L)[0]
    cap_lse.lse[t_lse, l_lse] = (cap_lse.lse[t_lse, l_lse].view(np.float32) + np.float32(0.5)).view(np.uint32)
    for g in range(64):
        R16, wit16 = RZ.make_wit(prf, model, cap_lse, com, sys_len=len(SYS), salts=((g + 1501, 0, 0, 0), (7, 0, 0, 0)))
        stmt16 = stmt_of(prf, com, R16, wit16)
        if (t_lse, l_lse) in slots(prf, stmt16, L):
            break
    ok, _ = run("lse_tampered", prf, stmt16, wit16, True)
    fails += [] if ok else ["lse_tampered"]

    # state anchor commits a hidden CoT other than the one the token chain (and H) holds
    R11, wit11 = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS))
    stmt11 = stmt_of(prf, com, R11, wit11, hist=[8])
    ok, _ = run("state_anchor_other_cot", prf, stmt11, wit11, True)
    fails += [] if ok else ["state_anchor_other_cot"]

    # replay: an honest proof presented for a request with another client nonce (anchors recomputed accordingly)
    R12, wit12 = RZ.make_wit(prf, model, cap, com, sys_len=len(SYS))
    stmt12 = stmt_of(prf, com, R12, wit12)
    Z.FORCE = False
    pf12, _ = RZ.prove(prf, stmt12, wit12, B=_BACKEND[0] or CPU)
    w_tmp = {}
    stmt12b = stmt_of(prf, com, R12, w_tmp, nc=b"\x33" * 16)
    res12 = RZ.try_verify(prf, stmt12b, pf12)
    ok = res12.startswith("rejected")
    print(f"{'PASS' if ok else 'FAIL'} replay_other_client_nonce: {res12}", flush=True)
    fails += [] if ok else ["replay_other_client_nonce"]

    print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
