# SPDX-License-Identifier: Apache-2.0
"""Fast CPU test suite (CI and laptop review): no GPU, no model download.

  1. Public policy Phi_pre and the JSON-Schema byte DFA (minimality, co-accessibility, prefix extensibility, token masks).
  2. Fixed-length AEAD state blobs: the ciphertext length is independent of the hidden history; tampering is rejected.
  3. Proof-bundle verifier: pinned profile registry, weakened profile, server-injected schema, fake EOS id, unrequested
     tool segments, wrong nonce.
  4. End-to-end proofs on a tiny deterministic dense model (TinyDense, M_C = 1, M_G = 1, T_pad = 16, V = 64):
       honest reply with a hidden system prompt and hidden reasoning, checked by the standalone CLI verifier in a
       CPU-only subprocess -> VERIFIED;
       white-box forgeries -> REJECTED: int4-quantized weights, a stop token hidden inside the reasoning, and a wrong
       constrained-decoding mask on a sampled generation step (next to an honest JSON-Schema reply -> accepted).

    python -m pytest tests/test_fast_cpu.py        or        python tests/test_fast_cpu.py
"""
import functools
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("FPN_OOC", "1")  # 1.51 B/cell streaming mode keeps host RSS < 2.5 GiB on 7 GB CI runners

from flashprove_next import api_zk as APZ  # noqa: E402
from flashprove_next import modelcommit as MC  # noqa: E402
from flashprove_next import reply_bundle as RB  # noqa: E402
from flashprove_next import reply_zk as RZ  # noqa: E402
from flashprove_next import sampling_v3 as S3  # noqa: E402
from flashprove_next import state_zk as SZ  # noqa: E402
from flashprove_next import zk_flat as Z  # noqa: E402
from flashprove_next.dense_engine import TinyDense  # noqa: E402
from flashprove_next.field import CPU  # noqa: E402
from flashprove_next.fp_ref import EPS_BITS  # noqa: E402

VERIFY_CLI = os.path.join(ROOT, "verifier", "fpn_verify.py")
PROFILE = "_ci_tiny"
V = 64
EOS = 62
TOOL_END = 63
SYS = [5, 6]                    # hidden system prompt (committed in model_id)
REQ = [7, 8]                    # public request tokens
NS = b"\x22" * 32               # server nonce
R_ST = (3, 5, 7, 11)            # state-anchor randomness
KPRF = [0x1234567, 0x89ABCDEF, 0x0F1E2D3C, 0x4B5A6978]
RULE = S3.Rule(0.6, 0.95, 32)
PRM = MC.Params(TO=64, TI=64, R_MAX=512, RATE_LOG=2, MQ=8, GROUP=8)
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
          "additionalProperties": False}


# ============================================================================ helpers
def tiny_spec():
    """Verifier-side profile spec of the tiny test model (JSON-serializable)."""
    return {
        "name": PROFILE, "security_tier": "test", "rope_recipe": "reference",
        "d": 128, "n_q": 4, "n_kv": 2, "dh": 32, "dff": 256,
        "L_MAX": 4, "T_pad": 16, "M_C": 1, "M_G": 1,
        "MQ_H": 16, "s_bits": 2, "SYS_CAP": 8, "VEC_PAD": 4096, "HIST_CAP": 16, "TOOL_CAP": 0,
        "QN_CAP": 4.0, "KN_CAP": 512.0, "EXP_BITS": 17,
        "V": V, "eps_bits": EPS_BITS, "rope_theta": 1000000.0,
        "eos_token_id": EOS, "tool_end_token_id": TOOL_END, "default_max_tokens": 32,
        "rule": RULE.as_dict(), "prm": asdict(PRM),
    }


def tiny_model(eos_boost=False):
    m = TinyDense(d=128, n_q=4, n_kv=2, dh=32, dff=256, L=2, V=V, seed=3)
    if eos_boost:                                  # EOS gets the logit row of token 0: sampled now and then
        m._tensors["lm_head.weight"][EOS] = m._tensors["lm_head.weight"][0]
    return m


def token_bytes():
    """Byte vocabulary for constrained decoding: JSON pieces of {"ok": true|false}, inert filler, EOS / TOOL_END special."""
    tb = [b"#" + str(i).encode() for i in range(V)]
    pieces = [b'{"ok":', b"true", b"false", b"true}", b"false}", b"}",
              b"t", b"r", b"u", b"e", b"f", b"a", b"l", b"s"]
    for k, p in enumerate(pieces):
        tb[40 + k] = p
    tb[EOS] = None
    tb[TOOL_END] = None
    return tb


def samp_ctx(nc):
    sess = SZ.sess_of(nc)
    return RZ.samp_ctx_of({"req_tokens": REQ, "st": {"N_c": nc.hex(), "N_s": NS.hex(), "sess": list(sess),
                                                       "v_rel": SZ.v_rel_of(RZ.VERSION), "turn": 0}})


def rollout(model, prefix, n_steps, pre, g_offset=0, stop_ids=(), allowed_fn=None):
    seq, emitted = list(prefix), []
    for k in range(n_steps):
        lg = np.asarray(model.forward(seq).logits[-1]).view(np.float32).copy()
        allow = allowed_fn(k, emitted) if allowed_fn is not None else None
        tok = int(S3.sample(pre, g_offset + k, lg, RULE, allowed=allow))
        seq.append(tok)
        emitted.append(tok)
        if tok in stop_ids:
            break
    return seq, emitted


def reply_statement(prf, model, com, cap, api, hist, nc, salts=None):
    R, wit = RZ.make_wit(prf, model, cap, com, salts=salts, sys_len=len(SYS), B=CPU)
    stp, stw, _ = RZ.turn_state(prf, com, api, hist, nc, NS, R_ST)
    wit["st"] = stw
    return RZ.statement(prf, com, R, api, stp), wit


def prove_and_verify(prf, stmt, wit, cheat, fast_honest_build_only=False):
    """Prove (forcing the prover past failed constraints for a cheater) and verify in-process."""
    if os.environ.get("FPN_CI_FAST") == "1":
        if cheat:
            Z.FORCE = False
            try:
                RZ.build(prf, stmt, wit, CPU)
            except Z.ProofError as e:
                return None, f"rejected before proving ({e})"
            raise AssertionError("expected dishonest witness to fail constraint check")
        if fast_honest_build_only:
            Z.FORCE = False
            em, _, _ = RZ.build(prf, stmt, wit, CPU)
            assert not em.fails and not em.ctx.notes, (em.fails, em.ctx.notes)
            return None, "accepted"
    Z.FORCE = bool(cheat)
    try:
        # CPU-only CI uses the vectorized NumPy/C++ backend for TinyDense (8.15M cells);
        # GPU/production callers use the default folded CUDA backend (`backend="folded"`).
        pf, _ = RZ.prove(prf, stmt, wit, B=CPU, backend="legacy")
    except Z.ProofError as e:
        assert cheat, f"honest prover failed: {e}"
        return None, f"rejected before proving ({e})"
    finally:
        Z.FORCE = False
    return pf, RZ.try_verify(prf, stmt, pf)


@functools.lru_cache(maxsize=None)
def fixture():
    spec = json.loads(json.dumps(tiny_spec()))
    if PROFILE not in RB.REGISTERED_PROFILES:
        RB.register_profile(spec)
    prf = RB.build_profile(spec)
    model = tiny_model()
    com = RZ.commit_model(model.tensors(), prf, L_real=model.L, V_real=V, sys_prompt=SYS, k_prf=KPRF)
    return spec, prf, model, com


@functools.lru_cache(maxsize=None)
def honest_reply():
    """Turn 0: hidden system prompt, one hidden reasoning token, two visible tokens, finish_reason 'length'."""
    spec, prf, model, com = fixture()
    for i in range(256):
        nc = (1000 + i).to_bytes(16, "little")
        seq, em = rollout(model, SYS + REQ, 3, S3.prefix_state(KPRF, samp_ctx(nc)))
        if EOS not in em and TOOL_END not in em:
            break
    else:
        raise AssertionError("no stop-free rollout found")
    cot, vis = em[:1], em[1:]
    api = {"req_tokens": REQ, "n_reason": len(cot), "vis_tokens": vis,
           "api_meta": {"finish_reason": "length", "max_tokens": len(em), "eos_token_id": EOS}}
    stmt, wit = reply_statement(prf, model, com, model.forward(seq), api, cot, nc)
    pf, res = prove_and_verify(prf, stmt, wit, cheat=False)
    assert res == "accepted", res
    return {"nc": nc, "seq": seq, "cot": cot, "vis": vis, "api": api, "stmt": stmt, "pf": pf,
            "blob": RB.pack_bundle(spec, stmt, pf)}


def client_request(h, **extra):
    return {"tokens": REQ, "max_tokens": len(h["cot"]) + len(h["vis"]), **extra}


def client_reply(h):
    return {"tokens": h["vis"], "n_reason": len(h["cot"]), "finish_reason": "length",
            "St_new": list(h["stmt"]["st"]["St_new"])}


def mid_bytes():
    return bytes.fromhex(RB.mid_to_hex(fixture()[3]["mid"]))


# ============================================================================ 1. public policy and schema DFA
def _equivalence_classes(trans, accept):
    """Number of Myhill-Nerode classes of a complete-ified DFA (Moore refinement; -1 = dead sink)."""
    n = trans.shape[0]
    cls = [1 if q in accept else 0 for q in range(n)] + [2]          # index n = dead sink
    t = np.where(trans >= 0, trans, n)
    t = np.vstack([t, np.full((1, 256), n, dtype=t.dtype)])
    while True:
        sig = {}
        new = []
        for q in range(n + 1):
            key = (cls[q],) + tuple(cls[int(x)] for x in t[q])
            new.append(sig.setdefault(key, len(sig)))
        if len(set(new)) == len(set(cls)):
            return len(set(new[:n]))
        cls = new


def test_policy_and_schema_dfa():
    ok, why = APZ.check_public_policy({"messages": [{"role": "user", "content": "hi"}], "stop": ["END"]}, RULE)
    assert not ok and "custom_stop_sequence" in why, (ok, why)
    assert not APZ.check_public_policy({"prompt": "hi", "logprobs": True}, RULE)[0]
    assert not APZ.check_public_policy({"prompt": "hi", "temperature": 0.1}, RULE)[0]
    assert not APZ.check_public_policy({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, RULE)[0]
    bad = {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
    assert not APZ.check_public_policy(
        {"response_format": {"type": "json_schema", "json_schema": {"schema": bad, "strict": True}}}, RULE)[0]
    sch = {"type": "object", "properties": {"ans": {"type": "integer"}, "ok": {"type": "boolean"}},
           "required": ["ans", "ok"], "additionalProperties": False}
    ok, why = APZ.check_public_policy({"messages": [{"role": "user", "content": "1+1"}], "response_format": {
        "type": "json_schema", "json_schema": {"schema": sch, "strict": True}}}, RULE)
    assert ok and why == "ok", (ok, why)

    dfa = APZ.compile_json_schema_dfa(sch)
    assert dfa.accepts_bytes(b'{"ans":42,"ok":true}') and dfa.accepts_bytes(b'{"ans":-7,"ok":false}')
    assert not dfa.accepts_bytes(b'{"ok":true,"ans":42}')            # declaration order is part of the dialect
    assert not dfa.accepts_bytes(b'{"ans":42,"ok":true,"x":1}')
    # every prefix of an accepted string is extensible; a dead prefix is not
    s = b'{"ans":42,"ok":true}'
    assert all(dfa.is_extensible_prefix(s[:k]) for k in range(len(s) + 1))
    assert not dfa.is_extensible_prefix(b'{"ans":true')
    # minimal: no two states are equivalent
    assert _equivalence_classes(dfa.trans, dfa.accept) == dfa.n_states
    # co-accessible: every state reaches an accepting state
    live = set(dfa.accept)
    changed = True
    while changed:
        changed = False
        for q in range(dfa.n_states):
            if q not in live and any(int(x) in live for x in dfa.trans[q] if x >= 0):
                live.add(q)
                changed = True
    assert live == set(range(dfa.n_states))
    # token masks: the stop token is allowed exactly in accepting states, and leads to the terminal state Q
    tb = token_bytes()
    dfa1 = APZ.compile_json_schema_dfa(SCHEMA)
    nxt, allow = dfa1.compile_token_masks(tb, (EOS,))
    Q = dfa1.n_states
    assert [bool(allow[q, EOS]) for q in range(Q)] == [q in dfa1.accept for q in range(Q)]
    assert all(int(nxt[q, EOS]) == Q for q in dfa1.accept)
    assert not allow[0, 40 + 1] and allow[0, 40]                      # first token must open the object


# ============================================================================ 2. fixed-length AEAD state blobs
def test_state_blob_fixed_length_and_tamper():
    key = b"\x55" * 32
    mid, sess, st = (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)
    base = {"st_pp": (1, 1, 1, 1), "hx": (2, 2, 2, 2), "hy": (3, 3, 3, 3), "r": (4, 4, 4, 4)}
    short = APZ.seal_state_blob(key, mid, sess, 0, st, {**base, "hist": [1]}, 16)
    long_ = APZ.seal_state_blob(key, mid, sess, 0, st, {**base, "hist": list(range(16))}, 16)
    assert len(short) == len(long_), (len(short), len(long_))
    assert APZ.open_state_blob(key, long_, mid, sess, 0, st, 16)["hist"] == list(range(16))

    def rejected(fn):
        try:
            fn()
        except Exception:
            return True
        return False
    raw = bytearray(bytes.fromhex(short))
    raw[20] ^= 1
    assert rejected(lambda: APZ.open_state_blob(key, bytes(raw).hex(), mid, sess, 0, st, 16))      # ciphertext bit
    assert rejected(lambda: APZ.open_state_blob(key, short, mid, sess, 1, st, 16))                 # other turn
    assert rejected(lambda: APZ.open_state_blob(key, short, mid, sess, 0, (9, 10, 11, 13), 16))    # other anchor
    assert rejected(lambda: APZ.open_state_blob(b"\x56" * 32, short, mid, sess, 0, st, 16))        # other key


# ============================================================================ 3. bundle verifier checks
def test_profile_registry():
    for name in ("qwen3-0.6b-sec315", "qwen3-0.6b-sec315-t1024", "qwen3-0.6b-demo", "qwen3-0.6b-mc4-mg2"):
        assert name in RB.BUILTIN_PROFILES, name
    demo = RB.get_registered_profile("qwen3-0.6b-demo")
    assert (demo["M_C"], demo["M_G"], demo["T_pad"], demo["security_tier"]) == (4, 2, 256, "demo_fast")
    prod = RB.get_registered_profile("qwen3-0.6b-sec315")
    assert (prod["M_C"], prod["M_G"]) == (210, 105)
    try:
        RB.register_profile({**prod, "M_C": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("a built-in profile was redefined")


def test_bundle_cross_checks():
    spec = fixture()[0]
    h = honest_reply()
    mid = mid_bytes()
    req, rep, nc = client_request(h), client_reply(h), h["nc"].hex()

    def check(stmt, req_=req, spec_=spec, nonce=nc, expect=None, allow_reduced=True):
        ok, why, _ = RB.verify_cli(
            mid, req_, rep, nonce, RB.pack_bundle(spec_, stmt, h["pf"]),
            allow_reduced_security=allow_reduced,
        )
        assert not ok and expect in why, (ok, why)

    # Default policy rejects non-production profiles unless --allow-reduced-security is explicitly set.
    check(h["stmt"], allow_reduced=False, expect="requires --allow-reduced-security")
    weak = {**RB.get_registered_profile("qwen3-0.6b-sec315"), "M_C": 2, "M_G": 2}
    check(h["stmt"], spec_=weak, expect="do not match registered profile")
    check(h["stmt"], spec_={**spec, "M_G": 0}, expect="do not match registered profile")
    injected = json.loads(json.dumps(h["stmt"]))
    injected["api_meta"]["schema"] = SCHEMA                       # the client asked for no schema
    check(injected, expect="schema")
    fake_eos = json.loads(json.dumps(h["stmt"]))
    fake_eos["api_meta"]["eos_token_id"] = h["vis"][0]
    check(fake_eos, expect="eos_token_id")
    tools = json.loads(json.dumps(h["stmt"]))
    tools["api_meta"]["tool_segments"] = [{"n_reason": 0, "call_tokens": [9, TOOL_END], "n_ret": 0, "ret_tokens": None}]
    check(tools, expect="tool_segments")
    check(h["stmt"], nonce=(b"\x33" * 16).hex(), expect="nonce")
    check(h["stmt"], req_=client_request(h, max_tokens=5), expect="max_tokens")


# ============================================================================ 4. end-to-end proofs
def _run_cli(tmp, h, nonce_hex, allow_reduced=True):
    paths = {k: os.path.join(tmp, k) for k in ("spec.json", "request.json", "reply.json", "reply.proof")}
    json.dump(fixture()[0], open(paths["spec.json"], "w"))
    json.dump(client_request(h), open(paths["request.json"], "w"))
    json.dump(client_reply(h), open(paths["reply.json"], "w"))
    open(paths["reply.proof"], "wb").write(h["blob"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, VERIFY_CLI, "--model-id", mid_bytes().hex(), "--request", paths["request.json"],
           "--reply", paths["reply.json"], "--nonce", nonce_hex, "--proof", paths["reply.proof"],
           "--extra-profile", paths["spec.json"], "--profile", PROFILE]
    if allow_reduced:
        cmd.append("--allow-reduced-security")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    lines = p.stdout.strip().splitlines()
    return p.returncode, (lines[-1] if lines else p.stderr.strip()[-500:])


def test_honest_reply_cli_verified():
    h = honest_reply()
    with tempfile.TemporaryDirectory() as tmp:
        rc0, out0 = _run_cli(tmp, h, h["nc"].hex(), allow_reduced=False)
        assert rc0 == 1 and "--allow-reduced-security" in out0, (rc0, out0)
        rc, out = _run_cli(tmp, h, h["nc"].hex(), allow_reduced=True)
        assert rc == 0 and out == "VERIFIED", (rc, out)
        rc, out = _run_cli(tmp, h, (b"\x33" * 16).hex(), allow_reduced=True)  # replay under another client nonce
        assert rc == 1 and out.startswith("REJECTED"), (rc, out)


def test_int4_weights_rejected():
    _, prf, model, com = fixture()
    h = honest_reply()
    cap = model.forward(h["seq"], cheat={"int4": True})        # the engine ran int4 weights; model_id commits bf16
    stmt, wit = reply_statement(prf, model, com, cap, h["api"], h["cot"], h["nc"])
    _, res = prove_and_verify(prf, stmt, wit, cheat=True)
    assert res.startswith("rejected"), res


def test_stop_token_hidden_in_reasoning_rejected():
    _, prf, _, _ = fixture()
    model = tiny_model(eos_boost=True)
    com = RZ.commit_model(model.tensors(), prf, L_real=model.L, V_real=V, sys_prompt=SYS, k_prf=KPRF)
    for i in range(500):
        nc = i.to_bytes(16, "little")
        pre = S3.prefix_state(KPRF, samp_ctx(nc))
        seq, em = rollout(model, SYS + REQ, 8, pre, stop_ids=(EOS, TOOL_END))
        if len(em) >= 2 and em[-1] == EOS and TOOL_END not in em:
            break
    else:
        raise AssertionError("no rollout ending in EOS found")
    # the engine sampled EOS, the provider hides it inside the reasoning and keeps generating one visible token
    _, extra = rollout(model, seq, 1, pre, g_offset=len(em))
    cot = list(em)
    vis = [extra[0]] if extra[0] not in (EOS, TOOL_END) else [10]
    api = {"req_tokens": REQ, "n_reason": len(cot), "vis_tokens": vis,
           "api_meta": {"finish_reason": "length", "max_tokens": len(cot) + len(vis), "eos_token_id": EOS}}
    stmt, wit = reply_statement(prf, model, com, model.forward(SYS + REQ + cot + vis), api, cot, nc)
    _, res = prove_and_verify(prf, stmt, wit, cheat=True)
    assert res.startswith("rejected"), res


def test_constrained_json_and_wrong_mask():
    spec, _, model, com = fixture()
    tb = token_bytes()
    prf = RB.build_profile(spec, token_bytes=tb)
    dfa = APZ.compile_json_schema_dfa(SCHEMA)
    next_q, allow = dfa.compile_token_masks(tb, (EOS,))
    for i in range(256):
        nc = (5000 + i).to_bytes(16, "little")
        pre = S3.prefix_state(KPRF, samp_ctx(nc))
        seq = list(SYS + REQ)
        cot_tok = int(S3.sample(pre, 0, np.asarray(model.forward(seq).logits[-1]).view(np.float32), RULE))
        if cot_tok in (EOS, TOOL_END):
            continue
        seq.append(cot_tok)
        vis, q = [], 0
        for k in range(1, 10):                               # constrained steps follow the hidden reasoning token
            lg = np.asarray(model.forward(seq).logits[-1]).view(np.float32)
            tok = int(S3.sample(pre, k, lg, RULE, allowed=allow[q, :V]))
            seq.append(tok)
            vis.append(tok)
            q = int(next_q[q, tok])
            if tok == EOS:
                break
        if vis and vis[-1] == EOS and len(seq) <= prf.T_pad:
            break
    else:
        raise AssertionError("no constrained rollout found")
    assert json.loads(b"".join(tb[t] for t in vis[:-1]).decode()) in ({"ok": True}, {"ok": False})
    api = {"req_tokens": REQ, "n_reason": 1, "vis_tokens": vis,
           "api_meta": {"finish_reason": "stop", "max_tokens": 12, "eos_token_id": EOS, "schema": SCHEMA}}
    cap = model.forward(seq)
    stmt, wit = reply_statement(prf, model, com, cap, api, [cot_tok], nc)
    _, res = prove_and_verify(prf, stmt, wit, cheat=False, fast_honest_build_only=True)
    assert res == "accepted", res

    # same tokens, but the prover feeds the sampler an all-ones mask on a drawn constrained step
    for g in range(64):
        stmt, wit = reply_statement(prf, model, com, cap, api, [cot_tok], nc, salts=((g + 301, 0, 0, 0), (7, 0, 0, 0)))
        drawn = RZ.gen_steps_of(stmt["u_gen"], 1 + len(vis))
        if any(s >= 1 for s in drawn):
            break
    else:
        raise AssertionError("no salt draws a constrained generation step")
    wit["gen_mask_override"] = {next(i for i, s in enumerate(drawn) if s >= 1): np.ones(V, dtype=bool)}
    _, res = prove_and_verify(prf, stmt, wit, cheat=True)
    assert res.startswith("rejected"), res


TESTS = [test_policy_and_schema_dfa, test_state_blob_fixed_length_and_tamper, test_profile_registry,
         test_bundle_cross_checks, test_honest_reply_cli_verified, test_int4_weights_rejected,
         test_stop_token_hidden_in_reasoning_rejected, test_constrained_json_and_wrong_mask]


def main():
    import time
    fails = []
    for t in TESTS:
        t0 = time.perf_counter()
        try:
            t()
            print(f"PASS {t.__name__} ({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}", flush=True)
            fails.append(t.__name__)
    print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
