# SPDX-License-Identifier: Apache-2.0
"""Wire bundle, pinned profile registry, deterministic public request tokenizer, and the standalone CPU verification
entry point for `flashprove_next.reply_zk` proofs (bundle magic `fpn-reply-zk-v1`, header `FPNZK1\\n`).

A bundle carries `(profile_spec, stmt, pf)`.  The verifier never trusts the bundle's profile: it looks the name up in
its own pinned registry and requires an exact match, so a server cannot weaken slot counts, the sampling rule or the
stop tokens.  Everything the statement claims about the API exchange is recomputed from the CLIENT's raw request
(tokens, nonce, turn, schema, max_tokens, tools) before the proof itself is checked.

Used by:
  * `fpn-verify` / `verifier/fpn_verify.py` (standalone CPU verifier, `CUDA_VISIBLE_DEVICES=""`, no weights or secret
    configuration);
  * `integrations/vllm/prover_daemon.py` (packages bundles and runs an isolated CPU self-check);
  * `integrations/vllm/end_to_end_vllm.py` (client verification straight from the client's raw request JSON).
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict

import numpy as np

from . import api_zk as APZ
from . import modelcommit as MC
from . import reply_zk as RZ
from . import sampling_v3 as S3
from . import state_zk as SZ
from . import zk_flat as Z
from .field import P
from .transcript import dumps, loads

BUNDLE_MAGIC = "fpn-reply-zk-v1"
HEADER_PREFIX = b"FPNZK1\n"
DEFAULT_MAX_TOKENS = 32
QWEN3_EOS_ID = 151645
QWEN3_TOOL_END_ID = 151658
QWEN3_THINK_END_ID = 151668


def mid_to_hex(mid_words) -> str:
    assert len(mid_words) == 4
    return b"".join(int(w).to_bytes(8, "little") for w in mid_words).hex()


def hex_to_mid(mid_hex: str) -> list[int]:
    raw = bytes.fromhex(mid_hex)
    if len(raw) != 32:
        raise ValueError("model_id must be 32 bytes (64 hex chars)")
    words = [int.from_bytes(raw[8 * i:8 * (i + 1)], "little") for i in range(4)]
    if any(w >= P for w in words):
        raise ValueError("model_id word not canonical mod P")
    return words


def qwen_rope_fn(dh: int, rope_theta: float):
    from .fp_ref import _exact_bf16_cos_sin, _exact_rope_inv_f32
    inv = _exact_rope_inv_f32(int(dh), float(rope_theta))
    f32 = lambda c: (np.asarray(c, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def rope(pos):
        p = int(pos)
        hit = cache.get(p)
        if hit is None:
            ang = np.float32(p) * inv
            cu, su = _exact_bf16_cos_sin(ang)
            hit = (f32(np.concatenate([cu, cu])), f32(np.concatenate([su, su])))
            cache[p] = hit
        return hit

    return rope


def first_eos_id(eos_cfg, fallback: int = QWEN3_EOS_ID) -> int:
    if isinstance(eos_cfg, (list, tuple)):
        return int(eos_cfg[0]) if eos_cfg else int(fallback)
    return int(fallback if eos_cfg is None else eos_cfg)


def render_public_chat_text(body: dict) -> tuple[str, bool]:
    """Deterministically render public OpenAI / Anthropic request fields (`messages` | `input` | `prompt` + optional
    top-level `system`) into the exact chat-template prompt text (EXCLUDING the server's hidden `sys_tokens`).
    Shared by both the serving endpoint and the standalone client verifier so the client never trusts server-echoed
    `prompt_token_ids`."""
    ta = body.get("fpn") or {}
    enable_thinking = bool(ta.get("thinking", body.get("enable_thinking", False)))

    parts: list[str] = []
    # Anthropic top-level `system` (public caller system instruction, kept distinct from hidden committed sys_tokens)
    top_sys = body.get("system")
    if isinstance(top_sys, str) and top_sys:
        parts.append(f"<|im_start|>user\n{top_sys}<|im_end|>\n")
    elif isinstance(top_sys, list):
        stxt = "".join(b.get("text", "") for b in top_sys if isinstance(b, dict))
        if stxt:
            parts.append(f"<|im_start|>user\n{stxt}<|im_end|>\n")

    msgs = body.get("messages")
    if msgs is None and "input" in body:
        inp = body.get("input")
        if isinstance(inp, str):
            msgs = [{"role": "user", "content": inp}]
        elif isinstance(inp, list):
            msgs = inp

    if isinstance(msgs, list):
        for m in msgs:
            role = m.get("role", "user")
            role_tag = "user" if role == "system" else role
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            parts.append(f"<|im_start|>{role_tag}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        if not enable_thinking:
            parts.append("<think>\n\n</think>\n\n")
        return "".join(parts), enable_thinking

    prompt_str = str(body.get("prompt", ""))
    parts.append(f"<|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n")
    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts), enable_thinking


def encode_text_tokens(tokenizer_dir: str, text: str) -> list[int]:
    """Encode `text` with `add_special_tokens=False` using fast Rust `tokenizers` (no torch/GPU) or HF fallback."""
    tok_json = os.path.join(tokenizer_dir, "tokenizer.json") if os.path.isdir(tokenizer_dir) else tokenizer_dir
    if os.path.exists(tok_json):
        try:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(tok_json)
            return [int(x) for x in tk.encode(text, add_special_tokens=False).ids]
        except ImportError:
            pass
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return [int(x) for x in tok.encode(text, add_special_tokens=False)]


def encode_public_request_tokens(tokenizer_dir: str, body: dict) -> tuple[list[int], bool]:
    ta = body.get("fpn") or {}
    enable_thinking = bool(ta.get("thinking", body.get("enable_thinking", False)))
    if isinstance(body.get("prompt_token_ids"), list):
        return [int(x) for x in body["prompt_token_ids"]], enable_thinking
    txt, enable_thinking = render_public_chat_text(body)
    return encode_text_tokens(tokenizer_dir, txt), enable_thinking


def token_bytes_from_dir(tokenizer_dir: str, V: int) -> list[bytes | None]:
    """Build canonical UTF-8 byte sequences for token IDs 0..V-1 directly from `tokenizer.json` (no torch)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}

    tok_json = os.path.join(tokenizer_dir, "tokenizer.json") if os.path.isdir(tokenizer_dir) else tokenizer_dir
    if os.path.exists(tok_json):
        data = json.load(open(tok_json, "r", encoding="utf-8"))
        added_special = {
            int(ent["id"]) for ent in (data.get("added_tokens") or [])
            if ent.get("special", False) and isinstance(ent.get("id"), int)
        }
        vocab = (data.get("model") or {}).get("vocab") or {}
        id_to_str: dict[int, str] = {int(idx): tok_s for tok_s, idx in vocab.items() if isinstance(idx, int)}
        for ent in data.get("added_tokens") or []:
            if isinstance(ent.get("id"), int) and isinstance(ent.get("content"), str):
                id_to_str[int(ent["id"])] = ent["content"]
        out: list[bytes | None] = [None] * V
        for i in range(V):
            if i in added_special:
                out[i] = None
                continue
            s = id_to_str.get(i)
            if s is None:
                out[i] = None
            elif all(ch in dec for ch in s):
                out[i] = bytes(dec[ch] for ch in s)
            else:
                out[i] = s.encode("utf-8")
        return out

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return APZ.token_bytes_from_hf_tokenizer(tok, V)


def _qwen3_06b_spec(name: str, *, m_c: int, m_g: int, t_pad: int = 256, security_tier: str = "production") -> dict:
    rule = S3.Rule(0.6, 0.95, 32)
    eps_bits = int(np.array([1e-6], dtype=np.float32).view(np.uint32)[0])
    return {
        "name": name,
        "security_tier": security_tier,
        "d": 1024, "n_q": 16, "n_kv": 8, "dh": 128, "dff": 3072,
        "L_MAX": 32,
        "T_pad": int(t_pad),
        "M_C": int(m_c),
        "M_G": int(m_g),
        "MQ_H": 268,
        "s_bits": 1,
        "SYS_CAP": 32,
        "VEC_PAD": 65536,
        "HIST_CAP": int(t_pad),
        "TOOL_CAP": 32,
        "QN_CAP": 12.0,
        "KN_CAP": 1100.0,
        "EXP_BITS": 17,
        "V": 151936,
        "eps_bits": eps_bits,
        "rope_theta": 1000000.0,
        "eos_token_id": QWEN3_EOS_ID,
        "tool_end_token_id": QWEN3_TOOL_END_ID,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "rule": rule.as_dict(),
        "prm": asdict(MC.QWEN_PARAMS_R),
    }


# Verifier-side profile registry.  A proof bundle names its profile; the verifier uses ITS OWN copy of that profile and
# rejects any bundle whose embedded spec differs, so an untrusted bundle cannot weaken M_C, M_G, MQ, MQ_H, the sampling
# rule or the stop token ids.  "production" profiles carry the full spot-check budget; "demo_fast" / "dev_fast" profiles
# use a handful of slots for demonstrations and integration tests and give correspondingly weaker soundness.
REGISTERED_PROFILES: dict[str, dict] = {
    "qwen3-0.6b-sec315": _qwen3_06b_spec("qwen3-0.6b-sec315", m_c=210, m_g=105, t_pad=256, security_tier="production"),
    "qwen3-0.6b-sec315-t1024": _qwen3_06b_spec("qwen3-0.6b-sec315-t1024", m_c=210, m_g=105, t_pad=1024, security_tier="production"),
    "qwen3-0.6b-mc4-mg2": _qwen3_06b_spec("qwen3-0.6b-mc4-mg2", m_c=4, m_g=2, t_pad=256, security_tier="dev_fast"),
    "qwen3-0.6b-demo": _qwen3_06b_spec("qwen3-0.6b-demo", m_c=4, m_g=2, t_pad=256, security_tier="demo_fast"),
}
BUILTIN_PROFILES = frozenset(REGISTERED_PROFILES)


def register_profile(spec: dict) -> str:
    """Add a verifier-chosen profile spec (e.g. a test-model profile loaded from a local file by `fpn-verify
    --extra-profile`).  The spec comes from the VERIFIER, never from a bundle; built-in profile names cannot be
    redefined.  Returns the profile name."""
    name = str(spec.get("name", ""))
    if not name:
        raise ValueError("profile spec has no name")
    if name in BUILTIN_PROFILES:
        raise ValueError(f"cannot redefine built-in profile {name!r}")
    REGISTERED_PROFILES[name] = copy.deepcopy(spec)
    return name


def get_registered_profile(name: str) -> dict:
    if name not in REGISTERED_PROFILES:
        raise KeyError(f"unknown or unregistered profile {name!r}")
    return copy.deepcopy(REGISTERED_PROFILES[name])


def profile_spec_from_hf_config(
    cfg: dict,
    *,
    name: str,
    vec_pad: int,
    t_pad: int,
    m_c: int,
    m_g: int,
    sys_cap: int = 32,
    hist_cap: int = 256,
    tool_cap: int = 32,
    tool_end_token_id: int = QWEN3_TOOL_END_ID,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    rule: S3.Rule | None = None,
    prm: MC.Params | None = None,
) -> dict:
    if name in REGISTERED_PROFILES:
        reg = get_registered_profile(name)
        assert reg["M_C"] == int(m_c) and reg["M_G"] == int(m_g) and reg["T_pad"] == int(t_pad), (
            f"registered profile {name!r} parameter mismatch"
        )
        return reg
    prm = MC.QWEN_PARAMS_R if prm is None else prm
    rule = S3.Rule(0.6, 0.95, 32) if rule is None else rule
    d = int(cfg["hidden_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    dh = int(cfg.get("head_dim") or d // n_q)
    dff = int(cfg["intermediate_size"])
    V = int(cfg["vocab_size"])
    eps_bits = int(np.array([cfg["rms_norm_eps"]], dtype=np.float32).view(np.uint32)[0])
    rope_theta = float(cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta", 1000000.0))
    return {
        "name": str(name),
        "security_tier": "custom",
        "d": d, "n_q": n_q, "n_kv": n_kv, "dh": dh, "dff": dff,
        "L_MAX": 32,
        "T_pad": int(t_pad),
        "M_C": int(m_c),
        "M_G": int(m_g),
        "MQ_H": 268,
        "s_bits": 1,
        "SYS_CAP": int(sys_cap),
        "VEC_PAD": int(vec_pad),
        "HIST_CAP": int(hist_cap),
        "TOOL_CAP": int(tool_cap),
        "QN_CAP": 12.0,
        "KN_CAP": 1100.0,
        "EXP_BITS": 17,
        "V": V,
        "eps_bits": eps_bits,
        "rope_theta": rope_theta,
        "eos_token_id": first_eos_id(cfg.get("eos_token_id")),
        "tool_end_token_id": int(tool_end_token_id),
        "default_max_tokens": int(default_max_tokens),
        "rule": rule.as_dict(),
        "prm": asdict(prm),
    }


def build_profile(spec: dict, token_bytes: list[bytes | None] | None = None) -> RZ.Profile:
    """Instantiate the prover / verifier profile of a spec.  Rotary tables follow the Hugging Face fp32 recipe of
    `rope_theta` (`qwen_rope_fn`); a spec with `"rope_recipe": "reference"` uses the float64 reference table of
    `fp_ref.rope_table` instead (the deterministic test engine `dense_engine.TinyDense`)."""
    prm = MC.Params(**spec["prm"])
    rd = spec["rule"]
    rule = S3.Rule(rd["temperature"], rd["top_p"], int(rd.get("beta", 32)))
    dh = int(spec["dh"])
    rope_theta = float(spec["rope_theta"])
    return RZ.Profile(
        d=int(spec["d"]),
        n_q=int(spec["n_q"]),
        n_kv=int(spec["n_kv"]),
        dh=dh,
        dff=int(spec["dff"]),
        L_MAX=int(spec["L_MAX"]),
        T_pad=int(spec["T_pad"]),
        M_C=int(spec["M_C"]),
        M_G=int(spec["M_G"]),
        prm=prm,
        MQ_H=int(spec["MQ_H"]),
        s_bits=int(spec["s_bits"]),
        SYS_CAP=int(spec["SYS_CAP"]),
        VEC_PAD=int(spec["VEC_PAD"]),
        QN_CAP=float(spec["QN_CAP"]),
        KN_CAP=float(spec["KN_CAP"]),
        EXP_BITS=int(spec["EXP_BITS"]),
        attention=True,
        V=int(spec["V"]),
        HIST_CAP=int(spec["HIST_CAP"]),
        TOOL_CAP=int(spec.get("TOOL_CAP", 0)),
        rule=rule,
        rope_fn=None if spec.get("rope_recipe") == "reference" else qwen_rope_fn(dh, rope_theta),
        eps_bits=int(spec["eps_bits"]),
        token_bytes=token_bytes,
    )


def pack_bundle(profile_spec: dict, stmt: dict, pf: dict, pad_to_bytes: int = 0) -> bytes:
    """Serialize `(profile_spec, stmt, pf)` into a `FPNZK1\\n` + gzip payload (optionally padded to fixed container size)."""
    body = dumps({"magic": BUNDLE_MAGIC, "profile_spec": profile_spec, "stmt": stmt, "pf": pf})
    hdr = HEADER_PREFIX + len(body).to_bytes(8, "little")
    raw = hdr + body
    if pad_to_bytes > 0:
        if len(raw) > pad_to_bytes:
            raise ValueError(f"proof bundle {len(raw)} B exceeds container size {pad_to_bytes} B")
        raw = raw + os.urandom(pad_to_bytes - len(raw))
    return raw


def is_reply_zk_blob(blob: bytes) -> bool:
    return isinstance(blob, (bytes, bytearray)) and bytes(blob[:len(HEADER_PREFIX)]) == HEADER_PREFIX


def unpack_bundle(blob: bytes) -> dict:
    b = bytes(blob)
    if not is_reply_zk_blob(b) or len(b) < len(HEADER_PREFIX) + 8:
        raise ValueError("not a FPNZK1 reply_zk bundle")
    ln = int.from_bytes(b[len(HEADER_PREFIX):len(HEADER_PREFIX) + 8], "little")
    off = len(HEADER_PREFIX) + 8
    if ln <= 0 or off + ln > len(b):
        raise ValueError("corrupted FPNZK1 bundle length")
    d = loads(b[off:off + ln])
    if not isinstance(d, dict) or d.get("magic") != BUNDLE_MAGIC:
        raise ValueError("invalid FPNZK1 magic")
    return d


def _expected_schema_spec(req: dict) -> tuple[dict | None, str, str, bool]:
    """Derive canonical `(schema, prefix_hex, suffix_hex, schema_on_tool_call)` strictly from the client's request."""
    sch = APZ.extract_request_schema(req)
    if sch is not None:
        return sch, "", "", False
    tsch = APZ.extract_strict_tool_schema(req)
    if tsch is not None:
        return tsch, b"<tool_call>\n".hex(), b"\n".hex(), True
    if req.get("schema") is not None:
        return (
            req["schema"],
            str(req.get("schema_prefix_hex", "")),
            str(req.get("schema_suffix_hex", "")),
            bool(req.get("schema_on_tool_call", False)),
        )
    return None, "", "", False


def verify_cli(
    model_id_bytes: bytes,
    req: dict,
    rep: dict,
    nonce_hex: str,
    blob: bytes,
    allowed_profiles: list[str] | None = None,
    tokenizer_dir: str | None = None,
    allow_reduced_security: bool = False,
) -> tuple[bool, str, dict]:
    """Standalone CPU verification of a `FPNZK1` proof bundle against public `(model_id, req, rep, nonce)`.
    By default only `security_tier == 'production'` profiles are accepted; non-production demo/dev/test profiles
    require `allow_reduced_security=True` so an untrusted server cannot downgrade the spot-check budget."""
    t0 = time.perf_counter()
    try:
        bundle = unpack_bundle(blob)
        untrusted_spec = bundle["profile_spec"]
        stmt = bundle["stmt"]
        pf = bundle["pf"]
    except Exception as e:
        return False, f"malformed bundle ({type(e).__name__}: {e})", {}

    prof_name = str((untrusted_spec or {}).get("name", ""))
    if allowed_profiles and prof_name not in allowed_profiles:
        return False, f"profile {prof_name!r} not in allowed profiles {allowed_profiles}", {"profile": prof_name}
    if prof_name not in REGISTERED_PROFILES:
        return False, f"unregistered profile {prof_name!r}", {"profile": prof_name}
    spec = get_registered_profile(prof_name)
    if untrusted_spec != spec:
        return False, f"bundle profile_spec parameters do not match registered profile {prof_name!r}", {"profile": prof_name}
    if spec.get("security_tier") != "production" and not allow_reduced_security:
        return (
            False,
            f"non-production profile {prof_name!r} (tier={spec.get('security_tier')!r}) requires --allow-reduced-security",
            {"profile": prof_name, "security_tier": spec.get("security_tier")},
        )

    try:
        exp_mid = hex_to_mid(model_id_bytes.hex())
    except Exception as e:
        return False, f"bad model_id ({e})", {}
    if [int(x) for x in stmt.get("model_id", [])] != exp_mid:
        return False, "statement model_id != expected model_id", {"profile": prof_name}

    # Client-side request tokenization when raw request fields (`messages` | `input` | `prompt`) are supplied.
    has_raw_prompt = any(k in req for k in ("messages", "input", "prompt", "prompt_token_ids"))
    if has_raw_prompt:
        if "prompt_token_ids" in req and not any(k in req for k in ("messages", "input", "prompt")):
            exp_req_toks = [int(x) for x in req["prompt_token_ids"]]
        else:
            if not tokenizer_dir:
                return False, "raw chat/prompt request verification requires --tokenizer DIR", {"profile": prof_name}
            exp_req_toks, _ = encode_public_request_tokens(tokenizer_dir, req)
            if "tokens" in req and [int(x) for x in req["tokens"]] != exp_req_toks:
                return False, "request tokens != client-rendered messages tokens", {"profile": prof_name}
    elif "tokens" in req:
        exp_req_toks = [int(x) for x in req["tokens"]]
    else:
        return False, "request JSON missing messages / prompt / tokens", {"profile": prof_name}

    rep_toks = [int(x) for x in rep["tokens"]]
    if [int(x) for x in stmt.get("req_tokens", [])] != exp_req_toks:
        return False, "statement req_tokens != request tokens", {"profile": prof_name}
    if [int(x) for x in stmt.get("vis_tokens", [])] != rep_toks:
        return False, "statement vis_tokens != reply tokens", {"profile": prof_name}
    if "n_reason" in rep and int(stmt.get("n_reason", -1)) != int(rep["n_reason"]):
        return False, "statement n_reason != reply n_reason", {"profile": prof_name}

    st_s = stmt.get("st") or {}
    nc_bytes = bytes.fromhex(nonce_hex)
    if bytes.fromhex(st_s.get("N_c", "")) != nc_bytes:
        return False, "statement client nonce N_c != expected nonce", {"profile": prof_name}
    ta_req = req.get("fpn") or {}
    exp_turn = int(req.get("turn", ta_req.get("turn", 0)))
    if int(st_s.get("turn", -1)) != exp_turn:
        return False, "statement turn != request turn", {"profile": prof_name}
    if exp_turn == 0:
        v_rel = SZ.v_rel_of(RZ.VERSION)
        exp_st_prev = list(SZ.genesis(v_rel, SZ.sess_of(nc_bytes)))
        if "St_prev" in req and [int(x) for x in req["St_prev"]] != exp_st_prev:
            return False, "turn-0 St_prev != canonical genesis anchor", {"profile": prof_name}
        if [int(x) for x in st_s.get("St_prev", [])] != exp_st_prev:
            return False, "statement turn-0 St_prev != canonical genesis anchor", {"profile": prof_name}
    else:
        req_st_prev = req.get("St_prev", ta_req.get("St_prev"))
        if req_st_prev is None:
            return False, "multi-turn verification (turn > 0) requires client's stored St_prev in request", {"profile": prof_name}
        if [int(x) for x in st_s.get("St_prev", [])] != [int(x) for x in req_st_prev]:
            return False, "statement St_prev != client's stored state anchor", {"profile": prof_name}
    if "St_new" in rep and [int(x) for x in st_s.get("St_new", [])] != [int(x) for x in rep["St_new"]]:
        return False, "statement St_new != reply St_new", {"profile": prof_name}

    # Strict cross-check of ALL `api_meta` fields against the client's request and pinned profile.
    am_stmt = stmt.get("api_meta") or {}
    want_sch, want_pre, want_suf, want_on_tool = _expected_schema_spec(req)
    if am_stmt.get("schema") != want_sch:
        return False, "statement schema != request schema", {"profile": prof_name}
    if str(am_stmt.get("schema_prefix_hex", "")) != want_pre or str(am_stmt.get("schema_suffix_hex", "")) != want_suf:
        return False, "statement schema prefix/suffix != canonical request wrapper", {"profile": prof_name}
    if bool(am_stmt.get("schema_on_tool_call", False)) != want_on_tool:
        return False, "statement schema_on_tool_call != request tool schema mode", {"profile": prof_name}

    allow_srv_tools = bool(req.get("server_tools") or ta_req.get("server_tools"))
    stmt_segs = am_stmt.get("tool_segments") or []
    if not allow_srv_tools and stmt_segs:
        return False, "statement contains tool_segments on a request without server_tools", {"profile": prof_name}
    if allow_srv_tools:
        if "tool_segments" in rep and rep["tool_segments"] is not None and stmt_segs != rep["tool_segments"]:
            return False, "statement tool_segments != reply tool_segments", {"profile": prof_name}
        for seg in stmt_segs:
            if seg.get("ret_tokens") is None and int(seg.get("n_ret", -1)) != int(spec["TOOL_CAP"]):
                return False, "private server-tool segment n_ret != pinned TOOL_CAP", {"profile": prof_name}

    has_tools = bool(req.get("tools")) or allow_srv_tools
    if am_stmt.get("finish_reason") is not None or ("finish_reason" in rep and rep["finish_reason"] is not None):
        if "finish_reason" in rep and rep["finish_reason"] is not None:
            if APZ.canon_finish_reason(am_stmt.get("finish_reason")) != APZ.canon_finish_reason(rep["finish_reason"]):
                return False, "statement finish_reason != reply finish_reason", {"profile": prof_name}
        raw_mt = req.get("max_tokens")
        if raw_mt is None:
            raw_mt = req.get("max_completion_tokens")
        if raw_mt is None:
            raw_mt = req.get("max_output_tokens")
        exp_mt = int(spec["default_max_tokens"]) if raw_mt is None else int(raw_mt)
        if int(am_stmt.get("max_tokens", -1)) != exp_mt:
            return False, f"statement max_tokens ({am_stmt.get('max_tokens')}) != expected ({exp_mt})", {"profile": prof_name}
        if int(am_stmt.get("eos_token_id", -1)) != int(spec["eos_token_id"]):
            return False, "statement eos_token_id != pinned profile eos_token_id", {"profile": prof_name}
        if has_tools:
            if int(am_stmt.get("tool_end_token_id", -1)) != int(spec["tool_end_token_id"]):
                return False, "statement tool_end_token_id != pinned profile tool_end_token_id", {"profile": prof_name}
        else:
            if am_stmt.get("tool_end_token_id") is not None:
                return False, "unexpected tool_end_token_id on request without tools", {"profile": prof_name}

    tok_bytes = None
    if am_stmt.get("schema") is not None:
        if not tokenizer_dir:
            return False, "constrained-decoding proof requires --tokenizer DIR", {"profile": prof_name}
        tok_bytes = token_bytes_from_dir(tokenizer_dir, int(spec["V"]))

    try:
        prf = build_profile(spec, token_bytes=tok_bytes)
        RZ.verify(prf, stmt, pf)
    except Z.ProofError as e:
        return False, str(e), {"profile": prof_name, "verify_s": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return False, f"malformed proof ({type(e).__name__}: {e})", {"profile": prof_name}

    return True, "accepted", {
        "profile": prof_name,
        "security_tier": spec["security_tier"],
        "M_C": int(spec["M_C"]),
        "M_G": int(spec["M_G"]),
        "T_pad": int(spec["T_pad"]),
        "St_new": list(st_s.get("St_new", [])),
        "verify_s": round(time.perf_counter() - t0, 3),
        "proof_bytes": len(blob),
    }
