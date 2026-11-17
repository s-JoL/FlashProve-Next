# SPDX-License-Identifier: Apache-2.0
"""API-level protocol semantics, constrained decoding (JSON Schema -> minimal byte DFA + in-circuit full-vocab mask
binding), termination / finish_reason verification, multi-segment tool-return witness helpers, public policy pre-check
Phi_pre, and AEAD multi-turn state sealing (specs/01 §6.2-6.4, specs/05 §4.1-4.6).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import gadgets as G
from . import mc_zk as MZ
from .field import CPU, ONE, P, ZERO, hadd, hmul, hsub
from .zk_flat import Ex, SelLt

Q_MAX = 128   # profile cap on compiled JSON-Schema DFA states (specs/05 §4.2)

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL = "tool_use"
FINISH_ALIASES = {
    "stop": FINISH_STOP,
    "end_turn": FINISH_STOP,
    "eos": FINISH_STOP,
    "length": FINISH_LENGTH,
    "max_tokens": FINISH_LENGTH,
    "max_output_tokens": FINISH_LENGTH,
    "tool_use": FINISH_TOOL,
    "tool_calls": FINISH_TOOL,
}

SUPPORTED_SCHEMA_KEYS = frozenset({
    "type", "properties", "required", "additionalProperties",
    "items", "minItems", "maxItems",
    "enum", "const", "anyOf",
    "minimum", "maximum", "minLength", "maxLength",
    "title", "description", "default",
    "$defs", "definitions", "$ref", "$schema",
})

ALLOWED_API_META_KEYS = frozenset({
    "finish_reason", "max_tokens", "eos_token_id", "tool_end_token_id",
    "tool_segments", "schema", "schema_on_tool_call",
    "schema_prefix_hex", "schema_suffix_hex",
})

MULTIMODAL_BLOCK_TYPES = frozenset({
    "image", "image_url", "input_image", "audio", "input_audio", "video", "file", "document",
})


def canon_finish_reason(fr: str | None) -> str | None:
    if fr is None:
        return None
    out = FINISH_ALIASES.get(str(fr).strip().lower())
    if out is None:
        raise ValueError(f"unsupported finish_reason: {fr!r}")
    return out


# ============================================================================ 1. Public Pre-Check Predicate Phi_pre
def extract_request_schema(req: dict) -> dict | None:
    """Extract strict JSON Schema from OpenAI Chat (`response_format`), OpenAI Responses (`text.format`),
    or Anthropic/OpenAI strict single-tool / union-tool schema when structured constraint is active."""
    rf = req.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        if js.get("strict", True) and isinstance(js.get("schema"), dict):
            return js["schema"]
    tf = (req.get("text") or {}).get("format") if isinstance(req.get("text"), dict) else None
    if isinstance(tf, dict) and tf.get("type") == "json_schema":
        if tf.get("strict", True) and isinstance(tf.get("schema"), dict):
            return tf["schema"]
    return None


def extract_strict_tool_schema(req: dict) -> dict | None:
    """If request declares `tools` with `strict: true`, build the canonical {"name": ..., "arguments": ...} union schema."""
    tools = req.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    strict_variants = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" and isinstance(t.get("function"), dict) else t
        if not fn.get("strict"):
            continue
        name = fn.get("name")
        params = fn.get("parameters") or fn.get("input_schema")
        if isinstance(name, str) and isinstance(params, dict):
            strict_variants.append({
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            })
    if not strict_variants:
        return None
    return strict_variants[0] if len(strict_variants) == 1 else {"anyOf": strict_variants}


def check_public_policy(req: dict, committed_rule=None, q_max: int = Q_MAX) -> tuple[bool, str]:
    """Deterministic public pre-check Phi_pre(req) (specs/05 §4.2-4.6).
    Depends ONLY on the public request and public model rule; never touches any secret."""
    if not isinstance(req, dict):
        return False, "invalid_request_object"
    # 1. Custom string stop sequences: publicly excluded from proving (specs/05 §4.6)
    for k in ("stop", "stop_sequences"):
        sv = req.get(k)
        if isinstance(sv, str) and len(sv) > 0:
            return False, "unsupported_public_policy:custom_stop_sequence"
        if isinstance(sv, (list, tuple)) and len(sv) > 0:
            return False, "unsupported_public_policy:custom_stop_sequence"
    # 2. logprobs
    if req.get("logprobs") or req.get("top_logprobs"):
        return False, "unsupported_public_policy:logprobs"
    inc = req.get("include")
    if isinstance(inc, list) and any("logprobs" in str(x) for x in inc):
        return False, "unsupported_public_policy:logprobs"
    # 3. User custom sampling overrides / penalties / logit_bias
    for k in ("logit_bias", "bad_words", "allowed_token_ids"):
        if req.get(k):
            return False, f"unsupported_public_policy:{k}"
    for k in ("presence_penalty", "frequency_penalty"):
        if req.get(k) not in (None, 0, 0.0):
            return False, f"unsupported_public_policy:{k}"
    if req.get("repetition_penalty") not in (None, 1, 1.0):
        return False, "unsupported_public_policy:repetition_penalty"
    if req.get("top_k") not in (None, -1):
        return False, "unsupported_public_policy:top_k"
    if committed_rule is not None:
        if req.get("temperature") is not None and abs(float(req["temperature"]) - float(committed_rule.temperature)) > 1e-6:
            return False, "unsupported_public_policy:custom_temperature"
        if req.get("top_p") is not None and abs(float(req["top_p"]) - float(committed_rule.top_p)) > 1e-6:
            return False, "unsupported_public_policy:custom_top_p"
    if req.get("n") not in (None, 1):
        return False, "unsupported_public_policy:multi_candidate_n"
    rf = req.get("response_format")
    if isinstance(rf, dict) and rf.get("type") not in (None, "text", "json_schema"):
        return False, f"unsupported_public_policy:response_format:{rf.get('type')}"
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        if not js.get("strict", True) or not isinstance(js.get("schema"), dict):
            return False, "unsupported_public_policy:non_strict_json_schema"
    if req.get("tool_choice") == "none" and req.get("tools"):
        return False, "unsupported_public_policy:tool_choice_none_with_tools"
    # 4. Multimodal inputs (not proven; specs/05 §4.6)
    for field in ("messages", "input"):
        items = req.get(field)
        if isinstance(items, list):
            for msg in items:
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") in MULTIMODAL_BLOCK_TYPES:
                    return False, "unsupported_public_policy:multimodal_input"
                content = msg.get("content")
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") in MULTIMODAL_BLOCK_TYPES:
                            return False, "unsupported_public_policy:multimodal_input"
    # 5. Validate any strict JSON Schema / tool schema compiles within dialect S_v and Q_MAX
    for sch in (extract_request_schema(req), extract_strict_tool_schema(req)):
        if sch is not None:
            try:
                compile_json_schema_dfa(sch, q_max=q_max)
            except ValueError as e:
                return False, f"unsupported_public_policy:schema:{e}"
    return True, "ok"


# ============================================================================ 2. JSON Schema -> Minimal Pruned Byte DFA
class _NFA:
    def __init__(self):
        self.eps: list[list[int]] = []
        self.tr: list[list[tuple[bytes, int]]] = []

    def new_state(self) -> int:
        s = len(self.eps)
        self.eps.append([])
        self.tr.append([])
        return s

    def add_eps(self, u: int, v: int):
        self.eps[u].append(v)

    def add_bytes(self, u: int, bs: bytes, v: int):
        self.tr[u].append((bs, v))

    def lit(self, data: bytes) -> tuple[int, int]:
        s = self.new_state()
        cur = s
        for b in data:
            nxt = self.new_state()
            self.add_bytes(cur, bytes([b]), nxt)
            cur = nxt
        return s, cur

    def charset(self, allowed: bytes) -> tuple[int, int]:
        s, e = self.new_state(), self.new_state()
        self.add_bytes(s, allowed, e)
        return s, e

    def seq(self, parts: list[tuple[int, int]]) -> tuple[int, int]:
        if not parts:
            s = self.new_state()
            return s, s
        s0, cur_e = parts[0]
        for s_i, e_i in parts[1:]:
            self.add_eps(cur_e, s_i)
            cur_e = e_i
        return s0, cur_e

    def alt(self, branches: list[tuple[int, int]]) -> tuple[int, int]:
        if len(branches) == 1:
            return branches[0]
        s, e = self.new_state(), self.new_state()
        for bs, be in branches:
            self.add_eps(s, bs)
            self.add_eps(be, e)
        return s, e

    def opt(self, frag: tuple[int, int]) -> tuple[int, int]:
        s, e = self.new_state(), self.new_state()
        self.add_eps(s, frag[0])
        self.add_eps(s, e)
        self.add_eps(frag[1], e)
        return s, e

    def star(self, frag: tuple[int, int]) -> tuple[int, int]:
        s, e = self.new_state(), self.new_state()
        self.add_eps(s, frag[0])
        self.add_eps(s, e)
        self.add_eps(frag[1], frag[0])
        self.add_eps(frag[1], e)
        return s, e

    def plus(self, frag: tuple[int, int]) -> tuple[int, int]:
        s, e = self.new_state(), self.new_state()
        self.add_eps(s, frag[0])
        self.add_eps(frag[1], frag[0])
        self.add_eps(frag[1], e)
        return s, e


_DIGITS = b"0123456789"
_NONZERO = b"123456789"
_HEX_BYTES = b"0123456789abcdefABCDEF"
_SAFE_STR_BYTES = bytes(b for b in range(0x20, 0x7F) if b not in (0x22, 0x5C))
_ESC_BYTES = b'"\\/bfnrt'


def _validate_schema_keys(sch: dict):
    bad = set(sch.keys()) - SUPPORTED_SCHEMA_KEYS
    if bad:
        raise ValueError(f"unsupported keywords {sorted(bad)}")


def _build_schema_nfa(nfa: _NFA, sch: dict, defs: dict, depth: int = 0) -> tuple[int, int]:
    if not isinstance(sch, dict):
        raise ValueError("schema node must be a dict")
    if depth > 16:
        raise ValueError("schema nesting/ref depth > 16")
    _validate_schema_keys(sch)

    if "$ref" in sch:
        ref = sch["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/") and not ref.startswith("#/definitions/"):
            raise ValueError(f"unsupported $ref: {ref!r}")
        key = ref.split("/", 2)[2]
        if key not in defs:
            raise ValueError(f"missing $ref target: {key!r}")
        return _build_schema_nfa(nfa, defs[key], defs, depth + 1)

    if "const" in sch:
        raw = json.dumps(sch["const"], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return nfa.lit(raw)

    if "enum" in sch:
        vals = sch["enum"]
        if not isinstance(vals, list) or not vals:
            raise ValueError("enum must be a non-empty list")
        return nfa.alt([
            nfa.lit(json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            for v in vals
        ])

    if "anyOf" in sch:
        subs = sch["anyOf"]
        if not isinstance(subs, list) or not subs:
            raise ValueError("anyOf must be a non-empty list")
        return nfa.alt([_build_schema_nfa(nfa, sub, defs, depth + 1) for sub in subs])

    typ = sch.get("type")
    if isinstance(typ, list):
        return nfa.alt([_build_schema_nfa(nfa, {**sch, "type": t}, defs, depth + 1) for t in typ])

    if typ == "null":
        return nfa.lit(b"null")
    if typ == "boolean":
        return nfa.alt([nfa.lit(b"true"), nfa.lit(b"false")])
    if typ in ("integer", "number"):
        if sch.get("multipleOf") is not None or sch.get("minimum") is not None or sch.get("maximum") is not None:
            # Small integer range [minimum, maximum] compiled exactly when span <= 64
            lo, hi = sch.get("minimum"), sch.get("maximum")
            if typ == "integer" and isinstance(lo, int) and isinstance(hi, int) and 0 <= hi - lo <= 64 and sch.get("multipleOf") is None:
                return nfa.alt([nfa.lit(str(v).encode("ascii")) for v in range(lo, hi + 1)])
            raise ValueError("unbounded numeric range/multipleOf not in v1 DFA dialect")
        sign = nfa.opt(nfa.lit(b"-"))
        int_body = nfa.alt([
            nfa.lit(b"0"),
            nfa.seq([nfa.charset(_NONZERO), nfa.star(nfa.charset(_DIGITS))]),
        ])
        if typ == "integer":
            return nfa.seq([sign, int_body])
        frac = nfa.opt(nfa.seq([nfa.lit(b"."), nfa.plus(nfa.charset(_DIGITS))]))
        return nfa.seq([sign, int_body, frac])

    if typ == "string":
        def ch_unit():
            # Valid JSON string character: printable ASCII, \x simple escape, \uXXXX hex escape, or UTF-8 multi-byte
            u_esc = nfa.seq([nfa.lit(b"\\u")] + [nfa.charset(_HEX_BYTES) for _ in range(4)])
            utf8_2 = nfa.seq([nfa.charset(bytes(range(0xC2, 0xE0))), nfa.charset(bytes(range(0x80, 0xC0)))])
            utf8_3 = nfa.seq([nfa.charset(bytes(range(0xE0, 0xF0))), nfa.charset(bytes(range(0x80, 0xC0))),
                              nfa.charset(bytes(range(0x80, 0xC0)))])
            utf8_4 = nfa.seq([nfa.charset(bytes(range(0xF0, 0xF5))), nfa.charset(bytes(range(0x80, 0xC0))),
                              nfa.charset(bytes(range(0x80, 0xC0))), nfa.charset(bytes(range(0x80, 0xC0)))])
            return nfa.alt([
                nfa.charset(_SAFE_STR_BYTES),
                nfa.seq([nfa.lit(b"\\"), nfa.charset(_ESC_BYTES)]),
                u_esc, utf8_2, utf8_3, utf8_4,
            ])
        min_l = int(sch.get("minLength", 0))
        max_l = sch.get("maxLength")
        if max_l is not None:
            max_l = int(max_l)
            if max_l < min_l or max_l > 64:
                raise ValueError("string maxLength must be in [minLength, 64]")
            parts = [nfa.lit(b'"')]
            for _ in range(min_l):
                parts.append(ch_unit())
            for _ in range(max_l - min_l):
                parts.append(nfa.opt(ch_unit()))
            parts.append(nfa.lit(b'"'))
            return nfa.seq(parts)
        parts = [nfa.lit(b'"')]
        for _ in range(min_l):
            parts.append(ch_unit())
        parts.append(nfa.star(ch_unit()))
        parts.append(nfa.lit(b'"'))
        return nfa.seq(parts)

    if typ == "array":
        item_sch = sch.get("items")
        if not isinstance(item_sch, dict):
            raise ValueError("array schema requires single dict 'items'")
        min_i = int(sch.get("minItems", 0))
        max_i = sch.get("maxItems")
        if max_i is not None:
            max_i = int(max_i)
            if max_i < min_i or max_i > 16:
                raise ValueError("array maxItems must be in [minItems, 16]")
            if max_i == 0:
                return nfa.lit(b"[]")
            branches = []
            if min_i == 0:
                branches.append(nfa.lit(b"[]"))
            for cnt in range(max(1, min_i), max_i + 1):
                seq_p = [nfa.lit(b"[")]
                for idx in range(cnt):
                    if idx > 0:
                        seq_p.append(nfa.lit(b","))
                    seq_p.append(_build_schema_nfa(nfa, item_sch, defs, depth + 1))
                seq_p.append(nfa.lit(b"]"))
                branches.append(nfa.seq(seq_p))
            return nfa.alt(branches)
        if min_i == 0:
            elem0 = _build_schema_nfa(nfa, item_sch, defs, depth + 1)
            elem_more = nfa.star(nfa.seq([nfa.lit(b","), _build_schema_nfa(nfa, item_sch, defs, depth + 1)]))
            non_empty = nfa.seq([nfa.lit(b"["), elem0, elem_more, nfa.lit(b"]")])
            return nfa.alt([nfa.lit(b"[]"), non_empty])
        seq_p = [nfa.lit(b"[")]
        for idx in range(min_i):
            if idx > 0:
                seq_p.append(nfa.lit(b","))
            seq_p.append(_build_schema_nfa(nfa, item_sch, defs, depth + 1))
        seq_p.append(nfa.star(nfa.seq([nfa.lit(b","), _build_schema_nfa(nfa, item_sch, defs, depth + 1)])))
        seq_p.append(nfa.lit(b"]"))
        return nfa.seq(seq_p)

    if typ == "object":
        if sch.get("additionalProperties") is not False:
            raise ValueError("strict object schema requires 'additionalProperties': false")
        props = sch.get("properties")
        if not isinstance(props, dict):
            raise ValueError("object schema requires 'properties' dict")
        req_set = set(sch.get("required") or [])
        if not req_set.issubset(props.keys()):
            raise ValueError("required keys must appear in properties")
        keys = list(props.keys())
        if not keys:
            return nfa.lit(b"{}")
        # Deterministic declaration-order serialization (specs/05 §4.2):
        # If all properties are required (standard OpenAI strict: true), single linear chain.
        if len(req_set) == len(keys):
            parts = [nfa.lit(b"{")]
            for idx, k in enumerate(keys):
                prefix = ("," if idx > 0 else "") + json.dumps(k, ensure_ascii=False) + ":"
                parts.append(nfa.lit(prefix.encode("utf-8")))
                parts.append(_build_schema_nfa(nfa, props[k], defs, depth + 1))
            parts.append(nfa.lit(b"}"))
            return nfa.seq(parts)
        # Optional keys (bounded <= 6 optional keys): enumerate valid subsets in declaration order
        opt_keys = [k for k in keys if k not in req_set]
        if len(opt_keys) > 6:
            raise ValueError("at most 6 optional properties supported in v1 DFA compiler")
        branches = []
        for mask in range(1 << len(opt_keys)):
            chosen_opt = {opt_keys[i] for i in range(len(opt_keys)) if (mask >> i) & 1}
            sub_keys = [k for k in keys if k in req_set or k in chosen_opt]
            if not sub_keys:
                branches.append(nfa.lit(b"{}"))
                continue
            parts = [nfa.lit(b"{")]
            for idx, k in enumerate(sub_keys):
                prefix = ("," if idx > 0 else "") + json.dumps(k, ensure_ascii=False) + ":"
                parts.append(nfa.lit(prefix.encode("utf-8")))
                parts.append(_build_schema_nfa(nfa, props[k], defs, depth + 1))
            parts.append(nfa.lit(b"}"))
            branches.append(nfa.seq(parts))
        return nfa.alt(branches)

    raise ValueError(f"unsupported schema type: {typ!r}")


@dataclass
class SchemaDFA:
    """Minimal, pruned, co-accessible byte-level DFA (C1-C3 of specs/05 §4.2)."""
    n_states: int
    trans: np.ndarray        # (n_states, 256) int16, -1 = invalid transition
    accept: frozenset[int]   # accepting states in 0 .. n_states - 1
    schema_sha256: str

    def step_bytes(self, q: int, data: bytes) -> int:
        cur = int(q)
        tr = self.trans
        for b in data:
            if cur < 0:
                return -1
            cur = int(tr[cur, b])
        return cur

    def accepts_bytes(self, data: bytes) -> bool:
        q = self.step_bytes(0, data)
        return q in self.accept

    def is_extensible_prefix(self, data: bytes) -> bool:
        """True iff `data` lands in a live co-accessible state (can be extended to an accepting JSON)."""
        return self.step_bytes(0, data) >= 0

    def compile_token_masks(self, token_bytes: list[bytes | None], stop_token_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Compile (next_state[Q+1, V] int16, allowed[Q+1, V] bool) where state `Q = n_states` is the post-stop
        terminal state (`C4` of specs/05 §4.2)."""
        V = len(token_bytes)
        Q = self.n_states
        next_q = np.full((Q + 1, V), -1, dtype=np.int16)
        by_len: dict[int, list[int]] = {}
        stop_set = {int(x) for x in stop_token_ids if x is not None and 0 <= int(x) < V}
        for tid, tb in enumerate(token_bytes):
            if tid in stop_set or tb is None:
                continue
            by_len.setdefault(len(tb), []).append(tid)
        if 0 in by_len:
            idxs0 = np.asarray(by_len.pop(0), dtype=np.int64)
            next_q[:Q, idxs0] = np.arange(Q, dtype=np.int16)[:, None]
        tr = self.trans
        q_init = np.arange(Q, dtype=np.int16)[:, None]
        for ln, tids in by_len.items():
            t_arr = np.asarray(tids, dtype=np.int64)
            b_mat = np.frombuffer(b"".join(token_bytes[i] for i in tids), dtype=np.uint8).reshape(len(tids), ln)
            cur = np.broadcast_to(q_init, (Q, len(tids))).copy()
            for j in range(ln):
                bj = b_mat[:, j][None, :]
                valid = cur >= 0
                nxt = tr[np.maximum(cur, 0), bj]
                cur = np.where(valid, nxt, np.int16(-1))
            next_q[:Q, t_arr] = cur
        for st_id in stop_set:
            for q in self.accept:
                next_q[q, st_id] = Q
        allowed = next_q >= 0
        return next_q, allowed


def compile_json_schema_dfa(schema: dict, prefix_bytes: bytes = b"", suffix_bytes: bytes = b"",
                            q_max: int = Q_MAX) -> SchemaDFA:
    """Compile a JSON Schema in dialect S_v (optionally wrapped in fixed prefix/suffix bytes such as
    `<tool_call>\\n` ... `\\n</tool_call>`) into a minimal co-accessible byte DFA with <= q_max states."""
    canon_str = json.dumps({"schema": schema, "pre": prefix_bytes.hex(), "suf": suffix_bytes.hex()},
                           sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon_str.encode("utf-8")).hexdigest()
    nfa = _NFA()
    defs = {}
    if isinstance(schema.get("$defs"), dict):
        defs.update(schema["$defs"])
    if isinstance(schema.get("definitions"), dict):
        defs.update(schema["definitions"])
    parts = []
    if prefix_bytes:
        parts.append(nfa.lit(prefix_bytes))
    parts.append(_build_schema_nfa(nfa, schema, defs, 0))
    if suffix_bytes:
        parts.append(nfa.lit(suffix_bytes))
    s_nfa, e_nfa = nfa.seq(parts)

    # Epsilon closures
    n_nfa = len(nfa.eps)
    eclose = []
    for i in range(n_nfa):
        seen = {i}
        dq = deque([i])
        while dq:
            u = dq.popleft()
            for v in nfa.eps[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        eclose.append(frozenset(seen))

    # Subset construction
    start_set = eclose[s_nfa]
    state_map = {start_set: 0}
    states_list = [start_set]
    raw_trans = []
    dq = deque([0])
    while dq:
        qi = dq.popleft()
        cur_set = states_list[qi]
        if len(states_list) > 4 * q_max + 256:
            raise ValueError(f"unminimized DFA exceeded state cap ({len(states_list)})")
        by_byte: dict[int, set[int]] = {}
        for u in cur_set:
            for bs, v in nfa.tr[u]:
                ec = eclose[v]
                for b in bs:
                    s_b = by_byte.get(b)
                    if s_b is None:
                        by_byte[b] = set(ec)
                    else:
                        s_b.update(ec)
        row = [-1] * 256
        for b, tgt in by_byte.items():
            fz = frozenset(tgt)
            ti = state_map.get(fz)
            if ti is None:
                ti = len(states_list)
                state_map[fz] = ti
                states_list.append(fz)
                dq.append(ti)
            row[b] = ti
        raw_trans.append(row)

    raw_acc = {i for i, st in enumerate(states_list) if e_nfa in st}
    if not raw_acc:
        raise ValueError("schema language is empty")

    # Co-accessibility pruning (C3: keep only states reachable from 0 AND able to reach raw_acc)
    rev = [[] for _ in range(len(states_list))]
    for u, row in enumerate(raw_trans):
        for v in set(row):
            if v >= 0:
                rev[v].append(u)
    coacc = set(raw_acc)
    dq = deque(raw_acc)
    while dq:
        v = dq.popleft()
        for u in rev[v]:
            if u not in coacc:
                coacc.add(u)
                dq.append(u)
    if 0 not in coacc:
        raise ValueError("schema start state cannot reach any accepting state")

    # Hopcroft DFA minimization over live co-accessible states (+ implicit dead state -1)
    live = sorted(coacc)
    lmap = {old: idx for idx, old in enumerate(live)}
    nl = len(live)
    tr_l = np.full((nl, 256), -1, dtype=np.int32)
    for idx, old in enumerate(live):
        for b, tgt in enumerate(raw_trans[old]):
            if tgt in lmap:
                tr_l[idx, b] = lmap[tgt]
    acc_l = frozenset(lmap[old] for old in raw_acc if old in lmap)
    non_acc_l = frozenset(set(range(nl)) - acc_l)
    parts_min = [p for p in (acc_l, non_acc_l) if p]
    changed = True
    while changed:
        changed = False
        cls_of = {}
        for ci, grp in enumerate(parts_min):
            for s in grp:
                cls_of[s] = ci
        new_parts = []
        for grp in parts_min:
            if len(grp) <= 1:
                new_parts.append(grp)
                continue
            sigs: dict[tuple, list[int]] = {}
            for s in grp:
                sig = tuple(cls_of[int(tr_l[s, b])] if tr_l[s, b] >= 0 else -1 for b in range(256))
                sigs.setdefault(sig, []).append(s)
            if len(sigs) > 1:
                changed = True
            for sub in sigs.values():
                new_parts.append(frozenset(sub))
        parts_min = new_parts

    # Order minimized states by BFS from the class containing 0 so state 0 is always the start state
    cls_of = {}
    reps = []
    for ci, grp in enumerate(parts_min):
        rep = min(grp)
        reps.append(rep)
        for s in grp:
            cls_of[s] = ci
    c0 = cls_of[0]
    order = [c0]
    seen_c = {c0}
    dq = deque([c0])
    while dq:
        ci = dq.popleft()
        rep = reps[ci]
        for b in range(256):
            t = int(tr_l[rep, b])
            if t >= 0:
                cj = cls_of[t]
                if cj not in seen_c:
                    seen_c.add(cj)
                    order.append(cj)
                    dq.append(cj)
    if len(order) > q_max:
        raise ValueError(f"minimized schema DFA has {len(order)} states > q_max={q_max}")
    remap_c = {ci: new_i for new_i, ci in enumerate(order)}
    n_final = len(order)
    final_tr = np.full((n_final, 256), -1, dtype=np.int16)
    for new_i, ci in enumerate(order):
        rep = reps[ci]
        for b in range(256):
            t = int(tr_l[rep, b])
            if t >= 0:
                final_tr[new_i, b] = remap_c[cls_of[t]]
    final_acc = frozenset(remap_c[ci] for ci in order if reps[ci] in acc_l)
    return SchemaDFA(n_states=n_final, trans=final_tr, accept=final_acc, schema_sha256=digest)


def token_bytes_from_hf_tokenizer(tok, V: int) -> list[bytes | None]:
    """Canonical raw UTF-8 byte sequence of each vocabulary ID 0 .. V-1 (`None` for special/control tokens)."""
    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    out: list[bytes | None] = [None] * V
    # Fast byte-level BPE decoder via convert_ids_to_tokens + ByteLevel map when available, else tok.decode([i])
    for i in range(V):
        if i in special_ids:
            out[i] = None
            continue
        try:
            s = tok.decode([i], clean_up_tokenization_spaces=False, skip_special_tokens=False)
            out[i] = s.encode("utf-8")
        except Exception:
            out[i] = None
    return out


# ============================================================================ 3. In-Circuit Constrained Decoding Binder
def bind_constrained_masks(em, prm_sz, VSs, As, q_per_slot, active_per_slot, allowed_mat: np.ndarray,
                           public_states: bool = True, tag="rzMask"):
    """Bind sampler_zk's Phase-1 committed boolean allowed-masks `As[p]` (`(K_G, V_p)`) to `allowed_mat[q_s, v]`
    (when `active_s == 1`) or `1[v < V]` (when `active_s == 0`) for every sampled generation step `s < K_G`!

    Soundness:
      * `As[p]` is committed in Phase 1 and constrained boolean with `As[p](s, v) = 0` for `v >= V`.
      * `e_q(s, q)` (one-hot of DFA state `q_s` on `0 .. Q_pad - 1`) and `act_c(s)` (`active_s in {0, 1}`) are committed
        in Phase 1.
      * In commit round `round_mask(tr)` (after Phase 1), a fresh Fiat-Shamir point `r_v in E^vb` (`vb = ceil(log2 V)`)
        is drawn.  Both prover and verifier compute the tiny `(Q_pad + 1)`-entry table
            m_q = sum_{v < V} eq(r_v, v) * allowed_mat[q, v]   (q < Q_real)
            m_unconstrained = sum_{v < V} eq(r_v, v)
        in ~10 ms on CPU, and enforce via `sumzero(..., eq_axes=['s'])` that for EVERY step `s < K_G`:
            sum_v eq(r_v, v) * As(s, v) == (1 - act_s) * m_unconstrained + act_s * sum_q e_q(s, q) * m_q.
      * By Schwartz-Zippel over `E = F_{p^3}`, any 1-token discrepancy in `As(s, .)` on `[0, V)` fails with probability
        >= 1 - vb / |E| > 1 - 2^-187.  Extra committed cells: O(K_G * Q_pad) (~8K cells, no per-vocabulary overhead).
    """
    pv = em.pv
    ctx = em.ctx
    K = prm_sz.K
    sb = prm_sz.sb
    V = prm_sz.V
    Q_states = int(allowed_mat.shape[0])
    qb = max(1, int(Q_states - 1).bit_length())
    Q_pad = 1 << qb
    vb = max(5, int(prm_sz.NS - 1).bit_length())

    em_s = em.sub_em([("s", sb)]) if sb > 0 else em
    SCs = em_s.space(tag + ".sc", [])
    QS = em_s.space(tag + ".q", [("q", qb)])

    q_pub_arr = np.asarray(q_per_slot, dtype=np.int64) if (pv or public_states) else None
    act_pub_arr = np.asarray(active_per_slot, dtype=np.int64) if (pv or public_states) else None
    if pv:
        q_arr = np.asarray(q_per_slot, dtype=np.int64)
        act_arr = np.asarray(active_per_slot, dtype=np.int64)
        assert len(q_arr) == K and len(act_arr) == K
        oh = np.zeros((K, Q_pad), dtype=np.int64)
        for s in range(K):
            qs = int(q_arr[s]) if act_arr[s] else 0
            assert 0 <= qs < Q_states, (s, qs, Q_states)
            oh[s, qs] = 1
        if sb == 0:
            act_v = np.repeat(act_arr, SCs.n)
            q_v = np.repeat(q_arr, SCs.n)
            oh_v = np.repeat(oh, QS.n // Q_pad, axis=0).ravel()
        else:
            act_v, q_v, oh_v = act_arr, q_arr, oh.ravel()
    else:
        act_v = q_v = oh_v = None

    act_c = em_s.col(SCs, "act", act_v, 1)
    q_c = em_s.col(SCs, "q", q_v, qb + 1)
    oh_c = em_s.col(QS, "oh", oh_v, 1)
    em_s.boolean(act_c, tag + ".act.b")
    em_s.boolean(oh_c, tag + ".oh.b")
    if public_states:
        p_act = np.repeat(act_pub_arr, SCs.n) if sb == 0 else act_pub_arr
        p_q = np.repeat(q_pub_arr, SCs.n) if sb == 0 else q_pub_arr
        pub_act_c = em_s.pub(SCs, "pact", p_act, 1)
        pub_q_c = em_s.pub(SCs, "pq", p_q, qb + 1)
        em_s.zero(SCs, lambda: act_c.ex - pub_act_c.ex, tag + ".act.pub")
        em_s.zero(SCs, lambda: q_c.ex - pub_q_c.ex, tag + ".q.pub")
    # Sum over q of oh(s, q) == 1 and sum_q q * oh(s, q) == (act_s ? q_s : 0)
    ind_q0 = QS.S.ind(q=0)
    s_eq = ["s"] if sb > 0 else []
    QS.S.sumzero(oh_c.ex - ind_q0, eq_axes=s_eq, tag=tag + ".oh.sum1")
    QS.S.sumzero(oh_c.ex * QS.S.idx("q") - ind_q0 * act_c.col.on(QS.S, {"s": "s"} if sb else {}) * q_c.col.on(QS.S, {"s": "s"} if sb else {}),
                 eq_axes=s_eq, tag=tag + ".oh.qval")

    allow_pad = np.zeros((Q_pad, 1 << vb), dtype=np.uint8)
    allow_pad[:Q_states, :V] = np.asarray(allowed_mat[:, :V], dtype=np.uint8)

    def round_mask(tr):
        from .zk_flat import EqP
        rv = tr.exts(vb)
        eq_v = CPU.eq_table(list(rv))    # (3, 2^vb) uint64
        # m_unc = sum_{v < V} eq_v[:, v]; m_q[q] = sum_{v < V} eq_v[:, v] * allow_pad[q, v]
        eq_real = eq_v[:, :V]
        m_unc = (int(np.bitwise_xor.reduce(eq_real[0]) * 0), 0, 0)  # placeholder replaced below
        m_unc = CPU.esum(eq_real)
        # Fast (3, V) @ (V, Q_states) in float64 chunks of 24-bit limbs or integer matmul:
        # Split uint64 Goldilocks elements into four 16-bit limbs -> exact int64 BLAS matmul `(12, V) @ (V, Q_states)`.
        eq16 = np.empty((12, V), dtype=np.float64)
        for l in range(3):
            u = eq_real[l]
            eq16[4 * l + 0] = (u & np.uint64(0xFFFF)).astype(np.float64)
            eq16[4 * l + 1] = ((u >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.float64)
            eq16[4 * l + 2] = ((u >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.float64)
            eq16[4 * l + 3] = ((u >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.float64)
        # Max entry of prod is 65535 * 151936 = 9.96e9 << 2^53 (100% exact IEEE-754 float64 integer arithmetic)
        prod = (eq16 @ allow_pad[:Q_states, :V].T.astype(np.float64)).astype(np.uint64)   # (12, Q_states) exact.
        mq_tab = np.zeros((3, Q_pad), dtype=np.uint64)
        for l in range(3):
            p0, p1, p2, p3 = prod[4 * l], prod[4 * l + 1], prod[4 * l + 2], prod[4 * l + 3]
            acc_l = [
                (int(p0[q]) + (int(p1[q]) << 16) + (int(p2[q]) << 32) + (int(p3[q]) << 48)) % P
                for q in range(Q_states)
            ]
            mq_tab[l, :Q_states] = np.asarray(acc_l, dtype=np.uint64)

        # Commit per-step partial MLE evaluations of As[p] and target sum on SCs
        piece_cols = []
        for p_idx, (off, bb) in enumerate(prm_sz.pieces):
            VSp = VSs[p_idx]
            S_ = VSp.S
            n_p = 1 << (bb + 5)
            lo = off * 32
            # High bits (bb+5 .. vb-1) of v = off*32 + 32*b + k are fixed constants for piece p_idx.
            hi_sc = ONE
            for t in range(bb + 5, vb):
                bit = (lo >> t) & 1
                hi_sc = hmul(hi_sc, rv[t] if bit else hsub(ONE, rv[t]))
            pv_val = None
            if pv:
                pv_val = np.zeros((3, SCs.S.n), dtype=np.uint64)
                a2 = As[p_idx].v.reshape(K, -1)[:, :n_p]
                eq_sub = eq_v[:, lo:min(lo + n_p, 1 << vb)]
                for s in range(K):
                    cnt_s = min(n_p, max(0, (1 << vb) - lo))
                    active_idx = np.flatnonzero(a2[s, :cnt_s])
                    val_s = CPU.esum(eq_sub[:, active_idx]) if len(active_idx) else ZERO
                    if sb == 0:
                        for rr in range(SCs.S.n):
                            pv_val[0, rr], pv_val[1, rr], pv_val[2, rr] = val_s
                    else:
                        pv_val[0, s], pv_val[1, s], pv_val[2, s] = val_s
            pcol = MZ.ecol(SCs.S, f"{tag}.ap{p_idx}", pv_val)
            piece_cols.append(pcol)
            v_vars = S_.vars("k") + (S_.vars("b") if VSp.has("b") else [])
            eq_low = Ex.atom(EqP(v_vars, list(rv[:bb + 5]))) * Ex.const(hi_sc)
            ind_v0 = Ex.atom(SelLt([], 1, fixed=tuple((v_, 0) for v_ in v_vars)))
            from . import sampler_zk as SAMPZ
            zsel = SAMPZ._zsel(VSp)
            S_.sumzero(zsel * (eq_low * As[p_idx].ex - ind_v0 * MZ.view(pcol, S_, {"s": "s"} if sb else {"z": [("c", 0)] * SCs.S.bits})),
                       eq_axes=s_eq, tag=f"{tag}.p{p_idx}")

        # Bind sum_q oh(s, q) * mq_tab[q] -> qsum_col(s), and check sum_p pcol_p(s) == (1 - act_s)*m_unc + act_s*qsum(s)
        qv_val = None
        if pv:
            qv_val = np.zeros((3, SCs.S.n), dtype=np.uint64)
            for s in range(K):
                qs = int(q_arr[s]) if act_arr[s] else 0
                vs = (int(mq_tab[0, qs]), int(mq_tab[1, qs]), int(mq_tab[2, qs]))
                if sb == 0:
                    for rr in range(SCs.S.n):
                        qv_val[0, rr], qv_val[1, rr], qv_val[2, rr] = vs
                else:
                    qv_val[0, s], qv_val[1, s], qv_val[2, s] = vs
        qsum_col = MZ.ecol(SCs.S, f"{tag}.qsum", qv_val)
        mq_ev = Ex.atom(MZ.EVec(f"{tag}.mq", mq_tab, QS.S.vars("q")))
        QS.S.sumzero(oh_c.ex * mq_ev - ind_q0 * MZ.view(qsum_col, QS.S, {"s": "s"} if sb else {"z": [("c", 0)] * SCs.S.bits}),
                     eq_axes=s_eq, tag=f"{tag}.qsum.eq")
        lhs_tot = Ex.const(0)
        for pcol in piece_cols:
            lhs_tot = lhs_tot + pcol.ex
        rhs_tot = (Ex.const(1) - act_c.ex) * Ex.const(m_unc) + act_c.ex * qsum_col.ex
        SCs.S.zero(lhs_tot - rhs_tot, tag=f"{tag}.eq")

    return {"act": act_c, "q": q_c, "round": round_mask}


# ============================================================================ 4. Multi-Turn AEAD State Blob (specs/01 §6.4, specs/05 §4.5)
def seal_state_blob(key_32: bytes, model_id, sess, turn: int, st_new, prev_bundle: dict, hist_cap: int) -> str:
    """Encrypt `(prev_bundle, zero_pad)` to a fixed byte length depending ONLY on `hist_cap` (IND$-CPA + length hiding)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    assert len(key_32) == 32
    raw = json.dumps({
        "hist": [int(x) for x in prev_bundle["hist"]],
        "st_pp": [int(x) for x in prev_bundle["st_pp"]],
        "hx": [int(x) for x in prev_bundle["hx"]],
        "hy": [int(x) for x in prev_bundle["hy"]],
        "r": [int(x) for x in prev_bundle["r"]],
    }, separators=(",", ":")).encode("utf-8")
    fixed_pt_len = 512 + 12 * int(hist_cap)
    if len(raw) + 4 > fixed_pt_len:
        raise ValueError(f"state plaintext {len(raw)} exceeds fixed container {fixed_pt_len}")
    pt = len(raw).to_bytes(4, "little") + raw + b"\x00" * (fixed_pt_len - 4 - len(raw))
    aad = json.dumps({
        "mid": [int(x) for x in model_id],
        "sess": [int(x) for x in sess],
        "turn": int(turn),
        "st": [int(x) for x in st_new],
        "cap": int(hist_cap),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(12)
    ct = AESGCM(bytes(key_32)).encrypt(nonce, pt, aad)
    return (nonce + ct).hex()


def open_state_blob(key_32: bytes, blob_hex: str, model_id, sess, prev_turn: int, st_prev, hist_cap: int) -> dict:
    """Decrypt and authenticate a previous turn's state blob against `(model_id, sess, prev_turn, st_prev, hist_cap)`."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    buf = bytes.fromhex(blob_hex)
    if len(buf) < 28:
        raise ValueError("truncated state blob")
    nonce, ct = buf[:12], buf[12:]
    aad = json.dumps({
        "mid": [int(x) for x in model_id],
        "sess": [int(x) for x in sess],
        "turn": int(prev_turn),
        "st": [int(x) for x in st_prev],
        "cap": int(hist_cap),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    pt = AESGCM(bytes(key_32)).decrypt(nonce, ct, aad)
    ln = int.from_bytes(pt[:4], "little")
    if ln < 0 or 4 + ln > len(pt):
        raise ValueError("corrupted state blob length")
    d = json.loads(pt[4:4 + ln].decode("utf-8"))
    return {
        "hist": [int(x) for x in d["hist"]],
        "st_pp": tuple(int(x) for x in d["st_pp"]),
        "hx": tuple(int(x) for x in d["hx"]),
        "hy": tuple(int(x) for x in d["hy"]),
        "r": tuple(int(x) for x in d["r"]),
    }
