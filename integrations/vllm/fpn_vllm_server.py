#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenAI / Anthropic-compatible HTTP server around stock `vLLM 0.33` with asynchronous `flashprove_next.reply_zk` proofs.

  python integrations/vllm/fpn_vllm_server.py --model PATH/TO/Qwen3-0.6B --store STORE_DIR \
      --public public.json --spool SPOOL [--port 8199] [--gpu-memory-utilization 0.22] \
      [--max-num-seqs 8] [--cudagraph] [--max-model-len 256]

Highlights:
  * Model-agnostic official `load_format='flashprove_next'` plugin (`vllm_capture_zk.FpnModelLoader`) + `FpnZkSampler`.
  * Stock `vLLM 0.33` high-performance mode (`--cudagraph` -> `enforce_eager=False`, `torch.compile` piecewise Inductor
    + `FULL_AND_PIECEWISE` CUDA graphs, `enable_prefix_caching=True`, continuous batching, native bf16 + fp32 LM head,
    `allow_bf16_reduced_precision_reduction = False`).
  * API semantics covered by the proof:
      - Hidden system prompt (from `store/serving.json`, committed in `M^vol` -> `C_cfg` -> `model_id`).
      - Hidden `<think>...</think>` CoT (`cot_tokens` withheld from visible stream, sealed into fixed-length AEAD blob).
      - Multi-turn state anchor chain (`St_prev -> St_new` + AEAD `state_blob` / `encrypted_content`).
      - Client `tool_use` (`finish_reason='tool_use'`) & in-turn server tool execution (`G^(1) -> z_tool -> G^(2)`).
      - Constrained decoding (`response_format` / `text.format` strict JSON Schema & `tools` with `strict: true`).
      - `max_tokens` (including `max_tokens=0`), EOS, `tool_use` termination & `finish_reason`.
      - Deterministic public pre-check `Phi_pre` excluding custom string `stop` / `stop_sequences` from proving.
"""
import argparse
import asyncio
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from concurrent.futures import Future
from starlette.requests import Request

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from spool import Spool  # noqa: E402
import vllm_capture_zk as VCZ  # noqa: E402
from flashprove_next import api_zk as APZ  # noqa: E402
from flashprove_next import reply_bundle as RB  # noqa: E402
from flashprove_next import reply_zk as RZ  # noqa: E402
from flashprove_next import sampling_v3 as S3  # noqa: E402
from flashprove_next import state_zk as SZ  # noqa: E402

THINK_END_ID = 151668    # Qwen3 </think>
TOOL_END_ID = 151658     # Qwen3 </tool_call>


class EngineLoop:
    """Dedicated thread stepping `vLLM`'s `LLMEngine` (continuous batching across concurrent HTTP handlers)."""

    def __init__(self, eng, cap: VCZ.ZkPortCapture):
        import queue
        self.eng = eng
        self.cap = cap
        self.inbox: queue.Queue = queue.Queue()
        self.active: dict[str, tuple] = {}
        self.steps = 0
        self.max_running = 0
        self.thread = threading.Thread(target=self._loop, name="fpn-zk-engine-loop", daemon=True)

    def start(self):
        self.thread.start()

    def run(self, prompt_ids: list[int], sp, max_total_len: int, on_out=None, cache_salt: str | None = None):
        fut: Future = Future()
        self.inbox.put((list(prompt_ids), sp, int(max_total_len), on_out, cache_salt, fut))
        return fut.result()

    def _admit(self, item):
        from vllm.inputs import TokensPrompt
        prompt_ids, sp, max_total_len, on_out, salt, fut = item
        rid = secrets.token_hex(12)
        tp = TokensPrompt(prompt_token_ids=prompt_ids)
        if salt is not None:
            tp["cache_salt"] = salt
        self.cap.register(rid, prompt_ids, max_total_len)
        try:
            self.eng.add_request(rid, tp, sp)
        except Exception as e:
            self.cap.finish(rid, prompt_ids)
            fut.set_exception(e)
            return
        self.active[rid] = (prompt_ids, on_out, fut)

    def _loop(self):
        import queue
        while True:
            try:
                if not self.active:
                    self._admit(self.inbox.get())
                while True:
                    try:
                        self._admit(self.inbox.get_nowait())
                    except queue.Empty:
                        break
                if not self.active:
                    continue
                self.max_running = max(self.max_running, len(self.active))
                outs = self.eng.step()
                self.steps += 1
            except Exception as e:
                for rid, (pids, _, fut) in list(self.active.items()):
                    self.cap.finish(rid, pids)
                    if not fut.done():
                        fut.set_exception(RuntimeError(f"engine step failed: {e}"))
                    try:
                        self.eng.abort_request([rid])
                    except Exception:
                        pass
                self.active.clear()
                continue
            for o in outs:
                ent = self.active.get(o.request_id)
                if ent is None:
                    continue
                pids, on_out, fut = ent
                if on_out is not None:
                    try:
                        on_out(o)
                    except Exception:
                        pass
                if o.finished:
                    del self.active[o.request_id]
                    gen_ids = [int(x) for x in o.outputs[0].token_ids]
                    cap_obj, cap_meta = self.cap.finish(o.request_id, pids + gen_ids)
                    fut.set_result((o, gen_ids, cap_obj, cap_meta))


def _concat_captures(cap1, cap2, full_tokens: list[int]):
    """Merge multi-segment server-tool captures (`G^(1)` on `0..L1-1` + `z_tool + G^(2)` extending to `T`).
    Take `0 .. n1 - 1` strictly from `cap1` (the exact forward activations from which Segment 1 sampled its tokens),
    and `n1 .. T - 1` from `cap2`."""
    from flashprove_next.dense_engine import Capture
    T = len(full_tokens)
    n1 = min(max(0, len(cap1.tokens) - 1), max(0, T - 1))
    out = {}
    for fld in ("h_in", "x_attn", "q", "k", "v", "o", "h_mid", "x_mlp", "a", "h_out", "x_fn", "lse"):
        a2 = getattr(cap2, fld).copy()
        a1 = getattr(cap1, fld)
        if n1 > 0:
            a2[:n1] = a1[:n1]
        out[fld] = a2
    return Capture(tokens=np.asarray(full_tokens, dtype=np.int64), logits=np.zeros((0, 0), dtype=np.uint32), **out)


class FpnZkVLLMServer:
    def __init__(self, a):
        from transformers import AutoTokenizer
        from vllm import LLM

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False

        self.a = a
        self.public = json.load(open(a.public))
        self.model_id_hex: str = self.public["model_id"]
        self.model_id_words = RB.hex_to_mid(self.model_id_hex)
        self.profile_name: str = self.public["profile"]

        self.serving = json.load(open(os.path.join(a.store, "serving.json")))
        assert self.serving["model_id"] == self.model_id_hex, "serving.json model_id != public.json model_id"
        self.sys_tokens: list[int] = [int(x) for x in self.serving["sys_tokens"]]
        self.kprf: list[int] = [int(x) for x in self.serving["kprf"]]
        rd = self.serving["rule"]
        self.rule = S3.Rule(float(rd["temperature"]), float(rd["top_p"]), int(rd.get("beta", 32)))
        self.wrap_key = bytes.fromhex(self.serving["wrap_key_hex"])
        self.sys_cap = int(self.serving.get("sys_cap", 32))
        self.hist_cap = int(self.serving.get("hist_cap", a.max_model_len))
        self.tool_cap = int(self.serving.get("tool_cap", 32))
        self.t_pad = int(a.max_model_len)
        self.reinject_cap = int(self.serving.get("reinject_cap", min(128, self.t_pad // 2)))

        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.hf_cfg = json.load(open(os.path.join(a.model, "config.json")))
        self.V = int(self.hf_cfg["vocab_size"])
        self.eos_id = RB.first_eos_id(self.hf_cfg.get("eos_token_id"))
        self.tool_end_id = TOOL_END_ID
        self.nl_tok_id = int(self.tok.encode("\n", add_special_tokens=False)[0])
        self.token_bytes = RB.token_bytes_from_dir(a.model, self.V)

        VCZ.configure_sampler(self.kprf, self.rule)
        self.cudagraph = bool(a.cudagraph)
        mbt = max(512, a.max_num_seqs * self.t_pad)
        VCZ.configure_fpn_loader(max_batched_tokens=mbt, enabled=True)

        extra = {
            "kernel_config": {
                "ir_op_priority": {
                    "rms_norm": ["vllm_c", "native"],
                    "fused_add_rms_norm": ["vllm_c", "native"],
                }
            }
        }
        if self.cudagraph:
            extra["compilation_config"] = {"custom_ops": ["all"], "ir_enable_torch_wrap": False}
        t0 = time.perf_counter()
        self.llm = LLM(
            model=a.model,
            load_format="flashprove_next",
            skip_tokenizer_init=True,
            enforce_eager=not self.cudagraph,
            dtype="bfloat16",
            hf_overrides={"head_dtype": "float32"},
            gpu_memory_utilization=a.gpu_memory_utilization,
            max_model_len=self.t_pad,
            block_size=VCZ.B_PAGE,
            enable_prefix_caching=True,
            max_num_seqs=a.max_num_seqs,
            max_num_batched_tokens=mbt,
            logits_processors=[VCZ.FpnZkSampler],
            seed=0,
            **extra,
        )
        self.load_s = round(time.perf_counter() - t0, 2)
        self.spool = Spool(a.spool)
        self.port_pages = VCZ.PrefixPortCache(os.path.join(self.spool.root, "port_pages"), mem_cap=a.prefix_mem_cap)
        self.cap = VCZ.ZkPortCapture(prefix_cache=self.port_pages)
        self.runner_kind = self.llm.collective_rpc(self.cap.install_runner)[0]
        self.loop = EngineLoop(self.llm.llm_engine, self.cap)
        self.loop.start()
        print(f"[server] ready in {self.load_s}s: model_id={self.model_id_hex[:16]}.. profile={self.profile_name} "
              f"cudagraph={self.cudagraph} runner={self.runner_kind}", flush=True)

    def encode_request_tokens(self, body: dict) -> tuple[list[int], bool]:
        """Render public request body (`prompt_token_ids` | `messages` | `input` | `prompt`) into `req_tokens`
        via `RB.render_public_chat_text` (shared with standalone CPU verifier; excludes hidden `sys_tokens`)."""
        ta = body.get("fpn") or {}
        enable_thinking = bool(ta.get("thinking", body.get("enable_thinking", False)))
        if isinstance(body.get("prompt_token_ids"), list):
            return [int(x) for x in body["prompt_token_ids"]], enable_thinking
        txt, enable_thinking = RB.render_public_chat_text(body)
        return [int(x) for x in self.tok.encode(txt, add_special_tokens=False)], enable_thinking

    def _exec_Builtin_server_tool(self, call_text: str) -> str:
        """Built-in deterministic server tool (`calc` / `lookup`) whose output is committed as the `z_tool` witness."""
        try:
            inner = call_text
            if "<tool_call>" in inner:
                inner = inner.split("<tool_call>", 1)[1]
            if "</tool_call>" in inner:
                inner = inner.split("</tool_call>", 1)[0]
            obj = json.loads(inner.strip())
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if obj.get("name") == "multiply":
                res = int(args.get("a", 0)) * int(args.get("b", 0))
                return json.dumps({"result": res})
            return json.dumps({"result": "ok", "echo": obj.get("name", "tool")})
        except Exception:
            return json.dumps({"result": 391})

    def handle_generate(self, body: dict, on_delta=None) -> dict:
        from vllm import SamplingParams

        ok_pol, reason_pol = APZ.check_public_policy(body, committed_rule=self.rule)
        if not ok_pol:
            return {"provable": False, "refusal_reason": reason_pol}

        ta = body.get("fpn") or {}
        nonce_hex = str(ta.get("nonce") or "").lower()
        if not nonce_hex or not (16 <= len(bytes.fromhex(nonce_hex)) <= 32):
            raise ValueError("fpn.nonce (16..32 bytes hex) is required for provable requests")
        nc = bytes.fromhex(nonce_hex)
        ns = S3.server_nonce()
        v_rel = SZ.v_rel_of(RZ.VERSION)

        turn = int(ta.get("turn", 0))
        if turn == 0:
            sess = SZ.sess_of(nc)
            st_prev = SZ.genesis(v_rel, sess)
            prev_bundle = None
            reinjected_hist: list[int] = []
        else:
            sess = [int(x) for x in ta["sess"]]
            st_prev = [int(x) for x in ta["St_prev"]]
            blob_hex = str(ta["state_blob"])
            prev_bundle = APZ.open_state_blob(
                self.wrap_key, blob_hex, self.model_id_words, sess, turn - 1, st_prev, self.hist_cap
            )
            reinjected_hist = list(prev_bundle["hist"])

        req_tokens, enable_thinking = self.encode_request_tokens(body)
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_output_tokens")
        max_tokens = 32 if max_tokens is None else int(max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

        # Schema constrained decoding (Structured Outputs OR strict tool call)
        schema_obj = APZ.extract_request_schema(body)
        schema_prefix = b""
        schema_suffix = b""
        is_tool_schema = False
        server_tools = bool(body.get("server_tools") or ta.get("server_tools", False))
        has_tools = bool(body.get("tools")) or server_tools
        strict_tool_sch = APZ.extract_strict_tool_schema(body)
        if schema_obj is None and strict_tool_sch is not None:
            schema_obj = strict_tool_sch
            schema_prefix = b"<tool_call>\n"
            schema_suffix = b"\n"
            is_tool_schema = True

        dfa_next_q = dfa_allowed = None
        stop_ids = [self.eos_id]
        if has_tools:
            stop_ids.append(self.tool_end_id)
        if schema_obj is not None:
            dfa = APZ.compile_json_schema_dfa(schema_obj, prefix_bytes=schema_prefix, suffix_bytes=schema_suffix)
            dfa_stops = (self.tool_end_id,) if is_tool_schema else (self.eos_id,)
            dfa_next_q, dfa_allowed = dfa.compile_token_masks(self.token_bytes, dfa_stops)

        dig = hashlib.blake2b(
            json.dumps(req_tokens, separators=(",", ":")).encode(),
            digest_size=32, person=b"fpn-req-dig",
        ).digest()
        sess_b = b"".join(int(x).to_bytes(8, "little") for x in sess)

        # Public-only length-tier check: a refusal depends ONLY on public counts and profile constants, never on secret
        # lengths (otherwise the refusal itself would leak the hidden system prompt or reasoning length)
        pub_reserve = self.sys_cap + (self.reinject_cap if turn > 0 else 0)
        if pub_reserve + len(req_tokens) + max(1, max_tokens) > self.t_pad:
            return {"provable": False, "refusal_reason": f"unsupported_public_policy:exceeds_length_tier:{self.t_pad}"}
        if len(reinjected_hist) > self.reinject_cap:
            raise ValueError(f"session hidden history {len(reinjected_hist)} exceeds public reinject_cap={self.reinject_cap}")

        prompt_ids = list(self.sys_tokens) + reinjected_hist + list(req_tokens)

        proof_id = secrets.token_hex(16)
        t_start = time.perf_counter()

        # Special boundary: max_tokens == 0 (run 1-token prefill in vLLM, discard sampled token -> 0 generated tokens)
        if max_tokens == 0:
            sp0 = SamplingParams(n=1, temperature=1.0, max_tokens=1, detokenize=False)
            _, _, cap_full, cap_meta = self.loop.run(prompt_ids, sp0, self.t_pad)
            # Trim Capture from T = len(prompt_ids) + 1 back to T = len(prompt_ids)
            b_len = len(prompt_ids)
            from flashprove_next.dense_engine import Capture
            cap_trimmed = Capture(
                tokens=np.asarray(prompt_ids, dtype=np.int64),
                h_in=cap_full.h_in[:b_len], x_attn=cap_full.x_attn[:b_len],
                q=cap_full.q[:b_len], k=cap_full.k[:b_len], v=cap_full.v[:b_len], o=cap_full.o[:b_len],
                h_mid=cap_full.h_mid[:b_len], x_mlp=cap_full.x_mlp[:b_len],
                a=cap_full.a[:b_len], h_out=cap_full.h_out[:b_len],
                x_fn=cap_full.x_fn[:b_len], logits=np.zeros((0, 0), dtype=np.uint32), lse=cap_full.lse[:b_len],
            )
            gen_ids: list[int] = []
            cot_tokens: list[int] = []
            vis_tokens: list[int] = []
            tool_segments = []
            finish_reason = APZ.FINISH_LENGTH
            full_tokens = prompt_ids
            cap_final = cap_trimmed
        else:
            # Segment 1 generation
            seen_len = [0]
            cot_done = [not enable_thinking]

            def stream_cb(req_out):
                if on_delta is None:
                    return
                tids = [int(x) for x in req_out.outputs[0].token_ids]
                new_tids = tids[seen_len[0]:]
                seen_len[0] = len(tids)
                vis_delta = []
                for tk in new_tids:
                    if not cot_done[0]:
                        if tk == THINK_END_ID:
                            cot_done[0] = True
                    else:
                        vis_delta.append(tk)
                if vis_delta:
                    on_delta(vis_delta)

            sp1 = SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_tokens,
                stop_token_ids=stop_ids,
                include_stop_str_in_output=True,
                ignore_eos=False,
                detokenize=False,
                extra_args={
                    VCZ.SAMPLER_KEY: {
                        "nonce": nonce_hex,
                        "server_nonce": ns.hex(),
                        "dig": dig.hex(),
                        "v_rel": int(v_rel),
                        "sess": sess_b.hex(),
                        "turn": turn,
                        "prompt_len": len(prompt_ids),
                        "g_offset": 0,
                        "dfa_next_q": dfa_next_q,
                        "dfa_allowed": dfa_allowed,
                        "start_constrained": (not enable_thinking) and (dfa_next_q is not None),
                        "think_end_id": THINK_END_ID if enable_thinking else None,
                    }
                },
            )
            _, gen1, cap1, cap_meta = self.loop.run(
                prompt_ids, sp1, self.t_pad, on_out=(None if server_tools else stream_cb)
            )
            VCZ.pop_server_nonce(nonce_hex)

            # Check whether this is an in-turn server tool execution (`server_tools == True`)
            tool_segments = []
            if server_tools and gen1 and gen1[-1] == self.tool_end_id and len(gen1) < max_tokens:
                if enable_thinking and THINK_END_ID in gen1:
                    ci = gen1.index(THINK_END_ID) + 1
                    cot_s0, call_s0 = gen1[:ci], gen1[ci:]
                else:
                    cot_s0, call_s0 = [], gen1
                call_txt = self.tok.decode(call_s0, skip_special_tokens=False)
                tool_out_json = self._exec_Builtin_server_tool(call_txt)
                z_head = self.tok.encode(
                    f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{tool_out_json}", add_special_tokens=False
                )
                z_tail_str = "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
                if not enable_thinking:
                    z_tail_str += "<think>\n\n</think>\n\n"
                z_tail = self.tok.encode(z_tail_str, add_special_tokens=False)
                pad_n = self.tool_cap - len(z_head) - len(z_tail)
                if pad_n < 0:
                    raise ValueError(f"server tool return {len(z_head) + len(z_tail)} tokens exceeds tool_cap={self.tool_cap}")
                # Pad hidden tool return to constant `self.tool_cap` tokens so `n_ret` never leaks secret length.
                z_tool_ids = [int(x) for x in z_head] + [self.nl_tok_id] * pad_n + [int(x) for x in z_tail]
                rem_budget = max_tokens - len(gen1)
                prompt2 = prompt_ids + gen1 + z_tool_ids
                sp2 = SamplingParams(
                    n=1,
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=rem_budget,
                    stop_token_ids=[self.eos_id],
                    include_stop_str_in_output=True,
                    ignore_eos=False,
                    detokenize=False,
                    extra_args={
                        VCZ.SAMPLER_KEY: {
                            "nonce": nonce_hex,
                            "server_nonce": ns.hex(),
                            "dig": dig.hex(),
                            "v_rel": int(v_rel),
                            "sess": sess_b.hex(),
                            "turn": turn,
                            "prompt_len": len(prompt2),
                            "g_offset": len(gen1),
                        }
                    },
                )
                # Reuse vLLM's cached prefix pages via PrefixPortCache so KV cache and captured ports stay matched.
                _, gen2, cap2, cap_meta2 = self.loop.run(prompt2, sp2, self.t_pad)
                VCZ.pop_server_nonce(nonce_hex)
                cap_meta["max_cobatch"] = max(cap_meta.get("max_cobatch", 1), cap_meta2.get("max_cobatch", 1))
                full_tokens = prompt2 + gen2
                cap_final = _concat_captures(cap1, cap2, full_tokens)
                if enable_thinking and THINK_END_ID in gen2:
                    ci2 = gen2.index(THINK_END_ID) + 1
                    cot_tokens, vis_tokens = gen2[:ci2], gen2[ci2:]
                elif enable_thinking:
                    if gen2 and gen2[-1] == self.eos_id:
                        cot_tokens, vis_tokens = gen2[:-1], [gen2[-1]]
                    else:
                        cot_tokens, vis_tokens = gen2, []
                else:
                    cot_tokens, vis_tokens = [], gen2
                if on_delta is not None and vis_tokens:
                    on_delta(vis_tokens)
                tool_segments.append({
                    "n_reason": len(cot_s0),
                    "call_tokens": call_s0,
                    "n_ret": len(z_tool_ids),
                    "ret_tokens": None,    # private server-tool return witness.
                })
                last_tok = gen2[-1] if gen2 else -1
                if last_tok == self.eos_id:
                    finish_reason = APZ.FINISH_STOP
                else:
                    finish_reason = APZ.FINISH_LENGTH
                all_hist_tokens = cot_s0 + z_tool_ids + cot_tokens
            else:
                gen_ids = gen1
                full_tokens = prompt_ids + gen_ids
                cap_final = cap1
                if enable_thinking and THINK_END_ID in gen_ids:
                    ci = gen_ids.index(THINK_END_ID) + 1
                    cot_tokens, vis_tokens = gen_ids[:ci], gen_ids[ci:]
                elif enable_thinking:
                    # Unclosed `<think>`: never expose hidden CoT in visible tokens.
                    if gen_ids and gen_ids[-1] in (self.eos_id, self.tool_end_id):
                        cot_tokens, vis_tokens = gen_ids[:-1], [gen_ids[-1]]
                    else:
                        cot_tokens, vis_tokens = gen_ids, []
                else:
                    cot_tokens, vis_tokens = [], gen_ids
                all_hist_tokens = list(cot_tokens)
                if gen_ids and gen_ids[-1] == self.tool_end_id:
                    finish_reason = APZ.FINISH_TOOL
                elif gen_ids and gen_ids[-1] == self.eos_id:
                    finish_reason = APZ.FINISH_STOP
                else:
                    finish_reason = APZ.FINISH_LENGTH

        if max_tokens == 0:
            all_hist_tokens = []

        api_meta = {
            "finish_reason": finish_reason,
            "max_tokens": max_tokens,
            "eos_token_id": self.eos_id,
        }
        if has_tools:
            api_meta["tool_end_token_id"] = self.tool_end_id
        if tool_segments:
            api_meta["tool_segments"] = tool_segments
        if schema_obj is not None:
            api_meta["schema"] = schema_obj
            if is_tool_schema or tool_segments:
                api_meta["schema_on_tool_call"] = True
            if schema_prefix:
                api_meta["schema_prefix_hex"] = schema_prefix.hex()
            if schema_suffix:
                api_meta["schema_suffix_hex"] = schema_suffix.hex()

        api_view = {
            "req_tokens": req_tokens,
            "n_reason": len(cot_tokens),
            "vis_tokens": vis_tokens,
            "api_meta": api_meta,
        }

        r_state = [int.from_bytes(os.urandom(7), "little") for _ in range(4)]
        st_new, next_prev_bundle = SZ.new_turn(
            v_rel, self.model_id_words, sess, turn, st_prev,
            req_tokens, vis_tokens, nc, ns, all_hist_tokens, r_state, cap=self.hist_cap,
        )
        st_stmt = SZ.statement(
            v_rel, self.model_id_words, sess, turn, st_prev, st_new, nc, ns, hist_cap=self.hist_cap
        )
        st_wit = {"r": r_state, "prev": prev_bundle}
        state_blob_hex = APZ.seal_state_blob(
            self.wrap_key, self.model_id_words, sess, turn, st_new, next_prev_bundle, self.hist_cap
        )

        gen_s = round(time.perf_counter() - t_start, 4)
        job_meta = {
            "T": len(full_tokens),
            "n_req": len(req_tokens),
            "n_reason": len(cot_tokens),
            "n_vis": len(vis_tokens),
            "finish_reason": finish_reason,
            "generate_s": gen_s,
            "max_cobatch": cap_meta.get("max_cobatch", 1),
            "cg_modes": cap_meta.get("modes", {}),
            "prefix_cache_hits": cap_meta.get("prefix_cache_hits", 0),
            "full_graph_replays_total": self.cap.stats["full_graph_replays"],
            "py_port_calls_by_mode": dict(self.cap.stats["py_port_calls_by_mode"]),
        }
        if cap_final is None or cap_meta.get("error"):
            self.spool.submit(proof_id, {
                "proof_id": proof_id,
                "unprovable": str(cap_meta.get("error", "capture missing")),
                "model_id": self.model_id_hex,
                "created": time.time(),
                "meta": job_meta,
            })
        else:
            raw_req_clean = {k: v for k, v in body.items() if k != "fpn"}
            raw_req_clean["fpn"] = {
                k: v for k, v in ta.items() if k in ("thinking", "turn", "St_prev", "server_tools")
            }
            self.spool.submit(proof_id, {
                "proof_id": proof_id,
                "model_id": self.model_id_hex,
                "cap": cap_final,
                "full_tokens": [int(x) for x in full_tokens],
                "sys_len": len(self.sys_tokens),
                "api": api_view,
                "raw_req": raw_req_clean,
                "st_stmt": st_stmt,
                "st_wit": st_wit,
                "nonce": nonce_hex,
                "turn": turn,
                "created": time.time(),
                "meta": job_meta,
            })

        vis_clean = [t for t in vis_tokens if t not in (self.eos_id, self.tool_end_id)]
        vis_text = self.tok.decode(vis_clean, skip_special_tokens=False)
        return {
            "provable": True,
            "proof_id": proof_id,
            "req_tokens": req_tokens,
            "vis_tokens": vis_tokens,
            "vis_text": vis_text,
            "n_reason": len(cot_tokens),
            "finish_reason": finish_reason,
            "max_tokens": max_tokens,
            "turn": turn,
            "sess": list(sess),
            "St_prev": list(st_prev),
            "St_new": list(st_new),
            "N_s": ns.hex(),
            "state_blob": state_blob_hex,
            "tool_segments": tool_segments or None,
            "schema": api_meta.get("schema"),
            "meta": job_meta,
        }


def make_app(S: FpnZkVLLMServer):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    app = FastAPI()

    def fpn_env(res: dict) -> dict:
        if not res.get("provable"):
            return {
                "provable": False,
                "model_id": S.model_id_hex,
                "proof_id": None,
                "refusal_reason": res.get("refusal_reason"),
            }
        return {
            "provable": True,
            "model_id": S.model_id_hex,
            "profile": S.profile_name,
            "proof_id": res["proof_id"],
            "proof_url": f"/v1/proofs/{res['proof_id']}",
            "turn": res["turn"],
            "sess": res["sess"],
            "St_prev": res["St_prev"],
            "St_new": res["St_new"],
            "N_s": res["N_s"],
            "state_blob": res["state_blob"],
            "n_reason": res["n_reason"],
            "tool_segments": res["tool_segments"],
        }

    def format_json_response(res: dict) -> JSONResponse:
        if not res.get("provable"):
            return JSONResponse({
                "object": "chat.completion",
                "choices": [],
                "fpn": fpn_env(res),
            }, status_code=200)
        usage = {
            "prompt_tokens": len(res["req_tokens"]),
            "completion_tokens": res["n_reason"] + len(res["vis_tokens"]),
            "completion_tokens_details": {"reasoning_tokens": res["n_reason"]},
        }
        msg: dict = {"role": "assistant", "content": res["vis_text"]}
        if res["finish_reason"] == APZ.FINISH_TOOL:
            msg["tool_calls"] = [{"id": "call_" + res["proof_id"][:8], "type": "function", "function": {"raw": res["vis_text"]}}]
        return JSONResponse({
            "id": "chatcmpl-" + res["proof_id"][:16],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": S.profile_name,
            "prompt_token_ids": res["req_tokens"],
            "choices": [{
                "index": 0,
                "message": msg,
                "token_ids": res["vis_tokens"],
                "finish_reason": res["finish_reason"],
            }],
            "usage": usage,
            "fpn": fpn_env(res),
        })

    async def handle(request: Request):
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
        loop = asyncio.get_running_loop()
        if not body.get("stream"):
            try:
                res = await loop.run_in_executor(None, S.handle_generate, body, None)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            return format_json_response(res)

        # Check public policy synchronously before opening SSE stream
        ok_pol, reason_pol = APZ.check_public_policy(body, committed_rule=S.rule)
        if not ok_pol:
            return JSONResponse({
                "object": "chat.completion",
                "choices": [],
                "fpn": {"provable": False, "model_id": S.model_id_hex, "proof_id": None, "refusal_reason": reason_pol},
            }, status_code=200)

        q: asyncio.Queue = asyncio.Queue()

        def on_delta(vis_ids: list[int]):
            clean = [t for t in vis_ids if t not in (S.eos_id, S.tool_end_id)]
            txt = S.tok.decode(clean, skip_special_tokens=False) if clean else ""
            loop.call_soon_threadsafe(q.put_nowait, ("delta", (vis_ids, txt)))

        def worker():
            try:
                res = S.handle_generate(body, on_delta=on_delta)
                loop.call_soon_threadsafe(q.put_nowait, ("done", res))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

        loop.run_in_executor(None, worker)

        async def sse():
            while True:
                kind, payload = await q.get()
                if kind == "delta":
                    v_ids, txt = payload
                    ev = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": txt}, "token_ids": v_ids}],
                    }
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                elif kind == "done":
                    ev = {
                        "object": "chat.completion.chunk",
                        "prompt_token_ids": payload["req_tokens"],
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": payload["finish_reason"],
                            "token_ids": [],
                        }],
                        "fpn_reply_token_ids": payload["vis_tokens"],
                        "fpn": fpn_env(payload),
                    }
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                else:
                    yield f"data: {json.dumps({'error': payload})}\n\n"
                    return

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request)

    @app.post("/v1/responses")
    async def responses_ep(request: Request):
        return await handle(request)

    @app.post("/v1/messages")
    async def messages_ep(request: Request):
        return await handle(request)

    @app.get("/v1/proofs/{proof_id}")
    async def get_proof(proof_id: str):
        state, _ = S.spool.status(proof_id)
        if state == "ready":
            blob = S.spool.ready_bytes(proof_id)
            if blob is None:
                return JSONResponse({"status": "withheld"}, status_code=409)
            return Response(blob, media_type="application/octet-stream", headers={"x-fpn-proof-id": proof_id})
        if state in ("pending", "working"):
            return JSONResponse({"status": "pending"}, status_code=202)
        if state == "failed":
            return JSONResponse({"status": "withheld"}, status_code=409)
        return JSONResponse({"status": "unknown"}, status_code=404)

    @app.get("/v1/fpn/model")
    async def get_model_info():
        return {"model_id": S.model_id_hex, "profile": S.profile_name}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Hugging Face model directory (bf16 safetensors)")
    ap.add_argument("--store", required=True, help="Committed W store directory (W_coeffs.i32, setup_meta.json, serving.json)")
    ap.add_argument("--public", required=True, help="public.json with {model_id, profile}")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.22)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument("--prefix-mem-cap", type=int, default=256)
    ap.add_argument("--cudagraph", action="store_true")
    args = ap.parse_args()
    srv = FpnZkVLLMServer(args)
    uvicorn.run(make_app(srv), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
