# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic `vLLM 0.33` extension for `flashprove_next.reply_zk`:

  1. Official `@register_model_loader("flashprove_next")` (`FpnModelLoader`): loads unmodified native bf16 weights via
     `DefaultModelLoader` and instruments `nn.Module` submodules by generic structural introspection (`layers[l]` norms,
     `Attention`, `act_fn`, `model.norm`) BEFORE `torch.compile` and CUDA-graph capture run.
     * Zero per-model-class monkey-patching (never imports `qwen3.py`, `llama.py`, etc.).
     * Zero VRAM reservation and zero runtime overhead when disabled (`load_format="auto"`).
  2. In-graph `torch.ops.fpn.port(x, slot)` custom op: opaque with `has_side_effect`, traced into Dynamo/Inductor
     piecewise graphs and replayed inside `FULL_AND_PIECEWISE` CUDA graphs with ZERO Python callbacks on replay.
  3. `PrefixPortCache` + `ZkPortCapture`: attributes continuous-batch rows per step with one pinned D2H copy and mounts
     prefix-cached 16-token KV/port blocks when `enable_prefix_caching=True` skips cached prefill blocks.
  4. `FpnZkSampler(LogitsProcessor)`: GPU bucketed keyed sampling (`sampling_v3_torch`) at 0-indexed generation step
     `g = ctx.pos + t_base[slot]` (hiding `sys_len` and `n_prev`), plus `SchemaDFA` constrained-decoding token masks.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flashprove_next import sampling_v3 as S3
from flashprove_next import sampling_v3_torch as T3
from flashprove_next.dense_engine import Capture

B_PAGE = 16
KINDS = ("h_in", "x_attn", "q", "k", "v", "o", "h_mid", "x_mlp", "a", "lse")
_P = {
    "enabled": False,
    "max_batched_tokens": 2048,
    "buf": None,
    "slots": {},
    "ids": {},
    "layout": None,
    "calls": 0,
    "lse_calls": 0,
    "cur_layer": None,
}


def _sid(key) -> int:
    ids = _P["ids"]
    hit = ids.get(key)
    if hit is None:
        hit = ids[key] = len(ids)
    return hit


@torch.library.custom_op("fpn::port", mutates_args=())
def _fpn_port(x: torch.Tensor, slot: int) -> None:
    _P["calls"] += 1
    buf = _P["buf"]
    if buf is None:
        return
    c0, w = _P["slots"][slot]
    n = min(int(x.shape[0]), int(buf.shape[0]))
    src = x[:n]
    if src.dtype != buf.dtype:
        src = src.contiguous().view(buf.dtype)
    src = src.reshape(n, -1)
    buf[:n, c0:c0 + w].copy_(src)


@_fpn_port.register_fake
def _fpn_port_fake(x: torch.Tensor, slot: int) -> None:
    return None


from torch.fx.node import has_side_effect  # noqa: E402
has_side_effect(torch.ops.fpn.port.default)


def port(x: torch.Tensor, key) -> None:
    torch.ops.fpn.port(x, _sid(key))


def _wrap_forward(mod, fn):
    orig_f = mod.forward
    mod.forward = lambda *a, **k: fn(orig_f, *a, **k)


def configure_fpn_loader(max_batched_tokens: int = 2048, enabled: bool = True) -> None:
    """Enable (or disable with 0 VRAM / 0 overhead) the `load_format='flashprove_next'` model loader plugin."""
    _P["enabled"] = bool(enabled)
    _P["max_batched_tokens"] = int(max_batched_tokens)
    if not enabled:
        _P["buf"] = None
        _P["slots"].clear()
        _P["ids"].clear()
        _P["layout"] = None
        return
    register_fpn_loader()


def _instrument_transformer_model(model: torch.nn.Module, max_tokens: int, device: torch.device) -> dict:
    """Model-agnostic structural instrumentation of any decoder-only Transformer (`model.model.layers` + `model.model.norm`)."""
    from vllm.v1.attention.backends import flash_attn as FAB

    backbone = getattr(model, "model", model)
    layers = list(backbone.layers)
    L = len(layers)
    norm_final = backbone.norm

    # Discover hidden dimensions structurally from layer 0
    l0 = layers[0]
    sa0 = l0.self_attn
    attn_mod0 = sa0.attn
    nq = int(getattr(sa0, "num_heads", getattr(attn_mod0, "num_heads", 0)))
    nkv = int(getattr(sa0, "num_kv_heads", getattr(attn_mod0, "num_kv_heads", 0)))
    dh = int(getattr(sa0, "head_dim", getattr(attn_mod0, "head_size", 0)))
    D = int(l0.input_layernorm.weight.shape[0])
    # Discover MLP intermediate width dff from down_proj input dim
    dff = int(l0.mlp.down_proj.weight.shape[1])

    width = dict(
        h_in=D, x_attn=D, q=nq * dh, k=nkv * dh, v=nkv * dh,
        o=nq * dh, h_mid=D, x_mlp=D, a=dff, lse=2 * nq,
    )
    _P["ids"].clear()
    off, slots = 0, {}
    for l in range(L):
        for kd in KINDS:
            slots[_sid((l, kd))] = (off, width[kd])
            off += width[kd]
    for nm, w in (("pos", 4), ("xf", D), ("hL", D)):
        slots[_sid(nm)] = (off, w)
        off += w

    _P["slots"] = slots
    _P["buf"] = torch.zeros((max_tokens, off), dtype=torch.bfloat16, device=device)
    layout = dict(
        L=L, D=D, dff=dff, nq=nq, nkv=nkv, dh=dh, W=off,
        max_tokens=max_tokens, bytes_per_token=off * 2,
    )
    _P["layout"] = layout

    for l, layer in enumerate(layers):
        layer.self_attn.attn._fpn_l = l

        def mk_norm1(l_):
            def norm1(of, x, residual=None):
                if residual is None:
                    port(x, (l_, "h_in"))
                    out = of(x)
                    port(out, (l_, "x_attn"))
                    return out
                out = of(x, residual)
                port(out[0], (l_, "x_attn"))
                port(out[1], (l_, "h_in"))
                return out
            return norm1

        def mk_norm2(l_):
            def norm2(of, x, residual):
                out = of(x, residual)
                port(out[0], (l_, "x_mlp"))
                port(out[1], (l_, "h_mid"))
                return out
            return norm2

        def mk_act(l_):
            def act(of, x):
                out = of(x)
                port(out, (l_, "a"))
                return out
            return act

        def mk_attn(l_):
            def attn(of, query, key, value, *args, **kwargs):
                port(query, (l_, "q"))
                port(key, (l_, "k"))
                port(value, (l_, "v"))
                out = of(query, key, value, *args, **kwargs)
                port(out, (l_, "o"))
                return out
            return attn

        _wrap_forward(layer.input_layernorm, mk_norm1(l))
        _wrap_forward(layer.post_attention_layernorm, mk_norm2(l))
        _wrap_forward(layer.mlp.act_fn, mk_act(l))
        _wrap_forward(layer.self_attn.attn, mk_attn(l))

        if l == 0:
            def pos_wrap(of, *args, **kwargs):
                pos_t = kwargs.get("positions", args[0] if args else None)
                if pos_t is not None:
                    port(pos_t, "pos")
                return of(*args, **kwargs)
            _wrap_forward(layer.self_attn, pos_wrap)

    def fnorm(of, x, residual=None):
        if residual is None:
            port(x, "hL")
            out = of(x)
            port(out, "xf")
            return out
        out = of(x, residual)
        port(out[0], "xf")
        port(out[1], "hL")
        return out

    _wrap_forward(norm_final, fnorm)

    # LSE from the exact same FlashAttention kernel call (recorded inside CUDA graphs as well)
    kern = FAB._FA4_DENSE_ATTENTION_KERNEL
    if not hasattr(kern, "_orig_kernel"):
        kern._orig_kernel = kern.kernel
    impl_cls = FAB.FlashAttentionImpl
    if not hasattr(impl_cls, "_orig_forward"):
        impl_cls._orig_forward = impl_cls.forward

    def impl_forward(self, layer, *args, **kwargs):
        _P["cur_layer"] = getattr(layer, "_fpn_l", None)
        try:
            return impl_cls._orig_forward(self, layer, *args, **kwargs)
        finally:
            _P["cur_layer"] = None

    def kernel(*a, **kw):
        l, buf = _P["cur_layer"], _P["buf"]
        if l is None or buf is None:
            return kern._orig_kernel(*a, **kw)
        kw["return_softmax_lse"] = True
        r = kern._orig_kernel(*a, **kw)
        lse = r[1]   # [nq, num_actual_tokens] fp32
        _P["lse_calls"] += 1
        n = min(int(lse.shape[1]), int(buf.shape[0]))
        c0, w = _P["slots"][_sid((l, "lse"))]
        buf[:n, c0:c0 + w].copy_(lse[:, :n].t().contiguous().view(torch.bfloat16))
        return r[0]

    impl_cls.forward = impl_forward
    kern.kernel = kernel
    return layout


def _apply_test_int4_downgrade(model: torch.nn.Module) -> None:
    """Test-only hook (`FPN_TEST_INT4_DOWNGRADE=1`): simulate a dishonest provider running int4-quantized weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
                w = param.float()
                o, i = w.shape
                g = 64 if i % 64 == 0 else i
                wg = w.reshape(o, i // g, g)
                s = torch.clamp(wg.abs().amax(dim=-1, keepdim=True), min=1e-12) / 7.0
                q = torch.clamp(torch.round(wg / s), -8, 7)
                param.copy_((q * s).reshape(o, i).to(param.dtype))


_LOADER_REGISTERED = False


def register_fpn_loader() -> None:
    global _LOADER_REGISTERED
    if _LOADER_REGISTERED:
        return
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import DefaultModelLoader, register_model_loader

    @register_model_loader("flashprove_next")
    class FpnModelLoader(DefaultModelLoader):
        def __init__(self, load_config: LoadConfig):
            auto_cfg = LoadConfig(
                load_format="auto",
                download_dir=load_config.download_dir,
                model_loader_extra_config=load_config.model_loader_extra_config,
                ignore_patterns=load_config.ignore_patterns,
                use_tqdm_on_load=load_config.use_tqdm_on_load,
                pt_load_map_location=load_config.pt_load_map_location,
                device=load_config.device,
            )
            super().__init__(auto_cfg)

        def load_model(self, vllm_config, model_config, prefix: str = "") -> torch.nn.Module:
            model = super().load_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            if os.environ.get("FPN_TEST_INT4_DOWNGRADE") == "1":
                _apply_test_int4_downgrade(model)
            if _P["enabled"]:
                dev = next(model.parameters()).device
                _instrument_transformer_model(model, _P["max_batched_tokens"], dev)
            return model

    _LOADER_REGISTERED = True


# ============================================================================ Prefix Port Page Cache (B_PAGE = 16)
class PrefixPortCache:
    """Stores 16-token pages of raw `PORT_BUF` uint16 rows `(B_PAGE, W)` keyed by exact prefix token tuple so
    `enable_prefix_caching=True` requests that skip cached prefill blocks recover the exact port activations."""

    def __init__(self, page_dir: str | None = None, mem_cap: int = 256):
        self.mem_cap = int(mem_cap)
        self.by_key: OrderedDict[str, tuple[list[int], np.ndarray]] = OrderedDict()
        self.lock = threading.Lock()
        self.page_dir = page_dir
        if page_dir:
            os.makedirs(page_dir, mode=0o700, exist_ok=True)

    @staticmethod
    def _key(prefix_tokens: list[int]) -> str:
        import hashlib
        return hashlib.sha256(np.asarray(prefix_tokens, dtype=np.int32).tobytes()).hexdigest()

    def store_single_page(self, prefix_tokens: list[int], page_arr: np.ndarray) -> None:
        """Store one `B_PAGE`-token page (`len(prefix_tokens)` a positive multiple of `B_PAGE`, `page_arr` shape `(B_PAGE, W)`).
        First-write-wins so `PrefixPortCache` stays bit-identical to `vLLM`'s in-VRAM prefix KV cache blocks."""
        assert len(prefix_tokens) >= B_PAGE and len(prefix_tokens) % B_PAGE == 0 and page_arr.shape[0] == B_PAGE
        toks_p = [int(x) for x in prefix_tokens]
        k = self._key(toks_p)
        with self.lock:
            if k in self.by_key and self.by_key[k][0] == toks_p:
                return
            arr_copy = page_arr.copy()
            if self.mem_cap > 0:
                self.by_key[k] = (toks_p, arr_copy)
                while len(self.by_key) > self.mem_cap:
                    self.by_key.popitem(last=False)
        if self.page_dir:
            dst = os.path.join(self.page_dir, f"{k}.npz")
            if os.path.exists(dst):
                return
            tmp = os.path.join(self.page_dir, f"{k}.tmp.{os.getpid()}.{secrets.token_hex(4)}.npz")
            try:
                np.savez(tmp, toks=np.asarray(toks_p, dtype=np.int32), rows=arr_copy)
                os.replace(tmp, dst)
            except OSError:
                pass

    def store_request_pages(self, full_tokens: list[int], raw_rows: np.ndarray, skip_pages: set[int] | None = None) -> None:
        """Save every complete `B_PAGE`-token prefix page of `full_tokens` from `raw_rows` `(>= n_pages*B_PAGE, W)`."""
        n_pages = min(len(full_tokens), raw_rows.shape[0]) // B_PAGE
        for p in range(n_pages):
            if skip_pages is not None and p in skip_pages:
                continue
            end = (p + 1) * B_PAGE
            self.store_single_page(full_tokens[:end], raw_rows[end - B_PAGE:end])

    def load_prefix_rows(self, prompt_tokens: list[int], n_cached_tokens: int) -> np.ndarray | None:
        """Load `[0 .. n_cached_tokens)` (`n_cached_tokens` a multiple of `B_PAGE`) from RAM or disk; None on miss."""
        if n_cached_tokens <= 0:
            return np.zeros((0, 0), dtype=np.uint16)
        if n_cached_tokens % B_PAGE != 0 or n_cached_tokens > len(prompt_tokens):
            return None
        pages = []
        for p in range(n_cached_tokens // B_PAGE):
            end = (p + 1) * B_PAGE
            toks_p = [int(x) for x in prompt_tokens[:end]]
            k = self._key(toks_p)
            hit = None
            with self.lock:
                ent = self.by_key.get(k)
                if ent is not None and ent[0] == toks_p:
                    hit = ent[1]
            if hit is None and self.page_dir:
                fp = os.path.join(self.page_dir, f"{k}.npz")
                if os.path.exists(fp):
                    try:
                        with np.load(fp, allow_pickle=False) as z:
                            if z["toks"].tolist() == toks_p:
                                hit = z["rows"].astype(np.uint16, copy=True)
                    except Exception:
                        hit = None
            if hit is None:
                return None
            pages.append(hit)
        return np.concatenate(pages, axis=0)


# ============================================================================ Host-Side Batch Harvester
class _ActiveReq:
    def __init__(self, rid: str, prompt_tokens: list[int], max_total_len: int, width: int):
        self.rid = rid
        self.prompt_tokens = list(prompt_tokens)
        self.raw = np.zeros((max_total_len, width), dtype=np.uint16)
        self.filled = np.zeros(max_total_len, dtype=bool)
        self.max_pos = -1
        self.max_cobatch = 0
        self.modes: dict[str, int] = {}
        self.error: str | None = None
        self.prefix_cache_hits = 0
        self.saved_pages: set[int] = set()


class ZkPortCapture:
    """Runner-side harvester turning `PORT_BUF` rows into `flashprove_next.dense_engine.Capture` per request."""

    def __init__(self, prefix_cache: PrefixPortCache | None = None):
        self.reqs: dict[str, _ActiveReq] = {}
        self.lock = threading.Lock()
        self.prefix_cache = prefix_cache
        self._pending = None
        self._staging = None
        self._mode = "NONE"
        self.stats = dict(
            steps=0, harvested_steps=0, steps_by_mode={},
            py_port_calls_by_mode={}, full_graph_replays=0, d2h_bytes=0,
        )

    @property
    def layout(self) -> dict:
        assert _P["layout"] is not None, "FpnModelLoader has not initialized layout"
        return _P["layout"]

    def register(self, rid: str, prompt_tokens: list[int], max_total_len: int) -> None:
        with self.lock:
            self.reqs[rid] = _ActiveReq(rid, prompt_tokens, max_total_len, self.layout["W"])

    def finish(self, rid: str, full_tokens: list[int]) -> tuple[Capture | None, dict]:
        """Assemble `Capture` for `full_tokens` (`T = len(full_tokens)` -> forward rows `0 .. T - 2` required)."""
        with self.lock:
            req = self.reqs.pop(rid, None)
        if req is None:
            return None, {"error": "unknown request id"}
        T = len(full_tokens)
        need_fwd = max(0, T - 1)
        # Fill any missing prefix-cached blocks `[0 .. first_filled)` from PrefixPortCache.
        if need_fwd > 0 and not req.filled[:need_fwd].all():
            first_miss_end = 0
            while first_miss_end < need_fwd and not req.filled[first_miss_end]:
                first_miss_end += 1
            if first_miss_end > 0 and self.prefix_cache is not None:
                cached_rows = self.prefix_cache.load_prefix_rows(req.prompt_tokens, first_miss_end)
                if cached_rows is not None and cached_rows.shape[0] == first_miss_end:
                    req.raw[:first_miss_end] = cached_rows
                    req.filled[:first_miss_end] = True
                    req.prefix_cache_hits = first_miss_end
        if req.error is None and need_fwd > 0 and not req.filled[:need_fwd].all():
            miss = np.flatnonzero(~req.filled[:need_fwd]).tolist()[:8]
            req.error = f"missing forward port positions {miss} (need 0..{need_fwd - 1})"
        meta = {
            "error": req.error,
            "max_cobatch": req.max_cobatch,
            "modes": dict(req.modes),
            "prefix_cache_hits": req.prefix_cache_hits,
        }
        if req.error is not None:
            return None, meta

        # Save complete pages (including generated-token pages for multi-turn KV prefix hits) into PrefixPortCache
        if self.prefix_cache is not None and need_fwd >= B_PAGE and req.filled[:need_fwd].all():
            self.prefix_cache.store_request_pages(full_tokens[:need_fwd], req.raw[:need_fwd], skip_pages=req.saved_pages)

        lay = self.layout
        L, D, dff, nq, nkv, dh = lay["L"], lay["D"], lay["dff"], lay["nq"], lay["nkv"], lay["dh"]
        sl = _P["slots"]
        raw_t = np.zeros((T, lay["W"]), dtype=np.uint16)
        if need_fwd > 0:
            raw_t[:need_fwd] = req.raw[:need_fwd]

        h_in = np.zeros((T, L, D), dtype=np.uint16)
        x_attn = np.zeros((T, L, D), dtype=np.uint16)
        q_a = np.zeros((T, L, nq * dh), dtype=np.uint16)
        k_a = np.zeros((T, L, nkv * dh), dtype=np.uint16)
        v_a = np.zeros((T, L, nkv * dh), dtype=np.uint16)
        o_a = np.zeros((T, L, nq * dh), dtype=np.uint16)
        h_mid = np.zeros((T, L, D), dtype=np.uint16)
        x_mlp = np.zeros((T, L, D), dtype=np.uint16)
        a_a = np.zeros((T, L, dff), dtype=np.uint16)
        h_out = np.zeros((T, L, D), dtype=np.uint16)
        lse_a = np.zeros((T, L, nq), dtype=np.uint32)

        for l in range(L):
            for dst, kd in (
                (h_in, "h_in"), (x_attn, "x_attn"), (q_a, "q"), (k_a, "k"),
                (v_a, "v"), (o_a, "o"), (h_mid, "h_mid"), (x_mlp, "x_mlp"), (a_a, "a"),
            ):
                c0, w = sl[_sid((l, kd))]
                dst[:, l, :] = raw_t[:, c0:c0 + w]
            c0, w = sl[_sid((l, "lse"))]
            lse_a[:, l, :] = np.ascontiguousarray(raw_t[:, c0:c0 + w]).view(np.uint32)
        for l in range(L - 1):
            h_out[:, l, :] = h_in[:, l + 1, :]
        c0_hl, w_hl = sl[_sid("hL")]
        h_out[:, L - 1, :] = raw_t[:, c0_hl:c0_hl + w_hl]
        c0_xf, w_xf = sl[_sid("xf")]
        x_fn = raw_t[:, c0_xf:c0_xf + w_xf].copy()

        cap = Capture(
            tokens=np.asarray(full_tokens, dtype=np.int64),
            h_in=h_in, x_attn=x_attn, q=q_a, k=k_a, v=v_a, o=o_a,
            h_mid=h_mid, x_mlp=x_mlp, a=a_a, h_out=h_out,
            x_fn=x_fn, logits=np.zeros((0, 0), dtype=np.uint32), lse=lse_a,
        )
        return cap, meta

    def install_runner(self, worker) -> str:
        runner = worker.model_runner
        o_prep, o_exec = runner.prepare_inputs, runner.execute_model
        cap = self

        def prepare_inputs(*a, **k):
            ib = o_prep(*a, **k)
            try:
                n = int(ib.num_reqs)
                cap._pending = (list(ib.req_ids[:n]), [int(x) for x in ib.query_start_loc_np[:n + 1]])
            except Exception:
                cap._pending = None
            return ib

        runner.prepare_inputs = prepare_inputs
        mgr = getattr(runner, "cudagraph_manager", None)
        if mgr is not None:
            o_dispatch, o_full = mgr.dispatch, mgr.run_fullgraph

            def dispatch(*a, **k):
                d = o_dispatch(*a, **k)
                cap._mode = getattr(d.cg_mode, "name", str(d.cg_mode))
                return d

            def run_fullgraph(*a, **k):
                cap.stats["full_graph_replays"] += 1
                return o_full(*a, **k)

            mgr.dispatch = dispatch
            mgr.run_fullgraph = run_fullgraph

        def execute_model(*a, **k):
            cap._mode = "NONE"
            c0 = _P["calls"]
            out = o_exec(*a, **k)
            pend, cap._pending = cap._pending, None
            if pend is not None and not k.get("dummy_run"):
                st, m = cap.stats, cap._mode
                st["steps"] += 1
                st["steps_by_mode"][m] = st["steps_by_mode"].get(m, 0) + 1
                st["py_port_calls_by_mode"][m] = st["py_port_calls_by_mode"].get(m, 0) + (_P["calls"] - c0)
                cap._harvest(pend, m)
            return out

        runner.execute_model = execute_model
        return f"{type(runner).__module__}.{type(runner).__name__}"

    def _harvest(self, pend, mode: str) -> None:
        rids, qsl = pend
        ntok, nb = qsl[-1], len(rids)
        with self.lock:
            mine = []
            for i, r in enumerate(rids):
                c = self.reqs.get(r) or self.reqs.get(r.rsplit("-", 1)[0])
                if c is not None:
                    mine.append((i, c))
        if not mine:
            return
        buf = _P["buf"]
        if buf is None or ntok > buf.shape[0]:
            for _, c in mine:
                c.error = f"batch tokens {ntok} exceed PORT_BUF {0 if buf is None else buf.shape[0]}"
            return
        if self._staging is None or self._staging.shape != buf.shape:
            self._staging = torch.empty(buf.shape, dtype=buf.dtype, pin_memory=True)
        torch.cuda.synchronize()
        self._staging[:ntok].copy_(buf[:ntok], non_blocking=True)
        torch.cuda.synchronize()
        self.stats["harvested_steps"] += 1
        self.stats["d2h_bytes"] += ntok * buf.shape[1] * 2
        hv = self._staging[:ntok].view(torch.int16).numpy().view(np.uint16)
        pc0, _ = _P["slots"][_sid("pos")]
        pos_all = np.ascontiguousarray(hv[:, pc0:pc0 + 4]).view(np.int64).reshape(-1)
        for i, c in mine:
            q0, q1 = qsl[i], qsl[i + 1]
            pos = pos_all[q0:q1]
            c.max_cobatch = max(c.max_cobatch, nb)
            c.modes[mode] = c.modes.get(mode, 0) + 1
            if c.error or len(pos) == 0:
                continue
            p0, p1 = int(pos[0]), int(pos[-1])
            if p1 - p0 + 1 != len(pos) or p0 < 0 or p1 >= c.raw.shape[0]:
                c.error = f"non-contiguous or out-of-range positions {p0}..{p1} (cap {c.raw.shape[0]})"
                continue
            c.raw[p0:p1 + 1] = hv[q0:q1]
            c.filled[p0:p1 + 1] = True
            c.max_pos = max(c.max_pos, p1)
            # Eagerly publish newly-completed 16-token prompt pages so concurrent co-batched requests hitting
            # vLLM's KV prefix cache in the very next step (and finishing earlier) find the page immediately.
            if self.prefix_cache is not None and len(c.prompt_tokens) >= B_PAGE:
                max_prompt_page = len(c.prompt_tokens) // B_PAGE
                for pg in range(p0 // B_PAGE, min(p1 // B_PAGE + 1, max_prompt_page)):
                    if pg in c.saved_pages:
                        continue
                    s0, s1 = pg * B_PAGE, (pg + 1) * B_PAGE
                    if c.filled[s0:s1].all():
                        self.prefix_cache.store_single_page(c.prompt_tokens[:s1], c.raw[s0:s1])
                        c.saved_pages.add(pg)


# ============================================================================ FpnZkSampler (LogitsProcessor)
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor  # noqa: E402

SAMPLER_KEY = "fpn_zk"
_SAMPLER_CFG: dict = {}
SERVER_NONCES: dict[str, str] = {}
_SAMPLER_LOCK = threading.Lock()


def configure_sampler(kprf: list[int], rule: S3.Rule) -> None:
    _SAMPLER_CFG["kprf"] = [int(x) for x in kprf]
    _SAMPLER_CFG["rule"] = rule


def pop_server_nonce(nonce_hex: str) -> str | None:
    with _SAMPLER_LOCK:
        return SERVER_NONCES.pop(nonce_hex, None)


class FpnZkSampler(LogitsProcessor):
    """Custom V2 GPU LogitsProcessor for `reply_zk`:
      * Samples at 0-indexed generation step `g = ctx.pos + t_base[slot]` (`t_base = g_offset - (prompt_len - 1)`).
      * Enforces optional `SchemaDFA` constrained-decoding token masks (`next_q`, `allowed_mat`) per request slot."""

    def __init__(self, vllm_config, req_states):
        self.kprf = S3.kprf_elems(_SAMPLER_CFG["kprf"])
        self.rule: S3.Rule = _SAMPLER_CFG["rule"]
        self.V = int(req_states.vocab_size)
        self.dev = req_states.device
        n = int(req_states.max_num_reqs)
        self.engine = T3.Engine(self.V, self.rule, self.dev, max_slots=n)
        self.active = np.zeros(n, dtype=bool)
        self._prefix_cpu = torch.zeros(n, 12, dtype=torch.int64).pin_memory()
        self._dt_cpu = torch.zeros(n, dtype=torch.int64).pin_memory()
        self._tbase_cpu = torch.zeros(n, dtype=torch.int64).pin_memory()
        self._active_cpu = torch.zeros(n, dtype=torch.bool).pin_memory()
        self.active_gpu = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.tbase_gpu = torch.zeros(n, dtype=torch.int64, device=self.dev)
        self.dfa_slots: dict[int, dict] = {}
        self._dirty = False

    @classmethod
    def validate_params(cls, sp):
        return

    def add_request(self, req_idx: int, sp) -> bool:
        ta = (sp.extra_args or {}).get(SAMPLER_KEY)
        if ta is None:
            self.dfa_slots.pop(req_idx, None)
            if self.active[req_idx]:
                self.active[req_idx] = False
                self._active_cpu[req_idx] = False
                self._dirty = True
            return False
        nc = bytes.fromhex(ta["nonce"])
        ns = bytes.fromhex(ta["server_nonce"]) if "server_nonce" in ta else S3.server_nonce()
        with _SAMPLER_LOCK:
            SERVER_NONCES[ta["nonce"]] = ns.hex()
        ctx = S3.SampCtx(
            nc, ns,
            dig=bytes.fromhex(ta["dig"]),
            v_rel=int(ta["v_rel"]) & 0xFFFFFFFF,
            epoch=int(ta.get("epoch", 0)),
            sess=bytes.fromhex(ta["sess"]),
            turn=int(ta.get("turn", 0)),
        )
        pre = S3.prefix_state(self.kprf, ctx)
        self._prefix_cpu[req_idx] = torch.tensor([T3._i64(x) for x in pre], dtype=torch.int64)
        self._dt_cpu[req_idx] = int(ta.get("d_t", 0))
        prompt_len = int(ta["prompt_len"])
        g_offset = int(ta.get("g_offset", 0))
        self._tbase_cpu[req_idx] = g_offset - (prompt_len - 1)
        self._active_cpu[req_idx] = True
        self.engine.set_slot(req_idx, pre, int(ta.get("d_t", 0)), upload=False)
        self.active[req_idx] = True
        # Optional constrained-decoding DFA state tracker for this slot
        if ta.get("dfa_next_q") is not None and ta.get("dfa_allowed") is not None:
            allow_np = np.asarray(ta["dfa_allowed"], dtype=bool)
            self.dfa_slots[req_idx] = {
                "next_q": np.asarray(ta["dfa_next_q"], dtype=np.int16),
                "allow_gpu": torch.as_tensor(allow_np, dtype=torch.bool, device=self.dev),
                "q": 0,
                "constrained": bool(ta.get("start_constrained", True)),
                "think_end_id": ta.get("think_end_id"),
                "skip_newlines_after_think": bool(ta.get("skip_newlines_after_think", False)),
            }
        else:
            self.dfa_slots.pop(req_idx, None)
        self._dirty = True
        return True

    def apply_staged_writes(self):
        if self._dirty:
            self.engine.upload_all(self._prefix_cpu, self._dt_cpu)
            self.active_gpu.copy_(self._active_cpu, non_blocking=True)
            self.tbase_gpu.copy_(self._tbase_cpu, non_blocking=True)
            self._dirty = False

    def apply(self, logits: torch.Tensor, ctx) -> torch.Tensor:
        if not self.active[ctx.idx_mapping_np].any():
            return logits
        slots = ctx.expanded_idx_mapping.long()
        step_t = ctx.pos.long() + self.tbase_gpu[slots]
        lg = logits.float()
        has_dfa = bool(self.dfa_slots) and any(int(s) in self.dfa_slots for s in ctx.idx_mapping_np)
        if has_dfa:
            lg = lg.clone()
            slots_cpu = slots.cpu().tolist()
            for r_i, sl_i in enumerate(slots_cpu):
                dsl = self.dfa_slots.get(sl_i)
                if dsl is not None and dsl["constrained"]:
                    q_cur = dsl["q"]
                    if 0 <= q_cur < dsl["allow_gpu"].shape[0]:
                        mask_row = dsl["allow_gpu"][q_cur]
                        if mask_row.shape[0] < lg.shape[1]:
                            lg[r_i, mask_row.shape[0]:] = float("-inf")
                        lg[r_i, :mask_row.shape[0]] = torch.where(
                            mask_row, lg[r_i, :mask_row.shape[0]], torch.tensor(float("-inf"), device=lg.device)
                        )
        out, tok, use = self.engine.force(lg, slots, step_t, self.active_gpu)
        if has_dfa:
            tok_cpu = tok.cpu().tolist()
            use_cpu = use.cpu().tolist()
            slots_cpu = slots.cpu().tolist()
            for r_i, sl_i in enumerate(slots_cpu):
                if not use_cpu[r_i]:
                    continue
                dsl = self.dfa_slots.get(sl_i)
                if dsl is None:
                    continue
                tk = int(tok_cpu[r_i])
                if not dsl["constrained"]:
                    if dsl["think_end_id"] is not None and tk == int(dsl["think_end_id"]):
                        dsl["constrained"] = True
                        dsl["q"] = 0
                else:
                    nq = dsl["next_q"]
                    if 0 <= dsl["q"] < nq.shape[0] and 0 <= tk < nq.shape[1]:
                        dsl["q"] = int(nq[dsl["q"], tk])
        return out
