# SPDX-License-Identifier: Apache-2.0
"""Non-invasive vLLM 0.33 capture plugin: zero-synchronization taps -> staging ring -> DMA sink -> page store.

STATUS: validated end-to-end against live ``vLLM 0.33`` on RTX 6000D across dense (``Qwen3-0.6B``, ``Qwen3-4B``) and
sparse MoE (``Qwen3-30B-A3B-FP8``) models via ``bench/tier1/bench_capture.py`` and ``bench/tier2/bench_live_vllm_*``.

Design
------
* ``register_loader()`` registers ``load_format="fpn_capture"``.  It wraps the stock ``DefaultModelLoader`` (unmodified
  weights) and, only when enabled, instruments ``nn.Module`` forwards by structural introspection BEFORE
  ``torch.compile`` / CUDA-graph capture.  Disabled => no allocation, no wrapping: zero VRAM, zero overhead.
* Taps are ``torch.ops.fpn.tap*`` custom ops registered ``has_side_effect``: Dynamo/Inductor keep them, the compiled
  piecewise graphs and ``FULL_AND_PIECEWISE`` CUDA graphs replay them with ZERO Python callbacks.  Each op launches an
  NVRTC kernel (``tap_kernels.cu``) that reads the real row count / ring cursor from a device control block, so a replay
  with capture-time launch arguments still writes exactly the step's rows.
* The attention epilogue (q, o, LSE; NO K/V -- those stay in vLLM's paged cache and are referenced by block hash) is
  launched directly from the wrapped ``FlashAttentionImpl.forward`` (run once during graph capture, recorded into it).
* Per step the runner wrappers do host-only bookkeeping (reserve ring rows, stream-ordered async copy of
  ``[cursor, n_valid]``), and after the forward record an event; the DMA sink copies on its own stream behind that event.
  There is no ``synchronize`` call anywhere in ``serve/``.
* Fail closed: ring overflow poisons the step's requests and the step runs with ``n_valid = 0``; an unresolvable token
  source poisons the request.
* Fused MoE : the routed LOGICAL ``topk_ids`` are tapped through vLLM's own ``BaseRouter.capture_fn`` hook
  (the ids the fused kernel routes with) into the ``expert_ids`` slot; the fused kernel exposes no down-projection
  input, so MoE models are captured with ``capture_act=False`` (slot ``a`` dropped).  Shared experts, monolithic MoE
  kernels and mixed dense / MoE stacks raise ``NotImplementedError``.

Usage (single process, ``VLLM_ENABLE_V1_MULTIPROCESSING=0``)::

    from fpn_serve.engines.vllm import capture_plugin as cp
    cp.configure_capture(cp.CaptureConfig(tier=cp.CaptureTier.TIER_B, ring_mib=128, max_batched_tokens=512))
    llm = LLM(model, load_format="fpn_capture", max_num_batched_tokens=512, ...)
    llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker  # -> cp.install_runner(worker)
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ...capture.ports import CaptureError, CaptureTier, ModelDims, MoEDims
from ...capture.session import CaptureConfig, CaptureSession

__all__ = ["CaptureConfig", "CaptureTier", "configure_capture", "disable_capture", "register_ops", "register_loader",
           "instrument_model", "install_runner", "get_session", "is_enabled"]

_S: dict[str, Any] = {
    "config": None,        # CaptureConfig
    "session": None,       # CaptureSession
    "launcher": None,      # TapLauncher
    "slots": [],           # list[PortSlot] indexed by the int ids baked into the traced graphs
    "slot_ids": {},        # (kind, layer) -> id
    "ops": False,
    "loader": False,
    "token_provider": None,
}


def is_enabled() -> bool:
    return _S["session"] is not None


def get_session() -> Optional[CaptureSession]:
    return _S["session"]


def configure_capture(config: CaptureConfig, token_provider: Optional[Callable[[Any, str], list[int]]] = None) -> None:
    """Set the capture config for the next model load.  ``enabled=False`` frees everything (zero VRAM)."""
    if not config.enabled:
        disable_capture()
        return
    _S["config"] = config
    _S["token_provider"] = token_provider
    register_loader()


def disable_capture() -> None:
    sess = _S["session"]
    if sess is not None:
        sess.close()
    launcher = _S["launcher"]
    if launcher is not None and hasattr(launcher, "close"):
        launcher.close()
    for k in ("config", "session", "launcher", "token_provider"):
        _S[k] = None
    _S["slots"], _S["slot_ids"] = [], {}


# ============================================================================================================ custom ops
def register_ops() -> None:
    """Register ``torch.ops.fpn.{tap,tap_pair,tap_ids}`` (idempotent; imports torch lazily)."""
    if _S["ops"]:
        return
    import torch
    from torch.fx.node import has_side_effect

    from ...capture.kernels import _load_cpp_ext

    def fake(*args):
        return None

    if _load_cpp_ext() is None:
        @torch.library.custom_op("fpn::tap", mutates_args=(), schema="(Tensor x, int slot) -> ()")
        def tap(x, slot):
            launcher = _S["launcher"]
            if launcher is not None:
                launcher.tap_port(x, _S["slots"][slot])

        @torch.library.custom_op(
            "fpn::tap_pair", mutates_args=(), schema="(Tensor res, Tensor nrm, int slot_res, int slot_nrm) -> ()"
        )
        def tap_pair(res, nrm, slot_res, slot_nrm):
            launcher = _S["launcher"]
            if launcher is not None:
                slots = _S["slots"]
                launcher.tap_residual_and_norm(res, slots[slot_res], nrm, slots[slot_nrm])

        @torch.library.custom_op(
            "fpn::tap_attn_no_kv", mutates_args=(),
            schema="(Tensor q, Tensor o, Tensor lse, int slot_q, int slot_o, int slot_lse) -> ()"
        )
        def tap_attn_no_kv(q, o, lse, slot_q, slot_o, slot_lse):
            launcher = _S["launcher"]
            if launcher is not None:
                slots = _S["slots"]
                launcher.tap_attn_no_kv(q, o, lse, slots[slot_q], slots[slot_o], slots[slot_lse])

        for op in (tap, tap_pair, tap_attn_no_kv):
            op.register_fake(fake)
    else:
        for name in ("tap", "tap_pair", "tap_attn_no_kv"):
            torch.library.register_fake(f"fpn::{name}", fake)

    @torch.library.custom_op("fpn::tap_ids", mutates_args=(), schema="(Tensor ids, int slot) -> ()")
    def _op_tap_ids(ids, slot):
        ln = _S["launcher"]
        if ln is not None:
            ln.tap_expert_ids(ids, _S["slots"][slot])

    @_op_tap_ids.register_fake
    def _op_tap_ids_fake(ids, slot):
        return None

    for op in (torch.ops.fpn.tap.default, torch.ops.fpn.tap_pair.default,
               torch.ops.fpn.tap_attn_no_kv.default, torch.ops.fpn.tap_ids.default):
        has_side_effect(op)
    _S["ops"] = True


def _sid(kind: str, layer: int) -> int:
    return _S["slot_ids"][(kind, layer)]


# ======================================================================================================= instrumentation
def _wrap_forward(mod: Any, fn: Callable[..., Any]) -> None:
    orig = mod.forward
    mod.forward = lambda *a, **k: fn(orig, *a, **k)


def _moe_router(mlp: Any) -> Any:
    """The vLLM ``BaseRouter`` of a fused-MoE block (``mlp.experts.router``), or ``None``.  Its ``capture_fn`` hook
    receives the LOGICAL ``topk_ids`` the fused kernel routes with (before any EPLB mapping)."""
    router = getattr(getattr(mlp, "experts", None), "router", None)
    return router if callable(getattr(router, "set_capture_fn", None)) else None


def _first_int(*vals: Any) -> int:
    for v in vals:
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return 0


def _discover_dims(model: Any) -> ModelDims:
    backbone = getattr(model, "model", model)
    layers = list(backbone.layers)
    l0 = layers[0]
    sa, attn = l0.self_attn, l0.self_attn.attn
    nq = int(getattr(sa, "num_heads", getattr(attn, "num_heads", 0)))
    nkv = int(getattr(sa, "num_kv_heads", getattr(attn, "num_kv_heads", 0)))
    dh = int(getattr(sa, "head_dim", getattr(attn, "head_size", 0)))
    d = int(l0.input_layernorm.weight.shape[0])
    if hasattr(l0.mlp, "down_proj"):
        dff = int(l0.mlp.down_proj.weight.shape[1])
        return ModelDims(len(layers), d, nq, nkv, dh, dff, None, type(model).__name__)
    experts = getattr(l0.mlp, "experts", None)
    routed = getattr(experts, "routed_experts", None)
    if experts is None or _moe_router(l0.mlp) is None:
        raise NotImplementedError("MLP block is neither dense (down_proj) nor a vLLM fused MoE with a BaseRouter")
    if getattr(l0.mlp, "shared_expert", None) is not None:
        raise NotImplementedError("MoE with shared experts: the shared-expert boundary is not wired")
    n_exp = _first_int(getattr(experts, "global_num_experts", None), getattr(routed, "global_num_experts", None),
                       getattr(l0.mlp, "n_routed_experts", None))
    top_k = _first_int(getattr(experts, "top_k", None), getattr(routed, "top_k", None))
    d_exp = _first_int(getattr(experts, "intermediate_size_per_partition", None),
                       getattr(routed, "intermediate_size_per_partition", None))
    if not (n_exp and top_k and d_exp):
        raise NotImplementedError(f"cannot read MoE dims from {type(experts).__name__} "
                                  f"(n_experts={n_exp}, top_k={top_k}, d_expert={d_exp})")
    if not all(_moe_router(layer.mlp) is not None for layer in layers):
        raise NotImplementedError("mixed dense / MoE layers are not wired")
    return ModelDims(len(layers), d, nq, nkv, dh, 0, MoEDims(n_exp, top_k, d_exp), type(model).__name__)


def instrument_model(model: Any, config: CaptureConfig, device: Any) -> CaptureSession:
    """Structural, model-agnostic instrumentation of a decoder-only dense Transformer (``model.model.layers``)."""
    import torch

    from ...capture.kernels import TapLauncher

    dims = _discover_dims(model)
    if dims.moe is not None and config.capture_act:
        raise CaptureError("fused MoE has no down-projection input boundary to tap: configure capture_act=False")
    sess = CaptureSession.create(dims, config, backend="cuda", device=str(device))
    lay = sess.layout
    _S["session"] = sess
    _S["slots"] = list(lay.slots)
    _S["slot_ids"] = {(s.kind, s.layer): i for i, s in enumerate(lay.slots)}
    _S["launcher"] = TapLauncher(sess.ring.backend, lay.row_u16, _S["slots"])

    register_ops()
    tap, tap_pair = torch.ops.fpn.tap.default, torch.ops.fpn.tap_pair.default
    backbone = getattr(model, "model", model)
    layers = list(backbone.layers)
    L = len(layers)
    full = config.tier in (CaptureTier.FULL_NO_KV, CaptureTier.FULL_NO_KV_ACT)
    norms = lay.has("x_f", L)

    def mk_norm1(l: int) -> Callable[..., Any]:
        h_in, x_attn = _sid("h_in", l), _sid("x_attn", l) if norms else -1
        def norm1(of: Any, x: Any, residual: Any = None) -> Any:
            if residual is None:                         # layer 0: x IS the embedding / residual stream
                tap(x, h_in)
                out = of(x)
                if norms:
                    tap(out, x_attn)
                return out
            out = of(x, residual)
            if norms:
                tap_pair(out[1], out[0], h_in, x_attn)
            else:
                tap(out[1], h_in)
            return out
        return norm1

    def mk_norm2(l: int) -> Callable[..., Any]:
        h_mid, x_mlp = _sid("h_mid", l), _sid("x_mlp", l) if norms else -1
        def norm2(of: Any, x: Any, residual: Any) -> Any:
            out = of(x, residual)
            if norms:
                tap_pair(out[1], out[0], h_mid, x_mlp)
            else:
                tap(out[1], h_mid)
            return out
        return norm2

    def mk_act(l: int) -> Callable[..., Any]:
        slot = _sid("a", l)
        def act(of: Any, x: Any) -> Any:
            out = of(x)
            tap(out, slot)
            return out
        return act

    tap_ids = torch.ops.fpn.tap_ids.default

    def mk_ids(l: int) -> Callable[..., Any]:
        slot = _sid("expert_ids", l)
        def ids(topk_ids: Any) -> None:                  # vLLM BaseRouter.capture_fn: the routed logical top-k ids
            tap_ids(topk_ids, slot)
        return ids

    for l, layer in enumerate(layers):
        if full:
            layer.self_attn.attn._fpn_attn_slots = (_sid("q", l), _sid("o", l), _sid("lse", l))
        _wrap_forward(layer.input_layernorm, mk_norm1(l))
        if lay.has("h_mid", l):
            _wrap_forward(layer.post_attention_layernorm, mk_norm2(l))
        if lay.has("a", l):
            _wrap_forward(layer.mlp.act_fn, mk_act(l))
        if lay.has("expert_ids", l):
            _moe_router(layer.mlp).set_capture_fn(mk_ids(l))

    h_L, x_f = _sid("h_L", L), _sid("x_f", L) if norms else -1

    def final_norm(of: Any, x: Any, residual: Any = None) -> Any:
        if residual is None:
            tap(x, h_L)
            out = of(x)
            if norms:
                tap(out, x_f)
            return out
        out = of(x, residual)
        if norms:
            tap_pair(out[1], out[0], h_L, x_f)
        else:
            tap(out[1], h_L)
        return out

    _wrap_forward(backbone.norm, final_norm)
    if full:
        _patch_flash_attention()
    return sess


def _patch_flash_attention() -> None:
    """Wrap ``FlashAttentionImpl.forward`` so q, o and the *same call's* LSE are tapped (no K/V).  The kernel call is
    unchanged except that it also returns the softmax LSE it computes anyway (as ``vllm_capture_zk`` does)."""
    from vllm.v1.attention.backends import flash_attn as FAB

    import torch

    tap_attn = torch.ops.fpn.tap_attn_no_kv.default
    kern = FAB._FA4_DENSE_ATTENTION_KERNEL
    impl_cls = FAB.FlashAttentionImpl
    if not hasattr(kern, "_fpn_orig_kernel"):
        kern._fpn_orig_kernel = kern.kernel
        impl_cls._fpn_orig_forward = impl_cls.forward
    original_kernel = kern._fpn_orig_kernel
    original_forward = impl_cls._fpn_orig_forward
    active = False
    lse = None

    def kernel(*a: Any, **kw: Any) -> Any:
        nonlocal lse
        if not active:
            return original_kernel(*a, **kw)
        kw["return_softmax_lse"] = True
        result = original_kernel(*a, **kw)
        lse = result[1]
        return result[0]

    def forward(self: Any, layer: Any, query: Any, key: Any, value: Any, kv_cache: Any, attn_metadata: Any,
                output: Any = None, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, lse
        slots = getattr(layer, "_fpn_attn_slots", None)
        prev_active, prev_lse = active, lse
        active, lse = slots is not None and is_enabled(), None
        try:
            ret = original_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                                   output=output, *args, **kwargs)
            if lse is not None and output is not None:
                # the kernel ran on query[:num_actual_tokens]: its LSE covers only those tokens, while query / output
                # may be the padded (CUDA-graph / chunked-prefill) buffers -- tap the rows the LSE covers (the tap
                # kernel writes only the step's n_valid <= num_actual_tokens rows from the control block anyway)
                n = lse.shape[-1]
                if query.shape[0] > n:
                    query, output = query[:n], output[:n]
                tap_attn(query, output, lse, *slots)
            return ret
        finally:
            active, lse = prev_active, prev_lse

    kern.kernel = kernel
    impl_cls.forward = forward


# ================================================================================================================ loader
def register_loader() -> None:
    if _S["loader"]:
        return
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import DefaultModelLoader, register_model_loader

    @register_model_loader("fpn_capture")
    class FpnCaptureModelLoader(DefaultModelLoader):
        def __init__(self, load_config: LoadConfig):
            extra = dict(load_config.model_loader_extra_config or {})
            if "fpn_capture" in extra:                       # optional per-LLM override of the module-level config
                _S["config"] = CaptureConfig.from_dict(extra.pop("fpn_capture"))
            super().__init__(LoadConfig(
                load_format="auto", download_dir=load_config.download_dir, model_loader_extra_config=extra,
                ignore_patterns=load_config.ignore_patterns, use_tqdm_on_load=load_config.use_tqdm_on_load,
                pt_load_map_location=load_config.pt_load_map_location, device=load_config.device))

        def load_model(self, vllm_config: Any, model_config: Any, prefix: str = "") -> Any:
            model = super().load_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            cfg = _S["config"]
            if cfg is not None and cfg.enabled:
                instrument_model(model, cfg, next(model.parameters()).device)
            return model

    _S["loader"] = True


# ============================================================================================================== runner
def _default_tokens(runner: Any, rid: str) -> list[int]:
    st = getattr(runner, "requests", {}).get(rid)
    if st is None:
        raise CaptureError(f"request {rid!r} unknown to the runner")
    return list(st.prompt_token_ids) + list(getattr(st, "output_token_ids", []))


def _null_span(sess: CaptureSession) -> Any:
    from ...capture.ring import Span

    return Span(sess.ring.write_wp, 0)


def install_runner(worker: Any) -> str:
    """Wrap ``prepare_inputs`` / ``execute_model`` of the worker's model runner: reserve ring rows + publish the control
    block before the forward, record the step event after it.  Host-only, stream-ordered, no sync.

    Token ids (for the page hash chain) come from ``token_provider(runner, request_id)``; the default reads
    ``runner.requests[rid]`` (prompt + output tokens).  A request whose tokens cannot be resolved is poisoned."""
    sess = _S["session"]
    if sess is None:
        raise CaptureError("capture is not enabled (configure_capture + load_format='fpn_capture' first)")
    runner = worker.model_runner
    backend = sess.ring.backend
    sess.sink.start()                        # without a live drain the ring fills after a few steps and capture is poisoned
    o_prep, o_exec = runner.prepare_inputs, runner.execute_model
    o_sample = getattr(runner, "sample_tokens", None)
    pend: dict[str, Any] = {"step": None}
    req_tokens: dict[str, list[int]] = {}
    req_computed: dict[str, int] = {}
    custom_tok = _S["token_provider"]

    def _get_tokens(rid: str) -> list[int]:
        if custom_tok is not None:
            return custom_tok(runner, rid)
        if rid in req_tokens:
            return req_tokens[rid]
        return _default_tokens(runner, rid)

    def _resolve_deferred(rid: str, pos: int, n: int) -> Optional[tuple[int, ...]]:
        toks = req_tokens.get(rid)
        if toks is None or len(toks) < pos + n:
            return None
        return tuple(toks[pos:pos + n])

    def _release_req_tokens(rid: str) -> None:
        req_tokens.pop(rid, None)
        req_computed.pop(rid, None)

    sess.sink.token_resolver = _resolve_deferred
    sess.sink.on_settled = _release_req_tokens
    runner._fpn_req_tokens = req_tokens              # exposed for bounded-memory / leak auditing in tests

    def _record_sampled(out: Any) -> None:
        if out is None:
            return
        rids_o = getattr(out, "req_ids", None)
        stoks_o = getattr(out, "sampled_token_ids", None)
        if not rids_o or not stoks_o:
            return
        for rid, tids in zip(rids_o, stoks_o):
            if rid in req_tokens and tids:
                req_tokens[rid].extend(int(t) for t in tids)

    def _wrap_async_out(out: Any) -> Any:
        if out is None:
            return out
        if getattr(out, "req_ids", None) and getattr(out, "sampled_token_ids", None):
            _record_sampled(out)
            return out
        orig_get = getattr(out, "get_output", None)
        if callable(orig_get) and not getattr(out, "_fpn_wrapped", False):
            def _get_output(*ga: Any, **gk: Any) -> Any:
                res = orig_get(*ga, **gk)
                _record_sampled(res)
                return res
            out.get_output = _get_output
            out._fpn_wrapped = True
        return out

    def _segments(rids: list[str], qsl: list[int]) -> list[tuple]:
        segs: list[tuple] = []
        for i, rid in enumerate(rids):
            n = qsl[i + 1] - qsl[i]
            if n <= 0:
                continue
            try:
                if rid not in sess.scheduled:
                    if rid in req_computed:
                        computed = req_computed[rid]
                    else:
                        st = getattr(runner, "requests", {}).get(rid)
                        computed = int(getattr(st, "num_computed_tokens", 0)) if st is not None else 0
                    if computed % 16:
                        raise CaptureError("prefix hit is not page aligned (block_size != 16?)")
                    sess.register_request(rid, _get_tokens(rid)[:computed])
                pos = sess.scheduled[rid]
                all_toks = _get_tokens(rid)
                toks = all_toks[pos:pos + n]
                if len(toks) != n:
                    if rid in req_tokens and custom_tok is None:
                        # Async scheduling: sampled token arrives via AsyncModelRunnerOutput.get_output() off critical path
                        segs.append((rid, n, pos))
                        continue
                    raise CaptureError(f"token source returned {len(toks)} < {n} tokens at pos={pos}")
            except Exception as e:                                # noqa: BLE001 - poison, never crash the engine
                h = sess.store.handle(rid)
                if h is not None:
                    h.poison(f"token resolution failed: {e}")
                toks = [0] * n
                sess.scheduled[rid] = sess.scheduled.get(rid, 0) + n
                segs.append((rid, toks))
                continue
            segs.append((rid, toks))
        return segs

    def prepare_inputs(*a: Any, **k: Any) -> Any:
        ib = o_prep(*a, **k)
        pend["step"] = None
        try:
            n_req = int(ib.num_reqs)
            rids = list(ib.req_ids[:n_req])
            qsl = [int(x) for x in ib.query_start_loc_np[:n_req + 1]]
        except Exception:                                          # noqa: BLE001 - unknown batch layout: capture nothing
            backend.begin_step(_null_span(sess), 0)
            return ib
        step = sess.begin_step(_segments(rids, qsl))
        if step is None:                                           # overflow: requests poisoned; taps write nothing
            backend.begin_step(_null_span(sess), 0)
        else:
            backend.begin_step(step.span, step.span.n_rows)
            pend["step"] = step
        return ib

    def execute_model(*a: Any, **k: Any) -> Any:
        so = a[0] if a else k.get("scheduler_output")
        if so is not None:
            for nr in getattr(so, "scheduled_new_reqs", ()) or ():
                rid = nr.req_id
                ptoks = getattr(nr, "prompt_token_ids", None)
                if ptoks is not None:
                    req_tokens[rid] = list(ptoks)
                    req_computed[rid] = int(getattr(nr, "num_computed_tokens", 0))
            done = list(getattr(so, "finished_req_ids", ()) or ())  # finished before this step => after all their rows
            if done:
                sess.finish_requests(done)
        out = o_exec(*a, **k)
        step, pend["step"] = pend["step"], None
        if step is not None and not k.get("dummy_run"):
            sess.end_step(step, backend.end_step())
        return _wrap_async_out(out)

    runner.prepare_inputs = prepare_inputs
    runner.execute_model = execute_model
    if o_sample is not None:
        def sample_tokens(*a: Any, **k: Any) -> Any:
            return _wrap_async_out(o_sample(*a, **k))
        runner.sample_tokens = sample_tokens
    return f"{type(runner).__module__}.{type(runner).__name__}"
