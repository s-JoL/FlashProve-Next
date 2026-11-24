# SPDX-License-Identifier: Apache-2.0
"""NVRTC loader + launch helpers for ``tap_kernels.cu`` (lazy torch; nothing here runs without a CUDA device).

All scalar kernel parameters are 32-bit ``int`` (see the .cu header), so the launcher only ever passes Python ints for
scalars and tensors for pointers -- no 32/64-bit marshalling ambiguity.  No ``synchronize`` anywhere: the launches are
ordinary stream-ordered kernels and are CUDA-graph capturable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from .ports import CaptureError, PortSlot

__all__ = ["KERNEL_NAMES", "kernel_source", "compile_kernels", "TapLauncher"]

KERNEL_NAMES = ("tap_single_port", "tap_residual_and_norm", "tap_attn_epilogue_no_kv", "tap_expert_ids_i32_to_u16",
                "advance_cursor_u64")
_BLOCK = 128
_CACHE: dict[str, Callable[..., Any]] = {}


def kernel_source() -> str:
    return Path(__file__).with_name("tap_kernels.cu").read_text(encoding="utf-8")


def compile_kernels() -> dict[str, Callable[..., Any]]:
    """Compile every kernel once per process (``torch.cuda._compile_kernel`` = NVRTC; needs no system nvcc)."""
    if not _CACHE:
        import torch
        import torch.utils.cpp_extension as ce

        if not getattr(ce, "CUDA_HOME", None):          # WSL has no toolkit: NVRTC ships with the torch wheels
            ce.CUDA_HOME = "/usr"
        src = kernel_source()
        for name in KERNEL_NAMES:
            _CACHE[name] = torch.cuda._compile_kernel(src, name)
    return _CACHE


def _u16_rows(x: Any) -> tuple[Any, int]:
    """View a bf16/fp16/int16 tensor as ``[rows, cols]`` int16 with unit inner stride; returns ``(view, row_stride)``."""
    import torch

    v = x if x.dtype == torch.int16 else x.view(torch.int16)
    if v.dim() != 2:
        try:
            v = v.view(v.shape[0], -1)
        except RuntimeError:
            v = v.reshape(v.shape[0], -1).contiguous()
    if v.stride(1) != 1:
        v = v.contiguous()
    return v, int(v.stride(0))


_EXT_MOD: Any = None
_EXT_TRIED: bool = False


def _load_cpp_ext() -> Any:
    """Compile a thin C++ launcher around tap_kernels.cu once via nvcc (~2 us/launch vs ~88 us for NVRTC Python).
    Falls back cleanly to None when nvcc/C++ toolchain is unavailable (e.g. minimal WSL containers)."""
    global _EXT_MOD, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT_MOD
    _EXT_TRIED = True
    try:
        import shutil
        import torch
        import torch.utils.cpp_extension as ce
        if not shutil.which("nvcc") and not getattr(ce, "CUDA_HOME", None):
            return None
        cu_src = kernel_source() + r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include <unordered_map>
#include <vector>

struct RingCtx {
    torch::Tensor buf;
    torch::Tensor ctl;
    int stride = 0;
    int capacity = 0;
    std::vector<int> slot_off;
    std::vector<int> slot_w;
    bool ready = false;
};
static std::mutex g_ctx_mu;
static std::unordered_map<int64_t, RingCtx> g_ctxs;
static int64_t g_next_ctx_id = 1;
static int64_t g_active_ctx_id = 0;

int64_t create_ring_ctx(torch::Tensor buf, torch::Tensor ctl, int stride, int capacity,
                        std::vector<int64_t> offsets, std::vector<int64_t> widths) {
    TORCH_CHECK(buf.is_cuda() && buf.is_contiguous() && buf.element_size() == 2,
                "fpn ring buf must be contiguous 16-bit CUDA tensor");
    TORCH_CHECK(ctl.is_cuda() && ctl.is_contiguous() && ctl.scalar_type() == torch::kInt64 && ctl.numel() >= 4,
                "fpn ring ctl must be contiguous int64[>=4] CUDA tensor");
    TORCH_CHECK(buf.device() == ctl.device(), "fpn ring buf and ctl must reside on the same CUDA device");
    TORCH_CHECK(stride > 0 && capacity > 0 && (int64_t)capacity * (int64_t)stride <= buf.numel(),
                "fpn ring geometry exceeds buf storage");
    TORCH_CHECK(offsets.size() == widths.size(), "fpn slot offsets/widths mismatch");
    for (size_t i = 0; i < offsets.size(); ++i) {
        TORCH_CHECK(offsets[i] >= 0 && offsets[i] <= (int64_t)stride
                    && widths[i] > 0 && widths[i] <= (int64_t)stride - offsets[i],
                    "fpn slot ", i, " out of row stride bounds");
    }
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    TORCH_CHECK(g_active_ctx_id == 0,
                "another native fpn capture context is already active; call close() before re-registering");
    int64_t cid = g_next_ctx_id++;
    RingCtx ctx;
    ctx.buf = buf;
    ctx.ctl = ctl;
    ctx.stride = stride;
    ctx.capacity = capacity;
    ctx.slot_off.assign(offsets.begin(), offsets.end());
    ctx.slot_w.assign(widths.begin(), widths.end());
    ctx.ready = true;
    g_ctxs[cid] = std::move(ctx);
    g_active_ctx_id = cid;
    return cid;
}

void release_ring_ctx(int64_t cid) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    g_ctxs.erase(cid);
    if (g_active_ctx_id == cid) g_active_ctx_id = 0;
}

void bind_ring_ctx(torch::Tensor buf, torch::Tensor ctl, int stride, int capacity,
                   std::vector<int64_t> offsets, std::vector<int64_t> widths) {
    create_ring_ctx(buf, ctl, stride, capacity, std::move(offsets), std::move(widths));
}

void clear_ring_ctx() {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    g_ctxs.clear();
    g_active_ctx_id = 0;
}

static inline RingCtx _get_ctx(int64_t cid) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    int64_t id = cid > 0 ? cid : g_active_ctx_id;
    auto it = g_ctxs.find(id);
    TORCH_CHECK(it != g_ctxs.end() && it->second.ready, "fpn ring context not bound");
    return it->second;
}

static inline int _gy_calc(int w) {
    int g = (w + 1023) / 1024;
    if (g < 1) g = 1;
    if (g > 16) g = 16;
    return g;
}

void fast_tap_slot_ctx(torch::Tensor x, int64_t slot_id, int64_t cid) {
    RingCtx ctx = _get_ctx(cid);
    TORCH_CHECK(slot_id >= 0 && slot_id < (int64_t)ctx.slot_off.size(), "invalid fpn tap slot");
    TORCH_CHECK(x.is_cuda() && x.element_size() == 2 && x.device() == ctx.buf.device(),
                "fpn tap input must be 16-bit CUDA tensor on ring device");
    int off = ctx.slot_off[slot_id];
    int w = ctx.slot_w[slot_id];
    int n_max = (int)x.size(0);
    if (n_max <= 0) return;
    torch::Tensor t = (x.dim() == 2 && x.stride(1) == 1) ? x : x.reshape({n_max, -1}).contiguous();
    TORCH_CHECK(t.size(1) == w, "fpn tap width mismatch: got ", t.size(1), " expected ", w);
    int st = (int)t.stride(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    tap_single_port<<<dim3(n_max, _gy_calc(w), 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const u16*>(t.data_ptr()), st,
        reinterpret_cast<u16*>(ctx.buf.data_ptr()), ctx.stride, ctx.capacity,
        reinterpret_cast<i64*>(ctx.ctl.data_ptr()), n_max, off, w);
}

void fast_tap_slot(torch::Tensor x, int64_t slot_id) { fast_tap_slot_ctx(x, slot_id, 0); }

void fast_tap_pair_slot_ctx(torch::Tensor res, torch::Tensor nrm, int64_t slot_res, int64_t slot_nrm, int64_t cid) {
    RingCtx ctx = _get_ctx(cid);
    TORCH_CHECK(slot_res >= 0 && slot_res < (int64_t)ctx.slot_off.size() &&
                slot_nrm >= 0 && slot_nrm < (int64_t)ctx.slot_off.size(), "invalid fpn pair slots");
    TORCH_CHECK(res.is_cuda() && res.element_size() == 2 && res.device() == ctx.buf.device(),
                "fpn res input must be 16-bit CUDA tensor on ring device");
    TORCH_CHECK(nrm.is_cuda() && nrm.element_size() == 2 && nrm.device() == ctx.buf.device(),
                "fpn nrm input must be 16-bit CUDA tensor on ring device");
    int off_r = ctx.slot_off[slot_res], w_r = ctx.slot_w[slot_res];
    int off_n = ctx.slot_off[slot_nrm], w_n = ctx.slot_w[slot_nrm];
    int n_max = (int)res.size(0);
    if (n_max <= 0) return;
    TORCH_CHECK((int)nrm.size(0) == n_max, "fpn pair row mismatch");
    torch::Tensor tr = (res.dim() == 2 && res.stride(1) == 1) ? res : res.reshape({n_max, -1}).contiguous();
    torch::Tensor tn = (nrm.dim() == 2 && nrm.stride(1) == 1) ? nrm : nrm.reshape({n_max, -1}).contiguous();
    TORCH_CHECK(tr.size(1) == w_r && tn.size(1) == w_n, "fpn pair width mismatch");
    auto stream = at::cuda::getCurrentCUDAStream();
    tap_residual_and_norm<<<dim3(n_max, _gy_calc(w_r > w_n ? w_r : w_n), 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const u16*>(tr.data_ptr()), (int)tr.stride(0), off_r, w_r,
        reinterpret_cast<const u16*>(tn.data_ptr()), (int)tn.stride(0), off_n, w_n,
        reinterpret_cast<u16*>(ctx.buf.data_ptr()), ctx.stride, ctx.capacity,
        reinterpret_cast<i64*>(ctx.ctl.data_ptr()), n_max);
}

void fast_tap_pair_slot(torch::Tensor res, torch::Tensor nrm, int64_t slot_res, int64_t slot_nrm) {
    fast_tap_pair_slot_ctx(res, nrm, slot_res, slot_nrm, 0);
}

void fast_tap_attn_slot_ctx(torch::Tensor q, torch::Tensor o, torch::Tensor lse,
                            int64_t slot_q, int64_t slot_o, int64_t slot_lse, int64_t cid) {
    RingCtx ctx = _get_ctx(cid);
    TORCH_CHECK(slot_q >= 0 && slot_q < (int64_t)ctx.slot_off.size() &&
                slot_o >= 0 && slot_o < (int64_t)ctx.slot_off.size() &&
                slot_lse >= 0 && slot_lse < (int64_t)ctx.slot_off.size(), "invalid fpn attention slots");
    TORCH_CHECK(q.is_cuda() && q.element_size() == 2 && q.device() == ctx.buf.device() &&
                o.is_cuda() && o.element_size() == 2 && o.device() == ctx.buf.device() &&
                lse.is_cuda() && lse.element_size() == 4 && lse.device() == ctx.buf.device(),
                "fpn attention tap tensors must be CUDA (16-bit q/o, 32-bit lse) on ring device");
    int off_q = ctx.slot_off[slot_q], w_q = ctx.slot_w[slot_q];
    int off_o = ctx.slot_off[slot_o], w_o = ctx.slot_w[slot_o];
    int off_l = ctx.slot_off[slot_lse], w_l = ctx.slot_w[slot_lse];
    int n_max = (int)q.size(0);
    if (n_max <= 0) return;
    TORCH_CHECK((int)o.size(0) == n_max && lse.dim() == 2 && (int)lse.size(1) >= n_max &&
                (int)lse.size(0) * 2 == w_l, "fpn attention tap shape mismatch");
    torch::Tensor tq = q.is_contiguous() ? q : q.reshape({n_max, -1}).contiguous();
    torch::Tensor to = o.is_contiguous() ? o : o.reshape({n_max, -1}).contiguous();
    torch::Tensor tl = (lse.stride(1) == 1) ? lse : lse.contiguous();
    TORCH_CHECK(tq.numel() == (int64_t)n_max * w_q && to.numel() == (int64_t)n_max * w_o,
                "fpn attention q/o width mismatch");
    int qst = (tq.dim() == 2) ? (int)tq.stride(0) : w_q;
    int ost = (to.dim() == 2) ? (int)to.stride(0) : w_o;
    auto stream = at::cuda::getCurrentCUDAStream();
    tap_attn_epilogue_no_kv<<<dim3(n_max, _gy_calc(w_q > w_o ? w_q : w_o), 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const u16*>(tq.data_ptr()), qst, off_q, w_q,
        reinterpret_cast<const u16*>(to.data_ptr()), ost, off_o, w_o,
        reinterpret_cast<const u32*>(tl.data_ptr()), (int)tl.stride(0), off_l, (int)tl.size(0),
        reinterpret_cast<u16*>(ctx.buf.data_ptr()), ctx.stride, ctx.capacity,
        reinterpret_cast<i64*>(ctx.ctl.data_ptr()), n_max);
}

void fast_tap_attn_slot(torch::Tensor q, torch::Tensor o, torch::Tensor lse,
                        int64_t slot_q, int64_t slot_o, int64_t slot_lse) {
    fast_tap_attn_slot_ctx(q, o, lse, slot_q, slot_o, slot_lse, 0);
}

void launch_tap_single_port(
    torch::Tensor src, int src_stride,
    torch::Tensor ring, int ring_stride, int capacity, torch::Tensor ctl, int n_max,
    int off, int w) {
    if (n_max <= 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(n_max, _gy_calc(w), 1);
    dim3 block(128, 1, 1);
    tap_single_port<<<grid, block, 0, stream>>>(
        reinterpret_cast<const u16*>(src.data_ptr()), src_stride,
        reinterpret_cast<u16*>(ring.data_ptr()), ring_stride, capacity,
        reinterpret_cast<i64*>(ctl.data_ptr()), n_max, off, w);
}

void launch_tap_residual_and_norm(
    torch::Tensor res, int res_stride, int off_res, int w_res,
    torch::Tensor nrm, int nrm_stride, int off_nrm, int w_nrm,
    torch::Tensor ring, int ring_stride, int capacity, torch::Tensor ctl, int n_max) {
    if (n_max <= 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(n_max, _gy_calc(w_res > w_nrm ? w_res : w_nrm), 1);
    dim3 block(128, 1, 1);
    tap_residual_and_norm<<<grid, block, 0, stream>>>(
        reinterpret_cast<const u16*>(res.data_ptr()), res_stride, off_res, w_res,
        reinterpret_cast<const u16*>(nrm.data_ptr()), nrm_stride, off_nrm, w_nrm,
        reinterpret_cast<u16*>(ring.data_ptr()), ring_stride, capacity,
        reinterpret_cast<i64*>(ctl.data_ptr()), n_max);
}

void launch_tap_attn_epilogue_no_kv(
    torch::Tensor q, int q_stride, int off_q, int w_q,
    torch::Tensor o, int o_stride, int off_o, int w_o,
    torch::Tensor lse, int lse_head_stride, int off_lse, int nq,
    torch::Tensor ring, int ring_stride, int capacity, torch::Tensor ctl, int n_max) {
    if (n_max <= 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(n_max, _gy_calc(w_q > w_o ? w_q : w_o), 1);
    dim3 block(128, 1, 1);
    tap_attn_epilogue_no_kv<<<grid, block, 0, stream>>>(
        reinterpret_cast<const u16*>(q.data_ptr()), q_stride, off_q, w_q,
        reinterpret_cast<const u16*>(o.data_ptr()), o_stride, off_o, w_o,
        reinterpret_cast<const u32*>(lse.data_ptr()), lse_head_stride, off_lse, nq,
        reinterpret_cast<u16*>(ring.data_ptr()), ring_stride, capacity,
        reinterpret_cast<i64*>(ctl.data_ptr()), n_max);
}

void launch_tap_expert_ids_i32_to_u16(
    torch::Tensor ids, int ids_stride, int off, int k,
    torch::Tensor ring, int ring_stride, int capacity, torch::Tensor ctl, int n_max) {
    if (n_max <= 0) return;
    TORCH_CHECK(ids.is_cuda() && ids.device() == ring.device(),
                "fpn expert_ids must be a CUDA tensor on ring device");
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.dim() == 2 &&
                (int)ids.size(1) == k && (int)ids.stride(1) == 1 && (int)ids.size(0) >= n_max &&
                (int)ids.stride(0) == ids_stride && ids_stride >= k,
                "fpn expert_ids must be row-contiguous 2D int32 [>=n_max, k] with stride(1)==1");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(n_max, 1, 1);
    dim3 block(128, 1, 1);
    tap_expert_ids_i32_to_u16<<<grid, block, 0, stream>>>(
        reinterpret_cast<const i32*>(ids.data_ptr()), ids_stride, off, k,
        reinterpret_cast<u16*>(ring.data_ptr()), ring_stride, capacity,
        reinterpret_cast<i64*>(ctl.data_ptr()), n_max);
}

void launch_advance_cursor_u64(torch::Tensor ctl) {
    auto stream = at::cuda::getCurrentCUDAStream();
    advance_cursor_u64<<<dim3(1, 1, 1), dim3(32, 1, 1), 0, stream>>>(
        reinterpret_cast<i64*>(ctl.data_ptr()));
}
"""
        cpp_src = """
#include <torch/library.h>
int64_t create_ring_ctx(torch::Tensor, torch::Tensor, int, int, std::vector<int64_t>, std::vector<int64_t>);
void release_ring_ctx(int64_t);
void bind_ring_ctx(torch::Tensor, torch::Tensor, int, int, std::vector<int64_t>, std::vector<int64_t>);
void clear_ring_ctx();
void fast_tap_slot(torch::Tensor, int64_t);
void fast_tap_slot_ctx(torch::Tensor, int64_t, int64_t);
void fast_tap_pair_slot(torch::Tensor, torch::Tensor, int64_t, int64_t);
void fast_tap_pair_slot_ctx(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
void fast_tap_attn_slot(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
void fast_tap_attn_slot_ctx(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);
void launch_tap_single_port(torch::Tensor, int, torch::Tensor, int, int, torch::Tensor, int, int, int);
void launch_tap_residual_and_norm(torch::Tensor, int, int, int, torch::Tensor, int, int, int, torch::Tensor, int, int, torch::Tensor, int);
void launch_tap_attn_epilogue_no_kv(torch::Tensor, int, int, int, torch::Tensor, int, int, int, torch::Tensor, int, int, int, torch::Tensor, int, int, torch::Tensor, int);
void launch_tap_expert_ids_i32_to_u16(torch::Tensor, int, int, int, torch::Tensor, int, int, torch::Tensor, int);
void launch_advance_cursor_u64(torch::Tensor);

TORCH_LIBRARY_FRAGMENT(fpn, m) {
    m.def("tap(Tensor x, int slot) -> ()");
    m.def("tap_pair(Tensor res, Tensor nrm, int slot_res, int slot_nrm) -> ()");
    m.def("tap_attn_no_kv(Tensor q, Tensor o, Tensor lse, int slot_q, int slot_o, int slot_lse) -> ()");
}
TORCH_LIBRARY_IMPL(fpn, CUDA, m) {
    m.impl("tap", &fast_tap_slot);
    m.impl("tap_pair", &fast_tap_pair_slot);
    m.impl("tap_attn_no_kv", &fast_tap_attn_slot);
}
"""
        _EXT_MOD = ce.load_inline(
            name="fpn_tap_kernels_ext_v6",
            cpp_sources=cpp_src,
            cuda_sources=cu_src,
            functions=[
                "create_ring_ctx",
                "release_ring_ctx",
                "bind_ring_ctx",
                "clear_ring_ctx",
                "fast_tap_slot",
                "fast_tap_slot_ctx",
                "fast_tap_pair_slot",
                "fast_tap_pair_slot_ctx",
                "fast_tap_attn_slot",
                "fast_tap_attn_slot_ctx",
                "launch_tap_single_port",
                "launch_tap_residual_and_norm",
                "launch_tap_attn_epilogue_no_kv",
                "launch_tap_expert_ids_i32_to_u16",
                "launch_advance_cursor_u64",
            ],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - fall back cleanly to NVRTC
        _EXT_MOD = None
    return _EXT_MOD


class TapLauncher:
    """Binds one ring backend (+ the layout's row geometry) to the compiled tap kernels."""

    def __init__(self, backend: Any, ring_stride_u16: int, slots: Any = None) -> None:
        self.ext = _load_cpp_ext()
        self.k = None if self.ext is not None else compile_kernels()
        self.buf = backend.buf
        self.ctl = backend.ctl
        self.capacity = int(backend.capacity_rows)
        self.stride = int(ring_stride_u16)
        self.ctx_id: int = 0
        if self.ext is not None and slots is not None:
            self.ctx_id = int(self.ext.create_ring_ctx(
                self.buf, self.ctl, self.stride, self.capacity,
                [int(s.offset) for s in slots], [int(s.width) for s in slots]))

    def close(self) -> None:
        if self.ext is not None and self.ctx_id:
            self.ext.release_ring_ctx(self.ctx_id)
            self.ctx_id = 0

    @staticmethod
    def _gy(width: int) -> int:
        return max(1, min(16, -(-width // (8 * _BLOCK))))

    def _launch(self, name: str, n_max: int, gy: int, args: tuple) -> None:
        if n_max > 0:
            self.k[name]((n_max, gy, 1), (_BLOCK, 1, 1), args)

    def tap_port(self, x: Any, slot: PortSlot) -> None:
        if self.ext is not None and x.dim() == 2 and x.stride(1) == 1 and x.shape[1] == slot.width:
            self.ext.launch_tap_single_port(
                x, int(x.stride(0)), self.buf, self.stride, self.capacity, self.ctl, int(x.shape[0]), slot.offset, slot.width)
            return
        v, st = _u16_rows(x)
        if v.shape[1] != slot.width:
            raise CaptureError(f"tap width {v.shape[1]} != slot {slot.kind}@{slot.layer} width {slot.width}")
        n = int(v.shape[0])
        if self.ext is not None:
            self.ext.launch_tap_single_port(v, st, self.buf, self.stride, self.capacity, self.ctl, n, slot.offset, slot.width)
            return
        self._launch("tap_single_port", n, self._gy(slot.width),
                     (v, st, self.buf, self.stride, self.capacity, self.ctl, n, slot.offset, slot.width))

    def tap_residual_and_norm(self, res: Any, res_slot: PortSlot, nrm: Optional[Any], nrm_slot: Optional[PortSlot]) -> None:
        if self.ext is not None and res.dim() == 2 and res.stride(1) == 1:
            n = int(res.shape[0])
            if nrm is None or nrm_slot is None:
                self.ext.launch_tap_residual_and_norm(
                    res, int(res.stride(0)), res_slot.offset, res_slot.width, res, int(res.stride(0)), 0, 0,
                    self.buf, self.stride, self.capacity, self.ctl, n)
                return
            if nrm.dim() == 2 and nrm.stride(1) == 1:
                self.ext.launch_tap_residual_and_norm(
                    res, int(res.stride(0)), res_slot.offset, res_slot.width,
                    nrm, int(nrm.stride(0)), nrm_slot.offset, nrm_slot.width,
                    self.buf, self.stride, self.capacity, self.ctl, n)
                return
        rv, rst = _u16_rows(res)
        if nrm is None or nrm_slot is None:
            nv, nst, noff, nw = rv, rst, 0, 0
        else:
            nv, nst = _u16_rows(nrm)
            noff, nw = nrm_slot.offset, nrm_slot.width
        n = int(rv.shape[0])
        if self.ext is not None:
            self.ext.launch_tap_residual_and_norm(
                rv, rst, res_slot.offset, res_slot.width, nv, nst, noff, nw,
                self.buf, self.stride, self.capacity, self.ctl, n)
            return
        self._launch("tap_residual_and_norm", n, self._gy(max(res_slot.width, nw)),
                     (rv, rst, res_slot.offset, res_slot.width, nv, nst, noff, nw,
                      self.buf, self.stride, self.capacity, self.ctl, n))

    def tap_attn_no_kv(self, q: Any, o: Any, lse: Any, q_slot: PortSlot, o_slot: PortSlot, lse_slot: PortSlot) -> None:
        """``lse`` is the attention kernel's fp32 ``[nq, tokens]`` (head-major, unit token stride)."""
        if self.ext is not None and q.is_contiguous() and o.is_contiguous() and lse.stride(1) == 1:
            n = int(q.shape[0])
            self.ext.launch_tap_attn_epilogue_no_kv(
                q, q_slot.width, q_slot.offset, q_slot.width, o, o_slot.width, o_slot.offset, o_slot.width,
                lse, int(lse.stride(0)), lse_slot.offset, int(lse.shape[0]),
                self.buf, self.stride, self.capacity, self.ctl, n)
            return
        import torch

        qv, qst = _u16_rows(q)
        ov, ost = _u16_rows(o)
        lv = lse.view(torch.int32) if lse.dtype == torch.float32 else lse
        if lv.stride(1) != 1:
            lv = lv.contiguous()
        n = int(qv.shape[0])
        if self.ext is not None:
            self.ext.launch_tap_attn_epilogue_no_kv(
                qv, qst, q_slot.offset, q_slot.width, ov, ost, o_slot.offset, o_slot.width,
                lv, int(lv.stride(0)), lse_slot.offset, int(lv.shape[0]),
                self.buf, self.stride, self.capacity, self.ctl, n)
            return
        self._launch("tap_attn_epilogue_no_kv", n, self._gy(max(q_slot.width, o_slot.width)),
                     (qv, qst, q_slot.offset, q_slot.width, ov, ost, o_slot.offset, o_slot.width,
                      lv, int(lv.stride(0)), lse_slot.offset, int(lv.shape[0]),
                      self.buf, self.stride, self.capacity, self.ctl, n))

    def tap_expert_ids(self, ids: Any, slot: PortSlot) -> None:
        import torch

        if ids.ndim != 2:
            raise ValueError(f"expert_ids must be 2D [rows, k], got shape {tuple(ids.shape)}")
        if int(ids.shape[1]) != slot.width:
            raise ValueError(f"expert_ids width mismatch: got {int(ids.shape[1])}, expected {slot.width}")
        if not ids.is_cuda or ids.device != self.buf.device:
            raise ValueError("expert_ids must be a CUDA tensor on the ring device")
        iv = ids if ids.dtype == torch.int32 else ids.to(torch.int32)
        n = int(iv.shape[0])
        if n <= 0:
            return
        if int(iv.stride(1)) != 1 or int(iv.stride(0)) < slot.width:
            # Always allocate a fresh row-major buffer (PyTorch treats size-1 dim0 with stride(0)==0 as contiguous).
            buf = torch.empty(iv.shape, dtype=torch.int32, device=iv.device)
            buf.copy_(iv)
            iv = buf
        if self.ext is not None:
            self.ext.launch_tap_expert_ids_i32_to_u16(
                iv, int(iv.stride(0)), slot.offset, slot.width, self.buf, self.stride, self.capacity, self.ctl, n)
            return
        self._launch("tap_expert_ids_i32_to_u16", n, 1,
                     (iv, int(iv.stride(0)), slot.offset, slot.width, self.buf, self.stride, self.capacity, self.ctl, n))

    def advance(self) -> None:
        if self.ext is not None:
            self.ext.launch_advance_cursor_u64(self.ctl)
            return
        self.k["advance_cursor_u64"]((1, 1, 1), (32, 1, 1), (self.ctl,))
