// SPDX-License-Identifier: Apache-2.0
// Portable tap kernels for the FlashProve-Next staging ring.  Header-free CUDA C++ (NVRTC-compilable: no #include).
//
// Conventions shared by every kernel
//   * The ring is ONE row-major u16 allocation [capacity rows x ring_stride u16].  Row r of the current step lands in
//     physical row  (cursor + r) % capacity  (64-bit unsigned arithmetic; capacity need not be a power of two).
//   * `ctl` is a 4 x int64 device control block:  ctl[0] = cursor (monotonic row watermark at step start),
//     ctl[1] = n_valid (rows really produced this step), ctl[2] = violation flag (set if n_valid > n_max), ctl[3] reserved.
//     Reading n_valid on-device is what makes the kernels CUDA-graph safe: launch arguments are frozen at capture time
//     (n_max is the capture size) while the real row count changes per replay.  Rows >= min(n_valid, n_max) write nothing.
//   * Every scalar parameter is a 32-bit `int` (pointers are device pointers): launchers that marshal Python ints as
//     C ints (torch NVRTC) therefore never feed a 32-bit slot to a 64-bit parameter.  Products are widened to 64 bit
//     inside the kernels.  `ctl` itself is int64 and lives on the device.
//   * Every captured port is copied in full: the 16-byte vector path handles floor(w/8)*8 columns and a scalar tail
//     handles the remaining w%8, so no column is ever dropped; a row whose src/dst pointers are not 16-byte aligned
//     takes the scalar path (no unaligned uint4 access).  All row/offset arithmetic is 64-bit.
//   * Nothing is ever zero-filled: unwritten ring memory stays whatever it was and is unreachable through the host
//     watermarks.
//   * Grid: (n_max, gy, 1), block: (128, 1, 1).  Each row is handled by gridDim.y blocks cooperatively.
//   * K and V are deliberately NOT captured (they stay in vLLM's paged KV cache and are referenced by block hash).

typedef unsigned short u16;
typedef unsigned int u32;
typedef int i32;
typedef long long i64;
typedef unsigned long long u64;

__device__ __forceinline__ void copy_row_u16_aligned(const u16* __restrict__ src, u16* __restrict__ dst, int w, int tid, int stride) {
    const int w8 = w >> 3;
    if (w8 > 0 && ((((u64)(const void*)src) | ((u64)(void*)dst)) & 15ull) == 0ull) {
        const uint4* s4 = (const uint4*)src;
        uint4* d4 = (uint4*)dst;
        for (int i = tid; i < w8; i += stride) d4[i] = s4[i];
        for (int i = (w8 << 3) + tid; i < w; i += stride) dst[i] = src[i];   // scalar tail: w % 8 columns
    } else {
        for (int i = tid; i < w; i += stride) dst[i] = src[i];
    }
}

__device__ __forceinline__ int tap_rows(i64* ctl, int n_max) {
    const i64 v = ctl[1];
    if (v > (i64)n_max) {                       // would advance the cursor past rows we do not write: flag it
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) ctl[2] = 1;
        return n_max;
    }
    return v < 0 ? 0 : (int)v;
}

__device__ __forceinline__ u16* tap_dst_row(u16* ring, int ring_stride, int capacity, const i64* ctl, int row) {
    const i64 phys = (i64)(((u64)ctl[0] + (u64)row) % (u64)capacity);
    return ring + phys * (i64)ring_stride;
}

#define TAP_PREAMBLE(NMAX)                                                    \
    const int row = blockIdx.x;                                               \
    if (row >= tap_rows(ctl, NMAX)) return;                                   \
    u16* drow = tap_dst_row(ring, ring_stride, capacity, ctl, row);           \
    const int tid = blockIdx.y * blockDim.x + threadIdx.x;                    \
    const int stride = gridDim.y * blockDim.x;

// One port: residual stream, MLP intermediate `a`, final hidden, ... ------------------------------------------------
extern "C" __global__ void tap_single_port(
    const u16* __restrict__ src, int src_stride,
    u16* __restrict__ ring, int ring_stride, int capacity, i64* ctl, int n_max,
    int off, int w) {
    TAP_PREAMBLE(n_max)
    copy_row_u16_aligned(src + (i64)row * src_stride, drow + off, w, tid, stride);
}

// Fused-add RMSNorm site: capture the residual output (h_in / h_mid / h_L) and, unless w_nrm == 0, the normed output
// (x_attn / x_mlp / x_f) in ONE launch. ------------------------------------------------------------------------------
extern "C" __global__ void tap_residual_and_norm(
    const u16* __restrict__ res, int res_stride, int off_res, int w_res,
    const u16* __restrict__ nrm, int nrm_stride, int off_nrm, int w_nrm,
    u16* __restrict__ ring, int ring_stride, int capacity, i64* ctl, int n_max) {
    TAP_PREAMBLE(n_max)
    copy_row_u16_aligned(res + (i64)row * res_stride, drow + off_res, w_res, tid, stride);
    if (w_nrm > 0) copy_row_u16_aligned(nrm + (i64)row * nrm_stride, drow + off_nrm, w_nrm, tid, stride);
}

// Attention epilogue WITHOUT K/V: q, o and the softmax LSE.  `lse` comes from the attention kernel as [nq, tokens]
// fp32 (head-major); it is transposed here, in registers, straight into the row's u32 pairs -- no .t().contiguous()
// temporary and no extra launch.  Requires off_lse even (slots are 16-byte aligned). ---------------------------------
extern "C" __global__ void tap_attn_epilogue_no_kv(
    const u16* __restrict__ q, int q_stride, int off_q, int w_q,
    const u16* __restrict__ o, int o_stride, int off_o, int w_o,
    const u32* __restrict__ lse, int lse_head_stride, int off_lse, int nq,
    u16* __restrict__ ring, int ring_stride, int capacity, i64* ctl, int n_max) {
    TAP_PREAMBLE(n_max)
    copy_row_u16_aligned(q + (i64)row * q_stride, drow + off_q, w_q, tid, stride);
    copy_row_u16_aligned(o + (i64)row * o_stride, drow + off_o, w_o, tid, stride);
    u32* dl = (u32*)(drow + off_lse);
    for (int h = tid; h < nq; h += stride) dl[h] = lse[(i64)h * lse_head_stride + row];
}

// MoE router top-k ids (int32 from the router) -> u16 slot.  Fail closed via ctl[2] on any out-of-range id. ---------
extern "C" __global__ void tap_expert_ids_i32_to_u16(
    const i32* __restrict__ ids, int ids_stride, int off, int k,
    u16* __restrict__ ring, int ring_stride, int capacity, i64* ctl, int n_max) {
    TAP_PREAMBLE(n_max)
    for (int j = tid; j < k; j += stride) {
        const i32 v = ids[(i64)row * ids_stride + j];
        if (v < 0 || v > 65535) {
            ctl[2] = 1;
            drow[off + j] = 0;
        } else {
            drow[off + j] = (u16)v;
        }
    }
}

// End-of-step: cursor += n_valid, entirely on the device.  The default host path writes [cursor, n_valid] before every
// step (self-correcting); this kernel is for engines that run several forward pieces per host step. ------------------
extern "C" __global__ void advance_cursor_u64(i64* ctl) {
    if (blockIdx.x == 0 && threadIdx.x == 0) ctl[0] = (i64)((u64)ctl[0] + (u64)(ctl[1] < 0 ? 0 : ctl[1]));
}
