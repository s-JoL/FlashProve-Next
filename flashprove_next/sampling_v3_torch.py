# SPDX-License-Identifier: Apache-2.0
"""Batched torch engine for sampling rule v3 (``flashprove_next.sampling_v3``): bit-identical tokens, no host round trip.

One call samples every logits row of a batch on the logits' device:
  * PRF: one Triton kernel evaluates ceil(B / 16) + 1 Poseidon2 permutations per row (Goldilocks arithmetic in
    uint64 with ``umulhi``) from the row's per-request prefix state, step index t and role tag d_t;
  * weights: fp32 max, binary64 (m - z) * c, floor, committed exp-table gather (identical IEEE operations to the
    NumPy reference, so the integer weights are identical);
  * top-p without sorting: per-row histogram of the integer weight mass by exp-table index j (scatter_add, exact
    int64), prefix sums over j, tie-end gather, first index reaching target(S) -> theta*;
  * buckets: int64 sums, integer floor-log2 + mantissa table (``ln_fx``), committed Gumbel-table gather, packed
    (key, -b) argmax (deterministic smallest-index tie break);
  * inverse CDF: int64 prefix sums inside the chosen bucket, tau = floor(u48 W / 2^48) by 24/31-bit limbs.
Rows with an empty allowed set or a non-finite allowed logit are flagged ``bad`` (failure branch; their token is
meaningless).  On a CPU device the PRF falls back to the NumPy reference (tests without a GPU).
"""
import numpy as np
import torch

from . import sampling_v3 as S3
from . import poseidon2 as PS

M48 = S3.M48
M24 = S3.M24
I64_MIN = -(1 << 63)

# ------------------------------------------------------------------------------------------------ Triton Poseidon2
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _addm(a, b, P_, E_):
        s = a + b
        s = tl.where(s < a, s + E_, s)
        return tl.where(s >= P_, s - P_, s)

    @triton.jit
    def _mulm(a, b, P_, E_):
        lo = a * b
        hi = tl.umulhi(a, b)
        hh = hi >> 32
        hl = hi & E_
        t0 = lo - hh
        t0 = tl.where(lo < hh, t0 - E_, t0)
        t1 = hl * E_
        r = t0 + t1
        r = tl.where(r < t1, r + E_, r)
        return tl.where(r >= P_, r - P_, r)

    @triton.jit
    def _sbox(x, P_, E_):
        x2 = _mulm(x, x, P_, E_)
        x3 = _mulm(x2, x, P_, E_)
        x6 = _mulm(x3, x3, P_, E_)
        return _mulm(x6, x, P_, E_)

    @triton.jit
    def _m4(a0, a1, a2, a3, P_, E_):
        t01 = _addm(a0, a1, P_, E_)
        t23 = _addm(a2, a3, P_, E_)
        t0123 = _addm(t01, t23, P_, E_)
        t01123 = _addm(t0123, a1, P_, E_)
        t01233 = _addm(t0123, a3, P_, E_)
        y3 = _addm(t01233, _addm(a0, a0, P_, E_), P_, E_)
        y1 = _addm(t01123, _addm(a2, a2, P_, E_), P_, E_)
        y0 = _addm(t01123, t01, P_, E_)
        y2 = _addm(t01233, t23, P_, E_)
        return y0, y1, y2, y3

    @triton.jit
    def _me(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, P_, E_):
        y0, y1, y2, y3 = _m4(s0, s1, s2, s3, P_, E_)
        y4, y5, y6, y7 = _m4(s4, s5, s6, s7, P_, E_)
        y8, y9, y10, y11 = _m4(s8, s9, s10, s11, P_, E_)
        z0 = _addm(_addm(y0, y4, P_, E_), y8, P_, E_)
        z1 = _addm(_addm(y1, y5, P_, E_), y9, P_, E_)
        z2 = _addm(_addm(y2, y6, P_, E_), y10, P_, E_)
        z3 = _addm(_addm(y3, y7, P_, E_), y11, P_, E_)
        return (_addm(y0, z0, P_, E_), _addm(y1, z1, P_, E_), _addm(y2, z2, P_, E_), _addm(y3, z3, P_, E_),
                _addm(y4, z0, P_, E_), _addm(y5, z1, P_, E_), _addm(y6, z2, P_, E_), _addm(y7, z3, P_, E_),
                _addm(y8, z0, P_, E_), _addm(y9, z1, P_, E_), _addm(y10, z2, P_, E_), _addm(y11, z3, P_, E_))

    @triton.jit
    def _rc(rc_ptr, k, i: tl.constexpr):
        return tl.load(rc_ptr + (k * 12 + i)).to(tl.uint64, bitcast=True)

    @triton.jit
    def _full(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, rc_ptr, k, P_, E_):
        s0 = _sbox(_addm(s0, _rc(rc_ptr, k, 0), P_, E_), P_, E_)
        s1 = _sbox(_addm(s1, _rc(rc_ptr, k, 1), P_, E_), P_, E_)
        s2 = _sbox(_addm(s2, _rc(rc_ptr, k, 2), P_, E_), P_, E_)
        s3 = _sbox(_addm(s3, _rc(rc_ptr, k, 3), P_, E_), P_, E_)
        s4 = _sbox(_addm(s4, _rc(rc_ptr, k, 4), P_, E_), P_, E_)
        s5 = _sbox(_addm(s5, _rc(rc_ptr, k, 5), P_, E_), P_, E_)
        s6 = _sbox(_addm(s6, _rc(rc_ptr, k, 6), P_, E_), P_, E_)
        s7 = _sbox(_addm(s7, _rc(rc_ptr, k, 7), P_, E_), P_, E_)
        s8 = _sbox(_addm(s8, _rc(rc_ptr, k, 8), P_, E_), P_, E_)
        s9 = _sbox(_addm(s9, _rc(rc_ptr, k, 9), P_, E_), P_, E_)
        s10 = _sbox(_addm(s10, _rc(rc_ptr, k, 10), P_, E_), P_, E_)
        s11 = _sbox(_addm(s11, _rc(rc_ptr, k, 11), P_, E_), P_, E_)
        return _me(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, P_, E_)

    @triton.jit
    def _partial(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, rc_ptr, dg_ptr, k, P_, E_):
        s0 = _sbox(_addm(s0, _rc(rc_ptr, k, 0), P_, E_), P_, E_)
        a = _addm(_addm(_addm(s0, s1, P_, E_), _addm(s2, s3, P_, E_), P_, E_),
                  _addm(_addm(s4, s5, P_, E_), _addm(s6, s7, P_, E_), P_, E_), P_, E_)
        tot = _addm(a, _addm(_addm(s8, s9, P_, E_), _addm(s10, s11, P_, E_), P_, E_), P_, E_)
        d0 = tl.load(dg_ptr + 0).to(tl.uint64, bitcast=True)
        d1 = tl.load(dg_ptr + 1).to(tl.uint64, bitcast=True)
        d2 = tl.load(dg_ptr + 2).to(tl.uint64, bitcast=True)
        d3 = tl.load(dg_ptr + 3).to(tl.uint64, bitcast=True)
        d4 = tl.load(dg_ptr + 4).to(tl.uint64, bitcast=True)
        d5 = tl.load(dg_ptr + 5).to(tl.uint64, bitcast=True)
        d6 = tl.load(dg_ptr + 6).to(tl.uint64, bitcast=True)
        d7 = tl.load(dg_ptr + 7).to(tl.uint64, bitcast=True)
        d8 = tl.load(dg_ptr + 8).to(tl.uint64, bitcast=True)
        d9 = tl.load(dg_ptr + 9).to(tl.uint64, bitcast=True)
        d10 = tl.load(dg_ptr + 10).to(tl.uint64, bitcast=True)
        d11 = tl.load(dg_ptr + 11).to(tl.uint64, bitcast=True)
        return (_addm(tot, _mulm(d0, s0, P_, E_), P_, E_), _addm(tot, _mulm(d1, s1, P_, E_), P_, E_),
                _addm(tot, _mulm(d2, s2, P_, E_), P_, E_), _addm(tot, _mulm(d3, s3, P_, E_), P_, E_),
                _addm(tot, _mulm(d4, s4, P_, E_), P_, E_), _addm(tot, _mulm(d5, s5, P_, E_), P_, E_),
                _addm(tot, _mulm(d6, s6, P_, E_), P_, E_), _addm(tot, _mulm(d7, s7, P_, E_), P_, E_),
                _addm(tot, _mulm(d8, s8, P_, E_), P_, E_), _addm(tot, _mulm(d9, s9, P_, E_), P_, E_),
                _addm(tot, _mulm(d10, s10, P_, E_), P_, E_), _addm(tot, _mulm(d11, s11, P_, E_), P_, E_))

    @triton.jit
    def _prf_kernel(pre_ptr, slot_ptr, t_ptr, dt_ptr, rc_ptr, dg_ptr, out_ptr, n_rows, NC, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        msk = offs < n_rows * NC
        row = offs // NC
        c = offs - row * NC
        slot = tl.load(slot_ptr + row, mask=msk, other=0)
        E_ = tl.full([BLOCK], 0xFFFFFFFF, tl.uint64)
        P_ = (E_ << 32) + 1
        base = pre_ptr + slot * 12
        s0 = tl.load(base + 0, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s1 = tl.load(base + 1, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s2 = tl.load(base + 2, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s3 = tl.load(base + 3, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s4 = tl.load(base + 4, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s5 = tl.load(base + 5, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s6 = tl.load(base + 6, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s7 = tl.load(base + 7, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s8 = tl.load(base + 8, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s9 = tl.load(base + 9, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s10 = tl.load(base + 10, mask=msk, other=0).to(tl.uint64, bitcast=True)
        s11 = tl.load(base + 11, mask=msk, other=0).to(tl.uint64, bitcast=True)
        t = tl.load(t_ptr + row, mask=msk, other=0).to(tl.uint64, bitcast=True)
        dt = tl.load(dt_ptr + slot, mask=msk, other=0).to(tl.uint64, bitcast=True)
        is_inv = c == NC - 1
        lab = tl.where(is_inv, 2, 1).to(tl.uint64)
        jj = tl.where(is_inv, 0, c).to(tl.uint64)
        s0 = _addm(s0, dt, P_, E_)
        s1 = _addm(s1, t, P_, E_)
        s2 = _addm(s2, lab, P_, E_)
        s3 = _addm(s3, jj, P_, E_)
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 = _me(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, P_, E_)
        # runtime loops (the round bodies are compiled once each; full unrolling made the JIT take minutes)
        for k in range(0, 4):
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 = _full(
                s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, rc_ptr, k, P_, E_)
        for k in range(4, 26):
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 = _partial(
                s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, rc_ptr, dg_ptr, k, P_, E_)
        for k in range(26, 30):
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 = _full(
                s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, rc_ptr, k, P_, E_)
        ob = out_ptr + offs * 8
        tl.store(ob + 0, s0.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 1, s1.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 2, s2.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 3, s3.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 4, s4.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 5, s5.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 6, s6.to(tl.int64, bitcast=True), mask=msk)
        tl.store(ob + 7, s7.to(tl.int64, bitcast=True), mask=msk)

    @triton.jit
    def _wj(z, m, c, J):
        """Rule step 1 for a tile: fp32 logits z, fp32 row max m, fp64 c -> exp-table index j (int64, J = none)."""
        a = z != float("-inf")
        x = (m.to(tl.float64) - z.to(tl.float64)) * c
        Jf = J.to(tl.float64)
        x = tl.where(a & (x == x), x, Jf)
        x = tl.minimum(x, Jf)
        j = tl.floor(x).to(tl.int64)
        return tl.minimum(tl.maximum(j, 0), J)

    @triton.jit
    def _weights_kernel(z_ptr, m_ptr, c_ptr, E_ptr, hist_ptr, S_ptr, V, J, stride_z, HW,
                        TOPP: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        msk = offs < V
        z = tl.load(z_ptr + row * stride_z + offs, mask=msk, other=float("-inf"))
        j = _wj(z, tl.load(m_ptr + row), tl.load(c_ptr), J)
        w = tl.load(E_ptr + j, mask=msk, other=0)
        tl.atomic_add(S_ptr + row, tl.sum(w, 0))
        if TOPP:
            tl.atomic_add(hist_ptr + row * HW + j, w, mask=msk & (w > 0))

    @triton.jit
    def _bucket_kernel(z_ptr, m_ptr, c_ptr, E_ptr, th_ptr, idx_ptr, G_ptr, LNT_ptr, W_ptr, key_ptr,
                       V, B, J, LN2, stride_z, BETA: tl.constexpr, BB: tl.constexpr, TIE: tl.constexpr):
        row = tl.program_id(0)
        b = tl.program_id(1) * BB + tl.arange(0, BB)
        bm = b < B
        v = b[:, None] * BETA + tl.arange(0, BETA)[None, :]
        msk = v < V
        z = tl.load(z_ptr + row * stride_z + v, mask=msk, other=float("-inf"))
        j = _wj(z, tl.load(m_ptr + row), tl.load(c_ptr), J)
        w = tl.load(E_ptr + j, mask=msk, other=0)
        w = tl.where(w >= tl.load(th_ptr + row), w, 0)
        W = tl.sum(w, 1)
        tl.store(W_ptr + row * B + b, W, mask=bm)
        # LN(W): floor(log2), 16-bit mantissa, table
        y = tl.maximum(W, 1)
        Wc = y
        e = tl.zeros_like(y)
        for s in tl.static_range(6):
            sh = 32 >> s
            t = y >> sh
            mk = t > 0
            e = tl.where(mk, e + sh, e)
            y = tl.where(mk, t, y)
        mant = tl.where(e >= 15, Wc >> tl.maximum(e - 15, 0), Wc << tl.maximum(15 - e, 0))
        ln = e * LN2 + tl.load(LNT_ptr + (mant - 32768), mask=bm, other=0)
        g = tl.load(G_ptr + tl.load(idx_ptr + row * B + b, mask=bm, other=0), mask=bm, other=0).to(tl.int64)
        key = ((ln + g) << TIE) + ((1 << TIE) - 1 - b)
        tl.store(key_ptr + row * B + b, tl.where(W > 0, key, -(1 << 62)), mask=bm)


def _i64(x):
    x = int(x)
    return x - (1 << 64) if x >= (1 << 63) else x


def _u64_as_i64(arr):
    return np.asarray(arr, dtype=np.uint64).view(np.int64)


assert PS.RF == 8 and PS.RP == 22 and PS.T == 12, "the Triton kernel hard-codes Poseidon2 t=12, R_F=8, R_P=22"


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    """Batched rule-v3 sampler for one committed rule and vocabulary size, with per-slot PRF prefix states."""

    def __init__(self, V, rule, device, max_slots=1, block=128):
        self.V, self.rule, self.dev = int(V), rule, torch.device(device)
        self.B = rule.n_buckets(V)
        self.NC = rule.n_bkt_calls(V) + 1
        assert self.B < (1 << S3.TIE_SHIFT)
        self.block = block
        d = self.dev
        self.J = S3.exp_J()
        self.E = torch.from_numpy(S3.exp_table().astype(np.int64)).to(d)
        self.TE = torch.from_numpy(S3.tie_end().astype(np.int64)).to(d)
        self.G = torch.from_numpy(S3.gumbel_table()).to(d)
        LNT, LN2 = S3.ln_table()
        self.LNT = torch.from_numpy(LNT.astype(np.int64)).to(d)
        self.LN2 = int(LN2)
        self.rc = torch.from_numpy(_u64_as_i64(PS._RC_FLAT)).to(d)
        self.dg = torch.from_numpy(_u64_as_i64(PS._DIAG)).to(d)
        self.tiebreak = (1 << S3.TIE_SHIFT) - 1 - torch.arange(self.B, device=d, dtype=torch.int64)
        self.bits32 = torch.arange(32, device=d, dtype=torch.int32)
        self.max_slots = max_slots
        self.prefix = torch.zeros(max_slots, 12, dtype=torch.int64, device=d)
        self.dt = torch.zeros(max_slots, dtype=torch.int64, device=d)
        self._prefix_host = [[0] * 12 for _ in range(max_slots)]
        self._dt_host = [0] * max_slots
        self.use_triton = HAVE_TRITON and self.dev.type == "cuda"

    # -- per-request state
    def set_slot(self, slot, prefix, d_t=0, upload=True):
        self._prefix_host[slot] = [int(x) for x in prefix]
        self._dt_host[slot] = int(d_t)
        if upload:
            self.prefix[slot] = torch.tensor([_i64(x) for x in prefix], dtype=torch.int64).to(self.dev)
            self.dt[slot] = int(d_t)

    def upload_all(self, pinned_prefix, pinned_dt):
        """Bulk H2D of staged host state (``pinned_*`` are CPU int64 tensors shaped like ``prefix`` / ``dt``)."""
        self.prefix.copy_(pinned_prefix, non_blocking=True)
        self.dt.copy_(pinned_dt, non_blocking=True)

    # -- PRF
    def prf(self, slots, t):
        """slots, t: int64 [R] on the device -> PRF outputs int64 (uint64 bits) [R, NC, 8]."""
        R = int(slots.shape[0])
        out = torch.empty(R, self.NC, 8, dtype=torch.int64, device=self.dev)
        if R == 0:
            return out
        if self.use_triton:
            n = R * self.NC
            _prf_kernel[(triton.cdiv(n, self.block),)](self.prefix, slots, t, self.dt, self.rc, self.dg, out, R,
                                                        self.NC, BLOCK=self.block)
            return out
        sl = slots.cpu().numpy(); tt = t.cpu().numpy()
        res = np.stack([S3.prf_calls(self._prefix_host[int(s)], self._dt_host[int(s)], [int(x)], self.NC - 1)[0]
                        for s, x in zip(sl, tt)])
        return torch.from_numpy(_u64_as_i64(res)).to(self.dev)

    def noise(self, outs):
        e = outs[:, :-1, :]
        low = e & M48
        idx = torch.stack([low & M24, low >> 24], dim=-1).reshape(outs.shape[0], -1)[:, :self.B]
        u48 = outs[:, -1, 0] & M48
        return idx, u48

    # -- arithmetic helpers
    def ln_fx(self, W):
        y = W.clone()
        e = torch.zeros_like(W)
        for s in (32, 16, 8, 4, 2, 1):
            t = y >> s
            mk = t > 0
            e += s * mk
            y = torch.where(mk, t, y)
        sh = S3.LN_MBITS - 1
        m = torch.where(e >= sh, torch.bitwise_right_shift(W, (e - sh).clamp(min=0)),
                        torch.bitwise_left_shift(W, (sh - e).clamp(min=0)))
        return e * self.LN2 + self.LNT[m - (1 << sh)]

    @staticmethod
    def mulshift48(u48, W):
        """floor(u48 * W / 2^48) exactly for u48 < 2^48, 0 <= W < 2^62 (int64 limbs, no overflow)."""
        uh, ul = u48 >> 24, u48 & M24
        Wh, Wl = W >> 31, W & ((1 << 31) - 1)
        A = Wl * ul
        Bp = Wl * uh
        C = Wh * ul
        D = Wh * uh
        lo = (A & M48) + ((Bp & M24) << 24) + ((C & ((1 << 17) - 1)) << 31)
        return (A >> 48) + (Bp >> 24) + (C >> 17) + (D << 7) + (lo >> 48)

    def unpack_bitmask(self, bm):
        """xgrammar-style packed mask int32 [R, ceil(V/32)] (bit k of word i <-> token 32 i + k, 1 = allowed)."""
        bits = (bm.to(torch.int32).unsqueeze(-1) >> self.bits32) & 1
        return bits.reshape(bm.shape[0], -1)[:, :self.V].bool()

    # -- the rule
    def sample(self, logits, slots, t, allowed=None, bitmask=None, info=False):
        """Fused Triton path on CUDA (``sample_fused``), plain torch otherwise (``sample_torch``); same tokens."""
        if self.use_triton and not info:
            return self.sample_fused(logits, slots, t, allowed, bitmask)
        return self.sample_torch(logits, slots, t, allowed, bitmask, info)

    def _wj_torch(self, z, m):
        a = z != float("-inf")
        x = (m.double() - z.double()) * self.rule.c
        x = torch.where(a & (x == x), x, float(self.J))
        return torch.floor(torch.clamp(x, max=float(self.J))).long().clamp(0, self.J)

    def sample_fused(self, logits, slots, t, allowed=None, bitmask=None):
        """Two fused passes over the fp32 logits (weights + histogram; buckets + keys); weights are recomputed, never
        materialised.  theta* = E[first j with C[j] >= target] (equal to the tie-run start's value)."""
        R, V = logits.shape
        z = logits.float()
        if allowed is not None or bitmask is not None:
            a = torch.ones_like(z, dtype=torch.bool) if allowed is None else allowed
            if bitmask is not None:
                a = a & self.unpack_bitmask(bitmask)
            z = torch.where(a, z, float("-inf"))
        z = z.contiguous()
        m = z.amax(-1)
        bad = ~torch.isfinite(m)
        if not hasattr(self, "_c"):
            self._c = torch.tensor([self.rule.c], dtype=torch.float64, device=self.dev)
        topp = self.rule.p16 < (1 << S3.P_BITS)
        S = torch.zeros(R, dtype=torch.int64, device=self.dev)
        HB = 512                                            # two-level search over the histogram
        HW = triton.cdiv(self.J + 1, HB) * HB
        hist = torch.zeros(R, HW, dtype=torch.int64, device=self.dev) if topp else S
        BLOCK = 1024
        _weights_kernel[(R, triton.cdiv(V, BLOCK))](z, m, self._c, self.E, hist, S, V, self.J, z.stride(0),
                                                    HW, TOPP=topp, BLOCK=BLOCK)
        if topp:
            p16 = self.rule.p16
            target = (S >> 16) * p16 + (((S & 0xFFFF) * p16 + 0xFFFF) >> 16)
            Hb = hist.view(R, HW // HB, HB)              # bin J (weight 0) never receives mass
            cb = Hb.sum(-1).cumsum(-1)
            k = (cb < target[:, None]).sum(-1)
            before = torch.where(k > 0, cb.gather(1, (k - 1).clamp(min=0)[:, None]).squeeze(1), 0)
            fine = Hb.gather(1, k[:, None, None].expand(R, 1, HB)).squeeze(1).cumsum(-1) + before[:, None]
            j0 = k * HB + (fine < target[:, None]).sum(-1)
            theta = self.E[j0.clamp(max=self.J)]
        else:
            theta = torch.ones(R, dtype=torch.int64, device=self.dev)
        idx, u48 = self.noise(self.prf(slots, t))
        idx = idx.contiguous()
        W = torch.empty(R, self.B, dtype=torch.int64, device=self.dev)
        key = torch.empty_like(W)
        beta = self.rule.beta
        BB = max(1, 2048 // beta)
        _bucket_kernel[(R, triton.cdiv(self.B, BB))](z, m, self._c, self.E, theta, idx, self.G, self.LNT, W, key,
                                                     V, self.B, self.J, self.LN2, z.stride(0), BETA=beta, BB=BB,
                                                     TIE=S3.TIE_SHIFT)
        b = key.argmax(-1)
        cols = (b * beta)[:, None] + torch.arange(beta, device=self.dev)[None, :]
        zin = z.gather(1, cols.clamp(max=V - 1))
        w = torch.where(cols < V, self.E[self._wj_torch(zin, m[:, None])], 0)
        w = torch.where(w >= theta[:, None], w, 0)
        Sv = w.cumsum(-1)
        Wb = W.gather(1, b[:, None]).squeeze(1)
        tau = self.mulshift48(u48, Wb)
        return b * beta + (Sv <= tau[:, None]).sum(-1), bad

    def sample_torch(self, logits, slots, t, allowed=None, bitmask=None, info=False):
        """logits fp32 [R, V]; slots, t int64 [R] (device); optional allowed bool [R, V] or packed bitmask.
        Returns (tokens int64 [R], bad bool [R]) -- and a dict of intermediates with info=True."""
        R, V = logits.shape
        assert V == self.V
        z = logits.float()
        a = z != float("-inf")
        if allowed is not None:
            a &= allowed
        if bitmask is not None:
            a &= self.unpack_bitmask(bitmask)
        bad = ~a.any(-1) | (a & ~torch.isfinite(z)).any(-1)
        m = torch.where(a, z, float("-inf")).amax(-1, keepdim=True)
        x = (m.double() - z.double()) * self.rule.c
        x = torch.where(a, x, float(self.J))
        x = torch.nan_to_num(x, nan=float(self.J))
        j = torch.floor(torch.clamp(x, max=float(self.J))).long()
        w = self.E[j]
        if self.rule.p16 < (1 << S3.P_BITS):
            p16 = self.rule.p16
            S = w.sum(-1)
            target = (S >> 16) * p16 + (((S & 0xFFFF) * p16 + 0xFFFF) >> 16)
            hist = torch.zeros(R, self.J + 1, dtype=torch.int64, device=z.device)
            hist.scatter_add_(1, j, w)
            C = hist[:, :self.J].cumsum(-1)
            jmin = (C[:, self.TE] < target[:, None]).sum(-1)
            theta = self.E[jmin]
            wN = torch.where(w >= theta[:, None], w, 0)
        else:
            wN = w
        beta = self.rule.beta
        if self.B * beta > V:
            wN = torch.nn.functional.pad(wN, (0, self.B * beta - V))
        blocks = wN.view(R, self.B, beta)
        W = blocks.sum(-1)
        pos = W > 0
        outs = self.prf(slots, t)
        idx, u48 = self.noise(outs)
        key = self.ln_fx(W.clamp(min=1)) + self.G[idx].long()
        packed = torch.where(pos, (key << S3.TIE_SHIFT) + self.tiebreak, I64_MIN)
        b = packed.argmax(-1)
        sel = blocks.gather(1, b.view(R, 1, 1).expand(R, 1, beta)).squeeze(1)
        Sv = sel.cumsum(-1)
        Wb = W.gather(1, b[:, None]).squeeze(1)
        tau = self.mulshift48(u48, Wb)
        tok = b * beta + (Sv <= tau[:, None]).sum(-1)
        if info:
            return tok, bad, {"bucket": b, "W_b": Wb, "tau": tau, "u48": u48, "outs": outs,
                              "nucleus_size": (wN > 0).sum(-1)}
        return tok, bad

    # -- CUDA-graph replay (the eager path is ~120 small kernels: launch-bound at decode batch sizes)
    def force(self, logits, slots, t, active):
        """Rule-v3 logits processing for a batch: rows with active[slot] and a valid draw become one-hot (0 at the
        token, -inf elsewhere); other rows are returned unchanged.  Returns (new_logits, tokens, use)."""
        tok, bad = self.sample(logits, slots, t)
        use = active[slots] & ~bad
        forced = torch.full_like(logits, float("-inf")).scatter_(1, tok[:, None], 0.0)
        return torch.where(use[:, None], forced, logits), tok, use

    def force_graphed(self, logits, slots, t, active):
        """``force`` replayed from a CUDA graph captured once per row count (static buffers; per-slot state and
        ``active`` are read in place, so later ``set_slot`` / ``upload_all`` / active updates are seen)."""
        R = int(logits.shape[0])
        if not hasattr(self, "_graphs"):
            self._graphs, self._pool = {}, torch.cuda.graph_pool_handle()
        g = self._graphs.get(R)
        if g is None:
            sl = torch.zeros(R, dtype=torch.int64, device=self.dev)
            tt = torch.zeros(R, dtype=torch.int64, device=self.dev)
            lg = torch.zeros(R, self.V, dtype=torch.float32, device=self.dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):                      # warm-up (Triton JIT, allocator) outside the capture
                    self.force(lg, sl, tt, active)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                outs = self.force(lg, sl, tt, active)
            g = self._graphs[R] = (graph, lg, sl, tt, outs)
        graph, lg, sl, tt, outs = g
        lg.copy_(logits)
        sl.copy_(slots)
        tt.copy_(t)
        graph.replay()
        return outs
