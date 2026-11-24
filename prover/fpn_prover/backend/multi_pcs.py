# SPDX-License-Identifier: Apache-2.0
"""Multi-group, multi-size batched rate-1/8 coset-4 folded ZK-Basefold (`multi_pcs`).

Proves simultaneously, in ONE fused coset-4 fold chain and ZERO Phase-2 commitments, that G >= 1 committed
groups F_0, ..., F_{G-1} of arbitrary even sizes n_0, ..., n_{G-1} (8 <= n_g <= n_max) satisfy linear claims

    <F_g, W_g> = y_g    for every g in 0 .. G-1,

where each W_g is any verifier-evaluable weight vector over E^{2^{n_g}}.

Key efficiency breakthrough — Multi-Size Layer Injection:
  * A small group F_g of size 2^{n_g} (e.g. a 2^12 interactive commit round or Mask group) is committed ONLY
    at its own size n_g (codeword length 8 * 2^{n_g}, Merkle depth n_g + 1), NOT padded to n_max = 2^24!
  * Its codeword lives on the EXACT same evaluation domain <omega_{t_g}> as fold layer t_g = (n_max - n_g)/2
    of the main n_max fold chain!
  * Step A (n_max-round degree-2 Libra-masked inner-product sumcheck):
      Group g acts on the high n_g variables (2*t_g .. n_max-1) multiplied by [x_lo == 0] on the low 2*t_g
      variables (so rounds 0..2*t_g-1 touch ZERO array entries of group g!).
      The Step-A Libra mask (3*n_max coeffs <= 84 Fp3) lives in a tiny dedicated n_lib = 8 group L (256 cells),
      discharged by a 0.4 ms, ~35 KB opening -> keeping Step B's weight PURE eq(r_ip)!
  * Step B (ONE batched rate-1/8 coset-4 Basefold chain with PURE weight W_H = eq(r_ip)):
      One full-length OS-CSPRNG uniform blinding group B (size 2^{n_max}) completely blinds H, every folded
      layer, and every Step-B sumcheck round.
      At the start of fold layer t (0 <= t < n_max/2), every group g with t_g == t is injected into the
      running state:
          h_cur    += theta^{g+1} * F_g
          cur_code += theta^{g+1} * C_g
          target_b += theta^{g+1} * eq(r_ip[:2*t], r_bf[:2*t]) * v_g.
      At query time, group g's coset leaf is opened at layer t_g (depth n_g + 1) and added before fold4!

Supports both pure-NumPy CPU and GPU (`CudaEngine`) with bit-identical transcripts and proofs.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from flashprove_next.field import ONE, P, ZERO, eq1, hadd, hf, hinv, hmul, hsub, lagrange_eval

from . import cuda_engine as CE
from . import folded_pcs as pcs
from .fpn2 import Fpn2Transcript, dec3, enc3, enc3s, leaf_hash, merkle_levels, merkle_path, node_hash

RATE = 8
LIBRA_DEG = 2


# ----------------------------------------------------------------------------- OS CSPRNG / AES-256-CTR blinding
def _csprng_bytes(nbytes: int) -> bytes:
    """Cryptographically secure random bytes: direct OS CSPRNG for small buffers, fresh-key AES-256-CTR
    (ephemeral 256-bit OS key per call, `A-PRG` computational ZK) for large buffers >= 32 KiB."""
    if nbytes < 32768:
        return secrets.token_bytes(nbytes)
    try:
        from flashprove_next.zk_flat import _aes_ctr_bytes
        key = secrets.token_bytes(32)
        arr = _aes_ctr_bytes(key, 0, nbytes)
        return arr.tobytes()
    except Exception:  # noqa: BLE001
        return secrets.token_bytes(nbytes)


def _os_rand_fp3(n_elems, rng=None):
    """Return (n_elems, 3) canonical uint64 Fp3 array. Uses OS CSPRNG (keyed AES-256-CTR for large
    arrays, immune to `FPN_TEST_RNG_SEED`) when rng is None."""
    if rng is not None:
        return np.asarray(rng.integers(0, P, size=(n_elems, 3), dtype=np.uint64), dtype=np.uint64)
    out = np.empty(n_elems * 3, dtype=np.uint64)
    filled = 0
    need = out.size
    while filled < need:
        batch = max(need - filled + 32, (need - filled) * 257 // 256 + 64)
        # Round up to multiple of 16 bytes for AES-CTR
        nbytes = ((batch * 8) + 15) & ~15
        raw = np.frombuffer(_csprng_bytes(nbytes), dtype="<u8")
        good = raw[raw < P]
        take = min(len(good), need - filled)
        out[filled:filled + take] = good[:take]
        filled += take
    return out.reshape(n_elems, 3)


def tail_len_for(queries: int) -> int:
    """Uniform OS-CSPRNG blinding tail per data group (>= 4*Q + 16, multiple of 16)."""
    return ((4 * queries + 16) + 15) & ~15


def group_data_capacity(n_g: int, queries: int) -> int:
    cap = (1 << n_g) - tail_len_for(queries)
    if cap <= 0:
        raise ValueError(f"n_g={n_g} too small for queries={queries} (tail={tail_len_for(queries)})")
    return cap


def libra_group_n(n_max: int, queries: int) -> int:
    """Smallest even n_lib >= 8 that holds 3*n_max Libra coeffs + tail_len_for(queries)."""
    need = 3 * n_max + tail_len_for(queries)
    n = 8
    while (1 << n) < need:
        n += 2
    return n


# ----------------------------------------------------------------------------- layout & byte accounting
@dataclass(frozen=True)
class MultiPcsLayout:
    n: int                                # max log2 group length (even, 8 <= n <= 28)
    n_groups: int                         # G >= 1 data groups
    queries: int                          # Q Basefold queries
    group_ns: tuple[int, ...] = ()        # optional per-group even sizes (all <= n); empty => all equal n

    def __post_init__(self):
        if type(self.n) is not int or self.n < 8 or self.n % 2 or self.n > 28:
            raise ValueError("MultiPcsLayout requires even n in [8, 28]")
        if type(self.n_groups) is not int or not 1 <= self.n_groups <= 128:
            raise ValueError("MultiPcsLayout requires 1 <= n_groups <= 128")
        if type(self.queries) is not int or not 1 <= self.queries <= 4096:
            raise ValueError("MultiPcsLayout requires 1 <= queries <= 4096")
        gns = tuple(self.group_ns) if self.group_ns else (self.n,) * self.n_groups
        if len(gns) != self.n_groups:
            raise ValueError("group_ns length must equal n_groups")
        for ng in gns:
            if type(ng) is not int or ng < 8 or ng % 2 or ng > self.n:
                raise ValueError(f"invalid group size n_g={ng} (must be even in [8, {self.n}])")
            if tail_len_for(self.queries) >= (1 << ng):
                raise ValueError(f"n_g={ng} too small for queries={self.queries}")
        object.__setattr__(self, "group_ns", gns)

    @property
    def group_len(self):
        return 1 << self.n

    @property
    def code_len(self):
        return RATE << self.n

    @property
    def folds(self):
        return self.n // 2

    @property
    def tail_len(self):
        return tail_len_for(self.queries)

    @property
    def data_capacity(self):
        return self.group_len - self.tail_len

    @property
    def n_lib(self):
        return libra_group_n(self.n, self.queries)

    def inject_layer(self, g: int) -> int:
        return (self.n - self.group_ns[g]) // 2

    def depth(self, t: int) -> int:
        return self.n + 1 - 2 * t

    def group_depth(self, g: int) -> int:
        return self.group_ns[g] + 1

    @property
    def step_a_bytes(self):
        # G_sum (24) + n rounds * 3 Fp3 (72) + (G + 2) finals (v_0..v_{G-1}, v_B, e_lib)
        return 24 + self.n * 72 + (self.n_groups + 2) * 24

    @property
    def step_b_header_bytes(self):
        # n rounds * 3 Fp3 (72) + (folds - 1) fold roots (32) + FINAL (24)
        return self.n * 72 + (self.folds - 1) * 32 + 24

    @property
    def main_query_bytes(self):
        # B at layer 0 + each group g at its own inject_layer (depth = n_g + 1) + combined layers 1..folds-1
        b0 = 96 + self.depth(0) * 32
        gs = sum(96 + self.group_depth(g) * 32 for g in range(self.n_groups))
        rest = sum(96 + self.depth(t) * 32 for t in range(1, self.folds))
        return b0 + gs + rest

    @property
    def libra_open_bytes(self):
        # Tiny standalone Basefold opening of group L (n_lib rounds, n_lib/2 folds, Q queries)
        nl = self.n_lib
        fl = nl // 2
        hdr = nl * 72 + (fl - 1) * 32 + 24
        q_b = sum(96 + (nl + 1 - 2 * t) * 32 for t in range(fl))
        return hdr + self.queries * q_b

    @property
    def proof_bytes(self):
        # root_B (32) + root_L (32) + step_a + libra_open + step_b_header + Q * main_query_bytes
        return 64 + self.step_a_bytes + self.libra_open_bytes + self.step_b_header_bytes + self.queries * self.main_query_bytes


# ----------------------------------------------------------------------------- Libra helpers
def libra_sum_from_coeffs(libra_coeffs, n):
    scale = pow(2, n - 1, P)
    acc = ZERO
    for i in range(n):
        c0 = tuple(int(v) for v in libra_coeffs[3 * i])
        c1 = tuple(int(v) for v in libra_coeffs[3 * i + 1])
        c2 = tuple(int(v) for v in libra_coeffs[3 * i + 2])
        s = hadd(hadd(c0, c0), hadd(c1, c2))
        acc = hadd(acc, s)
    return hmul(hf(scale), acc)


def libra_round_vals(libra_coeffs, n, i, prefix_sum):
    rem = n - 1 - i
    cnt = pow(2, rem, P)
    half_cnt = pow(2, rem - 1, P) if rem > 0 else 0
    tail_sum = ZERO
    if rem > 0:
        acc = ZERO
        for j in range(i + 1, n):
            c0 = tuple(int(v) for v in libra_coeffs[3 * j])
            c1 = tuple(int(v) for v in libra_coeffs[3 * j + 1])
            c2 = tuple(int(v) for v in libra_coeffs[3 * j + 2])
            acc = hadd(acc, hadd(hadd(c0, c0), hadd(c1, c2)))
        tail_sum = hmul(hf(half_cnt), acc)
    c0 = tuple(int(v) for v in libra_coeffs[3 * i])
    c1 = tuple(int(v) for v in libra_coeffs[3 * i + 1])
    c2 = tuple(int(v) for v in libra_coeffs[3 * i + 2])
    out = []
    for t in (0, 1, 2):
        tf = hf(t)
        gi_t = hadd(c0, hmul(tf, hadd(c1, hmul(tf, c2))))
        val = hadd(hmul(hf(cnt), hadd(prefix_sum, gi_t)), tail_sum)
        out.append(val)
    return out


def libra_eval_var(libra_coeffs, i, r_i):
    c0 = tuple(int(v) for v in libra_coeffs[3 * i])
    c1 = tuple(int(v) for v in libra_coeffs[3 * i + 1])
    c2 = tuple(int(v) for v in libra_coeffs[3 * i + 2])
    return hadd(c0, hmul(r_i, hadd(c1, hmul(r_i, c2))))


def libra_weight_entries(r_ip):
    out = []
    for r in r_ip:
        out.append(ONE)
        out.append(r)
        out.append(hmul(r, r))
    return out


def libra_weight_mle_at0(r_ip, r_lib, n_lib):
    """Evaluate W_lib(r_ip)~(r_lib) where the 3*len(r_ip) weights sit at indices 0 .. 3*len(r_ip)-1."""
    weights = libra_weight_entries(r_ip)
    acc = ZERO
    for k, w in enumerate(weights):
        eq_val = ONE
        for b in range(n_lib):
            rb = r_lib[b]
            eq_val = hmul(eq_val, rb if (k >> b) & 1 else hsub(ONE, rb))
        acc = hadd(acc, hmul(w, eq_val))
    return acc


def eq_eval_pt(a, b):
    acc = ONE
    for x, y in zip(a, b):
        acc = hmul(acc, eq1(x, y))
    return acc


# ----------------------------------------------------------------------------- Fp3 array arithmetic (NumPy & CuPy)
def _fp3_mul_scalar_host(arr_np, s):
    from flashprove_next.field import CPU
    a0, a1, a2 = arr_np[:, 0], arr_np[:, 1], arr_np[:, 2]
    s0, s1, s2 = s

    def smul(v, c):
        if c == 0:
            return np.zeros_like(v)
        if c == 1:
            return v.copy()
        return CPU.mul(v, np.full(len(v), c, dtype=np.uint64))

    t0 = CPU.add(smul(a1, s2), smul(a2, s1))
    c0 = CPU.add(smul(a0, s0), CPU.add(t0, t0))
    t1 = smul(a2, s2)
    c1 = CPU.add(CPU.add(smul(a0, s1), smul(a1, s0)), CPU.add(t1, t1))
    c2 = CPU.add(CPU.add(smul(a0, s2), smul(a1, s1)), smul(a2, s0))
    out = np.empty_like(arr_np)
    out[:, 0], out[:, 1], out[:, 2] = c0, c1, c2
    return out


def _fp3_mul_scalar_dev(engine, arr_dev, s):
    if engine is None or not hasattr(engine, "cp"):
        return _fp3_mul_scalar_host(np.asarray(arr_dev, dtype=np.uint64), s)
    from flashprove_next import field_gpu as FG
    a = engine.dev(arr_dev)
    M = len(a)
    s0 = engine.cp.full(M, np.uint64(s[0]), dtype=engine.cp.uint64)
    s1 = engine.cp.full(M, np.uint64(s[1]), dtype=engine.cp.uint64)
    s2 = engine.cp.full(M, np.uint64(s[2]), dtype=engine.cp.uint64)
    c0, c1, c2 = FG._emul(a[:, 0], a[:, 1], a[:, 2], s0, s1, s2)
    out = engine.cp.empty_like(a)
    out[:, 0], out[:, 1], out[:, 2] = c0, c1, c2
    return out


def _fp3_add_dev(engine, a_dev, b_dev):
    if engine is None or not hasattr(engine, "cp"):
        from flashprove_next.field import CPU
        out = np.empty_like(a_dev)
        for k in range(3):
            out[:, k] = CPU.add(a_dev[:, k], b_dev[:, k])
        return out
    from flashprove_next import field_gpu as FG
    c0, c1, c2 = FG._eadd(a_dev[:, 0], a_dev[:, 1], a_dev[:, 2], b_dev[:, 0], b_dev[:, 1], b_dev[:, 2])
    out = engine.cp.empty_like(a_dev)
    out[:, 0], out[:, 1], out[:, 2] = c0, c1, c2
    return out


_AXPY_SCALAR_KERNEL = None


def _get_axpy_scalar_kernel(cp):
    global _AXPY_SCALAR_KERNEL
    if _AXPY_SCALAR_KERNEL is None:
        from flashprove_next import field_gpu as FG
        _AXPY_SCALAR_KERNEL = cp.RawKernel(
            FG.PRE + r"""
extern "C" __global__ void fpn_axpy_scalar(
    u64* __restrict__ dst, const u64* __restrict__ src,
    u64 s0, u64 s1, u64 s2, unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long off = i * 3ULL;
    E3 a = { src[off], src[off + 1], src[off + 2] };
    E3 s = { s0, s1, s2 };
    E3 p = e_mul(a, s);
    dst[off]     = addm(dst[off],     p.a);
    dst[off + 1] = addm(dst[off + 1], p.b);
    dst[off + 2] = addm(dst[off + 2], p.c);
}
""",
            "fpn_axpy_scalar",
        )
    return _AXPY_SCALAR_KERNEL


def _fp3_axpy_dev(engine, dst_dev, src_dev, s):
    if s == ZERO:
        return dst_dev
    if engine is not None and hasattr(engine, "cp"):
        cp = engine.cp
        n_rows = len(dst_dev)
        k = _get_axpy_scalar_kernel(cp)
        block = 256
        grid = (n_rows + block - 1) // block
        k((grid,), (block,), (
            dst_dev, src_dev,
            np.uint64(s[0]), np.uint64(s[1]), np.uint64(s[2]), np.uint64(n_rows),
        ))
        return dst_dev
    if s == ONE:
        return _fp3_add_dev(engine, dst_dev, src_dev)
    term = _fp3_mul_scalar_dev(engine, src_dev, s)
    return _fp3_add_dev(engine, dst_dev, term)


def _pair_sumcheck_terms(engine, f_dev, w_dev):
    if engine is not None and hasattr(engine, "cp"):
        cp = engine.cp
        pairs = len(f_dev) // 2
        terms = cp.empty((3 * pairs, 3), dtype=cp.uint64)
        g, b = engine._grid(pairs)
        engine.k["basefold_round_terms"](g, b, (f_dev, w_dev, terms, np.uint64(pairs)))
        half = pairs // 2
        while half >= 1:
            nxt = cp.empty((3 * half, 3), dtype=cp.uint64)
            g, b = engine._grid(3 * half)
            engine.k["fp3_pair_sum"](g, b, (terms, nxt, np.uint64(half), np.uint64(3)))
            terms, half = nxt, half // 2
        arr = cp.asnumpy(terms).reshape(3, 3)
        return [tuple(int(x) for x in arr[t]) for t in range(3)]
    from flashprove_next.field import CPU
    f_h = np.asarray(CE._host(engine, f_dev) if engine is not None else f_dev, dtype=np.uint64)
    w_h = np.asarray(CE._host(engine, w_dev) if engine is not None else w_dev, dtype=np.uint64)
    f0, f1 = f_h[0::2], f_h[1::2]
    w0, w1 = w_h[0::2], w_h[1::2]

    def dot3(u, v):
        u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]
        v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]
        t0 = CPU.add(CPU.mul(u1, v2), CPU.mul(u2, v1))
        c0 = CPU.bsum(CPU.add(CPU.mul(u0, v0), CPU.add(t0, t0)))
        t1 = CPU.mul(u2, v2)
        c1 = CPU.bsum(CPU.add(CPU.add(CPU.mul(u0, v1), CPU.mul(u1, v0)), CPU.add(t1, t1)))
        c2 = CPU.bsum(CPU.add(CPU.add(CPU.mul(u0, v2), CPU.mul(u1, v1)), CPU.mul(u2, v0)))
        return (int(c0), int(c1), int(c2))

    f2 = np.empty_like(f0)
    w2 = np.empty_like(w0)
    for k in range(3):
        f2[:, k] = CPU.sub(CPU.add(f1[:, k], f1[:, k]), f0[:, k])
        w2[:, k] = CPU.sub(CPU.add(w1[:, k], w1[:, k]), w0[:, k])
    return [dot3(f0, w0), dot3(f1, w1), dot3(f2, w2)]


def _halve_vec(engine, f_dev, r):
    if engine is not None and hasattr(engine, "cp"):
        return engine.halve(f_dev, r)
    from flashprove_next.field import CPU
    f_h = np.asarray(CE._host(engine, f_dev) if engine is not None else f_dev, dtype=np.uint64)
    f0, f1 = f_h[0::2], f_h[1::2]
    diff = np.empty_like(f0)
    for k in range(3):
        diff[:, k] = CPU.sub(f1[:, k], f0[:, k])
    prod = _fp3_mul_scalar_host(diff, r)
    out = np.empty_like(f0)
    for k in range(3):
        out[:, k] = CPU.add(f0[:, k], prod[:, k])
    return out


def _eq_table_fp3(r):
    from flashprove_next.field import CPU
    if not r:
        return np.array([[1, 0, 0]], dtype=np.uint64)
    e = CPU.eq_table(list(r))
    return np.ascontiguousarray(e.T, dtype=np.uint64)


# ----------------------------------------------------------------------------- Group commitment
@dataclass
class GroupCommitment:
    n: int
    cells: np.ndarray          # host (2^n, 3) uint64
    code_dev: object           # device/host (8*2^n, 3) uint64 (None if released to save VRAM)
    levels_dev: list | None    # device/host Merkle levels (None if released to save VRAM)
    root: bytes                # 32-byte BLAKE2b Merkle root
    engine: object = None

    def release_device(self):
        """Release device codeword & Merkle levels while keeping host cells + 32B root (recomputed on demand)."""
        self.code_dev = None
        self.levels_dev = None


def _ensure_code_dev(grp: GroupCommitment, engine):
    """Return codeword on engine (recomputing if released); returns `(code_dev, ephemeral_bool)`."""
    if grp.code_dev is not None:
        return (engine.dev(grp.code_dev) if (engine is not None and hasattr(engine, "cp")) else grp.code_dev), False
    if engine is not None:
        return CE.encode_codeword_dev(engine, grp.cells), True
    tuples = [tuple(int(x) for x in row) for row in grp.cells]
    com = pcs.zk_commit(tuples)
    return np.array(com.code, dtype=np.uint64).reshape(-1, 3), True


def _commit_cells(cells_np, n, engine=None):
    cells_np = np.ascontiguousarray(cells_np, dtype=np.uint64).reshape(1 << n, 3)
    if engine is not None:
        code_dev, levels_dev = CE.commit_pipeline_dev(engine, cells_np)
        root = bytes(np.asarray(CE._host(engine, levels_dev[-1][0]), dtype=np.uint8).tobytes())
        return GroupCommitment(n=n, cells=cells_np, code_dev=code_dev, levels_dev=levels_dev, root=root, engine=engine)
    tuples = [tuple(int(x) for x in row) for row in cells_np]
    com = pcs.zk_commit(tuples)
    code_np = np.array(com.code, dtype=np.uint64).reshape(-1, 3)
    return GroupCommitment(n=n, cells=cells_np, code_dev=code_np, levels_dev=com.levels, root=com.root, engine=None)


def commit_data_group(layout_or_n, data_cells, *, queries: int | None = None, engine=None, rng=None) -> GroupCommitment:
    """Pack data cells (length <= capacity) with fresh OS-CSPRNG blinding in the tail and commit.

    `layout_or_n` may be a `MultiPcsLayout` (uses `layout.n`, `layout.queries`) or an even int `n_g`
    (requires `queries`).
    """
    if isinstance(layout_or_n, MultiPcsLayout):
        n_g = layout_or_n.n
        q_use = layout_or_n.queries
    else:
        n_g = int(layout_or_n)
        if queries is None:
            raise ValueError("queries required when passing integer n_g")
        q_use = int(queries)
    N = 1 << n_g
    tl = tail_len_for(q_use)
    cap = N - tl
    arr = np.asarray(data_cells, dtype=np.uint64)
    if arr.ndim == 1:
        if len(arr) > cap:
            raise ValueError(f"data length {len(arr)} exceeds capacity {cap}")
        cells = np.zeros((N, 3), dtype=np.uint64)
        cells[:len(arr), 0] = arr
    elif arr.ndim == 2 and arr.shape[1] == 3:
        if len(arr) > cap:
            raise ValueError(f"data length {len(arr)} exceeds capacity {cap}")
        cells = np.zeros((N, 3), dtype=np.uint64)
        cells[:len(arr)] = arr
    else:
        raise ValueError("data_cells must be 1-D uint64 or 2-D (M, 3) uint64")
    if cap > 0 and int(cells[:cap].max()) >= P:
        raise ValueError("noncanonical field element in data_cells")
    cells[cap:] = _os_rand_fp3(tl, rng=rng)
    return _commit_cells(cells, n_g, engine=engine)


def commit_blinding_group(n: int, *, engine=None, rng=None) -> GroupCommitment:
    """Commit one full-length OS-CSPRNG uniform Fp3 blinding group of size 2^n."""
    cells = _os_rand_fp3(1 << n, rng=rng)
    return _commit_cells(cells, n, engine=engine)


def _open_group_queries(grp: GroupCommitment, positions, engine=None):
    eng = grp.engine if grp.engine is not None else engine
    if grp.code_dev is None or grp.levels_dev is None:
        tmp = _commit_cells(grp.cells, grp.n, engine=eng)
        assert tmp.root == grp.root, "recomputed group root mismatch"
        try:
            return _open_group_queries(tmp, positions, engine=eng)
        finally:
            tmp.release_device()
            if eng is not None and hasattr(eng, "cp"):
                eng.cp.get_default_memory_pool().free_all_blocks()
    if grp.engine is not None:
        vals, sibs = CE.open_rows(grp.engine, grp.code_dev, grp.levels_dev, positions)
        depth = len(sibs)
        paths = [[bytes(sibs[d][q].tobytes()) for d in range(depth)] for q in range(len(positions))]
        return vals, paths
    quarter = len(grp.code_dev) // 4
    vals = np.empty((len(positions), 4, 3), dtype=np.uint64)
    paths = []
    for q, p in enumerate(positions):
        for k in range(4):
            vals[q, k] = grp.code_dev[p + k * quarter]
        paths.append(merkle_path(grp.levels_dev, p))
    return vals, paths


# ----------------------------------------------------------------------------- Tiny single-group Basefold for L
def _prove_single_open(grp: GroupCommitment, w_host: np.ndarray, queries: int, transcript: Fpn2Transcript) -> bytes:
    """Tiny CPU Basefold opening for the n_lib = 8..10 Libra group L (256..1024 cells -> < 1 ms)."""
    n = grp.n
    folds = n // 2
    code_len = RATE << n
    out = bytearray()
    h_cur = grp.cells.copy()
    w_cur = np.ascontiguousarray(w_host, dtype=np.uint64).reshape(1 << n, 3)
    cur_code = np.asarray(CE._host(grp.engine, grp.code_dev) if grp.engine is not None else grp.code_dev, dtype=np.uint64)
    l0_levels = [np.asarray(CE._host(grp.engine, lv), dtype=np.uint8) for lv in grp.levels_dev] if grp.engine is not None else grp.levels_dev

    omega = CE.root_of_unity(code_len)
    inv_iota = pow(CE.root_of_unity(4), P - 2, P)
    r_bf = []
    fold_layers = []
    fold_trees = []
    for i in range(n):
        vals = _pair_sumcheck_terms(None, h_cur, w_cur)
        msg = enc3s(vals)
        out += msg
        transcript.absorb_bytes(msg)
        ri = transcript.squeeze_fp3()
        r_bf.append(ri)
        h_cur = _halve_vec(None, h_cur, ri)
        w_cur = _halve_vec(None, w_cur, ri)
        if i % 2 == 1:
            t = i // 2
            omega_t = pow(omega, 4 ** t, P)
            inv_w_t = pow(omega_t, P - 2, P)
            cur_tuples = [tuple(int(x) for x in row) for row in cur_code]
            q_cur = len(cur_tuples) // 4
            nxt_tuples = []
            x_inv_p = 1
            for p in range(q_cur):
                coset = [cur_tuples[p + k * q_cur] for k in range(4)]
                nxt_tuples.append(pcs.fold4(coset, x_inv_p, inv_iota, r_bf[2 * t], r_bf[2 * t + 1]))
                x_inv_p = (x_inv_p * inv_w_t) % P
            cur_code = np.array(nxt_tuples, dtype=np.uint64).reshape(-1, 3)
            if t < folds - 1:
                quarter = len(nxt_tuples) // 4
                leaves_t = [leaf_hash([nxt_tuples[p + k * quarter] for k in range(4)]) for p in range(quarter)]
                lv = merkle_levels(leaves_t)
                root_t = lv[-1][0]
                out += root_t
                transcript.absorb_bytes(root_t)
                fold_layers.append(cur_code)
                fold_trees.append(lv)

    final_val = (int(h_cur[0, 0]), int(h_cur[0, 1]), int(h_cur[0, 2]))
    fin_b = enc3(final_val)
    out += fin_b
    transcript.absorb_bytes(fin_b)

    q_mask = (code_len // 4) - 1
    positions = [int.from_bytes(transcript.squeeze_bytes(8), "little") & q_mask for _ in range(queries)]
    l0_vals, l0_paths = _open_group_queries(grp, positions)

    for q, p0 in enumerate(positions):
        out += enc3s([tuple(int(x) for x in l0_vals[q, k]) for k in range(4)])
        for sib in l0_paths[q]:
            out += sib
        cur_p = p0
        for t in range(1, folds):
            q_t = (code_len >> (2 * t)) // 4
            cur_p = cur_p % q_t
            code_t = fold_layers[t - 1]
            out += enc3s([tuple(int(x) for x in code_t[cur_p + k * q_t]) for k in range(4)])
            for sib in merkle_path(fold_trees[t - 1], cur_p):
                out += sib
    return bytes(out)


def _verify_single_open(n: int, root: bytes, claim, w_mle_fn, queries: int, buf: bytes, off: int, transcript: Fpn2Transcript) -> int:
    folds = n // 2
    code_len = RATE << n
    target = claim
    r_bf = []
    fold_roots = []
    for i in range(n):
        rs = off
        p0, off = _read_fp3(buf, off)
        p1, off = _read_fp3(buf, off)
        p2, off = _read_fp3(buf, off)
        if hadd(p0, p1) != target:
            raise MultiPcsError("E_SUMCHECK", "Libra Basefold round sum mismatch")
        transcript.absorb_bytes(buf[rs:off])
        ri = transcript.squeeze_fp3()
        r_bf.append(ri)
        target = lagrange_eval([p0, p1, p2], ri)
        if i % 2 == 1 and (i // 2) < folds - 1:
            rt, off = _read_bytes(buf, off, 32)
            transcript.absorb_bytes(rt)
            fold_roots.append(rt)

    fs = off
    final_val, off = _read_fp3(buf, off)
    transcript.absorb_bytes(buf[fs:off])
    if hmul(final_val, w_mle_fn(r_bf)) != target:
        raise MultiPcsError("E_FINAL", "Libra Basefold final mismatch")

    q_mask = (code_len // 4) - 1
    positions = [int.from_bytes(transcript.squeeze_bytes(8), "little") & q_mask for _ in range(queries)]
    omega = CE.root_of_unity(code_len)
    inv_iota = pow(CE.root_of_unity(4), P - 2, P)
    for p0 in positions:
        cur_p = p0
        prev = None
        for t in range(folds):
            dt = n + 1 - 2 * t
            coset = []
            for _ in range(4):
                v, off = _read_fp3(buf, off)
                coset.append(v)
            path = []
            for _ in range(dt):
                s, off = _read_bytes(buf, off, 32)
                path.append(s)
            if t == 0:
                _check_coset_merkle(root, coset, cur_p, path)
            else:
                q_t = (code_len >> (2 * t)) // 4
                slot = cur_p // q_t
                cur_p = cur_p % q_t
                _check_coset_merkle(fold_roots[t - 1], coset, cur_p, path)
                if coset[slot] != prev:
                    raise MultiPcsError("E_FOLD", "Libra layer slot mismatch")
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(pow(omega_t, cur_p, P), P - 2, P)
            prev = pcs.fold4(coset, x_inv, inv_iota, r_bf[2 * t], r_bf[2 * t + 1])
        if prev != final_val:
            raise MultiPcsError("E_FOLD", "Libra final fold mismatch")
    return off


# ----------------------------------------------------------------------------- Prover
def prove_multi_open(
    layout: MultiPcsLayout,
    groups: list[GroupCommitment],
    weights: list[np.ndarray],
    claims: list[tuple[int, int, int]],
    transcript: Fpn2Transcript,
    *,
    engine=None,
    rng=None,
    blinding_group: GroupCommitment | None = None,
) -> bytes:
    """Prove `<F_g, W_g> = claims[g]` for all g in 0..G-1 (mixed sizes allowed!) in ONE fold chain."""
    G = layout.n_groups
    n = layout.n
    if len(groups) != G or len(weights) != G or len(claims) != G:
        raise ValueError("groups, weights, and claims must all have length layout.n_groups")
    for g in range(G):
        if groups[g].n != layout.group_ns[g]:
            raise ValueError(f"group {g} has n={groups[g].n}, expected {layout.group_ns[g]}")

    B_grp = blinding_group if blinding_group is not None else commit_blinding_group(n, engine=engine, rng=rng)
    # Dedicated tiny Libra group L (n_lib = 8..10, 0.4 ms to commit & open!)
    n_lib = layout.n_lib
    libra_coeffs = _os_rand_fp3(3 * n, rng=rng)
    L_grp = commit_data_group(n_lib, libra_coeffs, queries=layout.queries, engine=engine, rng=rng)
    g_sum = libra_sum_from_coeffs(libra_coeffs, n)

    out = bytearray()
    out += B_grp.root
    out += L_grp.root
    out += enc3(g_sum)

    for g in range(G):
        transcript.absorb_bytes(groups[g].root)
    transcript.absorb_bytes(B_grp.root)
    transcript.absorb_bytes(L_grp.root)
    transcript.absorb_bytes(enc3s(claims))
    transcript.absorb_bytes(enc3(g_sum))

    mu = transcript.squeeze_fp3()
    lam = transcript.squeeze_fp3()

    mu_pows = [ONE]
    for _ in range(1, G):
        mu_pows.append(hmul(mu_pows[-1], mu))

    # ---- Step A: n-round degree-2 Libra-masked multi-group inner-product sumcheck
    # Group g of size 2^{n_g} acts on high variables (start_g .. n-1) with [x_lo == 0] on (0 .. start_g-1)
    # where start_g = n - n_g = 2 * inject_layer(g).
    # VRAM-bounded: keeps active groups GPU-resident up to 24 GiB, streaming oversized early rounds 1 group at a time.
    use_dev = engine is not None and hasattr(engine, "cp")
    start_round = [n - layout.group_ns[g] for g in range(G)]
    f_cur = [None] * G
    w_cur = [None] * G
    on_dev = [False] * G
    z_lo = [ONE] * G
    step_a_vram_budget = 24 << 30
    step_a_vram_used = 0

    def _activate(g):
        nonlocal step_a_vram_used
        ng = layout.group_ns[g]
        wg = np.ascontiguousarray(weights[g], dtype=np.uint64).reshape(1 << ng, 3)
        need = 2 * (1 << ng) * 24
        if use_dev and step_a_vram_used + need <= step_a_vram_budget:
            f_cur[g] = engine.dev(groups[g].cells)
            w_cur[g] = engine.dev(wg)
            on_dev[g] = True
            step_a_vram_used += need
        else:
            f_cur[g] = groups[g].cells
            w_cur[g] = wg
            on_dev[g] = False

    for g in range(G):
        if start_round[g] == 0:
            _activate(g)

    r_ip = []
    libra_prefix = ZERO
    neg_one = hf(P - 1)
    for i in range(n):
        acc = [ZERO, ZERO, ZERO]
        for g in range(G):
            mg = mu_pows[g]
            if i < start_round[g]:
                c0 = hmul(hmul(mg, z_lo[g]), claims[g])
                acc[0] = hadd(acc[0], c0)
                acc[2] = hadd(acc[2], hmul(neg_one, c0))
            else:
                if use_dev and not on_dev[g]:
                    fd = engine.dev(f_cur[g])
                    wd = engine.dev(w_cur[g])
                    tg = _pair_sumcheck_terms(engine, fd, wd)
                    del fd, wd
                else:
                    tg = _pair_sumcheck_terms(engine if on_dev[g] else None, f_cur[g], w_cur[g])
                coeff = hmul(mg, z_lo[g])
                for t in range(3):
                    acc[t] = hadd(acc[t], hmul(coeff, tg[t]))
        lvals = libra_round_vals(libra_coeffs, n, i, libra_prefix)
        for t in range(3):
            acc[t] = hadd(acc[t], hmul(lam, lvals[t]))
        msg = enc3s(acc)
        out += msg
        transcript.absorb_bytes(msg)
        ri = transcript.squeeze_fp3()
        r_ip.append(ri)
        libra_prefix = hadd(libra_prefix, libra_eval_var(libra_coeffs, i, ri))
        one_minus_r = hsub(ONE, ri)
        for g in range(G):
            if i < start_round[g]:
                z_lo[g] = hmul(z_lo[g], one_minus_r)
                if i + 1 == start_round[g]:
                    _activate(g)
            else:
                if on_dev[g]:
                    old_b = 2 * len(f_cur[g]) * 24
                    f_cur[g] = _halve_vec(engine, f_cur[g], ri)
                    w_cur[g] = _halve_vec(engine, w_cur[g], ri)
                    step_a_vram_used -= old_b // 2
                elif use_dev:
                    fd = engine.dev(f_cur[g])
                    fh = engine.halve(fd, ri)
                    del fd
                    wd = engine.dev(w_cur[g])
                    wh = engine.halve(wd, ri)
                    del wd
                    need = 2 * len(fh) * 24
                    if step_a_vram_used + need <= step_a_vram_budget:
                        f_cur[g], w_cur[g] = fh, wh
                        on_dev[g] = True
                        step_a_vram_used += need
                    else:
                        f_cur[g] = CE._host(engine, fh)
                        w_cur[g] = CE._host(engine, wh)
                        del fh, wh
                else:
                    f_cur[g] = _halve_vec(None, f_cur[g], ri)
                    w_cur[g] = _halve_vec(None, w_cur[g], ri)
        if use_dev:
            engine.cp.get_default_memory_pool().free_all_blocks()

    v_groups = []
    for g in range(G):
        row = CE._host(engine, f_cur[g])[0] if on_dev[g] else f_cur[g][0]
        v_groups.append((int(row[0]), int(row[1]), int(row[2])))
    del f_cur, w_cur
    if use_dev:
        engine.cp.get_default_memory_pool().free_all_blocks()

    eq_ip = _eq_table_fp3(r_ip)
    if use_dev:
        tb = _pair_sumcheck_terms(engine, engine.dev(B_grp.cells), engine.dev(eq_ip))
    else:
        tb = _pair_sumcheck_terms(None, B_grp.cells, eq_ip)
    v_B = hadd(tb[0], tb[1])
    e_lib = libra_prefix

    finals_a_bytes = enc3s(v_groups + [v_B, e_lib])
    out += finals_a_bytes
    transcript.absorb_bytes(finals_a_bytes)

    # ---- Tiny standalone Basefold opening of L for <L, W_lib(r_ip)> = e_lib
    w_lib = np.zeros((1 << n_lib, 3), dtype=np.uint64)
    for k, wv in enumerate(libra_weight_entries(r_ip)):
        w_lib[k] = wv
    out += _prove_single_open(L_grp, w_lib, layout.queries, transcript)

    # ---- Step B: Single batched rate-1/8 coset-4 Basefold opening with PURE W_H = eq(r_ip)
    theta = transcript.squeeze_fp3()
    theta_pows = []
    cur_t = theta
    for _ in range(G):
        theta_pows.append(cur_t)
        cur_t = hmul(cur_t, theta)

    by_layer: dict[int, list[int]] = {}
    for g in range(G):
        by_layer.setdefault(layout.inject_layer(g), []).append(g)

    if use_dev:
        h_cur = engine.dev(B_grp.cells).copy()
        wh_cur = engine.dev(eq_ip)
        # At layer t=0, RS is linear: accumulate ALL t_g==0 groups into h_cur first and encode ONCE!
        if n >= 24:
            B_grp.release_device()
            for g in by_layer.get(0, ()):
                groups[g].release_device()
            engine.cp.get_default_memory_pool().free_all_blocks()
        for g in by_layer.get(0, ()):
            h_cur = _fp3_axpy_dev(engine, h_cur, engine.dev(groups[g].cells), theta_pows[g])
        cur_code = CE.encode_codeword_dev(engine, h_cur)
    else:
        h_cur = B_grp.cells.copy()
        wh_cur = eq_ip
        cur_code = B_grp.code_dev.copy()
        for g in by_layer.get(0, ()):
            h_cur = _fp3_axpy_dev(None, h_cur, groups[g].cells, theta_pows[g])
            cur_code = _fp3_axpy_dev(None, cur_code, groups[g].code_dev, theta_pows[g])

    omega = CE.root_of_unity(layout.code_len)
    inv_iota = pow(CE.root_of_unity(4), P - 2, P)
    r_bf = []
    fold_layers = []
    fold_trees = []

    for i in range(n):
        # At the start of fold layer t = i // 2 (t >= 1; t == 0 was already folded linearly above!), inject t_g == t
        if i % 2 == 0 and i > 0:
            t = i // 2
            inject_g = by_layer.get(t, ())
            if inject_g and use_dev:
                # Copy before in-place axpy so `fold_layers[t - 1]` keeps the exact pre-injection codeword!
                cur_code = cur_code.copy()
            for g in inject_g:
                tg = theta_pows[g]
                g_code, eph = _ensure_code_dev(groups[g], engine if use_dev else None)
                if use_dev:
                    h_cur = _fp3_axpy_dev(engine, h_cur, engine.dev(groups[g].cells), tg)
                    cur_code = _fp3_axpy_dev(engine, cur_code, g_code, tg)
                else:
                    h_cur = _fp3_axpy_dev(None, h_cur, groups[g].cells, tg)
                    cur_code = _fp3_axpy_dev(None, cur_code, g_code, tg)
                if eph:
                    del g_code
                    if use_dev:
                        engine.cp.get_default_memory_pool().free_all_blocks()

        vals = _pair_sumcheck_terms(engine if use_dev else None, h_cur, wh_cur)
        msg = enc3s(vals)
        out += msg
        transcript.absorb_bytes(msg)
        ri = transcript.squeeze_fp3()
        r_bf.append(ri)
        h_cur = _halve_vec(engine if use_dev else None, h_cur, ri)
        wh_cur = _halve_vec(engine if use_dev else None, wh_cur, ri)

        if i % 2 == 1:
            t = i // 2
            omega_t = pow(omega, 4 ** t, P)
            inv_w_t = pow(omega_t, P - 2, P)
            if use_dev:
                cur_code = engine.fold4(cur_code, inv_w_t, inv_iota, r_bf[2 * t], r_bf[2 * t + 1])
                if t < layout.folds - 1:
                    lv = CE.merkle_levels_dev(engine, cur_code)
                    root_t = bytes(np.asarray(CE._host(engine, lv[-1][0]), dtype=np.uint8).tobytes())
                    out += root_t
                    transcript.absorb_bytes(root_t)
                    fold_layers.append(cur_code)
                    fold_trees.append(lv)
            else:
                cur_tuples = [tuple(int(x) for x in row) for row in cur_code]
                q_cur = len(cur_tuples) // 4
                nxt_tuples = []
                x_inv_p = 1
                for p in range(q_cur):
                    coset = [cur_tuples[p + k * q_cur] for k in range(4)]
                    nxt_tuples.append(pcs.fold4(coset, x_inv_p, inv_iota, r_bf[2 * t], r_bf[2 * t + 1]))
                    x_inv_p = (x_inv_p * inv_w_t) % P
                cur_code = np.array(nxt_tuples, dtype=np.uint64).reshape(-1, 3)
                if t < layout.folds - 1:
                    quarter = len(nxt_tuples) // 4
                    leaves_t = [leaf_hash([nxt_tuples[p + k * quarter] for k in range(4)]) for p in range(quarter)]
                    lv = merkle_levels(leaves_t)
                    root_t = lv[-1][0]
                    out += root_t
                    transcript.absorb_bytes(root_t)
                    fold_layers.append(cur_code)
                    fold_trees.append(lv)

    final_row = CE._host(engine, h_cur)[0] if use_dev else h_cur[0]
    final_val = (int(final_row[0]), int(final_row[1]), int(final_row[2]))
    final_bytes = enc3(final_val)
    out += final_bytes
    transcript.absorb_bytes(final_bytes)
    del h_cur, wh_cur, cur_code
    if use_dev:
        engine.cp.get_default_memory_pool().free_all_blocks()

    q_mask = (layout.code_len // 4) - 1
    positions = [int.from_bytes(transcript.squeeze_bytes(8), "little") & q_mask for _ in range(layout.queries)]

    # Precompute query positions at each layer t = 0..folds-1
    layer_pos = [list(positions)]
    for t in range(1, layout.folds):
        q_t = (layout.code_len >> (2 * t)) // 4
        layer_pos.append([p % q_t for p in layer_pos[-1]])

    b_vals, b_paths = _open_group_queries(B_grp, layer_pos[0])
    g_openings = [_open_group_queries(groups[g], layer_pos[layout.inject_layer(g)]) for g in range(G)]

    fold_openings = []
    for t in range(1, layout.folds):
        q_t = (layout.code_len >> (2 * t)) // 4
        pos_t = layer_pos[t]
        if use_dev:
            vals_t, sibs_t = CE.open_rows(engine, fold_layers[t - 1], fold_trees[t - 1], pos_t)
            depth_t = len(sibs_t)
            paths_t = [[bytes(sibs_t[d][q].tobytes()) for d in range(depth_t)] for q in range(layout.queries)]
            fold_openings.append((vals_t, paths_t))
        else:
            code_t = fold_layers[t - 1]
            lv_t = fold_trees[t - 1]
            vals_t = np.empty((layout.queries, 4, 3), dtype=np.uint64)
            paths_t = []
            for q, p in enumerate(pos_t):
                for k in range(4):
                    vals_t[q, k] = code_t[p + k * q_t]
                paths_t.append(merkle_path(lv_t, p))
            fold_openings.append((vals_t, paths_t))

    # Serialize Q queries: for each q, layer t = 0..folds-1:
    #   at t=0: B coset+path, then groups with inject_layer==0;
    #   at t>0: fold_layer[t-1] coset+path, then groups with inject_layer==t.
    for q in range(layout.queries):
        for t in range(layout.folds):
            if t == 0:
                out += enc3s([tuple(int(x) for x in b_vals[q, k]) for k in range(4)])
                for sib in b_paths[q]:
                    out += sib
            else:
                fv, fp = fold_openings[t - 1]
                out += enc3s([tuple(int(x) for x in fv[q, k]) for k in range(4)])
                for sib in fp[q]:
                    out += sib
            for g in by_layer.get(t, ()):
                gv, gp = g_openings[g]
                out += enc3s([tuple(int(x) for x in gv[q, k]) for k in range(4)])
                for sib in gp[q]:
                    out += sib

    assert len(out) == layout.proof_bytes, (len(out), layout.proof_bytes)
    return bytes(out)


# ----------------------------------------------------------------------------- Verifier
class MultiPcsError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _read_fp3(buf, off):
    if off + 24 > len(buf):
        raise MultiPcsError("E_LENGTH", "truncated Fp3")
    try:
        return dec3(buf[off:off + 24]), off + 24
    except ValueError as e:
        raise MultiPcsError("E_NONCANONICAL", str(e)) from None


def _read_bytes(buf, off, count):
    if off + count > len(buf):
        raise MultiPcsError("E_LENGTH", "truncated bytes")
    return bytes(buf[off:off + count]), off + count


def _check_coset_merkle(root, values, idx, path):
    cur = leaf_hash(values)
    for d, sib in enumerate(path):
        cur = node_hash(sib, cur) if (idx >> d) & 1 else node_hash(cur, sib)
    if cur != root:
        raise MultiPcsError("E_MERKLE", f"coset Merkle mismatch at idx={idx}")


def verify_multi_open(
    layout: MultiPcsLayout,
    group_roots: list[bytes],
    claims: list[tuple[int, int, int]],
    w_eval_fn,
    proof: bytes,
    transcript: Fpn2Transcript,
) -> dict:
    """Verify a `prove_multi_open` proof against `group_roots`, `claims`, and `w_eval_fn(g, r_ip_g)`
    where `r_ip_g = r_ip[n - n_g :]` has length `layout.group_ns[g]`."""
    G = layout.n_groups
    n = layout.n
    if len(group_roots) != G or len(claims) != G:
        raise MultiPcsError("E_INVALID_INPUT", "group_roots and claims length mismatch")
    if len(proof) != layout.proof_bytes:
        raise MultiPcsError("E_LENGTH", f"got {len(proof)} B, expected {layout.proof_bytes} B")

    off = 0
    root_B, off = _read_bytes(proof, off, 32)
    root_L, off = _read_bytes(proof, off, 32)
    g_sum, off = _read_fp3(proof, off)

    for r in group_roots:
        transcript.absorb_bytes(r)
    transcript.absorb_bytes(root_B)
    transcript.absorb_bytes(root_L)
    transcript.absorb_bytes(enc3s(claims))
    transcript.absorb_bytes(enc3(g_sum))

    mu = transcript.squeeze_fp3()
    lam = transcript.squeeze_fp3()

    mu_pows = [ONE]
    for _ in range(1, G):
        mu_pows.append(hmul(mu_pows[-1], mu))

    # ---- Step A verification
    target = hmul(lam, g_sum)
    for g in range(G):
        target = hadd(target, hmul(mu_pows[g], claims[g]))

    r_ip = []
    for _ in range(n):
        raw_start = off
        p0, off = _read_fp3(proof, off)
        p1, off = _read_fp3(proof, off)
        p2, off = _read_fp3(proof, off)
        if hadd(p0, p1) != target:
            raise MultiPcsError("E_SUMCHECK", "Step-A round sum mismatch")
        transcript.absorb_bytes(proof[raw_start:off])
        ri = transcript.squeeze_fp3()
        r_ip.append(ri)
        target = lagrange_eval([p0, p1, p2], ri)

    finals_start = off
    v_groups = []
    for _ in range(G):
        vg, off = _read_fp3(proof, off)
        v_groups.append(vg)
    v_B, off = _read_fp3(proof, off)
    e_lib, off = _read_fp3(proof, off)
    transcript.absorb_bytes(proof[finals_start:off])

    expect_a = hmul(lam, e_lib)
    for g in range(G):
        s_g = n - layout.group_ns[g]
        z_lo_g = ONE
        for j in range(s_g):
            z_lo_g = hmul(z_lo_g, hsub(ONE, r_ip[j]))
        wg_eval = w_eval_fn(g, r_ip[s_g:])
        expect_a = hadd(expect_a, hmul(hmul(mu_pows[g], z_lo_g), hmul(v_groups[g], wg_eval)))
    if expect_a != target:
        raise MultiPcsError("E_SUMCHECK", "Step-A final evaluation mismatch")

    # ---- Verify tiny Libra Basefold opening for L
    n_lib = layout.n_lib
    off = _verify_single_open(
        n_lib, root_L, e_lib,
        lambda r_lib: libra_weight_mle_at0(r_ip, r_lib, n_lib),
        layout.queries, proof, off, transcript,
    )

    # ---- Step B verification (single chain with multi-size layer injection)
    theta = transcript.squeeze_fp3()
    theta_pows = []
    cur_t = theta
    for _ in range(G):
        theta_pows.append(cur_t)
        cur_t = hmul(cur_t, theta)

    by_layer: dict[int, list[int]] = {}
    for g in range(G):
        by_layer.setdefault(layout.inject_layer(g), []).append(g)

    target_b = v_B
    r_bf = []
    fold_roots = []
    c_lo = ONE
    for i in range(n):
        if i % 2 == 0:
            t = i // 2
            for g in by_layer.get(t, ()):
                target_b = hadd(target_b, hmul(theta_pows[g], hmul(c_lo, v_groups[g])))
        raw_start = off
        p0, off = _read_fp3(proof, off)
        p1, off = _read_fp3(proof, off)
        p2, off = _read_fp3(proof, off)
        if hadd(p0, p1) != target_b:
            raise MultiPcsError("E_SUMCHECK", "Step-B Basefold round sum mismatch")
        transcript.absorb_bytes(proof[raw_start:off])
        ri = transcript.squeeze_fp3()
        r_bf.append(ri)
        c_lo = hmul(c_lo, eq1(r_ip[i], ri))
        target_b = lagrange_eval([p0, p1, p2], ri)
        if i % 2 == 1 and (i // 2) < layout.folds - 1:
            rt, off = _read_bytes(proof, off, 32)
            transcript.absorb_bytes(rt)
            fold_roots.append(rt)

    final_start = off
    final_val, off = _read_fp3(proof, off)
    transcript.absorb_bytes(proof[final_start:off])

    if hmul(final_val, c_lo) != target_b:
        raise MultiPcsError("E_FINAL", "Step-B final Basefold evaluation mismatch")

    # ---- Query verification
    q_mask = (layout.code_len // 4) - 1
    positions = [int.from_bytes(transcript.squeeze_bytes(8), "little") & q_mask for _ in range(layout.queries)]

    omega = CE.root_of_unity(layout.code_len)
    inv_iota = pow(CE.root_of_unity(4), P - 2, P)

    for p0 in positions:
        cur_p = p0
        prev = None
        for t in range(layout.folds):
            dt = layout.depth(t)
            base_coset = []
            for _ in range(4):
                v, off = _read_fp3(proof, off)
                base_coset.append(v)
            base_path = []
            for _ in range(dt):
                s, off = _read_bytes(proof, off, 32)
                base_path.append(s)
            if t == 0:
                _check_coset_merkle(root_B, base_coset, cur_p, base_path)
            else:
                q_t = (layout.code_len >> (2 * t)) // 4
                slot = cur_p // q_t
                cur_p = cur_p % q_t
                _check_coset_merkle(fold_roots[t - 1], base_coset, cur_p, base_path)
                if base_coset[slot] != prev:
                    raise MultiPcsError("E_FOLD", f"layer {t} slot consistency mismatch")

            comb = list(base_coset)
            for g in by_layer.get(t, ()):
                g_coset = []
                for _ in range(4):
                    v, off = _read_fp3(proof, off)
                    g_coset.append(v)
                g_path = []
                for _ in range(dt):
                    s, off = _read_bytes(proof, off, 32)
                    g_path.append(s)
                _check_coset_merkle(group_roots[g], g_coset, cur_p, g_path)
                tg = theta_pows[g]
                for k in range(4):
                    comb[k] = hadd(comb[k], hmul(tg, g_coset[k]))

            omega_t = pow(omega, 4 ** t, P)
            x_inv_t = pow(pow(omega_t, cur_p, P), P - 2, P)
            prev = pcs.fold4(comb, x_inv_t, inv_iota, r_bf[2 * t], r_bf[2 * t + 1])

        if prev != final_val:
            raise MultiPcsError("E_FOLD", "final fold mismatch with FINAL")

    assert off == len(proof)
    return {
        "r_ip": r_ip,
        "r_bf": r_bf,
        "v_groups": v_groups,
        "final": final_val,
        "proof_bytes": len(proof),
    }
