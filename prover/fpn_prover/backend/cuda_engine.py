# SPDX-License-Identifier: Apache-2.0
"""Portable engine for the relation-3 prover kernels in `kernels/goldilocks_ntt.cu`.

Two interchangeable executors expose the same primitive operations on NumPy uint64 arrays:

  * `CudaEngine`   -- CuPy `RawModule` (NVRTC) on an NVIDIA GPU (RTX 4070 Ti SUPER, H100, ...);
  * `HostEngine`   -- the same .cu compiled for the CPU through `kernels/host_emulation.cpp`
                      (sequential thread emulation) so parity can be checked without CUDA.

`commit_pipeline` / `fold_chain` / `gkr_tree` build the folded-PCS commitment, the fused coset-4
fold chain with Merkle roots and the FracLogUp fraction tree from those primitives; their results
are bit-exact with `folded_pcs.zk_commit` / `zk_open` and `frac_logup_gkr.build_tree`.
Layouts: an Fp3 vector is a (n, 3) uint64 array (limb 0 = constant term); a codeword of length L is
stored in natural order (position j = evaluation at w^j).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

P = (1 << 64) - (1 << 32) + 1
HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "kernels", "goldilocks_ntt.cu")
HOST = os.path.join(HERE, "kernels", "host_emulation.cpp")
RATE = 8
FRAC_TILE = 128       # pairs per block of the tiled fraction_layer kernel (== FRAC_TILE in the .cu)


def root_of_unity(length):
    return pow(7, (P - 1) // length, P)


def stage_twiddles(n):
    out, half = [], 1
    while half < n:
        w = pow(7, (P - 1) // (2 * half), P)
        cur = 1
        for _ in range(half):
            out.append(cur)
            cur = cur * w % P
        half <<= 1
    return np.array(out, dtype=np.uint64)


# ---------------------------------------------------------------------------- Bailey SMEM NTT pass plan
TILE_LOG_MAX = 12      # <= 2^12-point shared-memory butterflies per tile (32 KiB of Words)
SMEM_LOG = 12          # tile words per block (2^m << c) <= 2^12 -> <= 32 KiB + 3 KiB factors < 48 KiB
COLS_LOG_MAX = 5       # <= 32 consecutive tiles per block (256-byte coalesced segments)


def bailey_plan(log_n, max_tile_log=TILE_LOG_MAX, smem_log=SMEM_LOG, max_cols_log=COLS_LOG_MAX):
    """Pass plan [(s, m, c), ...] of `bailey_pass` for a length-2^log_n NTT (shared by both engines).

    The log_n radix-2 stages are split into ceil(log_n / max_tile_log) balanced groups of m stages
    (e.g. 2^27 -> 9 + 9 + 9, 2^24 -> 12 + 12, 2^13 -> 7 + 6); pass (s, m, c) runs stages s .. s+m-1 on
    2^m-point shared-memory tiles, 2^c tiles per block.  The first pass also does the bit reversal.
    """
    if type(log_n) is not int or not 1 <= log_n <= 32:
        raise ValueError("bailey_plan needs 1 <= log_n <= 32")
    if not 1 <= max_tile_log <= TILE_LOG_MAX or not max_tile_log <= smem_log <= SMEM_LOG:
        raise ValueError("invalid tile / shared-memory limits")
    passes = -(-log_n // max_tile_log)
    sizes = [log_n // passes + (1 if i < log_n % passes else 0) for i in range(passes)]
    plan, s = [], 0
    for m in sizes:
        c = max(0, min(s, smem_log - m, max_cols_log))
        plan.append((s, m, c))
        s += m
    return plan


def check_plan(log_n, plan):
    """A plan must run every stage exactly once, in order, within the shared-memory budget."""
    s = 0
    for p in plan:
        if len(p) != 3 or any(type(v) is not int for v in p):
            raise ValueError("malformed Bailey pass")
        ps, m, c = p
        if ps != s or not 1 <= m <= TILE_LOG_MAX or not 0 <= c <= ps or m + c > SMEM_LOG + COLS_LOG_MAX \
                or (1 << m) << c > 1 << SMEM_LOG:
            raise ValueError(f"invalid Bailey pass {p}")
        s += m
    if s != log_n:
        raise ValueError("Bailey plan does not cover every stage")
    return [tuple(p) for p in plan]


def bailey_smem_bytes(m, c):
    return 8 * (((1 << m) << c) + (m << c))


LOCAL_TWIDDLES = None


def local_twiddles():
    """Stage-packed twiddles of length 2^12 (4095 words): serves every pass with m <= 12."""
    global LOCAL_TWIDDLES
    if LOCAL_TWIDDLES is None:
        LOCAL_TWIDDLES = stage_twiddles(1 << TILE_LOG_MAX)
    return LOCAL_TWIDDLES


def to_array(values):
    """List of Fp3 tuples -> (n, 3) uint64."""
    return np.array(values, dtype=np.uint64).reshape(-1, 3)


def to_tuples(arr):
    return [tuple(int(x) for x in row) for row in np.asarray(arr).reshape(-1, 3)]


# ============================================================================ executors
class HostEngine:
    """Runs the CUDA kernels on the CPU via the host-emulation harness (parity testing)."""

    name = "host-emulation"

    def __init__(self, workdir=None, compiler=None):
        compiler = compiler or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise RuntimeError("no C++ compiler for the host emulation")
        self.dir = workdir or tempfile.mkdtemp(prefix="fpn-host-emu-")
        self.exe = os.path.join(self.dir, "host_emulation")
        subprocess.run([compiler, "-std=c++17", "-O2", "-o", self.exe, HOST], check=True,
                       capture_output=True)

    def _run(self, op, words):
        data = np.concatenate([np.asarray(w, dtype=np.uint64).ravel() for w in words]).astype("<u8").tobytes()
        r = subprocess.run([self.exe, op], input=data, capture_output=True, check=True)
        return r.stdout

    def fold4(self, code, inv_omega, inv_iota, ra, rb):
        quarter = len(code) // 4
        out = self._run("fold4", [[quarter, inv_omega, inv_iota], list(ra) + list(rb), code])
        return np.frombuffer(out, dtype="<u8").reshape(quarter, 3).astype(np.uint64)

    def merkle_leaves(self, code):
        quarter = len(code) // 4
        out = self._run("leaf", [[quarter], code])
        return np.frombuffer(out, dtype=np.uint8).reshape(quarter, 32).copy()

    def merkle_layer(self, nodes):
        count = len(nodes) // 2
        words = np.frombuffer(np.ascontiguousarray(nodes).tobytes(), dtype="<u8")
        out = self._run("layer", [[count], words])
        return np.frombuffer(out, dtype=np.uint8).reshape(count, 32).copy()

    def ntt_rows(self, rows, plan=None):
        """Natural-order NTT of each row via the generalized Bailey SMEM passes (`bailey_pass`)."""
        rows = np.atleast_2d(np.asarray(rows, dtype=np.uint64))
        n = rows.shape[1]
        if n < 2 or n & (n - 1):
            raise ValueError("NTT length must be a power of two >= 2")
        plan = check_plan(n.bit_length() - 1, plan or bailey_plan(n.bit_length() - 1))
        out = self._run("bailey", [[n.bit_length() - 1, rows.shape[0], len(plan)],
                                   [w for p in plan for w in p], rows, local_twiddles()])
        return np.frombuffer(out, dtype="<u8").reshape(rows.shape).astype(np.uint64)

    def ntt_rows_radix2(self, rows):
        """Legacy reference path: ntt_bitrev + log2(n) global ntt_stage launches."""
        rows = np.atleast_2d(rows)
        n = rows.shape[1]
        out = self._run("ntt", [[n.bit_length() - 1, rows.shape[0]], rows, stage_twiddles(n)])
        return np.frombuffer(out, dtype="<u8").reshape(rows.shape).astype(np.uint64)

    def ntt8192(self, message):
        out = self._run("ntt8192", [[len(message)], message, stage_twiddles(8192)])
        return np.frombuffer(out, dtype="<u8").astype(np.uint64)

    def mobius(self, cells):
        out = self._run("mobius", [[len(cells)], cells])
        return np.frombuffer(out, dtype="<u8").reshape(-1, 3).astype(np.uint64)

    def fraction_layer(self, nodes):
        """nodes: (2 pairs, 6) = (num, den) Fp3 pairs -> (pairs, 6)."""
        pairs = len(nodes) // 2
        out = self._run("frac", [[pairs], nodes])
        return np.frombuffer(out, dtype="<u8").reshape(pairs, 6).astype(np.uint64)

    def basefold_round(self, f, w, r):
        n = len(f)
        out = np.frombuffer(self._run("sumcheck", [[n], list(r), f, w]), dtype="<u8").astype(np.uint64)
        return out[:9].reshape(3, 3), out[9:].reshape(n // 2, 3)

    def halve(self, f, r):
        return self.basefold_round(f, f, r)[1]

    @staticmethod
    def dev(a):
        return np.asarray(a)

    @staticmethod
    def host(a):
        return np.asarray(a)

    @staticmethod
    def fadd(a, b):
        from flashprove_next.field import CPU
        return CPU.add(a, b)

    @staticmethod
    def fsub(a, b):
        from flashprove_next.field import CPU
        return CPU.sub(a, b)

    def sync(self):
        pass


class CudaEngine:
    """CuPy/NVRTC executor (`--std=c++17`); arrays stay on the device between kernels."""

    name = "cuda"

    def __init__(self, block=256):
        import cupy as cp  # noqa: PLC0415  (GPU-only dependency)
        self.cp = cp
        self.device_id = int(cp.cuda.runtime.getDevice())
        with open(KERNEL) as f:
            src = f.read()
        self.mod = cp.RawModule(code=src, options=("--std=c++17",))
        self.k = {name: self.mod.get_function(name) for name in (
            "ntt_bitrev", "ntt_stage", "mobius_stage", "fold4_fp3", "merkle_leaf_blake2b",
            "merkle_layer_blake2b", "fraction_layer", "basefold_round_terms", "fp3_pair_sum", "halve_fp3",
            "ntt8192", "bailey_pass", "gkr_round_terms")}
        self.block = block
        self._tw = {}
        pp = "0xFFFFFFFF00000001ULL"
        self._fadd = cp.ElementwiseKernel(
            "uint64 a, uint64 b", "uint64 c",
            f"unsigned long long r = a + b; c = (r < a || r >= {pp}) ? r - {pp} : r;", "fpn_gl_add")
        self._fsub = cp.ElementwiseKernel(
            "uint64 a, uint64 b", "uint64 c", f"c = a >= b ? a - b : a + ({pp} - b);", "fpn_gl_sub")

    def fadd(self, a, b):
        """Element-wise Goldilocks a + b on the device (same values as `CPU.add`)."""
        return self._fadd(self.dev(a), self.dev(b))

    def fsub(self, a, b):
        return self._fsub(self.dev(a), self.dev(b))

    def _grid(self, n, rows=1):
        return ((n + self.block - 1) // self.block, rows), (self.block,)

    def dev(self, a):
        return self.cp.asarray(a)

    def host(self, a):
        return self.cp.asnumpy(a)

    def twiddles(self, n):
        if n not in self._tw:
            self._tw[n] = self.cp.asarray(stage_twiddles(n))
        return self._tw[n]

    def local_twiddles(self):
        if "local" not in self._tw:
            self._tw["local"] = self.cp.asarray(local_twiddles())
        return self._tw["local"]

    def ntt_rows(self, rows, plan=None):
        """Generalized Bailey SMEM NTT: ceil(log2 n / 12) shared-memory passes (3 at n = 2^27)."""
        cp = self.cp
        rows = cp.ascontiguousarray(cp.atleast_2d(self.dev(rows)))
        r, n = rows.shape
        if n < 2 or n & (n - 1):
            raise ValueError("NTT length must be a power of two >= 2")
        log_n = n.bit_length() - 1
        out = cp.empty_like(rows)
        tw = self.local_twiddles()
        kernel = self.k["bailey_pass"]
        for i, (s, m, c) in enumerate(check_plan(log_n, plan or bailey_plan(log_n))):
            threads = min(self.block, 1 << (m + c - 1))
            kernel((n >> (m + c), r), (threads,),
                   (rows if i == 0 else out, out, tw, np.int32(log_n), np.int32(s), np.int32(m),
                    np.int32(c), np.int32(1 if i == 0 else 0)),
                   shared_mem=bailey_smem_bytes(m, c))
        return out

    def ntt_rows_radix2(self, rows):
        """Legacy reference path: ntt_bitrev + log2(n) global ntt_stage launches."""
        cp = self.cp
        rows = cp.ascontiguousarray(cp.atleast_2d(self.dev(rows)))
        r, n = rows.shape
        out = cp.empty_like(rows)
        g, b = self._grid(n, r)
        self.k["ntt_bitrev"](g, b, (rows, out, np.int32(n.bit_length() - 1), np.uint64(n)))
        tw = self.twiddles(n)
        g, b = self._grid(n // 2, r)
        half = 1
        while half < n:
            self.k["ntt_stage"](g, b, (out, tw, np.uint64(n), np.uint64(half)))
            half <<= 1
        return out

    def mobius(self, cells):
        data = self.cp.array(self.dev(cells))
        n = len(data)
        g, b = self._grid(n // 2)
        bit = 1
        while bit < n:
            self.k["mobius_stage"](g, b, (data, np.uint64(n), np.uint64(bit)))
            bit <<= 1
        return data

    def fold4(self, code, inv_omega, inv_iota, ra, rb):
        code = self.dev(code)
        quarter = len(code) // 4
        out = self.cp.empty((quarter, 3), dtype=np.uint64)
        ch = self.cp.asarray(np.array(list(ra) + list(rb), dtype=np.uint64))
        g, b = self._grid(quarter)
        self.k["fold4_fp3"](g, b, (code, out, ch, np.uint64(inv_omega), np.uint64(inv_iota), np.uint64(quarter)))
        return out

    def merkle_leaves(self, code):
        code = self.dev(code)
        quarter = len(code) // 4
        out = self.cp.empty((quarter, 32), dtype=np.uint8)
        g, b = self._grid(quarter)
        self.k["merkle_leaf_blake2b"](g, b, (code, out, np.uint64(quarter)))
        return out

    def merkle_layer(self, nodes):
        nodes = self.dev(nodes)
        count = len(nodes) // 2
        out = self.cp.empty((count, 32), dtype=np.uint8)
        g, b = self._grid(count)
        self.k["merkle_layer_blake2b"](g, b, (nodes, out, np.uint64(count)))
        return out

    def fraction_layer(self, nodes):
        nodes = self.dev(nodes)
        pairs = len(nodes) // 2
        out = self.cp.empty((pairs, 6), dtype=np.uint64)
        # Tiled kernel: one block per FRAC_TILE (= 128) pairs, staged through shared memory.
        g, b = ((pairs + FRAC_TILE - 1) // FRAC_TILE,), (FRAC_TILE,)
        self.k["fraction_layer"](g, b, (nodes, out, np.uint64(pairs)))
        return out

    def basefold_round(self, f, w, r):
        cp = self.cp
        f, w = self.dev(f), self.dev(w)
        pairs = len(f) // 2
        terms = cp.empty((3 * pairs, 3), dtype=np.uint64)
        g, b = self._grid(pairs)
        self.k["basefold_round_terms"](g, b, (f, w, terms, np.uint64(pairs)))
        half = pairs // 2
        while half >= 1:
            nxt = cp.empty((3 * half, 3), dtype=np.uint64)
            g, b = self._grid(3 * half)
            self.k["fp3_pair_sum"](g, b, (terms, nxt, np.uint64(half), np.uint64(3)))
            terms, half = nxt, half // 2
        out = cp.empty((pairs, 3), dtype=np.uint64)
        g, b = self._grid(pairs)
        self.k["halve_fp3"](g, b, (f, out, cp.asarray(np.array(r, dtype=np.uint64)), np.uint64(pairs)))
        return terms, out

    def gkr_round(self, cols, alpha):
        """FracLogUp-GKR layer-sumcheck round on device columns (w, nl, dl, nr, dr), each (2 pairs, 3): the four
        samples t = 0..3 as a (4, 3) host array (same values as `folded_air._gkr_round_np`)."""
        cp = self.cp
        pairs = len(cols[0]) // 2
        terms = cp.empty((4 * pairs, 3), dtype=np.uint64)
        g, b = self._grid(pairs)
        self.k["gkr_round_terms"](g, b, (*cols, cp.asarray(np.array(alpha, dtype=np.uint64)), terms,
                                         np.uint64(pairs)))
        half = pairs // 2
        while half >= 1:
            nxt = cp.empty((4 * half, 3), dtype=np.uint64)
            g, b = self._grid(4 * half)
            self.k["fp3_pair_sum"](g, b, (terms, nxt, np.uint64(half), np.uint64(4)))
            terms, half = nxt, half // 2
        return cp.asnumpy(terms).reshape(4, 3)

    def halve(self, f, r):
        f = self.dev(f)
        out = self.cp.empty((len(f) // 2, 3), dtype=np.uint64)
        g, b = self._grid(len(out))
        self.k["halve_fp3"](g, b, (f, out, self.cp.asarray(np.array(r, dtype=np.uint64)), np.uint64(len(out))))
        return out

    def ntt8192(self, message):
        cp = self.cp
        self.k["ntt8192"].max_dynamic_shared_size_bytes = 65536
        out = cp.empty(8192, dtype=np.uint64)
        self.k["ntt8192"]((1,), (256,), (self.dev(np.asarray(message, dtype=np.uint64)), out,
                                         self.twiddles(8192), np.int32(len(message))), shared_mem=65536)
        return out

    def sync(self):
        self.cp.cuda.Stream.null.synchronize()


def default_engine():
    try:
        return CudaEngine()
    except Exception:  # noqa: BLE001  (no GPU / no CuPy): fall back to the host emulation
        return HostEngine()


# ============================================================================ pipelines
def _host(engine, a):
    return engine.host(a) if hasattr(engine, "host") else np.asarray(a)


def xp_of(engine):
    """Array module of the engine's device arrays (CuPy for CudaEngine, NumPy for HostEngine)."""
    return getattr(engine, "cp", np)


def encode_codeword_dev(engine, cells):
    """Mobius transform + rate-1/8 RS encoding (three limb NTTs of length 8 * 2^n); the (L, 3) codeword stays on
    the engine's device (no host round trip)."""
    xp = xp_of(engine)
    n = len(cells).bit_length() - 1
    coeffs = engine.mobius(cells)
    rows = xp.zeros((3, RATE << n), dtype=np.uint64)
    rows[:, :len(coeffs)] = coeffs.T
    del coeffs
    ntt = engine.ntt_rows(rows)
    del rows
    out = xp.ascontiguousarray(ntt.T)
    del ntt
    return out


def encode_codeword(engine, cells):
    return np.ascontiguousarray(_host(engine, encode_codeword_dev(engine, cells)))


def merkle_levels_dev(engine, code):
    """Merkle levels of a codeword, kept on the engine's device (leaves first, root last)."""
    level = engine.merkle_leaves(code)
    levels = [level]
    while len(level) > 1:
        level = engine.merkle_layer(level)
        levels.append(level)
    return levels


def merkle_root(engine, code):
    return [_host(engine, lv) for lv in merkle_levels_dev(engine, code)]


def commit_pipeline_dev(engine, cells):
    """(device codeword (L, 3), device Merkle levels). Accepts either host or device (N, 3) uint64."""
    xp = xp_of(engine)
    if isinstance(cells, np.ndarray) or not hasattr(cells, "__cuda_array_interface__"):
        arr = np.asarray(cells, dtype=np.uint64).reshape(-1, 3)
    else:
        arr = xp.ascontiguousarray(cells, dtype=xp.uint64).reshape(-1, 3)
    code = encode_codeword_dev(engine, arr)
    return code, merkle_levels_dev(engine, code)


def commit_pipeline(engine, cells):
    """Returns (codeword (L, 3), Merkle levels); root = bytes(levels[-1][0])."""
    code, levels = commit_pipeline_dev(engine, cells)
    return np.ascontiguousarray(_host(engine, code)), [_host(engine, lv) for lv in levels]


def open_rows(engine, code, levels, positions):
    """Batched query openings of a committed codeword on the device: for each position p the 4 coset values
    code[p + k L/4] (k < 4, as (Q, 4, 3) host array) and the Merkle siblings levels[d][(p >> d) ^ 1] (a list over
    d of (Q, 32) host arrays) -- one gather + one device->host copy per level instead of one per value."""
    xp = xp_of(engine)
    pos = np.asarray(positions, dtype=np.int64)
    quarter = len(code) // 4
    idx = (pos[:, None] + np.arange(4, dtype=np.int64)[None, :] * quarter).reshape(-1)
    vals = np.asarray(_host(engine, code[xp.asarray(idx)])).reshape(len(pos), 4, 3)
    sibs = [np.asarray(_host(engine, levels[d][xp.asarray((pos >> d) ^ 1)])) for d in range(len(levels) - 1)]
    return vals, sibs


def fold_chain_dev(engine, code, n, challenges):
    """Device-resident fused coset-4 fold chain (zero intermediate host copies).

    Returns (layers_dev, trees_dev, roots_bytes) where layers_dev[t] is the folded codeword at layer t+1
    (t = 0 .. n//2 - 1), trees_dev[t] is the device Merkle levels for t < n//2 - 1, and roots_bytes[t] is
    the 32-byte Merkle root."""
    omega = root_of_unity(RATE << n)
    inv_iota = pow(root_of_unity(4), P - 2, P)
    layers, trees, roots, cur = [], [], [], code
    for t in range(n // 2):
        omega_t = pow(omega, 4 ** t, P)
        cur = engine.fold4(cur, pow(omega_t, P - 2, P), inv_iota, challenges[2 * t], challenges[2 * t + 1])
        layers.append(cur)
        if t < n // 2 - 1:
            lv = merkle_levels_dev(engine, cur)
            trees.append(lv)
            roots.append(bytes(_host(engine, lv[-1][0]).tobytes()))
    return layers, trees, roots


def fold_chain(engine, code, n, challenges):
    """Fused coset-4 folds of the committed codeword with Basefold challenges r_0..r_{n-1}.

    Returns the list of folded codewords (layers 1 .. n/2) and the Merkle levels of each committed
    folded layer (1 .. n/2 - 1)."""
    layers_dev, trees_dev, _ = fold_chain_dev(engine, code, n, challenges)
    return [_host(engine, c) for c in layers_dev], [[_host(engine, lv) for lv in tr] for tr in trees_dev]


def gkr_tree_dev(engine, leaves):
    """Device-resident FracLogUp fraction tree on (num, den) Fp3 leaves -> list of device (m, 6) arrays."""
    xp = xp_of(engine)
    if isinstance(leaves, np.ndarray) or not hasattr(leaves, "__cuda_array_interface__"):
        cur = engine.dev(np.asarray(leaves, dtype=np.uint64).reshape(-1, 6))
    else:
        cur = xp.ascontiguousarray(leaves, dtype=xp.uint64).reshape(-1, 6)
    tree = [cur]
    while len(cur) > 1:
        cur = engine.fraction_layer(cur)
        tree.append(cur)
    return tree


def gkr_tree(engine, leaves):
    """FracLogUp fraction tree on (num, den) Fp3 leaves -> list of layers as (m, 6) arrays."""
    return [_host(engine, lv) for lv in gkr_tree_dev(engine, leaves)]
