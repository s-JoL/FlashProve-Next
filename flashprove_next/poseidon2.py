# SPDX-License-Identifier: Apache-2.0
"""Poseidon2 over Goldilocks, width 12, S-box x^7, R_F = 8 full rounds, R_P = 22 partial rounds.

Instantiation: official audited Plonky3 (``p3_goldilocks`` / ``p3_poseidon2``) width-12 specification:
  * external layer M_E = circ(2 M4, M4, M4) with ``MDSMat4 = circ(2, 3, 1, 1)``, applied once before round 0 and at the
    end of every full round;
  * internal layer M_I = J + diag(DIAG) with ``MATRIX_DIAG_12_GOLDILOCKS`` (J = all-ones);
  * Grain-LFSR width-12 round constants (``RC_INITIAL``, ``RC_INTERNAL``, ``RC_FINAL``).

Used for: 2-to-1 Merkle compression (model_id directory, tile trees, trace root R), the tile sponge, the token
sponge and the keyed sampling PRF.  The same permutation is constrained in-circuit by flashprove_next.gadgets (one row per
permutation with columns I, S1..S4, x0..x21, S26, S27..S30).

Every function takes a back end ``B`` (flashprove_next.field.CPU or gpu_backend()) and operates on (12, batch) uint64 arrays.
"""
import numpy as np
from .field import P, CPU

T = 12
RATE = 8
RF = 8
RP = 22
NR = RF + RP
M4 = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
TAG_NODE = 1        # last lane of a 2-to-1 compression
TAG_SPONGE = 2      # capacity lane 11 of a sponge

DIAG = [
    0xfffffffeffffffff, 0x0000000000000001, 0x0000000000000002, 0x7fffffff80000001,
    0x0000000000000003, 0x0000000000000004, 0x7fffffff80000000, 0xfffffffefffffffe,
    0xfffffffefffffffd, 0xbfffffff40000001, 0x3fffffffc0000000, 0xdfffffff20000001,
]

RC_INITIAL = [
    [0x13dcf33aba214f46, 0x30b3b654a1da6d83, 0x1fc634ada6159b56, 0x937459964dc03466,
     0xedd2ef2ca7949924, 0xede9affde0e22f68, 0x8515b9d6bac9282d, 0x6b5c07b4e9e900d8,
     0x1ec66368838c8a08, 0x9042367d80d1fbab, 0x400283564a3c3799, 0x4a00be0466bca75e],
    [0x7913beee58e3817f, 0xf545e88532237d90, 0x22f8cb8736042005, 0x6f04990e247a2623,
     0xfe22e87ba37c38cd, 0xd20e32c85ffe2815, 0x117227674048fe73, 0x4e9fb7ea98a6b145,
     0xe0866c232b8af08b, 0x00bbc77916884964, 0x7031c0fb990d7116, 0x240a9e87cf35108f],
    [0x2e6363a5a12244b3, 0x5e1c3787d1b5011c, 0x4132660e2a196e8b, 0x3a013b648d3d4327,
     0xf79839f49888ea43, 0xfe85658ebafe1439, 0xb6889825a14240bd, 0x578453605541382b,
     0x4508cda8f6b63ce9, 0x9c3ef35848684c91, 0x0812bde23c87178c, 0xfe49638f7f722c14],
    [0x8e3f688ce885cbf5, 0xb8e110acf746a87d, 0xb4b2e8973a6dabef, 0x9e714c5da3d462ec,
     0x6438f9033d3d0c15, 0x24312f7cf1a27199, 0x23f843bb47acbf71, 0x9183f11a34be9f01,
     0x839062fbb9d45dbf, 0x24b56e7e6c2e43fa, 0xe1683da61c962a72, 0xa95c63971a19bfa7],
]

RC_INTERNAL = [
    0x4adf842aa75d4316, 0xf8fbb871aa4ab4eb, 0x68e85b6eb2dd6aeb, 0x07a0b06b2d270380,
    0xd94e0228bd282de4, 0x8bdd91d3250c5278, 0x209c68b88bba778f, 0xb5e18cdab77f3877,
    0xb296a3e808da93fa, 0x8370ecbda11a327e, 0x3f9075283775dad8, 0xb78095bb23c6aa84,
    0x3f36b9fe72ad4e5f, 0x69bc96780b10b553, 0x3f1d341f2eb7b881, 0x4e939e9815838818,
    0xda366b3ae2a31604, 0xbc89db1e7287d509, 0x6102f411f9ef5659, 0x58725c5e7ac1f0ab,
    0x0df5856c798883e7, 0xf7bb62a8da4c961b,
]

RC_FINAL = [
    [0xc68be7c94882a24d, 0xaf996d5d5cdaedd9, 0x9717f025e7daf6a5, 0x6436679e6e7216f4,
     0x8a223d99047af267, 0xbb512e35a133ba9a, 0xfbbf44097671aa03, 0xf04058ebf6811e61,
     0x5cca84703fac7ffb, 0x9b55c7945de6469f, 0x8e05bf09808e934f, 0x2ea900de876307d7],
    [0x7748fff2b38dfb89, 0x6b99a676dd3b5d81, 0xac4bb7c627cf7c13, 0xadb6ebe5e9e2f5ba,
     0x2d33378cafa24ae3, 0x1e5b73807543f8c2, 0x09208814bfebb10f, 0x782e64b6bb5b93dd,
     0xadd5a48eac90b50f, 0xadd4c54c736ea4b1, 0xd58dbb86ed817fd8, 0x6d5ed1a533f34ddd],
    [0x28686aa3e36b7cb9, 0x591abd3476689f36, 0x047d766678f13875, 0xa2a11112625f5b49,
     0x21fd10a3f8304958, 0xf9b40711443b0280, 0xd2697eb8b2bde88e, 0x3493790b51731b3f,
     0x11caf9dd73764023, 0x7acfb8f72878164e, 0x744ec4db23cefc26, 0x1e00e58f422c6340],
    [0x21dd28d906a62dda, 0xf32a46ab5f465b5f, 0xbfce13201f3f7e6b, 0xf30d2e7adb5304e2,
     0xecdf4ee4abad48e9, 0xf94e82182d395019, 0x4ee52e3744d887c5, 0xa1341c7cac0083b2,
     0x2302fb26c30c834a, 0xaea3c587273bf7d3, 0xf798e24961823ec7, 0x962deba3e9a2cd94],
]

RC = [list(r) for r in RC_INITIAL] + [[c] + [0] * (T - 1) for c in RC_INTERNAL] + [list(r) for r in RC_FINAL]
assert len(RC) == NR and len(DIAG) == T and all(0 <= x < P for r in RC for x in r) and all(0 <= d < P for d in DIAG)

ME = [[(2 if i // 4 == j // 4 else 1) * M4[i % 4][j % 4] for j in range(T)] for i in range(T)]
MI = [[(DIAG[i] + 1) % P if i == j else 1 for j in range(T)] for i in range(T)]


def _matvec(B, M, s):
    xp = B.xp
    out = []
    for i in range(T):
        acc = None
        for j in range(T):
            if M[i][j] == 0:
                continue
            t = s[j] if M[i][j] == 1 else B.mul(s[j], np.uint64(M[i][j]))
            acc = t if acc is None else B.add(acc, t)
        out.append(acc)
    return xp.stack(out)


def _sbox(B, x):
    x2 = B.mul(x, x); x3 = B.mul(x2, x); x6 = B.mul(x3, x3)
    return B.mul(x6, x)


def _addc(B, s, k):
    xp = B.xp
    return xp.stack([B.add(s[i], np.uint64(RC[k][i])) if RC[k][i] else s[i] for i in range(T)])


_RC_FLAT = np.array([RC[k][i] for k in range(NR) for i in range(T)], dtype=np.uint64)
_DIAG = np.array(DIAG, dtype=np.uint64)
_TRACE_ROWS = {"S1": 0, "S2": 12, "S3": 24, "S4": 36, "S26": 70, "S27": 82, "S28": 94, "S29": 106, "S30": 118}
_NAT_INIT = []


def _cols_from_trace(state, tr, xp):
    cols = {"I": state}
    for k, r in _TRACE_ROWS.items():
        cols[k] = tr[r:r + 12]
    cols["x"] = tr[48:70]
    return cols


def permute(state, B=CPU, trace=False):
    """state: (12, n) uint64.  Returns the output (12, n); with trace=True also the circuit columns.
    Fast paths: fused CUDA kernel (GPU back end), native C++ (CPU back end, flashprove_next/_fast_cpu.so); NumPy otherwise."""
    st_check = B.xp.asarray(state, dtype=np.uint64)
    if st_check.ndim != 2 or st_check.shape[0] != T:
        raise ValueError(f"Poseidon2 state must have shape ({T}, n), got {st_check.shape}")
    if getattr(B, "name", "") == "cupy" and hasattr(B, "p2_permute"):
        st = B.xp.ascontiguousarray(st_check)
        out, tr = B.p2_permute(st, _RC_FLAT, _DIAG, trace)
        return (out, _cols_from_trace(st, tr, B.xp)) if trace else out
    if B is CPU:
        from .field import native, ptr
        lib = native()
        if lib is not None:
            if not _NAT_INIT:
                lib.p2_init(ptr(_RC_FLAT), ptr(_DIAG)); _NAT_INIT.append(1)
            st0 = np.ascontiguousarray(st_check, dtype=np.uint64)
            st = st0.copy(); n = st.shape[1]
            tr = np.empty((130, n), dtype=np.uint64) if trace else None
            if n >= _PAR_MIN:
                _permute_native_par(lib, st, tr)          # column chunks on worker threads (ctypes drops the GIL)
            elif n:
                lib.p2_permute(ptr(st), n, ptr(tr) if trace else None)
            return (st, _cols_from_trace(st0, tr, np)) if trace else st
    return _permute_np(state, B, trace)


_PAR_MIN = 1 << 14
_POOL = []


def _permute_native_par(lib, st, tr):
    """st (12, n) permuted in place (tr (130, n) filled) by the native kernel on column chunks in parallel threads."""
    from .field import ptr
    if not _POOL:
        from concurrent.futures import ThreadPoolExecutor
        import os
        _POOL.append(ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 4))))
    n = st.shape[1]
    w = max(_PAR_MIN // 4, (n + 4 * _POOL[0]._max_workers - 1) // (4 * _POOL[0]._max_workers))

    def job(a0):
        a1 = min(n, a0 + w)
        sc = np.ascontiguousarray(st[:, a0:a1])
        tc = np.empty((130, a1 - a0), dtype=np.uint64) if tr is not None else None
        lib.p2_permute(ptr(sc), a1 - a0, ptr(tc) if tc is not None else None)
        st[:, a0:a1] = sc
        if tc is not None:
            tr[:, a0:a1] = tc
    list(_POOL[0].map(job, range(0, n, w)))


def _permute_np(state, B=CPU, trace=False):
    xp = B.xp
    s = _matvec(B, ME, xp.asarray(state, dtype=np.uint64))
    cols = {"I": xp.asarray(state, dtype=np.uint64)} if trace else None
    xs = []
    for k in range(NR):
        s = _addc(B, s, k)
        if k < RF // 2 or k >= RF // 2 + RP:
            s = _matvec(B, ME, xp.stack([_sbox(B, s[i]) for i in range(T)]))
        else:
            xs.append(s[0])
            s = _matvec(B, MI, xp.concatenate([_sbox(B, s[0])[None], s[1:]]))
        if trace and (k + 1) in (1, 2, 3, 4, 26, 27, 28, 29, 30):
            cols["S%d" % (k + 1)] = s
    if trace:
        cols["x"] = xp.stack(xs)
        return s, cols
    return s


# ----------------------------------------------------------------------------- symbolic linear tracking (for the circuit)
def partial_round_linear_forms():
    """Express every partial-round S-box input x_j and the state S26 as affine forms in
    (S4[0..11], y_0..y_21, 1), where y_j = x_j^7.  Returns (xs, s26): lists of dicts {('S', i) | ('y', j) | 1: coef}."""
    st = [{("S", i): 1} for i in range(T)]
    xs = []
    for j in range(RP):
        k = RF // 2 + j
        lane0 = dict(st[0]); lane0[1] = (lane0.get(1, 0) + RC[k][0]) % P
        xs.append(lane0)
        vec = [{("y", j): 1}] + st[1:]
        new = []
        for i in range(T):
            acc = {}
            for jj in range(T):
                c = MI[i][jj]
                for key, v in vec[jj].items():
                    acc[key] = (acc.get(key, 0) + c * v) % P
            new.append({k2: v for k2, v in acc.items() if v})
        st = new
    return xs, st


# ----------------------------------------------------------------------------- hashing helpers
def compress(left, right, B=CPU):
    """2-to-1 compression: left, right are (4, n) arrays. Returns (4, n)."""
    xp = B.xp
    n = left.shape[1]
    st = xp.zeros((T, n), dtype=np.uint64)
    st[0:4] = left; st[4:8] = right; st[11] = TAG_NODE
    return permute(st, B)[0:4]


def compress1(l, r):
    """Host convenience: l, r are 4-tuples of ints."""
    out = compress(np.array(l, dtype=np.uint64)[:, None], np.array(r, dtype=np.uint64)[:, None])
    return tuple(int(v) for v in out[:, 0])


def sponge(blocks, B=CPU, domain=0, return_states=False):
    """Sponge over rows of 8 elements.  blocks: (nblk, 8, n) array (n independent messages of nblk blocks each).
    Capacity init: lane 10 = domain, lane 11 = TAG_SPONGE.  Returns digests (4, n) (and per-block inputs/outputs)."""
    xp = B.xp
    nblk, _, n = blocks.shape
    st = xp.zeros((T, n), dtype=np.uint64)
    st[10] = np.uint64(domain % P); st[11] = TAG_SPONGE
    ins, outs = [], []
    for b in range(nblk):
        st = st.copy()
        st[0:8] = xp.stack([B.add(st[i], blocks[b, i]) for i in range(8)])
        if return_states:
            ins.append(st)
        st = permute(st, B)
        if return_states:
            outs.append(st)
    if return_states:
        return st[0:4], ins, outs
    return st[0:4]


def digest_bytes(d):
    return b"".join(int(v).to_bytes(8, "little") for v in d)


def bytes_digest(b):
    assert len(b) == 32
    return tuple(int.from_bytes(b[8 * i:8 * i + 8], "little") for i in range(4))


# ----------------------------------------------------------------------------- sparse Merkle tree of fixed depth
class SparseMerkle:
    """Fixed-depth binary Poseidon2 tree over 4-element leaves; absent leaves are the zero digest.
    ``leaves``: dict index -> 4-tuple.  Computes only the populated part (default nodes are precomputed)."""

    def __init__(self, depth, leaves, B=CPU):
        self.depth = depth
        z = [(0, 0, 0, 0)]
        for _ in range(depth):
            z.append(compress1(z[-1], z[-1]))
        self.zero = z
        level = dict(leaves)
        self.levels = [level]
        for d in range(depth):
            parents = sorted({i >> 1 for i in level})
            if not parents:
                self.levels.append({}); level = {}; continue
            L = np.array([level.get(2 * p, z[d]) for p in parents], dtype=np.uint64).T
            R = np.array([level.get(2 * p + 1, z[d]) for p in parents], dtype=np.uint64).T
            if B is not CPU:
                out = B.tonumpy(compress(B.asarray(L), B.asarray(R), B))
            else:
                out = compress(L, R, B)
            level = {p: tuple(int(v) for v in out[:, k]) for k, p in enumerate(parents)}
            self.levels.append(level)
        self.root = self.levels[-1].get(0, z[depth])

    def path(self, idx):
        """Sibling digests from the leaf level upwards."""
        out = []
        for d in range(self.depth):
            out.append(self.levels[d].get(idx ^ 1, self.zero[d])); idx >>= 1
        return out


def verify_path(root, leaf, idx, path):
    cur = leaf
    for sib in path:
        cur = compress1(sib, cur) if idx & 1 else compress1(cur, sib)
        idx >>= 1
    return cur == tuple(root)


def merkle_root_dense(leaves, B=CPU):
    """leaves: (4, n) with n a power of two. Returns the root (4-tuple) and all levels."""
    lv = [leaves]
    while lv[-1].shape[1] > 1:
        a = lv[-1]
        lv.append(compress(a[:, 0::2], a[:, 1::2], B))
    r = B.tonumpy(lv[-1])
    return tuple(int(v) for v in r[:, 0]), lv
