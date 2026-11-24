# SPDX-License-Identifier: Apache-2.0
"""Poseidon2 over Goldilocks, width 12, S-box x^7, R_F = 8, R_P = 22 (Plonky3 p3_goldilocks instance).

Independent re-implementation (specs/11-wire-format.md §3).  The scalar path is exact Python-int arithmetic and has
no dependency beyond the standard library; ``permute_batch`` is an optional NumPy-vectorised path over (12, n) uint64
arrays used for building Merkle trees.  Never imports the prover package.
"""
from .field import P

T = 12
RATE = 8
RF = 8
RP = 22
NR = RF + RP
TAG_NODE = 1
TAG_SPONGE = 2
DOM_LEAF = 64

M4 = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

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
assert len(RC) == NR and len(DIAG) == T
assert all(0 <= x < P for r in RC for x in r) and all(0 <= d < P for d in DIAG)

ME = [[(2 if i // 4 == j // 4 else 1) * M4[i % 4][j % 4] for j in range(T)] for i in range(T)]
MI_DIAG = [(d + 1) % P for d in DIAG]


def _ext(s):
    return [sum(ME[i][j] * s[j] for j in range(T)) % P for i in range(T)]


def _int(s):
    tot = sum(s)
    return [(tot + DIAG[i] * s[i]) % P for i in range(T)]


def permute(state):
    """state: 12 ints in [0, P). Returns the permuted 12-element list."""
    if len(state) != T:
        raise ValueError("Poseidon2 state must have 12 lanes")
    s = _ext([int(x) % P for x in state])
    for k in range(NR):
        if k < RF // 2 or k >= RF // 2 + RP:
            s = _ext([pow((s[i] + RC[k][i]) % P, 7, P) for i in range(T)])
        else:
            s = list(s)
            s[0] = pow((s[0] + RC[k][0]) % P, 7, P)
            s = _int(s)
    return s


def compress(left, right):
    st = list(left) + list(right) + [0, 0, 0, TAG_NODE]
    return tuple(permute(st)[0:4])


def sponge_blocks(blocks, domain):
    """poseidon2.py::sponge for one message: blocks is a list of 8-element lists."""
    st = [0] * T
    st[10] = domain % P
    st[11] = TAG_SPONGE
    for blk in blocks:
        if len(blk) != RATE:
            raise ValueError("sponge block must have 8 elements")
        st = [(st[i] + blk[i]) % P if i < RATE else st[i] for i in range(T)]
        st = permute(st)
    return tuple(st[0:4])


def sponge_msg(msg, domain):
    """Length-in-capacity sponge (lane 8 = len): zero-padded to max(1, ceil(len/8)) blocks."""
    msg = [int(x) % P for x in msg]
    nb = max(1, (len(msg) + 7) // 8)
    padded = msg + [0] * (8 * nb - len(msg))
    st = [0] * T
    st[8] = len(msg) % P
    st[10] = domain % P
    st[11] = TAG_SPONGE
    for b in range(nb):
        st = [(st[i] + padded[8 * b + i]) % P if i < RATE else st[i] for i in range(T)]
        st = permute(st)
    return tuple(st[0:4])


def leaf_digest(v):
    return sponge_msg([v], DOM_LEAF)


def merkle_root(leaves):
    """leaves: list of 2^n digests (4-tuples). Returns (root, levels)."""
    n = len(leaves)
    if n == 0 or n & (n - 1):
        raise ValueError("Merkle tree needs a power-of-two leaf count")
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        a = levels[-1]
        levels.append([compress(a[2 * i], a[2 * i + 1]) for i in range(len(a) // 2)])
    return levels[-1][0], levels


def merkle_path(levels, idx):
    out = []
    for d in range(len(levels) - 1):
        out.append(levels[d][idx ^ 1])
        idx >>= 1
    return out


def verify_path(root, leaf, idx, path):
    cur = tuple(leaf)
    for sib in path:
        cur = compress(sib, cur) if idx & 1 else compress(cur, sib)
        idx >>= 1
    return idx == 0 and cur == tuple(root)


# ----------------------------------------------------------------------------- optional NumPy batch path
def permute_batch(states):
    """NumPy-vectorised permutation of a (12, n) uint64 array (requires NumPy)."""
    import numpy as np
    pu = np.uint64(P)
    m32 = np.uint64(0xFFFFFFFF)
    s32 = np.uint64(32)

    def red(lo, hi):
        hh = hi >> s32
        hl = hi & m32
        t = lo - hh
        t = np.where(lo < hh, t - m32, t)
        r = t + hl * m32
        r = np.where(r < t, r + m32, r)
        return np.where(r >= pu, r - pu, r)

    def mul(a, b):
        al, ah = a & m32, a >> s32
        bl, bh = b & m32, b >> s32
        ll, lh, hl, hh = al * bl, al * bh, ah * bl, ah * bh
        mid = lh + hl
        c1 = (mid < lh).astype(np.uint64)
        lo = ll + (mid << s32)
        c2 = (lo < ll).astype(np.uint64)
        hi = hh + (mid >> s32) + (c1 << s32) + c2
        return red(lo, hi)

    def add(a, b):
        r = a + b
        return np.where((r < a) | (r >= pu), r - pu, r)

    def sbox(x):
        x2 = mul(x, x)
        x3 = mul(x2, x)
        return mul(mul(x3, x3), x)

    def matvec(M, s):
        out = []
        for i in range(T):
            acc = None
            for j in range(T):
                c = M[i][j]
                t = s[j] if c == 1 else mul(s[j], np.full_like(s[j], c))
                acc = t if acc is None else add(acc, t)
            out.append(acc)
        return np.stack(out)

    MI = [[MI_DIAG[i] if i == j else 1 for j in range(T)] for i in range(T)]
    with np.errstate(over="ignore"):
        s = np.asarray(states, dtype=np.uint64)
        if s.ndim != 2 or s.shape[0] != T:
            raise ValueError("Poseidon2 batch state must have shape (12, n)")
        s = matvec(ME, s)
        for k in range(NR):
            if k < RF // 2 or k >= RF // 2 + RP:
                rc = np.array(RC[k], dtype=np.uint64)[:, None]
                s = matvec(ME, sbox(add(s, np.broadcast_to(rc, s.shape))))
            else:
                s = s.copy()
                s[0] = sbox(add(s[0], np.full_like(s[0], RC[k][0])))
                s = matvec(MI, s)
    return s
