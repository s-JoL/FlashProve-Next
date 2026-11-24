# SPDX-License-Identifier: Apache-2.0
"""End-to-end prover of `FPN2` relation 3 (`FOLDED_GKR_LOOKUP`, specs/11-wire-format.md §5.4).

Statement: the prover knows a committed vector F (Merkle root `root_0` of its rate-1/8 codeword)
whose query block q (2^(s-2) cells) lies entirely in the public table T = [0, 2^(s-2)).  The proof
is a ZK-masked FracLogUp-GKR whose leaf claims and Libra-mask evaluations are all discharged by one
fused two-round ZK-Basefold opening of F.  Output is the exact fixed-size `FPN2` container.

F layout (n = s + 3 variables, Q4 = 2^(s-2)):
  * data subcube {bit1 = bit3 = 1} (2^(s+1) cells), aligned blocks in data coordinates:
        [0, Q4) q private queries      [Q4, 2Q4) m multiplicities m_t = #{j : q_j = t}
        [2Q4, 3Q4) N1 | E1, [3Q4, 4Q4) N3 | E3   dummy quarters (left/right tree half): leaves
                                       (+N_t, E_t) and (-N_t, E_t) paired across the top query bit
        [2^s, +2^(s-1)) A              additive mask numerators (paired +A / -A across top bit)
        [.., +2^(s-1)) D               additive mask denominators
  * Libra coefficients g_{k,i,e} in the subcube {bit1 = bit3 = 0};
  * every other cell is reserved uniform blinding randomness (non-doubling ZK of the PCS).
The blinding meets all four patterns of every index-bit pair, so it reaches every monomial class
of every folded layer; z_blind = sum_{b not in data} F(b) eq(2, b) joins the batch so that the PCS
sumcheck messages are masked too (rank condition: `folded_pcs.require_joint_mask_ext`).
"""
import secrets

from flashprove_next.field import ONE, P, ZERO, eq1, hadd, hf, hmul, hsub  # noqa: F401

from . import folded_pcs as pcs
from . import frac_logup_gkr as gkr
from .fpn2 import (
    FOLDED_GKR_V1, MASK_SUBCUBES, Fpn2Transcript, Layout, enc3, header, profile_id,
)

TWO = (2, 0, 0)
MASK_POINTS = ((2, 0, 0), (3, 0, 0))   # u(b) = eq(2, b) + eq(3, b) on the non-data cells


class SecretRng:
    def fp3(self):
        return tuple(secrets.randbelow(P) for _ in range(3))


def _nonzero(rng):
    while True:
        v = rng.fp3()
        if v != ZERO:
            return v


def eq_vec(point):
    """eq(point, b) for b in {0,1}^len(point), bit i of b <-> point[i]."""
    out = [ONE]
    for r in point:
        out = [hmul(w, f) for f in (hsub(ONE, r), r) for w in out]
    return out


def eq_pt(a, b):
    acc = ONE
    for x, y in zip(a, b):
        acc = hmul(acc, eq1(x, y))
    return acc


def bits_point(value, count):
    return tuple(ONE if value >> i & 1 else ZERO for i in range(count))


def build_witness(layout, queries, rng):
    s, q4 = layout.s, layout.quarter
    if len(queries) != q4:
        raise ValueError(f"relation 3 with n={layout.n} takes exactly {q4} queries")
    cells = [rng.fp3() for _ in range(1 << layout.n)]       # blinding + Libra by default
    counts = [0] * q4
    for j, v in enumerate(queries):
        cells[layout.cell(layout.off_q + j)] = hf(v)
        if 0 <= v < q4:
            counts[v] += 1
    h = q4 // 2
    for t in range(q4):
        cells[layout.cell(layout.off_m + t)] = hf(counts[t])
    for t in range(h):
        for off in (layout.off_e1, layout.off_e3):
            cells[layout.cell(off + t)] = rng.fp3()              # N (numerator mask)
            cells[layout.cell(off + h + t)] = _nonzero(rng)      # E (denominator)
    for v in range(1 << (s - 1)):
        cells[layout.cell(layout.off_a + v)] = rng.fp3()
        cells[layout.cell(layout.off_d + v)] = _nonzero(rng)
    return cells


def leaves(layout, cells, beta):
    """GKR leaves: real leaf j at 2j, mask leaf v at 2v + 1."""
    s, q4 = layout.s, layout.quarter
    h = q4 // 2
    get = lambda off: cells[layout.cell(off)]  # noqa: E731

    def dummy(off, t):
        num, den = get(off + t % h), get(off + h + t % h)
        return (num if t < h else hsub(ZERO, num), den)

    out = [None] * (2 << s)
    for j in range(1 << s):
        qd, t = j >> (s - 2), j & (q4 - 1)
        if qd == 0:
            leaf = (ONE, hsub(beta, get(layout.off_q + t)))
        elif qd == 1:
            leaf = dummy(layout.off_e1, t)
        elif qd == 2:
            leaf = (hsub(ZERO, get(layout.off_m + t)), hsub(beta, hf(t)))
        else:
            leaf = dummy(layout.off_e3, t)
        out[2 * j] = leaf
    half = 1 << (s - 1)
    for v in range(1 << s):
        a, d = get(layout.off_a + v % half), get(layout.off_d + v % half)
        out[2 * v + 1] = (a if v < half else hsub(ZERO, a), d)
    return out


def leaf_claims(layout, point, beta):
    """Public parts and committed-block terms of num(rho), den(rho) (spec §5.4.4)."""
    s = layout.s
    rho0, rho = point[0], point[1:]
    low, ra, lowp = rho[:s - 2], rho[:s - 1], rho[:s - 3]
    top = (rho[s - 2], rho[s - 1])
    half = layout.quarter // 2
    pair = hsub(ONE, hadd(rho[s - 3], rho[s - 3]))           # 1 - 2 * (top query bit)
    eqq = [eq_pt(top, bits_point(qd, 2)) for qd in range(4)]
    real = hsub(ONE, rho0)
    t_ext = ZERO
    for i, r in enumerate(low):
        t_ext = hadd(t_ext, hmul(hf(1 << i), r))
    num_pub = hmul(real, eqq[0])
    den_pub = hmul(real, hadd(hmul(eqq[0], beta), hmul(eqq[2], hsub(beta, t_ext))))
    blk = layout.data_block
    num_blocks = [(*blk(layout.off_m, s - 2), hsub(ZERO, hmul(real, eqq[2])), low),
                  (*blk(layout.off_e1, s - 3), hmul(hmul(real, eqq[1]), pair), lowp),
                  (*blk(layout.off_e3, s - 3), hmul(hmul(real, eqq[3]), pair), lowp),
                  (*blk(layout.off_a, s - 1), hmul(rho0, hsub(ONE, hadd(rho[s - 1], rho[s - 1]))), ra)]
    den_blocks = [(*blk(layout.off_q, s - 2), hsub(ZERO, hmul(real, eqq[0])), low),
                  (*blk(layout.off_e1 + half, s - 3), hmul(real, eqq[1]), lowp),
                  (*blk(layout.off_e3 + half, s - 3), hmul(real, eqq[3]), lowp),
                  (*blk(layout.off_d, s - 1), rho0, ra)]
    return num_pub, den_pub, num_blocks, den_blocks


def mask_blocks(layout, coef):
    """u(b) = eq(2, b) + eq(3, b) on the complement of the data subcube, as subcube blocks.

    Two product terms with distinct slopes make the blinding part of every degree-2 PCS round
    polynomial span all quadratics (a single product term would only span (3X - 1)(a + bX))."""
    out = []
    for z in MASK_POINTS:
        for fixed in MASK_SUBCUBES:
            c = coef
            for _, val in fixed:
                c = hmul(c, z if val else hsub(ONE, z))
            fixed_bits = {b for b, _ in fixed}
            free = tuple(b for b in range(layout.n) if b not in fixed_bits)
            out.append((tuple(fixed), free, c, tuple(z for _ in free)))
    return out


def batch_terms(layout, point, beta, libra_evals, mu):
    """Subcube blocks (fixed, free, coefficient, point) and sparse cells of the batched claim
    Y = y_num + mu y_den + sum_k mu^(k+1) g_r(k) + mu^(s+2) z_blind."""
    num_pub, den_pub, nb, db = leaf_claims(layout, point, beta)
    blocks = list(nb) + [(f, fr, hmul(mu, c), z) for f, fr, c, z in db]
    sparse = []
    coef = hmul(mu, mu)
    for k, rs, _ in libra_evals:
        for i, r in enumerate(rs):
            pw = ONE
            for e in range(4):
                sparse.append((layout.libra_slot(k, i, e), hmul(coef, pw)))
                pw = hmul(pw, r)
        coef = hmul(coef, mu)
    blocks += mask_blocks(layout, coef)
    return num_pub, den_pub, blocks, sparse


def z_blind(layout, cells):
    acc = ZERO
    for idx, c in enumerate(cells):
        if not layout.is_data(idx):
            for z in MASK_POINTS:
                u = ONE
                for i in range(layout.n):
                    u = hmul(u, z if idx >> i & 1 else hsub(ONE, z))
                acc = hadd(acc, hmul(u, c))
    return acc


def weights_vector(layout, blocks, sparse):
    w = [ZERO] * (1 << layout.n)
    for fixed, free, c, z in blocks:
        base = sum(val << bit for bit, val in fixed)
        for b, e in enumerate(eq_vec(z)):
            idx = base
            for i, bit in enumerate(free):
                idx |= ((b >> i) & 1) << bit
            w[idx] = hadd(w[idx], hmul(c, e))
    for idx, c in sparse:
        w[idx] = hadd(w[idx], c)
    return w


def weights_eval(layout, blocks, sparse, r):
    n = layout.n
    acc = ZERO
    for fixed, free, c, z in blocks:
        term = hmul(c, eq_pt(z, [r[b] for b in free]))
        for bit, val in fixed:
            term = hmul(term, r[bit] if val else hsub(ONE, r[bit]))
        acc = hadd(acc, term)
    for idx, c in sparse:
        acc = hadd(acc, hmul(c, eq_pt(bits_point(idx, n), r)))
    return acc


def prove(queries, n_s=b"\x00" * 32, layout=None, rng=None, hooks=None, profile=FOLDED_GKR_V1,
          witness=None):
    """Return the exact C_p-byte FPN2 container proving `queries` in [0, 2^(s-2))."""
    layout = layout or Layout(profile["n_vars"], profile["n_queries"])
    rng = rng or SecretRng()
    hooks = hooks or {}
    if len(n_s) != 32:
        raise ValueError("N_s must be 32 bytes")
    for _ in range(8):     # resample blinding if beta hits a denominator (probability ~ 2^-180)
        cells = witness(layout, queries, rng) if witness else build_witness(layout, queries, rng)
        com = pcs.zk_commit(cells)
        hdr = header(profile_id(profile))
        stmt = com.root + bytes(n_s)
        tr = Fpn2Transcript()
        tr.absorb_bytes(hdr)
        tr.absorb_bytes(stmt)
        beta = tr.squeeze_fp3()
        tree = gkr.build_tree(leaves(layout, cells, beta))
        root = tree[-1][0]
        if root[1] == ZERO and not hooks.get("allow_zero_root"):
            continue
        if "root" in hooks:
            root = hooks["root"](root)
        tr.absorb_bytes(enc3(root[0]) + enc3(root[1]))
        libra = lambda k, i: tuple(cells[layout.libra_slot(k, i, e)] for e in range(4))  # noqa: E731
        tree[-1][0] = root
        layers, point, claim, evals = gkr.masked_prove(tr, tree, libra, hooks.get("gkr"))
        zb = z_blind(layout, cells)
        tr.absorb_bytes(enc3(zb))
        mu = tr.squeeze_fp3()
        num_pub, den_pub, blocks, sparse = batch_terms(layout, point, beta, evals, mu)
        w = weights_vector(layout, blocks, sparse)
        opening, rs, positions = pcs.zk_open(com, w, tr, layout.queries, hooks.get("pcs"))
        if "record" in hooks:
            hooks["record"].update(cells=cells, weights=w, rs=rs, positions=positions, layout=layout)
        body = (stmt + enc3(root[0]) + enc3(root[1]) + b"".join(l.wire() for l in layers)
                + enc3(zb) + opening.wire())
        if len(body) != layout.body_used:
            raise AssertionError("layout mismatch")
        out = hdr + body
        if len(out) > profile["c_p"]:
            raise ValueError("proof exceeds C_p")
        return out + b"\x00" * (profile["c_p"] - len(out))
    raise RuntimeError("could not find a nonzero root denominator")


# ----------------------------------------------------------------------------- prover-side verifier
class Reject(Exception):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _decode(layout, body):
    from .fpn2 import dec3
    off = 0

    def take(k):
        nonlocal off
        piece = body[off:off + k]
        off += k
        return piece

    def f3():
        return dec3(take(24))

    stmt = take(64)
    root = (f3(), f3())
    layers = [gkr.MaskedLayer(None, (), tuple(f3() for _ in range(4)), None)]
    for k in range(1, layout.s + 1):
        g_sum = f3()
        rounds = tuple(tuple(f3() for _ in range(4)) for _ in range(k))
        children = tuple(f3() for _ in range(4))
        layers.append(gkr.MaskedLayer(g_sum, rounds, children, f3()))
    zb = f3()
    rounds, roots = [], []
    for t in range(layout.folds):
        rounds.append(tuple(f3() for _ in range(3)))
        rounds.append(tuple(f3() for _ in range(3)))
        if t < layout.folds - 1:
            roots.append(take(32))
    final = f3()
    qs = []
    for _ in range(layout.queries):
        vals, paths = [], []
        for t in range(layout.folds):
            vals.append(tuple(f3() for _ in range(4)))
            paths.append(tuple(take(32) for _ in range(layout.depth(t))))
        qs.append(pcs.QueryOpening(tuple(vals), tuple(paths)))
    return stmt, root, layers, zb, pcs.ZkOpening(tuple(rounds), tuple(roots), final, tuple(qs))


def verify(data, profile=FOLDED_GKR_V1, layout=None):
    """Prover-side replay (the normative verifiers live in core/ and verifier/python/)."""
    layout = layout or Layout(profile["n_vars"], profile["n_queries"])
    if len(data) != profile["c_p"] or data[:64] != header(profile_id(profile)):
        raise Reject("E_LENGTH")
    body = data[64:64 + layout.body_used]
    if any(data[64 + layout.body_used:]):
        raise Reject("E_PADDING")
    try:
        stmt, root, layers, zb, opening = _decode(layout, body)
    except ValueError as e:
        raise Reject("E_NONCANONICAL", str(e))
    tr = Fpn2Transcript()
    tr.absorb_bytes(data[:64])
    tr.absorb_bytes(stmt)
    beta = tr.squeeze_fp3()
    tr.absorb_bytes(enc3(root[0]) + enc3(root[1]))
    try:
        point, claim, evals = gkr.masked_verify(tr, root, layers)
    except gkr.GkrReject as e:
        raise Reject(e.code, str(e))
    tr.absorb_bytes(enc3(zb))
    mu = tr.squeeze_fp3()
    num_pub, den_pub, blocks, sparse = batch_terms(layout, point, beta, evals, mu)
    y = hadd(hsub(claim[0], num_pub), hmul(mu, hsub(claim[1], den_pub)))
    coef = hmul(mu, mu)
    for _, _, g_r in evals:
        y = hadd(y, hmul(coef, g_r))
        coef = hmul(coef, mu)
    y = hadd(y, hmul(coef, zb))
    try:
        pcs.zk_verify(stmt[:32], layout.n, y, lambda r: weights_eval(layout, blocks, sparse, r),
                      opening, tr, layout.queries)
    except pcs.PcsReject as e:
        raise Reject(e.code, str(e))
    return "VERIFIED"


def verify_result(data, **kw):
    try:
        return verify(data, **kw)
    except Reject as e:
        return e.code
