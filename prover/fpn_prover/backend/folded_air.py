# SPDX-License-Identifier: Apache-2.0
"""End-to-end prover of `FPN2` relation 4 (`FOLDED_GKR_AIR`, specs/11-wire-format.md §5.5).

Statement (public: the circuit commitment `root_circ`, pinned by the verifier's caller): the prover
knows a variable vector V (K = 2^g entries, g = n - 5) such that, for the public Plonkish circuit C
committed under `root_circ`,

  (1) every gate row x < K satisfies
          q_M(x) w_a(x) w_b(x) + q_L(x) w_a(x) + q_R(x) w_b(x) + q_O(x) w_c(x) + q_C(x) = 0
      over F_p, where w_col(x) = V[sigma_col(x) - 1] (copy constraints / wiring);
  (2) every variable flagged by the circuit lies in the fixed limb table [0, 128).

One committed private vector F (Merkle root `root_0`) holds the gate columns w_a, w_b, w_c, the variable
vector V, the range multiplicities m, dummy fractions (N, E), additive GKR masks (A, D), every Libra
coefficient and reserved blinding.  The public circuit vector C (deterministic, no blinding, root
`root_circ`) holds q_M, q_L, q_R, q_O, q_C, sigma_a, sigma_b, sigma_c, the public variable
multiplicities `mult` and range flags `flag`.

  * Wiring + range: ONE ZK-masked FracLogUp-GKR tree (relation-3 machinery, `frac_logup_gkr`) over
        sum_cells 1/(beta - w - gamma sigma) - sum_k mult_k/(beta - V_k - gamma (k+1))
      + sum_k flag_k/(beta - V_k) - sum_t m_t/(beta - (t mod 128))  (+ cancelling dummy/mask leaves)
    with root_num == 0 and root_den != 0.  Wiring poles sit at beta = v + gamma*id with id >= 1, range
    poles at beta = v, so the two sub-sums vanish separately (two independent challenges beta, gamma).
  * Gates: a Libra-masked zerocheck sumcheck of sum_x eq(z, x) Gate(x) (gate degree 3, round degree 4).
  * Every resulting linear claim on F and C is discharged by ONE batched folded ZK-Basefold opening of
    the pair (F, C): two batching sumcheck rounds over G(x0, x1, y) = F(y) + x0 C(y) (r_0 is the
    batching coefficient, squeezed after both roots), then the fused coset-4 Basefold of
    H = F + r_0 C whose layer 0 is read from the two Merkle trees.
"""
import secrets
from dataclasses import dataclass

from flashprove_next.field import ONE, P, ZERO, eq1, hadd, hf, hinv, hmul, hsub, lagrange_eval

from . import folded_pcs as pcs
from . import frac_logup_gkr as gkr
from .fpn2 import (
    FOLDED_AIR_V1, Fpn2Transcript, deposit, enc3, enc3s, header, leaf_hash, merkle_levels, merkle_path,
    profile_id,
)

TABLE = 128                       # fixed limb table [0, 128) for every n
TABLE_BITS = 7
DATA_FIXED = ((1, 1),)            # data half of F / C: index bit 1 == 1
LIBRA_FIXED = ((1, 0),)           # Libra + blinding half: index bit 1 == 0
MASK_POINTS = ((2, 0, 0), (3, 0, 0))
ZC_DEGREE = 4                     # zerocheck round degree: eq * q_M * w_a * w_b
N_FINALS = 9                      # e_a, e_b, e_c, e_M, e_L, e_R, e_O, e_C, g_r
ZK_ROWS = 3                       # zero-selector gate rows wired to fresh uniform F_p variables (circuit.lower)


def free_bits(n):
    return [0] + list(range(2, n))


# ----------------------------------------------------------------------------- layout (§5.5.2)
@dataclass(frozen=True)
class AirLayout:
    """Relation-4 geometry, a pure function of (n, Q).

    Data coordinates (n - 1 bits -> cell bits 0, 2, 3, ..., n-1 of the half {bit1 = 1}), blocks of K:
        F: [0,K) w_a  [K,2K) w_b  [2K,3K) w_c  [3K,4K) V  [4K,5K) m  [5K,6K) N  [6K,7K) E  [7K,8K) spare
           [8K,12K) A  [12K,16K) D
        C: q_M q_L q_R q_O q_C sigma_a sigma_b sigma_c mult flag (blocks 0..9), everything else 0.
    """
    n: int
    queries: int

    def __post_init__(self):
        if self.n < 12 or self.n % 2 or self.n > 28 or not 1 <= self.queries <= 4096:
            raise ValueError("relation 4 needs even n in [12, 28] and 1 <= Q <= 4096")

    @property
    def g(self): return self.n - 5

    @property
    def K(self): return 1 << self.g

    @property
    def S(self): return self.g + 3          # real-leaf index bits (g row bits + 3 block bits)

    @property
    def folds(self): return self.n // 2

    def depth(self, t): return self.n + 1 - 2 * t

    # F blocks (data coordinates)
    W_A, W_B, W_C, V, M, N, E = range(7)

    def off(self, blk): return blk * self.K

    @property
    def off_a(self): return 8 * self.K

    @property
    def off_d(self): return 12 * self.K

    # C blocks
    QM, QL, QR, QO, QC, SA, SB, SC, MULT, FLAG = range(10)

    def cell(self, data_index):
        return deposit(data_index, DATA_FIXED, free_bits(self.n))

    def data_block(self, off, k):
        fb = free_bits(self.n)
        fixed = DATA_FIXED + tuple((fb[k + i], (off >> (k + i)) & 1) for i in range(len(fb) - k))
        return fixed, tuple(fb[:k])

    def libra_slot(self, k, i, e):
        return deposit(2 * k * (k - 1) + 4 * i + e, LIBRA_FIXED, free_bits(self.n))

    def zc_slot(self, i, e):
        return deposit(2 * self.S * (self.S + 1) + (ZC_DEGREE + 1) * i + e, LIBRA_FIXED, free_bits(self.n))

    def is_data(self, idx):
        return (idx >> 1) & 1

    # wire sizes (§5.5.7)
    STMT = 96

    def gkr_layer_len(self, k):
        return 24 + 96 * k + 120

    @property
    def gkr_len(self):
        return 48 + 96 + sum(self.gkr_layer_len(k) for k in range(1, self.S + 1))

    @property
    def zc_len(self):
        return 24 + self.g * 24 * (ZC_DEGREE + 1) + 24 * N_FINALS

    @property
    def pcs_len(self):
        return 24 + (self.n + 2) * 72 + 32 * (self.folds - 1) + 24

    @property
    def query_len(self):
        return (96 + 32 * self.depth(0)) + sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return self.STMT + self.gkr_len + self.zc_len + self.pcs_len + self.queries * self.query_len


def layout_of(profile):
    return AirLayout(profile["n_vars"], profile["n_queries"])


# ----------------------------------------------------------------------------- public circuit table
@dataclass(frozen=True)
class AirTable:
    """Public Plonkish circuit of exactly K rows / K variables (canonical F_p ints).

    rows[x] = (q_M, q_L, q_R, q_O, q_C, sigma_a, sigma_b, sigma_c), sigma = variable id in [1, K]
    (variable index + 1); mult[k] = number of gate cells wired to variable k; flag[k] in {0, 1}
    marks the variables range-checked against [0, 128)."""
    rows: tuple
    mult: tuple
    flag: tuple

    def check(self, K):
        if len(self.rows) != K or len(self.mult) != K or len(self.flag) != K:
            raise ValueError("circuit table must have exactly K rows and K variables")
        counts = [0] * K
        for row in self.rows:
            if len(row) != 8 or any(type(v) is not int or not 0 <= v < P for v in row):
                raise ValueError("bad circuit row")
            for sig in row[5:]:
                if not 1 <= sig <= K:
                    raise ValueError("wire id out of range")
                counts[sig - 1] += 1
        if tuple(counts) != tuple(self.mult) or any(f not in (0, 1) for f in self.flag):
            raise ValueError("variable multiplicities / flags inconsistent with the wiring")


def circuit_cells(layout, table):
    cells = [ZERO] * (1 << layout.n)
    K = layout.K
    for x, row in enumerate(table.rows):
        for blk, v in enumerate(row):                 # QM..QC, SA..SC are blocks 0..7 in row order
            cells[layout.cell(blk * K + x)] = hf(v)
    for k in range(K):
        cells[layout.cell(layout.MULT * K + k)] = hf(table.mult[k])
        cells[layout.cell(layout.FLAG * K + k)] = hf(table.flag[k])
    return cells


_CIRCUIT_CACHE = {}
_ENGINE_CIRCUIT_CACHE = {}      # array-backed EngineCommitment, never mixed with the tuple-backed CPU cache


def commit_circuit(layout, table, engine=None):
    """Deterministic commitment of the public circuit vector C (cached per (n, table)).

    engine=None: the CPU reference (`folded_pcs.zk_commit`, tuple-backed `Commitment`).  With a
    `cuda_engine.CudaEngine` / `HostEngine`: the same codeword and Merkle tree built by the kernels, returned as an
    array-backed `EngineCommitment` (identical root)."""
    if engine is not None:
        key = (layout.n, table, type(engine).__name__, getattr(engine, "device_id", None))     # never share across devices
        if key not in _ENGINE_CIRCUIT_CACHE:
            table.check(layout.K)
            if len(_ENGINE_CIRCUIT_CACHE) >= 8:                # bounded device memory
                _ENGINE_CIRCUIT_CACHE.clear()
            _ENGINE_CIRCUIT_CACHE[key] = engine_commit(engine, circuit_cells_np(layout, table))
        return _ENGINE_CIRCUIT_CACHE[key]
    key = (layout.n, table)
    if key not in _CIRCUIT_CACHE:
        table.check(layout.K)
        cells = circuit_cells(layout, table)
        _CIRCUIT_CACHE[key] = pcs.zk_commit(cells)
    return _CIRCUIT_CACHE[key]


def circuit_root(layout, table, engine=None):
    return commit_circuit(layout, table, engine).root


# ----------------------------------------------------------------------------- private witness F
def _os_rand_field(n):
    """Rejection-sampled uniform F_p words exclusively from OS CSPRNG (never honors FPN_TEST_RNG_SEED)."""
    np = _np()
    out = np.frombuffer(secrets.token_bytes(8 * n), dtype="<u8").copy()
    bad = np.flatnonzero(out >= P)
    while len(bad):
        out[bad] = np.frombuffer(secrets.token_bytes(8 * len(bad)), dtype="<u8")
        bad = bad[out[bad] >= P]
    return out


class SecretRng:
    """OS randomness.  `fp3()` draws one uniform F_p3 cell; `fp3_array(count)` draws `count` cells in bulk
    exclusively via rejection-sampled `os.urandom` words (immune to `FPN_TEST_RNG_SEED`).  A custom rng without
    `fp3_array` (seeded test rngs) keeps the exact per-cell draw order of the reference prover."""

    def fp3(self):
        return tuple(secrets.randbelow(P) for _ in range(3))

    def fp3_array(self, count):
        return _os_rand_field(3 * count).reshape(count, 3)


def draw_cells(rng, count, nonzero=False):
    """`count` uniform F_p3 cells as a (count, 3) uint64 array via `rng.fp3_array` (bulk rngs only); `nonzero`
    redraws all-zero cells (uniform on F_p3 minus {0}, as `_nonzero`)."""
    np = _np()
    out = np.ascontiguousarray(np.asarray(rng.fp3_array(count), dtype=np.uint64).reshape(count, 3))
    if nonzero:
        bad = np.flatnonzero(~out.any(axis=1))
        while len(bad):
            out[bad] = np.asarray(rng.fp3_array(len(bad)), dtype=np.uint64).reshape(len(bad), 3)
            bad = bad[~out[bad].any(axis=1)]
    return out


def _nonzero(rng):
    while True:
        v = rng.fp3()
        if v != ZERO:
            return v


def columns_of(table, V):
    """Honest gate columns w_col(x) = V[sigma_col(x) - 1]."""
    return [[V[row[5 + c] - 1] for row in table.rows] for c in range(3)]


def range_multiplicities(table, V):
    counts = [0] * TABLE
    for k, f in enumerate(table.flag):
        if f:
            if not 0 <= V[k] < TABLE:
                raise ValueError(f"variable {k} = {V[k]} outside [0, {TABLE})")
            counts[V[k]] += 1
    return counts


def build_witness(layout, table, V, rng, columns=None, m=None):
    """F cells.  `columns` / `m` override the honest gate columns / multiplicities (forgery tests)."""
    K = layout.K
    if len(V) != K or any(type(v) is not int or not 0 <= v < P for v in V):
        raise ValueError("V must hold K canonical F_p values")
    cols = columns if columns is not None else columns_of(table, V)
    mult = m if m is not None else range_multiplicities(table, V) + [0] * (K - TABLE)
    cells = [rng.fp3() for _ in range(1 << layout.n)]       # blinding / Libra / spare by default
    put = lambda blk, x, v: cells.__setitem__(layout.cell(blk * K + x), v)  # noqa: E731
    for x in range(K):
        for c in range(3):
            put(layout.W_A + c, x, hf(cols[c][x]))
        put(layout.V, x, hf(V[x]))
        put(layout.M, x, hf(mult[x]))
        put(layout.N, x, rng.fp3())
        put(layout.E, x, _nonzero(rng))
    for v in range(4 * K):
        cells[layout.cell(layout.off_a + v)] = rng.fp3()
        cells[layout.cell(layout.off_d + v)] = _nonzero(rng)
    return cells


def _get(layout, cells, blk, x):
    return cells[layout.cell(blk * layout.K + x)]


def leaves(layout, F, C, beta, gamma):
    """GKR leaves: real leaf j = blk * K + x at 2j, mask leaf v at 2v + 1 (§5.5.3)."""
    K, S = layout.K, layout.S
    out = [None] * (2 << S)
    for x in range(K):
        X = hf(x + 1)
        for c in range(3):
            w = _get(layout, F, layout.W_A + c, x)
            sig = _get(layout, C, layout.SA + c, x)
            out[2 * (c * K + x)] = (ONE, hsub(hsub(beta, w), hmul(gamma, sig)))
        v = _get(layout, F, layout.V, x)
        out[2 * (3 * K + x)] = (hsub(ZERO, _get(layout, C, layout.MULT, x)), hsub(hsub(beta, v), hmul(gamma, X)))
        out[2 * (4 * K + x)] = (_get(layout, C, layout.FLAG, x), hsub(beta, v))
        out[2 * (5 * K + x)] = (hsub(ZERO, _get(layout, F, layout.M, x)), hsub(beta, hf(x % TABLE)))
        nn, ee = _get(layout, F, layout.N, x), _get(layout, F, layout.E, x)
        out[2 * (6 * K + x)] = (nn, ee)
        out[2 * (7 * K + x)] = (hsub(ZERO, nn), ee)
    half = 1 << (S - 1)
    for v in range(1 << S):
        a = F[layout.cell(layout.off_a + v % half)]
        d = F[layout.cell(layout.off_d + v % half)]
        out[2 * v + 1] = (a if v < half else hsub(ZERO, a), d)
    return out


# ----------------------------------------------------------------------------- MLE helpers
def eq_vec(point):
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


def _scal(c, a):
    return tuple(c * x % P for x in a)


def int_ext(point, count=None):
    """MLE of b -> sum_i 2^i b_i over the first `count` coordinates of `point`."""
    acc = ZERO
    for i, r in enumerate(point[:count] if count is not None else point):
        acc = hadd(acc, _scal(1 << i, r))
    return acc


# ----------------------------------------------------------------------------- batched linear claim (§5.5.5)
def leaf_terms(layout, point, beta, gamma, mu):
    """(num_pub, den_pub, F-blocks, C-blocks) of Y_leaf = (num - num_pub) + mu (den - den_pub)."""
    g, S, K = layout.g, layout.S, layout.K
    rho0, rho = point[0], point[1:]
    low, top, ra = rho[:g], rho[g:g + 3], rho[:S - 1]
    eqb = [eq_pt(top, bits_point(k, 3)) for k in range(8)]
    real = hsub(ONE, rho0)
    neg = lambda x: hsub(ZERO, x)  # noqa: E731
    x_ext = int_ext(low)
    t_ext = int_ext(low, TABLE_BITS)
    num_pub = hmul(real, hadd(hadd(eqb[0], eqb[1]), eqb[2]))
    s05 = ZERO
    for k in range(6):
        s05 = hadd(s05, eqb[k])
    den_pub = hmul(real, hsub(hsub(hmul(beta, s05), hmul(hmul(gamma, eqb[3]), hadd(ONE, x_ext))),
                              hmul(eqb[5], t_ext)))
    blk = layout.data_block
    fb = [(*blk(layout.off(layout.M), g), neg(hmul(real, eqb[5])), low),
          (*blk(layout.off(layout.N), g), hmul(real, hsub(eqb[6], eqb[7])), low),
          (*blk(layout.off_a, S - 1), hmul(rho0, hsub(ONE, hadd(rho[S - 1], rho[S - 1]))), ra)]
    for c in range(3):
        fb.append((*blk(layout.off(layout.W_A + c), g), hmul(mu, neg(hmul(real, eqb[c]))), low))
    fb += [(*blk(layout.off(layout.V), g), hmul(mu, neg(hmul(real, hadd(eqb[3], eqb[4])))), low),
           (*blk(layout.off(layout.E), g), hmul(mu, hmul(real, hadd(eqb[6], eqb[7]))), low),
           (*blk(layout.off_d, S - 1), hmul(mu, rho0), ra)]
    cb = [(*blk(layout.off(layout.MULT), g), neg(hmul(real, eqb[3])), low),
          (*blk(layout.off(layout.FLAG), g), hmul(real, eqb[4]), low)]
    for c in range(3):
        cb.append((*blk(layout.off(layout.SA + c), g), hmul(mu, neg(hmul(hmul(real, gamma), eqb[c]))), low))
    return num_pub, den_pub, fb, cb


def mask_blocks(layout, coef):
    out = []
    for z in MASK_POINTS:
        free = tuple(b for b in range(layout.n) if b != 1)
        out.append((((1, 0),), free, hmul(coef, hsub(ONE, z)), tuple(z for _ in free)))
    return out


def batch_terms(layout, point, beta, gamma, gkr_evals, zc_point, zc_finals, mu):
    """Weights of Y = Y_leaf + sum_k mu^(k+1) g_r(k) + sum_j c_j zerocheck_j + c_z z_blind.

    Returns (y_pub_correction, F-blocks, F-sparse, C-blocks, coefficients) where the verifier's Y is
    (num - num_pub) + mu (den - den_pub) + sum coef * claimed value."""
    g = layout.g
    num_pub, den_pub, fb, cb = leaf_terms(layout, point, beta, gamma, mu)
    sparse = []
    coef = hmul(mu, mu)
    coefs = []
    for k, rs, _ in gkr_evals:
        for i, r in enumerate(rs):
            pw = ONE
            for e in range(4):
                sparse.append((layout.libra_slot(k, i, e), hmul(coef, pw)))
                pw = hmul(pw, r)
        coefs.append(coef)
        coef = hmul(coef, mu)
    blk = layout.data_block
    zc_coefs = []
    for j in range(N_FINALS):
        zc_coefs.append(coef)
        if j < 3:
            fb.append((*blk(layout.off(layout.W_A + j), g), coef, tuple(zc_point)))
        elif j < 8:
            cb.append((*blk(layout.off(layout.QM + j - 3), g), coef, tuple(zc_point)))
        else:
            for i, r in enumerate(zc_point):
                pw = ONE
                for e in range(ZC_DEGREE + 1):
                    sparse.append((layout.zc_slot(i, e), hmul(coef, pw)))
                    pw = hmul(pw, r)
        coef = hmul(coef, mu)
    fb += mask_blocks(layout, coef)
    return num_pub, den_pub, fb, sparse, cb, coefs, zc_coefs, coef


def weights_vector(n, blocks, sparse=()):
    w = [ZERO] * (1 << n)
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


def weights_eval(n, blocks, sparse, r):
    acc = ZERO
    for fixed, free, c, z in blocks:
        term = hmul(c, eq_pt(z, [r[b] for b in free]))
        for bit, val in fixed:
            term = hmul(term, r[bit] if val else hsub(ONE, r[bit]))
        acc = hadd(acc, term)
    for idx, c in sparse:
        acc = hadd(acc, hmul(c, eq_pt(bits_point(idx, n), r)))
    return acc


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


# ----------------------------------------------------------------------------- zerocheck (§5.5.4)
def gate_value(qm, ql, qr, qo, qc, a, b, c):
    return hadd(hadd(hadd(hmul(qm, hmul(a, b)), hmul(ql, a)), hadd(hmul(qr, b), hmul(qo, c))), qc)


def zerocheck_prove(tr, layout, F, C, hooks=None):
    """Libra-masked zerocheck of sum_x eq(z, x) Gate(x) == 0.  Returns (wire, point, finals)."""
    hooks = hooks or {}
    g, K = layout.g, layout.K
    z = [tr.squeeze_fp3() for _ in range(g)]
    h = [tuple(F[layout.zc_slot(i, e)] for e in range(ZC_DEGREE + 1)) for i in range(g)]
    z_sum = gkr.libra_sum(h)
    tr.absorb_bytes(enc3(z_sum))
    lam = tr.squeeze_fp3()
    cols = [eq_vec(z)]
    cols += [[_get(layout, C, layout.QM + j, x) for x in range(K)] for j in range(5)]
    cols += [[_get(layout, F, layout.W_A + j, x) for x in range(K)] for j in range(3)]
    wire = enc3(z_sum)
    rs = []
    target = hmul(lam, z_sum)
    for i in range(g):
        samples = []
        for t in range(ZC_DEGREE + 1):
            tt = hf(t)
            folded = [gkr._fold(col, tt) for col in cols]
            acc = ZERO
            for e, qm, ql, qr, qo, qc, a, b, c in zip(*folded):
                acc = hadd(acc, hmul(e, gate_value(qm, ql, qr, qo, qc, a, b, c)))
            samples.append(hadd(acc, hmul(lam, gkr.libra_round(h, i, rs, tt))))
        if "zc_round" in hooks:
            samples = hooks["zc_round"](i, samples)
        tr.absorb_bytes(enc3s(samples))
        wire += enc3s(samples)
        r = tr.squeeze_fp3()
        rs.append(r)
        target = lagrange_eval(samples, r)
        cols = [gkr._fold(col, r) for col in cols]
    g_r = ZERO
    for co, r in zip(h, rs):
        g_r = hadd(g_r, gkr.upoly(co, r))
    finals = [cols[6][0], cols[7][0], cols[8][0]] + [cols[1 + j][0] for j in range(5)] + [g_r]
    if "zc_finals" in hooks:
        finals = hooks["zc_finals"](finals)
    if hooks.get("zc_adaptive"):       # tests only: lie about w_a(r) and re-solve the Libra g_r so that
        finals[0] = hadd(finals[0], ONE)   # the zerocheck's own final equation still holds
        gate = gate_value(*finals[3:8], *finals[0:3])
        finals[8] = hmul(hsub(target, hmul(eq_pt(z, rs), gate)), hinv(lam))
    tr.absorb_bytes(enc3s(finals))
    wire += enc3s(finals)
    return wire, tuple(rs), tuple(finals)


# ----------------------------------------------------------------------------- batched folded opening (§5.5.6)
def pair_open(comF, comC, a_vec, c_vec, tr, queries, hooks=None, record=None):
    """Prove Y = sum F a + sum C c with one fold chain over H = F + r_0 C (r_0 = batching round 0)."""
    hooks = hooks or {}
    positions = []
    n = len(comF.values).bit_length() - 1
    size = 1 << n
    G, W = [None] * (4 * size), [None] * (4 * size)
    for y in range(size):
        f, c = comF.values[y], comC.values[y]
        G[4 * y], G[4 * y + 1], G[4 * y + 2], G[4 * y + 3] = f, hadd(f, c), f, hadd(f, c)
        W[4 * y], W[4 * y + 1] = hsub(a_vec[y], c_vec[y]), c_vec[y]
        W[4 * y + 2] = W[4 * y + 3] = ZERO
    rounds, rs, roots = [], [], []
    fv, wv = G, W
    for _ in range(2):
        hh = pcs._sc_round(fv, wv)
        if "round" in hooks:
            hh = hooks["round"](len(rounds), hh)
        tr.absorb_bytes(enc3s(hh))
        rounds.append(hh)
        r = tr.squeeze_fp3()
        rs.append(r)
        fv, wv = pcs._halve(fv, r), pcs._halve(wv, r)
    theta = rs[0]
    code = [hadd(f, hmul(theta, c)) for f, c in zip(comF.code, comC.code)]
    layers = []
    for t in range(n // 2):
        for _ in range(2):
            hh = pcs._sc_round(fv, wv)
            if "round" in hooks:
                hh = hooks["round"](len(rounds), hh)
            tr.absorb_bytes(enc3s(hh))
            rounds.append(hh)
            r = tr.squeeze_fp3()
            rs.append(r)
            fv, wv = pcs._halve(fv, r), pcs._halve(wv, r)
        code = pcs.fold_codeword(code, t, n, rs[-2], rs[-1])
        if "code" in hooks:
            code = hooks["code"](t, code)
        if t < n // 2 - 1:
            levels = tuple(map(tuple, merkle_levels(pcs.coset_leaves(code))))
            layers.append((tuple(code), levels))
            roots.append(levels[-1][0])
            tr.absorb_bytes(roots[-1])
    final = fv[0]
    if "code" not in hooks and any(v != final for v in code):
        raise AssertionError("final folded codeword is not the constant H~(r)")
    tr.absorb_bytes(enc3(final))
    wire = b""
    for i, hh in enumerate(rounds):
        wire += enc3s(hh)
        if i >= 3 and i % 2 == 1 and (i - 3) // 2 < len(roots):
            wire += roots[(i - 3) // 2]
    wire += enc3(final)
    c_levels = hooks.get("c_levels", comC.levels)
    c_code = hooks.get("c_code", comC.code)
    for _ in range(queries):
        pos = int.from_bytes(tr.squeeze_bytes(8), "little") % (len(comF.code) // 4)
        positions.append(pos)
        quarter = len(comF.code) // 4
        wire += enc3s([comF.code[pos + k * quarter] for k in range(4)]) + b"".join(merkle_path(comF.levels, pos))
        wire += enc3s([c_code[pos + k * quarter] for k in range(4)]) + b"".join(merkle_path(c_levels, pos))
        for lay_code, levels in layers:
            quarter = len(lay_code) // 4
            j = pos % quarter
            wire += enc3s([lay_code[j + k * quarter] for k in range(4)]) + b"".join(merkle_path(levels, j))
            pos = j
    if record is not None:
        record["positions"] = tuple(positions)
    return wire, tuple(rs), final


# ----------------------------------------------------------------------------- prover
def prove(table, V, n_s=b"\x00" * 32, profile=FOLDED_AIR_V1, rng=None, hooks=None, engine=None):
    """Exact C_p-byte FPN2 relation-4 container for the circuit `table` and variable vector `V`.

    The prover does NOT check the statement (that is `circuit.lower`'s job): a false statement yields
    bytes that the verifiers reject.  `hooks` (tests / adversarial vectors only):
      witness(layout, table, V, rng) -> F cells     root(r) -> forged GKR root
      gkr / pcs / zc_round / zc_finals -> sub-protocol tampering
      proof_table -> commit a different circuit table inside the proof than `table`
      stmt_root_circ -> bytes written as root_circ (proof still uses the committed table)

    `engine` (a `cuda_engine.CudaEngine` or `HostEngine`): run the heavy polynomial work on the kernels -- see
    `prove_engine`.  The output is byte-identical to engine=None for the same `rng` stream."""
    if engine is not None:
        return prove_engine(table, V, n_s, profile, rng, hooks, engine)
    layout = layout_of(profile)
    rng = rng or SecretRng()
    hooks = hooks or {}
    if len(n_s) != 32:
        raise ValueError("N_s must be 32 bytes")
    ptable = hooks.get("proof_table", table)
    comC = commit_circuit(layout, ptable)
    for _ in range(8):     # resample blinding if (beta, gamma) hit a denominator (probability ~ 2^-180)
        F = hooks["witness"](layout, ptable, V, rng) if "witness" in hooks else build_witness(layout, ptable, V, rng)
        comF = pcs.zk_commit(F)
        hdr = header(profile_id(profile))
        stmt = comF.root + hooks.get("stmt_root_circ", comC.root) + bytes(n_s)
        tr = Fpn2Transcript()
        tr.absorb_bytes(hdr)
        tr.absorb_bytes(stmt)
        beta = tr.squeeze_fp3()
        gamma = tr.squeeze_fp3()
        tree = gkr.build_tree(leaves(layout, F, comC.values, beta, gamma))
        root = tree[-1][0]
        if root[1] == ZERO and not hooks.get("allow_zero_root"):
            continue
        if "root" in hooks:
            root = hooks["root"](root)
        tree[-1][0] = root
        tr.absorb_bytes(enc3(root[0]) + enc3(root[1]))
        libra = lambda k, i: tuple(F[layout.libra_slot(k, i, e)] for e in range(4))  # noqa: E731
        layers, point, claim, evals = gkr.masked_prove(tr, tree, libra, hooks.get("gkr"))
        zc_wire, zc_point, zc_finals = zerocheck_prove(tr, layout, F, comC.values, hooks)
        zb = z_blind(layout, F)
        tr.absorb_bytes(enc3(zb))
        mu = tr.squeeze_fp3()
        _, _, fb, sparse, cb, _, _, _ = batch_terms(layout, point, beta, gamma, evals, zc_point, zc_finals, mu)
        a_vec = weights_vector(layout.n, fb, sparse)
        c_vec = weights_vector(layout.n, cb)
        pos_rec = {}
        pcs_wire, rs, final = pair_open(comF, comC, a_vec, c_vec, tr, layout.queries, hooks.get("pcs"), pos_rec)
        if "record" in hooks:
            hooks["record"].update(F=F, C=comC.values, a=a_vec, c=c_vec, rs=rs, point=point, zc_point=zc_point,
                                   gkr_evals=evals, beta=beta, gamma=gamma, positions=pos_rec["positions"])
        body = (stmt + enc3(root[0]) + enc3(root[1]) + b"".join(lay.wire() for lay in layers) + zc_wire
                + enc3(zb) + pcs_wire)
        if len(body) != layout.body_used:
            raise AssertionError(f"layout mismatch {len(body)} != {layout.body_used}")
        out = hdr + body
        if len(out) > profile["c_p"]:
            raise ValueError("proof exceeds C_p")
        return out + b"\x00" * (profile["c_p"] - len(out))
    raise RuntimeError("could not find a nonzero root denominator")


# ----------------------------------------------------------------------------- engine path 
# `prove(..., engine=E)` produces the SAME bytes as the CPU reference for the same rng stream: every value below
# is the same F_p3 element computed by a different (exact) algorithm, and every transcript absorb / squeeze happens
# in the same order.  Placement:
#   kernels (GPU on CudaEngine, CPU host emulation on HostEngine): Mobius + rate-1/8 Bailey NTT + BLAKE2b Merkle
#       tree of F and C, the FracLogUp fraction tree, every batched-opening sumcheck round (basefold_round_terms +
#       pair reduction) and halving of (G, W), H = F + theta C (one halve over interleaved (F, F + C) codewords),
#       every fused coset-4 fold and folded-layer Merkle tree;
#   host NumPy (vectorised, bit-exact): GKR leaves, the Libra-masked GKR layer sumchecks, the zerocheck, z_blind,
#       the batched weight vectors and the (F + C) codeword sum;
#   host Python: witness sampling (rng draw order is part of parity), transcript, wire assembly, query openings.
ENGINE_HOOKS = frozenset({"witness", "proof_table", "stmt_root_circ", "root", "allow_zero_root", "record", "timings"})


def _np():
    import numpy as np
    return np


def _cpu():
    from flashprove_next.field import CPU
    return CPU


def cell_np(data_index):
    """Vectorised `AirLayout.cell`: data bit 0 -> cell bit 0, cell bit 1 = 1, data bit i >= 1 -> cell bit i + 1."""
    d = _np().asarray(data_index, dtype=_np().int64)
    return (d & 1) | 2 | ((d >> 1) << 2)


def _tup(col):
    return tuple(int(v) for v in col)


class EngineCommitment:
    """Array-backed commitment: cells (3, 2^n) uint64 on the host; the codeword (L, 3) and the Merkle levels
    [(m, 32) uint8] stay on the engine's device (host copies are made lazily, only when `code` / `levels` are read).
    Query openings gather just the opened rows and siblings (`open_rows`)."""

    def __init__(self, cells, code, levels, engine=None, rows=None):
        from . import cuda_engine as CE
        self.cells, self.engine = cells, engine
        self.rows = rows if rows is not None else _np().ascontiguousarray(cells.T)    # (2^n, 3) layout
        self.code_dev, self.levels_dev = code, levels
        self._code = self._levels = None
        self._root = bytes(_np().asarray(CE._host(engine, levels[-1]))[0]) if engine is not None \
            else bytes(levels[-1][0])

    def _h(self, a):
        from . import cuda_engine as CE
        return _np().asarray(CE._host(self.engine, a)) if self.engine is not None else _np().asarray(a)

    @property
    def root(self):
        return self._root

    @property
    def code(self):
        if self._code is None:
            self._code = _np().ascontiguousarray(self._h(self.code_dev))
        return self._code

    @property
    def levels(self):
        if self._levels is None:
            self._levels = [self._h(lv) for lv in self.levels_dev]
        return self._levels

    @property
    def code_len(self):
        return len(self.code_dev)

    @property
    def values(self):
        return tuple(_tup(c) for c in self.cells.T)

    def code_at(self, i):
        return _tup(self._h(self.code_dev[i]).reshape(3))

    def path(self, i):
        return [bytes(self._h(self.levels_dev[k][(i >> k) ^ 1]).reshape(32)) for k in range(len(self.levels_dev) - 1)]

    def open_many(self, positions):
        """[(enc3s of the 4 coset values, Merkle path bytes)] for every position (batched device gathers)."""
        from . import cuda_engine as CE
        vals, sibs = CE.open_rows(self.engine, self.code_dev, self.levels_dev, positions)
        return [(enc3s([_tup(v) for v in vals[q]]), b"".join(bytes(sb[q]) for sb in sibs)) for q in range(len(vals))]


def engine_commit(engine, cells):
    """ZK-Basefold commitment of 2^n F_p3 cells ((2^n, 3) array or list of tuples) on `engine`."""
    from . import cuda_engine as CE
    np = _np()
    arr = np.ascontiguousarray(np.asarray(cells, dtype=np.uint64).reshape(-1, 3))
    n = len(arr).bit_length() - 1
    if len(arr) != 1 << n or n % 2 or n < 2:
        raise ValueError("ZK-Basefold needs 2^n cells with n even")
    if (arr >= P).any():
        raise ValueError("noncanonical field limb")
    code, levels = CE.commit_pipeline_dev(engine, arr)
    return EngineCommitment(np.ascontiguousarray(arr.T), code, levels, engine, rows=arr)


def circuit_cells_np(layout, table):
    np = _np()
    K = layout.K
    out = np.zeros((1 << layout.n, 3), dtype=np.uint64)
    rows = np.array(table.rows, dtype=np.uint64).reshape(K, 8)
    x = np.arange(K, dtype=np.int64)
    for blk in range(8):
        out[cell_np(blk * K + x), 0] = rows[:, blk]
    out[cell_np(layout.MULT * K + x), 0] = np.array(table.mult, dtype=np.uint64)
    out[cell_np(layout.FLAG * K + x), 0] = np.array(table.flag, dtype=np.uint64)
    return out


def build_witness_np(layout, table, V, rng):
    """`build_witness` as a (2^n, 3) array with the SAME rng draw order (default cells, then (N_x, E_x) per row x,
    then (A_v, D_v) per mask index v), vectorised placement."""
    np = _np()
    K = layout.K
    if len(V) != K or any(type(v) is not int or not 0 <= v < P for v in V):
        raise ValueError("V must hold K canonical F_p values")
    mult = range_multiplicities(table, V) + [0] * (K - TABLE)
    if hasattr(rng, "fp3_array"):
        # bulk path (SecretRng): independent uniform blocks, nonzero where `_nonzero` is used (E_x, D_v)
        cells = draw_cells(rng, 1 << layout.n)
        ne = np.stack([draw_cells(rng, K), draw_cells(rng, K, nonzero=True)], axis=1)
        ad = np.stack([draw_cells(rng, 4 * K), draw_cells(rng, 4 * K, nonzero=True)], axis=1)
    else:
        cells = np.array([rng.fp3() for _ in range(1 << layout.n)], dtype=np.uint64)
        ne = np.array([(rng.fp3(), _nonzero(rng)) for _ in range(K)], dtype=np.uint64)          # (K, 2, 3)
        ad = np.array([(rng.fp3(), _nonzero(rng)) for _ in range(4 * K)], dtype=np.uint64)      # (4K, 2, 3)
    x = np.arange(K, dtype=np.int64)
    Varr = np.array(V, dtype=np.uint64)
    sig = np.array([row[5:] for row in table.rows], dtype=np.int64).reshape(K, 3)

    def put_base(blk, vals):
        idx = cell_np(blk * K + x)
        cells[idx, 0] = vals
        cells[idx, 1:] = 0

    for c in range(3):
        put_base(layout.W_A + c, Varr[sig[:, c] - 1])
    put_base(layout.V, Varr)
    put_base(layout.M, np.array(mult, dtype=np.uint64))
    cells[cell_np(layout.N * K + x)] = ne[:, 0]
    cells[cell_np(layout.E * K + x)] = ne[:, 1]
    v = np.arange(4 * K, dtype=np.int64)
    cells[cell_np(layout.off_a + v)] = ad[:, 0]
    cells[cell_np(layout.off_d + v)] = ad[:, 1]
    return cells


def _const(c, m):
    np = _np()
    return np.repeat(np.array(c, dtype=np.uint64).reshape(3, 1), m, axis=1)


def _base(v):
    """int vector -> (3, m) F_p3 array embedding F_p."""
    return _cpu().ext(_np().asarray(v, dtype=_np().uint64))


def leaves_np(layout, Fa, Ca, beta, gamma):
    """`leaves` on (3, 2^n) cell arrays -> (2^(S+1), 6) [num | den] array (bit-exact)."""
    np, X = _np(), _cpu()
    K, S = layout.K, layout.S
    x = np.arange(K, dtype=np.int64)
    zero = X.zeros_e(K)
    neg = lambda a: X.esub(zero if a.shape[1] == K else X.zeros_e(a.shape[1]), a)  # noqa: E731
    getF = lambda blk: Fa[:, cell_np(blk * K + x)]  # noqa: E731
    getC = lambda blk: Ca[:, cell_np(blk * K + x)]  # noqa: E731
    bk = _const(beta, K)
    num = X.zeros_e(2 << S)
    den = X.zeros_e(2 << S)

    def put(blk, nn, dd):
        num[:, 2 * (blk * K + x)] = nn
        den[:, 2 * (blk * K + x)] = dd

    one = _const(ONE, K)
    for c in range(3):
        put(c, one, X.esub(X.esub(bk, getF(layout.W_A + c)), _escale(getC(layout.SA + c), gamma)))
    v = getF(layout.V)
    put(3, neg(getC(layout.MULT)), X.esub(X.esub(bk, v), _escale(_base(x + 1), gamma)))
    put(4, getC(layout.FLAG), X.esub(bk, v))
    put(5, neg(getF(layout.M)), X.esub(bk, _base(x % TABLE)))
    nn, ee = getF(layout.N), getF(layout.E)
    put(6, nn, ee)
    put(7, neg(nn), ee)
    half = 1 << (S - 1)
    vv = np.arange(1 << S, dtype=np.int64) % half
    a = Fa[:, cell_np(layout.off_a + vv)]
    d = Fa[:, cell_np(layout.off_d + vv)]
    a[:, half:] = neg(a[:, half:])
    num[:, 1::2] = a
    den[:, 1::2] = d
    return np.ascontiguousarray(np.concatenate([num.T, den.T], axis=1))


def _escale(a, c):
    """(3, m) times an F_p3 scalar (native ext_scale when available, else ONE ext_mul against a repeated constant)."""
    from flashprove_next.field import _c, _pts, native, ptr
    np = _np()
    lib = native()
    if lib is not None and a.ndim == 2 and a.shape[0] == 3:
        a = _c(a)
        out = np.empty_like(a)
        lib.ext_scale(ptr(a), a.shape[1], ptr(_pts([c])), ptr(out))
        return out
    return _cpu().emul(a, _const(c, a.shape[1]))


def _fold(col, r):
    return _cpu().efold(col, r)


def _small_folds(col, count):
    """[fold(col, t) for t = 0 .. count-1] with additions only (t is a small integer)."""
    X = _cpu()
    lo, hi = col[:, 0::2], col[:, 1::2]
    step = X.esub(hi, lo)
    out = [lo, hi]
    while len(out) < count:
        out.append(X.eadd(out[-1], step))
    return out[:count]


def _gkr_round_np(columns, alpha):
    from flashprove_next.field import _c, _pts, native, ptr
    X, np = _cpu(), _np()
    lib = native()
    if lib is not None and columns[0].shape[1] >= 2:
        cols = [_c(c) for c in columns]
        out = np.empty(12, dtype=np.uint64)
        lib.gkr_round(*(ptr(c) for c in cols), cols[0].shape[1], ptr(_pts([alpha])), ptr(out))
        return tuple(_tup(out[3 * t:3 * t + 3]) for t in range(4))
    folds = [_small_folds(col, 4) for col in columns]
    out = []
    for i in range(4):
        w, a, b, c, d = (f[i] for f in folds)
        num, den = X.eadd(X.emul(a, d), X.emul(c, b)), X.emul(b, d)
        out.append(X.esum(X.emul(w, X.eadd(num, _escale(den, alpha)))))
    return tuple(out)


GKR_DEVICE_MIN = 1 << 12     # layers with at least this many pairs run their rounds on the engine (gkr_round)


def masked_prove_np(tr, tree, libra, engine=None):
    """`frac_logup_gkr.masked_prove` (no hooks) on a tree of (m, 6) [num | den] arrays (bit-exact).  With an engine
    that has `gkr_round` (CudaEngine), the round evaluations and halvings of the large layers run on the device
    (same field values; the transcript stays on the host)."""
    np, X = _np(), _cpu()
    point = ()
    layers, evals = [], []
    on_dev = engine is not None and hasattr(engine, "gkr_round")
    for k, nodes in enumerate(reversed(tree[:-1])):
        nodes = np.asarray(nodes)
        alpha = tr.squeeze_fp3()
        g = g_sum = lam = None
        if k:
            g = [tuple(libra(k, i)) for i in range(k)]
            g_sum = gkr.libra_sum(g)
            tr.absorb_bytes(enc3(g_sum))
            lam = tr.squeeze_fp3()
        even, odd = nodes[0::2], nodes[1::2]
        dev = on_dev and k and len(even) >= GKR_DEVICE_MIN
        if dev:
            xp = engine.cp
            nd = engine.dev(nodes)
            columns = [engine.dev(np.ascontiguousarray(X.eq_table(list(point)).T)),
                       xp.ascontiguousarray(nd[0::2, :3]), xp.ascontiguousarray(nd[0::2, 3:]),
                       xp.ascontiguousarray(nd[1::2, :3]), xp.ascontiguousarray(nd[1::2, 3:])]
            del nd
        else:
            columns = [X.eq_table(list(point)), np.ascontiguousarray(even[:, :3].T),
                       np.ascontiguousarray(even[:, 3:].T), np.ascontiguousarray(odd[:, :3].T),
                       np.ascontiguousarray(odd[:, 3:].T)]
        rounds, rs = [], []
        for i in range(k):
            if dev:
                samples = tuple(_tup(v) for v in engine.gkr_round(columns, alpha))
            else:
                samples = _gkr_round_np(columns, alpha)
            samples = tuple(hadd(f, hmul(lam, gkr.libra_round(g, i, rs, hf(t)))) for t, f in enumerate(samples))
            tr.absorb_bytes(enc3s(samples))
            r = tr.squeeze_fp3()
            rounds.append(samples)
            rs.append(r)
            columns = [engine.halve(col, r) for col in columns] if dev else [_fold(col, r) for col in columns]
        if dev:
            children = tuple(_tup(np.asarray(engine.host(col)).reshape(3)) for col in columns[1:])
        else:
            children = tuple(_tup(col[:, 0]) for col in columns[1:])
        g_r = None
        if k:
            g_r = ZERO
            for co, r in zip(g, rs):
                g_r = hadd(g_r, gkr.upoly(co, r))
        tr.absorb_bytes(enc3s(children) + (enc3(g_r) if k else b""))
        if k:
            evals.append((k, tuple(rs), g_r))
        layers.append(gkr.MaskedLayer(g_sum, tuple(rounds), children, g_r))
        point = (tr.squeeze_fp3(), *rs)
    return tuple(layers), point, None, tuple(evals)


def zerocheck_np(tr, layout, Fa, Ca):
    """`zerocheck_prove` (no hooks) on (3, 2^n) cell arrays (bit-exact)."""
    np, X = _np(), _cpu()
    g, K = layout.g, layout.K
    z = [tr.squeeze_fp3() for _ in range(g)]
    h = [tuple(_tup(Fa[:, layout.zc_slot(i, e)]) for e in range(ZC_DEGREE + 1)) for i in range(g)]
    z_sum = gkr.libra_sum(h)
    tr.absorb_bytes(enc3(z_sum))
    lam = tr.squeeze_fp3()
    x = np.arange(K, dtype=np.int64)
    cols = [X.eq_table(z)]
    cols += [Ca[:, cell_np((layout.QM + j) * K + x)] for j in range(5)]
    cols += [Fa[:, cell_np((layout.W_A + j) * K + x)] for j in range(3)]
    wire = enc3(z_sum)
    rs = []
    for i in range(g):
        samples = []
        folds = [_small_folds(col, ZC_DEGREE + 1) for col in cols]
        for t in range(ZC_DEGREE + 1):
            tt = hf(t)
            e, qm, ql, qr, qo, qc, a, b, c = (f[t] for f in folds)
            gate = X.eadd(X.eadd(X.eadd(X.emul(qm, X.emul(a, b)), X.emul(ql, a)),
                                 X.eadd(X.emul(qr, b), X.emul(qo, c))), qc)
            samples.append(hadd(X.esum(X.emul(e, gate)), hmul(lam, gkr.libra_round(h, i, rs, tt))))
        tr.absorb_bytes(enc3s(samples))
        wire += enc3s(samples)
        r = tr.squeeze_fp3()
        rs.append(r)
        cols = [_fold(col, r) for col in cols]
    g_r = ZERO
    for co, r in zip(h, rs):
        g_r = hadd(g_r, gkr.upoly(co, r))
    finals = [_tup(cols[6][:, 0]), _tup(cols[7][:, 0]), _tup(cols[8][:, 0])]
    finals += [_tup(cols[1 + j][:, 0]) for j in range(5)] + [g_r]
    tr.absorb_bytes(enc3s(finals))
    wire += enc3s(finals)
    return wire, tuple(rs), tuple(finals)


_EQ_CONST = {}          # eq([z]^k, .) for the PUBLIC constant mask points z (witness-independent)
_DEPOSIT = {}           # (fixed, free) -> cell indices of a weight block (layout-only, witness-independent)


def _eq_const(z, k):
    key = (tuple(z), k)
    if key not in _EQ_CONST:
        if len(_EQ_CONST) > 16:
            _EQ_CONST.clear()
        _EQ_CONST[key] = _cpu().eq_table([tuple(z)] * k)
    return _EQ_CONST[key]


def _eq_block(z):
    """eq(z, .) over len(z) variables; the constant MASK_POINTS tables are cached."""
    z = tuple(tuple(v) for v in z)
    if z and all(v == z[0] for v in z) and z[0] in MASK_POINTS:
        return _eq_const(z[0], len(z))
    return _cpu().eq_table(list(z))


def _deposit(fixed, free):
    """Cell index of every b in [0, 2^len(free)): base(fixed) with bit i of b deposited at cell bit free[i]
    (vectorised per run of consecutive free bits)."""
    np = _np()
    key = (tuple(fixed), tuple(free))
    if key not in _DEPOSIT:
        if len(_DEPOSIT) > 64:
            _DEPOSIT.clear()
        k = len(free)
        b = np.arange(1 << k, dtype=np.int64)
        idx = np.full_like(b, sum(val << bit for bit, val in fixed))
        i = 0
        while i < k:
            j = i
            while j + 1 < k and free[j + 1] == free[j] + 1:
                j += 1
            idx |= ((b >> i) & ((1 << (j - i + 1)) - 1)) << free[i]
            i = j + 1
        _DEPOSIT[key] = idx
    return _DEPOSIT[key]


def z_blind_np(layout, Fa):
    """sum over the blinding half {bit1 = 0} of (eq(z^n, idx) for z in MASK_POINTS) * F[idx]: the bit-1 factor
    (1 - z) times the (n - 1)-variable constant table over the remaining bits (cached), in increasing cell order."""
    np, X = _np(), _cpu()
    free = tuple(b for b in range(layout.n) if b != 1)
    cells = np.ascontiguousarray(Fa[:, _deposit(((1, 0),), free)])
    acc = ZERO
    for z in MASK_POINTS:
        acc = hadd(acc, hmul(hsub(ONE, z), X.edot_dev(_eq_const(z, layout.n - 1), cells)))
    return acc


def _block_view(w, fixed, free):
    """Strided view of the cells of a block whose free bits are (0, 2, 3, ..., k) and whose fixed bits are bit 1 plus
    bits above k (every relation-4 / relation-5 data and mask block): shape (3, 2^(k-1), 2) with [:, h, l] the cell
    of b = 2 h + l.  None when the block has another shape (the caller then uses `_deposit`)."""
    k = len(free)
    if k < 2 or tuple(free) != (0,) + tuple(range(2, k + 1)):
        return None
    fx = dict(fixed)
    if 1 not in fx or any(bit <= k and bit != 1 for bit in fx):
        return None
    lo = sum(val << bit for bit, val in fixed if bit != 1)
    return w[:, lo:lo + (1 << (k + 1))].reshape(3, 1 << (k - 1), 2, 2)[:, :, fx[1], :]


def weights_vector_np(n, blocks, sparse=()):
    """`weights_vector` as a (3, 2^n) array (bit-exact)."""
    X = _cpu()
    w = X.zeros_e(1 << n)
    for fixed, free, c, z in blocks:
        if len(z) > len(free):
            raise ValueError("block point longer than its free bits")
        vals = _escale(_eq_block(z), c)
        view = _block_view(w, fixed, free[:len(z)])
        if view is not None:
            view[...] = X.eadd(_np().ascontiguousarray(view).reshape(3, -1), vals).reshape(view.shape)
        else:
            idx = _deposit(fixed, free[:len(z)])
            w[:, idx] = X.eadd(w[:, idx], vals)
    for idx, c in sparse:
        w[:, idx:idx + 1] = X.eadd(w[:, idx:idx + 1], _const(c, 1))
    return w


def _interleave(*arrs):
    """(m, 3) arrays a_0..a_{k-1} -> (k m, 3) with row k y + j = a_j[y]."""
    np = _np()
    return np.ascontiguousarray(np.stack(arrs, axis=1).reshape(-1, 3))


def pair_open_engine(engine, comF, comC, a, c, tr, queries, record=None):
    """`pair_open` (no hooks) with every sumcheck round, halving, fold and folded-layer tree on `engine`.  The
    batched vectors, the (F + theta C) codeword and the folded layers are built and kept on the device; only the
    round messages, layer roots and the queried rows / Merkle siblings come back to the host."""
    from . import cuda_engine as CE
    np = _np()
    xp = CE.xp_of(engine)
    n = comF.cells.shape[1].bit_length() - 1
    host = lambda v: np.asarray(CE._host(engine, v))  # noqa: E731
    Fd = engine.dev(comF.rows)
    Cd = engine.dev(comC.rows)
    FCd = engine.fadd(Fd, Cd)
    fv = xp.ascontiguousarray(xp.stack([Fd, FCd, Fd, FCd], axis=1).reshape(-1, 3))
    del Cd, FCd
    ad = engine.dev(np.ascontiguousarray(np.asarray(a).T))
    cd = engine.dev(np.ascontiguousarray(np.asarray(c).T))
    zero = xp.zeros_like(cd)
    wv = xp.ascontiguousarray(xp.stack([engine.fsub(ad, cd), cd, zero, zero], axis=1).reshape(-1, 3))
    del Fd, ad, cd, zero
    rounds, rs, roots, layers = [], [], [], []

    def sc_round():
        nonlocal fv, wv
        terms, _ = engine.basefold_round(fv, wv, ZERO)
        hh = tuple(_tup(row) for row in host(terms).reshape(3, 3))
        tr.absorb_bytes(enc3s(hh))
        rounds.append(hh)
        r = tr.squeeze_fp3()
        rs.append(r)
        fv, wv = engine.halve(fv, r), engine.halve(wv, r)

    for _ in range(2):
        sc_round()
    theta = rs[0]
    codeF, codeC = comF.code_dev, comC.code_dev
    cur = engine.halve(xp.ascontiguousarray(xp.stack([engine.dev(codeF), engine.fadd(codeF, codeC)], axis=1)
                                            .reshape(-1, 3)), theta)
    omega = CE.root_of_unity(CE.RATE << n)
    inv_iota = pow(CE.root_of_unity(4), P - 2, P)
    for t in range(n // 2):
        sc_round()
        sc_round()
        omega_t = pow(omega, 4 ** t, P)
        cur = engine.fold4(cur, pow(omega_t, P - 2, P), inv_iota, rs[-2], rs[-1])
        if t < n // 2 - 1:
            levels = CE.merkle_levels_dev(engine, cur)
            layers.append((cur, levels))
            roots.append(bytes(host(levels[-1])[0]))
            tr.absorb_bytes(roots[-1])
    final = _tup(host(fv).reshape(-1, 3)[0])
    last = host(cur).reshape(-1, 3)
    if any(_tup(v) != final for v in last):
        raise AssertionError("final folded codeword is not the constant H~(r)")
    tr.absorb_bytes(enc3(final))
    wire = b""
    for i, hh in enumerate(rounds):
        wire += enc3s(hh)
        if i >= 3 and i % 2 == 1 and (i - 3) // 2 < len(roots):
            wire += roots[(i - 3) // 2]
    wire += enc3(final)
    quarter0 = comF.code_len // 4
    positions = [int.from_bytes(tr.squeeze_bytes(8), "little") % quarter0 for _ in range(queries)]
    opened = [comF.open_many(positions), comC.open_many(positions)]
    pos_t = list(positions)
    for lay_code, levels in layers:
        quarter = len(lay_code) // 4
        pos_t = [p % quarter for p in pos_t]
        vals, sibs = CE.open_rows(engine, lay_code, levels, pos_t)
        opened.append([(enc3s([_tup(v) for v in vals[q]]), b"".join(bytes(sb[q]) for sb in sibs))
                       for q in range(len(pos_t))])
    parts = [wire]
    for q in range(queries):
        for op in opened:
            parts.append(op[q][0] + op[q][1])
    if record is not None:
        record["positions"] = tuple(positions)
    return b"".join(parts), tuple(rs), final


def _sync(engine):
    if hasattr(engine, "sync"):
        engine.sync()


def prove_engine(table, V, n_s, profile, rng, hooks, engine):
    """Engine-accelerated relation-4 prover (byte-identical to the CPU `prove` for the same rng stream).

    Supported hooks: witness, proof_table, stmt_root_circ, root, allow_zero_root, record, and `timings` (a dict
    filled with per-phase wall milliseconds, synchronised on the engine).  Sub-protocol tamper hooks (gkr, pcs,
    zc_*) are CPU-only and raise here, so a tamper request can never silently produce an honest proof."""
    import time
    from . import cuda_engine as CE
    layout = layout_of(profile)
    rng = rng or SecretRng()
    hooks = hooks or {}
    bad = set(hooks) - ENGINE_HOOKS
    if bad:
        raise ValueError(f"hooks {sorted(bad)} are CPU-only (engine=None)")
    if len(n_s) != 32:
        raise ValueError("N_s must be 32 bytes")
    tm = hooks.get("timings", {})
    clock = [time.perf_counter()]

    def lap(name):
        _sync(engine)
        now = time.perf_counter()
        tm[name] = tm.get(name, 0.0) + (now - clock[0]) * 1e3
        clock[0] = now

    ptable = hooks.get("proof_table", table)
    comC = commit_circuit(layout, ptable, engine)
    lap("commit_circuit")
    for _ in range(8):
        F = hooks["witness"](layout, ptable, V, rng) if "witness" in hooks else build_witness_np(layout, ptable, V, rng)
        lap("witness_sample")
        comF = engine_commit(engine, F)
        lap("commit_witness")
        Fa, Ca = comF.cells, comC.cells
        hdr = header(profile_id(profile))
        stmt = comF.root + hooks.get("stmt_root_circ", comC.root) + bytes(n_s)
        tr = Fpn2Transcript()
        tr.absorb_bytes(hdr)
        tr.absorb_bytes(stmt)
        beta = tr.squeeze_fp3()
        gamma = tr.squeeze_fp3()
        lv = leaves_np(layout, Fa, Ca, beta, gamma)
        lap("gkr_leaves")
        tree = CE.gkr_tree(engine, lv)
        lap("gkr_tree")
        root = (_tup(tree[-1][0][:3]), _tup(tree[-1][0][3:]))
        if root[1] == ZERO and not hooks.get("allow_zero_root"):
            continue
        if "root" in hooks:
            root = hooks["root"](root)
        tr.absorb_bytes(enc3(root[0]) + enc3(root[1]))
        libra = lambda k, i: tuple(_tup(Fa[:, layout.libra_slot(k, i, e)]) for e in range(4))  # noqa: E731
        layers, point, claim, evals = masked_prove_np(tr, tree, libra, engine)
        lap("gkr_sumcheck")
        zc_wire, zc_point, zc_finals = zerocheck_np(tr, layout, Fa, Ca)
        lap("zerocheck")
        zb = z_blind_np(layout, Fa)
        tr.absorb_bytes(enc3(zb))
        mu = tr.squeeze_fp3()
        _, _, fb, sparse, cb, _, _, _ = batch_terms(layout, point, beta, gamma, evals, zc_point, zc_finals, mu)
        a_vec = weights_vector_np(layout.n, fb, sparse)
        c_vec = weights_vector_np(layout.n, cb)
        lap("batch_weights")
        pos_rec = {}
        pcs_wire, rs, final = pair_open_engine(engine, comF, comC, a_vec, c_vec, tr, layout.queries, pos_rec)
        lap("pair_open")
        if "record" in hooks:
            hooks["record"].update(F=list(comF.values), C=comC.values, a=[_tup(v) for v in a_vec.T],
                                   c=[_tup(v) for v in c_vec.T], rs=rs, point=point, zc_point=zc_point,
                                   gkr_evals=evals, beta=beta, gamma=gamma, positions=pos_rec["positions"])
        body = (stmt + enc3(root[0]) + enc3(root[1]) + b"".join(lay.wire() for lay in layers) + zc_wire
                + enc3(zb) + pcs_wire)
        if len(body) != layout.body_used:
            raise AssertionError(f"layout mismatch {len(body)} != {layout.body_used}")
        out = hdr + body
        if len(out) > profile["c_p"]:
            raise ValueError("proof exceeds C_p")
        return out + b"\x00" * (profile["c_p"] - len(out))
    raise RuntimeError("could not find a nonzero root denominator")


# ----------------------------------------------------------------------------- size model
def byte_budget(n, queries):
    """Exact relation-4 container body size for (n, Q) and the gate/variable capacity K = 2^(n-5)."""
    lay = AirLayout(n, queries)
    return {"n_vars": n, "queries": queries, "rows": lay.K, "variables": lay.K,
            "gate_cells": 3 * lay.K, "body": lay.body_used, "container_min": 64 + lay.body_used}


# ----------------------------------------------------------------------------- ZK rank condition 
# The honest prover's LINEAR view.  For fixed challenges every plaintext value below is an F_p3-linear functional
# of the committed cells F (the public circuit vector C only adds constants, which are dropped):
#   claims : every GKR Libra sum g_sum(k) and evaluation g_r(k); the last GKR layer's leaf children
#            (nl, dl) = real-leaf (num, den) and (nr, dr) = mask-leaf (num, den) at the layer point; the
#            zerocheck Libra sum, its finals e_a, e_b, e_c = w_col~(r) and g_r; the blinding scalar z_blind;
#   pcs    : the 3 (n + 2) batched sumcheck samples h_i(0..2) and H~(r); per query the 4 layer-0 coset values
#            of F's codeword and the 4 coset values of every folded layer of H = F + theta C.
# Coordinates are what the prover actually samples, not raw cells: a variable k is ONE F_p coordinate whose
# column is the sum of its V cell and every gate cell wired to it (copy constraints make them equal); the
# range multiplicities m_t (t < 128) are F_p coordinates; every blinding / mask / Libra cell is a uniform F_p3
# coordinate (3 F_p columns: f, f u, f u^2 since E = F_p[u] / (u^3 - 2)).  Each F_p3 observation is 3 F_p rows.
#
# `require_joint_mask_air` checks im(A) <= im(B) over F_p (rank([B | A]) == rank(B)) where A = witness
# coordinates and B = mask coordinates.  It certifies the LINEAR view only: the GKR inner-layer messages and the
# zerocheck / GKR round polynomials are non-linear in F and rest on the mask-leaf (A, D, N, E) and Libra
# arguments, which consume some of the same mask coordinates.  `require_pcs_mask_air` is the stricter
# relation-3-style check of the opening alone: every non-reserved coordinate (witness AND every mask used by a
# non-linear message) is secret and only the cells no other message touches (the reserved half bit1 = 0 minus
# the Libra slots, and the spare block [7K, 8K)) may absorb it.
import os as _os
import random as _random
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile

_RANK_MAGIC = 0x4b4e4152504e4650


def _sparse_libra_sum(slot, polys, coeffs, scale):
    out = []
    for i in range(polys):
        out.append((slot(i, 0), _scal(2 * scale % P, ONE)))
        for e in range(1, coeffs):
            out.append((slot(i, e), _scal(scale, ONE)))
    return out


def _sparse_libra_eval(slot, rs, coeffs):
    out = []
    for i, r in enumerate(rs):
        pw = ONE
        for e in range(coeffs):
            out.append((slot(i, e), pw))
            pw = hmul(pw, r)
    return out


def _neg_blocks(blocks):
    return [(fx, fr, hsub(ZERO, c), z) for fx, fr, c, z in blocks]


def claim_functionals(layout, rec):
    """[(label, blocks, sparse)] for every plaintext claim that is linear in F (public C parts dropped)."""
    out = []
    for k, rs, _ in rec["gkr_evals"]:
        slot = lambda i, e, k=k: layout.libra_slot(k, i, e)  # noqa: E731
        out.append((f"gkr_libra_sum_{k}", [], _sparse_libra_sum(slot, k, 4, pow(2, k - 1, P))))
        out.append((f"gkr_g_r_{k}", [], _sparse_libra_eval(slot, rs, 4)))
    rs = tuple(rec["point"][1:])
    beta, gamma = rec["beta"], rec["gamma"]
    for b0, tag in ((ZERO, "l"), (ONE, "r")):
        num = leaf_terms(layout, (b0, *rs), beta, gamma, ZERO)[2]
        both = leaf_terms(layout, (b0, *rs), beta, gamma, ONE)[2]
        out.append((f"gkr_leaf_n{tag}", num, []))
        out.append((f"gkr_leaf_d{tag}", both + _neg_blocks(num), []))
    g = layout.g
    zc = tuple(rec["zc_point"])
    out.append(("zc_libra_sum", [], _sparse_libra_sum(layout.zc_slot, g, ZC_DEGREE + 1, pow(2, g - 1, P))))
    for c in range(3):
        out.append((f"zc_e_{'abc'[c]}", [(*layout.data_block(layout.off(layout.W_A + c), g), ONE, zc)], []))
    out.append(("zc_g_r", [], _sparse_libra_eval(layout.zc_slot, zc, ZC_DEGREE + 1)))
    out.append(("z_blind", mask_blocks(layout, ONE), []))
    return out


def _eq_table_np(point):
    from flashprove_next.field import CPU
    return CPU.eq_table(list(point))


def pcs_functionals(layout, rec):
    """Yield (label, dense (3, 2^n) uint64 row over the cells of F) for every value the opening reveals."""
    import numpy as np
    n, size = layout.n, 1 << layout.n
    a, c = rec["a"], rec["c"]
    W = [None] * (4 * size)
    for y in range(size):
        W[4 * y], W[4 * y + 1], W[4 * y + 2], W[4 * y + 3] = hsub(a[y], c[y]), c[y], ZERO, ZERO
    rs = list(rec["rs"])
    rows = pcs.observation_rows(n + 2, W, rs, ())          # 3 (n + 2) round samples + G~(r), over G's 4 size cells
    for i, row in enumerate(rows):                        # G[4y + k] = F[y] + (public C terms) for every k
        arr = np.array(row, dtype=np.uint64).T.reshape(3, size, 4)
        acc = arr[:, :, 0]
        from flashprove_next.field import CPU
        for k in range(1, 4):
            acc = CPU.eadd(acc, arr[:, :, k])
        yield (f"pcs_round_{i // 3}_{i % 3}" if i < 3 * (n + 2) else "pcs_final", acc)
    rest = rs[2:]
    length = pcs.RATE << n
    omega = pcs.root_of_unity(length)
    seen = set()
    for pos in rec["positions"]:
        for t in range(layout.folds):
            quarter = length >> (2 * t + 2)
            j = pos % quarter
            if (t, j) not in seen:
                seen.add((t, j))
                omega_t = pow(omega, 4 ** t, P)
                for k in range(4):
                    x = pow(omega_t, j + k * quarter, P)
                    pt = list(rest[:2 * t])
                    for _ in range(n - 2 * t):
                        pt.append(hf(x))
                        x = x * x % P
                    yield (f"query_t{t}_j{j}_k{k}", _eq_table_np(pt))
            pos = j


class Coordinates:
    """F_p coordinate model of the honest witness vector F (see the section comment)."""

    def __init__(self, layout, table, mask_vars):
        K = layout.K
        self.layout = layout
        self.var_cells = [[layout.cell(layout.V * K + k)] for k in range(K)]
        for x, row in enumerate(table.rows):
            for c in range(3):
                self.var_cells[row[5 + c] - 1].append(layout.cell((layout.W_A + c) * K + x))
        self.mask_vars = sorted(set(mask_vars))
        self.witness_vars = [k for k in range(K) if k not in set(self.mask_vars)]
        self.m_cells = [layout.cell(layout.M * K + t) for t in range(TABLE)]
        libra = {layout.libra_slot(k, i, e) for k in range(1, layout.S + 1) for i in range(k) for e in range(4)}
        libra |= {layout.zc_slot(i, e) for i in range(layout.g) for e in range(ZC_DEGREE + 1)}
        self.libra = sorted(libra)
        half = [idx for idx in range(1 << layout.n) if not layout.is_data(idx)]
        self.reserved = [idx for idx in half if idx not in libra]
        self.spare = [layout.cell(7 * K + x) for x in range(K)]
        self.gkr_masks = ([layout.cell(blk * K + x) for blk in (layout.N, layout.E) for x in range(K)]
                          + [layout.cell(layout.off_a + v) for v in range(4 * K)]
                          + [layout.cell(layout.off_d + v) for v in range(4 * K)])


def _fp3_columns(row, cells):
    """(3, 3 * len(cells)) block: for each F_p3 cell x the columns of f, f u, f u^2 (u^3 = 2)."""
    import numpy as np
    from flashprove_next.field import CPU
    idx = np.asarray(cells, dtype=np.int64)
    f0, f1, f2 = row[0][idx], row[1][idx], row[2][idx]
    t1, t2 = CPU.add(f1, f1), CPU.add(f2, f2)
    return np.concatenate([np.stack([f0, f1, f2]), np.stack([t2, f0, f1]), np.stack([t1, t2, f0])], axis=1)


def _var_columns(row, var_cells, ks):
    import numpy as np
    out = np.zeros((3, len(ks)), dtype=np.uint64)
    r = [row[i].tolist() for i in range(3)]
    for j, k in enumerate(ks):
        for comp in range(3):
            out[comp, j] = sum(r[comp][cidx] for cidx in var_cells[k]) % P
    return out


def _project(row, spec, coords):
    import numpy as np
    parts = []
    for kind, payload in spec:
        if kind == "var":
            parts.append(_var_columns(row, coords.var_cells, payload))
        elif kind == "fp":
            parts.append(row[:, np.asarray(payload, dtype=np.int64)])
        else:
            parts.append(_fp3_columns(row, payload))
    return np.concatenate(parts, axis=1)


def _spec_width(spec):
    return sum(len(p) * (3 if kind == "fp3" else 1) for kind, p in spec)


def _dense(layout, blocks, sparse):
    import numpy as np
    w = weights_vector(layout.n, blocks, sparse)
    return np.array(w, dtype=np.uint64).T.copy()


def _fp_rank_binary():
    here = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernels")
    src = _os.path.join(here, "fp_rank.cpp")
    cache = _os.environ.get("FPN_RANK_BIN") or _os.path.join(_tempfile.gettempdir(), "fpn_fp_rank_v1")
    if _os.path.exists(cache) and _os.path.getmtime(cache) >= _os.path.getmtime(src):
        return cache
    cxx = _shutil.which("c++") or _shutil.which("g++") or _shutil.which("clang++")
    if cxx is None:
        return None
    _subprocess.run([cxx, "-O3", "-std=c++17", "-pthread", src, "-o", cache], check=True, capture_output=True)
    return cache


def fp_rank_split(matrix, nb):
    """(rank(B), rank([B | A])) over F_p of a (rows, cols) uint64 matrix whose first nb columns are B."""
    import numpy as np
    matrix = np.ascontiguousarray(matrix, dtype=np.uint64)
    exe = _fp_rank_binary()
    if exe is None:
        return _fp_rank_split_numpy(matrix, nb)
    with _tempfile.NamedTemporaryFile(suffix=".fprank", delete=False) as f:
        np.array([_RANK_MAGIC, matrix.shape[0], matrix.shape[1], nb], dtype=np.uint64).tofile(f)
        matrix.tofile(f)
        path = f.name
    try:
        out = _subprocess.run([exe, path], check=True, capture_output=True, text=True).stdout.split()
    finally:
        _os.unlink(path)
    return int(out[0]), int(out[1])


def _fp_rank_split_numpy(matrix, nb):
    """Pure NumPy fallback (small matrices only)."""
    import numpy as np
    from flashprove_next.field import CPU
    a = matrix.copy()
    rows, cols = a.shape
    rank = rank_b = 0
    for col in range(cols):
        if rank == rows:
            break
        nz = np.nonzero(a[rank:, col])[0]
        if not len(nz):
            continue
        piv = rank + int(nz[0])
        a[[rank, piv]] = a[[piv, rank]]
        inv = pow(int(a[rank, col]), P - 2, P)
        a[rank] = CPU.mul(a[rank], np.uint64(inv))
        below = a[rank + 1:]
        fct = below[:, col:col + 1].copy()
        below[:] = CPU.sub(below, CPU.mul(fct, a[rank][None, :]))
        rank += 1
        if col < nb:
            rank_b = rank
    return rank_b, rank


def _rank_condition(layout, rows, b_spec, a_spec, coords):
    import numpy as np
    blocks = []
    labels = []
    spec = b_spec + a_spec
    for label, row in rows:
        blocks.append(_project(row, spec, coords))
        labels.append(label)
    matrix = np.concatenate(blocks, axis=0)
    nb = _spec_width(b_spec)
    rank_b, rank_ab = fp_rank_split(matrix, nb)
    return {"observations": len(labels), "fp_rows": int(matrix.shape[0]), "b_columns": nb,
            "a_columns": _spec_width(a_spec), "rank_B": rank_b, "rank_BA": rank_ab, "holds": rank_b == rank_ab}


def _claim_rows(layout, rec):
    for label, blocks, sparse in claim_functionals(layout, rec):
        yield label, _dense(layout, blocks, sparse)


def air_observations(layout, rec, claims=True):
    """[(label, dense (3, 2^n) row)] of the honest linear view: claims (optional) followed by the opening."""
    out = list(_claim_rows(layout, rec)) if claims else []
    return out + list(pcs_functionals(layout, rec))


def _sample(seq, count, rnd):
    seq = list(seq)
    return seq if count >= len(seq) else sorted(rnd.sample(seq, count))


def require_joint_mask_air(layout, table, rec, mask_vars, seed=0, rows=None, raise_on_fail=True, oversample=3):
    """im(A) <= im(B) over F_p for the joint linear view (claims + opening).

    A = witness coordinates (non-mask variables, multiplicities m_t, t < 128).  B = a subset of the mask
    coordinates (every mask variable and Libra / zerocheck slot, a sample of the GKR masks N, E, A, D, of the
    spare block and of the reserved half); im A <= im B_subset implies im A <= im B, so subsampling B can only
    make the check fail, never pass wrongly.  Returns the rank report; raises ValueError("under-rank mask")."""
    coords = Coordinates(layout, table, mask_vars)
    rows = rows if rows is not None else air_observations(layout, rec)
    rnd = _random.Random(seed)
    budget = int(oversample * len(rows)) + 64
    b_spec = [("var", coords.mask_vars), ("fp3", coords.libra),
              ("fp3", _sample(coords.gkr_masks + coords.spare, budget, rnd)),
              ("fp3", _sample(coords.reserved, budget, rnd))]
    a_spec = [("var", coords.witness_vars), ("fp", coords.m_cells)]
    report = _rank_condition(layout, rows, b_spec, a_spec, coords)
    report["check"] = "joint"
    if raise_on_fail and not report["holds"]:
        raise ValueError(f"under-rank mask: {report}")
    return report


def require_pcs_mask_air(layout, table, rec, mask_vars, seed=0, rows=None, raise_on_fail=True, strict=False, oversample=3):
    """Opening-only check with B = the cells NO other message touches (reserved half minus Libra / zerocheck
    slots, plus the spare block [7K, 8K)).

    strict=False: A = witness coordinates; every mask used by a non-linear message (mask variables, N, E, A,
    D, Libra slots) is dropped, i.e. treated as KNOWN -- the opening hides the witness even from an observer
    who learns all those masks.  strict=True: A = every non-B coordinate (the relation-3-style condition)."""
    coords = Coordinates(layout, table, mask_vars)
    rows = rows if rows is not None else list(pcs_functionals(layout, rec))
    rnd = _random.Random(seed)
    budget = int(oversample * len(rows)) + 64
    b_spec = [("fp3", _sample(coords.reserved, budget, rnd)), ("fp3", coords.spare)]
    a_spec = [("var", coords.witness_vars), ("fp", coords.m_cells)]
    if strict:
        a_spec = [("var", list(range(layout.K))), ("fp", coords.m_cells), ("fp3", coords.libra),
                  ("fp3", coords.gkr_masks)]
    report = _rank_condition(layout, rows, b_spec, a_spec, coords)
    report["check"] = "pcs_strict" if strict else "pcs_witness"
    if raise_on_fail and not report["holds"]:
        raise ValueError(f"under-rank mask: {report}")
    return report
