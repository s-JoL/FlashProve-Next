# SPDX-License-Identifier: Apache-2.0
"""Sampling rule v3: bucketed keyed sampling (specs/06 §3) with the Poseidon2 counter-mode PRF (specs/03 §5.3–§5.4).

This is the *reference* implementation (NumPy, exact integer arithmetic).  The batched torch / Triton engine
(``flashprove_next.sampling_v3_torch``) must agree with it bit for bit on the same fp32 logits.

Rule  Samp^bk_{beta}  (RULE_ID, committed per model_id with temperature T, top-p p_top, bucket size beta)
-------------------------------------------------------------------------------------------------------------
Inputs per generation step: fp32 logits z in R^V, allowed set A (constrained-decoding mask; -inf logits count as
masked), the per-request PRF prefix state (K_prf absorbed with the sampling context ctx, see ``SampCtx``), role tag
d_t and step index t.

 1. weights (integer, exact)      m = max_{v in A} z_v                                     (fp32, exact)
                                  j_v = floor( fl64( fl64(m - z_v) * c ) ),  c = fl64(2^EXP_K / T)
                                  w_v = E[min(j_v, J)] for v in A, else 0                  (E[J] = 0)
    E[j] = round(2^W_FRAC exp(-j / 2^EXP_K)) is a committed table (correctly rounded); J = first j with E[j] = 0.
    Every float operation is a single IEEE-754 binary64 operation (sub, mul, floor), so NumPy and CUDA agree.
 2. top-p threshold (no sorting)  S = sum_v w_v,  target(S) = ceil(S * P16 / 2^16),  P16 = round(p_top 2^16)
                                  theta* = max{ theta : sum_{w_v >= theta} w_v >= target(S) },  N = {v : w_v >= theta*}
    (all ties at the threshold are included -- the committed threshold form of specs/06 §3.1 step 2).
 3. buckets                       W_b = sum_{v in b cap N} w_v,  b = [b beta, (b+1) beta)  (V padded with zeros)
 4. bucket choice (Gumbel-max)    key_b = LN(W_b) + G[idx_b] for W_b > 0;  b* = argmax key_b, ties -> smallest b
    idx_b: PRF call j = b // 16 with label BKT gives 8 field elements e_0..e_7; bucket 16 j + 2 i + h uses bits
    [24 h, 24 h + 24) of e_i (the low 48 bits split into two 24-bit indices).  G[i] = round(2^G_FRAC *
    -ln(-ln((2 i + 1) / 2^25))) is the committed discrete Gumbel table; LN is the committed fixed-point natural log
    (``ln_fx``: exponent + 16-bit mantissa table, monotone).
 5. inverse CDF inside b*         u48 = (e_0 of the PRF call with label INV) mod 2^48,  U = u48 / 2^48
                                  tau = floor(u48 W_{b*} / 2^48);  token = first v in b* (index order) with S_v > tau,
                                  S_v = prefix sum of w over b* cap N.  (S_{v-1} <= U W < S_v  <=>  S_{v-1} <= tau < S_v.)

The distribution of the rule is, by definition, the distribution of this discrete process when the PRF outputs are
uniform field elements (specs/06 §3.1); with a continuous Gumbel and real arithmetic it is exactly softmax(z / T) truncated
to the nucleus.  Exactness gap of the discretisation: exp quantisation <= 2^-EXP_K / T in log-weight, Gumbel table
step 2^-24, LN mantissa truncation <= 2^-15 relative -- all part of the committed rule, none is a tolerance.

PRF (specs/03 §5.4, single ideal-permutation model): Poseidon2 (Goldilocks, width 12, rate 8, capacity 4).
    state_0 = (0^8 || K_prf);  absorb the fixed-length encoding Enc(DOM_PRF, ctx) block by block (add to the rate,
    permute) -> per-request prefix state;  per call: add (d_t, t, label, j, 0, 0, 0, 0) to the rate, permute, output
    the 8 rate elements.  This is the keyed sponge on the injective fixed-length message Enc(ctx) || (d_t, t, label, j);
    the shared ctx prefix is absorbed once per request (an implementation detail -- same outputs).
    Per step: ceil(B / 16) + 1 permutations (beta = 32, V = 151936: 297 + 1).
Nonce (specs/03 §5.3): ctx contains the client nonce N_c (16..32 bytes) and a fresh 256-bit server random N_s
(``server_nonce()``: CSPRNG, no counter); H_out is a separate primitive and does not appear here.

Verifier side (specs/06 §3.2): ``acceptable_set`` / ``accept`` take the *reference* logits z (the lmhead block output) and a
certified radius r (scalar or per-entry; only max_v r_v is used -- the global band of 06), and accept exactly the
tokens an engine whose logits satisfy |z_hat - z| <= r could have produced under this rule (a superset of them).
"""
import hashlib
import secrets
from dataclasses import dataclass, field
from decimal import Decimal, getcontext

import numpy as np

from .field import P
from . import poseidon2 as PS

RULE_ID = "fpn-samp-bk-v3"
BETA_DEFAULT = 32
EXP_K = 12                 # exp-table input resolution 2^-12 (log-weight units)
W_FRAC = 40                # weights are integers, max weight 2^40
G_FRAC = 24                # Gumbel table and LN output resolution 2^-24
G_BITS = 24                # Gumbel table index bits (2^24 entries)
LN_MBITS = 16              # LN mantissa bits
U_BITS = 48                # inverse-CDF uniform resolution
P_BITS = 16                # top-p resolution
NOISE_PER_CALL = 16        # 8 output elements x 2 indices
LBL_BKT = 1
LBL_INV = 2
TIE_SHIFT = 18             # bucket index bits for the deterministic tie break (B < 2^18)
M48 = (1 << 48) - 1
M24 = (1 << 24) - 1


def _dom(tag):
    return int.from_bytes(hashlib.sha256(tag).digest()[:4], "little")


DOM_PRF = _dom(b"fpn/prf/samp/v3")        # first rate element of every sampling-PRF message
DOM_XI = _dom(b"fpn/ctx/samp/v3")         # dom_xi inside ctx (specs/03 §5.3)
CTX_LEN = 39                                 # encoded ctx elements (incl. DOM_PRF and the length element)
CTX_BLOCKS = (CTX_LEN + PS.RATE - 1) // PS.RATE


class SamplingError(ValueError):
    """Rule failure branch (specs/01 §5.2 / §4.3): empty allowed set or a non-finite allowed logit."""


# ------------------------------------------------------------------------------------------------ committed tables
def _round_correct(x_f64, exact_fn, window):
    """Round x (float64 approximation of a real) to the nearest integer; entries within ``window`` of a .5 tie are
    recomputed with ``exact_fn(i)`` (Decimal, 50 digits) so the table does not depend on the platform's libm."""
    r = np.floor(x_f64 + 0.5)
    frac = x_f64 - np.floor(x_f64)
    near = np.nonzero(np.abs(frac - 0.5) < window)[0]
    for i in near:
        r[i] = float(int(exact_fn(int(i)).to_integral_value(rounding="ROUND_HALF_EVEN")))
    return r.astype(np.int64), len(near)


_TABLES = {}


def exp_table():
    """E[j] = round(2^W_FRAC exp(-j / 2^EXP_K)), j = 0..J (E[J] = 0 is the sentinel for 'no weight')."""
    if "E" not in _TABLES:
        getcontext().prec = 50
        jmax = int(np.ceil((W_FRAC + 1) * np.log(2) * (1 << EXP_K))) + 8
        j = np.arange(jmax, dtype=np.float64)
        x = np.exp(-j / (1 << EXP_K)) * float(1 << W_FRAC)
        two = Decimal(2) ** W_FRAC
        E, _ = _round_correct(x, lambda i: two * (Decimal(-i) / Decimal(1 << EXP_K)).exp(), 1e-3)
        J = int(np.argmax(E == 0))
        assert E[J] == 0 and E[J - 1] >= 1 and np.all(np.diff(E) <= 0)
        _TABLES["E"] = np.ascontiguousarray(E[:J + 1])
        # tie end: te[j] = max{j' : E[j'] = E[j]}  (E is non-increasing; equal runs appear in the tail)
        Ej = _TABLES["E"][:J]
        te = np.empty(J, dtype=np.int64)
        last = J - 1
        for i in range(J - 1, -1, -1):
            if i < J - 1 and Ej[i] != Ej[i + 1]:
                last = i
            te[i] = last
        _TABLES["TE"] = te
    return _TABLES["E"]


def exp_J():
    return len(exp_table()) - 1


def tie_end():
    exp_table()
    return _TABLES["TE"]


def gumbel_table():
    """G[i] = round(2^G_FRAC * -ln(-ln((2i + 1) / 2^(G_BITS + 1)))), i in [0, 2^G_BITS), int32."""
    if "G" not in _TABLES:
        getcontext().prec = 50
        n = 1 << G_BITS
        u = (2.0 * np.arange(n, dtype=np.float64) + 1.0) / float(1 << (G_BITS + 1))
        x = -np.log(-np.log(u)) * float(1 << G_FRAC)
        scale = Decimal(2) ** G_FRAC
        den = Decimal(2) ** (G_BITS + 1)

        def exact(i):
            ud = Decimal(2 * i + 1) / den
            return -scale * (-(ud.ln())).ln()
        G, _ = _round_correct(x, exact, 1e-5)
        _TABLES["G"] = G.astype(np.int32)
    return _TABLES["G"]


def ln_table():
    """LNT[m - 2^15] = round(2^G_FRAC ln(m / 2^15)), m in [2^15, 2^16); LN2 = round(2^G_FRAC ln 2)."""
    if "LN" not in _TABLES:
        getcontext().prec = 50
        h = 1 << (LN_MBITS - 1)
        m = np.arange(h, 2 * h, dtype=np.float64)
        x = np.log(m / h) * float(1 << G_FRAC)
        scale = Decimal(2) ** G_FRAC
        LNT, _ = _round_correct(x, lambda i: scale * (Decimal(h + i) / Decimal(h)).ln(), 1e-6)
        _TABLES["LN"] = LNT
        _TABLES["LN2"] = int((scale * Decimal(2).ln()).to_integral_value(rounding="ROUND_HALF_EVEN"))
        assert LNT[-1] < _TABLES["LN2"]          # monotone across exponent boundaries
    return _TABLES["LN"], _TABLES["LN2"]


def table_digests():
    """SHA-256 of each committed table (little-endian bytes) -- the values the rule commits to."""
    E = exp_table(); G = gumbel_table(); LNT, LN2 = ln_table()
    return {"exp": hashlib.sha256(E.astype("<i8").tobytes()).hexdigest(),
            "gumbel": hashlib.sha256(G.astype("<i4").tobytes()).hexdigest(),
            "ln": hashlib.sha256(LNT.astype("<i8").tobytes() + int(LN2).to_bytes(8, "little")).hexdigest(),
            "J": exp_J(), "LN2": int(LN2)}


def bitlen_floor_log2(x):
    """floor(log2 x) for an int64 array x >= 1 (integer shifts only; same steps in the torch engine)."""
    y = np.array(x, dtype=np.int64, copy=True)
    e = np.zeros_like(y)
    for s in (32, 16, 8, 4, 2, 1):
        t = y >> s
        mk = t > 0
        e += s * mk
        y = np.where(mk, t, y)
    return e


def ln_fx(W):
    """Committed fixed-point natural log of integers W >= 1: e * LN2 + LNT[mantissa16 - 2^15] (2^-24 units).
    Monotone non-decreasing in W (needed by the relaxation)."""
    LNT, LN2 = ln_table()
    W = np.asarray(W, dtype=np.int64)
    e = bitlen_floor_log2(W)
    sh = LN_MBITS - 1
    m = np.where(e >= sh, W >> np.maximum(e - sh, 0), W << np.maximum(sh - e, 0))
    return e * LN2 + LNT[m - (1 << sh)]


# ------------------------------------------------------------------------------------------------ rule parameters
@dataclass(frozen=True)
class Rule:
    """The committed per-model rule: temperature T > 0, top-p in (0, 1], bucket size beta (power of two)."""
    temperature: float = 1.0
    top_p: float = 1.0
    beta: int = BETA_DEFAULT

    def __post_init__(self):
        if not (self.temperature > 0 and np.isfinite(self.temperature)):
            raise ValueError("temperature must be finite and > 0")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.beta < 1 or self.beta & (self.beta - 1):
            raise ValueError("beta must be a power of two")
        if self.p16 < 1:
            raise ValueError("top_p rounds to 0 at 16-bit resolution")

    @property
    def c(self):
        return float(1 << EXP_K) / float(self.temperature)       # one IEEE division (committed as a float64)

    @property
    def T_bits(self):
        return int(np.float64(self.temperature).view(np.uint64)) % P

    @property
    def p16(self):
        return int(round(self.top_p * (1 << P_BITS)))

    def n_buckets(self, V):
        return (V + self.beta - 1) // self.beta

    def n_bkt_calls(self, V):
        return (self.n_buckets(V) + NOISE_PER_CALL - 1) // NOISE_PER_CALL

    def as_dict(self):
        return {"rule": RULE_ID, "temperature": self.temperature, "top_p": self.top_p, "p16": self.p16,
                "beta": self.beta, "c": self.c, "EXP_K": EXP_K, "W_FRAC": W_FRAC, "G_FRAC": G_FRAC, "U_BITS": U_BITS}


def rule_from_generation_config(path, beta=BETA_DEFAULT, default_temperature=1.0):
    """Fixed per-model rule from a Hugging Face generation_config.json (temperature, top_p; top_k is not part of
    rule v3 and is reported back so the caller can record the deviation)."""
    import json
    cfg = json.load(open(path))
    r = Rule(float(cfg.get("temperature", default_temperature)), float(cfg.get("top_p", 1.0)), beta)
    return r, {"ignored_top_k": cfg.get("top_k")}


# ------------------------------------------------------------------------------------------------ context and PRF
def server_nonce():
    """Fresh 256-bit server randomness N_s for one request attempt (CSPRNG, never a counter; specs/03 §5.3)."""
    return secrets.token_bytes(32)


def _limbs32(b, n=8):
    b = bytes(b)
    assert len(b) <= 4 * n
    b = b + b"\x00" * (4 * n - len(b))
    return [int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(n)]


@dataclass(frozen=True)
class SampCtx:
    """ctx = Enc(dom_xi, v_rel, e, dig(x_pub), N_c, N_s, sess, i_turn) (specs/01 §5.3 / specs/03 §5.3).  Byte strings are
    encoded as fixed-width 32-bit limbs (N_c with an explicit length), integers as single elements < 2^32."""
    client_nonce: bytes
    server_nonce: bytes
    dig: bytes = b"\x00" * 32          # digest of the public request core (computed outside the circuit)
    v_rel: int = 3
    epoch: int = 0
    sess: bytes = b"\x00" * 32
    turn: int = 0

    def __post_init__(self):
        if not (16 <= len(self.client_nonce) <= 32):
            raise ValueError("client nonce must be 16..32 bytes")
        if len(self.server_nonce) != 32:
            raise ValueError("server nonce must be 32 bytes")
        if len(self.dig) != 32 or len(self.sess) != 32:
            raise ValueError("dig and sess must be 32 bytes")
        for x in (self.v_rel, self.epoch, self.turn):
            if not (0 <= int(x) < (1 << 32)):
                raise ValueError("v_rel / epoch / turn must fit in 32 bits")

    def encode(self):
        el = [DOM_PRF, CTX_LEN, DOM_XI, int(self.v_rel), int(self.epoch)]
        el += _limbs32(self.dig)
        el += [len(self.client_nonce)] + _limbs32(self.client_nonce)
        el += _limbs32(self.server_nonce)
        el += _limbs32(self.sess)
        el += [int(self.turn)]
        assert len(el) == CTX_LEN and all(0 <= x < P for x in el)
        return el + [0] * (CTX_BLOCKS * PS.RATE - CTX_LEN)


def kprf_elems(k):
    """K_prf as 4 canonical field elements (from 4 ints or 32 bytes)."""
    if isinstance(k, (bytes, bytearray)):
        assert len(k) == 32
        k = [int.from_bytes(k[8 * i:8 * i + 8], "little") for i in range(4)]
    k = [int(x) % P for x in k]
    assert len(k) == 4
    return k


def prefix_state(kprf, ctx):
    """Keyed-sponge state after absorbing Enc(ctx): 12 field elements (secret; one per request)."""
    st = np.zeros((PS.T, 1), dtype=np.uint64)
    st[PS.RATE:, 0] = kprf_elems(kprf)
    enc = ctx.encode()
    for b in range(CTX_BLOCKS):
        blk = enc[PS.RATE * b:PS.RATE * (b + 1)]
        for i in range(PS.RATE):
            st[i, 0] = np.uint64((int(st[i, 0]) + blk[i]) % P)
        st = PS.permute(st)
    return [int(x) for x in st[:, 0]]


def prf_calls(prefix, d_t, ts, n_bkt_calls):
    """PRF outputs for steps ``ts``: uint64 array (len(ts), n_bkt_calls + 1, 8); the last call is the INV call."""
    ts = np.atleast_1d(np.asarray(ts, dtype=np.int64))
    nc = n_bkt_calls + 1
    n = len(ts) * nc
    st = np.empty((PS.T, n), dtype=np.uint64)
    for i in range(PS.T):
        st[i] = np.uint64(prefix[i])
    c = np.tile(np.arange(nc, dtype=np.int64), len(ts))
    tt = np.repeat(ts, nc)
    lab = np.where(c == nc - 1, LBL_INV, LBL_BKT)
    jj = np.where(c == nc - 1, 0, c)
    add = [np.full(n, int(d_t) % P, dtype=np.uint64), tt.astype(np.uint64), lab.astype(np.uint64), jj.astype(np.uint64)]
    for i in range(4):
        st[i] = _addp(st[i], add[i])
    out = PS.permute(st)
    return np.ascontiguousarray(out[:PS.RATE].T.reshape(len(ts), nc, PS.RATE))


def _addp(a, b):
    a = a.astype(np.uint64); b = b.astype(np.uint64)
    s = a + b
    s = np.where(s < a, s + np.uint64(0xFFFFFFFF), s)
    return np.where(s >= np.uint64(P), s - np.uint64(P), s)


def noise_indices(outs, B):
    """outs (..., nc, 8) uint64 -> (Gumbel table indices (..., B) int64, u48 (...) int64)."""
    e = outs[..., :-1, :]
    lo = (e & np.uint64(M24)).astype(np.int64)
    hi = ((e >> np.uint64(24)) & np.uint64(M24)).astype(np.int64)
    idx = np.stack([lo, hi], axis=-1).reshape(*e.shape[:-2], -1)[..., :B]
    u48 = (outs[..., -1, 0] & np.uint64(M48)).astype(np.int64)
    return idx, u48


# ------------------------------------------------------------------------------------------------ the rule
def allowed_vector(V, allowed=None, logits=None):
    a = np.ones(V, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool).copy()
    if logits is not None:
        a &= ~(np.asarray(logits) == -np.inf)
    return a


def quantize(z32, allowed, rule):
    """Step 1 inputs: j_v (int64, J for masked / underflowed entries) and m.  Raises SamplingError on the failure
    branch (empty allowed set, NaN / +inf allowed logit)."""
    z = np.asarray(z32, dtype=np.float32).astype(np.float64)
    a = allowed_vector(len(z), allowed, z)
    if not a.any():
        raise SamplingError("empty allowed set")
    if not np.all(np.isfinite(z[a])):
        raise SamplingError("non-finite allowed logit")
    J = exp_J()
    m = z[a].max()
    with np.errstate(invalid="ignore", over="ignore"):
        x = (m - z) * rule.c
    x = np.where(a, x, float(J))
    j = np.floor(np.minimum(x, float(J))).astype(np.int64)
    return j, m, a


def weights_from_j(j):
    return exp_table()[np.minimum(j, exp_J())]


def target_mass(S, p16):
    return -((-int(S) * int(p16)) >> P_BITS)          # ceil(S p16 / 2^16), exact Python int


def threshold(w, target):
    """theta = max{theta : sum_{w >= theta} w >= target} by bisection on the 41-bit weight domain (no sorting)."""
    th = 0
    for bit in range(W_FRAC, -1, -1):
        c = th | (1 << bit)
        if int(w[w >= c].sum()) >= target:
            th = c
    return th


def nucleus_weights(w, rule):
    if rule.p16 >= (1 << P_BITS):
        return w.copy(), 1
    th = threshold(w, target_mass(int(w.sum()), rule.p16))
    return np.where(w >= th, w, 0), th


def bucket_masses(wN, beta):
    V = len(wN)
    B = (V + beta - 1) // beta
    pad = np.zeros(B * beta, dtype=np.int64)
    pad[:V] = wN
    return pad.reshape(B, beta), pad.reshape(B, beta).sum(axis=1)


def bucket_keys(W, idx):
    keys = np.full(len(W), np.iinfo(np.int64).min, dtype=np.int64)
    pos = W > 0
    keys[pos] = ln_fx(W[pos]) + gumbel_table()[idx[pos]].astype(np.int64)
    return keys


def mulshift48(u48, W):
    return (int(u48) * int(W)) >> U_BITS


def sample_from(z32, rule, idx, u48, allowed=None, info=False):
    """The rule given precomputed PRF indices (idx (B,), u48).  Returns the token (and a debug dict)."""
    j, m, a = quantize(z32, allowed, rule)
    w = weights_from_j(j)
    wN, th = nucleus_weights(w, rule)
    blocks, W = bucket_masses(wN, rule.beta)
    keys = bucket_keys(W, idx)
    b = int(np.argmax(keys))                           # first maximum = smallest index among ties
    S = np.cumsum(blocks[b])
    tau = mulshift48(u48, W[b])
    v = int(np.searchsorted(S, tau, side="right"))
    tok = b * rule.beta + v
    assert a[tok] and wN[tok] > 0
    if info:
        return tok, {"bucket": b, "theta": int(th), "W_b": int(W[b]), "tau": tau, "u48": int(u48), "m": float(m),
                     "nucleus_size": int((wN > 0).sum())}
    return tok


def sample(prefix, t, z32, rule, allowed=None, d_t=0, info=False):
    """Token of generation step t: deterministic in (K_prf, ctx) [via ``prefix``], d_t, t and the fp32 logits."""
    V = len(z32)
    outs = prf_calls(prefix, d_t, [t], rule.n_bkt_calls(V))[0]
    idx, u48 = noise_indices(outs, rule.n_buckets(V))
    return sample_from(z32, rule, idx, int(u48), allowed, info)


def sample_many(prefix, ts, z32, rule, allowed=None, d_t=0):
    """Same logits row at many steps (one batched PRF evaluation) -- used by the distribution tests."""
    V = len(z32)
    outs = prf_calls(prefix, d_t, ts, rule.n_bkt_calls(V))
    idx, u48 = noise_indices(outs, rule.n_buckets(V))
    j, m, a = quantize(z32, allowed, rule)
    wN, _ = nucleus_weights(weights_from_j(j), rule)
    blocks, W = bucket_masses(wN, rule.beta)
    pos = W > 0
    lnW = np.full(len(W), 0, dtype=np.int64)
    lnW[pos] = ln_fx(W[pos])
    G = gumbel_table()
    S = np.cumsum(blocks, axis=1)
    toks = np.empty(len(ts), dtype=np.int64)
    for k in range(len(ts)):
        keys = np.where(pos, lnW + G[idx[k]], np.iinfo(np.int64).min)
        b = int(np.argmax(keys))
        tau = mulshift48(u48[k], W[b])
        toks[k] = b * rule.beta + int(np.searchsorted(S[b], tau, side="right"))
    return toks


def rule_distribution(z32, rule, allowed=None):
    """Probabilities w_v / sum_N w of the rule with ideal (uniform) noise -- the target of the chi-square test."""
    j, m, a = quantize(z32, allowed, rule)
    wN, _ = nucleus_weights(weights_from_j(j), rule)
    return wN.astype(np.float64) / float(wN.sum())


# ------------------------------------------------------------------------------------------------ verifier: relaxation
def band_shift(r_max, rule):
    """Delta (in exp-table steps) such that every admissible engine j_hat lies in [j - Delta, j + Delta].
    |m_hat - m| <= R and |z_hat_v - z_v| <= R give |(m_hat - z_hat_v) - (m - z_v)| <= 2R; +1 for each of the two
    floors (engine and verifier), float64 rounding of both is < 1e-6 steps."""
    return int(np.ceil(2.0 * float(r_max) * rule.c + 1e-6)) + 2


@dataclass
class Relaxed:
    accept: np.ndarray                  # bool (V,): tokens the relaxed predicate accepts at this step
    delta: int
    theta_lo: int
    theta_hi: int
    n_lo: int
    n_hi: int
    buckets_passing: list = field(default_factory=list)


def _sum_above(keys, vals, x):
    """For each x_v: sum of vals_u over u with keys_u > x_v (exact int64; keys sorted once)."""
    o = np.argsort(keys, kind="stable")
    ks, cs = keys[o], np.concatenate([[0], np.cumsum(vals[o])])
    i = np.searchsorted(ks, x, side="right")
    return cs[-1] - cs[i]


def _sum_at_most(keys, vals, x):
    o = np.argsort(keys, kind="stable")
    ks, cs = keys[o], np.concatenate([[0], np.cumsum(vals[o])])
    return cs[np.searchsorted(ks, x, side="right")]


def nucleus_bounds_tight(w_lo, w_hi, p16):
    """Sets N^lo (surely in the engine's nucleus) and N^hi (possibly in it) for every engine weight vector with
    w_lo <= w_hat <= w_hi.  v is in the engine nucleus iff sum_{w_hat_u > w_hat_v} w_hat_u < target(S_hat), i.e.
    (with integer sums) above_v (2^16 - p16) < rest_v p16, rest_v = S_hat - above_v.  Bounds:
      sure:     A_hi(v) = sum_{u != v, w_hi_u > w_lo_v} w_hi_u,  R_lo(v) = w_lo_v + sum_{u != v, w_hi_u <= w_lo_v} w_lo_u
      possible: A_lo(v) = sum_{w_lo_u > w_hi_v} w_lo_u,          R_hi(v) = sum_{w_lo_u <= w_hi_v} w_hi_u
    (The literal bound N^lo = {w_lo >= theta_hi} would always drop the threshold token itself.)"""
    if p16 >= (1 << P_BITS):
        return w_lo > 0, w_hi > 0
    q = (1 << P_BITS) - p16
    A_hi = _sum_above(w_hi, w_hi, w_lo) - np.where(w_hi > w_lo, w_hi, 0)
    R_lo = _sum_at_most(w_hi, w_lo, w_lo) + np.where(w_hi <= w_lo, 0, w_lo)
    A_lo = _sum_above(w_lo, w_lo, w_hi)
    R_hi = _sum_at_most(w_lo, w_hi, w_hi)
    big = lambda x: x.astype(object)                                     # noqa: E731  (exact > 2^63 products)
    sure = (w_lo > 0) & (big(A_hi) * q < big(R_lo) * p16).astype(bool)
    poss = (w_hi > 0) & (big(A_lo) * q < big(R_hi) * p16).astype(bool)
    return sure, poss


def acceptable_set(prefix, t, z_ref, r, rule, allowed=None, d_t=0, nucleus="tight"):
    """Relaxed acceptance (specs/06 §3.2) for reference logits z_ref and certified radius r (scalar or per entry; the
    global band uses max r).  Returns a ``Relaxed`` whose ``accept`` marks every token the verifier accepts.
    nucleus="tight" (default) uses ``nucleus_bounds_tight``; nucleus="06" uses the literal bound N^lo = {w_lo >= theta_hi}."""
    z = np.asarray(z_ref, dtype=np.float64)
    V = len(z)
    a = allowed_vector(V, allowed, z)
    if not a.any() or not np.all(np.isfinite(z[a])):
        return Relaxed(np.zeros(V, dtype=bool), 0, 0, 0, 0, 0)
    rmax = float(np.max(np.broadcast_to(np.asarray(r, dtype=np.float64), z.shape)[a]))
    assert 2.0 * rmax * rule.c < float(1 << 52), "outside binary64 step-1 domain (2 R c >= 2^52)"
    D = band_shift(rmax, rule)
    J = exp_J()
    E = exp_table()
    m = z[a].max()
    with np.errstate(invalid="ignore", over="ignore"):
        x = np.where(a, (m - z) * rule.c, float(J + D))
    j = np.floor(np.minimum(x, float(J + D))).astype(np.int64)
    w_hi = np.where(a, E[np.clip(j - D, 0, J)], 0)
    w_lo = np.where(a, E[np.minimum(j + D, J)], 0)
    S_hi, S_lo = int(w_hi.sum()), int(w_lo.sum())
    if rule.p16 >= (1 << P_BITS):
        th_lo = th_hi = 1
    else:
        tgt_hi, tgt_lo = target_mass(S_hi, rule.p16), target_mass(S_lo, rule.p16)
        th_lo = threshold(w_lo, tgt_hi) if S_lo >= tgt_hi else 1        # 1 = every positive weight
        th_lo = max(th_lo, 1)
        th_hi = max(threshold(w_hi, tgt_lo), 1)
    if nucleus == "06":
        in_lo, in_hi = w_lo >= th_hi, w_hi >= th_lo
    else:
        in_lo, in_hi = nucleus_bounds_tight(w_lo, w_hi, rule.p16)
    n_lo = np.where(in_lo & (w_lo > 0), w_lo, 0)            # N^lo with w^lo
    n_hi = np.where(in_hi & (w_hi > 0), w_hi, 0)            # N^hi with w^hi
    blo, Wlo = bucket_masses(n_lo, rule.beta)
    bhi, Whi = bucket_masses(n_hi, rule.beta)
    outs = prf_calls(prefix, d_t, [t], rule.n_bkt_calls(V))[0]
    idx, u48 = noise_indices(outs, rule.n_buckets(V))
    u48 = int(u48)
    G = gumbel_table().astype(np.int64)
    lo_keys = np.where(Wlo > 0, ln_fx(np.maximum(Wlo, 1)) + G[idx], np.iinfo(np.int64).min)
    hi_keys = np.where(Whi > 0, ln_fx(np.maximum(Whi, 1)) + G[idx], np.iinfo(np.int64).min)
    rhs = lo_keys.max()
    acc = np.zeros(rule.n_buckets(V) * rule.beta, dtype=bool)
    passing = [int(b) for b in np.nonzero((Whi > 0) & (hi_keys >= rhs))[0]]
    for b in passing:
        Shi = np.cumsum(bhi[b])
        Slo_prev = np.concatenate([[0], np.cumsum(blo[b])[:-1]])
        WH, WL = int(Whi[b]), int(Wlo[b])
        for k in range(rule.beta):
            if bhi[b, k] <= 0:
                continue
            c1 = int(Slo_prev[k]) << U_BITS <= u48 * WH
            c2 = u48 * WL < int(Shi[k]) << U_BITS
            acc[b * rule.beta + k] = c1 and c2
    acc = acc[:V] & a
    return Relaxed(acc, D, int(th_lo), int(th_hi), int((n_lo > 0).sum()), int((n_hi > 0).sum()), passing)


def accept(prefix, t, z_ref, r, rule, u, allowed=None, d_t=0, nucleus="tight"):
    """Verifier predicate S_t (specs/01 §5.4) for one generated token u."""
    rel = acceptable_set(prefix, t, z_ref, r, rule, allowed, d_t, nucleus)
    return bool(0 <= u < len(rel.accept) and rel.accept[u])
