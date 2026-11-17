# SPDX-License-Identifier: Apache-2.0
"""Reference arithmetic of the per-request trace commitment (specs/10 §2.1, §2.3; specs/06 §2.2 row 7 with the
exp-underflow term and the D-9 sum-of-squares bounds): lifting rule, tier geometry, the integer attention-slot witness and the public radius
constants.  Pure numpy + Python ints; shared by the prover (tc_zk) and the tests.

Integer attention slot (one decode slot, layer l, position t, full causal window tau <= t):
    q_int[h, i]  = round(q[h, i] * scale * 2^FQ)            (scaled query, local witness)
    K_int, V_int = Lift_K(K), Lift_V(V) = round(v * 2^F)     (entries of H)
    z[tau, h]    = sum_i q_int[h, i] K_int[tau, g(h), i]      (exact integer, |z| < 2^59)
    m[h] - z     = idx * 2^SH + rem - 2^(SH-1),  0 <= rem < 2^SH,  0 <= idx < 2^17      (unmasked keys)
                   i.e. idx = round(512 * x), x = (m - z) / 2^(FQ + FK) = scaled score gap
    E = EXPT[idx] = round(2^EB exp(-idx / 512)),  Ehi = ceil(E / 2^CB);  E = 0 on masked keys (and Ehi = 0 where E = 0)
    Z[h] = sum_tau E,  N[h, i] = sum_tau E V_int,  M2hi[h, i] = sum_tau Ehi V_int^2,
    Q2[g, i] = sum_{tau <= t} V_int^2   (causal mask: rows above t and padding rows never enter the radius)
    o_ref = N / Z.
Tolerance per output component (all quantities at scale 2^FO, FO = FV + 6):
    |o_eng - o_ref| <= A sqrt(Var) + B |o_ref| + Cq SQn + C0
    Var  <= kappa_hi M2hi 2^CB / (Z 2^(2 FV)) - kappa_lo o_ref^2                 (D-9 Jensen, E <= Ehi 2^CB pointwise)
    SQn  =  sqrt((t + 1) Q2) / (2 Z 2^FV)                                        (D-9 Cauchy-Schwarz, sum |v|)
Every per-key error term of the radius (exp-table rounding, Ehi ceiling, underflow) multiplies a VISIBLE key's v: E = 0 on
masked keys, so sum_{tau <= t} |v_tau| <= sqrt((t + 1) sum_{tau <= t} v_tau^2) = sqrt((t + 1) Q2) (Cauchy-Schwarz over the
t + 1 visible keys), and E <= ceil(E / 2^CB) 2^CB = Ehi 2^CB pointwise (with Ehi = 0 whenever E = 0, so zero-weight keys
cannot inflate M2hi).  Masking Q2 and range-checking Q2 < 2^(62 - nt) removes both future/padding-row inflation and
modulo-p wrap-around in (t + 1) Q2 + 1 < 2^62 + 1 < p/2.
LSE observation (M1, specs/06 §5.2): per query head h, |lse_eng[h] - L24[h] 2^-24| <= RL24 2^-24, where
L24[h] = floor(m[h] / 2^(FQ+FK-24)) + (e_Z[h] - EB) LN2 + LNT[mant16(Z[h]) - 2^15].
with A, B, Cq, C0 public constants of the tier (radius_constants)."""
import math
from dataclasses import dataclass, field

import numpy as np

EXP_R = 9                   # exp-table resolution 2^-9 in the scaled score gap x
EB = 24                     # E = round(2^EB exp(-x))
CB = 12                     # Ehi = E >> CB (12-bit weight for the sum of squares)


@dataclass(frozen=True)
class Tier:
    """Public profile + length-tier constants (specs/10 §2.1).  H rows are chunks of TC tokens, columns are
    (token in chunk, layer, port, group, dim) = TC * 2^16."""
    T_pad: int
    TC: int
    L_MAX: int = 32           # layer slots (Qwen3-0.6B uses 28; the rest are zero)
    G: int = 8                # KV groups (n_kv)
    R: int = 2                # query heads per group (n_q / n_kv)
    DH: int = 128
    MQ: int = 268             # opened columns (specs/10 D-6: rate 1/2, delta 0.24)
    FK: int = 17              # Lift_K scale (|K_int| < 2^27 for |K| < 1024)
    FV: int = 11              # Lift_V scale (|V_int| < 2^18 for |V| < 128); sets the bit budget of M2hi
    FQ: int = 24              # scaled-query scale
    # public caps used in the radius (profile constants)
    QN_CAP: float = 4.0       # max ||scale * q_h||_2
    KN_CAP: float = 512.0     # max ||k_tau||_2
    EXP_BITS: int = 17        # exp-table rows: idx < 2^EXP_BITS, i.e. scaled score gap x < 2^EXP_BITS / 512

    @property
    def nt(self): return self.T_pad.bit_length() - 1          # key bits

    @property
    def DB(self): return self.DH.bit_length() - 1

    @property
    def GB(self): return self.G.bit_length() - 1

    @property
    def RB(self): return self.R.bit_length() - 1

    @property
    def HB(self): return self.GB + self.RB                    # query-head bits, h = g R + rho

    @property
    def LB(self): return self.L_MAX.bit_length() - 1

    @property
    def B_G(self): return self.DB                             # bit positions inside the H index x (LSB first)

    @property
    def B_PORT(self): return self.DB + self.GB

    @property
    def B_L(self): return self.B_PORT + 1

    @property
    def B_TAU(self): return self.B_L + self.LB

    @property
    def VB(self): return self.B_TAU                           # log2 V_hist (Qwen tier: 7 + 3 + 1 + 5 = 16)

    @property
    def WB(self): return self.HB + self.DB                    # slot-vector segment bits

    @property
    def nc(self): return self.TC << self.VB

    @property
    def nr(self): return self.T_pad // self.TC

    @property
    def n(self): return 2 * self.nc                          # RS rate 1/2

    @property
    def depth(self): return self.n.bit_length() - 1

    @property
    def NH(self): return self.T_pad << self.VB

    @property
    def nH(self): return self.NH.bit_length() - 1            # variables of H~

    @property
    def SH(self): return self.FQ + self.FK - EXP_R             # remainder bits (32)

    @property
    def FO(self): return self.FV + 6


TIERS = {1024: Tier(1024, 8), 2048: Tier(2048, 8), 4096: Tier(4096, 16), 8192: Tier(8192, 16)}


# ============================================================================ H layout (specs/10 §2.1)
def h_index(tier, tau, layer, port, g, i):
    """flat index of H (row-major n_r x n_c): x = tau 2^VB + layer 2^B_L + port 2^B_PORT + g 2^B_G + i
    (Qwen tier: tau 2^16 + layer 2^11 + port 2^10 + g 2^7 + i)."""
    return (tau << tier.B_TAU) | (layer << tier.B_L) | (port << tier.B_PORT) | (g << tier.B_G) | i


def bf16(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def lift(x, F):
    """round-half-away of x * 2^F to int64 (x float32 / float64)."""
    y = np.asarray(x, dtype=np.float64) * float(2 ** F)
    return np.where(y >= 0, np.floor(y + 0.5), -np.floor(-y + 0.5)).astype(np.int64)


def build_H(tier, K, V):
    """K, V: (L, n_kv, T, dh) float.  Returns H as int32 (n_r, n_c) (unused layer slots and tau >= T are zero)."""
    L, G, T, D = K.shape
    assert G == tier.G and D == tier.DH and T <= tier.T_pad and L <= tier.L_MAX
    Kq, Vq = lift(K, tier.FK), lift(V, tier.FV)
    assert np.abs(Kq).max() < (1 << 27) and np.abs(Vq).max() < (1 << 18), (np.abs(Kq).max(), np.abs(Vq).max())
    H = np.zeros((tier.T_pad, tier.L_MAX, 2, G, D), dtype=np.int32)      # (tau, layer, port, g, i) = H index order
    H[:T, :L, 0] = Kq.transpose(2, 0, 1, 3)
    H[:T, :L, 1] = Vq.transpose(2, 0, 1, 3)
    return H.reshape(tier.nr, tier.nc)


_EXPT = {}


def expt(bits):
    """(E, Ehi) for idx < 2^bits: E = round(2^EB exp(-idx / 2^EXP_R)), Ehi = ceil(E / 2^CB) (0 where E = 0)."""
    if bits not in _EXPT:
        from decimal import Decimal
        from .certified import _dec_round_half_up
        idx = np.arange(1 << bits, dtype=np.float64)
        scale = Decimal(1 << EB)
        den = Decimal(1 << EXP_R)
        approx = np.exp(-idx / (1 << EXP_R)) * (1 << EB)
        E = _dec_round_half_up(approx, lambda k: scale * (Decimal(-int(k)) / den).exp())
        assert E[-1] == 0, "the last table row must have E = 0 (masked keys use it; needs EXP_BITS >= 14)"
        _EXPT[bits] = (E, (E + (1 << CB) - 1) >> CB)
    return _EXPT[bits]


# ============================================================================ slot witness
@dataclass
class Slot:
    tier: Tier
    layer: int
    t: int                      # slot position (keys 0..t visible)
    q_int: np.ndarray           # (16, 128) int64
    o_int: np.ndarray           # (16, 128) engine output at scale 2^FO
    K: np.ndarray               # (T_pad, 8, 128) int64 (from H, layer `layer`)
    V: np.ndarray
    tau_p: int = 0              # producer read position
    kp: np.ndarray = None       # (8, 128) producer's K output (int)
    vp: np.ndarray = None
    w: dict = field(default_factory=dict)


def slot_from_capture(tier, H, layer, q_bf16, o_bf16, t, scaling, tau_p):
    """Slot data from the dump: q, o (n_q, dh) uint16 bf16 codes of `layer`; H the committed history matrix."""
    Hv = H.reshape(tier.T_pad, tier.L_MAX, 2, tier.G, tier.DH)
    K = Hv[:, layer, 0].astype(np.int64)
    V = Hv[:, layer, 1].astype(np.int64)
    q = bf16(q_bf16).astype(np.float64) * scaling
    q_int = lift(q, tier.FQ)
    o_int = lift(bf16(o_bf16), tier.FO)
    return Slot(tier, layer, t, q_int, o_int, K, V, tau_p, K[tau_p].copy(), V[tau_p].copy())


def ihead(tier):
    """query head h = g R + rho -> KV group g."""
    return np.arange(tier.G * tier.R) // tier.R


def witness(s, mask_override=None, check=True):
    """Integer witness of the slot (int64 arrays; every quantity provably < 2^62, see the module docstring)."""
    tr = s.tier
    T, Hn = tr.T_pad, tr.G * tr.R
    gk = ihead(tr)
    Kh = s.K[:, gk, :]                                   # (T, 16, 128)
    z = np.einsum("hi,thi->th", s.q_int.astype(np.int64), Kh.astype(np.int64))     # |z| < 2^59: exact in int64
    mask = (np.arange(T) <= s.t).astype(np.int64) if mask_override is None else np.asarray(mask_override, dtype=np.int64)
    m = np.where(mask[:, None] == 1, z, np.iinfo(np.int64).min).max(axis=0)
    SH = tr.SH
    D = (m[None, :] - z) + (1 << (SH - 1))              # = idx 2^SH + rem on visible keys
    vis = mask[:, None] == 1
    if check and np.any(vis & (D < 0)):
        raise ValueError("m below a visible score")
    idx = np.where(vis, D >> SH, (1 << tr.EXP_BITS) - 1)
    rem = np.where(vis, D & ((1 << SH) - 1), 0)
    if check and idx.max() >= (1 << tr.EXP_BITS):
        raise ValueError("score gap beyond the exp table")
    idx = np.minimum(idx, (1 << tr.EXP_BITS) - 1)
    ET, EHT = expt(tr.EXP_BITS)
    E = ET[idx] * mask[:, None]; Ehi = EHT[idx] * mask[:, None]
    Z = E.sum(axis=0)
    Vh = s.V[:, gk, :].astype(np.int64)
    N = np.einsum("th,thi->hi", E, Vh)
    M2 = np.einsum("th,thi->hi", Ehi, Vh * Vh)
    Q2 = (mask[:, None, None] * s.V.astype(np.int64) ** 2).sum(axis=0)          # visible keys only
    w = dict(z=z, mask=mask, m=m, idx=idx, rem=rem, E=E, Ehi=Ehi, Z=Z, N=N, M2=M2, Q2=Q2)
    w.update(tolerance_witness(s, w))
    s.w = w
    return w


def isqrt_ceil(x):
    x = int(x)
    if x <= 0:
        return 0
    r = math.isqrt(x)
    return r if r * r == x else r + 1


# ============================================================================ radius constants (specs/06 §2.2 row 7)
def radius_constants(tr, W=16):
    """Public tier constants (A, B, Cq, C0) of |o_eng - o_ref| <= A sqrt(Var) + B |o_ref| + Cq SQn + C0 (v units) and
    the integer forms at precision 2^W.  Engine contract = FlashAttention-2 (SDPA) bf16, the AttnFA contract (fp32 accumulation, P stored bf16, exp2 with relative error 2^-21, B_c = 64 tiles)."""
    u32, ub, uP, uexp = 2.0 ** -24, 2.0 ** -8, 2.0 ** -8, 2.0 ** -21
    eta_exp = eta_P = 2.0 ** -126
    eta_acc, eta_b = 2.0 ** -149, 2.0 ** -133
    n = tr.T_pad
    nt = math.ceil(n / 64) + 8
    hPV = math.ceil(n / 16) + 64
    gam = (math.ceil(tr.DH / 16) + 64) * u32
    QK = tr.QN_CAP * tr.KN_CAP                                  # cap on |scale q|^T |k| and on |scale z|
    dx_k = 2.0 ** (-tr.FK - 1) * math.sqrt(tr.DH) * tr.QN_CAP     # engine bf16 K vs Lift_K (input box of the scores)
    dx_q = (0.5 + 2.0 ** -25) * 2.0 ** (-tr.FQ) * math.sqrt(tr.DH) * tr.KN_CAP   # q_int (incl. GL=33 guard) vs scale q
    rz = gam * QK + 4 * u32 * (2 * QK) + dx_k                   # score radius (specs/06 §2.2 row 2 + scaling + K lift)
    eps_s = math.exp(2 * rz) * (1 + uexp) ** (nt + 1) - 1
    eps_m = (1 + uP) * (1 + u32) ** (2 * nt + hPV) - 1
    eps_m2 = (1 + u32) ** (2 * nt + hPV) - 1
    eps_N = (1 + eps_s) * (1 + eps_m) - 1
    eps_D = (1 + eps_s) * (1 + eps_m2) - 1
    eta_w = (nt + 1) * (1 + eps_m) * eta_exp + eta_P
    eta_bar = math.exp(rz) * eta_w
    eta_v = 2.0 ** (-tr.FV - 1)
    D0 = 1 - eps_D - n * eta_bar
    k1 = (1 + ub) * (1 + u32) / D0
    c_o = (1 + eps_s) * (eps_m + eps_m2)
    # reference (integer exp table) error: x quantised to 2^-10 (round to nearest of 2^-9) + q_int rounding
    eps_r = math.exp(2.0 ** -(EXP_R + 1) + dx_q) - 1
    nz = n / 2.0 ** (EB + 1)                                    # n / (2 Z) <= T_pad / 2^(EB + 1)
    eps_r1 = eps_r * (1 + nz) / (1 - eps_r)
    lam = 1 / (1 - nz)
    a_J = k1 * eps_N + eps_r1
    a_o = k1 * (c_o + eta_bar * n) + (1 + ub) * u32 + ub + nz
    a_Q = k1 * eta_bar * 2 * n * 2.0 ** EB + 1                 # eta_bar * 2 Z <= eta_bar * 2 n 2^EB
    eta_o = 2.0 ** -tr.FO                                     # o_ref = floor(N 2^6 / Z) and o_eng rounding
    a_c = k1 * ((1 + eps_N) * eta_v + eta_bar * n * eta_v) + k1 * D0 * nt * eta_acc + eta_b + eta_o
    A = a_J + a_o * lam * eps_r1
    B = a_o * lam
    Cq = a_Q + a_o * lam
    C0 = a_c
    kap_hi = (1 + eps_r) / ((1 - eps_r) * (1 - nz))
    kap_lo = 1 - n / 2.0 ** EB
    # LSE certified radius (M1, natural log, scale 2^-24): engine online-softmax + log + fp32 add + reference EXPT/LN
    hB = math.ceil(64 / 16) + 64
    ea_lse = (1 + uexp) ** (nt + 1) * (1 + u32) ** (2 * nt + hB) - 1
    r_lse = ((rz + dx_q) - math.log(1.0 - ea_lse - n * eta_exp) + 2.0 * uexp * (math.log(n) + 1.0)
             + u32 * (QK + math.log(n) + 1.0) + 2.0 ** -(EXP_R + 1) - math.log(1.0 - nz) + 523.0 * 2.0 ** -24)
    rl24 = math.ceil(r_lse * (1 << 24)) + 1                     # +1 pays the observed-LSE magnitude floor at 2^-24
    ints = dict(A=math.ceil(A * 2 ** W), B=math.ceil(B * 2 ** W), Cq=math.ceil(Cq * 2 ** W),
                C0=math.ceil(C0 * 2 ** (W + tr.FO)), KH=math.ceil(kap_hi * 2 ** 12), KL=math.floor(kap_lo * 2 ** 12),
                W=W, RL24=rl24)
    return dict(A=A, B=B, Cq=Cq, C0=C0, kappa_hi=kap_hi, kappa_lo=kap_lo, eps_sigma=eps_s, eps_mu=eps_m, eps_N=eps_N,
                eps_D=eps_D, eps_r=eps_r, r_z=rz, r_lse=r_lse, eta_bar=eta_bar, n_t=nt, ints=ints)


FL_LSE = 24


def lse_witness(tr, w, lse_u32):
    """Per-head integer witness of the LSE check on HSp (scale 2^-24, matching sampling_v3.ln_table)."""
    from . import certified as CZ
    from . import sampling_v3 as S3
    CZ._f32hi_table()
    S_, Ahi_, E_ = CZ._TAB["f32hi_dec"]
    fin = CZ._TAB["f32hi_fin"]
    LNT, LN2 = S3.ln_table()
    RL24 = radius_constants(tr)["ints"]["RL24"]
    SM = tr.FQ + tr.FK - FL_LSE
    assert SM >= 0
    m = np.asarray(w["m"], dtype=np.int64)
    m24 = m >> SM
    rm = m - (m24 << SM)
    Z = np.asarray(w["Z"], dtype=np.int64)
    ez = S3.bitlen_floor_log2(np.maximum(Z, 1))
    sh = np.maximum(ez - 15, 0)
    Bz = np.left_shift(np.int64(1), sh)
    mant = np.clip(Z >> sh, 1 << 15, (1 << 16) - 1)
    rz_ln = Z - mant * Bz
    lnv = np.asarray(LNT, dtype=np.int64)[mant - (1 << 15)]
    L24 = m24 + (ez - EB) * int(LN2) + lnv
    cv = np.asarray(lse_u32, dtype=np.uint64).ravel()
    hi, lo = (cv >> np.uint64(16)).astype(np.int64), (cv & np.uint64(0xFFFF)).astype(np.int64)
    nonfin = ~fin[hi]
    s, ahi, E = S_[hi], Ahi_[hi], E_[hi]
    a = (ahi << 16) | lo
    k = E + FL_LSE
    out_rng = nonfin | (E < -149) | (E > -10)
    Al = np.where((k >= -24) & (k <= 14), np.left_shift(np.int64(1), np.clip(k, 0, 14)), 0)
    Bl = np.where((k >= -24) & (k <= 14), np.left_shift(np.int64(1), np.clip(-k, 0, 24)), 1)
    num = a * Al
    ly = num // Bl
    lrem = num - ly * Bl
    lso = ly * (1 - 2 * s)
    diff = lso - L24
    lp = (RL24 - 1) - diff
    lm = (RL24 - 1) + diff
    lse_ratio = (np.abs(diff) + 1).astype(np.float64) / float(RL24)
    return dict(m24=m24, rm=rm, ez=ez, Bz=Bz, mant=mant, rz_ln=rz_ln, lnv=lnv, L24=L24,
                lse_hi=hi, lse_lo=lo, ls_s=s, ls_ahi=ahi, ls_E=E, ls_A=Al, ls_B=Bl, ls_y=ly, ls_rem=lrem,
                ls_o24=lso, lse_sp=lp, lse_sm=lm, lse_bad=bool(out_rng.any() or (lp < 0).any() or (lm < 0).any()),
                lse_max_ratio=float(lse_ratio.max()), RL24=RL24)


def tolerance_witness(s, w):
    """Per-component integers of the tolerance check (scale 2^FO): oh = floor(N 2^6 / Z), W2 = ceil(M2 / Zc),
    SJ = ceil(sqrt(X / 2^12)), X = KH W2 2^12 - KL (|oh|^2 - 2|oh|) + 2^12, SQ0 = ceil(sqrt((t + 1) Q2 + 1)) (t + 1 = sum of the causal mask),
    SQ = ceil(SQ0 2^5 / Z),
    slack+ / slack- of 2^W |o - oh| <= A SJ + B |oh| + Cq SQ + C0."""
    tr = s.tier
    rc = radius_constants(tr)["ints"]
    gk = ihead(tr)
    Hn = tr.G * tr.R
    Z = [int(v) for v in w["Z"]]
    Zc = [z >> CB for z in Z]; rz = [z - (zc << CB) for z, zc in zip(Z, Zc)]
    out = {k: np.zeros((Hn, tr.DH), dtype=object) for k in ("oh", "rd", "aoh", "W2", "rW", "X", "SJ", "SQ", "rS", "sp", "sm")}
    SQ0 = np.zeros((tr.G, tr.DH), dtype=object)
    nvis = int(np.asarray(w["mask"]).sum())                  # t + 1 (the circuit's mask sum)
    for g in range(tr.G):
        for i in range(tr.DH):
            SQ0[g, i] = isqrt_ceil(nvis * int(w["Q2"][g, i]) + 1)
    W = rc["W"]
    for h in range(Hn):
        for i in range(tr.DH):
            N = int(w["N"][h, i]); M2 = int(w["M2"][h, i]); Q2 = int(w["Q2"][gk[h], i])
            oh = (N << 6) // Z[h]; rd = (N << 6) - oh * Z[h]
            W2 = -(-M2 // Zc[h]); rW = W2 * Zc[h] - M2
            X = rc["KH"] * W2 * (1 << 12) - rc["KL"] * (oh * oh - 2 * abs(oh)) + (1 << 12)
            SJ = isqrt_ceil(-(-X // (1 << 12))) if X > 0 else 0
            sq0 = int(SQ0[gk[h], i])
            SQ = -(-(sq0 << 5) // Z[h]); rS = SQ * Z[h] - (sq0 << 5)
            rhs = rc["A"] * SJ + rc["B"] * abs(oh) + rc["Cq"] * SQ + rc["C0"]
            d = int(s.o_int[h, i]) - oh
            for k, v in (("oh", oh), ("rd", rd), ("aoh", abs(oh)), ("W2", W2), ("rW", rW), ("X", X), ("SJ", SJ),
                         ("SQ", SQ), ("rS", rS), ("sp", rhs - (d << W)), ("sm", rhs + (d << W))):
                out[k][h, i] = v
    out["SQ0"] = SQ0
    out["Zc"] = np.array(Zc, dtype=object); out["rz"] = np.array(rz, dtype=object)
    ratio = np.zeros((Hn, tr.DH))
    for h in range(Hn):
        for i in range(tr.DH):
            rhs = rc["A"] * out["SJ"][h, i] + rc["B"] * out["aoh"][h, i] + rc["Cq"] * out["SQ"][h, i] + rc["C0"]
            ratio[h, i] = abs(int(s.o_int[h, i]) - int(out["oh"][h, i])) * (1 << W) / rhs
    out["ratio"] = ratio
    return out
