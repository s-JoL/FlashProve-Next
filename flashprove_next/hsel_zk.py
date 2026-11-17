# SPDX-License-Identifier: Apache-2.0
"""Reusable hidden-selector gadget into a per-request history matrix H (specs/10 §1.2 and §2.3).

tc_zk (the standalone slot proof) hard-wires the hidden-selector opening of H to the J_MAX = 8 claims of one
attention slot.  This module packages the same opening as a gadget (`HSel`) that a larger circuit instantiates
once per H: the caller registers any number (<= n_claims_max) of claims

    point claim   v = H~(P)                          P = (column bits LSB first, row bits LSB first) in H-index order;
                                                     every coordinate is hidden (a committed 0/1 cell of a caller
                                                     column), public (a challenge known when the claim is made) or a
                                                     constant bit
    vector bind   col_i = H[row, prefix || i]  for all i < 2^d   (row / prefix coordinates hidden, public or constant)

and HSel appends four commit rounds that discharge all of them with one opening of H:

  round P (pre)    per bind: r_dim <- FS (AFTER the caller committed col), hidden scalar c = sum_i eq(r_dim, i) col_i
                   (sumzero on the caller space), registered as the point claim c = H~(r_dim, prefix, row);
                   commits the coordinate copy column hc (all hidden coordinate bits of all claims, (j, b) layout), the
                   Libra mask coefficients of the explicit sumcheck, and zero scalars for unused claim slots
  round S          gamma, rho, lambda <- FS; explicit masked hidden-selector sumcheck of A(x) H(x),
                   A = sum_{j used} gamma^j eq(P_j, x) (row variables first) to the public point r = (r_row, r_col);
                   sigma <- FS; commits u_H = eq(r_row)^T H, u_px = sigma-powers^T H, the chain A(r) =
                   sum_j gamma^j prod_b eq(P_j[b], r_b) from hc; booleanity of hc; the lambda-link hc <-> caller cells
                   (sum_k lambda^k hc_k = sum_k lambda^k src_k, one sumzero per caller space)
  round C          Q <- FS (m_q column positions of the rate-1/2 codeword M = RS(H)); hidden columns M[:, Q] ->
                   Poseidon2 column leaves -> Merkle paths -> rt_H (in-circuit); R = trace_root(s_R, rt_obj, rt_H, T_pad)
                   (in-circuit, exactly tc_zk.trace_root; s_R, rt_obj, rt_H hidden)
  round D          beta, mu <- FS; batched column check, split through a hidden scalar X so that the column block need
                   not embed into the u cube:  <u_H + mu u_px, g> = X = sum_q beta_q sum_c (eq(r_row,c) + mu sigma^c) M[c,j_q]

Unused claim slots get A-weight 0 and a value scalar constrained to 0, and their chain rows are the constant 1, so
HSel's own vectors depend only on (tier, n_claims_max).  Which coordinates are hidden / public / constant, the number of
used claims and every bind's dim_bits are PUBLIC structure (the caller's circuit code), not witness.

Commit-order contract (soundness): every caller cell used as a hidden coordinate, every bound vector and every
"sum"-form claim value must be committed BEFORE HSel's round P (phase 1 or an earlier caller round), and public
coordinates must be challenges drawn before round P.  Caller spaces that HSel constrains must be padded
(gadgets.Space default), because hidden scalars are viewed onto them.

Standalone statement (tests): public tier + R; witness: H, s_R, rt_obj, the claims' hidden coordinates and data.
"""
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from . import poseidon2 as PS
from . import modelcommit as MC
from . import mc_zk as MZ
from . import tc_zk as TZ
from .zk_flat import Ex, EqP, SelLt, TB
from .field import P, CPU, ZERO, ONE, hadd, hsub, hmul, lagrange_eval, rand_field

VERSION = "fpn-hsel-1"
LABEL = b"fpn-hsel-v1"
HS_DEG = 2
SCB = TZ.SCB
DOM_HCOL, DOM_ROOT = TZ.DOM_HCOL, TZ.DOM_ROOT

_e = TZ._e
_lg = TZ._lg


# ============================================================================ profile
@dataclass(frozen=True)
class HTier:
    """Shape of H (profile constants only): n_r = 2^RB rows, n_c = 2^CB columns, rate-1/2 RS rows, MQ opened columns,
    T_pad = the public length-tier constant hashed into R (tc_zk.trace_root)."""
    RB: int
    CB: int
    MQ: int = 268
    T_pad: int = 0

    @property
    def nr(self): return 1 << self.RB

    @property
    def nc(self): return 1 << self.CB

    @property
    def n(self): return 2 * self.nc                          # RS rate 1/2 (tc_zk.commit_H)

    @property
    def depth(self): return self.n.bit_length() - 1

    @property
    def nH(self): return self.RB + self.CB                   # variables of H~

    @property
    def NH(self): return self.nr * self.nc


def check_tier(t):
    assert t.RB >= 3, "commit_H absorbs 8 rows per permutation: n_r must be a multiple of 8"
    assert t.CB >= 1 and t.MQ >= 1
    assert t.MQ <= t.n


class _Lay:
    """Public layout of HSel's own spaces (functions of the tier and n_claims_max only)."""

    def __init__(self, t, J):
        check_tier(t)
        assert J >= 2 and J & (J - 1) == 0, "n_claims_max must be a power of two >= 2"
        self.t, self.J = t, J
        self.JB = J.bit_length() - 1
        self.BB = max(1, _lg(t.nH))                          # coordinate index bits of the chain space
        self.QB = max(1, _lg(t.MQ))
        self.A = t.nr // 8                                   # absorb rows per column leaf
        self.depth = t.depth
        self.PCB = _lg(self.A + self.depth + 1)              # Poseidon rows per column block
        self.INIT = (1 << self.PCB) - 1
        self.ROOT = self.A + self.depth - 1
        self.RR = t.MQ << self.PCB                           # R preimage rows RR, RR + 1
        self.PSB = (self.RR + 1 + TB).bit_length()
        assert self.RR + 1 < (1 << self.PSB) - TB
        self.MKB = 2                                         # mask coefficient bits (deg 2 -> 3 coefficients)


# ============================================================================ commitment / statement
def commit(tier, H, B=CPU):
    """Same commitment as tc_zk.commit_H (rate-1/2 RS rows, unsalted Poseidon2 column leaves, Merkle root rt_H)."""
    check_tier(tier)
    H = np.asarray(H)
    assert H.shape == (tier.nr, tier.nc)
    return TZ.commit_H(tier, H, B)


def statement(tier, Rv, n_claims_max):
    return {"v": VERSION, "tier": asdict(tier), "J": int(n_claims_max), "R": [int(x) for x in Rv]}


def make_witness(tier, H, com, n_claims_max, salts=None):
    """-> (stmt, wit).  wit holds H, the commitment levels, rt_H and the hidden salts s_R, rt_obj."""
    rnd = lambda: tuple(int(x) for x in rand_field(4))
    s_R, rt_obj = salts or (rnd(), rnd())
    Rv = TZ.trace_root(s_R, rt_obj, com["root"], tier.T_pad)
    Hd = np.ascontiguousarray(H, dtype=np.int32).reshape(1, tier.nr, tier.nc)
    wit = {"H": np.asarray(H), "H_dev": Hd, "levels": com["levels"], "rt_H": com["root"], "s_R": s_R, "rt_obj": rt_obj}
    return statement(tier, Rv, n_claims_max), wit


# ============================================================================ helpers
def ind0(sp):
    """indicator of content row 0 of a space (all non-pad variables 0)."""
    return Ex.atom(SelLt([], 1, fixed=tuple((v, 0) for v in range(sp.bits))))


def _check_space(sp, what):
    assert sp.padded, f"{what} space {sp.name} must be padded"
    assert all(nm != "c" for nm, _ in sp.axes), f"{what} space {sp.name}: an axis named 'c' clashes with the ('c', bit) constant marker of Col views"


def _src_value(ctx, ex, row):
    """Prover: value of a plain committed column Ex (identity view) on content row `row`, else None."""
    try:
        (c, f), = ex.terms
        (l, p), = f
        (a, co), = l.items.values()
    except (ValueError, AttributeError):
        return None
    if c != ONE or p != 1 or co != ONE or l.const != ZERO or not isinstance(a, Z.View) or a.fixed or a.shift:
        return None
    if a.varmap != tuple(range(a.vec.nbits)):
        return None
    col = ctx.colreg.get(a.vec.id)
    if col is None or col.v is None or row >= len(col.v):
        return None
    return int(col.v[row]) % P


def _ex_key(ex):
    """structural key of an Ex (Col.ex builds a fresh Ex per access, so object identity would not group them)."""
    return tuple((c, tuple((l.key, p) for l, p in f)) for c, f in ex.terms)


def _fe1(x):
    """int / E tuple -> E tuple."""
    if isinstance(x, (tuple, list)):
        return _e(x)
    return (int(x) % P, 0, 0)


def _mle_rowvec(vals, r):
    """sum_i eq(r, i) vals_i for a signed-integer / field vector of length 2^len(r)."""
    v = np.asarray(vals)
    if v.dtype == np.uint64:
        f = v
    else:
        f = TZ.fe(v)
    return CPU.edot_dev(TZ.eq_t(r), TZ.ext(f))


def _mask_weights(n, r, lay):
    """weights on the MK space (b BB, k 2): sum_x g(x) and g(r) for g(x) = sum_b sum_k coef[b][k] x_b^k."""
    N = 1 << (lay.BB + lay.MKB)
    gw = np.zeros((3, N), dtype=np.uint64); rw = np.zeros((3, N), dtype=np.uint64)
    half = pow(2, n - 1, P)
    for b in range(n):
        x = ONE
        for kk in range(HS_DEG + 1):
            pos = (b << lay.MKB) | kk
            gw[0, pos] = (2 if kk == 0 else 1) * half % P
            rw[0, pos], rw[1, pos], rw[2, pos] = x
            x = hmul(x, _e(r[b]))
    return gw, rw


def _col_block(P_col):
    """Identify trailing boolean suffix of P_col (len CB): returns (d_low, c0, c1, eq_low (3, 2^d_low)).
    Exploits that port + layer bits in H claims are boolean, shrinking eq_t from 2^20 down to <= 2^12 (<= 4096)."""
    CB = len(P_col)
    d = CB
    while d > 0 and P_col[d - 1][1] == 0 and P_col[d - 1][2] == 0 and P_col[d - 1][0] in (0, 1):
        d -= 1
    blk = 0
    for b in range(d, CB):
        if P_col[b][0]:
            blk |= 1 << (b - d)
    c0 = blk << d
    return d, c0, c0 + (1 << d), TZ.eq_t(P_col[:d])


def _hs_prove(t, Pfull, wts, Hd, coef, rho, tr, B):
    """Explicit masked hidden-selector sumcheck prover (row variables first, LSB first).
    Exploits sparse boolean column blocks [c0, c1) of width 2^d_low <= 4096 and flattens (3, J * cur_len) on B
    so J = 16,384 claims at T_pad = 1024 use < 250 MB VRAM and ~0.1s instead of 214 GB."""
    CB, RB, n = t.CB, t.RB, t.nH
    nr = t.nr
    J = len(Pfull)
    gpu = MC._is_gpu(B)
    xp = B.xp
    dev = (lambda a: B.asarray(np.ascontiguousarray(a))) if gpu else (lambda a: np.ascontiguousarray(a))
    # 1. Compute hc (3, J, nr) grouped by aligned column slice (c0, c1)
    blks = [_col_block(Pj[:CB]) for Pj in Pfull]
    by_slice = {}
    for j, (d, c0, c1, eq_low) in enumerate(blks):
        by_slice.setdefault((c0, c1), []).append((j, eq_low))
    hc_host = np.zeros((3, J, nr), dtype=np.uint64)
    for (c0, c1), items in by_slice.items():
        m_s = len(items)
        # Slice H[:, c0:c1] -> shape (1, nr, width) int32 on B
        Hs = xp.ascontiguousarray(Hd[:, :, c0:c1])
        for a0 in range(0, m_s, 64):
            sub = items[a0:a0 + 64]
            E_sub = np.stack([it[1] for it in sub], axis=1)             # (3, len(sub), width)
            out = MC.tiles_matvec(B, Hs, [0] * len(sub), dev(E_sub))    # (3, len(sub), nr)
            out_h = B.tonumpy(out) if gpu else out
            for k_i, (j_idx, _) in enumerate(sub):
                hc_host[:, j_idx, :] = out_h[:, k_i, :]
        del Hs
    # 2. Vectorized RB row-variable rounds over flattened (3, J * cur_len) on B
    f0_host = np.stack([CPU.escale(TZ.eq_t(Pfull[j][CB:]), wts[j]) for j in range(J)], axis=1)  # (3, J, nr)
    F0 = dev(f0_host.reshape(3, J * nr))
    F1 = dev(hc_host.reshape(3, J * nr))
    del f0_host, hc_host
    S = B.esum(B.emul(F0, F1)) if J > 0 else ZERO
    Sp = hadd(S, hmul(rho, TZ.mask_sum(coef, n)))
    tr.absorb(TZ.canon([list(Sp)]))
    msgs, r = [], []
    prefix = ZERO

    def round_pair(A2, B2, i):
        a0, a1 = A2[:, 0::2], A2[:, 1::2]
        b0, b1 = B2[:, 0::2], B2[:, 1::2]
        vals = []
        for X in range(HS_DEG + 1):
            if A2.shape[1] == 0:
                acc = ZERO
            elif X == 0:
                acc = B.esum(B.emul(a0, b0))
            elif X == 1:
                acc = B.esum(B.emul(a1, b1))
            else:
                ax = B.eadd(a0, B.escale(B.esub(a1, a0), (X, 0, 0)))
                bx = B.eadd(b0, B.escale(B.esub(b1, b0), (X, 0, 0)))
                acc = B.esum(B.emul(ax, bx))
            vals.append(hadd(acc, hmul(rho, TZ._mask_part(coef, i, n, X, prefix))))
        return vals

    for i in range(RB):
        vals = round_pair(F0, F1, i)
        msgs.append([list(v) for v in vals]); tr.absorb(TZ.canon(msgs[-1]))
        ri = tr.ext(); r.append(ri)
        if J > 0:
            F0 = B.fold(F0, ri)
            F1 = B.fold(F1, ri)
        prefix = hadd(prefix, TZ._gsum(coef[i], ri))
    alpha_arr = B.tonumpy(F0) if (gpu and J > 0) else (F0 if J > 0 else np.zeros((3, 0), dtype=np.uint64))
    del F0, F1
    erow = TZ.eq_t(r[:RB])
    uH = MC.tiles_vecmat(B, Hd, [0], dev(erow.reshape(3, 1, -1)))[:, 0]
    uH = B.tonumpy(uH) if gpu else uH
    # 3. Sparse-block accumulation of Ar (3, n_c)
    Ar = np.zeros((3, t.nc), dtype=np.uint64)
    for j, (d, c0, c1, eq_low) in enumerate(blks):
        aj = (int(alpha_arr[0, j]), int(alpha_arr[1, j]), int(alpha_arr[2, j]))
        Ar[:, c0:c1] = CPU.eadd(Ar[:, c0:c1], CPU.escale(eq_low, aj))
    A_cur, U_cur = dev(Ar), dev(uH)
    del Ar
    for i in range(RB, n):
        vals = round_pair(A_cur, U_cur, i)
        msgs.append([list(v) for v in vals]); tr.absorb(TZ.canon(msgs[-1]))
        ri = tr.ext(); r.append(ri)
        A_cur = B.fold(A_cur, ri)
        U_cur = B.fold(U_cur, ri)
        prefix = hadd(prefix, TZ._gsum(coef[i], ri))
    return Sp, msgs, r, uH, S


def _ps_inputs(lay, Q, cols, paths, wit):
    """Poseidon2 inputs of every HSel Poseidon row (column leaves, Merkle paths, R); returns (inputs, roots, R)."""
    t = lay.t
    inp = np.zeros((12, (1 << lay.PSB) - TB), dtype=np.uint64)
    mq = t.MQ
    Qa = np.asarray(Q, dtype=np.uint64)
    base = np.arange(mq, dtype=np.int64) << lay.PCB
    st = np.zeros((12, mq), dtype=np.uint64)
    st[0] = Qa; st[1] = t.nr; st[10] = DOM_HCOL; st[11] = PS.TAG_SPONGE
    inp[:, base + lay.INIT] = st
    st = PS.permute(st, CPU)
    for b in range(lay.A):
        st = st.copy()
        st[0:8] = CPU.add(st[0:8], np.ascontiguousarray(cols[8 * b:8 * b + 8]))
        inp[:, base + b] = st
        st = PS.permute(st, CPU)
    cur = st[0:4].copy()
    sib = np.asarray(paths, dtype=np.uint64)
    for d in range(lay.depth):
        bit = ((Qa >> np.uint64(d)) & np.uint64(1)).astype(bool)
        s_ = np.ascontiguousarray(sib[:, d, :].T)
        st = np.zeros((12, mq), dtype=np.uint64)
        st[0:4] = np.where(bit, s_, cur); st[4:8] = np.where(bit, cur, s_); st[11] = PS.TAG_NODE
        inp[:, base + lay.A + d] = st
        cur = PS.permute(st, CPU)[0:4]
    roots = cur
    rt = roots[:, mq - 1]
    stA = np.zeros((12, 1), dtype=np.uint64)
    stA[0:4, 0] = wit["s_R"]; stA[4:8, 0] = wit["rt_obj"]; stA[10] = DOM_ROOT; stA[11] = PS.TAG_SPONGE
    inp[:, lay.RR] = stA[:, 0]
    stB = PS.permute(stA, CPU).copy()
    stB[0:4, 0] = CPU.add(stB[0:4, 0], rt)
    stB[4, 0] = CPU.add(stB[4:5, 0], np.array([t.T_pad], dtype=np.uint64))[0]
    inp[:, lay.RR + 1] = stB[:, 0]
    Rv = tuple(int(v) for v in PS.permute(stB, CPU)[0:4, 0])
    return inp, roots, Rv


# ============================================================================ the gadget
class HSel:
    """In-circuit hidden-selector opening of one committed H.

    ctx: gadgets.Ctx; tier: HTier; n_claims_max: claim slots J (power of two); wit: make_witness(...)[1] (prover) or
    None (verifier); stmt: the statement dict holding the public "R".  tag namespaces spaces, scalars and the proof key.
    Use: register claims (claim_point / bind_vector) during the caller's build or caller rounds, then append
    hs.rounds() to ctx.cir.rounds AFTER the caller's rounds."""

    def __init__(self, ctx, tier, n_claims_max, wit, stmt, tag="hs", B=CPU):
        self.ctx, self.t, self.J, self.wit, self.stmt, self.tag, self.B = ctx, tier, n_claims_max, wit, stmt, tag, B
        self.lay = _Lay(tier, n_claims_max)
        self.pv = ctx.prove
        self.SC = TZ.Scalars(ctx)
        self.claims = []            # dict(coords=[(kind, payload, bit)], vsrc=(sp, ex, row), vwit=E)
        self.binds = []
        self.bbinds = []
        self._closed = False
        self.S = {}
        self.info = {}

    # ---------------------------------------------------------------- claim registration
    def _name(self, s):
        return f"{self.tag}.{s}"

    def _H_dev(self):
        """H as (1, n_r, n_c) int32 on the prover backend B (a device copy of the host witness when B is a GPU)."""
        if getattr(self, "_Hd", None) is None:
            Hd = self.wit["H_dev"]
            self._Hd = self.B.xp.asarray(Hd) if MC._is_gpu(self.B) else Hd
        return self._Hd

    def scalar(self, name, val=None):
        """A fresh hidden replicated scalar (TZ.Scalars) committed in the current round; returns its name.
        View it on a padded space with hs.SC.on(name, sp)."""
        nm = self._name(name)
        self.SC.add(nm, val if self.pv else None)
        return nm

    def _norm_coords(self, coords):
        t = self.t
        if len(coords) != t.nH:
            raise ValueError(f"claim point needs {t.nH} coordinates (CB column bits, then RB row bits), got {len(coords)}")
        out = []
        for c in coords:
            kind = c[0]
            if kind == "hid":
                src, bit = c[1], (c[2] if len(c) > 2 else None)
                sp, ex, row = src
                _check_space(sp, "hidden-coordinate source")
                assert 0 <= row < sp.nrows
                out.append(("hid", (sp, ex, int(row)), bit))
            elif kind == "pub":
                out.append(("pub", _fe1(c[1]), None))
            elif kind == "const":
                assert c[1] in (0, 1)
                out.append(("const", (int(c[1]), 0, 0), None))
            else:
                raise ValueError(kind)
        return out

    def _pending_claims(self):
        return len(self.claims) + len(self.binds) + sum(b["n_active"] for b in self.bbinds)

    def claim_col_cell(self, coords, sp, ex, row, value_wit=None):
        """Register the claim ``ex[row] = H~(coords)`` directly from row ``row`` of a padded space ``sp`` (no TZ.Scalars)."""
        if self._closed:
            raise RuntimeError("claim registered after HSel's pre round")
        j = len(self.claims)
        if j >= self.J:
            raise ValueError(f"more than n_claims_max = {self.J} claims")
        _check_space(sp, "claim-cell")
        assert 0 <= int(row) < sp.nrows
        cs = self._norm_coords(coords)
        vw = _fe1(value_wit) if (self.pv and value_wit is not None) else (ZERO if self.pv else None)
        self.claims.append({"coords": cs, "vsrc": (sp, ex, int(row)), "vwit": vw})
        return j

    def claim_point(self, coords, value=None, value_wit=None):
        """Register the claim  value = H~(P).
        coords: t.nH entries in H-index order (coords[0:CB] column bits LSB first, coords[CB:] row bits LSB first):
            ("hid", (sp, ex, row), bit)  the coordinate is the value of Ex `ex` on content row `row` of the padded space
                                          sp (a committed 0/1 cell); bit = its witness value (prover; may be any int)
            ("pub", e)                    a public field / E element (a challenge already drawn)
            ("const", 0 | 1)
        value: None -> a fresh hidden scalar (returned name; constrain it via hs.SC.on(name, sp));
               (sp, ex)  -> the claimed value is sum_{content rows of padded sp} ex (a hidden scalar equal to it is
                            created and linked);
               str       -> an existing scalar name of hs.SC.
        value_wit: the claimed value (prover, int or E tuple).  Returns (slot index j, scalar name)."""
        if self._closed:
            raise RuntimeError("claim registered after HSel's pre round")
        j = len(self.claims)
        if j >= self.J:
            raise ValueError(f"more than n_claims_max = {self.J} claims")
        cs = self._norm_coords(coords)
        vw = _fe1(value_wit) if (self.pv and value_wit is not None) else (ZERO if self.pv else None)
        if isinstance(value, str):
            nm = value
        else:
            nm = self.scalar(f"v{j}", vw)
            if value is not None:
                sp, ex = value
                _check_space(sp, "claim-value")
                sp.sumzero(ex - self.SC.on(nm, sp) * ind0(sp), tag=self._name(f"val{j}"))
        sc_col = self.SC.c[nm]
        self.claims.append({"coords": cs, "val": nm, "vsrc": (sc_col.space, sc_col.ex, 0), "vwit": vw})
        return j, nm

    def bind_vector(self, src, dim_bits, row_coords, col_prefix_coords, vec_wit=None):
        """Bind a committed vector col (2^dim_bits entries) to H[row, prefix || i], i < 2^dim_bits."""
        if self._closed:
            raise RuntimeError("bind registered after HSel's pre round")
        t = self.t
        if isinstance(src, G.Col):
            sp = src.space
            assert sp.bits == dim_bits
            src = (sp, src.ex, list(range(dim_bits)), None)
        sp, ex, ivars, sel = src
        _check_space(sp, "bound-vector")
        assert len(ivars) == dim_bits and 1 <= dim_bits <= t.CB
        assert len(row_coords) == t.RB and len(col_prefix_coords) == t.CB - dim_bits
        if self._pending_claims() >= self.J:
            raise ValueError(f"more than n_claims_max = {self.J} claims")
        self.binds.append({"src": (sp, ex, list(ivars), sel), "d": dim_bits, "row": list(row_coords),
                           "pre": list(col_prefix_coords), "vec": vec_wit})
        return len(self.binds) - 1

    def bind_batched_vector(self, src, sb, dim_bits, rows_per_s, pres_per_s, vec_wits=None, n_active=None):
        """Bind a batched column on padded space ``sp`` (leading axis ``('s', sb)``, inner ``dim_bits`` bits) across
        the first ``n_active <= 2^sb`` slots using ONE shared random challenge ``r_dim`` and ONE ``eq_axes=['s']`` sumzero."""
        if self._closed:
            raise RuntimeError("bind registered after HSel's pre round")
        t = self.t
        K = 1 << sb
        n_act = K if n_active is None else int(n_active)
        assert 0 <= n_act <= K
        if isinstance(src, G.Col):
            sp = src.space
            assert sp.bits == sb + dim_bits
            src = (sp, src.ex, list(range(dim_bits)), None)
        sp, ex, ivars, sel = src
        _check_space(sp, "batched-bound-vector")
        assert sp.axes[0] == ("s", sb) and len(ivars) == dim_bits and 1 <= dim_bits <= t.CB
        assert len(rows_per_s) == n_act and len(pres_per_s) == n_act
        if self._pending_claims() + n_act > self.J:
            raise ValueError(f"more than n_claims_max = {self.J} claims")
        self.bbinds.append({"src": (sp, ex, list(ivars), sel), "sb": sb, "d": dim_bits,
                            "rows": [list(rc) for rc in rows_per_s], "pres": [list(pc) for pc in pres_per_s],
                            "vecs": vec_wits, "n_active": n_act})
        return len(self.bbinds) - 1

    # ---------------------------------------------------------------- prover helpers
    def _full_points(self):
        out = []
        for cl in self.claims:
            q = []
            for kind, pay, bit in cl["coords"]:
                q.append(_fe1(bit) if kind == "hid" else pay)
            out.append(q)
        return out

    def _hid_list(self):
        """hidden coordinates in link order: (k, j, b, src, bit)."""
        out = []
        for j, cl in enumerate(self.claims):
            for b, (kind, pay, bit) in enumerate(cl["coords"]):
                if kind == "hid":
                    out.append((j, b, pay, bit))
        return out

    # ---------------------------------------------------------------- rounds
    def rounds(self):
        return [self._round_pre, self._round_hs, self._round_cols, self._round_check]

    def _pf(self):
        pf = self.ctx.cir.pf
        return pf.setdefault("hsel", {}).setdefault(self.tag, {}) if self.pv else pf.get("hsel", {}).get(self.tag, {})

    def _round_pre(self, tr):
        t0 = time.time()
        ctx, t, lay, pv = self.ctx, self.t, self.lay, self.pv
        # 1. single-vector binds
        for k, bd in enumerate(self.binds):
            r = tr.exts(bd["d"])
            sp, ex, ivars, sel = bd["src"]
            cv = _mle_rowvec(bd["vec"], r) if pv else None
            nm = self.scalar(f"c{k}", cv)
            e = Ex.atom(EqP(ivars, r)) * ex
            if sel is not None:
                e = e * sel
            sp.sumzero(e - self.SC.on(nm, sp) * ind0(sp), tag=self._name(f"bind{k}"))
            coords = [("pub", x) for x in r] + bd["pre"] + bd["row"]
            self.claim_point(coords, value=nm, value_wit=cv)
        # 2. s-batched vector binds (one shared r_dim and one eq_axes=['s'] sumzero per port)
        for k, bd in enumerate(self.bbinds):
            r = tr.exts(bd["d"])
            sp, ex, ivars, sel = bd["src"]
            sb, d = bd["sb"], bd["d"]
            K = 1 << sb
            CS = ctx.space(self._name(f"CB{k}"), [("s", sb)])
            cb_v = None
            cvs = [None] * bd["n_active"]
            if pv:
                cb_v = np.zeros((3, K), dtype=np.uint64)
                vw2 = np.asarray(bd["vecs"]).reshape(K, 1 << d)
                eq_r = TZ.eq_t(r)
                for s_ in range(K):
                    row_v = vw2[s_]
                    f_v = row_v if row_v.dtype == np.uint64 else TZ.fe(row_v)
                    cv_s = CPU.edot_dev(eq_r, TZ.ext(f_v))
                    cb_v[0, s_], cb_v[1, s_], cb_v[2, s_] = cv_s
                    if s_ < bd["n_active"]:
                        cvs[s_] = cv_s
            cb = MZ.ecol(CS, "v", cb_v)
            e = Ex.atom(EqP(ivars, r)) * ex
            if sel is not None:
                e = e * sel
            ind_in0 = Ex.atom(SelLt([], 1, fixed=tuple((v, 0) for v in ivars)))
            if sel is not None:
                ind_in0 = ind_in0 * sel
            sp.sumzero(e - MZ.view(cb, sp, {"s": "s"}) * ind_in0, eq_axes=["s"], tag=self._name(f"bbind{k}"))
            for s_ in range(bd["n_active"]):
                coords = [("pub", x) for x in r] + bd["pres"][s_] + bd["rows"][s_]
                self.claim_col_cell(coords, CS, cb.ex, s_, value_wit=cvs[s_])
        self._closed = True
        self.n_used = len(self.claims)
        # 3. ONE claim-value vector cval on JC (j: JB) for all J slots (replaces O(J) TZ.Scalars + O(J) Lin)
        JC = ctx.space(self._name("JC"), [("j", lay.JB)])
        cval_v = None
        if pv:
            cval_v = np.zeros((3, self.J), dtype=np.uint64)
            for j, cl_ in enumerate(self.claims):
                vw = cl_["vwit"]
                cval_v[0, j], cval_v[1, j], cval_v[2, j] = vw
        cval = MZ.ecol(JC, "cval", cval_v)
        if self.n_used < self.J:
            JC.zero(cval.ex, sel=JC.rng("j", self.n_used, self.J), tag=self._name("cval.unused"))
        # coordinate copy column hc on AR (j, b) and the hidden mask hm (public)
        AR = ctx.space(self._name("A"), [("j", lay.JB), ("b", lay.BB)])
        hcv = None
        hm = np.zeros(1 << (lay.JB + lay.BB), dtype=np.int64)
        for j, b, src, bit in self._hid_list():
            hm[(j << lay.BB) | b] = 1
        if pv:
            hcv = np.zeros(AR.n, dtype=object)
            bad = False
            mism = False
            for j, b, src, bit in self._hid_list():
                hcv[(j << lay.BB) | b] = int(bit) % P
                bad |= int(bit) not in (0, 1)
                sv = _src_value(ctx, src[1], src[2])
                mism |= sv is not None and sv != int(bit) % P
            if bad:
                ctx.violation("non-boolean hidden coordinate")
            if mism:
                ctx.violation("hidden coordinate copy differs from its source cell")
            hcv = TZ.fe(hcv)
        hc = AR.col("hc", hcv)
        AR.zero(hc.ex * (hc.ex - Ex.const(1)), tag=self._name("hc.bool"))
        AR.zero((Ex.const(1) - AR.pub("hm", hm).ex) * hc.ex, tag=self._name("hc.mask"))
        # Libra mask coefficients of the explicit sumcheck (committed before rho)
        MK = ctx.space(self._name("MK"), [("b", lay.BB), ("k", lay.MKB)])
        coef = mkv = None
        if pv:
            mkv = np.zeros((3, 1 << (lay.BB + lay.MKB)), dtype=np.uint64)
            coef = []
            for b in range(t.nH):
                row = []
                for kk in range(HS_DEG + 1):
                    rf = rand_field(3)
                    v = (int(rf[0]), int(rf[1]), int(rf[2]))
                    row.append(v)
                    mkv[0, (b << lay.MKB) | kk], mkv[1, (b << lay.MKB) | kk], mkv[2, (b << lay.MKB) | kk] = v
                coef.append(row)
        mkc = MZ.ecol(MK, "g", mkv)
        self.S.update(JC=JC, cval=cval, AR=AR, hc=hc, MK=MK, mkc=mkc, coef=coef)
        self.info["pre_s"] = round(time.time() - t0, 3)

    def _round_hs(self, tr):
        t0 = time.time()
        ctx, t, lay, pv, B = self.ctx, self.t, self.lay, self.pv, self.B
        gam = tr.ext(); rho = tr.ext(); lam = tr.ext(); lam_v = tr.ext()
        Gam = MZ.epowers(gam, self.J)
        wts = [_e(Gam[:, j]) for j in range(self.n_used)]
        n = t.nH
        ent = self._pf()
        if pv:
            Pfull = self._full_points()
            coef = self.S["coef"]
            Hd = self._H_dev()
            Sp, msgs, r, uH, Strue = _hs_prove(t, Pfull, wts, Hd, coef, rho, tr, B)
            ent["hs"] = {"S": list(Sp), "msgs": msgs}
            vfin = lagrange_eval([_e(v) for v in msgs[-1]], r[-1])
            vs = ZERO
            for j in range(self.n_used):
                vs = hadd(vs, hmul(wts[j], self.claims[j]["vwit"]))
            if vs != Strue:
                ctx.violation("hidden-selector: claimed values disagree with H")
        else:
            if not isinstance(ent.get("hs"), dict):
                raise Z.ProofError("missing hidden-selector sumcheck")
            Sp = _e(ent["hs"]["S"])
            r, vfin = TZ.esc_verify(Sp, ent["hs"]["msgs"], n, HS_DEG, tr, self._name("HS"))
        r_row, r_col = r[:t.RB], r[t.RB:]
        rH = list(r_col) + list(r_row)                     # H-index order
        sig = tr.ext()
        US = ctx.space(self._name("U"), [("k", t.CB)])
        uH_v = upx_v = None
        if pv:
            uH_v = uH
            upx_v = TZ._combine(t, {"H_dev": Hd}, MZ.epowers(sig, t.nr), B)
            del Hd
            self._Hd = None                                   # the device copy is not needed after this round
        uHc = MZ.ecol(US, "uH", uH_v); upc = MZ.ecol(US, "upx", upx_v)
        # A(r) chain from the hidden coordinates
        JC, cval = self.S["JC"], self.S["cval"]
        AR, hc = self.S["AR"], self.S["hc"]
        NA = 1 << (lay.JB + lay.BB)
        C1 = np.zeros((3, NA), dtype=np.uint64); C0 = np.zeros((3, NA), dtype=np.uint64)
        for j in range(self.J):
            for b in range(1 << lay.BB):
                pos = (j << lay.BB) | b
                if j >= self.n_used or b >= n:
                    C0[0, pos] = 1
                    continue
                rb = _e(rH[b])
                two_r1 = hsub(hadd(rb, rb), ONE)
                kind, pay, _ = self.claims[j]["coords"][b]
                if kind == "hid":
                    C1[:, pos] = two_r1
                    C0[:, pos] = hsub(ONE, rb)
                else:
                    C0[:, pos] = hadd(hmul(pay, two_r1), hsub(ONE, rb))
        pp_v = None
        Ar_v = ZERO
        if pv:
            Pfull = self._full_points()
            pp_v = np.zeros((3, NA), dtype=np.uint64)
            for j in range(self.J):
                acc = ONE
                for b in range(1 << lay.BB):
                    if j < self.n_used and b < n:
                        acc = hmul(acc, TZ.eqv([Pfull[j][b]], [rH[b]]))
                    pp_v[0, (j << lay.BB) | b], pp_v[1, (j << lay.BB) | b], pp_v[2, (j << lay.BB) | b] = acc
                if j < self.n_used:
                    Ar_v = hadd(Ar_v, hmul(wts[j], acc))
        ppc = MZ.ecol(AR, "pp", pp_v)
        av = AR.vars("b") + AR.vars("j")
        e_ex = Ex.atom(MZ.EVec(self._name("C1"), C1, av)) * hc.ex + Ex.atom(MZ.EVec(self._name("C0"), C0, av))
        AR.zero(ppc.ex - e_ex, sel=AR.ind(b=0), tag=self._name("chain0"))
        AR.zero(ppc.ex - ppc.shifted(-1) * e_ex, sel=AR.rng("b", 1, 1 << lay.BB), tag=self._name("chain"))
        gj = np.zeros((3, self.J), dtype=np.uint64)
        for j in range(self.n_used):
            gj[:, j] = wts[j]
        # hidden scalars of this round
        vals = {}
        if pv:
            coef = self.S["coef"]
            cl_v = ZERO
            for j in range(self.n_used):
                cl_v = hadd(cl_v, hmul(wts[j], self.claims[j]["vwit"]))
            vals = {"Ghs": TZ.mask_sum(coef, n), "grhs": TZ.mask_eval(coef, r), "Ar": Ar_v,
                    "uHr": CPU.edot_dev(TZ.eq_t(r_col), uH), "Cl": cl_v}
        nm = {k: self.scalar(k, vals.get(k)) for k in ("Ghs", "grhs", "Ar", "uHr", "Cl")}
        JC.sumzero(Ex.atom(MZ.EVec(self._name("gam_jc"), gj, JC.vars("j"))) * cval.ex
                   - self.SC.on(nm["Cl"], JC) * ind0(JC), tag=self._name("Cl"))
        AR.sumzero(Ex.atom(MZ.EVec(self._name("gamj"), gj, AR.vars("j"), fixed=tuple((v, 1) for v in AR.vars("b"))))
                   * ppc.ex - self.SC.on(nm["Ar"], AR) * ind0(AR), tag=self._name("Ar"))
        US.sumzero(Ex.atom(EqP(US.vars("k"), r_col)) * (uHc.ex - self.SC.on(nm["uHr"], US)), tag=self._name("uHr"))
        MK, mkc = self.S["MK"], self.S["mkc"]
        gw, rw = _mask_weights(n, r, lay)
        mv = MK.vars("k") + MK.vars("b")
        MK.sumzero(Ex.atom(MZ.EVec(self._name("Gw"), gw, mv)) * mkc.ex - self.SC.on(nm["Ghs"], MK) * ind0(MK),
                   tag=self._name("Ghs"))
        MK.sumzero(Ex.atom(MZ.EVec(self._name("rw"), rw, mv)) * mkc.ex - self.SC.on(nm["grhs"], MK) * ind0(MK),
                   tag=self._name("grhs"))
        SY = ctx.space(self._name("Y"), [("x", SCB)])
        Y = lambda s: self.SC.on(s, SY)
        SY.zero(Y(nm["Cl"]) + Ex.const(rho) * Y(nm["Ghs"]) - Ex.const(_e(Sp)), tag=self._name("HS.sum"))
        SY.zero(Y(nm["Ar"]) * Y(nm["uHr"]) + Ex.const(rho) * Y(nm["grhs"]) - Ex.const(vfin), tag=self._name("HS.fin"))
        # lam_v-link: cval[j] <-> caller value cells (one sumzero per distinct caller space)
        if self.n_used:
            Lpv = MZ.epowers(lam_v, self.n_used)
            wjc = np.zeros((3, self.J), dtype=np.uint64)
            vgroups = {}
            for j in range(self.n_used):
                lk = _e(Lpv[:, j])
                wjc[:, j] = lk
                sp, ex, row = self.claims[j]["vsrc"]
                g = vgroups.setdefault(sp.name, {"sp": sp, "terms": {}, "val": ZERO})
                tm = g["terms"].setdefault(_ex_key(ex), {"ex": ex, "w": np.zeros((3, sp.nrows), dtype=np.uint64)})
                tm["w"][:, row] = CPU.eadd(tm["w"][:, row:row + 1], np.array(lk, dtype=np.uint64).reshape(3, 1))[:, 0]
                if pv:
                    g["val"] = hadd(g["val"], hmul(lk, self.claims[j]["vwit"]))
            lcval = ZERO
            if pv:
                for g in vgroups.values():
                    lcval = hadd(lcval, g["val"])
            nLv = self.scalar("Lcval", lcval if pv else None)
            JC.sumzero(Ex.atom(MZ.EVec(self._name("wlv"), wjc, JC.vars("j"))) * cval.ex
                       - self.SC.on(nLv, JC) * ind0(JC), tag=self._name("link.cval"))
            tot_v = Ex.const(0)
            for gi, g in enumerate(vgroups.values()):
                sp = g["sp"]
                nLvg = self.scalar(f"Lv{gi}", g["val"] if pv else None)
                e = Ex.const(0)
                for ti, tm in enumerate(g["terms"].values()):
                    e = e + Ex.atom(MZ.EVec(self._name(f"wlv{gi}.{ti}"), tm["w"], list(range(sp.bits)))) * tm["ex"]
                sp.sumzero(e - self.SC.on(nLvg, sp) * ind0(sp), tag=self._name(f"vlink{gi}"))
                tot_v = tot_v + Y(nLvg)
            SY.zero(Y(nLv) - tot_v, tag=self._name("vlink.sum"))
        # lambda-link: hc (copy) <-> caller cells, sum_k lam^k hc_k = sum_k lam^k src_k (one sumzero per caller space)
        hid = self._hid_list()
        if hid:
            Lp = MZ.epowers(lam, len(hid))
            war = np.zeros((3, NA), dtype=np.uint64)
            groups = {}
            for k, (j, b, src, bit) in enumerate(hid):
                lk = _e(Lp[:, k])
                war[:, (j << lay.BB) | b] = lk
                sp, ex, row = src
                g = groups.setdefault(sp.name, {"sp": sp, "terms": {}, "val": ZERO})
                tm = g["terms"].setdefault(_ex_key(ex), {"ex": ex, "w": np.zeros((3, sp.nrows), dtype=np.uint64)})
                tm["w"][:, row] = CPU.eadd(tm["w"][:, row:row + 1], np.array(lk, dtype=np.uint64).reshape(3, 1))[:, 0]
                if pv:
                    g["val"] = hadd(g["val"], hmul(lk, _fe1(bit)))
            lhc = ZERO
            if pv:
                for g in groups.values():
                    lhc = hadd(lhc, g["val"])
            nL = self.scalar("Lhc", lhc if pv else None)
            AR.sumzero(Ex.atom(MZ.EVec(self._name("wl"), war, av)) * hc.ex - self.SC.on(nL, AR) * ind0(AR),
                       tag=self._name("link.hc"))
            tot = Ex.const(0)
            for gi, g in enumerate(groups.values()):
                sp = g["sp"]
                nLg = self.scalar(f"L{gi}", g["val"] if pv else None)
                e = Ex.const(0)
                for ti, tm in enumerate(g["terms"].values()):
                    e = e + Ex.atom(MZ.EVec(self._name(f"wl{gi}.{ti}"), tm["w"], list(range(sp.bits)))) * tm["ex"]
                sp.sumzero(e - self.SC.on(nLg, sp) * ind0(sp), tag=self._name(f"link{gi}"))
                tot = tot + Y(nLg)
            SY.zero(Y(nL) - tot, tag=self._name("link.sum"))
        self.S.update(r=r, r_row=r_row, r_col=r_col, sig=sig, US=US, uHc=uHc, upc=upc)
        self.info["hs_round_s"] = round(time.time() - t0, 3)

    def _round_cols(self, tr):
        t0 = time.time()
        ctx, t, lay, pv = self.ctx, self.t, self.lay, self.pv
        Q = tr.indices(t.MQ, t.n)
        self.S["Q"] = Q
        QB, IB = lay.QB, t.RB
        MS = ctx.space(self._name("M"), [("q", QB), ("row", IB)])   # never name an axis "c": ("c", b) is a constant
        M_v = inp = None
        if pv:
            cols = TZ.columns(t, self.wit["H"], Q, self.B)                  # (n_r, m_q)
            if self.wit.get("flip") is not None:
                cols = cols.copy(); cols[self.wit["flip"], 0] ^= np.uint64(1)
            self.S["cols"] = cols
            paths = [TZ.path(self.wit["levels"], j, t.depth) for j in Q]
            M_v = np.zeros((1 << QB, 1 << IB), dtype=np.uint64)
            M_v[:t.MQ] = cols.T
            M_v = M_v.reshape(-1)
            inp, roots, Rv = _ps_inputs(lay, Q, cols, paths, self.wit)
            if any(tuple(int(x) for x in roots[:, q]) != tuple(self.wit["rt_H"]) for q in range(t.MQ)):
                ctx.violation("opened column does not authenticate to rt_H")
            if Rv != tuple(int(x) for x in self.stmt["R"]):
                ctx.violation("trace root mismatch")
            self._Hd = None
            if self.wit.get("release_h"):
                self.wit["H"] = None; self.wit["H_dev"] = None; self.wit["levels"] = None
        M = MS.col("M", M_v)
        PSp = ctx.space(self._name("PS"), [("q", lay.PSB - lay.PCB), ("b", lay.PCB)], padded=False)
        reg = G.PoseRegion(PSp, inp, B=self.B if pv else None)
        I = lambda k: reg.I(k).ex
        O = lambda k: reg.O(k).ex
        Osh = lambda k, d: reg.O(k).shifted(d)
        Ish = lambda k, d: reg.I(k).shifted(d)
        cst = lambda c: Ex.const(c % P)
        allv = list(range(PSp.nv))

        def pt(x):
            return Ex.atom(Z.SelLt([], 1, fixed=tuple((v, (x >> v) & 1) for v in allv)))

        jq = np.zeros(1 << PSp.nv, dtype=np.int64); pb = np.zeros(1 << PSp.nv, dtype=np.int64)
        for q, j in enumerate(Q):
            jq[(q << lay.PCB) + lay.INIT] = j
            for d in range(lay.depth):
                pb[(q << lay.PCB) + lay.A + d] = (j >> d) & 1
        jq_ex = PSp.pub("jq", jq).ex; pb_ex = PSp.pub("pbit", pb).ex
        colsel = PSp.lt("q", t.MQ)
        s_init = colsel * PSp.ind(b=lay.INIT)
        T_ = self._name
        PSp.zero(I(0) - jq_ex, sel=s_init, tag=T_("leaf.idx"))
        PSp.zero(I(1) - cst(t.nr), sel=s_init, tag=T_("leaf.nr"))
        for k in range(2, 10):
            PSp.zero(I(k), sel=s_init, tag=T_("leaf.zero"))
        PSp.zero(I(10) - cst(DOM_HCOL), sel=s_init, tag=T_("leaf.dom"))
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=s_init, tag=T_("leaf.tag"))
        first = PSp.ind(b=0); absorb = PSp.lt("b", lay.A); rest = absorb - first
        s_abs = colsel * absorb
        qmap = [("q", k) for k in range(QB)]
        for k in range(12):
            prev = first * Osh(k, lay.INIT) + rest * Osh(k, -1)
            e = I(k) - prev
            if k < 8:
                imap = [("c", (k >> b) & 1) for b in range(3)] + [("b", b) for b in range(IB - 3)]
                e = e - MZ.view(M, PSp, {"q": qmap, "row": imap}, pad_to=("b", IB - 3))
            PSp.zero(e, sel=s_abs, tag=T_("leaf.absorb"))
        s_path = colsel * PSp.rng("b", lay.A, lay.A + lay.depth)
        for k in range(4):
            cur = Osh(k, -1)
            PSp.zero((Ex.const(1) - pb_ex) * (I(k) - cur) + pb_ex * (I(4 + k) - cur), sel=s_path, tag=T_("path"))
        for k in (8, 9, 10):
            PSp.zero(I(k), sel=s_path, tag=T_("path.pad"))
        PSp.zero(I(11) - cst(PS.TAG_NODE), sel=s_path, tag=T_("path.tag"))
        xlast = ((t.MQ - 1) << lay.PCB) + lay.ROOT
        if t.MQ > 1:
            for k in range(4):
                PSp.zero(Osh(k, 1 << lay.PCB) - O(k), sel=PSp.lt("q", t.MQ - 1) * PSp.ind(b=lay.ROOT), tag=T_("root.chain"))
        # R = H(DOM_ROOT; s_R, rt_obj | rt_H, T_pad) on rows RR, RR + 1 (tc_zk.trace_root)
        RR = lay.RR
        for k in (8, 9):
            PSp.zero(I(k), sel=pt(RR), tag=T_("R.zero"))
        PSp.zero(I(10) - cst(DOM_ROOT), sel=pt(RR), tag=T_("R.dom"))
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=pt(RR), tag=T_("R.tag"))
        # rt_H (the output of the last column's root row xlast) enters the R preimage at row RR + 1.  The constraint
        # is anchored at row xlast with POSITIVE shifts (RR + 1 - xlast, RR - xlast in [1, 2^CB]): a shift of
        # xlast - (RR + 1) <= -TB would leave no random tail row of O in its support (zk_flat.zk_budget_check; T=1024
        # had -30).  Every other shift of this region is -1 or positive and < N - TB (asserted below).
        assert 1 <= RR - xlast <= (1 << lay.PCB) and RR + 1 + TB <= (1 << PSp.nv), (RR, xlast)
        for k in range(4):
            PSp.zero(Ish(k, RR + 1 - xlast) - Osh(k, RR - xlast) - O(k), sel=pt(xlast), tag=T_("R.rtH"))
        PSp.zero(I(4) - Osh(4, -1) - cst(t.T_pad), sel=pt(RR + 1), tag=T_("R.T"))
        for k in range(5, 12):
            PSp.zero(I(k) - Osh(k, -1), sel=pt(RR + 1), tag=T_("R.carry"))
        for k in range(4):
            PSp.zero(O(k) - cst(int(self.stmt["R"][k])), sel=pt(RR + 1), tag=T_("R.out"))
        self.S.update(M=M, MS=MS)
        self.info["cols_round_s"] = round(time.time() - t0, 3)

    def _round_check(self, tr):
        ctx, t, lay, pv = self.ctx, self.t, self.lay, self.pv
        beta = tr.exts(t.MQ); mu = tr.ext()
        US, MS, M = self.S["US"], self.S["MS"], self.S["M"]
        ac = CPU.eadd(TZ.eq_t(self.S["r_row"]), CPU.escale(MZ.epowers(self.S["sig"], t.nr), mu))
        bq = np.zeros((3, 1 << lay.QB), dtype=np.uint64)
        for q, b in enumerate(beta):
            bq[0, q], bq[1, q], bq[2, q] = b
        Xv = None
        if pv:
            cols = self.S["cols"]
            Xv = ZERO
            for q in range(t.MQ):
                Xv = hadd(Xv, hmul(_e(beta[q]), CPU.edot_dev(ac, TZ.ext(cols[:, q]))))
        nX = self.scalar("X", Xv)
        gq = Ex.atom(MZ.GQ(TZ._gq_params(t), self.S["Q"], beta, US.vars("k")))
        US.sumzero((self.S["uHc"].ex + Ex.const(mu) * self.S["upc"].ex) * gq - self.SC.on(nX, US) * ind0(US),
                   tag=self._name("colcheck.u"))
        MS.sumzero(Ex.atom(MZ.EVec(self._name("acoef"), ac, MS.vars("row"))) * Ex.atom(MZ.EVec(self._name("beta"), bq, MS.vars("q")))
                   * M.ex - self.SC.on(nX, MS) * ind0(MS), tag=self._name("colcheck.M"))


# ============================================================================ prove / verify (standalone circuits)
def prove_ctx(ctx, stmt, B=CPU):
    ctx.cir.pf = None
    return Z.prove(ctx.cir, stmt, B, label=LABEL)


def verify_ctx(ctx, stmt, pf):
    return Z.verify(ctx.cir, stmt, pf, label=LABEL)


def try_verify_ctx(ctx, stmt, pf):
    try:
        verify_ctx(ctx, stmt, pf)
        return "accepted"
    except Z.ProofError as e:
        return "rejected: " + str(e)
    except (KeyError, ValueError, IndexError, TypeError, AssertionError) as e:
        return f"rejected: malformed proof ({type(e).__name__}: {e})"


def cells(cir):
    return int(sum(v.N for v in cir.vecs))
