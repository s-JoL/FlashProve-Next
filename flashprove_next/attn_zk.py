# SPDX-License-Identifier: Apache-2.0
"""Attention-slot gadget on a SHARED hidden-selector opening of the per-request trace matrix H (specs/10 §2.3).

tc_zk (the standalone slot proof) proves ONE attention slot and opens H itself.  This module packages its
attention core (phase 1 + round A: causal mask, exp lookups, tolerance check of specs/06 §2.2 row 7 with the D-9
Jensen / Cauchy-Schwarz terms, four explicit masked sumchecks) as a gadget that a reply circuit instantiates once per
sampled attention slot.  Every read of H is registered on a caller-owned hsel_zk.HSel instance, so any number of
attention slots and the certified blocks of flashprove_next.reply_zk share ONE opening of H.

Per slot (hidden layer l and query position t given as the caller's committed 0/1 cells, reply_zk format
("hid", (space, ex, row), bit)):
  phase 1   q / o code vectors (bf16 codes, flattened heads h*dh + i) committed by the gadget and BOUND to H at
            (t, l) (hs.bind_vector through the caller's layout: the same H entries reply_zk's QKV / OPROJ blocks bind);
            in-circuit integer lifts q_int = round(a C_q[E] / 2^GL), o_int = Lift_{FO}(o) (bf16 decode lookup + exponent
            table, exact integer identities < 2^62); attention witness (mask, idx / E / Ehi, rem limbs, m, Z, N, M2hi, Q2,
            tolerance witness) exactly as tc_zk; a local copy t^ of the hidden t linked to the caller's bits;
            the producer read position tau_p = floor(u (t^ + 1) / 2^UB) (u: public draw supplied by the caller) with
            committed bits, so 0 <= tau_p <= t; the producer check at (tau_p, l) (below); Libra mask coefficients
  round     (attn_round(slots), one commit round shared by all slots, BEFORE hs.rounds()): per slot the points and
            the four explicit masked sumchecks of tc_zk round A; their final evaluations become hidden scalars c1..c4
            registered as point claims on hs:  c1 = K_lift~(tau*, l, g*, i*), c2..c4 = V_lift~(...)  (l hidden, the
            rest public challenges)
Producer check (producer_check, callable at any hidden (tau, l)): commits the k / v code vectors and their lifts
Lift_{FK}, Lift_{FV} (same lift gadget), binds the codes to H's code ports AND the lifts to H's lifted ports at (tau, l).
Accepting therefore implies H[K_lift / V_lift at (tau, l)] = Lift(H[k / v codes at (tau, l)]) entrywise (both bound
vectors are hidden-selector claims on the same extracted H; the lift is enforced in-circuit from the bound codes).

Soundness scope (be precise; specs/01 §4.5 key atoms, specs/10):
  * Covered per attention slot: the consistency lifted-port = Lift(code-port) at ONE row tau_p of layer l, tau_p
    uniform on [0, t] given (t, u) (bias <= (t + 1) / 2^UB).  If D_l = {tau : lifted K/V row (tau, l) != Lift(k/v codes
    at (tau, l))}, the slot's producer check fails with probability |D_l cap [0, t]| / (t + 1) (+ the Schwartz-Zippel
    terms of the binds), i.e. exactly the fraction of key atoms (t, tau, l), tau <= t, made bad by a non-canonical
    lift (their lineage contains the producer of k, v(tau, l)).  The reply circuit draws (t, l) with cost weight
    C_SELF + C_KEY (t + 1) (reply_zk.slot_of).
  * NOT covered by this gadget alone: (a) whether the k / v CODES at (tau_p, l) are the correct KV projection -- the
    caller must run a certified N1 + KV + EMB at the hidden (tau_p, l) bound to the same H row (flashprove_next.reply_zk
    does, through self.tau_ent / self.tpn); without it a cheater that writes wrong-but-consistent codes and
    lifts at row tau (e.g. all zero: Lift(0) = 0) passes the producer check.  (b) Rows other than tau_p: their
    lifted entries are only constrained through the attention sumchecks, i.e. they may be arbitrary field elements
    (no canonical-range argument) -- the reference attention is computed from whatever H holds, so a cheater may alter
    lifted K/V at rows it hopes are not drawn; each slot catches it with the probability above.
    (c) Q2 sums V^2 over the visible keys tau <= t only (SC4 multiplies by the causal mask) and SQ0 uses
    (t^ + 1) Q2, so rows above t and padding rows (which no attention sum and no producer read touches) cannot widen
    the radius (tc_ref module docstring for the bound).
  * q lift: q_int = round_half_away(a C_q[E] / 2^GL), C_q[E] = round(s 2^(E + FQ + GL)), s = 1/sqrt(dh) (float64 value,
    as tc_ref).  |q_int - s q 2^FQ| <= 1/2 + 255 2^(-GL-1) < 1/2 + 2^-26 (GL = 33); tc_ref's float64 path has a
    comparable excess (s q rounded to 53 bits: <= 2^27 2^-53 = 2^-26 units).  The radius term dx_q of
    tc_ref.radius_constants assumes exactly 1/2 unit: the < 2^-26 excess is not included in the radius (an idealisation
    of 2^-26 units).  K, V, o lifts are computed exactly as Lift_F of tc_ref.

Public shapes: tc_ref.Tier (T_pad, G, R, DH, L_MAX, scales, caps, EXP_BITS) and the caller's H layout only.
"""
import time
from fractions import Fraction

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from . import modelcommit as MC
from . import mc_zk as MZ
from . import hsel_zk as HZ
from . import tc_zk as TZ
from . import tc_ref as R
from . import certified as CZ
from .zk_flat import Ex, EqP, SelLt
from .field import P, CPU, ZERO, ONE, hadd, hmul, lagrange_eval, rand_field

VERSION = "fpn-attn-1"
LABEL = b"fpn-attn-v1"
GL = 33                      # guard bits of the lift identity  a C[E] + 2^(GL-1) = y 2^GL + rem
YB = 27                      # |lift| < 2^27 (specs/10 §2.1 b_H; tc_zk's q range)
UB = 32                      # bits of the public producer draw u
SC_DEG = TZ.SC_DEG
SCB = TZ.SCB
_e = TZ._e
_lg = TZ._lg


# ============================================================================ lifts (bf16 code -> integer)
_LT = {}


def lift_table(F, c=None):
    """(E rows, C[E]) of Lift_F with optional constant factor c (Fraction): C[E] = 2^(E+F+GL) (c None, exact; 0 below
    2^0) or floor(c 2^(E+F+GL) + 1/2).  Rows keep 255 C[E] < 2^61 (larger exponents are out of range for any normal
    significand: 128 C[E] / 2^GL >= 2^27)."""
    key = (F, c)
    if key not in _LT:
        Es, Cs = [], []
        for E in range(-133, 121):
            k = E + F + GL
            if c is None:
                C = (1 << k) if k >= 0 else 0
            else:
                x = Fraction(c) * (Fraction(2) ** k)
                C = int((x + Fraction(1, 2)) // 1)
            if 255 * C >= (1 << 61):
                break
            Es.append(E); Cs.append(C)
        _LT[key] = (np.array(Es, dtype=np.int64), np.array(Cs, dtype=np.int64))
    return _LT[key]


def q_scale(tier):
    """the score scale s = 1/sqrt(dh) as the exact rational of its float64 value (tc_ref multiplies in float64)."""
    return Fraction(float(1.0 / np.sqrt(tier.DH)))


def lift_parts(codes, F, c=None):
    """prover / reference: (s, a, E, C, y, rem, L) int64 arrays of the in-circuit lift of bf16 codes.
    Raises ValueError for non-finite or out-of-range codes (|L| >= 2^27)."""
    CZ._bf16_table()
    S_, A_, E_ = CZ._TAB["bf16_dec"]
    cv = np.asarray(codes, dtype=np.int64).ravel()
    if np.any(~CZ._TAB["bf16_fin"][cv]):
        raise ValueError("non-finite bf16 code")
    s, a, E = S_[cv], A_[cv], E_[cv]
    Es, Cs = lift_table(F, c)
    pos = E - Es[0]
    if np.any(pos >= len(Es)):
        raise ValueError("bf16 value out of the lift range")
    C = Cs[pos]
    num = a * C + (1 << (GL - 1))
    y = num >> GL
    rem = num - (y << GL)
    if np.any(y >= (1 << YB)):
        raise ValueError("lift out of range (|Lift| >= 2^27)")
    return s, a, E, C, y, rem, y * (1 - 2 * s)


def lift_np(codes, F, c=None):
    """integer lift of bf16 codes, exactly as the in-circuit gadget (c None: == tc_ref.lift(bf16(code), F))."""
    return lift_parts(codes, F, c)[-1]


def lifted_ports(k_codes, v_codes, tier):
    """H helper: the lifted K / V ports of one (tau, l) from the k / v bf16 code vectors (n_kv dh, flattened g dh + i),
    exactly as tc_ref.build_H (Lift_FK, Lift_FV) and as the in-circuit producer check."""
    return lift_np(k_codes, tier.FK), lift_np(v_codes, tier.FV)


def lift_gadget(ctx, sp, codes, F, c, name):
    """In-circuit lift of the code column on padded space sp (codes: prover values, length sp.n).  Returns (code Col,
    L Col).  Constraints: (code, s, a, E) in the bf16 table; (E, C) in the lift table of (F, c);
    a C + 2^(GL-1) = y 2^GL + rem, 0 <= rem < 2^GL, 0 <= y < 2^27; L = (1 - 2 s) y."""
    pv = ctx.prove
    G.table(ctx, "cz.bf16", CZ._bf16_table())
    tname = f"az.lift.{F}" if c is None else f"az.lift.{F}.{c.numerator}/{c.denominator}"
    Es, Cs = lift_table(F, c)
    G.table(ctx, tname, [Es, Cs])
    vals = None
    if pv:
        try:
            vals = lift_parts(codes, F, c)
        except ValueError as ex:
            ctx.violation(f"lift {name}: {ex}")
            cv = np.asarray(codes, dtype=np.int64).ravel()
            vals = tuple(np.zeros(len(cv), dtype=np.int64) for _ in range(7))
    s, a, E, C, y, rem, L = vals if pv else (None,) * 7
    code = sp.col(f"{name}.code", np.asarray(codes, dtype=np.int64).ravel() if pv else None)
    cs = sp.col(f"{name}.s", s); ca = sp.col(f"{name}.a", a); cE = sp.col(f"{name}.E", E)
    cC = sp.col(f"{name}.C", C); cy = sp.col(f"{name}.y", y); cL = sp.col(f"{name}.L", L)
    G.wire_lookup(ctx, "cz.bf16", sp, [code.ex, cs.ex, ca.ex, cE.ex],
                  [np.asarray(codes, dtype=np.int64).ravel(), s, a, E] if pv else None)
    G.wire_lookup(ctx, tname, sp, [cE.ex, cC.ex], [E, C] if pv else None)
    # rem is only ever its limbs: range_check commits them and constrains  a C + 2^(GL-1) - y 2^GL = sum of limbs
    G.range_check(sp, ca.ex * cC.ex + Ex.const(1 << (GL - 1)) - cy.ex * Ex.const(1 << GL), rem if pv else None, GL,
                  f"{name}.lift")
    G.range_check(sp, cy.ex, y if pv else None, YB, f"{name}.y")
    sp.zero(cL.ex - cy.ex + Ex.const(2) * cs.ex * cy.ex, tag=f"{name}.sign")
    return code, cL


# ============================================================================ reference H layout (reply_zk-compatible)
class RowLayout:
    """H[row, col]: row = position tau (RB = TB bits), col = ((l << PB) | port) << WB | idx  (flashprove_next.reply_zk.h_col).
    ports: name -> port id.  Code vectors 'q' / 'k' are stored split as reply_zk.port_vectors does
    (x.reshape(n, 2, dh/2): '<name>_lo' holds [:, 0, :], '<name>_hi' holds [:, 1, :]); 'o', 'v' and the lifted ports
    'K', 'V' (index g dh + i) are stored whole."""

    SPLIT = ("q", "k")

    def __init__(self, TB, LB, PB, WB, ports):
        self.TB, self.LB, self.PB, self.WB = TB, LB, PB, WB
        self.ports = dict(ports)
        self.CB = WB + PB + LB
        self.nH = self.CB + TB

    def col(self, l, port, i):
        return (((l << self.PB) | self.ports[port]) << self.WB) | i

    def _pre(self, d, port, l_ent):
        pid = self.ports[port]
        return ([("const", 0)] * (self.WB - d) + [("const", (pid >> k) & 1) for k in range(self.PB)] + list(l_ent))

    def binds(self, name, tau_ent, l_ent, sp, outer, vec=None):
        """binds of a vector stored on padded space sp with axes (outer, 'i') (flat index outer dh + i) to H at
        (tau, l): list of (src, dim_bits, row_coords, prefix_coords, vec_wit) for hs.bind_vector."""
        iv, ov = sp.vars("i"), sp.vars(outer)
        DB = len(iv)
        out = []
        v = None if vec is None else np.asarray(vec, dtype=np.int64)
        if name in self.SPLIT:
            for half, suf in ((0, "_lo"), (1, "_hi")):
                ivars = iv[:DB - 1] + ov
                sel = Ex.atom(SelLt([], 1, fixed=((iv[DB - 1], half),)))
                d = len(ivars)
                vw = None if v is None else v.reshape(-1, 2, 1 << (DB - 1))[:, half, :].ravel()
                out.append(((sp, None, ivars, sel), d, list(tau_ent), self._pre(d, name + suf, l_ent), vw))
        else:
            ivars = iv + ov
            d = len(ivars)
            out.append(((sp, None, ivars, None), d, list(tau_ent), self._pre(d, name, l_ent), v))
        return out

    def kv_point(self, tau, l_ent, port, g, i):
        """H coordinates of the lifted K / V entry (tau, l, g, i): column bits (i, g, 0.., port, l), then row bits."""
        zeros = [("const", 0)] * (self.WB - len(i) - len(g))
        pid = self.ports[port]
        return list(i) + list(g) + zeros + [("const", (pid >> k) & 1) for k in range(self.PB)] + list(l_ent) + list(tau)

    def put(self, H, tau, l, name, vec, dh=None):
        """prover helper: write a vector (code vector, split or whole, or a lifted port) into H."""
        v = np.asarray(vec, dtype=np.int64).ravel()
        if name in self.SPLIT:
            x = v.reshape(-1, 2, dh // 2)
            for half, suf in ((0, "_lo"), (1, "_hi")):
                seg = x[:, half, :].ravel()
                c0 = self.col(l, name + suf, 0)
                H[tau, c0:c0 + len(seg)] = seg
            return
        c0 = self.col(l, name, 0)
        H[tau, c0:c0 + len(v)] = v

    def get(self, H, tau, l, name, n, dh=None):
        if name in self.SPLIT:
            h = n // 2
            lo = H[tau, self.col(l, name + "_lo", 0):self.col(l, name + "_lo", 0) + h].reshape(-1, dh // 2)
            hi = H[tau, self.col(l, name + "_hi", 0):self.col(l, name + "_hi", 0) + h].reshape(-1, dh // 2)
            return np.concatenate([lo, hi], axis=1).ravel().astype(np.int64)
        c0 = self.col(l, name, 0)
        return np.asarray(H[tau, c0:c0 + n], dtype=np.int64)


# ============================================================================ small helpers
def rowsel(sp, row):
    """indicator of content row `row` of padded space sp."""
    return Ex.atom(SelLt([], 1, fixed=tuple((v, (row >> v) & 1) for v in range(sp.bits))))


def link_value(hs, entries, weights, name, val=None):
    """A hidden replicated scalar (hs.SC) equal to sum_k weights[k] * (caller cell of entries[k]) -- one sumzero per caller
    space (the claim_point(value=(sp, ex)) pattern of hsel_zk).  entries: ("hid", (sp, ex, row), bit).  Returns its name.
    val: the prover's value of the scalar (default: from the entries' bits)."""
    groups = {}
    for (kind, (sp, ex, row), bit), w in zip(entries, weights):
        assert kind == "hid"
        HZ._check_space(sp, "linked caller")
        g = groups.setdefault(sp.name, {"sp": sp, "e": Ex.const(0), "v": 0})
        g["e"] = g["e"] + ex * rowsel(sp, row) * Ex.const(int(w) % P)
        if hs.pv:
            g["v"] += int(w) * int(bit)
    if val is None and hs.pv:
        val = sum(g["v"] for g in groups.values())
    nm = hs.scalar(name, (int(val) % P, 0, 0) if hs.pv else None)
    parts = []
    for gi, g in enumerate(groups.values()):
        pn = nm if len(groups) == 1 else hs.scalar(f"{name}.p{gi}", (g["v"] % P, 0, 0) if hs.pv else None)
        g["sp"].sumzero(g["e"] - hs.SC.on(pn, g["sp"]) * HZ.ind0(g["sp"]), tag=f"{name}.link{gi}")
        parts.append(pn)
    if len(groups) > 1:
        SY = hs.ctx.space(f"{name}.Y", [("x", SCB)])
        tot = Ex.const(0)
        for pn in parts:
            tot = tot + hs.SC.on(pn, SY)
        SY.zero(hs.SC.on(nm, SY) - tot, tag=f"{name}.linksum")
    return nm


def key_draw(u, t):
    """producer read position tau_p = floor(u (t + 1) / 2^UB), 0 <= tau_p <= t (u < 2^UB public)."""
    assert 0 <= u < (1 << UB)
    return (int(u) * (int(t) + 1)) >> UB


def bind_all(hs, binds, ex):
    for (sp, _, ivars, sel), d, row, pre, vw in binds:
        hs.bind_vector((sp, ex, ivars, sel), d, row, pre, vec_wit=vw)


# ============================================================================ producer check
def producer_check(ctx, hs, tier, layout, tau_ent, l_ent, kcodes=None, vcodes=None, tag="pc", H_kv=None):
    """At the hidden (tau, l): bind the k / v code vectors to H's code ports and their in-circuit lifts to H's lifted
    ports K, V.  Accepting => H[K, V at (tau, l)] = (Lift_FK, Lift_FV)(H[k, v codes at (tau, l)]).
    kcodes / vcodes: the prover's code vectors (n_kv dh, flat g dh + i); H_kv: optional (K row, V row) the prover reads
    from its H for the attention (prover diagnostic only; the circuit binds both to H).  Must be called in phase 1 (before HSel's pre round).  Returns the space."""
    pv = ctx.prove
    t = tier
    sp = ctx.space(f"{tag}.P", [("g", t.GB), ("i", t.DB)])
    kc, kL = lift_gadget(ctx, sp, kcodes, t.FK, None, f"{tag}.k")
    vc, vL = lift_gadget(ctx, sp, vcodes, t.FV, None, f"{tag}.v")
    if pv and H_kv is not None:
        try:
            Kl, Vl = lifted_ports(kcodes, vcodes, t)
            if not (np.array_equal(Kl, np.asarray(H_kv[0]).ravel()) and np.array_equal(Vl, np.asarray(H_kv[1]).ravel())):
                ctx.violation("producer: the lifted K/V row the attention reads at tau_p differs from Lift(codes at (tau_p, l))")
        except ValueError:
            pass
    vec = lambda c: (np.asarray(c, dtype=np.int64) if pv else None)
    bind_all(hs, layout.binds("k", tau_ent, l_ent, sp, "g", vec(kcodes)), kc.ex)
    bind_all(hs, layout.binds("v", tau_ent, l_ent, sp, "g", vec(vcodes)), vc.ex)
    Kl = Vl = None
    if pv:
        Kl, Vl = kL.v, vL.v
    bind_all(hs, layout.binds("K", tau_ent, l_ent, sp, "g", Kl), kL.ex)
    bind_all(hs, layout.binds("V", tau_ent, l_ent, sp, "g", Vl), vL.ex)
    return sp


def _mask_weights(k, n, deg, r):
    """MK layout (sc 2, b 5, k 3): weights of sum_x g_k(x) and g_k(r) (tc_zk._mask_weights on a 2^10 cube)."""
    gw = np.zeros((3, 1 << 10), dtype=np.uint64); rw = np.zeros((3, 1 << 10), dtype=np.uint64)
    half = pow(2, n - 1, P)
    for b in range(n):
        x = ONE
        for kk in range(deg + 1):
            pos = (k << 8) | (b << 3) | kk
            gw[0, pos] = (2 if kk == 0 else 1) * half % P
            rw[0, pos], rw[1, pos], rw[2, pos] = x
            x = hmul(x, _e(r[b]))
    return gw, rw


def claims_per_slot(layout):
    """HSel claim slots used by one attention slot: 4 point claims + the code / lift (+ optional lse_lo/hi) binds."""
    nb = lambda nm: 2 if nm in layout.SPLIT else 1
    lse = 2 if "lse_lo" in layout.ports else 0
    return 4 + nb("q") + nb("o") + nb("k") + nb("v") + 2 + lse


_LSE_TABS = {}


def _lse_tables():
    if not _LSE_TABS:
        from . import sampling_v3 as S3
        LNT, LN2 = S3.ln_table()
        ez = np.arange(R.EB, 48, dtype=np.int64)
        Bz = np.left_shift(np.int64(1), ez - 15)
        Es = np.arange(-149, -9, dtype=np.int64)
        ks = Es + R.FL_LSE
        Al = np.where((ks >= -24) & (ks <= 14), np.left_shift(np.int64(1), np.clip(ks, 0, 14)), 0)
        Bl = np.where((ks >= -24) & (ks <= 14), np.left_shift(np.int64(1), np.clip(-ks, 0, 24)), 1)
        _LSE_TABS["lne"] = [ez, Bz]
        _LSE_TABS["lnt"] = [np.arange(1 << 15, 1 << 16, dtype=np.int64), np.asarray(LNT, dtype=np.int64)]
        _LSE_TABS["p2"] = [Es, Al, Bl]
        _LSE_TABS["LN2"] = int(LN2)
    return _LSE_TABS


# ============================================================================ the attention slot
class AttnSlot:
    """One attention slot (hidden t, l) on a shared HSel.

    ctx: gadgets.Ctx; hs: hsel_zk.HSel of the caller; tier: tc_ref.Tier (public profile shapes); layout: H layout
    (RowLayout API: binds / kv_point); t_ent, l_ent: the caller's hidden bit cells of t (tier.nt bits) and l
    (layout.LB bits), entries ("hid", (space, ex, row), bit); u: public producer draw (< 2^UB); wit (prover):
      q_codes, o_codes  (n_q dh,) bf16 codes at (t, l), flat h dh + i (h = g R + rho)
      K, V              (T_pad, G, DH) int64: lifted K / V rows of layer l as the prover's H holds them
      k_codes, v_codes  (T_pad, n_kv dh) code vectors of layer l (row tau_p is read)
      t, l              ints (t_att / tau_p_override: forgery tests only)
    Construct in phase 1; then put attn_round([slots...]) into ctx.cir.rounds BEFORE hs.rounds()."""

    def __init__(self, ctx, hs, tier, layout, t_ent, l_ent, u, wit=None, tag="att", B=CPU):
        t = tier
        self.ctx, self.hs, self.t, self.lay, self.tag, self.B = ctx, hs, t, layout, tag, B
        pv = self.pv = ctx.prove
        assert pv == (wit is not None)
        assert len(t_ent) == t.nt and len(l_ent) == layout.LB
        assert layout.TB == t.nt, "H rows must be the T_pad positions"
        assert t.nt + t.GB + t.DB <= 32 and t.HB + t.DB <= layout.WB and t.GB + t.DB <= layout.WB
        self.l_ent = list(l_ent)
        self.u = int(u)
        nt, GB, DB, HB = t.nt, t.GB, t.DB, t.HB
        S = self.S = {}
        info = self.info = {}
        t0 = time.time()
        s = None
        if pv:
            t_att = int(wit.get("t_att", wit["t"]))
            tau_p = int(wit.get("tau_p_override", key_draw(self.u, t_att)))   # override: forgery tests only
            q_int = lift_np(wit["q_codes"], t.FQ, q_scale(t)).reshape(t.G * t.R, t.DH)
            o_int = lift_np(wit["o_codes"], t.FO).reshape(t.G * t.R, t.DH)
            s = R.Slot(t, int(wit["l"]), t_att, q_int, o_int, np.asarray(wit["K"], dtype=np.int64),
                       np.asarray(wit["V"], dtype=np.int64), tau_p)
            R.witness(s, check=not Z.FORCE)
            if t_att != int(wit["t"]):
                ctx.violation("attention t differs from the caller's hidden t")
            self.tau_p = tau_p
        self.s = s
        w = s.w if pv else None
        nm = lambda x: f"{tag}.{x}"
        rcI = R.radius_constants(t)["ints"]
        Wc = rcI["W"]
        # ---- t^ (linked to the caller's bits), tau_p bits, key draw
        tn = link_value(hs, t_ent, [1 << k for k in range(nt)], nm("t"), val=(s.t if pv else None))
        SB = ctx.space(nm("B"), [("k", 5)])
        tpv = None
        if pv:
            tpv = np.zeros(32, dtype=np.int64)
            for k in range(nt):
                tpv[k] = (self.tau_p >> k) & 1
        tpb = SB.col("taup", tpv)
        SB.zero(tpb.ex * (tpb.ex - Ex.const(1)), tag=nm("tp.bool"))
        msk = np.zeros(32, dtype=np.int64); msk[:nt] = 1
        SB.zero((Ex.const(1) - SB.pub("tpm", msk).ex) * tpb.ex, tag=nm("tp.zero"))
        wtp = np.zeros(32, dtype=np.int64); wtp[:nt] = [1 << k for k in range(nt)]
        tpn = hs.scalar(nm("tp"), (self.tau_p, 0, 0) if pv else None)
        self.tpn = tpn                       # hidden scalar tau_p (callers link their own position columns to it)
        SB.sumzero(SB.pub("wtp", wtp).ex * tpb.ex - hs.SC.on(tpn, SB) * HZ.ind0(SB), tag=nm("tp.val"))
        SX = ctx.space(nm("X"), [("x", SCB)])
        X_ = lambda n_: hs.SC.on(n_, SX)
        kd_v = None
        if pv:
            kd_v = np.full(SX.n, self.u * (s.t + 1) - (self.tau_p << UB), dtype=object)
        # 0 <= u (t^ + 1) - tau_p 2^UB < 2^UB   <=>   tau_p = floor(u (t^ + 1) / 2^UB)   (no wrap: < 2^(UB + nt + 1))
        TZ.rc(SX, Ex.const(self.u) * (X_(tn) + Ex.const(1)) - X_(tpn) * Ex.const(1 << UB), kd_v, UB, nm("kdraw"))
        self.tau_ent = [("hid", (SB, tpb.ex, k), (self.tau_p >> k) & 1 if pv else None) for k in range(nt)]
        # ---- producer check at (tau_p, l)
        kc = vc = Hkv = None
        if pv:
            kc = np.asarray(wit["k_codes"][self.tau_p], dtype=np.int64)
            vc = np.asarray(wit["v_codes"][self.tau_p], dtype=np.int64)
            Hkv = (s.K[self.tau_p].ravel(), s.V[self.tau_p].ravel())
        producer_check(ctx, hs, t, layout, self.tau_ent, self.l_ent, kc, vc, tag=nm("pc"), H_kv=Hkv)
        # ---- q / o codes at (t, l): lifts, binds
        CS = ctx.space(nm("C"), [("h", HB), ("i", DB)])
        qcode, qL = lift_gadget(ctx, CS, wit["q_codes"] if pv else None, t.FQ, q_scale(t), nm("q"))
        ocode, oL = lift_gadget(ctx, CS, wit["o_codes"] if pv else None, t.FO, None, nm("o"))
        bind_all(hs, layout.binds("q", t_ent, self.l_ent, CS, "h", wit["q_codes"] if pv else None), qcode.ex)
        bind_all(hs, layout.binds("o", t_ent, self.l_ent, CS, "h", wit["o_codes"] if pv else None), ocode.ex)
        # ---- attention witness (tc_zk phase 1)
        TS = ctx.space(nm("T"), [("tau", nt)])
        mask = TS.col("mask", w["mask"] if pv else None)
        KS = ctx.space(nm("K"), [("tau", nt), ("h", HB)])
        ET, EHT = R.expt(t.EXP_BITS)
        etab = f"az.exp{t.EXP_BITS}"
        G.table(ctx, etab, [np.arange(1 << t.EXP_BITS, dtype=np.int64), ET, EHT])
        idx = KS.col("idx", w["idx"].ravel() if pv else None)
        E = KS.col("E", w["E"].ravel() if pv else None)
        Ehi = KS.col("Ehi", w["Ehi"].ravel() if pv else None)
        ctx.cir.lookup(etab, KS.nv, [idx.ex, E.ex, Ehi.ex], sel=KS.sel(),
                       vals=[w["idx"].ravel(), w["E"].ravel(), w["Ehi"].ravel()] if pv else None,
                       sel_vals=np.ones(KS.n, dtype=np.int64) if pv else None)
        SHb = t.SH
        lw = [16] * (SHb // 16) + ([SHb % 16] if SHb % 16 else [])
        rl = TZ.limbs(KS, "rem", [((w["rem"].ravel() >> (16 * k)) & ((1 << wd) - 1)) for k, wd in enumerate(lw)] if pv else None,
                      lw, ctx)
        rem_ex = Ex.const(0)
        for k, c in enumerate(rl):
            rem_ex = rem_ex + c.ex * Ex.const(1 << (16 * k))
        D_ex = idx.ex * Ex.const(1 << SHb) + rem_ex - Ex.const(1 << (SHb - 1))
        HSp = ctx.space(nm("H"), [("h", HB)])
        m_c = TZ.col_obj(HSp, "m", w["m"] if pv else None)
        Z_c = TZ.col_obj(HSp, "Z", w["Z"] if pv else None)
        Zc_c = TZ.col_obj(HSp, "Zc", w["Zc"] if pv else None)
        rz_c = TZ.col_obj(HSp, "rz", w["rz"] if pv else None)
        GS = ctx.space(nm("G"), [("g", GB), ("i", DB)])
        cs = {}
        for k_ in ["N", "M2", "oh", "rd", "aoh", "W2", "rW", "SJ", "SQ", "rS"]:
            cs[k_] = TZ.col_obj(CS, k_, w[k_].ravel() if pv else None)
        Q2_c = TZ.col_obj(GS, "Q2", w["Q2"].ravel() if pv else None)
        SQ0_c = TZ.col_obj(GS, "SQ0", w["SQ0"].ravel() if pv else None)
        MK = ctx.space(nm("M"), [("sc", 2), ("b", 5), ("k", 3)])
        coefs = {}
        mk_v = None
        self.n_sc = nt + GB + DB
        if pv:
            mk_v = np.zeros((3, 1 << 10), dtype=np.uint64)
            for k in range(4):
                cf = []
                for b in range(self.n_sc):
                    row = []
                    for kk in range(SC_DEG[k] + 1):
                        rf = rand_field(3)
                        v = (int(rf[0]), int(rf[1]), int(rf[2]))
                        row.append(v)
                        pos = (k << 8) | (b << 3) | kk
                        mk_v[0, pos], mk_v[1, pos], mk_v[2, pos] = v
                    cf.append(row)
                coefs[k] = cf
        mkc = MZ.ecol(MK, "g", mk_v)
        # mask: boolean, mask(0) = 1, monotone non-increasing, sum = t^ + 1  ==>  mask = [tau <= t^]
        TS.zero(mask.ex * (mask.ex - Ex.const(1)), tag=nm("mask.bool"))
        TS.zero(mask.ex - Ex.const(1), sel=TS.ind(tau=0), tag=nm("mask.first"))
        dlt = mask.ex - mask.shifted(1)
        TS.zero(dlt * (dlt - Ex.const(1)), sel=TS.lt("tau", t.T_pad - 1), tag=nm("mask.mono"))
        TS.sumzero(mask.ex - (hs.SC.on(tn, TS) + Ex.const(1)) * TS.ind(tau=0), tag=nm("mask.sum"))
        mask_ks = MZ.view(mask, KS, {"tau": "tau"})
        KS.zero((Ex.const(1) - mask_ks) * E.ex, tag=nm("E.masked"))
        Z_ks = MZ.view(Z_c, KS, {"h": "h"})
        KS.sumzero(E.ex - Z_ks * KS.ind(tau=0), eq_axes=("h",), tag=nm("Z"))
        TZ.rc(HSp, Z_c.ex - Ex.const(1 << R.EB), (w["Z"] - (1 << R.EB)) if pv else None, 38, "Z.lo")
        HSp.zero(Z_c.ex - Zc_c.ex * Ex.const(1 << R.CB) - rz_c.ex, tag=nm("Zc"))
        TZ.rc(HSp, rz_c.ex, w["rz"] if pv else None, R.CB, "rz")
        TZ.rc(HSp, Zc_c.ex, w["Zc"] if pv else None, 27, "Zc")
        # tolerance per component (specs/06 §2.2 row 7; same as tc_zk)
        Zcs = MZ.view(Z_c, CS, {"h": "h"}); Zccs = MZ.view(Zc_c, CS, {"h": "h"})
        gmap = [("h", t.RB + k) for k in range(GB)]
        Q2cs = MZ.view(Q2_c, CS, {"g": gmap, "i": "i"}); SQ0cs = MZ.view(SQ0_c, CS, {"g": gmap, "i": "i"})
        oi = oL.ex
        c_ = {k: v.ex for k, v in cs.items()}
        CS.zero(c_["N"] * Ex.const(64) - c_["oh"] * Zcs - c_["rd"], tag=nm("oh"))
        if pv:
            Zr = np.repeat(w["Z"].astype(object), t.DH); Zcr = np.repeat(w["Zc"].astype(object), t.DH)
            W_ = {k: (w[k].ravel().astype(object)) for k in ("oh", "rd", "aoh", "W2", "rW", "SJ", "SQ", "rS", "X", "sp", "sm")}
        TZ.rc(CS, c_["rd"], W_["rd"] if pv else None, 38, "rd.lo")
        TZ.rc(CS, Zcs - Ex.const(1) - c_["rd"], (Zr - 1 - W_["rd"]) if pv else None, 38, "rd.hi")
        CS.zero((c_["aoh"] - c_["oh"]) * (c_["aoh"] + c_["oh"]), tag=nm("aoh"))
        TZ.rc(CS, c_["aoh"], W_["aoh"] if pv else None, 26, "aoh")
        TZ.rc(CS, c_["M2"], w["M2"].ravel() if pv else None, 60, "M2")
        CS.zero(c_["W2"] * Zccs - c_["M2"] - c_["rW"], tag=nm("W2"))
        TZ.rc(CS, c_["rW"], W_["rW"] if pv else None, 27, "rW.lo")
        TZ.rc(CS, Zccs - Ex.const(1) - c_["rW"], (Zcr - 1 - W_["rW"]) if pv else None, 27, "rW.hi")
        TZ.rc(CS, c_["W2"], W_["W2"] if pv else None, 36, "W2")
        X_ex = (Ex.const(rcI["KH"] << 12) * c_["W2"] - Ex.const(rcI["KL"]) * (c_["aoh"] * c_["aoh"] - Ex.const(2) * c_["aoh"])
                + Ex.const(1 << 12))
        TZ.rc(CS, c_["SJ"] * c_["SJ"] * Ex.const(1 << 12) - X_ex,
              (W_["SJ"] * W_["SJ"] * 4096 - W_["X"]) if pv else None, 40, "SJ.up")
        TZ.rc(CS, X_ex - (c_["SJ"] - Ex.const(1)) * (c_["SJ"] - Ex.const(1)) * Ex.const(1 << 12) - Ex.const(1),
              (W_["X"] - (W_["SJ"] - 1) * (W_["SJ"] - 1) * 4096 - 1) if pv else None, 40, "SJ.tight")
        TZ.rc(CS, c_["SJ"], W_["SJ"] if pv else None, 26, "SJ")
        # SQ0 = ceil(sqrt((t^ + 1) Q2 + 1)); Q2 < 2^(62 - nt) over visible keys => (t^ + 1) Q2 + 1 < 2^62 + 1 < p/2 (no wrap)
        TZ.rc(GS, Q2_c.ex, w["Q2"].ravel() if pv else None, 62 - nt, "Q2")
        GQv = sq0 = None
        if pv:
            GQv = (int(np.asarray(w["mask"]).sum()) * w["Q2"].ravel().astype(object) + 1)
            sq0 = w["SQ0"].ravel().astype(object)
        gq_ex = Q2_c.ex * (hs.SC.on(tn, GS) + Ex.const(1)) + Ex.const(1)
        TZ.rc(GS, SQ0_c.ex * SQ0_c.ex - gq_ex, (sq0 * sq0 - GQv) if pv else None, 34, "SQ0.up")
        TZ.rc(GS, gq_ex - (SQ0_c.ex - Ex.const(1)) * (SQ0_c.ex - Ex.const(1)) - Ex.const(1),
              (GQv - (sq0 - 1) * (sq0 - 1) - 1) if pv else None, 34, "SQ0.tight")
        TZ.rc(GS, SQ0_c.ex, sq0 if pv else None, 31, "SQ0")
        CS.zero(c_["SQ"] * Zcs - SQ0cs * Ex.const(32) - c_["rS"], tag=nm("SQ"))
        TZ.rc(CS, c_["rS"], W_["rS"] if pv else None, 38, "rS.lo")
        TZ.rc(CS, Zcs - Ex.const(1) - c_["rS"], (Zr - 1 - W_["rS"]) if pv else None, 38, "rS.hi")
        TZ.rc(CS, c_["SQ"], W_["SQ"] if pv else None, 20, "SQ")
        rhs = (Ex.const(rcI["A"]) * c_["SJ"] + Ex.const(rcI["B"]) * c_["aoh"] + Ex.const(rcI["Cq"]) * c_["SQ"]
               + Ex.const(rcI["C0"]))
        dd = (oi - c_["oh"]) * Ex.const(1 << Wc)
        if pv and (any(int(x) < 0 for x in W_["sp"]) or any(int(x) < 0 for x in W_["sm"])):
            ctx.violation("attention output outside the certified radius")
        TZ.rc(CS, rhs - dd, W_["sp"] if pv else None, 48, "tol.p")
        TZ.rc(CS, rhs + dd, W_["sm"] if pv else None, 48, "tol.m")
        # ---- LSE observation (M1, specs/06 §5.2): bound to H ports lse_lo / lse_hi when present in layout
        if "lse_lo" in layout.ports:
            ltab = _lse_tables()
            G.table(ctx, "cz.f32hi", CZ._f32hi_table())
            G.table(ctx, "az.lne", ltab["lne"])
            G.table(ctx, "az.lnt", ltab["lnt"])
            G.table(ctx, "az.lsep2", ltab["p2"])
            SM = t.FQ + t.FK - R.FL_LSE
            lw_ = None
            if pv:
                lw_ = R.lse_witness(t, w, wit["lse_u32"])
                s.w["lse_max_ratio"] = lw_["lse_max_ratio"]
                if lw_["lse_bad"]:
                    ctx.violation("attention LSE outside the certified radius")
            m24_c = TZ.col_obj(HSp, "m24", lw_["m24"] if pv else None)
            rm_c = HSp.col("rm", lw_["rm"] if pv else None)
            HSp.zero(m_c.ex - m24_c.ex * Ex.const(1 << SM) - rm_c.ex, tag=nm("m24.def"))
            TZ.rc(HSp, rm_c.ex, lw_["rm"] if pv else None, SM, "m24.rem")
            TZ.rc(HSp, m24_c.ex + Ex.const(1 << 38), (lw_["m24"].astype(object) + (1 << 38)) if pv else None, 39, "m24.srng")
            ez_c = HSp.col("ez", lw_["ez"] if pv else None)
            Bz_c = HSp.col("Bz", lw_["Bz"] if pv else None)
            mant_c = HSp.col("mant", lw_["mant"] if pv else None)
            lnv_c = HSp.col("lnv", lw_["lnv"] if pv else None)
            G.wire_lookup(ctx, "az.lne", HSp, [ez_c.ex, Bz_c.ex], [lw_["ez"], lw_["Bz"]] if pv else None)
            G.wire_lookup(ctx, "az.lnt", HSp, [mant_c.ex, lnv_c.ex], [lw_["mant"], lw_["lnv"]] if pv else None)
            remz_ex = Z_c.ex - mant_c.ex * Bz_c.ex
            TZ.rc(HSp, remz_ex, lw_["rz_ln"] if pv else None, 33, "lnz.rlo")
            TZ.rc(HSp, Bz_c.ex - Ex.const(1) - remz_ex, (lw_["Bz"].astype(object) - 1 - lw_["rz_ln"]) if pv else None, 33, "lnz.rhi")
            L24_ex = m24_c.ex + (ez_c.ex - Ex.const(R.EB)) * Ex.const(ltab["LN2"]) + lnv_c.ex
            lse_lo = HSp.col("lse_lo", lw_["lse_lo"] if pv else None)
            lse_hi = HSp.col("lse_hi", lw_["lse_hi"] if pv else None)
            ls_s = HSp.col("ls_s", lw_["ls_s"] if pv else None)
            ls_ahi = HSp.col("ls_ahi", lw_["ls_ahi"] if pv else None)
            ls_E = TZ.col_obj(HSp, "ls_E", lw_["ls_E"] if pv else None)
            ls_A = HSp.col("ls_A", lw_["ls_A"] if pv else None)
            ls_B = HSp.col("ls_B", lw_["ls_B"] if pv else None)
            ls_y = HSp.col("ls_y", lw_["ls_y"] if pv else None)
            ls_rem = HSp.col("ls_rem", lw_["ls_rem"] if pv else None)
            ls_o24 = TZ.col_obj(HSp, "ls_o24", lw_["ls_o24"] if pv else None)
            TZ.rc(HSp, lse_lo.ex, lw_["lse_lo"] if pv else None, 16, "lse.lo")
            G.wire_lookup(ctx, "cz.f32hi", HSp, [lse_hi.ex, ls_s.ex, ls_ahi.ex, ls_E.ex],
                          [lw_["lse_hi"], lw_["ls_s"], lw_["ls_ahi"], lw_["ls_E"]] if pv else None)
            G.wire_lookup(ctx, "az.lsep2", HSp, [ls_E.ex, ls_A.ex, ls_B.ex],
                          [lw_["ls_E"], lw_["ls_A"], lw_["ls_B"]] if pv else None)
            lsa_ex = ls_ahi.ex * Ex.const(1 << 16) + lse_lo.ex
            HSp.zero(lsa_ex * ls_A.ex - ls_y.ex * ls_B.ex - ls_rem.ex, tag=nm("lse.shr"))
            TZ.rc(HSp, ls_rem.ex, lw_["ls_rem"] if pv else None, 25, "lse.rlo")
            TZ.rc(HSp, ls_B.ex - Ex.const(1) - ls_rem.ex, (lw_["ls_B"].astype(object) - 1 - lw_["ls_rem"]) if pv else None, 25, "lse.rhi")
            TZ.rc(HSp, ls_y.ex, lw_["ls_y"] if pv else None, 38, "lse.y")
            HSp.zero(ls_o24.ex - ls_y.ex + Ex.const(2) * ls_s.ex * ls_y.ex, tag=nm("lse.sgn"))
            RL24 = rcI["RL24"]
            TZ.rc(HSp, Ex.const(RL24 - 1) - (ls_o24.ex - L24_ex), lw_["lse_sp"] if pv else None, 41, "lse.p")
            TZ.rc(HSp, Ex.const(RL24 - 1) + (ls_o24.ex - L24_ex), lw_["lse_sm"] if pv else None, 41, "lse.m")
            hs.bind_vector(lse_lo, HB, list(t_ent), layout._pre(HB, "lse_lo", self.l_ent),
                           vec_wit=lw_["lse_lo"] if pv else None)
            hs.bind_vector(lse_hi, HB, list(t_ent), layout._pre(HB, "lse_hi", self.l_ent),
                           vec_wit=lw_["lse_hi"] if pv else None)
        S.update(TS=TS, KS=KS, HSp=HSp, CS=CS, GS=GS, MK=MK, SX=SX, mask=mask, E=E, Ehi=Ehi, D_ex=D_ex, m_c=m_c,
                 cs=cs, Q2=Q2_c, qL=qL, mkc=mkc, coefs=coefs, mask_ks=mask_ks)
        info["phase1_build_s"] = round(time.time() - t0, 3)

    # ---------------------------------------------------------------- round (tc_zk round A, claims on hs)
    def round(self, tr):
        t0 = time.time()
        t, hs, S, pv, B = self.t, self.hs, self.S, self.pv, self.B
        ctx, tag = self.ctx, self.tag
        nt, GB, DB = t.nt, t.GB, t.DB
        nm = lambda x: f"{tag}.{x}"
        pts = {1: dict(tau=tr.exts(nt), g=tr.exts(GB), r=tr.exts(t.RB)),
               2: dict(g=tr.exts(GB), r=tr.exts(t.RB), i=tr.exts(DB)),
               3: dict(g=tr.exts(GB), r=tr.exts(t.RB), i=tr.exts(DB)),
               4: dict(g=tr.exts(GB), i=tr.exts(DB))}
        rhos = tr.exts(4)
        pf = ctx.cir.pf
        ent = pf.setdefault("attn", {}).setdefault(tag, []) if pv else pf.get("attn", {}).get(tag)
        if not pv and (not isinstance(ent, list) or len(ent) != 4):
            raise Z.ProofError(f"{tag}: missing attention sumchecks")
        fin = {}
        for k in range(1, 5):
            if pv:
                tabs = TZ._sc_tables(t, k, pts[k], self.s, B)
                Sp, msgs, r, _ = TZ.esc_prove(B, tabs, SC_DEG[k - 1], S["coefs"][k - 1], rhos[k - 1], tr)
                ent.append({"S": list(Sp), "msgs": msgs})
                vfin = lagrange_eval([_e(v) for v in msgs[-1]], r[-1])
            else:
                if not isinstance(ent[k - 1], dict):
                    raise Z.ProofError(f"{tag}: malformed attention sumcheck")
                Sp = _e(ent[k - 1]["S"])
                r, vfin = TZ.esc_verify(Sp, ent[k - 1]["msgs"], self.n_sc, SC_DEG[k - 1], tr, f"{tag}.SC{k}")
            fin[k] = dict(Sp=_e(Sp), r=r, v=vfin)

        def split(r):
            return r[:DB], r[DB:DB + GB], r[DB + GB:]
        vals = self._values(pts, fin, split) if pv else {}
        names = {}
        for k_ in ("S1", "S2", "S3", "S4", "G1", "G2", "G3", "G4", "gr1", "gr2", "gr3", "gr4",
                   "a1", "b1", "c1", "b2", "c2", "b3", "c3", "a4", "c4"):
            names[k_] = hs.scalar(nm(k_), vals.get(k_))
        on = lambda k_, sp: hs.SC.on(names[k_], sp)
        SCX = S["SX"]
        X_ = lambda k_: on(k_, SCX)
        TS, KS, CS, GS, MK = S["TS"], S["KS"], S["CS"], S["GS"], S["MK"]
        mkc, cs = S["mkc"], S["cs"]
        for k in range(1, 5):
            SCX.zero(X_(f"S{k}") + Ex.const(rhos[k - 1]) * X_(f"G{k}") - Ex.const(fin[k]["Sp"]), tag=nm(f"S{k}'"))
        mv = MK.vars("k") + MK.vars("b") + MK.vars("sc")
        for k in range(1, 5):
            gw, rw = _mask_weights(k - 1, self.n_sc, SC_DEG[k - 1], fin[k]["r"])
            MK.sumzero(Ex.atom(MZ.EVec(nm(f"Gw{k}"), gw, mv)) * mkc.ex - on(f"G{k}", MK) * HZ.ind0(MK), tag=nm(f"G{k}"))
            MK.sumzero(Ex.atom(MZ.EVec(nm(f"rw{k}"), rw, mv)) * mkc.ex - on(f"gr{k}", MK) * HZ.ind0(MK), tag=nm(f"gr{k}"))
        kv = KS.vars("h") + KS.vars("tau")
        # SC1: S1 = sum_{tau,h} eq(pt1) mask (m - D);  a1 = [eq(tau_r) mask]~(tau*);  b1 = [eq(g_r) qq]~(g*, i*)
        p1 = list(pts[1]["r"]) + list(pts[1]["g"]) + list(pts[1]["tau"])
        m_ks = MZ.view(S["m_c"], KS, {"h": "h"})
        KS.sumzero(Ex.atom(EqP(kv, p1)) * S["mask_ks"] * (m_ks - S["D_ex"]) - on("S1", KS) * HZ.ind0(KS), tag=nm("S1"))
        i1, g1, ta1 = split(fin[1]["r"])
        tv = TS.vars("tau")
        TS.sumzero(Ex.atom(EqP(tv, ta1)) * Ex.atom(EqP(tv, pts[1]["tau"])) * S["mask"].ex - on("a1", TS) * HZ.ind0(TS),
                   tag=nm("a1"))
        wt1 = TZ._qq_weights(t, pts[1], g1, i1)
        CS.sumzero(Ex.atom(MZ.EVec(nm("wt1"), wt1[:, :1 << (t.HB + DB)], CS.vars("i") + CS.vars("h"))) * S["qL"].ex
                   - on("b1", CS) * HZ.ind0(CS), tag=nm("b1"))
        SCX.zero(X_("a1") * X_("b1") * X_("c1") + Ex.const(rhos[0]) * X_("gr1") - Ex.const(fin[1]["v"]), tag=nm("SC1.fin"))
        cv = CS.vars("i") + CS.vars("h")
        for k, Mc, Ec in ((2, cs["N"], S["E"]), (3, cs["M2"], S["Ehi"])):
            pk = pts[k]
            CS.sumzero(Ex.atom(EqP(cv, list(pk["i"]) + list(pk["r"]) + list(pk["g"]))) * (Mc.ex - on(f"S{k}", CS)),
                       tag=nm(f"S{k}"))
            ik, gk_, tk = split(fin[k]["r"])
            KS.sumzero(Ex.atom(EqP(kv, list(pk["r"]) + list(gk_) + list(tk))) * (Ec.ex - on(f"b{k}", KS)), tag=nm(f"b{k}"))
            cpub = hmul(TZ.eqv(pk["g"], gk_), TZ.eqv(pk["i"], ik))
            cl = X_(f"c{k}") if k == 2 else X_(f"c{k}") * X_(f"c{k}")
            SCX.zero(Ex.const(cpub) * X_(f"b{k}") * cl + Ex.const(rhos[k - 1]) * X_(f"gr{k}") - Ex.const(fin[k]["v"]),
                     tag=nm(f"SC{k}.fin"))
        gv = GS.vars("i") + GS.vars("g")
        GS.sumzero(Ex.atom(EqP(gv, list(pts[4]["i"]) + list(pts[4]["g"]))) * (S["Q2"].ex - on("S4", GS)), tag=nm("S4"))
        # SC4 sums eq(g''', i''') mask(tau) V^2 (causal mask: Q2 over the visible keys only); a4 = mask~(tau*)
        i4, g4, t4 = split(fin[4]["r"])
        TS.sumzero(Ex.atom(EqP(tv, t4)) * S["mask"].ex - on("a4", TS) * HZ.ind0(TS), tag=nm("a4"))
        cpub = hmul(TZ.eqv(pts[4]["g"], g4), TZ.eqv(pts[4]["i"], i4))
        SCX.zero(Ex.const(cpub) * X_("a4") * X_("c4") * X_("c4") + Ex.const(rhos[3]) * X_("gr4") - Ex.const(fin[4]["v"]),
                 tag=nm("SC4.fin"))
        # point claims on H: c1 = K_lift~(tau*, l, g*, i*), c2..c4 = V_lift~(...)   (l hidden, the rest public)
        pub = lambda xs: [("pub", _e(x)) for x in xs]
        for k, port in ((1, "K"), (2, "V"), (3, "V"), (4, "V")):
            ik, gk_, tk = split(fin[k]["r"])
            coords = self.lay.kv_point(pub(tk), self.l_ent, port, pub(gk_), pub(ik))
            hs.claim_point(coords, value=names[f"c{k}"], value_wit=vals.get(f"c{k}"))
        self.info["round_s"] = round(time.time() - t0, 3)

    def _values(self, pts, fin, split):
        """prover: the hidden scalars of the round (tc_zk._round_a_values without the producer reads)."""
        t, s = self.t, self.s
        w = s.w
        coefs = self.S["coefs"]
        mle = TZ._mle
        ext, fe = TZ.ext, TZ.fe
        v = {}
        for k in range(1, 5):
            v[f"G{k}"] = TZ.mask_sum(coefs[k - 1], len(fin[k]["r"]))
            v[f"gr{k}"] = TZ.mask_eval(coefs[k - 1], fin[k]["r"])
        p1 = pts[1]
        kpt = list(p1["r"]) + list(p1["g"]) + list(p1["tau"])
        zm = (w["mask"][:, None] * w["z"]).astype(object)
        v["S1"] = mle(ext(fe(zm.ravel())), kpt)
        v["S2"] = mle(ext(fe(w["N"].ravel())), list(pts[2]["i"]) + list(pts[2]["r"]) + list(pts[2]["g"]))
        v["S3"] = mle(ext(fe(w["M2"].ravel())), list(pts[3]["i"]) + list(pts[3]["r"]) + list(pts[3]["g"]))
        v["S4"] = mle(ext(fe(w["Q2"].ravel())), list(pts[4]["i"]) + list(pts[4]["g"]))
        i1, g1, ta1 = split(fin[1]["r"])
        v["a1"] = mle(CPU.emul(TZ.eq_t(p1["tau"]), ext(fe(w["mask"]))), ta1)
        v["b1"] = CPU.edot_dev(TZ._qq_weights(t, p1, g1, i1)[:, :t.G * t.R * t.DH], ext(fe(s.q_int.ravel())))
        v["c1"] = mle(ext(fe(s.K.reshape(-1))), list(i1) + list(g1) + list(ta1))
        for k, Ew in ((2, w["E"]), (3, w["Ehi"])):
            ik, gk_, tk = split(fin[k]["r"])
            v[f"b{k}"] = mle(ext(fe(Ew.ravel())), list(pts[k]["r"]) + list(gk_) + list(tk))
            v[f"c{k}"] = mle(ext(fe(s.V.reshape(-1))), list(ik) + list(gk_) + list(tk))
        i4, g4, t4 = split(fin[4]["r"])
        v["c4"] = mle(ext(fe(s.V.reshape(-1))), list(i4) + list(g4) + list(t4))
        v["a4"] = mle(ext(fe(w["mask"])), t4)
        return v


def attn_round(slots):
    """ONE commit round running every slot's attention sumchecks (append before hs.rounds())."""
    def fn(tr):
        for sl in slots:
            sl.round(tr)
    return fn


# ============================================================================ census (verifier-side build)
def verifier_cells(cir, rounds_pf):
    """Cells of a verifier-mode circuit including its commit rounds, LogUp helpers and Libra masks, without a proof:
    rounds_pf is a structurally valid all-zero proof dict for the rounds (zero sumcheck messages pass esc_verify).
    Returns (total cells, cells by space prefix)."""
    from .transcript import Transcript
    tr = Transcript(b"census")
    cir.pf = rounds_pf
    for k, fn in enumerate(cir.rounds):
        cir.cur_phase = 10 + k
        try:
            fn(tr)
        finally:
            cir.cur_phase = 1
    Z._finalize_lookups(cir, tr, CPU)
    sizes = sorted(set(cir.zeros) | set(cir.sums))
    degs = {nv: Z._degree(cir, nv) for nv in sizes}
    Z._masks(cir, sizes, degs)
    by = {}
    for v in cir.vecs:
        key = v.name.split(".")[0]
        if key.startswith("tcs:"):
            key = "tcs:" + ".".join(v.name[4:].split(".")[:2])
        by[key] = by.get(key, 0) + v.N
    return int(sum(v.N for v in cir.vecs)), by


def zero_round_pf(slots_tags, n_sc, hs_tag, nH):
    z = [0, 0, 0]
    pf = {"attn": {tg: [{"S": z, "msgs": [[z] * (SC_DEG[k] + 1)] * n_sc} for k in range(4)] for tg in slots_tags},
          "hsel": {hs_tag: {"hs": {"S": z, "msgs": [[z] * (HZ.HS_DEG + 1)] * nH}}}}
    return pf


# ============================================================================ cross-slot batched attention
B_SC_DEG = (4, 3, 4, 4)


def _b_mask_weights(k, n, deg, r):
    """MK layout (sc 2, b 6, k 3): weights of sum_x g_k(x) and g_k(r) on a 2^11 cube."""
    gw = np.zeros((3, 1 << 11), dtype=np.uint64); rw = np.zeros((3, 1 << 11), dtype=np.uint64)
    half = pow(2, n - 1, P)
    for b in range(n):
        x = ONE
        for kk in range(deg + 1):
            pos = (k << 9) | (b << 3) | kk
            gw[0, pos] = (2 if kk == 0 else 1) * half % P
            rw[0, pos], rw[1, pos], rw[2, pos] = x
            x = hmul(x, _e(r[b]))
    return gw, rw


def _b_binds(layout, name, sb, t_ents, l_ents, sp, outer, vec_all, n_act):
    """Build s-batched HSel binds for a port stored on `sp = [('s', sb), (outer, ...), ('i', DB)]`."""
    iv, ov = sp.vars("i"), sp.vars(outer)
    DB = len(iv)
    K = 1 << sb
    v = None if vec_all is None else np.asarray(vec_all, dtype=np.int64).reshape(K, -1)
    out = []
    if name in layout.SPLIT:
        for half, suf in ((0, "_lo"), (1, "_hi")):
            ivars = iv[:DB - 1] + ov
            sel = Ex.atom(SelLt([], 1, fixed=((iv[DB - 1], half),)))
            d = len(ivars)
            vw = None if v is None else v.reshape(K, -1, 2, 1 << (DB - 1))[:, :, half, :].reshape(K, -1)
            pres = [layout._pre(d, name + suf, l_ents[s_]) for s_ in range(n_act)]
            out.append(((sp, None, ivars, sel), d, list(t_ents[:n_act]), pres, vw))
    else:
        ivars = iv + ov
        d = len(ivars)
        pres = [layout._pre(d, name, l_ents[s_]) for s_ in range(n_act)]
        out.append(((sp, None, ivars, None), d, list(t_ents[:n_act]), pres, v))
    return out


class BatchedAttn:
    """Cross-slot batched attention + LSE + producer-lift gadget across `K_C = 1 << sb` slots (`M_C <= K_C` active,
    slots `s >= M_C` repeating slot 0).  Emits ONE set of Phase-1 spaces `[('s', sb), ...]` (~65 vectors total instead
    of `95 * M_C`) and runs 4 layer-factored / GPU-accelerated sumchecks `SC1..SC4` over `(i, g, tau, s)`."""

    def __init__(self, ctx, hs, tier, layout, sb, M_C, t_col, t_ents, l_ents, u_list,
                 wit_fn=None, kv_layer_fn=None, tag="batt", B=CPU):
        t = tier
        self.ctx, self.hs, self.t, self.lay, self.tag, self.B = ctx, hs, t, layout, tag, B
        self.sb, self.K = int(sb), 1 << int(sb)
        self.M_C = int(M_C)
        pv = self.pv = ctx.prove
        assert pv == (wit_fn is not None)
        assert self.K >= 2 and 1 <= self.M_C <= self.K
        K = self.K
        nt, GB, DB, HB = t.nt, t.GB, t.DB, t.HB
        Hn = t.G * t.R
        GD = t.G * t.DH
        HD = Hn * t.DH
        TH = t.T_pad * Hn
        nm = lambda x: f"{tag}.{x}"
        rcI = R.radius_constants(t)["ints"]
        Wc = rcI["W"]
        SHb = t.SH
        lw_bits = [16] * (SHb // 16) + ([SHb % 16] if SHb % 16 else [])
        has_lse = "lse_lo" in layout.ports
        cs_names = ("N", "M2", "oh", "rd", "aoh", "W2", "rW", "SJ", "SQ", "rS")
        t0 = time.time()

        # ---- Pre-allocate compact batched Phase-1 arrays (slot-by-slot streaming: 0 persistent R.Slot objects)
        if pv:
            tau_ps = np.zeros(K, dtype=np.int64)
            t_vals = np.zeros(K, dtype=np.int64)
            l_vals = np.zeros(K, dtype=np.int64)
            qc_all = np.zeros((K, HD), dtype=np.int64)
            oc_all = np.zeros((K, HD), dtype=np.int64)
            pkc_all = np.zeros((K, GD), dtype=np.int64)
            pvc_all = np.zeros((K, GD), dtype=np.int64)
            mask_all = np.zeros((K, t.T_pad), dtype=np.int8)
            idx_all = np.zeros((K, TH), dtype=np.int32)
            E_all = np.zeros((K, TH), dtype=np.int64)
            Ehi_all = np.zeros((K, TH), dtype=np.int32)
            rem_limbs = [np.zeros((K, TH), dtype=np.int32) for _ in lw_bits]
            m_all = np.zeros((K, Hn), dtype=np.int64)
            Z_all = np.zeros((K, Hn), dtype=np.int64)
            Zc_all = np.zeros((K, Hn), dtype=np.int64)
            rz_all = np.zeros((K, Hn), dtype=np.int64)
            cs_all = {k_: np.zeros((K, HD), dtype=object) for k_ in cs_names}
            sp_all = np.zeros((K, HD), dtype=object)
            sm_all = np.zeros((K, HD), dtype=object)
            X_all = np.zeros((K, HD), dtype=object)
            Q2_all = np.zeros((K, GD), dtype=object)
            SQ0_all = np.zeros((K, GD), dtype=object)
            q_int_all = np.zeros((K, t.G, t.R, t.DH), dtype=np.int64)
            if has_lse:
                lse_keys = ("m24", "rm", "ez", "Bz", "mant", "lnv", "rz_ln", "lse_lo", "lse_hi",
                            "ls_s", "ls_ahi", "ls_E", "ls_A", "ls_B", "ls_y", "ls_rem", "ls_o24", "lse_sp", "lse_sm")
                lse_all = {k_: np.zeros((K, Hn), dtype=object) for k_ in lse_keys}
            for s_i in range(K):
                ws = wit_fn(s_i)
                t_att = int(ws.get("t_att", ws["t"]))
                l_i = int(ws["l"])
                tp_i = int(ws.get("tau_p_override", key_draw(u_list[s_i], t_att)))
                tau_ps[s_i], t_vals[s_i], l_vals[s_i] = tp_i, t_att, l_i
                qc_all[s_i] = np.asarray(ws["q_codes"], dtype=np.int64).ravel()
                oc_all[s_i] = np.asarray(ws["o_codes"], dtype=np.int64).ravel()
                pkc_all[s_i] = np.asarray(ws["k_codes_tp"], dtype=np.int64).ravel()
                pvc_all[s_i] = np.asarray(ws["v_codes_tp"], dtype=np.int64).ravel()
                qi = lift_np(qc_all[s_i], t.FQ, q_scale(t)).reshape(Hn, t.DH)
                oi = lift_np(oc_all[s_i], t.FO).reshape(Hn, t.DH)
                q_int_all[s_i] = qi.reshape(t.G, t.R, t.DH)
                K_l, V_l = kv_layer_fn(l_i)
                sl = R.Slot(t, l_i, t_att, qi, oi, K_l, V_l, tp_i)
                R.witness(sl, check=not Z.FORCE)
                if t_att != int(ws["t"]):
                    ctx.violation("attention t differs from the caller's hidden t")
                try:
                    Kl_tp, Vl_tp = lifted_ports(pkc_all[s_i], pvc_all[s_i], t)
                    if not (np.array_equal(Kl_tp, K_l[tp_i].ravel()) and np.array_equal(Vl_tp, V_l[tp_i].ravel())):
                        ctx.violation("producer: lifted K/V row differs from Lift(codes at (tau_p, l))")
                except ValueError:
                    pass
                w = sl.w
                mask_all[s_i] = w["mask"]
                idx_all[s_i] = w["idx"].ravel()
                E_all[s_i] = w["E"].ravel()
                Ehi_all[s_i] = w["Ehi"].ravel()
                rem_flat = w["rem"].ravel()
                for li, wd in enumerate(lw_bits):
                    rem_limbs[li][s_i] = (rem_flat >> (16 * li)) & ((1 << wd) - 1)
                m_all[s_i], Z_all[s_i], Zc_all[s_i], rz_all[s_i] = w["m"], w["Z"], w["Zc"], w["rz"]
                for k_ in cs_names:
                    cs_all[k_][s_i] = w[k_].ravel()
                sp_all[s_i], sm_all[s_i], X_all[s_i] = w["sp"].ravel(), w["sm"].ravel(), w["X"].ravel()
                if any(int(x) < 0 for x in sp_all[s_i]) or any(int(x) < 0 for x in sm_all[s_i]):
                    ctx.violation("attention output outside the certified radius")
                Q2_all[s_i], SQ0_all[s_i] = w["Q2"].ravel(), w["SQ0"].ravel()
                if has_lse:
                    lw_ = R.lse_witness(t, w, ws["lse_u32"])
                    if lw_["lse_bad"]:
                        ctx.violation("attention LSE outside the certified radius")
                    for k_ in lse_keys:
                        lse_all[k_][s_i] = lw_[k_]
                del sl, w
            self.tau_ps = tau_ps
            self._wit_cache = dict(l_vals=l_vals, mask=mask_all, q_int=q_int_all, E=E_all, Ehi=Ehi_all,
                                   m=m_all, idx=idx_all, rem_limbs=rem_limbs, N=cs_all["N"], M2=cs_all["M2"],
                                   Q2=Q2_all, kv_layer_fn=kv_layer_fn)
        else:
            self.tau_ps = np.zeros(K, dtype=np.int64)
            self._wit_cache = None

        # ---- S_sc: per-slot producer draw tau_p[s] = floor(u_s (t_s + 1) / 2^UB) and its bits
        S_sc = ctx.space(nm("S"), [("s", sb)])
        t_on_S = t_col.on(S_sc, {"s": "s"})
        u_pub = S_sc.pub("u", np.asarray(u_list, dtype=np.int64))
        tp_col = S_sc.col("tp", tau_ps if pv else None)
        tpb = [S_sc.col(f"tpb{k}", ((tau_ps >> k) & 1) if pv else None) for k in range(nt)]
        for b_ in tpb:
            S_sc.zero(b_.ex * (b_.ex - Ex.const(1)), tag=nm("tp.bool"))
        S_sc.zero(tp_col.ex - sum((b_.ex * Ex.const(1 << k) for k, b_ in enumerate(tpb)), Ex.const(0)), tag=nm("tp.val"))
        kd_v = (np.asarray(u_list, dtype=object) * (t_vals.astype(object) + 1) - (tau_ps.astype(object) << UB)) if pv else None
        TZ.rc(S_sc, u_pub.ex * (t_on_S + Ex.const(1)) - tp_col.ex * Ex.const(1 << UB), kd_v, UB, nm("kdraw"))
        self.tp_col = tp_col
        self.S_sc = S_sc
        self.tau_ents = [[("hid", (S_sc, tpb[k].ex, s_i), int((self.tau_ps[s_i] >> k) & 1) if pv else 0)
                          for k in range(nt)] for s_i in range(self.M_C)]
        self.l_ents = list(l_ents[:self.M_C])

        # ---- Batched producer lift check at (tau_p[s], l_s)
        PS = ctx.space(nm("P"), [("s", sb), ("g", GB), ("i", DB)])
        kc, kL = lift_gadget(ctx, PS, pkc_all.ravel() if pv else None, t.FK, None, nm("pk"))
        vc, vL = lift_gadget(ctx, PS, pvc_all.ravel() if pv else None, t.FV, None, nm("pv"))
        for port_nm, col_x, wit_x in (("k", kc, pkc_all if pv else None),
                                      ("v", vc, pvc_all if pv else None),
                                      ("K", kL, kL.v if pv else None),
                                      ("V", vL, vL.v if pv else None)):
            for (sp_, _, ivars_, sel_), d_, rows_, pres_, vw_ in _b_binds(
                    layout, port_nm, sb, self.tau_ents, self.l_ents, PS, "g", wit_x, self.M_C):
                hs.bind_batched_vector((sp_, col_x.ex, ivars_, sel_), sb, d_, rows_, pres_, vec_wits=vw_, n_active=self.M_C)

        # ---- Batched q / o lifts & binds at (t_s, l_s)
        CS = ctx.space(nm("C"), [("s", sb), ("h", HB), ("i", DB)])
        qcode, qL = lift_gadget(ctx, CS, qc_all.ravel() if pv else None, t.FQ, q_scale(t), nm("q"))
        ocode, oL = lift_gadget(ctx, CS, oc_all.ravel() if pv else None, t.FO, None, nm("o"))
        for port_nm, col_x, wit_x in (("q", qcode, qc_all if pv else None),
                                      ("o", ocode, oc_all if pv else None)):
            for (sp_, _, ivars_, sel_), d_, rows_, pres_, vw_ in _b_binds(
                    layout, port_nm, sb, t_ents[:self.M_C], self.l_ents, CS, "h", wit_x, self.M_C):
                hs.bind_batched_vector((sp_, col_x.ex, ivars_, sel_), sb, d_, rows_, pres_, vec_wits=vw_, n_active=self.M_C)

        # ---- Batched attention Phase-1 columns & constraints
        TS = ctx.space(nm("T"), [("s", sb), ("tau", nt)])
        mask = TS.col("mask", mask_all.ravel() if pv else None)
        KS = ctx.space(nm("K"), [("s", sb), ("tau", nt), ("h", HB)])
        ET, EHT = R.expt(t.EXP_BITS)
        etab = f"az.exp{t.EXP_BITS}"
        G.table(ctx, etab, [np.arange(1 << t.EXP_BITS, dtype=np.int64), ET, EHT])
        idx = KS.col("idx", idx_all.ravel() if pv else None)
        E = KS.col("E", E_all.ravel() if pv else None)
        Ehi = KS.col("Ehi", Ehi_all.ravel() if pv else None)
        ctx.cir.lookup(etab, KS.nv, [idx.ex, E.ex, Ehi.ex], sel=KS.sel(),
                       vals=[idx_all.ravel(), E_all.ravel(), Ehi_all.ravel()] if pv else None,
                       sel_vals=np.ones(KS.n, dtype=np.int64) if pv else None)
        rl = TZ.limbs(KS, "rem", [rm.ravel() for rm in rem_limbs] if pv else None, lw_bits, ctx)
        rem_ex = sum((c.ex * Ex.const(1 << (16 * k)) for k, c in enumerate(rl)), Ex.const(0))
        D_ex = idx.ex * Ex.const(1 << SHb) + rem_ex - Ex.const(1 << (SHb - 1))
        HSp = ctx.space(nm("H"), [("s", sb), ("h", HB)])
        m_c = TZ.col_obj(HSp, "m", m_all.ravel() if pv else None)
        Z_c = TZ.col_obj(HSp, "Z", Z_all.ravel() if pv else None)
        Zc_c = TZ.col_obj(HSp, "Zc", Zc_all.ravel() if pv else None)
        rz_c = TZ.col_obj(HSp, "rz", rz_all.ravel() if pv else None)
        GS = ctx.space(nm("G"), [("s", sb), ("g", GB), ("i", DB)])
        cs = {k_: TZ.col_obj(CS, k_, cs_all[k_].ravel() if pv else None) for k_ in cs_names}
        Q2_c = TZ.col_obj(GS, "Q2", Q2_all.ravel() if pv else None)
        SQ0_c = TZ.col_obj(GS, "SQ0", SQ0_all.ravel() if pv else None)
        MK = ctx.space(nm("M"), [("sc", 2), ("b", 6), ("k", 3)])
        coefs = {}
        mk_v = None
        self.n_sc = DB + GB + nt + sb
        assert self.n_sc <= 64, f"BatchedAttn sumcheck rounds {self.n_sc} exceed 6-bit mask axis"
        if pv:
            mk_v = np.zeros((3, 1 << 11), dtype=np.uint64)
            for k in range(4):
                cf = []
                for b in range(self.n_sc):
                    row = []
                    for kk in range(B_SC_DEG[k] + 1):
                        rf = rand_field(3)
                        v = (int(rf[0]), int(rf[1]), int(rf[2]))
                        row.append(v)
                        pos = (k << 9) | (b << 3) | kk
                        mk_v[0, pos], mk_v[1, pos], mk_v[2, pos] = v
                    cf.append(row)
                coefs[k] = cf
        mkc = MZ.ecol(MK, "g", mk_v)

        # Causal mask per slot s: boolean, mask(s, 0) = 1, monotone for tau < T_pad - 1, sum_tau mask(s, tau) = t_s + 1
        TS.zero(mask.ex * (mask.ex - Ex.const(1)), tag=nm("mask.bool"))
        TS.zero(mask.ex - Ex.const(1), sel=TS.ind(tau=0), tag=nm("mask.first"))
        dlt = mask.ex - mask.shifted(1)
        TS.zero(dlt * (dlt - Ex.const(1)), sel=TS.lt("tau", t.T_pad - 1), tag=nm("mask.mono"))
        TS.sumzero(mask.ex - (t_col.on(TS, {"s": "s"}) + Ex.const(1)) * TS.ind(tau=0),
                   eq_axes=("s",), tag=nm("mask.sum"))
        mask_ks = mask.on(KS, {"s": "s", "tau": "tau"})
        KS.zero((Ex.const(1) - mask_ks) * E.ex, tag=nm("E.masked"))
        Z_ks = Z_c.on(KS, {"s": "s", "h": "h"})
        KS.sumzero(E.ex - Z_ks * KS.ind(tau=0), eq_axes=("s", "h"), tag=nm("Z"))
        TZ.rc(HSp, Z_c.ex - Ex.const(1 << R.EB), (Z_all.ravel() - (1 << R.EB)) if pv else None, 38, "Z.lo")
        HSp.zero(Z_c.ex - Zc_c.ex * Ex.const(1 << R.CB) - rz_c.ex, tag=nm("Zc"))
        TZ.rc(HSp, rz_c.ex, rz_all.ravel() if pv else None, R.CB, "rz")
        TZ.rc(HSp, Zc_c.ex, Zc_all.ravel() if pv else None, 27, "Zc")

        Zcs = Z_c.on(CS, {"s": "s", "h": "h"}); Zccs = Zc_c.on(CS, {"s": "s", "h": "h"})
        gmap = [("h", t.RB + k) for k in range(GB)]
        Q2cs = Q2_c.on(CS, {"s": "s", "g": gmap, "i": "i"})
        SQ0cs = SQ0_c.on(CS, {"s": "s", "g": gmap, "i": "i"})
        oi_ex = oL.ex
        c_ = {k: v.ex for k, v in cs.items()}
        CS.zero(c_["N"] * Ex.const(64) - c_["oh"] * Zcs - c_["rd"], tag=nm("oh"))
        if pv:
            Zr = np.repeat(Z_all.astype(object), t.DH, axis=1).ravel()
            Zcr = np.repeat(Zc_all.astype(object), t.DH, axis=1).ravel()
            W_ = {k_: cs_all[k_].ravel() for k_ in ("oh", "rd", "aoh", "W2", "rW", "SJ", "SQ", "rS")}
            W_["X"], W_["sp"], W_["sm"] = X_all.ravel(), sp_all.ravel(), sm_all.ravel()
        TZ.rc(CS, c_["rd"], W_["rd"] if pv else None, 38, "rd.lo")
        TZ.rc(CS, Zcs - Ex.const(1) - c_["rd"], (Zr - 1 - W_["rd"]) if pv else None, 38, "rd.hi")
        CS.zero((c_["aoh"] - c_["oh"]) * (c_["aoh"] + c_["oh"]), tag=nm("aoh"))
        TZ.rc(CS, c_["aoh"], W_["aoh"] if pv else None, 26, "aoh")
        TZ.rc(CS, c_["M2"], cs_all["M2"].ravel() if pv else None, 60, "M2")
        CS.zero(c_["W2"] * Zccs - c_["M2"] - c_["rW"], tag=nm("W2"))
        TZ.rc(CS, c_["rW"], W_["rW"] if pv else None, 27, "rW.lo")
        TZ.rc(CS, Zccs - Ex.const(1) - c_["rW"], (Zcr - 1 - W_["rW"]) if pv else None, 27, "rW.hi")
        TZ.rc(CS, c_["W2"], W_["W2"] if pv else None, 36, "W2")
        X_ex = (Ex.const(rcI["KH"] << 12) * c_["W2"] - Ex.const(rcI["KL"]) * (c_["aoh"] * c_["aoh"] - Ex.const(2) * c_["aoh"])
                + Ex.const(1 << 12))
        TZ.rc(CS, c_["SJ"] * c_["SJ"] * Ex.const(1 << 12) - X_ex,
              (W_["SJ"] * W_["SJ"] * 4096 - W_["X"]) if pv else None, 40, "SJ.up")
        TZ.rc(CS, X_ex - (c_["SJ"] - Ex.const(1)) * (c_["SJ"] - Ex.const(1)) * Ex.const(1 << 12) - Ex.const(1),
              (W_["X"] - (W_["SJ"] - 1) * (W_["SJ"] - 1) * 4096 - 1) if pv else None, 40, "SJ.tight")
        TZ.rc(CS, c_["SJ"], W_["SJ"] if pv else None, 26, "SJ")
        TZ.rc(GS, Q2_c.ex, Q2_all.ravel() if pv else None, 62 - nt, "Q2")
        GQv = sq0 = None
        if pv:
            t_p1 = (t_vals + 1)[:, None].astype(object)
            GQv = (t_p1 * Q2_all + 1).ravel()
            sq0 = SQ0_all.ravel()
        gq_ex = Q2_c.ex * (t_col.on(GS, {"s": "s"}) + Ex.const(1)) + Ex.const(1)
        TZ.rc(GS, SQ0_c.ex * SQ0_c.ex - gq_ex, (sq0 * sq0 - GQv) if pv else None, 34, "SQ0.up")
        TZ.rc(GS, gq_ex - (SQ0_c.ex - Ex.const(1)) * (SQ0_c.ex - Ex.const(1)) - Ex.const(1),
              (GQv - (sq0 - 1) * (sq0 - 1) - 1) if pv else None, 34, "SQ0.tight")
        TZ.rc(GS, SQ0_c.ex, sq0 if pv else None, 31, "SQ0")
        CS.zero(c_["SQ"] * Zcs - SQ0cs * Ex.const(32) - c_["rS"], tag=nm("SQ"))
        TZ.rc(CS, c_["rS"], W_["rS"] if pv else None, 38, "rS.lo")
        TZ.rc(CS, Zcs - Ex.const(1) - c_["rS"], (Zr - 1 - W_["rS"]) if pv else None, 38, "rS.hi")
        TZ.rc(CS, c_["SQ"], W_["SQ"] if pv else None, 20, "SQ")
        rhs = (Ex.const(rcI["A"]) * c_["SJ"] + Ex.const(rcI["B"]) * c_["aoh"] + Ex.const(rcI["Cq"]) * c_["SQ"]
               + Ex.const(rcI["C0"]))
        dd = (oi_ex - c_["oh"]) * Ex.const(1 << Wc)
        TZ.rc(CS, rhs - dd, W_["sp"] if pv else None, 48, "tol.p")
        TZ.rc(CS, rhs + dd, W_["sm"] if pv else None, 48, "tol.m")

        # ---- Batched LSE observation check (M1)
        if has_lse:
            ltab = _lse_tables()
            G.table(ctx, "cz.f32hi", CZ._f32hi_table())
            G.table(ctx, "az.lne", ltab["lne"])
            G.table(ctx, "az.lnt", ltab["lnt"])
            G.table(ctx, "az.lsep2", ltab["p2"])
            SM = t.FQ + t.FK - R.FL_LSE
            lf = (lambda k_: lse_all[k_].ravel().astype(np.int64)) if pv else (lambda k_: None)
            lo_obj = (lambda k_: lse_all[k_].ravel()) if pv else (lambda k_: None)
            m24_c = TZ.col_obj(HSp, "m24", lo_obj("m24"))
            rm_c = HSp.col("rm", lf("rm"))
            HSp.zero(m_c.ex - m24_c.ex * Ex.const(1 << SM) - rm_c.ex, tag=nm("m24.def"))
            TZ.rc(HSp, rm_c.ex, lo_obj("rm"), SM, "m24.rem")
            TZ.rc(HSp, m24_c.ex + Ex.const(1 << 38), (lo_obj("m24") + (1 << 38)) if pv else None, 39, "m24.srng")
            ez_c = HSp.col("ez", lf("ez"))
            Bz_c = HSp.col("Bz", lf("Bz"))
            mant_c = HSp.col("mant", lf("mant"))
            lnv_c = HSp.col("lnv", lf("lnv"))
            G.wire_lookup(ctx, "az.lne", HSp, [ez_c.ex, Bz_c.ex], [lf("ez"), lf("Bz")] if pv else None)
            G.wire_lookup(ctx, "az.lnt", HSp, [mant_c.ex, lnv_c.ex], [lf("mant"), lf("lnv")] if pv else None)
            remz_ex = Z_c.ex - mant_c.ex * Bz_c.ex
            TZ.rc(HSp, remz_ex, lo_obj("rz_ln"), 33, "lnz.rlo")
            TZ.rc(HSp, Bz_c.ex - Ex.const(1) - remz_ex, (lo_obj("Bz") - 1 - lo_obj("rz_ln")) if pv else None, 33, "lnz.rhi")
            L24_ex = m24_c.ex + (ez_c.ex - Ex.const(R.EB)) * Ex.const(ltab["LN2"]) + lnv_c.ex
            lse_lo = HSp.col("lse_lo", lf("lse_lo"))
            lse_hi = HSp.col("lse_hi", lf("lse_hi"))
            ls_s = HSp.col("ls_s", lf("ls_s"))
            ls_ahi = HSp.col("ls_ahi", lf("ls_ahi"))
            ls_E = TZ.col_obj(HSp, "ls_E", lo_obj("ls_E"))
            ls_A = HSp.col("ls_A", lf("ls_A"))
            ls_B = HSp.col("ls_B", lf("ls_B"))
            ls_y = HSp.col("ls_y", lf("ls_y"))
            ls_rem = HSp.col("ls_rem", lf("ls_rem"))
            ls_o24 = TZ.col_obj(HSp, "ls_o24", lo_obj("ls_o24"))
            TZ.rc(HSp, lse_lo.ex, lo_obj("lse_lo"), 16, "lse.lo")
            G.wire_lookup(ctx, "cz.f32hi", HSp, [lse_hi.ex, ls_s.ex, ls_ahi.ex, ls_E.ex],
                          [lf("lse_hi"), lf("ls_s"), lf("ls_ahi"), lo_obj("ls_E").astype(np.int64)] if pv else None)
            G.wire_lookup(ctx, "az.lsep2", HSp, [ls_E.ex, ls_A.ex, ls_B.ex],
                          [lo_obj("ls_E").astype(np.int64), lf("ls_A"), lf("ls_B")] if pv else None)
            lsa_ex = ls_ahi.ex * Ex.const(1 << 16) + lse_lo.ex
            HSp.zero(lsa_ex * ls_A.ex - ls_y.ex * ls_B.ex - ls_rem.ex, tag=nm("lse.shr"))
            TZ.rc(HSp, ls_rem.ex, lo_obj("ls_rem"), 25, "lse.rlo")
            TZ.rc(HSp, ls_B.ex - Ex.const(1) - ls_rem.ex, (lo_obj("ls_B") - 1 - lo_obj("ls_rem")) if pv else None, 25, "lse.rhi")
            TZ.rc(HSp, ls_y.ex, lo_obj("ls_y"), 38, "lse.y")
            HSp.zero(ls_o24.ex - ls_y.ex + Ex.const(2) * ls_s.ex * ls_y.ex, tag=nm("lse.sgn"))
            RL24 = rcI["RL24"]
            TZ.rc(HSp, Ex.const(RL24 - 1) - (ls_o24.ex - L24_ex), lo_obj("lse_sp"), 41, "lse.p")
            TZ.rc(HSp, Ex.const(RL24 - 1) + (ls_o24.ex - L24_ex), lo_obj("lse_sm"), 41, "lse.m")
            pres_lo = [layout._pre(HB, "lse_lo", self.l_ents[s_]) for s_ in range(self.M_C)]
            pres_hi = [layout._pre(HB, "lse_hi", self.l_ents[s_]) for s_ in range(self.M_C)]
            hs.bind_batched_vector(lse_lo, sb, HB, list(t_ents[:self.M_C]), pres_lo,
                                   vec_wits=lse_all["lse_lo"].astype(np.int64) if pv else None, n_active=self.M_C)
            hs.bind_batched_vector(lse_hi, sb, HB, list(t_ents[:self.M_C]), pres_hi,
                                   vec_wits=lse_all["lse_hi"].astype(np.int64) if pv else None, n_active=self.M_C)

        SX = ctx.space(nm("X"), [("x", SCB)])
        self.S = dict(S_sc=S_sc, TS=TS, KS=KS, HSp=HSp, CS=CS, GS=GS, MK=MK, SX=SX,
                      mask=mask, E=E, Ehi=Ehi, D_ex=D_ex, m_c=m_c, cs=cs, Q2=Q2_c, qL=qL,
                      mkc=mkc, coefs=coefs, mask_ks=mask_ks)
        self.info = {"phase1_build_s": round(time.time() - t0, 3)}

    def _prove_one_sc(self, k, pt, coef, rho, tr):
        """Two-stage exact sumcheck prover for SC k over `(i: DB, g: GB, tau: nt, s: sb)`:
        Stage 1 (`DB` rounds binding `i`): groups slots by layer `l_s` so each layer's KV cache is folded ONCE.
        Stage 2 (`GB + nt + sb` rounds binding `g, tau, s`): runs on the `DH`-free `(K, T_pad, G)` tensor."""
        t, B, wc = self.t, self.B, self._wit_cache
        K, sb = self.K, self.sb
        T, G_, D = t.T_pad, t.G, t.DH
        DB, GB, nt = t.DB, t.GB, t.nt
        n = self.n_sc
        deg = B_SC_DEG[k - 1]
        gpu = MC._is_gpu(B)
        dev = (lambda a: B.asarray(np.ascontiguousarray(a))) if gpu else (lambda a: np.ascontiguousarray(a))
        l_vals = wc["l_vals"]
        uniq_l = sorted(set(int(x) for x in l_vals))
        kv_fn = wc["kv_layer_fn"]
        es = TZ.eq_t(pt["s"])                                              # (3, K)

        # Prepare per-slot non-KV weights and per-layer Stage-1 factor lists
        layer_terms = {}
        if k == 1:
            etau = TZ.eq_t(pt["tau"])                                      # (3, T)
            er = TZ.eq_t(pt["r"]); eg = TZ.eq_t(pt["g"])
            a_st = np.zeros((3, K, T), dtype=np.uint64)
            qq_s = np.zeros((3, K, G_ * D), dtype=np.uint64)
            for s_i in range(K):
                a_st[:, s_i, :] = CPU.emul(etau, TZ.ext(TZ.fe(wc["mask"][s_i])))
                qf = TZ.fe(wc["q_int"][s_i])
                for g in range(G_):
                    acc = np.zeros((3, D), dtype=np.uint64)
                    for rr in range(t.R):
                        acc = CPU.eadd(acc, CPU.escale(TZ.ext(qf[g, rr]), _e(er[:, rr])))
                    qq_s[:, s_i, g * D:(g + 1) * D] = CPU.escale(acc, _e(eg[:, g]))
            for l in uniq_l:
                idxs = [s_i for s_i in range(K) if l_vals[s_i] == l]
                Wl = B.zeros_e(T * G_ * D)
                for s_i in idxs:
                    ea_s = CPU.escale(a_st[:, s_i, :], _e(es[:, s_i]))     # (3, T)
                    # Outer product on B: (3, T, 1) * (3, 1, G*D) -> (3, T * G * D)
                    da = dev(np.repeat(ea_s, G_ * D, axis=1))
                    dq = dev(np.tile(qq_s[:, s_i, :], (1, T)))
                    Wl = B.eadd(Wl, B.emul(da, dq))
                    del da, dq
                Kl_raw, _ = kv_fn(l)
                Kl = dev(TZ.ext(TZ.fe(Kl_raw.reshape(-1))))
                layer_terms[l] = [Wl, Kl]
        elif k in (2, 3):
            Ew_all = wc["E"] if k == 2 else wc["Ehi"]                      # (K, T * G * R)
            er = TZ.eq_t(pt["r"]); eg = TZ.eq_t(pt["g"]); ei = TZ.eq_t(pt["i"])
            egi = CPU.emul(np.repeat(eg, D, axis=1), np.tile(ei, (1, G_))) # (3, G * D)
            ep_s = np.zeros((3, K, T * G_), dtype=np.uint64)
            for s_i in range(K):
                Ef = TZ.fe(Ew_all[s_i]).reshape(T * G_, t.R)
                acc = np.zeros((3, T * G_), dtype=np.uint64)
                for rr in range(t.R):
                    acc = CPU.eadd(acc, CPU.escale(TZ.ext(Ef[:, rr]), _e(er[:, rr])))
                ep_s[:, s_i, :] = acc
            for l in uniq_l:
                idxs = [s_i for s_i in range(K) if l_vals[s_i] == l]
                w_tg = np.zeros((3, T * G_), dtype=np.uint64)
                for s_i in idxs:
                    w_tg = CPU.eadd(w_tg, CPU.escale(ep_s[:, s_i, :], _e(es[:, s_i])))
                ta_l = dev(CPU.emul(np.tile(egi, (1, T)), np.repeat(w_tg, D, axis=1)))
                _, Vl_raw = kv_fn(l)
                Vl = dev(TZ.ext(TZ.fe(Vl_raw.reshape(-1))))
                layer_terms[l] = [ta_l, Vl] + ([Vl] if k == 3 else [])
        else:
            eg = TZ.eq_t(pt["g"]); ei = TZ.eq_t(pt["i"])
            egi = CPU.emul(np.repeat(eg, D, axis=1), np.tile(ei, (1, G_)))
            for l in uniq_l:
                idxs = [s_i for s_i in range(K) if l_vals[s_i] == l]
                w_t = np.zeros((3, T), dtype=np.uint64)
                for s_i in idxs:
                    w_t = CPU.eadd(w_t, CPU.escale(TZ.ext(TZ.fe(wc["mask"][s_i])), _e(es[:, s_i])))
                ta_l = dev(CPU.emul(np.tile(egi, (1, T)), np.repeat(w_t, G_ * D, axis=1)))
                _, Vl_raw = kv_fn(l)
                Vl = dev(TZ.ext(TZ.fe(Vl_raw.reshape(-1))))
                layer_terms[l] = [ta_l, Vl, Vl]

        # True sum S across all layers
        S = ZERO
        for fs in layer_terms.values():
            prod = fs[0]
            for f in fs[1:]:
                prod = B.emul(prod, f)
            S = hadd(S, B.esum(prod))
        Sp = hadd(S, hmul(rho, TZ.mask_sum(coef, n)))
        tr.absorb(TZ.canon([list(Sp)]))
        msgs, r = [], []
        prefix = ZERO

        # Stage 1: DB rounds binding i_0 .. i_{DB-1}
        for i in range(DB):
            vals = []
            for X in range(deg + 1):
                acc = ZERO
                for fs in layer_terms.values():
                    prod = None
                    for f in fs:
                        lo, hi = f[:, 0::2], f[:, 1::2]
                        v = lo if X == 0 else (hi if X == 1 else B.eadd(lo, B.escale(B.esub(hi, lo), (X, 0, 0))))
                        prod = v if prod is None else B.emul(prod, v)
                    acc = hadd(acc, B.esum(prod))
                vals.append(hadd(acc, hmul(rho, TZ._mask_part(coef, i, n, X, prefix))))
            msgs.append([list(v) for v in vals]); tr.absorb(TZ.canon(msgs[-1]))
            ri = tr.ext(); r.append(ri)
            for l, fs in list(layer_terms.items()):
                if len(fs) == 3 and fs[1] is fs[2]:
                    f0 = B.fold(fs[0], ri); f1 = B.fold(fs[1], ri)
                    layer_terms[l] = [f0, f1, f1]
                else:
                    layer_terms[l] = [B.fold(f, ri) for f in fs]
            prefix = hadd(prefix, TZ._gsum(coef[i], ri))

        # Extract folded KV_i_star[l] (shape (3, T * G)) per layer & scatter to (3, K * T * G)
        i_star = r[:DB]
        kv_l_host = {l: (B.tonumpy(fs[1]) if gpu else fs[1]) for l, fs in layer_terms.items()}
        layer_terms.clear()
        kv_slot = np.zeros((3, K, T * G_), dtype=np.uint64)
        for s_i in range(K):
            kv_slot[:, s_i, :] = kv_l_host[int(l_vals[s_i])]
        kv_dev = dev(kv_slot.reshape(3, K * T * G_))
        es_dev = dev(np.repeat(es, T * G_, axis=1))
        if k == 1:
            ei_star = TZ.eq_t(i_star)
            qq_istar = np.zeros((3, K, G_), dtype=np.uint64)
            for s_i in range(K):
                for g in range(G_):
                    c_sg = CPU.edot_dev(ei_star, qq_s[:, s_i, g * D:(g + 1) * D])
                    qq_istar[0, s_i, g], qq_istar[1, s_i, g], qq_istar[2, s_i, g] = c_sg
            f_a = dev(np.repeat(a_st.reshape(3, K * T), G_, axis=1))
            f_q = dev(np.tile(qq_istar, (1, 1, T)).reshape(3, K * T * G_))
            st2 = [es_dev, f_a, f_q, kv_dev]
        elif k in (2, 3):
            ei_val = TZ.eqv(pt["i"], i_star)
            f_g = dev(np.tile(TZ.eq_t(pt["g"]), (1, K * T)))               # (3, K * T * G), deg 1 in g
            f_e = dev(CPU.escale(ep_s.reshape(3, K * T * G_), ei_val))     # (3, K * T * G), deg 1 in (g, tau, s)
            st2 = [es_dev, f_g, f_e, kv_dev] + ([kv_dev] if k == 3 else [])
        else:
            ei_val = TZ.eqv(pt["i"], i_star)
            eg_rep = np.tile(TZ.eq_t(pt["g"]), (1, K * T))
            m_flat = np.repeat(TZ.ext(TZ.fe(wc["mask"].reshape(K * T))), G_, axis=1)
            f_m = dev(CPU.emul(eg_rep, CPU.escale(m_flat, ei_val)))
            st2 = [es_dev, f_m, kv_dev, kv_dev]

        # Stage 2: GB + nt + sb rounds binding g, tau, s
        for i in range(DB, n):
            vals = []
            for X in range(deg + 1):
                prod = None
                for f in st2:
                    lo, hi = f[:, 0::2], f[:, 1::2]
                    v = lo if X == 0 else (hi if X == 1 else B.eadd(lo, B.escale(B.esub(hi, lo), (X, 0, 0))))
                    prod = v if prod is None else B.emul(prod, v)
                vals.append(hadd(B.esum(prod), hmul(rho, TZ._mask_part(coef, i, n, X, prefix))))
            msgs.append([list(v) for v in vals]); tr.absorb(TZ.canon(msgs[-1]))
            ri = tr.ext(); r.append(ri)
            if len(st2) >= 2 and st2[-1] is st2[-2]:
                fk = B.fold(st2[-1], ri)
                st2 = [B.fold(f, ri) for f in st2[:-2]] + [fk, fk]
            else:
                st2 = [B.fold(f, ri) for f in st2]
            prefix = hadd(prefix, TZ._gsum(coef[i], ri))

        # Also return per-slot c_k[s] = KV_i_star[l_s]~(g*, tau*) for s < K.
        gk, tk = r[DB:DB + GB], r[DB + GB:DB + GB + nt]
        egt = TZ.eq_t(list(gk) + list(tk))
        ck_l = {l: CPU.edot_dev(egt, kv_l_host[l]) for l in uniq_l}
        ck_per_s = [ck_l[int(l_vals[s_i])] for s_i in range(K)]
        return Sp, msgs, r, S, ck_per_s

    def round(self, tr):
        t0 = time.time()
        t, hs, S, pv = self.t, self.hs, self.S, self.pv
        ctx, tag = self.ctx, self.tag
        sb, K = self.sb, self.K
        nt, GB, DB = t.nt, t.GB, t.DB
        nm = lambda x: f"{tag}.{x}"
        pts = {1: dict(s=tr.exts(sb), tau=tr.exts(nt), g=tr.exts(GB), r=tr.exts(t.RB)),
               2: dict(s=tr.exts(sb), g=tr.exts(GB), r=tr.exts(t.RB), i=tr.exts(DB)),
               3: dict(s=tr.exts(sb), g=tr.exts(GB), r=tr.exts(t.RB), i=tr.exts(DB)),
               4: dict(s=tr.exts(sb), g=tr.exts(GB), i=tr.exts(DB))}
        rhos = tr.exts(4)
        pf = ctx.cir.pf
        ent = pf.setdefault("attn", {}).setdefault(tag, []) if pv else pf.get("attn", {}).get(tag)
        if not pv and (not isinstance(ent, list) or len(ent) != 4):
            raise Z.ProofError(f"{tag}: missing batched attention sumchecks")
        fin = {}
        ck_cols_v = {}
        for k in range(1, 5):
            deg = B_SC_DEG[k - 1]
            if pv:
                Sp, msgs, r, _, ck_s = self._prove_one_sc(k, pts[k], S["coefs"][k - 1], rhos[k - 1], tr)
                ent.append({"S": list(Sp), "msgs": msgs})
                vfin = lagrange_eval([_e(v) for v in msgs[-1]], r[-1])
                ck_cols_v[k] = ck_s
            else:
                if not isinstance(ent[k - 1], dict):
                    raise Z.ProofError(f"{tag}: malformed batched attention sumcheck")
                Sp = _e(ent[k - 1]["S"])
                r, vfin = TZ.esc_verify(Sp, ent[k - 1]["msgs"], self.n_sc, deg, tr, f"{tag}.SC{k}")
            fin[k] = dict(Sp=_e(Sp), r=r, v=vfin)

        def split_b(r):
            return r[:DB], r[DB:DB + GB], r[DB + GB:DB + GB + nt], r[DB + GB + nt:]

        vals = self._round_values(pts, fin, split_b, ck_cols_v) if pv else {}
        self._wit_cache = None                # release all prover-side attention caches right away.
        names = {}
        for k_ in ("S1", "S2", "S3", "S4", "G1", "G2", "G3", "G4", "gr1", "gr2", "gr3", "gr4",
                   "a1", "b1", "c1", "b2", "c2", "b3", "c3", "a4", "c4"):
            names[k_] = hs.scalar(nm(k_), vals.get(k_))
        on = lambda k_, sp: hs.SC.on(names[k_], sp)
        SCX = S["SX"]
        X_ = lambda k_: on(k_, SCX)
        S_sc, TS, KS, CS, GS, MK = S["S_sc"], S["TS"], S["KS"], S["CS"], S["GS"], S["MK"]
        mkc, cs = S["mkc"], S["cs"]
        for k in range(1, 5):
            SCX.zero(X_(f"S{k}") + Ex.const(rhos[k - 1]) * X_(f"G{k}") - Ex.const(fin[k]["Sp"]), tag=nm(f"S{k}'"))
        mv = MK.vars("k") + MK.vars("b") + MK.vars("sc")
        for k in range(1, 5):
            gw, rw = _b_mask_weights(k - 1, self.n_sc, B_SC_DEG[k - 1], fin[k]["r"])
            MK.sumzero(Ex.atom(MZ.EVec(nm(f"Gw{k}"), gw, mv)) * mkc.ex - on(f"G{k}", MK) * HZ.ind0(MK), tag=nm(f"G{k}"))
            MK.sumzero(Ex.atom(MZ.EVec(nm(f"rw{k}"), rw, mv)) * mkc.ex - on(f"gr{k}", MK) * HZ.ind0(MK), tag=nm(f"gr{k}"))

        kv_all = KS.vars("h") + KS.vars("tau") + KS.vars("s")
        cv_all = CS.vars("i") + CS.vars("h") + CS.vars("s")
        gv_all = GS.vars("i") + GS.vars("g") + GS.vars("s")
        tv_all = TS.vars("tau") + TS.vars("s")
        sv_s = S_sc.vars("s")

        # SC1
        p1 = list(pts[1]["r"]) + list(pts[1]["g"]) + list(pts[1]["tau"]) + list(pts[1]["s"])
        m_ks = S["m_c"].on(KS, {"s": "s", "h": "h"})
        KS.sumzero(Ex.atom(EqP(kv_all, p1)) * S["mask_ks"] * (m_ks - S["D_ex"]) - on("S1", KS) * HZ.ind0(KS), tag=nm("S1"))
        i1, g1, ta1, s1 = split_b(fin[1]["r"])
        TS.sumzero(Ex.atom(EqP(tv_all, list(ta1) + list(s1))) * Ex.atom(EqP(TS.vars("tau"), pts[1]["tau"])) * S["mask"].ex
                   - on("a1", TS) * HZ.ind0(TS), tag=nm("a1"))
        wt1 = TZ._qq_weights(t, pts[1], g1, i1)
        CS.sumzero(Ex.atom(EqP(CS.vars("s"), s1))
                   * Ex.atom(MZ.EVec(nm("wt1"), wt1[:, :1 << (t.HB + DB)], CS.vars("i") + CS.vars("h"))) * S["qL"].ex
                   - on("b1", CS) * HZ.ind0(CS), tag=nm("b1"))
        e1_pub = TZ.eqv(pts[1]["s"], s1)
        SCX.zero(Ex.const(e1_pub) * X_("a1") * X_("b1") * X_("c1") + Ex.const(rhos[0]) * X_("gr1") - Ex.const(fin[1]["v"]),
                 tag=nm("SC1.fin"))

        # SC2, SC3
        for k, Mc, Ec in ((2, cs["N"], S["E"]), (3, cs["M2"], S["Ehi"])):
            pk = pts[k]
            CS.sumzero(Ex.atom(EqP(cv_all, list(pk["i"]) + list(pk["r"]) + list(pk["g"]) + list(pk["s"])))
                       * (Mc.ex - on(f"S{k}", CS)), tag=nm(f"S{k}"))
            ik, gk_, tk, sk = split_b(fin[k]["r"])
            KS.sumzero(Ex.atom(EqP(kv_all, list(pk["r"]) + list(gk_) + list(tk) + list(sk)))
                       * (Ec.ex - on(f"b{k}", KS)), tag=nm(f"b{k}"))
            cpub = hmul(TZ.eqv(pk["s"], sk), hmul(TZ.eqv(pk["g"], gk_), TZ.eqv(pk["i"], ik)))
            cl = X_(f"c{k}") if k == 2 else X_(f"c{k}") * X_(f"c{k}")
            SCX.zero(Ex.const(cpub) * X_(f"b{k}") * cl + Ex.const(rhos[k - 1]) * X_(f"gr{k}") - Ex.const(fin[k]["v"]),
                     tag=nm(f"SC{k}.fin"))

        # SC4
        GS.sumzero(Ex.atom(EqP(gv_all, list(pts[4]["i"]) + list(pts[4]["g"]) + list(pts[4]["s"])))
                   * (S["Q2"].ex - on("S4", GS)), tag=nm("S4"))
        i4, g4, t4, s4 = split_b(fin[4]["r"])
        TS.sumzero(Ex.atom(EqP(tv_all, list(t4) + list(s4))) * S["mask"].ex - on("a4", TS) * HZ.ind0(TS), tag=nm("a4"))
        cpub4 = hmul(TZ.eqv(pts[4]["s"], s4), hmul(TZ.eqv(pts[4]["g"], g4), TZ.eqv(pts[4]["i"], i4)))
        SCX.zero(Ex.const(cpub4) * X_("a4") * X_("c4") * X_("c4") + Ex.const(rhos[3]) * X_("gr4") - Ex.const(fin[4]["v"]),
                 tag=nm("SC4.fin"))

        # Bind per-slot c1_col .. c4_col on S_sc: s < M_C claimed against H, s >= M_C locked to slot 0, c_k = c_k_col~(sk)
        pub = lambda xs: [("pub", _e(x)) for x in xs]
        for k, port in ((1, "K"), (2, "V"), (3, "V"), (4, "V")):
            ik, gk_, tk, sk = split_b(fin[k]["r"])
            cv_arr = None
            c0_val = None
            if pv:
                cv_arr = np.zeros((3, K), dtype=np.uint64)
                for s_i, vv in enumerate(ck_cols_v[k]):
                    cv_arr[0, s_i], cv_arr[1, s_i], cv_arr[2, s_i] = vv
                c0_val = ck_cols_v[k][0]
            ck_col = MZ.ecol(S_sc, f"c{k}", cv_arr)
            if self.M_C < K:
                c0_nm = hs.scalar(nm(f"c{k}0"), c0_val)
                S_sc.sumzero(S_sc.ind(s=0) * ck_col.ex - hs.SC.on(c0_nm, S_sc) * HZ.ind0(S_sc), tag=nm(f"c{k}.0"))
                S_sc.zero(ck_col.ex - hs.SC.on(c0_nm, S_sc), sel=S_sc.rng("s", self.M_C, K), tag=nm(f"c{k}.pad"))
            S_sc.sumzero(Ex.atom(EqP(sv_s, sk)) * (ck_col.ex - on(f"c{k}", S_sc)), tag=nm(f"c{k}.mle"))
            for s_i in range(self.M_C):
                coords = self.lay.kv_point(pub(tk), self.l_ents[s_i], port, pub(gk_), pub(ik))
                hs.claim_col_cell(coords, S_sc, ck_col.ex, s_i, value_wit=ck_cols_v[k][s_i] if pv else None)
        self.info["round_s"] = round(time.time() - t0, 3)

    def _round_values(self, pts, fin, split_b, ck_cols_v):
        t, wc = self.t, self._wit_cache
        K = self.K
        coefs = self.S["coefs"]
        mle, ext, fe = TZ._mle, TZ.ext, TZ.fe
        v = {}
        for k in range(1, 5):
            v[f"G{k}"] = TZ.mask_sum(coefs[k - 1], len(fin[k]["r"]))
            v[f"gr{k}"] = TZ.mask_eval(coefs[k - 1], fin[k]["r"])
        p1 = pts[1]
        SHb = t.SH
        rem_f = np.zeros_like(wc["idx"], dtype=np.int64)
        for li, rl_arr in enumerate(wc["rem_limbs"]):
            rem_f = rem_f + (rl_arr.astype(np.int64) << (16 * li))
        D_arr = wc["idx"].astype(object) * (1 << SHb) + rem_f.astype(object) - (1 << (SHb - 1))
        z_arr = (wc["m"].astype(object)[:, None, :] - D_arr.reshape(K, t.T_pad, t.G * t.R)).reshape(K, -1)
        zm = np.repeat(wc["mask"].astype(object), t.G * t.R, axis=1) * z_arr
        v["S1"] = mle(ext(fe(zm.ravel())), list(p1["r"]) + list(p1["g"]) + list(p1["tau"]) + list(p1["s"]))
        v["S2"] = mle(ext(fe(wc["N"].ravel())), list(pts[2]["i"]) + list(pts[2]["r"]) + list(pts[2]["g"]) + list(pts[2]["s"]))
        v["S3"] = mle(ext(fe(wc["M2"].ravel())), list(pts[3]["i"]) + list(pts[3]["r"]) + list(pts[3]["g"]) + list(pts[3]["s"]))
        v["S4"] = mle(ext(fe(wc["Q2"].ravel())), list(pts[4]["i"]) + list(pts[4]["g"]) + list(pts[4]["s"]))
        i1, g1, ta1, s1 = split_b(fin[1]["r"])
        m_e = CPU.emul( np.tile(TZ.eq_t(p1["tau"]), (1, K)), ext(fe(wc["mask"].ravel())) )
        v["a1"] = mle(m_e, list(ta1) + list(s1))
        wt1 = TZ._qq_weights(t, p1, g1, i1)[:, :t.G * t.R * t.DH]
        es1 = TZ.eq_t(s1)
        b1_acc = ZERO
        for s_i in range(K):
            bs = CPU.edot_dev(wt1, ext(fe(wc["q_int"][s_i].ravel())))
            b1_acc = hadd(b1_acc, hmul(_e(es1[:, s_i]), bs))
        v["b1"] = b1_acc
        for k, Ew in ((2, wc["E"]), (3, wc["Ehi"])):
            ik, gk_, tk, sk = split_b(fin[k]["r"])
            v[f"b{k}"] = mle(ext(fe(Ew.ravel())), list(pts[k]["r"]) + list(gk_) + list(tk) + list(sk))
        i4, g4, t4, s4 = split_b(fin[4]["r"])
        v["a4"] = mle(ext(fe(wc["mask"].ravel())), list(t4) + list(s4))
        for k in range(1, 5):
            _, _, _, sk = split_b(fin[k]["r"])
            esk = TZ.eq_t(sk)
            acc_c = ZERO
            for s_i, cv_s in enumerate(ck_cols_v[k]):
                acc_c = hadd(acc_c, hmul(_e(esk[:, s_i]), cv_s))
            v[f"c{k}"] = acc_c
        return v
