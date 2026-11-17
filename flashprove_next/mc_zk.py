# SPDX-License-Identifier: Apache-2.0
"""Per-proof model-commitment path (specs/09) in ZERO KNOWLEDGE: hidden column opening (HCO) inside flashprove_next.zk_flat.

Statement (public): profile constants (tile shape, R_MAX, rate, m_q), model_id, the point (xi, zeta) of the tile claims,
and -- standalone only -- cl_digest, a salted Poseidon2 digest of the hidden claim list (stand-in for the outer proof:
in the integrated proof the claim list is bound by the shared-challenge tile sumcheck of specs/09 §3.1, not built here).

Relation proven (specs/09 §3.1): there exist a claim list (t_j, c_j)_{j < J}, J claim slots (a public PROFILE
constant, power of two, default R_MAX, independent of the row count R_MAX; unused slots point to a zero row with c_j = 0,
so the number of touched tiles is hidden; several slots MAY name the same row), a static codeword M with salted column
leaves, and the model_id preimage (s_0, C_W, C_raw, C_cfg) such that, with gamma, Q, lambda, beta drawn by Fiat-Shamir
in this order:
    model_id = H(DOM_MID; s_0, C_W, C_raw, C_cfg)                            (in-circuit Poseidon2, C_W hidden)
    alpha_i  = sum_j gamma^j [t_j = i]                                       (weighted LogUp, challenge lambda)
    sum_j gamma^j c_j = u~(zeta, xi)            u = alpha^T W^cd committed BEFORE Q is drawn
    for each of the m_q public positions j_q in Q: column M[:, j_q] (hidden) hashes with its hidden salt to a leaf whose
        Merkle path (hidden siblings, public direction bits = bits of j_q) ends at C_W
    sum_q beta_q RS(u)[j_q] = <u, g> = sum_q beta_q <alpha, M[:, j_q]>,   g_k = sum_q beta_q w^{j_q k}  (verifier: g~ in
        O(m_q log n_c) via modelcommit.gtilde)
Everything except the public statement and the challenges is a committed (T1-blinded) vector: touched-tile indices,
claim values, weights, u, alpha, columns, salts, siblings, C_W, s_0, C_raw, C_cfg.  The circuit shape depends only on
the public Params (R_MAX, n_c, m_q, rate) and the public claim-slot count J, never on the model or on the touched set.

Claim side vs. row side.  The ROW side (alpha, u, the opened columns M[:, j_q], the Merkle tree, the LogUp table rows
i < R_MAX) is R_MAX-sized; only the CLAIM side (t, c, gamma^j, the LogUp slot rows) is J-sized, JB = log2 J <= KB (the
gamma-weighted claim sum lives on the first JB variables of the u cube).  The LogUp helper has XB = max(IB, JB) bits
per half: slot rows x >= J have gamma^x = 0 (forces H = 0 there), table rows x >= R_MAX read alpha masked to 0 (forces
H = 0 there); with J == R_MAX the claim and row sides coincide.

Soundness with repeated rows (paper argument; |E| = p^3 ~ 2^192):
  * LogUp (lambda after t, alpha committed): sum_j gamma^j / (lambda - t_j) = sum_{i < R_MAX} alpha_i / (lambda - i) as
    rational functions iff alpha_i = sum_{j : t_j = i} gamma^j for every i < R_MAX AND every t_j lies in [0, R_MAX)
    (a t_j outside the table leaves a pole the right side cannot cancel).  Repeated t_j simply add their gamma^j into
    one alpha_i.  Error <= (J + R_MAX) / |E| (degree of the cleared numerator), plus J / |E| for lambda hitting a t_j.
  * claims (gamma after t, c committed, u after gamma): given the alpha above, u = alpha^T W^cd (column check) and
    sum_j gamma^j c_j = u~(zeta, xi) = sum_j gamma^j W~_{t_j}(xi, zeta), i.e. sum_j gamma^j (c_j - W~_{t_j}(xi, zeta)) = 0.
    Each SLOT keeps its own power gamma^j even when rows repeat, so a single wrong c_j (next to honest slots of the same
    row) leaves a nonzero polynomial of degree < J in gamma: error <= (J - 1) / |E|.
  * column check / Merkle: unchanged (row side), error terms independent of J.
  Only the gamma and lambda terms grow with J (linearly, in |E|^-1 units): J = 2^16 costs < 2^-174.

Transcript order (specs/09 §3.1 / 02): statement -> phase 1 (claim list) -> gamma -> round A (u, alpha) -> Q, lambda ->
round B (columns, Poseidon2 rows, LogUp helper) -> beta -> LogUp / Libra phase -> sumchecks -> openings.  The extra
rounds use zk_flat.Circuit.rounds (zk_flat._commit_rounds).

Spaces (all padded = blinding half, except the Poseidon2 rows which keep TB uniform tail rows):
    mcCL (part 2, j JB)   cl: part 0 = t_j, 1 = Re c_j, 2 = Im c_j                         phase 1
    mcU  (k KB)           u in E^{n_c} (one E-valued vector)                                round A
    mcAL (i IB)           alpha in E^{R_MAX}                                                round A
    mcM  (q QB, i IB)     M[i, j_q]                                                         round B
    mcPS (q, b)           Poseidon2 rows: column block q < m_q of 2^CB rows (b < A: absorb of rows 8b..8b+7,
                          A <= b < A + depth: path, b = 2^CB - 1: leaf init); model_id rows; claim-list sponge rows
    mcLU (h 1, x XB)      weighted-LogUp helper (h = 0: slot rows j < J, h = 1: table rows i < R_MAX), E-valued

usage:
  python -m flashprove_next.mc_zk qwen --model PATH/TO/Qwen3-0.6B --store DIR --k1 210 --k2 105 --out FILE
  (DIR from `python -m flashprove_next.modelcommit commit ... --store DIR`, which saves the model_id preimage)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from . import poseidon2 as PS
from . import modelcommit as MC
from .zk_flat import Ex, EqP, SelLt, TB
from .field import P, CPU, ZERO, ONE, hadd, hsub, hmul, rand_field
from .hwinfo import describe as describe_machine
from .transcript import Transcript, dumps

VERSION = "fpn-mczk-1"
LABEL = b"fpn-mczk-v1"
DOM_CL = 24
XE1, XE2 = (0, 1, 0), (0, 0, 1)   # u, u^2 of E = F_p[u] / (u^3 - 2)


def _lg(n):
    return max(0, int(n - 1).bit_length())


class Shape:
    """Public layout constants derived from the profile Params and the profile claim-slot count J only."""

    def __init__(self, prm, anchor=True, vol_blocks=0, J=None):
        assert prm.R_MAX & (prm.R_MAX - 1) == 0 and prm.R_MAX >= 16, "R_MAX must be a power of two >= 16"
        J = prm.R_MAX if J is None else int(J)
        assert J & (J - 1) == 0 and J >= 16, "claim slots J must be a power of two >= 16"
        self.prm = prm
        self.anchor = anchor                            # standalone: salted claim-list digest rows in mcPS
        self.vol_blocks = int(vol_blocks)               # specs/09 §3.2: DOM_CFG sponge blocks (P_max, e.g. 64)
        self.IB = prm.R_MAX.bit_length() - 1            # rows of W (row side: alpha, columns, LogUp table)
        self.J = J                                      # claim slots (claim side: t, c, gamma^j, LogUp slot rows)
        self.JB = J.bit_length() - 1
        self.XB = max(self.IB, self.JB)                 # LogUp helper bits per half
        self.QB = _lg(prm.MQ)
        self.KB = prm.nc.bit_length() - 1
        assert self.QB + self.IB <= self.KB, "column block must embed into the u cube"
        assert self.JB <= self.KB, "claim slots must embed into the u cube"
        self.A = prm.R_MAX // 8                         # absorb rows per column leaf
        self.depth = prm.depth
        self.CB = _lg(self.A + self.depth + 1)          # rows per column block
        self.INIT = (1 << self.CB) - 1
        self.ROOT = self.A + self.depth - 1
        self.MID = prm.MQ << self.CB                    # model_id rows MID, MID + 1
        if anchor:
            self.X0 = ((self.MID + 2 + (1 << self.JB) - 1) >> self.JB) << self.JB   # claim-list sponge base (aligned)
            self.NB = 4 << (self.JB - 3)                # claim-list absorb rows ([t | c0 | c1 | c2], 8 per row)
            self.CLI = self.X0 + (1 << (self.JB - 1))   # claim-list sponge init row
            last = self.CLI
        else:
            self.X0 = self.NB = self.CLI = None
            last = self.MID + 1
        if self.vol_blocks > 0:
            vb = _lg(self.vol_blocks)
            assert self.vol_blocks == (1 << vb), "vol_blocks must be a power of two"
            self.VB_SP = vb
            step = 1 << (vb + 1)
            self.V0 = ((last + 1 + step - 1) // step) * step
            last = self.V0 + self.vol_blocks - 1
        else:
            self.VB_SP = self.V0 = None
        self.PSB = (last + TB).bit_length()
        assert last < (1 << self.PSB) - TB

    def xbit(self, p):
        """flat Poseidon-row bit p as (axis, bit) of the (q, b) space."""
        return ("b", p) if p < self.CB else ("q", p - self.CB)


# ============================================================================ public atoms
class EVec(Z.Pub):
    """E-valued public vector (challenge powers, beta) on the local variables `varmap`."""

    def __init__(self, name, vals, varmap, fixed=()):
        super().__init__(varmap, fixed)
        v = np.zeros((3, 1 << len(self.varmap)), dtype=np.uint64)
        a = np.asarray(vals, dtype=np.uint64)
        v[:, :a.shape[1]] = a
        self.v = v
        self._dev = None
        self.key = ("evec", name + ":" + hashlib.blake2b(v.tobytes(), digest_size=12).hexdigest(), self.varmap, self.fixed)

    def local_table(self, B):
        if getattr(B, "name", "") != "cupy":
            return self.v
        if self._dev is None:
            self._dev = B.asarray(self.v)
        return self._dev

    def local_mle(self, z):
        return CPU.edot_dev(CPU.eq_table(list(z)), self.v)


class GQ(Z.Pub):
    """g_k = sum_q beta_q w^{j_q k}, k < n_c (w = n-th root): <u, g> = sum_q beta_q RS(u)[j_q].  MLE in O(m_q log n_c)."""

    def __init__(self, prm, Q, beta, varmap):
        super().__init__(varmap, ())
        self.prm, self.Q, self.beta = prm, [int(j) for j in Q], [tuple(int(x) % P for x in b) for b in beta]
        assert len(self.varmap) == prm.nc.bit_length() - 1
        d = hashlib.blake2b(json.dumps([self.Q, self.beta]).encode(), digest_size=12).hexdigest()
        self.key = ("gq", d, self.varmap, self.fixed)
        self._tab = {}

    def local_table(self, B):
        k = getattr(B, "name", "")
        if k not in self._tab:
            prm = self.prm
            bv = np.zeros((3, prm.n), dtype=np.uint64)
            for j, b in zip(self.Q, self.beta):
                bv[0, j] = (int(bv[0, j]) + b[0]) % P
                bv[1, j] = (int(bv[1, j]) + b[1]) % P
                bv[2, j] = (int(bv[2, j]) + b[2]) % P
            if k == "cupy":
                xp = B.xp
                self._tab[k] = xp.ascontiguousarray(xp.stack([B.ntt(B.asarray(bv[l]), prm.n)[:prm.nc] for l in (0, 1, 2)]))
            else:
                self._tab[k] = np.stack([CPU.ntt(bv[l], prm.n)[:prm.nc] for l in (0, 1, 2)])
        return self._tab[k]

    def local_mle(self, z):
        return MC.gtilde(self.prm, self.Q, self.beta, z)


# ============================================================================ helpers
def ecol(sp, name, vals):
    """E-valued committed column on a padded space (vals: (3, <= sp.n) field elements, prover only).
    Stores content rows only (sp.content = v.N - v.tail), cutting E-witness RAM & spill in half."""
    cir = sp.ctx.cir
    v = cir.vec(f"{sp.name}.{name}", sp.nv, sp.content, tail=sp.tail)
    v.is_ext = True
    if sp.ctx.prove:
        a = np.asarray(vals, dtype=np.uint64)
        if a.shape[1] == sp.content:
            v.vals = np.ascontiguousarray(a)
        elif a.shape[1] > sp.content:
            v.vals = np.ascontiguousarray(a[:, :sp.content])
        else:
            full = np.zeros((3, sp.content), dtype=np.uint64)
            full[:, :a.shape[1]] = a
            v.vals = full
    c = G.Col(sp, v, None)
    sp.ctx.colreg[v.id] = c
    return c


def view(col, sp2, amap, pad_to=None, shift=0):
    """View of a committed column on another space with explicit per-bit maps (no small-column copies: every vector
    here has >= 16 blinding rows and at most a few claims; zk_flat.zk_budget_check verifies it on every proof)."""
    return col._on(sp2, col._bits(sp2, amap), shift, pad_to)


def epowers(g, n):
    out = np.zeros((3, n), dtype=np.uint64)
    x = ONE
    for j in range(n):
        out[0, j], out[1, j], out[2, j] = x; x = hmul(x, g)
    return out


def _e(t):
    return (int(t[0]) % P, int(t[1]) % P, int(t[2]) % P)


def tile_claim(row, xi, zeta):
    """c = W~_row(xi, zeta) for one tile row (n_c signed ints, index r * T_I + c; point order zeta (c bits), xi)."""
    e = CPU.eq_table(list(zeta) + list(xi))
    return Z._dot_base(CPU.fromi(np.asarray(row, dtype=np.int64)), e)


def cl_flat(sh, ts, cs):
    """[t | c0 | c1 | c2] as field elements (length 4 J)."""
    J = sh.J
    f = np.zeros(4 * J, dtype=np.uint64)
    f[:J] = np.asarray(ts, dtype=np.uint64)
    f[J:2 * J] = np.asarray([c[0] for c in cs], dtype=np.uint64)
    f[2 * J:3 * J] = np.asarray([c[1] for c in cs], dtype=np.uint64)
    f[3 * J:4 * J] = np.asarray([c[2] for c in cs], dtype=np.uint64)
    return f


def cl_digest(sh, ts, cs, salt):
    """Salted Poseidon2 sponge of the claim list (standalone anchor): block 0 = [salt, 0, 0, 0, 0], then [t | c0 | c1 | c2]."""
    f = cl_flat(sh, ts, cs)
    blocks = np.concatenate([np.array(list(salt) + [0, 0, 0, 0], dtype=np.uint64), f]).reshape(-1, 8, 1)
    d = PS.sponge(blocks, CPU, domain=DOM_CL)
    return tuple(int(v) for v in d[:, 0])


def pad_claims(sh, tiles, cvals):
    """Slots: touched tiles first (repeats allowed), then zero-row padding (t = R_MAX - 1, c = 0)."""
    J = sh.J
    assert len(tiles) <= J
    ts = list(tiles) + [sh.prm.R_MAX - 1] * (J - len(tiles))
    cs = [_e(c) for c in cvals] + [ZERO] * (J - len(tiles))
    return ts, cs


# ============================================================================ prover-side data
def _alpha(sh, ts, Gam):
    """alpha_i = sum_{j : t_j = i} gamma^j over the R_MAX table rows (row side; J slots may repeat rows)."""
    R = sh.prm.R_MAX
    a = np.zeros((3, R), dtype=np.uint64)
    for j, t in enumerate(ts):
        if not 0 <= t < R:                                  # only a cheating prover: no table row to put it in
            continue
        v = hadd((int(a[0, t]), int(a[1, t]), int(a[2, t])), (int(Gam[0, j]), int(Gam[1, j]), int(Gam[2, j])))
        a[0, t], a[1, t], a[2, t] = v
    return a


def _combined_row(prm, store, alpha, B):
    """u = alpha^T W over the used rows with alpha != 0 (O(touched rows * n_c); wide rows like PN / CFG0 / CFG1 handled
    in exact field arithmetic)."""
    rows = [i for i in range(min(store.n_used, prm.R_MAX)) if alpha[0, i] or alpha[1, i] or alpha[2, i]]
    if not rows:
        return np.zeros((3, prm.nc), dtype=np.uint64)
    wide_set = set(getattr(store, "wide", {}).keys())
    reg_rows = [i for i in rows if i not in wide_set]
    wide_rows = [i for i in rows if i in wide_set]
    out = np.zeros((3, prm.nc), dtype=np.uint64)
    if reg_rows:
        if MC._is_gpu(B):
            import cupy as cp
            u_acc = B.zeros_e(prm.nc)
            for c0 in range(0, len(reg_rows), 64):
                ch = reg_rows[c0:c0 + 64]
                Wh = np.stack([np.asarray(store.coeffs[i], dtype=np.int32) for i in ch]).reshape(1, len(ch), prm.nc)
                V = alpha[:, ch].reshape(3, 1, len(ch))
                u_ch = MC.tiles_vecmat(B, B.xp.asarray(Wh), [0], B.asarray(V))[:, 0]
                u_acc = B.eadd(u_acc, u_ch)
                del Wh, u_ch
                cp.get_default_memory_pool().free_all_blocks()
            out = B.tonumpy(u_acc)
            del u_acc
            cp.get_default_memory_pool().free_all_blocks()
        else:
            for c0 in range(0, len(reg_rows), 64):
                ch = reg_rows[c0:c0 + 64]
                Wh = np.stack([np.asarray(store.coeffs[i], dtype=np.int32) for i in ch]).reshape(1, len(ch), prm.nc)
                V = alpha[:, ch].reshape(3, 1, len(ch))
                out = CPU.eadd(out, MC.tiles_vecmat(B, Wh, [0], V)[:, 0])
                del Wh
    for i in wide_rows:
        rf = CPU.fromi(store.row(i))
        ai = (int(alpha[0, i]), int(alpha[1, i]), int(alpha[2, i]))
        out = CPU.eadd(out, CPU.escale(CPU.ext(rf), ai))
    return out


def ps_inputs(sh, Q, cols, salts, paths, pre, clf, salt_cl, vol_elems=None):
    """Poseidon2 permutation inputs (12, content rows) of every mcPS row; returns (inputs, roots (4, m_q), mid, digest, c_cfg_vol)."""
    prm = sh.prm
    n = (1 << sh.PSB) - TB
    inp = np.zeros((12, n), dtype=np.uint64)
    mq = prm.MQ
    Qa = np.asarray(Q, dtype=np.uint64)
    base = np.arange(mq, dtype=np.int64) << sh.CB
    st = np.zeros((12, mq), dtype=np.uint64)
    st[0] = Qa; st[1:5] = salts; st[5] = prm.R_MAX; st[10] = MC.DOM_COL; st[11] = PS.TAG_SPONGE
    inp[:, base + sh.INIT] = st
    st = PS.permute(st, CPU)
    for b in range(sh.A):
        st = st.copy()
        st[0:8] = CPU.add(st[0:8], np.ascontiguousarray(cols[8 * b:8 * b + 8]))
        inp[:, base + b] = st
        st = PS.permute(st, CPU)
    cur = st[0:4].copy()
    sib = np.asarray(paths, dtype=np.uint64)                 # (m_q, depth, 4)
    for d in range(sh.depth):
        bit = ((Qa >> np.uint64(d)) & np.uint64(1)).astype(bool)
        s = np.ascontiguousarray(sib[:, d, :].T)
        st = np.zeros((12, mq), dtype=np.uint64)
        st[0:4] = np.where(bit, s, cur); st[4:8] = np.where(bit, cur, s); st[11] = PS.TAG_NODE
        inp[:, base + sh.A + d] = st
        cur = PS.permute(st, CPU)[0:4]
    roots = cur
    cw = roots[:, mq - 1]
    c_cfg_vol = None
    if sh.vol_blocks > 0:
        assert vol_elems is not None and len(vol_elems) == sh.vol_blocks * 8
        vblks = np.asarray(vol_elems, dtype=np.uint64).reshape(sh.vol_blocks, 8)
        stv = np.zeros((12, 1), dtype=np.uint64)
        stv[10, 0] = MC.DOM_CFG; stv[11, 0] = PS.TAG_SPONGE
        for rho in range(sh.vol_blocks):
            stv = stv.copy()
            stv[0:8, 0] = CPU.add(stv[0:8, 0], vblks[rho])
            inp[:, sh.V0 + rho] = stv[:, 0]
            stv = PS.permute(stv, CPU)
        c_cfg_vol = tuple(int(v) for v in stv[0:4, 0])
        ccfg_vec = stv[0:4, 0].copy()
    else:
        ccfg_vec = np.asarray(pre["c_cfg"], dtype=np.uint64)
    stA = np.zeros((12, 1), dtype=np.uint64)
    stA[0:4, 0] = pre["s0"]; stA[4:8, 0] = cw; stA[10] = MC.DOM_MID; stA[11] = PS.TAG_SPONGE
    inp[:, sh.MID] = stA[:, 0]
    stB = PS.permute(stA, CPU).copy()
    stB[0:4, 0] = CPU.add(stB[0:4, 0], np.asarray(pre["c_raw"], dtype=np.uint64))
    stB[4:8, 0] = CPU.add(stB[4:8, 0], ccfg_vec)
    inp[:, sh.MID + 1] = stB[:, 0]
    mid = tuple(int(v) for v in PS.permute(stB, CPU)[0:4, 0])
    if not sh.anchor:
        return inp, roots, mid, None, c_cfg_vol
    st = np.zeros((12, 1), dtype=np.uint64)
    st[0:4, 0] = salt_cl; st[10] = DOM_CL; st[11] = PS.TAG_SPONGE
    inp[:, sh.CLI] = st[:, 0]
    st = PS.permute(st, CPU)
    blocks = clf[:4 * sh.J].reshape(sh.NB, 8)
    for rho in range(sh.NB):
        st = st.copy()
        st[0:8, 0] = CPU.add(st[0:8, 0], blocks[rho])
        inp[:, sh.X0 + rho] = st[:, 0]
        st = PS.permute(st, CPU)
    dig = tuple(int(v) for v in st[0:4, 0])
    return inp, roots, mid, dig, c_cfg_vol


class CfgExt:
    """specs/09 §3.2 in-circuit configuration opening:
      * (O1) volatile config M^vol: committed as `vol_col` (P_max * 8 field elements) in phase 1, absorbed through
             DOM_CFG Poseidon2 sponge rows inside mcPS, and wired directly into the DOM_MID model_id sponge rows.
             Callers (`sampler_zk`, `reply_zk`) read K_prf (indices 3..6) and sys_prompt (indices 19..) from `vol_col`.
      * (O2) stable tables Tt (8 cols), Vt (4 cols), Lt (8 cols): committed in phase 1 on padded spaces and bound to
             META_CFG0, META_CFG1 at the HCO point (xi, zeta) via two claim slots (no extra rounds)."""

    def __init__(self, ctx, prm, store=None, vol_elems=None, cfg_rows=False, vol_blocks=0, meta_cfg0=2, meta_cfg1=3):
        self.ctx, self.prm = ctx, prm
        self.pv = ctx.prove
        self.cfg_rows = bool(cfg_rows)
        self.vol_blocks = int(vol_blocks)
        self.meta_cfg0, self.meta_cfg1 = int(meta_cfg0), int(meta_cfg1)
        self.vol_col = None
        self.vol_elems = None
        if self.vol_blocks > 0:
            vb = _lg(self.vol_blocks)
            assert self.vol_blocks == (1 << vb)
            VS = ctx.space("mcVOL", [("x", vb + 3)])
            vv = None
            if self.pv:
                assert vol_elems is not None and len(vol_elems) == self.vol_blocks * 8
                self.vol_elems = np.asarray(vol_elems, dtype=np.uint64)
                # field values >= 2^63 are passed as negative representatives (Space.col reduces them mod p)
                vv = np.where(self.vol_elems >= (1 << 63),
                              self.vol_elems.astype(object) - int(P),
                              self.vol_elems.astype(object))
            self.VS = VS
            self.vol_col = VS.col("vol", vv)
        self.tt_cols = self.vt_cols = self.lt_cols = None
        self.TTS = self.VTS = self.LTS = None
        self._cfg0_row = self._cfg1_row = None
        if self.cfg_rows:
            NT, NV, NL = MC.cfg_sizes(prm)
            self.NT, self.NV, self.NL = NT, NV, NL
            NB, VB, LB = (NT - 1).bit_length(), (NV - 1).bit_length(), (NL - 1).bit_length()
            self.NB, self.VB, self.LB = NB, VB, LB
            self.TTS = ctx.space("mcTT", [("idx", NB)])
            self.VTS = ctx.space("mcVT", [("idx", VB)])
            self.LTS = ctx.space("mcLT", [("idx", LB)])
            tt_v = vt_v = lt_v = None
            if self.pv:
                c0 = store.row(self.meta_cfg0)
                c1 = store.row(self.meta_cfg1)
                self._cfg0_row, self._cfg1_row = c0, c1
                tt_v = [c0[f * NT:(f + 1) * NT].copy() for f in range(8)]
                vt_v = [c1[f * NV:(f + 1) * NV].copy() for f in range(4)]
                lt_v = [c1[4 * NV + f * NL:4 * NV + (f + 1) * NL].copy() for f in range(8)]
            self.tt_cols = [self.TTS.col(f"f{f}", tt_v[f] if self.pv else None) for f in range(8)]
            self.vt_cols = [self.VTS.col(f"f{f}", vt_v[f] if self.pv else None) for f in range(4)]
            self.lt_cols = [self.LTS.col(f"f{f}", lt_v[f] if self.pv else None) for f in range(8)]
            self.tt_vals, self.vt_vals, self.lt_vals = tt_v, vt_v, lt_v

    def cfg_claims(self, xi, zeta):
        """Prover helper: evaluate (c_cfg0, c_cfg1) = (CFG0~(xi, zeta), CFG1~(xi, zeta))."""
        assert self.pv and self.cfg_rows
        return tile_claim(self._cfg0_row, xi, zeta), tile_claim(self._cfg1_row, xi, zeta)

    def bind_claims(self, xi, zeta, c_cfg0_on, c_cfg1_on):
        """specs/09 §3.2: bind committed Tt, Vt, Lt columns to claim-slot expressions c_cfg0_on(sp), c_cfg1_on(sp)
        at point (xi, zeta) in O(NT + NV + NL) cells, zero extra commit rounds."""
        if not self.cfg_rows:
            return
        pt = [_e(z) for z in zeta] + [_e(x) for x in xi]
        NB, VB, LB = self.NB, self.VB, self.LB

        def eq_const(bits_pt, val):
            acc = ONE
            for b, z in enumerate(bits_pt):
                acc = hmul(acc, z if ((val >> b) & 1) else hsub(ONE, z))
            return acc

        TTS, VTS = self.TTS, self.VTS
        eq_tt = Ex.atom(EqP(TTS.vars("idx"), pt[:NB]))
        acc0 = Ex.const(0)
        for f in range(8):
            acc0 = acc0 + self.tt_cols[f].ex * Ex.const(eq_const(pt[NB:], f))
        TTS.sumzero(eq_tt * acc0 - TTS.ind(idx=0) * c_cfg0_on(TTS), tag="mc.cfg0.bind")

        eq_vt = Ex.atom(EqP(VTS.vars("idx"), pt[:VB]))
        acc_vt = Ex.const(0)
        for f in range(4):
            acc_vt = acc_vt + self.vt_cols[f].ex * Ex.const(eq_const(pt[VB:], f))
        eq_lt = Ex.atom(EqP(VTS.vars("idx")[:LB], pt[:LB], fixed=tuple((v, 0) for v in VTS.vars("idx")[LB:])))
        acc_lt = Ex.const(0)
        base_blk = (4 * self.NV) // self.NL
        for f in range(8):
            lv = view(self.lt_cols[f], VTS, {"idx": [("idx", s) for s in range(LB)]})
            acc_lt = acc_lt + lv * Ex.const(eq_const(pt[LB:], base_blk + f))
        VTS.sumzero(eq_vt * acc_vt + eq_lt * acc_lt - VTS.ind(idx=0) * c_cfg1_on(VTS), tag="mc.cfg1.bind")


# ============================================================================ circuit
def build(prm, stmt, wit=None, B=CPU, info=None, ctx=None, ext=None, cfg_ext=None):
    """Build the HCO circuit (prover: wit given; verifier: wit None).  Phase-1 vectors are declared here, the later
    rounds are registered as zk_flat commit rounds (run inside zk_flat.prove / verify).  Returns the gadgets Ctx.

    ext (integrated proof, flashprove_next.wsum_zk): the claim list comes from an earlier builder in the same circuit instead
    of the standalone phase-1 list + digest.  ext provides t_on(sp2, jmap) / c_on(sp2, jmap) (views of the hidden slot
    indices / E-valued claims, jmap: per-bit list over the JB slot bits), xi() / zeta() (the tile point, drawn by its
    rounds), ts() / cs() (prover values, J slots), J (claim slots, default R_MAX) and rounds (its commit rounds, run
    before ours).  ctx: the shared gadgets Ctx.  In this mode stmt carries no cl_digest / xi / zeta.
    Standalone: J = stmt["claim_slots"] (default R_MAX).
    cfg_ext (optional CfgExt, or auto-built when stmt carries `vol_blocks` / `cfg_rows` in standalone mode): opens
    M^vol in-circuit to C_cfg -> model_id and binds Tt, Vt, Lt to META_CFG0, META_CFG1."""
    pv = wit is not None
    ctx = G.Ctx(pv) if ctx is None else ctx
    if cfg_ext is None and (stmt.get("vol_blocks", 0) > 0 or stmt.get("cfg_rows", False)):
        cfg_ext = CfgExt(ctx, prm,
                         store=wit["store"] if pv else None,
                         vol_elems=wit.get("vol_elems") if pv else None,
                         cfg_rows=bool(stmt.get("cfg_rows", False)),
                         vol_blocks=int(stmt.get("vol_blocks", 0)),
                         meta_cfg0=int(stmt.get("meta_cfg0", 2)),
                         meta_cfg1=int(stmt.get("meta_cfg1", 3)))
    ctx.cfg_ext = cfg_ext
    vol_blocks = cfg_ext.vol_blocks if cfg_ext is not None else 0
    J_cl = int(getattr(ext, "J", prm.R_MAX)) if ext is not None else int(stmt.get("claim_slots", prm.R_MAX))
    if ext is not None and "claim_slots" in stmt and int(stmt["claim_slots"]) != J_cl:
        raise Z.ProofError("claim_slots of the statement and of the claim source differ")
    sh = Shape(prm, anchor=ext is None, vol_blocks=vol_blocks, J=J_cl)
    cir = ctx.cir
    info = {} if info is None else info
    J, IB, QB, KB, JB, XB = sh.J, sh.IB, sh.QB, sh.KB, sh.JB, sh.XB
    S = {}                                                  # objects shared between rounds
    if pv:
        assert wit["store"].n_used < prm.R_MAX, "padding slots need a zero row"

    def clk(tag, t0):
        info[tag] = round(info.get(tag, 0.0) + time.time() - t0, 4)

    if ext is None:
        xi = [_e(x) for x in stmt["xi"]]; zeta = [_e(z) for z in stmt["zeta"]]
        assert len(xi) == prm.ob and len(zeta) == prm.ib
        if pv:
            assert len(wit["t"]) == J
        # -------------------------------------------------------------- phase 1: hidden claim list
        CS = ctx.space("mcCL", [("part", 2), ("j", JB)])
        clf = cl_flat(sh, wit["t"], wit["c"]) if pv else None
        cl = CS.col("cl", clf)

        def cl_on(sp2, part, jmap, pad_to=None):
            pm = [("c", part & 1), ("c", part >> 1)] if isinstance(part, int) else part
            return view(cl, sp2, {"part": pm, "j": jmap}, pad_to=pad_to)

        t_on = lambda sp2, jm: cl_on(sp2, 0, jm)
        c_on = lambda sp2, jm: cl_on(sp2, 1, jm) + cl_on(sp2, 2, jm) * Ex.const(XE1) + cl_on(sp2, 3, jm) * Ex.const(XE2)
        get_t = lambda: wit["t"]
        get_c = lambda: wit["c"]
        get_pt = lambda: (xi, zeta)
        if cfg_ext is not None and cfg_ext.cfg_rows:
            s0_slot = int(stmt.get("cfg_slot0", J - 2))
            s1_slot = int(stmt.get("cfg_slot1", J - 1))
            jm0 = [("c", (s0_slot >> s) & 1) for s in range(JB)]
            jm1 = [("c", (s1_slot >> s) & 1) for s in range(JB)]
            CS.zero(t_on(CS, jm0) - Ex.const(cfg_ext.meta_cfg0), sel=CS.ind(part=0, j=0), tag="mc.cfg0.slot_t")
            CS.zero(t_on(CS, jm1) - Ex.const(cfg_ext.meta_cfg1), sel=CS.ind(part=0, j=0), tag="mc.cfg1.slot_t")
            cfg_ext.bind_claims(xi, zeta, lambda sp2: c_on(sp2, jm0), lambda sp2: c_on(sp2, jm1))
    else:
        clf = None
        t_on, c_on = ext.t_on, ext.c_on
        get_t, get_c = ext.ts, ext.cs
        get_pt = lambda: ([_e(x) for x in ext.xi()], [_e(z) for z in ext.zeta()])

    # ------------------------------------------------------------------ round A: gamma -> u, alpha
    def round_a(tr):
        t0 = time.time()
        gam = tr.ext()
        Gam = epowers(gam, J)
        xi, zeta = get_pt()
        AS = ctx.space("mcAL", [("i", IB)])
        US = ctx.space("mcU", [("k", KB)])
        al_v = u_v = None
        if pv:
            ts_, cs_ = get_t(), get_c()
            assert len(ts_) == J and len(cs_) == J
            al_v = _alpha(sh, ts_, Gam)
            u_v = _combined_row(prm, wit["store"], al_v, B)
            # prover-side check of the claim relation (a forced cheating prover keeps going)
            cv = np.array([_e(c) for c in cs_], dtype=np.uint64).T
            lhs = CPU.edot_dev(Gam, cv)
            rhs = CPU.esum(CPU.emul(CPU.eq_table(zeta + xi), u_v))
            if lhs != rhs:
                ctx.violation("sum_j gamma^j c_j != u~(zeta, xi)")
        al = ecol(AS, "alpha", al_v)
        u = ecol(US, "u", u_v)
        kv = US.vars("k")
        g_us = Ex.atom(EVec("gamma.u", Gam, kv[:JB], fixed=tuple((v, 0) for v in kv[JB:])))
        jm = [("k", s) for s in range(JB)]
        c_ex = c_on(US, jm)
        US.sumzero(u.ex * Ex.atom(EqP(kv, zeta + xi)) - g_us * c_ex, tag="mc.claims")
        if pv and hasattr(wit["store"], "drop_cache"):
            wit["store"].drop_cache()
        S.update(Gam=Gam, al=al, u=u, AS=AS, US=US, al_v=al_v)
        clk("round_a_s", t0)

    # ------------------------------------------------------------------ round B: Q, lambda -> columns, Poseidon2, LogUp
    def round_b(tr):
        t0 = time.time()
        Q = tr.indices(prm.MQ, prm.n)
        lam = tr.ext()
        S["Q"] = Q
        MS = ctx.space("mcM", [("q", QB), ("i", IB)])
        M_v = inp = None
        if pv:
            st = wit["store"]
            t1 = time.time()
            cols = (st.columns(Q, B) if hasattr(st, "columns")
                    else np.stack([np.asarray(st.column(j), dtype=np.uint64) for j in Q], axis=1))     # (R_MAX, m_q)
            salts = np.ascontiguousarray(np.stack([np.asarray(st.salts[:, j], dtype=np.uint64) for j in Q], axis=1))
            paths = [st.path(j) for j in Q]
            if hasattr(st, "drop_cache"):
                st.drop_cache()
            clk("column_read_s", t1)
            M_v = np.zeros((1 << QB, 1 << IB), dtype=np.uint64)
            M_v[:prm.MQ] = cols.T
            M_v = M_v.reshape(-1)
            t1 = time.time()
            vol_el = cfg_ext.vol_elems if (cfg_ext is not None and sh.vol_blocks > 0) else None
            inp, roots, mid, dig, c_cfg_vol = ps_inputs(sh, Q, cols, salts, paths, wit["pre"], clf, wit.get("salt_cl"),
                                                        vol_elems=vol_el)
            clk("ps_inputs_s", t1)
            if any(tuple(int(x) for x in roots[:, q]) != tuple(wit["pre"]["c_w"]) for q in range(prm.MQ)):
                ctx.violation("opened column does not authenticate to C_W")
            if c_cfg_vol is not None and "c_cfg" in wit["pre"] and c_cfg_vol != tuple(int(x) for x in wit["pre"]["c_cfg"]):
                ctx.violation("in-circuit volatile C_cfg does not match pre['c_cfg']")
            if mid != tuple(int(x) for x in stmt["model_id"]):
                ctx.violation("model_id preimage mismatch")
            if sh.anchor and dig != tuple(int(x) for x in stmt["cl_digest"]):
                ctx.violation("claim-list digest mismatch")
        M = MS.col("M", M_v)
        PSp = ctx.space("mcPS", [("q", sh.PSB - sh.CB), ("b", sh.CB)], padded=False)
        t1 = time.time()
        reg = G.PoseRegion(PSp, inp, B=B if pv else None)
        clk("poseidon_trace_s", t1)
        I = lambda k: reg.I(k).ex
        O = lambda k: reg.O(k).ex
        Osh = lambda k, d: reg.O(k).shifted(d)
        Ish = lambda k, d: reg.I(k).shifted(d)
        cst = lambda c: Ex.const(c % P)
        nv = PSp.nv
        allv = list(range(nv))

        def pt(x):
            return Ex.atom(SelLt([], 1, fixed=tuple((v, (x >> v) & 1) for v in allv)))

        def frng(lo, hi):
            return Ex.atom(SelLt(allv, hi)) - Ex.atom(SelLt(allv, lo))

        # public per-row data: leaf index j_q on init rows, path direction bits on path rows
        jq = np.zeros(1 << nv, dtype=np.int64); pb = np.zeros(1 << nv, dtype=np.int64)
        for q, j in enumerate(Q):
            jq[(q << sh.CB) + sh.INIT] = j
            for d in range(sh.depth):
                pb[(q << sh.CB) + sh.A + d] = (j >> d) & 1
        jq_ex = PSp.pub("jq", jq).ex; pb_ex = PSp.pub("pbit", pb).ex
        colsel = PSp.lt("q", prm.MQ)
        # column leaf sponge: init row
        s_init = colsel * PSp.ind(b=sh.INIT)
        PSp.zero(I(0) - jq_ex, sel=s_init, tag="mc.leaf.idx")
        PSp.zero(I(5) - cst(prm.R_MAX), sel=s_init, tag="mc.leaf.rmax")
        for k in (6, 7, 8, 9):
            PSp.zero(I(k), sel=s_init, tag="mc.leaf.zero")
        PSp.zero(I(10) - cst(MC.DOM_COL), sel=s_init, tag="mc.leaf.dom")
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=s_init, tag="mc.leaf.tag")
        # absorb rows: block b of column q adds M[8b .. 8b+7, j_q]
        first = PSp.ind(b=0); absorb = PSp.lt("b", sh.A); rest = absorb - first
        s_abs = colsel * absorb
        qmap = [("q", s) for s in range(QB)]
        for k in range(12):
            prev = first * Osh(k, sh.INIT) + rest * Osh(k, -1)
            e = I(k) - prev
            if k < 8:
                imap = [("c", (k >> s) & 1) for s in range(3)] + [("b", s) for s in range(IB - 3)]
                e = e - view(M, PSp, {"q": qmap, "i": imap}, pad_to=("b", IB - 3))
            PSp.zero(e, sel=s_abs, tag="mc.leaf.absorb")
        # Merkle path rows: public direction bit, hidden sibling (the free half of the input)
        s_path = colsel * PSp.rng("b", sh.A, sh.A + sh.depth)
        for k in range(4):
            cur = Osh(k, -1)
            PSp.zero((Ex.const(1) - pb_ex) * (I(k) - cur) + pb_ex * (I(4 + k) - cur), sel=s_path, tag="mc.path")
        for k in (8, 9, 10):
            PSp.zero(I(k), sel=s_path, tag="mc.path.pad")
        PSp.zero(I(11) - cst(PS.TAG_NODE), sel=s_path, tag="mc.path.tag")
        # every column's root equals the next one's; the last equals the C_W input of the model_id sponge
        xlast = ((prm.MQ - 1) << sh.CB) + sh.ROOT
        for k in range(4):
            if prm.MQ > 1:
                PSp.zero(Osh(k, 1 << sh.CB) - O(k), sel=PSp.lt("q", prm.MQ - 1) * PSp.ind(b=sh.ROOT), tag="mc.root.chain")
            PSp.zero(O(k) - Ish(4 + k, sh.MID - xlast), sel=pt(xlast), tag="mc.root.cw")
        # model_id = H(DOM_MID; s0, C_W, C_raw, C_cfg): rows MID (block [s0, C_W]) and MID + 1 (block [C_raw, C_cfg])
        for k in (8, 9):
            PSp.zero(I(k), sel=pt(sh.MID), tag="mc.mid.zero")
        PSp.zero(I(10) - cst(MC.DOM_MID), sel=pt(sh.MID), tag="mc.mid.dom")
        PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=pt(sh.MID), tag="mc.mid.tag")
        for k in range(8, 12):
            PSp.zero(I(k) - Osh(k, -1), sel=pt(sh.MID + 1), tag="mc.mid.cap")
        for k in range(4):
            PSp.zero(O(k) - cst(int(stmt["model_id"][k])), sel=pt(sh.MID + 1), tag="mc.mid.out")
        # volatile config sponge (specs/09 §3.2): absorb rows V0 .. V0 + vol_blocks - 1, wire output C_cfg into MID + 1
        if sh.vol_blocks > 0:
            v_first = pt(sh.V0); v_abs = frng(sh.V0, sh.V0 + sh.vol_blocks); v_rest = v_abs - v_first
            vxb = [sh.xbit(p) for p in range(sh.VB_SP)]
            vlast = sh.V0 + sh.vol_blocks - 1
            for k in range(12):
                if k < 8:
                    prev = v_rest * Osh(k, -1)
                    vm = [("c", (k >> s) & 1) for s in range(3)] + vxb
                    e = I(k) - prev - view(cfg_ext.vol_col, PSp, {"x": vm}, pad_to=sh.xbit(sh.VB_SP))
                elif k in (8, 9):
                    e = I(k) - v_rest * Osh(k, -1)
                elif k == 10:
                    e = I(10) - v_first * cst(MC.DOM_CFG) - v_rest * Osh(10, -1)
                else:
                    e = I(11) - v_first * cst(PS.TAG_SPONGE) - v_rest * Osh(11, -1)
                PSp.zero(e, sel=v_abs, tag="mc.vol.absorb")
            for k in range(4):
                PSp.zero(I(4 + k) - Osh(4 + k, -1) - Osh(k, vlast - (sh.MID + 1)), sel=pt(sh.MID + 1), tag="mc.mid.ccfg")
        # claim-list sponge (standalone anchor): init at CLI, absorb rows X0 .. X0 + NB - 1, digest = cl_digest
        for k in (range(4, 10) if sh.anchor else ()):
            PSp.zero(I(k), sel=pt(sh.CLI), tag="mc.cl.zero")
        if sh.anchor:
            PSp.zero(I(10) - cst(DOM_CL), sel=pt(sh.CLI), tag="mc.cl.dom")
            PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=pt(sh.CLI), tag="mc.cl.tag")
            c_first = pt(sh.X0); c_abs = frng(sh.X0, sh.X0 + sh.NB); c_rest = c_abs - c_first
            xb = [sh.xbit(p) for p in range(JB)]
            for k in range(12):
                prev = c_first * Osh(k, sh.CLI - sh.X0) + c_rest * Osh(k, -1)
                e = I(k) - prev
                if k < 8:
                    jm = [("c", (k >> s) & 1) for s in range(3)] + xb[:JB - 3]
                    e = e - cl_on(PSp, xb[JB - 3:JB - 1], jm, pad_to=xb[JB - 1])
                PSp.zero(e, sel=c_abs, tag="mc.cl.absorb")
            for k in range(4):
                PSp.zero(O(k) - cst(int(stmt["cl_digest"][k])), sel=pt(sh.X0 + sh.NB - 1), tag="mc.cl.out")
        # weighted LogUp: sum_j Gamma_j / (lam - t_j) = sum_i alpha_i / (lam - i)  =>  alpha_i = sum_{t_j = i} Gamma_j
        # (J slot rows on h = 0, R_MAX table rows on h = 1, each half XB = max(IB, JB) bits; see the module docstring)
        LS = ctx.space("mcLU", [("h", 1), ("x", XB)])
        H_v = None
        if pv:
            R = prm.R_MAX
            lamv = lambda n: np.stack([np.full(n, lam[l], dtype=np.uint64) for l in (0, 1, 2)])
            dt = CPU.esub(lamv(J), CPU.ext(CPU.fromi(np.asarray(get_t(), dtype=np.int64))))
            di = CPU.esub(lamv(R), CPU.ext(CPU.fromi(np.arange(R, dtype=np.int64))))
            h0 = np.zeros((3, 1 << XB), dtype=np.uint64)
            h1 = np.zeros((3, 1 << XB), dtype=np.uint64)
            h0[:, :J] = CPU.emul(S["Gam"], CPU.einv(dt))
            h1[:, :R] = CPU.esub(CPU.zeros_e(R), CPU.emul(S["al_v"], CPU.einv(di)))
            H_v = np.concatenate([h0, h1], axis=1)
        Hl = ecol(LS, "H", H_v)
        xm = [("x", s) for s in range(XB)]
        t_ls = t_on(LS, xm[:JB])                     # slot rows x >= J: gamma^x = 0 below forces H = 0 there
        a_ls = view(S["al"], LS, {"i": xm[:IB]})
        if XB > IB:                                  # table rows x >= R_MAX: alpha masked to 0 (forces H = 0 there)
            a_ls = a_ls * LS.lt("x", prm.R_MAX)
        g_ls = Ex.atom(EVec("gamma.ls", S["Gam"], LS.vars("x")))
        LS.zero(Hl.ex * (Ex.const(lam) - t_ls) - g_ls, sel=LS.ind(h=0), tag="mc.logup.slot")
        LS.zero(Hl.ex * (Ex.const(lam) - LS.idx("x")) + a_ls, sel=LS.ind(h=1), tag="mc.logup.table")
        LS.sumzero(Hl.ex, tag="mc.logup.sum")
        S.update(M=M, MS=MS)
        clk("round_b_s", t0)

    # ------------------------------------------------------------------ beta -> batched column check
    def round_c(tr):
        beta = tr.exts(prm.MQ)
        US = S["US"]
        kv = US.vars("k")
        bq = np.zeros((3, 1 << QB), dtype=np.uint64)
        for q, b in enumerate(beta):
            bq[0, q], bq[1, q], bq[2, q] = b
        bq_us = Ex.atom(EVec("beta", bq, kv[IB:IB + QB], fixed=tuple((v, 0) for v in kv[IB + QB:])))
        gq = Ex.atom(GQ(prm, S["Q"], beta, kv))
        a_us = view(S["al"], US, {"i": [("k", s) for s in range(IB)]})
        m_us = view(S["M"], US, {"q": [("k", IB + s) for s in range(QB)], "i": [("k", s) for s in range(IB)]})
        US.sumzero(S["u"].ex * gq - a_us * bq_us * m_us, tag="mc.colcheck")

    cir.rounds = (list(ext.rounds) if ext is not None else []) + [round_a, round_b, round_c]
    return ctx


# ============================================================================ prove / verify
def statement(prm, model_id, cl_dig, xi, zeta):
    return {"v": VERSION, "params": asdict(prm), "model_id": [int(x) for x in model_id],
            "cl_digest": [int(x) for x in cl_dig], "xi": [list(_e(x)) for x in xi], "zeta": [list(_e(z)) for z in zeta]}


def cells(cir):
    """Committed cells by space (E-valued vectors count one cell each, as packed in the commitment groups)."""
    out = {}
    for v in cir.vecs:
        sp = v.name.split(".")[0]
        out[sp] = out.get(sp, 0) + v.N
    return out


def prove(prm, stmt, wit, B=CPU):
    """Returns (proof dict, info).  wit: store (column / path / salts / coeffs / n_used), t, c (J slots), salt_cl,
    pre = dict(s0, c_w, c_raw, c_cfg)."""
    info = {}
    t0 = time.time()
    ctx = build(prm, stmt, wit, B, info)
    pf = Z.prove(ctx.cir, stmt, B, label=LABEL)
    info["prove_s"] = round(time.time() - t0, 3)
    info["cells_by_space"] = cells(ctx.cir)
    info["cells"] = int(sum(info["cells_by_space"].values()))
    info["vectors"] = len(ctx.cir.vecs)
    info["zk_budget"] = dict(Z.ZK_REPORT)
    info["violations"] = list(ctx.notes)
    return pf, info


def verify(prm, stmt, pf):
    ctx = build(prm, stmt, None)
    return Z.verify(ctx.cir, stmt, pf, label=LABEL)


def try_verify(prm, stmt, pf):
    try:
        verify(prm, stmt, pf)
        return "accepted"
    except Z.ProofError as e:
        return "rejected: " + str(e)


def shape_sig(x, _q=False):
    """Structure of a proof (types and lengths, no values): equal for any two witnesses of one profile.  The number of
    distinct Basefold query positions (lists under pf["open"]) depends on collisions of the public Fiat-Shamir query
    draw only, so list lengths inside openings are not compared."""
    if isinstance(x, dict):
        return {k: shape_sig(v, _q or k == "open") for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        head = "L" if _q else "L%d" % len(x)
        return [head] + [shape_sig(v, _q) for v in x[:4]] + ([shape_sig(x[-1], _q)] if len(x) > 4 else [])
    if isinstance(x, np.ndarray):
        return "A" + str(x.shape)
    if isinstance(x, (bytes, bytearray)):
        return "B%d" % len(x)
    return type(x).__name__


# ============================================================================ in-memory commitment (tests)
class DevStore:
    """Commitment of an (n_used, n_c) integer matrix with the modelcommit construction, kept in memory (codeword via
    B.ntt; prover-side accessors column / path / salts / coeffs / row)."""

    def __init__(self, prm, W, B=CPU):
        self.prm = prm
        W64 = np.asarray(W, dtype=np.int64)
        self.coeffs = W64.astype(np.int32)
        self.wide = {int(r): W64[int(r)] for r in np.flatnonzero((self.coeffs.astype(np.int64) != W64).any(axis=1))}
        self.n_used = self.coeffs.shape[0]
        com = MC.commit_matrix(MC.MatrixSource(W64), self.n_used, prm, B, log=lambda *a: None)
        self.levels, self.salts, self.root = com["levels"], com["salts"], com["root"]
        self.M = np.zeros((prm.R_MAX, prm.n), dtype=np.uint64)
        Wf = CPU.fromi(W64)
        for r in range(self.n_used):
            self.M[r] = B.tonumpy(B.ntt(B.asarray(Wf[r]), prm.n)) if MC._is_gpu(B) else CPU.ntt(Wf[r], prm.n)

    def row(self, i):
        if i in self.wide:
            return np.asarray(self.wide[i], dtype=np.int64)
        return np.asarray(self.coeffs[i], dtype=np.int64) if i < self.n_used else np.zeros(self.prm.nc, dtype=np.int64)

    def column(self, j):
        return self.M[:, j].copy()

    def path(self, j):
        out = []
        for d in range(self.prm.depth):
            out.append(tuple(int(v) for v in self.levels[d][:, j ^ 1])); j >>= 1
        return out


class ForgedStore:
    """Cheating prover's view: one row's coefficients replaced (it computes u with them) and/or one opened column entry
    bit-flipped; Merkle data is the honest commitment's."""

    def __init__(self, base, row=None, new=None, flip=None, rows=None):
        self.base, self.flip = base, flip
        self.rows = dict(rows or {})                              # several replaced rows: {row: coefficients}
        if row is not None:
            self.rows[row] = new
        self.prm, self.n_used, self.salts, self.root = base.prm, base.n_used, base.salts, base.root
        self.wide = dict(getattr(base, "wide", {}))
        for r_idx, r_val in self.rows.items():
            rv64 = np.asarray(r_val, dtype=np.int64)
            if (rv64.astype(np.int32).astype(np.int64) != rv64).any():
                self.wide[int(r_idx)] = rv64
            else:
                self.wide.pop(int(r_idx), None)
        self._flipped = False
        outer = self

        class _C:
            def __getitem__(s, t):
                return np.asarray(outer.rows[t], dtype=np.int32) if t in outer.rows else base.coeffs[t]
        self.coeffs = _C()

    def row(self, i):
        if i in self.rows:
            return np.asarray(self.rows[i], dtype=np.int64)
        if hasattr(self.base, "row"):
            return self.base.row(i)
        return np.asarray(self.base.coeffs[i], dtype=np.int64)

    def column(self, j):
        c = np.asarray(self.base.column(j), dtype=np.uint64).copy()
        if self.flip is not None and not self._flipped:          # first opened column, entry `flip`, bit 0
            c[self.flip] ^= np.uint64(1)
            self._flipped = True
        return c

    def path(self, j):
        return self.base.path(j)


def model_preimage(c_w, c_raw=None, c_cfg=None):
    s0 = tuple(int(x) for x in rand_field(4))
    c_raw = tuple(int(x) for x in (rand_field(4) if c_raw is None else c_raw))
    c_cfg = tuple(int(x) for x in (rand_field(4) if c_cfg is None else c_cfg))
    pre = {"s0": s0, "c_w": tuple(int(x) for x in c_w), "c_raw": c_raw, "c_cfg": c_cfg}
    return pre, MC.model_id(s0, pre["c_w"], c_raw, c_cfg)


def point(prm, seed):
    tr = Transcript(b"mczk-point")
    tr.absorb(seed)
    return tr.exts(prm.ob), tr.exts(prm.ib)


def make_case(prm, store, tiles, pre, mid, xi, zeta, rows_override=None, vol_elems=None, cfg_rows=False,
              meta_cfg0=2, meta_cfg1=3, J=None):
    """Honest (or overridden-row) claims for `tiles` (repeats allowed) at (xi, zeta); J claim slots (default R_MAX:
    no statement key); returns (stmt, wit)."""
    vol_blocks = (len(vol_elems) // 8) if vol_elems is not None else 0
    sh = Shape(prm, vol_blocks=vol_blocks, J=J)
    rows_override = rows_override or {}
    row_get = lambda t: rows_override[t] if t in rows_override else store.row(t)
    cv = [tile_claim(row_get(t), xi, zeta) for t in tiles]
    ts, cs = pad_claims(sh, tiles, cv)
    if cfg_rows:
        assert len(tiles) <= sh.J - 2, "need two slots for META_CFG0 and META_CFG1"
        ts[sh.J - 2] = meta_cfg0
        ts[sh.J - 1] = meta_cfg1
        cs[sh.J - 2] = tile_claim(row_get(meta_cfg0), xi, zeta)
        cs[sh.J - 1] = tile_claim(row_get(meta_cfg1), xi, zeta)
    salt_cl = tuple(int(x) for x in rand_field(4))
    dig = cl_digest(sh, ts, cs, salt_cl)
    stmt = statement(prm, mid, dig, xi, zeta)
    if vol_blocks > 0:
        stmt["vol_blocks"] = int(vol_blocks)
    if cfg_rows:
        stmt["cfg_rows"] = True
        stmt["meta_cfg0"] = int(meta_cfg0)
        stmt["meta_cfg1"] = int(meta_cfg1)
    if J is not None and int(J) != prm.R_MAX:
        stmt["claim_slots"] = int(J)
    wit = {"store": store, "t": ts, "c": cs, "salt_cl": salt_cl, "pre": pre}
    if vol_elems is not None:
        wit["vol_elems"] = np.asarray(vol_elems, dtype=np.uint64)
    return stmt, wit


# ============================================================================ full Qwen3-0.6B driver
def _rss_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def _gpu_claims(prm, store, tiles, xi, zeta, B):
    """c_t = W~_t(xi, zeta) for many tiles on the device: (W_t eq(zeta)) . eq(xi)."""
    ez = CPU.eq_table(list(zeta)); ex = CPU.eq_table(list(xi))
    out = []
    for s in range(0, len(tiles), 64):
        ch = tiles[s:s + 64]
        Wh = np.stack([np.asarray(store.coeffs[t], dtype=np.int32) for t in ch]).reshape(len(ch), prm.TO, prm.TI)
        H = np.broadcast_to(ez[:, None, :], (3, len(ch), prm.TI)).copy()
        A = B.tonumpy(MC.tiles_matvec(B, B.xp.asarray(Wh), list(range(len(ch))), B.asarray(H)))   # (3, nt, TO)
        for i in range(len(ch)):
            out.append(CPU.edot_dev(A[:, i], ex))
    return out


def cmd_qwen(a):
    from .field import gpu_backend
    import cupy as cp
    B = gpu_backend()
    prm = MC.QWEN_PARAMS
    sh = Shape(prm)
    ck = MC.Checkpoint(a.model)
    L = MC.Layout(MC.model_shapes(ck)[0], prm)
    store = MC.Store(a.store, prm, "W")
    meta = json.load(open(os.path.join(a.store, "commit.json")))
    sec = meta["secret"]
    pre = {"s0": tuple(sec["s0"]), "c_w": tuple(store.root), "c_raw": tuple(sec["c_raw"]), "c_cfg": tuple(sec["c_cfg"])}
    mid = tuple(meta["result"]["model_id"])
    assert MC.model_id(pre["s0"], pre["c_w"], pre["c_raw"], pre["c_cfg"]) == mid
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    claims, blocks = MC.build_claims(L, store, prm, a.k1, a.k2, rng, B)
    tiles = sorted({t for cl in claims for t in (cl.tiles if isinstance(cl, MC.GemmClaim) else [cl.tile])})
    del claims
    xi, zeta = point(prm, list(mid) + [a.seed])
    t1 = time.time()
    cv = _gpu_claims(prm, store, tiles, xi, zeta, B)
    t_claims = time.time() - t1
    ts, cs = pad_claims(sh, tiles, cv)
    salt_cl = tuple(int(x) for x in rand_field(4))
    stmt = statement(prm, mid, cl_digest(sh, ts, cs, salt_cl), xi, zeta)
    wit = {"store": store, "t": ts, "c": cs, "salt_cl": salt_cl, "pre": pre}
    out = {"model": os.path.basename(os.path.normpath(a.model)), "machine": a.machine or describe_machine(), "version": VERSION,
           "k1": a.k1, "k2": a.k2, "params": asdict(prm), "touched_tiles": len(tiles), "claim_slots": sh.J,
           "witness_select_s": round(t1 - t0, 2), "claims_eval_s": round(t_claims, 2),
           "layout": {k: getattr(sh, k) for k in ("IB", "QB", "KB", "A", "CB", "PSB", "MID", "X0", "NB", "CLI")}}
    runs = []
    for rep in range(a.reps):
        cp.get_default_memory_pool().free_all_blocks()
        with Z.PoolPeak() as pk:
            pf, info = prove(prm, stmt, wit, B)
        info["gpu_pool_peak_gib"] = round(pk.peak / (1 << 30), 3)
        info["gpu_pool_peak_used_gib"] = round(pk.peak_used / (1 << 30), 3)
        blob = dumps(pf)
        info["proof_bytes"] = len(blob)
        t1 = time.time()
        res = try_verify(prm, stmt, pf)
        info["verify_cpu_s"] = round(time.time() - t1, 3)
        info["verify"] = res
        info["rep"] = rep
        runs.append(info)
        print(json.dumps({k: v for k, v in info.items() if k != "cells_by_space"}), flush=True)
        del pf
    out["runs"] = runs
    out["cells"] = runs[0]["cells"]
    out["cells_by_space"] = runs[0]["cells_by_space"]
    hco = {k: v for k, v in runs[0]["cells_by_space"].items() if k != "mcCL" and not k.startswith("libra")}
    out["cells_hco_no_anchor_no_masks"] = int(sum(hco.values()))
    if a.forge:
        t = tiles[len(tiles) // 2]
        forged = MC.requant(np.asarray(store.coeffs[t]))
        fs = ForgedStore(store, row=t, new=forged)
        stmt_f, wit_f = make_case(prm, fs, tiles, pre, mid, xi, zeta, rows_override={t: forged})
        Z.FORCE = True
        try:
            pf_f, info_f = prove(prm, stmt_f, wit_f, B)
        finally:
            Z.FORCE = False
        out["forgery_int4"] = {"tile": int(t), "changed_entries": int((forged != np.asarray(store.coeffs[t])).sum()),
                               "prover_violations": info_f["violations"], "result": try_verify(prm, stmt_f, pf_f)}
        print(json.dumps(out["forgery_int4"]), flush=True)
    out["peak_rss_gib"] = round(_rss_gib(), 2)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("runs",)}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qwen")
    q.add_argument("--model", required=True); q.add_argument("--store", required=True)
    q.add_argument("--k1", type=int, default=210); q.add_argument("--k2", type=int, default=105)
    q.add_argument("--seed", type=int, default=1); q.add_argument("--reps", type=int, default=2)
    q.add_argument("--forge", action="store_true"); q.add_argument("--out", default=None)
    q.add_argument("--machine", default=None, help="host description (default: detected)")
    a = ap.parse_args(argv)
    {"qwen": cmd_qwen}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
