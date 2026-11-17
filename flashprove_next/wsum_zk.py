# SPDX-License-Identifier: Apache-2.0
"""Weight-side reduction of specs/09 §3.1 in ZERO KNOWLEDGE, chained: certified GEMM gadgets -> shared-point tile reduction -> hidden
column opening -> model_id, all in ONE flashprove_next.zk_flat proof with the weights hidden.

Every GEMM registered by a ``certified.TiledWeights`` provider contributes two products on committed activation
columns (phase 1): Y = W^cd X and A = W^r Z (W^r = |W^cd| + kappa_n [inexact], the certified radius matrix).  With
T = T_O = T_I the tile side and o = (a, o'), i = (b, i') the tile coordinates, the reduction is (FS order):

  phase 1  : gadget columns (X, Z, Y, A, ...) and the hidden slot -> tile-row table t (J claim slots, a public PROFILE
             constant independent of R_MAX: statement ``claim_slots``, default J = R_MAX; two halves: slot j < J/2 holds
             a W^cd tile, slot j + J/2 its W^r tile; t_{j+J/2} = t_j + R_MAX/2 is enforced, which with the LogUp range
             t < R_MAX of mc_zk (its table is exactly the R_MAX rows) also forces t_j < R_MAX/2: a W^r slot can only open
             the radius tile of ITS W^cd tile, and no slot can mix the two halves.  The offset is a profile constant:
             nothing is revealed).  Slots MAY repeat rows: M_C sampled slots of the same layer open the same tile row t
             from different slots; mc_zk's gamma-weighted LogUp aggregates them into alpha_t, while every slot keeps its
             own gamma^j in the claim check (soundness: see the mc_zk docstring; error terms grow linearly in J only).
  xi       : the shared row point of the tile cube (T variables).
  round W1 : folded rows w_{g,a}^h(b, i') = sum_{o'} eq(xi, o') W^h_{tile(g, a, b)}(o', i')  (E-valued, one committed
             vector on (h, slot, i')); per GEMM g and h in {cd, r} one zero-sum on the (a, b, i') cube:
                 for every a:  sum_{b, i'} w_{g,a}^h(b, i') X^h(b, i') = sum_{o'} eq(xi, o') Y^h(a, o')
             (X^cd = X, Y^cd = Y; X^r = Z, Y^r = A).  If Y != W X on tile row a, the left side computed with the TRUE
             folded row differs from the right side except with probability T/|E| over xi, so a cheating prover must
             commit a wrong folded row.
  zeta     : the shared column point.
  round W2 : hidden claims c_j = sum_{i'} eq(zeta, i') w_j(i'), committed (E-valued, J slots, unused slots forced 0;
             only the 2^(sb+1) planned slots carry folded rows, so J costs O(J) claim cells, not O(J T));
             a wrong folded row gives a wrong c_j = W~_{t_j}(xi, zeta) except with probability T/|E| over zeta.
  then the mc_zk rounds (gamma -> u = alpha^T W^cd, alpha by weighted LogUp over t; Q, lambda -> hidden columns,
             Poseidon2 Merkle paths to the hidden C_W, in-circuit model_id; beta -> batched column check), consuming
             (t, c, xi, zeta) instead of the standalone digest-pinned claim list.

Every value except the public statement (profile, model_id, block kinds and public shapes) and the challenges is a
T1-blinded committed vector: tile indices, folded rows, claims, u, alpha, columns, salts, siblings, C_W.  The circuit
depends on the public profile and the public gadget shapes only (the slot plan is a function of the shapes).

Hidden per-slot quantities of the gadget circuit (zero knowledge: layer count and positions must not leak; the verifier
builds the identical circuit without knowing them).  Under ``CfgTiledWeights`` + ``cfg_rows`` + ``vec_pad`` they are
ALL hidden committed columns, none is a public constant chosen by the caller:
  * the tensor exponent G_ten, per-tile exponents / kappa / s_align: Tt LogUp (specs/09 §3.2);
  * the RoPE rows: hidden position scalar ``pos`` and hidden committed rows, LogUp into one public table with rows
    (pos, j, cos/sin on the 2^-30 grid, abs + 1) for pos < T_pad (``certified.rope_hidden``);
  * the q/k score exponents: hidden scalars constrained EXACTLY to ``qk_exponent`` of the hidden gain vector
    (``certified.qk_exp_hidden``);
  * the norm gain vectors (q_norm, k_norm, input / post-attention layernorm): bound to ``model_id``.  A ``VecExt``
    commits SCALE (R_MAX entries) and the vector row prefix (``vec_pad`` entries) as hidden columns and binds each to
    its HCO claim slot at the shared point (xi, zeta) -- the same eq-sumzero as the Tt / Vt / Lt tables of CfgExt, two
    reserved slots next to the two cfg slots.  Every gain read (``CfgTiledWeights.vec_dec``) then
        Vt  :  (vec_key(role, lp1), vec_row, off, len) is a row of the committed Vt table (hidden offset),
        SCALE: (vec_row, G_v) is a row of the committed SCALE column (the row exponent G of the vec row),
        cd  :  v_j = floor(sigma_j 2^(E_j + G_v)) (exactly ``cd_decode``) is the committed integer at index off + j of the
               vec-row prefix column (LogUp), padded entries are zero codes.
    A forged gain vector (or a forged G_v: g' = 2 g with G_v' = G_v - 1 gives the same cd integers) leaves SCALE / the
    prefix column.  Profile constraints: ONE vec row (all gain vectors in one row, ``Layout.n_vec_rows == 1``) and the
    committed prefix length ``vec_pad`` >= ``Layout.vec_len`` is a profile constant; the cd image is the committed object,
    so two raw gain codes with the same floor(sigma 2^(E + G_v)) (below the 2^-G_v grid of the vec row) are not
    distinguished -- exact for every gain >= 2^-G_v, and the same precision the GEMM weights are committed at.
Scope: the one-time setup proof pi_setup (proximity, CD and W^r = |W^cd| + kappa m consistency, Lt / Vt derived
values) is flashprove_next.mc_setup; the Fiat-Shamir slot sampling and the hidden layer / position wiring are done by
the reply circuit (flashprove_next.reply_zk), which composes this module.
"""
import time
from dataclasses import asdict

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from . import certified as CZ
from . import mc_zk as MZ
from . import modelcommit as MC
from .zk_flat import Ex, EqP
from .field import CPU, P, ZERO

VERSION = "fpn-wsum-1"
LABEL = b"fpn-wsum-v1"


class HiddenG:
    """Symbolic wrapper around a hidden scalar CZ.X exponent column so certified.gemm (`-Gx`) and certified.blk_gemm_res
    (`Gx - 51`) transparently produce linear expressions via EmHG.lin without changes to flashprove_next.certified."""

    def __init__(self, x, sign=1, offset=0):
        self.x, self.sign, self.offset = x, int(sign), int(offset)

    def __neg__(self):
        return HiddenG(self.x, -self.sign, -self.offset)

    def __sub__(self, other):
        return HiddenG(self.x, self.sign, self.offset - int(other))

    def __add__(self, other):
        return HiddenG(self.x, self.sign, self.offset + int(other))

    def __int__(self):
        return int(self.x.v[0]) * self.sign + self.offset


class EmHG(CZ.Em):
    """CZ.Em subclass accepting HiddenG in the constant slot of `lin()` so G_ten_k is a hidden committed X."""

    def lin(self, terms, const=0, sp=None, bits=None):
        if isinstance(const, HiddenG):
            return super().lin(list(terms) + [(const.sign, const.x)], const.offset, sp=sp, bits=bits)
        return super().lin(terms, const, sp=sp, bits=bits)


class VecExt:
    """Binding of the hidden gain vectors (and of their row exponent) to model_id: specs/09 §3.2 'row reads' as hidden committed
    columns bound to HCO claim slots, exactly like the Tt / Vt / Lt columns of mc_zk.CfgExt (zero extra rounds):
      * SCALE column (R_MAX entries: G of every row, kappa of every row) bound to the claim of row META_SCALE;
      * vec-row prefix column (``vec_pad`` entries, profile constant >= Layout.vec_len) bound to the claim of the vec row;
      * wire tables ws.scale = {(i, SCALE[i])}, ws.vec = {(p, VEC[p])}, and ws.vt = the committed Vt table (cfg_ext).
    The slots (t = META_SCALE, t = vec_row) are reserved by ``WSum`` next to the two cfg slots."""

    def __init__(self, ctx, prm, store, vec_row, vec_pad, cfg_ext):
        assert cfg_ext is not None and cfg_ext.cfg_rows, "vector binding needs the cfg rows (Vt)"
        self.ctx, self.prm, self.pv = ctx, prm, ctx.prove
        self.vec_row, self.vec_pad = int(vec_row), int(vec_pad)
        self.cfg_ext = cfg_ext
        self.IB = prm.R_MAX.bit_length() - 1
        self.VB = _lg(self.vec_pad)
        assert (1 << self.VB) == self.vec_pad and self.vec_pad <= prm.nc and prm.R_MAX <= prm.nc
        self.SS = ctx.space("wsSCALE", [("idx", self.IB)])
        self.VV = ctx.space("wsVEC", [("idx", self.VB)])
        self.rows = {}
        sv = vv = None
        if self.pv:
            sr = np.asarray(store.row(MC.META_SCALE), dtype=np.int64)
            vr = np.asarray(store.row(self.vec_row), dtype=np.int64)
            assert not sr[prm.R_MAX:].any(), "SCALE row has entries beyond R_MAX"
            assert not vr[self.vec_pad:].any(), ("vec row has entries beyond vec_pad", self.vec_pad)
            sv, vv = sr[:prm.R_MAX], vr[:self.vec_pad]
            self.scale_v, self.vec_v = sv, vv
            self.rows = {MC.META_SCALE: sr, self.vec_row: vr}
        self.scale_col = self.SS.col("scale", sv)
        self.vec_col = self.VV.col("vec", vv)
        self._declared = False

    def _declare(self):
        """lookup tables over the committed columns, declared with the FIRST gain read (a table without any query is an
        error of zk_flat; the block list is public, so prover and verifier declare the same tables)."""
        if self._declared:
            return
        self._declared = True
        ctx, c = self.ctx, self.cfg_ext
        key_s = np.arange(self.SS.n, dtype=np.int64) if self.pv else None
        key_v = np.arange(self.VV.n, dtype=np.int64) if self.pv else None
        G.wire_table(ctx, "ws.scale", self.SS, [self.SS.idx("idx")], [self.scale_col], [key_s] if self.pv else None)
        G.wire_table(ctx, "ws.vec", self.VV, [self.VV.idx("idx")], [self.vec_col], [key_v] if self.pv else None)
        ctx.cir.lookup_table("ws.vt", c.VTS.nv, [x.ex for x in c.vt_cols], c.NV,
                             vals=c.vt_vals if self.pv else None)

    # ---------------------------------------------------------------- prover claims (slot t = META_SCALE / vec_row)
    def claims(self, xi, zeta):
        return (MZ.tile_claim(self.rows[MC.META_SCALE], xi, zeta), MZ.tile_claim(self.rows[self.vec_row], xi, zeta))

    def bind_claims(self, xi, zeta, c_scale_on, c_vec_on):
        """hidden SCALE / prefix columns = the committed rows: sum_i eq((xi, zeta), i) col[i] = claim (eq-sumzero over
        the column cube; the remaining high bits of the flat index are zero)."""
        from .field import ONE, hsub, hmul
        pt = [MZ._e(z) for z in zeta] + [MZ._e(x) for x in xi]

        def zero_tail(nb):
            acc = ONE
            for z in pt[nb:]:
                acc = hmul(acc, hsub(ONE, z))
            return acc
        for sp, col, nb, con, tag in ((self.SS, self.scale_col, self.IB, c_scale_on, "ws.scale.bind"),
                                      (self.VV, self.vec_col, self.VB, c_vec_on, "ws.vec.bind")):
            eq = Ex.atom(EqP(sp.vars("idx"), pt[:nb]))
            sp.sumzero(eq * col.ex * Ex.const(zero_tail(nb)) - sp.ind(idx=0) * con(sp), tag=tag)

    # ---------------------------------------------------------------- one gain read
    def read(self, em, prov, sp, name, dg, tag):
        """Constrain the hidden decoded gain vector ``dg`` (on sp, ``dg.code`` committed) to the committed vec-row segment
        at the hidden offset of its Vt entry, and its row exponent G_v to SCALE[vec_row].  Supports both single-slot and
        s-batched spaces (em.bbits >= 0)."""
        self._declare()
        ctx = self.ctx
        va = np.asarray(prov.vecs[name])
        ln = int(va.shape[-1])
        role = prov.vec_roles.get(name, MC.ROLE_IDS.get(name))
        assert role is not None and ln <= sp.in_n, ("vector role unknown", name)
        SC = prov.gsc
        n_sc = SC.n
        lp1 = prov.layer_p1
        lp1_ex = em.view(lp1, SC).ex if isinstance(lp1, CZ.X) else Ex.const(int(lp1))
        if isinstance(lp1, CZ.X):
            em.rng(lp1, 10, tag + ".lp1")      # vec_key = (role << 18) | (lp1 << 8) is injective only for lp1 < 2^10
        key_ex = Ex.const(role << 18) + lp1_ex * Ex.const(1 << 8)
        off_v = g_v = key_v = None
        if em.pv:
            lp1_arr = em.view(lp1, SC).v if isinstance(lp1, CZ.X) else np.full(n_sc, int(lp1), dtype=np.int64)
            key_v = np.zeros(n_sc, dtype=np.int64)
            off_v = np.zeros(n_sc, dtype=np.int64)
            vt = self.cfg_ext.vt_vals
            bad_vt = 0
            for ul in np.unique(lp1_arr):
                kv = MC.vec_key(role, int(ul), 0)
                hit = np.flatnonzero(vt[0] == kv)
                msk = (lp1_arr == ul)
                key_v[msk] = kv
                if len(hit):
                    off_v[msk] = int(vt[2][hit[0]])
                    if int(vt[1][hit[0]]) != self.vec_row or int(vt[3][hit[0]]) != ln:
                        bad_vt += int(msk.sum())
                else:
                    bad_vt += int(msk.sum())
            g_sc = int(self.scale_v[self.vec_row]) + int(prov.vec_g_shift.get(name, 0))
            g_v = np.full(n_sc, g_sc, dtype=np.int64)
            em.fail("vec:vt", bad_vt)
            em.fail("vec:scale", int(g_sc != int(self.scale_v[self.vec_row])))
        full = lambda val: None if not em.pv else np.full(n_sc, int(val), dtype=np.int64)
        off_x = em.col(SC, tag + ".off", off_v, 20)
        g_x = em.col(SC, tag + ".Gv", g_v, 7)
        G.wire_lookup(ctx, "ws.vt", SC.S, [key_ex, Ex.const(self.vec_row), off_x.ex, Ex.const(ln)],
                      [key_v, full(self.vec_row), off_v, full(ln)] if em.pv else None)
        G.wire_lookup(ctx, "ws.scale", SC.S, [Ex.const(self.vec_row), g_x.ex],
                      [full(self.vec_row), g_v] if em.pv else None)
        # cd integer of every gain: v = floor(sigma 2^(E + G_v)),  sigma = (1 - 2 s) a   (= modelcommit.cd_decode)
        sig = em.col(sp, tag + ".sig", (dg.a.v * (1 - 2 * dg.s.v)) if em.pv else None, 9)
        em.zero(sp, lambda: sig.ex - dg.a.ex + Ex.const(2) * dg.s.ex * dg.a.ex, tag + ".sig")
        k = em.lin([(1, dg.E), (1, g_x)], bits=10, sp=sp)
        kk = CZ.clamp(em, k, -53, 40, tag + ".k")
        A, B, f = CZ.pow2(em, kk, tag + ".p")
        A.bits = 41
        v = CZ.shr_hid(em, sig, A, B, 53, tag + ".v", 23, f=f)           # |v| < 2^22 (honest), exact floor
        # intra-slot flat index (bits 0 .. sp.in_bits - 1); padded entries (>= ln) are zero codes
        bnames = {ax for ax, _ in em.baxes}
        flat = Ex.const(0)
        for ax, _ in sp.axes:
            if ax not in bnames:
                flat = flat + sp.S.idx(ax, 1 << sp.S.vars(ax)[0])
        inln = Ex.atom(Z.SelLt(list(range(sp.in_bits)), ln))
        if ln < sp.in_n:
            em.zero(sp, lambda: (Ex.const(1) - inln) * dg.code.ex, tag + ".pad")
        key = em.view(off_x, sp).ex + flat
        kvals = sel_v = None
        if em.pv:
            K_ = 1 << em.bbits
            in_n = sp.in_n
            flat_v = np.tile(np.arange(in_n, dtype=np.int64), K_)
            off_sp = em.view(off_x, sp).v
            kvals = off_sp + flat_v
            sel_v = (flat_v < ln).astype(np.int64)
            vv2 = v.v.reshape(K_, in_n)[:, :ln]
            seg = off_sp.reshape(K_, in_n)[:, :ln] + np.arange(ln, dtype=np.int64)[None, :]
            ok = (seg < self.vec_pad) & (self.vec_v[np.minimum(seg, self.vec_pad - 1)] == vv2)
            em.fail("vec:cd", int((~ok).sum()))
        G.wire_lookup(ctx, "ws.vec", sp.S, [key, v.ex], [kvals, v.v] if em.pv else None,
                      sel=inln, sel_vals=sel_v)


class CfgTiledWeights(CZ.TiledWeights):
    """TiledWeights subclass where every registered GEMM's tensor exponent G_ten_k is a hidden committed CZ.X (not a
    public statement integer), and its output accumulator bitwidth reserves `s_bits` bits for per-tile exponent
    alignment s_j = 2^(G_ten - G_j).

    Hidden path (zero knowledge): RoPE rows from the hidden scalar ``pos_x`` (``t_pad`` public profile constant, ``rope``
    the public rope function pos -> (cos, sin)), q/k score exponents derived in-circuit from the hidden gain vectors
    (``qk_exp``), gain vectors bound to model_id when the statement carries ``vec_pad`` (``VecExt``).  Prover-only cheat
    hooks (tests): ``rope_wit`` (rows committed for the hidden position), ``qk_wit_delta`` (offset on the committed
    exponent), ``vec_g_shift`` (offset on the committed row exponent of a gain read)."""

    def __init__(self, em, mats=None, vecs=None, tensors=None, role_ids=None, layer_p1=None, expert=0, s_bits=1,
                 vec_roles=None, qk_wit_delta=None, vec_g_shift=None, qk_window=None, **kw):
        super().__init__(mats, vecs, tensors=tensors, **kw)
        self.qk_window = qk_window   # public profile window [lo, hi] of eq + ek (Lt score-exponent window), ATT only
        self.em = em
        self.s_bits = int(s_bits)
        self.extra_bits = self.s_bits         # read by certified.gemm_mb
        self.role_ids = dict(role_ids or {})
        self.vec_roles = dict(vec_roles or {})
        self.layer_p1 = layer_p1     # int or CZ.X on scalar space
        self.expert = expert         # int or CZ.X on scalar space
        self.require_hidden_rope = True
        self.qk_wit_delta = dict(qk_wit_delta or {})
        self.vec_g_shift = dict(vec_g_shift or {})
        self.qk_x = {}
        self._vec_dec = {}
        SC = em.space("ws.gsc", [])
        self.gsc = SC
        self.g_cols = {}
        for name, M in list(self.mats.items()):
            gv = np.full(SC.n, int(M.G), dtype=np.int64) if em.pv else None
            gc = em.col(SC, f"G.{name}", gv, 6)
            self.g_cols[name] = gc
            self.mats[name] = CZ.Mat(M.W, HiddenG(gc), M.Wr, M.n_in)

    def matvec(self, em, name, Xv, Zv, OS, tag):
        M = self.mats[name]
        XS = Xv.sp
        on, inn = OS.n, XS.n
        nb = CZ.bitlen(inn - 1) if inn > 1 else 0
        # s_bits of per-row alignment headroom (s_j W_j); exact static bounds, never clamped: the gadgets choose the
        # input width with certified.gemm_mb(prov) so that A fits LIM - 1 (mul_up) -- a clamp would under-state |A|
        yb = 22 + self.s_bits + Xv.bits + nb
        ab = 23 + self.s_bits + Zv.bits + nb
        assert yb <= CZ.LIM and ab <= CZ.LIM - 1, (tag, yb, ab)
        Y = A = None
        if em.pv:
            o, i = M.W.shape
            Y = np.zeros(on, dtype=np.int64); A = np.zeros(on, dtype=np.int64)
            Y[:o] = M.W.astype(np.int64) @ Xv.v[:i]
            A[:o] = M.Wr.astype(np.int64) @ Zv.v[:i]
        Xc = em.commit(Xv, tag + ".Xc")
        Zc = em.commit(Zv, tag + ".Zc")
        Yx = em.col(OS, tag + ".Y", Y, yb)
        Ax = em.col(OS, tag + ".A", A, ab)
        tname = self.tensors.get(name, name)
        role_id = self.role_ids.get(name)
        if role_id is None and tname:
            role_id = MC.tensor_role_layer_expert(tname)[0]
        self.gemms.append(dict(name=name, tensor=tname, X=Xc, Z=Zc, Y=Yx, A=Ax, OS=OS, XS=XS,
                               mat=M, G_x=self.g_cols[name], role_id=role_id,
                               layer_p1=self.layer_p1, expert=self.expert, tag=tag))
        return Yx, Ax, M.G, M.n_in

    def vec_dec(self, em, sp, name, tag):
        """Hidden gain vector, decoded on sp.  With a ``VecExt`` (statement ``vec_pad``) the hidden codes are bound to the
        committed vec-row segment (Vt offset, SCALE row exponent, cd image); one decode per (vector, space) so the
        norm and the exponent derivation read the same committed columns."""
        key = (name, id(sp))
        if key not in self._vec_dec:
            dg = CZ.CommittedWeights.vec_dec(self, em, sp, name, tag)
            ve = getattr(em, "vec_ext", None)
            if em.emit:
                # fail closed: an unbound hidden gain vector would be accepted by the verifier of the same statement
                if ve is None:
                    raise ValueError("CfgTiledWeights needs the gain-vector binding (statement vec_row / vec_pad)")
                ve.read(em, self, sp, name, dg, tag)
            self._vec_dec[key] = dg
        return self._vec_dec[key]

    def rope_wit(self, pos):
        return self.rope(pos)

    def on_qk_gain(self, em, gname, dg, dh, tag):
        """score exponent of this gain vector, hidden and EXACT (certified.qk_exp_hidden)."""
        if gname not in self.qk_x:
            self.qk_x[gname] = CZ.qk_exp_hidden(em, dg, dh, tag + ".qe", delta=self.qk_wit_delta.get(gname, 0))

    def qk_exp(self, layer):
        """(eq, ek): hidden committed scalars of the layer's q_norm / k_norm gains, derived where the gains are decoded
        (blk_qkv); when only the attention block runs they are derived here from the same (cached) gain decode."""
        em = self.em
        for gname in ("q_norm", "k_norm"):
            if gname not in self.qk_x:
                dh = len(self.vecs[gname])
                GD = em.space("qk.gd." + gname, [("hf", 1), ("dj", _lg(dh) - 1)])
                self.on_qk_gain(em, gname, self.vec_dec(em, GD, gname, "qk." + gname), dh, "qk." + gname)
        return self.qk_x["q_norm"], self.qk_x["k_norm"]


class BatchedCfgProvider(CfgTiledWeights):
    """Multi-slot provider on a child ``em_b = em.sub_em([('s', gsb)])`` (``K = 1 << gsb`` slots):
      * ``mat_getter(name, s_idx)`` returns the cached ``CZ.Mat`` for slot ``s_idx`` (prover only);
      * ``n_ins[name]`` gives ``n_in`` per matrix name;
      * ``matvec`` groups slots sharing the same ``CZ.Mat`` to run one BLAS matrix-matrix product per distinct layer/tile
        instead of ``K`` matrix-vector products, and registers ONE ``s_bits_ax = gsb`` batched GEMM in ``WSum``."""

    def __init__(self, em_b, n_ins, role_ids=None, vecs=None, vec_roles=None, layer_p1=None, expert=0, s_bits=1,
                 mat_getter=None, tensors_fn=None, **kw):
        assert em_b.bbits > 0 and len(em_b.baxes) == 1 and em_b.baxes[0][0] == "s"
        self.gsb = em_b.bbits
        self.K = 1 << self.gsb
        self.mat_getter = mat_getter
        self.tensors_fn = tensors_fn
        mats_stub = {nm: CZ.Mat(None, 0, None, int(nin)) for nm, nin in n_ins.items()}
        super().__init__(em_b, mats=mats_stub, vecs=vecs or {}, role_ids=role_ids, vec_roles=vec_roles,
                         layer_p1=layer_p1, expert=expert, s_bits=s_bits, **kw)
        if em_b.pv and mat_getter is not None:
            for nm, nin in n_ins.items():
                gv = np.array([int(mat_getter(nm, s_).G) for s_ in range(self.K)], dtype=np.int64)
                self.g_cols[nm].col.setv(gv)

    def matvec(self, em, name, Xv, Zv, OS, tag):
        M = self.mats[name]
        XS = Xv.sp
        K = self.K
        on, inn = OS.in_n, XS.in_n
        nb = CZ.bitlen(inn - 1) if inn > 1 else 0
        yb = 22 + self.s_bits + Xv.bits + nb
        ab = 23 + self.s_bits + Zv.bits + nb
        assert yb <= CZ.LIM and ab <= CZ.LIM - 1, (tag, yb, ab)
        Y = A = None
        if em.pv:
            X2 = Xv.v.reshape(K, inn)
            Z2 = Zv.v.reshape(K, inn)
            Y2 = np.zeros((K, on), dtype=np.int64)
            A2 = np.zeros((K, on), dtype=np.int64)
            groups = {}
            for s_ in range(K):
                ms = self.mat_getter(name, s_)
                groups.setdefault(id(ms), (ms, []))[1].append(s_)
            for ms, idxs in groups.values():
                idx_a = np.asarray(idxs, dtype=np.int64)
                o, i = ms.W.shape
                Y2[idx_a, :o] = X2[idx_a, :i] @ ms.W.T.astype(np.int64)
                A2[idx_a, :o] = Z2[idx_a, :i] @ ms.Wr.T.astype(np.int64)
            Y = Y2.reshape(-1)
            A = A2.reshape(-1)
        Xc = em.commit(Xv, tag + ".Xc")
        Zc = em.commit(Zv, tag + ".Zc")
        Yx = em.col(OS, tag + ".Y", Y, yb)
        Ax = em.col(OS, tag + ".A", A, ab)
        role_id = self.role_ids[name]
        self.gemms.append(dict(name=name, tensor=self.tensors.get(name, name), tensors_fn=self.tensors_fn,
                               X=Xc, Z=Zc, Y=Yx, A=Ax, OS=OS, XS=XS, s_bits_ax=self.gsb,
                               mat=M, G_x=self.g_cols[name], role_id=role_id,
                               layer_p1=self.layer_p1, expert=self.expert, tag=tag))
        return Yx, Ax, M.G, M.n_in


def _lg(n):
    return max(0, int(n - 1).bit_length())


def _flatmap(axes, f):
    """axes MSB first; f(p) -> (target axis, bit) for flat bit p (LSB = 0); returns amap axis -> per-bit list."""
    amap, p = {}, sum(b for _, b in axes)
    for nm, b in axes:
        p -= b
        amap[nm] = [f(p + k) for k in range(b)]
    return amap


class Plan:
    """Slot blocks: GEMM g gets 2^(gsb + ab + bb) consecutive slots (aligned), gsb = batch slot bits (0 if unbatched),
    ab = output tile bits, bb = input tile bits of its PUBLIC gadget shapes.
    Slot base + (s << (ab + bb)) + (a << bb) + b holds tile (a, b) of batch item s."""

    def __init__(self, prm, gemms, cfg_rows=False, vec_slots=0, J=None):
        assert prm.ob == prm.ib, "square tiles (T_O = T_I) assumed by the folded-row layout"
        self.prm = prm
        self.J = prm.R_MAX if J is None else int(J)
        self.cfg_rows = bool(cfg_rows)
        self.vec_slots = int(vec_slots)                  # 0, or 2 (SCALE, vec row) reserved below the cfg slots
        H = self.J // 2
        self.gsb, self.ab, self.bb, self.tb = [], [], [], []
        for g in gemms:
            gsb = int(g.get("s_bits_ax", 0))
            ab, bb = g["OS"].bits - gsb - prm.ob, g["XS"].bits - gsb - prm.ib
            assert ab >= 0 and bb >= 0, ("GEMM smaller than one tile", g["tag"])
            self.gsb.append(gsb); self.ab.append(ab); self.bb.append(bb); self.tb.append(gsb + ab + bb)
        order = sorted(range(len(gemms)), key=lambda k: -self.tb[k])
        self.base = [0] * len(gemms)
        b0 = 0
        for k in order:                                  # descending power-of-two sizes: every base stays aligned
            self.base[k] = b0
            b0 += 1 << self.tb[k]
        self.used = b0
        self.sb = max(1, _lg(b0))
        if self.cfg_rows:
            # Reserve 2 slots for CFG0, CFG1 (H - 2, H - 1) and, with a VecExt, 2 more below them (SCALE, vec row):
            # no GEMM slot there, and ws.claims skips them (their claims are bound by the cfg / vec tables instead)
            assert b0 <= self.res0 and (1 << self.sb) <= H, ("claim slots exceed the reserved region", b0, self.res0, H)
        else:
            assert (1 << self.sb) <= H, ("claim slots exceed the profile's J/2", b0, H)

    @property
    def res0(self):
        """first reserved slot index (lower half): GEMM slots live below it."""
        return self.J // 2 - 2 - self.vec_slots

    def jmap(self, k):
        gsb, ab, bb, tb, base = self.gsb[k], self.ab[k], self.bb[k], self.tb[k], self.base[k]
        return ([("b", x) for x in range(bb)] + [("a", x) for x in range(ab)] + [("s", x) for x in range(gsb)]
                + [("c", (base >> x) & 1) for x in range(tb, self.sb)])


class WSum:
    """Shared-point tile reduction over all GEMMs registered by `provs` (certified.TiledWeights / CfgTiledWeights), as
    two zk_flat commit rounds; also the claim source (`ext`) of mc_zk.build.  When `cfg_ext` (MZ.CfgExt with
    `cfg_rows=True`) is supplied, every GEMM slot (k, a, b) is bound to Tt via 6-column LogUp
    `(key, t_j, G_j, kappa_j, s_align_j, G_ten_k)`, folded-row zero-sums multiply by `s_align_j = 2^(G_ten - G_j)`, and
    two HCO claim slots bind Tt, Vt, Lt to META_CFG0, META_CFG1.
    J: claim slots (profile constant, power of two, default R_MAX); HJ = J / 2 slots per half, HR = R_MAX / 2 the row
    offset of the radius tiles (and HR - 1 / R_MAX - 1 the zero rows of the padding slots)."""

    def __init__(self, em, prm, provs, tile_of=None, B=CPU, info=None, cfg_ext=None, store=None, vec_ext=None, J=None):
        self.em, self.prm, self.B = em, prm, B
        self.cfg_ext = cfg_ext
        self.store = store
        self.has_cfg = cfg_ext is not None and cfg_ext.cfg_rows
        self.vec_ext = vec_ext
        self.info = {} if info is None else info
        self.gemms = [g for p in provs for g in p.gemms]
        J = prm.R_MAX if J is None else int(J)
        assert J & (J - 1) == 0 and J >= 16, "claim slots J must be a power of two >= 16"
        self.J = J
        self.plan = Plan(prm, self.gemms, cfg_rows=self.has_cfg, vec_slots=2 if vec_ext is not None else 0, J=J)
        ctx, pv = em.ctx, em.pv
        self.IB = J.bit_length() - 1                      # claim-slot bits JB (name kept: the mc_zk ext interface)
        H = J // 2                                        # slots per half
        HR = prm.R_MAX // 2                               # row offset W^cd -> W^r
        self.HR = HR
        sb = self.plan.sb
        nslot = 1 << sb
        self._ts = self._cs = self._xi = self._zeta = self._wv = None
        self.s_al = None
        if pv:
            ts = [HR - 1] * H                             # padding slots: zero row HR - 1 (its partner R_MAX - 1 is zero)
            if self.has_cfg:
                # Build lookup map from Tt key -> (t_row, G_t, kappa_t, s_align, G_ten)
                tt = cfg_ext.tt_vals
                tmap = {int(tt[0][idx]): (int(tt[1][idx]), int(tt[2][idx]), int(tt[3][idx]),
                                          int(tt[4][idx]), int(tt[5][idx]))
                        for idx in range(cfg_ext.NT)}
                g_v = np.full(nslot, prm.G_LO, dtype=np.int64)
                kap_v = np.zeros(nslot, dtype=np.int64)
                sal_v = np.ones(nslot, dtype=np.int64)
                for k, g in enumerate(self.gemms):
                    gsb, ab_k, bb_k = self.plan.gsb[k], self.plan.ab[k], self.plan.bb[k]
                    pS, pA, pB = 1 << gsb, 1 << ab_k, 1 << bb_k
                    role = int(g["role_id"])
                    lp1 = g["layer_p1"]
                    if isinstance(lp1, CZ.X):
                        lp1_arr = lp1.v[:pS] if gsb else [int(lp1.v[0])]
                    else:
                        dfl = int(lp1 if lp1 is not None else MC.tensor_role_layer_expert(g["tensor"])[1])
                        lp1_arr = [dfl] * pS
                    exp = g["expert"]
                    if isinstance(exp, CZ.X):
                        exp_arr = exp.v[:pS] if gsb else [int(exp.v[0])]
                    else:
                        exp_arr = [int(exp)] * pS
                    tfn = g.get("tensors_fn")
                    for s_i in range(pS):
                        lp1_i, exp_i = int(lp1_arr[s_i]), int(exp_arr[s_i])
                        for a in range(pA):
                            for b in range(pB):
                                slot = self.plan.base[k] + (s_i << (ab_k + bb_k)) + (a << bb_k) + b
                                ky = MC.tile_key(role, lp1_i, exp_i, a, b)
                                if ky in tmap:
                                    trw, gt_j, kp_j, sa_j, gten_j = tmap[ky]
                                else:
                                    tname_s = tfn(g["name"], s_i) if tfn is not None else g["tensor"]
                                    trw = tile_of(tname_s, a, b) if tile_of else None
                                    trw = HR - 1 if trw is None else int(trw)
                                    gt_j, kp_j, sa_j, gten_j = prm.G_LO, 0, 1, prm.G_LO
                                ts[slot] = trw
                                g_v[slot], kap_v[slot], sal_v[slot] = gt_j, kp_j, sa_j
                ts[H - 2] = cfg_ext.meta_cfg0
                ts[H - 1] = cfg_ext.meta_cfg1
                if vec_ext is not None:
                    ts[self.plan.res0] = MC.META_SCALE
                    ts[self.plan.res0 + 1] = vec_ext.vec_row
                self._sal_v = sal_v
            else:
                for k, g in enumerate(self.gemms):
                    o, i = g["mat"].W.shape
                    nA, nB = -(-o // prm.TO), -(-i // prm.TI)
                    for a in range(nA):
                        for b in range(nB):
                            t = tile_of(g["tensor"], a, b)
                            assert t is not None and 0 <= t < HR - 1, (g["tensor"], a, b, t)
                            ts[self.plan.base[k] + (a << self.plan.bb[k]) + b] = t
            self._ts = ts + [t + HR for t in ts]
        TS = ctx.space("wsT", [("h", 1), ("j", self.IB - 1)])
        self.t = TS.col("t", np.asarray(self._ts, dtype=np.int64) if pv else None)
        PP = ctx.space("wsTP", [("j", self.IB - 1)])
        # t_{j + J/2} = t_j + R_MAX/2; with mc_zk's LogUp range t < R_MAX on BOTH halves this forces t_j < R_MAX/2
        PP.zero(MZ.view(self.t, PP, {"h": ("c", 1)}) - MZ.view(self.t, PP, {"h": ("c", 0)}) - Ex.const(HR), tag="ws.pair")
        if self.has_cfg:
            # Constrain the two config slots in lower half: t[H - 2] = meta_cfg0, t[H - 1] = meta_cfg1
            jm_c0 = [("c", ((H - 2) >> s) & 1) for s in range(self.IB - 1)]
            jm_c1 = [("c", ((H - 1) >> s) & 1) for s in range(self.IB - 1)]
            PP.zero(MZ.view(self.t, PP, {"h": ("c", 0), "j": jm_c0}) - Ex.const(cfg_ext.meta_cfg0),
                    sel=PP.ind(j=0), tag="ws.cfg0.t")
            PP.zero(MZ.view(self.t, PP, {"h": ("c", 0), "j": jm_c1}) - Ex.const(cfg_ext.meta_cfg1),
                    sel=PP.ind(j=0), tag="ws.cfg1.t")
            if vec_ext is not None:
                for k_, (sl_, row_, tg_) in enumerate(((self.plan.res0, MC.META_SCALE, "ws.scale.t"),
                                                       (self.plan.res0 + 1, vec_ext.vec_row, "ws.vec.t"))):
                    jm_ = [("c", (sl_ >> s_) & 1) for s_ in range(self.IB - 1)]
                    PP.zero(MZ.view(self.t, PP, {"h": ("c", 0), "j": jm_}) - Ex.const(row_), sel=PP.ind(j=0), tag=tg_)
            # Commit per-slot (G_j, kappa_j, s_align_j) and bind every GEMM's (a, b) slots to Tt via 6-col LogUp.
            # SS_b: wsSL always has more rows than GEMMs.  A one-tile GEMM fixes ALL SS_b bits of its wsSL reads, in
            # round_xi and in the ws.tt lookup alike, so the two claims would compete for the ONE random tail row with
            # those bits; force-copying wsSL into every lookup (fresh_ex) gives the lookup claim its own vector, also
            # when SS_b >= 6 (tail >= Col.COPY_BELOW, where fresh_ex would otherwise stop copying)
            SS_b = max(5, sb, len(self.gemms).bit_length())
            SN = 1 << SS_b
            SS = ctx.space("wsSL", [("j", SS_b)])
            gp_v = kp_v = sp_v = None
            if pv:
                gp_v = np.full(SN, prm.G_LO, dtype=np.int64); gp_v[:nslot] = g_v
                kp_v = np.zeros(SN, dtype=np.int64); kp_v[:nslot] = kap_v
                sp_v = np.ones(SN, dtype=np.int64); sp_v[:nslot] = sal_v
            self.g_sl = SS.col("g", gp_v)
            self.kap_sl = SS.col("kap", kp_v)
            self.s_al = SS.col("sal", sp_v)
            self.SS_b = SS_b
            for c_ in (self.g_sl, self.kap_sl, self.s_al):
                ctx.force_copy.add(c_.vec.id)
            # Register 6-column Tt lookup table: (key, t_row, G_t, kappa_t, s_align, G_ten)
            ttc = cfg_ext.tt_cols
            if self.gemms:                    # a table without a single query is an error of zk_flat (ATT-only proofs)
                ctx.cir.lookup_table("ws.tt", cfg_ext.TTS.nv,
                                     [ttc[0].ex, ttc[1].ex, ttc[2].ex, ttc[3].ex, ttc[4].ex, ttc[5].ex],
                                     cfg_ext.NT,
                                     vals=[cfg_ext.tt_vals[0], cfg_ext.tt_vals[1], cfg_ext.tt_vals[2],
                                           cfg_ext.tt_vals[3], cfg_ext.tt_vals[4], cfg_ext.tt_vals[5]] if pv else None)
            for k, g in enumerate(self.gemms):
                gsb, ab, bb = self.plan.gsb[k], self.plan.ab[k], self.plan.bb[k]
                q_axes = ([("s", gsb)] if gsb else []) + ([("a", ab)] if ab else []) + ([("b", bb)] if bb else [])
                QS = em.space(f"ws.q{k}", q_axes)
                QSp = QS.S
                jm_sb = self.plan.jmap(k)
                jm_ib = jm_sb + [("c", 0)] * (self.IB - 1 - sb)
                jm_ss = jm_sb + [("c", 0)] * (SS_b - sb)
                role = int(g["role_id"])
                lp1 = g["layer_p1"]
                lp1_ex = em.view(lp1, QS).ex if isinstance(lp1, CZ.X) else Ex.const(int(lp1 if lp1 is not None else 0))
                exp = g["expert"]
                exp_ex = em.view(exp, QS).ex if isinstance(exp, CZ.X) else Ex.const(int(exp))
                a_ex = QSp.idx("a") if ab else Ex.const(0)
                b_ex = QSp.idx("b") if bb else Ex.const(0)
                key_ex = (Ex.const(role << 35) + lp1_ex * Ex.const(1 << 26)
                          + exp_ex * Ex.const(1 << 16) + a_ex * Ex.const(1 << 8) + b_ex)
                t_q = MZ.view(self.t, QSp, {"h": ("c", 0), "j": jm_ib})
                g_q = MZ.view(self.g_sl, QSp, {"j": jm_ss})
                kap_q = MZ.view(self.kap_sl, QSp, {"j": jm_ss})
                sal_q = MZ.view(self.s_al, QSp, {"j": jm_ss})
                gten_ex = em.view(g["G_x"], QS).ex if isinstance(g.get("G_x"), CZ.X) else Ex.const(int(g["mat"].G))
                qvals = None
                if pv:
                    qn = QSp.n
                    lp1_v = em.view(lp1, QS).v if isinstance(lp1, CZ.X) else np.full(qn, int(lp1 if lp1 is not None else 0), dtype=np.int64)
                    exp_v = em.view(exp, QS).v if isinstance(exp, CZ.X) else np.full(qn, int(exp), dtype=np.int64)
                    ss_ = QS.grid("s") if gsb else np.zeros(qn, dtype=np.int64)
                    aa = QS.grid("a") if ab else np.zeros(qn, dtype=np.int64)
                    bb_ = QS.grid("b") if bb else np.zeros(qn, dtype=np.int64)
                    sl_idx = self.plan.base[k] + (ss_ << (ab + bb)) + (aa << bb) + bb_
                    ky_v = (np.int64(role) << 35) | (lp1_v << 26) | (exp_v << 16) | (aa << 8) | bb_
                    gt_v = em.view(g["G_x"], QS).v if isinstance(g.get("G_x"), CZ.X) else np.full(qn, int(g["mat"].G), dtype=np.int64)
                    qvals = [ky_v, np.asarray(self._ts, dtype=np.int64)[sl_idx], gp_v[sl_idx], kp_v[sl_idx], sp_v[sl_idx], gt_v]
                G.wire_lookup(ctx, "ws.tt", QSp, [key_ex, t_q, g_q, kap_q, sal_q, gten_ex], qvals)
        self.rounds = [self.round_xi, self.round_zeta]

    # ---------------------------------------------------------------- mc_zk claim-source interface
    def ts(self): return self._ts
    def cs(self): return self._cs
    def xi(self): return self._xi
    def zeta(self): return self._zeta

    def _slot_view(self, col, sp2, jm):
        return MZ.view(col, sp2, {"h": [jm[self.IB - 1]], "j": list(jm[:self.IB - 1])})

    def t_on(self, sp2, jm): return self._slot_view(self.t, sp2, jm)
    def c_on(self, sp2, jm): return self._slot_view(self.c, sp2, jm)

    # ---------------------------------------------------------------- prover: folded rows
    def _fold(self, xi):
        prm, B, sb = self.prm, self.B, self.plan.sb
        T = prm.TI
        wv = np.zeros((3, 2 << (sb + prm.ib)), dtype=np.uint64)
        ex = CPU.eq_table(list(xi))                                     # (3, T_O)
        if self.has_cfg:
            # Fold the actual unscaled tile rows W^h_{t_j} committed in `store`; slots repeat rows (M_C sampled slots of
            # one layer), so every DISTINCT row is folded once (chunks of 64 rows) and scattered to its slots
            nslot = 1 << sb
            HJ, HR = self.J // 2, prm.R_MAX // 2
            R0 = self.plan.res0
            rows = [(HR - 1 if (h == 0 and R0 <= j < HJ) else self._ts[h * HJ + j]) for h in (0, 1) for j in range(nslot)]
            uniq = sorted(set(rows))
            fold = {}
            for c0 in range(0, len(uniq), 64):
                ch = uniq[c0:c0 + 64]
                Wt = np.stack([np.asarray(self.store.row(r), dtype=np.int32).reshape(prm.TO, T) for r in ch])
                V = np.broadcast_to(ex[:, None, :], (3, len(ch), prm.TO)).copy()
                if MC._is_gpu(B):
                    out = B.tonumpy(MC.tiles_vecmat(B, B.xp.asarray(Wt), list(range(len(ch))), B.asarray(V)))
                else:
                    out = MC.tiles_vecmat(B, Wt, list(range(len(ch))), V)
                for i, r in enumerate(ch):
                    fold[r] = out[:, i]
            for hj, r in enumerate(rows):
                off = hj << prm.ib
                wv[:, off:off + T] = fold[r]
            if hasattr(self.store, "drop_cache"):
                self.store.drop_cache()
            return wv
        for k, g in enumerate(self.gemms):
            for h in (0, 1):
                W = g["mat"].W if h == 0 else g["mat"].Wr
                o, i = W.shape
                nA, nB = -(-o // prm.TO), -(-i // T)
                Wp = np.zeros((nA * prm.TO, nB * T), dtype=np.int32)
                Wp[:o, :i] = W
                Wt = np.ascontiguousarray(Wp.reshape(nA, prm.TO, nB, T).transpose(0, 2, 1, 3).reshape(nA * nB, prm.TO, T))
                V = np.broadcast_to(ex[:, None, :], (3, nA * nB, prm.TO)).copy()
                if MC._is_gpu(B):
                    out = B.tonumpy(MC.tiles_vecmat(B, B.xp.asarray(Wt), list(range(nA * nB)), B.asarray(V)))
                else:
                    out = MC.tiles_vecmat(B, Wt, list(range(nA * nB)), V)
                for a in range(nA):
                    for b in range(nB):
                        j = self.plan.base[k] + (a << self.plan.bb[k]) + b
                        off = (((h << sb) | j) << prm.ib)
                        wv[:, off:off + T] = out[:, a * nB + b]
        return wv

    # ---------------------------------------------------------------- round W1: xi -> folded rows, GEMM zero-sums
    def round_xi(self, tr):
        t0 = time.time()
        em, prm, pl = self.em, self.prm, self.plan
        ib = prm.ib
        xi = tr.exts(prm.ob)
        self._xi = xi
        WV = em.ctx.space("wsW", [("h", 1), ("j", pl.sb), ("k", ib)])
        if em.pv:
            self._wv = self._fold(xi)
        self.wv = MZ.ecol(WV, "w", self._wv)
        self.WV = WV
        for k, g in enumerate(self.gemms):
            gsb, ab, bb = pl.gsb[k], pl.ab[k], pl.bb[k]
            axes = ([("s", gsb)] if gsb else []) + ([("a", ab)] if ab else []) + ([("b", bb)] if bb else []) + [("k", ib)]
            GS = em.space(f"ws.g{k}", axes)
            S = GS.S
            jm = pl.jmap(k)
            eqx = Ex.atom(EqP(S.vars("k"), xi))
            ind_b = S.ind(b=0) if bb else Ex.const(1)
            in_xb = g["XS"].bits - gsb
            in_ob = g["OS"].bits - gsb
            xmap = _flatmap(g["XS"].axes, lambda p, in_xb=in_xb: ("k", p) if p < ib else (("b", p - ib) if p < in_xb else ("s", p - in_xb)))
            ymap = _flatmap(g["OS"].axes, lambda p, in_ob=in_ob: ("k", p) if p < ib else (("a", p - ib) if p < in_ob else ("s", p - in_ob)))
            eq_ax = (["s"] if gsb else []) + (["a"] if ab else [])
            sal_ex = MZ.view(self.s_al, S, {"j": jm + [("c", 0)] * (self.SS_b - pl.sb)}) if self.has_cfg else None
            for h, (Xin, Yout) in enumerate(((g["X"], g["Y"]), (g["Z"], g["A"]))):
                wvv = MZ.view(self.wv, S, {"h": ("c", h), "j": jm, "k": "k"})
                xv = MZ.view(Xin, S, xmap) if isinstance(Xin, G.Col) else em.view(Xin, GS, xmap).ex
                yv = MZ.view(Yout, S, ymap) if isinstance(Yout, G.Col) else em.view(Yout, GS, ymap).ex
                lhs = (sal_ex * wvv * xv) if sal_ex is not None else (wvv * xv)
                S.sumzero(lhs - ind_b * eqx * yv, eq_axes=eq_ax, tag=f"ws.gemm{k}.{'cr'[h]}")
        self.info["ws_round_xi_s"] = round(time.time() - t0, 4)

    # ---------------------------------------------------------------- round W2: zeta -> hidden claims
    def round_zeta(self, tr):
        t0 = time.time()
        em, prm, pl = self.em, self.prm, self.plan
        sb, ib, IB = pl.sb, prm.ib, self.IB
        J, H = self.J, self.J // 2
        zeta = tr.exts(ib)
        self._zeta = zeta
        CS = em.ctx.space("wsC", [("h", 1), ("j", IB - 1)])
        cv = None
        if em.pv:
            ez = CPU.eq_table(list(zeta))
            W3 = self._wv.reshape(3, 2 << sb, prm.TI)
            cv = np.zeros((3, J), dtype=np.uint64)
            cs = [(0, 0, 0)] * J
            for hj in range(2 << sb):
                v = CPU.edot_dev(ez, np.ascontiguousarray(W3[:, hj]))
                h, j = hj >> sb, hj & ((1 << sb) - 1)
                cs[h * H + j] = v
                cv[0, h * H + j], cv[1, h * H + j], cv[2, h * H + j] = v
            if self.has_cfg:
                c0, c1 = self.cfg_ext.cfg_claims(self._xi, zeta)
                extra = [(H - 2, c0), (H - 1, c1)]
                if self.vec_ext is not None:
                    cs_, cv_ = self.vec_ext.claims(self._xi, zeta)
                    extra += [(pl.res0, cs_), (pl.res0 + 1, cv_)]
                for idx, val in extra:
                    cs[idx] = val
                    cv[0, idx], cv[1, idx], cv[2, idx] = val
            self._cs = cs
            self._wv = None
        self.c = MZ.ecol(CS, "c", cv)
        WV = self.WV
        cvw = MZ.view(self.c, WV, {"h": "h", "j": [("j", s) for s in range(sb)] + [("c", 0)] * (IB - 1 - sb)})
        R0 = pl.res0
        if self.has_cfg and (1 << sb) > R0:
            sel_wv = Ex.const(1) - WV.ind(h=0) * WV.rng("j", R0, 1 << sb)
            WV.sumzero(sel_wv * (Ex.atom(EqP(WV.vars("k"), zeta)) * self.wv.ex - WV.ind(k=0) * cvw),
                       eq_axes=["h", "j"], tag="ws.claims")
        else:
            WV.sumzero(Ex.atom(EqP(WV.vars("k"), zeta)) * self.wv.ex - WV.ind(k=0) * cvw,
                       eq_axes=["h", "j"], tag="ws.claims")
        if self.has_cfg:
            # Unused lower-half slots [2^sb, res0) and all upper-half slots [2^sb, H) (incl. the reserved ones) forced 0
            if (1 << sb) < R0:
                CS.zero(self.c.ex, sel=CS.ind(h=0) * CS.rng("j", 1 << sb, R0), tag="ws.cpad.lo")
            if (1 << sb) < H:
                CS.zero(self.c.ex, sel=CS.ind(h=1) * CS.rng("j", 1 << sb, H), tag="ws.cpad.hi")
            jm_c0 = [("c", ((H - 2) >> s) & 1) for s in range(IB - 1)]
            jm_c1 = [("c", ((H - 1) >> s) & 1) for s in range(IB - 1)]
            self.cfg_ext.bind_claims(
                self._xi, zeta,
                lambda sp2: MZ.view(self.c, sp2, {"h": ("c", 0), "j": jm_c0}),
                lambda sp2: MZ.view(self.c, sp2, {"h": ("c", 0), "j": jm_c1}),
            )
            if self.vec_ext is not None:
                jm_s = [("c", (R0 >> s_) & 1) for s_ in range(IB - 1)]
                jm_v = [("c", ((R0 + 1) >> s_) & 1) for s_ in range(IB - 1)]
                self.vec_ext.bind_claims(
                    self._xi, zeta,
                    lambda sp2: MZ.view(self.c, sp2, {"h": ("c", 0), "j": jm_s}),
                    lambda sp2: MZ.view(self.c, sp2, {"h": ("c", 0), "j": jm_v}),
                )
        elif (1 << sb) < H:
            CS.zero(self.c.ex, sel=Ex.const(1) - CS.lt("j", 1 << sb), tag="ws.cpad")
        self.info["ws_round_zeta_s"] = round(time.time() - t0, 4)


# ============================================================================ build / prove / verify
def statement(prm, model_id, public, cfg_rows=False, vol_blocks=0, meta_cfg0=2, meta_cfg1=3, vec_row=None, vec_pad=0,
              claim_slots=None):
    """vec_row / vec_pad (profile constants, requires cfg_rows): row index of the (single) vec row of W and the committed
    prefix length (power of two >= Layout.vec_len) of the gain-vector binding (``VecExt``).  Zero knowledge: vec_pad must
    be a PROFILE cap (like R_MAX), not derived from the model's own vec length -- the tests / cmd_qwen derive it from the
    Layout for convenience, which reveals ceil(log2) of the total gain length.
    claim_slots: J, the number of HCO claim slots (profile constant, power of two >= 16; J / 2 >= the planned GEMM slots
    + 4 reserved).  Omitted (or == R_MAX): J = R_MAX and no statement key, i.e. the original statement."""
    st = {"v": VERSION, "params": asdict(prm), "model_id": [int(x) for x in model_id], "public": public}
    if claim_slots is not None and int(claim_slots) != prm.R_MAX:
        st["claim_slots"] = int(claim_slots)
    if cfg_rows:
        st["cfg_rows"] = True
        st["meta_cfg0"] = int(meta_cfg0)
        st["meta_cfg1"] = int(meta_cfg1)
        if vec_pad:
            st["vec_row"] = int(vec_row)
            st["vec_pad"] = int(vec_pad)
    if vol_blocks > 0:
        st["vol_blocks"] = int(vol_blocks)
    return st


def build(prm, stmt, blocks_fn, wit=None, tile_of=None, B=CPU, info=None):
    """blocks_fn(em) -> list of TiledWeights / CfgTiledWeights providers (runs the gadgets).  wit (prover): store
    (prover's view), pre (model_id preimage), optional vol_elems.  Returns (em, wsum)."""
    info = {} if info is None else info
    pv = wit is not None
    em = EmHG(emit=True, prove=pv)
    cfg_rows = bool(stmt.get("cfg_rows", False))
    vol_blocks = int(stmt.get("vol_blocks", 0))
    cfg_ext = None
    if cfg_rows or vol_blocks > 0:
        cfg_ext = MZ.CfgExt(em.ctx, prm,
                            store=wit["store"] if pv else None,
                            vol_elems=wit.get("vol_elems") if pv else None,
                            cfg_rows=cfg_rows,
                            vol_blocks=vol_blocks,
                            meta_cfg0=int(stmt.get("meta_cfg0", 2)),
                            meta_cfg1=int(stmt.get("meta_cfg1", 3)))
        em.cfg_ext = cfg_ext
    vec_ext = None
    if cfg_rows and int(stmt.get("vec_pad", 0)) > 0:
        vec_ext = VecExt(em.ctx, prm, wit["store"] if pv else None, stmt["vec_row"], stmt["vec_pad"], cfg_ext)
    em.vec_ext = vec_ext
    t0 = time.time()
    provs = blocks_fn(em)
    info["gadgets_build_s"] = round(time.time() - t0, 3)
    ws = WSum(em, prm, provs, tile_of=tile_of, B=B, info=info, cfg_ext=cfg_ext, store=wit["store"] if pv else None,
              vec_ext=vec_ext, J=int(stmt.get("claim_slots", prm.R_MAX)))
    pre_rounds = list(em.ctx.cir.rounds)
    em.ctx.cir.rounds = []
    MZ.build(prm, stmt, wit, B, info, ctx=em.ctx, ext=ws, cfg_ext=cfg_ext)
    em.ctx.cir.rounds = pre_rounds + list(em.ctx.cir.rounds)
    return em, ws


def cells_by_kind(cir):
    out = {}
    for v in cir.vecs:
        sp = v.name.split(".")[0]
        k = "hco" if sp.startswith("mc") else "wsum" if sp.startswith("ws") else "libra" if sp.startswith("libra") else "gadgets"
        out[k] = out.get(k, 0) + v.N
    return out


def prove(prm, stmt, blocks_fn, wit, tile_of, B=CPU):
    info = {}
    t0 = time.time()
    em, ws = build(prm, stmt, blocks_fn, wit, tile_of, B, info)
    t1 = time.time()
    pf = Z.prove(em.ctx.cir, stmt, B, label=LABEL)
    info["build_s"] = round(t1 - t0, 3)
    info["prove_s"] = round(time.time() - t1, 3)
    cir = em.ctx.cir
    info["cells_by_kind"] = cells_by_kind(cir)
    info["cells"] = int(sum(v.N for v in cir.vecs))
    info["vectors"] = len(cir.vecs)
    info["gemms"] = len(ws.gemms)
    info["slots_used"] = ws.plan.used
    info["zk_budget"] = dict(Z.ZK_REPORT)
    info["violations"] = list(em.ctx.notes)
    info["gadget_fails"] = dict(em.fails)
    return pf, info


def verify(prm, stmt, blocks_fn, pf):
    em, _ = build(prm, stmt, blocks_fn, None)
    return Z.verify(em.ctx.cir, stmt, pf, label=LABEL)


def try_verify(prm, stmt, blocks_fn, pf):
    try:
        verify(prm, stmt, blocks_fn, pf)
        return "accepted"
    except Z.ProofError as e:
        return "rejected: " + str(e)


def forged_rows(prm, layout, tensors_codes, cfg_update=False, base_store=None):
    """Rows of W (cd and r halves) of the given tensors from (possibly cheating) bf16 codes using per-row maximal
    decode: {row: coefficients}.  When cfg_update=True and layout.cfg_rows, also recomputes SCALE and CFG0 (Tt) rows
    so the cheater's view has internally consistent per-row exponents and Tt entries (only the C_W column check fails)."""
    out = {}
    g_overrides = {}
    gten_overrides = {}
    for name, codes in tensors_codes.items():
        codes = np.asarray(codes, dtype=np.uint16)
        o, n_in = codes.shape
        kap = int(CZ.kappa(n_in))
        nz_gs = []
        for a in range(-(-o // prm.TO)):
            for b in range(-(-n_in // prm.TI)):
                t = layout.tile_of[(name, a, b)]
                blk = np.zeros((prm.TO, prm.TI), dtype=np.uint16)
                sub = codes[a * prm.TO:(a + 1) * prm.TO, b * prm.TI:(b + 1) * prm.TI]
                blk[:sub.shape[0], :sub.shape[1]] = sub
                g = MC.row_exponent(blk, prm.G_LO)
                if (blk & 0x7FFF).any():
                    nz_gs.append(int(g))
                w = MC.cd_decode(blk, g)
                wr = (np.abs(w.astype(np.int64)) + kap * MC.tile_inexact(blk, g).astype(np.int64)).astype(np.int32)
                out[t] = w.reshape(-1)
                if layout.radius:
                    out[layout.r_row(t)] = wr.reshape(-1)
                g_overrides[t] = int(g)
        gten_overrides[name] = max(nz_gs) if nz_gs else int(prm.G_LO)
    if cfg_update and layout.cfg_rows and base_store is not None:
        sc = base_store.row(MC.META_SCALE).copy()
        for t, g in g_overrides.items():
            sc[t] = g
        out[MC.META_SCALE] = sc
        c0 = base_store.row(layout.meta_cfg0).copy()
        NT = layout.NT
        keys = c0[0:NT]
        for name in tensors_codes:
            role, lp1, exp = MC.tensor_role_layer_expert(name)
            o, n_in = layout.shapes[name]
            gt = gten_overrides[name]
            nR, nC = -(-o // prm.TO), -(-n_in // prm.TI)
            pR = 1 << max(0, (nR - 1).bit_length()) if nR > 1 else 1
            pC = 1 << max(0, (nC - 1).bit_length()) if nC > 1 else 1
            for a in range(pR):
                for b in range(pC):
                    ky = MC.tile_key(role, lp1, exp, a, b)
                    pos = int(np.searchsorted(keys, ky))
                    if pos < NT and int(keys[pos]) == ky:
                        c0[5 * NT + pos] = gt
                        if a < nR and b < nC:
                            t = layout.tile_of[(name, a, b)]
                            gj = g_overrides[t]
                            c0[2 * NT + pos] = gj
                            c0[4 * NT + pos] = 1 << (gt - gj)
        out[layout.meta_cfg0] = c0
    return out


# ============================================================================ forgery helper
def int4_codes(codes, G=128):
    """RTN symmetric int4 re-quantization of bf16 weight codes (group G along the input axis, bf16 result codes)."""
    from .fp_ref import bf16
    W = (np.asarray(codes, dtype=np.uint32) << 16).view(np.float32)
    o, i = W.shape
    Wg = W.reshape(o, i // G, G)
    s = np.maximum(np.abs(Wg).max(-1, keepdims=True), np.float32(1e-12)) / np.float32(7.0)
    q = np.clip(np.round(Wg / s), -8, 7).astype(np.float32)
    return bf16((q * s).astype(np.float32).reshape(o, i))
