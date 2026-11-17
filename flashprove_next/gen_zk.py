# SPDX-License-Identifier: Apache-2.0
"""Generation slot composition (specs/01 §4.2 HEAD block + §5.4 generation predicate S_t, specs/06 §3.2):
final RMSNorm -> batched LM head (reference logits recomputed IN CIRCUIT, never objects) -> keyed sampling rule
``fpn-samp-bk-v3`` (``flashprove_next.sampler_zk``).

Three algorithmic cost reductions over naive per-slot ``blk_head`` + ``sampler_zk``:
  1. Merged HEAD and scalar radius: only ``(Y_v, A_v)`` (2 cells/vocab item) are committed per vocabulary
     entry instead of ``blk_head``'s ~75 columns/vocab item; the certified fp32 store radius upper bound ``Rmax`` is
     computed once per step on the scalar space ``SS`` from ``Amax = max_{a_v} A_v`` via ``sampler_zk._head_rmax_from_A``.
  2. Piecewise-linear Gumbel table with an exact tail: ``sampler_zk``'s 65,536-block linear + 131,072-row exact tail
     replaces the ``2^24``-row Gumbel table (>100x reduction in fixed table cells) with proven ``<= 16`` units of
     ``2^-24`` error absorbed into the bucket condition while the engine-side sampling rule stays unchanged.
  3. Batched LM-head reduction across all K generation steps: all ``K`` steps multiply by the same ``W_head``
     (``V x d``).  Instead of registering ``K`` copies of ``W_head`` in ``WSum`` (``105 x 149 = 15,645`` tile slots on
     ``Qwen3-0.6B``), a single commit round before ``WSum.round_xi`` draws ``K`` random extension-field scalars
     ``beta_s in E``, forms ``(X_bar, Z_bar, Y_bar, A_bar) = sum_s beta_s (X_s, Z_s, Y_s, A_s)``, constrains them
     entrywise, and registers ONE ``V x d`` GEMM (``149`` tiles) in ``WSum``.
"""
import numpy as np

from . import certified as CZ
from . import mc_zk as MZ
from . import modelcommit as MC
from . import sampler_zk as SZ
from . import sampling_v3 as S3
from . import zk_flat as Z
from .field import CPU, P
from .fp_ref import EPS_BITS
from .zk_flat import Ex, SelLt

VERSION = "fpn-gen-2"
LABEL = SZ.LABEL
HEAD_MB = 20          # 20-bit aligned input mantissa: exact for bf16 across 12 octaves, keeps kq in [0, 32]


def _lg(n):
    return max(0, int(n - 1).bit_length())


class _DecRec:
    """records (tag, space, Dec) of every certified.decode_bf16 call made while a block is built."""

    def __init__(self):
        self.recs = []
        self._orig = None

    def __enter__(self):
        self._orig = CZ.decode_bf16
        rec = self.recs

        def wrapped(em, sp, codes, tag):
            dec = self._orig(em, sp, codes, tag)
            rec.append((tag, sp, dec))
            return dec
        CZ.decode_bf16 = wrapped
        return self

    def __exit__(self, *a):
        CZ.decode_bf16 = self._orig

    def get(self, tag):
        hit = [(sp, dec) for t, sp, dec in self.recs if t == tag]
        assert len(hit) == 1, (tag, [t for t, _, _ in self.recs])
        return hit[0]


# ============================================================================ statement / witness helpers
def make_statement(prm, kcom, ctx, ds, ts, toks=None, hidden_t=False, meta=None):
    st = SZ.make_statement(prm, kcom, ctx, ds, None if hidden_t else ts, toks)
    st["gen"] = VERSION
    if meta:
        st.update(meta)
    return st


def head_align(em, prov, gname, head_name, xin, d, prm_sz, tag):
    """Align decoded bf16 ``x_fn`` for the LM head with ``MB = min(HEAD_MB, gemm_mb)`` and ``efloor = G_head - KQMAX - Fq``
    so ``kq = -e_y - Fq`` always lies in ``[0, KQMAX]`` even on tiny-norm hidden states.
    Returns ``(XS, dxi, Xc, Zc, ey)``."""
    XS = em.space(tag + ".x", [("i", _lg(d))])
    SC = em.space(tag + ".sc", [])
    dxi = CZ.decode_bf16(em, XS, CZ._pad_sp(xin, XS), tag + ".xi")
    mb = min(HEAD_MB, CZ.gemm_mb(prov, XS.in_bits))
    M = prov.mats[head_name]
    Gx = M.G
    # e_y = e_x - G_head >= -(KQMAX + Fq)  <=>  e_x >= G_head - (KQMAX + Fq)
    efl = em.lin([], Gx - (KQMAX_FLOOR := (SZ.KQMAX + prm_sz.Fq)), sp=SC, bits=10)
    efl = CZ.minmax2(em, efl, em.const(SC, CZ.EFX), tag + ".fl", True)
    Xs, mX, t, ex = CZ.align(em, dxi, SC, mb, tag + ".al", efloor=efl)
    kap = CZ.kappa(M.n_in)
    Z_in = em.lin([(1, mX), (kap, t)], bits=mX.bits + 1)
    assert kap < (1 << mX.bits), (tag, kap, mX.bits)
    Xc = em.commit(Xs, tag + ".Xc")
    Zc = em.commit(Z_in, tag + ".Zc")
    ey = em.commit(em.lin([(1, ex)], -Gx, bits=10), tag + ".ey")
    return XS, dxi, Xc, Zc, ey


def ref_logits(prov_eval, h_out, x_fn, d, V, prm_sz=None, head_name="lm", norm_name="final_norm",
               eps_bits=EPS_BITS, x_head=None):
    """Evaluator pass of ``blk_N`` + merged ``HEAD`` on one slot -> ``(Y, A, Rmax, ey, fails)``."""
    if prm_sz is None:
        prm_sz = SZ.Prm(V, S3.Rule(0.6, 0.95, 32), 2)
    em = CZ.Em(emit=False, prove=True)
    CZ.blk_N(em, prov_eval, norm_name, h_out, x_fn, d, eps_bits, tag="N")
    xh = x_fn if x_head is None else x_head
    XS, dxi, Xc, Zc, ey = head_align(em, prov_eval, norm_name, head_name, xh, d, prm_sz, tag="H")
    OS = em.space("H.o", [("o", _lg(V))])
    Yx, Ax, _, _ = prov_eval.matvec(em, head_name, Xc, Zc, OS, "H.mm")
    Yv, Av, ey_i = Yx.v[:V].copy(), Ax.v[:V].copy(), int(ey.v[0])
    rmax, rfails = SZ.head_rmax_val(int(Av.max()), ey_i, d)
    fails = dict(em.fails); fails.update(rfails)
    return Yv, Av, rmax, ey_i, fails


def make_wit(prm, kprf, salt, ctx, slots, hidden_t=False, hidden_tok=False, hidden_kcom=False):
    ds = [s_.get("d_role", 0) for s_ in slots]
    ts = [int(s_["t"]) for s_ in slots]
    toks = [int(s_["tok"]) for s_ in slots]
    kcom = None if hidden_kcom else SZ.kcom_of(kprf, salt)
    stmt = make_statement(prm, kcom, ctx, ds, ts, None if hidden_tok else toks, hidden_t)
    return stmt, dict(slots=slots, kprf=kprf, salt=salt, ctx=ctx)


def check_ref(prm, prov_eval, kprf, salt, ctx, slots, d, head_name="lm", norm_name="final_norm", eps_bits=EPS_BITS):
    steps, fails = [], []
    for s_ in slots:
        Y, A, _, ey, f = ref_logits(prov_eval, s_["h_out"], s_["x_fn"], d, prm.V, prm, head_name, norm_name,
                                    eps_bits, s_.get("x_head"))
        fails.append(f)
        steps.append(dict(Y=Y, R=A, ey=ey, a=s_.get("a", np.ones(prm.V, dtype=bool)),
                          d=s_.get("d_role", 0), t=s_["t"], tok=s_["tok"]))
    _, _, oks = SZ.make_wit(prm, kprf, salt, ctx, steps, head_d=d)
    return oks, fails


# ============================================================================ Batched LM-Head Reduction
class BatchedHead:
    """Batches ``K`` LM-head evaluations ``Y_s = W_head X_s, A_s = W_head^r Z_s`` across all generation slots into ONE
    ``V x d`` GEMM registration in ``WSum`` using random extension-field scalars ``beta_0 .. beta_{K-1}`` drawn in a
    commit round BEFORE ``WSum.round_xi``."""

    def __init__(self, em, prm, prov, head_name, X_cols, Z_cols, Ys, Rs, VSs, V, d, tag="bhead"):
        self.em, self.prm, self.prov, self.head_name = em, prm, prov, head_name
        self.X_cols, self.Z_cols = X_cols, Z_cols
        self.Ys, self.Rs, self.VSs = Ys, Rs, VSs
        self.V, self.d, self.tag = int(V), int(d), tag
        self.K = prm.K
        self.XS = em.space(tag + ".x", [("i", _lg(d))])
        self.OS = em.space(tag + ".o", [("o", _lg(V))])
        self._xb_col = self._zb_col = self._yb_col = self._ab_col = None
        # Register the SINGLE batched LM-head GEMM on `prov` with placeholder E-valued columns filled in round_beta.
        # Phase-1 columns are committed in phase 1, whereas X_bar, Z_bar, Y_bar, A_bar depend on beta_s drawn after
        # phase 1, so we declare X_bar, Z_bar, Y_bar, A_bar inside `round_beta(tr)` (phase 10+k) and point the
        # single GEMM descriptor on `prov.gemms` at those round-committed E-valued columns.
        M = prov.mats[head_name]
        tname = prov.tensors.get(head_name, head_name)
        role_id = prov.role_ids.get(head_name)
        if role_id is None and tname:
            role_id = MC.tensor_role_layer_expert(tname)[0]
        self.gemm_entry = dict(
            name=head_name, tensor=tname,
            X=None, Z=None, Y=None, A=None,
            OS=self.OS, XS=self.XS, mat=M,
            G_x=prov.g_cols[head_name] if hasattr(prov, "g_cols") else M.G,
            role_id=role_id, layer_p1=getattr(prov, "layer_p1", 0), expert=getattr(prov, "expert", 0),
            tag=tag,
        )
        prov.gemms.append(self.gemm_entry)

    def round_beta(self, tr):
        em, prm, K = self.em, self.prm, self.K
        pv = em.pv
        betas = tr.exts(K)
        XS, OS = self.XS, self.OS
        xb_v = zb_v = yb_v = ab_v = None
        if pv:
            xb_v = np.zeros((3, XS.n), dtype=np.uint64)
            zb_v = np.zeros((3, XS.n), dtype=np.uint64)
            yb_v = np.zeros((3, OS.n), dtype=np.uint64)
            ab_v = np.zeros((3, OS.n), dtype=np.uint64)
            M = self.prov.mats[self.head_name]
            o, inn = M.W.shape
            if isinstance(self.X_cols, CZ.X):
                x2 = self.X_cols.v.reshape(K, XS.n)
                z2 = self.Z_cols.v.reshape(K, XS.n)
                for s, bt in enumerate(betas):
                    xv_s = np.zeros(XS.n, dtype=np.int64); xv_s[:inn] = x2[s, :inn]
                    zv_s = np.zeros(XS.n, dtype=np.int64); zv_s[:inn] = z2[s, :inn]
                    xb_v = CPU.eadd(xb_v, CPU.escale(CPU.ext(CPU.fromi(xv_s)), bt))
                    zb_v = CPU.eadd(zb_v, CPU.escale(CPU.ext(CPU.fromi(zv_s)), bt))
            else:
                for s, bt in enumerate(betas):
                    xv_s = np.zeros(XS.n, dtype=np.int64); xv_s[:inn] = self.X_cols[s].v[:inn]
                    zv_s = np.zeros(XS.n, dtype=np.int64); zv_s[:inn] = self.Z_cols[s].v[:inn]
                    xb_v = CPU.eadd(xb_v, CPU.escale(CPU.ext(CPU.fromi(xv_s)), bt))
                    zb_v = CPU.eadd(zb_v, CPU.escale(CPU.ext(CPU.fromi(zv_s)), bt))
            for p, (off, bb) in enumerate(prm.pieces):
                n_p = 1 << (bb + 5)
                y_all = self.Ys[p].v.reshape(K, -1)[:, :n_p]
                r_all = self.Rs[p].v.reshape(K, -1)[:, :n_p]
                lo = off * 32
                cnt = max(0, min(self.V - lo, n_p))
                if cnt > 0:
                    for s, bt in enumerate(betas):
                        yb_v[:, lo:lo + cnt] = CPU.eadd(yb_v[:, lo:lo + cnt],
                                                        CPU.escale(CPU.ext(CPU.fromi(y_all[s, :cnt])), bt))
                        ab_v[:, lo:lo + cnt] = CPU.eadd(ab_v[:, lo:lo + cnt],
                                                        CPU.escale(CPU.ext(CPU.fromi(r_all[s, :cnt])), bt))
        Xb = MZ.ecol(XS.S, "Xb", xb_v)
        Zb = MZ.ecol(XS.S, "Zb", zb_v)
        Yb = MZ.ecol(OS.S, "Yb", yb_v)
        Ab = MZ.ecol(OS.S, "Ab", ab_v)
        self.gemm_entry.update(X=Xb, Z=Zb, Y=Yb, A=Ab)

        bv = np.zeros((3, 1 << prm.sb), dtype=np.uint64)
        for s, bt in enumerate(betas):
            bv[0, s], bv[1, s], bv[2, s] = bt
        # 1) Constrain Xb = sum_s beta_s X_s and Zb = sum_s beta_s Z_s (0 on i >= d)
        if isinstance(self.X_cols, CZ.X):
            XSp = self.X_cols.sp.S
            b_x = Ex.atom(MZ.EVec(f"{self.tag}.beta.x", bv, XSp.vars("s")))
            ind_s0_x = XSp.ind(s=0)
            in_dx = Ex.atom(SelLt(list(range(XS.bits)), self.d)) if self.d < XS.n else Ex.const(1)
            XSp.sumzero(in_dx * b_x * self.X_cols.ex - ind_s0_x * MZ.view(Xb, XSp, {"i": "i"}),
                        eq_axes=["i"], tag=self.tag + ".Xb")
            XSp.sumzero(in_dx * b_x * self.Z_cols.ex - ind_s0_x * MZ.view(Zb, XSp, {"i": "i"}),
                        eq_axes=["i"], tag=self.tag + ".Zb")
        else:
            x_acc = Ex.const(0)
            z_acc = Ex.const(0)
            for s, bt in enumerate(betas):
                x_acc = x_acc + Ex.const(bt) * em.view(self.X_cols[s], XS).ex
                z_acc = z_acc + Ex.const(bt) * em.view(self.Z_cols[s], XS).ex
            XS.S.zero(Xb.ex - x_acc, tag=self.tag + ".Xb")
            XS.S.zero(Zb.ex - z_acc, tag=self.tag + ".Zb")
        if self.d < XS.n:
            in_d = Ex.atom(SelLt(list(range(XS.bits)), self.d))
            XS.S.zero((Ex.const(1) - in_d) * Xb.ex, tag=self.tag + ".Xpad")
            XS.S.zero((Ex.const(1) - in_d) * Zb.ex, tag=self.tag + ".Zpad")
        if self.V < OS.n:
            in_vo = Ex.atom(SelLt(list(range(OS.bits)), self.V))
            OS.S.zero((Ex.const(1) - in_vo) * Yb.ex, tag=self.tag + ".Ypad")
            OS.S.zero((Ex.const(1) - in_vo) * Ab.ex, tag=self.tag + ".Apad")

        # 2) Constrain Yb[v] = sum_s beta_s Y_s[v] and Ab[v] = sum_s beta_s R_s[v] for all v < V.
        ob = OS.bits
        for p, (off, bb) in enumerate(prm.pieces):
            if off * 32 >= self.V:
                continue
            VSp = self.VSs[p]
            S_ = VSp.S
            vbits = [("k", t) for t in range(5)] + ([("b", t) for t in range(bb)] if bb else [])
            hi = [("c", ((off * 32) >> t) & 1) for t in range(bb + 5, ob)]
            omap = {"o": (vbits + hi)[:ob]}
            vs = S_.vars("k") + (S_.vars("b") if VSp.has("b") else [])
            in_v = Ex.atom(SelLt(vs, max(0, min(self.V - off * 32, 1 << (bb + 5)))))
            zsel = SZ._zsel(VSp)
            # Sum over s of beta_s * Ys[p](s, b, k) == Yb(off*32 + 32*b + k) for every (b, k) with v < V.
            if prm.sb:
                b_atom = Ex.atom(MZ.EVec(f"{self.tag}.beta.{p}", bv, S_.vars("s")))
                ind_s0 = S_.ind(s=0)
                eq_ax = (["b"] if VSp.has("b") else []) + ["k"]
            else:
                b_atom = Ex.const(betas[0])
                ind_s0 = Ex.const(1)
                eq_ax = (["b"] if VSp.has("b") else []) + ["k"]
            yb_on = MZ.view(Yb, S_, omap)
            ab_on = MZ.view(Ab, S_, omap)
            S_.sumzero(zsel * in_v * (b_atom * self.Ys[p].ex - ind_s0 * yb_on),
                       eq_axes=eq_ax, tag=f"{self.tag}.Yb.{p}")
            S_.sumzero(zsel * in_v * (b_atom * self.Rs[p].ex - ind_s0 * ab_on),
                       eq_axes=eq_ax, tag=f"{self.tag}.Ab.{p}")


# ============================================================================ circuit
def build_gen(em, prm, prov, stmt, wit=None, d=None, V=None, head_name="lm", norm_name="final_norm",
              eps_bits=EPS_BITS, tag="gen", hidden_t=False, link_x=True, kprf_on_ab=None, all_allowed=True):
    """Emit ``K`` generation slots on the existing emitting ``em`` with weight provider ``prov``.
      * With ``TiledWeights`` / ``CfgTiledWeights`` (``hasattr(prov, 'gemms')``), uses ``BatchedHead`` so all ``K``
        generation steps share ONE ``V x d`` GEMM in ``WSum`` (appends ``bhead.round_beta`` to ``out['rounds']``).
      * With ``PublicWeights`` (standalone unit tests), evaluates ``prov.matvec`` per step and links ``(Y, A, ey)``.
    Returns dict with ``h_out``, ``x_fn``, ``tok``, ``t``, ``SS``, ``wbits``, ``oks``, ``rounds``."""
    assert em.emit
    pv = wit is not None
    V = prm.V if V is None else int(V)
    K = prm.K
    z16 = lambda n: np.zeros(n, dtype=np.uint16)
    batched = hasattr(prov, "gemms")
    M_head = prov.mats[head_name]
    hs, xs, wb = [], [], None
    h_batched = x_batched = None
    pub_refs = []
    if batched and prm.sb > 0:
        # Vectorize final RMSNorm + head_align across all K generation steps on ONE ('s', sb) child Em.
        em_g = em.sub_em([("s", prm.sb)])
        if pv:
            h_all = np.stack([sl["h_out"] for sl in wit["slots"]])
            x_all = np.stack([sl["x_fn"] for sl in wit["slots"]])
            xh_all = np.stack([sl["x_fn"] if sl.get("x_head") is None else sl["x_head"] for sl in wit["slots"]])
        else:
            h_all = x_all = xh_all = np.zeros((K, d), dtype=np.uint16)
        with _DecRec() as rec:
            CZ.blk_N(em_g, prov, norm_name, h_all, x_all, d, eps_bits, tag=f"{tag}.N")
            XS_g, dxh, X_col, Z_col, ey_col = head_align(em_g, prov, norm_name, head_name, xh_all, d, prm, tag=f"{tag}.H")
        sp_h, dh = rec.get(f"{tag}.N.h")
        sp_x, dx = rec.get(f"{tag}.N.x")
        if link_x:
            em_g.zero(sp_x, lambda dx=dx, dxh=dxh, sp_x=sp_x: dx.code.ex - em_g.view(dxh.code, sp_x).ex, f"{tag}.xlink")
        h_batched, x_batched = dh.code, dx.code
        wb = sp_x.in_bits
        X_cols, Z_cols = X_col, Z_col
    else:
        X_cols, Z_cols, ey_cols = [], [], []
        for s in range(K):
            sl = wit["slots"][s] if pv else None
            h = sl["h_out"] if pv else z16(d)
            x = sl["x_fn"] if pv else z16(d)
            xh = (x if sl.get("x_head") is None else sl["x_head"]) if pv else z16(d)
            with _DecRec() as rec:
                CZ.blk_N(em, prov, norm_name, h, x, d, eps_bits, tag=f"{tag}.N{s}")
                XS_s, dxh, Xc, Zc, ey_c = head_align(em, prov, norm_name, head_name, xh, d, prm, tag=f"{tag}.H{s}")
            sp_h, dh = rec.get(f"{tag}.N{s}.h")
            sp_x, dx = rec.get(f"{tag}.N{s}.x")
            if link_x:
                em.zero(sp_x, lambda dx=dx, dxh=dxh, sp_x=sp_x: dx.code.ex - em.view(dxh.code, sp_x).ex, f"{tag}.xlink{s}")
            hs.append(dh.code); xs.append(dx.code)
            X_cols.append(Xc); Z_cols.append(Zc); ey_cols.append(ey_c)
            wb = sp_x.S.bits
            if not batched:
                OS = em.space(f"{tag}.Ho{s}", [("o", _lg(V))])
                Yx, Ax, _, _ = prov.matvec(em, head_name, Xc, Zc, OS, f"{tag}.Hm{s}")
                pub_refs.append(dict(Y=Yx, R=Ax, ey=ey_c, OS=OS))
    sw, oks = None, None
    if pv:
        steps = []
        if batched and prm.sb > 0:
            o, inn = M_head.W.shape
            X2 = X_col.v.reshape(K, -1)[:, :inn]
            Z2 = Z_col.v.reshape(K, -1)[:, :inn]
            Y_mat = X2 @ M_head.W[:V, :inn].T.astype(np.int64)
            A_mat = Z2 @ M_head.Wr[:V, :inn].T.astype(np.int64)
            ey_arr = ey_col.v[:K]
            for s, sl in enumerate(wit["slots"]):
                Yv = Y_mat[s] if sl.get("y_override") is None else np.asarray(sl["y_override"], dtype=np.int64)[:V].copy()
                steps.append(dict(Y=Yv, R=A_mat[s], ey=int(ey_arr[s]), a=sl.get("a", np.ones(V, dtype=bool)),
                                  d=sl.get("d_role", 0), t=sl["t"], tok=sl["tok"]))
        else:
            for s, sl in enumerate(wit["slots"]):
                if batched:
                    o, inn = M_head.W.shape
                    Yv = M_head.W[:V, :inn].astype(np.int64) @ X_cols[s].v[:inn]
                    Av = M_head.Wr[:V, :inn].astype(np.int64) @ Z_cols[s].v[:inn]
                    if sl.get("y_override") is not None:
                        Yv = np.asarray(sl["y_override"], dtype=np.int64)[:V].copy()
                else:
                    Yv = pub_refs[s]["Y"].v[:V].copy()
                    Av = pub_refs[s]["R"].v[:V].copy()
                ey_i = int(ey_cols[s].v[0])
                steps.append(dict(Y=Yv, R=Av, ey=ey_i, a=sl.get("a", np.ones(V, dtype=bool)),
                                  d=sl.get("d_role", 0), t=sl["t"], tok=sl["tok"]))
        _, sw, oks = SZ.make_wit(prm, wit["kprf"], wit.get("salt", (0, 0, 0, 0)), wit["ctx"], steps,
                                 head_d=d, hidden_kcom=(kprf_on_ab is not None),
                                 hidden_t=hidden_t, hidden_tok=(stmt.get("tok") is None))
    out = SZ.build_on(em, prm, stmt, sw, head_refs=(None if batched else pub_refs),
                      commit_tok=True, hidden_t=hidden_t, all_allowed=all_allowed,
                      head_d=d, kprf_on_ab=kprf_on_ab)
    rounds = []
    if batched:
        SS = out["SS"]
        if prm.sb > 0:
            SS.S.zero(out["ey"].ex - em.view(ey_col, SS, {"s": "s"}).ex, tag=f"{tag}.ey")
        else:
            for s in range(K):
                SS.S.zero(out["ey"].ex - em.view(ey_cols[s], SS).ex, tag=f"{tag}.ey.{s}")
        bh = BatchedHead(em, prm, prov, head_name, X_cols, Z_cols, out["Ys"], out["Rs"], out["VSs"], V, d,
                         tag=f"{tag}.bh")
        rounds.append(bh.round_beta)
    out.update(h_out=hs, x_fn=xs, h_out_batched=h_batched, x_fn_batched=x_batched,
               wbits=wb, K=K, oks=oks, rounds=rounds)
    return out


# ============================================================================ standalone proof (tests)
def build(prm, stmt, prov, wit=None, d=None, **kw):
    pv = wit is not None
    em = CZ.Em(emit=True, prove=pv)
    out = build_gen(em, prm, prov, stmt, wit, d=d, **kw)
    return em, out
