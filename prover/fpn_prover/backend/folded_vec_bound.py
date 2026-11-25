# SPDX-License-Identifier: Apache-2.0
"""End-to-end prover of `FPN2` relation 6 (`FOLDED_VEC_BOUND`, specs/11-wire-format.md §5.7).

Relation 5 proves `Y = W X` for an `X` that lives INSIDE the private witness vector `F` (`root_0`): nothing ties that
`X` to anything the verifier (or `C_pre`) has seen, so a prover can open honest trace rows and an honest `N_s` while
proving a privately substituted `(X', Y' = W X')` (relation-5 "case (c)").  Relation 6 closes the gap:

  * `X_com` -- a SEPARATE ZK-blinded vector (same geometry / codeword length as `F` and `C`) that commits ONLY the
    activation tile(s) `X` in its data half plus independent uniform blinding everywhere else; its root `x_root` is
    computed BEFORE any challenge, anchored into `C_pre` (`precommit_x_root`) and PINNED by the verifier;
  * `F` (`root_0`) keeps `Y`, the Libra mask `G` and the blinding; the old `X` block of `F` becomes uniform spare
    (it MUST stay random: the batched opening exposes <F + X_com, xw> on that block);
  * the statement is `root_0 || weight_root || x_root || N_s`; all three roots are absorbed before any challenge;
  * the inner-product sumcheck is relation 5's, byte for byte; its final claim (never sent in the clear, exactly as
    in relation 5) becomes ONE linear claim on THREE committed vectors,
        y = target + mu e_W + mu^2 z_blind = <F, a> + <C, c> + <X_com, xw>,
  * discharged by ONE batched folded ZK-Basefold opening with ZERO extra fold layers: the two batching rounds fold
    the 4-slot table [F, F + C, F + X_com, F + C + X_com] with weights [a - c - xw, c, xw, 0] by `(theta, r_1)`, so
    H = F + theta C + r_1 X_com is the vector the coset-4 fold chain runs on; per query layer 0 additionally reads
    the `X_com` coset from `x_root`'s tree.
    (Not `F + mu C + mu^2 X_com`: `theta = r_0`, `r_1` are the batching challenges of the two existing rounds.)

`commit_x` draws its blinding from the dedicated OS CSPRNG (`SecretRng` / `_os_rand_field`, `secrets.token_bytes`),
never from `FPN_TEST_RNG_SEED`.  `engine=None` runs the PCS on the pure-Python reference (`folded_pcs.zk_commit` +
`pair_open3`); `engine=CudaEngine()/HostEngine()` runs commitments, batched sumcheck and fold chain on the kernels
(`pair_open3_engine`); both give identical bytes for the same rng stream.
"""
import hashlib
import time
from dataclasses import dataclass

import numpy as np

from flashprove_next.field import CPU as X
from flashprove_next.field import P, hadd, hf, hmul, hsub

from . import folded_air as FA
from . import folded_pcs as pcs
from . import folded_vec as FV
from .folded_vec import ROWS, ROW_BITS, IP_DEGREE, SecretRng, _os_rand_field, cell_np, draw_cells  # noqa: F401
from .fpn2 import (FOLDED_VEC_BOUND_N16_V1, Fpn2Transcript, enc3, enc3s, header, merkle_path, profile_id)

STMT = 128                  # root_0 || weight_root || x_root || N_s
_XROOT_TAG = b"FPN2-XROOT-v1"
_XROOT_TPR_PERSON = b"FPN2-XROOT-TPR\x00\x00"


# ----------------------------------------------------------------------------- layout (§5.7.2)
@dataclass(frozen=True)
class BoundLayout(FV.VecLayout):
    """Relation-6 geometry = relation 5's `(n, Q, k)`; every query additionally carries the `X_com` layer-0 leaf.

    F:     half {bit1 = 1}: `[0, B K)` uniform SPARE (X's old block) . `[B K, B K + B R)` Y . rest uniform spare;
           half {bit1 = 0}: Libra coefficients (slot 4 l + e) + uniform blinding.
    X_com: half {bit1 = 1}: `[0, B K)` X (index b K + j) . rest uniform spare; half {bit1 = 0}: uniform blinding.
    C:     `[0, B R K)` W (index (b R + i) K + j), the whole data half; the other half is 0 (deterministic).
    """
    STMT = STMT

    @property
    def query_len(self):
        return 2 * (96 + 32 * self.depth(0)) + sum(96 + 32 * self.depth(t) for t in range(self.folds))

    @property
    def body_used(self):
        return STMT + self.ip_len + self.pcs_len + self.queries * self.query_len


def layout_of(profile):
    lay = FV.layout_of(profile)
    return BoundLayout(lay.n, lay.queries, lay.k)


# ----------------------------------------------------------------------------- commitments
def commit_weights(layout, W, engine=None):
    """Deterministic commitment of C (relation 5's, cached)."""
    return FV.commit_weights(layout, W, engine)


def weight_root(layout, W, engine=None):
    return commit_weights(layout, W, engine).root


@dataclass
class XCommitment:
    """PRIVATE prover-side state of the pre-committed activations: the blinded cells and the commitment trees.
    Only `root` (32 bytes) ever leaves the prover before the challenge."""
    layout: BoundLayout
    cells: np.ndarray          # (2^n, 3) uint64: X in the data half, uniform everywhere else
    x_values: np.ndarray       # (B, K) uint64
    com: object                # folded_pcs.Commitment or folded_air.EngineCommitment
    engine: object = None

    @property
    def root(self):
        return self.com.root


def _check_x(layout, Xv):
    FV._check_tiles(layout, None, Xv, None)


def build_x_cells_np(layout, Xv, rng):
    """X_com cells (2^n, 3): one bulk uniform F_p3 draw (blinding half, data spare), then the X cells."""
    _check_x(layout, Xv)
    cells = draw_cells(rng, 1 << layout.n)
    xi = cell_np(np.arange(layout.B * layout.K, dtype=np.int64))
    cells[xi, 0] = np.asarray(Xv, dtype=np.uint64).reshape(-1)
    cells[xi, 1:] = 0
    return cells


def commit_x(layout, Xv, rng=None, engine=None):
    """Commit the activations `X` (B, K) under a fresh ZK-blinded `X_com`; returns the private `XCommitment` whose
    `.root` is `x_root`.  Blinding comes from the OS CSPRNG (`SecretRng`) unless a test passes an explicit `rng`;
    it never reads `FPN_TEST_RNG_SEED`.  Call BEFORE the external seed is drawn and bind `root` into `C_pre`
    (`precommit_x_root`)."""
    Xv = np.asarray(Xv, dtype=np.uint64)
    cells = build_x_cells_np(layout, Xv, rng or SecretRng())
    if engine is not None:
        com = FA.engine_commit(engine, cells)
    else:
        com = pcs.zk_commit([tuple(int(v) for v in row) for row in cells])
    return XCommitment(layout, cells, Xv.copy(), com, engine)


def precommit_x_root(setup_cert_bytes, x_root, response_tokens, trace_payload_root=None, policy_bytes=b""):
    """32-byte `C_pre` that binds `x_root` (and so the pre-committed activations) with ZERO bytes of plaintext
    activation disclosure: the only activation-derived input is the 32-byte hiding commitment.

    `x_root` is domain-tagged (`FPN2-XROOT-v1`) inside the policy bytes, so it can never alias a
    `trace_payload_root`; without a trace payload root the slot is `B256_{FPN2-XROOT-TPR}(x_root)`.  `C_pre`
    changes iff the setup, the trace root, the response, the policy or `x_root` changes."""
    if type(x_root) is not bytes or len(x_root) != 32:
        raise ValueError("x_root must be the 32-byte root of commit_x(...)")
    if trace_payload_root is None:
        trace_payload_root = hashlib.blake2b(x_root, digest_size=32, person=_XROOT_TPR_PERSON).digest()
    from ..runtime.pipeline import precommit_request
    return precommit_request(setup_cert_bytes, trace_payload_root, response_tokens,
                             _XROOT_TAG + x_root + bytes(policy_bytes))


# ----------------------------------------------------------------------------- witness F
def build_witness_np(layout, Y, rng):
    """F cells (2^n, 3): one bulk uniform draw (blinding / Libra / spare incl. X's old block), then the Y cells."""
    FV._check_tiles(layout, None, None, Y)
    cells = draw_cells(rng, 1 << layout.n)
    yi = cell_np(layout.off_y + np.arange(layout.B * ROWS, dtype=np.int64))
    cells[yi, 0] = np.asarray(Y, dtype=np.uint64).reshape(-1)
    cells[yi, 1:] = 0
    return cells


def batch_terms(layout, r_i, r_b, s, lam, e_w, mu):
    """(F-blocks, F-sparse, X-blocks, C-blocks) of  target + mu e_W + mu^2 z_blind  (§5.7.4)."""
    k, beta = layout.k, layout.beta
    s_j, s_b = list(s[:k]), list(s[k:])
    eqrb = FA.eq_pt(r_b, s_b)
    e_s = FA.eq_pt(r_i, s_j[:ROW_BITS])
    for t in s_j[ROW_BITS:]:
        e_s = hmul(e_s, hsub(FA.ONE, t))
    fb = [(*layout.data_block(layout.off_y, ROW_BITS + beta), hsub(FA.ZERO, hmul(eqrb, e_s)),
           tuple(s_j[:ROW_BITS]) + tuple(s_b))]
    xb = [(*layout.data_block(0, layout.d), hmul(eqrb, e_w), tuple(s))]
    sparse = []
    for l, r in enumerate(s):
        pw = lam
        for e in range(IP_DEGREE + 1):
            sparse.append((layout.libra_slot(l, e), pw))
            pw = hmul(pw, r)
    fb += FA.mask_blocks(layout, hmul(mu, mu))
    cb = [(*layout.data_block(0, layout.n - 1), mu, tuple(s_j) + tuple(r_i) + tuple(s_b))]
    return fb, sparse, xb, cb


# ----------------------------------------------------------------------------- batched opening of (F, C, X_com)
def pair_open3(comF, comC, comX, a_vec, c_vec, x_vec, tr, queries, hooks=None, record=None):
    """Prove y = <F, a> + <C, c> + <X_com, xw> with one fold chain over H = F + theta C + r_1 X_com (the two batching
    rounds fold the slots [F, F + C, F + X, F + C + X] with weights [a - c - xw, c, xw, 0])."""
    hooks = hooks or {}
    positions = []
    n = len(comF.values).bit_length() - 1
    size = 1 << n
    G, W = [None] * (4 * size), [None] * (4 * size)
    for y in range(size):
        f, c, x = comF.values[y], comC.values[y], comX.values[y]
        fc, fx = hadd(f, c), hadd(f, x)
        G[4 * y], G[4 * y + 1], G[4 * y + 2], G[4 * y + 3] = f, fc, fx, hadd(fc, x)
        W[4 * y] = hsub(hsub(a_vec[y], c_vec[y]), x_vec[y])
        W[4 * y + 1], W[4 * y + 2], W[4 * y + 3] = c_vec[y], x_vec[y], FA.ZERO
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
    theta, r1 = rs[0], rs[1]
    code = [hadd(hadd(f, hmul(theta, c)), hmul(r1, x)) for f, c, x in zip(comF.code, comC.code, comX.code)]
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
            levels = tuple(map(tuple, FA.merkle_levels(pcs.coset_leaves(code))))
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
    c_levels, c_code = hooks.get("c_levels", comC.levels), hooks.get("c_code", comC.code)
    x_levels, x_code = hooks.get("x_levels", comX.levels), hooks.get("x_code", comX.code)
    for _ in range(queries):
        pos = int.from_bytes(tr.squeeze_bytes(8), "little") % (len(comF.code) // 4)
        positions.append(pos)
        quarter = len(comF.code) // 4
        wire += enc3s([comF.code[pos + k * quarter] for k in range(4)]) + b"".join(merkle_path(comF.levels, pos))
        wire += enc3s([c_code[pos + k * quarter] for k in range(4)]) + b"".join(merkle_path(c_levels, pos))
        wire += enc3s([x_code[pos + k * quarter] for k in range(4)]) + b"".join(merkle_path(x_levels, pos))
        for lay_code, levels in layers:
            quarter = len(lay_code) // 4
            j = pos % quarter
            wire += enc3s([lay_code[j + k * quarter] for k in range(4)]) + b"".join(merkle_path(levels, j))
            pos = j
    if record is not None:
        record["positions"] = tuple(positions)
    return wire, tuple(rs), final


def pair_open3_engine(engine, comF, comC, comX, a, c, xw, tr, queries, record=None):
    """`pair_open3` (no hooks) with every sumcheck round, halving, fold and folded-layer tree on `engine`."""
    from . import cuda_engine as CE
    xp = CE.xp_of(engine)
    n = comF.cells.shape[1].bit_length() - 1
    host = lambda v: np.asarray(CE._host(engine, v))  # noqa: E731
    Fd, Cd, Xd = engine.dev(comF.rows), engine.dev(comC.rows), engine.dev(comX.rows)
    FCd = engine.fadd(Fd, Cd)
    fv = xp.ascontiguousarray(xp.stack([Fd, FCd, engine.fadd(Fd, Xd), engine.fadd(FCd, Xd)], axis=1).reshape(-1, 3))
    del Cd, FCd, Xd
    ad = engine.dev(np.ascontiguousarray(np.asarray(a).T))
    cd = engine.dev(np.ascontiguousarray(np.asarray(c).T))
    xd = engine.dev(np.ascontiguousarray(np.asarray(xw).T))
    zero = xp.zeros_like(cd)
    wv = xp.ascontiguousarray(xp.stack([engine.fsub(engine.fsub(ad, cd), xd), cd, xd, zero], axis=1).reshape(-1, 3))
    del Fd, ad, cd, xd, zero
    rounds, rs, roots, layers = [], [], [], []

    def sc_round():
        nonlocal fv, wv
        terms, _ = engine.basefold_round(fv, wv, FA.ZERO)
        hh = tuple(FA._tup(row) for row in host(terms).reshape(3, 3))
        tr.absorb_bytes(enc3s(hh))
        rounds.append(hh)
        r = tr.squeeze_fp3()
        rs.append(r)
        fv, wv = engine.halve(fv, r), engine.halve(wv, r)

    for _ in range(2):
        sc_round()
    theta, r1 = rs[0], rs[1]
    codeF, codeC, codeX = comF.code_dev, comC.code_dev, comX.code_dev
    if hasattr(engine, "cp"):
        tmp = engine.halve(xp.ascontiguousarray(xp.stack([engine.dev(codeF), engine.fadd(codeF, codeC)], axis=1)
                                                .reshape(-1, 3)), theta)                   # F + theta C
        cur = engine.halve(xp.ascontiguousarray(xp.stack([tmp, engine.fadd(tmp, codeX)], axis=1).reshape(-1, 3)), r1)
        del tmp
    else:
        # Chunked F + theta C + r1 X on CPU/HostEngine to avoid multi-GB stacked copies at n = 22.
        total = len(codeF)
        cur = np.empty((total, 3), dtype=np.uint64)
        chunk = 1 << 19
        for i in range(0, total, chunk):
            end = min(i + chunk, total)
            cf = codeF[i:end]
            cc = codeC[i:end]
            cx = codeX[i:end]
            t1 = engine.halve(np.ascontiguousarray(np.stack([cf, engine.fadd(cf, cc)], axis=1).reshape(-1, 3)), theta)
            cur[i:end] = engine.halve(np.ascontiguousarray(np.stack([t1, engine.fadd(t1, cx)], axis=1).reshape(-1, 3)), r1)
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
    final = FA._tup(host(fv).reshape(-1, 3)[0])
    del fv, wv
    last = host(cur).reshape(-1, 3)
    if any(FA._tup(v) != final for v in last):
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
    opened = [comF.open_many(positions), comC.open_many(positions), comX.open_many(positions)]
    pos_t = list(positions)
    for lay_code, levels in layers:
        quarter = len(lay_code) // 4
        pos_t = [p % quarter for p in pos_t]
        vals, sibs = CE.open_rows(engine, lay_code, levels, pos_t)
        opened.append([(enc3s([FA._tup(v) for v in vals[q]]), b"".join(bytes(sb[q]) for sb in sibs))
                       for q in range(len(pos_t))])
    parts = [wire]
    for q in range(queries):
        for op in opened:
            parts.append(op[q][0] + op[q][1])
    if record is not None:
        record["positions"] = tuple(positions)
    return b"".join(parts), tuple(rs), final


# ----------------------------------------------------------------------------- prover
def prove_bound(profile, W, Xv, Y, x_commitment, n_s=b"\x00" * 32, rng=None, hooks=None, engine=None):
    """Exact C_p-byte FPN2 relation-6 container.  `W` (B, 32, K), `X` (B, K), `Y` (B, 32) canonical F_p uint64;
    `x_commitment` is the `XCommitment` of `commit_x(layout, X)` made BEFORE the challenge (its `.root` is the pinned
    `x_root`).  The prover does NOT check Y = W X: a false statement yields bytes the verifiers reject.  `X` must be
    the committed activations unless `hooks["allow_x_mismatch"]` (adversarial vectors: the prover runs the sumcheck on
    X' but can only open the committed X).

    hooks (tests / adversarial vectors): stmt_weight_root / stmt_x_root -> bytes written into the statement (the
    proof still uses the real commitments); ip_round(l, samples) / e_w(value) / ip_adaptive -> as relation 5; pcs ->
    `pair_open3` hooks (CPU only); record (dict) -> filled with internals; timings (dict) -> per-phase milliseconds."""
    layout = layout_of(profile)
    rng = rng or SecretRng()
    hooks = hooks or {}
    if len(n_s) != 32:
        raise ValueError("N_s must be 32 bytes")
    if x_commitment.layout != layout:
        raise ValueError("x_commitment was made for a different layout")
    if x_commitment.engine is not engine:
        raise ValueError("commit_x and prove_bound must use the same engine (None = pure-Python reference)")
    if engine is not None and "pcs" in hooks:
        raise ValueError("pcs tamper hooks are CPU-only (engine=None)")
    W = np.asarray(W, dtype=np.uint64)
    Xv = np.asarray(Xv, dtype=np.uint64)
    Y = np.asarray(Y, dtype=np.uint64)
    FV._check_tiles(layout, W, Xv, Y)
    if not hooks.get("allow_x_mismatch") and not np.array_equal(Xv, x_commitment.x_values):
        raise ValueError("X differs from the committed activations (x_commitment)")
    tm = hooks.get("timings", {})
    clock = [time.perf_counter()]

    def lap(name):
        if engine is not None and hasattr(engine, "sync"):
            engine.sync()
        now = time.perf_counter()
        tm[name] = tm.get(name, 0.0) + (now - clock[0]) * 1e3
        clock[0] = now

    comC = commit_weights(layout, W, engine)
    lap("commit_weights")
    F = build_witness_np(layout, Y, rng)
    lap("witness_sample")
    comF = FA.engine_commit(engine, F) if engine is not None else pcs.zk_commit([tuple(int(v) for v in r) for r in F])
    lap("commit_witness")
    comX = x_commitment.com
    Fa = np.ascontiguousarray(F.T)
    Xa = np.ascontiguousarray(x_commitment.cells.T)
    Fip = Fa.copy()                                    # the inner-product view: X from X_com, Y / Libra from F
    xi = cell_np(np.arange(layout.B * layout.K, dtype=np.int64))
    if hooks.get("allow_x_mismatch"):                  # the cheater runs the sumcheck on its own X'
        Fip[:, xi] = 0
        Fip[0, xi] = Xv.reshape(-1)
    else:
        Fip[:, xi] = Xa[:, xi]
    hdr = header(profile_id(profile))
    stmt = (comF.root + hooks.get("stmt_weight_root", comC.root) + hooks.get("stmt_x_root", comX.root)
            + bytes(n_s))
    tr = Fpn2Transcript()
    tr.absorb_bytes(hdr)
    tr.absorb_bytes(stmt)
    ip_wire, r_i, r_b, s, lam, e_w, _ = FV.ip_prove(tr, layout, Fip, W, hooks)
    lap("ip_sumcheck")
    zb = FA.z_blind_np(layout, Fa)
    del Fa, Xa, Fip
    if "record" not in hooks:
        del F
    tr.absorb_bytes(enc3(zb))
    mu = tr.squeeze_fp3()
    fb, sparse, xb, cb = batch_terms(layout, r_i, r_b, s, lam, e_w, mu)
    a_vec = FA.weights_vector_np(layout.n, fb, sparse)
    x_vec = FA.weights_vector_np(layout.n, xb)
    c_vec = FA.weights_vector_np(layout.n, cb)
    lap("batch_weights")
    rec = {}
    if engine is not None:
        pcs_wire, rs, final = pair_open3_engine(engine, comF, comC, comX, a_vec, c_vec, x_vec, tr,
                                                layout.queries, rec)
    else:
        pcs_wire, rs, final = pair_open3(comF, comC, comX, [FV._tup(v) for v in a_vec.T],
                                         [FV._tup(v) for v in c_vec.T], [FV._tup(v) for v in x_vec.T], tr,
                                         layout.queries, hooks.get("pcs"), rec)
    lap("pair_open")
    if "record" in hooks:
        hooks["record"].update(F=F, Xcells=x_commitment.cells, a=a_vec, c=c_vec, xw=x_vec, rs=rs, s=s, r_i=r_i,
                               r_b=r_b, lam=lam, mu=mu, e_w=e_w, z_blind=zb, final=final,
                               positions=rec.get("positions"))
    body = stmt + ip_wire + enc3(zb) + pcs_wire
    if len(body) != layout.body_used:
        raise AssertionError(f"layout mismatch {len(body)} != {layout.body_used}")
    out = hdr + body
    if len(out) > profile["c_p"]:
        raise ValueError("proof exceeds C_p")
    return out + b"\x00" * (profile["c_p"] - len(out))


def byte_budget(n, queries, k=10):
    lay = BoundLayout(n, queries, k)
    return {"n_vars": n, "queries": queries, "K": lay.K, "tiles": lay.B, "rows": lay.B * ROWS, "macs": lay.macs,
            "body": lay.body_used, "container_min": 64 + lay.body_used}


# ----------------------------------------------------------------------------- ZK rank condition 
# JOINT linear view over the two committed private vectors (F, X_com) of one relation-6 transcript.  W is PUBLIC, so for
# fixed challenges every plaintext value of the container is an F_p3-linear functional of (F, X_com) (the public C
# terms dropped):
#   claims : the Libra sum g_sum; ALL 4 d inner-product round samples (rows split: X part on X_com, Y + Libra on F);
#            z_blind (F's blinding half).  e_W is a function of the public W only; e_X, e_Y, e_G are NEVER sent.
#   pcs    : the 3 (n + 2) batched sumcheck samples h_i(0..2) and G~(r) (F-part = sum of the 4 slot weights, X-part =
#            slots 2 + 3), per query the 4 layer-0 coset values of F's codeword (F rows) AND of X_com's codeword
#            (X rows), and the 4 coset values of every folded layer of H = F + theta C + r_1 X_com (F row + r_1 * X row).
# Coordinates: X(b, j), Y(b, i) are F_p coordinates; every Libra / blinding / spare cell of F and of X_com is a uniform F_p3
# coordinate (3 F_p columns).  `require_joint_mask_vec_bound` checks im(A) <= im(B) over F_p (rank_B == rank_BA) with A
# the honest witness manifold (Y = W X): one column per X(b, j) = X_com's X cell plus sum_i W[b][i][j] times F's Y(b, i).
def _escale_rows(row, c):
    return FA._escale(row, c)


def _split_rows(layout, row, xcells):
    """Relation-5-shaped dense row over a single vector -> (F-part, X_com-part): the X block goes to X_com."""
    rowx = X.zeros_e(1 << layout.n)
    rowx[:, xcells] = row[:, xcells]
    rowf = row.copy()
    rowf[:, xcells] = 0
    return rowf, rowx


def pcs_rows_bound_np(layout, rec, with_public=False):
    """Yield (label, rowF, rowX, rowC) for every value the opening reveals; rowC (public C part) only with
    `with_public` (numeric validation), else None."""
    n, size = layout.n, 1 << layout.n
    a = np.asarray(rec["a"], dtype=np.uint64)
    c = np.asarray(rec["c"], dtype=np.uint64)
    xw = np.asarray(rec["xw"], dtype=np.uint64)
    if a.shape[0] != 3:
        a, c, xw = a.T, c.T, xw.T
    N = n + 2
    wv = X.zeros_e(4 * size)
    wv[:, 0::4] = X.esub(X.esub(a, c), xw)
    wv[:, 1::4] = c
    wv[:, 2::4] = xw
    rs = list(rec["rs"])
    r1 = rs[1]

    def to_rows(g):
        g = g.reshape(3, size, 4)
        parts = [np.ascontiguousarray(g[:, :, k]) for k in range(4)]
        rf = X.eadd(X.eadd(parts[0], parts[1]), X.eadd(parts[2], parts[3]))
        rx = X.eadd(parts[2], parts[3])
        rc = X.eadd(parts[1], parts[3]) if with_public else None
        return rf, rx, rc

    b = np.arange(1 << N, dtype=np.int64)
    for i in range(N):
        lo, hi = wv[:, 0::2], wv[:, 1::2]
        step = X.esub(hi, lo)
        high = lo
        for t in range(3):
            low = X.eq_table(list(rs[:i]) + [hf(t)])
            g = X.emul(low[:, b & ((2 << i) - 1)], high[:, b >> (i + 1)])
            yield (f"pcs_round_{i}_{t}", *to_rows(g))
            high = X.eadd(high, step)
        wv = FA._fold(wv, rs[i])
    yield ("pcs_final", *to_rows(X.eq_table(rs)))
    rest = rs[2:]
    length = pcs.RATE << n
    omega = pcs.root_of_unity(length)
    seen = set()
    zero = X.zeros_e(size)
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
                    eqrow = X.eq_table(pt)
                    if t == 0:
                        yield (f"query_t0_j{j}_k{k}_F", eqrow, zero, None)
                        yield (f"query_t0_j{j}_k{k}_X", zero, eqrow, None)
                    else:
                        rc = _escale_rows(eqrow, rec["rs"][0]) if with_public else None
                        yield (f"query_t{t}_j{j}_k{k}", eqrow, _escale_rows(eqrow, r1), rc)
            pos = j


def bound_claim_rows(layout, rec, W):
    """[(label, rowF, rowX)]: g_sum, the 4 d inner-product samples, z_blind (all linear in (F, X_com))."""
    xcells = cell_np(np.arange(layout.B * layout.K, dtype=np.int64))
    for label, row in FV.vec_claim_rows(layout, rec, W):
        rf, rx = _split_rows(layout, row, xcells)
        yield label, rf, rx


def bound_observations(layout, rec, W, claims=True, pcs_view=True):
    if claims:
        for label, rf, rx in bound_claim_rows(layout, rec, W):
            yield label, rf, rx
    if pcs_view:
        for label, rf, rx, _ in pcs_rows_bound_np(layout, rec):
            yield label, rf, rx


class BoundCoordinates:
    """F_p coordinate model of the honest relation-6 vectors (F, X_com)."""

    def __init__(self, layout):
        B, K, n = layout.B, layout.K, layout.n
        self.layout = layout
        self.x_cells = cell_np(np.arange(B * K, dtype=np.int64))
        self.y_cells = cell_np(layout.off_y + np.arange(B * ROWS, dtype=np.int64))
        self.libra = np.array(sorted(layout.libra_slot(l, e) for l in range(layout.d) for e in range(IP_DEGREE + 1)),
                              dtype=np.int64)
        idx = np.arange(1 << n, dtype=np.int64)
        blind = idx[((idx >> 1) & 1) == 0]
        data = idx[((idx >> 1) & 1) == 1]
        self.f_pool = np.concatenate([np.setdiff1d(blind, self.libra), np.setdiff1d(data, self.y_cells)])
        self.f_xblock = np.intersect1d(self.f_pool, self.x_cells)          # F's old X block (uniform spare)
        self.x_pool = np.concatenate([blind, np.setdiff1d(data, self.x_cells)])


def _proj_bound(rowf, rowx, bf, bx, W, coords):
    """(3, nb + a) block of one observation: [F fp3 columns | X_com fp3 columns | honest X coordinates]."""
    Wt = np.asarray(W, dtype=np.uint64)
    Bn, _, K = Wt.shape
    ry = rowf[:, coords.y_cells].reshape(3, Bn, ROWS)
    acc = rowx[:, coords.x_cells].reshape(3, Bn, K).copy()
    for i in range(ROWS):
        for comp in range(3):
            acc[comp] = X.add(acc[comp], X.mul(ry[comp, :, i:i + 1], Wt[:, i, :]))
    parts = []
    if len(bf):
        parts.append(FA._fp3_columns(rowf, bf))
    if len(bx):
        parts.append(FA._fp3_columns(rowx, bx))
    parts.append(acc.reshape(3, Bn * K))
    return np.concatenate(parts, axis=1)


def require_joint_mask_vec_bound(layout, rec, W, seed=0, drop_libra=False, drop_x_blinding=False,
                                 zero_f_x_block=False, raise_on_fail=True, oversample=1.0, claims=True,
                                 pcs_view=True):
    """im(A) <= im(B) over F_p for the joint linear view of relation 6 over (F, X_com).

    B = every Libra slot of F (unless `drop_libra`) plus a seeded sample of F's blinding / spare cells and of
    X_com's blinding / spare cells; im A <= im B_subset implies im A <= im B, so subsampling B (never the observation
    rows) can only make the check fail, never pass wrongly.  Negative controls that MUST fail: `drop_libra`,
    `drop_x_blinding` (X_com gets no random coordinates) and `zero_f_x_block` (F's old X block is NOT random: removed
    from B).  `rec` is `prove_bound(..., hooks={"record": rec})`.  Returns the rank report; raises
    ValueError("under-rank mask") when it fails and raise_on_fail.  `layout` may also be the profile dict."""
    import random as _random
    if isinstance(layout, dict):
        layout = layout_of(layout)
    t0 = time.perf_counter()
    coords = BoundCoordinates(layout)
    n_obs = (2 + layout.d * (IP_DEGREE + 1) if claims else 0)
    if pcs_view:
        n_obs += 3 * (layout.n + 2) + 1 + _count_queries_bound(layout, rec["positions"])
    rnd = _random.Random(seed)
    budget = int(oversample * n_obs) + 64
    f_pool = coords.f_pool
    if zero_f_x_block:
        f_pool = np.setdiff1d(f_pool, coords.f_xblock)
    bf = sorted(rnd.sample([int(v) for v in f_pool], min(budget, len(f_pool))))
    bx = [] if drop_x_blinding else sorted(rnd.sample([int(v) for v in coords.x_pool], min(budget, len(coords.x_pool))))
    if not drop_libra:
        bf = sorted(set(bf) | set(int(v) for v in coords.libra))
    elif drop_libra:
        bf = sorted(set(bf) - set(int(v) for v in coords.libra))
    bf, bx = np.array(bf, dtype=np.int64), np.array(bx, dtype=np.int64)
    blocks, labels = [], []
    for label, rf, rx in bound_observations(layout, rec, W, claims, pcs_view):
        blocks.append(_proj_bound(rf, rx, bf, bx, W, coords))
        labels.append(label)
    if len(labels) != n_obs:
        raise AssertionError(f"observation count {len(labels)} != {n_obs}")
    matrix = np.concatenate(blocks, axis=0)
    nb = 3 * (len(bf) + len(bx))
    t1 = time.perf_counter()
    rank_b, rank_ab = FA.fp_rank_split(matrix, nb)
    report = {"check": "joint_vec_bound", "observations": n_obs,
              "claim_observations": (2 + layout.d * (IP_DEGREE + 1)) if claims else 0,
              "fp_rows": int(matrix.shape[0]), "b_columns": nb, "a_columns": int(layout.B * layout.K),
              "b_columns_F": 3 * len(bf), "b_columns_X": 3 * len(bx),
              "rank_B": rank_b, "rank_BA": rank_ab, "holds": rank_b == rank_ab,
              "drop_libra": drop_libra, "drop_x_blinding": drop_x_blinding, "zero_f_x_block": zero_f_x_block,
              "build_s": round(t1 - t0, 2), "rank_s": round(time.perf_counter() - t1, 2)}
    if raise_on_fail and not report["holds"]:
        raise ValueError(f"under-rank mask: {report}")
    return report


def _count_queries_bound(layout, positions):
    """Number of PCS query observations: 4 per unique (t, j) coset, layer 0 twice (F and X_com)."""
    length = pcs.RATE << layout.n
    seen = set()
    for pos in positions:
        for t in range(layout.folds):
            j = pos % (length >> (2 * t + 2))
            seen.add((t, j))
            pos = j
    return 4 * len(seen) + 4 * len({j for (t, j) in seen if t == 0})


# ----------------------------------------------------------------------------- numeric validation of the rows
def bound_wire_values(layout, data):
    """Plaintext claim values of a relation-6 container in observation order: [g_sum, h_l(t)..., z_blind]."""
    from .fpn2 import dec3
    off = 64 + STMT
    vals = [dec3(bytes(data[off:off + 24]))]
    off += 24
    for _ in range(layout.d * (IP_DEGREE + 1)):
        vals.append(dec3(bytes(data[off:off + 24])))
        off += 24
    off += 24                                               # e_W (public)
    vals.append(dec3(bytes(data[off:off + 24])))
    return vals


def eval_row(row, cells):
    """F_p3 value  sum_c row[:, c] * cells[:, c]  of a dense functional over (3, 2^n) arrays."""
    return tuple(int(v) for v in X.esum(X.emul(np.asarray(row, dtype=np.uint64), np.asarray(cells, dtype=np.uint64))))


# ----------------------------------------------------------------------------- re-proof leakage (M2-3 style)
def section_map(layout, c_p):
    """[(name, start, end, public)] covering [0, c_p): header | root_0 (private) weight_root, x_root (public: pinned)
    N_s (public) | inner-product sumcheck + z_blind + PCS rounds (private) | per query: F opening (private), C opening
    (public), X_com opening (private), folded layers (private) | padding (public)."""
    secs = [("header", 0, 64, True), ("stmt.root_0", 64, 96, False), ("stmt.weight_root", 96, 128, True),
            ("stmt.x_root", 128, 160, True), ("stmt.N_s", 160, 192, True)]
    q_start = 64 + layout.body_used - layout.queries * layout.query_len
    secs.append(("ip_zb_pcs_rounds", 192, q_start, False))
    pos = q_start
    open0 = 96 + 32 * layout.depth(0)
    for q in range(layout.queries):
        secs.append((f"q{q}.F", pos, pos + open0, False))
        secs.append((f"q{q}.C", pos + open0, pos + 2 * open0, True))
        secs.append((f"q{q}.X", pos + 2 * open0, pos + 3 * open0, False))
        pos += 3 * open0
        for t in range(1, layout.folds):
            ln = 96 + 32 * layout.depth(t)
            secs.append((f"q{q}.L{t}", pos, pos + ln, False))
            pos += ln
    if pos != 64 + layout.body_used:
        raise AssertionError("section map does not sum to body_used")
    secs.append(("padding", pos, c_p, True))
    return secs


def shared_blocks(a, b, layout, c_p, block=64):
    """Re-proof leakage report over 64-byte blocks of two containers of the SAME statement, split public / private
    by `section_map` (aligned blocks + unaligned 64-byte windows at every 8-byte offset of the private bytes)."""
    secs = section_map(layout, c_p)
    priv = [(s, e) for _, s, e, pub in secs if not pub]

    def private(off):
        return all(not pub for _, s, e, pub in secs if s < off + block and off < e)

    def windows(x):
        out = set()
        for s, e in priv:
            for o in range(s, e - block + 1, 8):
                out.add(bytes(x[o:o + block]))
        return out

    offs = range(0, len(a), block)
    priv_bytes = sum(e - s for s, e in priv)
    return {"block_bytes": block, "private_bytes": priv_bytes, "public_bytes": len(a) - priv_bytes,
            "private_aligned_blocks": sum(1 for o in offs if private(o)),
            "private_aligned_shared": sum(1 for o in offs if private(o) and a[o:o + block] == b[o:o + block]),
            "private_window_set_intersection": len(windows(a) & windows(b)),
            "total_aligned_shared_incl_public": sum(1 for o in offs if a[o:o + block] == b[o:o + block])}
