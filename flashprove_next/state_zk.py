# SPDX-License-Identifier: Apache-2.0
"""In-circuit multi-turn state chain (specs/01 §6.4, §5.5) for flashprove_next.reply_zk.

Relation (turn i; statement-public: v_rel, model_id, sess, turn i, St_prev, St_new, client nonce N_c, server nonce N_s;
prover-hidden: hist_i, r_i (4 field elements = 256 bits) and, for i > 0, the opening of St_prev):

    St_i   = H(DOM_ST;  v_rel, model_id[4], sess[4], i, St_prev[4], Hx[4], Hy[4], Hhist[4], r_i[4])      (30 elements)
    Hx     = H(DOM_X;   req_tokens ++ enc(N_c))        public, recomputed from the statement (never a statement field)
    Hy     = H(DOM_Y;   vis_tokens ++ enc(N_s))        public, recomputed from the statement
    Hhist  = H(DOM_HIST; tok[b .. c))                  hidden CoT of THIS turn, read IN-CIRCUIT from the token column
    sess   = H(DOM_SESS; enc(N_c^(1)))                 St_0 = H(DOM_GEN; v_rel, sess)   (both public)

Every sponge here is the Poseidon2 sponge of flashprove_next.poseidon2 (rate 8, capacity lane 10 = domain, lane 11 = TAG_SPONGE, zero-padded
message blocks) with the MESSAGE LENGTH in capacity lane 8 (lane 9 = 0): [5] and [5, 0] never collide, the hidden
history length is part of the preimage, and variable-length Hx / Hy need no extra padding rule.  Nonces are encoded
injectively as 32-bit little-endian limbs (a 128-bit nonce is 4 elements, a 256-bit one 8; never reduced mod p).

Domain separators 33..38 (already allocated: 3..6, modelcommit 21..23, mc_zk 24, tc_zk 25..27, 28..32 reserved).

Turn i > 0 -- opening of St_prev and the exact rendering of the previous hidden history
  * St_prev = H(DOM_ST; v_rel, model_id, sess, i-1, St_pp, Hx', Hy', Hhist', r')   with HIDDEN St_pp = St_{i-2}, Hx', Hy',
    r' (free capacity-less rate lanes of the sponge rows: no extra columns) and the SAME public v_rel, model_id, sess as
    this turn (they are constants of both messages), turn index i-1 (constant).
  * RENDERING (the choice, v_cap-fixed): the previous turn's hidden history tokens hist_{i-1} (its hidden CoT; the
    previous turn's tool returns etc. are out of scope here) are re-injected IMMEDIATELY AFTER THE HIDDEN SYSTEM PROMPT:
        [0, a)               hidden system prompt (a = sys_len, committed in M^vol)
        [a, a + l')          hist_{i-1}, l' = len(hist_{i-1}) HIDDEN, 0 <= l' <= HIST_CAP, l' = 0 at turn 0
        [a + l', b)          public request tokens, b = a + l' + len(req)
        [b, c)               hidden CoT of this turn, c = b + n_reason (public usage count)
        [c, T)               public visible reply, T = c + n_vis
    Hhist' = H(DOM_HIST; hist_{i-1}) (length l' in lane 8) is recomputed in-circuit from the tokens at [a, a + l') by
    LogUp of (a + k, h'_k) into the token table 'rz.tokt', k < l' (hidden boolean monotone mask m'_k, sum m' = l').
  * Hx', Hy' (public x/y digests of the previous turn) stay HIDDEN (the verifier of turn i need not know them: St_prev
    commits to them).  Scope: specs/01 §6.4(b) (the history part of THIS request equals the previous
    request / reply cores) is not enforced by this gadget; it requires tying Hx' / Hy' to the public request tokens.
  * l' is a hidden scalar 'st.np' owned by this module (declare_scalars); reply_zk.token_chain must shift its request /
    CoT / reply bases by it (reply_zk does).

Circuit shape.  One Poseidon2 region on a PADDED space [g x b] (group g = one sponge, row b = one block):
    g0 Hhist (this turn), g1 St_new, g2 Hhist' and g3 St_prev (turn > 0 only).
Padded (not the non-padded PoseRegion default) because TZ.Scalars views (hidden preimage lanes, l') need a padded
target.  Hidden message elements are NEVER committed twice: they are the free I(k) lanes of the sponge rows, constrained
only through (a) the public constants (pt(row) selects), (b) digest hand-offs by shifted O(k) (as hsel_zk R.rtH), (c) the
hidden-token lanes through views of the hidden token columns hc / hp on the space 'stH' (8 lane views with constant
lane bits, block index -> PS b axis), and (d) hidden scalars for r, St_pp, Hx', Hy', r', l'.  The columns on 'stH' are
read by lookups into 'rz.tokt' (multiplicities: reply_zk computes them AFTER state_chain, `Z._logup_mult`).
"""
import hashlib
import json
import time

import numpy as np

from . import gadgets as G
from . import mc_zk as MZ
from . import poseidon2 as PS
from . import tc_zk as TZ
from . import zk_flat as Z
from .field import P, CPU
from .zk_flat import Ex

DOM_ST, DOM_X, DOM_Y, DOM_HIST, DOM_GEN, DOM_SESS = 33, 34, 35, 36, 37, 38
ST_LEN = 30                  # v_rel, model_id[4], sess[4], turn, St_prev[4], Hx[4], Hy[4], Hhist[4], r[4]
HIST_CAP = 16                # default profile cap on the (hidden) history length; power of two >= 8


# ============================================================================ host helpers (prover / verifier / server)
def _lg(n):
    return max(0, int(n - 1).bit_length())


def sponge_trace(msg, domain, length=None):
    """Host sponge with the message length in capacity lane 8.  Returns (digest (4 ints), [(in_state, out_state)] per block;
    states are 12 Python ints)."""
    msg = [int(x) % P for x in msg]
    ln = len(msg) if length is None else int(length)
    nb = max(1, (len(msg) + 7) // 8)
    msg = msg + [0] * (8 * nb - len(msg))
    st = [0] * 12
    st[8], st[10], st[11] = ln % P, domain % P, PS.TAG_SPONGE
    trace = []
    for b in range(nb):
        st = [(st[k] + msg[8 * b + k]) % P if k < 8 else st[k] for k in range(12)]
        pin = list(st)
        out = PS.permute(np.array(st, dtype=np.uint64).reshape(12, 1))
        st = [int(v) for v in out[:, 0]]
        trace.append((pin, list(st)))
    return tuple(st[:4]), trace


def sponge(msg, domain):
    return sponge_trace(msg, domain)[0]


def nonce_fe(nonce):
    """Injective encoding of a nonce (bytes) as 32-bit little-endian limbs (each < 2^32 < p)."""
    b = bytes(nonce)
    b = b + b"\x00" * (-len(b) % 4)
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def v_rel_of(rv):
    """field element of the relation version string (reply_zk.VERSION): first 7 bytes of its blake2b, little endian."""
    return int.from_bytes(hashlib.blake2b(rv.encode(), digest_size=7, person=b"fpn-state-v").digest(), "little")


def sess_of(nonce_c1):
    """sess = H(DOM_SESS; N_c^(1))."""
    return sponge(nonce_fe(nonce_c1) + [len(bytes(nonce_c1))], DOM_SESS)


def genesis(v_rel, sess):
    """St_0 = H(DOM_GEN; v_rel, sess) (public)."""
    return sponge([int(v_rel)] + list(sess), DOM_GEN)


def x_digest(req_tokens, nonce_c):
    """Hx = H(DOM_X; request core tokens ++ enc(N_c) ++ len(N_c))."""
    return sponge([int(t) for t in req_tokens] + nonce_fe(nonce_c) + [len(bytes(nonce_c))], DOM_X)


def y_digest(vis_tokens, nonce_s):
    """Hy = H(DOM_Y; visible reply tokens ++ enc(N_s) ++ len(N_s))."""
    return sponge([int(t) for t in vis_tokens] + nonce_fe(nonce_s) + [len(bytes(nonce_s))], DOM_Y)


def hist_digest(tokens, cap=HIST_CAP):
    """Hhist = H(DOM_HIST; hidden history tokens zero-padded to the profile cap), true length in capacity lane 8.  The
    circuit absorbs cap/8 blocks (the hidden length cannot change the circuit shape), so the host MUST pad to the same cap:
    St depends on the cap, which is therefore a profile constant fixed for a whole session."""
    toks = [int(t) for t in tokens]
    assert len(toks) <= cap and cap >= 8 and cap & (cap - 1) == 0
    return sponge_trace(toks + [0] * (cap - len(toks)), DOM_HIST, len(toks))[0]


def state_msg(v_rel, model_id, sess, turn, st_prev, hx, hy, hhist, r):
    m = [int(v_rel)] + list(model_id) + list(sess) + [int(turn)] + list(st_prev) + list(hx) + list(hy) + list(hhist) + list(r)
    assert len(m) == ST_LEN
    return m


def state_root(v_rel, model_id, sess, turn, st_prev, hx, hy, hhist, r):
    """St_i = H(DOM_ST; v_rel, model_id, sess, i, St_prev, Hx, Hy, Hhist, r_i)."""
    return sponge(state_msg(v_rel, model_id, sess, turn, st_prev, hx, hy, hhist, r), DOM_ST)


def new_turn(v_rel, model_id, sess, turn, st_prev, req_tokens, vis_tokens, nonce_c, nonce_s, hist, r, cap=HIST_CAP):
    """Server / prover helper: St_turn and the bundle the NEXT turn needs to open it (its `prev` witness; this is the
    plaintext of specs/01 §6.4's blob_i = AEAD(hist_i, r_i, ...))."""
    hx, hy = x_digest(req_tokens, nonce_c), y_digest(vis_tokens, nonce_s)
    st = state_root(v_rel, model_id, sess, turn, st_prev, hx, hy, hist_digest(hist, cap), r)
    return st, {"hist": [int(t) for t in hist], "st_pp": tuple(st_prev), "hx": hx, "hy": hy, "r": tuple(r)}


def statement(v_rel, model_id, sess, turn, st_prev, st_new, nonce_c, nonce_s, hist_cap=HIST_CAP):
    """The public part ('st' key of the reply statement).  Hx / Hy are NOT fields: the circuit recomputes them."""
    return {"v_rel": int(v_rel), "model_id": [int(x) for x in model_id], "sess": [int(x) for x in sess], "turn": int(turn),
            "St_prev": [int(x) for x in st_prev], "St_new": [int(x) for x in st_new],
            "N_c": bytes(nonce_c).hex(), "N_s": bytes(nonce_s).hex(), "hist_cap": int(hist_cap)}


def check_statement(stmt, req_tokens=None, n_reason=0):
    """Public checks outside the circuit (the verifier runs them; state_chain repeats the structural ones):
      * well-formed field elements / nonce lengths (N_c >= 128 bit, N_s = 256 bit), turn >= 0, hist cap;
      * turn 0: St_prev == genesis(v_rel, sess) and sess == sess_of(N_c) (N_c is the FIRST client nonce);
      * n_reason <= hist_cap (completeness limit of the profile).
    For turn > 0 the verifier can only check that St_prev is the anchor the client stored (specs/01 §5.5)."""
    s = stmt["st"] if "st" in stmt else stmt
    for k in ("sess", "St_prev", "St_new", "model_id"):
        if len(s[k]) != 4 or any(not (0 <= int(x) < P) for x in s[k]):
            raise Z.ProofError(f"state statement: bad {k}")
    if int(s["turn"]) < 0 or not (0 <= int(s["v_rel"]) < P):
        raise Z.ProofError("state statement: bad turn / v_rel")
    nc, ns = bytes.fromhex(s["N_c"]), bytes.fromhex(s["N_s"])
    if len(nc) < 16 or len(ns) != 32:
        raise Z.ProofError("state statement: nonce length (N_c >= 128 bit, N_s = 256 bit)")
    cap = int(s["hist_cap"])
    if cap < 8 or cap & (cap - 1):
        raise Z.ProofError("state statement: hist_cap must be a power of two >= 8")
    if int(n_reason) > cap:
        raise Z.ProofError("state statement: n_reason exceeds the history cap")
    if int(s["turn"]) == 0:
        if tuple(s["sess"]) != sess_of(nc):
            raise Z.ProofError("state statement: sess != H(DOM_SESS; N_c) at turn 0")
        if tuple(s["St_prev"]) != genesis(s["v_rel"], s["sess"]):
            raise Z.ProofError("state statement: St_prev is not the genesis anchor at turn 0")


def n_prev(wit):
    """prover: l' = len(hist_{i-1}) (0 at turn 0)."""
    p = (wit or {}).get("st", {}).get("prev") if wit is not None else None
    return len(p["hist"]) if p else 0


def declare_scalars(SC, stmt, wit):
    """Add the hidden scalars of this module to the circuit's TZ.Scalars (idempotent).  reply_zk.token_chain calls it right
    after creating SC so that it can shift its bases by 'st.np'.  Names: st.np (turn > 0 only), st.r0..3 and, turn > 0,
    st.pp0..3 (St_{i-2}), st.hx0..3, st.hy0..3, st.rp0..3."""
    if "st.r0" in SC.c:
        return
    pv = wit is not None
    w = (wit or {}).get("st", {})
    turn = int(stmt["st"]["turn"])

    def add(name, v):
        SC.add(name, (int(v), 0, 0) if pv else None)
    for j in range(4):
        add(f"st.r{j}", w["r"][j] if pv else 0)
    if turn > 0:
        add("st.np", n_prev(wit))
        pr = w["prev"] if pv else None
        for j in range(4):
            add(f"st.pp{j}", pr["st_pp"][j] if pv else 0)
            add(f"st.hx{j}", pr["hx"][j] if pv else 0)
            add(f"st.hy{j}", pr["hy"][j] if pv else 0)
            add(f"st.rp{j}", pr["r"][j] if pv else 0)


def start_ex(SC, stmt, sp, a_name="rz.a"):
    """Expression a + l' (start of the public request) on space sp: reply_zk.token_chain's base of req / CoT / reply."""
    e = SC.on(a_name, sp)
    if int(stmt["st"]["turn"]) > 0:
        e = e + SC.on("st.np", sp)
    return e


# ============================================================================ the gadget
def state_chain(ctx, prf, stmt, SP_tok, tok_col, masks, a_scalar_name, SC, wit=None, hist_rel_offs=None):
    """Constrain St_new = St_i (module docstring) from the hidden token column.

    stmt     reply statement with stmt['st'] (state_zk.statement) and the public stmt['req_tokens'], ['n_reason'],
             ['vis_tokens'] (Hx / Hy are recomputed from them and the nonces, on both prover and verifier paths).
    prf      profile: prf.HIST_CAP (power of two >= 8, same for the whole session) must equal stmt['st']['hist_cap'].
    SP_tok, tok_col, masks   the token chain space / hidden column / prefix masks (reply_zk.token_chain); the column is
             read through the table 'rz.tokt' (G.wire_table must have been declared); masks are only sanity-checked.
    a_scalar_name   name of the hidden scalar a (sys_len) in SC ('rz.a'), SC the TZ.Scalars instance.
    wit      prover: {'tokens', 'sys_len', 'st': {'r': 4, 'prev': None | {'hist','st_pp','hx','hy','r'}, 'hist_override'?}}.
             `hist_override` (forgery tests only): committed hist tokens fed to the sponge instead of tok[b..c).
    Returns info {groups, rows, cells...}.  Must be called BEFORE reply_zk's `Z._logup_mult(..., 'rz.tokt')`."""
    pv = wit is not None
    s = stmt["st"]
    turn = int(s["turn"])
    req = [int(t) for t in stmt["req_tokens"]]
    n_r = int(stmt["n_reason"]) if hist_rel_offs is None else len(hist_rel_offs)
    vis = [int(t) for t in stmt["vis_tokens"]]
    cap = int(s["hist_cap"])
    if hasattr(prf, "HIST_CAP") and int(prf.HIST_CAP) != cap:
        raise Z.ProofError("state statement: hist_cap differs from the profile's HIST_CAP")
    check_statement(stmt, req, n_r)
    assert len(masks) == 4
    nc, ns = bytes.fromhex(s["N_c"]), bytes.fromhex(s["N_s"])
    hx = x_digest(req, nc)                              # recomputed: the statement carries no digests
    hy = y_digest(vis, ns)
    # model_id and the relation version are the REPLY statement's (one source of truth); the st copy must agree
    model_id = [int(x) for x in stmt.get("model_id", s["model_id"])]
    if model_id != [int(x) for x in s["model_id"]]:
        raise Z.ProofError("state statement: model_id differs from the reply statement's")
    if "rv" in stmt and int(s["v_rel"]) != v_rel_of(stmt["rv"]):
        raise Z.ProofError("state statement: v_rel is not derived from the reply statement's relation version")
    sess, st_prev = [int(x) for x in s["sess"]], [int(x) for x in s["St_prev"]]
    v_rel = int(s["v_rel"])
    cells0 = int(sum(v.N for v in ctx.cir.vecs))
    hasp = turn > 0
    declare_scalars(SC, stmt, wit)
    one = Ex.const(1)
    cst = lambda c: Ex.const(int(c) % P)

    # ---------------------------------------------------------------- history token columns on 'stH'
    KB = max(5, _lg(cap))
    HS = ctx.space("stH", [("k", KB)])
    NH = HS.n
    a_ex = SC.on(a_scalar_name, HS)
    sys_len = int(wit["sys_len"]) if pv else 0
    npv = n_prev(wit) if pv else 0
    toks = list(wit["tokens"]) if pv else []
    tokat = lambda i: int(toks[i]) if 0 <= i < len(toks) else 0
    b0 = sys_len + npv + len(req)                       # start of post-request sequence (prover)
    st_w = wit["st"] if pv else None
    rel_list = list(range(n_r)) if hist_rel_offs is None else [int(x) for x in hist_rel_offs]
    hist_cur = None
    if pv:
        hist_cur = list(st_w.get("hist_override") or [tokat(b0 + r_off) for r_off in rel_list])
        assert len(hist_cur) == n_r
    hc_v = None
    if pv:
        hc_v = np.zeros(NH, dtype=np.int64); hc_v[:n_r] = hist_cur
    hc = HS.col("hc", hc_v)
    pc_v = None
    if pv:
        pc_v = np.zeros(NH, dtype=np.int64); pc_v[:n_r] = [b0 + r_off for r_off in rel_list]
    pc = HS.col("pc", pc_v)
    selc = HS.lt("k", n_r)
    if hist_rel_offs is None:
        off_ex = HS.idx("k")
    else:
        hoff_arr = np.zeros(NH, dtype=np.int64); hoff_arr[:n_r] = rel_list
        off_ex = HS.pub("hoff", hoff_arr).ex
    pos_c = a_ex + len_ex(SC, HS, hasp) + cst(len(req)) + off_ex
    HS.zero(pc.ex - pos_c, sel=selc, tag="st.hc.pos")
    HS.zero(hc.ex, sel=HS.rng("k", n_r, NH), tag="st.hc.zero")
    if n_r:
        sv = np.zeros(NH, dtype=np.int64); sv[:n_r] = 1
        G.wire_lookup(ctx, "rz.tokt", HS, [pc.ex, hc.ex], [pc_v, hc_v] if pv else None, sel=selc,
                      sel_vals=sv if pv else None)
    hp = mp = None
    if hasp:
        prev = st_w["prev"] if pv else None
        hp_v = mp_v = pp_v = None
        if pv:
            hp_v = np.zeros(NH, dtype=np.int64); hp_v[:npv] = prev["hist"]
            mp_v = np.zeros(NH, dtype=np.int64); mp_v[:npv] = 1
            pp_v = np.zeros(NH, dtype=np.int64); pp_v[:npv] = [sys_len + k for k in range(npv)]
            assert npv <= cap, "previous history longer than HIST_CAP"
        hp = HS.col("hp", hp_v)
        mp = HS.col("mp", mp_v)
        pp = HS.col("pp", pp_v)
        HS.zero(mp.ex * (mp.ex - one), tag="st.mp.bool")
        dm = mp.ex - mp.shifted(1)
        HS.zero(dm * (dm - one), sel=HS.lt("k", NH - 1), tag="st.mp.mono")          # non-increasing 1..1 0..0
        HS.zero(mp.ex, sel=HS.rng("k", cap, NH), tag="st.mp.cap")                    # l' <= HIST_CAP
        HS.sumzero(mp.ex - SC.on("st.np", HS) * HS.ind(k=0), tag="st.mp.sum")        # sum m' = l'
        HS.zero(hp.ex * (one - mp.ex), tag="st.hp.zero")                             # zero beyond l'
        HS.zero(mp.ex * (pp.ex - a_ex - HS.idx("k")), tag="st.hp.pos")               # positions a + k
        G.wire_lookup(ctx, "rz.tokt", HS, [pp.ex, hp.ex], [pp_v, hp_v] if pv else None, sel=mp.ex,
                      sel_vals=mp_v if pv else None)

    # ---------------------------------------------------------------- Poseidon region
    groups = ["hc", "st"] + (["hp", "sp"] if hasp else [])
    gi = {n: i for i, n in enumerate(groups)}
    nbh = cap // 8
    nb = {"hc": nbh, "st": 4, "hp": nbh, "sp": 4}
    bb = max(2, _lg(max(nbh, 4)), KB - 3)
    gb = _lg(len(groups))
    PSp = ctx.space("stPS", [("g", gb), ("b", bb)])
    row = lambda g, b: (gi[g] << bb) | b
    NR = PSp.n

    # prover: host sponge states per group
    inputs = None
    host = {}
    if pv:
        prev = st_w.get("prev")
        # tokens are absorbed zero-padded to cap (the column is zero beyond n)
        hh = sponge_trace(hist_cur + [0] * (cap - len(hist_cur)), DOM_HIST, len(hist_cur))
        host["hc"] = hh
        r = [int(x) for x in st_w["r"]]
        host["st"] = sponge_trace(state_msg(v_rel, model_id, sess, turn, st_prev, hx, hy, hh[0], r), DOM_ST, ST_LEN)
        if hasp:
            hph = sponge_trace(list(prev["hist"]) + [0] * (cap - len(prev["hist"])), DOM_HIST, len(prev["hist"]))
            host["hp"] = hph
            host["sp"] = sponge_trace(state_msg(v_rel, model_id, sess, turn - 1, prev["st_pp"], prev["hx"], prev["hy"], hph[0],
                                                prev["r"]), DOM_ST, ST_LEN)
        inputs = np.zeros((12, NR), dtype=np.uint64)
        for g, (_, trace) in host.items():
            for b, (pin, _) in enumerate(trace):
                inputs[:, row(g, b)] = np.array(pin, dtype=np.uint64)
    reg = G.PoseRegion(PSp, inputs, B=CPU if pv else None)
    I = lambda k: reg.I(k).ex
    O = lambda k: reg.O(k).ex
    Osh = lambda k, d: reg.O(k).shifted(d)
    allv = list(range(PSp.nv))

    def pt(x):
        return Ex.atom(Z.SelLt([], 1, fixed=tuple((v, (x >> v) & 1) for v in allv)))

    first_g = {g: PSp.ind(g=gi[g]) * PSp.ind(b=0) for g in groups}
    rows_g = {g: PSp.ind(g=gi[g]) * PSp.lt("b", nb[g]) for g in groups}
    used = sum((rows_g[g] for g in groups[1:]), rows_g[groups[0]])
    rest = used - sum((first_g[g] for g in groups[1:]), first_g[groups[0]])
    firsts = sum((first_g[g] for g in groups[1:]), first_g[groups[0]])
    # capacity lanes: first row [len, 0, DOM, TAG], later rows carry
    lens = {"hc": n_r, "st": ST_LEN, "hp": None, "sp": ST_LEN}
    doms = {"hc": DOM_HIST, "st": DOM_ST, "hp": DOM_HIST, "sp": DOM_ST}
    for g in groups:
        if lens[g] is not None:
            PSp.zero(I(8) - cst(lens[g]), sel=first_g[g], tag=f"st.{g}.len")
        else:
            PSp.zero(I(8) - SC.on("st.np", PSp), sel=first_g[g], tag=f"st.{g}.len")
        PSp.zero(I(10) - cst(doms[g]), sel=first_g[g], tag=f"st.{g}.dom")
    PSp.zero(I(9), sel=firsts, tag="st.cap9")
    PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=firsts, tag="st.tag")
    for k in (8, 9, 10, 11):
        PSp.zero(I(k) - Osh(k, -1), sel=rest, tag=f"st.carry{k}")

    # message lanes: public constants + hidden scalars + digest wires + history token views
    pubm = [np.zeros(NR, dtype=np.uint64) for _ in range(8)]
    hid = [[] for _ in range(8)]          # per lane: (selector, expr)

    def put(g, i, val=None, expr=None):
        b, k = divmod(i, 8)
        if val is not None:
            pubm[k][row(g, b)] = int(val) % P
        else:
            hid[k].append((pt(row(g, b)), expr))

    def put_state(g, turn_, stp, stp_hidden, hx_, hy_, hx_hidden, hist_src, rnames):
        put(g, 0, v_rel)
        for j in range(4):
            put(g, 1 + j, model_id[j]); put(g, 5 + j, sess[j])
        put(g, 9, turn_)
        for j in range(4):
            if stp_hidden:
                put(g, 10 + j, expr=SC.on(f"st.pp{j}", PSp))
                put(g, 14 + j, expr=SC.on(f"st.hx{j}", PSp)); put(g, 18 + j, expr=SC.on(f"st.hy{j}", PSp))
            else:
                put(g, 10 + j, stp[j]); put(g, 14 + j, hx_[j]); put(g, 18 + j, hy_[j])
            # Hhist wire: element j of the digest = O(j) of the last row of the history group
            src = row(hist_src, nb[hist_src] - 1)
            i = 22 + j
            put(g, i, expr=Osh(j, src - row(g, i // 8)))
            put(g, 26 + j, expr=SC.on(rnames[j], PSp))

    put_state("st", turn, st_prev, False, hx, hy, False, "hc", [f"st.r{j}" for j in range(4)])
    if hasp:
        put_state("sp", turn - 1, None, True, None, None, True, "hp", [f"st.rp{j}" for j in range(4)])
    pcols = [PSp.pub(f"m{k}", pubm[k]) for k in range(8)]
    # history token lanes: row b of a history group, lane k <- column[8 b + k] (lane bits const, block bits -> PS 'b')
    bvars = KB - 3
    for hg, col in (("hc", hc), ("hp", hp)):
        if col is None:
            continue
        for k in range(8):
            amap = [("c", (k >> j) & 1) for j in range(3)] + [("b", j) for j in range(bvars)]
            hid[k].append((PSp.ind(g=gi[hg]), MZ.view(col, PSp, {"k": amap})))
    for k in range(8):
        e = I(k) - rest * Osh(k, -1) - pcols[k].ex
        for sel, ex in hid[k]:
            e = e - sel * ex
        PSp.zero(e, sel=used, tag=f"st.msg{k}")
    # outputs equal the public anchors
    for k in range(4):
        PSp.zero(O(k) - cst(int(s["St_new"][k])), sel=pt(row("st", 3)), tag="st.out.new")
        if hasp:
            PSp.zero(O(k) - cst(st_prev[k]), sel=pt(row("sp", 3)), tag="st.out.prev")
    if pv:
        if host["st"][0] != tuple(int(x) for x in s["St_new"]):
            ctx.violation("state chain: recomputed St_new != public St_new")
        if hasp and host["sp"][0] != tuple(st_prev):
            ctx.violation("state chain: opening of St_prev mismatch")
    return {"groups": groups, "ps_rows": NR, "hist_cap": cap, "turn": turn, "hist_space_bits": KB,
            "state_cells": int(sum(v.N for v in ctx.cir.vecs)) - cells0}


def len_ex(SC, sp, hasp):
    """expression l' on sp (0 at turn 0)."""
    return SC.on("st.np", sp) if hasp else Ex.const(0)
