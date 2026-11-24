# SPDX-License-Identifier: Apache-2.0
"""Reply proof: certified blocks of Fiat-Shamir-sampled slots, bound to the per-request trace commitment H, to the
generated tokens and to model_id, in ONE flashprove_next.zk_flat proof.

  statement  model_id (specs/09 commitment with radius and configuration rows), R = trace_root(s_R, rt_obj, rt_H, T_pad),
             the public API view (request tokens, visible reply tokens, reasoning-token count, api_meta), the
             multi-turn state part (state_zk) and the profile's slot counts M_C, M_G.  The sequence length T (which
             includes the hidden system prompt and hidden reasoning) and the layer count L are NOT public: every draw
             is reduced in-circuit.  rt_H commits the trace matrix H of every port code of every (t, l) (hsel_zk:
             rate-1/2 rows, Poseidon2 column leaves).
  draws      u_s = 192-bit hashes of the statement, fixed before any prover message.  Compute slots draw (t, l) with
             cost weight C_SELF + C_KEY (t + 1) from two pools (input positions, generated positions; specs/01 §4.4),
             reduced in-circuit by an exact Euclidean division against the hidden pool total (``slot_of``,
             ``euclid_draw``).  Generation slots draw a public step g in [0, N_gen).
  blocks     certified.blk_N (N1, N2), blk_qkv, blk_gemm_res (OPROJ, DOWN), blk_gu at (t_s, l_s); weights through
             wsum_zk.CfgTiledWeights with the hidden layer_p1 = l_s + 1 -> Tt LogUp -> WSum -> mc_zk HCO -> model_id.
  binding    EVERY decoded input / output code column of every block is bound to H[t_s, (l_s, port, .)] with
             hsel_zk.bind_vector (hidden row t_s, hidden column prefix l_s, constant port bits).
  attention  attn_zk slots read layer l_s's lifted K / V of every key tau <= t_s from H; the producer row (tau_p, l_s)
             gets a certified N1 + KV + EMB check, every decoded port bound to H at row tau_p, so the k / v codes
             the slot lifts are the correct KV projection output.
  tokens     one global token chain (hidden system prompt, re-injected history, request, reasoning, tool segments,
             visible reply) is committed once; EMB binds H's layer-0 input to E[tok[t]] at every drawn and every
             injection position.
  generation gen_zk / sampler_zk: at each drawn step the LM head over the full vocabulary, the certified logit radius,
             the keyed bucketed sampling predicate S_t (specs/01 §5.4, specs/06 §3.2) with the committed K_prf, and
             the constrained-decoding mask of the request's JSON-Schema DFA (api_zk).
  api        finish_reason, max_tokens, stop / tool-end tokens and tool segments are checked against the token chain;
             a stop token inside hidden reasoning is rejected deterministically.

usage:  python tests/test_reply_zk.py   and   python tests/test_api_zk.py
"""
import hashlib
import json
import time

import numpy as np

from . import api_zk as APZ
from . import certified as CZ
from . import hsel_zk as HZ
from . import tc_zk as TZ
from . import tc_ref as R
from . import attn_zk as AZ
from . import gen_zk as GZ
from . import sampler_zk as SAMPZ
from . import sampling_v3 as S3
from . import state_zk as SZ
from . import mc_zk as MZ
from . import modelcommit as MC
from . import wsum_zk as WS
from . import zk_flat as Z
from .field import CPU, P
from .zk_flat import Ex
from .fp_ref import EPS_BITS

VERSION = "fpn-reply-1"
LABEL = b"fpn-reply-v1"

def emb_block(k):
    """E^T column block k (d x T_I): token tok lives in block tok // T_I at column tok % T_I.  The block index is the
    tile key's EXPERT field, so an EMB check opens ONE hidden tile (expert = hidden tok // T_I) instead of all V/T_I."""
    return f"model.embed_tokens_t.experts.{int(k)}.weight"


class LazyCommitTensors:
    """Zero-copy Checkpoint-backed mapping of committed tensors + E^T T_I-column blocks + tied lm_head fallback."""

    def __init__(self, ck, TI):
        self.ck = ck
        self.TI = int(TI)
        sh = dict(ck.shapes())
        V, d = sh["model.embed_tokens.weight"]
        self.V, self.d = int(V), int(d)
        self.n_emb = -(-self.V // self.TI)
        if "lm_head.weight" not in sh:
            sh["lm_head.weight"] = (self.V, self.d)
        for k in range(self.n_emb):
            sh[emb_block(k)] = (self.d, self.TI)
        self._shapes = sh
        self.config = getattr(ck, "config", {})

    def shapes(self):
        return dict(self._shapes)

    def __contains__(self, name):
        return name in self._shapes

    def __getitem__(self, name):
        if name.startswith("model.embed_tokens_t.experts.") and name.endswith(".weight"):
            k = int(name.split(".")[3])
            E = self.ck.codes("model.embed_tokens.weight")
            blk = np.zeros((self.d, self.TI), dtype=np.uint16)
            cols = np.asarray(E[k * self.TI:(k + 1) * self.TI]).T
            blk[:, :cols.shape[1]] = cols
            return blk
        if name == "lm_head.weight" and "lm_head.weight" not in self.ck.shapes():
            return self.ck.codes("model.embed_tokens.weight")
        return self.ck.codes(name)

    def codes(self, name):
        return self[name]


def commit_tensors(tensors, TI):
    """the tensors committed under model_id: the checkpoint plus E^T split into T_I-column blocks (EMB block)."""
    if isinstance(tensors, LazyCommitTensors):
        return tensors
    if hasattr(tensors, "codes") and hasattr(tensors, "shapes"):
        return LazyCommitTensors(tensors, TI)
    out = dict(tensors)
    E = np.asarray(tensors["model.embed_tokens.weight"])
    if "lm_head.weight" not in out:
        out["lm_head.weight"] = E
    V, d = E.shape
    for k in range(-(-V // TI)):
        blk = np.zeros((d, TI), dtype=E.dtype)
        cols = E[k * TI:(k + 1) * TI].T
        blk[:, :cols.shape[1]] = cols
        out[emb_block(k)] = blk
    return out


MAT_MAP = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}

# H ports: (name, capture field, slice) -- the code vectors the certified blocks decode
PORTS = ["x_attn", "q_lo", "q_hi", "k_lo", "k_hi", "v", "o", "emb", "h_mid", "x_mlp", "a", "h_out", "K", "V", "lse_lo", "lse_hi"]
# residual input h_in(t, l) is NOT a port of its own (specs/01 §4.1 zero-freedom wiring): it IS h_out(t, l - 1), or for l = 0
# the embedding row stored in port "emb" of the layer-0 record; "h_in" below is that virtual source
# K / V: integer-lifted k / v (attn_zk.lifted_ports, tc_ref Lift_FK / Lift_FV) read by the attention slots
PORT_ID = {p: i for i, p in enumerate(PORTS)}
# decode tag suffix (per block kind) -> port
BIND = {
    "N1": {"h": "h_in", "x": "x_attn"},
    "QKV": {"xi": "x_attn", "vo": "v", "q.olo": "q_lo", "q.ohi": "q_hi", "k.olo": "k_lo", "k.ohi": "k_hi"},
    "KV": {"xi": "x_attn", "vo": "v", "k.olo": "k_lo", "k.ohi": "k_hi"},
    "OPROJ": {"xi": "o", "hr": "h_in", "ho": "h_mid"},
    "N2": {"h": "h_mid", "x": "x_mlp"},
    "GU": {"xi": "x_mlp", "ao": "a"},
    "DOWN": {"xi": "a", "hr": "h_mid", "ho": "h_out"},
}
# decodes of weight-side vectors (norm gains): bound to model_id by the gain-binding work, not to H
WEIGHT_SIDE = {"g", "gq", "gk"}
BLOCKS = ["N1", "QKV", "OPROJ", "N2", "GU", "DOWN"]
CLAIMS_PER_SLOT = sum(len(BIND[b]) for b in BLOCKS)         # 18
# key producer of an attention slot (lineage depth 1): N1 + KV (k, v only; q is unused by key-producer lineage) + EMB
PRODUCER_BLOCKS = ["N1", "KV"]
PRODUCER_CLAIMS = sum(len(BIND[b]) for b in PRODUCER_BLOCKS) + 1   # 6 port binds + 1 EMB output bind = 7


def _lg(n):
    return max(0, int(n - 1).bit_length())


# ============================================================================ profile
class PackedLayout(AZ.RowLayout):
    """attn_zk.RowLayout API over reply_zk's packed (t, l) records: col = (l << WB) | (poff[port] + i)."""

    def __init__(self, prf):
        super().__init__(prf.TB, prf.LB, 0, prf.WB, {p: i for i, p in enumerate(PORTS)})
        self.prf = prf

    def col(self, l, port, i):
        return (l << self.WB) | (self.prf.poff[port] + i)

    def _pre(self, d, port, l_ent):
        hi = self.prf.poff[port] >> d
        assert self.prf.poff[port] % (1 << d) == 0 and d <= self.prf.pbits[port]
        return [("const", (hi >> k) & 1) for k in range(self.WB - d)] + list(l_ent)

    def kv_point(self, tau, l_ent, port, g, i):
        d = len(i) + len(g)
        return list(i) + list(g) + self._pre(d, port, l_ent) + list(tau)


class Profile:
    """Public profile constants: layer widths, padded bounds L_MAX / T_pad, slot counts and commitment parameters.
    The real layer count, the sequence length and every weight stay hidden."""

    def __init__(self, d, n_q, n_kv, dh, dff, L_MAX, T_pad, M_C, prm, MQ_H=16, s_bits=1, SYS_CAP=8, VEC_PAD=2048,
                 QN_CAP=4.0, KN_CAP=64.0, EXP_BITS=14, attention=True, V=64, K0=None, C_SELF=None, C_KEY=None, HIST_CAP=16,
                 claim_slots=None, M_G=0, rule=None, rope_fn=None, eps_bits=EPS_BITS, TOOL_CAP=0, token_bytes=None):
        self.rope_fn = rope_fn
        self.eps_bits = int(eps_bits)
        self.HIST_CAP = int(HIST_CAP)
        self.TOOL_CAP = int(TOOL_CAP)
        assert self.TOOL_CAP == 0 or (self.TOOL_CAP >= 2 and self.TOOL_CAP & (self.TOOL_CAP - 1) == 0)
        self.token_bytes = token_bytes
        # Pi_2 pools: K0 compute slots from the input layer, M_C - K0 from the generation layer (specs/01 §4.4); unit costs
        # c_s (GEMM work of one (t, l)) and c_k (attention work per key) are public profile constants
        self.K0 = (M_C + 1) // 2 if K0 is None else int(K0)
        attn_k = 4 * n_q * dh
        self.C_SELF = int(C_SELF) if C_SELF else max(1, (2 * d * (n_q + 2 * n_kv) * dh + 2 * n_q * dh * d + 6 * d * dff) // attn_k)
        self.C_KEY = int(C_KEY) if C_KEY else 1
        max_pool_c = int(L_MAX) * (int(T_pad) * self.C_SELF + int(T_pad) * (int(T_pad) + 1) // 2 * self.C_KEY)
        assert max_pool_c < (1 << 46), f"pool cost cap {max_pool_c} must stay < 2^46 for non-wrapping Euclidean draw"
        self.V = int(V)                          # vocabulary size (EMB + LM head + sampler_zk)
        self.M_G = int(M_G)                      # Pool-3 generation spot-check draws (105 at full security; 0 disables)
        self.K_G = 0 if self.M_G == 0 else max(2, 1 << _lg(self.M_G))
        self.rule = S3.Rule(0.6, 0.95, 32) if rule is None else rule
        self.s_bits = int(s_bits)
        self.VEC_PAD = int(VEC_PAD)
        self.SYS_CAP = int(SYS_CAP)
        self.d, self.n_q, self.n_kv, self.dh, self.dff = d, n_q, n_kv, dh, dff
        self.L_MAX, self.T_pad, self.M_C, self.prm = L_MAX, T_pad, M_C, prm
        self.K_C = max(2, 1 << _lg(max(1, M_C)))
        self.attention = bool(attention)
        # HCO claim slots of WSum / mc_zk (power of two, independent of R_MAX; auto-sized when None).  NOT self.J (H claims)
        if claim_slots is None:
            tq = 1 << (max(0, _lg(n_q * dh) - prm.ob) + max(0, _lg(d) - prm.ib))
            tkv = 1 << (max(0, _lg(n_kv * dh) - prm.ob) + max(0, _lg(d) - prm.ib))
            to_ = 1 << (max(0, _lg(d) - prm.ob) + max(0, _lg(n_q * dh) - prm.ib))
            tgu = 1 << (max(0, _lg(dff) - prm.ob) + max(0, _lg(d) - prm.ib))
            tdn = 1 << (max(0, _lg(d) - prm.ob) + max(0, _lg(dff) - prm.ib))
            temb = 1 << (max(0, _lg(d) - prm.ob) + max(0, _lg(prm.TI) - prm.ib))
            thd = (1 << (max(0, _lg(self.V) - prm.ob) + max(0, _lg(d) - prm.ib))) if self.M_G > 0 else 0
            kinj = max(2, 1 << _lg(max(1, SYS_CAP)))
            ktool = max(2, 1 << _lg(self.TOOL_CAP)) if self.TOOL_CAP > 0 else 0
            ws_need = (self.K_C * (tq + 2 * tkv + to_ + 2 * tgu + tdn + temb)
                       + (self.K_C * (2 * tkv + temb) if self.attention else 0)
                       + (kinj + ktool) * temb + thd + 4)
            self.claim_slots = max(prm.R_MAX, 2 << _lg(ws_need))
        else:
            self.claim_slots = int(claim_slots)
        sizes = {"x_attn": d, "q_lo": n_q * dh // 2, "q_hi": n_q * dh // 2, "k_lo": n_kv * dh // 2,
                 "k_hi": n_kv * dh // 2, "v": n_kv * dh, "o": n_q * dh, "emb": d, "h_mid": d, "x_mlp": d,
                 "a": dff, "h_out": d, "K": n_kv * dh, "V": n_kv * dh, "lse_lo": n_q, "lse_hi": n_q}
        # packed port layout: each port at an offset aligned to its own power-of-two width (largest first)
        self.pbits = {p: _lg(n) for p, n in sizes.items()}
        self.poff, off = {}, 0
        for p in sorted(PORTS, key=lambda p: (-self.pbits[p], PORT_ID[p])):
            self.poff[p] = off
            off += 1 << self.pbits[p]
        self.WB = _lg(off)                       # column bits of one (t, l) record
        self.LB = _lg(L_MAX)
        self.TB = _lg(T_pad)
        self.tier = HZ.HTier(RB=self.TB, CB=self.WB + self.LB, MQ=MQ_H, T_pad=T_pad)
        # attention slots (flashprove_next.attn_zk): tc_ref tier of the public shapes, H layout adapter
        self.atier = R.Tier(T_pad=T_pad, TC=1, L_MAX=L_MAX, G=n_kv, R=n_q // n_kv, DH=dh, MQ=MQ_H, QN_CAP=QN_CAP,
                            KN_CAP=KN_CAP, EXP_BITS=EXP_BITS)
        self.layout = PackedLayout(self)
        per = CLAIMS_PER_SLOT + 1 + ((AZ.claims_per_slot(self.layout) + PRODUCER_CLAIMS) if self.attention else 0)
        self.J = 1 << _lg(max(2, per * M_C + SYS_CAP + self.TOOL_CAP + self.K_G))

    def gprm(self):
        return dict(d=self.d, n_q=self.n_q, n_kv=self.n_kv, dh=self.dh, tb=max(4, self.TB))


# ============================================================================ H
def port_vectors(cap, t, l, prf):
    """the code vector of every port at (t, l), in the flattening the certified blocks decode."""
    dh = prf.dh
    q = np.asarray(cap.q[t, l]).reshape(-1, 2, dh // 2)
    k = np.asarray(cap.k[t, l]).reshape(-1, 2, dh // 2)
    lse = np.asarray(cap.lse[t, l], dtype=np.uint32).ravel()
    return {"x_attn": cap.x_attn[t, l], "q_lo": q[:, 0, :].ravel(), "q_hi": q[:, 1, :].ravel(),
            "k_lo": k[:, 0, :].ravel(), "k_hi": k[:, 1, :].ravel(), "v": cap.v[t, l], "o": cap.o[t, l],
            "emb": cap.h_in[t, 0] if l == 0 else np.zeros(0, dtype=np.int64), "h_mid": cap.h_mid[t, l], "x_mlp": cap.x_mlp[t, l], "a": cap.a[t, l],
            "h_out": cap.h_out[t, l], "lse_lo": (lse & 0xFFFF).astype(np.int64), "lse_hi": (lse >> 16).astype(np.int64),
            **dict(zip(("K", "V"), AZ.lifted_ports(cap.k[t, l], cap.v[t, l], prf.atier)))}


def h_col(prf, l, port, i):
    return (l << prf.WB) | (prf.poff[port] + i)


def build_H(cap, prf):
    """H[t, (l, port, i)] = code (int32, 0 outside the real shapes; 2x smaller than int64)."""
    tier = prf.tier
    H = np.zeros((tier.nr, tier.nc), dtype=np.int32)
    T, L = cap.h_in.shape[0], cap.h_in.shape[1]
    for t in range(T):
        for l in range(L):
            for p, vec in port_vectors(cap, t, l, prf).items():
                v = np.asarray(vec, dtype=np.int32).ravel()
                assert len(v) <= (1 << prf.pbits[p]), p
                c0 = h_col(prf, l, p, 0)
                H[t, c0:c0 + len(v)] = v
    return H


# ============================================================================ model commitment (09, radius + cfg rows)
def commit_model(tensors, prf, seed_vol=7, L_real=1, V_real=1024, sys_prompt=(101, 202, 303), k_prf=None):
    tensors = commit_tensors(tensors, prf.prm.TI)
    rng = np.random.default_rng(seed_vol)
    shapes = {k: tuple(v.shape) for k, v in tensors.items()}
    lay = MC.Layout(shapes, prf.prm, radius=True, cfg_rows=True)
    src = MC.RowSource(MC.ArrayModel(tensors), lay)
    Wd = np.zeros((lay.n_span, prf.prm.nc), dtype=np.int64)
    for lo, hi in lay.row_ranges:
        Wd[lo:hi] = src.rows(lo, hi)
    store = MZ.DevStore(prf.prm, Wd, CPU)
    kp = [int(x) for x in rng.integers(1, 1 << 30, size=4)] if k_prf is None else [int(x) % P for x in k_prf]
    sc = [int(x) for x in rng.integers(1, 1 << 30, size=4)]
    vol = MC.encode_volatile_cfg({
        "k_prf": kp, "salt_cfg": sc,
        "sys_prompt": list(sys_prompt), "L_real": L_real, "V_real": V_real,
        "T_bits": prf.rule.T_bits, "p16": prf.rule.p16, "beta": prf.rule.beta,
    })
    ccfg = MC.volatile_config_root(vol)
    pre, mid = MZ.model_preimage(store.root, c_cfg=ccfg)
    return {"layout": lay, "store": store, "pre": pre, "mid": mid, "vol": vol, "k_prf": kp, "salt_cfg": sc}


# ============================================================================ draws
def draws(stmt_core, M_C, C):
    """slot draws J_s in [0, C) from the statement only (u_s: 192-bit hash, J_s = floor(u_s C / 2^192))."""
    seed = hashlib.blake2b(json.dumps(stmt_core, sort_keys=True, separators=(",", ":")).encode(),
                           digest_size=32, person=b"fpn-reply-dr").digest()
    out = []
    for s in range(M_C):
        u = int.from_bytes(hashlib.blake2b(seed + s.to_bytes(4, "little"), digest_size=24).digest(), "little")
        out.append((u * C) >> 192)
    return out


CORE_KEYS = ("rv", "R", "model_id", "M_C", "M_G", "req_tokens", "n_reason", "vis_tokens", "st")
VOL_T_BITS, VOL_P16, VOL_BETA = 8, 9, 10
VOL_L_REAL = 11                          # modelcommit.encode_volatile_cfg: L_real


def draw_seeds(stmt_core, M_C):
    """192-bit draw seeds u_s from the public statement only (fixed before any prover message; re-salting R to regrind
    them is the adaptive-query term of the lifetime bound).  Returned as 12 little-endian 16-bit limbs each."""
    seed = hashlib.blake2b(json.dumps(stmt_core, sort_keys=True, separators=(",", ":")).encode(),
                           digest_size=32, person=b"fpn-reply-dr").digest()
    out = []
    for s in range(M_C):
        u = int.from_bytes(hashlib.blake2b(seed + s.to_bytes(4, "little"), digest_size=24).digest(), "little")
        out.append([(u >> (16 * k)) & 0xFFFF for k in range(12)])
    return out


def gen_draw_seeds(stmt_core, M_G):
    """Pool-3 generation-step 192-bit draw seeds u_gen[s] from the public statement only."""
    if M_G <= 0:
        return []
    seed = hashlib.blake2b(json.dumps(stmt_core, sort_keys=True, separators=(",", ":")).encode(),
                           digest_size=32, person=b"fpn-reply-gn").digest()
    out = []
    for s in range(M_G):
        u = int.from_bytes(hashlib.blake2b(seed + s.to_bytes(4, "little"), digest_size=24).digest(), "little")
        out.append([(u >> (16 * k)) & 0xFFFF for k in range(12)])
    return out


def gen_steps_of(u_gen_draws, n_gen):
    """Public generation step indices g_s = floor(u_gen[s] * N_gen / 2^192) in [0, N_gen) (N_gen is public)."""
    if n_gen <= 0:
        return []
    return [(u_of(u) * int(n_gen)) >> 192 for u in u_gen_draws]


def samp_ctx_of(st):
    """Canonical SampCtx derived from the public reply statement (specs/01 §5.3): binds req_tokens, N_c, N_s, sess, turn."""
    s = st["st"]
    nc = bytes.fromhex(s["N_c"])
    ns = bytes.fromhex(s["N_s"])
    dig = hashlib.blake2b(json.dumps(st["req_tokens"], separators=(",", ":")).encode(),
                          digest_size=32, person=b"fpn-req-dig").digest()
    sess_b = b"".join(int(x).to_bytes(8, "little") for x in s["sess"])
    return S3.SampCtx(nc, ns, dig=dig, v_rel=int(s["v_rel"]) & 0xFFFFFFFF, epoch=0, sess=sess_b, turn=int(s["turn"]))


def u_of(limbs):
    return sum(int(x) << (16 * k) for k, x in enumerate(limbs))


def _base_core_dict(st):
    core = {k: st[k] for k in CORE_KEYS}
    if st.get("api_meta") is not None:
        core["api_meta"] = st["api_meta"]
    return core


def precommit_cpre(st):
    """Protocol v2 pre-challenge commitment C_pre = B256(base_core) returned before external seed c_ext."""
    return hashlib.blake2b(
        json.dumps(_base_core_dict(st), sort_keys=True, separators=(",", ":")).encode(),
        digest_size=32, person=b"FPN2-CPRE-V2\0\0\0\0",
    ).digest()


def _core_dict(st):
    core = _base_core_dict(st)
    if st.get("challenge_mode") == "external_seed":
        core["challenge_mode"] = "external_seed"
        core["c_pre"] = st["c_pre"]
        core["c_ext"] = st["c_ext"]
    return core


def _api_layout(st):
    """Canonical segment layout after `b = a + n_prev + len(req)` (single-segment when `api_meta.tool_segments` empty)."""
    req = [int(x) for x in st["req_tokens"]]
    vis = [int(x) for x in st["vis_tokens"]]
    n_r = int(st["n_reason"])
    am = st.get("api_meta") or {}
    segs = am.get("tool_segments") or []
    pub_binds = [(0, k, x) for k, x in enumerate(req)]
    gen_rel_offs = []
    hist_rel_offs = []
    tool_ret_rel_offs = []
    seg_end_steps = {}
    rel = 0
    for seg in segs:
        snr = int(seg.get("n_reason", 0))
        call = [int(x) for x in seg["call_tokens"]]
        nret = int(seg["n_ret"])
        ret_pub = seg.get("ret_tokens")
        for k in range(snr):
            hist_rel_offs.append(rel + k)
            gen_rel_offs.append(rel + k)
        rel += snr
        for k, x in enumerate(call):
            pub_binds.append((len(req) + rel, k, x))
            gen_rel_offs.append(rel + k)
        if call:
            seg_end_steps[len(gen_rel_offs) - 1] = APZ.FINISH_TOOL
        rel += len(call)
        for k in range(nret):
            tool_ret_rel_offs.append(rel + k)
            if ret_pub is None:
                hist_rel_offs.append(rel + k)
        if ret_pub is not None:
            for k, x in enumerate(ret_pub):
                pub_binds.append((len(req) + rel, k, int(x)))
        rel += nret
    con_step_start = len(gen_rel_offs) + n_r
    for k in range(n_r):
        hist_rel_offs.append(rel + k)
        gen_rel_offs.append(rel + k)
    rel += n_r
    for k, x in enumerate(vis):
        pub_binds.append((len(req) + rel, k, x))
        gen_rel_offs.append(rel + k)
    rel += len(vis)
    fr = APZ.canon_finish_reason(am.get("finish_reason"))
    if fr is not None and len(gen_rel_offs) > 0:
        seg_end_steps[len(gen_rel_offs) - 1] = fr
    return {
        "n_span": rel,
        "n_gen": len(gen_rel_offs),
        "gen_rel_offs": gen_rel_offs,
        "hist_rel_offs": None if not segs else hist_rel_offs,
        "tool_ret_rel_offs": tool_ret_rel_offs,
        "seg_end_steps": seg_end_steps,
        "pub_binds": pub_binds,
        "con_step_start": con_step_start,
        "finish_reason": fr,
    }


def turn_state(prf, com, api, hist, nonce_c, nonce_s, r, turn=0, sess=None, st_prev=None, prev_bundle=None):
    """state part of a reply statement (state_zk, specs/01 §6.4) + the prover's state witness.  Turn 0: sess = H(N_c),
    St_prev = genesis.  hist = this turn's hidden history tokens.  Returns (st_stmt, st_wit, next_prev_bundle)."""
    v_rel = SZ.v_rel_of(VERSION)
    sess = SZ.sess_of(nonce_c) if sess is None else tuple(sess)
    st_prev = SZ.genesis(v_rel, sess) if st_prev is None else tuple(st_prev)
    st_new, nxt = SZ.new_turn(v_rel, com["mid"], sess, turn, st_prev, api["req_tokens"], api["vis_tokens"], nonce_c,
                              nonce_s, hist, r, cap=prf.HIST_CAP)
    return (SZ.statement(v_rel, com["mid"], sess, turn, st_prev, st_new, nonce_c, nonce_s, hist_cap=prf.HIST_CAP),
            {"r": list(r), "prev": prev_bundle}, nxt)


def statement(prf, com, R, api=None, st_part=None, *, c_ext=None):
    """api: the standard-API view {req_tokens, n_reason (usage.reasoning_tokens), vis_tokens, optional api_meta}.  T (which
    includes the hidden system prompt) and L (architecture) are NOT public: the draws are computed in-circuit.
    c_ext: optional 32-byte external challenge seed (Protocol v2 external-seed mode with C_pre anti-retry)."""
    api = api or {"req_tokens": [], "n_reason": 0, "vis_tokens": []}
    lay = com["layout"]
    assert lay.n_vec_rows == 1 and lay.vec_len <= prf.VEC_PAD, "gain vectors must fit the profile's single vec row cap"
    st = WS.statement(prf.prm, com["mid"], {"blocks": BLOCKS, "reply": VERSION}, cfg_rows=True,
                      vol_blocks=len(com["vol"]) // 8, meta_cfg0=lay.meta_cfg0, meta_cfg1=lay.meta_cfg1,
                      vec_row=lay.n_meta, vec_pad=prf.VEC_PAD, claim_slots=prf.claim_slots)
    st.update({"rv": VERSION, "R": [int(x) for x in R], "tier": HZ.statement(prf.tier, R, prf.J)["tier"], "J": prf.J,
               "M_C": prf.M_C, "M_G": prf.M_G, "req_tokens": [int(x) for x in api["req_tokens"]],
               "n_reason": int(api["n_reason"]), "vis_tokens": [int(x) for x in api["vis_tokens"]]})
    if api.get("api_meta") is not None:
        st["api_meta"] = json.loads(json.dumps(api["api_meta"]))
    assert st_part is not None, "statement needs the state part (turn_state)"
    st["st"] = st_part
    if c_ext is not None:
        ce = bytes.fromhex(c_ext) if isinstance(c_ext, str) else bytes(c_ext)
        if len(ce) != 32:
            raise ValueError("c_ext must be 32 bytes")
        st["challenge_mode"] = "external_seed"
        st["c_pre"] = precommit_cpre(st).hex()
        st["c_ext"] = ce.hex()
    core = _core_dict(st)
    st["u_draws"] = draw_seeds(core, prf.M_C)
    st["u_att"] = att_draws(st["u_draws"])
    st["u_gen"] = gen_draw_seeds(core, prf.M_G)
    return st


def att_draws(u_draws):
    """public producer-read draws (attn_zk.UB bits) of the attention slots, derived from the slot seeds."""
    return [int.from_bytes(hashlib.blake2b(json.dumps(u).encode(), digest_size=8, person=b"fpn-reply-at").digest(),
                           "little") & ((1 << AZ.UB) - 1) for u in u_draws]


def check_statement(prf, st, *, expected_c_pre=None, expected_c_ext=None, require_external_pins=False):
    if int(st.get("M_G", 0)) != prf.M_G:
        raise Z.ProofError("M_G of the statement differs from the profile")
    cmode = st.get("challenge_mode")
    if cmode is not None:
        if cmode != "external_seed":
            raise Z.ProofError(f"unsupported challenge_mode {cmode!r}")
        recomputed = precommit_cpre(st).hex()
        if not isinstance(st.get("c_pre"), str) or st["c_pre"] != recomputed:
            raise Z.ProofError("E_CPRE_MISMATCH: statement c_pre does not match committed base core")
        raw_cext = st.get("c_ext")
        if not isinstance(raw_cext, str):
            raise Z.ProofError("invalid c_ext type in statement (expected 64-hex string)")
        try:
            ce = bytes.fromhex(raw_cext)
        except ValueError:
            raise Z.ProofError("invalid c_ext hex in statement") from None
        if len(ce) != 32 or raw_cext != ce.hex():
            raise Z.ProofError("c_ext must be canonical 32-byte lowercase hex")
        if require_external_pins and (expected_c_pre is None or expected_c_ext is None):
            raise Z.ProofError("E_CPRE_UNPINNED: external_seed verification requires pre-recorded expected_c_pre and expected_c_ext")
        if expected_c_pre is not None:
            exp_p = expected_c_pre.hex() if isinstance(expected_c_pre, (bytes, bytearray)) else str(expected_c_pre)
            if st["c_pre"] != exp_p:
                raise Z.ProofError("E_CPRE_MISMATCH: statement c_pre differs from pre-recorded commitment")
        if expected_c_ext is not None:
            exp_e = expected_c_ext.hex() if isinstance(expected_c_ext, (bytes, bytearray)) else str(expected_c_ext)
            if raw_cext != exp_e:
                raise Z.ProofError("E_CEXT_MISMATCH: statement c_ext differs from released external seed")
    elif expected_c_pre is not None or expected_c_ext is not None:
        raise Z.ProofError("E_CPRE_MISMATCH: expected external_seed mode when expected_c_pre/expected_c_ext pinned")
    core = _core_dict(st)
    if (st["u_draws"] != draw_seeds(core, prf.M_C) or st["u_att"] != att_draws(st["u_draws"])
            or st.get("u_gen", []) != gen_draw_seeds(core, prf.M_G)):
        raise Z.ProofError("slot draw seeds do not match the statement")
    if int(st.get("claim_slots", prf.prm.R_MAX)) != prf.claim_slots:
        raise Z.ProofError("claim_slots of the statement differ from the profile")
    if int(st.get("n_reason", -1)) < 0:
        raise Z.ProofError("negative n_reason in statement")
    am = st.get("api_meta") or {}
    if not isinstance(am, dict):
        raise Z.ProofError("api_meta must be a dict when present")
    bad_am = set(am.keys()) - APZ.ALLOWED_API_META_KEYS
    if bad_am:
        raise Z.ProofError(f"unknown api_meta keys {sorted(bad_am)}")
    for seg in am.get("tool_segments") or []:
        if not isinstance(seg, dict):
            raise Z.ProofError("tool_segment must be a dict")
        if int(seg.get("n_reason", 0)) < 0 or int(seg.get("n_ret", -1)) < 0:
            raise Z.ProofError("negative length in tool_segment")
    lay_a = _api_layout(st)
    goffs = lay_a["gen_rel_offs"]
    if any(a >= b for a, b in zip(goffs, goffs[1:])):
        raise Z.ProofError("non-increasing generated token offsets")
    if len(st["req_tokens"]) + lay_a["n_span"] > prf.T_pad:
        raise Z.ProofError("public token counts exceed the length tier")
    if len(lay_a["tool_ret_rel_offs"]) > prf.TOOL_CAP:
        raise Z.ProofError("tool-return witness length exceeds profile TOOL_CAP")
    fr = lay_a["finish_reason"]
    if fr is not None:
        max_tok = am.get("max_tokens")
        if max_tok is None or int(max_tok) < 0:
            raise Z.ProofError("finish_reason requires non-negative max_tokens in api_meta")
        max_tok = int(max_tok)
        n_gen = lay_a["n_gen"]
        if n_gen > max_tok:
            raise Z.ProofError(f"generated tokens {n_gen} exceed max_tokens {max_tok}")
        if n_gen == 0 and (max_tok != 0 or fr != APZ.FINISH_LENGTH):
            raise Z.ProofError("empty generation is valid only when max_tokens == 0 and finish_reason == 'length'")
        if fr == APZ.FINISH_LENGTH and n_gen != max_tok:
            raise Z.ProofError(f"finish_reason 'length' requires n_gen ({n_gen}) == max_tokens ({max_tok})")
        if fr == APZ.FINISH_STOP and am.get("eos_token_id") is None:
            raise Z.ProofError("finish_reason 'stop' requires eos_token_id in api_meta")
        if fr == APZ.FINISH_TOOL and am.get("tool_end_token_id") is None:
            raise Z.ProofError("finish_reason 'tool_use' requires tool_end_token_id in api_meta")
    if (am.get("tool_segments") or []) and am.get("tool_end_token_id") is None:
        raise Z.ProofError("tool_segments require tool_end_token_id in api_meta")
    for seg in am.get("tool_segments") or []:
        call = seg.get("call_tokens") or []
        if not call:
            raise Z.ProofError("empty tool_call segment")
        if int(call[-1]) != int(am["tool_end_token_id"]):
            raise Z.ProofError("intermediate tool_call segment does not end with tool_end_token_id")
        if seg.get("ret_tokens") is not None and len(seg["ret_tokens"]) != int(seg["n_ret"]):
            raise Z.ProofError("tool_segment ret_tokens length != n_ret")
    if list(st["st"]["model_id"]) != list(st["model_id"]) or int(st["st"]["hist_cap"]) != prf.HIST_CAP:
        raise Z.ProofError("state statement does not match the reply statement / profile")
    n_hist = int(st["n_reason"]) if lay_a["hist_rel_offs"] is None else len(lay_a["hist_rel_offs"])
    SZ.check_statement(st, st["req_tokens"], n_hist)


def pool_sums(b, T, cs, ck):
    """(S0, S1): total unit cost of pool 0 (t < b - 1) and pool 1 (b - 1 <= t < T - 1); S(t) = t cs + ck t (t + 1) / 2."""
    S = lambda t: t * cs + ck * t * (t + 1) // 2
    return S(b - 1), S(T - 1) - S(b - 1)


def slot_pool(prf, n_gen, s):
    """public pool of compute slot s: the first K0 slots draw from pool 0, the rest from pool 1 (pool 0 if N = 0)."""
    return 0 if (s < prf.K0 or n_gen == 0) else 1


def slot_of(u_limbs, b, T, L, pool, cs, ck):
    """prover: (J, t, l, J') of one draw from a pool: J = floor(u L S_pool / 2^192), (J', l) = divmod(J, L), t the unique
    position of the pool with S(t) - S(t_0) <= J' < S(t + 1) - S(t_0)."""
    S = lambda t: t * cs + ck * t * (t + 1) // 2
    t0, t1 = (0, b - 1) if pool == 0 else (b - 1, T - 1)
    Sp = S(t1) - S(t0)
    J = (u_of(u_limbs) * L * Sp) >> 192
    Jp, l = divmod(J, L)
    t = t0
    while S(t + 1) - S(t0) <= Jp:
        t += 1
    return J, t, l, Jp


def euclid_draw(ctx, SCa, u_limbs, C_val, tag, cname="rz.C0"):
    """In-circuit J = floor(u C / 2^192) for public u (12 x 16-bit limbs) and hidden C < 2^46:
    u_k C + carry_{k-1} = rem_k + carry_k 2^16 (k = 0..11), 0 <= rem_k < 2^16, 0 <= carry_k < 2^46; carry_11 = J.
    Because rem_k + carry_k 2^16 < 2^62 < P and u_k C + carry_{k-1} < 2^63 < P, every limb equation holds over Z
    without modular wrap-around, so telescoping gives u C = rem + J 2^192 with 0 <= rem < 2^192 and J is unique."""
    G_ = __import__("flashprove_next.gadgets", fromlist=["x"])
    DS = ctx.space(tag, [("k", 4)])
    uv = np.zeros(DS.n, dtype=np.int64); uv[:12] = u_limbs
    act = np.zeros(DS.n, dtype=np.int64); act[:12] = 1
    rem_v = car_v = None
    if ctx.prove:
        rem_v = np.zeros(DS.n, dtype=np.int64); car_v = np.zeros(DS.n, dtype=np.int64)
        c = 0
        for k in range(12):
            x = int(u_limbs[k]) * int(C_val) + c
            rem_v[k], c = x & 0xFFFF, x >> 16
            car_v[k] = c
    rem = DS.col("rem", rem_v)
    car = DS.col("car", car_v)
    u_p = DS.pub("u", uv)
    a_p = DS.pub("act", act)
    C_ex = SCa.on(cname, DS)
    prev = (Ex.const(1) - DS.ind(k=0)) * car.shifted(-1)
    DS.zero(a_p.ex * (u_p.ex * C_ex + prev - rem.ex - car.ex * Ex.const(1 << 16)), tag=tag + ".step")
    DS.zero(rem.ex, sel=DS.rng("k", 12, 16), tag=tag + ".rem0")
    DS.zero(car.ex, sel=DS.rng("k", 12, 16), tag=tag + ".car0")
    G_.range_check(DS, rem.ex, rem_v, 16, tag + ".rrem")
    G_.range_check(DS, car.ex, car_v, 46, tag + ".rcar")
    return DS, car


# ============================================================================ global token chain (specs/01 §6.1)
VOL_SYS_LEN, VOL_SYS0 = 7, MC.VOL_SYS0   # modelcommit.encode_volatile_cfg: sys_len, first sys_prompt token (32)


def token_chain(ctx, prf, stmt, cfg_ext, wit=None):
    """Hidden token column tok[t] over T_pad rows with nested prefix masks minj <= mpre <= mcot <= mtot:
      [0, a)  hidden injection = M^vol sys_prompt (a = sys_len, both read in-circuit from the committed vol column, which
              the HCO opens to C_cfg -> model_id);
      [a, b)  public request tokens, b = a + n_req;   [b, c) hidden CoT, c = b + n_reason (public usage count);
      [c, T)  public visible reply tokens, T = c + n_vis.
    T (and a) stay hidden: only n_req, n_reason, n_vis and T_pad are public (standard API).  Returns (SP, tok col, T col)."""
    pv = wit is not None
    G_ = __import__("flashprove_next.gadgets", fromlist=["x"])
    TB, N = prf.TB, 1 << prf.TB
    SP = ctx.space("rzTok", [("t", TB)])
    req, vis, n_r = stmt["req_tokens"], stmt["vis_tokens"], int(stmt["n_reason"])
    vb = cfg_ext.vol_col.space.bits

    def vol_at(sp, idx):
        return MZ.view(cfg_ext.vol_col, sp, {"x": [("c", (idx >> k) & 1) for k in range(vb)]})

    lay_a = _api_layout(stmt)
    n_span = lay_a["n_span"]
    n_pre_vis = n_span - len(vis)
    if pv:
        toks = list(wit["tokens"]); a = int(wit["sys_len"])
        assert len(toks) <= N
        tv = np.zeros(N, dtype=np.int64); tv[:len(toks)] = toks
        n_p0 = SZ.n_prev(wit)                      # previous hidden CoT re-injected at [a, a + n_p0)
        b, c, T = a + n_p0 + len(req), a + n_p0 + len(req) + n_pre_vis, len(toks)
        assert T == c + len(vis), (T, c, len(vis))
        mk = lambda n: (np.arange(N) < n).astype(np.int64)
        mv = [mk(a), mk(b), mk(c), mk(T)]
    else:
        tv = None; mv = [None] * 4
    tok = SP.col("tok", tv)
    m = [SP.col(nm, v) for nm, v in zip(("minj", "mpre", "mcot", "mtot"), mv)]
    one = Ex.const(1)
    for k, mc in enumerate(m):
        SP.zero(mc.ex * (mc.ex - one), tag=f"rz.m{k}.bool")
        dm = mc.ex - mc.shifted(1)
        SP.zero(dm * (dm - one), sel=SP.lt("t", N - 1), tag=f"rz.m{k}.mono")
    for k in range(3):
        dd = m[k + 1].ex - m[k].ex
        SP.zero(dd * (dd - one), tag=f"rz.nest{k}")
    i0 = SP.ind(t=0)
    # a = sys_len: ONE pinned claim on vol (zk_budget: every pinned evaluation needs its own random row); every other
    # use reads the hidden replicated scalar a
    SCa = TZ.Scalars(ctx)
    SCa.add("rz.a", (int(wit["sys_len"]), 0, 0) if pv else None)
    SZ.declare_scalars(SCa, stmt, wit)
    a_ex = SCa.on("rz.a", SP)
    a2_ex = SZ.start_ex(SCa, stmt, SP)          # a + l' (previous hidden CoT re-injected after the system prompt)
    n_p = SZ.n_prev(wit) if pv else 0
    SP.zero((a_ex - vol_at(SP, VOL_SYS_LEN)) * i0, tag="rz.a.vol")
    SP.sumzero(m[0].ex - a_ex * i0, tag="rz.sum.a")
    SP.sumzero(m[1].ex - (a2_ex + Ex.const(len(req))) * i0, tag="rz.sum.b")
    SP.sumzero(m[2].ex - (a2_ex + Ex.const(len(req) + n_pre_vis)) * i0, tag="rz.sum.c")
    SP.sumzero(m[3].ex - (a2_ex + Ex.const(len(req) + n_span)) * i0, tag="rz.sum.T")
    # L = L_real from M^vol (one pinned claim); Pi_2 pools over the forward positions [0, T - 1):
    #   pool 0 (input layer)      t in [0, b - 1),   pool 1 (generation layer)  t in [b - 1, T - 1)  (N = T - b public)
    # unit (t, l) costs c(t) = c_s + (t + 1) c_k (self + key atoms, specs/01 §4.5); S0 = S(b - 1), S1 = S(T - 1) - S(b - 1)
    # with S(t) = t c_s + c_k t (t + 1) / 2; draw modulus of pool j: C_j = L S_j (hidden scalars)
    n_req, N = len(req), n_span
    cs, ck = prf.C_SELF, prf.C_KEY
    if pv:
        Lv = int(wit["L"]); bv = int(wit["sys_len"]) + n_p + n_req; Tv = bv + N
        S0v, S1v = pool_sums(bv, Tv, cs, ck)
    names = {"rz.L": Lv if pv else 0, "rz.b2": ((bv - 1) * bv // 2) if pv else 0,
             "rz.T2": ((Tv - 1) * Tv // 2) if pv else 0, "rz.S0": S0v if pv else 0, "rz.S1": S1v if pv else 0,
             "rz.C0": Lv * S0v if pv else 0, "rz.C1": Lv * S1v if pv else 0}
    for nm, v in names.items():
        SCa.add(nm, (int(v), 0, 0) if pv else None)
    on = lambda nm: SCa.on(nm, SP)
    L_ex = on("rz.L")
    b_ex = a2_ex + Ex.const(n_req)
    T_ex = b_ex + Ex.const(N)
    SP.zero((L_ex - vol_at(SP, VOL_L_REAL)) * i0, tag="rz.L.vol")
    # Bind committed sampling rule (T_bits, p16, beta) in M^vol -> C_cfg -> model_id to the profile's public rule.
    SP.zero((vol_at(SP, VOL_T_BITS) - Ex.const(int(prf.rule.T_bits))) * i0, tag="rz.rule.T")
    SP.zero((vol_at(SP, VOL_P16) - Ex.const(int(prf.rule.p16))) * i0, tag="rz.rule.p")
    SP.zero((vol_at(SP, VOL_BETA) - Ex.const(int(prf.rule.beta))) * i0, tag="rz.rule.b")
    SP.zero((on("rz.b2") * Ex.const(2) - (b_ex - Ex.const(1)) * b_ex) * i0, tag="rz.b2")
    SP.zero((on("rz.T2") * Ex.const(2) - (T_ex - Ex.const(1)) * T_ex) * i0, tag="rz.T2")
    SP.zero((on("rz.S0") - (b_ex - Ex.const(1)) * Ex.const(cs) - on("rz.b2") * Ex.const(ck)) * i0, tag="rz.S0")
    SP.zero((on("rz.S1") - Ex.const(N * cs) - (on("rz.T2") - on("rz.b2")) * Ex.const(ck)) * i0, tag="rz.S1")
    SP.zero((on("rz.C0") - L_ex * on("rz.S0")) * i0, tag="rz.C0")
    SP.zero((on("rz.C1") - L_ex * on("rz.S1")) * i0, tag="rz.C1")
    # hidden injection equals the committed sys_prompt tokens: one view vol[VOL_SYS0 + t] (aligned offset, free t bits)
    sb = _lg(prf.SYS_CAP)
    assert prf.SYS_CAP == 1 << sb and VOL_SYS0 % prf.SYS_CAP == 0 and sb <= TB
    xmap = [TZ.var_ref(SP, k) for k in range(sb)] + [("c", (VOL_SYS0 >> k) & 1) for k in range(sb, vb)]
    sysv = MZ.view(cfg_ext.vol_col, SP, {"x": xmap})
    SP.zero(m[0].ex * (tok.ex - sysv), sel=SP.lt("t", prf.SYS_CAP), tag="rz.sys")
    if prf.SYS_CAP < (1 << TB):
        SP.zero(m[0].ex, sel=SP.rng("t", prf.SYS_CAP, 1 << TB), tag="rz.sys.cap")
    # (t, tok[t]) table: public request / visible-reply tokens (below) and the EMB blocks look up into it
    G_.wire_table(ctx, "rz.tokt", SP, [SP.idx("t")], [tok], key_vals=[np.arange(SP.n, dtype=np.int64)] if pv else None)
    # public request / tool / visible-reply tokens at hidden offsets: LogUp of (pos, value) into (t, tok[t])
    pub = lay_a["pub_binds"]
    if pub:
        KB = max(5, _lg(len(pub)))
        BS = ctx.space("rzBind", [("k", KB)])
        offs = np.zeros(BS.n, dtype=np.int64); vals = np.zeros(BS.n, dtype=np.int64)
        for k, (base, j, x) in enumerate(pub):
            offs[k], vals[k] = base + j, x
        pos_v = (offs + int(wit["sys_len"]) + n_p) if pv else None
        pos = BS.col("pos", pos_v)
        off_p = BS.pub("off", offs)
        val_p = BS.pub("val", vals)
        sel = BS.lt("k", len(pub))
        BS.zero(pos.ex - (SZ.start_ex(SCa, stmt, BS) + off_p.ex), sel=sel, tag="rz.bind.pos")
        sv = (np.arange(BS.n) < len(pub)).astype(np.int64) if pv else None
        G_.wire_lookup(ctx, "rz.tokt", BS, [pos.ex, val_p.ex], [pos_v, vals] if pv else None, sel=sel, sel_vals=sv)

    # In-circuit termination & finish_reason verification across ALL generated steps (including hidden CoT)
    am = stmt.get("api_meta") or {}
    fr = lay_a["finish_reason"]
    n_gen_tot = lay_a["n_gen"]
    if fr is not None and n_gen_tot > 0:
        eos_id = am.get("eos_token_id")
        tend_id = am.get("tool_end_token_id")
        seg_ends = lay_a["seg_end_steps"]
        gen_offs = lay_a["gen_rel_offs"]
        TB_g = max(5, _lg(n_gen_tot))
        TSg = ctx.space("rzTerm", [("g", TB_g)])
        NT = TSg.n
        goff_arr = np.zeros(NT, dtype=np.int64); goff_arr[:n_gen_tot] = gen_offs
        must_eos = np.zeros(NT, dtype=np.int64)
        must_not_eos = np.zeros(NT, dtype=np.int64)
        must_tend = np.zeros(NT, dtype=np.int64)
        must_not_tend = np.zeros(NT, dtype=np.int64)
        for g_i in range(n_gen_tot):
            ek = seg_ends.get(g_i)
            if eos_id is not None:
                if ek == APZ.FINISH_STOP:
                    must_eos[g_i] = 1
                else:
                    must_not_eos[g_i] = 1
            if tend_id is not None:
                if ek == APZ.FINISH_TOOL:
                    must_tend[g_i] = 1
                else:
                    must_not_tend[g_i] = 1
        tpos_v = tval_v = inv_eos_v = inv_tend_v = None
        if pv:
            b_base = int(wit["sys_len"]) + n_p + n_req
            tpos_v = np.zeros(NT, dtype=np.int64)
            tval_v = np.zeros(NT, dtype=np.int64)
            for g_i in range(n_gen_tot):
                p_i = b_base + gen_offs[g_i]
                tpos_v[g_i] = p_i
                tval_v[g_i] = int(toks[p_i]) if p_i < len(toks) else 0
            if eos_id is not None:
                de = np.array([(int(tval_v[i]) - int(eos_id)) % P for i in range(NT)], dtype=np.uint64)
                inv_eos_v = np.ascontiguousarray(CPU.einv(CPU.ext(de))[0])
                if np.any((must_eos[:n_gen_tot] == 1) & (tval_v[:n_gen_tot] != int(eos_id))):
                    ctx.violation("finish_reason 'stop' violated: terminal generated token != eos_token_id")
                if np.any((must_not_eos[:n_gen_tot] == 1) & (tval_v[:n_gen_tot] == int(eos_id))):
                    ctx.violation("premature or forbidden eos_token_id in generated tokens")
            if tend_id is not None:
                dt = np.array([(int(tval_v[i]) - int(tend_id)) % P for i in range(NT)], dtype=np.uint64)
                inv_tend_v = np.ascontiguousarray(CPU.einv(CPU.ext(dt))[0])
                if np.any((must_tend[:n_gen_tot] == 1) & (tval_v[:n_gen_tot] != int(tend_id))):
                    ctx.violation("tool_use violated: segment terminal token != tool_end_token_id")
                if np.any((must_not_tend[:n_gen_tot] == 1) & (tval_v[:n_gen_tot] == int(tend_id))):
                    ctx.violation("premature or forbidden tool_end_token_id in generated tokens")
        tpos_c = TSg.col("pos", tpos_v)
        tval_c = TSg.col("tok", tval_v)
        goff_p = TSg.pub("off", goff_arr)
        sel_g = TSg.lt("g", n_gen_tot)
        sv_g = (np.arange(NT) < n_gen_tot).astype(np.int64) if pv else None
        TSg.zero(tpos_c.ex - (SZ.start_ex(SCa, stmt, TSg) + Ex.const(n_req) + goff_p.ex), sel=sel_g, tag="rz.term.pos")
        G_.wire_lookup(ctx, "rz.tokt", TSg, [tpos_c.ex, tval_c.ex], [tpos_v, tval_v] if pv else None,
                       sel=sel_g, sel_vals=sv_g)
        if eos_id is not None:
            ie_c = TSg.col("inveos", inv_eos_v)
            me_p = TSg.pub("meos", must_eos)
            mne_p = TSg.pub("mneos", must_not_eos)
            TSg.zero(me_p.ex * (tval_c.ex - Ex.const(int(eos_id))), tag="rz.term.eos")
            TSg.zero(mne_p.ex * ((tval_c.ex - Ex.const(int(eos_id))) * ie_c.ex - Ex.const(1)), tag="rz.term.noeos")
        if tend_id is not None:
            it_c = TSg.col("invtend", inv_tend_v)
            mt_p = TSg.pub("mtend", must_tend)
            mnt_p = TSg.pub("mntend", must_not_tend)
            TSg.zero(mt_p.ex * (tval_c.ex - Ex.const(int(tend_id))), tag="rz.term.tend")
            TSg.zero(mnt_p.ex * ((tval_c.ex - Ex.const(int(tend_id))) * it_c.ex - Ex.const(1)), tag="rz.term.notend")
    return SP, tok, m, SCa


# ============================================================================ circuit
class _DecRec:
    """records the code column of every decode_bf16 call made while a block is built."""

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


def _provider(em, prf, tens, l, lp1_x, pos_x, prover):
    pre = f"model.layers.{l}."
    mats = {}
    for k, hf in MAT_MAP.items():
        c = tens[pre + hf + ".weight"] if prover else None
        mats[k] = MC.aligned_mat(c, prf.prm)[0] if prover else CZ.Mat(None, 0, None, {
            "q": prf.d, "k": prf.d, "v": prf.d, "o": prf.n_q * prf.dh, "gate": prf.d, "up": prf.d,
            "down": prf.dff}[k])
    z = lambda n: np.zeros(n, dtype=np.uint16)
    g = (lambda nm, n: tens[pre + nm + ".weight"]) if prover else (lambda nm, n: z(n))
    vecs = {"input_layernorm": g("input_layernorm", prf.d), "post_attention_layernorm": g("post_attention_layernorm", prf.d),
            "q_norm": g("self_attn.q_norm", prf.dh), "k_norm": g("self_attn.k_norm", prf.dh)}
    from .fp_ref import EPS_BITS, rope_table
    return WS.CfgTiledWeights(em, mats, vecs, tensors={k: pre + MAT_MAP[k] + ".weight" for k in MAT_MAP},
                              role_ids={k: MC.ROLE_IDS[MAT_MAP[k].split(".")[-1]] for k in MAT_MAP},
                              layer_p1=lp1_x, s_bits=prf.s_bits, eps_bits=EPS_BITS,
                              rope=lambda p: tuple(a.astype(np.float64) for a in rope_table(p, prf.dh)),
                              pos_x=pos_x, t_pad=prf.T_pad, scale=1.0 / np.sqrt(prf.dh))


def build(prf, stmt, wit=None, B=CPU, info=None, *, expected_c_pre=None, expected_c_ext=None):
    G_ = __import__("flashprove_next.gadgets", fromlist=["x"])
    """prover: wit = {cap, tensors, store, pre, vol, hs_wit}; verifier: wit None.  Returns (em, ws, hs)."""
    from .fp_ref import rope_table
    info = {} if info is None else info
    check_statement(
        prf, stmt,
        expected_c_pre=expected_c_pre,
        expected_c_ext=expected_c_ext,
        require_external_pins=(wit is None),
    )
    pv = wit is not None
    em = WS.EmHG(emit=True, prove=pv)
    em.B = B
    ctx = em.ctx
    cfg_ext = MZ.CfgExt(ctx, prf.prm, store=wit["store"] if pv else None, vol_elems=wit["vol"] if pv else None,
                        cfg_rows=True, vol_blocks=int(stmt["vol_blocks"]), meta_cfg0=int(stmt["meta_cfg0"]),
                        meta_cfg1=int(stmt["meta_cfg1"]))
    em.cfg_ext = cfg_ext
    em.vec_ext = WS.VecExt(ctx, prf.prm, wit["store"] if pv else None, stmt["vec_row"], stmt["vec_pad"], cfg_ext)
    hs = HZ.HSel(ctx, prf.tier, prf.J, wit["hs_wit"] if pv else None, stmt, tag="rhs", B=B)
    gp = prf.gprm()
    d, dff, nqd = prf.d, prf.dff, prf.n_q * prf.dh
    nkd = prf.n_kv * prf.dh
    eps_b = prf.eps_bits
    rope_fn = prf.rope_fn or (lambda p: tuple(a.astype(np.float64) for a in rope_table(p, prf.dh)))
    t0 = time.time()

    att_rounds = []
    mat_cache = {}

    def aligned_cached(tname):
        hit = mat_cache.get(tname)
        if hit is None:
            while len(mat_cache) >= 24:
                mat_cache.pop(next(iter(mat_cache)))
            hit = mat_cache[tname] = MC.aligned_mat(wit["tensors"][tname], prf.prm)[0]
        return hit

    def emb_check_batched(em_b, provs, SC_b, pos_x, pos_ents, pos_vs, n_act, tag):
        """Batched EMB across K = 1 << gsb positions on child Em `em_b` (`SC_b` has axes `[('s', gsb)]`):
        H[pos_s, (layer 0, emb)] = E[tok[pos_s]] with (pos_s, tok_s) LogUp-bound to `rz.tokt`."""
        TI = prf.prm.TI
        gsb = em_b.bbits
        K_b = 1 << gsb
        if pv:
            toks_w = wit["tokens"]
            tok_vs = np.array([int(toks_w[p]) if p < len(toks_w) else 0 for p in pos_vs], dtype=np.int64)
            kv_arr, jv_arr = np.divmod(tok_vs, TI)
        else:
            tok_vs = kv_arr = jv_arr = None
        tokc = em_b.col(SC_b, tag + ".tok", tok_vs, 20)
        kc = em_b.col(SC_b, tag + ".k", kv_arr, 10)
        jc = em_b.col(SC_b, tag + ".j", jv_arr, _lg(TI) + 1)
        em_b.zero(SC_b, lambda: tokc.ex - kc.ex * Ex.const(TI) - jc.ex, tag + ".kj")
        em_b.rng(jc, max(1, _lg(TI)), tag + ".jr")
        em_b.rng(kc, 10, tag + ".kr")
        G_.wire_lookup(ctx, "rz.tokt", SC_b.S, [pos_x.ex, tokc.ex],
                       [np.asarray(pos_vs, dtype=np.int64), tok_vs] if pv else None)
        prov_e = WS.BatchedCfgProvider(
            em_b, {"embT": TI},
            role_ids={"embT": MC.ROLE_IDS["embed_tokens_t"]},
            layer_p1=0, expert=kc, s_bits=prf.s_bits,
            mat_getter=(lambda nm, s_: aligned_cached(emb_block(int(kv_arr[s_])))) if pv else None,
            tensors_fn=(lambda nm, s_: emb_block(int(kv_arr[s_]))) if pv else None,
            eps_bits=eps_b, rope=lambda p: (np.zeros(prf.dh), np.zeros(prf.dh)),
            pos_x=pos_x, t_pad=prf.T_pad, scale=1.0 / np.sqrt(prf.dh),
        )
        provs.append(prov_e)
        if pv:
            cap = wit["cap"]
            oh = np.zeros((K_b, TI), dtype=np.uint16)
            oh[np.arange(K_b), jv_arr] = 0x3F80
            ho_v = np.stack([cap.h_in[int(p), 0] for p in pos_vs])
        else:
            oh = np.zeros((K_b, TI), dtype=np.uint16)
            ho_v = np.zeros((K_b, d), dtype=np.uint16)
        zr_v = np.zeros((K_b, d), dtype=np.uint16)
        with _DecRec() as rec:
            CZ.blk_gemm_res(em_b, prov_e, "embT", oh, zr_v, ho_v, TI, d, tag)
        decs = {dtag.split(".")[-1]: (sp, dec) for dtag, sp, dec in rec.recs}
        assert set(decs) == {"xi", "hr", "ho"}, decs.keys()
        sx, dx = decs["xi"]
        C = dx.code.col.ex
        sx.S.zero(C * (C - Ex.const(0x3F80)), tag=tag + ".oh")
        sx.S.sumzero(C - Ex.const(0x3F80) * sx.S.ind(i=0), eq_axes=["s"], tag=tag + ".one")
        sx.S.sumzero(C * sx.S.idx("i") - Ex.const(0x3F80) * em_b.view(jc, sx).ex * sx.S.ind(i=0),
                     eq_axes=["s"], tag=tag + ".at")
        sh, dhr = decs["hr"]
        sh.S.zero(dhr.code.col.ex, tag=tag + ".nores")
        so, dho = decs["ho"]
        db = so.in_bits
        assert db == prf.pbits["emb"]
        hi = prf.poff["emb"] >> db
        pre_e = [("const", (hi >> k) & 1) for k in range(prf.WB - db)] + [("const", 0)] * prf.LB
        hs.bind_batched_vector(dho.code.col, gsb, db, pos_ents[:n_act], [pre_e] * n_act,
                               vec_wits=np.asarray(dho.code.v, dtype=np.int64).reshape(K_b, -1) if pv else None,
                               n_active=n_act)

    def blocks_fn(em_):
        provs = []
        # ---- 1. Batched hidden injection region: EMB at EVERY position p < a (a = hidden sys_len <= SYS_CAP)
        K_inj = max(2, 1 << _lg(max(1, prf.SYS_CAP)))
        sb_inj = _lg(K_inj)
        em_inj = em_.sub_em([("s", sb_inj)])
        SCi = em_inj.space("inj", [])
        p_idx = np.array([p_ if p_ < prf.SYS_CAP else 0 for p_ in range(K_inj)], dtype=np.int64)
        p_pub = em_inj.pub(SCi, "p", p_idx, 16)
        if pv:
            a_v = int(wit["sys_len"])
            fv_i = (p_idx < a_v).astype(np.int64)
            rv_i = p_idx * fv_i
            gv_i = np.where(fv_i == 1, a_v - 1 - p_idx, p_idx - a_v)
        else:
            fv_i = rv_i = gv_i = None
        fc_i = em_inj.col(SCi, "f", fv_i, 1)
        rc_i = em_inj.col(SCi, "r", rv_i, prf.TB + 1)
        rb_i = [em_inj.col(SCi, f"rb{k}", ((rv_i >> k) & 1) if pv else None, 1) for k in range(prf.TB)]
        for x_ in [fc_i] + rb_i:
            em_inj.boolean(x_, "inj.bit")
        em_inj.zero(SCi, lambda: rc_i.ex - fc_i.ex * p_pub.ex, "inj.r")
        em_inj.zero(SCi, lambda: rc_i.ex - sum((x_.ex * Ex.const(1 << k) for k, x_ in enumerate(rb_i)), Ex.const(0)),
                    "inj.rsum")
        a_on_i = SCa.on("rz.a", SCi.S)
        gc_i = em_inj.col(SCi, "g", gv_i, 17)
        em_inj.zero(SCi, lambda: gc_i.ex - (fc_i.ex * (a_on_i - (Ex.const(1) + p_pub.ex))
                                            + (Ex.const(1) - fc_i.ex) * (p_pub.ex - a_on_i)), "inj.g")
        em_inj.rng(gc_i, 16, "inj.gr")
        inj_ents = [[("hid", (SCi.S, rb_i[k].ex, s_), int((rv_i[s_] >> k) & 1) if pv else 0)
                     for k in range(prf.TB)] for s_ in range(prf.SYS_CAP)]
        emb_check_batched(em_inj, provs, SCi, rc_i, inj_ents,
                          rv_i.tolist() if pv else [0] * K_inj, prf.SYS_CAP, "inj.EMB")

        # ---- 1b. Deterministic EMB binding on committed tool-return witness slots (when prf.TOOL_CAP > 0)
        lay_a = _api_layout(stmt)
        n_req = len(stmt["req_tokens"])
        Nspan = lay_a["n_span"]
        Ngen = lay_a["n_gen"]
        gen_rel_offs = lay_a["gen_rel_offs"]
        if prf.TOOL_CAP > 0:
            K_tl = max(2, 1 << _lg(prf.TOOL_CAP))
            em_tl = em_.sub_em([("s", _lg(K_tl))])
            SCtl = em_tl.space("tool", [])
            t_rets = lay_a["tool_ret_rel_offs"]
            n_tr = len(t_rets)
            act_tl = np.zeros(K_tl, dtype=np.int64); act_tl[:n_tr] = 1
            off_tl = np.zeros(K_tl, dtype=np.int64); off_tl[:n_tr] = t_rets
            act_tl_p = em_tl.pub(SCtl, "act", act_tl, 1)
            off_tl_p = em_tl.pub(SCtl, "off", off_tl, 32)
            if pv:
                b_tl = int(wit["sys_len"]) + SZ.n_prev(wit) + n_req
                rv_tl = np.where(act_tl == 1, b_tl + off_tl, 0)
            else:
                rv_tl = None
            rc_tl = em_tl.col(SCtl, "r", rv_tl, prf.TB + 1)
            rb_tl = [em_tl.col(SCtl, f"rb{k}", ((rv_tl >> k) & 1) if pv else None, 1) for k in range(prf.TB)]
            for x_ in rb_tl:
                em_tl.boolean(x_, "tool.bit")
            b_on_tl = SZ.start_ex(SCa, stmt, SCtl.S) + Ex.const(n_req)
            em_tl.zero(SCtl, lambda: rc_tl.ex - act_tl_p.ex * (b_on_tl + off_tl_p.ex), "tool.r")
            em_tl.zero(SCtl, lambda: rc_tl.ex - sum((x_.ex * Ex.const(1 << k) for k, x_ in enumerate(rb_tl)), Ex.const(0)),
                       "tool.rsum")
            tl_ents = [[("hid", (SCtl.S, rb_tl[k].ex, s_), int((rv_tl[s_] >> k) & 1) if pv else 0)
                        for k in range(prf.TB)] for s_ in range(prf.TOOL_CAP)]
            emb_check_batched(em_tl, provs, SCtl, rc_tl, tl_ents,
                              rv_tl.tolist() if pv else [0] * K_tl, prf.TOOL_CAP, "tool.EMB")

        # ---- 2. Cross-slot batched compute slots (Pool 0 + Pool 1, K_C = 2^sb_c >= M_C)
        cs, ck = prf.C_SELF, prf.C_KEY
        M_C, K_C = prf.M_C, prf.K_C
        sb_c = _lg(K_C)
        em_c = em_.sub_em([("s", sb_c)])
        SCc = em_c.space("slotC", [])

        u_pad = [stmt["u_draws"][s_ if s_ < M_C else 0] for s_ in range(K_C)]
        pools_c = np.array([slot_pool(prf, Nspan, s_ if s_ < M_C else 0) for s_ in range(K_C)], dtype=np.int64)
        # Batched euclid_draw across all K_C slots on one space [s, k]
        DS = ctx.space("rzDr", [("s", sb_c), ("k", 4)])
        uv_2d = np.zeros((K_C, 16), dtype=np.int64)
        act_2d = np.zeros((K_C, 16), dtype=np.int64); act_2d[:, :12] = 1
        p_2d = np.repeat(pools_c[:, None], 16, axis=1)
        for s_ in range(K_C):
            uv_2d[s_, :12] = u_pad[s_]
        if pv:
            Lw = int(wit["L"]); bw = int(wit["sys_len"]) + SZ.n_prev(wit) + n_req; Tw = bw + Nspan
            S0w, S1w = pool_sums(bw, Tw, cs, ck)
            rem_2d = np.zeros((K_C, 16), dtype=np.int64)
            car_2d = np.zeros((K_C, 16), dtype=np.int64)
            t_arr = np.zeros(K_C, dtype=np.int64)
            l_arr = np.zeros(K_C, dtype=np.int64)
            Jp_arr = np.zeros(K_C, dtype=np.int64)
            wrap_hook = wit.get("draw_wrap_limb", {})
            j_arr = np.zeros(K_C, dtype=np.int64)
            c_arr = np.zeros(K_C, dtype=np.int64)
            for s_ in range(K_C):
                idx_orig = s_ if s_ < M_C else 0
                pl_s = int(pools_c[s_])
                C_s = Lw * (S0w if pl_s == 0 else S1w)
                c_arr[s_] = C_s
                c_run = 0
                w_limb = wrap_hook.get(idx_orig) if s_ < M_C else None
                for k_ in range(12):
                    xx = int(u_pad[s_][k_]) * C_s + c_run
                    if w_limb is not None and k_ == int(w_limb):
                        xx = xx + P
                    rem_2d[s_, k_], c_run = xx & 0xFFFF, xx >> 16
                    car_2d[s_, k_] = c_run
                Jv_actual = int(car_2d[s_, 11])
                j_arr[s_] = Jv_actual
                if w_limb is not None:
                    Jpv = Jv_actual // Lw
                    l_s = Jv_actual % Lw
                    s0_b = (bw - 1) * cs + ck * (bw - 1) * bw // 2 if pl_s == 1 else 0
                    target = s0_b + Jpv
                    t_s = 0
                    while t_s + 1 < Tw - 1 and (t_s + 1) * cs + ck * (t_s + 1) * (t_s + 2) // 2 <= target:
                        t_s += 1
                else:
                    Jv, t_s, l_s, Jpv = slot_of(u_pad[s_], bw, Tw, Lw, pl_s, cs, ck)
                    if idx_orig in wit.get("slot_override", {}):
                        t_s, l_s = wit["slot_override"][idx_orig]
                        Jpv = Jv // Lw
                t_arr[s_], l_arr[s_], Jp_arr[s_] = t_s, l_s, Jpv
            rem_v, car_v = rem_2d.ravel(), car_2d.ravel()
        else:
            Lw = bw = Tw = S0w = S1w = 0
            rem_v = car_v = t_arr = l_arr = Jp_arr = j_arr = c_arr = None

        rem_c = DS.col("rem", rem_v)
        car_c = DS.col("car", car_v)
        u_pub_d = DS.pub("u", uv_2d.ravel())
        a_pub_d = DS.pub("act", act_2d.ravel())
        p_pub_d = DS.pub("pool", p_2d.ravel())
        C_ex_d = (Ex.const(1) - p_pub_d.ex) * SCa.on("rz.C0", DS) + p_pub_d.ex * SCa.on("rz.C1", DS)
        prev_d = (Ex.const(1) - DS.ind(k=0)) * car_c.shifted(-1)
        DS.zero(a_pub_d.ex * (u_pub_d.ex * C_ex_d + prev_d - rem_c.ex - car_c.ex * Ex.const(1 << 16)), tag="rzDr.step")
        DS.zero(rem_c.ex, sel=DS.rng("k", 12, 16), tag="rzDr.rem0")
        DS.zero(car_c.ex, sel=DS.rng("k", 12, 16), tag="rzDr.car0")
        G_.range_check(DS, rem_c.ex, rem_v, 16, "rzDr.rrem")
        G_.range_check(DS, car_c.ex, car_v, 46, "rzDr.rcar")

        tc = em_c.col(SCc, "t", t_arr, prf.TB + 1)
        lc = em_c.col(SCc, "l", l_arr, prf.LB + 1)
        tb = [em_c.col(SCc, f"tb{k}", ((t_arr >> k) & 1) if pv else None, 1) for k in range(prf.TB)]
        lb = [em_c.col(SCc, f"lb{k}", ((l_arr >> k) & 1) if pv else None, 1) for k in range(prf.LB)]
        for x_ in tb + lb:
            em_c.boolean(x_, "slotC.bit")
        em_c.zero(SCc, lambda: tc.ex - sum((x_.ex * Ex.const(1 << k) for k, x_ in enumerate(tb)), Ex.const(0)), "slotC.tsum")
        em_c.zero(SCc, lambda: lc.ex - sum((x_.ex * Ex.const(1 << k) for k, x_ in enumerate(lb)), Ex.const(0)), "slotC.lsum")
        J_ex = MZ.view(car_c, SCc.S, {"s": "s", "k": [("c", (11 >> k) & 1) for k in range(4)]})
        L_ex = SCa.on("rz.L", SCc.S)
        jp = em_c.col(SCc, "jp", Jp_arr, 46)
        em_c.zero(SCc, lambda: jp.ex * L_ex + lc.ex - J_ex, "slotC.draw")
        lgap = em_c.col(SCc, "lgap", (Lw - 1 - l_arr) if pv else None, prf.LB + 1)
        em_c.zero(SCc, lambda: lgap.ex - (L_ex - Ex.const(1) - lc.ex), "slotC.lgap")
        em_c.rng(lgap, 8, "slotC.lrange")
        tt = em_c.col(SCc, "tt", (t_arr * (t_arr + 1) // 2) if pv else None, 2 * prf.TB + 1)
        em_c.zero(SCc, lambda: tt.ex * Ex.const(2) - tc.ex * (tc.ex + Ex.const(1)), "slotC.tt")
        p_sc = em_c.pub(SCc, "pool", pools_c, 1)
        C_sc_ex = lambda: (Ex.const(1) - p_sc.ex) * SCa.on("rz.C0", SCc.S) + p_sc.ex * SCa.on("rz.C1", SCc.S)
        jgap = em_c.col(SCc, "jgap", (c_arr - 1 - j_arr) if pv else None, 46)
        em_c.zero(SCc, lambda: jgap.ex - (C_sc_ex() - Ex.const(1) - J_ex), "slotC.jgap")
        em_c.rng(jgap, 46, "slotC.jrange")
        St_ex = lambda: tc.ex * Ex.const(cs) + tt.ex * Ex.const(ck)
        S0_ex = lambda: p_sc.ex * ((SZ.start_ex(SCa, stmt, SCc.S) + Ex.const(n_req - 1)) * Ex.const(cs)
                                   + SCa.on("rz.b2", SCc.S) * Ex.const(ck))
        if pv:
            s0_base = (bw - 1) * cs + ck * (bw - 1) * bw // 2
            s0_arr = pools_c * s0_base
            St_arr = t_arr * cs + ck * t_arr * (t_arr + 1) // 2
            g1_v = Jp_arr - (St_arr - s0_arr)
            g2_v = (St_arr - s0_arr) + cs + (t_arr + 1) * ck - 1 - Jp_arr
            is0_arr = (l_arr == 0).astype(np.int64)
            lp_arr = l_arr - 1 + is0_arr
        else:
            g1_v = g2_v = is0_arr = lp_arr = None
        g1 = em_c.col(SCc, "g1", g1_v, 46)
        g2 = em_c.col(SCc, "g2", g2_v, 46)
        em_c.zero(SCc, lambda: g1.ex - (jp.ex - (St_ex() - S0_ex())), "slotC.g1")
        em_c.zero(SCc, lambda: g2.ex - (St_ex() - S0_ex() + Ex.const(cs) + (tc.ex + Ex.const(1)) * Ex.const(ck)
                                        - Ex.const(1) - jp.ex), "slotC.g2")
        em_c.rng(g1, 46, "slotC.g1r")
        em_c.rng(g2, 46, "slotC.g2r")
        lp1 = em_c.commit(em_c.lin([(1, lc)], 1, bits=prf.LB + 1), "slotC.lp1")
        is0 = em_c.col(SCc, "is0", is0_arr, 1)
        n0 = em_c.col(SCc, "n0", (1 - is0_arr) if pv else None, 1)
        lpb = [em_c.col(SCc, f"lpb{k}", ((lp_arr >> k) & 1) if pv else None, 1) for k in range(prf.LB)]
        for x_ in [is0, n0] + lpb:
            em_c.boolean(x_, "slotC.rbit")
        em_c.zero(SCc, lambda: is0.ex + n0.ex - Ex.const(1), "slotC.n0")
        em_c.zero(SCc, lambda: is0.ex * lc.ex, "slotC.is0")
        em_c.zero(SCc, lambda: sum((x_.ex * Ex.const(1 << k) for k, x_ in enumerate(lpb)), Ex.const(0))
                  - (lc.ex - Ex.const(1) + is0.ex), "slotC.lp")

        hid_rows = [[("hid", (SCc.S, tb[k].ex, s_), int((t_arr[s_] >> k) & 1) if pv else 0)
                     for k in range(prf.TB)] for s_ in range(M_C)]
        hid_ls = [[("hid", (SCc.S, lb[k].ex, s_), int((l_arr[s_] >> k) & 1) if pv else 0)
                   for k in range(prf.LB)] for s_ in range(M_C)]
        hid_lps = [[("hid", (SCc.S, lpb[k].ex, s_), int((lp_arr[s_] >> k) & 1) if pv else 0)
                    for k in range(prf.LB)] for s_ in range(M_C)]
        sel_ents = [{0: ("hid", (SCc.S, n0.ex, s_), int(1 - is0_arr[s_]) if pv else 0),
                     1: ("hid", (SCc.S, is0.ex, s_), int(is0_arr[s_]) if pv else 0)} for s_ in range(M_C)]

        # Batched provider for compute slots
        n_ins_c = {"q": d, "k": d, "v": d, "o": nqd, "gate": d, "up": d, "down": dff}
        role_ids_c = {k: MC.ROLE_IDS[MAT_MAP[k].split(".")[-1]] for k in MAT_MAP}
        if pv:
            tens = wit["tensors"]
            vg = lambda nm: np.stack([tens[f"model.layers.{int(l_arr[s_])}.{nm}.weight"] for s_ in range(K_C)])
            vecs_c = {"input_layernorm": vg("input_layernorm"), "post_attention_layernorm": vg("post_attention_layernorm"),
                      "q_norm": vg("self_attn.q_norm"), "k_norm": vg("self_attn.k_norm")}
            mg_c = lambda nm, s_: aligned_cached(f"model.layers.{int(l_arr[s_])}.{MAT_MAP[nm]}.weight")
            tf_c = lambda nm, s_: f"model.layers.{int(l_arr[s_])}.{MAT_MAP[nm]}.weight"
        else:
            z2 = lambda n: np.zeros((K_C, n), dtype=np.uint16)
            vecs_c = {"input_layernorm": z2(d), "post_attention_layernorm": z2(d),
                      "q_norm": z2(prf.dh), "k_norm": z2(prf.dh)}
            mg_c = tf_c = None
        prov_c = WS.BatchedCfgProvider(
            em_c, n_ins_c, role_ids=role_ids_c, vecs=vecs_c, layer_p1=lp1, s_bits=prf.s_bits,
            mat_getter=mg_c, tensors_fn=tf_c, eps_bits=eps_b, rope=rope_fn,
            pos_x=tc, t_pad=prf.T_pad, scale=1.0 / np.sqrt(prf.dh),
        )
        provs.append(prov_c)

        def bind_decoded_ports(bk, tag, recs, row_ents_list):
            bound = set()
            for dtag, sp, dec in recs:
                suf = dtag[len(tag) + 1:]
                port = BIND[bk].get(suf)
                if port is None:
                    if suf.split(".")[-1] in WEIGHT_SIDE:
                        info.setdefault("weight_side_decodes", []).append(dtag)
                        continue
                    raise AssertionError(f"unbound decode column {dtag}")
                col = dec.code.col
                db = sp.in_bits
                if port == "h_in":
                    assert db == prf.pbits["h_out"] == prf.pbits["emb"]
                    ho, he = prf.poff["h_out"] >> db, prf.poff["emb"] >> db
                    pres = []
                    for s_ in range(M_C):
                        pc_s = []
                        for k in range(prf.WB - db):
                            bo, be = (ho >> k) & 1, (he >> k) & 1
                            pc_s.append(("const", bo) if bo == be else (sel_ents[s_][1] if be else sel_ents[s_][0]))
                        pres.append(pc_s + hid_lps[s_])
                else:
                    assert db == prf.pbits[port], (dtag, db, prf.pbits[port])
                    hi = prf.poff[port] >> db
                    c_pre = [("const", (hi >> k) & 1) for k in range(prf.WB - db)]
                    pres = [c_pre + hid_ls[s_] for s_ in range(M_C)]
                vw = np.asarray(dec.code.v, dtype=np.int64).reshape(K_C, -1) if pv else None
                hs.bind_batched_vector(col, sb_c, db, row_ents_list[:M_C], pres, vec_wits=vw, n_active=M_C)
                bound.add(suf)
            missing = set(BIND[bk]) - bound
            assert not missing, (bk, missing)

        cap = wit["cap"] if pv else None
        z_kc = lambda n: np.zeros((K_C, n), dtype=np.uint16)
        xg_c = (lambda f: np.stack([getattr(cap, f)[int(t_arr[s_]), int(l_arr[s_])] for s_ in range(K_C)])) if pv else None
        for bk in BLOCKS:
            tg = f"sC.{bk}"
            with _DecRec() as rec:
                if bk == "N1":
                    CZ.blk_N(em_c, prov_c, "input_layernorm", xg_c("h_in") if pv else z_kc(d),
                             xg_c("x_attn") if pv else z_kc(d), d, eps_b, tag=tg)
                elif bk == "QKV":
                    a_ = (xg_c("x_attn"), xg_c("q"), xg_c("k"), xg_c("v")) if pv else (z_kc(d), z_kc(nqd), z_kc(nkd), z_kc(nkd))
                    CZ.blk_qkv(em_c, prov_c, *a_, tc, gp, tag=tg)
                elif bk == "OPROJ":
                    a_ = (xg_c("o"), xg_c("h_in"), xg_c("h_mid")) if pv else (z_kc(nqd), z_kc(d), z_kc(d))
                    CZ.blk_gemm_res(em_c, prov_c, "o", *a_, nqd, d, tg)
                elif bk == "N2":
                    CZ.blk_N(em_c, prov_c, "post_attention_layernorm", xg_c("h_mid") if pv else z_kc(d),
                             xg_c("x_mlp") if pv else z_kc(d), d, eps_b, tag=tg)
                elif bk == "GU":
                    a_ = (xg_c("x_mlp"), xg_c("a")) if pv else (z_kc(d), z_kc(dff))
                    CZ.blk_gu(em_c, prov_c, *a_, d, dff, tag=tg)
                elif bk == "DOWN":
                    a_ = (xg_c("a"), xg_c("h_mid"), xg_c("h_out")) if pv else (z_kc(dff), z_kc(d), z_kc(d))
                    CZ.blk_gemm_res(em_c, prov_c, "down", *a_, dff, d, tg)
            bind_decoded_ports(bk, tg, rec.recs, hid_rows)

        emb_check_batched(em_c, provs, SCc, tc, hid_rows,
                          t_arr.tolist() if pv else [0] * K_C, M_C, "sC.EMB")

        # ---- 3. Cross-slot batched attention + key-producer (N1 + KV + EMB)
        if prf.attention:
            u_att_pad = [stmt["u_att"][s_ if s_ < M_C else 0] for s_ in range(K_C)]
            awit_fn = kv_l_fn = None
            if pv:
                lay, Hm, at = prf.layout, wit["H"], prf.atier
                kv_cache = {}

                def kv_l_fn(l_idx):
                    l_idx = int(l_idx)
                    hit = kv_cache.get(l_idx)
                    if hit is None:
                        cK = lay.col(l_idx, "K", 0); cV = lay.col(l_idx, "V", 0)
                        K_m = np.ascontiguousarray(Hm[:, cK:cK + nkd], dtype=np.int64).reshape(prf.T_pad, at.G, at.DH)
                        V_m = np.ascontiguousarray(Hm[:, cV:cV + nkd], dtype=np.int64).reshape(prf.T_pad, at.G, at.DH)
                        hit = kv_cache[l_idx] = (K_m, V_m)
                    return hit

                def awit_fn(s_idx):
                    t_s, l_s = int(t_arr[s_idx]), int(l_arr[s_idx])
                    tp_s = AZ.key_draw(u_att_pad[s_idx], t_s)
                    lse_u32 = ((lay.get(Hm, t_s, l_s, "lse_hi", prf.n_q).astype(np.uint32) << np.uint32(16))
                               | lay.get(Hm, t_s, l_s, "lse_lo", prf.n_q).astype(np.uint32))
                    return {"t": t_s, "l": l_s, "lse_u32": lse_u32,
                            "q_codes": lay.get(Hm, t_s, l_s, "q", nqd, prf.dh),
                            "o_codes": lay.get(Hm, t_s, l_s, "o", nqd),
                            "k_codes_tp": lay.get(Hm, tp_s, l_s, "k", nkd, prf.dh),
                            "v_codes_tp": lay.get(Hm, tp_s, l_s, "v", nkd)}

            batt = AZ.BatchedAttn(ctx, hs, prf.atier, prf.layout, sb_c, M_C, tc.col, hid_rows, hid_ls,
                                  u_att_pad, wit_fn=awit_fn, kv_layer_fn=kv_l_fn, tag="batt", B=B)
            att_rounds.append(batt.round)

            SPp = em_c.space("prodC", [])
            tpc = em_c.col(SPp, "tp", batt.tau_ps if pv else None, prf.TB + 1)
            SPp.S.zero(tpc.ex - MZ.view(batt.tp_col, SPp.S, {"s": "s"}), tag="sC.P.tp")
            if pv:
                vecs_p = {"input_layernorm": vecs_c["input_layernorm"], "k_norm": vecs_c["k_norm"]}
                mg_p = lambda nm, s_: aligned_cached(f"model.layers.{int(l_arr[s_])}.{MAT_MAP[nm]}.weight")
                tf_p = lambda nm, s_: f"model.layers.{int(l_arr[s_])}.{MAT_MAP[nm]}.weight"
            else:
                vecs_p = {"input_layernorm": vecs_c["input_layernorm"], "k_norm": vecs_c["k_norm"]}
                mg_p = tf_p = None
            prov_p = WS.BatchedCfgProvider(
                em_c, {"k": d, "v": d},
                role_ids={"k": MC.ROLE_IDS["k_proj"], "v": MC.ROLE_IDS["v_proj"]},
                vecs=vecs_p, layer_p1=lp1, s_bits=prf.s_bits,
                mat_getter=mg_p, tensors_fn=tf_p, eps_bits=eps_b, rope=rope_fn,
                pos_x=tpc, t_pad=prf.T_pad, scale=1.0 / np.sqrt(prf.dh),
            )
            provs.append(prov_p)
            xg_p = (lambda f: np.stack([getattr(cap, f)[int(batt.tau_ps[s_]), int(l_arr[s_])] for s_ in range(K_C)])) if pv else None
            for bk in PRODUCER_BLOCKS:
                tg = f"sC.P.{bk}"
                with _DecRec() as rec:
                    if bk == "N1":
                        CZ.blk_N(em_c, prov_p, "input_layernorm", xg_p("h_in") if pv else z_kc(d),
                                 xg_p("x_attn") if pv else z_kc(d), d, eps_b, tag=tg)
                    elif bk == "KV":
                        a_ = (xg_p("x_attn"), xg_p("k"), xg_p("v")) if pv else (z_kc(d), z_kc(nkd), z_kc(nkd))
                        CZ.blk_kv(em_c, prov_p, *a_, tpc, gp, tag=tg)
                bind_decoded_ports(bk, tg, rec.recs, batt.tau_ents)
            emb_check_batched(em_c, provs, SPp, tpc, batt.tau_ents,
                              batt.tau_ps.tolist() if pv else [0] * K_C, M_C, "sC.P.EMB")

        # ---- 4. Pool 3: generation slots (specs/01 §5.4, specs/06 §3.2) via BatchedHead + sampler_zk
        am_cfg = stmt.get("api_meta") or {}
        schema_obj = am_cfg.get("schema")
        dfa_info = None
        if schema_obj is not None:
            if prf.token_bytes is None or len(prf.token_bytes) != prf.V:
                raise Z.ProofError("constrained decoding requires profile token_bytes of length V")
            pre_b = bytes.fromhex(am_cfg.get("schema_prefix_hex", ""))
            suf_b = bytes.fromhex(am_cfg.get("schema_suffix_hex", ""))
            dfa = APZ.compile_json_schema_dfa(schema_obj, prefix_bytes=pre_b, suffix_bytes=suf_b)
            if am_cfg.get("schema_on_tool_call") and am_cfg.get("tool_end_token_id") is not None:
                stop_ids = (int(am_cfg["tool_end_token_id"]),)
            elif am_cfg.get("eos_token_id") is not None and not am_cfg.get("schema_on_tool_call"):
                stop_ids = (int(am_cfg["eos_token_id"]),)
            else:
                stop_ids = tuple(int(x) for x in (am_cfg.get("eos_token_id"), am_cfg.get("tool_end_token_id")) if x is not None)
            next_q, allowed_mat = dfa.compile_token_masks(prf.token_bytes, stop_ids)
            # Verify public constrained token sequence against DFA (C2 / C4 / C7 of specs/05 §4.2)
            if am_cfg.get("schema_on_tool_call") and (am_cfg.get("tool_segments") or []):
                seg0 = am_cfg["tool_segments"][0]
                con_toks = [int(x) for x in seg0["call_tokens"]]
                con_start_step = int(seg0.get("n_reason", 0))
                fr_con = APZ.FINISH_TOOL
            else:
                con_toks = [int(x) for x in stmt["vis_tokens"]]
                con_start_step = lay_a["con_step_start"]
                fr_con = lay_a["finish_reason"]
            q_trace = []
            cur_q = 0
            for tid in con_toks:
                if not (0 <= tid < prf.V) or cur_q < 0 or cur_q > dfa.n_states:
                    raise Z.ProofError("constrained decoding token out of DFA state bounds")
                q_trace.append(cur_q)
                nxt_q = int(next_q[cur_q, tid])
                if nxt_q < 0:
                    raise Z.ProofError(f"constrained decoding violation: token {tid} disallowed in DFA state {cur_q}")
                cur_q = nxt_q
            if fr_con in (APZ.FINISH_STOP, APZ.FINISH_TOOL):
                if cur_q != dfa.n_states and cur_q not in dfa.accept:
                    raise Z.ProofError(f"constrained decoding incomplete at finish_reason={fr_con!r} (state {cur_q})")
            dfa_info = (dfa, next_q, allowed_mat, q_trace, con_start_step)

        if prf.M_G > 0 and Ngen > 0:
            K_G = prf.K_G
            g_draws = gen_steps_of(stmt["u_gen"], Ngen)
            assert len(g_draws) == prf.M_G
            g_steps = list(g_draws) + [g_draws[0]] * (K_G - len(g_draws))
            rel_steps = [gen_rel_offs[g] for g in g_steps]
            SCg = em_.space("genL", [])
            repg = lambda v: np.full(SCg.n, int(v), dtype=np.int64) if pv else None
            lm1_v = (Lw - 1) if pv else 0
            lm1 = em_.col(SCg, "lm1", repg(lm1_v), prf.LB + 1)
            lm1b = [em_.col(SCg, f"lb{k}", repg((lm1_v >> k) & 1), 1) for k in range(prf.LB)]
            for b_ in lm1b:
                em_.boolean(b_, "genL.bit")
            em_.zero(SCg, lambda: lm1.ex + Ex.const(1) - SCa.on("rz.L", SCg.S), "genL.def")
            em_.zero(SCg, lambda: lm1.ex - sum((b_.ex * Ex.const(1 << k) for k, b_ in enumerate(lm1b)), Ex.const(0)),
                     "genL.sum")
            hid_last_l = [("hid", (SCg.S, b_.ex, 0), (lm1_v >> k) & 1 if pv else 0) for k, b_ in enumerate(lm1b)]
            mat_cache.clear()
            if pv:
                t_tens = wit["tensors"]
                lm_key = "lm_head.weight" if "lm_head.weight" in t_tens else "model.embed_tokens.weight"
                M_lm = aligned_cached(lm_key)
                fn_vec = t_tens["model.norm.weight"]
            else:
                M_lm = CZ.Mat(None, 0, None, prf.d)
                fn_vec = np.zeros(prf.d, np.uint16)
            prov_g = WS.CfgTiledWeights(
                em_, {"lm": M_lm}, {"final_norm": fn_vec},
                tensors={"lm": "lm_head.weight"},
                role_ids={"lm": MC.ROLE_IDS["lm_head"]},
                vec_roles={"final_norm": MC.ROLE_IDS["norm"]},
                layer_p1=0, s_bits=prf.s_bits, eps_bits=eps_b,
                rope=lambda p: (np.zeros(prf.dh), np.zeros(prf.dh)),
                pos_x=lm1, t_pad=prf.T_pad, scale=1.0 / np.sqrt(prf.dh),
            )
            provs.append(prov_g)
            sctx = samp_ctx_of(stmt)
            prm_sz = SAMPZ.Prm(prf.V, prf.rule, K_G)
            gstmt = GZ.make_statement(prm_sz, None, sctx, [0] * K_G, g_steps, toks=None, hidden_t=False)
            q_per_slot = []
            act_per_slot = []
            if dfa_info is not None:
                _, _, allowed_mat, q_trace, con_start = dfa_info
                for g_idx in g_steps:
                    if g_idx >= con_start and (g_idx - con_start) < len(q_trace):
                        act_per_slot.append(1)
                        q_per_slot.append(int(q_trace[g_idx - con_start]))
                    else:
                        act_per_slot.append(0)
                        q_per_slot.append(0)
            gwit = None
            pos_vs = []
            if pv:
                toks_w = wit["tokens"]
                kprf_w = wit.get("kprf_override")
                if kprf_w is None:
                    kprf_w = [int(wit["vol"][3 + k]) % P for k in range(4)]
                slots_g = []
                mask_ovr = wit.get("gen_mask_override", {})
                for idx_s, g_idx in enumerate(g_steps):
                    pos_s = bw - 1 + rel_steps[idx_s]
                    if idx_s in wit.get("gen_pos_override", {}):
                        pos_s = int(wit["gen_pos_override"][idx_s])
                    pos_vs.append(pos_s)
                    tok_s = int(wit.get("gen_tok_override", {}).get(idx_s, toks_w[pos_s + 1]))
                    sl_d = dict(
                        h_out=cap.h_out[pos_s, Lw - 1],
                        x_fn=cap.x_fn[pos_s],
                        t=g_idx,
                        tok=tok_s,
                    )
                    if dfa_info is not None:
                        if idx_s in mask_ovr:
                            sl_d["a"] = np.asarray(mask_ovr[idx_s], dtype=bool)
                        elif act_per_slot[idx_s]:
                            sl_d["a"] = allowed_mat[q_per_slot[idx_s], :prf.V].copy()
                        else:
                            sl_d["a"] = np.ones(prf.V, dtype=bool)
                    slots_g.append(sl_d)
                gwit = dict(slots=slots_g, kprf=kprf_w, salt=(0, 0, 0, 0), ctx=sctx)
            vb = cfg_ext.vol_col.space.bits
            kprf_fn = lambda sp: [MZ.view(cfg_ext.vol_col, sp,
                                          {"x": [("c", ((3 + k) >> b) & 1) for b in range(vb)]}) for k in range(4)]
            gout = GZ.build_gen(em_, prm_sz, prov_g, gstmt, gwit, d=prf.d, V=prf.V,
                                eps_bits=eps_b, hidden_t=False, kprf_on_ab=kprf_fn,
                                all_allowed=(dfa_info is None), tag="rzG")
            gen_rounds = list(gout["rounds"])
            if dfa_info is not None:
                _, _, allowed_mat, _, _ = dfa_info
                mout = APZ.bind_constrained_masks(
                    em_, prm_sz, gout["VSs"], gout["As"], q_per_slot, act_per_slot, allowed_mat, tag="rzMask"
                )
                gen_rounds.append(mout["round"])
            info["gen_rounds"] = gen_rounds
            info["gen_slots"] = K_G
            tok_g = gout["tok"]
            em_gs = em_.sub_em([("s", prm_sz.sb)])
            SCgs = em_gs.space("genP", [])
            rel_pub = em_gs.pub(SCgs, "rel", np.asarray(rel_steps, dtype=np.int64), 32)
            pos_arr_g = np.asarray(pos_vs, dtype=np.int64) if pv else None
            pc_g = em_gs.col(SCgs, "pos", pos_arr_g, prf.TB + 1)
            pb_g = [em_gs.col(SCgs, f"pb{k}", ((pos_arr_g >> k) & 1) if pv else None, 1) for k in range(prf.TB)]
            for b_ in pb_g:
                em_gs.boolean(b_, "genP.bit")
            em_gs.zero(SCgs, lambda: pc_g.ex - sum((b_.ex * Ex.const(1 << k) for k, b_ in enumerate(pb_g)), Ex.const(0)),
                       "genP.psum")
            b_on_g = SZ.start_ex(SCa, stmt, SCgs.S) + Ex.const(n_req)
            em_gs.zero(SCgs, lambda: pc_g.ex - (b_on_g + rel_pub.ex - Ex.const(1)), "genP.pdef")
            hid_grows = [[("hid", (SCgs.S, pb_g[k].ex, idx_s), int((pos_arr_g[idx_s] >> k) & 1) if pv else 0)
                          for k in range(prf.TB)] for idx_s in range(K_G)]
            db_h = prf.pbits["h_out"]
            hi_h = prf.poff["h_out"] >> db_h
            pre_h = [("const", (hi_h >> k) & 1) for k in range(prf.WB - db_h)] + hid_last_l
            hb_col = gout["h_out_batched"]
            hs.bind_batched_vector(hb_col.col, prm_sz.sb, db_h, hid_grows, [pre_h] * K_G,
                                   vec_wits=np.asarray(hb_col.v, dtype=np.int64).reshape(K_G, -1) if pv else None,
                                   n_active=K_G)
            tok_on = em_gs.view(tok_g, SCgs, {"s": "s"})
            G_.wire_lookup(ctx, "rz.tokt", SCgs.S, [pc_g.ex + Ex.const(1), tok_on.ex],
                           [pos_arr_g + 1, tok_on.v[:K_G]] if pv else None)
        return provs

    SPt, tokc_, masks_, SCa = token_chain(ctx, prf, stmt, cfg_ext, wit)
    info["state"] = SZ.state_chain(ctx, prf, stmt, SPt, tokc_, masks_, "rz.a", SCa, wit,
                                   hist_rel_offs=_api_layout(stmt)["hist_rel_offs"])
    provs = blocks_fn(em)
    mat_cache.clear()
    if pv and hasattr(wit.get("tensors"), "ck") and hasattr(wit["tensors"].ck, "drop_cache"):
        wit["tensors"].ck.drop_cache()
    if pv:
        Z._logup_mult(ctx.cir, ctx.cir.lookups["rz.tokt"])      # after every lookup into the token table
    info["gadgets_build_s"] = round(time.time() - t0, 3)
    pre_rounds = list(info.get("gen_rounds", [])) + list(ctx.cir.rounds)
    ctx.cir.rounds = []
    ws = WS.WSum(em, prf.prm, provs, tile_of=wit["tile_of"] if pv else None, B=B, info=info, cfg_ext=cfg_ext,
                 store=wit["store"] if pv else None, vec_ext=em.vec_ext, J=prf.claim_slots)
    mz_wit = {"store": wit["store"], "pre": wit["pre"], "vol_elems": wit["vol"]} if pv else None
    MZ.build(prf.prm, stmt, mz_wit, B, info, ctx=ctx, ext=ws, cfg_ext=cfg_ext)
    ctx.cir.rounds = pre_rounds + list(ctx.cir.rounds) + att_rounds + hs.rounds()
    info["attention_slots"] = prf.M_C if prf.attention else 0
    return em, ws, hs


# ============================================================================ prove / verify
def make_wit(prf, model, cap, com, salts=None, H_cap=None, tokens=None, sys_len=0, B=CPU):
    """H_cap: the capture committed in H (forgery tests: differs from the capture the slots are computed on)."""
    H = build_H(cap if H_cap is None else H_cap, prf)
    hcom = HZ.commit(prf.tier, H, B=B)
    hstmt, hwit = HZ.make_witness(prf.tier, H, hcom, prf.J, salts=salts)
    lay = com["layout"]
    src_t = model.tensors() if hasattr(model, "tensors") else model
    wit = {"cap": cap, "tensors": commit_tensors(src_t, prf.prm.TI), "store": com["store"], "pre": com["pre"], "vol": com["vol"],
           "hs_wit": hwit, "tile_of": lambda t, R, C: lay.tile_of.get((t, R, C)), "H": H,
           "tokens": list(cap.tokens) if tokens is None else list(tokens), "sys_len": int(sys_len),
           "L": int(cap.h_in.shape[1])}
    return hstmt["R"], wit


_CUDA_ENGINE_CACHE = [None]


def _get_folded_cuda_engine():
    if _CUDA_ENGINE_CACHE[0] is None:
        from prover.fpn_prover.backend import cuda_engine as CE
        _CUDA_ENGINE_CACHE[0] = CE.CudaEngine()
    return _CUDA_ENGINE_CACHE[0]


def prove(prf, stmt, wit, B=CPU, *, backend="folded", queries=100, gkr_masks=16, engine=None):
    info = {}
    t0 = time.time()
    em, ws, hs = build(prf, stmt, wit, B, info)
    t1 = time.time()
    em.ctx.cir.pf = None
    cir = em.ctx.cir
    if backend == "folded":
        from prover.fpn_prover.backend import folded_engine as FE
        eng = engine
        if eng is None and getattr(B, "name", "cpu") == "cupy":
            eng = _get_folded_cuda_engine()
        pf, finfo = FE.prove_folded(
            cir, stmt, engine=eng, queries=int(queries), gkr_masks=int(gkr_masks),
            label=LABEL if isinstance(LABEL, bytes) else str(LABEL).encode("ascii"),
        )
        info.update(
            build_s=round(t1 - t0, 3),
            prove_s=round(time.time() - t1, 3),
            cells=int(sum(v.N for v in cir.vecs)),
            phase2_cells=0,
            cells_by_kind=WS.cells_by_kind(cir),
            gadget_fails=dict(em.fails),
            violations=list(em.ctx.notes),
            slots_used=ws.plan.used,
            zk_budget={"engine": "fpn2-folded-v1", "queries": int(queries), "gkr_masks": int(gkr_masks),
                       "phase2_cells": 0, "n_groups": finfo.get("n_groups", 0), "n_group": finfo.get("n_group", 0)},
            engine="fpn2-folded-v1",
            folded_info=finfo,
        )
        return pf, info
    if backend != "legacy":
        raise ValueError(f"unknown reply_zk backend {backend!r} (expected 'folded' or 'legacy')")
    pf = Z.prove(cir, stmt, B, label=LABEL)
    info.update(build_s=round(t1 - t0, 3), prove_s=round(time.time() - t1, 3), cells=int(sum(v.N for v in cir.vecs)),
                cells_by_kind=WS.cells_by_kind(cir), gadget_fails=dict(em.fails), violations=list(em.ctx.notes),
                slots_used=ws.plan.used, zk_budget=dict(Z.ZK_REPORT), engine="legacy-zk-flat")
    return pf, info


def verify(prf, stmt, pf, *, expected_c_pre=None, expected_c_ext=None, allow_reduced_queries=False):
    em, _, _ = build(prf, stmt, None, expected_c_pre=expected_c_pre, expected_c_ext=expected_c_ext)
    if isinstance(pf, dict) and pf.get("version") == "fpn2-folded-v1":
        from prover.fpn_prover.backend import folded_engine as FE, multi_gkr as MG, multi_pcs as MP
        min_q = 1 if allow_reduced_queries else FE.MIN_PRODUCTION_QUERIES
        try:
            return FE.verify_folded(
                em.ctx.cir, stmt, pf,
                label=LABEL if isinstance(LABEL, bytes) else str(LABEL).encode("ascii"),
                min_queries=min_q,
            )
        except (FE.FoldedEngineError, MP.MultiPcsError, MG.MultiGkrError) as e:
            raise Z.ProofError(str(e)) from None
    return Z.verify(em.ctx.cir, stmt, pf, label=LABEL)


def try_verify(prf, stmt, pf, *, expected_c_pre=None, expected_c_ext=None, allow_reduced_queries=False):
    try:
        verify(prf, stmt, pf, expected_c_pre=expected_c_pre, expected_c_ext=expected_c_ext,
               allow_reduced_queries=allow_reduced_queries)
        return "accepted"
    except Z.ProofError as e:
        return "rejected: " + str(e)
    except (KeyError, ValueError, IndexError, TypeError, AssertionError) as e:
        return f"rejected: malformed proof ({type(e).__name__}: {e})"
