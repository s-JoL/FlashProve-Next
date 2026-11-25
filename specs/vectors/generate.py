# SPDX-License-Identifier: Apache-2.0
"""Deterministic generator of the FPN2 cross-language golden vectors.

    python3.14 specs/vectors/generate.py            # (re)write fpn2_primitives.json, fpn2_containers.json,
                                                    #           fpn2_folded_gkr.json, fpn2_folded_air.json,
                                                    #           fpn2_folded_vec.json, fpn2_folded_vec_bound.json
    python3.14 specs/vectors/generate.py --check    # fail unless the committed files regenerate byte-identically
    python3.14 specs/vectors/generate.py --only=fpn2_folded_vec_bound.json[,...]   # (re)generate / check just those

Expected values for Fp / Fp3 / Poseidon2 are computed by the prover package ``flashprove_next`` (the existing
implementation) and BLAKE2b expectations by Python's ``hashlib``; the detached reference verifier
``verifier/python/fpn_verify_ref`` must agree with every one of them (asserted here and again in both test suites).
Relation-1 containers are built by a small harness prover living only in this script; relation-3
containers (`fpn2_folded_gkr.json`) by the real backend prover `prover/fpn_prover/backend/zk_lookup.py`
with a deterministic DRBG in place of its secret randomness; relation-4 containers (`fpn2_folded_air.json`)
by `prover/fpn_prover/backend/folded_air.py` on circuits lowered by `prover/fpn_prover/circuit/lower.py`;
relation-5 containers (`fpn2_folded_vec.json`) by `prover/fpn_prover/backend/folded_vec.py` (pure-Python PCS path);
relation-6 containers (`fpn2_folded_vec_bound.json`) by `prover/fpn_prover/backend/folded_vec_bound.py`.
"""
import hashlib
import json
import os
import sys

os.environ["FPN_NO_NATIVE"] = "1"          # never build/load the native .so inside flashprove_next/
sys.dont_write_bytecode = True             # leave no __pycache__ behind in flashprove_next/

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verifier", "python"))

import numpy as np                                                    # noqa: E402
from flashprove_next import field as OF                               # noqa: E402
from flashprove_next import poseidon2 as OP                           # noqa: E402
from flashprove_next import state_zk as OS                            # noqa: E402

from fpn_verify_ref import field as F                                 # noqa: E402
from fpn_verify_ref import poseidon2 as P2                            # noqa: E402
from fpn_verify_ref import profile as PR                              # noqa: E402
from fpn_verify_ref import wire as W                                  # noqa: E402
from fpn_verify_ref.transcript import Fpn2Transcript, LABEL_V2, PERSONAL_FS   # noqa: E402
from fpn_verify_ref.verify import verify_result                       # noqa: E402

P = F.P
assert P == OF.P


def hx(x):
    return "0x%016x" % x


def hx3(a):
    return [hx(x) for x in a]


def hd(d):
    return [hx(x) for x in d]


class DRBG:
    """Deterministic BLAKE2b counter stream (vector generation only)."""

    def __init__(self, seed):
        self.seed = seed
        self.ctr = 0

    def u64(self):
        self.ctr += 1
        return int.from_bytes(hashlib.blake2b(self.seed + self.ctr.to_bytes(8, "little"), digest_size=8).digest(),
                              "little")

    def fp(self):
        while True:
            v = self.u64()
            if v < P:
                return v

    def fp3(self):
        return (self.fp(), self.fp(), self.fp())

    def bytes(self, n):
        out = b""
        while len(out) < n:
            out += self.u64().to_bytes(8, "little")
        return out[:n]


# ============================================================================ primitives
EDGES = [0, 1, 2, P - 1, P - 2, 1 << 32, (1 << 32) - 1, (1 << 32) + 1, 1 << 63, (P - 1) // 2]


def o_fp(op, a, b=None):
    A = np.array([a], dtype=np.uint64)
    B = np.array([b if b is not None else 0], dtype=np.uint64)
    if op == "add":
        return int(OF.CPU.add(A, B)[0])
    if op == "sub":
        return int(OF.CPU.sub(A, B)[0])
    if op == "mul":
        return int(OF.CPU.mul(A, B)[0])
    raise KeyError(op)


def gen_field(rng):
    out = []
    vals = EDGES + [rng.fp() for _ in range(6)]
    for a in vals:
        for b in vals[:8]:
            for op in ("add", "sub", "mul"):
                exp = o_fp(op, a, b)
                mine = {"add": F.fp_add, "sub": F.fp_sub, "mul": F.fp_mul}[op](a, b)
                assert exp == mine, (op, a, b)
                out.append({"op": "fp_" + op, "a": hx(a), "b": hx(b), "out": hx(exp)})
    for a in vals:
        out.append({"op": "fp_neg", "a": hx(a), "out": hx(F.fp_neg(a))})
        if a:
            inv = pow(a, P - 2, P)
            assert F.fp_inv(a) == inv and o_fp("mul", a, inv) == 1
            out.append({"op": "fp_inv", "a": hx(a), "out": hx(inv)})
        for e in (0, 1, 2, 7, P - 2, P - 1, (1 << 64) - 1):
            out.append({"op": "fp_pow", "a": hx(a), "e": hx(e), "out": hx(pow(a, e, P))})
    out.append({"op": "fp_inv", "a": hx(0), "out": "error"})
    for raw in (0, 1, P - 1, P, P + 1, (1 << 64) - 1, 1 << 32):
        out.append({"op": "fp_decode", "bytes": raw.to_bytes(8, "little").hex(),
                    "out": hx(raw) if raw < P else "noncanonical"})

    e_vals = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (P - 1, P - 1, P - 1), (1 << 32, 1, P - 1)]
    e_vals += [rng.fp3() for _ in range(6)]
    for a in e_vals:
        for b in e_vals[:7]:
            m = OF.hmul(a, b)
            assert m == F.e_mul(a, b)
            assert OF.hadd(a, b) == F.e_add(a, b) and OF.hsub(a, b) == F.e_sub(a, b)
            out.append({"op": "fp3_mul", "a": hx3(a), "b": hx3(b), "out": hx3(m)})
            out.append({"op": "fp3_add", "a": hx3(a), "b": hx3(b), "out": hx3(OF.hadd(a, b))})
            out.append({"op": "fp3_sub", "a": hx3(a), "b": hx3(b), "out": hx3(OF.hsub(a, b))})
        out.append({"op": "fp3_neg", "a": hx3(a), "out": hx3(OF.hneg(a))})
        if a != (0, 0, 0):
            inv = OF.hinv(a)
            assert inv == F.e_inv(a) and OF.hmul(a, inv) == (1, 0, 0)
            out.append({"op": "fp3_inv", "a": hx3(a), "out": hx3(inv)})
        for e in (0, 1, 2, 3, 5, P - 1, P ** 3 - 2):
            pw = OF.hpow(a, e)
            assert pw == F.e_pow(a, e)
            out.append({"op": "fp3_pow", "a": hx3(a), "e": "0x%x" % e, "out": hx3(pw)})
    out.append({"op": "fp3_inv", "a": hx3((0, 0, 0)), "out": "error"})
    for d in (1, 2, 3, 4):
        for _ in range(3):
            vals = [rng.fp3() for _ in range(d + 1)]
            r = rng.fp3()
            le = OF.lagrange_eval(vals, r)
            assert le == F.lagrange_eval(vals, r)
            out.append({"op": "fp3_lagrange_eval", "vals": [hx3(v) for v in vals], "r": hx3(r), "out": hx3(le)})
        vals = [rng.fp3() for _ in range(d + 1)]
        out.append({"op": "fp3_lagrange_eval", "vals": [hx3(v) for v in vals], "r": hx3((1, 0, 0)),
                    "out": hx3(vals[1])})
    for n in (0, 1, 3, 5):
        a = [rng.fp3() for _ in range(n)]
        b = [rng.fp3() for _ in range(n)]
        q = OF.eq_pt(a, b)
        assert q == F.eq_pt(a, b)
        out.append({"op": "fp3_eq_pt", "a": [hx3(x) for x in a], "b": [hx3(x) for x in b], "out": hx3(q)})
    for raw in [(1, 2, 3), (P - 1, 0, P - 1), (P, 0, 0), (0, P, 0), (0, 0, (1 << 64) - 1)]:
        enc = b"".join(x.to_bytes(8, "little") for x in raw)
        ok = all(x < P for x in raw)
        out.append({"op": "fp3_decode", "bytes": enc.hex(), "out": hx3(raw) if ok else "noncanonical"})
    return out


def o_permute(st):
    return [int(v) for v in OP.permute(np.array(st, dtype=np.uint64).reshape(12, 1))[:, 0]]


def gen_poseidon2(rng):
    perm = []
    states = [[0] * 12, list(range(12)), [P - 1] * 12, [1 << 32] * 12] + [[rng.fp() for _ in range(12)] for _ in range(4)]
    for s in states:
        exp = o_permute(s)
        assert exp == P2.permute(s)
        perm.append({"in": hd(s), "out": hd(exp)})
    batch = np.array(states, dtype=np.uint64).T
    assert (OP.permute(batch) == P2.permute_batch(batch)).all()

    comp = []
    for _ in range(4):
        l = tuple(rng.fp() for _ in range(4)); r = tuple(rng.fp() for _ in range(4))
        exp = OP.compress1(l, r)
        assert exp == P2.compress(l, r)
        comp.append({"left": hd(l), "right": hd(r), "out": hd(exp)})

    sb = []
    for nblk, dom in ((1, 0), (2, 5), (3, 33)):
        blocks = [[rng.fp() for _ in range(8)] for _ in range(nblk)]
        arr = np.array(blocks, dtype=np.uint64).reshape(nblk, 8, 1)
        exp = tuple(int(v) for v in OP.sponge(arr, domain=dom)[:, 0])
        assert exp == P2.sponge_blocks(blocks, dom)
        sb.append({"blocks": [hd(b) for b in blocks], "domain": dom, "out": hd(exp)})

    sm = []
    for ln, dom in ((0, 33), (1, 64), (7, 34), (8, 35), (9, 64), (17, 38)):
        msg = [rng.fp() for _ in range(ln)]
        exp = tuple(OS.sponge(msg, dom))
        assert exp == P2.sponge_msg(msg, dom)
        sm.append({"msg": hd(msg), "domain": dom, "out": hd(exp)})

    mk = []
    for depth in (0, 1, 3, 4):
        leaves = [P2.leaf_digest(rng.fp()) for _ in range(1 << depth)]
        exp, _ = OP.merkle_root_dense(np.array(leaves, dtype=np.uint64).T)
        root, levels = P2.merkle_root(leaves)
        assert tuple(exp) == root
        paths = []
        for idx in range(1 << depth):
            pth = P2.merkle_path(levels, idx)
            assert OP.verify_path(root, leaves[idx], idx, pth) and P2.verify_path(root, leaves[idx], idx, pth)
            paths.append({"idx": idx, "path": [hd(s) for s in pth]})
        mk.append({"leaves": [hd(x) for x in leaves], "root": hd(root), "paths": paths[:4]})
    return {"permute": perm, "compress": comp, "sponge_blocks": sb, "sponge_msg": sm, "merkle": mk,
            "dom_leaf": P2.DOM_LEAF}


def gen_blake2b():
    out = [{"nn": 64, "person": "", "msg": b"abc".hex(),
            "out": ("ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
                    "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923")}]   # RFC 7693 Appendix A
    assert hashlib.blake2b(b"abc").hexdigest() == out[0]["out"]
    msgs = [bytes((i * 7 + 3) & 0xFF for i in range(n)) for n in (0, 1, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000)]
    for nn in (32, 64):
        for person in (b"", PERSONAL_FS, b"fpn-state-v"):
            for m in msgs:
                d = hashlib.blake2b(m, digest_size=nn, person=person).hexdigest()
                out.append({"nn": nn, "person": person.hex(), "msg": m.hex(), "out": d})
    return out


def gen_transcript(rng):
    scripts = []
    for s in range(2):
        tr = Fpn2Transcript(LABEL_V2 if s == 0 else b"fpn2-test")
        steps = [{"op": "init", "label": (LABEL_V2 if s == 0 else b"fpn2-test").hex(), "state": tr.state.hex()}]
        plan = ["absorb_bytes", "squeeze_fp3", "absorb_fp", "absorb_fp3", "squeeze_bytes", "squeeze_u192",
                "absorb_bytes", "squeeze_bytes", "squeeze_fp3", "absorb_bytes", "squeeze_bytes"]
        sizes = iter([0, 33, 200, 64, 129, 1, 7])
        for op in plan:
            st = {"op": op}
            if op == "absorb_bytes":
                b = rng.bytes(next(sizes)); tr.absorb_bytes(b); st["data"] = b.hex()
            elif op == "absorb_fp":
                x = rng.fp(); tr.absorb_fp(x); st["x"] = hx(x)
            elif op == "absorb_fp3":
                a = rng.fp3(); tr.absorb_fp3(a); st["a"] = hx3(a)
            elif op == "squeeze_bytes":
                n = next(sizes); st["n"] = n; st["out"] = tr.squeeze_bytes(n).hex()
            elif op == "squeeze_fp3":
                st["out"] = hx3(tr.squeeze_fp3())
            elif op == "squeeze_u192":
                st["out"] = "0x%048x" % tr.squeeze_u192()
            st["state"] = tr.state.hex()
            steps.append(st)
        scripts.append(steps)
    return scripts


def gen_profiles():
    out = []
    for name, (prof, pin) in PR.PINNED.items():
        cb = prof.canonical_bytes()
        pid = hashlib.blake2b(cb, digest_size=32).hexdigest()
        assert pid == pin, (name, pid)
        assert PR.decode_profile(cb) == prof
        out.append({"name": name, "canonical_hex": cb.hex(), "profile_id": pid, "c_p": prof.c_p,
                    "security_tier": prof.security_tier, "relation_id": prof.relation_id})
    return out


# ============================================================================ harness prover (generation only)
def prove_harness(prof, table, n_s, st_new):
    n = prof.n_vars
    assert len(table) == 1 << n
    leaves = [P2.leaf_digest(v) for v in table]
    r_tab, levels = P2.merkle_root(leaves)
    claim = (sum(v * v for v in table) % P, 0, 0)
    hdr = W.Header(prof.profile_id).encode()
    stmt = b"".join(x.to_bytes(8, "little") for x in r_tab) + n_s + b"".join(x.to_bytes(8, "little") for x in st_new) \
        + F.e_encode(claim)
    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(hdr)
    tr.absorb_bytes(stmt)
    A = [(v, 0, 0) for v in table]
    rounds = []
    for _ in range(n):
        g = []
        for t in range(prof.sc_degree + 1):
            acc = (0, 0, 0)
            for j in range(len(A) // 2):
                lo, hi = A[2 * j], A[2 * j + 1]
                x = F.e_add(lo, F.e_mul((t, 0, 0), F.e_sub(hi, lo)))
                acc = F.e_add(acc, F.e_mul(x, x))
            g.append(acc)
        rounds.append(g)
        tr.absorb_bytes(b"".join(F.e_encode(v) for v in g))
        r = tr.squeeze_fp3()
        A = [F.e_add(A[2 * j], F.e_mul(r, F.e_sub(A[2 * j + 1], A[2 * j]))) for j in range(len(A) // 2)]
    final = A[0]
    tr.absorb_bytes(F.e_encode(final))
    openings = []
    for _ in range(prof.n_queries):
        idx = int.from_bytes(tr.squeeze_bytes(8), "little") & ((1 << n) - 1)
        openings.append((table[idx], P2.merkle_path(levels, idx)))
    body = W.HarnessBody(r_tab, n_s, st_new, claim, rounds, final, openings)
    return W.encode_harness(prof, body)


def gen_containers(rng):
    H = PR.HARNESS_SQ_N4
    S = PR.LLM_STD_V1
    hpid = H.profile_id.hex()
    spid = S.profile_id.hex()
    vecs = []

    def add(name, data, expect, allow_reduced=True, profile=None, note=""):
        data = bytes(data)
        got = verify_result(data, allow_reduced, bytes.fromhex(profile) if profile else None)
        assert got == expect, (name, got, expect)
        stripped = data.rstrip(b"\x00")
        vecs.append({"name": name, "prefix_hex": stripped.hex(), "total_len": len(data),
                     "allow_reduced": allow_reduced, "expected_profile_id": profile, "expect": expect, "note": note})

    tables = [[rng.fp() for _ in range(16)], [0] * 15 + [P - 1], [rng.u64() % 1000 for _ in range(16)]]
    valid = []
    for k, t in enumerate(tables):
        c = prove_harness(H, t, rng.bytes(32), tuple(rng.fp() for _ in range(4)))
        valid.append(c)
        add(f"valid_harness_{k}", c, "VERIFIED", note="honest harness proof")
    v0 = valid[0]
    add("valid_harness_0_pinned", v0, "VERIFIED", profile=hpid)
    add("downgrade_reduced_without_flag", v0, "E_REDUCED_SECURITY", allow_reduced=False,
        note="reduced-tier profile rejected without --allow-reduced-security")
    add("pinned_profile_mismatch", v0, "E_PROFILE_PINNED", profile=spid)

    def mut(off, val):
        b = bytearray(v0); b[off] = val; return bytes(b)

    def setle(off, n, x):
        b = bytearray(v0); b[off:off + n] = x.to_bytes(n, "little"); return bytes(b)

    add("wrong_magic", mut(0, ord("X")), "E_MAGIC")
    add("wrong_version_1", setle(4, 2, 1), "E_VERSION")
    add("wrong_version_3", setle(4, 2, 3), "E_VERSION")
    add("nonzero_flags", setle(6, 2, 1), "E_FLAGS")
    add("nonzero_flags_high", setle(6, 2, 0x8000), "E_FLAGS")
    add("nonzero_reserved_first", mut(40, 1), "E_RESERVED")
    add("nonzero_reserved_last", mut(63, 0x80), "E_RESERVED")
    b = bytearray(v0); b[8:40] = rng.bytes(32)
    add("wrong_profile_id", bytes(b), "E_PROFILE_ID")
    b = bytearray(v0); b[8] ^= 1
    add("profile_id_bitflip", bytes(b), "E_PROFILE_ID")
    add("truncated_by_1", v0[:-1], "E_LENGTH")
    add("extended_by_1_zero", v0 + b"\x00", "E_LENGTH")
    add("extended_by_1_nonzero", v0 + b"\x01", "E_LENGTH")
    add("header_only", v0[:64], "E_LENGTH")
    add("short_63", v0[:63], "E_LENGTH")
    add("empty", b"", "E_LENGTH")
    add("nonzero_pad_first", mut(64 + H.body_used, 1), "E_PADDING")
    add("nonzero_pad_last", mut(H.c_p - 1, 0x01), "E_PADDING")
    add("noncanonical_fp_eq_p_rtab", setle(64, 8, P), "E_NONCANONICAL", note="R_tab limb 0 = P")
    add("noncanonical_fp_max_stnew", setle(64 + 64, 8, (1 << 64) - 1), "E_NONCANONICAL")
    add("noncanonical_fp3_claim", setle(64 + 96 + 16, 8, P), "E_NONCANONICAL")
    off_open = 64 + 120 + H.n_vars * H.round_len + 24
    add("noncanonical_leaf_value", setle(off_open, 8, P), "E_NONCANONICAL")
    add("flip_bit_stmt_ns", mut(64 + 40, v0[64 + 40] ^ 0x04), "E_SUMCHECK", note="N_s bound via transcript")
    add("flip_bit_sumcheck_g0", mut(64 + 120, v0[64 + 120] ^ 0x01), "E_SUMCHECK")
    add("flip_bit_final", mut(off_open - 24 + 3, v0[off_open - 24 + 3] ^ 0x10), "E_FINAL")
    add("flip_bit_opening_sibling", mut(off_open + 8 + 5, v0[off_open + 8 + 5] ^ 0x02), "E_MERKLE")
    add("flip_bit_opening_leaf", mut(off_open + 1, v0[off_open + 1] ^ 0x01), "E_MERKLE")
    # body swap: valid proof of table 1 with statement of table 0
    b = bytearray(valid[1]); b[64:184] = v0[64:184]
    add("stmt_from_other_proof", bytes(b), "E_SUMCHECK")

    std = W.Header(S.profile_id).encode() + b"\x00" * (S.c_p - 64)
    add("std_profile_backend_unavailable", std, "E_BACKEND_UNAVAILABLE", allow_reduced=False,
        note="standard tier is never VERIFIED until the M1-3 backend is wired")
    add("std_profile_truncated", std[:-1], "E_LENGTH", allow_reduced=False)
    return vecs


# ============================================================================ relation 3 (folded GKR)
class DrbgRng:
    """Deterministic stand-in for the prover's secret randomness (vector generation only)."""

    def __init__(self, drbg):
        self.drbg = drbg

    def fp3(self):
        return self.drbg.fp3()


def _prof_dict(p):
    return dict(name=p.name, tier=p.security_tier, relation=p.relation_id, bits=p.security_bits,
                n_vars=p.n_vars, sc_degree=p.sc_degree, n_queries=p.n_queries, c_p=p.c_p, max_hidden=p.max_hidden)


def folded_offsets(lay):
    """Absolute byte offsets of the relation-3 sections (spec §5.4.7)."""
    o = {"stmt": 64, "root_num": 128, "root_den": 152, "layer0": 176}
    off = 176 + 96
    o["layers"] = {}
    for k in range(1, lay.s + 1):
        o["layers"][k] = {"g_sum": off, "round": lambda i, base=off: base + 24 + 96 * i,
                          "children": off + 24 + 96 * k, "g_r": off + 24 + 96 * k + 96}
        off += lay.layer_len(k)
    o["z_blind"] = off
    off += 24
    o["pcs_round"], o["pcs_root"] = [], []
    for t in range(lay.folds):
        o["pcs_round"] += [off, off + 72]
        off += 144
        if t < lay.folds - 1:
            o["pcs_root"].append(off)
            off += 32
    o["final"] = off
    off += 24
    o["query"] = []
    for _ in range(lay.queries):
        q = []
        for t in range(lay.folds):
            q.append({"values": off, "sibs": off + 96})
            off += 96 + 32 * lay.depth(t)
        o["query"].append(q)
    o["end"] = off
    assert off == 64 + lay.body_used
    return o


def gen_folded():
    from fpn_verify_ref.folded_gkr import layout_of
    from flashprove_next.field import ONE as HONE, ZERO as HZERO, hadd as HADD
    from prover.fpn_prover.backend import zk_lookup as ZL
    from prover.fpn_prover.backend.fpn2 import Layout

    drbg = DRBG(b"FPN2 folded-gkr vectors v1")
    rng = DrbgRng(drbg)
    V1, N8 = PR.FOLDED_GKR_V1, PR.FOLDED_GKR_N8
    vecs, bases = [], {}

    def prove(prof, queries, **kw):
        p = _prof_dict(prof)
        return ZL.prove(queries, n_s=drbg.bytes(32), layout=Layout(prof.n_vars, prof.n_queries), rng=rng,
                        profile=p, **kw)

    def add(name, data, expect, prof, allow_reduced=None, pin=None, note="", base=None, patches=(),
            truncate=None, append=b""):
        allow = prof.security_tier == PR.TIER_REDUCED if allow_reduced is None else allow_reduced
        got = verify_result(data, allow, bytes.fromhex(pin) if pin else None)
        assert got == expect, (name, got, expect)
        v = {"name": name, "allow_reduced": allow, "expected_profile_id": pin, "expect": expect, "note": note}
        if base is None:
            stripped = data.rstrip(b"\x00")
            v.update(prefix_hex=stripped.hex(), total_len=len(data))
            bases[name] = data
        else:
            v.update(base=base, patches=[[o, b.hex()] for o, b in patches], truncate=truncate,
                     append_hex=append.hex())
        vecs.append(v)

    def patched(base, patches, truncate=None, append=b""):
        b = bytearray(bases[base])
        for off, by in patches:
            b[off:off + len(by)] = by
        out = bytes(b)
        if truncate is not None:
            out = out[:truncate]
        return out + append

    def mut(name, base, prof, patches, expect, note="", truncate=None, append=b"", **kw):
        add(name, patched(base, patches, truncate, append), expect, prof, note=note, base=base,
            patches=patches, truncate=truncate, append=append, **kw)

    def flip(base, off, mask=0x01):
        return [(off, bytes([bases[base][off] ^ mask]))]

    le = lambda x, n=8: x.to_bytes(n, "little")  # noqa: E731

    # ---------------------------------------------------------------- standard profile fpn2-folded-gkr-v1
    lay = layout_of(V1)
    o = folded_offsets(lay)
    q1 = [drbg.u64() % 128 for _ in range(128)]
    add("folded_v1_valid_0", prove(V1, q1), "VERIFIED", V1, note="honest 128-query range proof, STANDARD tier")
    v0 = "folded_v1_valid_0"
    other = prove(V1, [127] * 64 + [0] * 64)
    add("folded_v1_valid_0_pinned", bases[v0], "VERIFIED", V1, pin=V1.profile_id.hex(), base=v0)
    add("folded_v1_pinned_mismatch", bases[v0], "E_PROFILE_PINNED", V1, pin=N8.profile_id.hex(), base=v0)
    mut("folded_v1_wrong_magic", v0, V1, [(0, b"X")], "E_MAGIC")
    mut("folded_v1_wrong_version", v0, V1, [(4, le(3, 2))], "E_VERSION")
    mut("folded_v1_truncated_by_1", v0, V1, [], "E_LENGTH", truncate=V1.c_p - 1)
    mut("folded_v1_extended_by_1", v0, V1, [], "E_LENGTH", append=b"\x00")
    mut("folded_v1_nonzero_pad_first", v0, V1, [(o["end"], b"\x01")], "E_PADDING")
    mut("folded_v1_nonzero_pad_last", v0, V1, [(V1.c_p - 1, b"\x80")], "E_PADDING")
    mut("folded_v1_noncanonical_root_num", v0, V1, [(o["root_num"], le(P))], "E_NONCANONICAL")
    mut("folded_v1_noncanonical_gsum", v0, V1, [(o["layers"][4]["g_sum"] + 16, le((1 << 64) - 1))],
        "E_NONCANONICAL")
    mut("folded_v1_noncanonical_query_value", v0, V1, [(o["query"][-1][-1]["values"] + 8, le(P + 1))],
        "E_NONCANONICAL", note="non-canonical limb in the very last field of the body")
    mut("folded_v1_noncanonical_pcs_final", v0, V1, [(o["final"], le(P))], "E_NONCANONICAL")
    mut("folded_v1_root_num_nonzero", v0, V1, [(o["root_num"], le(1))], "E_GKR_ROOT")
    mut("folded_v1_root_den_zero", v0, V1, [(o["root_den"], b"\x00" * 24)], "E_GKR_ROOT",
        note="root_den == 0 would let a zero leaf denominator hide a nonzero sum")
    mut("folded_v1_flip_stmt_root0", v0, V1, flip(v0, o["stmt"] + 7), "E_GKR",
        note="commitment root is bound before beta")
    mut("folded_v1_flip_stmt_ns", v0, V1, flip(v0, o["stmt"] + 40, 0x10), "E_GKR")
    mut("folded_v1_stmt_from_other_proof", v0, V1, [(64, other[64:128])], "E_GKR")
    mut("folded_v1_flip_gkr_mask_gsum", v0, V1, flip(v0, o["layers"][5]["g_sum"] + 3), "E_GKR",
        note="Libra mask sum is bound before lambda")
    mut("folded_v1_flip_gkr_round", v0, V1, flip(v0, o["layers"][7]["round"](3) + 48 + 2), "E_GKR")
    mut("folded_v1_flip_gkr_children", v0, V1, flip(v0, o["layers"][lay.s]["children"] + 30), "E_GKR")
    mut("folded_v1_flip_gkr_mask_eval", v0, V1, flip(v0, o["layers"][lay.s]["g_r"] + 1), "E_GKR")
    mut("folded_v1_flip_z_blind", v0, V1, flip(v0, o["z_blind"] + 9), "E_SUMCHECK")
    mut("folded_v1_flip_pcs_round_0", v0, V1, flip(v0, o["pcs_round"][0] + 5), "E_SUMCHECK")
    mut("folded_v1_flip_pcs_round_7", v0, V1, flip(v0, o["pcs_round"][7] + 50), "E_SUMCHECK")
    mut("folded_v1_flip_pcs_fold_root_2", v0, V1, flip(v0, o["pcs_root"][2] + 31, 0x80), "E_SUMCHECK")
    mut("folded_v1_flip_pcs_final", v0, V1, flip(v0, o["final"] + 4), "E_FINAL")
    mut("folded_v1_tamper_coset_value", v0, V1, flip(v0, o["query"][10][1]["values"] + 25), "E_MERKLE")
    mut("folded_v1_tamper_coset_merkle_path", v0, V1, flip(v0, o["query"][40][3]["sibs"] + 2 * 32 + 6),
        "E_MERKLE")
    mut("folded_v1_tamper_root0_layer_path", v0, V1, flip(v0, o["query"][0][0]["sibs"]), "E_MERKLE")

    # ---------------------------------------------------------------- reduced test profile fpn2-folded-gkr-n8
    lay8 = layout_of(N8)
    o8 = folded_offsets(lay8)
    qn = [drbg.u64() % lay8.quarter for _ in range(lay8.quarter)]
    add("folded_n8_valid_0", prove(N8, qn), "VERIFIED", N8)
    add("folded_n8_valid_1", prove(N8, [lay8.quarter - 1] * lay8.quarter), "VERIFIED", N8,
        note="all queries hit the last table entry")
    n0 = "folded_n8_valid_0"
    add("folded_n8_without_flag", bases[n0], "E_REDUCED_SECURITY", N8, allow_reduced=False, base=n0)
    s8 = lay8.s
    last = lambda k, ch: (HADD(ch[0], HONE),) + ch[1:] if k == s8 else ch  # noqa: E731
    add("folded_n8_adaptive_mask_tamper",
        prove(N8, qn, hooks={"gkr": {"children": last, "adaptive_g_r": True}}), "E_SUMCHECK", N8,
        note="forged leaf claim with adaptively solved Libra g_r passes every GKR check; the PCS rejects")
    add("folded_n8_forged_children", prove(N8, qn, hooks={"gkr": {"children": last}}), "E_GKR", N8)
    add("folded_n8_fold_layer_shift",
        prove(N8, qn, hooks={"pcs": {"code": lambda t, c: [HADD(x, HONE) for x in c] if t == 0 else c}}),
        "E_FOLD", N8, note="well-formed but inconsistent folded layer 1 under a valid Merkle root")
    bad = list(qn)
    bad[3] = lay8.quarter
    add("folded_n8_missing_table_entry", prove(N8, bad), "E_GKR_ROOT", N8, note="query outside the table")
    add("folded_n8_missing_table_entry_forged_root",
        prove(N8, bad, hooks={"root": lambda r: (HZERO, r[1])}), "E_GKR", N8)

    def wrong_m(layout, queries, rng_):
        cells = ZL.build_witness(layout, queries, rng_)
        i = layout.cell(layout.off_m + 1)
        cells[i] = HADD(cells[i], HONE)
        return cells

    def zero_den(layout, queries, rng_):
        cells = ZL.build_witness(layout, queries, rng_)
        cells[layout.cell(layout.off_e3 + layout.quarter // 2 + 2)] = HZERO
        return cells

    add("folded_n8_wrong_multiplicity", prove(N8, qn, witness=wrong_m), "E_GKR_ROOT", N8)
    add("folded_n8_wrong_multiplicity_forged_root",
        prove(N8, qn, witness=wrong_m, hooks={"root": lambda r: (HZERO, r[1])}), "E_GKR", N8)
    add("folded_n8_zero_denominator_leaf", prove(N8, qn, witness=zero_den, hooks={"allow_zero_root": True}),
        "E_GKR_ROOT", N8, note="zero leaf denominator forces root_den == 0")
    mut("folded_n8_root_den_zero", n0, N8, [(o8["root_den"], b"\x00" * 24)], "E_GKR_ROOT")
    mut("folded_n8_flip_gkr_mask_gsum", n0, N8, flip(n0, o8["layers"][2]["g_sum"]), "E_GKR")
    mut("folded_n8_flip_pcs_round", n0, N8, flip(n0, o8["pcs_round"][3] + 30), "E_SUMCHECK")
    mut("folded_n8_flip_pcs_final", n0, N8, flip(n0, o8["final"]), "E_FINAL")
    mut("folded_n8_tamper_coset_merkle_path", n0, N8, flip(n0, o8["query"][1][2]["sibs"] + 33), "E_MERKLE")
    mut("folded_n8_noncanonical_limb", n0, N8, [(o8["layers"][1]["round"](0) + 8, le(P))], "E_NONCANONICAL")
    mut("folded_n8_nonzero_pad", n0, N8, [(o8["end"] + 5, b"\x02")], "E_PADDING")
    return {"spec": "fpn-wire-2.0", "relation": "FOLDED_GKR_LOOKUP",
            "profiles": {V1.name: V1.profile_id.hex(), N8.name: N8.profile_id.hex()},
            "offsets": {N8.name: {"z_blind": o8["z_blind"], "final": o8["final"], "end": o8["end"]},
                        V1.name: {"z_blind": o["z_blind"], "final": o["final"], "end": o["end"]}},
            "vectors": vecs}


# ============================================================================ relation 4 (folded GKR AIR)
def air_offsets(lay):
    """Absolute byte offsets of the relation-4 sections (spec §5.5.7)."""
    o = {"stmt": 64, "root_circ": 96, "n_s": 128, "root_num": 160, "root_den": 184}
    off = 160 + 48 + 96
    o["layers"] = {}
    for k in range(1, lay.S + 1):
        o["layers"][k] = {"g_sum": off, "round": lambda i, base=off: base + 24 + 96 * i,
                          "children": off + 24 + 96 * k, "g_r": off + 24 + 96 * k + 96}
        off += lay.layer_len(k)
    o["zc_sum"] = off
    o["zc_round"] = [off + 24 + 120 * i for i in range(lay.g)]
    o["zc_finals"] = off + 24 + 120 * lay.g
    off += lay.zc_len
    o["z_blind"] = off
    off += 24
    o["pcs_round"], o["pcs_root"] = [], []
    for i in range(lay.n + 2):
        o["pcs_round"].append(off)
        off += 72
        if i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1:
            o["pcs_root"].append(off)
            off += 32
    o["final"] = off
    off += 24
    o["query"] = []
    for _ in range(lay.queries):
        q = []
        for t in range(lay.folds):
            q.append({"values": off, "sibs": off + 96})
            off += 96 + 32 * lay.depth(t)
            if t == 0:
                q[0]["c_values"], q[0]["c_sibs"] = off, off + 96
                off += 96 + 32 * lay.depth(0)
        o["query"].append(q)
    o["end"] = off
    assert off == 64 + lay.body_used
    return o


def _bf16(x):
    import struct
    word = struct.unpack(">I", struct.pack(">f", x))[0]
    return (word + 0x7fff + ((word >> 16) & 1)) >> 16


def air_demo_program(opcode=0):
    """rmsnorm_block (width 2) + gemm_tile_row_block (mxfp4, 1x1) + 5-opcode microprogram slot."""
    from fractions import Fraction
    from prover.fpn_prover.circuit.blocks import gemm_tile_row_block, rmsnorm_block
    from prover.fpn_prover.circuit.microprogram import install_microprogram
    from prover.fpn_prover.ir.arith import bf16_decode, norm_rsqrt
    from prover.fpn_prover.ir.format import get_contract
    from prover.fpn_prover.ir.gadget_ir import Program
    branches = ((0, 256, 256, 0, 512, 0, 512), (0, 256, 256, 0, 512, 0, 512),
                (0, 0, 1, 1, 1, 0, 256, 256, 0, 0, 0, 1, 0, 512, 0), (256, 0, 256, 512, 0, 0, 0, 512),
                (0, 1, 0, 256, 0, 256, 0, 256, 256, 256, 0))
    p = Program()
    norm = rmsnorm_block(p, "norm", 2)
    contract = get_contract("mxfp4_e2m1")
    gemm = gemm_tile_row_block(p, "gemm", 1, 1, contract, operand_bits=38)   # radius product is a gate
    slot = install_microprogram(p, "slot", tokens=2, experts=2, top_k=2, window=1, sink=1, conv_width=1, extent=1)
    one = _bf16(1)
    x = bf16_decode(one).value
    ref = norm_rsqrt((x * x + x * x) / 2 + Fraction(1, 10**6))
    w = dict(zip(norm, (one, one, one, one, _bf16(1.0), _bf16(1.0), ref.k, ref.low, ref.high, ref.high - ref.low),
                 strict=True))
    z = 8 * contract.dot_relative_bound(1).numerator
    w.update(zip(gemm, (8, 0, z, 3, 3, 24, 3 * z, 24, 0), strict=True))
    lanes = branches[opcode] + (0,) * (len(slot.lanes) - len(branches[opcode]))
    w.update(zip(slot.selectors + slot.lanes, tuple(int(i == opcode) for i in range(5)) + lanes, strict=True))
    assert not p.failures(w)
    return p, w


def air_small_program(tamper=False):
    """Tiny relation-4 circuit for the n = 12 test profile: z = x * y, s boolean, x + y = 12."""
    from prover.fpn_prover.ir.gadget_ir import Program
    p = Program()
    x = p.wire("x", 9, "operand", "demo")
    y = p.wire("y", 9, "operand", "demo")
    z = p.wire("z", 16, "product", "demo")
    s = p.wire("s", 2, "boolean", "demo")
    p.equal("demo.mul", z - x * y)
    p.equal("demo.bool", s * (s - 1))
    p.equal("demo.sum", x + y - (13 if tamper else 12))
    return p, {"x": 5, "y": 7, "z": 35, "s": 1}


def gen_folded_air():
    from fpn_verify_ref.folded_air import layout_of as air_layout_of
    from flashprove_next.field import ONE as HONE, ZERO as HZERO, hadd as HADD
    from prover.fpn_prover.backend import folded_air as FA
    from prover.fpn_prover.circuit import lower as LW

    drbg = DRBG(b"FPN2 folded-air vectors v1")
    rng = DrbgRng(drbg)
    V1, N12 = PR.FOLDED_AIR_V1, PR.FOLDED_AIR_N12
    vecs, bases = [], {}

    def add(name, data, expect, prof, root, allow_reduced=None, pin=None, note="", base=None, patches=(),
            truncate=None, append=b""):
        allow = prof.security_tier == PR.TIER_REDUCED if allow_reduced is None else allow_reduced
        got = verify_result(data, allow, bytes.fromhex(pin) if pin else None, root)
        assert got == expect, (name, got, expect)
        v = {"name": name, "allow_reduced": allow, "expected_profile_id": pin,
             "expected_circuit_root": None if root is None else root.hex(), "expect": expect, "note": note}
        if base is None:
            v.update(prefix_hex=data.rstrip(b"\x00").hex(), total_len=len(data))
            bases[name] = data
        else:
            v.update(base=base, patches=[[o, b.hex()] for o, b in patches], truncate=truncate,
                     append_hex=append.hex())
        vecs.append(v)

    def patched(base, patches, truncate=None, append=b""):
        b = bytearray(bases[base])
        for off, by in patches:
            b[off:off + len(by)] = by
        out = bytes(b)
        return (out if truncate is None else out[:truncate]) + append

    def mut(name, base, prof, root, patches, expect, note="", truncate=None, append=b"", **kw):
        add(name, patched(base, patches, truncate, append), expect, prof, root, note=note, base=base,
            patches=patches, truncate=truncate, append=append, **kw)

    def flip(base, off, mask=0x01):
        return [(off, bytes([bases[base][off] ^ mask]))]

    le = lambda x, n=8: x.to_bytes(n, "little")  # noqa: E731
    p12 = _prof_dict(N12)
    p1 = _prof_dict(V1)

    # ---------------------------------------------------------------- reduced test profile fpn2-folded-air-n12
    lay = air_layout_of(N12)
    o = air_offsets(lay)
    prog, wit = air_small_program()
    circ = LW.compile_air(prog, p12)
    root = circ.root
    honest = LW.assign_air(circ, prog, wit).values

    def prove(values, **kw):
        return LW.prove_air_unchecked(circ, values, server_nonce=drbg.bytes(32), rng=rng, **kw)

    add("air_n12_valid_0", prove(honest), "VERIFIED", N12, root, note="honest z = x*y, s(s-1) = 0, x + y = 12")
    n0 = "air_n12_valid_0"
    add("air_n12_without_flag", bases[n0], "E_REDUCED_SECURITY", N12, root, allow_reduced=False, base=n0,
        note="downgrade: REDUCED profile rejected without --allow-reduced-security")
    add("air_n12_unpinned_circuit", bases[n0], "E_CIRCUIT_UNPINNED", N12, None, base=n0,
        note="relation 4 fails closed without a pinned circuit root")
    add("air_n12_wrong_circuit_pin", bases[n0], "E_CIRCUIT_PINNED", N12, bytes(32), base=n0)
    add("air_n12_pinned_profile", bases[n0], "VERIFIED", N12, root, pin=N12.profile_id.hex(), base=n0)
    add("air_n12_pinned_profile_mismatch", bases[n0], "E_PROFILE_PINNED", N12, root, pin=V1.profile_id.hex(),
        base=n0)
    # adversarial provers (each breaks exactly one thing; all witness values in range unless stated)
    bad = LW.air_values(circ, {**wit, "z": 36})
    add("air_n12_bad_gate", prove(bad), "E_ZEROCHECK", N12, root, note="z != x*y (in range): zerocheck")
    bad = LW.air_values(circ, {**wit, "s": -1, })
    add("air_n12_bad_selector_gate", prove(bad), "E_ZEROCHECK", N12, root, note="s = -1: s(s-1) = 2")
    limb = LW.limb_name("z", 0)
    bad = LW.air_values(circ, wit, overrides={limb: honest[limb] ^ 1})
    add("air_n12_bad_recompose", prove(bad), "E_ZEROCHECK", N12, root, note="limb does not recombine")
    l0, l1 = LW.limb_name("z", 0), LW.limb_name("z", 1)
    moved = LW.air_values(circ, wit, overrides={l0: honest[l0] + 128, l1: honest[l1] - 1})
    v_cols = LW.air_vector(circ, moved)
    miswired = lambda lay_, tab, V, r: FA.build_witness(lay_, tab, V, r, columns=FA.columns_of(tab, v_cols))  # noqa: E731
    add("air_n12_copy_violation", prove(honest, hooks={"witness": miswired}), "E_GKR_ROOT", N12, root,
        note="gate cells disagree with the range-checked variables; every row holds on the cells")

    def clipped(lay_, tab, V, r):
        m = FA.range_multiplicities(tab, [v if not f or v < FA.TABLE else 0 for v, f in zip(V, tab.flag)])
        return FA.build_witness(lay_, tab, V, r, m=m + [0] * (lay_.K - FA.TABLE))

    add("air_n12_out_of_table_limb", prove(moved, hooks={"witness": clipped}), "E_GKR_ROOT", N12, root,
        note="limb 0 + 128 / limb 1 - 1: recomposition holds, limb outside [0, 128)")

    def wrong_m(lay_, tab, V, r):
        m = FA.range_multiplicities(tab, V) + [0] * (lay_.K - FA.TABLE)
        t = next(i for i, x in enumerate(m) if x)
        m[t] -= 1
        m[t + 1] += 1
        return FA.build_witness(lay_, tab, V, r, m=m)

    add("air_n12_wrong_multiplicity", prove(honest, hooks={"witness": wrong_m}), "E_GKR_ROOT", N12, root)
    add("air_n12_wrong_multiplicity_forged_root",
        prove(honest, hooks={"witness": wrong_m, "root": lambda r: (HZERO, r[1])}), "E_GKR", N12, root)
    tprog, twit = air_small_program(tamper=True)
    tcirc = LW.compile_air(tprog, p12)
    tvals = LW.assign_air(tcirc, tprog, {**twit, "y": 8, "z": 40}).values
    tdata = LW.prove_air_unchecked(tcirc, tvals, server_nonce=drbg.bytes(32), rng=rng)
    add("air_n12_tampered_circuit", tdata, "E_CIRCUIT_PINNED", N12, root,
        note="valid proof of a circuit with one altered gate constant, verifier pinned to the honest root")
    add("air_n12_tampered_circuit_own_root", tdata, "VERIFIED", N12, tcirc.root, base="air_n12_tampered_circuit",
        note="the same container verifies only against the tampered circuit's own root")
    add("air_n12_tampered_selectors_under_honest_root",
        FA.prove(tcirc.table, LW.air_vector(tcirc, tvals), n_s=drbg.bytes(32), profile=p12, rng=rng,
                 hooks={"stmt_root_circ": root}), "E_MERKLE", N12, root,
        note="tampered selector table smuggled under the honest root_circ: circuit-leaf openings fail")
    hc = FA.commit_circuit(FA.layout_of(p12), circ.table)
    add("air_n12_tampered_selectors_honest_openings",
        FA.prove(tcirc.table, LW.air_vector(tcirc, tvals), n_s=drbg.bytes(32), profile=p12, rng=rng,
                 hooks={"stmt_root_circ": root, "pcs": {"c_code": hc.code, "c_levels": hc.levels}}),
        "E_FOLD", N12, root,
        note="tampered table used in GKR / zerocheck / folding, honest circuit leaves opened: fold check fails")
    add("air_n12_adaptive_zerocheck_claim", prove(honest, hooks={"zc_adaptive": True}), "E_SUMCHECK", N12, root,
        note="w_a(r_gate) shifted, Libra g_r re-solved: the batched opening of (F, C) rejects")
    add("air_n12_fold_layer_shift",
        prove(honest, hooks={"pcs": {"code": lambda t, c: [HADD(x, HONE) for x in c] if t == 0 else c}}),
        "E_FOLD", N12, root, note="well-formed but inconsistent folded layer 1 under a valid Merkle root")
    # single mutations of the honest container
    mut("air_n12_nonzero_pad", n0, N12, root, [(o["end"] + 3, b"\x04")], "E_PADDING")
    mut("air_n12_truncated_by_1", n0, N12, root, [], "E_LENGTH", truncate=N12.c_p - 1)
    mut("air_n12_noncanonical_zc_final", n0, N12, root, [(o["zc_finals"] + 8, le(P))], "E_NONCANONICAL")
    mut("air_n12_flip_stmt_root_circ", n0, N12, root, flip(n0, o["root_circ"] + 5), "E_CIRCUIT_PINNED")
    mut("air_n12_flip_stmt_root0", n0, N12, root, flip(n0, o["stmt"] + 5), "E_GKR",
        note="witness root is bound before (beta, gamma)")
    mut("air_n12_root_num_nonzero", n0, N12, root, [(o["root_num"], le(1))], "E_GKR_ROOT")
    mut("air_n12_root_den_zero", n0, N12, root, [(o["root_den"], b"\x00" * 24)], "E_GKR_ROOT")
    mut("air_n12_flip_gkr_round", n0, N12, root, flip(n0, o["layers"][6]["round"](2) + 30), "E_GKR")
    mut("air_n12_flip_gkr_mask_eval", n0, N12, root, flip(n0, o["layers"][lay.S]["g_r"] + 2), "E_GKR")
    mut("air_n12_flip_zc_sum", n0, N12, root, flip(n0, o["zc_sum"] + 1), "E_ZEROCHECK")
    mut("air_n12_flip_zc_round", n0, N12, root, flip(n0, o["zc_round"][3] + 50), "E_ZEROCHECK")
    mut("air_n12_flip_zc_selector_eval", n0, N12, root, flip(n0, o["zc_finals"] + 3 * 24 + 4), "E_ZEROCHECK")
    mut("air_n12_flip_z_blind", n0, N12, root, flip(n0, o["z_blind"] + 9), "E_SUMCHECK")
    mut("air_n12_flip_batch_round", n0, N12, root, flip(n0, o["pcs_round"][1] + 5), "E_SUMCHECK")
    mut("air_n12_flip_pcs_round", n0, N12, root, flip(n0, o["pcs_round"][6] + 40), "E_SUMCHECK")
    mut("air_n12_flip_pcs_fold_root", n0, N12, root, flip(n0, o["pcs_root"][1] + 31, 0x80), "E_SUMCHECK")
    mut("air_n12_flip_pcs_final", n0, N12, root, flip(n0, o["final"] + 4), "E_FINAL")
    mut("air_n12_tamper_witness_leaf", n0, N12, root, flip(n0, o["query"][0][0]["values"] + 7), "E_MERKLE")
    mut("air_n12_tamper_circuit_leaf", n0, N12, root, flip(n0, o["query"][1][0]["c_values"] + 7), "E_MERKLE")
    mut("air_n12_tamper_circuit_path", n0, N12, root, flip(n0, o["query"][0][0]["c_sibs"] + 33), "E_MERKLE")
    mut("air_n12_tamper_fold_path", n0, N12, root, flip(n0, o["query"][1][3]["sibs"] + 2), "E_MERKLE")

    # ---------------------------------------------------------------- standard profile fpn2-folded-air-v1
    lay1 = air_layout_of(V1)
    o1 = air_offsets(lay1)
    dprog, dwit = air_demo_program()
    dcirc = LW.compile_air(dprog, p1)
    droot = dcirc.root
    dvals = LW.assign_air(dcirc, dprog, dwit).values
    add("air_v1_valid_0", LW.prove_air_unchecked(dcirc, dvals, server_nonce=drbg.bytes(32), rng=rng), "VERIFIED",
        V1, droot, note="rmsnorm + gemm tile + 5-opcode microprogram slot lowered by circuit/lower.py")
    a0 = "air_v1_valid_0"
    add("air_v1_valid_0_pinned", bases[a0], "VERIFIED", V1, droot, pin=V1.profile_id.hex(), base=a0)
    add("air_v1_unpinned_circuit", bases[a0], "E_CIRCUIT_UNPINNED", V1, None, base=a0)
    add("air_v1_other_circuit_root", bases[a0], "E_CIRCUIT_PINNED", V1, root, base=a0,
        note="a relation-4 container never verifies against another circuit's root")
    mut("air_v1_nonzero_pad", a0, V1, droot, [(V1.c_p - 1, b"\x80")], "E_PADDING")
    mut("air_v1_noncanonical_query_value", a0, V1, droot, [(o1["query"][-1][-1]["values"] + 8, le(P + 1))],
        "E_NONCANONICAL")
    mut("air_v1_flip_zc_round", a0, V1, droot, flip(a0, o1["zc_round"][0] + 3), "E_ZEROCHECK")
    mut("air_v1_flip_gkr_children", a0, V1, droot, flip(a0, o1["layers"][lay1.S]["children"] + 30), "E_GKR")
    mut("air_v1_flip_pcs_final", a0, V1, droot, flip(a0, o1["final"] + 1), "E_FINAL")
    mut("air_v1_tamper_circuit_leaf", a0, V1, droot, flip(a0, o1["query"][40][0]["c_values"] + 50), "E_MERKLE")
    return {"spec": "fpn-wire-2.0", "relation": "FOLDED_GKR_AIR",
            "profiles": {V1.name: V1.profile_id.hex(), N12.name: N12.profile_id.hex()},
            "circuits": {"air_small_program": root.hex(), "air_small_program_tampered": tcirc.root.hex(),
                         "air_demo_program": droot.hex()},
            "offsets": {N12.name: {"zc_sum": o["zc_sum"], "z_blind": o["z_blind"], "final": o["final"],
                                   "end": o["end"]},
                        V1.name: {"zc_sum": o1["zc_sum"], "z_blind": o1["z_blind"], "final": o1["final"],
                                  "end": o1["end"]}},
            "vectors": vecs}


# ============================================================================ relation 5 (§5.6)
def vec_offsets(lay):
    """Absolute byte offsets of the relation-5 sections (spec §5.6.6)."""
    o = {"stmt": 64, "weight_root": 96, "n_s": 128, "g_sum": 160}
    o["ip_round"] = [184 + 96 * i for i in range(lay.d)]
    off = 184 + 96 * lay.d
    o["e_w"] = off
    o["z_blind"] = off + 24
    off += 48
    o["pcs_round"], o["pcs_root"] = [], []
    for i in range(lay.n + 2):
        o["pcs_round"].append(off)
        off += 72
        if i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1:
            o["pcs_root"].append(off)
            off += 32
    o["final"] = off
    off += 24
    o["query"] = []
    for _ in range(lay.queries):
        q = []
        for t in range(lay.folds):
            q.append({"values": off, "sibs": off + 96})
            off += 96 + 32 * lay.depth(t)
            if t == 0:
                q[0]["w_values"], q[0]["w_sibs"] = off, off + 96
                off += 96 + 32 * lay.depth(0)
        q.append({"end": off})
        o["query"].append(q)
    o["end"] = off
    assert off == 64 + lay.body_used
    return o


def gen_folded_vec():
    from fpn_verify_ref.folded_vec import layout_of as vec_layout_of
    from flashprove_next.field import ONE as HONE, hadd as HADD
    from prover.fpn_prover.backend import folded_vec as FV

    drbg = DRBG(b"FPN2 folded-vec vectors v1")
    rng = DrbgRng(drbg)
    V16, V22 = PR.FOLDED_VEC_N16_V1, PR.FOLDED_VEC_N22_V1
    p16 = _prof_dict(V16)
    lay = vec_layout_of(V16)
    flay = FV.layout_of(p16)
    o = vec_offsets(lay)
    vecs, bases = [], {}

    def add(name, data, expect, root, pin=None, note="", base=None, patches=(), truncate=None, append=b"",
            zero_after=None):
        if zero_after is not None:          # bytes the verifier never reads before rejecting are stored as zero
            data = data[:zero_after] + b"\x00" * (len(data) - zero_after)
            note = (note + "; " if note else "") + f"bytes from offset {zero_after} zeroed (rejected earlier)"
        got = verify_result(data, False, bytes.fromhex(pin) if pin else None, root)
        assert got == expect, (name, got, expect)
        v = {"name": name, "allow_reduced": False, "expected_profile_id": pin,
             "expected_circuit_root": None if root is None else root.hex(), "expect": expect, "note": note}
        if base is None:
            v.update(prefix_hex=data.rstrip(b"\x00").hex(), total_len=len(data))
            bases[name] = data
        else:
            v.update(base=base, patches=[[o_, b.hex()] for o_, b in patches], truncate=truncate,
                     append_hex=append.hex())
        vecs.append(v)

    def patched(base, patches, truncate=None, append=b""):
        b = bytearray(bases[base])
        for off, by in patches:
            b[off:off + len(by)] = by
        out = bytes(b)
        return (out if truncate is None else out[:truncate]) + append

    def mut(name, base, root, patches, expect, note="", truncate=None, append=b""):
        add(name, patched(base, patches, truncate, append), expect, root, note=note, base=base,
            patches=patches, truncate=truncate, append=append)

    def flip(base, off, mask=0x01):
        return [(off, bytes([bases[base][off] ^ mask]))]

    le = lambda x, n=8: x.to_bytes(n, "little")  # noqa: E731
    sint = lambda: (drbg.u64() % (1 << 16)) - (1 << 15)  # noqa: E731
    B, K = flay.B, flay.K
    Wi = np.array([[[sint() for _ in range(K)] for _ in range(FV.ROWS)] for _ in range(B)], dtype=np.int64)
    Xi = np.array([[sint() for _ in range(K)] for _ in range(B)], dtype=np.int64)
    Yi, bound_bits = FV.exact_products(Wi, Xi)
    assert bound_bits < 63
    W, Xv, Yv = FV.to_field(Wi), FV.to_field(Xi), FV.to_field(Yi)
    root = FV.weight_root(flay, W)

    def prove(Wf, Xf, Yf, **kw):
        return FV.prove(Wf, Xf, Yf, n_s=drbg.bytes(32), profile=p16, rng=rng, **kw)

    end_ip, end_q0 = o["e_w"] + 24, o["query"][0][-1]["end"]
    add("vec_n16_valid_0", prove(W, Xv, Yv), "VERIFIED", root,
        note="one full-width 32 x 1024 tile of 16-bit signed integers, Y = W X exactly")
    v0 = "vec_n16_valid_0"
    add("vec_n16_valid_0_pinned", bases[v0], "VERIFIED", root, pin=V16.profile_id.hex(), base=v0)
    add("vec_n16_pinned_profile_mismatch", bases[v0], "E_PROFILE_PINNED", root, pin=V22.profile_id.hex(), base=v0)
    add("vec_n16_unpinned_weights", bases[v0], "E_CIRCUIT_UNPINNED", None, base=v0,
        note="relation 5 fails closed without a pinned weight_root")
    add("vec_n16_wrong_weight_root", bases[v0], "E_CIRCUIT_PINNED", bytes(32), base=v0)
    # adversarial provers
    Xt = Xv.copy()
    Xt[0, 7] = (int(Xt[0, 7]) + 1) % P
    add("vec_n16_tampered_x", prove(W, Xt, Yv), "E_ZEROCHECK", root, zero_after=end_ip,
        note="one activation X[0][7] + 1 with the honest outputs: the inner-product zero-claim fails")
    Yt = Yv.copy()
    Yt[0, 3] = (int(Yt[0, 3]) + 1) % P
    add("vec_n16_tampered_y", prove(W, Xv, Yt), "E_ZEROCHECK", root, zero_after=end_ip,
        note="one output Y[0][3] + 1")
    Wt = Wi.copy()
    Wt[0, 5, 9] += 1
    Wtf = FV.to_field(Wt)
    Ytf = FV.to_field(FV.exact_products(Wt, Xi)[0])
    troot = FV.weight_root(flay, Wtf)
    add("vec_n16_tampered_weights", prove(Wtf, Xv, Ytf), "E_CIRCUIT_PINNED", root, zero_after=128,
        note="valid proof for W with W[0][5][9] + 1 (Y recomputed), verifier pinned to the honest weight_root")
    add("vec_n16_tampered_weights_under_honest_root", prove(Wtf, Xv, Ytf, hooks={"stmt_weight_root": root}),
        "E_MERKLE", root, zero_after=end_q0,
        note="tampered weights smuggled under the honest weight_root: weight-leaf openings fail")
    add("vec_n16_adaptive_e_w", prove(W, Xv, Yv, hooks={"e_w": lambda v: HADD(v, HONE)}), "E_SUMCHECK", root,
        zero_after=o["pcs_round"][1], note="public evaluation e_W shifted by one: the batched opening rejects")
    add("vec_n16_adaptive_ip_rounds", prove(W, Xv, Yt, hooks={"ip_adaptive": True}), "E_SUMCHECK", root,
        zero_after=o["pcs_round"][1],
        note="false Y with every inner-product round shifted so each round sum passes: the batched claim rejects")
    add("vec_n16_fold_layer_shift",
        prove(W, Xv, Yv, hooks={"pcs": {"code": lambda t, c: [HADD(x, HONE) for x in c] if t == 0 else c}}),
        "E_FOLD", root, zero_after=end_q0,
        note="well-formed but inconsistent folded layer 1 under a valid Merkle root")
    # single mutations of the honest container
    mut("vec_n16_nonzero_pad", v0, root, [(o["end"] + 3, b"\x04")], "E_PADDING")
    mut("vec_n16_truncated_by_1", v0, root, [], "E_LENGTH", truncate=V16.c_p - 1)
    mut("vec_n16_noncanonical_ip_round", v0, root, [(o["ip_round"][2] + 8, le(P))], "E_NONCANONICAL")
    mut("vec_n16_noncanonical_query_value", v0, root, [(o["query"][-1][-2]["values"] + 8, le(P + 1))],
        "E_NONCANONICAL")
    mut("vec_n16_flip_stmt_weight_root", v0, root, flip(v0, o["weight_root"] + 5), "E_CIRCUIT_PINNED")
    mut("vec_n16_flip_stmt_root0", v0, root, flip(v0, o["stmt"] + 5), "E_ZEROCHECK",
        note="witness root is bound before the row point and lambda")
    mut("vec_n16_flip_n_s", v0, root, flip(v0, o["n_s"] + 1), "E_ZEROCHECK")
    mut("vec_n16_flip_g_sum", v0, root, flip(v0, o["g_sum"] + 1), "E_ZEROCHECK")
    mut("vec_n16_flip_ip_round", v0, root, flip(v0, o["ip_round"][4] + 50), "E_ZEROCHECK")
    mut("vec_n16_flip_ip_last_round", v0, root, flip(v0, o["ip_round"][-1] + 75), "E_SUMCHECK",
        note="last inner-product round: its value at s enters the batched claim")
    mut("vec_n16_flip_e_w", v0, root, flip(v0, o["e_w"] + 2), "E_SUMCHECK")
    mut("vec_n16_flip_z_blind", v0, root, flip(v0, o["z_blind"] + 9), "E_SUMCHECK")
    mut("vec_n16_flip_batch_round", v0, root, flip(v0, o["pcs_round"][1] + 5), "E_SUMCHECK")
    mut("vec_n16_flip_pcs_round", v0, root, flip(v0, o["pcs_round"][6] + 40), "E_SUMCHECK")
    mut("vec_n16_flip_pcs_fold_root", v0, root, flip(v0, o["pcs_root"][1] + 31, 0x80), "E_SUMCHECK")
    mut("vec_n16_flip_pcs_final", v0, root, flip(v0, o["final"] + 4), "E_FINAL")
    mut("vec_n16_tamper_witness_leaf", v0, root, flip(v0, o["query"][0][0]["values"] + 7), "E_MERKLE")
    mut("vec_n16_tamper_weight_leaf", v0, root, flip(v0, o["query"][1][0]["w_values"] + 7), "E_MERKLE")
    mut("vec_n16_tamper_weight_path", v0, root, flip(v0, o["query"][0][0]["w_sibs"] + 33), "E_MERKLE")
    mut("vec_n16_tamper_fold_path", v0, root, flip(v0, o["query"][1][3]["sibs"] + 2), "E_MERKLE")
    mut("vec_n16_tamper_fold_value", v0, root, flip(v0, o["query"][5][2]["values"] + 2), "E_MERKLE")
    return {"spec": "fpn-wire-2.0", "relation": "FOLDED_VEC_MATMUL",
            "profiles": {V16.name: V16.profile_id.hex(), V22.name: V22.profile_id.hex()},
            "weights": {"tile_n16": root.hex(), "tile_n16_tampered": troot.hex()},
            "offsets": {V16.name: {"g_sum": o["g_sum"], "e_w": o["e_w"], "z_blind": o["z_blind"],
                                   "final": o["final"], "end": o["end"]}},
            "vectors": vecs}


# ============================================================================ relation 6 (§5.7)
def bound_offsets(lay):
    """Absolute byte offsets of the relation-6 sections (spec §5.7.6)."""
    o = {"stmt": 64, "weight_root": 96, "x_root": 128, "n_s": 160, "g_sum": 192}
    o["ip_round"] = [216 + 96 * i for i in range(lay.d)]
    off = 216 + 96 * lay.d
    o["e_w"] = off
    o["z_blind"] = off + 24
    off += 48
    o["pcs_round"], o["pcs_root"] = [], []
    for i in range(lay.n + 2):
        o["pcs_round"].append(off)
        off += 72
        if i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1:
            o["pcs_root"].append(off)
            off += 32
    o["final"] = off
    off += 24
    o["query"] = []
    for _ in range(lay.queries):
        q = []
        for t in range(lay.folds):
            q.append({"values": off, "sibs": off + 96})
            off += 96 + 32 * lay.depth(t)
            if t == 0:
                q[0]["w_values"], q[0]["w_sibs"] = off, off + 96
                off += 96 + 32 * lay.depth(0)
                q[0]["x_values"], q[0]["x_sibs"] = off, off + 96
                off += 96 + 32 * lay.depth(0)
                q[0]["x_end"] = off
        q.append({"end": off})
        o["query"].append(q)
    o["end"] = off
    assert off == 64 + lay.body_used
    return o


def gen_folded_vec_bound():
    from fpn_verify_ref.folded_vec_bound import layout_of as bound_layout_of
    from flashprove_next.field import ONE as HONE, hadd as HADD
    from prover.fpn_prover.backend import folded_vec as FV
    from prover.fpn_prover.backend import folded_vec_bound as FB

    drbg = DRBG(b"FPN2 folded-vec-bound vectors v1")
    rng = DrbgRng(drbg)
    B16, B22 = PR.FOLDED_VEC_BOUND_N16_V1, PR.FOLDED_VEC_BOUND_N22_V1
    R5_16 = PR.FOLDED_VEC_N16_V1
    p16 = _prof_dict(B16)
    lay = bound_layout_of(B16)
    flay = FB.layout_of(p16)
    o = bound_offsets(lay)
    vecs, bases = [], {}

    def add(name, data, expect, wroot, xroot, pin=None, note="", base=None, patches=(), truncate=None, append=b"",
            zero_after=None):
        if zero_after is not None:          # bytes the verifier never reads before rejecting are stored as zero
            data = data[:zero_after] + b"\x00" * (len(data) - zero_after)
            note = (note + "; " if note else "") + f"bytes from offset {zero_after} zeroed (rejected earlier)"
        got = verify_result(data, False, bytes.fromhex(pin) if pin else None, wroot, xroot)
        assert got == expect, (name, got, expect)
        v = {"name": name, "allow_reduced": False, "expected_profile_id": pin,
             "expected_circuit_root": None if wroot is None else wroot.hex(),
             "expected_x_root": None if xroot is None else xroot.hex(), "expect": expect, "note": note}
        if base is None:
            v.update(prefix_hex=data.rstrip(b"\x00").hex(), total_len=len(data))
            bases[name] = data
        else:
            v.update(base=base, patches=[[o_, b.hex()] for o_, b in patches], truncate=truncate,
                     append_hex=append.hex())
        vecs.append(v)

    def patched(base, patches, truncate=None, append=b""):
        b = bytearray(bases[base])
        for off, by in patches:
            b[off:off + len(by)] = by
        out = bytes(b)
        return (out if truncate is None else out[:truncate]) + append

    def mut(name, base, wroot, xroot, patches, expect, note="", truncate=None, append=b""):
        add(name, patched(base, patches, truncate, append), expect, wroot, xroot, note=note, base=base,
            patches=patches, truncate=truncate, append=append)

    def flip(base, off, mask=0x01):
        return [(off, bytes([bases[base][off] ^ mask]))]

    le = lambda x, n=8: x.to_bytes(n, "little")  # noqa: E731
    sint = lambda: (drbg.u64() % (1 << 16)) - (1 << 15)  # noqa: E731
    Bn, K = flay.B, flay.K
    Wi = np.array([[[sint() for _ in range(K)] for _ in range(FV.ROWS)] for _ in range(Bn)], dtype=np.int64)

    def acts():
        Xi = np.array([[sint() for _ in range(K)] for _ in range(Bn)], dtype=np.int64)
        Yi, bits = FV.exact_products(Wi, Xi)
        assert bits < 63
        return Xi, FV.to_field(Xi), Yi

    Xi, Xv, Yi = acts()
    Yv = FV.to_field(Yi)
    W = FV.to_field(Wi)
    wroot = FB.weight_root(flay, W)
    xc = FB.commit_x(flay, Xv, rng=rng)          # committed BEFORE the proof (and before any challenge)
    xroot = xc.root

    def prove(Wf, Xf, Yf, xcom, **kw):
        return FB.prove_bound(p16, Wf, Xf, Yf, xcom, n_s=drbg.bytes(32), rng=rng, **kw)

    end_ip, end_q0 = o["e_w"] + 24, o["query"][0][-1]["end"]
    add("vec_bound_n16_valid_0", prove(W, Xv, Yv, xc), "VERIFIED", wroot, xroot,
        note="one 32 x 1024 tile of 16-bit signed integers, X pre-committed under x_root (ZK-blinded), Y = W X")
    v0 = "vec_bound_n16_valid_0"
    add("vec_bound_n16_valid_0_pinned", bases[v0], "VERIFIED", wroot, xroot, pin=B16.profile_id.hex(), base=v0)
    # a second honest statement over DIFFERENT activations: its own x_root, valid under it, wrong under the first pin
    Xi2, Xv2, Yi2 = acts()
    xc2 = FB.commit_x(flay, Xv2, rng=rng)
    add("vec_bound_n16_valid_other_x", prove(W, Xv2, FV.to_field(Yi2), xc2), "VERIFIED", wroot, xc2.root,
        note="second honest proof over other activations X2 under its own x_root")
    vx = "vec_bound_n16_valid_other_x"
    # ---- pins (fail closed) and their precedence
    add("vec_bound_n16_pinned_profile_mismatch", bases[v0], "E_PROFILE_PINNED", wroot, xroot,
        pin=B22.profile_id.hex(), base=v0)
    add("vec_bound_n16_unpinned_weight_root", bases[v0], "E_CIRCUIT_UNPINNED", None, xroot, base=v0,
        note="relation 6 fails closed without a pinned weight_root")
    add("vec_bound_n16_unpinned_x_root", bases[v0], "E_X_ROOT_UNPINNED", wroot, None, base=v0,
        note="relation 6 fails closed without a pinned x_root")
    add("vec_bound_n16_unpinned_both", bases[v0], "E_CIRCUIT_UNPINNED", None, None, base=v0,
        note="both pins missing: the weight pin is reported first")
    add("vec_bound_n16_wrong_weight_root", bases[v0], "E_CIRCUIT_PINNED", bytes(32), xroot, base=v0)
    add("vec_bound_n16_wrong_x_root", bases[v0], "E_X_ROOT_PINNED", wroot, bytes(32), base=v0)
    add("vec_bound_n16_wrong_x_root_other_statement", bases[v0], "E_X_ROOT_PINNED", wroot, xc2.root, base=v0,
        note="a valid x_root of ANOTHER honest statement is still the wrong pin")
    add("vec_bound_n16_wrong_both_roots", bases[v0], "E_CIRCUIT_PINNED", bytes(32), bytes(32), base=v0,
        note="both pins wrong: the weight pin is reported first")
    add("vec_bound_n16_x_root_pin_of_other_proof", bases[vx], "E_X_ROOT_PINNED", wroot, xroot, base=vx,
        note="honest proof of X2 checked against the pin of X: rejected, x_root is the activation binding")
    # ---- case (c): relation 5's gap -- honest trace / N_s, privately substituted (X', Y' = W X')
    Xs_i = Xi.copy()
    Xs_i[0, 11] += 1
    Xs = FV.to_field(Xs_i)
    Ys = FV.to_field(FV.exact_products(Wi, Xs_i)[0])
    xcs = FB.commit_x(flay, Xs, rng=rng)
    add("vec_bound_n16_case_c_substituted_x_under_honest_x_root",
        prove(W, Xs, Ys, xcs, hooks={"stmt_x_root": xroot}), "E_MERKLE", wroot, xroot,
        zero_after=o["query"][0][0]["x_end"],
        note="case (c): a valid-in-itself proof for (X', Y' = W X') whose statement claims the HONEST x_root; "
             "the X_com openings come from X' and do not hash to the pinned x_root")
    add("vec_bound_n16_case_c_substituted_x_own_x_root", prove(W, Xs, Ys, xcs), "E_X_ROOT_PINNED", wroot, xroot,
        zero_after=192,
        note="case (c): the same substitution proved under the cheater's own x_root' (verifier pinned to the honest x_root)")
    add("vec_bound_n16_case_c_forged_e_x", prove(W, Xs, Ys, xc, hooks={"allow_x_mismatch": True}), "E_SUMCHECK",
        wroot, xroot, zero_after=o["pcs_round"][1],
        note="case (c): sumcheck run on (X', Y' = W X') but opened against the HONEST X_com: the batched claim "
             "(which would need a forged X~(s)) breaks in the first batching round")
    # ---- adversarial provers (as relation 5), pinned to their own x_root
    Xt_i = Xi.copy()
    Xt_i[0, 7] += 1
    Xt = FV.to_field(Xt_i)
    xct = FB.commit_x(flay, Xt, rng=rng)
    add("vec_bound_n16_tampered_x", prove(W, Xt, Yv, xct), "E_ZEROCHECK", wroot, xct.root, zero_after=end_ip,
        note="one activation X[0][7] + 1 (committed) with the honest outputs: the inner-product zero-claim fails")
    Yt = Yv.copy()
    Yt[0, 3] = (int(Yt[0, 3]) + 1) % P
    add("vec_bound_n16_tampered_y", prove(W, Xv, Yt, xc), "E_ZEROCHECK", wroot, xroot, zero_after=end_ip,
        note="one output Y[0][3] + 1")
    Wt = Wi.copy()
    Wt[0, 5, 9] += 1
    Wtf = FV.to_field(Wt)
    Ytf = FV.to_field(FV.exact_products(Wt, Xi)[0])
    troot = FB.weight_root(flay, Wtf)
    add("vec_bound_n16_tampered_weights", prove(Wtf, Xv, Ytf, xc), "E_CIRCUIT_PINNED", wroot, xroot,
        zero_after=128, note="valid proof for W with W[0][5][9] + 1, verifier pinned to the honest weight_root")
    add("vec_bound_n16_tampered_weights_under_honest_root",
        prove(Wtf, Xv, Ytf, xc, hooks={"stmt_weight_root": wroot}), "E_MERKLE", wroot, xroot,
        zero_after=end_q0, note="tampered weights smuggled under the honest weight_root: weight-leaf openings fail")
    add("vec_bound_n16_adaptive_e_w", prove(W, Xv, Yv, xc, hooks={"e_w": lambda v: HADD(v, HONE)}), "E_SUMCHECK",
        wroot, xroot, zero_after=o["pcs_round"][1], note="public evaluation e_W shifted by one: the batched opening rejects")
    add("vec_bound_n16_adaptive_ip_rounds", prove(W, Xv, Yt, xc, hooks={"ip_adaptive": True}), "E_SUMCHECK",
        wroot, xroot, zero_after=o["pcs_round"][1],
        note="false Y with every inner-product round shifted so each round sum passes: the batched claim rejects")
    add("vec_bound_n16_fold_layer_shift",
        prove(W, Xv, Yv, xc, hooks={"pcs": {"code": lambda t, c: [HADD(x, HONE) for x in c] if t == 0 else c}}),
        "E_FOLD", wroot, xroot, zero_after=end_q0, note="well-formed but inconsistent folded layer 1 under a valid Merkle root")
    # ---- single mutations of the honest container
    mut("vec_bound_n16_nonzero_pad", v0, wroot, xroot, [(o["end"] + 3, b"\x04")], "E_PADDING")
    mut("vec_bound_n16_truncated_by_1", v0, wroot, xroot, [], "E_LENGTH", truncate=B16.c_p - 1)
    mut("vec_bound_n16_relabeled_as_relation5_profile", v0, wroot, xroot, [(8, R5_16.profile_id)], "E_PADDING",
        note="header profile id swapped to fpn2-folded-vec-n16-v1 (same C_p): the relation-5 layout leaves nonzero bytes in PAD")
    mut("vec_bound_n16_noncanonical_ip_round", v0, wroot, xroot, [(o["ip_round"][2] + 8, le(P))], "E_NONCANONICAL")
    mut("vec_bound_n16_noncanonical_query_value", v0, wroot, xroot,
        [(o["query"][-1][-2]["values"] + 8, le(P + 1))], "E_NONCANONICAL")
    mut("vec_bound_n16_noncanonical_x_leaf_value", v0, wroot, xroot,
        [(o["query"][3][0]["x_values"] + 16, le(P + 2))], "E_NONCANONICAL")
    mut("vec_bound_n16_flip_stmt_weight_root", v0, wroot, xroot, flip(v0, o["weight_root"] + 5), "E_CIRCUIT_PINNED")
    mut("vec_bound_n16_flip_stmt_x_root", v0, wroot, xroot, flip(v0, o["x_root"] + 5), "E_X_ROOT_PINNED")
    mut("vec_bound_n16_flip_stmt_root0", v0, wroot, xroot, flip(v0, o["stmt"] + 5), "E_ZEROCHECK",
        note="witness root is bound before the row point and lambda")
    mut("vec_bound_n16_flip_n_s", v0, wroot, xroot, flip(v0, o["n_s"] + 1), "E_ZEROCHECK")
    mut("vec_bound_n16_flip_g_sum", v0, wroot, xroot, flip(v0, o["g_sum"] + 1), "E_ZEROCHECK")
    mut("vec_bound_n16_flip_ip_round", v0, wroot, xroot, flip(v0, o["ip_round"][4] + 50), "E_ZEROCHECK")
    mut("vec_bound_n16_flip_ip_last_round", v0, wroot, xroot, flip(v0, o["ip_round"][-1] + 75), "E_SUMCHECK",
        note="last inner-product round: its value at s enters the batched claim")
    mut("vec_bound_n16_flip_e_w", v0, wroot, xroot, flip(v0, o["e_w"] + 2), "E_SUMCHECK")
    mut("vec_bound_n16_flip_z_blind", v0, wroot, xroot, flip(v0, o["z_blind"] + 9), "E_SUMCHECK")
    mut("vec_bound_n16_flip_batch_round", v0, wroot, xroot, flip(v0, o["pcs_round"][1] + 5), "E_SUMCHECK")
    mut("vec_bound_n16_flip_pcs_round", v0, wroot, xroot, flip(v0, o["pcs_round"][6] + 40), "E_SUMCHECK")
    mut("vec_bound_n16_flip_pcs_fold_root", v0, wroot, xroot, flip(v0, o["pcs_root"][1] + 31, 0x80), "E_SUMCHECK")
    mut("vec_bound_n16_flip_pcs_final", v0, wroot, xroot, flip(v0, o["final"] + 4), "E_FINAL")
    mut("vec_bound_n16_tamper_witness_leaf", v0, wroot, xroot, flip(v0, o["query"][0][0]["values"] + 7), "E_MERKLE")
    mut("vec_bound_n16_tamper_weight_leaf", v0, wroot, xroot, flip(v0, o["query"][1][0]["w_values"] + 7), "E_MERKLE")
    mut("vec_bound_n16_tamper_weight_path", v0, wroot, xroot, flip(v0, o["query"][0][0]["w_sibs"] + 33), "E_MERKLE")
    mut("vec_bound_n16_tamper_x_leaf", v0, wroot, xroot, flip(v0, o["query"][2][0]["x_values"] + 7), "E_MERKLE",
        note="one bit of an opened X_com coset value")
    mut("vec_bound_n16_tamper_x_path", v0, wroot, xroot, flip(v0, o["query"][0][0]["x_sibs"] + 33), "E_MERKLE",
        note="one bit of an X_com Merkle sibling")
    mut("vec_bound_n16_tamper_x_path_last_query", v0, wroot, xroot, flip(v0, o["query"][-1][0]["x_sibs"] + 1),
        "E_MERKLE")
    mut("vec_bound_n16_tamper_fold_path", v0, wroot, xroot, flip(v0, o["query"][1][3]["sibs"] + 2), "E_MERKLE")
    mut("vec_bound_n16_tamper_fold_value", v0, wroot, xroot, flip(v0, o["query"][5][2]["values"] + 2), "E_MERKLE")
    # ---- multi-tile (64 tiles, n = 22): honest proof by the engine path, byte-identical to the CPU reference
    multi = gen_bound_multi_tile(drbg, B22)
    if multi is not None:
        data, wroot22, xroot22 = multi
        add("vec_bound_n22_valid_multi_tile", data, "VERIFIED", wroot22, xroot22,
            note="64 full-width 32 x 1024 tiles (2,097,152 multiply-adds), HostEngine kernel-emulation path")
        add("vec_bound_n22_wrong_x_root", data, "E_X_ROOT_PINNED", wroot22, xroot, base="vec_bound_n22_valid_multi_tile")
    return {"spec": "fpn-wire-2.0", "relation": "FOLDED_VEC_BOUND",
            "profiles": {B16.name: B16.profile_id.hex(), B22.name: B22.profile_id.hex()},
            "weights": {"tile_n16": wroot.hex(), "tile_n16_tampered": troot.hex()},
            "x_roots": {"x_n16": xroot.hex(), "x_n16_other": xc2.root.hex(), "x_n16_substituted": xcs.root.hex()},
            "offsets": {B16.name: {"g_sum": o["g_sum"], "e_w": o["e_w"], "z_blind": o["z_blind"],
                                   "final": o["final"], "end": o["end"]}},
            "vectors": vecs}


def gen_bound_multi_tile(drbg, prof):
    """Valid 64-tile (n = 22) relation-6 container: (data, weight_root, x_root).  Uses the HostEngine (kernel
    host emulation), whose bytes equal the pure-Python reference (checked at n = 16 in the unit tests)."""
    from prover.fpn_prover.backend import cuda_engine as CE
    from prover.fpn_prover.backend import folded_vec as FV
    from prover.fpn_prover.backend import folded_vec_bound as FB

    pd = _prof_dict(prof)
    flay = FB.layout_of(pd)
    sint = lambda: (drbg.u64() % (1 << 16)) - (1 << 15)  # noqa: E731
    Wi = np.array([sint() for _ in range(flay.B * FV.ROWS * flay.K)], dtype=np.int64).reshape(flay.B, FV.ROWS, flay.K)
    Xi = np.array([sint() for _ in range(flay.B * flay.K)], dtype=np.int64).reshape(flay.B, flay.K)
    Yi = np.einsum("bij,bj->bi", Wi, Xi)
    assert int(np.abs(Yi).max()) < 1 << 40
    W, Xv, Yv = FV.to_field(Wi), FV.to_field(Xi), FV.to_field(Yi)
    del Wi, Xi, Yi
    eng = CE.HostEngine()
    rng = DrbgRng(drbg)
    xc = FB.commit_x(flay, Xv, rng=rng, engine=eng)
    wroot = FB.weight_root(flay, W, eng)
    data = FB.prove_bound(pd, W, Xv, Yv, xc, n_s=drbg.bytes(32), rng=rng, engine=eng)
    return data, wroot, xc.root


def main():
    check = "--check" in sys.argv
    only = None                                    # --only name[,name]: generate / check just those files
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
    rng = DRBG(b"FPN2 golden vectors v1")
    prim = {"spec": "fpn-wire-2.0", "p": hx(P), "field": gen_field(rng), "poseidon2": gen_poseidon2(rng),
            "blake2b": gen_blake2b(), "transcript": gen_transcript(rng), "profiles": gen_profiles()}
    cont = {"spec": "fpn-wire-2.0", "transcript_label": LABEL_V2.hex(), "vectors": gen_containers(rng)}
    builders = {"fpn2_primitives.json": lambda: prim, "fpn2_containers.json": lambda: cont,
                "fpn2_folded_gkr.json": gen_folded, "fpn2_folded_air.json": gen_folded_air,
                "fpn2_folded_vec.json": gen_folded_vec, "fpn2_folded_vec_bound.json": gen_folded_vec_bound}
    bad = False
    for fn, build in builders.items():
        if only is not None and fn not in only:
            continue
        obj = build()
        txt = json.dumps(obj, indent=1, sort_keys=True) + "\n"
        path = os.path.join(HERE, fn)
        if check:
            with open(path) as f:
                if f.read() != txt:
                    print(f"MISMATCH {fn}")
                    bad = True
                else:
                    print(f"OK {fn}")
        else:
            with open(path, "w") as f:
                f.write(txt)
            print(f"wrote {fn} ({len(txt)} bytes)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
