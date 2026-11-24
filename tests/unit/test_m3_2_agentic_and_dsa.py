# SPDX-License-Identifier: Apache-2.0
"""(M3-2): incremental multi-turn state chain `St_i` + committed block-sparse / DSA attention gadget.

    PYTHONPATH=.:prover:serve:verifier/python python3.14 -W error -m unittest \
        tests/unit/test_m3_2_agentic_and_dsa.py tests/unit/test_capture_and_format.py

Part 1 (`TestAgenticStateChain`): domain-separated `St_i`, 20-turn honest 32k -> 125k-token session with prefix-reuse
accounting, fork / rollback / equivocation, the seven exact reject codes (+ the additional `E_PREFIX_ROOT`) and the
C_pre binding.

Part 2 (`TestDsaBlockSparse`): the block-sparse gadget (N_blk = 16, B = 4, sink + 2 local mandatory blocks) lowered with
`compile_air` / `assign_air` onto `fpn2-folded-air-v1`, proven by `folded_air.prove`, and judged by BOTH the Rust
`core/target/release/fpn-verify --circuit-root` and the detached Python `fpn_verify_ref`.  Adversarial cheats are
rejected by the IR AND by both verifiers; every witness-carrying variable is perturbed by +-1 on the committed
table; the key gate rows are deleted one at a time (a forged witness the intact circuit rejects is admitted by the
mutated circuit and verified under that circuit's own root).
"""
import dataclasses
import hashlib
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("FPN_NO_NATIVE", "1")
for p in (ROOT, HERE, os.path.join(ROOT, "tests", "audit"), os.path.join(ROOT, "verifier", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flashprove_next.field import P  # noqa: E402
from prover.fpn_prover.backend import folded_air as FA  # noqa: E402
from prover.fpn_prover.backend.fpn2 import FOLDED_AIR_N12, FOLDED_AIR_V1  # noqa: E402
from prover.fpn_prover.circuit import dsa_block_sparse as D  # noqa: E402
from prover.fpn_prover.circuit import e2e_prover as E2E  # noqa: E402
from prover.fpn_prover.circuit import lower as LW  # noqa: E402
from prover.fpn_prover.runtime import agentic_state as AS  # noqa: E402
from prover.fpn_prover.runtime.pipeline import precommit_request  # noqa: E402
from prover.fpn_prover.runtime.setup_cert import SetupCertificate  # noqa: E402
from fpn_verify_ref.verify import verify_result as py_verify  # noqa: E402
import lowered_air as A  # noqa: E402
from underconstraint import static_audit  # noqa: E402
from test_ir_lowering_and_ntt import SeededRng  # noqa: E402
from test_relation4_air_and_wbforge import FPN_VERIFY, rust_verifier  # noqa: E402

CERT = SetupCertificate(b"\x01" * 32, b"\x02" * 32, b"\x03" * 32, b"\x04" * 32, 7).encode()
LAYOUT_FP = hashlib.blake2b(b"layout", digest_size=32).digest()
ROW_BYTES = 4096                      # stand-in capture row width for the byte accounting (scales linearly)
REPORT = {}


def H(domain, *fields):
    """Test-side re-implementation of the documented `H(domain, fields...)` encoding."""
    h = hashlib.blake2b(digest_size=32)
    h.update(struct.pack("<Q", len(domain)))
    h.update(domain)
    for f in fields:
        h.update(struct.pack("<Q", f) if isinstance(f, int) else f)
    return h.digest()


def mth(leaves):
    """RFC 6962 Merkle tree hash with the module's node domain (independent recursion)."""
    n = len(leaves)
    if n == 0:
        return H(b"fpn2/agentic-state/empty/v1")
    if n == 1:
        return leaves[0]
    k = 1 << ((n - 1).bit_length() - 1)
    return H(b"fpn2/agentic-state/node/v1", mth(leaves[:k]), mth(leaves[k:]))


# ============================================================================================ Part 1: St_i chain
def pages_for(tag, first, fresh, pt=16):
    return tuple(AS.page_hash(first + k * pt, min(pt, fresh - k * pt), b"%s/p%d" % (tag, k))
                 for k in range(-(-fresh // pt)))


def run_session(fresh_per_turn, *, session=None, tag=b"s", response=lambda i: [i, i + 1], prefix_of=None):
    """Drive `AgenticSession` through len(fresh_per_turn) turns; returns (session, turns, pages per turn)."""
    s = session or AS.AgenticSession(CERT, LAYOUT_FP)
    turns, pages, total = [], [], s.verifier.tip_node.total_tokens
    for i, fresh in enumerate(fresh_per_turn):
        prefix = (total // 16) * 16 if prefix_of is None else prefix_of(i, total)
        ph = pages_for(b"%s%d" % (tag, i), prefix, fresh)
        t = s.begin_turn(prefix, fresh, ph, response(i))
        turns.append(t)
        pages.append(ph)
        total = t.total_tokens
    return s, turns, pages


def rebuild(turn, **changes):
    """A malicious re-chainer: change fields and recompute a self-consistent St_i."""
    f = {k: getattr(turn, k) for k in ("turn_idx", "prefix_tokens", "fresh_tokens", "total_tokens",
                                       "cached_prefix_root", "fresh_pages_root", "response_digest", "prev_state")}
    f.update(changes)
    state = AS.turn_state(f["prev_state"], f["turn_idx"], f["prefix_tokens"], f["fresh_tokens"], f["total_tokens"],
                          f["cached_prefix_root"], f["fresh_pages_root"], f["response_digest"])
    return AS.AgenticTurn(**f, state=state)


class TestAgenticStateChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session, cls.turns, cls.pages = run_session([32768] + [4864] * 19)     # 20 turns, 32k -> 125,184 tokens

    def verify(self, turns, **kw):
        return AS.verify_agentic_chain(CERT, LAYOUT_FP, turns, **kw)

    # ------------------------------------------------------------------ construction
    def test_state_formulas_match_independent_encoding(self):
        cert_id = bytes.fromhex(SetupCertificate.decode(CERT).id)
        st0 = H(b"fpn2/agentic-state/genesis/v1", cert_id, LAYOUT_FP)
        self.assertEqual(AS.genesis_state(SetupCertificate.decode(CERT).id, LAYOUT_FP), st0)
        self.assertEqual(AS.genesis_state(cert_id, LAYOUT_FP), st0)
        t1 = self.turns[0]
        self.assertEqual(t1.prev_state, st0)
        self.assertEqual((t1.turn_idx, t1.prefix_tokens, t1.fresh_tokens, t1.total_tokens), (1, 0, 32768, 32768))
        pages1 = self.pages[0]
        fresh_root = H(b"fpn2/agentic-state/fresh-root/v1", 0, 32768, len(pages1), mth(list(pages1)))
        self.assertEqual(t1.fresh_pages_root, fresh_root)
        empty_prefix = H(b"fpn2/agentic-state/prefix-root/v1", 0, 0, mth([]))
        self.assertEqual(t1.cached_prefix_root, empty_prefix)
        resp = H(b"fpn2/agentic-state/response/v1", 2, struct.pack("<2I", 0, 1))
        self.assertEqual(t1.response_digest, resp)
        self.assertEqual(t1.state, H(b"fpn2/agentic-state/turn/v1", st0, 1, 0, 32768, 32768, empty_prefix, fresh_root,
                                     resp))
        # domain separation: genesis and turn domains never collide on the same fields
        self.assertNotEqual(H(b"fpn2/agentic-state/turn/v1", st0), H(b"fpn2/agentic-state/genesis/v1", st0))
        # golden vectors (regression pins of St_0 and St_1)
        self.assertEqual(st0.hex(), GOLDEN["st0"])
        self.assertEqual(t1.state.hex(), GOLDEN["st1"])

    def test_record_roundtrip_and_type_discipline(self):
        for t in self.turns[:3]:
            raw = t.encode()
            self.assertEqual(len(raw), AS.AgenticTurn.SIZE)
            self.assertEqual(AS.AgenticTurn.decode(raw), t)
        with self.assertRaises(ValueError):
            AS.AgenticTurn.decode(raw[:-1])
        with self.assertRaises(TypeError):
            dataclasses.replace(self.turns[0], turn_idx=True)             # bool is not an exact int
        with self.assertRaises(TypeError):
            dataclasses.replace(self.turns[0], state=b"short")

    def test_twenty_turn_honest_chain_and_prefix_reuse_accounting(self):
        turns, pages = self.turns, self.pages
        self.assertEqual([t.turn_idx for t in turns], list(range(1, 21)))
        self.assertTrue(all(b.prefix_tokens <= a.total_tokens and b.prev_state == a.state
                            for a, b in zip(turns, turns[1:])))
        self.assertGreater(turns[-1].total_tokens, 120_000)
        self.assertEqual(self.verify(turns, fresh_pages=pages)["verdict"], "VERIFIED")         # ledger mode
        out = self.verify(turns)                                                                # public records only
        self.assertEqual((out["verdict"], out["turns"]), ("VERIFIED", 20))
        self.assertEqual(out["tip_state"], turns[-1].state)
        # prefix pages are committed ONCE: the cached prefix root of turn i+1 is the root of the first
        # prefix_tokens/16 pages written by earlier turns (independent recursion over the page hashes).
        written = []
        for t, ph in zip(turns, pages):
            k = t.prefix_tokens // 16
            self.assertEqual(t.cached_prefix_root, H(b"fpn2/agentic-state/prefix-root/v1", t.prefix_tokens, k,
                                                     mth(written[:k])))
            written = written[:k] + list(ph)
        acc = AS.session_accounting(turns, ROW_BYTES)
        self.assertEqual(acc["deduped_capture_bytes"], sum(t.fresh_tokens for t in turns) * ROW_BYTES)
        self.assertEqual(acc["naive_capture_bytes"], sum(t.total_tokens for t in turns) * ROW_BYTES)
        self.assertGreater(acc["dedup_ratio"], 12.0)
        retained = self.session.verifier.retained_digests()
        fresh_pages_total = sum(len(p) for p in pages)
        self.assertLessEqual(retained, 2 * fresh_pages_total)           # every page hash / node is stored once
        seconds = self.session.turn_seconds
        REPORT["agentic"] = {
            "turns": 20, "first_total_tokens": turns[0].total_tokens, "last_total_tokens": turns[-1].total_tokens,
            "row_bytes_standin": ROW_BYTES, **acc, "ledger_digests_retained": retained,
            "ledger_bytes_retained": 32 * retained, "max_turn_update_ms": 1e3 * max(seconds),
            "mean_turn_update_ms": 1e3 * sum(seconds) / len(seconds),
            "naive_pages_rehashed": sum(-(-t.total_tokens // 16) for t in turns), "fresh_pages_hashed": fresh_pages_total}
        print("\nAGENTIC", REPORT["agentic"])

    def test_ledger_prefix_roots_match_reference_recursion(self):
        leaves = [H(b"leaf", i) for i in range(100)]
        ledger = AS.PageLedger().extended(leaves)
        for k in (0, 1, 2, 3, 5, 8, 13, 37, 64, 99, 100):
            self.assertEqual(ledger.tree_root(k), mth(leaves[:k]), k)
        grown = ledger.truncated(37).extended([H(b"new", i) for i in range(10)])
        self.assertEqual(grown.tree_root(47), mth(leaves[:37] + [H(b"new", i) for i in range(10)]))
        self.assertEqual(ledger.tree_root(100), mth(leaves))             # persistent: the parent is unchanged
        with self.assertRaises(ValueError):
            ledger.tree_root(101)

    def test_partial_prefix_reuse_is_page_aligned_and_rewrites_the_tail(self):
        # turn 2 reuses only 6400 of 7000 tokens (tool output replaced the tail): the tail is fresh again.
        s, turns, pages = run_session([7000, 900], prefix_of=lambda i, total: 0 if i == 0 else 6400)
        self.assertEqual((turns[1].prefix_tokens, turns[1].fresh_tokens, turns[1].total_tokens), (6400, 900, 7300))
        self.assertEqual(self.verify(turns, fresh_pages=pages)["verdict"], "VERIFIED")
        with self.assertRaises(AS.AgenticChainError) as cm:
            s.begin_turn(7005, 5, pages_for(b"x", 7005, 5), [1])           # not page aligned
        self.assertEqual(cm.exception.code, "E_TOKEN_COUNT")

    # ------------------------------------------------------------------ fork / rollback / equivocation
    def test_branch_from_an_earlier_verified_state(self):
        s, turns, pages = run_session([2048] + [512] * 7, tag=b"m")        # 8 turns
        fork_at = turns[4].state                                           # retry from turn 5
        branch = s.fork(fork_at)
        self.assertEqual([t.turn_idx for t in branch.turns], [1, 2, 3, 4, 5])
        bs, bturns, bpages = run_session([640, 256], session=branch, tag=b"b", response=lambda i: [900 + i])
        self.assertEqual([t.turn_idx for t in bturns], [6, 7])
        self.assertEqual(bturns[0].prev_state, fork_at)
        self.assertNotEqual(bturns[0].state, turns[5].state)               # same index, same parent, new content
        chain_a = self.verify(turns, fresh_pages=pages)
        chain_b = self.verify(turns[:5] + bturns, fresh_pages=pages[:5] + bpages)
        self.assertEqual((chain_a["verdict"], chain_b["verdict"]), ("VERIFIED", "VERIFIED"))
        self.assertNotEqual(chain_a["tip_state"], chain_b["tip_state"])
        # the forked branch re-uses the verified history: only its own fresh pages are new digests
        before = s.verifier.retained_digests()
        self.assertLess(before, 2 * (sum(len(p) for p in pages) + sum(len(p) for p in bpages)) + 1)
        # explicit verifier fork: accept the alternative turn 6 on a fork, never on the main chain
        v = AS.AgenticChainVerifier(CERT, LAYOUT_FP)
        for t in turns:
            self.assertEqual(v.accept(t)["verdict"], "VERIFIED")
        self.assertEqual(v.accept(bturns[0])["verdict"], "E_TURN_ORDER")      # equivocation on the main chain
        fork = v.fork(fork_at)
        self.assertEqual(fork.accept(bturns[0])["verdict"], "VERIFIED")
        self.assertEqual(fork.accept(bturns[1])["verdict"], "VERIFIED")
        self.assertEqual(v.tip, turns[-1].state)                              # the main chain is untouched
        with self.assertRaises(AS.AgenticChainError) as cm:
            v.fork(b"\x07" * 32)
        self.assertEqual(cm.exception.code, "E_STATE_CHAIN")
        # Security audit MAJOR: sibling state added to fork must NEVER leak into parent's _nodes or session.fork
        self.assertFalse(v.known(bturns[0].state))
        self.assertFalse(s.verifier.known(bturns[0].state))
        with self.assertRaises(AS.AgenticChainError) as cm:
            v.fork(bturns[0].state)
        self.assertEqual(cm.exception.code, "E_STATE_CHAIN")
        with self.assertRaises(AS.AgenticChainError) as cm:
            s.fork(bturns[0].state)
        self.assertEqual(cm.exception.code, "E_STATE_CHAIN")

    def test_batch_verify_commitments_strict_and_exact_input_encoding(self):
        """Security audit BLOCKER + MINOR regressions: batch commitments fail closed on None/length mismatch; exact int encoding."""
        t = self.turns[:3]
        p = self.pages[:3]
        comms = [
            AS.TurnCommitment(
                b"\xab" * 32,
                [i, i + 1],
                b"",
                AS.precommit_agentic_turn(CERT, t[i], b"\xab" * 32, [i, i + 1], b""),
            )
            for i in range(3)
        ]
        self.assertEqual(self.verify(t, fresh_pages=p, commitments=comms)["verdict"], "VERIFIED")
        # BLOCKER: a None entry inside commitments must NOT silently skip C_pre verification
        self.assertEqual(self.verify(t, fresh_pages=p, commitments=[comms[0], None, comms[2]])["verdict"],
                         "E_CPRE_MISMATCH")
        # Short or surplus commitments / fresh_pages must fail closed cleanly (no IndexError, no silent ignore)
        self.assertEqual(self.verify(t, fresh_pages=p, commitments=comms[:2])["verdict"], "E_CPRE_MISMATCH")
        self.assertEqual(self.verify(t, fresh_pages=p, commitments=comms + [comms[0]])["verdict"], "E_CPRE_MISMATCH")
        self.assertEqual(self.verify(t, fresh_pages=p[:2])["verdict"], "E_FRESH_ROOT")
        self.assertEqual(self.verify(t, fresh_pages=p + [p[0]])["verdict"], "E_FRESH_ROOT")
        # MINOR: response_digest & block_summary_root require exact int types and bounds
        for bad in ([True], [1.5], [-1], [1 << 32]):
            with self.assertRaises((TypeError, ValueError)):
                AS.response_digest(bad)
        cfg = D.DsaConfig(n_blocks=4, top_b=1, mandatory=(0,))
        self.assertEqual(len(D.block_summary_root(cfg, [1, 2, 3, 4], [1, 1, 1, 1], [0, 0, 0, 0])), 32)
        with self.assertRaises(ValueError):
            D.block_summary_root(cfg, [1, 2, 3], [1, 1, 1, 1], [0, 0, 0, 0])
        for bad_val in (True, 1.5, 1 << 63):
            with self.assertRaises((TypeError, ValueError)):
                D.block_summary_root(cfg, [bad_val, 2, 3, 4], [1, 1, 1, 1], [0, 0, 0, 0])

    def test_rollback_and_equivocation_are_rejected_inside_one_chain(self):
        t = self.turns
        self.assertEqual(self.verify([t[0], t[1], t[2], t[1]])["verdict"], "E_TURN_ORDER")            # replay turn 2
        self.assertEqual(self.verify([t[0], t[1], t[2], t[0]])["verdict"], "E_TURN_ORDER")            # roll back to 1
        alt = rebuild(t[2], response_digest=AS.response_digest([123456]))                              # 2nd child of t[1]
        self.assertEqual(self.verify([t[0], t[1], t[2], alt])["verdict"], "E_TURN_ORDER")             # equivocation
        v = AS.AgenticChainVerifier(CERT, LAYOUT_FP)
        for x in t[:3]:
            v.accept(x)
        self.assertEqual(v.accept(alt)["verdict"], "E_TURN_ORDER")
        self.assertEqual(v.tip, t[2].state)                                  # atomic: nothing changed
        self.assertEqual(v.accept(t[3])["verdict"], "VERIFIED")
        self.assertEqual(v.accept(rebuild(t[0], response_digest=AS.response_digest([5])))["verdict"], "E_TURN_ORDER")

    # ------------------------------------------------------------------ the seven reject codes
    def test_E_GENESIS(self):
        other_fp = hashlib.blake2b(b"other layout", digest_size=32).digest()
        self.assertEqual(AS.verify_agentic_chain(CERT, other_fp, self.turns[:3])["verdict"], "E_GENESIS")
        other_cert = SetupCertificate(b"\x09" * 32, b"\x02" * 32, b"\x03" * 32, b"\x04" * 32, 7).encode()
        self.assertEqual(AS.verify_agentic_chain(other_cert, LAYOUT_FP, self.turns[:3])["verdict"], "E_GENESIS")
        forged_first = rebuild(self.turns[0], prev_state=b"\x00" * 32)
        self.assertEqual(self.verify([forged_first] + list(self.turns[1:3]))["verdict"], "E_GENESIS")
        late_first = rebuild(self.turns[1], prev_state=b"\x00" * 32)      # chain that does not start at St_0
        self.assertEqual(self.verify([late_first])["verdict"], "E_GENESIS")
        self.assertEqual(self.verify([])["verdict"], "E_GENESIS")

    def test_E_TURN_ORDER(self):
        t = self.turns
        self.assertEqual(self.verify([t[0], rebuild(t[1], turn_idx=3)])["verdict"], "E_TURN_ORDER")    # skip
        self.assertEqual(self.verify([t[0], rebuild(t[1], turn_idx=1)])["verdict"], "E_TURN_ORDER")    # repeat
        self.assertEqual(self.verify([rebuild(t[0], turn_idx=2)])["verdict"], "E_TURN_ORDER")

    def test_E_PREFIX_EXCEEDS_PREV(self):
        t = self.turns
        bad = rebuild(t[1], prefix_tokens=t[0].total_tokens + 16, total_tokens=t[0].total_tokens + 16 + t[1].fresh_tokens)
        out = self.verify([t[0], bad])
        self.assertEqual(out["verdict"], "E_PREFIX_EXCEEDS_PREV")
        self.assertEqual(out["turn_idx"], 2)
        first = rebuild(t[0], prefix_tokens=16, total_tokens=16 + t[0].fresh_tokens)     # nothing to reuse at turn 1
        self.assertEqual(self.verify([first])["verdict"], "E_PREFIX_EXCEEDS_PREV")

    def test_E_TOKEN_COUNT(self):
        t = self.turns
        self.assertEqual(self.verify([t[0], rebuild(t[1], total_tokens=t[1].total_tokens + 1)])["verdict"],
                         "E_TOKEN_COUNT")
        self.assertEqual(self.verify([t[0], rebuild(t[1], fresh_tokens=t[1].fresh_tokens - 1)])["verdict"],
                         "E_TOKEN_COUNT")
        self.assertEqual(self.verify([t[0], rebuild(t[1], prefix_tokens=t[1].prefix_tokens - 1,
                                                    fresh_tokens=t[1].fresh_tokens + 1)])["verdict"], "E_TOKEN_COUNT")
        zero = rebuild(t[1], prefix_tokens=t[0].total_tokens, fresh_tokens=0, total_tokens=t[0].total_tokens)
        self.assertEqual(self.verify([t[0], zero])["verdict"], "E_TOKEN_COUNT")             # a turn forwards >= 1 token

    def test_E_STATE_CHAIN(self):
        t = self.turns
        self.assertEqual(self.verify([t[0], dataclasses.replace(t[1], state=b"\x00" * 32)])["verdict"], "E_STATE_CHAIN")
        for field, value in (("response_digest", b"\x11" * 32), ("fresh_pages_root", b"\x12" * 32),
                             ("cached_prefix_root", b"\x13" * 32)):                # edited WITHOUT re-chaining
            self.assertEqual(self.verify([t[0], dataclasses.replace(t[1], **{field: value})])["verdict"],
                             "E_STATE_CHAIN", field)
        unknown_parent = rebuild(t[2], prev_state=b"\x14" * 32)                    # broken link mid-chain
        self.assertEqual(self.verify([t[0], t[1], unknown_parent])["verdict"], "E_STATE_CHAIN")
        self.assertEqual(self.verify([t[0], t[2]])["verdict"], "E_STATE_CHAIN")    # a turn dropped from the chain
        out = self.verify(list(t[:5]) + [dataclasses.replace(t[5], total_tokens=t[5].total_tokens)] + list(t[6:8]))
        self.assertEqual(out["verdict"], "VERIFIED")

    def test_E_FRESH_ROOT_and_E_PREFIX_ROOT(self):
        t, pages = self.turns[:6], [list(p) for p in self.pages[:6]]
        self.assertEqual(self.verify(t, fresh_pages=pages)["verdict"], "VERIFIED")
        swapped = [list(p) for p in pages]
        swapped[3][17] = swapped[3][16]                                     # one page substituted
        out = self.verify(t, fresh_pages=swapped)
        self.assertEqual((out["verdict"], out["turn_idx"]), ("E_FRESH_ROOT", 4))
        dropped = [list(p) for p in pages]
        dropped[2].pop()
        self.assertEqual(self.verify(t, fresh_pages=dropped)["verdict"], "E_FRESH_ROOT")
        missing = [list(p) for p in pages]
        missing[1] = None
        self.assertEqual(self.verify(t, fresh_pages=missing)["verdict"], "E_FRESH_ROOT")
        rechained = rebuild(t[4], fresh_pages_root=b"\x21" * 32)             # self-consistent St, wrong fresh root
        self.assertEqual(self.verify(list(t[:4]) + [rechained], fresh_pages=pages[:5])["verdict"], "E_FRESH_ROOT")
        forged_prefix = rebuild(t[4], cached_prefix_root=b"\x22" * 32)
        self.assertEqual(self.verify(list(t[:4]) + [forged_prefix], fresh_pages=pages[:5])["verdict"], "E_PREFIX_ROOT")

    def test_E_CPRE_MISMATCH_and_cpre_binds_state(self):
        t = self.turns[:4]
        roots = [hashlib.blake2b(b"trace%d" % i, digest_size=32).digest() for i in range(4)]
        resp = [[i, i + 1] for i in range(4)]
        coms = [AS.TurnCommitment(roots[i], tuple(resp[i]), b"policy", AS.precommit_agentic_turn(CERT, t[i], roots[i],
                                                                                              resp[i], b"policy"))
                for i in range(4)]
        self.assertEqual(self.verify(t, commitments=coms)["verdict"], "VERIFIED")
        # St_i is bound: the C_pre is exactly precommit_request with the St_i-carrying policy bytes ...
        self.assertEqual(coms[1].c_pre, precommit_request(CERT, roots[1], resp[1], AS.agentic_policy_bytes(t[1], b"policy")))
        # ... and differs from the plain (state-free) C_pre, from another turn's, and from a forked branch's.
        plain = precommit_request(CERT, roots[1], resp[1], b"policy")
        self.assertNotEqual(coms[1].c_pre, plain)
        alt = rebuild(t[1], response_digest=AS.response_digest([9, 9]))
        self.assertNotEqual(AS.precommit_agentic_turn(CERT, alt, roots[1], resp[1], b"policy"), coms[1].c_pre)
        for bad in (dataclasses.replace(coms[2], c_pre=plain),                                  # state-free C_pre
                    dataclasses.replace(coms[2], trace_payload_root=roots[3]),                  # swapped trace root
                    dataclasses.replace(coms[2], policy_bytes=b"other"),
                    dataclasses.replace(coms[2], response_tokens=(2, 4)),                       # wrong response tokens
                    dataclasses.replace(coms[2], c_pre=bytes(32))):
            out = self.verify(t, commitments=coms[:2] + [bad] + coms[3:])
            self.assertEqual((out["verdict"], out["turn_idx"]), ("E_CPRE_MISMATCH", 3))
        with self.assertRaises(TypeError):
            AS.precommit_agentic_turn(CERT, t[0].encode(), roots[0], resp[0])


GOLDEN = {"st0": "164b013e1044594ed2f0099ed08e1769be0d851b91001dc856de68d3d1de36e5",
          "st1": "b14399c57fb5a9db1cd9984db12641844b4789c44b623eb3cf9059a9ddf36d65"}


# ====================================================================================== Part 2: DSA block-sparse
def nearest(z, num):
    return (2 * num + z) // (2 * z) if z > 0 else 0


class DsaFixture:
    """N_blk = 16, B = 4, mandatory = sink 0 + local 14, 15 on `fpn2-folded-air-v1` (K = 512)."""

    def __init__(self):
        self.cfg = D.DsaConfig()
        self.program, self.nm = D.build_program(self.cfg)
        t0 = time.perf_counter()
        self.circuit = LW.compile_air(self.program)
        self.compile_s = time.perf_counter() - t0
        rng = random.Random(5)
        n = self.cfg.n_blocks
        self.scores = rng.sample(range(-60, 61), n)
        self.weights = [rng.randrange(1, 32) for _ in range(n)]
        self.values = [rng.randrange(-60, 61) for _ in range(n)]
        self.base = D.honest_witness(self.cfg, self.scores, self.weights, self.values)
        self.complete = self.program.complete(self.base)
        nm = self.nm
        free = self.cfg.free_blocks
        self.selected = sorted((j for j in free if self.base[nm["sel"][j]]), key=lambda j: -self.scores[j])
        self.unselected = sorted((j for j in free if not self.base[nm["sel"][j]]), key=lambda j: -self.scores[j])
        self.m = self.selected[-1]                    # lowest selected: score == tau (permitted tie)
        self.k = self.unselected[0]                   # best unselected block
        self.tau = self.base[nm["tau"]]
        assert self.tau == self.scores[self.m] and len(self.selected) == 4
        self.z, self.num, self.o = (self.complete[nm[x]] for x in ("Z", "num", "o"))

    # witness builders --------------------------------------------------------------------------------------
    def witness(self, changes=None, aux=None, fix_o=True):
        """Primary witness with `changes`; `o` refitted to the forged selection (nearest rounding) unless given."""
        nm, w = self.nm, dict(self.base)
        w.update(changes or {})
        if fix_o and nm["o"] not in (changes or {}):
            z = sum(w[nm["sel"][j]] * w[nm["w"][j]] for j in range(self.cfg.n_blocks))
            num = sum(w[nm["sel"][j]] * w[nm["w"][j]] * w[nm["val"][j]] for j in range(self.cfg.n_blocks))
            w[nm["o"]] = nearest(z, num)
        w.update(aux or {})
        return w

    def vector(self, w, seed=0):
        return A.vector(self.circuit, LW.air_values(self.circuit, w), SeededRng(seed))

    def judge(self, w, seed=0):
        """(set of violated IR gate origins, out-of-range cells) of the committed table on the witness's vector."""
        rows, cells = A.table_failures(self.circuit.table, self.vector(w, seed))
        return {self.circuit.rows[r].origin for r in rows}, cells

    def prove(self, V, seed, table=None):
        return FA.prove(table or self.circuit.table, V, n_s=LW.air_binding_nonce(self.circuit),
                        profile=FOLDED_AIR_V1, rng=SeededRng(seed))


def rust_cli(data, root, reduced=False):
    with tempfile.NamedTemporaryFile(suffix=".fpn2", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        args = [FPN_VERIFY, path, "--circuit-root", root.hex()] + (["--allow-reduced-security"] if reduced else [])
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()
    finally:
        os.unlink(path)


class TestDsaBlockSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rust = rust_verifier()
        if not cls.rust and not os.environ.get("FPN_ALLOW_NO_RUST"):
            raise RuntimeError("Rust fpn-verify missing or stale and cargo unavailable "
                               "(set FPN_ALLOW_NO_RUST=1 only for explicit Python-only runs)")
        cls.fx = DsaFixture()
        fx = cls.fx
        cls.assignment = LW.assign_air(fx.circuit, fx.program, fx.base)
        t0 = time.perf_counter()
        cls.container = LW.prove_air(fx.circuit, cls.assignment, rng=SeededRng(77))
        cls.prove_s = time.perf_counter() - t0
        cls.table = fx.circuit.table

    # ------------------------------------------------------------------ helpers
    def both(self, data, code, root=None, reduced=False):
        root = root or self.fx.circuit.root
        self.assertEqual(py_verify(data, reduced, None, root), code)
        if self.rust:
            self.assertEqual(rust_cli(data, root, reduced), "VERIFIED" if code == "VERIFIED" else f"REJECTED: {code}")

    def ir_rejects(self, w, gate):
        """The IR (`assign_air`) refuses the witness and names `gate` among the violated relations."""
        with self.assertRaises(LW.LoweringError) as cm:
            LW.assign_air(self.fx.circuit, self.fx.program, w)
        self.assertIn(gate, cm.exception.failures)

    def cheat(self, label, w, gate, seed):
        """IR rejects (naming `gate`); the unchecked proof of the same witness is rejected by both verifiers."""
        fx = self.fx
        self.ir_rejects(w, gate)
        gates, cells = fx.judge(w, seed)
        self.assertIn(gate, gates, label)
        self.assertEqual(cells, [], label)                      # an in-range forgery: a Plonkish row, not a range cell
        V = fx.vector(w, seed)
        self.both(fx.prove(V, seed), "E_ZEROCHECK")
        return gates

    # ------------------------------------------------------------------ shape and honest end to end
    def test_circuit_shape_is_pure_relation4_rows(self):
        fx, c = self.fx, self.fx.circuit
        s = c.summary()
        used = A.table_integrity(c)
        self.assertEqual(c.predicates, ())                      # no LookupSpec: nothing is prover-side
        self.assertEqual(set(c.manifest.values()), {LW.LOOKUP_RANGE, LW.ALGEBRAIC_GATE})
        self.assertEqual((s["rows"], used), (len(c.rows), len(c.rows)))
        self.assertLessEqual(s["rows"] + FA.ZK_ROWS, s["capacity"])
        self.assertLessEqual(s["variables"], s["capacity"])
        self.assertEqual(max(g.degree for g in c.gates), 2)
        families = {g.name.split(".")[1] for g in c.gates if g.origin == "constraint"}
        self.assertEqual(families, {"sel_boolean", "mand_boolean", "mand_public", "mand_selected", "d_def", "a",
                                    "topb_card", "u_thr", "v_thr", "Z_def", "num_def", "Z_positive", "o_hi", "o_lo"})
        static_audit(fx.program)                               # every column produced and consumed, no duplicate gate
        REPORT["dsa_shape"] = {k: s[k] for k in ("rows", "capacity", "variables", "range_checked", "ir_gates")}
        REPORT["dsa_shape"]["gate_rows_by_family"] = {
            fam: sum(len(r) for g, r in A.gate_rows(c).items() if g.split(".")[1] == fam) for fam in sorted(families)}
        REPORT["dsa_timing_s"] = {"compile_air": round(fx.compile_s, 2), "prove_air_v1": round(self.prove_s, 2)}
        print("\nDSA SHAPE", REPORT["dsa_shape"], REPORT["dsa_timing_s"])

    def test_honest_end_to_end_both_verifiers(self):
        fx, c = self.fx, self.fx.circuit
        self.assertEqual(fx.program.failures(fx.base), [])
        self.assertEqual(A.table_failures(c.table, A.vector(c, self.assignment.values, SeededRng(1))), ([], []))
        self.assertEqual(len(self.container), FOLDED_AIR_V1["c_p"])
        self.both(self.container, "VERIFIED")
        py_line, _, _ = E2E.python_verify_detached(self.container, c.root)
        self.assertEqual(E2E.verdict_code(py_line), "VERIFIED")                 # isolated `-I` interpreter
        if self.rust:
            rs_line, _, _ = E2E.rust_verify(FPN_VERIFY, self.container, c.root)
            self.assertEqual(E2E.verdict_code(rs_line), "VERIFIED")
            self.assertEqual(rust_cli(self.container, bytes(32)), "REJECTED: E_CIRCUIT_PINNED")   # wrong pin
        self.assertTrue(LW.check_binding_air(c, self.container))
        # selection structure of the honest witness
        nm = fx.nm
        sel = [fx.base[x] for x in nm["sel"]]
        self.assertEqual(sum(sel), 4 + 3)
        self.assertTrue(all(sel[j] == 1 for j in fx.cfg.mandatory))
        chosen = set(fx.selected)
        self.assertEqual(chosen, set(sorted(fx.cfg.free_blocks, key=lambda j: -fx.scores[j])[:4]))
        self.assertTrue(all(fx.scores[j] >= fx.tau for j in chosen))
        self.assertTrue(all(fx.scores[j] <= fx.tau for j in fx.unselected))
        self.assertLessEqual(2 * abs(fx.z * fx.o - fx.num), fx.z)                # nearest-rounding radius

    def test_committed_summary_digest_binds_the_statement_nonce(self):
        fx, c = self.fx, self.fx.circuit
        root = D.block_summary_root(fx.cfg, fx.scores, fx.weights, fx.values)
        c_pre = hashlib.blake2b(b"c_pre", digest_size=32).digest()
        nonce = D.dsa_statement_nonce(root, c_pre)
        data = LW.prove_air(c, self.assignment, server_nonce=nonce, rng=SeededRng(78))
        self.both(data, "VERIFIED")
        self.assertTrue(LW.check_binding_air(c, data, nonce))
        tampered = list(fx.scores)
        tampered[fx.k] += 1                                                       # one committed score changed
        root2 = D.block_summary_root(fx.cfg, tampered, fx.weights, fx.values)
        self.assertNotEqual(root, root2)
        self.assertFalse(LW.check_binding_air(c, data, D.dsa_statement_nonce(root2, c_pre)))
        self.assertFalse(LW.check_binding_air(c, data, D.dsa_statement_nonce(root, bytes(32))))
        self.assertFalse(LW.check_binding_air(c, self.container, nonce))          # default-nonce proof is not bound
        other = D.DsaConfig(top_b=3)
        self.assertNotEqual(root, D.block_summary_root(other, fx.scores, fx.weights, fx.values))

    def test_unselected_blocks_do_not_influence_the_output(self):
        fx, nm = self.fx, self.fx.nm
        rng = random.Random(9)
        w2, v2 = list(fx.weights), list(fx.values)
        chosen = set(fx.selected) | set(fx.cfg.mandatory)
        for j in range(fx.cfg.n_blocks):
            if j not in chosen:
                w2[j], v2[j] = rng.randrange(0, 32), rng.randrange(-64, 64)       # arbitrary unselected KV summary
        wit = D.honest_witness(fx.cfg, fx.scores, w2, v2)
        self.assertEqual(wit[nm["o"]], fx.o)
        self.assertEqual(fx.program.complete(wit)[nm["Z"]], fx.z)
        self.assertEqual(fx.program.complete(wit)[nm["num"]], fx.num)
        data = LW.prove_air(fx.circuit, LW.assign_air(fx.circuit, fx.program, wit), rng=SeededRng(79))
        self.both(data, "VERIFIED")                                               # same public circuit, same root

    def test_ties_at_tau_are_permitted_not_a_soundness_hole(self):
        fx, nm = self.fx, self.fx.nm
        scores = list(fx.scores)
        scores[fx.k] = scores[fx.m]                                               # best unselected ties the lowest selected
        for chosen in (fx.m, fx.k):
            other = fx.m if chosen == fx.k else fx.k
            w = D.honest_witness(fx.cfg, scores, fx.weights, fx.values, tau=scores[fx.m])
            w[nm["sel"][chosen]], w[nm["sel"][other]] = 1, 0
            z = sum(w[nm["sel"][j]] * fx.weights[j] for j in range(16))
            num = sum(w[nm["sel"][j]] * fx.weights[j] * fx.values[j] for j in range(16))
            w[nm["o"]] = nearest(z, num)
            self.assertEqual(fx.program.failures(w), [])                          # either tied block may be the 4th

    # ------------------------------------------------------------------ adversarial cheats
    def test_cheat_a_drop_a_top_scoring_block(self):
        fx, nm = self.fx, self.fx.nm
        top = fx.selected[0]
        self.assertGreater(fx.scores[top], fx.tau)
        w = fx.witness({nm["sel"][top]: 0, nm["sel"][fx.k]: 1})                    # top block out, runner-up in (|S| = 4)
        gates = self.cheat("drop top", w, f"dsa.v_thr.{top}", 201)
        self.assertIn(f"dsa.u_thr.{fx.k}", gates)                                  # the replacement is below tau too
        w = fx.witness({nm["sel"][top]: 0})                                        # plain drop: cardinality broken as well
        self.ir_rejects(w, f"dsa.v_thr.{top}")
        self.ir_rejects(w, "dsa.topb_card")

    def test_cheat_b_drop_a_mandatory_sink_block(self):
        fx, nm = self.fx, self.fx.nm
        sink = fx.cfg.mandatory[0]
        w = fx.witness({nm["sel"][sink]: 0})
        self.ir_rejects(w, f"dsa.mand_selected.{sink}")
        # a careful cheater compensates cardinality, tau and the dropped block's own thresholds: ONLY the
        # mandatory gate is left to reject him.
        sk = fx.scores[fx.k]
        w = fx.witness({nm["sel"][sink]: 0, nm["sel"][fx.k]: 1, nm["tau"]: sk, nm["score"][sink]: sk})
        gates = self.cheat("drop sink", w, f"dsa.mand_selected.{sink}", 202)
        self.assertEqual(gates, {f"dsa.mand_selected.{sink}"})
        local = fx.cfg.mandatory[-1]                                               # a local-window block is mandatory too
        w = fx.witness({nm["sel"][local]: 0, nm["sel"][fx.k]: 1, nm["tau"]: sk, nm["score"][local]: sk})
        self.assertEqual(self.cheat("drop local", w, f"dsa.mand_selected.{local}", 203), {f"dsa.mand_selected.{local}"})
        w = fx.witness({nm["mand"][sink]: 0})                                      # or un-mark it: the mask is public
        self.ir_rejects(w, f"dsa.mand_public.{sink}")

    def test_cheat_c_wrong_cardinality(self):
        fx, nm = self.fx, self.fx.nm
        w = fx.witness({nm["sel"][fx.m]: 0})                                       # B - 1 selected (tau = score_m still valid)
        self.assertEqual(self.cheat("B-1", w, "dsa.topb_card", 204), {"dsa.topb_card"})
        w = fx.witness({nm["sel"][fx.k]: 1, nm["tau"]: fx.scores[fx.k]})           # B + 1 selected (tau lowered to the tie)
        self.assertEqual(self.cheat("B+1", w, "dsa.topb_card", 205), {"dsa.topb_card"})
        w = fx.witness({nm["sel"][j]: 1 for j in fx.unselected[:2]} | {nm["tau"]: fx.scores[fx.unselected[1]]})
        self.ir_rejects(w, "dsa.topb_card")                                        # B + 2

    def test_cheat_d_non_boolean_selection(self):
        fx, nm = self.fx, self.fx.nm
        j = min(fx.selected[:-1], key=lambda x: fx.scores[x])                      # smallest margin above tau
        self.assertLessEqual(2 * (fx.scores[j] - fx.tau), 127)
        w = fx.witness({nm["sel"][j]: 2, nm["sel"][fx.m]: 0})                      # sel = 2 pays for the dropped tie block
        self.assertEqual(self.cheat("sel=2", w, f"dsa.sel_boolean.{j}", 206), {f"dsa.sel_boolean.{j}"})
        w = fx.witness({nm["sel"][fx.k]: -1})                                      # sel = -1
        self.ir_rejects(w, f"dsa.sel_boolean.{fx.k}")

    def test_cheat_e_output_beyond_tolerance(self):
        fx, nm = self.fx, self.fx.nm
        sign = 1 if fx.o + 3 <= 63 else -1
        gate = "dsa.o_hi" if sign > 0 else "dsa.o_lo"
        w = fx.witness({nm["o"]: fx.o + 3 * sign})
        self.assertEqual(self.cheat("o+3", w, gate, 207), {gate})
        # inside the radius the output is accepted (nearest rounding has no other tolerance)
        self.assertLess(2 * abs(fx.z * fx.o - fx.num), fx.z)
        for delta in (1, -1):
            self.assertTrue(fx.program.failures(fx.witness({nm["o"]: fx.o + delta})))     # off by one: beyond Z / 2
        w = fx.witness({nm["w"][j]: 0 for j in range(16) if fx.base[nm["sel"][j]]})        # Z = 0 normaliser
        self.assertEqual(self.cheat("Z=0", w, "dsa.Z_positive", 208), {"dsa.Z_positive"})

    def test_cheat_f_unselected_score_above_tau(self):
        fx, nm = self.fx, self.fx.nm
        w = fx.witness({nm["score"][fx.k]: fx.tau + 3})
        self.assertEqual(self.cheat("unselected > tau", w, f"dsa.v_thr.{fx.k}", 209), {f"dsa.v_thr.{fx.k}"})
        w = fx.witness({nm["score"][fx.m]: fx.tau - 1})                            # selected block below tau
        self.assertEqual(self.cheat("selected < tau", w, f"dsa.u_thr.{fx.m}", 210), {f"dsa.u_thr.{fx.m}"})
        self.assertTrue(fx.program.failures(fx.witness({nm["score"][fx.cfg.mandatory[0]]: 63})) == [])  # mandatory: free

    # ------------------------------------------------------------------ under-constraint perturbation scan
    def test_raw_perturbation_scan_every_variable_plus_minus_one(self):
        fx, c = self.fx, self.fx.circuit
        values = LW.air_values(c, fx.complete)
        named = [c.variables[k] for k in A.named_variables(c)]
        self.assertEqual(len(named), len(c.variables) - 1 - 3 * FA.ZK_ROWS)      # all but pad and #zk
        flagged = {n for plan in c.limb_plans for n in plan.names()}
        trials = rejected = by_row = by_range = 0
        silent = []
        for name in named:
            for delta in (1, -1):
                v = LW.air_values(c, fx.complete, overrides={name: (values[name] + delta) % P})
                rows, cells = A.table_failures(c.table, A.vector(c, v))
                trials += 1
                if rows or cells:
                    rejected += 1
                    by_row += bool(rows)
                    by_range += bool(cells)
                else:
                    silent.append((name, delta))
        self.assertEqual(silent, [])
        self.assertEqual(rejected, trials)
        self.assertEqual(trials, 2 * len(named))
        classes = {"wires": sum(1 for n in named if n in fx.program.columns), "limbs_or_tight": len(flagged),
                   "accumulators": sum(1 for n in named if n in c.defs)}
        self.assertEqual(sum(classes.values()), len(named))
        REPORT["dsa_raw_scan"] = {"variables": len(named), "trials": trials, "rejected": rejected,
                                  "by_row": by_row, "by_range": by_range, "silent": 0, **classes}
        print("\nDSA RAW SCAN", REPORT["dsa_raw_scan"])

    def test_column_perturbation_audit_pins_exactly_the_unconstrained_columns(self):
        """Layer-1 audit of EVERY wire (primary and auxiliary): +1, -1, 0, sign flip, range minimum / maximum, with
        limbs / accumulators re-derived.  STALE: all other columns keep their honest values -> the columns that
        survive are exactly those NO gate reads at this witness (documented allowlist).  REGENERATED: auxiliary
        cells are re-derived from the forged input -> survivors are alternative valid witnesses inside an inequality
        or rounding band, never selection / mask / output / auxiliary cells."""
        fx, c, nm = self.fx, self.fx.circuit, self.fx.nm
        primary = [n for n in fx.program.columns if n not in fx.program.hints]
        aux = [n for n in fx.program.columns if n in fx.program.hints]

        def accepted(w):
            rows, cells = A.table_failures(c.table, A.vector(c, LW.air_values(c, w)))
            return not rows and not cells

        def candidates(name):
            v, (lo, hi) = fx.complete[name], fx.program.columns[name].bounds
            return sorted({v + 1, v - 1, 0, -v, lo, hi} - {v})

        stale, regen, trials = {}, {}, 0
        for name in fx.program.columns:
            for value in candidates(name):
                w = dict(fx.complete)
                w[name] = value
                trials += 1
                if accepted(w):
                    stale.setdefault(name, []).append(value)
                if name in primary:
                    w = dict(fx.base)
                    w[name] = value
                    trials += 1
                    if accepted(w):
                        regen.setdefault(name, []).append(value)
        chosen = set(fx.selected) | set(fx.cfg.mandatory)
        unsel = [j for j in range(16) if j not in chosen]
        # documented allowlist: columns no gate reads at this witness.  Unselected block: a_j = sel_j w_j = 0 whatever
        # w_j, val_j are (zero dependence on unselected KV blocks).  Mandatory block: d_j = 0 and sel_j = 1, so
        # both threshold products vanish whatever its score is (a mandatory block never competes).
        allow = {*(nm["w"][j] for j in unsel), *(nm["val"][j] for j in unsel), *(nm["score"][j] for j in fx.cfg.mandatory)}
        self.assertEqual(set(stale), allow, sorted(set(stale) ^ allow))
        for j in unsel:                                              # EVERY candidate value is accepted there
            for col in (nm["w"][j], nm["val"][j]):
                self.assertEqual(sorted(stale[col]), candidates(col))
        pinned_stale = [n for n in fx.program.columns if n not in stale]
        self.assertEqual(len(pinned_stale) + len(allow), len(fx.program.columns))
        # REGENERATED: an alternative valid witness may exist only for inequality / band columns
        band = {*nm["score"], nm["tau"], *nm["w"], *nm["val"]}
        self.assertTrue(set(regen) <= band, sorted(set(regen) - band))
        self.assertTrue({*nm["sel"], *nm["mand"], nm["o"]} <= set(primary) - set(regen))   # never free
        for name in aux:
            self.assertNotIn(name, stale)                            # every auxiliary cell is pinned by its definition
        REPORT["dsa_column_audit"] = {
            "columns": len(fx.program.columns), "primary": len(primary), "auxiliary": len(aux), "trials": trials,
            "stale_pinned": len(pinned_stale), "stale_free_allowlist": len(allow),
            "stale_free": {"unselected_w_val": 2 * len(unsel), "mandatory_score": len(fx.cfg.mandatory)},
            "regenerated_alternative_witness_columns": len(regen),
            "regenerated_pinned_primary": len(primary) - len(regen)}
        print("\nDSA COLUMN AUDIT", REPORT["dsa_column_audit"])

    # ------------------------------------------------------------------ row-deletion mutation test
    def forged(self):
        """gate name -> forged witness violating (only) that gate."""
        fx, nm = self.fx, self.fx.nm
        sink, sk = fx.cfg.mandatory[0], fx.scores[fx.k]
        j = min(fx.selected[:-1], key=lambda x: fx.scores[x])
        other = fx.selected[0]
        z, num = fx.z, fx.num
        a_other = fx.complete[nm["a"][other]]
        zp, nump = z + 1, num + fx.values[other]
        return {
            f"dsa.sel_boolean.{j}": fx.witness({nm["sel"][j]: 2, nm["sel"][fx.m]: 0}),
            f"dsa.mand_selected.{sink}": fx.witness({nm["sel"][sink]: 0, nm["sel"][fx.k]: 1, nm["tau"]: sk,
                                                     nm["score"][sink]: sk}),
            f"dsa.mand_public.{fx.m}": fx.witness({nm["mand"][fx.m]: 1, nm["sel"][fx.k]: 1, nm["tau"]: sk}),
            "dsa.topb_card": fx.witness({nm["sel"][fx.m]: 0}),
            f"dsa.u_thr.{fx.m}": fx.witness({nm["tau"]: fx.tau + 1}),
            f"dsa.v_thr.{fx.k}": fx.witness({nm["tau"]: sk - 1}),
            "dsa.o_hi": fx.witness({nm["o"]: fx.o + 3}) if fx.o + 3 <= 63 else None,
            "dsa.o_lo": fx.witness({nm["o"]: fx.o - 3}) if fx.o - 3 >= -64 else None,
            "dsa.Z_positive": fx.witness({nm["w"][x]: 0 for x in range(16) if fx.base[nm["sel"][x]]}),
            "dsa.Z_def": fx.witness({nm["o"]: nearest(z + 1, num)}, aux={nm["Z"]: z + 1}, fix_o=False),
            "dsa.num_def": fx.witness({nm["o"]: nearest(z, num + 1)}, aux={nm["num"]: num + 1}, fix_o=False),
            f"dsa.a.{other}.def": fx.witness({nm["o"]: nearest(zp, nump)},
                                             aux={nm["a"][other]: a_other + 1, nm["Z"]: zp, nm["num"]: nump},
                                             fix_o=False),
        }

    def test_every_row_of_the_key_gates_is_individually_necessary(self):
        fx, c = self.fx, self.fx.circuit
        gate_rows = A.gate_rows(c)
        forged = {g: w for g, w in self.forged().items() if w is not None}
        self.assertGreaterEqual(len(forged), 11)
        total_rows = 0
        for gate, w in forged.items():
            rows = gate_rows[gate]
            values = LW.air_values(c, w)
            self.assertEqual(A.table_failures(c.table, A.vector(c, values, SeededRng(3))),
                             ([rows[-1]], []), gate)                       # exactly this gate's closing row rejects it
            for r in rows:
                vec = A.row_deletion_vector(c, values, rows, r)
                V = A.vector(c, vec, SeededRng(r))
                self.assertEqual(A.table_failures(c.table, V), ([r], []), (gate, r))      # intact circuit: rejected
                mutated = A.delete_row(c.table, r)
                self.assertEqual(A.table_failures(mutated, V), ([], []), (gate, r))       # row deleted: admitted
                total_rows += 1
        REPORT["dsa_row_deletion"] = {"gates": len(forged), "rows_individually_necessary": total_rows,
                                      "gate_names": sorted(forged)}
        print("\nDSA ROW DELETION", REPORT["dsa_row_deletion"])

    def test_mand_boolean_rows_are_implied_by_the_public_pin(self):
        """`mand_j (mand_j - 1) = 0` follows from `mand_j = c_j`, c_j in {0, 1}: no vector violates only it."""
        fx, c, nm = self.fx, self.fx.circuit, self.fx.nm
        for j in (0, fx.m, 14):
            for bad in (-3, -1, 2, 5):
                w = fx.witness({nm["mand"][j]: bad})
                gates, cells = fx.judge(w)
                if f"dsa.mand_boolean.{j}" in gates:
                    self.assertIn(f"dsa.mand_public.{j}", gates)
        # ... while deleting the pin AND keeping the boolean row still rejects mand = 2, and vice versa
        j = fx.m
        rows_pin = A.gate_rows(c)[f"dsa.mand_public.{j}"]
        rows_bool = A.gate_rows(c)[f"dsa.mand_boolean.{j}"]
        mutated = A.delete_row(c.table, rows_pin[-1])
        w = fx.witness({nm["mand"][j]: 2})
        V = fx.vector(w)
        intact, after = set(A.table_failures(c.table, V)[0]), set(A.table_failures(mutated, V)[0])
        self.assertTrue(set(rows_pin + rows_bool) <= intact)
        self.assertTrue(set(rows_bool) <= after)                 # the boolean rows alone still reject mand = 2
        self.assertFalse(after & {rows_pin[-1]})                 # and the deleted pin row no longer does

    def test_row_deleted_circuits_prove_the_forgery_under_their_own_root(self):
        fx, c = self.fx, self.fx.circuit
        gate_rows = A.gate_rows(c)
        forged = self.forged()
        sink = fx.cfg.mandatory[0]
        j = min(fx.selected[:-1], key=lambda x: fx.scores[x])
        checked = 0
        for gate, pick in ((f"dsa.sel_boolean.{j}", -1), (f"dsa.mand_selected.{sink}", -1), ("dsa.topb_card", -1),
                           ("dsa.topb_card", 3), (f"dsa.u_thr.{fx.m}", -1)):
            rows = gate_rows[gate]
            r = rows[pick]
            values = LW.air_values(c, forged[gate])
            V = A.vector(c, A.row_deletion_vector(c, values, rows, r), SeededRng(r))
            self.assertEqual(A.table_failures(self.table, V), ([r], []))
            mutated = A.delete_row(self.table, r)
            own_root = FA.circuit_root(c.layout, mutated)
            self.assertNotEqual(own_root, c.root)
            self.both(fx.prove(V, 300 + r), "E_ZEROCHECK")                                  # committed circuit rejects
            self.both(fx.prove(V, 301 + r, table=mutated), "VERIFIED", root=own_root)       # deleted row admits it
            self.both(fx.prove(V, 302 + r, table=mutated), "E_CIRCUIT_PINNED")              # but not under the pinned root
            checked += 1
        self.assertEqual(checked, 5)

    # ------------------------------------------------------------------ reduced N12 profile (small config)
    def test_small_config_on_the_reduced_n12_profile(self):
        cfg = D.DsaConfig(n_blocks=3, top_b=1, mandatory=(0,))
        program, nm = D.build_program(cfg)
        circuit = LW.compile_air(program, FOLDED_AIR_N12)
        s = circuit.summary()
        self.assertLessEqual(s["rows"] + FA.ZK_ROWS, s["capacity"])
        self.assertLessEqual(s["variables"], s["capacity"])
        scores, weights, values = [10, 40, -5], [7, 9, 3], [30, -20, 11]
        wit = D.honest_witness(cfg, scores, weights, values)
        assignment = LW.assign_air(circuit, program, wit)
        data = LW.prove_air(circuit, assignment, rng=SeededRng(5))
        self.assertEqual(py_verify(data, False, None, circuit.root), "E_REDUCED_SECURITY")      # tier 0 refuses by default
        self.both(data, "VERIFIED", root=circuit.root, reduced=True)
        cheat = dict(wit)
        cheat[nm["sel"][1]], cheat[nm["sel"][2]] = 0, 1                                        # drop the top block (40) for -5
        with self.assertRaises(LW.LoweringError) as cm:
            LW.assign_air(circuit, program, cheat)
        self.assertIn("dsa.u_thr.2", cm.exception.failures)          # the replacement block scores below tau
        vals = LW.air_values(circuit, cheat)
        V = A.vector(circuit, vals, SeededRng(6))
        rows, cells = A.table_failures(circuit.table, V)
        self.assertTrue(rows)
        bad = LW.prove_air_unchecked(circuit, vals, rng=SeededRng(6))
        self.both(bad, "E_ZEROCHECK", root=circuit.root, reduced=True)
        REPORT["dsa_small"] = {k: s[k] for k in ("rows", "capacity", "variables", "range_checked")}

    def test_config_validation_and_scaling(self):
        for bad in (dict(mandatory=(3, 1)), dict(mandatory=(16,)), dict(top_b=0), dict(top_b=14), dict(n_blocks=1),
                    dict(round_den=0)):
            with self.assertRaises(ValueError):
                D.DsaConfig(**bad)
        with self.assertRaises(TypeError):
            D.DsaConfig(top_b=4.0)
        rows = {}
        for n in (8, 12, 16):
            cfg = D.DsaConfig(n_blocks=n, top_b=2, mandatory=(0, n - 1))
            rows[n] = LW.compile_air(D.build_program(cfg)[0]).summary()["rows"]
        self.assertEqual(rows[12] - rows[8], rows[16] - rows[12])                     # rows are linear in N_blk ...
        a = LW.compile_air(D.build_program(D.DsaConfig(top_b=2))[0]).summary()["rows"]
        b = LW.compile_air(D.build_program(D.DsaConfig(top_b=4))[0]).summary()["rows"]
        self.assertEqual(a, b)                                                         # ... and independent of B and of T
        REPORT["dsa_rows_by_blocks"] = rows


if __name__ == "__main__":
    unittest.main()
