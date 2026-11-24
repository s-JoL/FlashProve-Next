# SPDX-License-Identifier: Apache-2.0
"""structural `compile_air` cache isolation + Merkle trace-row commitment / openings.

    PYTHONPATH=prover:serve:verifier/python python3.14 -W error -m unittest tests/unit/test_trace_commit_and_lower_cache.py
"""
import struct
import unittest

from fpn_prover.backend.fpn2 import FOLDED_AIR_N12, FOLDED_AIR_V1
from fpn_prover.circuit import e2e_prover as E2E
from fpn_prover.circuit import lower as LW
from fpn_prover.circuit import protocol
from fpn_prover.ir.gadget_ir import ConstraintSpec, LookupSpec, Program
from fpn_prover.runtime import trace_commit as TC
from fpn_prover.runtime.pipeline import (
    commit_trace_payload,
    precommit_request,
    prove_captured_request,
    verify_drawn_trace_rows,
)
from fpn_prover.runtime.setup_cert import SetupCertificate
from fpn_serve.capture import CaptureConfig, CaptureSession, ModelDims


# ============================================================================== compile_air cache
def tiny(coeff=1, const=0, a_name="a", a_bits=8, order=0, lookup_cols=("a",), producer="t", hint_offset=0):
    """A small program on FOLDED_AIR_N12: c = coeff * a * b + const, d = a + b (+ one reference predicate)."""
    p = Program()
    a = p.wire(a_name, a_bits, "x", producer)
    b = p.wire("b", 8, "x", "t")
    c = p.wire("c", 21, "y", "t")
    d = p.aux("d", 14, "y", "t", lambda w: w[a_name] + w["b"] + hint_offset)
    gates = [("mul", c - a * b * coeff - const), ("sum", d - a - b)]
    for name, diff in (gates if order == 0 else gates[::-1]):
        p.equal(name, diff)
    p.lookups.append(LookupSpec("lk", tuple(x if x != "a" else a_name for x in lookup_cols), lambda *v: True))
    return p


class CompileAirCacheTests(unittest.TestCase):
    def setUp(self):
        LW.clear_air_cache()

    def fp(self, program, profile=FOLDED_AIR_N12):
        return LW.air_program_fingerprint(program, profile)

    def test_hit_returns_identical_immutable_circuit_with_callers_hints(self):
        p1, p2 = tiny(), tiny()
        self.assertIsNot(p1, p2)
        c1 = LW.compile_air(p1, FOLDED_AIR_N12)
        hits = LW.AIR_CACHE_STATS["hits"]
        c2 = LW.compile_air(p2, FOLDED_AIR_N12)
        self.assertEqual(LW.AIR_CACHE_STATS["hits"], hits + 1)
        self.assertIs(c2.table, c1.table)
        self.assertEqual((c2.root, c2.digest, c2.rows, c2.variables), (c1.root, c1.digest, c1.rows, c1.variables))
        self.assertIs(c2.hints["d"], p2.hints["d"])          # prover-side hints come from the CALLER's program
        self.assertIs(c1.hints["d"], p1.hints["d"])
        ref = LW._compile_air_uncached(tiny(), FOLDED_AIR_N12)
        self.assertEqual((ref.root, ref.digest, ref.table, ref.rows, ref.defs, ref.manifest),
                         (c2.root, c2.digest, c2.table, c2.rows, c2.defs, c2.manifest))
        # a different hint (prover-side only) is NOT part of the public circuit: same entry, caller's hint used
        p3 = tiny(hint_offset=1)
        c3 = LW.compile_air(p3, FOLDED_AIR_N12)
        self.assertEqual(c3.root, c1.root)
        self.assertIs(c3.hints["d"], p3.hints["d"])
        w = {"a": 3, "b": 5, "c": 15}
        self.assertEqual(LW.assign_air(c2, p2, w).values["d"], 8)
        with self.assertRaises(LW.LoweringError):           # p3's hint gives d = 9: the `sum` gate rejects it
            LW.assign_air(c3, p3, w)

    def test_returned_dicts_cannot_contaminate_the_cache(self):
        c1 = LW.compile_air(tiny(), FOLDED_AIR_N12)
        c1.manifest.clear()
        c1.defs["bogus"] = ("lin", ())
        c2 = LW.compile_air(tiny(), FOLDED_AIR_N12)
        self.assertIn("mul", c2.manifest)
        self.assertNotIn("bogus", c2.defs)

    def test_linear_accumulator_defs_are_immutable_tuples(self):
        """Security audit MINOR regression: linear accumulator terms in `defs` must be tuples, not shared lists."""
        p = Program()
        ws = [p.wire(f"x{i}", 8, "x", "t") for i in range(4)]
        out = p.wire("out", 12, "y", "t")
        p.equal("sum4", out - ws[0] - ws[1] - ws[2] - ws[3])
        c1 = LW.compile_air(p, FOLDED_AIR_N12)
        lin_entries = [d for d in c1.defs.values() if d[0] == "lin"]
        self.assertTrue(lin_entries)
        for _, terms in lin_entries:
            self.assertIsInstance(terms, tuple)
        c2 = LW.compile_air(p, FOLDED_AIR_N12)
        w = {"x0": 1, "x1": 2, "x2": 3, "x3": 4, "out": 10}
        self.assertEqual(LW.assign_air(c1, p, w).values, LW.assign_air(c2, p, w).values)

    def test_widened_column_spec_invalidates_bounds_memo_and_never_contaminates_cache(self):
        """Security audit MAJOR regression: widening an existing column spec must re-run check_bounds()."""
        from dataclasses import replace as dc_replace

        p = tiny(a_bits=8)
        LW.compile_air(p, FOLDED_AIR_N12)                        # populates p._bounds_checked
        # Replace column spec 'a' in-place with a wider bound so `a * b` exceeds 2^62.
        p.columns["a"] = dc_replace(p.columns["a"], signed_bound_bits=56)
        with self.assertRaisesRegex(ValueError, "signed lift bound"):
            LW.compile_air(p, FOLDED_AIR_N12)
        # A fresh widened program with the same structure must also be rejected (no poisoned cache entry).
        fresh_wide = tiny(a_bits=56)
        with self.assertRaisesRegex(ValueError, "signed lift bound"):
            LW.compile_air(fresh_wide, FOLDED_AIR_N12)

    def test_expr_args_are_immutable_tuples_and_profile_is_isolated(self):
        """Security audit regressions: Expr coerces list args to tuples and AirCircuit.profile is copied."""
        from fpn_prover.ir.gadget_ir import Expr

        p = Program()
        wa = p.wire("a", 8, "x", "t")
        wb = p.wire("b", 8, "x", "t")
        raw_args = [wa, wb]
        expr = Expr("+", raw_args)
        self.assertIsInstance(expr.args, tuple)
        raw_args[1] = Expr("const", [1 << 63])                   # mutating caller's list cannot alter Expr
        self.assertEqual(expr.args, (wa, wb))
        p.equal("sum", expr - wa - wb)
        c1 = LW.compile_air(p, FOLDED_AIR_N12)
        orig_n = FOLDED_AIR_N12["n_vars"]
        c1.profile["n_vars"] = 999
        self.assertEqual(FOLDED_AIR_N12["n_vars"], orig_n)       # caller mutation cannot alter global default
        c2 = LW.compile_air(p, FOLDED_AIR_N12)
        self.assertEqual(c2.profile["n_vars"], orig_n)           # nor subsequent cached returns

        class EvilStr(str):
            pass

        with self.assertRaises(TypeError):
            Expr(EvilStr("+"), (wa, wb))
        with self.assertRaises(TypeError):
            p.wire("evil_meta", 8, EvilStr("x"), "t")
        with self.assertRaises(TypeError):
            p.wire("evil_prod", 8, "x", EvilStr("t"))

    def test_every_structural_mutation_gets_a_distinct_key_and_circuit(self):
        base = tiny()
        c0 = LW.compile_air(base, FOLDED_AIR_N12)
        k0 = self.fp(base)
        # Mutations that change the lowered Plonkish table: distinct key AND distinct circuit_root.
        for label, prog in (("coefficient (same shape, D12)", tiny(coeff=2)),
                            ("constant (same shape)", tiny(const=1)),
                            ("wire bits", tiny(a_bits=15)),
                            ("constraint order", tiny(order=1))):
            with self.subTest(label):
                self.assertNotEqual(self.fp(prog), k0)
                c = LW.compile_air(prog, FOLDED_AIR_N12)
                self.assertNotEqual(c.root, c0.root)
                self.assertNotEqual(c.digest, c0.digest)
                self.assertIsNot(c.table, c0.table)
                ref = LW._compile_air_uncached(prog, FOLDED_AIR_N12)
                self.assertEqual((c.root, c.digest), (ref.root, ref.digest))
        # Renames / public metadata outside the index-wired table (wire names, reference-predicate columns,
        # producer): distinct key AND distinct circuit digest (the digest feeds the relation-4 binding nonce).
        for label, prog in (("wire rename", tiny(a_name="a2")), ("lookup columns", tiny(lookup_cols=("b",))),
                            ("producer", tiny(producer="u"))):
            with self.subTest(label):
                self.assertNotEqual(self.fp(prog), k0)
                c = LW.compile_air(prog, FOLDED_AIR_N12)
                self.assertNotEqual(c.digest, c0.digest)
                self.assertEqual(c.digest, LW._compile_air_uncached(prog, FOLDED_AIR_N12).digest)
        # Profile: distinct key and distinct root (different K).
        self.assertNotEqual(self.fp(base, FOLDED_AIR_V1), k0)
        self.assertNotEqual(LW.compile_air(tiny(), FOLDED_AIR_V1).root, c0.root)

    def test_in_place_mutation_of_a_compiled_program_never_hits_stale_entry(self):
        p = tiny()
        c0 = LW.compile_air(p, FOLDED_AIR_N12)
        k0 = self.fp(p)
        a, b, c = (p.columns[n] for n in "abc")
        from fpn_prover.ir.gadget_ir import Expr
        wa, wb, wc = (Expr("wire", (n,)) for n in "abc")
        p.constraints[0] = ConstraintSpec("mul", wc - wa * wb * 3)
        self.assertNotEqual(self.fp(p), k0)
        c1 = LW.compile_air(p, FOLDED_AIR_N12)
        self.assertNotEqual(c1.root, c0.root)
        self.assertEqual(c1.root, LW._compile_air_uncached(p, FOLDED_AIR_N12).root)
        p.columns["b"] = type(b)("b", 1, "x", "t", 9)        # widen one wire in place
        c2 = LW.compile_air(p, FOLDED_AIR_N12)
        self.assertNotIn(c2.root, (c0.root, c1.root))
        self.assertEqual(c2.root, LW._compile_air_uncached(p, FOLDED_AIR_N12).root)

    def test_encoding_is_self_delimiting(self):
        # wire names that would collide under naive concatenation stay distinct
        p1, p2 = Program(), Program()
        x1, y1 = p1.wire("ab", 8, "x", "t"), p1.wire("c", 8, "x", "t")
        x2, y2 = p2.wire("a", 8, "x", "t"), p2.wire("bc", 8, "x", "t")
        p1.equal("g", x1 - y1)
        p2.equal("g", x2 - y2)
        self.assertNotEqual(self.fp(p1), self.fp(p2))
        self.assertNotEqual(LW._expr_code(x1 - y1), LW._expr_code(x2 - y2))

    def test_real_block_program_caches_and_matches_uncached(self):
        prog, _ = E2E.build_program()
        c1 = LW.compile_air(prog)
        c2 = LW.compile_air(E2E.build_program()[0])
        self.assertIs(c2.table, c1.table)
        self.assertEqual(c1.root, LW._compile_air_uncached(E2E.build_program()[0], FOLDED_AIR_V1).root)


# ============================================================================== Merkle trace commitment
class _FakeHandle:
    FINISHED = "finished"

    def __init__(self, pages, trace):
        self.pages, self.state, self.trace = pages, self.FINISHED, trace

    def finish(self):
        return self.trace

    def iter_verified_pages(self, token_ids):
        yield from self.pages


class _FakeLayout:
    def fingerprint(self):
        return b"L" * 32


class _FakeStore:
    def __init__(self, rows, page=4):
        self.row_bytes = len(rows[0])
        self.layout = _FakeLayout()
        n = len(rows)
        trace = type("T", (), {"request_id": "r", "n_tokens": n})()
        pages = [(bytes(32), min(page, n - i), b"".join(rows[i:i + page])) for i in range(0, n, page)]
        self._h = _FakeHandle(pages, trace)
        self.trace = trace

    def handle(self, rid):
        return self._h if rid == "r" else None


def rows_of(n, width=24):
    return [bytes((t * 7 + j) % 256 for j in range(width)) for t in range(n)]


class TraceMerkleTests(unittest.TestCase):
    def test_openings_verify_for_every_size_and_reject_tampering(self):
        for n in (1, 2, 3, 5, 8, 17):
            with self.subTest(n=n):
                store = _FakeStore(rows_of(n))
                tree = TC.commit_trace_merkle(store, store.trace, "r", list(range(n)))
                self.assertEqual(tree.n_tokens, n)
                positions = sorted({0, n - 1, n // 2})
                bundle = TC.open_trace_rows(tree, positions)
                enc = bundle.encode()
                self.assertEqual(TC.TraceRowOpenings.decode(enc), bundle)
                ok = TC.verify_trace_row_openings(tree.root, n, b"L" * 32, enc, row_bytes=24,
                                                  expected_positions=positions)
                self.assertEqual(ok["verdict"], "VERIFIED")
                self.assertEqual(ok["rows"], {t: rows_of(n)[t] for t in positions})
                # one flipped activation byte in an opened row
                o = bundle.openings[0]
                bad_row = bytes([o.row[0] ^ 1]) + o.row[1:]
                bad = TC.TraceRowOpenings(n, b"L" * 32, 24, (TC.TraceRowOpening(o.position, bad_row, o.path),)
                                          + bundle.openings[1:])
                self.assertEqual(TC.verify_trace_row_openings(tree.root, n, b"L" * 32, bad)["verdict"],
                                 TC.E_TRACE_ROW_PATH)
                # missing opening / wrong public parameters
                if len(positions) > 1:
                    short = TC.TraceRowOpenings(n, b"L" * 32, 24, bundle.openings[1:])
                    self.assertEqual(TC.verify_trace_row_openings(tree.root, n, b"L" * 32, short,
                                                                  expected_positions=positions)["verdict"],
                                     TC.E_TRACE_POSITION)
                self.assertEqual(TC.verify_trace_row_openings(tree.root, n + 1, b"L" * 32, enc)["verdict"],
                                 TC.E_TRACE_HEADER)
                self.assertEqual(TC.verify_trace_row_openings(tree.root, n, b"M" * 32, enc)["verdict"],
                                 TC.E_TRACE_HEADER)
                self.assertEqual(TC.verify_trace_row_openings(tree.root, n, b"L" * 32, enc, row_bytes=23)["verdict"],
                                 TC.E_TRACE_HEADER)
                self.assertEqual(TC.verify_trace_row_openings(tree.root, n, b"L" * 32, enc[:-1])["verdict"],
                                 TC.E_TRACE_DECODE)

    def test_row_moved_to_another_position_is_rejected(self):
        rows = rows_of(8)
        store = _FakeStore(rows)
        tree = TC.commit_trace_merkle(store, store.trace, "r", list(range(8)))
        b = TC.open_trace_rows(tree, [2, 3])
        o2, o3 = b.openings
        swapped = TC.TraceRowOpenings(8, b"L" * 32, 24, (TC.TraceRowOpening(2, o3.row, o2.path), o3))
        self.assertEqual(TC.verify_trace_row_openings(tree.root, 8, b"L" * 32, swapped)["verdict"],
                         TC.E_TRACE_ROW_PATH)

    def test_root_binds_count_layout_and_every_row(self):
        rows = rows_of(6)
        s0 = _FakeStore(rows)
        r0 = TC.commit_trace_merkle(s0, s0.trace, "r", []).root
        s = _FakeStore(rows, page=2)                          # different page split, same logical rows
        self.assertEqual(TC.commit_trace_merkle(s, s.trace, "r", []).root, r0)
        for t in range(6):
            alt = list(rows)
            alt[t] = bytes([alt[t][5] ^ 0x80]).join([alt[t][:5], alt[t][6:]])
            s = _FakeStore(alt)
            self.assertNotEqual(TC.commit_trace_merkle(s, s.trace, "r", []).root, r0)
        s = _FakeStore(rows[:5])
        self.assertNotEqual(TC.commit_trace_merkle(s, s.trace, "r", []).root, r0)
        self.assertNotEqual(TC.payload_root(bytes(32), 6, b"L" * 32), TC.payload_root(bytes(32), 6, b"M" * 32))

    def test_binding_digest_is_input_exact(self):
        a = TC.activation_binding_digest([("x_attn", 5, 3, b"\x01\x02")], b"D")
        self.assertNotEqual(a, TC.activation_binding_digest([("x_attn", 5, 3, b"\x01\x03")], b"D"))
        self.assertNotEqual(a, TC.activation_binding_digest([("x_attn", 6, 3, b"\x01\x02")], b"D"))
        self.assertNotEqual(a, TC.activation_binding_digest([("x_mlp", 5, 3, b"\x01\x02")], b"D"))
        self.assertNotEqual(a, TC.activation_binding_digest([("x_attn", 5, 3, b"\x01\x02")], b"E"))


class IncrementalTraceCommitTests(unittest.TestCase):
    """write-once incremental trace commitment == commit_trace_merkle, hashing only fresh pages, and its
    page leaves driving the St_i ledger + per-turn C_pre (verify_agentic_chain with fresh pages AND commitments)."""

    @staticmethod
    def chain(root, toks):
        from fpn_serve.capture.store import block_hash_chain, partial_block_hash
        hs = block_hash_chain(root, toks)
        tail = len(toks) % 16
        if tail:
            hs.append(partial_block_hash(hs[-1] if hs else root, toks[len(toks) - tail:]))
        return hs

    def test_incremental_root_equals_full_and_hashes_only_fresh_pages(self):
        from fpn_prover.runtime import agentic_state as AS
        session = CaptureSession.create(ModelDims(2, 8, 2, 1, 4, 16), CaptureConfig())
        self.addCleanup(session.close)
        rb = session.store.row_bytes

        def rows(toks, start):
            out = bytearray()
            for i, t in enumerate(toks):
                r = bytearray(rb)
                for slot in session.layout.slots:
                    if slot.kind == "lse":
                        struct.pack_into("<f", r, 2 * slot.offset, 1.0 + (start + i) / 64)
                    else:
                        struct.pack_into("<H", r, 2 * slot.offset, 0x3F80 + ((start + i) * 7 + t) % 97)
                out += r
            return bytes(out)

        com = TC.IncrementalTraceCommitter(session.store, self.chain, threads=4)
        self.addCleanup(com.close)
        setup = SetupCertificate(b"w" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 7).encode()
        ags = AS.AgenticSession(setup, session.layout.fingerprint())
        toks, prev_total, records, fresh_pages, commitments = [], 0, [], [], []
        for k, (grow, resp) in enumerate(((40, [5, 6]), (27, [7]), (50, [8, 9, 10]))):
            new = [(1000 * k + i) % 251 for i in range(grow)]
            all_toks = toks + new
            prefix = (len(toks) // 16) * 16                     # full pages of the previous context are mounted
            rid = f"turn{k}"
            session.register_request(rid, all_toks[:prefix])
            self.assertTrue(session.step_cpu([(rid, all_toks[prefix:])], rows(all_toks[prefix:], prefix)))
            trace = session.settle()[rid]
            inc = com.commit(trace, rid, all_toks, retain_fresh_rows=True)
            full = TC.commit_trace_merkle(session.store, trace, rid, all_toks)
            self.assertEqual(inc.tree.root, full.root)
            self.assertEqual(inc.tree.levels, full.levels)
            self.assertEqual(inc.prefix_pages * 16, prefix)
            self.assertEqual(inc.stats["rows_hashed"], len(all_toks) - prefix)       # O(fresh) payload hashing
            self.assertLessEqual(prefix, prev_total)
            # openings of fresh positions from the incremental tree verify against the same root
            pos = sorted({prefix, len(all_toks) - 1})
            ok = TC.verify_trace_row_openings(full.root, len(all_toks), session.layout.fingerprint(),
                                              TC.open_trace_rows(inc.tree, pos).encode(), row_bytes=rb,
                                              expected_positions=set(pos))
            self.assertEqual(ok["verdict"], "VERIFIED")
            fph = [AS.page_hash(j * 16, min(16, len(all_toks) - j * 16), inc.page_digest_payload(j))
                   for j in range(inc.prefix_pages, len(inc.page_leaves))]
            turn = ags.begin_turn(prefix, len(all_toks) - prefix, fph, resp)
            c_pre = ags.precommit(turn, inc.tree.root, resp)
            self.assertNotEqual(c_pre, precommit_request(setup, inc.tree.root, resp))   # St_i is bound in
            records.append(turn)
            fresh_pages.append(fph)
            commitments.append(AS.TurnCommitment(inc.tree.root, tuple(resp), b"", c_pre))
            toks, prev_total = all_toks, len(all_toks)
        ok = AS.verify_agentic_chain(setup, session.layout.fingerprint(), records, fresh_pages=fresh_pages,
                                     commitments=commitments)
        self.assertEqual(ok["verdict"], "VERIFIED")
        bad_c = list(commitments)
        bad_c[1] = AS.TurnCommitment(bad_c[1].trace_payload_root, bad_c[1].response_tokens, b"",
                                     bytes([bad_c[1].c_pre[0] ^ 1]) + bad_c[1].c_pre[1:])
        self.assertEqual(AS.verify_agentic_chain(setup, session.layout.fingerprint(), records,
                                                 fresh_pages=fresh_pages, commitments=bad_c)["verdict"],
                         "E_CPRE_MISMATCH")
        bad_p = [list(x) for x in fresh_pages]
        bad_p[2][0] = bytes([bad_p[2][0][0] ^ 1]) + bad_p[2][0][1:]
        self.assertEqual(AS.verify_agentic_chain(setup, session.layout.fingerprint(), records,
                                                 fresh_pages=bad_p, commitments=commitments)["verdict"],
                         "E_FRESH_ROOT")
        # a tampered chain (page hash list of another token sequence) is refused before any row is hashed
        with self.assertRaises(ValueError):
            com.commit(session.store.handle("turn2").finish(), "turn2", [t ^ 1 for t in toks])


class CapturedRequestOpeningTests(unittest.TestCase):
    def test_live_capture_drawn_rows_open_against_c_pre_and_tampered_row_is_rejected(self):
        session = CaptureSession.create(ModelDims(2, 8, 2, 1, 4, 16), CaptureConfig())
        self.addCleanup(session.close)
        prefix, fresh, response = list(range(16)), [16, 17, 18], [18, 19]
        row = bytearray(session.store.row_bytes)
        for slot in session.layout.slots:
            if slot.kind == "lse":
                struct.pack_into("<f", row, 2 * slot.offset, 1.25)
            else:
                struct.pack_into("<H", row, 2 * slot.offset, 0x3FC0)
        session.register_request("prefix")
        self.assertTrue(session.step_cpu([("prefix", prefix)], bytes(row) * len(prefix)))
        session.settle()
        session.register_request("reply", prefix)
        self.assertTrue(session.step_cpu([("reply", fresh)], bytes(row) * len(fresh)))
        trace = session.settle()["reply"]
        token_ids = prefix + fresh

        prog, _ = E2E.build_program()
        setup = SetupCertificate(b"w" * 32, LW.compile_air(prog).root, b"r" * 32, b"f" * 32, 7).encode()
        s_prof = protocol.SamplerProfile(2, ((8, 8),), 2, 1, 4, attn_gemms=1)
        nonce = b"\x42" * 32
        res = prove_captured_request(
            session.store, trace, "reply", token_ids=token_ids, response_tokens=response,
            cache_read_input_tokens=len(prefix), setup_cert_bytes=setup, sampler_profile=s_prof,
            weights_provider=lambda _d, ports: {"norm_gain": (E2E.bf16_word(1.0), E2E.bf16_word(1.0)),
                                                "gemm_w": E2E.bf16_word(0.5),
                                                "gemm_observed": E2E.bf16_word(ports["h_in"][0] * 0.5)},
            external_nonce=nonce)
        self.assertEqual(res.trace_payload_root, commit_trace_payload(session.store, trace, "reply", token_ids))
        self.assertEqual(res.c_pre, precommit_request(setup, res.trace_payload_root, response))
        public = dict(token_ids=token_ids, response_tokens=response, cache_read_input_tokens=len(prefix),
                      sampler_profile=s_prof, layout_fingerprint=session.layout.fingerprint(),
                      row_bytes=session.store.row_bytes, external_nonce=nonce, expected_c_pre=res.c_pre)
        ok = verify_drawn_trace_rows(setup, res.trace_payload_root, res.trace_row_openings, **public)
        self.assertEqual(ok["verdict"], "VERIFIED", ok["detail"])
        self.assertEqual(ok["seed"], res.seed)
        self.assertTrue(ok["rows"])
        for t, r in ok["rows"].items():
            self.assertGreaterEqual(t, len(prefix))
            self.assertEqual(r, bytes(row))

        # tamper ONE activation byte of a drawn row inside the opening bundle -> E_TRACE_ROW_PATH
        bundle = TC.TraceRowOpenings.decode(res.trace_row_openings)
        o = bundle.openings[0]
        forged = TC.TraceRowOpenings(bundle.n_tokens, bundle.layout_fingerprint, bundle.row_bytes,
                                     (TC.TraceRowOpening(o.position, bytes([o.row[0] ^ 1]) + o.row[1:], o.path),)
                                     + bundle.openings[1:]).encode()
        self.assertEqual(verify_drawn_trace_rows(setup, res.trace_payload_root, forged, **public)["verdict"],
                         TC.E_TRACE_ROW_PATH)
        # a root recomputed over the tampered payload no longer reproduces the reply's C_pre
        self.assertEqual(verify_drawn_trace_rows(setup, bytes(32), res.trace_row_openings, **public)["verdict"],
                         "E_CPRE_MISMATCH")
        # the reply's pre-challenge C_pre is mandatory
        no_cpre = dict(public, expected_c_pre=None)
        self.assertEqual(verify_drawn_trace_rows(setup, res.trace_payload_root, res.trace_row_openings,
                                                 **no_cpre)["verdict"], "E_CPRE_REQUIRED")
        # dropping an opening (or opening a non-drawn row) -> E_TRACE_POSITION
        short = TC.TraceRowOpenings(bundle.n_tokens, bundle.layout_fingerprint, bundle.row_bytes,
                                    bundle.openings[1:]).encode()
        self.assertEqual(verify_drawn_trace_rows(setup, res.trace_payload_root, short, **public)["verdict"],
                         TC.E_TRACE_POSITION)

        # Security audit MAJOR regression: IncrementalTraceCommitter must reject non-FULL_STRIPE stores
        inc = TC.IncrementalTraceCommitter(session.store, IncrementalTraceCommitTests.chain)
        try:
            orig_stripe = session.store.stripe
            session.store.stripe = (0, 1)
            with self.assertRaisesRegex(ValueError, "full-width"):
                inc.commit(trace, "reply", token_ids)
            session.store.stripe = orig_stripe
            res_inc = inc.commit(trace, "reply", token_ids)
            self.assertEqual(res_inc.tree.root, res.trace_payload_root)
        finally:
            inc.close()


if __name__ == "__main__":
    unittest.main()
