# SPDX-License-Identifier: Apache-2.0
"""generalized Bailey SMEM NTT parity + IR -> relation-3 lowering.

    PYTHONPATH=prover:serve:verifier/python python3.14 -W error -m unittest \
        tests/unit/test_ir_lowering_and_ntt.py

Rust verification uses core/target/release/fpn-verify (built on demand with cargo).
"""
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("FPN_NO_NATIVE", "1")
for p in (ROOT, os.path.join(ROOT, "verifier", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from flashprove_next.field import P  # noqa: E402
from prover.fpn_prover.backend import cuda_engine as CE  # noqa: E402
from prover.fpn_prover.backend import folded_pcs as pcs  # noqa: E402
from prover.fpn_prover.backend import zk_lookup as ZL  # noqa: E402
from prover.fpn_prover.circuit import lower as LW  # noqa: E402
from prover.fpn_prover.circuit.blocks import gemm_tile_row_block, rmsnorm_block  # noqa: E402
from prover.fpn_prover.circuit.microprogram import install_microprogram  # noqa: E402
from prover.fpn_prover.ir.arith import bf16_decode, norm_rsqrt  # noqa: E402
from prover.fpn_prover.ir.format import get_contract  # noqa: E402
from prover.fpn_prover.ir.gadget_ir import Program  # noqa: E402
from fpn_verify_ref.verify import verify_result as py_verify  # noqa: E402

HAVE_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))
FPN_VERIFY = os.path.join(ROOT, "core", "target", "release", "fpn-verify")


class SeededRng:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return tuple(self.r.randrange(P) for _ in range(3))


def rust_verifier():
    if not os.path.exists(FPN_VERIFY) and shutil.which("cargo"):
        subprocess.run(["cargo", "build", "--release", "-q", "--manifest-path",
                        os.path.join(ROOT, "core", "Cargo.toml")], check=True, capture_output=True)
    return FPN_VERIFY if os.path.exists(FPN_VERIFY) else None


def rust_verify(data):
    with tempfile.NamedTemporaryFile(suffix=".fpn2", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        r = subprocess.run([FPN_VERIFY, path], capture_output=True, text=True)
        return r.stdout.strip()
    finally:
        os.unlink(path)


# ============================================================================ Deliverable 1: Bailey NTT
@unittest.skipUnless(HAVE_CXX, "needs a C++17 compiler for the host emulation")
class TestBaileyNtt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = CE.HostEngine()
        cls.rng = random.Random(2026)

    def rows(self, n, count=3):
        return np.array([[self.rng.randrange(P) for _ in range(n)] for _ in range(count)], dtype=np.uint64)

    def test_plan_shapes(self):
        self.assertEqual(CE.bailey_plan(27), [(0, 9, 0), (9, 9, 3), (18, 9, 3)])
        self.assertEqual(CE.bailey_plan(24), [(0, 12, 0), (12, 12, 0)])
        self.assertEqual(CE.bailey_plan(13), [(0, 7, 0), (7, 6, 5)])
        self.assertEqual(CE.bailey_plan(12), [(0, 12, 0)])
        for k in range(12, 30):
            plan = CE.check_plan(k, CE.bailey_plan(k))
            self.assertEqual(len(plan), -(-k // 12))
            for s, m, c in plan:
                self.assertLessEqual(m, 12)
                self.assertLessEqual(CE.bailey_smem_bytes(m, c), 48 * 1024)
                self.assertLess(1 << (k - m - c), 1 << 31)          # grid.x limit
        with self.assertRaises(ValueError):
            CE.check_plan(13, [(0, 7, 0)])
        with self.assertRaises(ValueError):
            CE.check_plan(13, [(0, 7, 0), (6, 7, 0)])

    def test_parity_against_reference_ntt(self):
        """Square 2^(2m) and non-square 2^(2m+1) lengths vs folded_pcs._ntt_limb and legacy radix-2."""
        for log_n in (10, 12, 13, 14, 16, 17):
            n = 1 << log_n
            with self.subTest(n=n, plan=CE.bailey_plan(log_n)):
                rows = self.rows(n)
                got = self.eng.ntt_rows(rows)
                self.assertTrue(np.array_equal(got, self.eng.ntt_rows_radix2(rows)))
                if log_n <= 13:      # pure-Python reference is slow beyond 8192
                    omega = CE.root_of_unity(n)
                    for r in range(len(rows)):
                        self.assertEqual([int(x) for x in got[r]],
                                         pcs._ntt_limb([int(x) for x in rows[r]], omega))

    def test_forced_multi_pass_plans(self):
        """Three- and more-pass plans (the 2^25..2^27 path) exercised at small lengths."""
        for log_n, tile in ((12, 4), (13, 5), (14, 5), (15, 5), (10, 1), (16, 6)):
            plan = CE.bailey_plan(log_n, max_tile_log=tile)
            self.assertGreaterEqual(len(plan), 3)
            with self.subTest(log_n=log_n, plan=plan):
                rows = self.rows(1 << log_n, 2)
                self.assertTrue(np.array_equal(self.eng.ntt_rows(rows, plan), self.eng.ntt_rows_radix2(rows)))

    def test_encode_codeword_matches_zk_commit(self):
        rng = SeededRng(9)
        for n in (2, 4, 6, 10):        # codewords 2^5, 2^7, 2^9, 2^13
            cells = [rng.fp3() for _ in range(1 << n)]
            code = CE.encode_codeword(self.eng, CE.to_array(cells))
            self.assertEqual(CE.to_tuples(code), list(pcs.zk_commit(cells).code))


# ============================================================================ Deliverable 2: lowering
def bf16(x):
    word = struct.unpack(">I", struct.pack(">f", x))[0]
    return (word + 0x7fff + ((word >> 16) & 1)) >> 16


SLOT_BRANCHES = (     # one valid lane row per opcode (tokens=2, experts=2, top_k=2, window=1, sink=1, conv=1)
    (0, 256, 256, 0, 512, 0, 512),
    (0, 256, 256, 0, 512, 0, 512),
    (0, 0, 1, 1, 1, 0, 256, 256, 0, 0, 0, 1, 0, 512, 0),
    (256, 0, 256, 512, 0, 0, 0, 512),
    (0, 1, 0, 256, 0, 256, 0, 256, 256, 256, 0),
)


def build(opcode=0):
    """rmsnorm_block (width 2) + gemm_tile_row_block (mxfp4, 1x1) + 5-opcode microprogram slot."""
    p = Program()
    norm = rmsnorm_block(p, "norm", 2)
    contract = get_contract("mxfp4_e2m1")
    # operand_bits=38: the declared operand width that lets radius_numerator = z . wr be a relation-4 gate.
    gemm = gemm_tile_row_block(p, "gemm", 1, 1, contract, operand_bits=38)
    slot = install_microprogram(p, "slot", tokens=2, experts=2, top_k=2, window=1, sink=1,
                                conv_width=1, extent=1)
    one = bf16(1)
    x = bf16_decode(one).value
    ref = norm_rsqrt((x * x + x * x) / 2 + Fraction(1, 10**6))
    w = dict(zip(norm, (one, one, one, one, bf16(1.0), bf16(1.0), ref.k, ref.low, ref.high,
                        ref.high - ref.low), strict=True))
    z = 8 * contract.dot_relative_bound(1).numerator
    w.update(zip(gemm, (8, 0, z, 3, 3, 24, 3 * z, 24, 0), strict=True))
    branch = SLOT_BRANCHES[opcode]
    lanes = branch + (0,) * (len(slot.lanes) - len(branch))
    w.update(zip(slot.selectors + slot.lanes,
                 tuple(int(i == opcode) for i in range(5)) + lanes, strict=True))
    return p, p.complete(w), slot      # + auxiliary wires (gated products, slack cells) from their hints


class TestLowering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rust = rust_verifier()
        if not cls.rust and not os.environ.get("FPN_ALLOW_NO_RUST"):
            raise unittest.SkipTest("Rust fpn-verify unavailable (set FPN_ALLOW_NO_RUST=1 for Python-only)")
        cls.program, cls.witness, cls.slot = build()
        assert not cls.program.failures(cls.witness), cls.program.failures(cls.witness)
        cls.circuit, cls.assignment = LW.lower(cls.program, cls.witness)
        cls.containers = LW.prove(cls.circuit, cls.assignment, rng=SeededRng(1))

    def verify_both(self, data):
        py = py_verify(data)
        rs = rust_verify(data) if self.rust else None
        return py, rs

    def assert_rejected_by_both(self, data, code=None):
        py, rs = self.verify_both(data)
        self.assertNotEqual(py, "VERIFIED")
        if code:
            self.assertEqual(py, code)
        if self.rust:
            self.assertTrue(rs.startswith("REJECTED"), rs)
            if code:
                self.assertEqual(rs, f"REJECTED: {code}")

    # ------------------------------------------------------------------ structure / honesty
    def test_manifest_covers_every_relation_honestly(self):
        c, p = self.circuit, self.program
        names = {r.name for r in (*p.constraints, *p.lookups)}
        self.assertEqual(set(c.manifest), names | {f"{n}#range" for n in p.columns})
        for k in p.constraints:
            self.assertEqual(c.manifest[k.name], LW.ALGEBRAIC_GATE)
        for k in p.lookups:
            self.assertEqual(c.manifest[k.name], LW.REFERENCE_PREDICATE)
        for slot_rel in (*self.slot.branch_names, "slot.exhaustive"):
            self.assertIn(slot_rel, c.manifest)
        self.assertIn("norm.norm", c.manifest)
        self.assertEqual(c.manifest["norm.norm"], LW.REFERENCE_PREDICATE)
        self.assertTrue(VERIFIER_CHECKED_HONEST(c.summary()["verifier_checked"]))
        # Selector booleanity is a degree-2 gate, the rest linear.
        self.assertEqual(c.summary()["max_gate_degree"], 2)

    def test_lowered_queries_and_multiplicities(self):
        c, a = self.circuit, self.assignment
        self.assertEqual(c.table_size, 128)
        self.assertEqual(c.chunks, -(-c.query_count // 128))
        self.assertGreaterEqual(c.chunks, 2)
        self.assertEqual(len(a.queries), 128 * c.chunks)
        for i in range(c.chunks):
            chunk = a.chunk(i, 128)
            self.assertEqual(sum(a.multiplicities[i]), 128)
            self.assertEqual(list(a.multiplicities[i]), [chunk.count(t) for t in range(128)])
        # Recomposition of every wire from its limbs.
        for plan in c.limb_plans:
            limbs = [a.values[name] for name in plan.names()]
            u = sum(l << (7 * i) for i, l in enumerate(limbs[:plan.limbs]))
            self.assertEqual(u - plan.offset, self.witness[plan.wire])
        # Public circuit digest is witness-independent; different shapes differ.
        p2, w2, _ = build(opcode=3)
        self.assertEqual(LW.compile_program(p2).digest, c.digest)
        self.assertNotEqual(LW.lower(p2, w2)[1].queries, a.queries)
        p3 = Program()
        rmsnorm_block(p3, "norm", 3)
        self.assertNotEqual(LW.compile_program(p3).digest, c.digest)

    # ------------------------------------------------------------------ end-to-end acceptance
    def test_containers_verify_with_python_and_rust(self):
        self.assertEqual(len(self.containers), self.circuit.chunks)
        self.assertTrue(LW.check_binding(self.circuit, self.containers))
        for data in self.containers:
            self.assertEqual(len(data), 163840)
            self.assertEqual(ZL.verify_result(data), "VERIFIED")
            py, rs = self.verify_both(data)
            self.assertEqual(py, "VERIFIED")
            if self.rust:
                self.assertEqual(rs, "VERIFIED")
            else:
                self.skipTest("Rust fpn-verify unavailable")

    def test_every_microprogram_opcode_lowers_and_verifies(self):
        for opcode in range(5):
            with self.subTest(opcode=opcode):
                p, w, _ = build(opcode)
                circuit, assignment = LW.lower(p, w)
                containers = LW.prove(circuit, assignment, rng=SeededRng(10 + opcode))
                self.assertTrue(LW.check_binding(circuit, containers))
                for data in containers:
                    py, rs = self.verify_both(data)
                    self.assertEqual(py, "VERIFIED")
                    if self.rust:
                        self.assertEqual(rs, "VERIFIED")

    def test_binding_rejects_swapped_or_foreign_containers(self):
        c, cs = self.circuit, self.containers
        self.assertFalse(LW.check_binding(c, cs[::-1]))
        self.assertFalse(LW.check_binding(c, cs[:-1]))
        self.assertFalse(LW.check_binding(c, cs, server_nonce=b"\x01" * 32))
        p3 = Program()
        rmsnorm_block(p3, "norm", 3)
        self.assertFalse(LW.check_binding(LW.compile_program(p3), cs))

    # ------------------------------------------------------------------ under-constraint audit (layer 1)
    def test_lowering_rejects_exactly_what_the_reference_rejects(self):
        """Single-column perturbations of every IR wire: lowering adds no bypass of Program.failures."""
        c, p, w = self.circuit, self.program, self.witness
        checked = flagged = 0
        for name, spec in p.columns.items():
            v = w[name]
            lo, hi = spec.bounds
            for change in sorted({v + 1, v - 1, 0, -v, lo, hi, lo - 1, hi + 1, v + P, v - P} - {v}):
                cand = {**w, name: change}
                ref_rejects = bool(p.failures(cand))
                try:
                    LW.assign(c, p, cand)
                    lowered_rejects = False
                except LW.LoweringError:
                    lowered_rejects = True
                self.assertEqual(lowered_rejects, ref_rejects, (name, change))
                checked += 1
                flagged += ref_rejects
        self.assertGreater(flagged, checked // 2)

    def test_every_limb_is_constrained_by_a_gate(self):
        """Perturbing any single lowered limb / tightening cell fails a recompose or tighten gate."""
        c, base = self.circuit, self.assignment.values
        for plan in c.limb_plans:
            for limb in plan.names():
                for delta in (1, -1, 128):
                    values = {**base, limb: base[limb] + delta}
                    failed = [g.name for g in c.gates if g.evaluate(values)]
                    self.assertTrue(failed, (limb, delta))
                    self.assertTrue(all(f.startswith(plan.wire + "#") for f in failed), failed)

    def test_tampered_constraint_changes_digest_and_breaks_binding(self):
        """Altering or deleting one equality in the Program changes the circuit digest, so the
        honest containers no longer bind to the tampered circuit."""
        from dataclasses import replace
        for mode in ("alter", "delete"):
            with self.subTest(mode=mode):
                p, _, _ = build()
                i = next(k for k, eq in enumerate(p.constraints) if eq.name == "gemm.difference.0")
                if mode == "alter":
                    p.constraints[i] = replace(p.constraints[i], difference=p.constraints[i].difference + 1)
                else:
                    del p.constraints[i]
                tampered = LW.compile_program(p)
                self.assertNotEqual(tampered.digest, self.circuit.digest)
                self.assertFalse(LW.check_binding(tampered, self.containers))

    # ------------------------------------------------------------------ forgeries: lowering rejects
    def test_tampered_wire_rejected_by_lowering(self):
        p, w = self.program, self.witness
        for name, value, expect in (
                ("gemm.observed.0.0", 25, "gemm.difference.0"),          # equality constraint
                ("norm.output.0", bf16(1.5), "norm.norm"),               # RMSNorm reference predicate
                ("slot.lane6", 513, "slot.dense_gqa.relation"),          # dense-GQA branch relation
                ("norm.span.0", 5, "range: norm.span.0"),                # out of its 2-bit range
                ("slot.dense_gqa", 2, "range: slot.dense_gqa")):         # non-boolean selector
            with self.subTest(wire=name):
                self.assertIn(name, w)
                with self.assertRaises(LW.LoweringError) as ctx:
                    LW.lower(p, {**w, name: value})
                self.assertIn(expect, ctx.exception.failures)

    def test_tampered_equality_constraint_rejected_by_lowering(self):
        p, w = self.program, self.witness
        bad = {**w, "gemm.observed.0.0": 25, "gemm.delta.0.0": 0}
        with self.assertRaises(LW.LoweringError) as ctx:
            LW.lower(p, bad)
        self.assertIn("gemm.difference.0", ctx.exception.failures)
        # Gate-level check independent of the reference interpreter.
        gate = next(g for g in self.circuit.gates if g.name == "gemm.difference.0")
        self.assertEqual(gate.evaluate(w), 0)
        self.assertNotEqual(gate.evaluate(bad), 0)
        # A forged recomposition (limb not matching its wire) fails its recompose gate.
        values = dict(self.assignment.values)
        values[LW.limb_name("gemm.center.0.0", 0)] ^= 1
        bad_gates = [g.name for g in self.circuit.gates if g.evaluate(values)]
        self.assertEqual(bad_gates, ["gemm.center.0.0#recompose"])

    def test_tampered_multiplicity_rejected(self):
        c, a = self.circuit, self.assignment
        mult = [list(m) for m in a.multiplicities]
        t = next(i for i, m in enumerate(mult[0]) if m > 0)
        mult[0][t] -= 1
        mult[0][(t + 1) % 128] += 1          # still sums to 128: only the GKR can catch it
        forged = LW.Assignment(a.values, a.queries, tuple(map(tuple, mult)))
        with self.assertRaises(LW.LoweringError):
            LW.prove(c, forged)
        # Bypassing the lowering check: real bytes reach both verifiers and are rejected.
        data = LW.prove(c, forged, rng=SeededRng(5), unchecked=True)[0]
        self.assert_rejected_by_both(data, "E_GKR_ROOT")

    def test_out_of_range_wire_rejected_by_both_verifiers(self):
        """A 17-bit wire forced to 2^16 overflows its tightened top limb to T = 128 (outside the table)."""
        c = self.circuit
        forged = {**self.witness, "norm.input.0": 1 << 16}
        with self.assertRaises(LW.LoweringError):
            LW.lower(self.program, forged)
        queries = LW.limb_queries_unchecked(c, forged)
        self.assertIn(128, queries)
        chunk = next(i for i in range(c.chunks) if 128 in queries[i * 128:(i + 1) * 128])
        data = ZL.prove(queries[chunk * 128:(chunk + 1) * 128], n_s=LW.binding_nonce(c, chunk),
                        rng=SeededRng(6))
        self.assert_rejected_by_both(data, "E_GKR_ROOT")

    def test_byte_tampering_rejected_by_both_verifiers(self):
        data = bytearray(self.containers[0])
        for off in (64 + 64 + 10, 64 + 64 + 48 + 200, 64 + 2000, 64 + 150000):
            with self.subTest(offset=off):
                bad = bytearray(data)
                bad[off] ^= 0x01
                self.assert_rejected_by_both(bytes(bad))

    # ------------------------------------------------------------------ documented limitation
    def test_limitation_gate_violations_are_not_verifier_enforced(self):
        """HONESTY PIN: relation 3 has no arithmetic component.  An in-range witness violating an
        equality gate, proven by a prover that skips `lower.assign`, is ACCEPTED by both verifiers.
        If this test ever fails, the relation gained a wiring check -- update the lower.py docstring."""
        c = self.circuit
        forged = {**self.witness, "gemm.observed.0.0": 25}     # in range, violates gemm.difference.0
        self.assertIn("gemm.difference.0", self.program.failures(forged))
        queries = LW.limb_queries_unchecked(c, forged)
        self.assertTrue(all(0 <= q < 128 for q in queries))
        self.assertNotEqual(queries[:128], self.assignment.chunk(0, 128))   # the forged chunk is proven
        data = ZL.prove(queries[:128], n_s=LW.binding_nonce(c, 0), rng=SeededRng(7))
        py, rs = self.verify_both(data)
        self.assertEqual(py, "VERIFIED")
        if self.rust:
            self.assertEqual(rs, "VERIFIED")


def VERIFIER_CHECKED_HONEST(table):
    return (table[LW.ALGEBRAIC_GATE].startswith("no") and table[LW.REFERENCE_PREDICATE].startswith("no")
            and "only" in table[LW.LOOKUP_RANGE])


if __name__ == "__main__":
    unittest.main()
