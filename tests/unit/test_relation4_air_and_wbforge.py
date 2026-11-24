# SPDX-License-Identifier: Apache-2.0
"""relation 4 (`FOLDED_GKR_AIR`) lowering + witness-only forgery suite (wbforge).

    PYTHONPATH=prover:serve:verifier/python python3.14 -W error -m unittest \
        tests/unit/test_relation4_air_and_wbforge.py

Every container is checked by BOTH the Rust `core/target/release/fpn-verify` (built on demand) and the
detached Python `fpn_verify_ref`, with the circuit root pinned (`--circuit-root`).  The forgeries bypass
`lower.assign_air` entirely: they hand the backend prover an IN-RANGE (or wiring-inconsistent) variable
vector and assert the exact reject code of both verifiers.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("FPN_NO_NATIVE", "1")
for p in (ROOT, HERE, os.path.join(ROOT, "verifier", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flashprove_next.field import ONE, P, hadd  # noqa: E402
from prover.fpn_prover.backend import folded_air as FA  # noqa: E402
from prover.fpn_prover.backend.fpn2 import FOLDED_AIR_V1  # noqa: E402
from prover.fpn_prover.circuit import lower as LW  # noqa: E402
from fpn_verify_ref.verify import verify_result as py_verify  # noqa: E402
from test_ir_lowering_and_ntt import SeededRng, build  # noqa: E402

FPN_VERIFY = os.path.join(ROOT, "core", "target", "release", "fpn-verify")


def rust_verifier():
    """Let Cargo validate/rebuild `fpn-verify` unconditionally; never reuse an unverified binary without Cargo."""
    if not shutil.which("cargo"):
        return None
    subprocess.run(["cargo", "build", "--release", "-q", "--manifest-path",
                    os.path.join(ROOT, "core", "Cargo.toml")], check=True, capture_output=True)
    return FPN_VERIFY if os.path.exists(FPN_VERIFY) else None


def rust_verify(data, root=None):
    with tempfile.NamedTemporaryFile(suffix=".fpn2", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        args = [FPN_VERIFY, path] + (["--circuit-root", root.hex()] if root is not None else [])
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()
    finally:
        os.unlink(path)


class TestRelation4Air(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rust = rust_verifier()
        if not cls.rust and not os.environ.get("FPN_ALLOW_NO_RUST"):
            raise RuntimeError(
                "Rust fpn-verify missing or stale and cargo unavailable "
                "(set FPN_ALLOW_NO_RUST=1 only for explicit Python-only runs)"
            )
        cls.program, cls.witness, cls.slot = build()
        cls.circuit, cls.assignment = LW.lower_air(cls.program, cls.witness)
        cls.container = LW.prove_air(cls.circuit, cls.assignment, rng=SeededRng(41))

    # ------------------------------------------------------------------ helpers
    def verify_both(self, data, root="pinned"):
        root = self.circuit.root if root == "pinned" else root
        py = py_verify(data, False, None, root)
        rs = rust_verify(data, root) if self.rust else None
        return py, rs

    def assert_both(self, data, code, root="pinned"):
        py, rs = self.verify_both(data, root)
        self.assertEqual(py, code)
        if self.rust:
            self.assertEqual(rs, "VERIFIED" if code == "VERIFIED" else f"REJECTED: {code}")

    def forge(self, values, seed, hooks=None):
        """wbforge: prove arbitrary variable values, no lowering check."""
        return LW.prove_air_unchecked(self.circuit, values, rng=SeededRng(seed), hooks=hooks)

    def pick_wire(self):
        """A wire with >= 3 limbs whose limb 1 is nonzero (so limb 1 - 1 stays in range)."""
        v = self.assignment.values
        return next(pl.wire for pl in self.circuit.limb_plans
                    if pl.limbs >= 3 and v[LW.limb_name(pl.wire, 1)] > 0)

    def in_range(self, values):
        return all(0 <= values[n] < 128 for plan in self.circuit.limb_plans for n in plan.names())

    # ------------------------------------------------------------------ structure / honesty
    def test_compiled_circuit_shape_and_manifest(self):
        c, p = self.circuit, self.program
        s = c.summary()
        self.assertLessEqual(s["rows"], c.layout.K)
        self.assertLessEqual(s["variables"], c.layout.K)
        self.assertEqual(s["range_checked"], sum(pl.query_count for pl in c.limb_plans))
        self.assertTrue(s["verifier_checked"][LW.ALGEBRAIC_GATE].startswith("yes"))
        self.assertTrue(s["verifier_checked"][LW.LOOKUP_RANGE].startswith("yes"))
        self.assertTrue(s["verifier_checked"][LW.REFERENCE_PREDICATE].startswith("no"))
        # Every equality / recomposition / tightening gate lowered to >= 1 Plonkish row.
        origins = {r.origin for r in c.rows}
        self.assertEqual(origins, {g.name for g in c.gates})
        for eq in p.constraints:
            self.assertEqual(c.manifest[eq.name], LW.ALGEBRAIC_GATE)
            self.assertIn(eq.name, origins)
        for name in ("gemm.center_dot.0", "gemm.difference.0", "norm.span", "slot.exhaustive",
                     *(s + ".boolean" for s in self.slot.selectors)):
            self.assertIn(name, origins)
        for lk in p.lookups:
            self.assertEqual(c.manifest[lk.name], LW.REFERENCE_PREDICATE)
        # Booleanity s(s-1) = 0 is a single multiplication row q_M = 1, q_L = -1.
        boolean = [r for r in c.rows if r.origin == "slot.dense_gqa.boolean"]
        self.assertEqual([(r.q, r.wires[:2]) for r in boolean],
                         [((1, P - 1, 0, 0, 0), ("slot.dense_gqa", "slot.dense_gqa"))])
        self.assertEqual(LW.air_row_failures(c, self.assignment.values), [])
        self.assertEqual(len(self.container), FOLDED_AIR_V1["c_p"])

    def test_circuit_root_is_public_and_witness_independent(self):
        for opcode in (1, 4):
            p2, _, _ = build(opcode)
            c2 = LW.compile_air(p2)
            self.assertEqual(c2.root, self.circuit.root)
            self.assertEqual(c2.digest, self.circuit.digest)
        p3, _, _ = build()
        i = next(k for k, eq in enumerate(p3.constraints) if eq.name == "gemm.difference.0")
        p3.constraints[i] = replace(p3.constraints[i], difference=p3.constraints[i].difference + 1)
        self.assertNotEqual(LW.compile_air(p3).root, self.circuit.root)

    # ------------------------------------------------------------------ honest acceptance (both verifiers)
    def test_honest_blocks_and_every_opcode_verify_in_rust_and_python(self):
        self.assert_both(self.container, "VERIFIED")
        self.assertTrue(LW.check_binding_air(self.circuit, self.container))
        self.assertFalse(LW.check_binding_air(self.circuit, self.container, server_nonce=b"\x01" * 32))
        for opcode in range(1, 5):
            with self.subTest(opcode=opcode):
                p, w, _ = build(opcode)
                circuit, assignment = LW.lower_air(p, w)
                data = LW.prove_air(circuit, assignment, rng=SeededRng(50 + opcode))
                self.assertTrue(LW.check_binding_air(circuit, data))
                self.assert_both(data, "VERIFIED")

    def test_circuit_root_must_be_pinned(self):
        self.assert_both(self.container, "E_CIRCUIT_UNPINNED", root=None)
        self.assert_both(self.container, "E_CIRCUIT_PINNED", root=b"\x00" * 32)

    def test_lowering_rejects_forged_witnesses(self):
        for name, value, expect in (("gemm.observed.0.0", 25, "gemm.difference.0"),
                                    ("gemm.center.0.0", 25, "gemm.center_dot.0"),
                                    ("slot.swa_sink", 1, "slot.exhaustive")):
            with self.subTest(wire=name):
                with self.assertRaises(LW.LoweringError) as ctx:
                    LW.lower_air(self.program, {**self.witness, name: value})
                self.assertIn(expect, ctx.exception.failures)

    # ------------------------------------------------------------------ wbforge: witness-only forgeries
    def test_wbforge_a_wire_equality_gate(self):
        forged = {**self.witness, "gemm.observed.0.0": 25}          # in range, violates observed - center = delta
        values = LW.air_values(self.circuit, forged)
        self.assertTrue(self.in_range(values))
        self.assertEqual(LW.air_row_failures(self.circuit, values), ["gemm.difference.0"])
        self.assert_both(self.forge(values, 1), "E_ZEROCHECK")

    def test_wbforge_b_limb_recombination_gate(self):
        limb = LW.limb_name("gemm.center.0.0", 0)
        base = self.assignment.values
        values = LW.air_values(self.circuit, self.witness, overrides={limb: base[limb] ^ 1})
        self.assertTrue(self.in_range(values))
        self.assertEqual(LW.air_row_failures(self.circuit, values), ["gemm.center.0.0#recompose"])
        self.assert_both(self.forge(values, 2), "E_ZEROCHECK")

    def test_wbforge_c_multiplication_selector_gate(self):
        # dense = -1 (in its 2-bit range), swa = mla = 1: sum = 1 holds, s(s - 1) = 2 != 0 fails.
        forged = {**self.witness, "slot.dense_gqa": -1, "slot.swa_sink": 1, "slot.mla_absorbed": 1}
        values = LW.air_values(self.circuit, forged)
        self.assertTrue(self.in_range(values))
        self.assertEqual(LW.air_row_failures(self.circuit, values), ["slot.dense_gqa.boolean"])
        self.assert_both(self.forge(values, 3), "E_ZEROCHECK")

    def test_wbforge_d_lookup_multiplicity(self):
        values = dict(self.assignment.values)

        def wrong_m(layout, table, V, rng):
            m = FA.range_multiplicities(table, V) + [0] * (layout.K - FA.TABLE)
            t = next(i for i, x in enumerate(m) if x)
            m[t] -= 1
            m[(t + 1) % FA.TABLE] += 1                  # total unchanged: only the GKR can see it
            return FA.build_witness(layout, table, V, rng, m=m)

        self.assert_both(self.forge(values, 4, hooks={"witness": wrong_m}), "E_GKR_ROOT")

    def test_wbforge_d2_out_of_table_limb(self):
        """Limb 0 -> limb 0 + 128 and limb 1 -> limb 1 - 1: recomposition holds, limb leaves [0, 128)."""
        wire = self.pick_wire()
        l0, l1 = LW.limb_name(wire, 0), LW.limb_name(wire, 1)
        base = self.assignment.values
        values = LW.air_values(self.circuit, self.witness, overrides={l0: base[l0] + 128, l1: base[l1] - 1})
        self.assertEqual(LW.air_row_failures(self.circuit, values), [])
        self.assertFalse(self.in_range(values))

        def counts_in_table(layout, table, V, rng):
            clipped = [v if not f or v < FA.TABLE else 0 for v, f in zip(V, table.flag)]
            m = FA.range_multiplicities(table, clipped) + [0] * (layout.K - FA.TABLE)
            return FA.build_witness(layout, table, V, rng, m=m)

        self.assert_both(self.forge(values, 5, hooks={"witness": counts_in_table}), "E_GKR_ROOT")

    def test_wbforge_copy_constraint(self):
        """Every Plonkish row holds on the gate columns, every range-checked variable is in [0, 128),
        but the gate cells of two limbs disagree with the committed (range-checked) variables."""
        wire = self.pick_wire()
        l0, l1 = LW.limb_name(wire, 0), LW.limb_name(wire, 1)
        honest = self.assignment.values
        cols_values = LW.air_values(self.circuit, self.witness,
                                    overrides={l0: honest[l0] + 128, l1: honest[l1] - 1})
        self.assertEqual(LW.air_row_failures(self.circuit, cols_values), [])
        self.assertEqual(LW.air_row_failures(self.circuit, honest), [])
        V_cols = LW.air_vector(self.circuit, cols_values)

        def miswired(layout, table, V, rng):
            return FA.build_witness(layout, table, V, rng, columns=FA.columns_of(table, V_cols))

        self.assert_both(self.forge(honest, 6, hooks={"witness": miswired}), "E_GKR_ROOT")

    def test_wbforge_e_tampered_circuit_gate(self):
        p, _, _ = build()
        i = next(k for k, eq in enumerate(p.constraints) if eq.name == "gemm.difference.0")
        p.constraints[i] = replace(p.constraints[i], difference=p.constraints[i].difference + 1)
        tampered = LW.compile_air(p)
        self.assertNotEqual(tampered.root, self.circuit.root)
        forged = {**self.witness, "gemm.observed.0.0": 23}     # satisfies observed - center - delta + 1 = 0
        self.assertEqual(p.failures(forged), [])
        values = LW.assign_air(tampered, p, forged).values
        data = LW.prove_air_unchecked(tampered, values, rng=SeededRng(7))
        # Proof of the tampered circuit is valid for ITS root, rejected under the pinned honest root.
        self.assert_both(data, "VERIFIED", root=tampered.root)
        self.assert_both(data, "E_CIRCUIT_PINNED")
        # Tampered selectors smuggled under the honest root: the circuit-leaf openings fail.
        smuggled = FA.prove(tampered.table, LW.air_vector(tampered, values), profile=FOLDED_AIR_V1,
                            rng=SeededRng(8), hooks={"stmt_root_circ": self.circuit.root})
        self.assert_both(smuggled, "E_MERKLE")
        # Smarter forger: honest root_circ in STMT AND honest circuit leaves / paths opened, the tampered
        # table used everywhere else (GKR, zerocheck, folded codeword): fold consistency rejects.
        honest_c = FA.commit_circuit(self.circuit.layout, self.circuit.table)
        smart = FA.prove(tampered.table, LW.air_vector(tampered, values), profile=FOLDED_AIR_V1,
                         rng=SeededRng(9), hooks={"stmt_root_circ": self.circuit.root,
                                                  "pcs": {"c_code": honest_c.code, "c_levels": honest_c.levels}})
        self.assert_both(smart, "E_FOLD")

    def test_wbforge_tampered_zerocheck_final(self):
        values = dict(self.assignment.values)
        data = self.forge(values, 9, hooks={"zc_finals": lambda f: [hadd(f[0], ONE)] + list(f[1:])})
        self.assert_both(data, "E_ZEROCHECK")

    def test_wbforge_adaptive_zerocheck_claim_caught_by_batched_opening(self):
        """Claimed w_a(r_gate) shifted by one and the Libra g_r re-solved so the zerocheck's own final
        equation still holds: the lie is caught by the single batched opening of (F, C)."""
        data = self.forge(dict(self.assignment.values), 10, hooks={"zc_adaptive": True})
        self.assert_both(data, "E_SUMCHECK")


if __name__ == "__main__":
    unittest.main()
