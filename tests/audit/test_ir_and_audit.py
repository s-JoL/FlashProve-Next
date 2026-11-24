import json
import struct
import unittest
from dataclasses import replace
from fractions import Fraction

from flashprove_next.field import ZERO, hf
from prover.fpn_prover.backend import folded_pcs as pcs
from prover.fpn_prover.backend import frac_logup_gkr as gkr
from prover.fpn_prover.ir.arith import (
    bf16_decode, install_bf16, install_rsqrt, norm_rsqrt, signed_range_check,
)
from prover.fpn_prover.ir.gadget_ir import (
    MicroprogramSlot, Program, attaining_rmx, topk_tau,
)
from underconstraint import audit, mutations, static_audit


class ArithmeticTests(unittest.TestCase):
    def test_bf16_full_range(self):
        program = Program()
        names = install_bf16(program)
        for word in range(65536):
            if ((word >> 7) & 255) == 255:
                with self.assertRaises(ValueError):
                    bf16_decode(word)
                continue
            decoded = bf16_decode(word)
            expected = struct.unpack(">f", struct.pack(">I", word << 16))[0]
            self.assertEqual(decoded.value, Fraction(expected))
        cases = (1, 127, 128, 0x7f7f, 0xff7f, 0x8000)
        cases += tuple(struct.unpack(">I", struct.pack(">f", x))[0] >> 16 for x in (1e8, 1e20))
        for word in cases:
            d = bf16_decode(word)
            witness = dict(zip(names, (word, d.sign, d.exponent, d.mantissa, d.significand, d.power)))
            audit(program, witness, fixed=names[:1])
            self.assertTrue(all(mutations(program, witness).values()))
        self.assertEqual(len(json.loads(json.dumps(program.ledger()))), 6)

    def test_rsqrt(self):
        program = Program()
        names = install_rsqrt(program)
        for x in (Fraction(10)**9, Fraction(10)**-20, Fraction(2)**500, Fraction(2)**-500):
            enclosure = norm_rsqrt(x)
            lo, hi = enclosure.enclosure
            self.assertLessEqual(x * lo**2, 1)
            self.assertGreaterEqual(x * hi**2, 1)
            self.assertTrue(enclosure.accepts(x))
        self.assertGreater(norm_rsqrt(Fraction(10)**-20).enclosure[0], 4096)
        for c, p in ((1953125, 9), (3, -68), (1, 500), (1, -500), (9, 0)):
            enclosure = norm_rsqrt(c * Fraction(2)**p, precision=16)
            n, d = enclosure.u.as_integer_ratio()
            k, l, h, scale = enclosure.k, enclosure.low, enclosure.high, enclosure.scale
            row = dict(zip(names, (c, p, k, n, d, l, h, -16-k,
                                    d*scale**2 - n*l*l, n*h*h - d*scale**2)))
            audit(program, row, fixed=names[:2])
            targeted = []
            for name in names[-2:]:
                if row[name]:
                    targeted.append({**row, name: row[name]+1})
                else:
                    # Equal endpoints pin both slacks through the tightness relation.
                    break
            if len(targeted) == 2:
                self.assertTrue(all(mutations(program, row, fixed=names[:2], targeted=targeted).values()))

    def test_signed_range_and_static_errors(self):
        program = Program()
        x = program.wire("x", 17, "integer", "input")
        with self.assertRaises(ValueError):
            static_audit(program)
        limbs = signed_range_check(program, "x", 16)
        self.assertEqual(limbs, signed_range_check(program, "x", 16))
        audit(program, dict(zip(("x", *limbs), (-1, 255, 127))), fixed=("x",))
        program.equal("overflow", x*x*x*x)
        with self.assertRaises(ValueError):
            static_audit(program)

    def test_audit_reports_correlated_freedom(self):
        program = Program()
        x = program.wire("x", 8, "integer", "fixture")
        y = program.wire("y", 8, "integer", "fixture")
        program.equal("agreement", x, y)
        with self.assertRaisesRegex(AssertionError, "silent strict perturbation"):
            audit(program, {"x": 1, "y": 1})
        self.assertIsNotNone(mutations(program, {"x": 1, "y": 1})["agreement"])

    def test_slot(self):
        program = Program()
        slot = MicroprogramSlot.install(program, "slot", (
            ("topk", 9, lambda *v: topk_tau(v[:3], v[3:6], v[6:8], v[8])),
            ("rmx", 10, lambda *v: attaining_rmx(v[:2], v[2:4], v[4:6], v[6], v[7:9], v[9])),
        ))
        self.assertEqual(len(slot.lanes), 10)
        topk = dict(zip(slot.selectors + slot.lanes, (1, 0, 7, 6, 5, 1, 1, 0, 0, 1, 0, 0)))
        rmx = dict(zip(slot.selectors + slot.lanes, (0, 1, 7, 6, 1, 1, 1, 0, 7, 8, 3, 11)))
        self.assertFalse(program.failures(topk))
        self.assertFalse(program.failures(rmx))
        duplicate = {**topk, slot.lanes[7]: 0}
        unattained = {**rmx, slot.lanes[6]: 8}
        self.assertTrue(program.failures(duplicate))
        self.assertTrue(program.failures(unattained))
        # The opcode and upstream operands are fixed public data for these fixtures.
        audit(program, topk, fixed=slot.selectors + slot.lanes[:6] + (slot.lanes[8],),
              correlated=(duplicate, {**topk, slot.lanes[6]: 2, slot.lanes[7]: 2}))
        audit(program, rmx, fixed=slot.selectors + slot.lanes[:4] + slot.lanes[7:9],
              correlated=(unattained,))
        self.assertTrue(mutations(program, rmx, targeted=(
            {**rmx, **dict.fromkeys(slot.selectors, 0)},
        ))["slot.exhaustive"])
        program.lookups.pop()
        with self.assertRaises(ValueError):
            static_audit(program)


class BackendTests(unittest.TestCase):
    def test_gkr(self):
        queries, table, counts = (3, 6, 3), (3, 6), (2, 1)
        salt = b"test"
        proof = gkr.prove(queries, table, counts, salt)
        self.assertTrue(gkr.verify(queries, table, counts, proof, salt))
        for bad_table, bad_counts in (((3, 7), counts), (table, (1, 2))):
            bad = gkr.prove(queries, bad_table, bad_counts, salt)
            self.assertFalse(gkr.verify(queries, bad_table, bad_counts, bad, salt))
        self.assertFalse(gkr.verify(queries, table, counts, replace(proof, root=(ZERO, ZERO)), salt))
        layer = proof.layers[1]
        samples = (*layer.rounds[0][:2], hf(123), layer.rounds[0][3])
        changed = replace(layer, rounds=(samples,))
        self.assertFalse(gkr.verify(queries, table, counts, replace(
            proof, layers=(proof.layers[0], changed, *proof.layers[2:]),
        ), salt))

    def test_pcs(self):
        for size in (2, 4, 8):
            polys = (pcs.reserve(tuple(map(hf, range(size-1))), size, [size-1]),
                     tuple(hf(i*i) for i in range(size)))
            point = (7, 6, 5)
            root, claims, proof = pcs.prove(polys, point)
            self.assertEqual(root, pcs.commit(polys))
            self.assertTrue(pcs.verify(root, size, point, claims, proof))
            self.assertFalse(pcs.verify(root, size, point, (hf(1), *claims[1:]), proof))
            chain = proof.openings[0]
            opening = replace(chain[0], row=((chain[0].row[0][0] ^ 1, *chain[0].row[0][1:]),
                                             *chain[0].row[1:]))
            forged = replace(proof, openings=((opening, *chain[1:]), *proof.openings[1:]))
            self.assertFalse(pcs.verify(root, size, point, claims, forged))
            expected = 32 * len(proof.roots) + 24
            expected += 8 * sum(24*len(item.row) + 32*len(item.path) for item in chain)
            self.assertEqual(proof.byte_count, expected)
        pcs.require_joint_mask([[1], [2]], [[3], [6]])
        with self.assertRaisesRegex(ValueError, "under-rank"):
            pcs.require_joint_mask([[1], [2]], [[3], [7]])


if __name__ == "__main__":
    unittest.main()
