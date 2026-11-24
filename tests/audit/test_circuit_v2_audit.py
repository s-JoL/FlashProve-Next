import struct
import unittest
from dataclasses import replace
from fractions import Fraction

from prover.fpn_prover.circuit.blocks import (
    gemm_tile_row_block, gqa_group_attn_block, rmsnorm_block, swiglu_block,
)
from prover.fpn_prover.circuit.microprogram import install_microprogram, swa_indices
from prover.fpn_prover.circuit.protocol import external_seed, fresh_input_stratum
from prover.fpn_prover.ir.arith import bf16_decode, norm_rsqrt
from prover.fpn_prover.ir.format import get_contract
from prover.fpn_prover.ir.gadget_ir import Program
from underconstraint import audit, lookup_mutations, mutations


def bf16(x):
    word = struct.unpack(">I", struct.pack(">f", x))[0]
    return (word + 0x7fff + ((word >> 16) & 1)) >> 16


class CircuitAudit(unittest.TestCase):
    def test_rmsnorm(self):
        p = Program()
        names = rmsnorm_block(p, "norm", 1)
        for value in (1, 1e8):
            word = bf16(value)
            x = bf16_decode(word).value
            ref = norm_rsqrt(x*x + Fraction(1, 10**6))
            row = dict(zip(names, (word, bf16(1), bf16(1), ref.k, ref.low,
                                   ref.high, ref.high-ref.low), strict=True))
            audit(p, row, fixed=names[:3])
            self.assertTrue(all(mutations(p, row, fixed=names[:3]).values()))
            zero = norm_rsqrt(Fraction(1, 10**6))
            candidate = dict(zip(names, (0, bf16(1), 0, zero.k, zero.low,
                                         zero.high, zero.high-zero.low), strict=True))
            self.assertEqual(p.failures(candidate), ["norm.norm"])
            self.assertTrue(lookup_mutations(p, row, [candidate])["norm.norm"])
        # Q/K normalization has no residual non-degeneracy floor.
        p = Program()
        epsilon = Fraction(1, 10**12)
        names = rmsnorm_block(p, "qnorm", 1, epsilon=epsilon, residual=False)
        ref = norm_rsqrt(epsilon)
        self.assertGreater(ref.enclosure[0], 4096)
        row = dict(zip(names, (0, bf16(1), 0, ref.k, ref.low, ref.high,
                               ref.high-ref.low), strict=True))
        audit(p, row, fixed=names[:3])
        self.assertTrue(all(mutations(p, row, fixed=names[:3]).values()))

    def test_gemm_contracts(self):
        for contract_name in ("mxfp4_e2m1", "fp8_e4m3_block128"):
            with self.subTest(contract=contract_name):
                p = Program()
                contract = get_contract(contract_name)
                names = gemm_tile_row_block(p, "gemm", 1, 1, contract)
                z = 8*contract.dot_relative_bound(1).numerator
                row = p.complete(dict(zip(names, (8, 0, z, 3, 3, 24, 3*z, 24, 0), strict=True)))
                fixed = names[:2] + names[3:5] + (names[-2],)
                audit(p, row, fixed=fixed)
                # r_x >= 0, wr >= |w| and (when it keeps the lift bound) the operand
                # z = num |x| + (den + num) r_x are gates.  The operand cells are pinned by the fixed
                # inputs, so their two gates need the operand (and the radius product) to move.
                lifted = "gemm.abs.0" in p.columns
                self.assertEqual(lifted, contract_name == "mxfp4_e2m1")
                targeted = []
                if lifted:
                    num = contract.dot_relative_bound(1).numerator
                    targeted = [
                        {**row, names[2]: z+1, names[6]: 3*(z+1)},
                        {**row, "gemm.abs.0": row["gemm.abs.0"]+1, names[2]: z+num, names[6]: 3*(z+num)},
                    ]
                deleted = mutations(p, row, fixed=names[:2] + names[3:5], targeted=targeted)
                self.assertTrue(all(deleted.values()))
                self.assertLessEqual(
                    {"gemm.input_radius_nonneg.0", "gemm.weight_radius_upper.0.0",
                     "gemm.weight_radius_lower.0.0", "gemm.center_dot.0", "gemm.difference.0"}
                    | ({"gemm.abs_square.0", "gemm.operand.0"} if lifted else set()), deleted.keys())
                # Each lifted inequality rejects its own violation by name (hint-regenerated slack).
                for change, gate in (({names[1]: -1}, "gemm.input_radius_nonneg.0"),
                                     ({names[4]: 2}, "gemm.weight_radius_upper.0.0"),
                                     ({names[3]: -4, names[5]: -32, names[-1]: 56},
                                      "gemm.weight_radius_lower.0.0")):
                    base = {k: v for k, v in row.items() if k not in p.hints}
                    self.assertIn(gate, p.failures({**base, **change}))
                candidates = [
                    {**row, names[2]: z+1, names[6]: 3*(z+1)},
                    {**row, names[6]: 3*z+1},
                ]
                # Alternate observations leave the committed weight and input cells fixed.
                candidates.append({**row, names[-2]: 16, names[-1]: -8})
                candidates.append({**row, names[-2]: 20, names[-1]: -4})
                self.assertTrue(all(lookup_mutations(p, row, candidates).values()))

    def test_swiglu(self):
        p = Program()
        names = swiglu_block(p, "gate", 1, get_contract("bf16_fp32acc"))
        row = dict(zip(names, (bf16(3), bf16(2), bf16(2.859375), bf16(5.71875)), strict=True))
        audit(p, row, fixed=names[:2])
        self.assertEqual(mutations(p, row, fixed=names[:2]), {})
        bad = {**row, names[-1]: bf16(1)}
        self.assertTrue(lookup_mutations(p, row, [bad])["gate.gate.0"])

    def test_gqa(self):
        unit = 1
        for contract_name in ("bf16_fp32acc", "fp16_attn_strict"):
            p = Program()
            names = gqa_group_attn_block(p, "att", 2, 1, 1, 1, get_contract(contract_name), unit=unit)
            row = dict(zip(names, (unit, 0, unit, -unit, 0, 0, unit, -unit), strict=True))
            audit(p, row, fixed=names[:4])
            self.assertEqual(mutations(p, row, fixed=names[:4]), {})
            candidates = [{**row, name: 0} for name in names[-2:]]
            for candidate in candidates:
                self.assertTrue(p.failures(candidate))
            self.assertTrue(all(lookup_mutations(p, row, candidates).values()))

    def test_universal_slot(self):
        p = Program()
        slot = install_microprogram(p, "slot", tokens=2, experts=2, top_k=2,
                                    window=1, sink=1, conv_width=1, extent=1)
        self.assertEqual(swa_indices(2, 3, 2), (0, 1, 2))
        self.assertEqual(swa_indices(4, 2, 1), (0, 3, 4))
        branches = (
            (0, 256, 256, 0, 512, 0, 512),
            (0, 256, 256, 0, 512, 0, 512),
            (0, 0, 1, 1, 1, 0, 256, 256, 0, 0, 0, 1, 0, 512, 0),
            (256, 0, 256, 512, 0, 0, 0, 512),
            (0, 1, 0, 256, 0, 256, 0, 256, 256, 256, 0),
        )
        self.assertEqual(len(slot.lanes), max(slot.widths))
        # Lanes and selectors are the opened cells; every other column is an auxiliary wire of a lifted
        # gate (gated product / slack / one-hot index cell) with a witness-generation hint.
        self.assertEqual(len(p.columns) - len(p.hints), max(slot.widths)+len(branches))
        self.assertEqual(set(p.columns) - set(p.hints), set(slot.selectors + slot.lanes))
        geometry = p.ledger()
        moe, gdn = "slot.moe_router_expert.", "slot.gdn_lrec."
        for i, branch in enumerate(branches):
            with self.subTest(branch=i):
                values = branch + (0,)*(len(slot.lanes)-len(branch))
                opened = dict(zip(slot.selectors+slot.lanes,
                                  tuple(int(i == j) for j in range(len(branches)))+values,
                                  strict=True))
                row = p.complete(opened)
                # Opened branch operands and selected opcode are held fixed.
                fixed = slot.selectors + slot.lanes[:len(branch)-1]
                audit(p, row, fixed=fixed)
                deleted = mutations(p, row, fixed=slot.lanes)
                self.assertIsNotNone(deleted["slot.exhaustive"])
                peer = Program()
                install_microprogram(peer, "slot", tokens=2, experts=2, top_k=2,
                                     window=1, sink=1, conv_width=1, extent=1)
                self.assertFalse(peer.failures(row))
                self.assertEqual(peer.ledger(), geometry)
                forged = [({**row, slot.lanes[len(branch)-1]: branch[-1]+1}, [slot.branch_names[i]])]
                # these forgeries used to be caught only by the prover-side branch predicate;
                # they are now rejected by named relation-4 gates (auxiliary cells regenerated by hint).
                if i == 2:
                    forged += [
                        (p.complete({**opened, slot.lanes[11]: 0}),          # index list (0, 0)
                         [moe + "index_order.1", moe + "index_column.0", moe + "index_column.1"]),
                        (p.complete({**opened, slot.lanes[12]: 1}),          # maximum above every score
                         [moe + "argmax_attains.0", slot.branch_names[i]]),
                        (p.complete({**opened, slot.lanes[2]: 0, slot.lanes[10]: 1, slot.lanes[11]: 1}),
                         [moe + "index_order.1", moe + "index_column.1"]),  # dropped expert
                    ]
                if i == 4:
                    forged.append((p.complete({**opened, slot.lanes[1]: 0, slot.lanes[9]: 0}),
                                   [gdn + "in_range_compare"]))
                for candidate, expected in forged:
                    self.assertEqual(p.failures(candidate), expected)
                self.assertTrue(lookup_mutations(p, row, [c for c, _ in forged])[slot.branch_names[i]])

        empty = dict.fromkeys(slot.selectors+slot.lanes, 0)
        empty[slot.selectors[0]] = 1
        selector_mutants = []
        compatible = (0, 1, 3)
        for i, selector in enumerate(slot.selectors):
            left, right = [j for j in compatible if j != i][:2]
            selector_mutants.append(p.complete({
                **dict.fromkeys(empty, 0), selector: -1,
                slot.selectors[left]: 1, slot.selectors[right]: 1,
            }))
        deleted = mutations(p, empty, fixed=slot.lanes, targeted=selector_mutants)
        # The selector gates (all the non-`.unused` equalities).  The lifted branch gates
        # are selector-gated, so their deletion mutants need live lanes: every one of them is deleted
        # row by row on the lowered relation-4 table in tests/audit/test_lowered_air_mutation_audit.py.
        structural = {selector + ".boolean" for selector in slot.selectors} | {"slot.exhaustive"}
        self.assertLessEqual(structural, deleted.keys())
        # Two selector equalities became IMPLIED by lifted gates over the 2-bit selector range [-2, 1],
        # so no row can violate them alone:
        #   * MoE: `index_row.r` equates the selector to a sum of `index_onehot_boolean` cells (>= 0);
        #   * GDN: `position_low` / `position_high` need s p >= 0 and s (tokens - 1 - p) >= 0, which
        #     has no solution for s < 0 once tokens >= 2.
        implied = {"slot.moe_router_expert.boolean": [moe + "argmax_one", moe + "index_row.0",
                                                      moe + "index_row.1"],
                   "slot.gdn_lrec.boolean": [gdn + "position_high"]}
        self.assertEqual({name for name in structural if not deleted[name]}, set(implied))
        for index, (name, also) in zip((2, 4), implied.items(), strict=True):
            self.assertEqual(selector_mutants[index][name.removesuffix(".boolean")], -1)
            self.assertEqual(p.failures(selector_mutants[index]), [name] + also)
        self.assertFalse(any(-pos >= 0 and -(2 - 1 - pos) >= 0 for pos in range(-64, 64)))
        # the unused-lane guard s_k * lane_i = 0 (i >= width_k) is an algebraic gate (lowered to a
        # relation-4 Plonkish row) and the branch predicate no longer re-checks it, so each gate is necessary:
        # a nonzero unused lane of the selected branch is rejected by exactly that gate.
        unused = [c.name for c in p.constraints if ".unused" in c.name]
        self.assertEqual(len(unused), sum(len(slot.lanes) - w for w in slot.widths))
        found = set()
        for i, branch in enumerate(branches):
            row = p.complete(dict(zip(slot.selectors + slot.lanes,
                                      tuple(int(i == j) for j in range(len(branches)))
                                      + branch + (0,) * (len(slot.lanes) - len(branch)), strict=True)))
            for lane in range(slot.widths[i], len(slot.lanes)):
                name = f"{slot.selectors[i]}.unused{lane}"
                mutant = {**row, slot.lanes[lane]: 1}
                self.assertEqual(p.failures(mutant), [name])
                weakened = replace(p, constraints=[c for c in p.constraints if c.name != name])
                self.assertFalse(weakened.failures(mutant))
                found.add(name)
        self.assertEqual(found, set(unused))


    def test_external_seed_and_fresh_input(self):
        self.assertNotEqual(external_seed(b"x", b"r", b"pre", b"c"),
                            external_seed(b"x", b"r", b"other", b"c"))
        self.assertEqual(list(fresh_input_stratum(5, 3)), [3, 4])
