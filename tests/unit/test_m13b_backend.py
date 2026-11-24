# SPDX-License-Identifier: Apache-2.0
"""M1-3b backend: ZK-masked FracLogUp-GKR, fused two-round folded ZK-Basefold, relation-3 prover,
CUDA kernel parity (host emulation), and the byte budget.

    python3.14 -W error -m unittest discover -s prover/fpn_prover/backend/tests
"""
import os
import random
import shutil
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("FPN_NO_NATIVE", "1")
for p in (ROOT, os.path.join(ROOT, "verifier", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from flashprove_next.field import ONE, P, ZERO, hadd, hmul  # noqa: E402
from prover.fpn_prover.backend import cuda_engine as CE  # noqa: E402
from prover.fpn_prover.backend import folded_pcs as pcs  # noqa: E402
from prover.fpn_prover.backend import frac_logup_gkr as gkr  # noqa: E402
from prover.fpn_prover.backend import fpn2 as BF  # noqa: E402
from prover.fpn_prover.backend import zk_lookup as ZL  # noqa: E402
from fpn_verify_ref.verify import verify_result as ref_verify  # noqa: E402


class SeededRng:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return tuple(self.r.randrange(P) for _ in range(3))


N8 = BF.FOLDED_GKR_N8
L8 = BF.Layout(8, 2)


def prove8(queries=None, seed=1, **kw):
    queries = queries if queries is not None else [i % L8.quarter for i in range(L8.quarter)]
    return ZL.prove(queries, profile=N8, layout=L8, rng=SeededRng(seed), **kw)


def both(c):
    return ZL.verify_result(c, profile=N8, layout=L8), ref_verify(c, True)


class TestMaskedGkr(unittest.TestCase):
    def _instance(self, seed=3, s=4):
        rng = SeededRng(seed)
        leaves = []
        for j in range(1 << s):
            a, d = rng.fp3(), rng.fp3()
            leaves += [(a, d), (hmul(a, (P - 1, 0, 0)), d)]        # cancelling pairs: root_num = 0
        coeffs = {(k, i): tuple(rng.fp3() for _ in range(4)) for k in range(1, s + 2) for i in range(k)}
        return gkr.build_tree(leaves), (lambda k, i: coeffs[(k, i)])

    def _run(self, tree, libra, hooks=None, tamper=None):
        tp, tv = BF.Fpn2Transcript(), BF.Fpn2Transcript()
        root = tree[-1][0]
        self.assertEqual(root[0], ZERO)
        tp.absorb_bytes(BF.enc3(root[0]) + BF.enc3(root[1]))
        layers, point, claim, evals = gkr.masked_prove(tp, tree, libra, hooks)
        if tamper:
            root, layers = tamper(root, list(layers))
        tv.absorb_bytes(BF.enc3(root[0]) + BF.enc3(root[1]))
        return (point, claim, evals), gkr.masked_verify(tv, root, layers)

    def test_honest_round_trip_and_libra_algebra(self):
        tree, libra = self._instance()
        (point, claim, evals), (vp, vc, ve) = self._run(tree, libra)
        self.assertEqual((point, claim, evals), (vp, vc, ve))
        g = [libra(3, i) for i in range(3)]
        self.assertEqual(hadd(gkr.libra_round(g, 0, [], ZERO), gkr.libra_round(g, 0, [], ONE)), gkr.libra_sum(g))

    def test_forgeries_rejected(self):
        tree, libra = self._instance()
        cases = {
            "root_den_zero": lambda r, ls: ((r[0], ZERO), ls),
            "root_num_nonzero": lambda r, ls: ((ONE, r[1]), ls),
            "mask_sum_tamper": lambda r, ls: (r, ls[:2] + [gkr.MaskedLayer(hadd(ls[2].g_sum, ONE), ls[2].rounds,
                                                                             ls[2].children, ls[2].g_r)] + ls[3:]),
            "mask_eval_tamper": lambda r, ls: (r, ls[:-1] + [gkr.MaskedLayer(ls[-1].g_sum, ls[-1].rounds,
                                                                            ls[-1].children, hadd(ls[-1].g_r, ONE))]),
            "round_tamper": lambda r, ls: (r, ls[:3] + [gkr.MaskedLayer(
                ls[3].g_sum, (ls[3].rounds[0][:3] + (hadd(ls[3].rounds[0][3], ONE),),) + ls[3].rounds[1:],
                ls[3].children, ls[3].g_r)] + ls[4:]),
        }
        for name, tamper in cases.items():
            with self.subTest(name):
                with self.assertRaises(gkr.GkrReject) as cm:
                    self._run(tree, libra, tamper=tamper)
                self.assertEqual(cm.exception.code, "E_GKR_ROOT" if name.startswith("root") else "E_GKR")

    def test_adaptive_mask_passes_gkr_alone(self):
        """Adaptively solving g_r makes a forged leaf claim pass every GKR check: the PCS must bind g_r."""
        tree, libra = self._instance()
        s = len(tree) - 2
        hooks = {"children": lambda k, ch: (hadd(ch[0], ONE),) + ch[1:] if k == s else ch, "adaptive_g_r": True}
        honest, _ = self._run(tree, libra)
        _, (vp, vc, ve) = self._run(tree, libra, hooks=hooks)
        self.assertNotEqual(vc, honest[1])         # GKR accepted a wrong leaf claim ...
        self.assertNotEqual(ve[-1][2], honest[2][-1][2])   # ... with a g_r the commitment will refute


class TestRelation3EndToEnd(unittest.TestCase):
    def test_honest(self):
        self.assertEqual(both(prove8()), ("VERIFIED", "VERIFIED"))
        self.assertEqual(both(prove8([L8.quarter - 1] * L8.quarter, seed=2)), ("VERIFIED", "VERIFIED"))

    def test_negative_forgeries(self):
        bad = [i % L8.quarter for i in range(L8.quarter)]
        bad[0] = L8.quarter

        def wrong_m(layout, queries, rng):
            cells = ZL.build_witness(layout, queries, rng)
            i = layout.cell(layout.off_m + 2)
            cells[i] = hadd(cells[i], ONE)
            return cells

        def zero_den(layout, queries, rng):
            cells = ZL.build_witness(layout, queries, rng)
            cells[layout.cell(layout.off_e1 + layout.quarter // 2)] = ZERO
            return cells

        s = L8.s
        last = lambda k, ch: (hadd(ch[0], ONE),) + ch[1:] if k == s else ch  # noqa: E731
        cases = {
            "missing_table_entry": (dict(queries=bad), "E_GKR_ROOT"),
            "missing_table_entry_forged_root": (dict(queries=bad, hooks={"root": lambda r: (ZERO, r[1])}), "E_GKR"),
            "wrong_multiplicity": (dict(witness=wrong_m), "E_GKR_ROOT"),
            "forged_root_den_zero": (dict(witness=zero_den, hooks={"allow_zero_root": True}), "E_GKR_ROOT"),
            "adaptive_mask_tamper": (dict(hooks={"gkr": {"children": last, "adaptive_g_r": True}}), "E_SUMCHECK"),
            "fold_layer_tamper": (dict(hooks={"pcs": {"code": lambda t, c: [hadd(x, ONE) for x in c] if t == 1 else c}}),
                                  "E_FOLD"),
            "pcs_round_tamper": (dict(hooks={"pcs": {"round": lambda i, h: (hadd(h[0], ONE),) + h[1:] if i == 3 else h}}),
                                 "E_SUMCHECK"),
        }
        for name, (kw, code) in cases.items():
            with self.subTest(name):
                self.assertEqual(both(prove8(**kw)), (code, code))

    def test_last_layer_children_do_not_reveal_multiplicities(self):
        """Regression: with zero-numerator dummies, n_l at the leaf layer equalled eqQ0 - eqQ2 * m~(low)."""
        from flashprove_next.field import hinv, hsub
        rec = {}
        c = prove8(hooks={"record": rec})
        stmt, root, layers, _, _ = ZL._decode(L8, c[64:64 + L8.body_used])
        tr = BF.Fpn2Transcript()
        tr.absorb_bytes(c[:64])
        tr.absorb_bytes(stmt)
        tr.squeeze_fp3()
        tr.absorb_bytes(BF.enc3(root[0]) + BF.enc3(root[1]))
        point, _, _ = gkr.masked_verify(tr, root, layers)
        s, rho = L8.s, point[1:]
        eqq = [ZL.eq_pt((rho[s - 2], rho[s - 1]), ZL.bits_point(qd, 2)) for qd in range(4)]
        extracted = hmul(hsub(eqq[0], layers[-1].children[0]), hinv(eqq[2]))
        m_tilde = ZERO
        for t, e in enumerate(ZL.eq_vec(rho[:s - 2])):
            m_tilde = hadd(m_tilde, hmul(e, rec["cells"][L8.cell(L8.off_m + t)]))
        self.assertNotEqual(extracted, m_tilde)

    def test_wire_is_fixed_size_and_raw(self):
        c = prove8()
        self.assertEqual(len(c), N8["c_p"])
        self.assertFalse(any(c[64 + L8.body_used:]))
        self.assertEqual(c[:64], BF.header(BF.profile_id(N8)))
        b = bytearray(c)
        b[64 + 64 + 24 + 8] ^= 0xFF                 # inside root_den (raw bytes are transcript input)
        self.assertNotEqual(both(bytes(b))[0], "VERIFIED")

    def test_profile_bytes_match_reference(self):
        from fpn_verify_ref import profile as PR
        self.assertEqual(BF.profile_bytes(BF.FOLDED_GKR_V1), PR.FOLDED_GKR_V1.canonical_bytes())
        self.assertEqual(BF.profile_bytes(N8), PR.FOLDED_GKR_N8.canonical_bytes())
        lay = BF.Layout(12, 72)
        self.assertEqual(lay.body_used + 64, 159024)
        self.assertGreaterEqual(lay.blind_len, lay.observations)


class TestFoldedZkPcs(unittest.TestCase):
    def test_open_verify_and_tamper(self):
        rng = SeededRng(9)
        n = 6
        cells = [rng.fp3() for _ in range(1 << n)]
        w = [rng.fp3() for _ in range(1 << n)]
        com = pcs.zk_commit(cells)
        claim = ZERO
        for f, x in zip(cells, w):
            claim = hadd(claim, hmul(f, x))
        tp, tv = BF.Fpn2Transcript(), BF.Fpn2Transcript()
        opening, rs, _ = pcs.zk_open(com, w, tp, 3)
        w_eval = lambda r: sum_eval(w, r)  # noqa: E731
        self.assertEqual(len(pcs.zk_verify(com.root, n, claim, w_eval, opening, tv, 3)), n)
        with self.assertRaises(pcs.PcsReject) as cm:
            pcs.zk_verify(com.root, n, hadd(claim, ONE), w_eval, opening, BF.Fpn2Transcript(), 3)
        self.assertEqual(cm.exception.code, "E_SUMCHECK")

    def test_closed_form_observations_match_brute_force(self):
        rec = {}
        prove8(hooks={"record": rec})
        rows = pcs.observation_rows(L8.n, rec["weights"], rec["rs"], rec["positions"])
        for col in random.Random(4).sample(range(1 << L8.n), 4):
            e = [ZERO] * (1 << L8.n)
            e[col] = ONE
            self.assertEqual(pcs.observations(e, rec["weights"], rec["rs"], rec["positions"]),
                             [r[col] for r in rows])

    def _rank_split(self, layout, rec):
        size = 1 << layout.n
        libra = {layout.libra_slot(k, i, e) for k in range(1, layout.s + 1) for i in range(k) for e in range(4)}
        secret = [i for i in range(size) if layout.is_data(i) or i in libra]
        blind = [i for i in range(size) if not (layout.is_data(i) or i in libra)]
        rows = pcs.observation_rows(layout.n, rec["weights"], rec["rs"], rec["positions"])
        return rows, secret, blind

    def test_reserved_mask_rank_condition(self):
        """Explicit A(x - x') in im B check of the reserved non-doubling blinding cells."""
        for n, q in ((8, 2), (10, 4)):
            with self.subTest(n=n, q=q):
                lay = BF.Layout(n, q)
                prof = dict(N8, n_vars=n, n_queries=q, c_p=64 + lay.body_used)
                rec = {}
                ZL.prove([0] * lay.quarter, profile=prof, layout=lay, rng=SeededRng(n), hooks={"record": rec})
                rows, secret, blind = self._rank_split(lay, rec)
                pcs.require_joint_mask_ext(pcs.split_columns(rows, secret), pcs.split_columns(rows, blind))

    def test_rank_check_catches_clustered_mask(self):
        """Blinding confined to one low index range cannot reach the upper monomial classes."""
        rec = {}
        prove8(hooks={"record": rec})
        rows, secret, blind = self._rank_split(L8, rec)
        cluster = blind[:48]
        others = [i for i in range(1 << L8.n) if i not in cluster]
        with self.assertRaisesRegex(ValueError, "under-rank"):
            pcs.require_joint_mask_ext(pcs.split_columns(rows, others), pcs.split_columns(rows, cluster))

    def test_byte_budget(self):
        b = pcs.byte_budget(300_000_000, cohorts=2, queries=256)
        self.assertTrue(b["fits_8mib"])
        self.assertLessEqual(b["total"], 8 * pcs.MIB)
        self.assertEqual(b["total"], sum(b["parts"].values()))
        self.assertEqual((b["n_vars"], b["codeword"], b["folds"]), (28, 1 << 31, 14))
        self.assertEqual(b["total"], 2457144)
        self.assertGreater(pcs.byte_budget(2 ** 31, queries=256)["total"], b["total"])


def sum_eval(w, r):
    acc = ZERO
    for b, e in enumerate(ZL.eq_vec(r)):
        acc = hadd(acc, hmul(w[b], e))
    return acc


@unittest.skipUnless(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"), "needs a C++ compiler")
class TestKernelParityHostEmulation(unittest.TestCase):
    """goldilocks_ntt.cu compiled for the CPU must be bit-exact with the Python reference."""

    @classmethod
    def setUpClass(cls):
        cls.eng = CE.HostEngine()
        cls.rng = SeededRng(77)

    def rand(self, n):
        return CE.to_array([self.rng.fp3() for _ in range(n)])

    def test_fold4(self):
        code = self.rand(64)
        ra, rb = self.rng.fp3(), self.rng.fp3()
        omega = CE.root_of_unity(64)
        got = CE.to_tuples(self.eng.fold4(code, pow(omega, P - 2, P), pow(CE.root_of_unity(4), P - 2, P), ra, rb))
        n = 3                                          # 64 = RATE * 2^3, layer t = 0
        self.assertEqual(got, pcs.fold_codeword(CE.to_tuples(code), 0, n, ra, rb))

    def test_merkle(self):
        code = self.rand(32)
        leaves = self.eng.merkle_leaves(code)
        tup = CE.to_tuples(code)
        self.assertEqual([bytes(x) for x in leaves], pcs.coset_leaves(tup))
        nodes = self.eng.merkle_layer(leaves)
        self.assertEqual([bytes(x) for x in nodes], [BF.node_hash(bytes(leaves[2 * i]), bytes(leaves[2 * i + 1]))
                                                     for i in range(4)])

    def test_ntt_and_mobius(self):
        rows = np.array([[self.rng.r.randrange(P) for _ in range(256)] for _ in range(3)], dtype=np.uint64)
        got = self.eng.ntt_rows(rows)
        for k in range(3):
            self.assertEqual([int(x) for x in got[k]], pcs._ntt_limb([int(x) for x in rows[k]], CE.root_of_unity(256)))
        msg = [self.rng.r.randrange(P) for _ in range(5000)]
        self.assertEqual([int(x) for x in self.eng.ntt8192(np.array(msg, dtype=np.uint64))],
                         pcs._ntt_limb(msg + [0] * (8192 - len(msg)), CE.root_of_unity(8192)))
        cells = self.rand(64)
        self.assertEqual(CE.to_tuples(self.eng.mobius(cells)), pcs.mobius(CE.to_tuples(cells)))

    def test_fraction_layer_and_sumcheck(self):
        nodes = self.rand(32).reshape(16, 6)
        got = self.eng.fraction_layer(nodes)
        for i in range(8):
            a, b = CE.to_tuples(nodes[2 * i]), CE.to_tuples(nodes[2 * i + 1])
            self.assertEqual(CE.to_tuples(got[i]), list(gkr.combine(tuple(a), tuple(b))))
        f, w, r = self.rand(32), self.rand(32), self.rng.fp3()
        sums, halved = self.eng.basefold_round(f, w, r)
        self.assertEqual(tuple(CE.to_tuples(sums)), pcs._sc_round(CE.to_tuples(f), CE.to_tuples(w)))
        self.assertEqual(CE.to_tuples(halved), pcs._halve(CE.to_tuples(f), r))

    def test_fraction_layer_multi_tile_bit_exact(self):
        """Tiled fraction_layer (shared-memory staged, 16-byte loads): several full FRAC_TILE tiles plus a
        ragged last tile, and a whole GKR tree, bit-exact with frac_logup_gkr.build_tree."""
        pairs = 2 * CE.FRAC_TILE + 37
        nodes = self.rand(4 * pairs).reshape(2 * pairs, 6)
        fp3s = CE.to_tuples(nodes)                    # node i = (fp3s[2i], fp3s[2i + 1]) = (num, den)
        got = CE.to_tuples(self.eng.fraction_layer(nodes))
        self.assertEqual(len(got), 2 * pairs)
        for i in range(pairs):
            n, d = gkr.combine((fp3s[4 * i], fp3s[4 * i + 1]), (fp3s[4 * i + 2], fp3s[4 * i + 3]))
            self.assertEqual((got[2 * i], got[2 * i + 1]), (n, d), i)
        leaves = [(self.rng.fp3(), self.rng.fp3()) for _ in range(1024)]
        tree = CE.gkr_tree(self.eng, np.array([a + b for a, b in leaves], dtype=np.uint64))
        ref = gkr.build_tree(leaves)
        self.assertEqual(len(tree), len(ref))
        for level, ref_level in zip(tree, ref):
            self.assertEqual(CE.to_tuples(level), [x for node in ref_level for x in node])

    def test_full_commit_and_fold_chain(self):
        """GPU pipeline (emulated) reproduces the prover's commitment, folded roots and final value."""
        n = 6
        cells = [self.rng.fp3() for _ in range(1 << n)]
        w = [self.rng.fp3() for _ in range(1 << n)]
        com = pcs.zk_commit(cells)
        code, levels = CE.commit_pipeline(self.eng, CE.to_array(cells))
        self.assertEqual(CE.to_tuples(code), list(com.code))
        self.assertEqual(bytes(levels[-1][0]), com.root)
        opening, rs, _ = pcs.zk_open(com, w, BF.Fpn2Transcript(), 2)
        layers, trees = CE.fold_chain(self.eng, code, n, rs)
        self.assertEqual([bytes(t[-1][0]) for t in trees], list(opening.roots))
        self.assertTrue(all(v == opening.final for v in CE.to_tuples(layers[-1])))
        leaves = [(self.rng.fp3(), self.rng.fp3()) for _ in range(16)]
        tree = CE.gkr_tree(self.eng, np.array([a + b for a, b in leaves], dtype=np.uint64))
        ref = gkr.build_tree(leaves)
        self.assertEqual(CE.to_tuples(tree[-1][0]), [ref[-1][0][0], ref[-1][0][1]])


if __name__ == "__main__":
    unittest.main()
