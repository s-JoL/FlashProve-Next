# SPDX-License-Identifier: Apache-2.0
"""Tests for `multi_pcs` (multi-group batched coset-4 ZK-Basefold) and `multi_gkr` (multi-table FracLogUp-GKR)."""
from __future__ import annotations

import unittest
import numpy as np

from flashprove_next.field import ONE, P, ZERO, hadd, hf, hmul, hsub
from prover.fpn_prover.backend import multi_gkr as MG
from prover.fpn_prover.backend import multi_pcs as MP
from prover.fpn_prover.backend.fpn2 import Fpn2Transcript, enc3


def _dot_fp3(a_np, b_np):
    acc = ZERO
    for i in range(len(a_np)):
        u = (int(a_np[i, 0]), int(a_np[i, 1]), int(a_np[i, 2]))
        v = (int(b_np[i, 0]), int(b_np[i, 1]), int(b_np[i, 2]))
        acc = hadd(acc, hmul(u, v))
    return acc


def _mle_eval_host(arr_np, pt):
    eq = MP._eq_table_fp3(pt)
    return _dot_fp3(arr_np, eq)


class TestMultiPcs(unittest.TestCase):
    def _build_instance(self, n=8, n_groups=3, queries=4, seed=1234):
        layout = MP.MultiPcsLayout(n=n, n_groups=n_groups, queries=queries)
        rng = np.random.default_rng(seed)
        N = layout.group_len
        cap = layout.data_capacity
        groups = []
        weights = []
        claims = []
        w_pts = []
        for g in range(n_groups):
            if g == 0:
                # Base-field data vector
                data = rng.integers(0, 10000, size=cap, dtype=np.uint64)
            else:
                # Extension-field data vector
                data = rng.integers(0, P, size=(cap, 3), dtype=np.uint64)
            grp = MP.commit_data_group(layout, data, rng=rng)
            # Distinct evaluation point per group so W_g = eq(pt_g, .) differs across groups!
            pt_g = [tuple(int(x) for x in rng.integers(0, P, size=3, dtype=np.uint64)) for _ in range(n)]
            w_g = MP._eq_table_fp3(pt_g)
            # Zero weight on blinding tail is typical for circuit data, or full-vector weight also works
            w_g[cap:] = 0
            y_g = _dot_fp3(grp.cells, w_g)
            groups.append(grp)
            weights.append(w_g)
            claims.append(y_g)
            w_pts.append(pt_g)
        return layout, groups, weights, claims, rng

    def test_multi_group_honest_and_exact_size(self):
        layout, groups, weights, claims, rng = self._build_instance(n=8, n_groups=3, queries=4, seed=7)
        tr_p = Fpn2Transcript(b"test/multi_pcs")
        proof = MP.prove_multi_open(layout, groups, weights, claims, tr_p, rng=rng)
        self.assertEqual(len(proof), layout.proof_bytes)

        tr_v = Fpn2Transcript(b"test/multi_pcs")
        w_eval_fn = lambda g, r_ip: _mle_eval_host(weights[g], r_ip)
        res = MP.verify_multi_open(layout, [g.root for g in groups], claims, w_eval_fn, proof, tr_v)
        self.assertEqual(res["proof_bytes"], layout.proof_bytes)
        self.assertEqual(tr_p.state, tr_v.state)

    def test_single_group_honest(self):
        layout, groups, weights, claims, rng = self._build_instance(n=8, n_groups=1, queries=3, seed=99)
        tr_p = Fpn2Transcript(b"test/multi_pcs/1")
        proof = MP.prove_multi_open(layout, groups, weights, claims, tr_p, rng=rng)
        tr_v = Fpn2Transcript(b"test/multi_pcs/1")
        res = MP.verify_multi_open(
            layout, [groups[0].root], claims, lambda g, r: _mle_eval_host(weights[g], r), proof, tr_v
        )
        self.assertEqual(res["proof_bytes"], layout.proof_bytes)

    def test_reject_false_claim_and_tamper(self):
        layout, groups, weights, claims, rng = self._build_instance(n=8, n_groups=2, queries=3, seed=42)
        tr_p = Fpn2Transcript(b"test/multi_pcs/bad")
        proof = MP.prove_multi_open(layout, groups, weights, claims, tr_p, rng=rng)
        w_eval_fn = lambda g, r_ip: _mle_eval_host(weights[g], r_ip)
        roots = [g.root for g in groups]

        # 1. Wrong claim
        bad_claims = list(claims)
        bad_claims[1] = hadd(bad_claims[1], ONE)
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, roots, bad_claims, w_eval_fn, proof, Fpn2Transcript(b"test/multi_pcs/bad"))
        self.assertEqual(cm.exception.code, "E_SUMCHECK")

        # 2. Wrong group root changes Fiat-Shamir (mu, lam) immediately -> E_SUMCHECK
        bad_roots = list(roots)
        bad_roots[0] = bytes([roots[0][0] ^ 1]) + roots[0][1:]
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, bad_roots, claims, w_eval_fn, proof, Fpn2Transcript(b"test/multi_pcs/bad"))
        self.assertEqual(cm.exception.code, "E_SUMCHECK")

        # 3. Wrong length
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, roots, claims, w_eval_fn, proof[:-1], Fpn2Transcript(b"test/multi_pcs/bad"))
        self.assertEqual(cm.exception.code, "E_LENGTH")

        # 4. Noncanonical limb in g_sum (offset 64 = root_B + root_L)
        bad_nc = bytearray(proof)
        bad_nc[64:72] = P.to_bytes(8, "little")
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, roots, claims, w_eval_fn, bytes(bad_nc), Fpn2Transcript(b"test/multi_pcs/bad"))
        self.assertEqual(cm.exception.code, "E_NONCANONICAL")

        # 5. Tamper FINAL (offset 64 + step_a_bytes + libra_open_bytes + step_b_header_bytes - 24)
        fin_off = 64 + layout.step_a_bytes + layout.libra_open_bytes + layout.step_b_header_bytes - 24
        bad_fin = bytearray(proof)
        bad_fin[fin_off] ^= 1
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, roots, claims, w_eval_fn, bytes(bad_fin), Fpn2Transcript(b"test/multi_pcs/bad"))
        self.assertEqual(cm.exception.code, "E_FINAL")

        # 6. Tamper query coset byte -> E_MERKLE
        q_off = 64 + layout.step_a_bytes + layout.libra_open_bytes + layout.step_b_header_bytes
        bad_q = bytearray(proof)
        bad_q[q_off] ^= 1
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, roots, claims, w_eval_fn, bytes(bad_q), Fpn2Transcript(b"test/multi_pcs/bad"))
        self.assertEqual(cm.exception.code, "E_MERKLE")

    def test_mixed_size_layer_injection(self):
        """Groups of different even sizes (n=12, 10, 8) opened in ONE fold chain via layer injection."""
        gns = (12, 10, 8)
        layout = MP.MultiPcsLayout(n=12, n_groups=3, queries=4, group_ns=gns)
        rng = np.random.default_rng(2026)
        groups, weights, claims, pts = [], [], [], []
        for ng in gns:
            cap = MP.group_data_capacity(ng, layout.queries)
            data = rng.integers(0, 5000, size=cap, dtype=np.uint64)
            grp = MP.commit_data_group(ng, data, queries=layout.queries, rng=rng)
            pt = [tuple(int(x) for x in rng.integers(0, P, size=3, dtype=np.uint64)) for _ in range(ng)]
            w = MP._eq_table_fp3(pt)
            w[cap:] = 0
            claims.append(_dot_fp3(grp.cells, w))
            groups.append(grp)
            weights.append(w)
            pts.append(pt)

        tr_p = Fpn2Transcript(b"test/mixed")
        proof = MP.prove_multi_open(layout, groups, weights, claims, tr_p, rng=rng)
        self.assertEqual(len(proof), layout.proof_bytes)

        tr_v = Fpn2Transcript(b"test/mixed")
        w_eval_fn = lambda g, r_ip_g: _mle_eval_host(weights[g], r_ip_g)
        MP.verify_multi_open(layout, [g.root for g in groups], claims, w_eval_fn, proof, tr_v)
        self.assertEqual(tr_p.state, tr_v.state)

        # False claim on the smallest (n=8) injected group MUST reject
        bad_c = list(claims)
        bad_c[2] = hadd(bad_c[2], ONE)
        with self.assertRaises(MP.MultiPcsError) as cm:
            MP.verify_multi_open(layout, [g.root for g in groups], bad_c, w_eval_fn, proof, Fpn2Transcript(b"test/mixed"))
        self.assertEqual(cm.exception.code, "E_SUMCHECK")

    def test_reproving_zero_shared_64b_blocks_os_csprng(self):
        layout = MP.MultiPcsLayout(n=8, n_groups=2, queries=3)
        data = np.arange(layout.data_capacity, dtype=np.uint64)
        g0_a = MP.commit_data_group(layout, data)
        g1_a = MP.commit_data_group(layout, data * 2)
        g0_b = MP.commit_data_group(layout, data)
        g1_b = MP.commit_data_group(layout, data * 2)
        w = [np.zeros((layout.group_len, 3), dtype=np.uint64) for _ in range(2)]
        w[0][0, 0] = 1
        w[1][1, 0] = 1
        claims = [hf(0), hf(2)]
        p1 = MP.prove_multi_open(layout, [g0_a, g1_a], w, claims, Fpn2Transcript(b"zk"))
        p2 = MP.prove_multi_open(layout, [g0_b, g1_b], w, claims, Fpn2Transcript(b"zk"))
        self.assertEqual(len(p1), len(p2))
        b1 = {p1[i:i + 64] for i in range(0, len(p1) - 63, 64)}
        b2 = {p2[i:i + 64] for i in range(0, len(p2) - 63, 64)}
        self.assertEqual(len(b1 & b2), 0)


class TestMultiGkr(unittest.TestCase):
    def _build_multi_table_leaves(self, beta, gamma, cheat_query=False, cheat_mult=False):
        """Pack TWO distinct tables + weighted queries into one leaf array:
          Table 0 (range [0, 8)): 10 unweighted queries
          Table 1 (2-column pairs (k, k^2) for k in 0..3): 6 weighted queries (Fp3 weights like HCO!)
        """
        def fp(tid, cols):
            acc = hf(tid)
            gp = gamma
            for c in cols:
                acc = hadd(acc, hmul(gp, hf(c)))
                gp = hmul(gp, gamma)
            return acc

        nums, dens = [], []
        # Table 0 queries
        q0 = [0, 3, 7, 2, 3, 5, 1, 7, 0, 4]
        if cheat_query:
            q0[3] = 9  # out of table!
        m0 = [q0.count(t) for t in range(8)]
        if cheat_mult:
            m0[0] += 1
        for x in q0:
            nums.append(ONE)
            dens.append(hsub(beta, fp(0, [x])))
        for t in range(8):
            nums.append(hsub(ZERO, hf(m0[t])))
            dens.append(hsub(beta, fp(0, [t])))

        # Table 1 weighted queries (2 columns)
        q1_idx = [0, 2, 1, 2, 3, 0]
        q1_w = [(i + 1, i + 2, i + 3) for i in range(len(q1_idx))]
        m1 = [ZERO] * 4
        for idx, w in zip(q1_idx, q1_w):
            nums.append(w)
            dens.append(hsub(beta, fp(1, [idx, idx * idx])))
            m1[idx] = hadd(m1[idx], w)
        for t in range(4):
            nums.append(hsub(ZERO, m1[t]))
            dens.append(hsub(beta, fp(1, [t, t * t])))

        return np.array(nums, dtype=np.uint64), np.array(dens, dtype=np.uint64)

    def test_multi_table_gkr_honest_and_leaf_mle_match(self):
        layout = MG.MultiGkrLayout(s=6, m_mask=4)
        rng = np.random.default_rng(2026)
        libra_coeffs, mask_a, mask_d = MG.sample_gkr_masks(layout, rng=rng)
        tr_p = Fpn2Transcript(b"test/multi_gkr")
        beta = tr_p.squeeze_fp3()
        gamma = tr_p.squeeze_fp3()
        real_num, real_den = self._build_multi_table_leaves(beta, gamma)

        res = MG.prove_multi_gkr(
            layout, real_num, real_den, tr_p,
            libra_coeffs=libra_coeffs, mask_a=mask_a, mask_d=mask_d,
        )
        self.assertEqual(len(res.proof), layout.proof_bytes)

        tr_v = Fpn2Transcript(b"test/multi_gkr")
        self.assertEqual(tr_v.squeeze_fp3(), beta)
        self.assertEqual(tr_v.squeeze_fp3(), gamma)
        ver = MG.verify_multi_gkr(layout, res.proof, tr_v)
        self.assertEqual(tr_p.state, tr_v.state)
        self.assertEqual(ver.r_gkr, res.r_gkr)
        self.assertEqual(ver.num_eval, res.num_eval)
        self.assertEqual(ver.den_eval, res.den_eval)

        # Verify leaf MLE decomposition: full_leaf~(r_gkr) == real_part~(r_gkr) + mask_leaf_contributions
        full_leaves = MG.pack_real_leaves(layout, real_num, real_den, mask_a, mask_d)
        self.assertEqual(_mle_eval_host(full_leaves[:, :3], ver.r_gkr), ver.num_eval)
        self.assertEqual(_mle_eval_host(full_leaves[:, 3:], ver.r_gkr), ver.den_eval)

        real_only = full_leaves.copy()
        l_idx, r_idx = layout.mask_leaf_indices
        for idx in l_idx + r_idx:
            real_only[idx] = 0
        rn = _mle_eval_host(real_only[:, :3], ver.r_gkr)
        rd = _mle_eval_host(real_only[:, 3:], ver.r_gkr)
        mn, md = MG.mask_leaf_contributions(
            layout, ver.r_gkr,
            [tuple(int(v) for v in row) for row in mask_a],
            [tuple(int(v) for v in row) for row in mask_d],
        )
        self.assertEqual(hadd(rn, mn), ver.num_eval)
        self.assertEqual(hadd(rd, md), ver.den_eval)

    def test_gkr_layer0_denominator_quotient_is_blinded_counterexample(self):
        """Regression for layer-0 denominator blinding: minimal counterexample `0/x` on left half vs `(0,1)` on right.
        With independent denominator blinders (0, d_L)/(0, d_R), published `dl0 / dr0` MUST NOT equal `x`."""
        from flashprove_next.field import hinv
        from prover.fpn_prover.backend.fpn2 import dec3

        layout = MG.MultiGkrLayout(s=4, m_mask=2)
        for x in (1, 2, 7):
            rn = np.array([[0, 0, 0]], dtype=np.uint64)
            rd = np.array([[x, 0, 0]], dtype=np.uint64)
            quotients = set()
            for _ in range(5):
                lc, ma, md = MG.sample_gkr_masks(layout)
                res = MG.prove_multi_gkr(
                    layout, rn, rd, Fpn2Transcript(b"test/gkr_denom_blind"),
                    libra_coeffs=lc, mask_a=ma, mask_d=md,
                )
                MG.verify_multi_gkr(layout, res.proof, Fpn2Transcript(b"test/gkr_denom_blind"))
                # Layer 0 children sit right after 48B root: (nl0, dl0, nr0, dr0)
                dl0 = dec3(res.proof[48 + 24:48 + 48])
                dr0 = dec3(res.proof[48 + 72:48 + 96])
                q = hmul(dl0, hinv(dr0))
                self.assertNotEqual(q, hf(x), "LEAK: dl0 / dr0 revealed real denominator x!")
                quotients.add(q)
            self.assertEqual(len(quotients), 5, "dl0 / dr0 must vary independently across proofs")

    def test_gkr_rejects_out_of_table_and_wrong_mult_and_tamper(self):
        layout = MG.MultiGkrLayout(s=6, m_mask=4)
        rng = np.random.default_rng(77)
        libra_coeffs, mask_a, mask_d = MG.sample_gkr_masks(layout, rng=rng)

        for kw in (dict(cheat_query=True), dict(cheat_mult=True)):
            tr_p = Fpn2Transcript(b"test/multi_gkr/cheat")
            beta, gamma = tr_p.squeeze_fp3(), tr_p.squeeze_fp3()
            rn, rd = self._build_multi_table_leaves(beta, gamma, **kw)
            res = MG.prove_multi_gkr(layout, rn, rd, tr_p, libra_coeffs=libra_coeffs, mask_a=mask_a, mask_d=mask_d)
            tr_v = Fpn2Transcript(b"test/multi_gkr/cheat")
            tr_v.squeeze_fp3(); tr_v.squeeze_fp3()
            with self.assertRaises(MG.MultiGkrError) as cm:
                MG.verify_multi_gkr(layout, res.proof, tr_v)
            self.assertEqual(cm.exception.code, "E_GKR_ROOT")

        # Tamper layer-1 round polynomial on an honest proof -> E_GKR
        tr_p = Fpn2Transcript(b"test/multi_gkr/tamper")
        beta, gamma = tr_p.squeeze_fp3(), tr_p.squeeze_fp3()
        rn, rd = self._build_multi_table_leaves(beta, gamma)
        res = MG.prove_multi_gkr(layout, rn, rd, tr_p, libra_coeffs=libra_coeffs, mask_a=mask_a, mask_d=mask_d)
        bad = bytearray(res.proof)
        bad[48 + 96 + 24] ^= 1
        tr_v = Fpn2Transcript(b"test/multi_gkr/tamper")
        tr_v.squeeze_fp3(); tr_v.squeeze_fp3()
        with self.assertRaises(MG.MultiGkrError) as cm:
            MG.verify_multi_gkr(layout, bytes(bad), tr_v)
        self.assertEqual(cm.exception.code, "E_GKR")

    def test_combined_gkr_plus_multi_pcs_zero_phase2(self):
        """End-to-end check: GKR masks + leaf arrays discharged in ONE multi_pcs opening (0 Phase-2!)."""
        gkr_lay = MG.MultiGkrLayout(s=6, m_mask=4)
        pcs_lay = MP.MultiPcsLayout(n=8, n_groups=2, queries=4)
        rng = np.random.default_rng(314)

        libra_coeffs, mask_a, mask_d = MG.sample_gkr_masks(gkr_lay, rng=rng)
        # Pack (libra_coeffs, mask_a, mask_d) into group 0 BEFORE squeezing GKR challenges
        n_lib = gkr_lay.libra_coeff_count
        m_m = gkr_lay.m_mask
        aux = np.zeros((pcs_lay.data_capacity, 3), dtype=np.uint64)
        aux[:n_lib] = libra_coeffs
        aux[n_lib:n_lib + m_m] = mask_a
        aux[n_lib + m_m:n_lib + 2 * m_m] = mask_d
        grp0 = MP.commit_data_group(pcs_lay, aux, rng=rng)

        # Group 1 holds dummy witness values (e.g. Phase-1 data)
        wit = rng.integers(0, 1000, size=pcs_lay.data_capacity, dtype=np.uint64)
        grp1 = MP.commit_data_group(pcs_lay, wit, rng=rng)

        tr_p = Fpn2Transcript(b"test/combined")
        tr_p.absorb_bytes(grp0.root + grp1.root)
        beta = tr_p.squeeze_fp3()
        gamma = tr_p.squeeze_fp3()
        rn, rd = self._build_multi_table_leaves(beta, gamma)
        gkr_res = MG.prove_multi_gkr(
            gkr_lay, rn, rd, tr_p, libra_coeffs=libra_coeffs, mask_a=mask_a, mask_d=mask_d,
        )

        # Batch the GKR Libra evaluations + mask_a/mask_d contributions as ONE linear claim on grp0
        eta = tr_p.squeeze_fp3()
        w0 = np.zeros((pcs_lay.group_len, 3), dtype=np.uint64)
        claim0 = ZERO
        ep = ONE
        for k, (r_k, g_r) in enumerate(zip(gkr_res.layer_points, gkr_res.layer_libra_evals), start=1):
            off_l = gkr_lay.layer_libra_offset(k)
            for i, ri in enumerate(r_k):
                r2 = hmul(ri, ri)
                r3 = hmul(r2, ri)
                for j, pw in enumerate((ONE, ri, r2, r3)):
                    idx = off_l + 4 * i + j
                    cur = (int(w0[idx, 0]), int(w0[idx, 1]), int(w0[idx, 2]))
                    w0[idx] = hadd(cur, hmul(ep, pw))
            claim0 = hadd(claim0, hmul(ep, g_r))
            ep = hmul(ep, eta)

        # Also check group 1 at a random point
        pt1 = [tr_p.squeeze_fp3() for _ in range(pcs_lay.n)]
        w1 = MP._eq_table_fp3(pt1)
        claim1 = _dot_fp3(grp1.cells, w1)

        pcs_proof = MP.prove_multi_open(pcs_lay, [grp0, grp1], [w0, w1], [claim0, claim1], tr_p, rng=rng)

        # Verifier replay
        tr_v = Fpn2Transcript(b"test/combined")
        tr_v.absorb_bytes(grp0.root + grp1.root)
        self.assertEqual(tr_v.squeeze_fp3(), beta)
        self.assertEqual(tr_v.squeeze_fp3(), gamma)
        ver_gkr = MG.verify_multi_gkr(gkr_lay, gkr_res.proof, tr_v)
        eta_v = tr_v.squeeze_fp3()
        pt1_v = [tr_v.squeeze_fp3() for _ in range(pcs_lay.n)]
        self.assertEqual(eta_v, eta)
        self.assertEqual(pt1_v, pt1)

        # Recompute claim0 purely from ver_gkr.layer_libra_evals
        c0_v = ZERO
        ep = ONE
        for g_r in ver_gkr.layer_libra_evals:
            c0_v = hadd(c0_v, hmul(ep, g_r))
            ep = hmul(ep, eta_v)
        self.assertEqual(c0_v, claim0)

        w_eval_fn = lambda g, r_ip: _mle_eval_host(w0, r_ip) if g == 0 else MP.eq_eval_pt(pt1_v, r_ip)
        MP.verify_multi_open(pcs_lay, [grp0.root, grp1.root], [c0_v, claim1], w_eval_fn, pcs_proof, tr_v)
        self.assertEqual(tr_p.state, tr_v.state)


if __name__ == "__main__":
    unittest.main()
