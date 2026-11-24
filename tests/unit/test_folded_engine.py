# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for `folded_engine` (unified Folded ZK Engine with 0 Phase-2 LogUp + batched coset-4 PCS)."""
from __future__ import annotations

import os
import unittest
import numpy as np

from flashprove_next import gadgets as G
from flashprove_next import reply_zk as RZ
from flashprove_next import zk_flat as Z
from flashprove_next.field import CPU, ONE, P, ZERO
from prover.fpn_prover.backend import folded_engine as FE
from prover.fpn_prover.backend import multi_gkr as MG
from prover.fpn_prover.backend import multi_pcs as MP


class TestFoldedEngineCore(unittest.TestCase):
    def _build_synth_circuit(self, prove=True, cheat_range=False, cheat_gate=False, cheat_deg2_sel=False):
        """Synthetic circuit exercising zerocheck, sumzero, 1-col range lookup, 2-col table lookup,
        a degree-2 lookup selector, and an interactive commit round."""
        ctx = G.Ctx(prove)
        sp = ctx.space("m64", [("x", 6)], padded=False)   # 64 rows, content = 32
        N = sp.content
        a_v = np.arange(N, dtype=np.int64) % 8
        b_v = (np.arange(N, dtype=np.int64) * 3 + 1) % 8
        c_v = a_v * b_v
        if cheat_gate:
            c_v[5] += 1
        if cheat_range:
            a_v[3] = 50  # outside [0, 32)

        ca = sp.col("a", a_v if prove else None)
        cb = sp.col("b", b_v if prove else None)
        cc = sp.col("c", c_v if prove else None)
        sp.zero(ca.ex * cb.ex - cc.ex, tag="mul")

        # 1-col lookup table [0, 32)
        G.table(ctx, "rng32", [np.arange(32, dtype=np.int64)])
        G.wire_lookup(ctx, "rng32", sp, [ca.ex], [a_v] if prove else None)
        G.wire_lookup(ctx, "rng32", sp, [cb.ex], [b_v] if prove else None)

        # 2-col lookup table (x, x^2) for x in [0, 32) with a degree-2 boolean selector s1 * s2
        tx = np.arange(32, dtype=np.int64)
        G.table(ctx, "sq32", [tx, tx * tx])

        s1_v = (np.arange(N, dtype=np.int64) % 2 == 0).astype(np.int64)
        s2_v = (np.arange(N, dtype=np.int64) < 20).astype(np.int64)
        cs1 = sp.col("s1", s1_v if prove else None)
        cs2 = sp.col("s2", s2_v if prove else None)
        sq_v = a_v * a_v
        if cheat_deg2_sel:
            sq_v[2] = 999  # row 2 has s1=1, s2=1 -> active bad lookup!
        csq = sp.col("sq", sq_v if prove else None)
        # Active only when s1 * s2 == 1 (degree-2 selector!)
        G.wire_lookup(
            ctx, "sq32", sp, [ca.ex, csq.ex], [a_v, sq_v] if prove else None,
            sel=cs1.ex * cs2.ex, sel_vals=(s1_v * s2_v) if prove else None,
        )

        # Interactive commit round: draw challenge gamma, commit d = a + gamma * b, check zero
        def rnd(tr):
            gam = tr.ext()
            dv = None
            if prove:
                from flashprove_next.field import hadd, hf, hmul
                arr = np.zeros((3, 1 << sp.nv), dtype=np.uint64)
                for i in range(N):
                    val = hadd(hf(int(a_v[i])), hmul(gam, hf(int(b_v[i]))))
                    arr[0, i], arr[1, i], arr[2, i] = val
                dv = arr
            v_d = ctx.cir.vec("rnd.d", sp.nv, N, phase=ctx.cir.cur_phase)
            v_d.is_ext = True
            if prove:
                v_d.vals = dv
            sp.zero(Z.Ex.atom(Z.View(v_d)) - ca.ex - cb.ex * Z.Ex.const(gam), tag="rnd_lin")

        ctx.cir.rounds.append(rnd)
        return ctx.cir

    def test_synth_honest_zero_phase2_and_verify(self):
        for gkr_max_s in (26, 8):
            cir_p = self._build_synth_circuit(prove=True)
            stmt = {"test": f"synth_v1_{gkr_max_s}"}
            rng = np.random.default_rng(101)
            pf, info = FE.prove_folded(cir_p, stmt, queries=4, gkr_masks=4, gkr_max_s=gkr_max_s, rng=rng)
            self.assertEqual(info["phase2_cells"], 0)
            self.assertGreater(len(info["gkr_s"]), 0)
            if gkr_max_s == 8:
                self.assertGreaterEqual(len(info["gkr_s"]), 2)

            cir_v = self._build_synth_circuit(prove=False)
            self.assertTrue(FE.verify_folded(cir_v, stmt, pf, min_queries=4))
            # Default production min_queries=72 MUST reject weak queries=4 proof
            with self.assertRaises(FE.FoldedEngineError) as cm_q:
                FE.verify_folded(self._build_synth_circuit(prove=False), stmt, pf)
            self.assertEqual(cm_q.exception.code, "E_REDUCED_SECURITY")

    def test_synth_rejects_cheating_witnesses(self):
        stmt = {"test": "synth_cheat"}
        # 1. Gate violation (c != a*b)
        cir_p = self._build_synth_circuit(prove=True, cheat_gate=True)
        pf, _ = FE.prove_folded(cir_p, stmt, queries=4, gkr_masks=4, rng=np.random.default_rng(1))
        with self.assertRaises((FE.FoldedEngineError, MP.MultiPcsError, MG.MultiGkrError)):
            FE.verify_folded(self._build_synth_circuit(prove=False), stmt, pf, min_queries=4)

        # 2. Out-of-table range lookup (a[3] = 13 not in [0, 8))
        old_force = Z.FORCE
        Z.FORCE = True
        try:
            cir_p = self._build_synth_circuit(prove=True, cheat_range=True)
            pf, _ = FE.prove_folded(cir_p, stmt, queries=4, gkr_masks=4, rng=np.random.default_rng(2))
        finally:
            Z.FORCE = old_force
        with self.assertRaises(MG.MultiGkrError) as cm:
            FE.verify_folded(self._build_synth_circuit(prove=False), stmt, pf, min_queries=4)
        self.assertEqual(cm.exception.code, "E_GKR_ROOT")

        # 3. Degree-2 selector lookup cheat (sq[2] = 999 while s1*s2 == 1)
        Z.FORCE = True
        try:
            cir_p = self._build_synth_circuit(prove=True, cheat_deg2_sel=True)
            pf, _ = FE.prove_folded(cir_p, stmt, queries=4, gkr_masks=4, rng=np.random.default_rng(3))
        finally:
            Z.FORCE = old_force
        with self.assertRaises(MG.MultiGkrError) as cm:
            FE.verify_folded(self._build_synth_circuit(prove=False), stmt, pf, min_queries=4)
        self.assertEqual(cm.exception.code, "E_GKR_ROOT")


class TestFoldedEngineExternalSeed(unittest.TestCase):
    def test_protocol_v2_external_seed_c_pre_binding_and_rejection(self):
        """Protocol v2 external-seed mode: C_pre binds base core before c_ext; tampering c_pre or c_ext fails closed."""
        from tests import test_fast_cpu as TFC
        from flashprove_next import sampling_v3 as S3

        _spec, prf, model, com = TFC.fixture()
        nc = (1000).to_bytes(16, "little")
        seq, em = TFC.rollout(model, TFC.SYS + TFC.REQ, 3, S3.prefix_state(TFC.KPRF, TFC.samp_ctx(nc)))
        cot, vis = em[:1], em[1:]
        api = {
            "req_tokens": TFC.REQ, "n_reason": len(cot), "vis_tokens": vis,
            "api_meta": {"finish_reason": "length", "max_tokens": len(em), "eos_token_id": TFC.EOS},
        }
        R, _wit = RZ.make_wit(prf, model, model.forward(seq), com, tokens=seq, sys_len=len(TFC.SYS))
        stp, _, _ = RZ.turn_state(prf, com, api, cot, nc, TFC.NS, TFC.R_ST)
        c_ext_1 = bytes.fromhex("ab" * 32)
        c_ext_2 = bytes.fromhex("cd" * 32)

        st1 = RZ.statement(prf, com, R, api=api, st_part=stp, c_ext=c_ext_1)
        st2 = RZ.statement(prf, com, R, api=api, st_part=stp, c_ext=c_ext_2)
        RZ.check_statement(prf, st1)
        RZ.check_statement(prf, st2)
        # Same C_pre before c_ext, different draws after different c_ext!
        self.assertEqual(st1["c_pre"], st2["c_pre"])
        self.assertNotEqual(st1["u_draws"], st2["u_draws"])

        # Tampered C_pre MUST reject fail-closed
        bad_cpre = dict(st1)
        bad_cpre["c_pre"] = "00" * 32
        with self.assertRaises(Z.ProofError):
            RZ.check_statement(prf, bad_cpre)

        # Swapped c_ext without re-deriving draws MUST reject fail-closed
        bad_cext = dict(st1)
        bad_cext["c_ext"] = c_ext_2.hex()
        with self.assertRaises(Z.ProofError):
            RZ.check_statement(prf, bad_cext)

        # Finding #6 (Round-2): verifier entry point `RZ.build(prf, stmt, None)` AUTOMATICALLY enforces
        # pre-recorded `expected_c_pre` and `expected_c_ext` in external_seed mode!
        recorded_c_pre = st1["c_pre"]
        with self.assertRaises(Z.ProofError) as cm:
            RZ.build(prf, st1, None)
        self.assertIn("E_CPRE_UNPINNED", str(cm.exception))
        # Providing both recorded pins succeeds
        RZ.build(prf, st1, None, expected_c_pre=recorded_c_pre, expected_c_ext=c_ext_1)

        # Late replacement commitment (different R -> different C_pre') after seeing c_ext MUST fail!
        R_late = [int(R[0]) ^ 1, int(R[1]), int(R[2]), int(R[3])]
        st_late = RZ.statement(prf, com, R_late, api=api, st_part=stp, c_ext=c_ext_1)
        self.assertNotEqual(st_late["c_pre"], recorded_c_pre)
        with self.assertRaises(Z.ProofError):
            RZ.build(prf, st_late, None, expected_c_pre=recorded_c_pre, expected_c_ext=c_ext_1)
        with self.assertRaises(Z.ProofError):
            RZ.build(prf, st1, None, expected_c_pre=recorded_c_pre, expected_c_ext=c_ext_2)

        # MINOR (Round-2): non-string JSON `c_ext` MUST raise `ProofError`, never `TypeError`
        for bad_val in (None, 123, ["aa"], {"hex": "00"}):
            st_bad_type = dict(st1)
            st_bad_type["c_ext"] = bad_val
            with self.assertRaises(Z.ProofError):
                RZ.check_statement(prf, st_bad_type)


class TestFoldedEngineFullReply(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("FPN_FULL_CPU_REPLY") == "1",
        "full TinyDense n=22 reply runs on GPU (GPU) or when FPN_FULL_CPU_REPLY=1",
    )
    def test_tiny_dense_full_reply_honest_and_tamper(self):
        """Full end-to-end LLM reply circuit (TinyDense: hidden sys prompt + hidden CoT + HSel + WSum + HCO
        + attention + bucketed sampling + multi-turn state) proved and verified by `folded_engine` with
        ZERO Phase-2 commitments!"""
        from tests import test_fast_cpu as TFC
        from flashprove_next import sampling_v3 as S3

        _spec, prf, model, com = TFC.fixture()
        for i in range(256):
            nc = (1000 + i).to_bytes(16, "little")
            seq, em = TFC.rollout(model, TFC.SYS + TFC.REQ, 3, S3.prefix_state(TFC.KPRF, TFC.samp_ctx(nc)))
            if TFC.EOS not in em and TFC.TOOL_END not in em:
                break
        else:
            raise AssertionError("no stop-free rollout found")
        cot, vis = em[:1], em[1:]
        api = {
            "req_tokens": TFC.REQ, "n_reason": len(cot), "vis_tokens": vis,
            "api_meta": {"finish_reason": "length", "max_tokens": len(em), "eos_token_id": TFC.EOS},
        }
        stmt, wit = TFC.reply_statement(prf, model, com, model.forward(seq), api, cot, nc)

        # Build full reply circuit and prove with folded_engine (0 Phase-2!)
        em_p, _, _ = RZ.build(prf, stmt, wit, B=CPU)
        pf, info = FE.prove_folded(em_p.ctx.cir, stmt, queries=8, gkr_masks=8, rng=np.random.default_rng(77))
        self.assertEqual(info["phase2_cells"], 0)

        # Independent verify-mode build (wit=None)
        em_v, _, _ = RZ.build(prf, stmt, None, B=CPU)
        self.assertTrue(FE.verify_folded(em_v.ctx.cir, stmt, pf, min_queries=8))

        # Tamper statement (modify visible reply token) -> MUST reject
        bad_stmt = dict(stmt)
        bad_stmt["vis_tokens"] = [int(vis[0]) ^ 1] + list(vis[1:])
        em_v2, _, _ = RZ.build(prf, stmt, None, B=CPU)
        with self.assertRaises((FE.FoldedEngineError, MP.MultiPcsError, MG.MultiGkrError)):
            FE.verify_folded(em_v2.ctx.cir, bad_stmt, pf, min_queries=8)


class TestFPNZK2ContainerFraming(unittest.TestCase):
    def test_fpnzk2_fixed_size_zero_padding_and_tamper_rejection(self):
        from flashprove_next import reply_bundle as RB
        spec = RB.get_registered_profile("qwen3-0.6b-extseed153")
        self.assertEqual(spec["security_tier"], "production")
        self.assertEqual(spec["M_C"], 102)
        self.assertEqual(spec["M_G"], 51)
        spec4 = RB.get_registered_profile("qwen3-4b-extseed153")
        self.assertEqual(spec4["L_MAX"], 64)
        self.assertEqual(spec4["d"], 2560)

        stmt = {"dummy": "stmt"}
        pf = {"version": "fpn2-folded-v1", "payload": b"\xaa\xbb\xcc"}
        blob = RB.pack_bundle(spec, stmt, pf, pad_to_bytes=4096, format="v2")
        self.assertEqual(len(blob), 4096)
        self.assertTrue(RB.is_reply_zk_blob(blob))
        unpacked = RB.unpack_bundle(blob)
        self.assertEqual(unpacked["pf"], pf)
        # Non-zero byte in tail padding -> E_PADDING
        tampered = bytearray(blob)
        tampered[-1] ^= 1
        with self.assertRaises(ValueError) as cm_pad:
            RB.unpack_bundle(bytes(tampered))
        self.assertIn("E_PADDING", str(cm_pad.exception))
        # Truncated container -> E_LENGTH
        with self.assertRaises(ValueError) as cm_len:
            RB.unpack_bundle(blob[:-1])
        self.assertIn("E_LENGTH", str(cm_len.exception))
        # Self-consistent FPNZK2 envelope whose declared & actual size != trusted profile size (8 MiB) -> E_LENGTH in verify_cli!
        ok, why, _info = RB.verify_cli(
            b"\x01" * 32,
            {"tokens": [1, 2]},
            {"tokens": [3]},
            "00" * 16,
            blob,
            allowed_profiles=["qwen3-0.6b-extseed153"],
        )
        self.assertFalse(ok)
        self.assertIn("E_LENGTH", why)
        # Malformed non-dict stmt / profile_spec inside FPNZK2 must return structured rejection, never raise!
        bad_blob = RB.pack_bundle(["not-a-dict"], "not-a-dict", {}, pad_to_bytes=4096, format="v2")
        ok_b, why_b, _ = RB.verify_cli(b"\x01" * 32, {"tokens": [1]}, {"tokens": [2]}, "00" * 16, bad_blob)
        self.assertFalse(ok_b)
        self.assertIn("malformed", why_b)
        # Oversized FPNZK2 container (> MAX_FPNZK2_BYTES = 16 MiB) rejected BEFORE JSON decode with E_LENGTH!
        huge_hdr = RB.HEADER_PREFIX_V2 + (RB.MAX_FPNZK2_BYTES + 1).to_bytes(8, "little") + (10).to_bytes(8, "little")
        huge_blob = huge_hdr + bytes(RB.MAX_FPNZK2_BYTES + 1 - len(huge_hdr))
        ok_h, why_h, _ = RB.verify_cli(b"\x01" * 32, {"tokens": [1]}, {"tokens": [2]}, "00" * 16, huge_blob)
        self.assertFalse(ok_h)
        self.assertIn("E_LENGTH", why_h)

    def test_sharded_checkpoint_resolves_weight_map_and_rejects_conflicts(self):
        """Checkpoint resolves indexed tensors via weight_map and rejects duplicate/missing shard entries."""
        import json as _json
        import struct as _struct
        import tempfile
        from flashprove_next import modelcommit as MC

        def _write_safetensors(path, tensors_dict):
            hdr = {}
            payload = bytearray()
            for name, arr in tensors_dict.items():
                raw = np.ascontiguousarray(arr, dtype=np.uint16).tobytes()
                a = len(payload)
                payload.extend(raw)
                hdr[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [a, len(payload)]}
            hb = _json.dumps(hdr).encode("utf-8")
            with open(path, "wb") as f:
                f.write(_struct.pack("<Q", len(hb)) + hb + payload)

        with tempfile.TemporaryDirectory() as td:
            a1 = np.array([[0x3F80, 0x4000]], dtype=np.uint16)
            b1 = np.array([[0x4040, 0x4080]], dtype=np.uint16)
            _write_safetensors(os.path.join(td, "shard-00001.safetensors"), {"w.a": a1})
            _write_safetensors(os.path.join(td, "shard-00002.safetensors"), {"w.b": b1})
            idx_p = os.path.join(td, "model.safetensors.index.json")
            with open(idx_p, "w") as f:
                _json.dump({"weight_map": {"w.a": "shard-00001.safetensors", "w.b": "shard-00002.safetensors"}}, f)
            ck = MC.Checkpoint(td)
            self.assertTrue(np.array_equal(ck.codes("w.a"), a1))
            self.assertTrue(np.array_equal(ck.codes("w.b"), b1))

            # Duplicate tensor across two shards -> MUST reject!
            _write_safetensors(os.path.join(td, "shard-00002.safetensors"), {"w.a": b1, "w.b": b1})
            with self.assertRaises(ValueError):
                MC.Checkpoint(td)

            # Indexed tensor missing from designated shard -> MUST reject!
            _write_safetensors(os.path.join(td, "shard-00002.safetensors"), {"w.b": b1})
            with open(idx_p, "w") as f:
                _json.dump({"weight_map": {"w.a": "shard-00002.safetensors", "w.b": "shard-00002.safetensors"}}, f)
            with self.assertRaises(ValueError):
                MC.Checkpoint(td)

            # Ambiguous mixed directory containing BOTH model.safetensors AND model.safetensors.index.json -> MUST reject!
            _write_safetensors(os.path.join(td, "model.safetensors"), {"w.a": a1, "w.b": b1})
            with self.assertRaises(ValueError):
                MC.Checkpoint(td)

    def test_rsqrt_enc_supports_d_gt_1024_for_qwen3_4b(self):
        """Generalized rsqrt_enc (K0 = 48 - d_shift) supports Qwen3-4B d=2560 with exact enclosure and no overflow."""
        from flashprove_next import certified as CZ, gadgets as G, zk_flat as Z
        for d in (1024, 2560, 4096):
            ctx = G.Ctx(prove=True)
            em = CZ.Em(ctx)
            SC = em.space("sc", [])
            Pl = em.col(SC, "Pl", np.full(SC.n, 1 << 19, dtype=np.int64), 22)
            Ph = em.col(SC, "Ph", np.full(SC.n, (1 << 19) + 2, dtype=np.int64), 22)
            eP = em.col(SC, "eP", np.full(SC.n, -20, dtype=np.int64), 11)
            T, et = CZ.rsqrt_enc(em, Pl, Ph, eP, d, f"rs.{d}")
            self.assertEqual(len(em.fails), 0, em.fails)
            self.assertEqual(len(ctx.notes), 0, ctx.notes)
            # Check numerical enclosure: T*2^et <= sqrt(d / (Ph*2^eP)) <= sqrt(d / (Pl*2^eP)) < (T+8)*2^et
            t_lo = float(int(T.v[0])) * (2.0 ** int(et.v[0]))
            t_hi = float(int(T.v[0]) + CZ.RSQ_W) * (2.0 ** int(et.v[0]))
            p_lo = float(int(Pl.v[0])) * (2.0 ** int(eP.v[0]))
            p_hi = float(int(Ph.v[0])) * (2.0 ** int(eP.v[0]))
            self.assertLessEqual(t_lo, (d / p_hi) ** 0.5 + 1e-12)
            self.assertGreater(t_hi, (d / p_lo) ** 0.5)


if __name__ == "__main__":
    unittest.main()
