# SPDX-License-Identifier: Apache-2.0
"""Relation 6 (`FOLDED_VEC_BOUND`): pre-committed, ZK-blinded activations `x_root` + soundness / zero-knowledge audit.

* `commit_x` (OS CSPRNG, immune to FPN_TEST_RNG_SEED) -> `C_pre` (`precommit_x_root`, 0 plaintext bytes) ->
  `prove_bound` -> native Rust `fpn-verify --weight-root --x-root` AND the detached `-I` Python reference VERIFIED;
* relation-5 "case (c)": the SAME substitution `(X', Y' = W X')` that relation 5 accepts is rejected by relation 6
  under EVERY cheat strategy, identically by Rust and Python;
* joint linear zero-knowledge over (F, X_com): `rank(B) == rank([B | A])` on the full observation set, every
  observation row validated against the wire bytes, three negative controls that must fail;
* re-proofs (fresh `commit_x` each) share 0 private 64-byte blocks; exhaustive byte-flip rejection in Rust (every byte)
  and Python (dense + strided here; the full 393,216-byte Python sweep is the opt-in `FPN_EXHAUSTIVE_FLIPS=1` test of
  `verifier/python/tests/test_folded_vec_bound.py`).
"""
import hashlib
import os
import random
import shutil
import statistics
import struct
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from flashprove_next.field import P
from fpn_prover.backend import cuda_engine as CE
from fpn_prover.backend import folded_vec as FV
from fpn_prover.backend import folded_vec_bound as FB
from fpn_prover.backend.fpn2 import (FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1, FOLDED_VEC_N16_V1,
                                     dec3, profile_id)
from fpn_prover.circuit import e2e_prover as E2E
from fpn_prover.runtime.setup_cert import SetupCertificate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUST = os.path.join(ROOT, "core", "target", "release", "fpn-verify")
PROF = FOLDED_VEC_BOUND_N16_V1


class _Rng:
    """Deterministic test rng (explicit argument only; the production default is the OS CSPRNG)."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return (self.r.randrange(P), self.r.randrange(P), self.r.randrange(P))


def _tile(seed, lo=-(1 << 15), hi=1 << 15):
    lay = FB.layout_of(PROF)
    g = np.random.default_rng(seed)
    Wi = g.integers(lo, hi, size=(lay.B, FV.ROWS, lay.K))
    Xi = g.integers(lo, hi, size=(lay.B, lay.K))
    Yi, bits = FV.exact_products(Wi, Xi)
    return lay, Wi, Xi, Yi, bits


def _rust(data, w, x, extra=()):
    args = [RUST]
    if w is not None:
        args += ["--weight-root", w.hex()]
    if x is not None:
        args += ["--x-root", x.hex()]
    return E2E._run_cli(args + list(extra), data)


def _py(data, w, x):
    vdir = os.path.join(ROOT, "verifier", "python")
    boot = ("import runpy, sys; " f"sys.path.insert(0, {vdir!r}); "
            "sys.argv = ['fpn_verify_ref', *sys.argv[1:]]; runpy.run_module('fpn_verify_ref', run_name='__main__')")
    args = [os.sys.executable, "-W", "error", "-I", "-c", boot]
    if w is not None:
        args += ["--weight-root", w.hex()]
    if x is not None:
        args += ["--x-root", x.hex()]
    return E2E._run_cli(args, data, cwd=vdir)


def _verdicts(data, w, x):
    py = E2E.verdict_code(_py(data, w, x)[0])
    rs = E2E.verdict_code(_rust(data, w, x)[0]) if os.path.exists(RUST) else None
    return rs, py


class Relation6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CE.HostEngine()
        cls.lay, cls.Wi, cls.Xi, cls.Yi, cls.bits = _tile(11)
        cls.W, cls.X, cls.Y = FV.to_field(cls.Wi), FV.to_field(cls.Xi), FV.to_field(cls.Yi)
        cls.wroot = FB.weight_root(cls.lay, cls.W, cls.engine)
        cls.xc = FB.commit_x(cls.lay, cls.X, rng=_Rng(1), engine=cls.engine)
        cls.xroot = cls.xc.root
        cls.rec = {}
        cls.honest = FB.prove_bound(PROF, cls.W, cls.X, cls.Y, cls.xc, n_s=b"\x07" * 32, rng=_Rng(3),
                                    engine=cls.engine, hooks={"record": cls.rec})

    # ------------------------------------------------------------------ geometry, profiles, bytes
    def test_layout_and_profiles(self):
        n16, n22 = FB.layout_of(PROF), FB.layout_of(FOLDED_VEC_BOUND_N22_V1)
        self.assertEqual((n16.B, n16.K, n16.macs), (1, 1024, 32768))
        self.assertEqual((n22.B, n22.K, n22.macs), (64, 1024, 2_097_152))
        self.assertEqual((n16.body_used, n22.body_used), (334_480, 529_120))
        self.assertLessEqual(64 + n16.body_used, PROF["c_p"])
        self.assertLessEqual(64 + n22.body_used, FOLDED_VEC_BOUND_N22_V1["c_p"])
        self.assertGreater(64 + n22.body_used, 524_288)              # relation 5's C_p does not fit: 1 MiB
        self.assertEqual(profile_id(PROF).hex(), "9492aa70db8c3d79fb242001f178a43b31ba5e610114d2606c87d6350eb0472e")
        self.assertNotEqual(profile_id(PROF), profile_id(FOLDED_VEC_N16_V1))
        self.assertLess(self.bits, 62)

    def test_host_engine_matches_cpu_reference(self):
        self.assertEqual(FB.weight_root(self.lay, self.W), self.wroot)
        xc_cpu = FB.commit_x(self.lay, self.X, rng=_Rng(1))
        self.assertEqual(xc_cpu.root, self.xroot)
        cpu = FB.prove_bound(PROF, self.W, self.X, self.Y, xc_cpu, n_s=b"\x07" * 32, rng=_Rng(3))
        self.assertEqual(len(cpu), PROF["c_p"])
        self.assertEqual(cpu, self.honest)

    def test_honest_dual_verify_and_timing(self):
        self.assertEqual(len(self.honest), PROF["c_p"])
        rs, py = _verdicts(self.honest, self.wroot, self.xroot)
        self.assertEqual(py, "VERIFIED")
        if rs is not None:
            self.assertEqual(rs, "VERIFIED")
            walls = [_rust(self.honest, self.wroot, self.xroot)[1] for _ in range(5)]
            print(f"\nrelation 6 n16: container {len(self.honest)} B (body {self.lay.body_used} B), "
                  f"Rust fpn-verify wall {statistics.median(walls):.1f} ms (median of 5, process start included)")

    # ------------------------------------------------------------------ C_pre binds x_root, 0 plaintext bytes
    def test_precommit_x_root_binds_without_plaintext(self):
        cert = SetupCertificate(b"w" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 0).encode()
        resp = [11, 22, 33]
        c_pre = FB.precommit_x_root(cert, self.xroot, resp)
        self.assertEqual(len(c_pre), 32)
        self.assertEqual(c_pre, FB.precommit_x_root(cert, self.xroot, resp))
        self.assertNotEqual(c_pre, FB.precommit_x_root(cert, self.xroot, resp + [4]))
        other = FB.commit_x(self.lay, self.X, rng=_Rng(2), engine=self.engine)   # same X, other blinding
        self.assertNotEqual(self.xroot, other.root)
        self.assertNotEqual(c_pre, FB.precommit_x_root(cert, other.root, resp))
        other_cert = SetupCertificate(b"v" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 0).encode()
        self.assertNotEqual(c_pre, FB.precommit_x_root(other_cert, self.xroot, resp))
        # domain separation: the x_root can never alias a trace payload root
        from fpn_prover.runtime.pipeline import precommit_request
        self.assertNotEqual(c_pre, precommit_request(cert, self.xroot, resp))
        tpr = hashlib.blake2b(b"trace", digest_size=32).digest()
        self.assertNotEqual(FB.precommit_x_root(cert, self.xroot, resp, tpr), precommit_request(cert, tpr, resp))
        with self.assertRaises(ValueError):
            FB.precommit_x_root(cert, b"short", resp)
        # the plaintext X is not an input: a different X under the same x_root bytes is indistinguishable here
        import inspect
        self.assertNotIn("X", inspect.signature(FB.precommit_x_root).parameters)
        self.assertEqual(len(struct.pack("<32s", self.xroot)), 32)           # 32 bytes of hiding commitment only

    # ------------------------------------------------------------------ OS CSPRNG, never FPN_TEST_RNG_SEED
    def test_commit_x_uses_os_csprng_never_the_test_seed(self):
        import fpn_prover.backend.folded_vec as fv_mod
        with patch.dict(os.environ, {"FPN_TEST_RNG_SEED": "123456"}):
            a = FB.build_x_cells_np(self.lay, self.X, FB.SecretRng())
            b = FB.build_x_cells_np(self.lay, self.X, FB.SecretRng())
        xi = FV.cell_np(np.arange(self.lay.B * self.lay.K, dtype=np.int64))
        mask = np.ones(1 << self.lay.n, dtype=bool)
        mask[xi] = False
        self.assertFalse(np.array_equal(a[mask], b[mask]))                   # blinding differs between draws
        self.assertTrue((a[mask] != 0).any(axis=1).mean() > 0.99)
        self.assertTrue(np.array_equal(a[xi, 0], self.X.reshape(-1)))        # X sits in the data cells, F_p only
        self.assertFalse(a[xi, 1:].any())
        calls = []
        real = fv_mod.secrets.token_bytes
        with patch.object(fv_mod.secrets, "token_bytes", lambda n: (calls.append(n), real(n))[1]):
            with patch.dict(os.environ, {"FPN_TEST_RNG_SEED": "123456"}):
                FB.build_x_cells_np(self.lay, self.X, FB.SecretRng())
        try:
            import cryptography  # noqa: F401
            # With `cryptography`, large draws expand a fresh ephemeral 256-bit OS-CSPRNG key via AES-256-CTR.
            self.assertGreaterEqual(sum(calls), 32)
        except ImportError:
            self.assertGreaterEqual(sum(calls), 24 * (1 << self.lay.n))      # raw OS bytes fallback

    # ------------------------------------------------------------------ case (c)
    def test_case_c_forgeries_rejected_by_rust_and_python(self):
        """Relation 5 accepts a private substitution (X', Y' = W X'); relation 6 rejects it under every strategy."""
        Xs_i = self.Xi.copy()
        Xs_i[0, 11] += 1
        Xs = FV.to_field(Xs_i)
        Ys = FV.to_field(FV.exact_products(self.Wi, Xs_i)[0])
        # the gap in relation 5: the substituted proof verifies (nothing binds X to anything the verifier holds)
        r5 = FV.prove(self.W, Xs, Ys, n_s=b"\x07" * 32, rng=_Rng(3), engine=self.engine)
        r5_root = FV.weight_root(self.lay, self.W, self.engine)
        self.assertEqual(E2E.verdict_code(E2E.python_verify_detached(r5, r5_root)[0]), "VERIFIED")
        # relation 6: honest X committed (x_root pinned, in C_pre) BEFORE the challenge
        xcs = FB.commit_x(self.lay, Xs, rng=_Rng(5), engine=self.engine)
        strategies = {
            # (1) re-prove (X', Y') but claim the honest x_root in the statement
            "reprove_under_honest_x_root": (FB.prove_bound(PROF, self.W, Xs, Ys, xcs, rng=_Rng(6), engine=self.engine,
                                                          hooks={"stmt_x_root": self.xroot}), "E_MERKLE"),
            # (2) substitute the cheater's own x_root'
            "substitute_x_root": (FB.prove_bound(PROF, self.W, Xs, Ys, xcs, rng=_Rng(6), engine=self.engine),
                                  "E_X_ROOT_PINNED"),
            # (3) forge the X evaluation: sumcheck on X' but the only openable X_com is the honest one
            "forged_x_evaluation": (FB.prove_bound(PROF, self.W, Xs, Ys, self.xc, rng=_Rng(6), engine=self.engine,
                                                   hooks={"allow_x_mismatch": True}), "E_SUMCHECK"),
            # (4) forged X evaluation with an adaptive e_W shift on top
            "forged_x_evaluation_adaptive": (FB.prove_bound(
                PROF, self.W, Xs, Ys, self.xc, rng=_Rng(6), engine=self.engine,
                hooks={"allow_x_mismatch": True, "ip_adaptive": True,
                       "e_w": lambda v: tuple((x + 1) % P if i == 0 else x for i, x in enumerate(v))}),
                                             "E_SUMCHECK"),
            # (5) honest X and honest Y but a tampered Y' (not a substitution of X): inner-product zero-claim
            "honest_x_false_y": (FB.prove_bound(PROF, self.W, self.X, np.where(np.arange(32) == 3, (self.Y + 1) % P,
                                                                                self.Y).astype(np.uint64),
                                                self.xc, rng=_Rng(6), engine=self.engine), "E_ZEROCHECK"),
        }
        for name, (data, want) in strategies.items():
            with self.subTest(name):
                rs, py = _verdicts(data, self.wroot, self.xroot)
                self.assertEqual(py, want)
                if rs is not None:
                    self.assertEqual(rs, want)
        # the cheater's proof is internally valid: it verifies under ITS OWN pins (so the pin is what binds X)
        rs, py = _verdicts(strategies["substitute_x_root"][0], self.wroot, xcs.root)
        self.assertEqual(py, "VERIFIED")
        if rs is not None:
            self.assertEqual(rs, "VERIFIED")
        # pin precedence and fail-closed behaviour through the CLIs
        for w, x, want in ((None, self.xroot, "E_CIRCUIT_UNPINNED"), (self.wroot, None, "E_X_ROOT_UNPINNED"),
                           (None, None, "E_CIRCUIT_UNPINNED"), (bytes(32), self.xroot, "E_CIRCUIT_PINNED"),
                           (self.wroot, bytes(32), "E_X_ROOT_PINNED")):
            rs, py = _verdicts(self.honest, w, x)
            self.assertEqual(py, want)
            if rs is not None:
                self.assertEqual(rs, want)

    def test_x_com_tampering_rejected(self):
        """A flipped bit in an opened X_com coset value / Merkle sibling at any queried position fails closed."""
        lay = self.lay
        q0 = 64 + lay.body_used - lay.queries * lay.query_len
        open0 = 96 + 32 * lay.depth(0)
        for qi in (0, 17, lay.queries - 1):
            x_leaf = q0 + qi * lay.query_len + 2 * open0
            for off in (7, 96 + 33):
                b = bytearray(self.honest)
                b[x_leaf + off] ^= 1
                rs, py = _verdicts(bytes(b), self.wroot, self.xroot)
                self.assertEqual(py, "E_MERKLE", (qi, off))
                if rs is not None:
                    self.assertEqual(rs, "E_MERKLE", (qi, off))

    # ------------------------------------------------------------------ joint linear zero knowledge
    def test_rows_reproduce_the_wire_values(self):
        """Every observation row of the rank model, evaluated on the actual private cells, equals the value the
        container reveals (claims, batching rounds, FINAL and every opened coset at every layer) -- a rank pass on
        mis-modelled rows would prove nothing."""
        lay, rec, out = self.lay, self.rec, self.honest
        F = rec["F"].T.copy()
        Xc = rec["Xcells"].T.copy()
        C = FV.weight_cells_np(lay, self.W).T.copy()

        def add3(*xs):
            return tuple(sum(int(x[i]) for x in xs if x is not None) % P for i in range(3))

        vals = FB.bound_wire_values(lay, out)
        rows = list(FB.bound_claim_rows(lay, rec, self.W))
        self.assertEqual(len(rows), len(vals))
        for (label, rf, rx), v in zip(rows, vals):
            self.assertEqual(add3(FB.eval_row(rf, F), FB.eval_row(rx, Xc)), v, label)
        o = 64 + FB.STMT + lay.ip_len + 24
        rw = []
        for i in range(lay.n + 2):
            rw.append([dec3(out[o + 24 * t:o + 24 * t + 24]) for t in range(3)])
            o += 72 + (32 if i >= 3 and i % 2 == 1 and (i - 3) // 2 < lay.folds - 1 else 0)
        final = dec3(out[o:o + 24])
        o += 24
        length = FB.pcs.RATE << lay.n
        wire, open0 = {}, 96 + 32 * lay.depth(0)
        for pos in rec["positions"]:
            for t in range(lay.folds):
                j = pos % (length >> (2 * t + 2))
                vv = [dec3(out[o + 24 * k:o + 24 * k + 24]) for k in range(4)]
                o += 96 + 32 * lay.depth(t)
                if t == 0:
                    wire[("F", j)] = vv
                    o += open0
                    wire[("X", j)] = [dec3(out[o + 24 * k:o + 24 * k + 24]) for k in range(4)]
                    o += open0
                else:
                    wire[(t, j)] = vv
                pos = j
        checked = 0
        for label, rf, rx, rc in FB.pcs_rows_bound_np(lay, rec, with_public=True):
            got = add3(FB.eval_row(rf, F), FB.eval_row(rx, Xc), FB.eval_row(rc, C) if rc is not None else None)
            if label.startswith("pcs_round"):
                _, _, i, t = label.split("_")
                exp = rw[int(i)][int(t)]
            elif label == "pcs_final":
                exp = final
            else:
                parts = label.split("_")
                t, j, k = int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])
                exp = wire[(("F" if label.endswith("_F") else "X", j) if t == 0 else (t, j))][k]
            self.assertEqual(got, exp, label)
            checked += 1
        self.assertEqual(checked, 3 * (lay.n + 2) + 1 + FB._count_queries_bound(lay, rec["positions"]))

    def test_joint_linear_zk_full_rank_and_negative_controls(self):
        rep = FB.require_joint_mask_vec_bound(self.lay, self.rec, self.W)
        print("\njoint linear ZK over (F, X_com):", {k: rep[k] for k in (
            "observations", "fp_rows", "b_columns_F", "b_columns_X", "a_columns", "rank_B", "rank_BA", "holds")})
        self.assertTrue(rep["holds"])
        self.assertEqual(rep["rank_B"], rep["rank_BA"])
        self.assertGreater(rep["rank_B"], 4000)
        for kw in ({"drop_libra": True}, {"drop_x_blinding": True}, {"zero_f_x_block": True}):
            r = FB.require_joint_mask_vec_bound(self.lay, self.rec, self.W, raise_on_fail=False, **kw)
            print(f"negative control {kw}: rank_B={r['rank_B']} rank_BA={r['rank_BA']}")
            self.assertFalse(r["holds"], kw)
            self.assertLess(r["rank_B"], r["rank_BA"], kw)
        with self.assertRaises(ValueError):
            FB.require_joint_mask_vec_bound(self.lay, self.rec, self.W, drop_x_blinding=True)

    # ------------------------------------------------------------------ re-proofs
    def test_reproofs_with_fresh_commit_x_share_no_private_blocks(self):
        lay = self.lay
        xc2 = FB.commit_x(lay, self.X, engine=self.engine)                  # fresh OS-CSPRNG blinding
        again = FB.prove_bound(PROF, self.W, self.X, self.Y, xc2, n_s=b"\x07" * 32, engine=self.engine)
        base = FB.prove_bound(PROF, self.W, self.X, self.Y, FB.commit_x(lay, self.X, engine=self.engine),
                              n_s=b"\x07" * 32, engine=self.engine)
        rep = FB.shared_blocks(base, again, lay, PROF["c_p"])
        print("\nre-proof leakage (fresh commit_x each):", rep)
        self.assertEqual(rep["private_aligned_shared"], 0)
        self.assertEqual(rep["private_window_set_intersection"], 0)
        # honest limit: re-proving under the SAME x_commitment re-opens the same X_com tree (upper Merkle siblings
        # repeat; each proof also spends fresh blinding rank cumulatively, which the per-proof rank check does not cover)
        reuse = FB.prove_bound(PROF, self.W, self.X, self.Y, self.xc, n_s=b"\x07" * 32, engine=self.engine)
        rep2 = FB.shared_blocks(self.honest, reuse, lay, PROF["c_p"])
        print("re-proof leakage (SAME x_commitment reused):", rep2)
        self.assertEqual(rep2["private_aligned_shared"] > 0 or rep2["private_window_set_intersection"] > 0, True)

    # ------------------------------------------------------------------ byte flips
    def test_byte_flips_python_and_rust(self):
        base = self.honest
        dense = 64 + 128 + 24 + 96 * 10 + 48 + 72 * 18 + 32 * 7 + 24
        from fpn_verify_ref.verify import verify_result
        w, x = self.wroot, self.xroot
        for i in list(range(0, dense, 2)) + list(range(dense, len(base), 997)):
            b = bytearray(base)
            b[i] ^= 1 << (i % 8)
            self.assertNotEqual(verify_result(bytes(b), False, None, w, x), "VERIFIED", i)
        cargo = shutil.which("cargo")
        if cargo is None or not os.path.exists(RUST):
            self.skipTest("cargo not available for the exhaustive Rust sweep")
        r = subprocess.run([cargo, "test", "--release", "--test", "folded_vec_bound", "bound_every_byte_flip_rejects"],
                           cwd=os.path.join(ROOT, "core"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
