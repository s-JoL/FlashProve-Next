# SPDX-License-Identifier: Apache-2.0
"""Soundness audit for `FPN2` Relation 2 (`LLM_INFERENCE_V2`):
  1. Exhaustive per-column ±1 perturbation scan across every content cell of every column;
  2. Constraint-deletion mutation testing (deleting any gate or lookup MUST allow a witness-only forgery
     that the full circuit rejects);
  3. Witness-only `wbforge` forgeries verified against BOTH standalone Rust `fpn-verify` and detached Python
     `fpn_verify_ref` (both MUST exit 1 with the exact expected reject code).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import numpy as np

from flashprove_next import gadgets as G
from flashprove_next import zk_flat as Z
from prover.fpn_prover.backend import relation2 as R2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN = os.path.join(ROOT, "core", "target", "release", "fpn-verify")


def _verify_both(container: bytes, desc: bytes, circuit_root: bytes, stmt_digest: bytes) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "p.fpn2")
        dp = os.path.join(td, "d.bin")
        with open(pf, "wb") as f:
            f.write(container)
        with open(dp, "wb") as f:
            f.write(desc)
        args = [
            "--profile", "fpn2-llm-std-v1",
            "--circuit-desc", dp,
            "--circuit-root", circuit_root.hex(),
            "--stmt-digest", stmt_digest.hex(),
        ]
        rs = subprocess.run([BIN, pf] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "verifier", "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        py_bin = "/opt/homebrew/bin/python3.14" if os.path.exists("/opt/homebrew/bin/python3.14") else "python3"
        py = subprocess.run([py_bin, "-m", "fpn_verify_ref", pf] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        assert (rs.returncode, rs.stdout.strip()) == (py.returncode, py.stdout.strip()), (
            f"Rust {(rs.returncode, rs.stdout.strip())} != Python {(py.returncode, py.stdout.strip())}"
        )
        return rs.returncode, rs.stdout.strip()


def _build_audited_circuit(
    *,
    drop_mul: bool = False,
    drop_add: bool = False,
    drop_lookup: bool = False,
    override: dict | None = None,
):
    """3-column audited Relation-2 circuit:
      a, b in [0, 64) (range lookup)
      c = a * b       (mul gate)
      d = c + a       (add gate)
    Every cell of (a, b, c, d) is constrained.
    """
    ctx = G.Ctx(True)
    sp = ctx.space("aud", [("x", 6)], padded=False)
    N = sp.content
    a_v = (np.arange(N, dtype=np.int64) * 2 + 1) % 32
    b_v = (np.arange(N, dtype=np.int64) * 3 + 2) % 32
    c_v = a_v * b_v
    d_v = c_v + a_v

    cols = {"a": a_v.copy(), "b": b_v.copy(), "c": c_v.copy(), "d": d_v.copy()}
    if override:
        for col_name, (idx, delta) in override.items():
            cols[col_name][idx] += delta

    ca = sp.col("a", cols["a"])
    cb = sp.col("b", cols["b"])
    cc = sp.col("c", cols["c"])
    cd = sp.col("d", cols["d"])

    if not drop_mul:
        sp.zero(ca.ex * cb.ex - cc.ex, tag="mul")
    if not drop_add:
        sp.zero(cc.ex + ca.ex - cd.ex, tag="add")

    if not drop_lookup:
        tx = np.arange(64, dtype=np.int64)
        idx_ex = Z.Ex.atom(Z.IndexP(list(range(6)), 1))
        lk = ctx.cir.lookups["rng64"] = Z.Lookup("rng64")
        lk.table = (6, [idx_ex])
        lk.table_vals = [tx]
        lk.mult = ctx.cir.vec("mult:rng64", 6, 64, phase=1, tail=0)
        ctx.cir.lookup("rng64", sp.nv, [ca.ex], sel=sp.sel(), vals=[cols["a"]], sel_vals=np.ones(N, dtype=np.int64))
        ctx.cir.lookup("rng64", sp.nv, [cb.ex], sel=sp.sel(), vals=[cols["b"]], sel_vals=np.ones(N, dtype=np.int64))

    return ctx.cir


class TestRelation2SoundnessAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["cargo", "build", "--manifest-path", os.path.join(ROOT, "core", "Cargo.toml"), "--release", "--bin", "fpn-verify"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def test_exhaustive_per_column_perturbation_scan(self):
        """Every single content cell (4 columns * 32 rows * ±1 = 256 perturbations) MUST violate constraints."""
        N = 32
        checked = 0
        for col_name in ("a", "b", "c", "d"):
            for idx in range(N):
                for delta in (+1, -1):
                    cir = _build_audited_circuit(override={col_name: (idx, delta)})
                    # Direct witness check (all zerochecks + sumzeros + lookups)
                    try:
                        fails = Z.check_witness(cir)
                    except Z.ProofError as e:
                        fails = [str(e)]
                    self.assertGreater(len(fails), 0, f"UNDER-CONSTRAINED: {col_name}[{idx}] {delta:+d} passed!")
                    checked += 1
        self.assertEqual(checked, 256)

    def test_constraint_deletion_mutation_and_wbforge_dual_verifiers(self):
        """Deleting each constraint family MUST allow a witness-only forgery accepted by the weakened
        circuit, while the intact circuit rejects the exact same witness-only forgery in BOTH Rust & Python."""
        cases = [
            ("drop_mul", dict(drop_mul=True), {"c": (3, +1), "d": (3, +1)}, "E_SUMCHECK"),
            ("drop_add", dict(drop_add=True), {"d": (5, +1)}, "E_SUMCHECK"),
            ("drop_lookup", dict(drop_lookup=True), {"a": (1, +80), "c": (1, +80 * 5), "d": (1, +80 * 6)}, "E_GKR_ROOT"),
        ]
        for name, drop_kw, ovr, expected_code in cases:
            # 1. Weakened circuit accepts the forgery (proving the mutation test is non-vacuous!)
            weak_cir = _build_audited_circuit(**drop_kw, override=ovr)
            art_w = R2.prove_relation2(weak_cir, f"mut-{name}".encode(), queries=72, gkr_masks=4, rng=np.random.default_rng(9))
            rc_w, out_w = _verify_both(art_w.container, art_w.desc, art_w.circuit_root, art_w.stmt_digest)
            self.assertEqual((rc_w, out_w), (0, "VERIFIED"), f"weakened {name} should accept")

            # 2. Full circuit MUST reject the exact same witness-only forgery in BOTH Rust & Python!
            old_force = Z.FORCE
            Z.FORCE = True
            try:
                full_cir = _build_audited_circuit(override=ovr)
                art_f = R2.prove_relation2(full_cir, f"wb-{name}".encode(), queries=72, gkr_masks=4, rng=np.random.default_rng(10))
            finally:
                Z.FORCE = old_force
            rc_f, out_f = _verify_both(art_f.container, art_f.desc, art_f.circuit_root, art_f.stmt_digest)
            self.assertEqual((rc_f, out_f), (1, f"REJECTED: {expected_code}"), f"full circuit failed to reject {name}")


if __name__ == "__main__":
    unittest.main()
