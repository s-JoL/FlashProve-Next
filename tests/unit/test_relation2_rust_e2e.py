# SPDX-License-Identifier: Apache-2.0
"""End-to-end cross-language test for `FPN2` Relation 2 (`LLM_INFERENCE_V2`):
Python prover (`relation2.prove_relation2`) -> exact 8 MiB zero-padded `fpn2-llm-std-v1` container ->
standalone zero-dependency Rust CLI (`core/target/release/fpn-verify --circuit-desc ... --circuit-root ...`).
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


def _ensure_rust_bin():
    subprocess.run(
        ["cargo", "build", "--manifest-path", os.path.join(ROOT, "core", "Cargo.toml"), "--release", "--bin", "fpn-verify"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert os.path.isfile(BIN), BIN


def _run_cli(proof_bytes: bytes, desc_bytes: bytes, extra_args: list[str]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as td:
        pf_path = os.path.join(td, "proof.fpn2")
        desc_path = os.path.join(td, "desc.bin")
        with open(pf_path, "wb") as f:
            f.write(proof_bytes)
        with open(desc_path, "wb") as f:
            f.write(desc_bytes)
        cmd = [BIN, pf_path, "--circuit-desc", desc_path] + extra_args
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout.strip()


def _run_py_ref(proof_bytes: bytes, desc_bytes: bytes, extra_args: list[str]) -> tuple[int, str]:
    """Run completely detached Python reference verifier (`-I`, zero prover imports)."""
    with tempfile.TemporaryDirectory() as td:
        pf_path = os.path.join(td, "proof.fpn2")
        desc_path = os.path.join(td, "desc.bin")
        with open(pf_path, "wb") as f:
            f.write(proof_bytes)
        with open(desc_path, "wb") as f:
            f.write(desc_bytes)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "verifier", "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [
            "/opt/homebrew/bin/python3.14" if os.path.exists("/opt/homebrew/bin/python3.14") else "python3",
            "-m", "fpn_verify_ref", pf_path, "--circuit-desc", desc_path,
        ] + extra_args
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        return p.returncode, p.stdout.strip()


def _run_both(proof_bytes: bytes, desc_bytes: bytes, extra_args: list[str]) -> tuple[int, str]:
    rc_rs, out_rs = _run_cli(proof_bytes, desc_bytes, extra_args)
    rc_py, out_py = _run_py_ref(proof_bytes, desc_bytes, extra_args)
    assert (rc_rs, out_rs) == (rc_py, out_py), f"Rust ({rc_rs}, {out_rs!r}) != Python ({rc_py}, {out_py!r})"
    return rc_rs, out_rs
    with tempfile.TemporaryDirectory() as td:
        pf_path = os.path.join(td, "proof.fpn2")
        desc_path = os.path.join(td, "desc.bin")
        with open(pf_path, "wb") as f:
            f.write(proof_bytes)
        with open(desc_path, "wb") as f:
            f.write(desc_bytes)
        cmd = [BIN, pf_path, "--circuit-desc", desc_path] + extra_args
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout.strip()


def _build_r2_circuit(cheat_gate=False, cheat_range=False):
    """Self-contained structured hypercube circuit with zerocheck, sumzero, and linear-numerator lookup."""
    ctx = G.Ctx(True)
    sp = ctx.space("m64", [("x", 6)], padded=False)  # 64 rows, content = 32
    N = sp.content
    a_v = np.arange(N, dtype=np.int64) % 16
    b_v = (np.arange(N, dtype=np.int64) * 3 + 1) % 16
    c_v = a_v * b_v
    if cheat_gate:
        c_v[4] += 1
    if cheat_range:
        a_v[2] = 77  # outside [0, 64)

    ca = sp.col("a", a_v)
    cb = sp.col("b", b_v)
    cc = sp.col("c", c_v)
    sp.zero(ca.ex * cb.ex - cc.ex, tag="mul")
    # Shifted identity on a: a[i] - a[i] == 0 via shifted view sumzero
    sp.sumzero(ca.ex - ca.ex, tag="sum0")

    # Full-power-of-two table [0, 64) with tail=0 (group-level multi_pcs tail blinds it!)
    tx = np.arange(64, dtype=np.int64)
    idx_ex = Z.Ex.atom(Z.IndexP(list(range(6)), 1))
    lk = ctx.cir.lookups["rng64"] = Z.Lookup("rng64")
    lk.table = (6, [idx_ex])
    lk.table_vals = [tx]
    lk.mult = ctx.cir.vec("mult:rng64", 6, 64, phase=1, tail=0)
    # Lookup query on a_v and b_v (with linear SelLt content mask)
    ctx.cir.lookup("rng64", sp.nv, [ca.ex], sel=sp.sel(), vals=[a_v], sel_vals=np.ones(N, dtype=np.int64))
    ctx.cir.lookup("rng64", sp.nv, [cb.ex], sel=sp.sel(), vals=[b_v], sel_vals=np.ones(N, dtype=np.int64))
    return ctx.cir


class TestRelation2RustE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_rust_bin()

    def test_honest_8mib_fixed_container_verified_by_rust_cli(self):
        cir = _build_r2_circuit()
        art = R2.prove_relation2(cir, b"stmt-v1-honest", queries=72, gkr_masks=4, rng=np.random.default_rng(42))
        self.assertEqual(len(art.container), 8 * 1024 * 1024)
        pin = art.circuit_root.hex()
        sdig = art.stmt_digest.hex()

        rc, out = _run_both(
            art.container, art.desc,
            ["--profile", "fpn2-llm-std-v1", "--circuit-root", pin, "--stmt-digest", sdig],
        )
        self.assertEqual((rc, out), (0, "VERIFIED"))

        # Negative 1: missing --circuit-root -> E_CIRCUIT_UNPINNED
        rc, out = _run_both(art.container, art.desc, ["--profile", "fpn2-llm-std-v1", "--stmt-digest", sdig])
        self.assertEqual((rc, out), (1, "REJECTED: E_CIRCUIT_UNPINNED"))

        # Negative 2: wrong --circuit-root -> E_CIRCUIT_PINNED
        rc, out = _run_both(art.container, art.desc, ["--circuit-root", "00" * 32, "--stmt-digest", sdig])
        self.assertEqual((rc, out), (1, "REJECTED: E_CIRCUIT_PINNED"))

        # Negative 3 (Finding #5): missing --stmt-digest -> E_STMT_UNPINNED
        rc, out = _run_both(art.container, art.desc, ["--circuit-root", pin])
        self.assertEqual((rc, out), (1, "REJECTED: E_STMT_UNPINNED"))

        # Negative 4 (Finding #5): same circuit, DIFFERENT response digest -> E_STMT_PINNED
        other_sdig = R2.stmt_digest(b"stmt-v1-DIFFERENT-response").hex()
        rc, out = _run_both(art.container, art.desc, ["--circuit-root", pin, "--stmt-digest", other_sdig])
        self.assertEqual((rc, out), (1, "REJECTED: E_STMT_PINNED"))

        # Negative 5: unsupported --x-root pin -> E_X_ROOT_PINNED
        rc, out = _run_both(
            art.container, art.desc,
            ["--circuit-root", pin, "--stmt-digest", sdig, "--x-root", "11" * 32],
        )
        self.assertEqual((rc, out), (1, "REJECTED: E_X_ROOT_PINNED"))

        # Negative 6: nonzero byte in padding tail -> E_PADDING
        bad_pad = bytearray(art.container)
        bad_pad[-1] = 1
        rc, out = _run_both(bytes(bad_pad), art.desc, ["--circuit-root", pin, "--stmt-digest", sdig])
        self.assertEqual((rc, out), (1, "REJECTED: E_PADDING"))

        # Negative 7: truncated container -> E_LENGTH
        rc, out = _run_both(art.container[:-1], art.desc, ["--circuit-root", pin, "--stmt-digest", sdig])
        self.assertEqual((rc, out), (1, "REJECTED: E_LENGTH"))

        # Negative 8: tamper stmt_digest inside container -> E_STMT_PINNED
        bad_stmt = bytearray(art.container)
        bad_stmt[96] ^= 1
        rc, out = _run_both(bytes(bad_stmt), art.desc, ["--circuit-root", pin, "--stmt-digest", sdig])
        self.assertEqual((rc, out), (1, "REJECTED: E_STMT_PINNED"))

        # Negative 9 (Finding #2): queries=1 descriptor under STANDARD profile -> E_REDUCED_SECURITY
        art_weak = R2.prove_relation2(
            _build_r2_circuit(), b"stmt-weak-q1", queries=1, gkr_masks=4,
            rng=np.random.default_rng(19), _allow_weak_queries_for_test=True,
        )
        rc, out = _run_both(
            art_weak.container, art_weak.desc,
            ["--profile", "fpn2-llm-std-v1", "--circuit-root", art_weak.circuit_root.hex(),
             "--stmt-digest", art_weak.stmt_digest.hex()],
        )
        self.assertEqual((rc, out), (1, "REJECTED: E_REDUCED_SECURITY"))

        # Negative 10 (Finding #5 Round-2): BOTH --stmt-digest and --stmt-file in EITHER order -> exit 2 in BOTH CLIs!
        with tempfile.NamedTemporaryFile(delete=False) as sf:
            sf.write(b"stmt-v1-honest")
            sf_path = sf.name
        try:
            # First confirm --stmt-file alone succeeds in both CLIs
            rc, out = _run_both(
                art.container, art.desc,
                ["--profile", "fpn2-llm-std-v1", "--circuit-root", pin, "--stmt-file", sf_path],
            )
            self.assertEqual((rc, out), (0, "VERIFIED"))
            # Order A: --stmt-digest then --stmt-file -> exit 2 in both
            rc_rs, _ = _run_cli(art.container, art.desc, ["--circuit-root", pin, "--stmt-digest", sdig, "--stmt-file", sf_path])
            rc_py, _ = _run_py_ref(art.container, art.desc, ["--circuit-root", pin, "--stmt-digest", sdig, "--stmt-file", sf_path])
            self.assertEqual((rc_rs, rc_py), (2, 2))
            # Order B: --stmt-file then --stmt-digest -> exit 2 in both
            rc_rs, _ = _run_cli(art.container, art.desc, ["--circuit-root", pin, "--stmt-file", sf_path, "--stmt-digest", sdig])
            rc_py, _ = _run_py_ref(art.container, art.desc, ["--circuit-root", pin, "--stmt-file", sf_path, "--stmt-digest", sdig])
            self.assertEqual((rc_rs, rc_py), (2, 2))
            # Repeated identical flags (--stmt-digest x2 or --stmt-file x2) -> exit 2 in both
            other_dig = "00" * 32
            rc_rs, _ = _run_cli(art.container, art.desc, ["--circuit-root", pin, "--stmt-digest", other_dig, "--stmt-digest", sdig])
            rc_py, _ = _run_py_ref(art.container, art.desc, ["--circuit-root", pin, "--stmt-digest", other_dig, "--stmt-digest", sdig])
            self.assertEqual((rc_rs, rc_py), (2, 2))
            rc_rs, _ = _run_cli(art.container, art.desc, ["--circuit-root", pin, "--stmt-file", sf_path, "--stmt-file", sf_path])
            rc_py, _ = _run_py_ref(art.container, art.desc, ["--circuit-root", pin, "--stmt-file", sf_path, "--stmt-file", sf_path])
            self.assertEqual((rc_rs, rc_py), (2, 2))
        finally:
            os.unlink(sf_path)

    def test_cheating_witnesses_rejected_by_rust_cli(self):
        # 1. Gate violation -> E_SUMCHECK
        cir_g = _build_r2_circuit(cheat_gate=True)
        art_g = R2.prove_relation2(cir_g, b"stmt-bad-gate", queries=72, gkr_masks=4, rng=np.random.default_rng(7))
        rc, out = _run_both(
            art_g.container, art_g.desc,
            ["--circuit-root", art_g.circuit_root.hex(), "--stmt-digest", art_g.stmt_digest.hex()],
        )
        self.assertEqual((rc, out), (1, "REJECTED: E_SUMCHECK"))

        # 2. Out-of-range lookup -> E_GKR_ROOT
        old_force = Z.FORCE
        Z.FORCE = True
        try:
            cir_r = _build_r2_circuit(cheat_range=True)
            art_r = R2.prove_relation2(cir_r, b"stmt-bad-range", queries=72, gkr_masks=4, rng=np.random.default_rng(8))
        finally:
            Z.FORCE = old_force
        rc, out = _run_both(
            art_r.container, art_r.desc,
            ["--circuit-root", art_r.circuit_root.hex(), "--stmt-digest", art_r.stmt_digest.hex()],
        )
        self.assertEqual((rc, out), (1, "REJECTED: E_GKR_ROOT"))

    def test_reproving_fixed_length_and_zero_shared_64b_blocks(self):
        art1 = R2.prove_relation2(_build_r2_circuit(), b"stmt-zk", queries=72, gkr_masks=4)
        art2 = R2.prove_relation2(_build_r2_circuit(), b"stmt-zk", queries=72, gkr_masks=4)
        self.assertEqual(len(art1.container), len(art2.container))
        self.assertEqual(art1.circuit_root, art2.circuit_root)
        # Private proof body (excluding 64B header + 64B circuit_root/stmt_digest and zero padding tail)
        b1 = art1.container[128:64 + art1.body_used]
        b2 = art2.container[128:64 + art2.body_used]
        s1 = {b1[i:i + 64] for i in range(0, len(b1) - 63, 64)}
        s2 = {b2[i:i + 64] for i in range(0, len(b2) - 63, 64)}
        self.assertEqual(len(s1 & s2), 0)


if __name__ == "__main__":
    unittest.main()
