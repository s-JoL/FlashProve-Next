#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline laptop CPU demo for the v1.0 `FPN2` Relation 2 (`LLM_INFERENCE_V2`) fixed-size 8 MiB container.

Demonstrates (in ~3 seconds on any CPU, zero GPU, zero weights):
  1. Compiling a structured hypercube circuit into a canonical binary `CircuitDesc` pinned by
     `circuit_root = BLAKE2b-256("FPN2-CDESC-v2", desc)`;
  2. Proving with ZERO Phase-2 LogUp commitments (`multi_gkr`) + ONE batched multi-size rate-1/8
     coset-4 folded ZK-Basefold opening (`multi_pcs`) into an exact 8,388,608-byte (8 MiB) zero-padded
     `fpn2-llm-std-v1` container;
  3. Verifying independently via BOTH:
       - zero-dependency `#![forbid(unsafe_code)]` Rust CLI (`core/target/release/fpn-verify`), AND
       - detached pure-Python reference verifier (`verifier/python/fpn_verify_ref`, zero prover imports);
  4. Rejecting gate forgery (`E_SUMCHECK`), out-of-range lookup forgery (`E_GKR_ROOT`), nonzero padding
     (`E_PADDING`), truncation (`E_LENGTH`), missing/wrong `--circuit-root` pin (`E_CIRCUIT_UNPINNED` /
     `E_CIRCUIT_PINNED`), and confirming zero shared 64B private blocks across re-proving.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from flashprove_next import gadgets as G, zk_flat as Z  # noqa: E402
from prover.fpn_prover.backend import relation2 as R2  # noqa: E402

RUST_BIN = ROOT / "core" / "target" / "release" / "fpn-verify"


def _build_circuit(cheat_gate: bool = False, cheat_range: bool = False):
    ctx = G.Ctx(True)
    sp = ctx.space("demo", [("x", 6)], padded=False)
    n = sp.content
    a_v = (np.arange(n, dtype=np.int64) * 2 + 1) % 32
    b_v = (np.arange(n, dtype=np.int64) * 3 + 2) % 32
    c_v = a_v * b_v
    if cheat_gate:
        c_v[4] += 1
    if cheat_range:
        a_v[2] = 99
    ca, cb, cc = sp.col("a", a_v), sp.col("b", b_v), sp.col("c", c_v)
    sp.zero(ca.ex * cb.ex - cc.ex, tag="mul")
    tx = np.arange(64, dtype=np.int64)
    idx_ex = Z.Ex.atom(Z.IndexP(list(range(6)), 1))
    lk = ctx.cir.lookups["rng64"] = Z.Lookup("rng64")
    lk.table = (6, [idx_ex])
    lk.table_vals = [tx]
    lk.mult = ctx.cir.vec("mult:rng64", 6, 64, phase=1, tail=0)
    ctx.cir.lookup("rng64", sp.nv, [ca.ex], sel=sp.sel(), vals=[a_v], sel_vals=np.ones(n, dtype=np.int64))
    ctx.cir.lookup("rng64", sp.nv, [cb.ex], sel=sp.sel(), vals=[b_v], sel_vals=np.ones(n, dtype=np.int64))
    return ctx.cir


def _verify_both(container: bytes, desc: bytes, extra_args: list[str]) -> tuple[str, float, float]:
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "proof.fpn2"
        dp = Path(td) / "desc.bin"
        pf.write_bytes(container)
        dp.write_bytes(desc)

        t0 = time.perf_counter()
        rs = subprocess.run(
            [str(RUST_BIN), str(pf), "--circuit-desc", str(dp), *extra_args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        t_rs = (time.perf_counter() - t0) * 1e3

        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "verifier" / "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        t1 = time.perf_counter()
        py = subprocess.run(
            [sys.executable, "-m", "fpn_verify_ref", str(pf), "--circuit-desc", str(dp), *extra_args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        t_py = (time.perf_counter() - t1) * 1e3

        assert (rs.returncode, rs.stdout.strip()) == (py.returncode, py.stdout.strip()), (
            f"Rust {(rs.returncode, rs.stdout.strip())} != Python {(py.returncode, py.stdout.strip())}"
        )
        return rs.stdout.strip(), t_rs, t_py


def main() -> int:
    if not RUST_BIN.is_file():
        print("Building zero-dependency Rust fpn-verify (--release)...")
        subprocess.run(
            ["cargo", "build", "--manifest-path", str(ROOT / "core" / "Cargo.toml"), "--release", "--bin", "fpn-verify"],
            check=True,
        )

    print("=== FlashProve-Next v1.0: FPN2 Relation 2 (LLM_INFERENCE_V2, 8 MiB Fixed-Size, Q=72) CPU Demo ===")
    art = R2.prove_relation2(_build_circuit(), b"v1.0-demo-statement", queries=72, gkr_masks=4)
    pin = art.circuit_root.hex()
    sdig = art.stmt_digest.hex()
    print(f"Container size: {len(art.container):,} B (exact 8 MiB, body_used={art.body_used:,} B, desc={len(art.desc):,} B)")
    print(f"Circuit root:   {pin}")
    print(f"Stmt digest:    {sdig}")

    checks = [
        ("Honest proof (fpn2-llm-std-v1, Q=72)", art.container, ["--profile", "fpn2-llm-std-v1", "--circuit-root", pin, "--stmt-digest", sdig], "VERIFIED"),
        ("Missing --circuit-root pin", art.container, ["--profile", "fpn2-llm-std-v1", "--stmt-digest", sdig], "REJECTED: E_CIRCUIT_UNPINNED"),
        ("Wrong --circuit-root pin", art.container, ["--circuit-root", "00" * 32, "--stmt-digest", sdig], "REJECTED: E_CIRCUIT_PINNED"),
        ("Missing --stmt-digest pin", art.container, ["--circuit-root", pin], "REJECTED: E_STMT_UNPINNED"),
        ("Different response --stmt-digest", art.container, ["--circuit-root", pin, "--stmt-digest", R2.stmt_digest(b"other-reply").hex()], "REJECTED: E_STMT_PINNED"),
        ("Nonzero padding byte", art.container[:-1] + b"\x01", ["--circuit-root", pin, "--stmt-digest", sdig], "REJECTED: E_PADDING"),
        ("Truncated container", art.container[:-1], ["--circuit-root", pin, "--stmt-digest", sdig], "REJECTED: E_LENGTH"),
    ]
    for label, blob, args, exp in checks:
        out, t_rs, t_py = _verify_both(blob, art.desc, args)
        assert out == exp, f"{label}: got {out!r}, expected {exp!r}"
        print(f"  [PASS] {label:36s} -> {out:28s} (Rust {t_rs:5.1f} ms | Python {t_py:5.1f} ms)")

    # Gate forgery
    art_g = R2.prove_relation2(_build_circuit(cheat_gate=True), b"v1.0-bad-gate", queries=72, gkr_masks=4)
    out, t_rs, t_py = _verify_both(art_g.container, art_g.desc, ["--circuit-root", art_g.circuit_root.hex(), "--stmt-digest", art_g.stmt_digest.hex()])
    assert out == "REJECTED: E_SUMCHECK", out
    print(f"  [PASS] {'Witness-only gate forgery':36s} -> {out:28s} (Rust {t_rs:5.1f} ms | Python {t_py:5.1f} ms)")

    # Out-of-table lookup forgery
    Z.FORCE = True
    try:
        art_r = R2.prove_relation2(_build_circuit(cheat_range=True), b"v1.0-bad-range", queries=72, gkr_masks=4)
    finally:
        Z.FORCE = False
    out, t_rs, t_py = _verify_both(art_r.container, art_r.desc, ["--circuit-root", art_r.circuit_root.hex(), "--stmt-digest", art_r.stmt_digest.hex()])
    assert out == "REJECTED: E_GKR_ROOT", out
    print(f"  [PASS] {'Witness-only range forgery':36s} -> {out:28s} (Rust {t_rs:5.1f} ms | Python {t_py:5.1f} ms)")

    # Zero-knowledge re-proving check (0 shared 64B blocks in private body)
    art2 = R2.prove_relation2(_build_circuit(), b"v1.0-demo-statement", queries=72, gkr_masks=4)
    assert len(art.container) == len(art2.container) == 8 * 1024 * 1024
    b1 = art.container[128:64 + art.body_used]
    b2 = art2.container[128:64 + art2.body_used]
    shared = len({b1[i:i + 64] for i in range(0, len(b1) - 63, 64)} & {b2[i:i + 64] for i in range(0, len(b2) - 63, 64)})
    assert shared == 0, f"shared 64B blocks: {shared}"
    print(f"  [PASS] {'ZK re-proving (0 shared 64B)':32s} -> shared={shared}, identical_len={len(art2.container):,} B")
    print("ALL v1.0 RELATION-2 CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
