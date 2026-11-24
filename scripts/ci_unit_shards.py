#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one shard of `tests/unit` so CI can spread the slow suites over parallel jobs.

Usage:
    python scripts/ci_unit_shards.py --list            # print the shard -> module assignment
    python scripts/ci_unit_shards.py <shard>           # run one shard
    python scripts/ci_unit_shards.py all               # run every module (same set as `unittest discover`)

Every `tests/unit/test_*.py` module belongs to exactly one shard: the slow modules are named below and
every other module, including any added later, falls into `fast`.  A named module that no longer exists
is an error, so a rename cannot drop a suite from CI silently.  Each module runs in its own interpreter
with `-W error` and the same `PYTHONPATH` as a plain `unittest discover -s tests/unit` run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "tests" / "unit"
PYTHONPATH = ("prover", "serve", "verifier/python")

# Slow modules only; everything else is `fast`.
SLOW_SHARDS: dict[str, tuple[str, ...]] = {
    "relation6": ("test_relation6_vec_bound_zk",),
    "agentic-e2e": ("test_m3_2_agentic_and_dsa", "test_e2e_real_model_and_zk"),
    "relations-4-5": (
        "test_weight_commit_and_vec_zk",
        "test_relation4_air_and_wbforge",
        "test_relation5_vec_matmul",
    ),
}
FAST = "fast"


def assignment() -> dict[str, list[str]]:
    present = sorted(p.stem for p in UNIT.glob("test_*.py"))
    named = [m for mods in SLOW_SHARDS.values() for m in mods]
    if len(named) != len(set(named)):
        raise SystemExit("FAIL: a module is listed in more than one shard")
    missing = sorted(set(named) - set(present))
    if missing:
        raise SystemExit(f"FAIL: shard table names modules that do not exist in tests/unit: {missing}")
    shards = {name: list(mods) for name, mods in SLOW_SHARDS.items()}
    shards[FAST] = [m for m in present if m not in set(named)]
    return shards


def run_module(module: str) -> tuple[int, float]:
    env = dict(os.environ)
    paths = [str(ROOT / p) for p in PYTHONPATH]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    t0 = time.monotonic()
    rc = subprocess.run(
        [sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", "tests/unit", "-p", f"{module}.py"],
        cwd=ROOT,
        env=env,
    ).returncode
    return rc, time.monotonic() - t0


def main(argv: list[str]) -> int:
    shards = assignment()
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    if argv[1] == "--list":
        for name, mods in shards.items():
            print(f"{name}: {' '.join(mods)}")
        return 0
    if argv[1] == "all":
        modules = sorted(m for mods in shards.values() for m in mods)
    elif argv[1] in shards:
        modules = shards[argv[1]]
    else:
        print(f"FAIL: unknown shard {argv[1]!r}; choose one of {sorted(shards)} or 'all'", file=sys.stderr)
        return 2
    if not modules:
        print(f"FAIL: shard {argv[1]!r} is empty", file=sys.stderr)
        return 1
    # Several suites cross-check the Rust verifier and skip that half when cargo is absent.
    if shutil.which("cargo") is None and not os.environ.get("FPN_ALLOW_NO_RUST"):
        print("FAIL: cargo not found (set FPN_ALLOW_NO_RUST=1 only for explicit Python-only runs)", file=sys.stderr)
        return 1

    failed: list[str] = []
    for module in modules:
        print(f"== {module}", flush=True)
        rc, secs = run_module(module)
        print(f"== {module}: exit {rc} in {secs:.1f} s", flush=True)
        if rc != 0:
            failed.append(module)
    if failed:
        print(f"FAIL: {len(failed)} of {len(modules)} module(s) failed: {' '.join(failed)}", file=sys.stderr)
        return 1
    print(f"OK: {len(modules)} module(s) in shard {argv[1]!r} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
