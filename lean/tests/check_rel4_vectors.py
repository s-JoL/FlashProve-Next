#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Executable parity check of the Lean `verifyBytesRel4` against `specs/vectors/fpn2_folded_air.json`.

Runs the *Lean* verifier (`lean/FlashProveNext/Fpn2Rel4.lean`, oracles instantiated with the BLAKE2b
of `lean/tests/FpnTestLib.lean`, the one `check_rel3_vectors.py` checks against `hashlib`) on every
golden relation-4 container, with the vector's `allow_reduced`, `expected_profile_id` and
`expected_circuit_root`, and compares the printed `VERIFIED` / reject code with the vector's
`expect`. Every container goes through both entry points:

* `verifyBytesRel4Policy` (the relation-4 verifier the refinement theorems are about), and
* `verifyBytesAny` (the relation dispatcher mirroring `core/src/verify.rs::verify_bytes`);

both must print the expected result. In the same Lean run it also checks:

* `B256(canonicalBytes p) == p.profileId` for the three relation-4 profiles of the Lean registry, and
  that each id is the one pinned in the vector file (`core/src/profile.rs::REGISTRY`);
* `64 + Rel4Layout.bodyUsed` against the vector file's `offsets.<profile>.end`;
* `fpn2-folded-air-n20-v1` has no golden container (a 512 KiB proof needs the GPU prover), so its id
  and layout end are pinned here in `EXTRA_PROFILES`, copied from `core/src/profile.rs::REGISTRY`
  and `core/src/folded_air.rs::tests::layout_of_registered_profiles`;
* unless `--no-rel3`: the dispatcher on all relation-3 containers of `fpn2_folded_gkr.json`.

Relation 1 (`fpn2_containers.json`) is not run: its Merkle anchors need Poseidon2, which has no
Lean implementation.

The verifier modules are Lean core only, so only the `lean` binary is needed (no Mathlib, no
`lake`): they are compiled into a temporary directory and the harness is interpreted.

    python3 lean/tests/check_rel4_vectors.py            # all vectors
    python3 lean/tests/check_rel4_vectors.py -k n12     # only vectors whose name contains "n12"
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_rel3_vectors import MODULES as REL3_MODULES, TESTS, VECTORS, build, compile_modules, find_lean  # noqa: E402

MODULES = REL3_MODULES + ["Fpn2Rel4", "Fpn2Rel5"]   # the dispatcher `verifyBytesAny` is in Fpn2Rel5
# Registered relation-4 profiles without a golden container: pinned id (`core/src/profile.rs::REGISTRY`)
# and `64 + body_used` (`core/src/folded_air.rs::tests::layout_of_registered_profiles`: 424 104 + 64).
EXTRA_PROFILES = {
    "fpn2-folded-air-n20-v1": {"id": "966dec06ad5fb40ef7843fe91310684a12378990bbe53483ce7718ef606ba60d",
                               "end": 424168},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="only run vectors whose name contains this substring")
    ap.add_argument("--lean", default=os.environ.get("LEAN", "lean"), help="lean binary (default: $LEAN or lean)")
    ap.add_argument("--no-rel3", action="store_true", help="skip the relation-3 containers through verifyBytesAny")
    ap.add_argument("--keep", action="store_true", help="keep the temporary build directory")
    args = ap.parse_args()

    lean = find_lean(args.lean)
    if lean is None:
        print("SKIP: no `lean` binary found (install elan; toolchain in lean/lean-toolchain)")
        return 2
    with open(os.path.join(VECTORS, "fpn2_folded_air.json")) as f:
        air = json.load(f)
    vectors = air["vectors"]
    selected = [v for v in vectors if args.k in v["name"]]
    pins = {**air["profiles"], **{k: v["id"] for k, v in EXTRA_PROFILES.items()}}
    ends = {**{k: v["end"] for k, v in air["offsets"].items()},
            **{k: v["end"] for k, v in EXTRA_PROFILES.items()}}
    profiles = sorted(pins)
    rel3_vectors, rel3_selected = [], []
    if not args.no_rel3:
        with open(os.path.join(VECTORS, "fpn2_folded_gkr.json")) as f:
            rel3_vectors = json.load(f)["vectors"]
        rel3_selected = [v for v in rel3_vectors if args.k in v["name"]]

    tmp = tempfile.mkdtemp(prefix="fpn-rel4-")
    try:
        env = compile_modules(lean, tmp, MODULES)
        if env is None:
            return 1

        lines = [f"profile {name}" for name in profiles] + [f"layout {name}" for name in profiles]
        for i, v in enumerate(selected):
            path = os.path.join(tmp, f"v{i}.bin")
            with open(path, "wb") as f:
                f.write(build(vectors, v))
            lines.append(f"verify4 {v['name']} {int(v['allow_reduced'])} {v['expected_profile_id'] or '-'} "
                         f"{v['expected_circuit_root'] or '-'} {path}")
        for i, v in enumerate(rel3_selected):
            path = os.path.join(tmp, f"g{i}.bin")
            with open(path, "wb") as f:
                f.write(build(rel3_vectors, v))
            lines.append(f"any {v['name']} {int(v['allow_reduced'])} {v['expected_profile_id'] or '-'} - {path}")
        manifest = os.path.join(tmp, "manifest.txt")
        with open(manifest, "w") as f:
            f.write("\n".join(lines) + "\n")

        t0 = time.time()
        r = subprocess.run([lean, "--run", os.path.join(TESTS, "Rel4Vectors.lean"), manifest],
                           env=env, cwd=os.path.dirname(TESTS), capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            print("FAIL: the Lean harness did not run")
            return 1
        out = [ln for ln in r.stdout.splitlines() if ln.split(" ")[0] in ("profile", "layout", "verify4", "any")]
        if len(out) != len(lines):
            print(r.stdout + r.stderr)
            print(f"FAIL: expected {len(lines)} result lines, got {len(out)}")
            return 1

        bad = 0
        for name, got in zip(profiles, out):
            want = f"profile {name} {pins[name]} {pins[name]}"
            if got != want:
                bad += 1
                print(f"MISMATCH profile id {name}: expected {want!r}, Lean printed {got!r}")
        off = len(profiles)
        for name, got in zip(profiles, out[off:]):
            want = f"layout {name} {ends[name]}"
            if got != want:
                bad += 1
                print(f"MISMATCH layout {name}: expected {want!r}, Lean printed {got!r}")
        off += len(profiles)
        codes, ok4, ok_any = {}, 0, 0
        for v, got in zip(selected, out[off:]):
            f = got.split(" ")
            codes[v["expect"]] = codes.get(v["expect"], 0) + 1
            if len(f) != 4 or f[1] != v["name"]:
                bad += 1
                print(f"MISMATCH {v['name']}: malformed result line {got!r}")
                continue
            if f[2] == v["expect"]:
                ok4 += 1
            else:
                bad += 1
                print(f"MISMATCH {v['name']}: expected {v['expect']}, Lean verifyBytesRel4 printed {f[2]}")
            if f[3] == v["expect"]:
                ok_any += 1
            else:
                bad += 1
                print(f"MISMATCH {v['name']}: expected {v['expect']}, Lean verifyBytesAny printed {f[3]}")
        off += len(selected)
        ok_rel3 = 0
        for v, got in zip(rel3_selected, out[off:]):
            if got == f"any {v['name']} {v['expect']}":
                ok_rel3 += 1
            else:
                bad += 1
                print(f"MISMATCH {v['name']}: expected {v['expect']}, Lean verifyBytesAny printed: {got}")
        print(f"ran {len(profiles)} profile pins, {len(profiles)} layouts, {len(selected)} relation-4 vectors "
              f"(x2 entry points), {len(rel3_selected)} relation-3 vectors (dispatcher) in {time.time() - t0:.1f} s")
        print("expected codes: " + ", ".join(f"{k}×{n}" for k, n in sorted(codes.items())))
        print(f"verifyBytesRel4: {ok4}/{len(selected)}   verifyBytesAny (relation 4): {ok_any}/{len(selected)}   "
              f"verifyBytesAny (relation 3): {ok_rel3}/{len(rel3_selected)}")
        if bad:
            print(f"FAIL: {bad} mismatches")
            return 1
        print(f"OK: Lean verifyBytesRel4 agrees with all {len(selected)} golden vectors "
              f"({ok4}/{len(selected)}; the file holds {len(vectors)})")
        return 0
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
