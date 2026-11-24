#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Executable parity check of the Lean `verifyBytesRel3` against `specs/vectors/fpn2_folded_gkr.json`.

Runs the *Lean* verifier (`lean/FlashProveNext/Fpn2Rel3.lean`, oracles instantiated with the
BLAKE2b of `lean/tests/Rel3Vectors.lean`) on every golden relation-3 container and compares the
printed `VERIFIED` / reject code with the vector's `expect`. It also checks, in the same Lean run:

* the Lean BLAKE2b against `hashlib` (the BLAKE2b KATs of `fpn2_primitives.json` plus personalized
  and multi-block messages), so a wrong oracle cannot hide behind a wrong verifier;
* `B256(canonicalBytes p) == p.profileId` for every profile of the Lean registry (all nine);
* `Fp3.inv` (norm-map inverse in `F_p[u]/(u^3 - 2)`) against an independent Python computation.

The four verifier modules are Lean core only, so only the `lean` binary is needed (no Mathlib, no
`lake`): they and the shared test library `tests/FpnTestLib.lean` (the BLAKE2b) are compiled into a
temporary directory and the harness is interpreted.

    python3 lean/tests/check_rel3_vectors.py            # all vectors
    python3 lean/tests/check_rel3_vectors.py -k n8      # only vectors whose name contains "n8"
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

TESTS = os.path.dirname(os.path.abspath(__file__))
LEAN_DIR = os.path.dirname(TESTS)
REPO = os.path.dirname(LEAN_DIR)
VECTORS = os.path.join(REPO, "specs", "vectors")
MODULES = ["Fpn2Wire", "Fpn2Transcript", "VerifyBytes", "Fpn2Rel3"]
TEST_LIB = "FpnTestLib"   # tests/FpnTestLib.lean: the Lean BLAKE2b shared with Rel4Vectors.lean
PROFILES = ["fpn2-harness-sq-n4", "fpn2-llm-std-v1", "fpn2-folded-gkr-v1", "fpn2-folded-gkr-n8",
            "fpn2-folded-air-v1", "fpn2-folded-air-n12", "fpn2-folded-air-n20-v1",
            "fpn2-folded-vec-n16-v1", "fpn2-folded-vec-n22-v1"]


def find_lean(name):
    return shutil.which(name) or shutil.which(os.path.expanduser("~/.elan/bin/lean"))


def compile_modules(lean, tmp, modules):
    """Compile the Lean-core verifier modules and the shared test library into `tmp/olean`.
    Returns the environment for `lean --run`, or None after printing the compiler output."""
    olean = os.path.join(tmp, "olean")
    os.makedirs(os.path.join(olean, "FlashProveNext"))
    os.makedirs(os.path.join(olean, "tests"))
    env = dict(os.environ, LEAN_PATH=olean)
    t0 = time.time()
    for sub, m in [("FlashProveNext", m) for m in modules] + [("tests", TEST_LIB)]:
        r = subprocess.run([lean, "--root=" + LEAN_DIR, "-o", os.path.join(olean, sub, m + ".olean"),
                            os.path.join(LEAN_DIR, sub, m + ".lean")],
                           env=env, cwd=LEAN_DIR, capture_output=True, text=True)
        if r.returncode != 0 or "error" in r.stdout:
            print(r.stdout + r.stderr)
            print(f"FAIL: {m}.lean does not compile")
            return None
    print(f"compiled {len(modules)} modules + tests/{TEST_LIB} in {time.time() - t0:.1f} s")
    return env


def build(vectors, v):
    """Materialize a vector: `prefix_hex` + zero fill, or `base` + patches / truncate / append."""
    if "prefix_hex" in v:
        pre = bytes.fromhex(v["prefix_hex"])
        return pre + b"\x00" * (v["total_len"] - len(pre))
    b = bytearray(build(vectors, next(x for x in vectors if x["name"] == v["base"])))
    for off, hx in v["patches"]:
        by = bytes.fromhex(hx)
        b[off:off + len(by)] = by
    out = bytes(b)
    if v["truncate"] is not None:
        out = out[:v["truncate"]]
    return out + bytes.fromhex(v["append_hex"])


def hash_cases():
    """(nn, person, msg) triples: the committed KATs plus personalized / block-boundary cases."""
    with open(os.path.join(VECTORS, "fpn2_primitives.json")) as f:
        prim = json.load(f)
    cases = [(k["nn"], bytes.fromhex(k["person"]), bytes.fromhex(k["msg"])) for k in prim["blake2b"]]
    for person in (b"FPN2-FS-v2", b"FPN2-MERKLE-v1"):
        for nn in (32, 64):
            for ln in (0, 1, 64, 97, 127, 128, 129, 255, 256, 257, 400):
                cases.append((nn, person, bytes((7 * i + ln) & 0xFF for i in range(ln))))
    return cases


P = 2**64 - 2**32 + 1
INV_CASES = [(3, 5, 7), (1, 0, 0), (0, 1, 0), (P - 1, 2**32, 123456789), (0, 0, 0)]


def e_mul(a, b):
    return ((a[0] * b[0] + 2 * (a[1] * b[2] + a[2] * b[1])) % P,
            (a[0] * b[1] + a[1] * b[0] + 2 * a[2] * b[2]) % P,
            (a[0] * b[2] + a[1] * b[1] + a[2] * b[0]) % P)


def e_inv(a):
    """a^(P^3 - 2) by square-and-multiply (independent of the norm-map formula)."""
    if a == (0, 0, 0):
        return None
    r, b, e = (1, 0, 0), a, P**3 - 2
    while e:
        if e & 1:
            r = e_mul(r, b)
        b = e_mul(b, b)
        e >>= 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="only run vectors whose name contains this substring")
    ap.add_argument("--lean", default=os.environ.get("LEAN", "lean"), help="lean binary (default: $LEAN or lean)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary build directory")
    args = ap.parse_args()

    lean = find_lean(args.lean)
    if lean is None:
        print("SKIP: no `lean` binary found (install elan; toolchain in lean/lean-toolchain)")
        return 2
    with open(os.path.join(VECTORS, "fpn2_folded_gkr.json")) as f:
        fold = json.load(f)
    vectors = fold["vectors"]
    selected = [v for v in vectors if args.k in v["name"]]

    tmp = tempfile.mkdtemp(prefix="fpn-rel3-")
    try:
        env = compile_modules(lean, tmp, MODULES)
        if env is None:
            return 1

        cases = hash_cases()
        lines = []
        for nn, person, msg in cases:
            lines.append(f"hash {nn} {person.hex() or '-'} {msg.hex() or '-'}")
        lines += [f"fp3inv {a[0]} {a[1]} {a[2]}" for a in INV_CASES]
        lines += [f"profile {name}" for name in PROFILES]
        for i, v in enumerate(selected):
            path = os.path.join(tmp, f"v{i}.bin")
            with open(path, "wb") as f:
                f.write(build(vectors, v))
            lines.append(f"verify {v['name']} {int(v['allow_reduced'])} {v['expected_profile_id'] or '-'} {path}")
        manifest = os.path.join(tmp, "manifest.txt")
        with open(manifest, "w") as f:
            f.write("\n".join(lines) + "\n")

        t0 = time.time()
        r = subprocess.run([lean, "--run", os.path.join(TESTS, "Rel3Vectors.lean"), manifest],
                           env=env, cwd=LEAN_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            print("FAIL: the Lean harness did not run")
            return 1
        out = [ln for ln in r.stdout.splitlines() if ln.split(" ")[0] in ("hash", "fp3inv", "profile", "verify")]
        if len(out) != len(lines):
            print(r.stdout + r.stderr)
            print(f"FAIL: expected {len(lines)} result lines, got {len(out)}")
            return 1

        bad = 0
        for (nn, person, msg), got in zip(cases, out):
            want = hashlib.blake2b(msg, digest_size=nn, person=person).hexdigest()
            if got != f"hash {want}":
                bad += 1
                print(f"MISMATCH blake2b nn={nn} person={person!r} len={len(msg)}: {got}")
        off = len(cases)
        for a, got in zip(INV_CASES, out[off:]):
            inv = e_inv(a)
            want = "fp3inv none" if inv is None else f"fp3inv {inv[0]} {inv[1]} {inv[2]} true"
            if got != want:
                bad += 1
                print(f"MISMATCH Fp3.inv{a}: expected {want!r}, Lean printed {got!r}")
        off += len(INV_CASES)
        for name, got in zip(PROFILES, out[off:]):
            f = got.split()
            if len(f) != 4 or f[2] != f[3]:
                bad += 1
                print(f"MISMATCH profile id {name}: {got}")
        off += len(PROFILES)
        codes = {}
        for v, got in zip(selected, out[off:]):
            want = f"verify {v['name']} {v['expect']}"
            codes[v["expect"]] = codes.get(v["expect"], 0) + 1
            if got != want:
                bad += 1
                print(f"MISMATCH {v['name']}: expected {v['expect']}, Lean printed: {got}")
        print(f"ran {len(cases)} BLAKE2b cases, {len(INV_CASES)} Fp3.inv cases, {len(PROFILES)} profile pins, {len(selected)} vectors "
              f"in {time.time() - t0:.1f} s")
        print("expected codes: " + ", ".join(f"{k}×{n}" for k, n in sorted(codes.items())))
        if bad:
            print(f"FAIL: {bad} mismatches")
            return 1
        print(f"OK: Lean verifyBytesRel3 agrees with all {len(selected)} golden vectors")
        return 0
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
