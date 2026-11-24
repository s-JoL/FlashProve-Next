#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Executable parity check of the Lean `verifyBytesRel5` against `specs/vectors/fpn2_folded_vec.json`.

Runs the *Lean* verifier (`lean/FlashProveNext/Fpn2Rel5.lean`, oracles instantiated with the BLAKE2b
of `lean/tests/FpnTestLib.lean`, the one `check_rel3_vectors.py` checks against `hashlib`) on every
golden relation-5 container, with the vector's `allow_reduced`, `expected_profile_id` and
`expected_circuit_root` (the pinned `weight_root`), and compares the printed `VERIFIED` / reject
code with the vector's `expect`. Every container goes through both entry points:

* `verifyBytesRel5Policy` (the relation-5 verifier the refinement theorems are about), and
* `verifyBytesAny` (the four-relation dispatcher mirroring `core/src/verify.rs::verify_bytes`);

both must print the expected result. In the same Lean run it also checks:

* all nine profiles of the Lean registry: `B256(canonicalBytes p) == p.profileId`, that this id is
  the one pinned in `core/src/profile.rs::REGISTRY` (copied into `REGISTRY` below; for the two
  relation-5 profiles also compared with the vector file's `profiles`), and the relation number;
* `64 + Rel5Layout.bodyUsed` for both relation-5 profiles: `fpn2-folded-vec-n16-v1` against the
  vector file's `offsets.<profile>.end`; `fpn2-folded-vec-n22-v1` has no golden container (a 512 KiB
  proof of 64 tiles needs the GPU prover), so its end `469 184 + 64` is pinned here, copied from
  `core/src/folded_vec.rs::tests::layout_of_registered_profiles`;
* unless `--no-weights` or the Python reference verifier is absent: `finalWeight5 (rel5Terms ..)`
  against `verifier/python/fpn_verify_ref/folded_vec.py::_terms` + `_weights_eval` on random
  challenges, at `(n, k) = (16, 10)` and at `(22, 10)`, `(14, 5)`, `(18, 8)` where `beta > 0`.
  All 33 golden containers have `beta = 0` (one tile), so apart from the next item this is the only
  execution of the tile-dependent parts of `rel5Terms` (`s = s_j ‖ s_b` split, `eq(r_b, s_b)`, the
  `B·K` offset);
* the multi-tile fixture `lean/tests/rel5_multitile.json` (self-contained: no file outside `lean/` is
  read for it). It holds an accepted container and a false-statement container of a test-only,
  UNREGISTERED relation-5 profile with `n = 14`, `K = 64`, `beta = 2` (four tiles, `d = 8`), produced
  once by `gen_rel5_multitile.py` with the CPU prover, a third container whose folded layer 1 is
  inconsistent under a valid Merkle root (`E_FOLD`), plus tamper cases whose expected codes come
  from the Python reference verifier. The Lean `verifyParsedRel5` (`parseContainerRel5`, weight pin
  and `rel5Check`: what `verifyBytesRel5` returns on a registered profile whose id the header names,
  `verifyBytesRel5_eq_verifyParsedRel5`) must return the expected result on every case, and the challenges Lean squeezes on the accepted container
  (`r_i`, the two tile challenges `r_b`, `lambda`, the eight `s`, `mu`) must equal the prover's.
  This executes the non-empty `r_b` squeeze, the multi-tile body parser and `finalWeight5` with
  `beta > 0` end to end. `fpn2-folded-vec-n22-v1` bytes (`beta = 6`) are still not executed;
  `verifyBytesRel5_foldedVecN22` proves that they would go through these same functions.

The verifier modules are Lean core only, so only the `lean` binary is needed (no Mathlib, no
`lake`): they are compiled into a temporary directory and the harness is interpreted.

    python3 lean/tests/check_rel5_vectors.py            # all vectors
    python3 lean/tests/check_rel5_vectors.py -k flip    # only vectors whose name contains "flip"
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_rel3_vectors import MODULES as REL3_MODULES, TESTS, VECTORS, build, compile_modules, find_lean  # noqa: E402

MODULES = REL3_MODULES + ["Fpn2Rel4", "Fpn2Rel5"]
# `core/src/profile.rs::REGISTRY`: name -> (pinned id, relation number).
REGISTRY = {
    "fpn2-harness-sq-n4": ("e24f718cf8c2f4d21417f9e8a3038b9993aa418ca0f7205a5cf6a8c53a3442db", 1),
    "fpn2-llm-std-v1": ("588659a6e46c51c021cd65dd287c8968562b637d20afaf0d68f4ad57b16db9a1", 2),
    "fpn2-folded-gkr-v1": ("e238db306348de19561c2fb4896ea4a249bd8ce7bd41beb1f715b6fcb305e986", 3),
    "fpn2-folded-gkr-n8": ("d7a3d169b65f8c5e328911f4a7edb3750c3244ca0202a9b259a7325d5705c147", 3),
    "fpn2-folded-air-v1": ("f6c6b4ca3f4292e6e8917d1285f6105bde168ed7517b71aa65e5c562933d0a97", 4),
    "fpn2-folded-air-n12": ("5f661ec4ea1ab8b67de2ed9d13a13d148b4ed639a242d30c81fdb4f2705ced7d", 4),
    "fpn2-folded-air-n20-v1": ("966dec06ad5fb40ef7843fe91310684a12378990bbe53483ce7718ef606ba60d", 4),
    "fpn2-folded-vec-n16-v1": ("a48dfb0a1b0f17580cbab6b756678238200c1b3028a26f92898430d8da801327", 5),
    "fpn2-folded-vec-n22-v1": ("ba72066888c8fc3cf0ca31c86527a5292759eb629c0eed131d812623b3e6b3ef", 5),
}
# Registered relation-5 profile without a golden container: `64 + body_used`
# (`core/src/folded_vec.rs::tests::layout_of_registered_profiles`: 469 184 + 64).
EXTRA_ENDS = {"fpn2-folded-vec-n22-v1": 469248}
# (n, k, seed) of the `finalWeight5` differential cases; beta = n - 6 - k.
WEIGHT_CASES = [(16, 10, 1), (22, 10, 2), (22, 10, 3), (14, 5, 4), (18, 8, 5)]
P = 0xFFFFFFFF00000001


def weight_cases():
    """[(name, manifest line, expected result line)] from the Python reference verifier, or None."""
    ref = os.path.join(os.path.dirname(os.path.dirname(TESTS)), "verifier", "python")
    if not os.path.isdir(os.path.join(ref, "fpn_verify_ref")):
        return None
    sys.path.insert(0, ref)
    from fpn_verify_ref.field import ONE3, e_add, e_mul, e_sub
    from fpn_verify_ref.folded_air import _weights_eval
    from fpn_verify_ref.folded_vec import VecLayout, _terms
    out = []
    for n, k, seed in WEIGHT_CASES:
        rng = random.Random(seed)
        lay = VecLayout(n, 1, k)
        el = lambda: tuple(rng.randrange(P) for _ in range(3))  # noqa: E731
        r_i = [el() for _ in range(5)]
        r_b = [el() for _ in range(lay.beta)]
        s = [el() for _ in range(lay.d)]
        lam, e_w, mu = el(), el(), el()
        rs = [el() for _ in range(n + 2)]
        fb, sparse, cb = _terms(lay, r_i, r_b, s, lam, e_w, mu)
        theta, r1, rest = rs[0], rs[1], rs[2:]
        a_r = _weights_eval(n, fb, sparse, rest)
        c_r = _weights_eval(n, cb, [], rest)
        w = e_mul(e_sub(ONE3, r1), e_add(e_mul(e_sub(ONE3, theta), a_r), e_mul(e_sub(e_add(theta, theta), ONE3), c_r)))
        name = f"n{n}_k{k}_beta{lay.beta}_seed{seed}"
        limbs = " ".join(str(x) for e in r_i + r_b + s + [lam, e_w, mu] + rs for x in e)
        out.append((name, f"weight {name} {n} {k} {limbs}", f"weight {name} {w[0]} {w[1]} {w[2]}"))
    return out


def multitile_cases(tmp):
    """(fixture, [manifest lines], [expected result lines], number of tile5 cases) of the multi-tile fixture."""
    path = os.path.join(TESTS, "rel5_multitile.json")
    with open(path) as f:
        fx = json.load(f)
    pr, lay = fx["profile"], fx["layout"]
    args = f"{pr['name']} {pr['tier']} {pr['bits']} {pr['n_vars']} {pr['n_queries']} {pr['max_hidden']} {pr['c_p']}"
    bases = {k: bytes.fromhex(v["prefix_hex"]) + b"\x00" * (v["total_len"] - len(v["prefix_hex"]) // 2)
             for k, v in fx["containers"].items()}
    lines = [f"tprof {args}"]
    want = [f"tprof {pr['name']} {pr['id']} {lay['beta']} {lay['tiles']} {lay['d']} {lay['end']}"]
    for i, c in enumerate(fx["cases"]):
        b = bytearray(bases[c["base"]])
        for off, hx in c["patches"]:
            by = bytes.fromhex(hx)
            b[off:off + len(by)] = by
        out = bytes(b)
        if c["truncate"] is not None:
            out = out[:c["truncate"]]
        out += bytes.fromhex(c["append_hex"])
        fpath = os.path.join(tmp, f"mt{i}.bin")
        with open(fpath, "wb") as f:
            f.write(out)
        lines.append(f"tile5 {c['name']} {args} {c['pin'] or '-'} {fpath}")
        want.append(f"tile5 {c['name']} {c['expect']}")
    good = os.path.join(tmp, "mt_good.bin")
    with open(good, "wb") as f:
        f.write(bases["good"])
    ch = fx["challenges"]
    seq = ch["r_i"] + ch["r_b"] + [ch["lam"]] + ch["s"] + [ch["mu"]]
    lines.append(f"tchal {args} {good}")
    want.append(f"tchal {pr['name']} " + " ".join(str(x) for e in seq for x in e))
    return fx, lines, want, len(fx["cases"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="only run vectors whose name contains this substring")
    ap.add_argument("--lean", default=os.environ.get("LEAN", "lean"), help="lean binary (default: $LEAN or lean)")
    ap.add_argument("--no-weights", action="store_true", help="skip the finalWeight5 differential cases")
    ap.add_argument("--keep", action="store_true", help="keep the temporary build directory")
    args = ap.parse_args()

    lean = find_lean(args.lean)
    if lean is None:
        print("SKIP: no `lean` binary found (install elan; toolchain in lean/lean-toolchain)")
        return 2
    with open(os.path.join(VECTORS, "fpn2_folded_vec.json")) as f:
        vec = json.load(f)
    vectors = vec["vectors"]
    selected = [v for v in vectors if args.k in v["name"]]
    profiles = list(REGISTRY)
    ends = {**{k: v["end"] for k, v in vec["offsets"].items()}, **EXTRA_ENDS}
    layouts = sorted(vec["profiles"])
    weights = [] if args.no_weights else weight_cases()
    if weights is None:
        print("NOTE: verifier/python/fpn_verify_ref not found, finalWeight5 differential cases not run")
        weights = []

    bad = 0
    for name, pin in vec["profiles"].items():
        if REGISTRY.get(name, ("", 0))[0] != pin:
            bad += 1
            print(f"MISMATCH {name}: vector file pins {pin}, REGISTRY here {REGISTRY.get(name)}")

    tmp = tempfile.mkdtemp(prefix="fpn-rel5-")
    try:
        env = compile_modules(lean, tmp, MODULES)
        if env is None:
            return 1

        lines = [f"profile {name}" for name in profiles] + [f"layout {name}" for name in layouts]
        for i, v in enumerate(selected):
            path = os.path.join(tmp, f"v{i}.bin")
            with open(path, "wb") as f:
                f.write(build(vectors, v))
            lines.append(f"verify5 {v['name']} {int(v['allow_reduced'])} {v['expected_profile_id'] or '-'} "
                         f"{v['expected_circuit_root'] or '-'} {path}")
        lines += [w[1] for w in weights]
        fx, mt_lines, mt_want, mt_n = multitile_cases(tmp)
        lines += mt_lines
        manifest = os.path.join(tmp, "manifest.txt")
        with open(manifest, "w") as f:
            f.write("\n".join(lines) + "\n")

        t0 = time.time()
        r = subprocess.run([lean, "--run", os.path.join(TESTS, "Rel5Vectors.lean"), manifest],
                           env=env, cwd=os.path.dirname(TESTS), capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            print("FAIL: the Lean harness did not run")
            return 1
        out = [ln for ln in r.stdout.splitlines() if ln.split(" ")[0] in ("profile", "layout", "verify5", "weight", "tprof", "tile5", "tchal")]
        if len(out) != len(lines):
            print(r.stdout + r.stderr)
            print(f"FAIL: expected {len(lines)} result lines, got {len(out)}")
            return 1

        ok_pins = 0
        for name, got in zip(profiles, out):
            pin, rel = REGISTRY[name]
            want = f"profile {name} {pin} {pin} {rel}"
            if got == want:
                ok_pins += 1
            else:
                bad += 1
                print(f"MISMATCH profile id {name}: expected {want!r}, Lean printed {got!r}")
        off = len(profiles)
        for name, got in zip(layouts, out[off:]):
            want = f"layout {name} {ends[name]}"
            if got != want:
                bad += 1
                print(f"MISMATCH layout {name}: expected {want!r}, Lean printed {got!r}")
        off += len(layouts)
        codes, ok5, ok_any = {}, 0, 0
        for v, got in zip(selected, out[off:]):
            f = got.split(" ")
            codes[v["expect"]] = codes.get(v["expect"], 0) + 1
            if len(f) != 4 or f[1] != v["name"]:
                bad += 1
                print(f"MISMATCH {v['name']}: malformed result line {got!r}")
                continue
            if f[2] == v["expect"]:
                ok5 += 1
            else:
                bad += 1
                print(f"MISMATCH {v['name']}: expected {v['expect']}, Lean verifyBytesRel5 printed {f[2]}")
            if f[3] == v["expect"]:
                ok_any += 1
            else:
                bad += 1
                print(f"MISMATCH {v['name']}: expected {v['expect']}, Lean verifyBytesAny printed {f[3]}")
        off += len(selected)
        ok_w = 0
        for (name, _, want), got in zip(weights, out[off:]):
            if got == want:
                ok_w += 1
            else:
                bad += 1
                print(f"MISMATCH finalWeight5 {name}: expected {want!r}, Lean printed {got!r}")
        off += len(weights)
        ok_mt, mt_codes = 0, {}
        for want, got in zip(mt_want, out[off:]):
            f = want.split(" ")
            if f[0] == "tile5":
                mt_codes[f[2]] = mt_codes.get(f[2], 0) + 1
            if got == want:
                ok_mt += f[0] == "tile5"
            else:
                bad += 1
                show = (lambda x: x if len(x) < 200 else x[:200] + " ...")
                print(f"MISMATCH multi-tile {f[0]} {f[1]}: expected {show(want)!r}, Lean printed {show(got)!r}")
        ok_prof = out[off] == mt_want[0]
        ok_chal = out[off + len(mt_want) - 1] == mt_want[-1]
        lay = fx["layout"]
        print(f"ran {len(profiles)} profile pins, {len(layouts)} layouts, {len(selected)} relation-5 vectors "
              f"(x2 entry points), {len(weights)} finalWeight5 cases in {time.time() - t0:.1f} s")
        print("expected codes: " + ", ".join(f"{k}×{n}" for k, n in sorted(codes.items())))
        print(f"profile pins: {ok_pins}/{len(profiles)}   verifyBytesRel5: {ok5}/{len(selected)}   "
              f"verifyBytesAny (relation 5): {ok_any}/{len(selected)}   "
              f"finalWeight5 vs Python reference: {ok_w}/{len(weights)}")
        print(f"multi-tile fixture {fx['profile']['name']} (n = {lay['n']}, k = {lay['k']}, beta = {lay['beta']}, "
              f"{lay['tiles']} tiles, d = {lay['d']}; unregistered test profile): "
              f"layout / profile id {'ok' if ok_prof else 'MISMATCH'}, "
              f"parseContainerRel5 + pin + rel5Check: {ok_mt}/{mt_n} "
              f"({', '.join(f'{k}×{n}' for k, n in sorted(mt_codes.items()))}), "
              f"transcript challenges r_i (5), r_b ({lay['beta']}), lambda, s ({lay['d']}), mu vs prover: "
              f"{'equal' if ok_chal else 'MISMATCH'}")
        if bad:
            print(f"FAIL: {bad} mismatches")
            return 1
        print(f"OK: Lean verifyBytesRel5 agrees with all {len(selected)} golden vectors "
              f"({ok5}/{len(selected)}; the file holds {len(vectors)}), all {len(profiles)} profile pins hold")
        print(f"OK: multi-tile relation 5 (beta = {lay['beta']}, {lay['tiles']} tiles): {ok_mt}/{mt_n} cases, "
              f"non-empty r_b squeeze equal to the prover's")
        return 0
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
