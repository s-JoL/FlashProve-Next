# SPDX-License-Identifier: Apache-2.0
"""Build the optional native CPU accelerator ``_fast_cpu.so`` from ``_fast_cpu.cpp``.

    python -m flashprove_next.build_ext [--cxx g++] [--quiet]

The library speeds up Goldilocks vector arithmetic, extension-field dot products, eq tables, FRI folding and the
Poseidon2 permutation on the CPU.  Every entry point has a NumPy fallback, so a missing compiler is not an error:
the command reports it and exits 0.  ``flashprove_next.field`` calls :func:`compile_native` on first use when the
library is missing or older than its source (disable with ``FPN_NO_BUILD=1``).
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_fast_cpu.cpp")
SO = os.path.join(HERE, "_fast_cpu.so")
BASE_FLAGS = ["-O3", "-shared", "-std=c++17", "-fPIC"]


def find_cxx(cxx=None):
    return cxx or os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


def compile_native(cxx=None, verbose=False, timeout=300):
    """Compile ``_fast_cpu.cpp`` into ``_fast_cpu.so`` (atomic replace).  Returns (ok, message)."""
    cxx = find_cxx(cxx)
    if not cxx:
        return False, "no C++ compiler found (set CXX); the NumPy fallback will be used"
    if not os.path.exists(SRC):
        return False, f"source not found: {SRC}"
    if not os.access(HERE, os.W_OK):
        return False, f"package directory is not writable: {HERE}"
    attempts = [BASE_FLAGS]
    if sys.platform.startswith("linux"):
        attempts.insert(0, BASE_FLAGS + ["-fopenmp"])
    tmp = SO + ".%d.tmp" % os.getpid()
    last = ""
    for flags in attempts:
        cmd = [cxx, *flags, SRC, "-o", tmp]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            last = f"{type(e).__name__}: {e}"
            continue
        if r.returncode == 0:
            os.replace(tmp, SO)
            return True, " ".join(cmd[:-2] + ["-o", SO])
        last = (r.stderr or r.stdout).strip()
        if verbose:
            print(f"build failed with {' '.join(flags)}:\n{last}", file=sys.stderr)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False, f"compilation failed: {last.splitlines()[-1] if last else 'unknown error'}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cxx", default=None, help="C++ compiler (default: $CXX, g++, c++, clang++)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    ok, msg = compile_native(a.cxx, verbose=not a.quiet)
    if not a.quiet:
        print(("built: " if ok else "skipped: ") + msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
