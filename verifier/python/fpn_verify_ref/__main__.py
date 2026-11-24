# SPDX-License-Identifier: Apache-2.0
"""python -m fpn_verify_ref <file> [--allow-reduced-security] [--profile NAME] [--circuit-root|--weight-root HEX64]
[--x-root HEX64]"""
import argparse
import sys

from .profile import PINNED
from .verify import verify_result


_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _parse_hex32(s: str):
    if len(s) != 64 or any(c not in _HEX_CHARS for c in s):
        return None
    return bytes.fromhex(s)


class _SingleUseStore(argparse.Action):
    """Reject repeating the same flag (matching Rust CLI's strict once-only parsing)."""
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"argument {option_string}: cannot be specified more than once")
        setattr(namespace, self.dest, values)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fpn_verify_ref")
    ap.add_argument("file")
    ap.add_argument("--allow-reduced-security", action="store_true")
    ap.add_argument("--profile")
    ap.add_argument("--circuit-root", "--weight-root", dest="circuit_root",
                    help="pinned public commitment: relation-2/4 circuit root or relation-5/6 weight root (64 hex chars)")
    ap.add_argument("--x-root", help="pinned pre-committed activation root of relation 6 (64 hex chars)")
    ap.add_argument("--circuit-desc", dest="circuit_desc",
                    help="canonical binary CircuitDesc for relation 2 (LLM_INFERENCE_V2)")
    stmt_grp = ap.add_mutually_exclusive_group()
    stmt_grp.add_argument("--stmt-digest", dest="stmt_digest", action=_SingleUseStore,
                          help="pinned 64-hex statement digest of relation 2 (LLM_INFERENCE_V2)")
    stmt_grp.add_argument("--stmt-file", dest="stmt_file", action=_SingleUseStore,
                          help="path to canonical statement bytes to hash for relation 2 (LLM_INFERENCE_V2)")
    a = ap.parse_args(argv)
    circ = None
    if a.circuit_root is not None:
        circ = _parse_hex32(a.circuit_root)
        if circ is None:
            print("error: --circuit-root needs 64 hex characters", file=sys.stderr)
            return 2
    xroot = None
    if a.x_root is not None:
        xroot = _parse_hex32(a.x_root)
        if xroot is None:
            print("error: --x-root needs 64 hex characters", file=sys.stderr)
            return 2
    stmt_dig = None
    if a.stmt_digest is not None:
        stmt_dig = _parse_hex32(a.stmt_digest)
        if stmt_dig is None:
            print("error: --stmt-digest needs 64 hex characters", file=sys.stderr)
            return 2
    if a.stmt_file is not None:
        from .relation2 import stmt_digest_bytes
        try:
            with open(a.stmt_file, "rb") as f:
                stmt_dig = stmt_digest_bytes(f.read())
        except OSError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    pin = None
    if a.profile is not None:
        if a.profile not in PINNED:
            print(f"error: unknown profile {a.profile}", file=sys.stderr)
            return 2
        pin = bytes.fromhex(PINNED[a.profile][1])
    try:
        with open(a.file, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.circuit_desc is not None:
        from .relation2 import verify_relation2
        from .wire import Reject
        try:
            with open(a.circuit_desc, "rb") as f:
                desc = f.read()
        except OSError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            res = verify_relation2(
                data, desc,
                allow_reduced=a.allow_reduced_security,
                expected_profile_id=pin,
                expected_circuit_root=circ,
                expected_x_root=xroot,
                expected_stmt_digest=stmt_dig,
            )
        except Reject as r:
            res = r.code
    else:
        if stmt_dig is not None:
            res = "E_STMT_PINNED"
        else:
            res = verify_result(data, a.allow_reduced_security, pin, circ, xroot)
    if res == "VERIFIED":
        print("VERIFIED")
        return 0
    print(f"REJECTED: {res}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
