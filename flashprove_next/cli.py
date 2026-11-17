# SPDX-License-Identifier: Apache-2.0
"""fpn-verify: standalone CPU verifier for FlashProve-Next reply proofs.

    fpn-verify --model-id HEX --request request.json --reply reply.json --nonce HEX --proof reply.proof
               [--profile NAME ...] [--tokenizer DIR] [--extra-profile SPEC.json ...] [--json]

request.json  the client's own request: OpenAI / Anthropic fields (`messages` | `input` | `prompt`, plus `system`,
              `max_tokens`, `response_format`, `tools`) or pre-tokenized `tokens`; the `fpn` object carries `thinking`,
              `turn` and, for turn > 0, the client's stored `St_prev`.
reply.json    {"tokens": [...], "n_reason": int, "finish_reason": str, "St_new": [...], "tool_segments": [...]}
              (the visible reply tokens and the usage / state fields of the response envelope).

Exit status 0 prints VERIFIED; exit status 1 prints REJECTED: <reason>.  No weights, no architecture configuration and
no GPU are needed (run with CUDA_VISIBLE_DEVICES="").  Request-side rendering of chat messages and constrained-decoding
masks need the model's public tokenizer (`--tokenizer`, a directory or tokenizer.json).

`--extra-profile` adds a profile spec chosen by the VERIFIER (for example the profile of a test model); it can never
redefine a built-in profile, and a bundle is accepted only if it matches the spec byte for byte.
"""
import argparse
import json
import os
import sys


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_parser():
    ap = argparse.ArgumentParser(prog="fpn-verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True, help="published model_id (64 hex characters)")
    ap.add_argument("--request", required=True, help="client request JSON")
    ap.add_argument("--reply", required=True, help="reply JSON (visible tokens and envelope fields)")
    ap.add_argument("--nonce", required=True, help="client nonce N_c (hex) sent with the request")
    ap.add_argument("--proof", required=True, help="proof bundle (FPNZK1)")
    ap.add_argument("--profile", action="append", default=None,
                    help="accept only this profile (repeatable), e.g. qwen3-0.6b-sec315; default: production profiles only")
    ap.add_argument("--allow-reduced-security", action="store_true",
                    help="explicitly permit non-production (demo_fast / dev_fast / test) profiles")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer directory or tokenizer.json (chat rendering and constrained decoding)")
    ap.add_argument("--extra-profile", action="append", default=None, metavar="SPEC.json",
                    help="register an additional verifier-chosen profile spec (repeatable)")
    ap.add_argument("--json", action="store_true", help="also print a JSON result line")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    from . import reply_bundle as RB

    try:
        mid = bytes.fromhex(a.model_id)
        if len(mid) != 32:
            raise ValueError("model_id must be 32 bytes")
        bytes.fromhex(a.nonce)
        req = _load_json(a.request)
        rep = _load_json(a.reply)
        with open(a.proof, "rb") as f:
            proof = f.read()
        for path in a.extra_profile or ():
            RB.register_profile(_load_json(path))
    except Exception as e:
        print(f"REJECTED: bad input ({type(e).__name__}: {e})")
        return 1

    if not RB.is_reply_zk_blob(proof):
        ok, why, info = False, "not an FPNZK1 proof bundle", {}
    else:
        tok = os.path.expanduser(a.tokenizer) if a.tokenizer else None
        ok, why, info = RB.verify_cli(
            mid, req, rep, a.nonce, proof,
            allowed_profiles=a.profile,
            tokenizer_dir=tok,
            allow_reduced_security=bool(a.allow_reduced_security),
        )
    if a.json:
        print(json.dumps({"ok": ok, "reason": why, **info}))
    print("VERIFIED" if ok else f"REJECTED: {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
