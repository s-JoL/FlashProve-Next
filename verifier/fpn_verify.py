#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU verifier for FlashProve-Next reply proofs, runnable from a source checkout without installation.

Same interface as the installed `fpn-verify` command (see `flashprove_next.cli`):

    python verifier/fpn_verify.py --model-id HEX --request request.json --reply reply.json --nonce HEX \
        --proof reply.proof [--profile NAME] [--tokenizer DIR] [--extra-profile SPEC.json] [--json]

Exit status 0 prints VERIFIED, 1 prints REJECTED: <reason>.  NumPy and the Python standard library only; no weights,
no architecture configuration, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flashprove_next.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
