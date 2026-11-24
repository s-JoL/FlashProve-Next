# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(TESTS)
REPO = os.path.dirname(os.path.dirname(PYROOT))
VECTORS = os.path.join(REPO, "specs", "vectors")
if PYROOT not in sys.path:
    sys.path.insert(0, PYROOT)


def load(name):
    with open(os.path.join(VECTORS, name)) as f:
        return json.load(f)


def h(x):
    return int(x, 16)


def h3(a):
    return tuple(int(x, 16) for x in a)


def container_bytes(v):
    pre = bytes.fromhex(v["prefix_hex"])
    return pre + b"\x00" * (v["total_len"] - len(pre))
