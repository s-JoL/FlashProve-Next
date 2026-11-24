# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import tempfile
import unittest

from _common import PYROOT, container_bytes, load

CONT = load("fpn2_containers.json")


def run(data, *args):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "proof.fpn2")
        with open(p, "wb") as f:
            f.write(data)
        return subprocess.run([sys.executable, "-m", "fpn_verify_ref", p, *args], cwd=PYROOT,
                              capture_output=True, text=True)


class TestCli(unittest.TestCase):
    def test_verified_and_rejected(self):
        v = next(x for x in CONT["vectors"] if x["name"] == "valid_harness_0")
        r = run(container_bytes(v), "--allow-reduced-security", "--profile", "fpn2-harness-sq-n4")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "VERIFIED"))
        r = run(container_bytes(v))
        self.assertEqual((r.returncode, r.stdout.strip()), (1, "REJECTED: E_REDUCED_SECURITY"))
        r = run(container_bytes(v), "--allow-reduced-security", "--profile", "fpn2-llm-std-v1")
        self.assertEqual((r.returncode, r.stdout.strip()), (1, "REJECTED: E_PROFILE_PINNED"))
        r = run(b"FPN2")
        self.assertEqual((r.returncode, r.stdout.strip()), (1, "REJECTED: E_LENGTH"))

    def test_usage_errors(self):
        r = subprocess.run([sys.executable, "-m", "fpn_verify_ref", "/nonexistent/x.fpn2"], cwd=PYROOT,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
