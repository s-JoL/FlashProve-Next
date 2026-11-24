# SPDX-License-Identifier: Apache-2.0
"""The reference verifier is detached from the prover: no import (static or dynamic) of prover packages."""
import ast
import os
import subprocess
import sys
import unittest

from _common import PYROOT

PKG = os.path.join(PYROOT, "fpn_verify_ref")
FORBIDDEN = ("flashprove_next", "prover", "fpn_prover", "serve", "fpn_serve", "integrations", "cupy", "torch")
ALLOWED_TOP = {"argparse", "dataclasses", "hashlib", "struct", "sys", "typing", "numpy"}


class TestIndependence(unittest.TestCase):
    def _files(self):
        out = [os.path.join(PKG, f) for f in sorted(os.listdir(PKG)) if f.endswith(".py")]
        self.assertGreaterEqual(len(out), 7)
        return out

    def test_static_imports(self):
        for path in self._files():
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        continue                       # intra-package relative import
                    mods = [node.module or ""]
                else:
                    mods = []
                for m in mods:
                    top = m.split(".")[0]
                    self.assertNotIn(top, FORBIDDEN, f"{path}: import {m}")
                    self.assertIn(top, ALLOWED_TOP, f"{path}: unexpected import {m}")

    def test_no_dynamic_imports(self):
        for path in self._files():
            with open(path, encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src, path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, ("__import__", "importlib", "exec", "eval"), path)
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ("import_module", "__import__"), path)
            for bad in FORBIDDEN[:4]:
                self.assertNotIn(bad + ".", src.replace("fpn_verify_ref.", ""), path)

    def test_runtime_modules(self):
        code = ("import sys; sys.path.insert(0, %r); import fpn_verify_ref.verify, fpn_verify_ref.__main__; "
                "bad = [m for m in sys.modules if m.split('.')[0] in %r]; print(bad); sys.exit(1 if bad else 0)"
                % (PYROOT, FORBIDDEN))
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
