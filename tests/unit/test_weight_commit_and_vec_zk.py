# SPDX-License-Identifier: Apache-2.0
"""global model-weight commitment + drawn-tile openings, and the relation-5 joint-rank ZK certificate.

* `commit_model_weights`: exact per-tile integer encoding (== folded_vec.bf16_words_to_ints), deterministic root,
  safetensors reader == in-memory checkpoint, any single weight element / tensor name / K changes the root;
* `open_drawn_tiles` -> `verify_drawn_tiles_against_setup` with ONLY the SetupCertificate: an honest relation-5
  (n = 16) container + opening VERIFIES and the returned weight_root is the one Rust / Python fpn-verify accept;
  forgeries (one element swapped in either direction, a genuine tile of another layer substituted or relabelled,
  another model's certificate, tampered manifest, wrong coordinates / tile count) are REJECTED with stable codes;
* `require_joint_mask_vec`: every plaintext claim functional reproduces the container's bytes, the NumPy PCS
  functionals equal `folded_air.pcs_functionals`, rank_B == rank_BA on `fpn2-folded-vec-n16-v1` (honest witness
  manifold Y = W X), the free (X, Y) model fails by exactly the mask-free zero-claim residual (3 F_p ranks), dropping
  the Libra slots fails (negative control), and re-proving shares 0 private 64-byte blocks.
"""
import ast
import json
import os
import random
import struct
import tempfile
import unittest

import numpy as np

from flashprove_next.field import CPU as X
from flashprove_next.field import P
from fpn_prover.backend import cuda_engine as CE
from fpn_prover.backend import folded_air as FA
from fpn_prover.backend import folded_vec as FV
from fpn_prover.backend.fpn2 import FOLDED_VEC_N16_V1
from fpn_prover.circuit import e2e_prover as E2E
from fpn_prover.runtime import weight_commit as WC
from fpn_prover.runtime.setup_cert import SetupCertificate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUST = os.path.join(ROOT, "core", "target", "release", "fpn-verify")
SMALL = dict(FOLDED_VEC_N16_V1, name="test-folded-vec-n12", n_vars=12, max_hidden=32, n_queries=4, c_p=1 << 20)


class _Rng:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return (self.r.randrange(P), self.r.randrange(P), self.r.randrange(P))


def _bf16(g, shape, scale=0.02):
    f = (g.standard_normal(shape) * scale).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def _model(seed, layers=3):
    g = np.random.default_rng(seed)
    t = {}
    for l in range(layers):
        t[f"model.layers.{l}.self_attn.q_proj.weight"] = _bf16(g, (128, 1024))
        t[f"model.layers.{l}.mlp.down_proj.weight"] = _bf16(g, (64, 1500))       # ragged: K-edge zero padding
        t[f"model.layers.{l}.input_layernorm.weight"] = _bf16(g, (1024,))         # 1-D: skipped
    return t


def _write_safetensors(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        hdr[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.astype("<u2").tobytes())
        off += a.nbytes
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _verdicts(data, root):
    py = E2E.verdict_code(E2E.python_verify_detached(data, root)[0])
    rs = E2E.verdict_code(E2E.rust_verify(RUST, data, root)[0]) if os.path.exists(RUST) else None
    return rs, py


# ===================================================================================== global weight commitment
class GlobalWeightCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CE.HostEngine()
        cls.T = _model(1)
        cls.com = WC.commit_model_weights(cls.T)
        cls.setup = SetupCertificate(cls.com.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        cls.lay = FV.layout_of(FOLDED_VEC_N16_V1)                                 # B = 1 tile of 32 x 1024
        cls.coords = [("model.layers.1.self_attn.q_proj.weight", 64, 0)]
        cls.opening = WC.open_drawn_tiles(cls.com, cls.T, cls.coords)
        g = np.random.default_rng(7)
        cls.W_int = np.stack([t.units for t in cls.opening.tiles])
        cls.X_int = np.stack([FV.bf16_words_to_ints(_bf16(g, (1024,), 1.0))[0]])
        Y_int, bits = FV.exact_products(cls.W_int, cls.X_int)
        assert bits <= 62
        cls.W, cls.Xf, cls.Y = cls.opening.weights_field(), FV.to_field(cls.X_int), FV.to_field(Y_int)
        cls.container = FV.prove(cls.W, cls.Xf, cls.Y, n_s=b"\x05" * 32, rng=_Rng(4), engine=cls.engine)

    def check(self, opening, container=None, coords=None, setup=None):
        return WC.verify_drawn_tiles_against_setup(setup or self.setup, opening, container or self.container,
                                                   coords if coords is not None else self.coords,
                                                   engine=self.engine)

    # ---------------------------------------------------------------- encoding + commitment
    def test_tile_units_match_relation5_encoding(self):
        g = np.random.default_rng(3)
        tiles = _bf16(g, (6, 32, 1024))
        tiles[1] = 0                                                               # all-zero tile
        tiles[2] = 0
        tiles[2, 0, :6] = np.array([1, 2, 0x8001, 0x0080, 0x8000, 0x1a40], dtype=np.uint16)  # subnormals, -0
        tiles[3, 0, 0] = 0x4f00                                                    # large exponent, spread < 54
        units, shifts = WC.tile_units(tiles)
        for i in range(len(tiles)):
            ref, sh = FV.bf16_words_to_ints(tiles[i])
            self.assertNotEqual(ref.dtype, object)
            self.assertTrue((ref == units[i]).all())
            self.assertEqual(sh, shifts[i])
        with self.assertRaises(WC.WeightCommitError):
            WC.tile_units(np.full((1, 32, 1024), 0x7f80, dtype=np.uint16))      # inf
        spread = np.zeros((1, 32, 1024), dtype=np.uint16)
        spread[0, 0, :2] = (0x0001, 0x7f7f)                                       # subnormal + max: > int64 units
        with self.assertRaises(WC.WeightCommitError):
            WC.tile_units(spread)

    def test_commitment_shape_and_determinism(self):
        st = self.com.stats
        self.assertEqual(st["tensors"], 6)
        self.assertEqual(len(st["skipped"]), 3)
        self.assertEqual(st["leaves"], 3 * (4 * 1 + 2 * 2))                       # q: 4 row tiles; down: 2 x 2
        self.assertEqual(WC.commit_model_weights(self.T, threads=1).root, self.com.root)

    def test_safetensors_reader_equals_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            _write_safetensors(path, self.T)
            self.assertEqual(WC.commit_model_weights(path).root, self.com.root)
            self.assertEqual(WC.commit_model_weights(d).root, self.com.root)
            op = WC.open_drawn_tiles(self.com, path, self.coords)
            self.assertEqual(op.encode(), self.opening.encode())

    def test_root_binds_every_element_name_and_K(self):
        T2 = dict(self.T)
        w = T2["model.layers.2.mlp.down_proj.weight"].copy()
        w[63, 1499] ^= 1
        T2["model.layers.2.mlp.down_proj.weight"] = w
        self.assertNotEqual(WC.commit_model_weights(T2).root, self.com.root)
        T3 = {k.replace("layers.2.", "layers.9."): v for k, v in self.T.items()}
        self.assertNotEqual(WC.commit_model_weights(T3).root, self.com.root)
        self.assertNotEqual(WC.commit_model_weights(self.T, K=512).root, self.com.root)
        T4 = dict(self.T)
        nw = T4["model.layers.0.input_layernorm.weight"].copy()
        nw[3] ^= 1                                                                 # untiled 1-D tensor
        T4["model.layers.0.input_layernorm.weight"] = nw
        self.assertNotEqual(WC.commit_model_weights(T4).root, self.com.root)

    def test_module_never_reads_the_environment(self):
        with open(WC.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "os" and n.attr in ("environ", "getenv", "putenv")]
        self.assertEqual(hits, [])

    # ---------------------------------------------------------------- honest opening
    def test_honest_opening_verifies_with_setup_only(self):
        enc = self.opening.encode()
        self.assertEqual(WC.TileOpening.decode(enc).encode(), enc)
        rep = self.check(enc)
        self.assertEqual(rep["verdict"], "VERIFIED", rep)
        self.assertEqual(rep["weight_root"], bytes(self.container[96:128]).hex())
        self.assertEqual(rep["tree_depth"], WC.tree_depth(self.com.stats["leaves"]))
        rs, py = _verdicts(self.container, bytes.fromhex(rep["weight_root"]))
        self.assertEqual(py, "VERIFIED")
        if rs is not None:
            self.assertEqual(rs, "VERIFIED")

    # ---------------------------------------------------------------- forgeries
    def _with_tile(self, tile):
        return WC.TileOpening(self.opening.K, self.opening.manifest, (tile,))

    def test_one_weight_element_swapped_in_opening_rejected(self):
        t = self.opening.tiles[0]
        u = t.units.copy()
        u[5, 9] += 1
        rep = self.check(self._with_tile(WC.OpenedTile(t.name, t.r0, t.c0, t.shift, u, t.path)))
        self.assertEqual(rep["verdict"], "REJECTED: E_TILE_PATH")

    def test_container_on_one_altered_weight_rejected(self):
        """Prover proves on W' (one element changed) with consistent Y': the container verifies under ITS OWN
        root, but the honest opening's weight_root differs -> E_WEIGHT_ROOT; an opening altered to match W' fails
        inclusion -> E_TILE_PATH."""
        W2 = self.W_int.copy()
        W2[0, 3, 17] += 1
        Y2, _ = FV.exact_products(W2, self.X_int)
        bad = FV.prove(FV.to_field(W2), self.Xf, FV.to_field(Y2), n_s=b"\x05" * 32, rng=_Rng(4), engine=self.engine)
        self.assertEqual(_verdicts(bad, bytes(bad[96:128]))[1], "VERIFIED")
        self.assertEqual(self.check(self.opening, container=bad)["verdict"], "REJECTED: E_WEIGHT_ROOT")
        t = self.opening.tiles[0]
        matched = self._with_tile(WC.OpenedTile(t.name, t.r0, t.c0, t.shift, W2[0], t.path))
        self.assertEqual(self.check(matched, container=bad)["verdict"], "REJECTED: E_TILE_PATH")

    def test_genuine_tile_of_another_layer_rejected(self):
        other = WC.open_drawn_tiles(self.com, self.T, [("model.layers.2.self_attn.q_proj.weight", 64, 0)])
        self.assertEqual(self.check(other)["verdict"], "REJECTED: E_TILE_COORDS")           # true coords
        o = other.tiles[0]
        t = self.opening.tiles[0]
        relabelled = self._with_tile(WC.OpenedTile(t.name, t.r0, t.c0, o.shift, o.units, o.path))
        self.assertEqual(self.check(relabelled)["verdict"], "REJECTED: E_TILE_PATH")     # lied coords
        relabelled2 = self._with_tile(WC.OpenedTile(t.name, t.r0, t.c0, o.shift, o.units, t.path))
        self.assertEqual(self.check(relabelled2)["verdict"], "REJECTED: E_TILE_PATH")

    def test_tile_from_a_different_model_rejected(self):
        T_other = _model(2)
        com2 = WC.commit_model_weights(T_other)
        op2 = WC.open_drawn_tiles(com2, T_other, self.coords)
        self.assertEqual(self.check(op2)["verdict"], "REJECTED: E_TILE_PATH")
        setup2 = SetupCertificate(com2.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        self.assertEqual(self.check(self.opening, setup=setup2)["verdict"], "REJECTED: E_TILE_PATH")

    def test_manifest_coords_and_shape_tampering_rejected(self):
        m = bytearray(self.opening.manifest)
        m[-1] ^= 1                                                                 # first_leaf of the last tensor
        rep = self.check(WC.TileOpening(self.opening.K, bytes(m), self.opening.tiles))
        self.assertIn(rep["verdict"], ("REJECTED: E_MANIFEST", "REJECTED: E_TILE_PATH"))
        self.assertEqual(self.check(self.opening, coords=[("model.layers.1.self_attn.q_proj.weight", 32, 0)])
                         ["verdict"], "REJECTED: E_TILE_COORDS")
        t = self.opening.tiles[0]
        mis = self._with_tile(WC.OpenedTile(t.name, t.r0 + 1, t.c0, t.shift, t.units, t.path))
        self.assertEqual(self.check(mis, coords=[(t.name, t.r0 + 1, 0)])["verdict"], "REJECTED: E_TILE_RANGE")
        two = WC.open_drawn_tiles(self.com, self.T, self.coords * 2)
        self.assertEqual(self.check(two, coords=self.coords * 2)["verdict"], "REJECTED: E_PROFILE")
        self.assertEqual(self.check(self.opening, container=b"\x00" * len(self.container))["verdict"],
                         "REJECTED: E_PROFILE")

    def test_production_secret_rng_ignores_fpn_test_rng_seed(self):
        """Security audit BLOCKER #1 regression: SecretRng.fp3_array must NEVER honor FPN_TEST_RNG_SEED."""
        from unittest.mock import patch
        from fpn_prover.backend import folded_air as FA

        with patch.dict(os.environ, {"FPN_TEST_RNG_SEED": "123456"}):
            a1 = FA.SecretRng().fp3_array(64)
            a2 = FA.SecretRng().fp3_array(64)
            v1 = FV.SecretRng().fp3_array(64)
            v2 = FV.SecretRng().fp3_array(64)
        self.assertFalse(np.array_equal(a1, a2))
        self.assertFalse(np.array_equal(v1, v2))
        self.assertTrue(bool((a1 < FA.P).all()) and bool((v1 < FV.P).all()))



# ===================================================================================== block-128 FP8 E4M3
def _fp8_bytes(g, shape):
    """Random finite e4m3fn bytes (never 0x7f / 0xff = NaN)."""
    b = g.integers(0, 256, size=shape, dtype=np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 0x01, b).astype(np.uint8)


def _fp8_model(seed, layers=2):
    g = np.random.default_rng(seed)
    t, dt = {}, {}
    for l in range(layers):
        for name, (r, c) in ((f"model.layers.{l}.self_attn.q_proj.weight", (256, 2048)),
                             (f"model.layers.{l}.mlp.experts.3.down_proj.weight", (192, 768))):   # ragged K
            t[name] = _fp8_bytes(g, (r, c))
            dt[name] = "F8_E4M3"
            sc = (2.0 ** g.integers(-14, -4, size=(-(-r // 128), -(-c // 128)))
                  * (1 + g.random(size=(-(-r // 128), -(-c // 128))))).astype(np.float32)
            t[name + "_scale_inv"] = sc
            dt[name + "_scale_inv"] = "F32"
        t[f"model.layers.{l}.mlp.gate.weight"] = _bf16(g, (128, 1024))              # BF16 router: tiled
        t[f"model.layers.{l}.input_layernorm.weight"] = _bf16(g, (1024,))           # residual
    return t, dt


def _write_safetensors_typed(path, tensors, dtypes):
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        dtn = dtypes.get(name, "BF16")
        raw = a.astype({"BF16": "<u2", "F8_E4M3": "u1", "F32": "<f4"}[dtn]).tobytes()
        hdr[name] = {"dtype": dtn, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class Fp8WeightCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CE.HostEngine()
        cls.T, cls.DT = _fp8_model(11)
        cls.ck = WC.ArrayCheckpoint(cls.T, cls.DT)
        cls.com = WC.commit_model_weights(cls.ck)
        cls.setup = SetupCertificate(cls.com.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        cls.coords = [("model.layers.1.self_attn.q_proj.weight", 160, 1024)]
        cls.opening = WC.open_drawn_tiles(cls.com, cls.ck, cls.coords)
        g = np.random.default_rng(12)
        W_int, cls.w_shift = cls.opening.tiles[0].window(WC.WINDOW_INT_BITS)
        cls.W_int = W_int[None]
        xv = WC.bf16_values(_bf16(g, (1024,), 1.0))
        xv[:7] = (2.0 ** -40, -2.0 ** -38, 3.0 * 2 ** -30, 2.0 ** 9, -1.5 * 2 ** 9, 0.0, -0.0)   # wide spread
        cls.xv = xv
        X_int, cls.x_shift = WC.window_ints(xv)
        cls.X_int = X_int[None]
        Y_int, cls.bits = FV.exact_products(cls.W_int, cls.X_int)
        cls.container = FV.prove(FV.to_field(cls.W_int), FV.to_field(cls.X_int), FV.to_field(Y_int),
                                 n_s=b"\x06" * 32, rng=_Rng(5), engine=cls.engine)

    def check(self, opening, container=None, coords=None):
        return WC.verify_drawn_tiles_against_setup(self.setup, opening, container or self.container,
                                                   coords if coords is not None else self.coords, engine=self.engine)

    def test_e4m3_table_matches_the_format(self):
        t = WC.E4M3_TABLE
        self.assertEqual(t[0x00], 0.0)
        self.assertEqual(t[0x01], 2.0 ** -9)                    # min subnormal
        self.assertEqual(t[0x07], 7 * 2.0 ** -9)                # max subnormal
        self.assertEqual(t[0x08], 2.0 ** -6)                    # min normal
        self.assertEqual(t[0x38], 1.0)
        self.assertEqual(t[0x7E], 448.0)                        # e4m3fn max
        self.assertEqual(t[0xFE], -448.0)
        self.assertTrue(np.isnan(t[0x7F]) and np.isnan(t[0xFF]))
        self.assertEqual(int(np.isnan(t).sum()), 2)
        with self.assertRaises(WC.WeightCommitError):
            WC.fp8_dequant(np.full((1, 128), 0x7F, dtype=np.uint8), np.ones(1, dtype=np.float32))

    def test_commitment_kinds_manifest_v2_and_scales_consumed(self):
        st = self.com.stats
        self.assertEqual(st["fp8_tensors"], 4)
        self.assertEqual(st["fp8_scale_tensors_in_leaves"], 4)
        self.assertEqual(st["manifest_version"], 2)
        self.assertEqual(st["residual_tensors"], 2)                           # the two 1-D norms only
        self.assertEqual(st["tensors"], 6)                                    # 4 FP8 + 2 BF16 routers
        # q: 8 row tiles x 2 col tiles; down: 6 row tiles x 1 (K-padded); router 4 x 1
        self.assertEqual(st["leaves"], 2 * (16 + 6 + 4))
        self.assertEqual(WC.manifest_window_int_bits(self.com.manifest), WC.WINDOW_INT_BITS)
        self.assertEqual(WC.commit_model_weights(self.ck, threads=1).root, self.com.root)
        # BF16-only checkpoints keep the v1 manifest 
        bf = {k: v for k, v in self.T.items() if self.DT.get(k) is None}
        self.assertTrue(WC.commit_model_weights(bf).manifest.startswith(b"FPN2_WMANIFEST\x01"))

    def test_safetensors_fp8_reader_equals_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            _write_safetensors_typed(path, self.T, self.DT)
            self.assertEqual(WC.commit_model_weights(path).root, self.com.root)
            op = WC.open_drawn_tiles(self.com, path, self.coords)
            self.assertEqual(op.encode(), self.opening.encode())

    def test_root_binds_fp8_byte_scale_and_window(self):
        T2 = dict(self.T)
        w = T2["model.layers.0.mlp.experts.3.down_proj.weight"].copy()
        w[191, 767] ^= 0x01
        T2["model.layers.0.mlp.experts.3.down_proj.weight"] = w
        self.assertNotEqual(WC.commit_model_weights(WC.ArrayCheckpoint(T2, self.DT)).root, self.com.root)
        T3 = dict(self.T)
        sc = T3["model.layers.1.self_attn.q_proj.weight_scale_inv"].copy()
        sc[1, 15] = np.nextafter(sc[1, 15], np.float32(1))
        T3["model.layers.1.self_attn.q_proj.weight_scale_inv"] = sc
        self.assertNotEqual(WC.commit_model_weights(WC.ArrayCheckpoint(T3, self.DT)).root, self.com.root)
        self.assertNotEqual(WC.commit_model_weights(self.ck, window_int_bits=22).root, self.com.root)
        bad = dict(self.T)
        bad["model.layers.0.self_attn.q_proj.weight_scale_inv"] = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(WC.WeightCommitError):
            WC.commit_model_weights(WC.ArrayCheckpoint(bad, self.DT))

    def test_window_guarantees_integer_faithful_and_discloses_tail(self):
        g = np.random.default_rng(5)
        for trial in range(20):
            v = np.ldexp(g.standard_normal((32, 1024)), g.integers(-60, 20, size=(32, 1024)))  # absurd spread
            u, s = WC.window_ints(v)
            self.assertLessEqual(int(np.abs(u).max()), 1 << WC.WINDOW_INT_BITS)
            err, mx = WC.window_tail(v, u, s)
            self.assertLessEqual(err, 2.0 ** -(WC.WINDOW_INT_BITS + 0) * mx)   # half a grid step <= 2^-23 max
        # round-half-even can reach exactly 2^23 (the documented <= edge)
        v = np.array([2.0 ** 23 - 0.5, 1.0])
        u, s = WC.window_ints(v)
        self.assertEqual(s, 0)
        self.assertEqual(int(u[0]), 1 << 23)
        # exact when the tile already fits: BF16 values within 16 binades of the top binade
        words = _bf16(g, (32, 1024), 0.02)
        vals = WC.bf16_values(words)
        u, s = WC.window_ints(vals)
        top = np.frexp(np.abs(vals).max())[1]
        keep = np.abs(vals) >= 2.0 ** (top - 16)
        self.assertTrue((np.ldexp(u.astype(np.float64), -s)[keep] == vals[keep]).all())
        zu, zs = WC.window_ints(np.zeros(5))
        self.assertEqual(zs, 0)
        self.assertFalse(zu.any())
        with self.assertRaises(WC.WeightCommitError):
            WC.window_ints(np.array([1.0, np.inf]))
        # the FP8 tile + wide-spread X of setUpClass: sum |W||X| < 2^62 -> integer faithful
        self.assertLessEqual(self.bits, 57)
        self.assertLessEqual(int(np.abs(self.X_int).max()), 1 << 23)
        self.assertLessEqual(int(np.abs(self.W_int).max()), 1 << 23)
        xerr, xmax = WC.window_tail(self.xv, self.X_int[0], self.x_shift)
        self.assertGreater(xerr, 0.0)                                         # the 2^-40 entries were rounded
        self.assertLessEqual(xerr, 2.0 ** -23 * xmax)

    def test_bf16_window_of_exact_units_equals_window_of_values(self):
        com = WC.commit_model_weights(_model(1))
        op = WC.open_drawn_tiles(com, _model(1), [("model.layers.1.self_attn.q_proj.weight", 64, 0)])
        t = op.tiles[0]
        self.assertTrue((t.window(None)[0] == t.units).all())
        u, s = t.window(WC.WINDOW_INT_BITS)
        ref = WC.window_ints(WC.bf16_values(_tensor_bf16_tile(_model(1), "model.layers.1.self_attn.q_proj.weight",
                                                               64)))
        self.assertEqual(s, ref[1])
        self.assertTrue((u == ref[0]).all())

    def test_honest_fp8_opening_verifies_with_setup_only(self):
        enc = self.opening.encode()
        self.assertTrue(enc.startswith(b"FPN2_WOPEN\x02"))
        self.assertEqual(WC.TileOpening.decode(enc).encode(), enc)
        self.assertLess(len(enc), 64 * 1024)                                  # 32 KiB raw e4m3 + scales + path
        rep = self.check(enc)
        self.assertEqual(rep["verdict"], "VERIFIED", rep)
        self.assertEqual(rep["fp8_tiles"], 1)
        self.assertEqual(rep["weight_root"], bytes(self.container[96:128]).hex())
        rs, py = _verdicts(self.container, bytes.fromhex(rep["weight_root"]))
        self.assertEqual(py, "VERIFIED")
        if rs is not None:
            self.assertEqual(rs, "VERIFIED")

    def _with(self, **kw):
        t = self.opening.tiles[0]
        f = dict(name=t.name, r0=t.r0, c0=t.c0, shift=t.shift, units=t.units, path=t.path, kind=t.kind, raw=t.raw,
                 scales=t.scales)
        f.update(kw)
        return WC.TileOpening(self.opening.K, self.opening.manifest, (WC.OpenedTile(**f),))

    def test_fp8_forgeries_rejected(self):
        raw = self.opening.tiles[0].raw.copy()
        raw[3, 5] ^= 0x01
        self.assertEqual(self.check(self._with(raw=raw))["verdict"], "REJECTED: E_TILE_PATH")
        sc = self.opening.tiles[0].scales.copy()
        sc[2] *= np.float32(2)
        self.assertEqual(self.check(self._with(scales=sc))["verdict"], "REJECTED: E_TILE_PATH")
        # a genuine FP8 tile of another expert / layer at its true coordinates
        other = WC.open_drawn_tiles(self.com, self.ck, [("model.layers.0.self_attn.q_proj.weight", 160, 1024)])
        self.assertEqual(self.check(other)["verdict"], "REJECTED: E_TILE_COORDS")
        o = other.tiles[0]
        self.assertEqual(self.check(self._with(raw=o.raw, scales=o.scales, path=o.path))["verdict"],
                         "REJECTED: E_TILE_PATH")
        # the FP8 tile presented as a BF16 tile (kind confusion)
        u, s = self.opening.tiles[0].window(WC.WINDOW_INT_BITS)
        self.assertEqual(self.check(self._with(kind=WC.KIND_BF16, units=u, shift=s, raw=None, scales=None))
                         ["verdict"], "REJECTED: E_TILE_KIND")
        # a container proved on the EXACT-but-unwindowed / differently windowed W -> E_WEIGHT_ROOT
        W2, _ = self.opening.tiles[0].window(WC.WINDOW_INT_BITS - 1)
        Y2, _ = FV.exact_products(W2[None], self.X_int)
        bad = FV.prove(FV.to_field(W2[None]), FV.to_field(self.X_int), FV.to_field(Y2), n_s=b"\x06" * 32,
                       rng=_Rng(5), engine=self.engine)
        self.assertEqual(self.check(self.opening, container=bad)["verdict"], "REJECTED: E_WEIGHT_ROOT")
        # manifest with a different window width -> the root changes -> E_TILE_PATH
        m = bytearray(self.opening.manifest)
        off = len(b"FPN2_WMANIFEST\x02") + 32 + 24
        m[off] ^= 0x01
        rep = self.check(WC.TileOpening(self.opening.K, bytes(m), self.opening.tiles))
        self.assertIn(rep["verdict"], ("REJECTED: E_TILE_PATH", "REJECTED: E_MANIFEST"))
        # truncated opening bytes
        enc = self.opening.encode()
        self.assertEqual(self.check(enc[:-3])["verdict"], "REJECTED: E_OPENING")


def _tensor_bf16_tile(T, name, r0):
    return np.asarray(T[name][r0:r0 + 32, :1024], dtype=np.uint16)

# ===================================================================================== relation-5 ZK rank
def _prove_recorded(profile, seed, engine=None, rng=None):
    lay = FV.layout_of(profile)
    g = np.random.default_rng(seed)
    Wi = g.integers(-(1 << 15), 1 << 15, size=(lay.B, FV.ROWS, lay.K))
    Xi = g.integers(-(1 << 15), 1 << 15, size=(lay.B, lay.K))
    Yi, _ = FV.exact_products(Wi, Xi)
    W, Xv, Y = FV.to_field(Wi), FV.to_field(Xi), FV.to_field(Yi)
    rec = {}
    data = FV.prove(W, Xv, Y, profile=profile, rng=rng or _Rng(seed), engine=engine, hooks={"record": rec})
    return lay, W, Xv, Y, rec, data


class Relation5ZkRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CE.HostEngine()
        cls.small = _prove_recorded(SMALL, 21)
        cls.n16 = _prove_recorded(FOLDED_VEC_N16_V1, 22, engine=cls.engine)

    def test_claim_functionals_reproduce_container_bytes(self):
        for lay, W, _, _, rec, data in (self.small, self.n16):
            Fa = np.ascontiguousarray(np.asarray(rec["F"]).T)
            rows = list(FV.vec_claim_rows(lay, rec, W))
            vals = FV.vec_claim_values(lay, data)
            self.assertEqual(len(rows), 2 + 4 * lay.d)
            for (label, row), v in zip(rows, vals, strict=True):
                self.assertEqual(tuple(int(x) for x in X.esum(X.emul(row, Fa))), tuple(v), label)

    def test_numpy_pcs_functionals_equal_folded_air(self):
        lay, _, _, _, rec, _ = self.small
        rec2 = dict(rec, a=[tuple(int(x) for x in v) for v in np.asarray(rec["a"]).T],
                    c=[tuple(int(x) for x in v) for v in np.asarray(rec["c"]).T])
        a = list(FV.pcs_rows_np(lay, rec))
        b = list(FA.pcs_functionals(lay, rec2))
        self.assertEqual([x for x, _ in a], [x for x, _ in b])
        for (label, r1), (_, r2) in zip(a, b, strict=True):
            self.assertTrue((np.asarray(r1, dtype=np.uint64) == np.asarray(r2, dtype=np.uint64)).all(), label)
        self.assertEqual(len(a), FV.vec_observation_count(lay, rec, claims=False))

    def test_joint_rank_holds_n16(self):
        lay, W, _, _, rec, _ = self.n16
        rep = FV.require_joint_mask_vec(lay, rec, W)
        self.assertTrue(rep["holds"], rep)
        self.assertEqual(rep["rank_B"], rep["rank_BA"])
        self.assertEqual(rep["observations"], 2 + 4 * lay.d + 3 * (lay.n + 2) + 1
                         + FV._count_queries(lay, rec["positions"]))

    def test_free_witness_gap_is_exactly_the_zero_claim(self):
        lay, W, _, _, rec, _ = self.small
        honest = FV.require_joint_mask_vec(lay, rec, W, witness="honest")
        free = FV.require_joint_mask_vec(lay, rec, W, witness="free", raise_on_fail=False)
        self.assertTrue(honest["holds"])
        self.assertFalse(free["holds"])
        self.assertEqual(free["rank_BA"] - free["rank_B"], 3)                    # one F_p3 observation
        rows = dict(FV.vec_claim_rows(lay, rec, W))
        lam = rec["lam"]
        z = X.esub(X.eadd(rows["ip_round_0_0"], rows["ip_round_0_1"]), FA._escale(rows["libra_sum"], lam))
        coords = FV.VecCoordinates(lay)
        mask_cells = coords.libra + coords.reserved + coords.spare
        self.assertFalse(z[:, mask_cells].any())                                 # mask-free functional ...
        self.assertTrue(z[:, coords.y_cells].any())                              # ... of the witness,
        proj = FV._project_vec(z, [("xw", (coords.x_cells, coords.y_cells))], W)
        self.assertFalse(proj.any())                                             # identically 0 when Y = W X

    def test_negative_control_without_libra_fails(self):
        lay, W, _, _, rec, _ = self.small
        rep = FV.require_joint_mask_vec(lay, rec, W, drop_libra=True, raise_on_fail=False)
        self.assertFalse(rep["holds"], rep)
        with self.assertRaises(ValueError):
            FV.require_joint_mask_vec(lay, rec, W, drop_libra=True)

    def test_reproof_shares_no_private_64_byte_block(self):
        lay, W, Xv, Y, _, _ = self.n16
        a = FV.prove(W, Xv, Y, engine=self.engine)                              # fresh SecretRng randomness
        b = FV.prove(W, Xv, Y, engine=self.engine)
        self.assertNotEqual(a, b)
        rep = FV.shared_blocks(a, b, lay, FOLDED_VEC_N16_V1["c_p"])
        self.assertEqual(rep["private_aligned_shared"], 0, rep)
        self.assertEqual(rep["private_window_set_intersection"], 0, rep)
        self.assertGreater(rep["private_aligned_blocks"], 3000)
        self.assertEqual(bytes(a[96:160]), bytes(b[96:160]))                    # weight_root, N_s public
        root = bytes(a[96:128])
        for c in (a, b):
            self.assertEqual(_verdicts(c, root)[1], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
