# SPDX-License-Identifier: Apache-2.0
"""lossless OCP MXFP4 (E2M1 packed nibbles + E8M0 block-32 scales) weight commitment -> relation 5 / 6.

* format: the exact 16-entry E2M1 table, the E8M0 exponent range (0xFF = NaN is refused), the nibble order of the
  `gpt-oss-20b` `Mxfp4GptOssExperts` layout re-expressed in numpy from `bench/tier2/bench_frontier_calibrate.py`
  (low nibble = earlier element), and an independent `fractions.Fraction` reference for the integer materialisation
  (round-half-even ties, -0, 60-binade tiles);
* guarantees: strict |W_int| < 2^w, half-step bound 2^-w max|W|, sum |W||X| < 2^62 at the default window and the
  COMPUTED (not assumed) `integer_faithful` flag turning false at w = 52; 17,408 B of leaf payload against 65,536 B of
  BF16 (3.76x); strict malformed-input rejection at every public boundary;
* end to end: MXFP4 -> `SetupCertificate.weight_commitment_root` -> `verify_mxfp4_tiles_against_setup` -> relation 6
  (`fpn2-folded-vec-bound-n16-v1`) -> native Rust `fpn-verify --weight-root --x-root` and the detached `-I` Python
  reference both VERIFIED, relation 5 likewise, and a forgery suite with one deterministic rejection code each.
"""
import ast
import json
import math
import os
import random
import struct
import tempfile
import unittest
from fractions import Fraction

import numpy as np

from flashprove_next.field import P
from fpn_prover.backend import cuda_engine as CE
from fpn_prover.backend import folded_vec as FV
from fpn_prover.backend import folded_vec_bound as FB
from fpn_prover.backend.fpn2 import FOLDED_VEC_BOUND_N16_V1
from fpn_prover.circuit import e2e_prover as E2E
from fpn_prover.runtime import mxfp4_commit as M
from fpn_prover.runtime import weight_commit as WC
from fpn_prover.runtime.setup_cert import SetupCertificate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUST = os.path.join(ROOT, "core", "target", "release", "fpn-verify")
PROF = FOLDED_VEC_BOUND_N16_V1
K = 1024

# literal copy of `lut` in bench/tier2/bench_frontier_calibrate.py (the gpt-oss-20b MXFP4 dequant reference)
BENCH_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class _Rng:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def fp3(self):
        return (self.r.randrange(P), self.r.randrange(P), self.r.randrange(P))


def _blocks(g, rows, cols):
    return g.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)


def _scales(g, rows, cols, lo=118, hi=135):
    return g.integers(lo, hi, size=(rows, cols // 32), dtype=np.uint8)


def _gpt_oss_tensors(seed, layers=2):
    """Synthetic tensors in the exact gpt-oss-20b layout: blocks (E, rows, in/32, 16) + scales (E, rows, in/32)."""
    g = np.random.default_rng(seed)
    t = {}
    for l in range(layers):
        for proj, rows, cols in (("gate_up_proj", 64, 1024), ("down_proj", 32, 1504)):    # 1504: ragged K edge
            base = f"model.layers.{l}.mlp.experts.{proj}"
            t[base + "_blocks"] = g.integers(0, 256, size=(2, rows, cols // 32, 16), dtype=np.uint8)
            t[base + "_scales"] = g.integers(118, 135, size=(2, rows, cols // 32), dtype=np.uint8)
    return t


def _wide_tile(seed, span=20):
    """A 32 x 1024 tile whose block scales span 2^-span ... 2^+span (e = 127 -+ span) inside ONE tile."""
    g = np.random.default_rng(seed)
    b = _blocks(g, 32, K)
    s = g.choice(np.array([127 - span, 127 + span, 127, 127 - span // 2, 127 + span // 2], dtype=np.uint8),
                 size=(32, K // 32))
    s[0, 0], s[0, 1] = 127 - span, 127 + span
    return b, s.astype(np.uint8)


def _source(seed, layers=2, wide=None):
    items = M.experts_from_checkpoint(WC.ArrayCheckpoint(_gpt_oss_tensors(seed, layers)))
    if wide is not None:
        items.append(("wide", wide[0], wide[1]))
    return items


def _write_u8_safetensors(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name], dtype=np.uint8)
        hdr[name] = {"dtype": "U8", "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _ref_window(blocks, scales, w):
    """Independent pure-Python reference: Fraction values, exact binade search, round-half-even (Fraction.__round__)."""
    vals = []
    for r in range(blocks.shape[0]):
        for c in range(blocks.shape[1] * 2):
            byte = int(blocks[r, c // 2])
            nib = (byte & 0xF) if c % 2 == 0 else (byte >> 4)
            sign, e, m = nib >> 3, (nib >> 1) & 3, nib & 1
            mag = Fraction(m, 2) if e == 0 else (1 + Fraction(m, 2)) * Fraction(2) ** (e - 1)
            v = mag * Fraction(2) ** (int(scales[r, c // 32]) - 127)
            vals.append(-v if sign else v)
    top = max(abs(v) for v in vals)
    if top == 0:
        return [0] * len(vals), 0
    E = 0
    while top >= Fraction(2) ** E:
        E += 1
    while top < Fraction(2) ** (E - 1):
        E -= 1
    shift = w - E
    return [int(round(v * Fraction(2) ** shift)) for v in vals], shift


def _rust(data, w, x):
    return E2E._run_cli([RUST, "--weight-root", w.hex(), "--x-root", x.hex()], data)[0]


def _py(data, w, x):
    vdir = os.path.join(ROOT, "verifier", "python")
    boot = ("import runpy, sys; " f"sys.path.insert(0, {vdir!r}); "
            "sys.argv = ['fpn_verify_ref', *sys.argv[1:]]; runpy.run_module('fpn_verify_ref', run_name='__main__')")
    args = [os.sys.executable, "-W", "error", "-I", "-c", boot, "--weight-root", w.hex(), "--x-root", x.hex()]
    return E2E._run_cli(args, data, cwd=vdir)[0]


def _verdicts(data, w, x):
    rs = E2E.verdict_code(_rust(data, w, x)) if os.path.exists(RUST) else None
    return rs, E2E.verdict_code(_py(data, w, x))


# ===================================================================================== formats + commitment
class Mxfp4FormatTests(unittest.TestCase):
    def test_e2m1_table_is_the_exact_16_value_format(self):
        self.assertEqual(len(M.E2M1_TABLE), 16)
        for nib, (sign, d) in enumerate(M.E2M1_TABLE):
            self.assertEqual(M.e2m1_nibble_x2(nib), (sign, d))
            self.assertEqual(sign, nib >> 3)
            self.assertIs(type(d), int)
            # value = (-1)^s d / 2 reproduces the literal calibration LUT bit for bit (including -0.0)
            self.assertEqual(math.copysign(d / 2, -1.0 if sign else 1.0), BENCH_LUT[nib])
            # independent derivation from the bit fields: 1 sign, 2 exponent (bias 1), 1 mantissa bit
            e, m = (nib >> 1) & 3, nib & 1
            ref = m * 0.5 if e == 0 else (1 + m * 0.5) * 2.0 ** (e - 1)
            self.assertEqual(d / 2, ref)
        self.assertEqual([d for _, d in M.E2M1_TABLE[:8]], [0, 1, 2, 3, 4, 6, 8, 12])
        self.assertEqual(M.E2M1_SIGNED_X2[8], 0)                              # nibble 8 is -0 -> integer 0
        for bad in (-1, 16, True, 1.0, "3"):
            with self.assertRaises(ValueError):
                M.e2m1_nibble_x2(bad)

    def test_e8m0_exponent_range_and_nan(self):
        for e in range(0, 255):
            self.assertEqual(M.e8m0_exponent(e), e - 127)
            self.assertEqual(M.e8m0_scale(e), 2.0 ** (e - 127))               # exact power of two
            m, ex = math.frexp(M.e8m0_scale(e))
            self.assertEqual((m, ex), (0.5, e - 126))
            self.assertEqual(M.e8m0_exponent(np.uint8(e)), e - 127)
        with self.assertRaises(ValueError):
            M.e8m0_scale(255)
        with self.assertRaises(M.WeightCommitError) as cm:
            M.e8m0_exponent(255)
        self.assertEqual(cm.exception.code, "E_TILE_ENCODING")
        for bad in (-1, 256, True, 1.0, "1", None):
            with self.assertRaises((ValueError, TypeError)):
                M.e8m0_exponent(bad)
        with self.assertRaises(ValueError):
            M.scale_exponents(np.array([[127, 255]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            M.scale_exponents(np.array([[127, 1]], dtype=np.int64))

    def test_nibble_order_matches_gpt_oss_convention(self):
        """numpy re-expression of bench_frontier_calibrate.mxfp4_dequant on the (rows, in/32, 16) layout."""
        lut = np.array(BENCH_LUT, dtype=np.float64)
        g = np.random.default_rng(3)
        rows, cols = 5, 96
        b4 = g.integers(0, 256, size=(rows, cols // 32, 16), dtype=np.uint8)
        sc = g.integers(100, 150, size=(rows, cols // 32), dtype=np.uint8)
        lo, hi = (b4 & 0x0F).astype(np.int64), (b4 >> 4).astype(np.int64)
        vals = np.stack([lut[lo], lut[hi]], axis=-1).reshape(rows, cols // 32, 32)
        vals = vals * np.power(2.0, sc.astype(np.float64) - 127.0)[..., None]
        ref = vals.reshape(rows, -1)
        got = M.dequant_values(b4.reshape(rows, cols // 2), sc)
        np.testing.assert_array_equal(got, ref)
        # one hand-computed byte: 0x21 -> element 0 = low nibble 1 (0.5), element 1 = high nibble 2 (1.0); e = 128
        b = np.zeros((1, 16), dtype=np.uint8)
        b[0, 0], b[0, 1] = 0x21, 0x9F                                         # 0x9F: lo = F (-6), hi = 9 (-0.5)
        v = M.dequant_values(b, np.array([[128]], dtype=np.uint8))
        self.assertEqual(v[0, :4].tolist(), [1.0, 2.0, -12.0, -1.0])
        self.assertEqual(M.unpack_nibbles(b)[0, :4].tolist(), [1, 2, 15, 9])

    def test_materialisation_equals_an_independent_fraction_reference(self):
        g = np.random.default_rng(8)
        for w in (8, 12, 23, 40, 52):
            for trial in range(6):
                b = _blocks(g, 32, 64)
                s = g.integers(0, 255, size=(32, 2), dtype=np.uint8)           # full E8M0 range e = 0..254
                units, shift = M.materialize_tile(b, s, w)
                ref, rshift = _ref_window(b, s, w)
                self.assertEqual(shift, rshift)
                self.assertEqual(units.reshape(-1).tolist(), ref)
        # hand-built round-half-even ties incl. negative values and -0 (w = 8)
        # row 0: scale 2^k, nibble 7 (6.0) -> top = 0.75 * 2^(k+3): E = k + 3, shift = 5 - k;
        # row 1 at scale 2^(k-5): nibble 1 (0.5) -> unit 0.5 -> 0 ; nibble 3 (1.5) -> 1.5 -> 2 ; negatives mirror ; -0
        k = 4
        b = np.zeros((32, 16), dtype=np.uint8)
        s = np.zeros((32, 1), dtype=np.uint8)
        b[0, 0], s[0, 0] = 0x07, 127 + k
        s[1, 0] = 127 + k - 5
        b[1, 0] = 0x31                                                        # [0.5, 1.5]
        b[1, 1] = 0xB9                                                        # [-0.5, -1.5]  (lo = 9, hi = B)
        b[1, 2] = 0x08                                                        # [-0.0, +0.0]
        units, shift = M.materialize_tile(b, s, 8)
        self.assertEqual(shift, 8 - (k + 3))
        self.assertEqual(units[0, 0], 3 << 6)
        self.assertEqual(units[1, :6].tolist(), [0, 2, 0, -2, 0, 0])          # ties go to even, signed zero -> 0
        ref, rshift = _ref_window(b, s, 8)
        self.assertEqual((units.reshape(-1).tolist(), shift), (ref, rshift))
        z, zs = M.materialize_tile(np.zeros((32, 16), np.uint8), np.zeros((32, 1), np.uint8), 23)
        self.assertEqual((int(np.abs(z).max()), zs), (0, 0))                  # all-zero tile: no 0/0 under -W error
        self.assertEqual(M.window_report(np.zeros((32, 16), np.uint8), np.zeros((32, 1), np.uint8))
                         ["quant_window_tail_rel_err"], 0.0)

    def test_strict_window_half_step_bound_and_integer_faithfulness(self):
        g = np.random.default_rng(21)
        tiles = [_wide_tile(1, 20), _wide_tile(2, 30),                         # 2^-20..2^20 and 2^-30..2^30 (60 binades)
                 (_blocks(g, 32, K), g.integers(0, 255, size=(32, K // 32), dtype=np.uint8))]   # e = 0..254
        for b, s in tiles:
            vals = M.dequant_values(b, s)
            top = float(np.abs(vals).max())
            for w in (8, 16, 23, 31, 52):
                units, shift = M.materialize_tile(b, s, w)
                mx = int(np.abs(units).max())
                self.assertLess(mx, 1 << w)                                      # STRICT: E2M1 tops are 2^(w-1) or 3 2^(w-2)
                self.assertIn(mx, (1 << (w - 1), 3 << (w - 2)))
                rep = M.window_report(b, s, w)
                self.assertLessEqual(rep["tail_abs"], 2.0 ** -w * top)           # analytical half-step bound
                self.assertLessEqual(rep["quant_window_tail_rel_err"], 2.0 ** -w)
                self.assertEqual(rep["max_abs_unit"], mx)
                keep = np.abs(vals) >= 2.0 ** (math.frexp(top)[1] - w + 1)     # entries within w - 1 binades are exact
                np.testing.assert_array_equal(np.ldexp(units.astype(np.float64), -shift)[keep], vals[keep])
        # the F_p statement is the integer statement at the default window: sum |W||X| < 2^62 (X windowed likewise)
        for b, s in tiles:
            Wu, _ = M.materialize_tile(b, s, 23)
            xv = np.ldexp(g.standard_normal(K), g.integers(-20, 20, K))
            Xu, _ = WC.window_ints(xv, 23)
            _, bits = FV.exact_products(Wu[None], Xu[None])
            self.assertLess(bits, 62)
            self.assertLessEqual(bits, 56)
            self.assertTrue(M.integer_faithful(K, 23))
        # ...and the flag is COMPUTED: at w = 52 the same tile is NOT faithful, and the bound really is violated
        self.assertFalse(M.integer_faithful(K, 52))
        self.assertTrue(M.integer_faithful(K, 23, 30, max_w=1 << 21))
        self.assertFalse(M.integer_faithful(K, 23, 30))
        b, s = tiles[0]
        Wu, _ = M.materialize_tile(b, s, 52)
        Xu, _ = WC.window_ints(g.standard_normal(K), 23)
        self.assertGreaterEqual(FV.exact_products(Wu[None], Xu[None])[1], 62)

    def test_leaf_payload_is_3_76x_smaller_than_bf16(self):
        com = M.commit_mxfp4_weights(_source(1))
        self.assertEqual(com.stats["leaf_payload_bytes"], 17408)
        self.assertEqual(com.stats["bf16_tile_bytes"], 65536)
        self.assertEqual(32 * K // 2, 16384)
        self.assertEqual(32 * K // 32, 1024)
        self.assertGreater(com.stats["compression_vs_bf16"], 3.76)
        self.assertAlmostEqual(com.stats["compression_vs_bf16"], 65536 / 17408, places=12)
        op = M.open_mxfp4_tiles(com, _source(1), [("model.layers.1.mlp.experts.gate_up_proj[1]", 32, 0)])
        enc = op.encode()
        t = op.tiles[0]
        self.assertEqual(len(enc), len(M._OPENING_MAGIC) + 16 + len(op.manifest) + 2 + len(t.name.encode()) + 20
                         + 32 * len(t.path) + 17408)
        # the WC BF16 leaf hashes int64 units: 262,144 B for the same tile (the 3.76x is against RAW bf16 bytes)
        self.assertEqual(32 * K * 8, 262144)

    def test_commitment_binds_every_byte_name_k_window_and_residual(self):
        src = _source(1)
        com = M.commit_mxfp4_weights(src)
        self.assertEqual(M.commit_mxfp4_weights(src, threads=1).root, com.root)
        self.assertEqual(M.commit_mxfp4_weights(list(reversed(src))).root, com.root)       # order independent
        self.assertEqual(M.commit_mxfp4_weights(dict((n, (b, s)) for n, b, s in src)).root, com.root)
        self.assertEqual(com.stats["tensors"], 8)
        self.assertEqual(com.stats["leaves"], sum(e.leaves for e in com.entries))
        roots = {com.root}

        def variant(i, which, pos):
            out = [(n, b.copy(), s.copy()) for n, b, s in src]
            arr = out[i][1 if which == "b" else 2]
            arr[pos] ^= 1
            return out
        for i in (0, 3, 7):
            for which, pos in (("b", (31, 5)), ("b", (0, 0)), ("s", (31, 0)), ("s", (0, 3))):
                roots.add(M.commit_mxfp4_weights(variant(i, which, pos)).root)
        roots.add(M.commit_mxfp4_weights([("x" + n, b, s) for n, b, s in src]).root)
        roots.add(M.commit_mxfp4_weights(src, K=512).root)
        roots.add(M.commit_mxfp4_weights(src, window_bits=24).root)
        roots.add(M.commit_mxfp4_weights(src, residual=b"\x01" * 32).root)
        self.assertEqual(len(roots), 1 + 12 + 4)
        # ragged edge: 1504 columns -> 2 column tiles, the second zero-padded (blocks and scales)
        e = com.entry("model.layers.0.mlp.experts.down_proj[0]")
        self.assertEqual((e.rows, e.cols, e.col_tiles), (32, 1504, 2))
        op = M.open_mxfp4_tiles(com, src, [(e.name, 0, K)])
        t = op.tiles[0]
        self.assertFalse(t.blocks[:, (1504 - K) // 2:].any())
        self.assertFalse(t.scales[:, (1504 - K) // 32:].any())
        self.assertTrue(t.blocks[:, :(1504 - K) // 2].any())

    def test_ragged_rows_zero_padded_and_bench_convention_pinned(self):
        g = np.random.default_rng(9)
        b, s = _blocks(g, 40, K), _scales(g, 40, K)                              # 40 rows: second row tile is 8 rows + padding
        com = M.commit_mxfp4_weights([("r", b, s)])
        self.assertEqual((com.entry("r").row_tiles, com.stats["leaves"]), (2, 2))
        t = M.open_mxfp4_tiles(com, [("r", b, s)], [("r", 32, 0)]).tiles[0]
        np.testing.assert_array_equal(t.blocks[:8], b[32:])
        np.testing.assert_array_equal(t.scales[:8], s[32:])
        self.assertFalse(t.blocks[8:].any() or t.scales[8:].any())
        self.assertEqual(M.verify_mxfp4_tiles_against_setup(
            SetupCertificate(com.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode(),
            M.open_mxfp4_tiles(com, [("r", b, s)], [("r", 32, 0)]), [("r", 32, 0)], b"\x00" * 5)["verdict"],
            "REJECTED: E_PROFILE")                                               # container check precedes everything
        # the dequant convention this module re-implements is still the one in the calibration bench source
        with open(os.path.join(ROOT, "bench", "tier2", "bench_frontier_calibrate.py"), encoding="utf-8") as fh:
            bench = fh.read()
        self.assertIn("[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]", bench)
        self.assertIn("lo, hi = (blocks & 0x0F).long(), (blocks >> 4).long()", bench)
        self.assertIn("torch.stack([lut[lo], lut[hi]], dim=-1)", bench)
        self.assertIn("scales.double() - 127.0", bench)

    def test_helper_arguments_are_strictly_typed(self):
        for args in ((1024, 23, None, -5), (1024, 23, None, True), (1024, 23, None, 1.5), (1024, 23, None, None, -1),
                     (1024, 99), (1000, 23), (True, 23), (1024, True)):
            with self.assertRaises(WC.WeightCommitError):
                M.integer_faithful(*args)
        g = np.random.default_rng(1)
        b, s = _blocks(g, 32, K), _scales(g, 32, K)
        for kw in ({"threads": 0}, {"threads": -1}, {"threads": "8"}, {"threads": True}, {"chunk_row_tiles": 0},
                   {"chunk_row_tiles": -3}, {"chunk_row_tiles": 1.0}):
            with self.assertRaises(WC.WeightCommitError):
                M.commit_mxfp4_weights([("a", b, s)], **kw)
        e = M.Mxfp4Entry("a", 32, K, 0, K)
        for bad in (M.Mxfp4Entry("a", -32, K, 0, K), M.Mxfp4Entry("a", 32, K, -1, K), M.Mxfp4Entry("a", 32.0, K, 0, K)):
            with self.assertRaises(WC.WeightCommitError):
                M.encode_manifest(K, [bad])
        com = M.commit_mxfp4_weights([("a", b, s)])
        t = M.open_mxfp4_tiles(com, [("a", b, s)], [("a", 0, 0)]).tiles[0]
        for bad in (M.OpenedMxfp4Tile("a", -1, 0, t.blocks, t.scales, t.path),
                    M.OpenedMxfp4Tile("a", 0, True, t.blocks, t.scales, t.path),
                    M.OpenedMxfp4Tile("a", 0, 0, t.blocks, t.scales, None)):
            with self.assertRaises(WC.WeightCommitError):
                M.Mxfp4TileOpening(K, com.manifest, (bad,)).encode()
        for bad in (M.Mxfp4TileOpening(K, com.manifest, None), M.Mxfp4TileOpening("x", com.manifest, ()),
                    M.Mxfp4TileOpening(K, "m", ())):
            with self.assertRaises(WC.WeightCommitError):
                bad.encode()
        self.assertTrue(e.leaves == 1)

    def test_commit_rejects_every_malformed_input(self):
        g = np.random.default_rng(5)
        b, s = _blocks(g, 32, K), _scales(g, 32, K)

        def code(src, **kw):
            with self.assertRaises(WC.WeightCommitError) as cm:
                M.commit_mxfp4_weights(src, **kw)
            return cm.exception.code
        self.assertEqual(code([("a", b, s), ("a", b, s)]), "E_DUPLICATE")                      # duplicate names
        self.assertEqual(code([("", b, s)]), "E_TILE_RANGE")
        self.assertEqual(code([(b"a", b, s)]), "E_TILE_RANGE")
        self.assertEqual(code([("\ud800", b, s)]), "E_TILE_RANGE")
        self.assertEqual(code([("a" * 70000, b, s)]), "E_TILE_RANGE")
        self.assertEqual(code([]), "E_MANIFEST")
        self.assertEqual(code({}), "E_MANIFEST")
        self.assertEqual(code({"a": (b, s, s)}), "E_TILE_SHAPE")
        self.assertEqual(code([("a", b)]), "E_TILE_SHAPE")
        for bad_b in (b.astype(np.int8), b.astype(np.uint16), b.astype(np.int64), b.astype(bool), b.tolist()):
            self.assertEqual(code([("a", bad_b, s)]), "E_TILE_SHAPE")                         # no silent coercion
        for bad_s in (s.astype(np.int8), s.astype(np.uint16), s.astype(np.float32), s.tolist()):
            self.assertEqual(code([("a", b, bad_s)]), "E_TILE_SHAPE")
        self.assertEqual(code([("a", b[:, :K // 2 - 8], s[:, :K // 32 - 1])]), "E_TILE_SHAPE")     # width 1008: not 32 | K
        self.assertEqual(code([("a", b, s[:, :-1])]), "E_TILE_SHAPE")                          # scales mismatch
        self.assertEqual(code([("a", b, s[:-1])]), "E_TILE_SHAPE")
        self.assertEqual(code([("a", b.reshape(32, K // 32, 16)[..., :15], s)]), "E_TILE_SHAPE")
        self.assertEqual(code([("a", b[:0], s[:0])]), "E_TILE_SHAPE")                          # zero rows
        self.assertEqual(code([("a", b[:, :0], s[:, :0])]), "E_TILE_SHAPE")                    # zero columns
        self.assertEqual(code([("a", b.reshape(2, 16, K // 2, 1), s)]), "E_TILE_SHAPE")
        nan = s.copy()
        nan[7, 3] = 255
        self.assertEqual(code([("a", b, nan)]), "E_TILE_ENCODING")                              # NaN scale
        with self.assertRaises(ValueError):
            M.commit_mxfp4_weights([("a", b, nan)])
        for bad_k in (0, 16, 31, 48, 1000, K + 1, True, 32.0, "1024", -32, 1 << 21):
            self.assertEqual(code([("a", b, s)], K=bad_k), "E_TILE_SHAPE")
        for bad_w in (7, 0, 53, True, 23.0, "23", None):
            self.assertEqual(code([("a", b, s)], window_bits=bad_w), "E_WINDOW")
        self.assertEqual(code([("a", b, s)], residual=b"\x00" * 31), "E_MANIFEST")
        self.assertEqual(code([("a", b, s)], residual="x" * 32), "E_MANIFEST")
        for bad in (1, 2.5, None):
            with self.assertRaises((TypeError, WC.WeightCommitError)):
                M.commit_mxfp4_weights(bad)
        # open: a source that is not the committed checkpoint, NaN injected after commit, missing / unknown tensors
        com = M.commit_mxfp4_weights([("a", b, s)])
        flipped = b.copy()
        flipped[3, 3] ^= 4
        with self.assertRaises(WC.WeightCommitError) as cm:
            M.open_mxfp4_tiles(com, [("a", flipped, s)], [("a", 0, 0)])
        self.assertEqual(cm.exception.code, "E_TILE_PATH")
        with self.assertRaises(WC.WeightCommitError) as cm:
            M.open_mxfp4_tiles(com, [("a", b, nan)], [("a", 0, 0)])
        self.assertEqual(cm.exception.code, "E_TILE_ENCODING")
        with self.assertRaises(WC.WeightCommitError) as cm:
            M.open_mxfp4_tiles(com, [("a", b[:, :K // 2 - 16], s[:, :K // 32 - 1])], [("a", 0, 0)])
        self.assertEqual(cm.exception.code, "E_TILE_SHAPE")
        for coords in ([("zz", 0, 0)], [("a", 1, 0)], [("a", 0, 32)], [("a", 32, 0)], [("a", 0, True)], [("a", -32, 0)]):
            with self.assertRaises(WC.WeightCommitError):
                M.open_mxfp4_tiles(com, [("a", b, s)], coords)
        with self.assertRaises(WC.WeightCommitError) as cm:
            M.open_mxfp4_tiles(com, [], [("a", 0, 0)])
        self.assertEqual(cm.exception.code, "E_TILE_RANGE")

    def test_manifest_and_opening_decoders_are_strict(self):
        src = _source(1, layers=1)
        com = M.commit_mxfp4_weights(src)
        K_, w, count, entries, residual = M.decode_manifest(com.manifest)
        self.assertEqual((K_, w, count, residual), (K, 23, com.stats["leaves"], b"\x00" * 32))
        self.assertEqual(entries, com.entries)
        self.assertEqual(M.manifest_window_bits(com.manifest), 23)
        mag = len(M._MANIFEST_MAGIC) + 32
        woff = mag + 4 + 4 + 8 + 4

        def mcode(data):
            with self.assertRaises(WC.WeightCommitError) as cm:
                M.decode_manifest(data)
            return cm.exception.code
        self.assertEqual(mcode(com.manifest + b"\x00"), "E_MANIFEST")                          # trailing byte
        self.assertEqual(mcode(com.manifest[:-1]), "E_MANIFEST")                               # truncated
        self.assertEqual(mcode(b"x" + com.manifest[1:]), "E_MANIFEST")
        self.assertEqual(mcode(None), "E_MANIFEST")
        for bad_w in (7, 53, 0, 1 << 20):
            m = bytearray(com.manifest)
            m[woff:woff + 4] = struct.pack("<I", bad_w)
            self.assertEqual(mcode(bytes(m)), "E_MANIFEST")
        m = bytearray(com.manifest)
        m[mag + 4:mag + 8] = struct.pack("<I", 48)                                             # K not 32 | 1024... K = 48
        self.assertEqual(mcode(bytes(m)), "E_MANIFEST")
        m = bytearray(com.manifest)
        m[mag:mag + 4] = struct.pack("<I", 16)                                                 # rows != 32
        self.assertEqual(mcode(bytes(m)), "E_MANIFEST")
        e0 = com.entries[0]
        dup = M.encode_manifest(K, [e0, M.Mxfp4Entry(e0.name, e0.rows, e0.cols, e0.leaves, K)])
        self.assertEqual(mcode(dup), "E_MANIFEST")                                             # duplicate tensor
        nc = M.encode_manifest(K, [e0, M.Mxfp4Entry("b", 32, 1024, e0.leaves + 1, K)])
        self.assertEqual(mcode(nc), "E_MANIFEST")                                              # leaf gap
        odd = M.encode_manifest(K, [M.Mxfp4Entry("b", 32, 1000, 0, K)])
        self.assertEqual(mcode(odd), "E_MANIFEST")                                             # 32 does not divide cols
        zero = M.encode_manifest(K, [M.Mxfp4Entry("b", 0, 1024, 0, K)])
        self.assertEqual(mcode(zero), "E_MANIFEST")
        cnt = bytearray(com.manifest)
        cnt[mag + 8:mag + 16] = struct.pack("<Q", com.stats["leaves"] + 1)
        self.assertEqual(mcode(bytes(cnt)), "E_MANIFEST")
        # opening
        op = M.open_mxfp4_tiles(com, src, [(com.entries[2].name, 32, 0), (com.entries[0].name, 0, 0)])
        enc = op.encode()
        back = M.Mxfp4TileOpening.decode(enc)
        self.assertEqual(back.encode(), enc)
        self.assertEqual(back.coords(), op.coords())

        def ocode(data):
            with self.assertRaises(WC.WeightCommitError) as cm:
                M.Mxfp4TileOpening.decode(data)
            return cm.exception.code
        self.assertEqual(ocode(enc + b"\x00"), "E_OPENING")
        self.assertEqual(ocode(enc[:-1]), "E_OPENING")
        self.assertEqual(ocode(enc[:len(M._OPENING_MAGIC) + 8]), "E_OPENING")
        self.assertEqual(ocode(b"\x00" + enc[1:]), "E_OPENING")
        self.assertEqual(ocode("x"), "E_OPENING")
        hdr = len(M._OPENING_MAGIC)
        bomb = bytearray(enc)                                                                  # K / tile-count / depth bombs
        bomb[hdr + 4:hdr + 8] = struct.pack("<I", 0xFFFFFFFF)
        self.assertEqual(ocode(bytes(bomb)), "E_OPENING")
        bomb = bytearray(enc)
        bomb[hdr:hdr + 4] = struct.pack("<I", 48)
        self.assertEqual(ocode(bytes(bomb)), "E_OPENING")
        nm = len(com.entries[2].name.encode())
        depth_off = hdr + 16 + len(op.manifest) + 2 + nm + 16
        bomb = bytearray(enc)
        bomb[depth_off:depth_off + 4] = struct.pack("<I", 0xFFFFFFFF)                          # 4 G iterations if unbounded
        self.assertEqual(ocode(bytes(bomb)), "E_OPENING")
        bomb[depth_off:depth_off + 4] = struct.pack("<I", 40)                                  # in bounds, not in the data
        self.assertEqual(ocode(bytes(bomb)), "E_OPENING")
        # encode refuses malformed tiles instead of coercing them
        t = op.tiles[0]
        for bad in (M.OpenedMxfp4Tile(t.name, t.r0, t.c0, t.blocks.astype(np.int8), t.scales, t.path),
                    M.OpenedMxfp4Tile(t.name, t.r0, t.c0, t.blocks, t.scales[:, :-1], t.path),
                    M.OpenedMxfp4Tile(t.name, t.r0, t.c0, t.blocks, t.scales, t.path[:-1] + (b"\x00" * 31,))):
            with self.assertRaises(WC.WeightCommitError):
                M.Mxfp4TileOpening(op.K, op.manifest, (bad,)).encode()

    def test_gpt_oss_safetensors_layout_equals_in_memory(self):
        T = _gpt_oss_tensors(4, layers=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            _write_u8_safetensors(path, T)
            ck = WC.SafetensorsCheckpoint(path)
            items = M.experts_from_checkpoint(ck)
            self.assertEqual(len(items), 4)                                                    # 2 projections x 2 experts
            self.assertEqual(M.commit_mxfp4_weights(items).root,
                             M.commit_mxfp4_weights(_source(4, layers=1)).root)
            only = M.experts_from_checkpoint(ck, include=lambda n: "down_proj" in n)
            self.assertEqual([n for n, _, _ in only], ["model.layers.0.mlp.experts.down_proj[0]",
                                                       "model.layers.0.mlp.experts.down_proj[1]"])
            bad = dict(T)
            del bad["model.layers.0.mlp.experts.down_proj_scales"]
            _write_u8_safetensors(path, bad)
            with self.assertRaises(WC.WeightCommitError):
                M.experts_from_checkpoint(WC.SafetensorsCheckpoint(path))
        with self.assertRaises(WC.WeightCommitError):
            M.expand_experts("x", np.zeros((2, 32, 32, 16), np.uint8), np.zeros((3, 32, 32), np.uint8))

    def test_module_never_reads_the_environment_nor_imports_fpn_serve(self):
        with open(M.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        env = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == "os" and n.attr in ("environ", "getenv", "putenv", "environb")]
        self.assertEqual(env, [])
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                imports.append(n.module or "")
        self.assertFalse([i for i in imports if i.split(".")[0] in ("fpn_serve", "os")], imports)
        with open(M.__file__, encoding="utf-8") as fh:
            self.assertNotIn("environ", fh.read().replace("environment", ""))

    def test_domain_separation_tags(self):
        tags = (M._T_LEAF, M._T_NODE, M._T_MANI, M._T_ROOT, M._T_PAD)
        self.assertEqual(tags[:4], (b"FPN2-MXFP4-LEAF-v1", b"FPN2-MXFP4-NODE-v1", b"FPN2-MXFP4-MANI-v1",
                                    b"FPN2-MXFP4-ROOT-v1"))
        self.assertEqual(len(set(tags)), 5)
        self.assertTrue(all(len(t) > 16 for t in tags[:4]))              # longer than BLAKE2b `person`: message prefix
        self.assertEqual(len({M._h(t, b"x") for t in tags}), 5)
        # the same bytes under the BF16 / FP8 trees can never collide with an MXFP4 hash
        self.assertNotEqual(M._h(M._T_LEAF, b"x"), WC._h(WC._P_LEAF8, b"x"))


# ===================================================================================== end to end + forgeries
class Mxfp4RelationEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CE.HostEngine()
        cls.wide = _wide_tile(7, 20)
        cls.src = _source(1, wide=cls.wide)
        cls.com = M.commit_mxfp4_weights(cls.src)
        cls.setup = SetupCertificate(cls.com.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        cls.lay = FB.layout_of(PROF)
        cls.coords = [("wide", 0, 0)]
        cls.opening = M.open_mxfp4_tiles(cls.com, cls.src, cls.coords)
        g = np.random.default_rng(12)
        cls.W_int = cls.opening.weights_int()[0][0][None]
        cls.xv = np.ldexp(g.standard_normal(K), g.integers(-6, 6, K))
        cls.X_int = WC.window_ints(cls.xv, 23)[0][None]
        Y_int, cls.bits = FV.exact_products(cls.W_int, cls.X_int)
        cls.W, cls.X, cls.Y = FV.to_field(cls.W_int), FV.to_field(cls.X_int), FV.to_field(Y_int)
        cls.xc = FB.commit_x(cls.lay, cls.X, rng=_Rng(1), engine=cls.engine)
        cls.xroot = cls.xc.root
        cls.container = FB.prove_bound(PROF, cls.W, cls.X, cls.Y, cls.xc, n_s=b"\x07" * 32, rng=_Rng(3),
                                       engine=cls.engine)
        cls.wroot = bytes(cls.container[96:128])

    def check(self, opening=None, container=None, coords=None, setup=None, **kw):
        return M.verify_mxfp4_tiles_against_setup(setup or self.setup, opening or self.opening,
                                                  coords if coords is not None else self.coords,
                                                  container or self.container, engine=self.engine, **kw)

    def _tile(self, opening=None, **kw):
        t = (opening or self.opening).tiles[0]
        f = dict(name=t.name, r0=t.r0, c0=t.c0, blocks=t.blocks, scales=t.scales, path=t.path)
        f.update(kw)
        return M.Mxfp4TileOpening(self.opening.K, self.opening.manifest, (M.OpenedMxfp4Tile(**f),))

    # ---------------------------------------------------------------- honest
    def test_honest_mxfp4_to_relation6_with_rust_and_detached_python(self):
        enc = self.opening.encode()
        self.assertEqual(M.Mxfp4TileOpening.decode(enc).encode(), enc)
        rep = self.check(enc)
        self.assertEqual(rep["verdict"], "VERIFIED", rep)
        self.assertEqual((rep["relation"], rep["profile"]), (6, "fpn2-folded-vec-bound-n16-v1"))
        self.assertEqual(rep["weight_root"], self.wroot.hex())                              # STMT[32..64]
        self.assertEqual(rep["x_root"], self.xroot.hex())
        self.assertEqual(rep["window_bits"], 23)
        self.assertTrue(rep["integer_faithful"])
        self.assertTrue(rep["window_strict"])
        self.assertLess(rep["max_abs_unit"], 1 << 23)
        self.assertLessEqual(rep["quant_window_tail_rel_err"], 2.0 ** -23)
        self.assertGreater(rep["quant_window_tail_rel_err"], 0.0)       # the 2^-20..2^20 tile does lose its small blocks
        self.assertEqual(rep["leaf_payload_bytes"], 17408)
        self.assertLess(self.bits, 62)
        self.assertEqual(self.check(enc, expected_window_bits=23)["verdict"], "VERIFIED")
        rs, py = _verdicts(self.container, bytes.fromhex(rep["weight_root"]), self.xroot)
        self.assertEqual(py, "VERIFIED")
        if rs is not None:
            self.assertEqual(rs, "VERIFIED")

    def test_rust_binary_leg_ran(self):
        if not os.path.exists(RUST):
            self.skipTest("core/target/release/fpn-verify not built: `cargo build --release --locked` in core/")
        out = _rust(self.container, self.wroot, self.xroot)
        self.assertEqual(out, "VERIFIED")

    def test_relation5_container_over_the_same_tile_also_verifies(self):
        c5 = FV.prove(self.W, self.X, self.Y, n_s=b"\x05" * 32, rng=_Rng(4), engine=self.engine)
        rep = self.check(container=c5)
        self.assertEqual(rep["verdict"], "VERIFIED", rep)
        self.assertEqual((rep["relation"], rep["profile"]), (5, "fpn2-folded-vec-n16-v1"))
        self.assertIsNone(rep["x_root"])
        self.assertEqual(rep["weight_root"], bytes(c5[96:128]).hex())
        self.assertEqual(E2E.verdict_code(E2E.python_verify_detached(c5, bytes.fromhex(rep["weight_root"]))[0]),
                         "VERIFIED")

    # ---------------------------------------------------------------- forgeries (each with ONE deterministic code)
    def test_f01_one_nibble_flip_rejected(self):
        b = self.opening.tiles[0].blocks.copy()
        b[5, 9] ^= 0x01
        self.assertEqual(self.check(self._tile(blocks=b))["verdict"], "REJECTED: E_TILE_PATH")
        b = self.opening.tiles[0].blocks.copy()
        b[31, 511] ^= 0x80                                               # the very last element (high nibble)
        self.assertEqual(self.check(self._tile(blocks=b))["verdict"], "REJECTED: E_TILE_PATH")

    def test_f02_one_scale_byte_flip_rejected(self):
        s = self.opening.tiles[0].scales.copy()
        s[3, 2] ^= 0x01
        self.assertEqual(self.check(self._tile(scales=s))["verdict"], "REJECTED: E_TILE_PATH")

    def test_f03_e8m0_nan_scale_rejected_three_ways(self):
        s = self.opening.tiles[0].scales.copy()
        s[3, 2] = 255
        self.assertEqual(self.check(self._tile(scales=s))["verdict"], "REJECTED: E_TILE_PATH")      # injected into an opening
        with self.assertRaises(ValueError):                                                        # the committer refuses
            M.commit_mxfp4_weights([("wide", self.wide[0], s)])
        # a malicious committer builds a self-consistent tree OVER a NaN tile: inclusion passes, materialisation refuses
        b = self.opening.tiles[0].blocks
        entries = [M.Mxfp4Entry("wide", 32, K, 0, K)]
        levels = M.build_levels([M.leaf_hash("wide", 0, 0, K, b, s)])
        manifest = M.encode_manifest(K, entries, 23)
        root = M.global_root(manifest, levels[-1][0])
        setup = SetupCertificate(root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        op = M.Mxfp4TileOpening(K, manifest, (M.OpenedMxfp4Tile("wide", 0, 0, b, s, ()),))
        rep = self.check(op, setup=setup)
        self.assertEqual(rep["verdict"], "REJECTED: E_TILE_ENCODING")
        self.assertEqual(rep["tiles"], 1)                                                          # inclusion DID pass
        s_ok = s.copy()
        s_ok[3, 2] = 254                                                                           # largest legal exponent
        self.assertIsNotNone(M.materialize_tile(b, s_ok, 23))

    def test_f04_manifest_window_bits_tamper_rejected(self):
        off = len(M._MANIFEST_MAGIC) + 32 + 4 + 4 + 8 + 4

        def with_window(v):
            m = bytearray(self.opening.manifest)
            m[off:off + 4] = struct.pack("<I", v)
            return M.Mxfp4TileOpening(self.opening.K, bytes(m), self.opening.tiles)
        self.assertEqual(self.check(with_window(24))["verdict"], "REJECTED: E_TILE_PATH")          # valid width, root changes
        self.assertEqual(self.check(with_window(22))["verdict"], "REJECTED: E_TILE_PATH")
        self.assertEqual(self.check(with_window(7))["verdict"], "REJECTED: E_MANIFEST")            # out of range
        self.assertEqual(self.check(with_window(53))["verdict"], "REJECTED: E_MANIFEST")
        # a commitment made at another window is a different root
        other = M.commit_mxfp4_weights(self.src, window_bits=24)
        self.assertNotEqual(other.root, self.com.root)
        op24 = M.open_mxfp4_tiles(other, self.src, self.coords)
        self.assertEqual(self.check(op24)["verdict"], "REJECTED: E_TILE_PATH")                      # not under OUR root

    def test_f05_genuine_tile_of_another_layer_rejected(self):
        other = M.open_mxfp4_tiles(self.com, self.src, [("model.layers.1.mlp.experts.gate_up_proj[0]", 32, 0)])
        self.assertEqual(self.check(other)["verdict"], "REJECTED: E_TILE_COORDS")

    def test_f06_relabelled_tile_rejected(self):
        other = M.open_mxfp4_tiles(self.com, self.src, [("model.layers.1.mlp.experts.gate_up_proj[0]", 32, 0)])
        o = other.tiles[0]
        self.assertEqual(self.check(self._tile(blocks=o.blocks, scales=o.scales, path=o.path))["verdict"],
                         "REJECTED: E_TILE_PATH")                       # lied coordinates, genuine data + path
        self.assertEqual(self.check(self._tile(blocks=o.blocks, scales=o.scales))["verdict"],
                         "REJECTED: E_TILE_PATH")                       # other layer's data under our path
        self.assertEqual(self.check(self._tile(path=o.path))["verdict"], "REJECTED: E_TILE_PATH")

    def test_f07_different_models_setup_certificate_rejected(self):
        com2 = M.commit_mxfp4_weights(_source(2, wide=self.wide))
        setup2 = SetupCertificate(com2.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        self.assertEqual(self.check(setup=setup2)["verdict"], "REJECTED: E_TILE_PATH")
        op2 = M.open_mxfp4_tiles(com2, _source(2, wide=self.wide), self.coords)       # same wide tile, other model
        self.assertEqual(self.check(op2)["verdict"], "REJECTED: E_TILE_PATH")
        self.assertEqual(self.check(setup=b"\x00" * 10)["verdict"], "REJECTED: E_SETUP")
        self.assertEqual(self.check(setup="x")["verdict"], "REJECTED: E_SETUP")

    def test_f08_proof_on_altered_weights_rejected(self):
        t = self.opening.tiles[0]
        # one E2M1 element of the largest-scale block changed (a flip below the window would round to the same W:
        # only the lossless leaf sees it, see test_window_blind_flip_is_caught_by_the_lossless_leaf)
        r, blk = np.unravel_index(int(np.argmax(t.scales)), t.scales.shape)
        b2 = t.blocks.copy()
        b2[r, blk * 16 + 3] ^= 0x04
        W2 = M.materialize_tile(b2, t.scales, 23)[0][None]
        self.assertFalse(np.array_equal(W2, self.W_int))
        Y2, _ = FV.exact_products(W2, self.X_int)
        bad = FB.prove_bound(PROF, FV.to_field(W2), self.X, FV.to_field(Y2), self.xc, n_s=b"\x07" * 32, rng=_Rng(3),
                             engine=self.engine)
        rs, py = _verdicts(bad, bytes(bad[96:128]), self.xroot)
        self.assertEqual(py, "VERIFIED")                                 # internally valid under ITS OWN weight root
        self.assertEqual(self.check(container=bad)["verdict"], "REJECTED: E_WEIGHT_ROOT")
        self.assertEqual(self.check(self._tile(blocks=b2), container=bad)["verdict"], "REJECTED: E_TILE_PATH")

    def test_window_blind_flip_is_caught_by_the_lossless_leaf(self):
        """An element 40 binades below the tile top rounds to 0 in W (the window), yet its nibble is committed."""
        t = self.opening.tiles[0]
        r, blk = np.unravel_index(int(np.argmin(t.scales)), t.scales.shape)
        b2 = t.blocks.copy()
        b2[r, blk * 16 + 3] ^= 0x04
        self.assertTrue(np.array_equal(M.materialize_tile(b2, t.scales, 23)[0], self.opening.weights_int()[0][0]))
        self.assertEqual(self.check(self._tile(blocks=b2))["verdict"], "REJECTED: E_TILE_PATH")

    def test_f09_case_c_substituted_activations_rejected_by_relation6(self):
        Xs_i = self.X_int.copy()
        Xs_i[0, 11] += 1
        Xs = FV.to_field(Xs_i)
        Ys = FV.to_field(FV.exact_products(self.W_int, Xs_i)[0])
        # relation 5 accepts the substituted (X', Y' = W X'): the gap relation 6 closes
        r5 = FV.prove(self.W, Xs, Ys, n_s=b"\x07" * 32, rng=_Rng(3), engine=self.engine)
        self.assertEqual(E2E.verdict_code(E2E.python_verify_detached(r5, self.wroot)[0]), "VERIFIED")
        xcs = FB.commit_x(self.lay, Xs, rng=_Rng(5), engine=self.engine)
        # (1) re-prove (X', Y') but claim the honest x_root in the statement
        c1 = FB.prove_bound(PROF, self.W, Xs, Ys, xcs, rng=_Rng(6), engine=self.engine,
                            hooks={"stmt_x_root": self.xroot})
        # (2) the cheater's own x_root'
        c2 = FB.prove_bound(PROF, self.W, Xs, Ys, xcs, rng=_Rng(6), engine=self.engine)
        for data, want in ((c1, "E_MERKLE"), (c2, "E_X_ROOT_PINNED")):
            rs, py = _verdicts(data, self.wroot, self.xroot)                  # honest pins: weight_root AND x_root
            self.assertEqual(py, want)
            if rs is not None:
                self.assertEqual(rs, want)
        rs, py = _verdicts(c2, self.wroot, xcs.root)                          # the cheater's proof is valid under ITS pin
        self.assertEqual(py, "VERIFIED")
        # the MXFP4 opening check binds W ONLY: the X-substituted container passes it -- x_root pinning is the caller's job
        self.assertEqual(self.check(container=c2)["verdict"], "VERIFIED")
        self.assertEqual(self.check(container=c2)["x_root"], xcs.root.hex())
        self.assertNotEqual(xcs.root, self.xroot)

    def test_f10_trailing_bytes_rejected(self):
        enc = self.opening.encode()
        rep = self.check(enc + b"\x00")
        self.assertEqual(rep["verdict"], "REJECTED: E_OPENING")
        self.assertEqual(self.check(enc[:-1])["verdict"], "REJECTED: E_OPENING")
        op = M.Mxfp4TileOpening(self.opening.K, self.opening.manifest + b"\x00", self.opening.tiles)
        self.assertEqual(self.check(op)["verdict"], "REJECTED: E_MANIFEST")
        self.assertEqual(self.check(op.encode())["verdict"], "REJECTED: E_MANIFEST")

    def test_f11_mismatched_window_bits_rejected(self):
        self.assertEqual(self.check(expected_window_bits=24)["verdict"], "REJECTED: E_WINDOW")     # verifier pins 24
        self.assertEqual(self.check(expected_window_bits=8)["verdict"], "REJECTED: E_WINDOW")
        self.assertEqual(self.check(expected_window_bits=7)["verdict"], "REJECTED: E_WINDOW")
        self.assertEqual(self.check(expected_window_bits=True)["verdict"], "REJECTED: E_WINDOW")
        # container proved on W materialised at ANOTHER width (manifest says 23)
        W22 = self.opening.weights_int(22)[0][0][None]
        Y22, _ = FV.exact_products(W22, self.X_int)
        bad = FB.prove_bound(PROF, FV.to_field(W22), self.X, FV.to_field(Y22), self.xc, n_s=b"\x07" * 32,
                             rng=_Rng(3), engine=self.engine)
        self.assertEqual(self.check(container=bad)["verdict"], "REJECTED: E_WEIGHT_ROOT")
        # ...but a commitment bound at 22 verifies that same container
        com22 = M.commit_mxfp4_weights(self.src, window_bits=22)
        setup22 = SetupCertificate(com22.root, b"\x11" * 32, b"\x22" * 32, b"\x33" * 32, 0).encode()
        op22 = M.open_mxfp4_tiles(com22, self.src, self.coords)
        rep = self.check(op22, container=bad, setup=setup22, expected_window_bits=22)
        self.assertEqual(rep["verdict"], "VERIFIED", rep)

    def test_f12_wrong_profile_tile_count_coords_and_shapes_rejected(self):
        two = M.open_mxfp4_tiles(self.com, self.src, self.coords * 2)
        self.assertEqual(self.check(two, coords=self.coords * 2)["verdict"], "REJECTED: E_PROFILE")
        self.assertEqual(self.check(container=b"\x00" * len(self.container))["verdict"], "REJECTED: E_PROFILE")
        self.assertEqual(self.check(container=self.container[:-1])["verdict"], "REJECTED: E_PROFILE")
        self.assertEqual(self.check(container="x")["verdict"], "REJECTED: E_PROFILE")
        self.assertEqual(self.check(coords=[("wide", 32, 0)])["verdict"], "REJECTED: E_TILE_COORDS")
        self.assertEqual(self.check(coords=5)["verdict"], "REJECTED: E_TILE_COORDS")
        self.assertEqual(self.check(opening=5)["verdict"], "REJECTED: E_OPENING")
        t = self.opening.tiles[0]
        for broken in (M.Mxfp4TileOpening(K, self.opening.manifest, None), M.Mxfp4TileOpening("x", b"", ()),
                       M.Mxfp4TileOpening(K, self.opening.manifest, (5,)), self._tile(path=None)):
            self.assertEqual(self.check(broken)["verdict"], "REJECTED: E_OPENING")
        self.assertEqual(self.check(self._tile(r0=1), coords=[("wide", 1, 0)])["verdict"], "REJECTED: E_TILE_RANGE")
        self.assertEqual(self.check(self._tile(name="nope"), coords=[("nope", 0, 0)])["verdict"],
                         "REJECTED: E_TILE_RANGE")
        self.assertEqual(self.check(self._tile(path=t.path[:-1]))["verdict"], "REJECTED: E_TILE_PATH")
        self.assertEqual(self.check(self._tile(path=t.path + (b"\x00" * 32,)))["verdict"], "REJECTED: E_TILE_PATH")
        self.assertEqual(self.check(self._tile(path=t.path[:-1] + (b"\x00" * 31,)))["verdict"], "REJECTED: E_TILE_PATH")
        self.assertEqual(self.check(self._tile(blocks=t.blocks.astype(np.int8)))["verdict"], "REJECTED: E_TILE_SHAPE")
        self.assertEqual(self.check(self._tile(scales=t.scales[:, :-1]))["verdict"], "REJECTED: E_TILE_SHAPE")
        self.assertEqual(self.check(M.Mxfp4TileOpening(K // 2, self.opening.manifest, self.opening.tiles))["verdict"],
                         "REJECTED: E_PROFILE")
        self.assertEqual(self.check(M.Mxfp4TileOpening(K, self.opening.manifest[:-1], self.opening.tiles))["verdict"],
                         "REJECTED: E_MANIFEST")

    def test_f13_byte_flips_of_the_honest_container_never_verify_with_both_pins(self):
        """Sample of single-byte container flips (the exhaustive sweep lives in the relation-6 suite)."""
        used = self.lay.body_used + 64
        for pos in (0, 70, 100, 130, 170, 5000, 150000, used - 1):
            data = bytearray(self.container)
            data[pos] ^= 0x01
            rs, py = _verdicts(bytes(data), self.wroot, self.xroot)
            self.assertNotEqual(py, "VERIFIED", pos)
            if rs is not None:
                self.assertNotEqual(rs, "VERIFIED", pos)
        flipped = bytearray(self.container)
        flipped[100] ^= 0x01                                                   # inside the pinned weight_root
        self.assertEqual(self.check(container=bytes(flipped))["verdict"], "REJECTED: E_WEIGHT_ROOT")


if __name__ == "__main__":
    unittest.main()
