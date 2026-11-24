# SPDX-License-Identifier: Apache-2.0
"""pre-challenge ZK-blinded activation chunk commitment (`runtime/x_commit.py`) -> C_pre -> relation 6
(`FOLDED_VEC_BOUND`) -> zero-plaintext offline verifier `verify_rel6_zk_bundle` (Rust `fpn-verify --weight-root
--x-root` + detached Python), on the fast n = 16 profile (B = 1) with `HostEngine`.

Covers: x_chunks_root binding + domain separation, exact integer validation and zero padding (B = 64 partition,
non-power-of-two chunk counts), OS-CSPRNG blinding, the honest end-to-end bundle, and every forgery: relation-5
case (c) under the honest C_pre / x_root (E_MERKLE, E_SUMCHECK), a post-challenge x_root' (E_X_CHUNK_PATH) or a
rebuilt x_chunks_root (E_CPRE_MISMATCH), the relation-5 downgrade (E_X_ROOT_PINNED), a tampered chunk sibling
(E_X_CHUNK_PATH), a wrong chunk index (E_X_CHUNK_INDEX), a tampered weight tile (E_TILE_PATH), missing C_pre
(E_CPRE_REQUIRED), plus routing metadata (E_X_CHUNK_META) and N_s (E_NS_MISMATCH).
"""
import ast
import hashlib
import inspect
import os
import pathlib
import struct
import unittest
from unittest.mock import patch

import numpy as np

from fpn_prover.backend import cuda_engine as CE
from fpn_prover.backend import folded_vec as FV
from fpn_prover.backend import folded_vec_bound as FB
from fpn_prover.backend.fpn2 import FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1, FOLDED_VEC_N16_V1
from fpn_prover.runtime import weight_commit as WC
from fpn_prover.runtime import x_commit as XC
from fpn_prover.runtime.pipeline import precommit_request
from fpn_prover.runtime.setup_cert import SetupCertificate

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUST = str(ROOT / "core" / "target" / "release" / "fpn-verify")
PROF = FOLDED_VEC_BOUND_N16_V1
K = 1024
LAYERS = 2
TENSORS = {"q_proj": ("self_attn", 64), "k_proj": ("self_attn", 32), "gate_proj": ("mlp", 96)}
NONCE = hashlib.blake2b(b"fpn2 unit test nonce", digest_size=32).digest()
FPRINT = hashlib.blake2b(b"unit capture layout", digest_size=32).digest()
TOKENS = list(range(100, 140))
RESP = [7, 8, 9, 10]


def _name(layer, proj):
    return f"model.layers.{layer}.{TENSORS[proj][0]}.{proj}.weight"


def tile_rule(desc, draw, routing):
    """PUBLIC rule: x_attn rows (slot 0) -> q_proj | k_proj, x_mlp rows (slot 1) -> gate_proj; r0 from the draw;
    c0 = 1024 * column chunk.  Pad rows -> layer 0 q_proj.  `routing` (MoE) picks among the row's routed ids."""
    pos, layer, slot, chunk = desc
    if desc == XC.PAD_ROW:
        layer, slot, chunk = 0, 0, 0
    proj = ("q_proj" if draw[0] & 1 else "k_proj") if slot == 0 else "gate_proj"
    if routing is not None and slot == 1:                     # routed rows: the drawn slot's id selects a layer
        layer = int(routing[draw[2] % len(routing)]) % LAYERS
    rows = TENSORS[proj][1]
    r0 = 32 * (int.from_bytes(draw[4:12], "little") % (rows // 32))
    return (_name(layer, proj), r0, K * chunk)


def _bf16_words(g, shape, lo=-12, hi=2):
    """Random normal-range BF16 words (exponents in [lo, hi])."""
    sign = g.integers(0, 2, size=shape, dtype=np.uint16) << 15
    exp = (g.integers(lo, hi + 1, size=shape) + 127).astype(np.uint16) << 7
    man = g.integers(0, 128, size=shape, dtype=np.uint16)
    return (sign | exp | man).astype(np.uint16)


def _rust(data, w, x):
    return XC._code(XC.rust_verify_rel6(RUST, data, w, x)[0])


class XCommitRel6E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(RUST):
            raise unittest.SkipTest("core/target/release/fpn-verify not built (cargo build --release --bin fpn-verify)")
        g = np.random.default_rng(14)
        cls.engine = CE.HostEngine()
        tensors = {_name(l, p): _bf16_words(g, (r, K)) for l in range(LAYERS) for p, (_, r) in TENSORS.items()}
        cls.ck = WC.ArrayCheckpoint(tensors)
        cls.wcom = WC.commit_model_weights(cls.ck, K=K, threads=2)
        cls.setup = SetupCertificate(cls.wcom.root, b"c" * 32, b"r" * 32, b"f" * 32, 0).encode()
        # public schedule: positions 30..32, both layers, slots x_attn / x_mlp, one column chunk -> 12 rows = 12 chunks
        cls.rows = [(p, l, s, 0) for p in range(30, 33) for l in range(LAYERS) for s in range(2)]
        xw = [WC.window_ints(WC.bf16_values(_bf16_words(g, (K,))), XC.DEFAULT_INT_BITS)[0] for _ in cls.rows]
        cls.X = np.array(xw, dtype=np.int64)
        cls.xcc = XC.commit_x_chunks(PROF, cls.X, cls.rows, FPRINT, engine=cls.engine)
        cls.c_pre = XC.precommit_x_chunks(cls.setup, cls.xcc.root, RESP)
        cls.bundle = XC.prove_rel6_zk_bundle(cls.xcc, setup_cert=cls.setup, weight_commitment=cls.wcom,
                                             checkpoint=cls.ck, token_ids=TOKENS, response_tokens=RESP,
                                             external_nonce=NONCE, tile_rule=tile_rule)

    def verify(self, **over):
        b = self.bundle
        kw = dict(c_pre=self.c_pre, token_ids=TOKENS, response_tokens=RESP, external_nonce=NONCE, profile=PROF,
                  row_desc=self.rows, layout_fingerprint=FPRINT, x_chunks_root=b.x_chunks_root,
                  chunk_opening=b.chunk_opening, routing_opening=b.routing_opening, tile_opening=b.tile_opening,
                  container=b.container, tile_rule=tile_rule, rust_binary=RUST, check_python=True,
                  engine=self.engine)
        kw.update(over)
        return XC.verify_rel6_zk_bundle(self.setup, **kw)

    def code(self, **over):
        return self.verify(**over)["verdict"].removeprefix("REJECTED: ")

    # ------------------------------------------------------------------ honest end to end
    def test_honest_bundle_rust_and_python_zero_plaintext(self):
        rep = self.verify(rust_runs=3)
        self.assertEqual(rep["verdict"], "VERIFIED", rep)
        self.assertEqual((rep["rust"], rep["python"]), ("VERIFIED", "VERIFIED"))
        self.assertEqual(rep["plaintext_activation_bytes_received"], 0)
        self.assertLess(rep["x_chunk_opening_bytes"], 1024)
        self.assertEqual(len(self.bundle.container), PROF["c_p"])
        self.assertTrue(all(self.bundle.report["integer_faithful"]))
        self.assertEqual(self.bundle.report["chunk_idx"], rep["chunk_idx"])
        # the verifier's signature takes no activation rows; the opening carries none of their bytes
        params = set(inspect.signature(XC.verify_rel6_zk_bundle).parameters)
        self.assertFalse(params & {"X", "X_rows", "x_rows", "activations", "rows"})
        c = rep["chunk_idx"]
        # the x-chunk opening is exactly: magic + 4 integers + x_root + meta + depth + path (no room for an 8 KiB row)
        op = XC.XChunkOpening.decode(self.bundle.chunk_opening)
        self.assertEqual(len(self.bundle.chunk_opening), len(XC._OPEN_MAGIC) + 24 + 64 + 1 + 32 * len(op.path))
        self.assertEqual((op.x_root, op.meta), (self.xcc.x_roots[c], XC.NO_META))
        # the tile opening re-encodes byte-identically from (manifest, drawn tiles) alone (decode rejects extra bytes)
        self.assertEqual(WC.TileOpening.decode(self.bundle.tile_opening).encode(), self.bundle.tile_opening)
        self.assertEqual(self.bundle.routing_opening, b"")
        # the pinned x_root is exactly the opened chunk's pre-committed root
        self.assertEqual(self.bundle.container[128:160], self.xcc.x_roots[c])

    # ------------------------------------------------------------------ binding + domain separation
    def test_x_chunks_root_binding_and_domain_separation(self):
        x = self.xcc
        base = dict(n_chunks=x.n_chunks, B=x.B, K=x.K, n_rows=x.n_rows, int_bits=x.int_bits, profile=PROF,
                    layout_fingerprint=FPRINT, sched_digest=XC.schedule_digest(self.rows))
        tree = x.levels[-1][0]
        self.assertEqual(XC.chunks_root(tree, **base), x.root)
        for k, v in (("n_chunks", x.n_chunks + 1), ("n_rows", x.n_rows - 1), ("int_bits", 22),
                     ("profile", FOLDED_VEC_BOUND_N22_V1), ("layout_fingerprint", bytes(32)),
                     ("sched_digest", XC.schedule_digest(self.rows[::-1]))):
            self.assertNotEqual(XC.chunks_root(tree, **{**base, k: v}), x.root, k)
        # leaf / node / pad / root personalisations are distinct
        a, b = b"\x01" * 32, b"\x02" * 32
        hs = {XC._leaf(0, a, b), XC._h(XC._P_NODE, a, b), XC._pad_leaf(), XC._h(XC._P_ROOT, a, b),
              XC._h(XC._P_META, a, b), XC._h(XC._P_SCHED, a, b), XC._h(XC._P_TPR, a)}
        self.assertEqual(len(hs), 7)
        # C_pre domain separation: x_chunks_root vs relation 6's single x_root vs a trace payload root
        r = x.root
        cp = XC.precommit_x_chunks(self.setup, r, RESP)
        self.assertNotEqual(cp, FB.precommit_x_root(self.setup, r, RESP))
        self.assertNotEqual(cp, precommit_request(self.setup, r, RESP))
        self.assertNotEqual(cp, XC.precommit_x_chunks(self.setup, r, RESP + [1]))
        self.assertNotEqual(cp, XC.precommit_x_chunks(self.setup, r, RESP, b"policy"))
        with self.assertRaises(XC.XCommitError):
            XC.precommit_x_chunks(self.setup, r[:31], RESP)
        # a different (blinded) commitment of the SAME rows gives a different root: hiding, OS CSPRNG
        with patch.dict(os.environ, {"FPN_TEST_RNG_SEED": "123"}):
            x2 = XC.commit_x_chunks(PROF, self.X[:2], self.rows[:2], FPRINT, engine=self.engine)
            x3 = XC.commit_x_chunks(PROF, self.X[:2], self.rows[:2], FPRINT, engine=self.engine)
        self.assertNotEqual(x2.x_roots[0], x3.x_roots[0])
        self.assertNotEqual(x2.root, x3.root)

    def test_partition_padding_and_exact_integer_validation(self):
        g = np.random.default_rng(3)
        rows = g.integers(-(1 << 23), (1 << 23) + 1, size=(130, K))
        n, chunks = XC.partition_rows(rows, 64, K)
        self.assertEqual((n, len(chunks)), (3, 3))                    # 130 rows -> 3 chunks of 64, 62 zero rows
        self.assertTrue(np.array_equal(XC.field_to_int(chunks[2][:2]), rows[128:]))
        self.assertFalse(chunks[2][2:].any())
        self.assertTrue(np.array_equal(XC.field_to_int(np.concatenate(chunks)[:130]), rows))
        self.assertEqual(XC.chunk_rows([(i, 0, 0, 0) for i in range(130)], 64, 2)[2:], [XC.PAD_ROW] * 62)
        for bad in (rows.astype(np.float64), rows > 0, rows.astype(object), rows[:, :K - 1], rows[:0],
                    np.where(rows == rows[0, 0], (1 << 23) + 1, rows), rows.tolist()):
            with self.assertRaises(XC.XCommitError):
                XC.partition_rows(bad, 64, K)
        with self.assertRaises(XC.XCommitError):
            XC.partition_rows(np.full((1, K), -(1 << 23) - 1), 64, K)
        for bad_sched in ([(0, 0, 0, 0)] * 2, [XC.PAD_ROW], [(0, 0, 0)], [(1 << 32, 0, 0, 0)], []):
            with self.assertRaises(XC.XCommitError):
                XC._check_schedule(bad_sched)
        with self.assertRaises(XC.XCommitError):                       # schedule length must equal the row count
            XC.commit_x_chunks(PROF, self.X[:3], self.rows[:2], FPRINT, engine=self.engine)
        # non-power-of-two chunk count (12) -> PAD leaves; every chunk's path verifies, depth = 4
        self.assertEqual((self.xcc.n_chunks, len(self.xcc.levels) - 1), (12, 4))
        for c in range(self.xcc.n_chunks):
            op = XC.XChunkOpening.decode(self.xcc.opening(c).encode())
            self.assertEqual(XC._root_from_path(XC._leaf(c, op.x_root, op.meta), c, op.path), self.xcc.levels[-1][0])

    def test_module_hygiene(self):
        src = pathlib.Path(XC.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                self.assertNotIn(node.attr, ("environ", "getenv", "putenv"))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                self.assertFalse(any(m.startswith("fpn_serve") for m in mods), mods)

    # ------------------------------------------------------------------ forgeries
    def _substituted(self):
        """(c*, W, X', Y' = W X') for the drawn tiles of the honest chunk."""
        c = self.bundle.report["chunk_idx"]
        to = WC.TileOpening.decode(self.bundle.tile_opening)
        W_int = np.stack([w for w, _ in to.weights_int(XC.DEFAULT_INT_BITS)]).astype(np.int64)
        Xs = XC.field_to_int(self.xcc.chunk_x(c)).copy()
        Xs[0, 11] += 1
        Ys, _ = FV.exact_products(W_int, Xs)
        return c, FV.to_field(W_int), FV.to_field(Xs), FV.to_field(Ys)

    def test_case_c_substitution_under_honest_cpre_and_x_root(self):
        c, W, Xs, Ys = self._substituted()
        n_s = self.bundle.container[160:192]
        xcs = FB.commit_x(self.xcc.layout, Xs, engine=self.engine)
        # (a1) re-prove (X', Y') under its own X_com but claim the honest x_root in the statement
        forged = FB.prove_bound(PROF, W, Xs, Ys, xcs, n_s=n_s, engine=self.engine,
                                hooks={"stmt_x_root": self.xcc.x_roots[c]})
        self.assertEqual(self.code(container=forged), "E_MERKLE")
        # (a2) run the sumcheck on X' but only the honest X_com can be opened
        forged2 = FB.prove_bound(PROF, W, Xs, Ys, self.xcc.x_commitment(c, _allow_reuse=True), n_s=n_s, engine=self.engine,
                                 hooks={"allow_x_mismatch": True})
        self.assertEqual(self.code(container=forged2), "E_SUMCHECK")
        # (b1) commit x_root' AFTER the challenge and open it in place of x_root_{c*}: not under x_chunks_root
        honest = XC.XChunkOpening.decode(self.bundle.chunk_opening)
        sub = XC.XChunkOpening(c, honest.n_chunks, honest.B, honest.K, xcs.root, honest.meta, honest.path)
        reprove = FB.prove_bound(PROF, W, Xs, Ys, xcs, n_s=n_s, engine=self.engine)
        self.assertEqual(self.code(chunk_opening=sub.encode(), container=reprove), "E_X_CHUNK_PATH")
        # (b2) rebuild x_chunks_root around x_root': the published C_pre no longer matches
        Xall = self.X.copy()
        Xall[c] = XC.field_to_int(Xs)[0]
        x2 = XC.commit_x_chunks(PROF, Xall, self.rows, FPRINT, engine=self.engine)
        self.assertEqual(self.code(x_chunks_root=x2.root, chunk_opening=x2.opening(c).encode(), container=reprove),
                         "E_CPRE_MISMATCH")

    def test_relation5_downgrade_rejected_by_rust_and_python(self):
        c, W, _, _ = self._substituted()
        X = self.xcc.chunk_x(c)
        to = WC.TileOpening.decode(self.bundle.tile_opening)
        Y = FV.to_field(FV.exact_products(np.stack([w for w, _ in to.weights_int(XC.DEFAULT_INT_BITS)]),
                                          XC.field_to_int(X))[0])
        FV._W_CACHE.clear()
        r5 = FV.prove(W, X, Y, n_s=b"\x07" * 32, profile=FOLDED_VEC_N16_V1, engine=self.engine)
        FV._W_CACHE.clear()
        self.assertEqual(r5[96:128], self.bundle.container[96:128])  # same honest weight root
        rep = self.verify(container=r5)
        self.assertEqual(rep["verdict"], "REJECTED: E_X_ROOT_PINNED")
        w, x = self.bundle.container[96:128], self.xcc.x_roots[c]
        self.assertEqual(_rust(r5, w, x), "E_X_ROOT_PINNED")
        self.assertEqual(XC._code(XC.python_verify_rel6(r5, w, x)[0]), "E_X_ROOT_PINNED")

    def test_tampered_openings_and_missing_cpre(self):
        b = self.bundle
        op = XC.XChunkOpening.decode(b.chunk_opening)
        sib = bytearray(op.path[1])
        sib[5] ^= 1
        path = op.path[:1] + (bytes(sib),) + op.path[2:]
        self.assertEqual(self.code(chunk_opening=XC.XChunkOpening(op.chunk_idx, op.n_chunks, op.B, op.K, op.x_root,
                                                                  op.meta, path).encode()), "E_X_CHUNK_PATH")
        other = (op.chunk_idx + 1) % op.n_chunks                     # a genuine opening of ANOTHER chunk
        self.assertEqual(self.code(chunk_opening=self.xcc.opening(other).encode()), "E_X_CHUNK_INDEX")
        wb = bytearray(b.tile_opening)
        wb[-9] ^= 1                                                   # a unit of the last opened tile
        self.assertEqual(self.code(tile_opening=bytes(wb)), "E_TILE_PATH")
        to = WC.TileOpening.decode(b.tile_opening)
        t0 = to.tiles[0]
        alt_name = next(n for n in self.ck.names() if n != t0.name)
        wrong = WC.open_drawn_tiles(self.wcom, self.ck, [(alt_name, 0, 0)])
        self.assertNotEqual(wrong.coords(), to.coords())
        self.assertEqual(self.code(tile_opening=wrong.encode()), "E_TILE_COORDS")
        # Container weight_root differs from opened tiles' weight_root -> E_WEIGHT_ROOT
        bad_w_container = bytearray(b.container)
        bad_w_container[96] ^= 1
        self.assertEqual(self.code(container=bytes(bad_w_container)), "E_WEIGHT_ROOT")
        self.assertEqual(self.code(c_pre=None), "E_CPRE_REQUIRED")
        self.assertEqual(self.code(c_pre=b.c_pre[:16]), "E_CPRE_REQUIRED")
        bad = bytearray(b.c_pre)
        bad[0] ^= 1
        self.assertEqual(self.code(c_pre=bytes(bad)), "E_CPRE_MISMATCH")
        # a different public schedule (row order) is a different x_chunks_root header
        self.assertEqual(self.code(row_desc=self.rows[::-1]), "E_X_CHUNK_PATH")
        # another external nonce re-draws the challenge: the honest opening no longer answers it
        self.assertIn(self.code(external_nonce=bytes(32)), ("E_X_CHUNK_INDEX", "E_TILE_COORDS", "E_NS_MISMATCH"))

    def test_n_s_binds_the_challenge(self):
        c, W, _, _ = self._substituted()
        xcom = self.xcc.x_commitment(c, _allow_reuse=True)
        to = WC.TileOpening.decode(self.bundle.tile_opening)
        Y = FV.to_field(FV.exact_products(np.stack([w for w, _ in to.weights_int(XC.DEFAULT_INT_BITS)]),
                                          XC.field_to_int(xcom.x_values))[0])
        FV._W_CACHE.clear()
        stale = FB.prove_bound(PROF, W, xcom.x_values, Y, xcom, n_s=b"\x00" * 32, engine=self.engine)
        self.assertEqual(self.code(container=stale), "E_NS_MISMATCH")

    def test_single_issuance_retry_conflict_and_concurrency(self):
        """Security audit BLOCKER regression: single-proof blinding budget per chunk (cached retry, fail-closed conflict)."""
        from concurrent.futures import ThreadPoolExecutor

        xcc = XC.commit_x_chunks(PROF, self.X, self.rows, FPRINT, engine=self.engine)
        kw = dict(setup_cert=self.setup, weight_commitment=self.wcom, checkpoint=self.ck,
                  token_ids=TOKENS, response_tokens=RESP, external_nonce=NONCE, tile_rule=tile_rule)
        # Concurrent identical calls on the same chunk must issue at most ONE proof and return the same cached bundle.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(XC.prove_rel6_zk_bundle, xcc, **kw) for _ in range(4)]
            bundles = [f.result() for f in futs]
        b0 = bundles[0]
        for bi in bundles[1:]:
            self.assertIs(bi, b0)
        c = b0.report["chunk_idx"]
        # Direct second x_commitment(c) without explicit _allow_reuse=True must fail closed.
        with self.assertRaises(XC.XCommitError) as cm:
            xcc.x_commitment(c)
        self.assertEqual(cm.exception.code, "E_CHUNK_BLINDING_EXHAUSTED")
        # Negative-index / bool / out-of-range aliases must NEVER bypass lock or consumption checks.
        for bad_c in (-1, c - xcc.n_chunks, True, False, xcc.n_chunks):
            with self.assertRaises(XC.XCommitError) as cm_idx:
                xcc.x_commitment(bad_c)
            self.assertEqual(cm_idx.exception.code, "E_X_CHUNK_INDEX", bad_c)
            with self.assertRaises(XC.XCommitError) as cm_op:
                xcc.opening(bad_c)
            self.assertEqual(cm_op.exception.code, "E_X_CHUNK_INDEX", bad_c)
            with self.assertRaises(XC.XCommitError) as cm_x:
                xcc.chunk_x(bad_c)
            self.assertEqual(cm_x.exception.code, "E_X_CHUNK_INDEX", bad_c)
        # Find a distinct external nonce that maps to the SAME chunk c -> MUST fail closed with E_CHUNK_BLINDING_EXHAUSTED.
        c_pre = b0.c_pre
        alt_nonce = None
        for i in range(1, 200):
            cand = hashlib.blake2b(b"alt-%d" % i, digest_size=32).digest()
            _, ci, _ = XC.draw_challenge(c_pre, TOKENS, RESP, cand, xcc.n_chunks, xcc.B)
            if ci == c and cand != NONCE:
                alt_nonce = cand
                break
        self.assertIsNotNone(alt_nonce)
        with self.assertRaises(XC.XCommitError) as cm2:
            XC.prove_rel6_zk_bundle(xcc, **{**kw, "external_nonce": alt_nonce})
        self.assertEqual(cm2.exception.code, "E_CHUNK_BLINDING_EXHAUSTED")

    def test_malformed_public_inputs_rejected_cleanly(self):
        """Security audit MINOR regression: exact u32 discipline on schedule & token IDs; no raw exception escape."""
        for bad_row in ((1.5, 0, 0, 0), ("1", 0, 0, 0), (True, 0, 0, 0), (-1, 0, 0, 0), (1 << 32, 0, 0, 0)):
            self.assertEqual(self.code(row_desc=[bad_row] + list(self.rows[1:])), "E_SCHEDULE")
        for bad_tok in (1.5, "10", True, -1, 1 << 32):
            self.assertEqual(self.code(token_ids=[bad_tok, 2]), "E_INVALID_INPUT")
            self.assertEqual(self.code(response_tokens=[bad_tok]), "E_INVALID_INPUT")
        self.assertEqual(self.code(external_nonce=b"short"), "E_INVALID_INPUT")
        self.assertEqual(self.code(int_bits=True), "E_INVALID_INPUT")
        self.assertEqual(self.code(bf16_window=23.0), "E_INVALID_INPUT")

    def test_routing_metadata_binding(self):
        R = np.random.default_rng(9).integers(0, 128, size=(len(self.rows), 8), dtype=np.uint8)
        xr = XC.commit_x_chunks(PROF, self.X, self.rows, FPRINT, engine=self.engine, routing_rows=R)
        self.assertNotEqual(xr.root, self.xcc.root)
        cp = XC.precommit_x_chunks(self.setup, xr.root, RESP)
        bun = XC.prove_rel6_zk_bundle(xr, setup_cert=self.setup, weight_commitment=self.wcom, checkpoint=self.ck,
                                      token_ids=TOKENS, response_tokens=RESP, external_nonce=NONCE,
                                      tile_rule=tile_rule)
        kw = dict(c_pre=cp, x_chunks_root=xr.root, chunk_opening=bun.chunk_opening,
                  routing_opening=bun.routing_opening, tile_opening=bun.tile_opening, container=bun.container)
        self.assertEqual(self.verify(**kw)["verdict"], "VERIFIED")
        ro = XC.RoutingOpening.decode(bun.routing_opening)
        flipped = XC.RoutingOpening(ro.salt, ro.width, bytes([ro.data[0] ^ 1]) + ro.data[1:]).encode()
        self.assertEqual(self.code(**{**kw, "routing_opening": flipped}), "E_X_CHUNK_META")
        self.assertEqual(self.code(**{**kw, "routing_opening": b""}), "E_X_CHUNK_META")


if __name__ == "__main__":
    unittest.main()
