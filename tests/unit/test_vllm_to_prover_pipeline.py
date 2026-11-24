# SPDX-License-Identifier: Apache-2.0
"""CPU capture-reader and response framing tests."""

import base64
import os
import struct
import unittest
from unittest.mock import Mock

from fpn_prover.backend.fpn2 import FOLDED_AIR_V1, header, profile_id
from fpn_prover.circuit import e2e_prover as E2E
from fpn_prover.circuit import lower as LW
from fpn_prover.circuit import protocol
from fpn_prover.runtime.api_envelope import attach_verification, release_deadline_s
from fpn_prover.runtime.pipeline import (
    extract_samples_from_trace,
    prove_captured_request,
    verify_reply_with_setup,
)
from fpn_prover.runtime.setup_cert import SetupCertificate
from fpn_serve.capture import CaptureConfig, CaptureSession, CaptureTier, ModelDims


class _DetRng:
    def __init__(self, seed=17):
        import random
        self._r = random.Random(seed)

    def fp3(self):
        from flashprove_next.field import P
        return (self._r.randrange(P), self._r.randrange(P), self._r.randrange(P))


class PipelineTests(unittest.TestCase):
    def test_settled_capture_with_mounted_prefix(self):
        session = CaptureSession.create(ModelDims(2, 8, 2, 1, 4, 16), CaptureConfig())
        self.addCleanup(session.close)
        prefix = list(range(16))
        fresh = [16, 17, 18]
        row = bytearray(session.store.row_bytes)
        for slot in session.layout.slots:
            if slot.kind == "lse":
                struct.pack_into("<f", row, 2 * slot.offset, 1.25)
            else:
                struct.pack_into("<H", row, 2 * slot.offset, 0x3FC0)
        session.register_request("prefix")
        self.assertTrue(session.step_cpu([("prefix", prefix)], bytes(row) * len(prefix)))
        prefix_trace = session.settle()["prefix"]

        session.register_request("reply", prefix)
        self.assertTrue(session.step_cpu([("reply", fresh)], bytes(row) * len(fresh)))
        trace = session.settle()["reply"]
        self.assertEqual(trace.tip, prefix_trace.tip)
        self.assertIsNotNone(trace.tail)

        draws = protocol.sample_request(
            b"s" * 32, protocol.SamplerProfile(2, ((8, 8),), 2, 1, 4, attn_gemms=1),
            input_tokens=18, cache_read_input_tokens=16, output_tokens=2, draws=3,
        )
        samples = extract_samples_from_trace(session.store, trace, "reply", token_ids=prefix + fresh, draws=draws)
        self.assertTrue(samples)
        for draw, ports in samples.items():
            self.assertGreaterEqual(draw.position, len(prefix))
            self.assertEqual(set(ports), {"h_in", "h_mid", "q", "o", "a", "lse", "h_L"})
            self.assertEqual(ports["lse"][0], 1.25)
            self.assertEqual(ports["h_in"][0], 1.5)
        with self.assertRaisesRegex(ValueError, "invalid capture token chain"):
            extract_samples_from_trace(session.store, trace, "reply", token_ids=prefix + [100, 17, 18], draws=draws)

    def test_small_capture_never_submits_copy_work(self):
        session = CaptureSession.create(
            ModelDims(8, 512, 8, 2, 64, 2048), CaptureConfig(ring_mib=24, max_batched_tokens=128),
        )
        self.addCleanup(session.close)
        pool = Mock()
        session.sink._copy_pool = pool
        session.register_request("decode")
        # The pages are individually eligible under the previous 1 MiB check.
        self.assertGreaterEqual(16 * session.store.row_bytes, 1 << 20)
        self.assertLess(32 * session.store.row_bytes, 32 << 20)
        self.assertTrue(session.step_cpu([("decode", list(range(32)))], bytes(32 * session.store.row_bytes)))
        self.assertEqual(session.settle()["decode"].n_full_pages, 2)
        pool.submit.assert_not_called()

    def test_tier_b_small_chunks_stay_serial_while_large_chunks_parallelize(self):
        session = CaptureSession.create(
            ModelDims(40, 512, 8, 2, 64, 2048),
            CaptureConfig(tier=CaptureTier.TIER_B, ring_mib=40, max_batched_tokens=480),
        )
        self.addCleanup(session.close)
        self.assertEqual(session.sink.copy_workers, 4)
        pool = session.sink._copy_pool
        self.assertIsNotNone(pool)
        submit = Mock(wraps=pool.submit)
        pool.submit = submit

        from fpn_serve.capture.store import PAGE_TOKENS, PARALLEL_COPY_MIN_BYTES

        page_bytes = PAGE_TOKENS * session.store.row_bytes
        self.assertGreaterEqual(page_bytes, 1 << 20)
        large_pages = -(-PARALLEL_COPY_MIN_BYTES // page_bytes)
        for request, pages in (("decode", large_pages - 1), ("prefill", large_pages)):
            count = pages * PAGE_TOKENS
            session.register_request(request)
            self.assertTrue(session.step_cpu([(request, range(count))], bytes(count * session.store.row_bytes)))
            self.assertEqual(session.settle()[request].n_full_pages, pages)
            if request == "decode":
                submit.assert_not_called()
            else:
                self.assertEqual(submit.call_count, pages)

    def test_fixed_setup_encoding_and_provider_extension(self):
        original = SetupCertificate(b"w" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 0)
        config_update = SetupCertificate(b"w" * 32, b"d" * 32, b"r" * 32, b"f" * 32, 1)
        weight_update = SetupCertificate(b"v" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 1)
        # Framing fixture with the production header. It carries no proof.
        receipt = header(profile_id(FOLDED_AIR_V1)).ljust(FOLDED_AIR_V1["c_p"], b"\0")
        expected = {"fpn_receipt_b64", "setup_cert_id", "challenge_mode", "c_pre_hex"}
        for update in (config_update, weight_update):
            encoded = update.encode()
            self.assertEqual(len(encoded), len(original.encode()))
            self.assertEqual(SetupCertificate.decode(encoded), update)
            self.assertNotEqual(update.id, original.id)
            self.assertNotIn(b"update_type", encoded)
            for response in ({"type": "message", "content": []}, {"object": "chat.completion", "choices": []}):
                envelope = attach_verification(
                    response, receipt=receipt, setup_cert=encoded, challenge_mode=protocol.MODE_NI,
                    c_pre=b"p" * 32,
                )
                self.assertEqual(envelope.keys() - response.keys(), expected)
                self.assertEqual(base64.b64decode(envelope["fpn_receipt_b64"], validate=True), receipt)
                self.assertEqual(envelope["setup_cert_id"], update.id)
        self.assertEqual(release_deadline_s(100.5, 2.5), 103.0)
        with self.assertRaises(ValueError):
            release_deadline_s(100.5, -1)

    def test_end_to_end_capture_to_relation4_api_and_dual_verify(self):
        """Full loop: CaptureSession (mounted prefix) -> PageStore -> Protocol v2 -> SetupCert -> Relation 4 -> Rust + Python."""
        session = CaptureSession.create(ModelDims(2, 8, 2, 1, 4, 16), CaptureConfig())
        self.addCleanup(session.close)
        prefix = list(range(16))
        fresh = [16, 17, 18]
        response_tokens = [18, 19]
        row = bytearray(session.store.row_bytes)
        for slot in session.layout.slots:
            if slot.kind == "lse":
                struct.pack_into("<f", row, 2 * slot.offset, 1.25)
            else:
                # 0x3FC0 = 1.5 in bf16
                struct.pack_into("<H", row, 2 * slot.offset, 0x3FC0)
        session.register_request("prefix")
        self.assertTrue(session.step_cpu([("prefix", prefix)], bytes(row) * len(prefix)))
        session.settle()

        session.register_request("reply", prefix)
        self.assertTrue(session.step_cpu([("reply", fresh)], bytes(row) * len(fresh)))
        trace = session.settle()["reply"]

        prog, _ = E2E.build_program()
        circuit = LW.compile_air(prog)
        setup_cert = SetupCertificate(b"w" * 32, circuit.root, b"r" * 32, b"f" * 32, 7).encode()
        s_prof = protocol.SamplerProfile(2, ((8, 8),), 2, 1, 4, attn_gemms=1)

        def weights_provider(_draw, ports):
            x_val = ports["h_in"][0]
            w_val = 0.5
            return {
                "norm_gain": (E2E.bf16_word(1.0), E2E.bf16_word(1.0)),
                "gemm_w": E2E.bf16_word(w_val),
                "gemm_observed": E2E.bf16_word(x_val * w_val),
            }

        res = prove_captured_request(
            session.store,
            trace,
            "reply",
            token_ids=prefix + fresh,
            response_tokens=response_tokens,
            cache_read_input_tokens=len(prefix),
            setup_cert_bytes=setup_cert,
            sampler_profile=s_prof,
            weights_provider=weights_provider,
            challenge_mode=protocol.MODE_EXTERNAL_SEED,
            external_nonce=b"\x42" * 32,
            rng=_DetRng(99),
        )
        self.assertEqual(len(res.receipt), FOLDED_AIR_V1["c_p"])
        self.assertGreaterEqual(res.extracted_count, 1)

        # Attach to both Anthropic & OpenAI zero-leakage envelopes
        for base_resp in ({"type": "message", "content": [{"type": "text", "text": "ok"}]},
                          {"object": "chat.completion", "choices": []}):
            env = attach_verification(
                base_resp,
                receipt=res.receipt,
                setup_cert=setup_cert,
                challenge_mode=res.challenge_mode,
                c_pre=res.c_pre,
            )
            self.assertEqual(env["setup_cert_id"], res.setup_cert_id)
            for forbidden in ("architecture", "n_layers", "d_model", "n_experts", "update_type", "cache_read"):
                self.assertNotIn(forbidden, env)

        # Dual verify with BOTH native Rust fpn-verify AND isolated -I detached Python!
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        rust_bin = os.environ.get("FPN_VERIFY_BIN") or os.path.join(root_dir, "core", "target", "release", "fpn-verify")
        self.assertTrue(os.path.exists(rust_bin), "expected release fpn-verify binary")
        ver = verify_reply_with_setup(
            setup_cert,
            res.receipt,
            expected_c_pre=res.c_pre,
            circuit_digest=res.circuit_digest,
            rust_binary=rust_bin,
            check_python=True,
        )
        self.assertTrue(ver["verified"])
        self.assertEqual(ver["rust"], "VERIFIED")
        self.assertEqual(ver["python"], "VERIFIED")

        # Tampered setup cert circuit_root -> BOTH Rust and Python reject with E_CIRCUIT_PINNED!
        bad_setup = SetupCertificate(b"w" * 32, b"\xaa" * 32, b"r" * 32, b"f" * 32, 7).encode()
        bad_ver = verify_reply_with_setup(
            bad_setup,
            res.receipt,
            expected_c_pre=res.c_pre,
            circuit_digest=res.circuit_digest,
            rust_binary=rust_bin,
            check_python=True,
        )
        self.assertFalse(bad_ver["verified"])
        self.assertEqual(bad_ver["rust"], "E_CIRCUIT_PINNED")
        self.assertEqual(bad_ver["python"], "E_CIRCUIT_PINNED")

        # Tampered C_pre -> E_CPRE_MISMATCH!
        cpre_ver = verify_reply_with_setup(
            setup_cert,
            res.receipt,
            expected_c_pre=b"\xff" * 32,
            circuit_digest=res.circuit_digest,
            rust_binary=rust_bin,
            check_python=False,
        )
        self.assertFalse(cpre_ver["verified"])
        self.assertEqual(cpre_ver["error"], "E_CPRE_MISMATCH")

        # Security audit MAJOR #2 regression: mutating 1 activation byte in the partial tail
        # (with IDENTICAL token_ids and IDENTICAL trace.tip!) MUST change trace_payload_root, C_pre, and seed!
        session2 = CaptureSession.create(ModelDims(2, 8, 2, 1, 4, 16), CaptureConfig())
        self.addCleanup(session2.close)
        session2.register_request("prefix")
        self.assertTrue(session2.step_cpu([("prefix", prefix)], bytes(row) * len(prefix)))
        session2.settle()
        row_alt = bytearray(row)
        # Flip 1 bit of a BF16 activation inside the fresh tail rows
        row_alt[0] ^= 0x01
        session2.register_request("reply", prefix)
        self.assertTrue(session2.step_cpu([("reply", fresh)], bytes(row_alt) * len(fresh)))
        trace2 = session2.settle()["reply"]
        self.assertEqual(trace2.tip, trace.tip)
        from fpn_prover.runtime.pipeline import commit_trace_payload, precommit_request
        pay1 = commit_trace_payload(session.store, trace, "reply", prefix + fresh)
        pay2 = commit_trace_payload(session2.store, trace2, "reply", prefix + fresh)
        self.assertNotEqual(pay1, pay2)
        cpre1 = precommit_request(setup_cert, pay1, response_tokens)
        cpre2 = precommit_request(setup_cert, pay2, response_tokens)
        self.assertNotEqual(cpre1, cpre2)
        # And omitting expected_c_pre fails closed with E_CPRE_REQUIRED
        unbound_ver = verify_reply_with_setup(
            setup_cert, res.receipt, expected_c_pre=None, rust_binary=rust_bin, check_python=False
        )
        self.assertFalse(unbound_ver["verified"])
        self.assertEqual(unbound_ver["error"], "E_CPRE_REQUIRED")

        # Tampered captured activation (observed product mismatches captured h_in * w) -> rejected by lowering!
        with self.assertRaises(LW.LoweringError):
            prove_captured_request(
                session.store,
                trace,
                "reply",
                token_ids=prefix + fresh,
                response_tokens=response_tokens,
                cache_read_input_tokens=len(prefix),
                setup_cert_bytes=setup_cert,
                sampler_profile=s_prof,
                weights_provider=lambda _d, _p: {
                    "norm_gain": (E2E.bf16_word(1.0), E2E.bf16_word(1.0)),
                    "gemm_w": E2E.bf16_word(0.5),
                    "gemm_observed": E2E.bf16_word(9.0),  # forged product!
                },
                challenge_mode=protocol.MODE_EXTERNAL_SEED,
                external_nonce=b"\x42" * 32,
                rng=_DetRng(99),
            )
