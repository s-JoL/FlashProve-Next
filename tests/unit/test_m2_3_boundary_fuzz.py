# SPDX-License-Identifier: Apache-2.0
"""Deterministic v2 boundary cases, with three native-verifier spot checks."""

import base64
import json
import math
import os
import struct
import unittest
from fractions import Fraction
from pathlib import Path

from fpn_prover.backend.fpn2 import FOLDED_AIR_V1, header, profile_id
from fpn_prover.circuit import e2e_prover as e2e, lower, protocol
from fpn_prover.circuit.blocks import rmsnorm_block
from fpn_prover.circuit.microprogram import _exp_cell, _softmax
from fpn_prover.ir.arith import bf16_decode, norm_rsqrt
from fpn_prover.ir.gadget_ir import Program, attaining_rmx, topk_tau
from fpn_prover.runtime.api_envelope import attach_verification, release_deadline_s
from fpn_prover.runtime.pipeline import (
    extract_samples_from_trace, prove_captured_request, verify_reply_with_setup,
)
from fpn_prover.runtime.setup_cert import SetupCertificate
from fpn_serve.capture import CaptureConfig, CaptureSession, ModelDims


PROFILE = protocol.SamplerProfile(2, ((8, 8),), 2, 1, 4, attn_gemms=1)
EXTENSION = {"fpn_receipt_b64", "setup_cert_id", "challenge_mode", "c_pre_hex"}
FORBIDDEN = {"architecture", "n_layers", "d_model", "n_experts", "update_type", "cache_read",
             "cache_read_input_tokens", "system_prompt", "hidden_cot", "precision"}


def keys_in(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from keys_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys_in(child)


class BoundaryFuzz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.norm = Program()
        cls.norm_wires = rmsnorm_block(cls.norm, "norm", 2)
        cls.circuit = lower.compile_air(e2e.build_program()[0])
        cls.setup = SetupCertificate(b"w" * 32, cls.circuit.root, b"r" * 32, b"f" * 32, 0)

    def session(self):
        session = CaptureSession.create(
            ModelDims(2, 8, 2, 1, 4, 16), CaptureConfig(ring_mib=1, max_batched_tokens=32),
        )
        self.addCleanup(session.close)
        return session

    def capture(self, session, name, tokens, *, prefix=(), words=(0x3FC0, 0x3FC0)):
        row = bytearray(session.store.row_bytes)
        for slot in session.layout.slots:
            offset = 2 * slot.offset
            if slot.kind == "lse":
                struct.pack_into("<f", row, offset, 1.25)
            else:
                for i in range(slot.width):
                    word = words[i % 2] if slot.kind == "h_in" else 0x3FC0
                    struct.pack_into("<H", row, offset + 2 * i, word)
        session.register_request(name, prefix)
        fresh = tokens[len(prefix):]
        if fresh:
            self.assertTrue(session.step_cpu([(name, fresh)], bytes(row) * len(fresh)))
        trace = session.settle()[name]
        self.assertEqual(trace.n_tokens, len(tokens))
        return trace

    def assert_norm(self, words, accepted):
        values = tuple(bf16_decode(word).value for word in words)
        squares = sum(x * x for x in values)
        self.assertEqual(protocol.residual_floor_holds(squares, 2), accepted)
        radicand = squares / 2 + Fraction(1, 10**6)
        cell = norm_rsqrt(radicand)
        self.assertTrue(cell.accepts(radicand))
        outputs = tuple(e2e.bf16_word(float(x) / math.sqrt(radicand)) for x in values)
        witness = dict(zip(self.norm_wires, (*words, 0x3F80, 0x3F80, *outputs,
                                           cell.k, cell.low, cell.high, cell.high - cell.low), strict=True))
        self.assertEqual(self.norm.failures(witness), [] if accepted else ["norm.norm"])

    def assert_rust_proof(self, session, trace, tokens, cache_read):
        def weights(_draw, ports):
            return {
                "norm_gain": (0x3F80, 0x3F80),
                "gemm_w": 0x3F00,
                "gemm_observed": e2e.bf16_word(ports["h_in"][0] / 2),
                "slot": {"normalize": 1, "query": 1, "key0": 1, "key1": 1,
                         "value0": 1, "value1": 1},
            }

        encoded = self.setup.encode()
        result = prove_captured_request(
            session.store, trace, trace.request_id, token_ids=tokens, response_tokens=[97],
            cache_read_input_tokens=cache_read, setup_cert_bytes=encoded,
            sampler_profile=PROFILE, weights_provider=weights, challenge_mode=protocol.MODE_NI,
        )
        binary = os.environ.get("FPN_VERIFY_BIN") or (
            Path(__file__).resolve().parents[2] / "core/target/release/fpn-verify"
        )
        verdict = verify_reply_with_setup(
            encoded, result.receipt, expected_c_pre=result.c_pre,
            circuit_digest=result.circuit_digest, rust_binary=str(binary), check_python=False,
        )
        self.assertEqual(verdict["rust"], "VERIFIED")
        self.assertTrue(verdict["verified"])

    def test_outlier_activations_25(self):
        session = self.session()
        draws = protocol.sample_request(b"a" * 32, PROFILE, input_tokens=1,
                                        cache_read_input_tokens=0, output_tokens=1, draws=2)
        tails = (0x8000, 0x0001, 0x807F, 0xBF80, 0x0080)
        for i in range(25):
            with self.subTest(case=i):
                high = ((143 + 83 * i // 24) << 7) | (1 + (17 * i) % 127)
                if i % 2:
                    high |= 0x8000
                tail = tails[i % len(tails)] if i % 5 != 4 else high ^ 0x8000
                words = (high, tail)
                self.assertGreater(abs(bf16_decode(high).value), 65536)
                self.assert_norm(words, True)
                tokens = [i + 1]
                name = f"outlier-{i}"
                trace = self.capture(session, name, tokens, words=words)
                samples = extract_samples_from_trace(session.store, trace, name, token_ids=tokens, draws=draws)
                self.assertTrue(samples)
                for ports in samples.values():
                    actual = struct.pack("<2f", *ports["h_in"][:2])
                    self.assertEqual(actual, struct.pack("<2I", *(word << 16 for word in words)))
                if i == 0:
                    self.assert_rust_proof(session, trace, tokens, 0)

    def test_tiny_residual_norms_20(self):
        # Round the nearest-above-floor target upward if BF16 rounds it below the floor.
        targets = (0.001001, 0.00105, 0.0015, 0.01, 0.00099)
        for i in range(20):
            with self.subTest(case=i):
                target = targets[i // 4] if i != 19 else 0
                single = i % 2 == 1
                word = e2e.bf16_word(target * (math.sqrt(2) if single else 1))
                accepted = i < 16
                if accepted and bf16_decode(word).value**2 < Fraction(1 + single, 10**6):
                    word += 1
                words = (word, (1 if target else 0) if single else word ^ (0x8000 if i % 4 == 2 else 0))
                self.assert_norm(words, accepted)
                if i == 4:
                    session = self.session()
                    trace = self.capture(session, "tiny", [1], words=words)
                    self.assert_rust_proof(session, trace, [1], 0)

    def test_ties_and_normalizer_boundaries_25(self):
        for i, (tau, margin) in enumerate(((0, 0), (0, 1), (1, 1), (2, 2), (3, 3),
                                           (3, 6), (3, 7), (64, 128), (64, 129))):
            with self.subTest(router=i):
                scores = (11, 11 + margin, 11 + margin)
                self.assertEqual(topk_tau(scores, (1, 0, 0), (0,), tau), margin <= 2 * tau)
                self.assertTrue(topk_tau(scores, (0, 1, 1), (1, 2), tau))
                self.assertFalse(topk_tau(scores, (0, 1, 1), (2, 1), tau))
                for pick in ((0, 1, 0), (0, 0, 1)):
                    self.assertTrue(attaining_rmx(scores, (0, 1, 1), pick, scores[1], (0, 256, 256), 512))
                self.assertFalse(attaining_rmx(scores, (0, 1, 1), (0, 1, 1), scores[1], (0, 256, 256), 512))
                # Reference nucleus cutoff on the real exponential cells of the two tied maxima.
                weights = tuple(_exp_cell(score - scores[1], 256) for score in scores[1:])
                first_mass = Fraction(weights[0], sum(weights))
                step = Fraction(1, 2 ** (8 + i))
                for cutoff, size in ((Fraction(1, 2) - step, 1), (Fraction(1, 2), 1),
                                     (Fraction(1, 2) + step, 2)):
                    self.assertEqual(1 + (first_mass < cutoff), size)
        for scale in (16, 64):
            with self.subTest(saturation_scale=scale):
                self.assertEqual(_exp_cell(0, scale), scale)
                self.assertEqual(_exp_cell(-512, scale), 0)
                self.assertEqual(_softmax((0, -512), (13, -100), scale), 13)
                self.assertEqual(_softmax((0, 0), (13, -9), scale), 2)
        for tokens in (1, 16, 4096, 32768, 65536, 131072):
            for precision in ("bf16", "fp16"):
                with self.subTest(tokens=tokens, precision=precision):
                    radius = protocol.certified_lse_radius(tokens, precision=precision)
                    self.assertEqual(radius.norm_rel, Fraction(tokens, 2**32))
                    self.assertGreater(radius.circuit, radius.floating)
                    self.assertLess(radius.e_den, 1)
                    if precision == "fp16":
                        base = protocol.certified_lse_radius(tokens)
                        self.assertGreater(radius.e_den - base.e_den, Fraction(tokens, 2**26))
        for kwargs in ({"tokens": 1 << 26, "precision": "fp16"}, {"tokens": 4096, "score_radius_max": Fraction(1, 2)}):
            with self.subTest(nonpositive_denominator=kwargs):
                with self.assertRaisesRegex(protocol.LseDomainError, "denominator .* <= 0"):
                    protocol.certified_lse_radius(**kwargs)

    def test_api_and_agentic_boundaries_50(self):
        session = self.session()
        for n in range(1, 4):
            tokens = list(range(n * 16))
            self.capture(session, f"turn-{n}", tokens, prefix=tokens[:-16])

        # The envelope validates receipt framing. Native verification is exercised separately.
        receipt = header(profile_id(FOLDED_AIR_V1)).ljust(FOLDED_AIR_V1["c_p"], b"\0")
        original = self.setup.encode()
        for i in range(50):
            with self.subTest(case=i):
                prefix = [] if i % 5 == 0 else list(range(16 * (1 + i // 5 % 3)))
                tokens = prefix if i % 5 in (1, 2) else [*prefix, 700 + i]
                name = f"reply-{i}"
                trace = self.capture(session, name, tokens, prefix=prefix)
                if prefix:
                    earlier = session.store.handle(f"turn-{len(prefix) // 16}").finish()
                    self.assertEqual(trace.tip, earlier.tip)
                max_tokens = i % 2
                output = max_tokens
                draws = protocol.sample_request(
                    i.to_bytes(32, "little"), PROFILE, input_tokens=len(tokens),
                    cache_read_input_tokens=len(prefix), output_tokens=output, draws=3,
                )
                for draw in draws[protocol.STRATUM_FRESH_INPUT]:
                    self.assertEqual(draw.empty, len(prefix) >= len(tokens) - output)
                    if not draw.empty:
                        self.assertEqual(draw.position, len(prefix))
                for draw in draws[protocol.STRATUM_GENERATION]:
                    self.assertEqual(draw.empty, output == 0)
                    if not draw.empty:
                        self.assertEqual(draw.position, len(tokens) - 1)
                self.assertEqual(draws[protocol.POOL_TOKEN_SAMPLING], (output - 1,) * 3)
                if i == 11:
                    self.assert_rust_proof(session, trace, tokens, len(prefix))

                args = {"value": i}
                strict_result = json.dumps({"ok": True, "value": i})
                # Closed object shapes used by both providers' strict-schema fixtures.
                schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "value": {"type": "integer"}},
                          "required": ["ok", "value"], "additionalProperties": False}
                tool_id = f"call-{i}"
                call = {"id": tool_id, "type": "function",
                        "function": {"name": "lookup", "arguments": json.dumps(args)}}
                request = {
                    "max_tokens": max_tokens,
                    "response_format": {"strict": True, "schema": schema},
                    "messages": [
                        {"role": "user", "content": "Look up the value."},
                        {"role": "assistant", "tool_calls": [call]},
                        {"role": "tool", "tool_call_id": tool_id, "content": strict_result},
                    ],
                }
                assistant, tool = request["messages"][1:]
                self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])
                self.assertEqual(json.loads(call["function"]["arguments"]), args)
                parsed = json.loads(strict_result)
                self.assertEqual(set(parsed), set(request["response_format"]["schema"]["required"]))
                self.assertIs(type(parsed["ok"]), bool)
                self.assertIs(type(parsed["value"]), int)
                responses = (
                    {"type": "message", "content": [
                        {"type": "tool_use", "id": tool_id, "name": "lookup", "input": args},
                        {"type": "text", "text": strict_result},
                    ]},
                    {"object": "chat.completion", "choices": [{"message": {
                        "content": strict_result,
                        "tool_calls": [call],
                    }}]},
                )
                for weight, cfg in ((b"v" * 32, self.setup.model_format_digest), (self.setup.weight_commitment_root, b"c" * 32)):
                    update = SetupCertificate(weight, self.circuit.root, self.setup.rounding_contract_digest, cfg, i + 1)
                    encoded = update.encode()
                    self.assertEqual(len(encoded), len(original))
                    self.assertEqual(SetupCertificate.decode(encoded), update)
                    self.assertNotEqual(update.id, self.setup.id)
                    for response in responses:
                        envelope = attach_verification(response, receipt=receipt, setup_cert=encoded,
                                                       challenge_mode=protocol.MODE_NI, c_pre=b"p" * 32)
                        self.assertEqual(envelope.keys() - response.keys(), EXTENSION)
                        self.assertEqual({k: envelope[k] for k in response}, response)
                        self.assertFalse(FORBIDDEN.intersection(keys_in(envelope)))
                        self.assertEqual(envelope["setup_cert_id"], update.id)
                        self.assertEqual(base64.b64decode(envelope["fpn_receipt_b64"], validate=True), receipt)
                deadline = release_deadline_s(i / 4, 2.5)
                self.assertEqual(deadline - i / 4, 2.5)
                self.assertLess(release_deadline_s((i - 1) / 4, 2.5), deadline)
