# SPDX-License-Identifier: Apache-2.0
"""CPU verification of logical capture across simulated TP, EP and PP ranks."""

import struct
import threading
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

from fpn_prover.backend.fpn2 import FOLDED_AIR_V1, header, profile_id
from fpn_prover.circuit.protocol import MODE_NI
from fpn_prover.runtime.api_envelope import attach_verification
from fpn_prover.runtime.pipeline import commit_trace_payload, precommit_request
from fpn_prover.runtime.setup_cert import SetupCertificate
from fpn_serve.capture import store as capture_store
from fpn_serve.capture.ports import (
    PAGE_TOKENS, QWEN3_4B, QWEN3_30B_A3B, CaptureError, CaptureLayout, CaptureOptions, CaptureTier,
    merge_row_stripes,
)
from fpn_serve.capture.ring import Span, _ReadyRead
from fpn_serve.capture.sink import CHUNK_PARALLEL_COPY_MIN_BYTES, AsyncDmaSink, Segment, StepRecord
from fpn_serve.capture.store import PageStore


def logical_rows(layout, count):
    for token in range(count):
        pattern = bytes((column * 13 + token * 29) % 251 for column in range(251))
        row = bytearray((pattern * ((layout.stride_bytes + 250) // 251))[:layout.stride_bytes])
        if layout.dims.moe:
            moe = layout.dims.moe
            for layer in range(layout.dims.n_layers):
                slot = layout.slot("expert_ids", layer)
                ids = [(token * 11 + layer + position * 9) % moe.n_experts for position in range(slot.width)]
                struct.pack_into(f"<{slot.width}H", row, 2 * slot.offset, *ids)
        yield bytes(row)


def assemble(layout, row, groups, tp, ep, *, corrupt=False):
    pieces = []
    for group in reversed(groups):
        width = group.col_hi - group.col_lo
        payload = bytearray(row[2 * group.col_lo:2 * group.col_hi])
        if layout.dims.moe:
            moe = layout.dims.moe
            for layer in range(group.layer_lo, min(group.layer_hi, layout.dims.n_layers)):
                slot = layout.slot("expert_ids", layer)
                offset = 2 * (slot.offset - group.col_lo)
                ids = struct.unpack_from(f"<{slot.width}H", payload, offset)
                routed = [[] for _ in range(ep)]
                for position, expert in enumerate(ids):
                    routed[expert // (moe.n_experts // ep)].append((position, expert))
                payload[offset:offset + 2 * slot.width] = bytes(2 * slot.width)
                for rank in reversed(routed):
                    for position, expert in reversed(rank):
                        struct.pack_into("<H", payload, offset + 2 * position, expert)

        stripes = []
        for rank in reversed(range(tp)):
            lo, hi = width * rank // tp, width * (rank + 1) // tp
            shard = payload[2 * lo:2 * hi]
            if corrupt and group.layer_lo == 0 and rank == 3:
                shard[0] ^= 1
            stripes.append((lo, bytes(shard)))
        pieces.append((group.col_lo, merge_row_stripes(width, stripes)))
    return merge_row_stripes(layout.row_u16, pieces)


class MultiGpuTopologyHidingTests(unittest.TestCase):
    tokens = tuple(range(2 * PAGE_TOKENS + 3))

    def capture(self, layout, rows, group_size, *, verify=True):
        store = PageStore(layout, layer_group_size=group_size, verify_payload_hash=verify)
        handle = store.open_request("reply")
        handle.append(b"".join(rows), self.tokens)
        return store, handle, handle.finish()

    def test_tp_ep_pp_preserve_logical_pages_commitment_and_envelopes(self):
        for dims in (QWEN3_4B, QWEN3_30B_A3B):
            layout = CaptureLayout.build(dims, CaptureOptions(tier=CaptureTier.TIER_B))
            rows = list(logical_rows(layout, len(self.tokens)))
            canonical_store, canonical_handle, canonical_ref = self.capture(layout, rows, 0)
            canonical_pages = list(canonical_handle.iter_verified_pages(self.tokens))
            root = commit_trace_payload(canonical_store, canonical_ref, "reply", self.tokens)
            cert = SetupCertificate(b"w" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 0)
            encoded = cert.encode()
            self.assertEqual(SetupCertificate.decode(encoded), cert)
            self.assertEqual({f.name for f in fields(cert)}, {
                "weight_commitment_root", "circuit_root", "rounding_contract_digest", "model_format_digest",
                "update_epoch",
            })
            self.assertEqual(len(encoded), len(b"FPN2_SETUP\x01") + 4 * 32 + 8)
            c_pre = precommit_request(encoded, root, [101, 102])
            # Valid wire shape suffices here: attach_verification does not verify the proof.
            receipt = header(profile_id(FOLDED_AIR_V1)).ljust(FOLDED_AIR_V1["c_p"], b"\0")

            for tp, pp, ep in ((1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1), (1, 2, 1), (1, 4, 1),
                               (4, 2, 4 if dims.moe else 1)):
                with self.subTest(model=dims.name, tp=tp, pp=pp, ep=ep):
                    group_size = (layout.n_layer_rows + pp - 1) // pp
                    groups = layout.layer_groups(group_size)
                    self.assertEqual(len(groups), pp)
                    merged = [assemble(layout, row, groups, tp, ep) for row in rows]
                    self.assertEqual(merged, rows)
                    store, handle, trace = self.capture(layout, merged, group_size)
                    self.assertEqual(trace, canonical_ref)
                    self.assertEqual(list(handle.iter_verified_pages(self.tokens)), canonical_pages)
                    merged_root = commit_trace_payload(store, trace, "reply", self.tokens)
                    self.assertEqual(merged_root, root)
                    merged_c_pre = precommit_request(encoded, merged_root, [101, 102])
                    self.assertEqual(merged_c_pre, c_pre)
                    for response in ({"type": "message"}, {"object": "chat.completion"}):
                        envelope = attach_verification(
                            response, receipt=receipt, setup_cert=encoded, challenge_mode=MODE_NI, c_pre=merged_c_pre
                        )
                        self.assertEqual(envelope, attach_verification(
                            response, receipt=receipt, setup_cert=encoded, challenge_mode=MODE_NI, c_pre=c_pre
                        ))
                        self.assertEqual(envelope.keys(), response.keys() | {
                            "fpn_receipt_b64", "setup_cert_id", "challenge_mode", "c_pre_hex"
                        })

    def test_rank_three_byte_is_committed_and_post_store_corruption_is_rejected(self):
        layout = CaptureLayout.build(QWEN3_30B_A3B, CaptureOptions(tier=CaptureTier.TIER_B))
        rows = list(logical_rows(layout, len(self.tokens)))
        group_size = (layout.n_layer_rows + 1) // 2
        groups = layout.layer_groups(group_size)
        store, handle, trace = self.capture(layout, rows, group_size)
        root = commit_trace_payload(store, trace, "reply", self.tokens)
        bad_row = assemble(layout, rows[0], groups, 4, 4, corrupt=True)
        offset = 2 * (groups[0].col_hi * 3 // 4)
        self.assertEqual(sum(a != b for a, b in zip(bad_row, rows[0])), 1)
        self.assertEqual(bad_row[offset], rows[0][offset] ^ 1)
        altered, _, bad_trace = self.capture(layout, [bad_row, *rows[1:]], group_size, verify=False)
        self.assertFalse(altered.payload_digests)
        self.assertEqual(bad_trace, trace)
        bad_root = commit_trace_payload(altered, bad_trace, "reply", self.tokens)
        cert = SetupCertificate(b"w" * 32, b"c" * 32, b"r" * 32, b"f" * 32, 0).encode()
        self.assertNotEqual(bad_root, root)
        self.assertNotEqual(precommit_request(cert, bad_root, [101]), precommit_request(cert, root, [101]))

        key = next(iter(handle.iter_verified_pages(self.tokens)))[0]
        self.assertEqual(len(store.payload_digests), 3 * len(groups))
        store.verify_page_digests(key)
        group = groups[0]
        part = (group.layer_lo, group.layer_hi, store.stripe)
        damaged = bytearray(store._pages[key].parts[part])
        damaged[offset] ^= 1
        store._pages[key].parts[part] = damaged
        with self.assertRaisesRegex(CaptureError, "payload digest mismatch"):
            store.verify_page_digests(key)
        with self.assertRaisesRegex(CaptureError, "payload digest mismatch"):
            commit_trace_payload(store, trace, "reply", self.tokens)
        store.close_request("reply")
        store.trim_chain(trace.tail)
        self.assertFalse(store.payload_digests)


class ChunkDrainTests(unittest.TestCase):
    @unittest.skipIf(capture_store._np is None, "parallel page copies require NumPy")
    def test_aggregate_full_pages_cross_threshold_and_decode_stays_serial(self):
        for batch in (4, 8):
            with self.subTest(batch=batch):
                self.check_chunk_drain(batch)

    def check_chunk_drain(self, batch):
        layout = CaptureLayout.build(QWEN3_4B, CaptureOptions(tier=CaptureTier.TIER_B))
        store = PageStore(layout)
        ring = SimpleNamespace(row_bytes=layout.stride_bytes, backend=None, capacity_rows=192)
        sink = AsyncDmaSink(ring, store, copy_workers=4)
        self.addCleanup(sink.stop)
        barrier = threading.Barrier(4, timeout=5)
        original = store._copy_into_pooled
        requests = [f"req-{i}" for i in range(batch)]
        prefill = 192 // batch
        full_pages = prefill // PAGE_TOKENS
        for request in requests:
            store.open_request(request)

        def ingest(count, start):
            segments = tuple(Segment(request, i * count, count, tuple(range(start + i * 100, start + i * 100 + count)))
                             for i, request in enumerate(requests))
            span = Span(0, count * len(requests))
            data = b"".join(bytes([i + 1]) * (count * layout.stride_bytes) for i in range(len(requests)))
            sink._ingest_chunk(StepRecord(span, segments), span, _ReadyRead(data), False)

        def concurrent(buf):
            barrier.wait()
            return original(buf)

        self.assertLess(prefill * layout.stride_bytes, capture_store.PARALLEL_COPY_MIN_BYTES)
        self.assertGreaterEqual(batch * full_pages * PAGE_TOKENS * layout.stride_bytes,
                                CHUNK_PARALLEL_COPY_MIN_BYTES)
        with patch.object(store, "_copy_into_pooled", side_effect=concurrent) as copy:
            ingest(prefill, 0)
            self.assertEqual(copy.call_count, batch * full_pages)
        with patch.object(sink._copy_pool, "submit", side_effect=AssertionError("decode woke the copy pool")):
            ingest(1, prefill)
        for i, request in enumerate(requests):
            handle = store.handle(request)
            handle.finish()
            tokens = list(range(i * 100, i * 100 + prefill + 1))
            pages = list(handle.iter_verified_pages(tokens))
            self.assertEqual([n for _, n, _ in pages], [PAGE_TOKENS] * full_pages + [prefill % PAGE_TOKENS + 1])
            self.assertEqual(b"".join(row for _, _, row in pages),
                             bytes([i + 1]) * ((prefill + 1) * layout.stride_bytes))
