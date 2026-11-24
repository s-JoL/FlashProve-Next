# SPDX-License-Identifier: Apache-2.0
"""``CaptureConfig`` (plain, serve-side) and ``CaptureSession`` (layout + ring + page store + DMA sink glue).

``serve/`` never sees the prover's ``ResourceBudget``: whoever builds the session (the bench driver, the vLLM plugin's
``model_loader_extra_config``) passes the numbers in explicitly.  No environment variable is read in this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .ports import CaptureError, CaptureLayout, CaptureOptions, CaptureTier, ModelDims
from .ring import CpuRingBackend, RingOverflowError, Span, StagingRing
from .sink import AsyncDmaSink, Segment, StepRecord
from .store import PageStore

__all__ = ["CaptureConfig", "CaptureSession", "StepHandle"]

TIER_RING_CAP_MIB = {
    CaptureTier.FULL_NO_KV: 350,
    CaptureTier.FULL_NO_KV_ACT: 350,
    CaptureTier.TIER_B: 128,
}


@dataclass(frozen=True)
class CaptureConfig:
    tier: CaptureTier = CaptureTier.FULL_NO_KV
    include_h_mid: bool = True               # TIER_B only
    capture_norm_outputs: bool = True        # FULL_NO_KV only (drop after gate M0-G1)
    capture_act: bool = True                 # FULL_NO_KV only: False for fused-MoE models (no ``a`` boundary)
    ring_mib: int = 320
    dma_chunk_mib: int = 64
    max_batched_tokens: int = 2048           # largest legal step: must fit in the ring
    max_inflight_dma: int = 8
    page_store_mib: Optional[int] = None     # host page-store cap (None = unbounded; set from the host budget)
    layer_group_size: int = 0
    retain_payload: bool = True              # False = count-only measurement mode (bench); never for proving
    enabled: bool = True
    copy_workers: int = 4

    def options(self) -> CaptureOptions:
        return CaptureOptions(self.tier, self.include_h_mid, self.capture_norm_outputs, self.capture_act)

    def ring_cap_mib(self) -> int:
        return min(self.ring_mib, TIER_RING_CAP_MIB[self.tier])

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureConfig":
        d = dict(d)
        if isinstance(d.get("tier"), str):
            d["tier"] = CaptureTier(d["tier"])
        return cls(**d)


@dataclass(frozen=True)
class StepHandle:
    span: Span
    record: StepRecord


class CaptureSession:
    def __init__(self, layout: CaptureLayout, ring: StagingRing, store: PageStore, sink: AsyncDmaSink, config: CaptureConfig) -> None:
        self.layout, self.ring, self.store, self.sink, self.config = layout, ring, store, sink, config
        self.scheduled: dict[str, int] = {}          # request_id -> rows already reserved (= next token position)
        self._finish_queued: set[str] = set()

    @classmethod
    def create(cls, dims: ModelDims, config: CaptureConfig, *, backend: str = "cpu", device: str = "cuda") -> "CaptureSession":
        layout = CaptureLayout.build(dims, config.options())
        stride = layout.stride_bytes
        rows = (config.ring_cap_mib() << 20) // stride
        if rows < 1:
            raise CaptureError(f"one row ({stride} B) does not fit in a {config.ring_cap_mib()} MiB ring")
        if backend == "cpu":
            be = CpuRingBackend(rows, layout.row_u16)
        elif backend == "cuda":
            from .ring import CudaRingBackend
            be = CudaRingBackend(rows, layout.row_u16, device=device, dma_chunk_mib=config.dma_chunk_mib)
        else:
            raise CaptureError(f"unknown backend {backend!r}")
        ring = StagingRing(be)
        ring.require_step_capacity(config.max_batched_tokens)       # fail at configuration time, not mid-serving
        store = PageStore(layout, layer_group_size=config.layer_group_size,
                          max_bytes=None if config.page_store_mib is None else config.page_store_mib << 20,
                          retain_payload=config.retain_payload)
        sink = AsyncDmaSink(ring, store, max_inflight=config.max_inflight_dma, copy_workers=config.copy_workers)
        return cls(layout, ring, store, sink, config)

    # ---------------------------------------------------------------------------------------------------- requests
    def register_request(self, request_id: str, prefix_tokens: Sequence[int] = (), extra: bytes = b"") -> None:
        """Open a trace handle; ``prefix_tokens`` are the tokens a prefix-cache hit let the engine skip (a multiple of
        16).  Their pages must already be in the store; if not, the request is poisoned (fail closed) -- not raised, the
        engine keeps serving."""
        h = self.store.open_request(request_id, extra)
        self.scheduled[request_id] = len(prefix_tokens)
        if prefix_tokens:
            try:
                h.mount_prefix(prefix_tokens)
            except CaptureError as e:
                h.poison(f"prefix mount failed: {e}")

    def finish_requests(self, request_ids: Sequence[str]) -> None:
        """Queue end-of-request markers behind every row already submitted for them."""
        ids = tuple(r for r in request_ids if self.store.handle(r) is not None and r not in self._finish_queued)
        for r in ids:
            self.scheduled.pop(r, None)
            self._finish_queued.add(r)
        if ids:
            self.sink.submit(StepRecord(Span(self.ring.write_wp, 0), (), None, ids))

    # ------------------------------------------------------------------------------------------------------ steps
    def begin_step(self, segments: Sequence[tuple], *, touched_rows: Optional[int] = None) -> Optional[StepHandle]:
        """Reserve ring rows for one engine step.  ``segments`` entries are either ``(request_id, input_token_ids)`` or
        deferred ``(request_id, n_rows, pos_start)`` (resolved asynchronously by the sink's ``token_resolver``).

        On overflow every request of this step is poisoned (their trace is now incomplete) and ``None`` is returned: the
        caller must make the step's taps write nothing (device ``n_valid = 0``).  Rows of earlier, still-undrained steps
        are untouched because the reservation was refused."""
        n = sum(item[1] if len(item) == 3 else len(item[1]) for item in segments)
        try:
            span = self.ring.reserve(n, touched_rows)
        except RingOverflowError as e:
            for item in segments:
                rid = item[0]
                h = self.store.handle(rid)
                if h is not None:
                    h.poison(f"ring overflow: {e}")
            return None
        segs, r0 = [], 0
        for item in segments:
            if len(item) == 3:
                rid, nrows, pos0 = item
                segs.append(Segment(rid, r0, int(nrows), (), pos_start=int(pos0)))
                self.scheduled[rid] = self.scheduled.get(rid, 0) + int(nrows)
                r0 += int(nrows)
            else:
                rid, toks = item
                segs.append(Segment(rid, r0, len(toks), tuple(int(x) for x in toks)))
                self.scheduled[rid] = self.scheduled.get(rid, 0) + len(toks)
                r0 += len(toks)
        return StepHandle(span, StepRecord(span, tuple(segs)))

    def end_step(self, step: StepHandle, ready_token: object = None) -> None:
        """The step's taps are enqueued (``ready_token`` = device event): hand the span to the sink."""
        self.ring.mark_ready(step.span.end)
        rec = step.record if ready_token is None else StepRecord(step.record.span, step.record.segments, ready_token, step.record.finish)
        self.sink.submit(rec)

    # --------------------------------------------------------------------------------------------------- mock path
    def step_cpu(self, segments: Sequence[tuple[str, Sequence[int]]], rows: bytes) -> bool:
        """CPU-mock producer: reserve, write ``rows`` into the mock ring, mark ready, submit.  ``False`` if refused."""
        step = self.begin_step(segments)
        if step is None:
            return False
        self.ring.publish(step.span, rows)
        self.sink.submit(step.record)
        return True

    def settle(self, timeout_s: float = 120.0) -> dict[str, object]:
        """Finish every open request, drain the sink and report ``{rid: TraceRef | error string}`` (never the step path)."""
        ids = self.store.open_ids()
        self.finish_requests(ids)
        self.sink.flush(timeout_s)
        return {r: self.sink.finished.get(r) or self.sink.failed.get(r, "unfinished") for r in ids}

    def forget(self, request_ids: Sequence[str]) -> None:
        """Release pins, free unshared pages and drop all per-request bookkeeping."""
        for r in request_ids:
            h = self.store.handle(r)
            if h is not None:
                h.drop()
            self.store.close_request(r)
            self._finish_queued.discard(r)
            self.sink.finished.pop(r, None)
            self.sink.failed.pop(r, None)
            self.scheduled.pop(r, None)

    def close(self) -> None:
        self.sink.stop()
