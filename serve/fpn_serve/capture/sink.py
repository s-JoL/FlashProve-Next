# SPDX-License-Identifier: Apache-2.0
"""``AsyncDmaSink``: drains completed staging-ring slabs into the page store, off the decode step path.

The engine thread only calls :meth:`submit` (append a tiny ``StepRecord`` to a deque -- no device interaction).  The
sink (a worker thread, or ``pump()`` driven manually in tests) then, strictly in step order:

1. starts ``ring.read_async`` for each <=64 MiB chunk of the record's span (device backend: ``cudaMemcpyAsync`` on the
   dedicated DMA stream behind the step's event; CPU mock: immediate);
2. polls completion (``done()`` = ``Event.query()``; never a synchronize);
3. splits the bytes per request segment and appends them to that request's :class:`RequestTraceHandle`
   (which seals 16-token pages into the :class:`PageStore`);
4. release the pinned slab after ingestion. Ring rows can be reused as soon as DMA is complete.

Fail-closed: rows lost to an overwrite, a store that cannot accept a page, or any unexpected error poison the affected
requests (``RequestTraceHandle.poison``) instead of dropping or zero-filling data; an unexpected sink fault poisons every
queued request and makes ``submit`` raise.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

from .ports import PAGE_TOKENS, CaptureError
from .ring import AsyncRead, RingNotReadyError, RingOverflowError, Span, StagingRing
from .store import PageStore, TraceRef

__all__ = ["Segment", "StepRecord", "AsyncDmaSink", "SinkFaultedError"]

CHUNK_PARALLEL_COPY_MIN_BYTES = 16 << 20


class SinkFaultedError(CaptureError):
    """The DMA sink hit an unexpected error; capture is no longer trustworthy."""


@dataclass(frozen=True)
class Segment:
    request_id: str
    row_start: int                    # first row of this request inside the step's span
    n_rows: int
    token_ids: tuple[int, ...]        # the input tokens of those rows (empty if deferred via pos_start)
    pos_start: Optional[int] = None   # optional deferred token offset for async scheduling (resolved at sink ingest)


@dataclass(frozen=True)
class StepRecord:
    span: Span
    segments: tuple[Segment, ...]
    ready_token: Any = None           # device backend: the step-completion event
    finish: tuple[str, ...] = ()      # requests that end after this record's rows (processed in step order)


@dataclass
class _Inflight:
    rec: StepRecord
    todo: Deque[Span]                                  # chunks whose DMA has not been enqueued yet
    reads: list[tuple[Span, AsyncRead]] = field(default_factory=list)
    lost: bool = False
    wait_since: Optional[float] = None                 # monotonic timestamp when first blocked on deferred tokens


class AsyncDmaSink:
    def __init__(self, ring: StagingRing, store: PageStore, *, max_inflight: int = 8, idle_sleep_s: float = 0.0001,
                 token_resolver: Any = None, on_settled: Any = None, deferred_timeout_s: float = 5.0, copy_workers: Optional[int] = None) -> None:
        self.ring = ring
        self.store = store
        self.max_inflight = max(1, max_inflight)
        self.idle_sleep_s = idle_sleep_s
        self.token_resolver = token_resolver
        self.on_settled = on_settled
        self.deferred_timeout_s = float(deferred_timeout_s)
        self.chunk_rows = int(getattr(ring.backend, "chunk_rows", ring.capacity_rows))
        if copy_workers is None:
            copy_workers = 4
        self.copy_workers = max(1, copy_workers)
        self._copy_pool = self._make_copy_pool()
        self._queue: Deque[StepRecord] = deque()
        self._inflight: Deque[_Inflight] = deque()
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self.fault: Optional[BaseException] = None
        self.finished: dict[str, TraceRef] = {}     # request_id -> O(1) reference for the prover side
        self.failed: dict[str, str] = {}            # request_id -> why it can never be proven
        self.stats = dict(records=0, rows=0, bytes=0, lost_records=0, orphan_segments=0, poisoned_requests=0)

    def _make_copy_pool(self) -> Optional[ThreadPoolExecutor]:
        if self.copy_workers == 1:
            return None
        return ThreadPoolExecutor(max_workers=self.copy_workers, thread_name_prefix="fpn-page-copy")

    # ------------------------------------------------------------------------------------------------ producer side
    def _settle_record_ids(self, rec: StepRecord) -> None:
        if self.on_settled is None:
            return
        seen: set[str] = set()
        for seg in rec.segments:
            if seg.request_id not in seen:
                seen.add(seg.request_id)
                self.on_settled(seg.request_id)
        for rid in rec.finish:
            if rid not in seen:
                seen.add(rid)
                self.on_settled(rid)

    def submit(self, rec: StepRecord) -> None:
        with self._cv:
            if self.fault is not None:
                self._settle_record_ids(rec)
                raise SinkFaultedError(f"DMA sink faulted: {self.fault!r}") from self.fault
            self._queue.append(rec)
            self._cv.notify()

    def check(self) -> None:
        with self._cv:
            if self.fault is not None:
                raise SinkFaultedError(f"DMA sink faulted: {self.fault!r}") from self.fault

    @property
    def pending(self) -> int:
        with self._cv:
            return len(self._queue) + len(self._inflight)

    # ------------------------------------------------------------------------------------------------ worker side
    def _poison(self, rec: StepRecord, reason: str) -> None:
        for seg in rec.segments:
            h = self.store.handle(seg.request_id)
            if h is not None:
                h.poison(reason)
                self.stats["poisoned_requests"] += 1

    def _advance(self, infl: _Inflight) -> None:
        """Enqueue as many pending chunk DMAs as currently possible (in order)."""
        while infl.todo and not infl.lost:
            try:
                infl.reads.append((infl.todo[0], self.ring.read_async(infl.todo[0], infl.rec.ready_token)))
            except RingNotReadyError:
                return                                  # rows not enqueueable yet / no free pinned slab: retry next pump
            except RingOverflowError as e:
                infl.lost = True
                infl.todo.clear()
                self._poison(infl.rec, f"ring overflow: {e}")
                return
            infl.todo.popleft()

    def _release_completed(self) -> None:
        """Reclaim the contiguous DMA-complete prefix independently of host page assembly."""
        copied = self.ring.copied_wp
        upto = copied
        for infl in self._inflight:
            for span, read in infl.reads:
                if span.end <= upto:
                    continue
                if span.start != upto or not read.done():
                    break
                upto = span.end
            if upto < infl.rec.span.end and infl.rec.span.n_rows:
                break
        if upto != copied:
            self.ring.release(upto)

    def _copy_chunk_pages(self, rec: StepRecord, c0: int, c1: int, mv: memoryview) -> dict[int, dict[int, Any]]:
        """Copy independently sealable pages across requests before publishing any of them."""
        pool = self._copy_pool
        if (pool is None or len(rec.segments) < 2 or len(mv) < CHUNK_PARALLEL_COPY_MIN_BYTES
                or not self.store._can_parallel_copy()):
            return {}

        rb = self.store.row_bytes
        occurrences = Counter(seg.request_id for seg in rec.segments)
        spans = []
        for index, seg in enumerate(rec.segments):
            lo, hi = max(c0, seg.row_start), min(c1, seg.row_start + seg.n_rows)
            if hi - lo < PAGE_TOKENS or occurrences[seg.request_id] != 1:
                continue
            handle = self.store.handle(seg.request_id)
            if handle is None or handle.state != handle.OPEN:
                continue
            # An existing tail needs a serial join before the independently copyable pages.
            first = (-len(handle._tail_tokens)) % PAGE_TOKENS
            for offset in range(first, hi - lo - PAGE_TOKENS + 1, PAGE_TOKENS):
                spans.append((index, offset, (lo - c0 + offset) * rb))

        page_bytes = PAGE_TOKENS * rb
        if len(spans) * page_bytes < CHUNK_PARALLEL_COPY_MIN_BYTES:
            return {}
        futures = {}
        try:
            for index, offset, start in spans:
                page = mv[start:start + page_bytes]
                futures[index, offset] = pool.submit(self.store._copy_into_pooled, page)
            copied: dict[int, dict[int, Any]] = {}
            for (index, offset), future in futures.items():
                copied.setdefault(index, {})[offset] = future.result()
            return copied
        finally:
            # Even a failed submit or copy must leave no workers reading the DMA slab.
            wait(futures.values())

    def _ingest_chunk(self, rec: StepRecord, sp: Span, r: Any, lost: bool) -> None:
        """Zero-intermediate-copy ingestion of one completed DMA chunk + immediate ring & slab release."""
        rb = self.ring.row_bytes
        if hasattr(r, "as_memoryview"):
            mv = r.as_memoryview()
        else:
            mv = memoryview(r.take())
        nbytes = len(mv)
        try:
            if not lost:
                if nbytes != sp.n_rows * rb:  # pragma: no cover - defensive
                    self._poison(rec, "short DMA read")
                else:
                    c0 = sp.start - rec.span.start
                    c1 = c0 + sp.n_rows
                    copied = self._copy_chunk_pages(rec, c0, c1, mv)
                    for index, seg in enumerate(rec.segments):
                        s0 = seg.row_start
                        s1 = s0 + seg.n_rows
                        ov0 = max(c0, s0)
                        ov1 = min(c1, s1)
                        if ov0 >= ov1:
                            continue
                        h = self.store.handle(seg.request_id)
                        if h is None:
                            self.stats["orphan_segments"] += 1
                            continue
                        seg_off = ov0 - s0
                        n_ov = ov1 - ov0
                        if seg.token_ids:
                            toks = seg.token_ids[seg_off:seg_off + n_ov]
                        elif seg.pos_start is not None and self.token_resolver is not None:
                            resolved = self.token_resolver(seg.request_id, seg.pos_start + seg_off, n_ov)
                            if resolved is None or len(resolved) != n_ov:
                                h.poison(f"deferred token resolution failed at pos={seg.pos_start + seg_off}")
                                self.stats["poisoned_requests"] += 1
                                continue
                            toks = tuple(int(t) for t in resolved)
                        else:
                            h.poison("missing token_ids")
                            self.stats["poisoned_requests"] += 1
                            continue
                        try:
                            h.append(mv[(ov0 - c0) * rb:(ov1 - c0) * rb], toks,
                                     copy_pool=self._copy_pool if not copied else None, copy_workers=self.copy_workers,
                                     _copied_pages=copied.get(index))
                        except CaptureError:
                            self.stats["poisoned_requests"] += 1
                    self.stats["rows"] += sp.n_rows
                    self.stats["bytes"] += nbytes
        finally:
            mv.release()
            if hasattr(r, "release"):
                r.release()

    def _finish_record(self, infl: _Inflight) -> None:
        rec = infl.rec
        if rec.span.n_rows and rec.span.end > self.ring.copied_wp:
            self.ring.release(rec.span.end)
        if infl.lost:
            self.stats["lost_records"] += 1
        for rid in rec.finish:
            h = self.store.handle(rid)
            if h is None or rid in self.finished or rid in self.failed:
                if self.on_settled is not None:
                    self.on_settled(rid)
                continue                                          # unknown / already settled: finishing twice is a no-op
            try:
                self.finished[rid] = h.finish()
            except CaptureError as e:
                self.failed[rid] = f"{type(e).__name__}: {e}"
            if self.on_settled is not None:
                self.on_settled(rid)
        self.stats["records"] += 1

    def pump(self) -> int:
        """One non-blocking pass; returns the number of records fully drained."""
        self.check()
        drained = 0
        try:
            with self._cv:
                while self._queue and len(self._inflight) < self.max_inflight:
                    rec = self._queue.popleft()
                    self._inflight.append(_Inflight(rec, deque(rec.span.split(self.chunk_rows) if rec.span.n_rows else [])))
            for infl in self._inflight:
                self._advance(infl)
            self._release_completed()
            while self._inflight:
                head = self._inflight[0]
                unresolved = (
                    self.token_resolver is not None and not head.lost and any(
                        not s.token_ids and s.pos_start is not None
                        and self.token_resolver(s.request_id, s.pos_start, s.n_rows) is None
                        for s in head.rec.segments
                    )
                )
                if unresolved:
                    now = time.monotonic()
                    if head.wait_since is None:
                        head.wait_since = now
                    if (now - head.wait_since) < self.deferred_timeout_s:
                        break                              # async scheduling: wait briefly for sampled tokens (no sync)
                    # Timeout expired (e.g. cancelled request whose sampled token will never arrive):
                    # fall through so _ingest_chunk poisons ONLY the missing segment(s), releases the ring, and never deadlocks!
                else:
                    head.wait_since = None
                # Stream completed chunks off head IMMEDIATELY so ring rows & pinned slabs are freed mid-step!
                while head.reads and head.reads[0][0].end <= self.ring.copied_wp:
                    sp, r = head.reads.pop(0)
                    self._ingest_chunk(head.rec, sp, r, head.lost)
                    if head.todo:
                        self._advance(head)
                        self._release_completed()
                if head.todo or head.reads:
                    break
                self._finish_record(head)
                self._inflight.popleft()
                drained += 1
        except Exception as e:     # noqa: BLE001 - any unexpected error fails the whole sink closed
            with self._cv:
                self.fault = e
                faulted_recs = list(self._queue) + [infl.rec for infl in self._inflight]
                self._queue.clear()
                self._inflight.clear()
                for rec in faulted_recs:
                    try:
                        self._poison(rec, f"sink fault: {e!r}")
                    except Exception:                      # noqa: BLE001 - store itself may have faulted
                        pass
                    self._settle_record_ids(rec)
            raise SinkFaultedError(f"DMA sink faulted: {e!r}") from e
        return drained

    # ------------------------------------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = False
        if self._copy_pool is None:
            self._copy_pool = self._make_copy_pool()

        def loop() -> None:
            while not self._stop:
                try:
                    n = self.pump()
                except SinkFaultedError:
                    return
                if not n:
                    with self._cv:
                        if not self._stop and (not self._queue or self._inflight):
                            self._cv.wait(self.idle_sleep_s if self._inflight else None)

        self._thread = threading.Thread(target=loop, name="fpn-dma-sink", daemon=True)
        self._thread.start()

    def flush(self, timeout_s: float = 30.0) -> None:
        """Block the *caller* until everything submitted so far is stored (tests / shutdown; never on the step path)."""
        deadline = time.monotonic() + timeout_s
        while self.pending:
            self.check()
            if self._thread is None:
                self.pump()
            if time.monotonic() > deadline:
                raise TimeoutError("DMA sink flush timed out")
            time.sleep(self.idle_sleep_s)
        self.check()

    def stop(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._copy_pool is not None:
            self._copy_pool.shutdown()
            self._copy_pool = None
