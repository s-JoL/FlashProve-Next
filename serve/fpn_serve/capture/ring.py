# SPDX-License-Identifier: Apache-2.0
"""Staging ring: a *small* fail-closed handoff buffer between the GPU taps and the DMA sink (NOT reply storage).

Watermarks (all monotonic 64-bit row counters, ``physical row = counter % capacity_rows``)::

        copied_wp  <=  ready_wp  <=  write_wp
        |-- rows already DMA'd to host (safe to overwrite)
                       |-- rows whose tap kernels have completed (readable)
                                      |-- rows reserved for steps already scheduled

Invariant (checked on the HOST before a step is scheduled)::

        write_wp + max(committed_rows, touched_rows) - copied_wp  <=  capacity_rows

``committed_rows`` is how many rows the step really produces; ``touched_rows`` is how many the launch will write
(a CUDA-graph replay launches with capture-time arguments, so it may touch the padded capture size).  Violating the
invariant raises :class:`RingOverflowError` and leaves the ring unchanged.  A reader that asks for rows which were
overwritten, never produced, or already released also raises -- missing rows are NEVER zero-filled, truncated or
returned partially.

Backends
--------
``CpuRingBackend``   bytearray-backed mock, pre-filled with a non-zero poison pattern so tests can prove that no read
                     ever surfaces unwritten bytes.  Every edge case of the ring is exercised on it without a GPU.
``CudaRingBackend``  one VRAM tensor + a 4-slot int64 control block ``ctl = [cursor, n_valid, flag, 0]`` that the tap kernels
                     read on-device (so CUDA graphs replay the right offsets), and a dedicated DMA stream that waits on
                     a step event and copies slabs (<= ``dma_chunk_mib``) into pinned host buffers with
                     ``cudaMemcpyAsync``.  Completion is polled with ``Event.query()``.  This module contains NO
                     ``synchronize`` call anywhere: the decode step path is sync-free by construction (an AST test
                     enforces it across ``serve/``).
"""
from __future__ import annotations

import ctypes
import struct
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .ports import CaptureError, U16_BYTES

__all__ = [
    "RingOverflowError", "RingNotReadyError", "RingStateError", "Span", "StagingRing", "CpuRingBackend",
    "CudaRingBackend", "AsyncRead", "RingBackend",
]

MAX_WP = (1 << 63) - 1          # 64-bit monotonic watermark: refuse (fail closed) rather than ever wrap


class RingOverflowError(CaptureError):
    """Capture would overwrite (or already overwrote) rows that have not been copied out.  Fail closed."""


class RingNotReadyError(CaptureError):
    """Rows requested before their tap kernels completed."""


class RingStateError(CaptureError):
    """Programming error against the ring protocol (release past ready, double-release, bad span, ...)."""


@dataclass(frozen=True)
class Span:
    """Rows ``[start, start + n_rows)`` in watermark coordinates (not physical slots)."""

    start: int
    n_rows: int

    @property
    def end(self) -> int:
        return self.start + self.n_rows

    def split(self, max_rows: int) -> list["Span"]:
        if max_rows < 1:
            raise RingStateError("max_rows must be >= 1")
        return [Span(s, min(max_rows, self.end - s)) for s in range(self.start, self.end, max_rows)]


class AsyncRead(Protocol):
    def done(self) -> bool: ...
    def take(self) -> bytes: ...          # only valid once done(); releases the slab


class RingBackend(Protocol):
    capacity_rows: int
    row_u16: int

    def read_async(self, span: Span, after: Any = None) -> AsyncRead: ...


# ==================================================================================================== CPU mock backend
class _ReadyRead:
    def __init__(self, data: bytes) -> None:
        self._d = data

    def done(self) -> bool:
        return True

    def take(self) -> bytes:
        return self._d

    def as_memoryview(self) -> memoryview:
        return memoryview(self._d)

    def release(self) -> None:
        pass


class CpuRingBackend:
    """Mock device ring in host memory.  ``poison`` fills all slots (non-zero) so a stale read is detectable."""

    def __init__(self, capacity_rows: int, row_u16: int, poison: int = 0xA5A5) -> None:
        if capacity_rows < 1 or row_u16 < 1:
            raise RingStateError("capacity_rows and row_u16 must be >= 1")
        self.capacity_rows = capacity_rows
        self.row_u16 = row_u16
        self.row_bytes = row_u16 * U16_BYTES
        self._poison = poison & 0xFFFF
        pat = self._poison.to_bytes(2, "little") * row_u16
        self.buf = bytearray(pat * capacity_rows)

    # -- device-control mock: lets the engine-side glue (plugin runner wrappers) run unchanged against the CPU ring ----------
    last_ctl: tuple[int, int] = (0, 0)

    def begin_step(self, span: "Span", n_valid: int) -> None:
        self.last_ctl = (span.start, n_valid)

    def end_step(self) -> Any:
        self.last_ctl = (self.last_ctl[0], 0)         # like the device: n_valid is zeroed so later forwards write nothing
        return None

    def device_write(self, rows: bytes) -> None:
        """What the tap kernels would do for the step published by the last ``begin_step`` (tests only)."""
        cursor, n_valid = self.last_ctl
        if len(rows) != n_valid * self.row_bytes:
            raise RingStateError("emulated device write does not match n_valid")
        self.write_rows(cursor, rows)

    def write_rows(self, start: int, rows: bytes) -> None:
        rb = self.row_bytes
        if len(rows) % rb:
            raise RingStateError("rows must be whole rows")
        n = len(rows) // rb
        for i in range(n):
            p = (start + i) % self.capacity_rows
            self.buf[p * rb:(p + 1) * rb] = rows[i * rb:(i + 1) * rb]

    def read_rows(self, start: int, n_rows: int) -> bytes:
        rb = self.row_bytes
        out = bytearray()
        for i in range(n_rows):
            p = (start + i) % self.capacity_rows
            out += self.buf[p * rb:(p + 1) * rb]
        return bytes(out)

    def read_async(self, span: Span, after: Any = None) -> _ReadyRead:
        return _ReadyRead(self.read_rows(span.start, span.n_rows))


# ======================================================================================================= watermark ring
class StagingRing:
    """Host-side watermark bookkeeping + the invariant; data movement is delegated to a backend."""

    def __init__(self, backend: RingBackend) -> None:
        self.backend = backend
        self.capacity_rows = int(backend.capacity_rows)
        self.row_u16 = int(backend.row_u16)
        self.row_bytes = self.row_u16 * U16_BYTES
        self._lock = threading.Lock()
        self.write_wp = 0
        self.ready_wp = 0
        self.copied_wp = 0
        self._touched_hwm = 0           # highest row (exclusive) any launch may have written
        self._lost_below = 0            # rows < this were overwritten before being copied (poisoned)
        self.overflow_events = 0

    @classmethod
    def cpu(cls, capacity_rows: int, row_u16: int, poison: int = 0xA5A5) -> "StagingRing":
        return cls(CpuRingBackend(capacity_rows, row_u16, poison))

    # ------------------------------------------------------------------------------------------------ producer side
    def require_step_capacity(self, max_rows_per_step: int) -> None:
        """Configuration-time guard: a step can never legally exceed the ring."""
        if max_rows_per_step > self.capacity_rows:
            raise RingOverflowError(
                f"max rows per step {max_rows_per_step} exceeds ring capacity {self.capacity_rows}: "
                "lower max_num_batched_tokens or enlarge the staging ring"
            )

    def reserve(self, committed_rows: int, touched_rows: Optional[int] = None, *, poison_on_overflow: bool = False) -> Span:
        """Reserve ``committed_rows`` for the next step.

        ``touched_rows >= committed_rows`` accounts for launches that write more rows than they commit.
        Default: overflow raises and nothing changes.  ``poison_on_overflow=True`` is for GPU graphs whose kernels will
        run (and overwrite) regardless: the watermark advances, every un-copied row that gets clobbered is marked lost
        (unreadable forever) and the call still raises nothing -- the caller inspects :attr:`lost_below`."""
        touched = committed_rows if touched_rows is None else touched_rows
        if committed_rows < 0 or touched < committed_rows:
            raise RingStateError("need 0 <= committed_rows <= touched_rows")
        with self._lock:
            if committed_rows > self.capacity_rows or touched > self.capacity_rows:
                self.overflow_events += 1
                raise RingOverflowError(f"step of {max(committed_rows, touched)} rows exceeds ring capacity {self.capacity_rows}")
            if self.write_wp + committed_rows > MAX_WP:
                raise RingOverflowError("64-bit watermark exhausted")
            end_touched = self.write_wp + touched
            over = end_touched - self.copied_wp - self.capacity_rows
            if over > 0:
                self.overflow_events += 1
                if not poison_on_overflow:
                    raise RingOverflowError(
                        f"ring full: wp={self.write_wp} copied={self.copied_wp} step={touched} cap={self.capacity_rows}")
                self._lost_below = max(self._lost_below, self.copied_wp + over)
            span = Span(self.write_wp, committed_rows)
            self.write_wp += committed_rows
            self._touched_hwm = max(self._touched_hwm, end_touched)
            return span

    def publish(self, span: Span, rows: bytes) -> None:
        """Mock/CPU producers: write the step's rows then mark them ready.  (GPU producers write via kernels and call
        :meth:`mark_ready` from the step-completion event.)"""
        if not isinstance(self.backend, CpuRingBackend):
            raise RingStateError("publish() is only for the CPU mock backend")
        if len(rows) != span.n_rows * self.row_bytes:
            raise RingStateError("published rows do not match the span")
        with self._lock:
            if span.start < self._lost_below or self._touched_hwm - span.start > self.capacity_rows:
                raise RingOverflowError("span was overwritten before it was published")
        self.backend.write_rows(span.start, rows)
        self.mark_ready(span.end)

    def mark_ready(self, upto_wp: int) -> None:
        with self._lock:
            if upto_wp > self.write_wp:
                raise RingStateError("cannot mark rows ready that were never reserved")
            self.ready_wp = max(self.ready_wp, upto_wp)

    # ------------------------------------------------------------------------------------------------ consumer side
    def check_readable(self, span: Span) -> None:
        with self._lock:
            if span.start < 0 or span.n_rows < 0:
                raise RingStateError("bad span")
            if span.start < self.copied_wp:
                raise RingStateError(f"rows [{span.start},{span.end}) were already released")
            if span.start < self._lost_below or self._touched_hwm - span.start > self.capacity_rows:
                raise RingOverflowError(f"rows [{span.start},{span.end}) were overwritten before being copied out")
            if span.end > self.ready_wp:
                raise RingNotReadyError(f"rows up to {span.end} requested but only {self.ready_wp} are ready")

    def read(self, span: Span) -> bytes:
        """Synchronous read (mock / tests): guarded, exact, all-or-nothing."""
        self.check_readable(span)
        if not isinstance(self.backend, CpuRingBackend):
            raise RingStateError("use read_async() with a device backend")
        return self.backend.read_rows(span.start, span.n_rows)

    def read_async(self, span: Span, after: Any = None) -> AsyncRead:
        """Start copying ``span`` out of the ring (guarded).  The data must be consumed (``take``) before ``release``."""
        self.check_readable(span)
        return self.backend.read_async(span, after)

    def release(self, upto_wp: int) -> None:
        """Mark rows ``< upto_wp`` as copied out (safe to overwrite).  Monotonic; may not pass ``ready_wp``."""
        with self._lock:
            if upto_wp < self.copied_wp:
                raise RingStateError("release watermark must not go backwards")
            if upto_wp > self.ready_wp:
                raise RingStateError("cannot release rows that are not ready")
            self.copied_wp = upto_wp

    @property
    def lost_below(self) -> int:
        return self._lost_below

    @property
    def free_rows(self) -> int:
        with self._lock:
            return self.capacity_rows - (self.write_wp - self.copied_wp)


# ===================================================================================================== CUDA device backend
class _CudaRead:
    def __init__(self, backend: "CudaRingBackend", slab: int, nbytes: int, event: Any) -> None:
        self._b, self._slab, self._n, self._ev = backend, slab, nbytes, event
        self._taken = False
        self._complete = False

    def done(self) -> bool:
        if not self._complete:
            self._complete = bool(self._ev.query())
        return self._complete

    def as_memoryview(self) -> memoryview:
        if self._taken:
            raise RingStateError("AsyncRead already taken/released")
        if not self.done():
            raise RingNotReadyError("DMA still in flight")
        return self._b._slab_views[self._slab][:self._n]

    def release(self) -> None:
        if not self._taken:
            self._taken = True
            self._b._free_slab(self._slab)

    def take(self) -> bytes:
        mv = self.as_memoryview()
        out = mv.tobytes()
        mv.release()
        self.release()
        return out


class CudaRingBackend:
    """VRAM ring + control block + event-fenced DMA stream (see module docstring).  Lazy ``torch`` import.

    The ring memory is deliberately NOT initialised (no zero-fill): rows are only ever read through watermark-guarded
    spans, so unwritten memory is unreachable.  The control block ``ctl`` (int64[4] = ``[cursor, n_valid, violation, 0]``) is the only
    zero-initialised allocation."""

    def __init__(self, capacity_rows: int, row_u16: int, device: str = "cuda", dma_chunk_mib: int = 64, n_slabs: int = 16) -> None:
        import torch  # lazy: serve/ must import on machines without torch

        self.torch = torch
        self.capacity_rows = int(capacity_rows)
        self.row_u16 = int(row_u16)
        self.row_bytes = self.row_u16 * U16_BYTES
        self.device = torch.device(device)
        self.chunk_rows = max(1, (dma_chunk_mib << 20) // self.row_bytes)
        if self.chunk_rows > self.capacity_rows:
            self.chunk_rows = self.capacity_rows
        self.buf = torch.empty((self.capacity_rows, self.row_u16), dtype=torch.int16, device=self.device)
        self.ctl = torch.zeros(4, dtype=torch.int64, device=self.device)
        self.dma_stream = torch.cuda.Stream(device=self.device)
        self._slabs = [torch.empty((self.chunk_rows, self.row_u16), dtype=torch.int16, pin_memory=True) for _ in range(n_slabs)]
        self._slab_views = [memoryview(s.numpy()).cast("B") for s in self._slabs]
        self._free = list(range(n_slabs))
        self._slab_lock = threading.Lock()
        # pinned staging for [cursor, n_valid]; rotated so an in-flight H2D is never overwritten by the next step
        self._ctl_host = [torch.zeros(2, dtype=torch.int64).pin_memory() for _ in range(64)]
        self._ctl_views = [memoryview(h.numpy()) for h in self._ctl_host]
        self._ctl_input = self.ctl[:2]
        self._ctl_valid = self.ctl[1:2]
        self._ctl_i = 0

    # -- control block ----------------------------------------------------------------------------------------------
    def begin_step(self, span: Span, n_valid: int) -> None:
        """Stream-ordered (no host sync): publish ``[cursor, n_valid]`` to the device before the step's kernels."""
        i = self._ctl_i % len(self._ctl_host)
        self._ctl_i += 1
        struct.pack_into("<qq", self._ctl_views[i], 0, span.start, n_valid)
        self._ctl_input.copy_(self._ctl_host[i], non_blocking=True)

    def end_step(self) -> Any:
        """Stream-ordered: zero ``n_valid`` (so later dummy/warm-up forwards write nothing into rows the DMA stream may
        still be reading), then record the step-completion event on the *current* (compute) stream."""
        self._ctl_valid.zero_()
        ev = self.torch.cuda.Event()
        ev.record()
        return ev

    # -- DMA ----------------------------------------------------------------------------------------------------------
    def _take_slab(self) -> Optional[int]:
        with self._slab_lock:
            return self._free.pop() if self._free else None

    def _free_slab(self, i: int) -> None:
        with self._slab_lock:
            self._free.append(i)

    def read_async(self, span: Span, after: Any = None) -> _CudaRead:
        if span.n_rows > self.chunk_rows:
            raise RingStateError(f"span of {span.n_rows} rows exceeds the DMA chunk of {self.chunk_rows}: split it first")
        slab = self._take_slab()
        if slab is None:
            raise RingNotReadyError("no free pinned DMA slab (sink is behind)")
        torch = self.torch
        with torch.cuda.stream(self.dma_stream):
            if after is not None:
                self.dma_stream.wait_event(after)                # device-side wait, no host block
            p0 = span.start % self.capacity_rows
            first = min(span.n_rows, self.capacity_rows - p0)
            dst = self._slabs[slab]
            dst[:first].copy_(self.buf[p0:p0 + first], non_blocking=True)                      # cudaMemcpyAsync
            if first < span.n_rows:
                dst[first:span.n_rows].copy_(self.buf[:span.n_rows - first], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.dma_stream)
        return _CudaRead(self, slab, span.n_rows * self.row_bytes, done)
