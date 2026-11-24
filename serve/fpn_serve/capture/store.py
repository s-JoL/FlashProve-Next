# SPDX-License-Identifier: Apache-2.0
"""16-token content-addressed, write-once page store + O(new-tokens) per-request trace handles (host memory).

Addressing
----------
A *page* holds the captured rows of 16 consecutive tokens.  Its address is the hash chain
``h_i = BLAKE2b(h_{i-1} || tokens_i)`` seeded with the capture-layout fingerprint (``CaptureLayout.fingerprint``) and an
optional per-tenant ``extra`` salt, so identical prefixes under the same layout share pages and different layouts can
never collide.  Each page is stored as one *part* per ``(layer_range, tp_stripe)`` -- the store key is
``(block_hash, layer_range, tp_stripe)``.  (Hashing here is plain stdlib BLAKE2b content addressing: serve/ does no field
arithmetic and no Poseidon2.)

Rules
-----
* **Write-once, first write wins, no byte comparison.**  Batch non-invariance makes honest recomputes differ by 1 ulp,
  and the first writer's bytes are exactly what the engine's prefix-cached KV block corresponds to.  A second write of an
  existing part is a dedup hit (counted), never an error and never an overwrite.
* **Fail closed.**  Resolving a missing page raises :class:`PageMissingError`; a poisoned request cannot be finished;
  a full store with nothing evictable raises :class:`PageStoreFullError`.  Nothing is ever zero-filled.
* **Chain-linked.**  A page records its parent.  A request keeps only the *tip* hash (+ count + a <16-row tail), i.e.
  ``O(new tokens)`` state regardless of prefix length; every ancestor is protected from eviction because it has a child.
  Pages are evictable only when unpinned *and* childless (leaf-first, like the KV cache), in LRU order.
* **Eviction hook** :meth:`PageStore.evict` mirrors a KV-cache block eviction: it frees the page when allowed and
  otherwise defers (``pending_evict``) until the page becomes a free leaf.
* A trailing partial page (<16 tokens) is request-private: stored under a domain-separated hash that also commits to the
  token count, and never reachable as a prefix.
"""
from __future__ import annotations

import hashlib
import struct
import threading
from collections import OrderedDict, deque
from concurrent.futures import wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence

from .ports import PAGE_TOKENS, U16_BYTES, CaptureError, CaptureLayout, LayerGroup

__all__ = [
    "PageMissingError", "PageStoreFullError", "RequestPoisonedError", "PageKey", "PageStore", "RequestTraceHandle",
    "TraceRef", "chain_seed", "next_block_hash", "partial_block_hash", "block_hash_chain", "FULL_STRIPE",
]

PARALLEL_COPY_MIN_BYTES = 32 << 20
FULL_STRIPE = (0, 0)          # (chunk_lo, chunk_hi) in canonical chunk space; (0, 0) = the whole logical row


class PageMissingError(CaptureError):
    """A required page is not in the store (never zero-filled)."""


class PageStoreFullError(CaptureError):
    """Host budget exhausted and no unpinned leaf page can be evicted."""


class RequestPoisonedError(CaptureError):
    """Capture for this request is incomplete (ring overflow, sink fault, missing KV page): it must not be proven."""


class PageKey(NamedTuple):
    block_hash: bytes
    layer_lo: int
    layer_hi: int
    stripe: tuple[int, int]


# ============================================================================================================ hashing
def chain_seed(layout_fingerprint: bytes, extra: bytes = b"") -> bytes:
    return hashlib.blake2b(extra, key=layout_fingerprint, digest_size=32, person=b"fpn-page-root").digest()


def _tok_bytes(tokens: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(tokens)}I", *tokens)


def next_block_hash(parent: bytes, tokens: Sequence[int]) -> bytes:
    if len(tokens) != PAGE_TOKENS:
        raise ValueError("a full page hashes exactly 16 tokens")
    return hashlib.blake2b(_tok_bytes(tokens), key=parent, digest_size=32, person=b"fpn-page-full").digest()


def partial_block_hash(parent: bytes, tokens: Sequence[int]) -> bytes:
    if not 0 < len(tokens) < PAGE_TOKENS:
        raise ValueError("a partial page has 1..15 tokens")
    return hashlib.blake2b(bytes([len(tokens)]) + _tok_bytes(tokens), key=parent, digest_size=32, person=b"fpn-page-tail").digest()


def block_hash_chain(root: bytes, tokens: Sequence[int]) -> list[bytes]:
    """Hashes of every *full* 16-token block of ``tokens`` (a trailing partial block has no hash)."""
    out, h = [], root
    for i in range(0, len(tokens) - len(tokens) % PAGE_TOKENS, PAGE_TOKENS):
        h = next_block_hash(h, tokens[i:i + PAGE_TOKENS])
        out.append(h)
    return out


# ============================================================================================================== store
@dataclass
class _Page:
    ntokens: int
    parent: Optional[bytes]
    parts: dict[tuple[int, int, tuple[int, int]], Any] = field(default_factory=dict)
    pins: int = 0
    dropped: int = 0              # bytes seen but not retained (retain_payload=False)
    children: int = 0
    pending_evict: bool = False

    @property
    def nbytes(self) -> int:
        return sum(len(b) for b in self.parts.values())


try:
    import numpy as _np  # optional: releases the Python GIL during 32 MiB page copies on GPU hosts
except ImportError:
    _np = None


def _copy_payload(buf: Any) -> Any:
    """Single-copy buffer materialisation; uses NumPy copy (releases GIL) when available, else stdlib bytes."""
    if isinstance(buf, bytes):
        return buf
    if _np is not None:
        return _np.frombuffer(buf, dtype=_np.uint8).copy()
    return bytes(buf)


def _as_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if _np is not None and isinstance(payload, _np.ndarray):
        return payload.tobytes()
    return bytes(payload)


class PageStore:
    def __init__(self, layout: CaptureLayout, *, layer_group_size: int = 0, stripe: tuple[int, int] = FULL_STRIPE,
                 max_bytes: Optional[int] = None, on_evict: Optional[Callable[[bytes], None]] = None,
                 retain_payload: bool = True, verify_payload_hash: bool = False) -> None:
        self.layout = layout
        self.groups: list[LayerGroup] = layout.layer_groups(layer_group_size)
        self.stripe = stripe
        self.max_bytes = max_bytes
        self.on_evict = on_evict
        self._retain_payload = retain_payload
        self._verify_payload_hash = verify_payload_hash
        self.payload_digests: dict[PageKey, bytes] = {}
        self.row_bytes = layout.stride_bytes
        self._fingerprint = layout.fingerprint()
        self.seed = chain_seed(self._fingerprint)
        self._pages: "OrderedDict[bytes, _Page]" = OrderedDict()
        self._lock = threading.RLock()
        self._handles: dict[str, RequestTraceHandle] = {}
        self.bytes_stored = 0
        self.stats = dict(pages_written=0, dedup_hits=0, bytes_written=0, bytes_deduped=0, evictions=0,
                          deferred_evictions=0)
        self._full_page_bytes = PAGE_TOKENS * self.row_bytes
        self._buf_pool: list[Any] = []
        if _np is not None and max_bytes is not None and max_bytes >= (2 << 30) and self._full_page_bytes > 0:
            self._buf_pool_cap = min(8192, max(64, (max_bytes // 2) // self._full_page_bytes))
        else:
            self._buf_pool_cap = min(512, max(32, (256 << 20) // self._full_page_bytes)) if _np is not None else 0
        # Pre-touch a small pool on large GPU hosts so the very first prefill never stalls on cold mmap faults.
        if (
            _np is not None
            and retain_payload
            and max_bytes is not None
            and max_bytes >= (2 << 30)
            and self._full_page_bytes >= (1 << 20)
        ):
            n_warm = min(32, max(1, (max_bytes // 8) // self._full_page_bytes))
            for _ in range(n_warm):
                buf = _np.empty(self._full_page_bytes, dtype=_np.uint8)
                buf[::4096] = 0
                self._buf_pool.append(buf)

    def _acquire_buf(self, size: int) -> Any:
        """Reuse pre-touched full-page NumPy buffers to avoid repeated mmap/munmap TLB shootdowns."""
        if _np is not None and size == self._full_page_bytes:
            with self._lock:
                if self._buf_pool:
                    return self._buf_pool.pop()
            return _np.empty(size, dtype=_np.uint8)
        return _np.empty(size, dtype=_np.uint8) if _np is not None else bytearray(size)

    def _recycle_buf(self, buf: Any) -> None:
        # Called under the store lock after removing the part from its page.
        # Public reads always materialize immutable bytes under that same lock.
        if (_np is not None and isinstance(buf, _np.ndarray) and buf.flags.owndata
                and buf.dtype == _np.uint8 and buf.shape == (self._full_page_bytes,)
                and len(self._buf_pool) < self._buf_pool_cap):
            self._buf_pool.append(buf)

    def _copy_into_pooled(self, buf: Any) -> Any:
        if isinstance(buf, bytes):
            return buf
        if _np is not None:
            src = _np.frombuffer(buf, dtype=_np.uint8)
            dst = self._acquire_buf(src.size)
            _np.copyto(dst, src)
            return dst
        return bytes(buf)

    def _can_parallel_copy(self) -> bool:
        return (
            _np is not None and self.retain_payload and self._full_page_bytes >= (1 << 20)
            and len(self.groups) == 1 and self.groups[0].col_lo == 0
            and self.groups[0].col_hi * U16_BYTES == self.row_bytes
        )

    @property
    def retain_payload(self) -> bool:
        return self._retain_payload

    @retain_payload.setter
    def retain_payload(self, retain: bool) -> None:
        with self._lock:
            if retain != self._retain_payload and (self._handles or self._pages):
                raise CaptureError("retention can only change on an empty store")
            self._retain_payload = retain

    # ------------------------------------------------------------------------------------------------ writes
    def _split_rows(self, rows: Any, n: int, *, owned: bool = False) -> dict[tuple[int, int, tuple[int, int]], Any]:
        rb = self.row_bytes
        if not self.retain_payload:
            return {(g.layer_lo, g.layer_hi, self.stripe): n * (g.col_hi - g.col_lo) * U16_BYTES
                    for g in self.groups}
        if len(rows) != n * rb:
            raise ValueError(f"expected {n} rows of {rb} bytes, got {len(rows)} bytes")
        if len(self.groups) == 1:
            g = self.groups[0]
            c0, c1 = g.col_lo * U16_BYTES, g.col_hi * U16_BYTES
            if c0 == 0 and c1 == rb:
                key = (g.layer_lo, g.layer_hi, self.stripe)
                return {key: rows if owned else self._copy_into_pooled(rows)}
        mv = memoryview(rows)
        matrix = _np.frombuffer(mv, dtype=_np.uint8).reshape(n, rb) if _np is not None else None
        out = {}
        for g in self.groups:
            c0, c1 = g.col_lo * U16_BYTES, g.col_hi * U16_BYTES
            key = (g.layer_lo, g.layer_hi, self.stripe)
            if matrix is not None:
                out[key] = matrix[:, c0:c1].copy().reshape(-1)
            else:
                out[key] = b"".join(mv[i * rb + c0:i * rb + c1].tobytes() for i in range(n))
        return out

    def put_page(self, block_hash: bytes, parent: Optional[bytes], ntokens: int, rows: Any) -> bool:
        """Write-once insert.  Returns ``True`` if new bytes were stored, ``False`` on a pure dedup hit."""
        if len(rows) != ntokens * self.row_bytes:
            raise ValueError("row byte length does not match page token count")
        return self._put_page(block_hash, parent, ntokens, rows)

    def _put_page(self, block_hash: bytes, parent: Optional[bytes], ntokens: int, rows: Any, *,
                  owned: bool = False, pin: bool = False) -> bool:
        if not 1 <= ntokens <= PAGE_TOKENS:
            raise ValueError("ntokens must be 1..16")
        with self._lock:
            page = self._pages.get(block_hash)
            if page is not None and all((g.layer_lo, g.layer_hi, self.stripe) in page.parts for g in self.groups):
                if page.ntokens != ntokens or page.parent != parent:
                    raise CaptureError("block hash collision with different page metadata")
                self._pages.move_to_end(block_hash)
                if pin:
                    page.pins += 1
                self.stats["bytes_deduped"] += ntokens * self.row_bytes
                self.stats["dedup_hits"] += 1
                if owned:
                    self._recycle_buf(rows)
                return False
        parts = self._split_rows(rows, ntokens, owned=owned)
        with self._lock:
            page = self._pages.get(block_hash)
            if page is None:
                if parent is not None and parent not in self._pages:
                    raise PageMissingError("parent page missing: refusing to build an unlinked chain")
                need = sum(len(b) for b in parts.values()) if self.retain_payload else 0
                self._make_room(need)
                page = _Page(ntokens, parent)
                self._pages[block_hash] = page
                if parent is not None:
                    self._pages[parent].children += 1
            elif page.ntokens != ntokens or page.parent != parent:
                raise CaptureError("block hash collision with different page metadata")
            wrote = 0
            for k, payload in parts.items():
                if not self.retain_payload:
                    plen = payload if isinstance(payload, int) else len(payload)
                    if k not in page.parts:
                        page.parts[k] = b""
                        page.dropped += plen
                        wrote += plen
                    else:
                        self.stats["bytes_deduped"] += plen
                    continue
                if k in page.parts:
                    self.stats["bytes_deduped"] += len(payload)
                    if owned:
                        self._recycle_buf(payload)
                    continue
                if self._verify_payload_hash:
                    self.payload_digests[PageKey(block_hash, *k)] = hashlib.blake2b(payload, digest_size=32).digest()
                page.parts[k] = payload
                wrote += len(payload)
            self._pages.move_to_end(block_hash)
            if pin:
                page.pins += 1
            if wrote:
                if self.retain_payload:
                    self.bytes_stored += wrote
                self.stats["pages_written"] += 1
                self.stats["bytes_written"] += wrote
                return True
            self.stats["dedup_hits"] += 1
            return False

    # ------------------------------------------------------------------------------------------------ reads
    def has(self, block_hash: bytes) -> bool:
        with self._lock:
            return block_hash in self._pages

    def n_pages(self) -> int:
        with self._lock:
            return len(self._pages)

    def get_part(self, key: PageKey) -> bytes:
        with self._lock:
            page = self._pages.get(key.block_hash)
            if page is None or (key.layer_lo, key.layer_hi, key.stripe) not in page.parts:
                raise PageMissingError(f"page part {key.block_hash.hex()[:12]}.. {key[1:]} not in store")
            if not self.retain_payload:
                raise PageMissingError("payload not retained (store is in count-only measurement mode)")
            self._pages.move_to_end(key.block_hash)
            return _as_bytes(page.parts[(key.layer_lo, key.layer_hi, key.stripe)])

    def verify_page_digests(self, block_hash: bytes) -> None:
        """Check retained page parts against their write-time digests when enabled."""
        if not self._verify_payload_hash:
            return
        with self._lock:
            page = self._pages.get(block_hash)
            if page is None or not self.retain_payload:
                raise PageMissingError("no retained payload for digest verification")
            for part_key, payload in page.parts.items():
                key = PageKey(block_hash, *part_key)
                if self.payload_digests.get(key) != hashlib.blake2b(payload, digest_size=32).digest():
                    raise CaptureError(f"payload digest mismatch for {key}")

    def get_rows(self, block_hash: bytes) -> tuple[int, bytes]:
        """``(ntokens, rows)`` re-assembled from every layer-group part of this store's stripe."""
        with self._lock:
            page = self._pages.get(block_hash)
            if page is None:
                raise PageMissingError(f"page {block_hash.hex()[:12]}.. not in store")
            if not self.retain_payload:
                raise PageMissingError("payload not retained (store is in count-only measurement mode)")
            if self._verify_payload_hash:
                self.verify_page_digests(block_hash)
            n = page.ntokens
            try:
                cols = [_as_bytes(page.parts[(g.layer_lo, g.layer_hi, self.stripe)]) for g in self.groups]
            except KeyError:
                raise PageMissingError(f"page {block_hash.hex()[:12]}.. is missing a layer-group part") from None
            self._pages.move_to_end(block_hash)
        widths = [(g.col_hi - g.col_lo) * U16_BYTES for g in self.groups]
        if len(cols) == 1:
            return n, cols[0]
        out = bytearray()
        for i in range(n):
            for part, w in zip(cols, widths):
                out += part[i * w:(i + 1) * w]
        return n, bytes(out)

    def walk_to_root(self, tip: Optional[bytes]) -> list[bytes]:
        """Block hashes root -> ``tip`` (inclusive) by following parent links; raises if any link is gone."""
        out: list[bytes] = []
        with self._lock:
            h = tip
            while h is not None:
                page = self._pages.get(h)
                if page is None:
                    raise PageMissingError(f"chain broken at {h.hex()[:12]}..")
                out.append(h)
                h = page.parent
        out.reverse()
        return out

    # ------------------------------------------------------------------------------------------------ pin / evict
    def pin(self, block_hash: bytes) -> None:
        with self._lock:
            page = self._pages.get(block_hash)
            if page is None:
                raise PageMissingError("cannot pin a missing page")
            page.pins += 1

    def unpin(self, block_hash: bytes) -> None:
        with self._lock:
            page = self._pages.get(block_hash)
            if page is None or page.pins < 1:
                return
            page.pins -= 1
            if page.pending_evict:
                self._try_evict(block_hash)

    def _evictable(self, page: _Page) -> bool:
        return page.pins == 0 and page.children == 0

    def _try_evict(self, block_hash: bytes) -> bool:
        page = self._pages.get(block_hash)
        if page is None or not self._evictable(page):
            return False
        parent = page.parent
        del self._pages[block_hash]
        self.bytes_stored -= page.nbytes
        while page.parts:
            part_key, part = page.parts.popitem()
            if self._verify_payload_hash:
                self.payload_digests.pop(PageKey(block_hash, *part_key), None)
            self._recycle_buf(part)
        self.stats["evictions"] += 1
        if self.on_evict:
            self.on_evict(block_hash)
        if parent is not None and parent in self._pages:
            pp = self._pages[parent]
            pp.children -= 1
            if pp.pending_evict:
                self._try_evict(parent)
        return True

    def trim_chain(self, *tips: Optional[bytes]) -> int:
        """Free every currently-evictable page walking leaf -> root from each tip (no deferred flags: a page still shared
        with a live request simply stays).  Use after a request's trace has been consumed."""
        freed = 0
        with self._lock:
            for tip in tips:
                h = tip
                while h is not None:
                    page = self._pages.get(h)
                    if page is None:
                        break
                    parent = page.parent
                    if not self._try_evict(h):
                        break
                    freed += 1
                    h = parent
        return freed

    def evict(self, block_hash: bytes) -> bool:
        """KV-cache eviction hook.  Frees the page now if it is an unpinned leaf, else defers until it becomes one."""
        with self._lock:
            page = self._pages.get(block_hash)
            if page is None:
                return False
            if self._try_evict(block_hash):
                return True
            page.pending_evict = True
            self.stats["deferred_evictions"] += 1
            return False

    def _make_room(self, need: int) -> None:
        if self.max_bytes is None:
            return
        if need > self.max_bytes:
            raise PageStoreFullError("a single page exceeds the store budget")
        while self.bytes_stored + need > self.max_bytes:
            victim = next((h for h, p in self._pages.items() if self._evictable(p)), None)   # LRU order
            if victim is None:
                raise PageStoreFullError(
                    f"store budget {self.max_bytes} B exhausted by pinned/linked pages (stored {self.bytes_stored} B)")
            self._try_evict(victim)

    # ------------------------------------------------------------------------------------------------ requests
    def open_request(self, request_id: str, extra: bytes = b"") -> "RequestTraceHandle":
        with self._lock:
            if request_id in self._handles:
                raise CaptureError(f"request {request_id!r} already open")
            h = RequestTraceHandle(self, request_id, extra)
            self._handles[request_id] = h
            return h

    def handle(self, request_id: str) -> Optional["RequestTraceHandle"]:
        with self._lock:
            return self._handles.get(request_id)

    def open_ids(self) -> list[str]:
        with self._lock:
            return list(self._handles)

    def close_request(self, request_id: str) -> None:
        with self._lock:
            h = self._handles.pop(request_id, None)
        if h is not None:
            h.release()


@dataclass(frozen=True)
class TraceRef:
    """What the prover side receives for a finished request: ``O(1)`` regardless of context length."""

    request_id: str
    root: bytes
    tip: Optional[bytes]            # hash of the last FULL page (None if the request has < 16 tokens)
    n_full_pages: int
    tail: Optional[bytes]           # hash of the trailing partial page, if any
    n_tokens: int


class RequestTraceHandle:
    """Per-request capture state: tip hash + count + a < 16-row tail.  Never a dense ``max_len x W`` array."""

    OPEN, FINISHED, POISONED, RELEASED = "open", "finished", "poisoned", "released"

    def __init__(self, store: PageStore, request_id: str, extra: bytes = b"") -> None:
        self.store = store
        self.request_id = request_id
        self.root = chain_seed(store._fingerprint, extra) if extra else store.seed
        self.state = self.OPEN
        self.poison_reason: Optional[str] = None
        self._tip: Optional[bytes] = None
        self.n_full_pages = 0
        self.n_tokens = 0
        self._tail_rows = None
        self._tail_tokens: list[int] = []
        self._tail_hash: Optional[bytes] = None
        self._ref: Optional["TraceRef"] = None

    # -- state ------------------------------------------------------------------------------------------------------
    def state_bytes(self) -> int:
        """Approximate private host state (tail only; independent of the number of pages)."""
        return (0 if self._tail_rows is None else len(self._tail_rows)) + 8 * len(self._tail_tokens) + 32 * 3

    def _require_open(self) -> None:
        if self.state == self.POISONED:
            raise RequestPoisonedError(f"request {self.request_id!r} poisoned: {self.poison_reason}")
        if self.state != self.OPEN:
            raise CaptureError(f"request {self.request_id!r} is {self.state}")

    def poison(self, reason: str) -> None:
        """Mark capture incomplete (idempotent).  A poisoned request can never be finished -> never proven."""
        if self.state in (self.OPEN, self.FINISHED):
            self.state = self.POISONED
            self.poison_reason = reason

    # -- prefix mounting ------------------------------------------------------------------------------------------
    def mount_prefix(self, prefix_tokens: Sequence[int]) -> int:
        """Attach the already-captured pages of a prefix-cache hit (``len`` a multiple of 16).  Every page must exist;
        a missing one poisons the request and raises (fail closed).  Only the tip is retained."""
        self._require_open()
        if self.n_tokens or self._tip is not None:
            raise CaptureError("mount_prefix must come first")
        if len(prefix_tokens) % PAGE_TOKENS:
            raise ValueError("prefix must be whole 16-token pages")
        tip = None
        n = 0
        for h in block_hash_chain(self.root, prefix_tokens):
            if not self.store.has(h):
                self.poison(f"prefix page {n} missing from the page store")
                raise PageMissingError(f"prefix page {n} ({h.hex()[:12]}..) missing: KV page without captured trace")
            tip, n = h, n + 1
        if tip is not None:
            try:
                self.store.pin(tip)
            except PageMissingError:
                self.poison("prefix tip evicted while mounting")
                raise
            self._tip = tip
        self.n_full_pages = n
        self.n_tokens = n * PAGE_TOKENS
        return n

    # -- appending rows ---------------------------------------------------------------------------------------------
    def _buffer_tail(self, rows: memoryview) -> None:
        if not self.store.retain_payload:
            return
        if self._tail_rows is None:
            size = PAGE_TOKENS * self.store.row_bytes
            self._tail_rows = self.store._acquire_buf(size)
        start = len(self._tail_tokens) * self.store.row_bytes
        if _np is not None:
            _np.copyto(self._tail_rows[start:start + len(rows)], _np.frombuffer(rows, dtype=_np.uint8))
        else:
            self._tail_rows[start:start + len(rows)] = rows

    def _write_page(self, h: bytes, n: int, rows: Any, *, owned: bool = False, pin: bool = False) -> None:
        if owned and self.store.retain_payload:
            self.store._put_page(h, self._tip, n, rows, owned=True, pin=pin)
        elif self.store.retain_payload:
            with self.store._lock:
                self.store.put_page(h, self._tip, n, rows)
                if pin:
                    self.store.pin(h)
        else:
            self.store._put_page(h, self._tip, n, None, pin=pin)

    def append(self, rows: bytes, tokens: Sequence[int], *, copy_pool: Any = None, copy_workers: int = 1,
               _copied_pages: Optional[dict[int, Any]] = None) -> int:
        """Append captured rows for their input tokens and return the number of sealed pages."""
        self._require_open()
        rb = self.store.row_bytes
        n_in = len(tokens)
        if len(rows) != n_in * rb:
            self.poison("row/token count mismatch")
            raise CaptureError("rows do not match tokens")
        if n_in == 0:
            return 0

        sealed = 0
        page_bytes = PAGE_TOKENS * rb
        mv = memoryview(rows)
        t_list = [int(t) for t in tokens]
        idx = 0

        def _commit_full(toks_16: Sequence[int], blob: Any, *, owned: bool = False) -> None:
            nonlocal sealed
            parent = self._tip if self._tip is not None else self.root
            h = next_block_hash(parent, toks_16)
            self._write_page(h, PAGE_TOKENS, blob, owned=owned, pin=True)
            if self._tip is not None:
                self.store.unpin(self._tip)
            self._tip = h
            self.n_full_pages += 1
            sealed += 1

        try:
            # 1. If a partial tail exists, complete it first (at most 1 page copy)
            if self._tail_tokens:
                need = PAGE_TOKENS - len(self._tail_tokens)
                take = min(need, n_in)
                self._buffer_tail(mv[:take * rb])
                self._tail_tokens.extend(t_list[:take])
                idx = take
                if len(self._tail_tokens) == PAGE_TOKENS:
                    tail, self._tail_rows = self._tail_rows, None
                    _commit_full(self._tail_tokens, tail, owned=True)
                    self._tail_tokens.clear()

            # 2. Slice full 16-token pages directly from `mv` in O(N) (zero intermediate bytes copy!)
            parallel = (
                copy_pool is not None and self.store._can_parallel_copy()
                and ((n_in - idx) // PAGE_TOKENS) * page_bytes >= PARALLEL_COPY_MIN_BYTES
            )
            while parallel and idx + PAGE_TOKENS <= n_in:
                count = min(copy_workers, (n_in - idx) // PAGE_TOKENS)
                stripes = copy_workers // count if page_bytes >= (8 << 20) else 1
                if count == stripes == 1:
                    break
                futures = []
                copied = deque()
                try:
                    for i in range(count):
                        start = (idx + i * PAGE_TOKENS) * rb
                        src = mv[start:start + page_bytes]
                        if stripes == 1:
                            futures.append(copy_pool.submit(self.store._copy_into_pooled, src))
                        else:
                            dst = self.store._acquire_buf(page_bytes)
                            copied.append(dst)
                            src_array = _np.frombuffer(src, dtype=_np.uint8)
                            for j in range(stripes):
                                lo, hi = j * page_bytes // stripes, (j + 1) * page_bytes // stripes
                                futures.append(copy_pool.submit(_np.copyto, dst[lo:hi], src_array[lo:hi]))
                            del dst
                    for future in futures:
                        result = future.result()
                        if stripes == 1:
                            copied.append(result)
                    del result, future
                finally:
                    # A fault still has to wait for every reader of the DMA slab.
                    wait(futures)
                futures.clear()
                while copied:
                    _commit_full(t_list[idx:idx + PAGE_TOKENS], copied.popleft(), owned=True)
                    idx += PAGE_TOKENS

            while idx + PAGE_TOKENS <= n_in:
                p0 = idx * rb
                owned_page = _copied_pages.pop(idx, None) if _copied_pages else None
                if owned_page is None:
                    _commit_full(t_list[idx:idx + PAGE_TOKENS], mv[p0:p0 + page_bytes])
                else:
                    _commit_full(t_list[idx:idx + PAGE_TOKENS], owned_page, owned=True)
                idx += PAGE_TOKENS

            # 3. Buffer any < 16-token remainder in tail
            if idx < n_in:
                self._buffer_tail(mv[idx * rb:])
                self._tail_tokens.extend(t_list[idx:])
        except CaptureError as e:
            self.poison(f"{type(e).__name__}: {e}")
            raise
        self.n_tokens = self.n_full_pages * PAGE_TOKENS + len(self._tail_tokens)
        return sealed

    def finish(self) -> TraceRef:
        """Seal the trailing partial page (request-private) and return the O(1) reference.  Idempotent once finished;
        raises if poisoned."""
        if self.state == self.FINISHED and self._ref is not None:
            return self._ref
        self._require_open()
        if self._tail_tokens:
            parent = self._tip if self._tip is not None else self.root
            h = partial_block_hash(parent, self._tail_tokens)
            try:
                n = len(self._tail_tokens)
                rows = None if self._tail_rows is None else self._tail_rows[:n * self.store.row_bytes]
                self._write_page(h, n, rows)
                self.store.pin(h)
            except CaptureError as e:
                self.poison(f"{type(e).__name__}: {e}")
                raise
            self._tail_hash = h
            self._tail_rows = None                                  # the partial page now lives in the store
            self._tail_tokens.clear()
        self.state = self.FINISHED
        self._ref = TraceRef(self.request_id, self.root, self._tip, self.n_full_pages, self._tail_hash, self.n_tokens)
        return self._ref

    def iter_pages(self) -> Iterator[tuple[bytes, int, bytes]]:
        """``(block_hash, ntokens, rows)`` root -> end, including the partial tail once finished.  Missing -> raise."""
        if self.state == self.POISONED:
            raise RequestPoisonedError(f"request {self.request_id!r} poisoned: {self.poison_reason}")
        hashes = self.store.walk_to_root(self._tail_hash if self._tail_hash is not None else self._tip)
        for h in hashes:
            n, rows = self.store.get_rows(h)
            yield h, n, rows

    def iter_verified_pages(self, token_ids: Sequence[int]) -> Iterator[tuple[bytes, int, bytes]]:
        """Read settled full-width pages with their token chain checked against the forwarded token IDs."""
        if self.state != self.FINISHED or self.store.stripe != FULL_STRIPE or len(token_ids) != self.n_tokens:
            raise ValueError("expected a settled full-width trace and its forwarded token IDs")
        ref = self.finish()
        parent = ref.root
        tip = tail = None
        offset = full_pages = 0
        for key, count, rows in self.iter_pages():
            if count != min(PAGE_TOKENS, ref.n_tokens - offset) or count < 1:
                raise ValueError("invalid captured page token count")
            if len(rows) != count * self.store.row_bytes:
                raise ValueError("invalid captured page size")
            tokens = token_ids[offset:offset + count]
            if count == PAGE_TOKENS:
                expected = next_block_hash(parent, tokens)
                tip = key
                full_pages += 1
            else:
                expected = partial_block_hash(parent, tokens)
                tail = key
            if key != expected:
                raise ValueError("invalid capture token chain")
            yield key, count, rows
            parent = key
            offset += count
        if (offset, full_pages, tip, tail) != (ref.n_tokens, ref.n_full_pages, ref.tip, ref.tail):
            raise ValueError("incomplete capture token chain")

    def drop(self) -> int:
        """Release pins and free every page that is no longer shared (host-RAM hygiene once the trace was consumed)."""
        tip, tail = self._tip, self._tail_hash
        self.release()
        return self.store.trim_chain(tail, tip)

    def release(self) -> None:
        """Drop this request's pins (pages stay cached until LRU/KV eviction)."""
        if self.state == self.RELEASED:
            return
        for h in (self._tip, self._tail_hash):
            if h is not None:
                self.store.unpin(h)
        self.state = self.RELEASED
