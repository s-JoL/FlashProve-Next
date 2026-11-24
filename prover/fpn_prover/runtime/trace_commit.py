# SPDX-License-Identifier: Apache-2.0
"""Merkle commitment of a settled capture trace (every token row) and compact openings of drawn rows.

`commit_trace_merkle` streams every verified page of a settled full-width trace (`iter_verified_pages`, token chain
checked) and builds a BLAKE2b-256 Merkle tree with one leaf per captured token row:

    leaf_t = BLAKE2b-256(person="FPN2-TROW-v1",  le64(t) || le64(len(row)) || row)            t < n_tokens
    pad    = BLAKE2b-256(person="FPN2-TPAD-v1",  "")                                           t >= n_tokens
    node   = BLAKE2b-256(person="FPN2-TNODE-v1", left || right)
    trace_payload_root = BLAKE2b-256(person="FPN2-TROOT-v1",
                                     "FPN2-TRPAY-v2" || le64(n_tokens) || layout_fingerprint (32) || tree_root)

over a complete binary tree of depth ceil(log2 n_tokens).  The root binds the token count and the capture-layout
fingerprint, so the same logical rows give the same root whatever the TP / PP / EP page assembly (topology hiding).
`trace_payload_root` is what `pipeline.precommit_request` folds into C_pre BEFORE the sampling seed is derived;
`open_trace_rows` then emits inclusion proofs for the drawn positions and `verify_trace_row_openings` checks them
against the root using only public data (n_tokens = forwarded token count, layout fingerprint, row width).

Scope: the leaves authenticate the payload bytes the store held WHEN the commitment was computed.  An edit to the
store before `commit_trace_merkle` runs is committed as-is (capture-time payload authentication is a serving-side
property; see `PageStore(verify_payload_hash=True)`).  An opening discloses the WHOLE row of a drawn position to the
holder of the opening (the auditor), not only the slots a relation consumes.  Unlike the flat payload hash,
the root does not include the capture page-chain hashes: the forwarded token ids are bound through the Protocol v2
seed (prompt bytes) and the response through C_pre, not through `trace_payload_root`.
"""
import hashlib
import struct
from dataclasses import dataclass

LEAF_PERSON = b"FPN2-TROW-v1\0\0\0\0"
NODE_PERSON = b"FPN2-TNODE-v1\0\0\0"
PAD_PERSON = b"FPN2-TPAD-v1\0\0\0\0"
ROOT_PERSON = b"FPN2-TROOT-v1\0\0\0"
ROOT_DOMAIN = b"FPN2-TRPAY-v2"
BIND_PERSON = b"FPN2-XBIND-v1\0\0\0"
OPENING_MAGIC = b"FPN2TROP\x01"
PAD_LEAF = hashlib.blake2b(b"", digest_size=32, person=PAD_PERSON).digest()

E_TRACE_DECODE = "E_TRACE_DECODE"
E_TRACE_HEADER = "E_TRACE_HEADER"
E_TRACE_POSITION = "E_TRACE_POSITION"
E_TRACE_ROW_LEN = "E_TRACE_ROW_LEN"
E_TRACE_ROW_PATH = "E_TRACE_ROW_PATH"


def row_leaf(t, row):
    h = hashlib.blake2b(digest_size=32, person=LEAF_PERSON)
    h.update(struct.pack("<QQ", t, len(row)))
    h.update(row)
    return h.digest()


def _node(left, right):
    return hashlib.blake2b(left + right, digest_size=32, person=NODE_PERSON).digest()


def tree_depth(n_tokens):
    if type(n_tokens) is not int or n_tokens < 1:
        raise ValueError("a trace commitment needs n_tokens >= 1")
    return (n_tokens - 1).bit_length()


def payload_root(tree_root, n_tokens, layout_fingerprint):
    if type(layout_fingerprint) is not bytes or len(layout_fingerprint) != 32:
        raise ValueError("layout_fingerprint must be 32 bytes")
    h = hashlib.blake2b(digest_size=32, person=ROOT_PERSON)
    h.update(ROOT_DOMAIN)
    h.update(struct.pack("<Q", n_tokens))
    h.update(layout_fingerprint)
    h.update(tree_root)
    return h.digest()


@dataclass(frozen=True)
class TraceTree:
    """Prover-side Merkle tree of one settled trace (levels[0] = padded leaves, levels[-1] = [tree_root])."""
    root: bytes                  # trace_payload_root (what C_pre binds)
    n_tokens: int
    layout_fingerprint: bytes
    row_bytes: int
    levels: tuple
    pages: tuple                 # ((first_position, count, rows), ...) -- retained page payloads for openings

    def row(self, t):
        for first, count, rows in self.pages:
            if first <= t < first + count:
                start = (t - first) * self.row_bytes
                return bytes(rows[start:start + self.row_bytes])
        raise ValueError("position not retained in this trace tree")


def _build_levels(leaves):
    depth = tree_depth(len(leaves))
    level = list(leaves) + [PAD_LEAF] * ((1 << depth) - len(leaves))
    levels = [tuple(level)]
    while len(level) > 1:
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(tuple(level))
    return tuple(levels)


def commit_trace_merkle(store, trace, request_id, token_ids, *, retain_rows=True):
    """TraceTree over every captured token row of a settled full-width trace (read via `iter_verified_pages`)."""
    handle = store.handle(request_id)
    if (
        handle is None or handle.state != handle.FINISHED or handle.finish() != trace
        or trace.request_id != request_id
    ):
        raise ValueError("expected a matching settled full-width trace")
    row_bytes = store.row_bytes
    fingerprint = store.layout.fingerprint()
    leaves, pages = [], []
    offset = 0
    for _page_hash, count, rows in handle.iter_verified_pages(token_ids):   # drained: completeness check at the end
        if len(rows) != count * row_bytes:
            raise ValueError("invalid captured page size")
        view = memoryview(rows)
        for i in range(count):
            leaves.append(row_leaf(offset + i, view[i * row_bytes:(i + 1) * row_bytes]))
        if retain_rows:
            pages.append((offset, count, rows))
        offset += count
    if offset != trace.n_tokens or offset < 1:
        raise ValueError("trace commitment does not cover every captured token")
    levels = _build_levels(leaves)
    return TraceTree(payload_root(levels[-1][0], offset, fingerprint), offset, fingerprint, row_bytes, levels,
                     tuple(pages))



# ----------------------------------------------------------------------------- incremental (write-once sessions)
@dataclass(frozen=True)
class IncrementalCommit:
    """Result of `IncrementalTraceCommitter.commit`: the SAME TraceTree / root as `commit_trace_merkle`, plus the
    page split (leading reused pages, fresh pages) and the per-page row leaves (for the St_i page ledger)."""
    tree: TraceTree
    prefix_pages: int            # leading pages whose row leaves were reused from an earlier commit
    page_leaves: tuple           # per page: tuple of its row leaves (32 B each)
    stats: dict

    def page_digest_payload(self, j):
        """Ledger payload of page j: the concatenation of its FPN2-TROW-v1 row leaves (so the St_i ledger and
        trace_payload_root are built from the SAME per-row leaves)."""
        return b"".join(self.page_leaves[j])


class IncrementalTraceCommitter:
    """Prover-side incremental `commit_trace_merkle` for a growing session whose prefix pages are written once.

    The capture page store is content-addressed by a TOKEN-chain hash (`block_hash`), and every page of a trace sits
    at a fixed absolute position, so the row leaves `row_leaf(t, row)` of a page that an earlier committed trace
    already contained are identical.  `commit` therefore (1) checks the trace's page chain against
    `chain_hashes(ref.root, token_ids)` (no row is read for that), (2) reuses the cached leaves of every already
    committed page and reads / hashes ONLY the rows of the pages it has not seen (O(fresh tokens) payload hashing,
    `threads` workers; hashlib releases the GIL), and (3) rebuilds the upper Merkle levels over all leaves (O(n)
    32-byte node hashes, timed separately).  The returned root is bit-identical to `commit_trace_merkle`'s (the
    bench cross-checks it at the last turn).  Scope: like `commit_trace_merkle`, the leaves authenticate the payload
    the store held when the page was FIRST committed (pages are write-once in the store)."""

    def __init__(self, store, chain_hashes, *, threads=16, page_tokens=16):
        from concurrent.futures import ThreadPoolExecutor
        self.store, self.chain_hashes, self.page_tokens = store, chain_hashes, page_tokens
        self.cache = {}              # block hash -> (first position, token count, leaves tuple)
        self.pool = ThreadPoolExecutor(max_workers=max(1, threads))

    def close(self):
        self.pool.shutdown(wait=True)

    def _hash_page(self, h, first, count, retain):
        n, rows = self.store.get_rows(h)
        rb = self.store.row_bytes
        if n != count or len(rows) != count * rb:
            raise ValueError("invalid captured page size")
        view = memoryview(rows)
        leaves = tuple(row_leaf(first + i, view[i * rb:(i + 1) * rb]) for i in range(count))
        return leaves, (rows if retain else None)

    def commit(self, trace, request_id, token_ids, *, retain_fresh_rows=False):
        import time
        store = self.store
        handle = store.handle(request_id)
        if (
            handle is None or handle.state != handle.FINISHED or getattr(store, "stripe", None) != (0, 0)
            or handle.finish() != trace or trace.request_id != request_id
            or len(token_ids) != trace.n_tokens or trace.n_tokens < 1
        ):
            raise ValueError("expected a matching settled full-width trace and its forwarded token IDs")
        t0 = time.perf_counter()
        ref = handle.finish()
        hashes = store.walk_to_root(ref.tail if ref.tail is not None else ref.tip)
        if list(hashes) != list(self.chain_hashes(ref.root, token_ids)):
            raise ValueError("invalid capture token chain")
        n, pt = trace.n_tokens, self.page_tokens
        if len(hashes) != -(-n // pt):
            raise ValueError("trace page count does not cover every captured token")
        t1 = time.perf_counter()
        page_leaves = [None] * len(hashes)
        todo = []
        for j, h in enumerate(hashes):
            first, count = j * pt, min(pt, n - j * pt)
            c = self.cache.get(h)
            if c is not None and c[0] == first and c[1] == count:
                page_leaves[j] = c[2]
            else:
                todo.append((j, h, first, count))
        prefix_pages = todo[0][0] if todo else len(hashes)
        if any(page_leaves[j] is None for j in range(prefix_pages)) or \
                any(page_leaves[j] is not None for j in range(prefix_pages, len(hashes))):
            raise ValueError("reused pages are not a leading run of the trace")
        futs = [(j, h, first, count, self.pool.submit(self._hash_page, h, first, count, retain_fresh_rows))
                for j, h, first, count in todo]
        pages = []
        for j, h, first, count, f in futs:
            leaves, rows = f.result()
            page_leaves[j] = leaves
            self.cache[h] = (first, count, leaves)
            if retain_fresh_rows:
                pages.append((first, count, rows))
        t2 = time.perf_counter()
        leaves = [x for pl in page_leaves for x in pl]
        if len(leaves) != n:
            raise ValueError("trace commitment does not cover every captured token")
        levels = _build_levels(leaves)
        fingerprint = store.layout.fingerprint()
        tree = TraceTree(payload_root(levels[-1][0], n, fingerprint), n, fingerprint, store.row_bytes, levels,
                         tuple(pages))
        t3 = time.perf_counter()
        fresh_rows = sum(c for _, _, _, c in todo)
        stats = {"pages": len(hashes), "pages_reused": prefix_pages, "pages_fresh": len(todo),
                 "rows_hashed": fresh_rows, "payload_bytes_hashed": fresh_rows * store.row_bytes,
                 "chain_check_ms": (t1 - t0) * 1e3, "fresh_row_leaves_ms": (t2 - t1) * 1e3,
                 "tree_levels_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}
        return IncrementalCommit(tree, prefix_pages, tuple(page_leaves), stats)

    def forget(self, block_hashes):
        for h in block_hashes:
            self.cache.pop(h, None)

# ----------------------------------------------------------------------------- openings
@dataclass(frozen=True)
class TraceRowOpening:
    position: int
    row: bytes
    path: tuple                  # sibling hashes, leaf level first


@dataclass(frozen=True)
class TraceRowOpenings:
    n_tokens: int
    layout_fingerprint: bytes
    row_bytes: int
    openings: tuple              # TraceRowOpening, strictly increasing positions

    def encode(self):
        out = [OPENING_MAGIC, struct.pack("<Q", self.n_tokens), self.layout_fingerprint,
               struct.pack("<II", self.row_bytes, len(self.openings))]
        for o in self.openings:
            out += [struct.pack("<Q", o.position), o.row, *o.path]
        return b"".join(out)

    @classmethod
    def decode(cls, data):
        data = bytes(data)
        m = len(OPENING_MAGIC)
        if len(data) < m + 8 + 32 + 8 or data[:m] != OPENING_MAGIC:
            raise ValueError("bad trace-row opening header")
        (n_tokens,) = struct.unpack_from("<Q", data, m)
        fingerprint = data[m + 8:m + 40]
        row_bytes, count = struct.unpack_from("<II", data, m + 40)
        depth = tree_depth(n_tokens)
        off = m + 48
        item = 8 + row_bytes + 32 * depth
        if len(data) != off + count * item:
            raise ValueError("bad trace-row opening length")
        openings = []
        for _ in range(count):
            (t,) = struct.unpack_from("<Q", data, off)
            row = data[off + 8:off + 8 + row_bytes]
            p0 = off + 8 + row_bytes
            openings.append(TraceRowOpening(t, row, tuple(data[p0 + 32 * i:p0 + 32 * (i + 1)] for i in range(depth))))
            off += item
        return cls(n_tokens, fingerprint, row_bytes, tuple(openings))


def open_trace_rows(tree, positions):
    """Inclusion proofs of the rows at `positions` (deduplicated, sorted) against `tree.root`."""
    out = []
    for t in sorted(set(int(p) for p in positions)):
        if not 0 <= t < tree.n_tokens:
            raise ValueError("opening position outside the committed trace")
        row = tree.row(t)
        if row_leaf(t, row) != tree.levels[0][t]:
            raise ValueError("retained row no longer matches its committed leaf")
        path, k = [], t
        for level in tree.levels[:-1]:
            path.append(level[k ^ 1])
            k >>= 1
        out.append(TraceRowOpening(t, row, tuple(path)))
    return TraceRowOpenings(tree.n_tokens, tree.layout_fingerprint, tree.row_bytes, tuple(out))


def _fail(code, detail):
    return {"verdict": code, "detail": detail, "rows": {}}


def verify_trace_row_openings(trace_payload_root, n_tokens, layout_fingerprint, openings, *, row_bytes=None,
                              expected_positions=None):
    """Check every opened row against `trace_payload_root` using only PUBLIC parameters.

    `n_tokens` (the forwarded token count), `layout_fingerprint` and `row_bytes` (the public capture layout) come
    from the verifier, never from the bundle; the bundle header must merely agree.  With `expected_positions` the
    opened set must equal it exactly (a missing or extra opening fails).  Returns {"verdict": "VERIFIED", "rows":
    {position: row bytes}} or {"verdict": <error code>, "detail": ...}."""
    if type(trace_payload_root) is not bytes or len(trace_payload_root) != 32:
        return _fail(E_TRACE_HEADER, "trace_payload_root must be 32 bytes")
    try:
        bundle = openings if isinstance(openings, TraceRowOpenings) else TraceRowOpenings.decode(openings)
        depth = tree_depth(n_tokens)
    except (ValueError, struct.error) as e:
        return _fail(E_TRACE_DECODE, str(e))
    if (bundle.n_tokens != n_tokens or bundle.layout_fingerprint != layout_fingerprint
            or (row_bytes is not None and bundle.row_bytes != row_bytes)):
        return _fail(E_TRACE_HEADER, "opening header disagrees with the public trace parameters")
    width = bundle.row_bytes if row_bytes is None else row_bytes
    positions = [o.position for o in bundle.openings]
    if any(type(t) is not int or not 0 <= t < n_tokens for t in positions) or positions != sorted(set(positions)):
        return _fail(E_TRACE_POSITION, "positions must be distinct, increasing and inside the trace")
    if expected_positions is not None and set(positions) != {int(t) for t in expected_positions}:
        return _fail(E_TRACE_POSITION, "opened positions differ from the re-derived draw positions")
    rows = {}
    for o in bundle.openings:
        if type(o.row) is not bytes or len(o.row) != width:
            return _fail(E_TRACE_ROW_LEN, f"row {o.position} has {len(o.row)} bytes, expected {width}")
        if len(o.path) != depth or any(type(s) is not bytes or len(s) != 32 for s in o.path):
            return _fail(E_TRACE_ROW_PATH, f"row {o.position}: path length / sibling size")
        acc, k = row_leaf(o.position, o.row), o.position
        for sib in o.path:
            acc = _node(sib, acc) if k & 1 else _node(acc, sib)
            k >>= 1
        if payload_root(acc, n_tokens, layout_fingerprint) != trace_payload_root:
            return _fail(E_TRACE_ROW_PATH, f"row {o.position} does not open against trace_payload_root")
        rows[o.position] = o.row
    return {"verdict": "VERIFIED", "detail": f"{len(rows)} rows", "rows": rows}


# ----------------------------------------------------------------------------- declared-input binding digest
def activation_binding_digest(items, domain=b""):
    """32-byte digest of the exact activation words a relation consumed: items = ((tag, position, layer, words), ...)
    with `words` the raw little-endian BF16 bytes.  Callers fold it into a relation's public statement nonce and the
    verifier recomputes it from rows opened against `trace_payload_root` (never from prover-supplied words)."""
    h = hashlib.blake2b(digest_size=32, person=BIND_PERSON)
    h.update(struct.pack("<Q", len(domain)))
    h.update(domain)
    h.update(struct.pack("<Q", len(items)))
    for tag, position, layer, words in items:
        tag = tag.encode() if type(tag) is str else bytes(tag)
        words = bytes(words)
        h.update(struct.pack("<Q", len(tag)))
        h.update(tag)
        h.update(struct.pack("<qqQ", int(position), int(layer), len(words)))
        h.update(words)
    return h.digest()
