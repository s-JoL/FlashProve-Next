# SPDX-License-Identifier: Apache-2.0
"""Incremental multi-turn transcript state chain `St_i` for agentic long-context sessions (M3-2).

An agentic session is a sequence of turns; turn i forwards `total_i` tokens of which the first `prefix_i` are a
cache-read prefix of an earlier verified context (their pages were captured and committed ONCE, when they were
fresh) and `fresh_i = total_i - prefix_i` are new.  Every turn publishes a 192-byte record whose state

    St_0 = H("fpn2/agentic-state/genesis/v1", setup_cert_id, layout_fp)
    St_i = H("fpn2/agentic-state/turn/v1", St_{i-1}, turn_idx_u64, prefix_tokens_u64, fresh_tokens_u64,
             total_tokens_u64, cached_prefix_root, fresh_pages_root, response_digest)

(H = BLAKE2b-256 over `le64(len(domain)) || domain || fields`; integers are 8-byte little endian, roots / states
32 raw bytes) chains it to its predecessor.  `fresh_pages_root` commits the page hashes of the fresh tokens only
(bound to their first position, token count and page count); `cached_prefix_root` commits the first `prefix_i`
tokens of the verified parent context through an append-only page-level Merkle tree (`PageLedger`), so the page
hashes of a session are written to the ledger exactly once and a later turn re-commits its prefix by one O(log n)
prefix-root evaluation instead of re-hashing it.  `precommit_agentic_turn` folds `St_i` into the per-turn C_pre
(`pipeline.precommit_request`), i.e. BEFORE the Protocol v2 challenge seed exists.

`AgenticChainVerifier` checks turns incrementally (O(fresh pages) per turn, state kept between turns):

    E_GENESIS             first record is not chained to St_0 of (setup certificate, layout fingerprint)
    E_TURN_ORDER          turn_idx != parent.turn_idx + 1, or a rollback / equivocation on one chain
    E_PREFIX_EXCEEDS_PREV prefix_tokens > parent.total_tokens
    E_TOKEN_COUNT         total != prefix + fresh, fresh < 1, or prefix not page aligned
    E_STATE_CHAIN         St_i does not recompute / prev_state is not the verified parent
    E_FRESH_ROOT          fresh_pages_root does not recompute from the supplied fresh page hashes
    E_CPRE_MISMATCH       published C_pre does not recompute from (St_i, trace root, response, policy)
    E_PREFIX_ROOT         cached_prefix_root differs from the verified parent context's prefix root (additional)

Forking: `fork(state)` opens a new branch rooted at ANY previously verified state (agentic tool-retry), sharing the
verified history.  Inside one chain a record whose parent is an earlier state than the tip (rollback) or a second
child of an already extended state (equivocation) is rejected `E_TURN_ORDER`; the only way to branch is `fork`.

Scope: this module authenticates turn structure and page-hash accounting.  The ledger leaves (`page_hash`) are
digests of the same captured rows that `trace_commit` commits token-by-token, but nothing here proves that a
`trace_payload_root` and the page ledger describe the same rows -- a verifier holding the rows recomputes both.
The chain says nothing about the correctness of the model output; it prevents re-ordering, truncation-and-replay
and silent substitution of the transcript a proof is about.
"""
import hashlib
import struct
import time
from dataclasses import dataclass

from .pipeline import precommit_request
from .setup_cert import SetupCertificate

GENESIS_DOMAIN = b"fpn2/agentic-state/genesis/v1"
TURN_DOMAIN = b"fpn2/agentic-state/turn/v1"
RESPONSE_DOMAIN = b"fpn2/agentic-state/response/v1"
PAGE_DOMAIN = b"fpn2/agentic-state/page/v1"
NODE_DOMAIN = b"fpn2/agentic-state/node/v1"
EMPTY_DOMAIN = b"fpn2/agentic-state/empty/v1"
FRESH_ROOT_DOMAIN = b"fpn2/agentic-state/fresh-root/v1"
PREFIX_ROOT_DOMAIN = b"fpn2/agentic-state/prefix-root/v1"
CPRE_DOMAIN = b"fpn2/agentic-state/cpre/v1"
RECORD_MAGIC = b"FPN2AGST\x01"
PAGE_TOKENS = 16

E_GENESIS = "E_GENESIS"
E_TURN_ORDER = "E_TURN_ORDER"
E_PREFIX_EXCEEDS_PREV = "E_PREFIX_EXCEEDS_PREV"
E_TOKEN_COUNT = "E_TOKEN_COUNT"
E_STATE_CHAIN = "E_STATE_CHAIN"
E_FRESH_ROOT = "E_FRESH_ROOT"
E_CPRE_MISMATCH = "E_CPRE_MISMATCH"
E_PREFIX_ROOT = "E_PREFIX_ROOT"
VERIFIED = "VERIFIED"


class AgenticChainError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _u64(x, what):
    if type(x) is not int or not 0 <= x < 1 << 64:
        raise TypeError(f"{what} must be an exact int in [0, 2^64)")
    return x


def _b32(x, what):
    if type(x) is not bytes or len(x) != 32:
        raise TypeError(f"{what} must be 32 bytes")
    return x


def _h(domain, *fields):
    h = hashlib.blake2b(digest_size=32)
    h.update(struct.pack("<Q", len(domain)))
    h.update(domain)
    for f in fields:
        h.update(struct.pack("<Q", f) if type(f) is int else f)
    return h.digest()


def genesis_state(setup_cert_id, layout_fingerprint):
    """St_0.  `setup_cert_id` is `SetupCertificate.id` (hex) or its 32 raw bytes."""
    cert = bytes.fromhex(setup_cert_id) if type(setup_cert_id) is str else setup_cert_id
    return _h(GENESIS_DOMAIN, _b32(cert, "setup_cert_id"), _b32(layout_fingerprint, "layout_fingerprint"))


def turn_state(prev_state, turn_idx, prefix_tokens, fresh_tokens, total_tokens, cached_prefix_root,
               fresh_pages_root, response_digest):
    return _h(TURN_DOMAIN, _b32(prev_state, "prev_state"), _u64(turn_idx, "turn_idx"),
              _u64(prefix_tokens, "prefix_tokens"), _u64(fresh_tokens, "fresh_tokens"),
              _u64(total_tokens, "total_tokens"), _b32(cached_prefix_root, "cached_prefix_root"),
              _b32(fresh_pages_root, "fresh_pages_root"), _b32(response_digest, "response_digest"))


def response_digest(response_tokens):
    toks = list(response_tokens)
    for t in toks:
        if type(t) is not int:
            raise TypeError(f"response token must be an exact int, got {type(t).__name__}")
        if not 0 <= t < (1 << 32):
            raise ValueError(f"response token {t} out of u32 range")
    return _h(RESPONSE_DOMAIN, len(toks), struct.pack(f"<{len(toks)}I", *toks))


def page_hash(first_position, n_tokens, payload):
    """Ledger leaf of one captured page: bound to its first token position, token count and payload bytes."""
    return _h(PAGE_DOMAIN, _u64(first_position, "first_position"), _u64(n_tokens, "n_tokens"), len(payload),
              bytes(payload))


# ----------------------------------------------------------------------------- append-only page ledger
def _node(left, right):
    return _h(NODE_DOMAIN, left, right)


class PageLedger:
    """Immutable append-only Merkle tree of page hashes (RFC 6962 shape).  levels[l][i] is the root of the complete
    subtree of 2^l pages starting at page i * 2^l, so the root over ANY prefix of k pages is a bag of O(log k)
    already-computed nodes.  `extended` / `truncated` return new ledgers that share every 32-byte digest."""

    __slots__ = ("levels",)

    def __init__(self, levels=((),)):
        self.levels = levels

    @property
    def n_pages(self):
        return len(self.levels[0])

    def truncated(self, k):
        if not 0 <= k <= self.n_pages:
            raise ValueError("prefix longer than the ledger")
        top = max(1, k.bit_length())
        return PageLedger(tuple(self.levels[l][:k >> l] if l < len(self.levels) else () for l in range(top)))

    def extended(self, leaves):
        levels = [list(l) for l in self.levels]
        for leaf in leaves:
            _b32(leaf, "page hash")
            levels[0].append(leaf)
            l = 0
            while len(levels[l]) % 2 == 0:
                if l + 1 == len(levels):
                    levels.append([])
                levels[l + 1].append(_node(levels[l][-2], levels[l][-1]))
                l += 1
        return PageLedger(tuple(tuple(l) for l in levels))

    def tree_root(self, k):
        if not 0 <= k <= self.n_pages:
            raise ValueError("prefix longer than the ledger")
        if k == 0:
            return _h(EMPTY_DOMAIN)
        peaks, pos = [], 0
        for l in range(k.bit_length() - 1, -1, -1):
            if k >> l & 1:
                peaks.append(self.levels[l][pos >> l])
                pos += 1 << l
        acc = peaks[-1]
        for p in reversed(peaks[:-1]):
            acc = _node(p, acc)
        return acc

    def digests(self):
        return [d for level in self.levels for d in level]


def fresh_pages_root(prefix_tokens, fresh_tokens, fresh_page_hashes):
    """Commitment to the fresh pages of a turn, bound to their position, token count and page count."""
    tree = PageLedger().extended(fresh_page_hashes).tree_root(len(fresh_page_hashes))
    return _h(FRESH_ROOT_DOMAIN, _u64(prefix_tokens, "prefix_tokens"), _u64(fresh_tokens, "fresh_tokens"),
              len(fresh_page_hashes), tree)


def cached_prefix_root(ledger, prefix_tokens, page_tokens=PAGE_TOKENS):
    """Commitment to the first `prefix_tokens` tokens (page aligned) of a verified context."""
    if prefix_tokens % page_tokens:
        raise ValueError("prefix is not page aligned")
    k = prefix_tokens // page_tokens
    return _h(PREFIX_ROOT_DOMAIN, _u64(prefix_tokens, "prefix_tokens"), k, ledger.tree_root(k))


# ----------------------------------------------------------------------------- public per-turn record
@dataclass(frozen=True)
class AgenticTurn:
    turn_idx: int
    prefix_tokens: int
    fresh_tokens: int
    total_tokens: int
    cached_prefix_root: bytes
    fresh_pages_root: bytes
    response_digest: bytes
    prev_state: bytes
    state: bytes

    SIZE = 4 * 8 + 5 * 32 + len(RECORD_MAGIC)

    def __post_init__(self):
        for f in ("turn_idx", "prefix_tokens", "fresh_tokens", "total_tokens"):
            _u64(getattr(self, f), f)
        for f in ("cached_prefix_root", "fresh_pages_root", "response_digest", "prev_state", "state"):
            _b32(getattr(self, f), f)

    def recompute_state(self):
        return turn_state(self.prev_state, self.turn_idx, self.prefix_tokens, self.fresh_tokens, self.total_tokens,
                          self.cached_prefix_root, self.fresh_pages_root, self.response_digest)

    def encode(self):
        return (RECORD_MAGIC + struct.pack("<4Q", self.turn_idx, self.prefix_tokens, self.fresh_tokens,
                                           self.total_tokens)
                + self.cached_prefix_root + self.fresh_pages_root + self.response_digest + self.prev_state
                + self.state)

    @classmethod
    def decode(cls, data):
        data = bytes(data)
        if len(data) != cls.SIZE or not data.startswith(RECORD_MAGIC):
            raise ValueError("invalid agentic turn record")
        m = len(RECORD_MAGIC)
        ints = struct.unpack_from("<4Q", data, m)
        roots = [data[m + 32 + 32 * i:m + 64 + 32 * i] for i in range(5)]
        return cls(*ints, *roots)


@dataclass(frozen=True)
class TurnCommitment:
    """What a published turn's C_pre was computed from (all public once the reply is published)."""
    trace_payload_root: bytes
    response_tokens: tuple
    policy_bytes: bytes
    c_pre: bytes


def agentic_policy_bytes(turn, policy_bytes=b""):
    return CPRE_DOMAIN + turn.state + struct.pack("<Q", len(policy_bytes)) + bytes(policy_bytes)


def precommit_agentic_turn(setup_cert_bytes, turn, trace_payload_root, response_tokens, policy_bytes=b""):
    """Per-turn C_pre with `St_i` bound in (as part of the policy bytes of `pipeline.precommit_request`)."""
    if type(turn) is not AgenticTurn:
        raise TypeError("turn must be an AgenticTurn")
    return precommit_request(setup_cert_bytes, trace_payload_root, response_tokens,
                             agentic_policy_bytes(turn, policy_bytes))


# ----------------------------------------------------------------------------- incremental verifier
@dataclass(frozen=True)
class _Node:
    turn_idx: int
    total_tokens: int
    ledger: object          # PageLedger of the context AFTER this turn, or None (light mode)
    parent_state: object = None


def _verdict(code, detail="", turn=None, tip=None):
    return {"verdict": code, "detail": detail, "turn_idx": None if turn is None else turn.turn_idx,
            "tip_state": tip}


class AgenticChainVerifier:
    """Stateful, incremental verifier of one chain (branch).  `accept` is atomic: a rejected record changes
    nothing.  `track_pages=True` additionally keeps the page ledger and checks `E_FRESH_ROOT` / `E_PREFIX_ROOT`."""

    def __init__(self, setup_cert_bytes, layout_fingerprint, *, page_tokens=PAGE_TOKENS, track_pages=False):
        if type(page_tokens) is not int or page_tokens < 1:
            raise ValueError("page_tokens must be a positive int")
        self.setup_cert_bytes = bytes(setup_cert_bytes)
        self.layout_fingerprint = _b32(layout_fingerprint, "layout_fingerprint")
        self.page_tokens, self.track_pages = page_tokens, track_pages
        self.genesis = genesis_state(SetupCertificate.decode(self.setup_cert_bytes).id, self.layout_fingerprint)
        self._nodes = {self.genesis: _Node(0, 0, PageLedger() if track_pages else None, None)}
        self.tip = self.genesis

    # -- helpers
    @property
    def tip_node(self):
        return self._nodes[self.tip]

    def known(self, state):
        return state in self._nodes

    def fork(self, state):
        """New branch rooted at verified `state` with an isolated ancestor-only node map."""
        if state not in self._nodes:
            raise AgenticChainError(E_STATE_CHAIN, "cannot fork from an unverified state")
        ancestors = {}
        cur = state
        while cur is not None:
            node = self._nodes[cur]
            ancestors[cur] = node
            cur = node.parent_state
        other = object.__new__(AgenticChainVerifier)
        other.__dict__.update(self.__dict__)
        other._nodes = ancestors
        other.tip = state
        return other

    def expected_cached_prefix_root(self, prefix_tokens):
        node = self.tip_node
        if node.ledger is None:
            raise ValueError("verifier is not tracking pages")
        return cached_prefix_root(node.ledger, prefix_tokens, self.page_tokens)

    # -- the check
    def accept(self, turn, *, fresh_page_hashes=None, commitment=None):
        t = turn
        if type(t) is not AgenticTurn:
            raise TypeError("turn must be an AgenticTurn")
        parent_state = self.tip
        parent = self._nodes[parent_state]
        if t.prev_state != parent_state:
            if parent_state == self.genesis:
                return _verdict(E_GENESIS, "first record is not chained to St_0 of this setup / layout", t, self.tip)
            if t.prev_state in self._nodes:
                return _verdict(E_TURN_ORDER, "rollback / equivocation: parent is not the chain tip (use fork())",
                                t, self.tip)
            return _verdict(E_GENESIS if t.turn_idx == 1 else E_STATE_CHAIN,
                            "prev_state is not a verified state", t, self.tip)
        if t.turn_idx != parent.turn_idx + 1:
            return _verdict(E_TURN_ORDER, f"turn_idx {t.turn_idx} != {parent.turn_idx} + 1", t, self.tip)
        if t.prefix_tokens > parent.total_tokens:
            return _verdict(E_PREFIX_EXCEEDS_PREV, f"prefix {t.prefix_tokens} > previous total "
                            f"{parent.total_tokens}", t, self.tip)
        if t.total_tokens != t.prefix_tokens + t.fresh_tokens or t.fresh_tokens < 1:
            return _verdict(E_TOKEN_COUNT, "total != prefix + fresh (or no fresh token)", t, self.tip)
        if t.prefix_tokens % self.page_tokens:
            return _verdict(E_TOKEN_COUNT, "prefix is not page aligned", t, self.tip)
        if t.recompute_state() != t.state:
            return _verdict(E_STATE_CHAIN, "St_i does not recompute from the record", t, self.tip)
        ledger = None
        if self.track_pages:
            n_fresh = -(-t.fresh_tokens // self.page_tokens)
            if (fresh_page_hashes is None or len(fresh_page_hashes) != n_fresh
                    or any(type(x) is not bytes or len(x) != 32 for x in fresh_page_hashes)
                    or fresh_pages_root(t.prefix_tokens, t.fresh_tokens, fresh_page_hashes) != t.fresh_pages_root):
                return _verdict(E_FRESH_ROOT, "fresh_pages_root does not recompute from the fresh page hashes",
                                t, self.tip)
            if cached_prefix_root(parent.ledger, t.prefix_tokens, self.page_tokens) != t.cached_prefix_root:
                return _verdict(E_PREFIX_ROOT, "cached_prefix_root is not the parent context's prefix root",
                                t, self.tip)
            ledger = parent.ledger.truncated(t.prefix_tokens // self.page_tokens).extended(fresh_page_hashes)
        if commitment is not None:
            ok = (type(commitment) is TurnCommitment
                  and response_digest(commitment.response_tokens) == t.response_digest
                  and precommit_agentic_turn(self.setup_cert_bytes, t, commitment.trace_payload_root,
                                             commitment.response_tokens, commitment.policy_bytes) == commitment.c_pre)
            if not ok:
                return _verdict(E_CPRE_MISMATCH, "C_pre does not recompute from (St_i, trace root, response, policy)",
                                t, self.tip)
        self._nodes[t.state] = _Node(t.turn_idx, t.total_tokens, ledger, parent_state)
        self.tip = t.state
        return _verdict(VERIFIED, f"turn {t.turn_idx}", t, self.tip)

    def retained_digests(self):
        """Number of DISTINCT 32-byte digests held by the ledgers of every verified state (shared ones once)."""
        seen = set()
        for node in self._nodes.values():
            if node.ledger is not None:
                seen.update(id(d) for d in node.ledger.digests())
        return len(seen)


def verify_agentic_chain(setup_cert_bytes, layout_fingerprint, turns, *, fresh_pages=None, commitments=None,
                         page_tokens=PAGE_TOKENS):
    """Verify a linear chain of records from St_0.  `fresh_pages[i]` (page hashes of turn i) switches on the
    ledger checks; `commitments[i]` (`TurnCommitment`) checks the published C_pre.  Returns the verdict of the first
    failing record, or {"verdict": "VERIFIED", "tip_state": St_n, "turns": n}."""
    turns = list(turns)
    v = AgenticChainVerifier(setup_cert_bytes, layout_fingerprint, page_tokens=page_tokens,
                             track_pages=fresh_pages is not None)
    if not turns:
        return {"verdict": E_GENESIS, "detail": "empty chain", "turns": 0, "tip_state": v.tip}
    if fresh_pages is not None:
        fresh_pages = list(fresh_pages)
        if len(fresh_pages) != len(turns):
            return _verdict(E_FRESH_ROOT, f"fresh_pages length {len(fresh_pages)} != {len(turns)}", None, v.tip)
    if commitments is not None:
        commitments = list(commitments)
        if len(commitments) != len(turns):
            return _verdict(E_CPRE_MISMATCH, f"commitments length {len(commitments)} != {len(turns)}", None, v.tip)
    out = _verdict(E_GENESIS, "empty chain", None, v.tip)
    for i, t in enumerate(turns):
        if commitments is not None and type(commitments[i]) is not TurnCommitment:
            return _verdict(E_CPRE_MISMATCH, f"commitment[{i}] missing or not a TurnCommitment", t, v.tip)
        out = v.accept(t, fresh_page_hashes=None if fresh_pages is None else fresh_pages[i],
                       commitment=None if commitments is None else commitments[i])
        if out["verdict"] != VERIFIED:
            return out
    return {"verdict": out["verdict"], "detail": out["detail"], "turns": len(turns), "tip_state": v.tip}


# ----------------------------------------------------------------------------- prover-side session
class AgenticSession:
    """Prover-side builder: every emitted record has already passed an internal `AgenticChainVerifier`."""

    def __init__(self, setup_cert_bytes, layout_fingerprint, *, page_tokens=PAGE_TOKENS, _verifier=None):
        self.verifier = _verifier or AgenticChainVerifier(setup_cert_bytes, layout_fingerprint,
                                                          page_tokens=page_tokens, track_pages=True)
        self.turns = []              # records of THIS branch, oldest first
        self.turn_seconds = []

    @property
    def tip(self):
        return self.verifier.tip

    @property
    def page_tokens(self):
        return self.verifier.page_tokens

    def begin_turn(self, prefix_tokens, fresh_tokens, fresh_page_hashes, response_tokens):
        t0 = time.perf_counter()
        v = self.verifier
        parent = v.tip_node
        fresh_page_hashes = tuple(fresh_page_hashes)
        if prefix_tokens > parent.total_tokens or prefix_tokens % v.page_tokens:
            cached = bytes(32)          # let accept() name the exact error
        else:
            cached = v.expected_cached_prefix_root(prefix_tokens)
        fresh_root = fresh_pages_root(prefix_tokens, fresh_tokens, fresh_page_hashes)
        total = prefix_tokens + fresh_tokens
        idx = parent.turn_idx + 1
        resp = response_digest(response_tokens)
        state = turn_state(v.tip, idx, prefix_tokens, fresh_tokens, total, cached, fresh_root, resp)
        turn = AgenticTurn(idx, prefix_tokens, fresh_tokens, total, cached, fresh_root, resp, v.tip, state)
        verdict = v.accept(turn, fresh_page_hashes=fresh_page_hashes)
        if verdict["verdict"] != VERIFIED:
            raise AgenticChainError(verdict["verdict"], verdict["detail"])
        self.turns.append(turn)
        self.turn_seconds.append(time.perf_counter() - t0)
        return turn

    def fork(self, state):
        """Branch from genesis or a verified turn state on this session's recorded branch."""
        known = {t.state: i for i, t in enumerate(self.turns)}
        if state != self.verifier.genesis and state not in known:
            raise AgenticChainError(E_STATE_CHAIN, "cannot fork from a state not on this session branch")
        branch = AgenticSession(None, None, _verifier=self.verifier.fork(state))
        branch.turns = list(self.turns[:known[state] + 1]) if state in known else []
        return branch

    def precommit(self, turn, trace_payload_root, response_tokens, policy_bytes=b""):
        return precommit_agentic_turn(self.verifier.setup_cert_bytes, turn, trace_payload_root, response_tokens,
                                      policy_bytes)


def session_accounting(turns, row_bytes, page_tokens=PAGE_TOKENS):
    """Capture / retention accounting of one chain.  `naive`: every turn re-captures its whole context;
    `deduped`: prefix pages are written once (only fresh tokens are captured).  `ledger_bytes_logical`: 32 bytes per
    page hash + per internal node of the final transcript; `record_bytes`: the public chain (192 B per turn)."""
    naive = sum(t.total_tokens for t in turns) * row_bytes
    deduped = sum(t.fresh_tokens for t in turns) * row_bytes
    final_pages = -(-turns[-1].total_tokens // page_tokens) if turns else 0
    return {"turns": len(turns), "naive_capture_bytes": naive, "deduped_capture_bytes": deduped,
            "dedup_ratio": naive / deduped if deduped else None,
            "ledger_bytes_logical": 32 * (2 * final_pages), "record_bytes": AgenticTurn.SIZE * len(turns)}
