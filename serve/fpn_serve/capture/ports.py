# SPDX-License-Identifier: Apache-2.0
"""Capture ports, tiers and the canonical ``(page, layer, 128-col chunk)`` layout (stdlib only; no torch, no prover).

Everything here is *data and integer math*: it decides which tensors a capture tier records, where each lands in a
ring row, how many exact bytes per token that costs, and how a physical shard (TP stripe / EP expert / PP stage) maps
back to the one logical tensor without leaking the deployment topology.

Units: widths and offsets are counted in ``u16`` elements (bf16 / fp16 / bit-patterns).  An ``fp32`` LSE costs 2 u16 per
head.  A ring *row* is one token across every captured layer.

Tiers
-----
``FULL_NO_KV``  the full per-layer port set minus K/V (K/V stay in vLLM's paged KV cache and are *referenced*, never
                copied): ``h_in, x_attn, q, o, h_mid, x_mlp, a, lse`` per layer (+ ``expert_ids`` for MoE) and
                ``x_f, h_L`` once.  ``capture_norm_outputs=False`` drops ``x_attn / x_mlp / x_f`` -- allowed only once
                gate M0-G1 (bitwise RMSNorm recompute) passes, hence a flag and not a constant.
``FULL_NO_KV_ACT`` captures h_in, h_mid, Q, O, LSE, h_L and the exact down-projection input a.
                Proof consumers must reproduce the omitted RMSNorm outputs at the serving batch shape
                and establish that the captured a follows from gate/up projection and activation.

``TIER_B``      residual boundaries only: ``h_in`` per layer, optionally ``h_mid``, plus ``h_L``, MoE ``expert_ids`` and
                a KV *reference* (page-level block hash, ``KV_REF_BYTES_PER_PAGE`` per 16 tokens, not row payload).

Bytes/token reconciliation (design memo 03 §1): the memo's Tier-B headline numbers (189 KB 4B, 201 KB 30B-A3B,
1.33 MB 70B, 502 KB 400B) equal ``(L+1) * d * 2`` -- one residual per layer plus ``h_L``.  The port list in the memo
text (``h_in / h_mid / h_L``) would double that.  ``TIER_B`` here is data-driven: ``include_h_mid=True`` is the
as-specified port list, ``include_h_mid=False`` reproduces the memo's numbers exactly; the test pins both.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional

__all__ = [
    "CHUNK_COLS", "ALIGN_U16", "PAGE_TOKENS", "KV_REF_BYTES_PER_PAGE", "U16_BYTES",
    "CaptureTier", "CaptureOptions", "MoEDims", "ModelDims", "PortSlot", "LayerGroup", "CaptureLayout",
    "ChunkAddress", "chunk_count", "chunk_span", "stripe_chunks", "merge_row_stripes", "reference_full_bytes_per_token",
    "CaptureError", "StripeCoverageError", "LayoutError",
    "QWEN3_0_6B", "QWEN3_4B", "QWEN3_30B_A3B", "LLAMA3_70B", "ASSUMED_400B_MOE",
]

CHUNK_COLS = 128            # canonical chunk width, in u16 columns (256 B)
ALIGN_U16 = 8               # 16-byte alignment of every slot (uint4 vector stores)
PAGE_TOKENS = 16            # block size of vLLM's prefix cache and of the page store
KV_REF_BYTES_PER_PAGE = 32  # one BLAKE2b-256 block hash per page references the KV block (no K/V bytes are copied)
U16_BYTES = 2


class CaptureError(RuntimeError):
    """Base class of every fail-closed capture error."""


class LayoutError(CaptureError, ValueError):
    pass


class StripeCoverageError(CaptureError):
    """Shard stripes overlap or leave a gap: refusing to assemble (never zero-fill the hole)."""


class CaptureTier(Enum):
    FULL_NO_KV = "full_no_kv"
    FULL_NO_KV_ACT = "full_no_kv_act"
    TIER_B = "tier_b"


@dataclass(frozen=True)
class MoEDims:
    n_experts: int
    top_k: int
    d_expert: int
    n_shared: int = 0


@dataclass(frozen=True)
class ModelDims:
    n_layers: int
    d_model: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    moe: Optional[MoEDims] = None
    name: str = ""

    def __post_init__(self) -> None:
        if min(self.n_layers, self.d_model, self.n_q_heads, self.n_kv_heads, self.head_dim) < 1:
            raise LayoutError("model dims must be positive")
        if self.moe is not None and not 1 <= self.moe.n_experts <= 0xFFFF:
            raise LayoutError("expert ids are captured as u16: need 1 <= n_experts <= 65535")

    @property
    def act_width(self) -> int:
        """Width of the down-proj input ``a`` (active experts concatenated for MoE)."""
        if self.moe is not None:
            return (self.moe.top_k + self.moe.n_shared) * self.moe.d_expert
        return self.d_ff

    @classmethod
    def from_hf_config(cls, cfg: Mapping[str, object], name: str = "") -> "ModelDims":
        g = cfg.get
        nq = int(g("num_attention_heads"))  # type: ignore[arg-type]
        d = int(g("hidden_size"))  # type: ignore[arg-type]
        n_exp = g("num_experts") or g("num_local_experts") or g("n_routed_experts")
        moe = None
        if n_exp:
            moe = MoEDims(int(n_exp), int(g("num_experts_per_tok") or g("top_k") or 1),  # type: ignore[arg-type]
                          int(g("moe_intermediate_size") or g("intermediate_size")),  # type: ignore[arg-type]
                          int(g("n_shared_experts") or 0))  # type: ignore[arg-type]
        return cls(int(g("num_hidden_layers")), d, nq, int(g("num_key_value_heads") or nq),  # type: ignore[arg-type]
                   int(g("head_dim") or d // nq), int(g("intermediate_size") or 0), moe, name)  # type: ignore[arg-type]


# Reference dimensions (public configs).  ASSUMED_400B_MOE is an *assumed* Llama-4-Maverick-like shape used only to
# reproduce the memo's 502 KB figure; it is not read from any checkpoint.
QWEN3_0_6B = ModelDims(28, 1024, 16, 8, 128, 3072, name="Qwen3-0.6B")
QWEN3_4B = ModelDims(36, 2560, 32, 8, 128, 9728, name="Qwen3-4B")
QWEN3_30B_A3B = ModelDims(48, 2048, 32, 4, 128, 0, MoEDims(128, 8, 768), name="Qwen3-30B-A3B")
LLAMA3_70B = ModelDims(80, 8192, 64, 8, 128, 28672, name="Llama-3-70B")
ASSUMED_400B_MOE = ModelDims(48, 5120, 40, 8, 128, 0, MoEDims(128, 1, 8192, 1), name="assumed-400B-MoE")


@dataclass(frozen=True)
class CaptureOptions:
    tier: CaptureTier = CaptureTier.FULL_NO_KV
    include_h_mid: bool = True          # TIER_B only
    capture_norm_outputs: bool = True   # FULL_NO_KV only: x_attn / x_mlp / x_f (drop after gate M0-G1)
    # FULL_NO_KV only: the down-projection input ``a``.  A fused MoE kernel (vLLM FusedMoE) gate/up-projects,
    # activates and down-projects the routed experts in one kernel, so no ``a`` boundary exists to tap; such a model
    # is captured with ``capture_act=False`` (the slot is dropped from the layout, never left unwritten).
    capture_act: bool = True


@dataclass(frozen=True)
class PortSlot:
    kind: str
    layer: int          # 0..L-1 for layer ports; L for the final row (h_L / x_f)
    offset: int         # u16 offset inside the ring row
    width: int          # payload u16 columns


@dataclass(frozen=True)
class LayerGroup:
    layer_lo: int       # inclusive; layer index L is the final pseudo-layer row
    layer_hi: int       # exclusive
    col_lo: int         # u16 column range inside the ring row (padding included, so groups tile the row)
    col_hi: int


@dataclass(frozen=True, order=True)
class ChunkAddress:
    page: int
    layer: int
    chunk: int


def _align(n: int, a: int = ALIGN_U16) -> int:
    return (n + a - 1) // a * a


def chunk_count(width: int) -> int:
    return -(-width // CHUNK_COLS)


def chunk_span(width: int, chunk: int) -> tuple[int, int]:
    """Column range ``[lo, hi)`` of canonical chunk ``chunk`` of a ``width``-column layer row (last one is short)."""
    if not 0 <= chunk < chunk_count(width):
        raise LayoutError(f"chunk {chunk} out of range for width {width}")
    return chunk * CHUNK_COLS, min(width, (chunk + 1) * CHUNK_COLS)


def stripe_chunks(width: int, col_lo: int, col_hi: int) -> list[tuple[int, int, int]]:
    """Split a shard's column stripe ``[col_lo, col_hi)`` of a layer row at canonical chunk boundaries.

    Returns ``[(chunk, lo, hi)]`` (absolute columns).  The decomposition depends only on the columns, never on how many
    shards there are, so TP=1, 2, 4, 8 writers all address the same logical chunks."""
    if not 0 <= col_lo < col_hi <= width:
        raise LayoutError(f"bad stripe [{col_lo},{col_hi}) for width {width}")
    out = []
    for c in range(col_lo // CHUNK_COLS, -(-col_hi // CHUNK_COLS)):
        lo, hi = chunk_span(width, c)
        out.append((c, max(lo, col_lo), min(hi, col_hi)))
    return out


def merge_row_stripes(width: int, pieces: Iterable[tuple[int, bytes]]) -> bytes:
    """Assemble one layer-row from shard pieces ``(col_lo, u16-bytes)``.  Requires an exact, gap-free, overlap-free
    cover of ``[0, width)``; anything else raises :class:`StripeCoverageError` (never zero-fills a missing stripe)."""
    ps = sorted(((lo, bytes(b)) for lo, b in pieces), key=lambda t: t[0])
    cur = 0
    out = bytearray()
    for lo, b in ps:
        if len(b) % U16_BYTES:
            raise StripeCoverageError("stripe bytes not a multiple of 2")
        if lo != cur:
            raise StripeCoverageError(f"stripe gap/overlap at column {cur}: next stripe starts at {lo}")
        out += b
        cur = lo + len(b) // U16_BYTES
    if cur != width:
        raise StripeCoverageError(f"stripes cover {cur} of {width} columns")
    return bytes(out)


@dataclass(frozen=True)
class CaptureLayout:
    """Exact ring-row layout for a model + capture options."""

    dims: ModelDims
    options: CaptureOptions
    slots: tuple[PortSlot, ...]
    row_u16: int                    # aligned ring-row stride, in u16
    payload_u16: int                # sum of slot widths (no padding)
    layer_base: tuple[int, ...]     # per layer row (length L+1), offset in u16
    layer_cols: tuple[int, ...]     # per layer row, payload columns incl. inner slot alignment padding
    _index: Mapping[tuple[str, int], PortSlot] = field(repr=False, compare=False, default_factory=dict)

    # ------------------------------------------------------------------------------------------------ construction
    @classmethod
    def build(cls, dims: ModelDims, options: Optional[CaptureOptions] = None) -> "CaptureLayout":
        opt = options or CaptureOptions()
        D, nq, dh = dims.d_model, dims.n_q_heads, dims.head_dim
        moe_ids = dims.moe.top_k if dims.moe else 0
        if opt.tier in (CaptureTier.FULL_NO_KV, CaptureTier.FULL_NO_KV_ACT):
            nrm = opt.tier is CaptureTier.FULL_NO_KV and opt.capture_norm_outputs
            per_layer = [("h_in", D)] + ([("x_attn", D)] if nrm else []) + [("q", nq * dh), ("o", nq * dh), ("h_mid", D)]
            if not opt.capture_act and opt.tier is CaptureTier.FULL_NO_KV_ACT:
                raise LayoutError("FULL_NO_KV_ACT is defined by the captured down-projection input a")
            per_layer += ([("x_mlp", D)] if nrm else []) + ([("a", dims.act_width)] if opt.capture_act else [])
            per_layer += [("lse", 2 * nq)]
            final = ([("x_f", D)] if nrm else []) + [("h_L", D)]
        else:
            per_layer = [("h_in", D)] + ([("h_mid", D)] if opt.include_h_mid else [])
            final = [("h_L", D)]
        if moe_ids:
            per_layer = per_layer + [("expert_ids", moe_ids)]
        slots: list[PortSlot] = []
        bases: list[int] = []
        cols: list[int] = []
        off = 0
        for layer in range(dims.n_layers + 1):
            group = per_layer if layer < dims.n_layers else final
            bases.append(off)
            start = off
            for kind, w in group:
                off = _align(off)
                slots.append(PortSlot(kind, layer, off, w))
                off += w
            cols.append(off - start)
        row = _align(off)
        payload = sum(s.width for s in slots)
        idx = {(s.kind, s.layer): s for s in slots}
        return cls(dims, opt, tuple(slots), row, payload, tuple(bases), tuple(cols), idx)

    # ------------------------------------------------------------------------------------------------ bytes / token
    @property
    def tier(self) -> CaptureTier:
        return self.options.tier

    @property
    def payload_bytes_per_token(self) -> int:
        """Exact bytes of captured tensor data per token (no alignment padding, no KV reference)."""
        return self.payload_u16 * U16_BYTES

    @property
    def stride_bytes(self) -> int:
        """Ring-row stride in bytes (payload + 16-byte slot alignment padding)."""
        return self.row_u16 * U16_BYTES

    @property
    def kv_ref_bytes_per_token(self) -> float:
        return KV_REF_BYTES_PER_PAGE / PAGE_TOKENS if self.tier is CaptureTier.TIER_B else 0.0

    def breakdown(self) -> dict[str, int]:
        """Payload bytes per token by port kind."""
        out: dict[str, int] = {}
        for s in self.slots:
            out[s.kind] = out.get(s.kind, 0) + s.width * U16_BYTES
        return out

    def ring_bytes(self, capacity_rows: int) -> int:
        return capacity_rows * self.stride_bytes

    def slot(self, kind: str, layer: int) -> PortSlot:
        try:
            return self._index[(kind, layer)]
        except KeyError:
            raise LayoutError(f"no port {kind!r} at layer {layer} in tier {self.tier.value}") from None

    def has(self, kind: str, layer: int) -> bool:
        return (kind, layer) in self._index

    # ------------------------------------------------------------------------------------------------ canonical layout
    @property
    def n_layer_rows(self) -> int:
        return self.dims.n_layers + 1

    def layer_chunks(self, layer: int) -> int:
        return chunk_count(self.layer_cols[layer])

    @property
    def chunks_per_page(self) -> int:
        return sum(self.layer_chunks(l) for l in range(self.n_layer_rows))

    def chunk_linear_index(self, a: ChunkAddress) -> int:
        """Topology-independent linear index of ``(page, layer, chunk)``: ``page``-major, then layer, then chunk."""
        if not 0 <= a.layer < self.n_layer_rows or not 0 <= a.chunk < self.layer_chunks(a.layer) or a.page < 0:
            raise LayoutError(f"bad chunk address {a}")
        return a.page * self.chunks_per_page + sum(self.layer_chunks(l) for l in range(a.layer)) + a.chunk

    def chunk_columns(self, layer: int, chunk: int) -> tuple[int, int]:
        """Absolute u16 column range inside the ring row of canonical chunk ``chunk`` of layer row ``layer``."""
        lo, hi = chunk_span(self.layer_cols[layer], chunk)
        return self.layer_base[layer] + lo, self.layer_base[layer] + hi

    def layer_groups(self, group_size: int = 0) -> list[LayerGroup]:
        """Partition the ring row into contiguous layer groups (``0`` = one group) that tile ``[0, row_u16)``."""
        n = self.n_layer_rows
        step = n if group_size <= 0 else group_size
        out = []
        for lo in range(0, n, step):
            hi = min(n, lo + step)
            c0 = 0 if lo == 0 else self.layer_base[lo]
            c1 = self.row_u16 if hi == n else self.layer_base[hi]
            out.append(LayerGroup(lo, hi, c0, c1))
        return out

    def fingerprint(self) -> bytes:
        """32-byte identity of tier + dims + slot geometry; seeds the page hash chain so layouts never collide."""
        h = hashlib.blake2b(digest_size=32, person=b"fpn-capture-lay")
        h.update(repr((self.tier.value, self.options.include_h_mid, self.options.capture_norm_outputs, self.dims,
                       self.row_u16, [(s.kind, s.layer, s.offset, s.width) for s in self.slots])).encode())
        return h.digest()


def reference_full_bytes_per_token(dims: ModelDims, *, with_kv: bool = True, capture_norm_outputs: bool = True) -> int:
    """Payload bytes/token of the *old* full-port prototype (K/V and norm outputs included) for ratio reporting."""
    D, nq, nkv, dh = dims.d_model, dims.n_q_heads, dims.n_kv_heads, dims.head_dim
    per = 2 * D + nq * dh * 2 + dims.act_width + 2 * nq + (2 * D if capture_norm_outputs else 0)
    if with_kv:
        per += 2 * nkv * dh
    if dims.moe:
        per += dims.moe.top_k
    tail = D * (2 if capture_norm_outputs else 1)
    return (dims.n_layers * per + tail) * U16_BYTES
