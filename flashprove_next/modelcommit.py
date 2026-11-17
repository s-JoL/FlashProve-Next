# SPDX-License-Identifier: Apache-2.0
"""Model commitment (specs/09): static Reed-Solomon interleaved weight commitment + hidden column opening.

Commitment (specs/09 §2), done once per checkpoint (layout v2, `fpn-modelcom-2`):
  * every 2-D weight tensor (out, in) is cut into public-shape tiles T_O x T_I; tile (R, C) is ONE row of the matrix
    W in F_p^{n_r x n_c}, n_c = T_O T_I, addressed (output row r, input column c) -> k = r T_I + c.  1-D tensors
    (norm gains) are concatenated into "vec" rows.  n_r = R_MAX (profile); rows past the used ones are zero.
  * META rows at fixed public positions: row 0 = SCALE (SCALE[i] = exponent G_i of row i, i < R_MAX), rows 1.. = PN
    (per-tile-row squared norms: PN[i T_O + r] = sum_c W^cd[i, r T_I + c]^2, the norms specs/06 needs; ceil(R_MAX T_O /
    n_c) rows, one for the Qwen profile; meta rows hold 0 there).  Data rows follow (vec rows first, then tiles).  Row R_MAX - 1 stays zero (padding row of flashprove_next.mc_zk).
  * W^cd = CD(W^raw): canonical decode of the deployed bf16 codes to integers |w| < 2^22 with ONE exponent per row of
    W (tile-granular block floating point; the vec row shares one): G_i = the largest G with max |sigma| 2^(E+G) < 2^22
    over the row (cd_exponent), G_i = G_LO for an all-zero row; G_i must lie in [G_LO, G_LO + 16) (setup table range).
    W^raw = the 16-bit codes themselves (meta rows zero), same row indices.
  * each row is RS-encoded at rate 1/4 (n = 4 n_c evaluations at the n-th roots of unity, natural order); column j of
    the codeword matrix M gets a 256-bit salt s_j and the leaf H_col(j, s_j, R_MAX, M[:, j]) (Poseidon2 sponge, domain
    DOM_COL, 1 + R_MAX / 8 permutations); C_W is the root of the depth-log2(n) Poseidon2 tree over the leaves.
    C_raw is the same construction over W^raw; model_id = H(DOM_MID; s_0, C_W, C_raw, C_cfg).
    The one-time setup proof pi_setup (specs/09 §2.2) is flashprove_next.mc_setup.

Per-proof path (specs/09 §3.1), evaluated here in the clear as the reference for the algebra and the cost model
(the zero-knowledge version used by reply proofs is flashprove_next.mc_zk):
  1. claims: GEMM claims y = sum_C W[t_C] h_C (sampled blocks; the LM head over all vocabulary tiles with the K2
     steps batched by r_s) and lookup claims x = W[t][r0, :] (embedding rows); FS gives xi', zeta', lambda.
  2. ONE batched sumcheck over the 20 tile variables, output (row) variables first, shared by every term:
         sum_u sum_{r,c} g_u(r) W_{t_u}(r, c) h_u(c) = sum_k v_k
     ends at one point (xi, zeta) and yields one claim c_t = W~_t(xi, zeta) per DISTINCT touched tile t.
  3. gamma; alpha_{t_j} = gamma^j on touched rows only; u = alpha^T W^cd in E^{n_c}; check sum_j gamma^j c_j = u~(xi,zeta).
  4. u is committed (absorbed) BEFORE the m_q column positions Q are drawn; the prover opens M[:, j], s_j and the
     Merkle path for j in Q; checks: Merkle, and RS(u)[j] = <alpha, M[:, j]> (per column, and batched with beta as
     <u, g> with g~(z) = sum_j beta_j prod_b (1 - z_b + z_b w^{j 2^b}) evaluated in O(m_q log n_c)).

usage (CUDA GPU):
  python -m flashprove_next.modelcommit commit --model PATH/TO/Qwen3-0.6B --store DIR [--store-cw [--store-raw]]
  python -m flashprove_next.modelcommit prove  --model ... --store DIR --k1 78 --k2 32 [--out FILE]
"""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

from .field import P, CPU, ZERO, ONE, hadd, hsub, hmul, hpow, lagrange_eval, rand_field
from .hwinfo import describe as describe_machine
from . import poseidon2 as PS
from .transcript import Transcript

DOM_COL, DOM_CFG, DOM_MID = 21, 22, 23
VERSION = "fpn-modelcom-2"
META_SCALE, META_PN = 0, 1                 # fixed meta rows of W^cd (layout v2): SCALE, then PN row(s)
G_SPAN = 16                                # per-row exponents lie in [G_LO, G_LO + G_SPAN)
VOL_SYS0 = 32                              # first sys_prompt token in M^vol (aligned to 32)
P_MAX_CFG = 64                             # volatile config sponge blocks (64 * 8 = 512 field elements)
SENTINEL_BASE = 1 << 40                    # strictly-increasing sentinel key base for unused Tt / Vt rows

ROLE_IDS = {
    "embed_tokens": 0, "lm_head": 1,
    "q_proj": 2, "k_proj": 3, "v_proj": 4, "o_proj": 5,
    "gate_proj": 6, "up_proj": 7, "down_proj": 8,
    "norm": 16, "input_layernorm": 17, "post_attention_layernorm": 18, "q_norm": 19, "k_norm": 20,
    "embed_tokens_t": 21,          # transposed embedding E^T (d x V), committed for the in-circuit EMB row read
}


def tensor_role_layer_expert(name):
    """Parse canonical (role_id, layer_plus1, expert) from a checkpoint tensor name."""
    parts = name.split(".")
    layer_p1 = int(parts[parts.index("layers") + 1]) + 1 if "layers" in parts else 0
    expert = int(parts[parts.index("experts") + 1]) if "experts" in parts else 0
    role = 31
    for p in reversed(parts):
        if p in ROLE_IDS:
            role = ROLE_IDS[p]
            break
    if role == 31 and "embed" in name:
        role = 0
    return role, layer_p1, expert


def tile_key(role_id, layer_p1, expert, R, C):
    """Strictly-orderable 40-bit integer key for Tt: (role_id:5, layer_p1:9, expert:10, R:8, C:8)."""
    assert 0 <= role_id < 32 and 0 <= layer_p1 < 512 and 0 <= expert < 1024 and 0 <= R < 256 and 0 <= C < 256
    return (role_id << 35) | (layer_p1 << 26) | (expert << 16) | (R << 8) | C


def vec_key(role_id, layer_p1, chunk=0):
    """Strictly-orderable integer key for Vt: (role_id:6, layer_p1:10, chunk:8)."""
    assert 0 <= role_id < 64 and 0 <= layer_p1 < 1024 and 0 <= chunk < 256
    return (role_id << 18) | (layer_p1 << 8) | chunk


def cfg_sizes(prm):
    """Profile-derived table lengths (NT, NV, NL) fitting inside two configuration rows of width prm.nc >= 1024."""
    assert prm.nc >= 1024, f"cfg_rows requires nc >= 1024, got {prm.nc}"
    return min(4096, prm.nc // 8), min(512, prm.nc // 16), min(128, prm.nc // 16)


def tile_inexact(codes, G):
    """Boolean array [w_bf16 * 2^G not in Z] (exact formula of certified.cd_matrix at per-row exponent G)."""
    sig_t, E_t, _ = _bf_tables()
    c = np.asarray(codes, dtype=np.int64)
    k = E_t[c] + int(G)
    inexact = k < 0
    if inexact.any():
        sig = sig_t[c]
        sh = np.clip(-k, 0, 62)
        inexact = (sig != 0) & inexact & (((np.abs(sig) & ((np.int64(1) << sh) - 1)) != 0) | (sh > 20))
    return inexact


def mq_for(rate_log=2, delta_ca=0.3, bits=106):
    """specs/09: columns needed so that (rate + delta_ca)^m_q <= 2^-bits."""
    return math.ceil(bits / -math.log2(2.0 ** -rate_log + delta_ca))


@dataclass(frozen=True)
class Params:
    TO: int = 1024          # tile output rows
    TI: int = 1024          # tile input columns
    R_MAX: int = 1024       # rows of W (profile bound; unused rows are zero)
    RATE_LOG: int = 2       # rate 1/4
    MQ: int = 123           # opened columns per proof
    GROUP: int = 64         # rows encoded per device batch (multiple of 8 = sponge rate)
    G_LO: int = 12          # per-row CD exponents lie in [G_LO, G_LO + 16)

    @property
    def nc(self): return self.TO * self.TI

    @property
    def n(self): return self.nc << self.RATE_LOG

    @property
    def depth(self): return self.n.bit_length() - 1

    @property
    def ob(self): return self.TO.bit_length() - 1

    @property
    def ib(self): return self.TI.bit_length() - 1


QWEN_PARAMS = Params()
QWEN_PARAMS_R = Params(R_MAX=2048)     # with the radius matrix W^r (rows R_MAX/2 + t, wsum_zk / certified gadgets)


# ============================================================================ checkpoint access (bf16 safetensors)
class Checkpoint:
    """Minimal safetensors reader (memory-mapped bf16 codes as uint16; no torch)."""

    def __init__(self, path):
        f = os.path.join(path, "model.safetensors") if os.path.isdir(path) else path
        with open(f, "rb") as fh:
            hl = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(hl))
        self.meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
        self.mm = np.memmap(f, dtype=np.uint8, mode="r")
        self.base = 8 + hl
        cfgp = os.path.join(os.path.dirname(f), "config.json")
        self.config = json.load(open(cfgp)) if os.path.exists(cfgp) else {}

    def shapes(self):
        return {k: tuple(v["shape"]) for k, v in self.meta.items()}

    def codes(self, name):
        m = self.meta[name]
        assert m["dtype"] == "BF16", (name, m["dtype"])
        a, b = m["data_offsets"]
        return self.mm[self.base + a:self.base + b].view(np.uint16).reshape(m["shape"])

    def drop_cache(self):
        import mmap
        try:
            if hasattr(self.mm, "_mmap") and hasattr(mmap, "MADV_DONTNEED"):
                self.mm._mmap.madvise(mmap.MADV_DONTNEED)
        except (OSError, AttributeError, ValueError):
            pass


def model_shapes(ck):
    """Tensor shapes to commit.  A tied LM head stored as a byte-identical copy of the embedding (Qwen3-0.6B ships
    lm_head.weight == model.embed_tokens.weight with tie_word_embeddings = true) is committed once: the deployed model
    multiplies by the embedding; the alias is recorded in the config root."""
    sh = dict(ck.shapes())
    aliases = {}
    if ck.config.get("tie_word_embeddings") and "lm_head.weight" in sh and "model.embed_tokens.weight" in sh:
        if np.array_equal(np.asarray(ck.codes("lm_head.weight")), np.asarray(ck.codes("model.embed_tokens.weight"))):
            del sh["lm_head.weight"]
            aliases["lm_head.weight"] = "model.embed_tokens.weight"
    return sh, aliases


class ArrayModel:
    """In-memory stand-in with the Checkpoint API (tests): dict name -> uint16 bf16 code array."""

    def __init__(self, tensors, config=None):
        self.t = tensors
        self.config = config or {}

    def shapes(self):
        return {k: tuple(v.shape) for k, v in self.t.items()}

    def codes(self, name):
        return self.t[name]


# ============================================================================ canonical decode (vectorised)
_BF_SIG = _BF_E = _BF_OK = None


def _bf_tables():
    global _BF_SIG, _BF_E, _BF_OK
    if _BF_SIG is None:
        c = np.arange(1 << 16, dtype=np.int64)
        s = (c >> 15) & 1; e = (c >> 7) & 0xFF; m = c & 0x7F
        sig = np.where(e == 0, m, 128 + m); E = np.where(e == 0, -133, e - 134)
        _BF_SIG = np.where(s == 1, -sig, sig).astype(np.int64)
        _BF_E = E.astype(np.int64)
        _BF_OK = e != 0xFF
    return _BF_SIG, _BF_E, _BF_OK


def cd_exponent(codes):
    """Largest G with max |sigma| 2^(E+G) < 2^22 over the tensor."""
    sig_t, E_t, ok_t = _bf_tables()
    c = np.asarray(codes).reshape(-1)
    # the max of |sigma| 2^E is reached on the largest-magnitude code (bf16 order = magnitude order without sign)
    mag = c & 0x7FFF
    if not ok_t[c].all():
        raise ValueError("non-finite weight code")
    cm = int(mag.max())
    if cm == 0:
        return 0
    sig, E = abs(int(sig_t[cm])), int(E_t[cm])
    G = 21 - int(math.floor(math.log2(sig) + E))
    while (sig << max(0, E + G)) >> max(0, -(E + G)) >= (1 << 22):
        G -= 1
    return G


def cd_decode(codes, G):
    """floor(sigma 2^(E+G)) as int32 (|w| < 2^22)."""
    sig_t, E_t, _ = _bf_tables()
    c = np.asarray(codes).astype(np.int64)
    sig = sig_t[c]; k = E_t[c] + G
    out = np.where(k >= 0, sig << np.clip(k, 0, 62), sig >> np.clip(-k, 0, 63))
    assert out.size == 0 or int(np.abs(out).max()) < (1 << 22)
    return out.astype(np.int32)


# ============================================================================ layout
def _natkey(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


_MAT_ORDER = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "o_proj": 3, "gate_proj": 4, "up_proj": 5, "down_proj": 6}


def _tensor_key(name):
    parts = name.split(".")
    mk = next((_MAT_ORDER[p] for p in parts if p in _MAT_ORDER), 99)
    layer = int(parts[parts.index("layers") + 1]) if "layers" in parts else -1
    return (0 if "embed" in name else 1, layer, mk, _natkey(name))


class Layout:
    """Public-shape tiling of a checkpoint (layout v2).  rows[i] = dict(kind, ...) for every used row of W (private tile
    table): rows 0, 1.. = meta (SCALE, PN, and when cfg_rows=True also CFG0, CFG1), then vec rows, then tiles."""

    def __init__(self, shapes, prm, radius=False, cfg_rows=None, aliases=None, layer_meta=None):
        self.prm = prm
        self.radius = bool(radius)
        self.cfg_rows = (bool(radius) and prm.nc >= 1024) if cfg_rows is None else bool(cfg_rows)
        self.aliases = dict(aliases or {})
        self.layer_meta = dict(layer_meta or {})
        mats = sorted([k for k, s in shapes.items() if len(s) == 2], key=_tensor_key)
        vecs = sorted([k for k, s in shapes.items() if len(s) == 1], key=_natkey)
        self.mats, self.vecs, self.shapes = mats, vecs, dict(shapes)
        assert prm.R_MAX <= prm.nc, "SCALE must fit one row"
        H = prm.R_MAX // 2 if self.radius else prm.R_MAX
        if self.radius:
            assert 2 * H <= prm.nc, "SCALE + KAPPA_META must fit one row"
        self.n_pn = -(-(H * prm.TO) // prm.nc)                  # PN rows over the lower-half / data span (1 for Qwen)
        self.n_cfg = 2 if self.cfg_rows else 0
        self.meta_cfg0 = (1 + self.n_pn) if self.cfg_rows else None
        self.meta_cfg1 = (2 + self.n_pn) if self.cfg_rows else None
        self.n_meta = 1 + self.n_pn + self.n_cfg
        if self.cfg_rows:
            self.NT, self.NV, self.NL = cfg_sizes(prm)
        else:
            self.NT = self.NV = self.NL = 0
        rows = [{"kind": "scale"}] + [{"kind": "pnorm", "chunk": k} for k in range(self.n_pn)]
        if self.cfg_rows:
            rows.append({"kind": "cfg0"})
            rows.append({"kind": "cfg1"})
        self.vec_off = {}
        off = 0
        for name in vecs:
            self.vec_off[name] = off; off += shapes[name][0]
        self.n_vec_rows = -(-off // prm.nc) if off else 0
        self.vec_len = off
        for k in range(self.n_vec_rows):
            rows.append({"kind": "vec", "chunk": k})
        self.tile_of = {}                       # (tensor, R, C) -> row
        for name in mats:
            o, i = shapes[name]
            for R in range(-(-o // prm.TO)):
                for C in range(-(-i // prm.TI)):
                    self.tile_of[(name, R, C)] = len(rows)
                    rows.append({"kind": "tile", "tensor": name, "R": R, "C": C})
        self.raw_rows = len(rows)               # W^raw uses the same row indices (meta & cfg rows are zero codes)
        self.rows = rows
        self.n_tiles = len(self.tile_of)
        if self.radius:
            # radius matrix W^r = |W^cd| + kappa_n [inexact] of tile row t at row R_MAX/2 + t;
            # rows n_rows .. R_MAX/2 - 1 stay zero (row R_MAX/2 - 1 is the padding row of every claim slot, and its
            # partner R_MAX - 1 is zero too).  The offset is a profile constant, so the pairing t^r = t^cd + R_MAX/2
            # can be enforced in circuit without revealing anything (flashprove_next.wsum_zk).
            assert len(rows) < prm.R_MAX // 2, (len(rows), prm.R_MAX)
            self.row_ranges = [(0, len(rows)), (prm.R_MAX // 2, prm.R_MAX // 2 + len(rows))]
        else:
            assert len(rows) < prm.R_MAX, (len(rows), prm.R_MAX)      # row R_MAX - 1 stays zero (mc_zk padding slot)
            self.row_ranges = [(0, len(rows))]

    def r_row(self, t):
        """row of the radius tile of tile row t."""
        assert self.radius and 0 <= t < self.n_rows and self.rows[t]["kind"] == "tile"
        return self.prm.R_MAX // 2 + t

    @property
    def zero_row(self):
        """Canonical all-zero padding row index (H - 1 when radius=True, else R_MAX - 1)."""
        return (self.prm.R_MAX // 2 - 1) if self.radius else (self.prm.R_MAX - 1)

    @property
    def n_span(self):
        """1 + highest used row (all rows of W at or above it are zero)."""
        return self.row_ranges[-1][1]

    @property
    def n_rows(self): return len(self.rows)

    def table_json(self):
        return {"params": asdict(self.prm), "mats": [[m, list(self.shapes[m])] for m in self.mats],
                "vecs": [[v, list(self.shapes[v])] for v in self.vecs], "n_rows": self.n_rows, "raw_rows": self.raw_rows,
                "radius": bool(self.radius), "cfg_rows": bool(self.cfg_rows),
                "row_ranges": [list(r) for r in self.row_ranges]}


def row_exponent(codes, G_LO):
    """Canonical per-row CD exponent (layout v2): cd_exponent over the row, G_LO for a row without nonzero magnitude."""
    c = np.asarray(codes)
    if not (c & 0x7FFF).any():
        return G_LO
    return cd_exponent(c)


class RowSource:
    """Rows of W^cd (signed int64) or W^raw (codes) for the commitment.  Layout v2 needs every data row decoded before
    the meta rows (which come first), so the whole matrix is decoded once into RAM (int32: 4 B per entry) by prepare();
    Grow[i] = exponent of row i (R_MAX entries), raw/cd share one decode pass."""

    def __init__(self, ckpt, layout, raw=False, shared=None):
        self.ck, self.L, self.raw = ckpt, layout, raw
        self.shared = shared if shared is not None else {}

    def _tile_codes(self, rw):
        prm = self.L.prm
        c = self.ck.codes(rw["tensor"])
        r0, c0 = rw["R"] * prm.TO, rw["C"] * prm.TI
        blk = np.zeros((prm.TO, prm.TI), dtype=np.int64)
        sub = np.asarray(c[r0:r0 + prm.TO, c0:c0 + prm.TI])
        blk[:sub.shape[0], :sub.shape[1]] = sub
        return blk.reshape(-1)

    def _vec_codes(self, k):
        prm, L = self.L.prm, self.L
        out = np.zeros(prm.nc, dtype=np.int64)
        lo, hi = k * prm.nc, (k + 1) * prm.nc
        for name in L.vecs:
            a = L.vec_off[name]; b = a + L.shapes[name][0]
            s, e = max(a, lo), min(b, hi)
            if s < e:
                out[s - lo:e - lo] = np.asarray(self.ck.codes(name)).astype(np.int64)[s - a:e - a]
        return out

    def raw_row(self, i):
        rw = self.L.rows[i]
        if rw["kind"] == "tile":
            return self._tile_codes(rw)
        if rw["kind"] == "vec":
            return self._vec_codes(rw["chunk"])
        return np.zeros(self.L.prm.nc, dtype=np.int64)

    def _build_cfg_tables(self, G, kap_row, G_ten):
        """Build specs/09 §3.2 stable configuration tables (Tt in CFG0, Vt + Lt in CFG1) as int64 arrays."""
        from .certified import kappa, qk_exponent
        prm, L = self.L.prm, self.L
        NT, NV, NL = L.NT, L.NV, L.NL
        zrow = L.zero_row
        # ---- Tt (Tile Table): 8 segments of length NT in META_CFG0
        # fields: 0=key, 1=t_row, 2=G_t, 3=kappa_t, 4=s_align, 5=G_ten, 6=n_in, 7=flags
        entries_t = []
        all_mats = [(m, m) for m in L.mats] + [(al, tgt) for al, tgt in L.aliases.items() if tgt in L.shapes]
        for qname, tname in all_mats:
            role, lp1, exp = tensor_role_layer_expert(qname)
            o, n_in = L.shapes[tname]
            gt = int(G_ten[tname])
            kap = int(kappa(n_in))
            nR, nC = -(-o // prm.TO), -(-n_in // prm.TI)
            for R in range(nR):
                for C in range(nC):
                    t = L.tile_of[(tname, R, C)]
                    gt_row = int(G[t])
                    s_al = 1 << (gt - gt_row)
                    entries_t.append((tile_key(role, lp1, exp, R, C), t, gt_row, kap, s_al, gt, n_in, 1))
            pR = 1 << max(0, (nR - 1).bit_length()) if nR > 1 else 1
            pC = 1 << max(0, (nC - 1).bit_length()) if nC > 1 else 1
            for R in range(pR):
                for C in range(pC):
                    if R >= nR or C >= nC:
                        # zero-tile padding slot for power-of-2 gadget slot grid: t_row=zrow, G_t=G_LO, kap=0, s_align=1
                        entries_t.append((tile_key(role, lp1, exp, R, C), zrow, int(prm.G_LO), 0, 1, gt, n_in, 0))
        entries_t.sort(key=lambda e: e[0])
        assert len(entries_t) <= NT, f"Tt entries {len(entries_t)} exceed NT={NT}"
        for idx in range(len(entries_t) - 1):
            assert entries_t[idx + 1][0] > entries_t[idx][0], f"duplicate Tt key at {idx}: {entries_t[idx]} vs {entries_t[idx+1]}"
        tt_cols = [np.zeros(NT, dtype=np.int64) for _ in range(8)]
        for idx in range(NT):
            if idx < len(entries_t):
                row_tup = entries_t[idx]
            else:
                row_tup = (SENTINEL_BASE + idx, zrow, int(prm.G_LO), 0, 1, int(prm.G_LO), 0, 0)
            for f in range(8):
                tt_cols[f][idx] = row_tup[f]
        cfg0 = np.zeros(prm.nc, dtype=np.int64)
        for f in range(8):
            cfg0[f * NT:(f + 1) * NT] = tt_cols[f]

        # ---- Vt (Vector Table): 4 segments of length NV at [0, 4*NV) in META_CFG1
        # fields: 0=key, 1=vec_row, 2=offset, 3=length
        entries_v = []
        vec_base = L.n_meta
        for name in L.vecs:
            role, lp1, _ = tensor_role_layer_expert(name)
            off = L.vec_off[name]
            ln = L.shapes[name][0]
            k_chunk = off // prm.nc
            entries_v.append((vec_key(role, lp1, 0), vec_base + k_chunk, off % prm.nc, ln))
        entries_v.sort(key=lambda e: e[0])
        assert len(entries_v) <= NV, f"Vt entries {len(entries_v)} exceed NV={NV}"
        for idx in range(len(entries_v) - 1):
            assert entries_v[idx + 1][0] > entries_v[idx][0], f"duplicate Vt key at {idx}"
        vt_cols = [np.zeros(NV, dtype=np.int64) for _ in range(4)]
        for idx in range(NV):
            row_tup = entries_v[idx] if idx < len(entries_v) else (SENTINEL_BASE + idx, zrow, 0, 0)
            for f in range(4):
                vt_cols[f][idx] = row_tup[f]

        # ---- Lt (Layer Table): 8 segments of length NL at [4*NV, 4*NV + 8*NL) in META_CFG1
        # fields: 0=layer_idx, 1=valid, 2=attn_kind, 3=mlp_kind, 4=win, 5=nsink, 6=has_qknorm, 7=eq_ek_packed
        layers_seen = set()
        for m in L.mats:
            _, lp1, _ = tensor_role_layer_expert(m)
            if lp1 > 0:
                layers_seen.add(lp1 - 1)
        lt_cols = [np.zeros(NL, dtype=np.int64) for _ in range(8)]
        for l_idx in range(NL):
            lt_cols[0][l_idx] = l_idx
            if l_idx in layers_seen:
                lm = L.layer_meta.get(l_idx, {})
                qn = f"model.layers.{l_idx}.self_attn.q_norm.weight"
                kn = f"model.layers.{l_idx}.self_attn.k_norm.weight"
                has_qk = int(qn in L.shapes and kn in L.shapes)
                if "eq_ek_packed" in lm:
                    eq_ek = int(lm["eq_ek_packed"])
                elif has_qk:
                    dh = int(L.shapes[qn][0])
                    eq = qk_exponent(self.ck.codes(qn), dh)
                    ek = qk_exponent(self.ck.codes(kn), dh)
                    eq_ek = ((int(eq) + 128) << 8) | ((int(ek) + 128) & 0xFF)
                else:
                    eq_ek = 0
                has_moe = any(f"layers.{l_idx}." in m and "experts" in m for m in L.mats)
                lt_cols[1][l_idx] = 1
                lt_cols[2][l_idx] = int(lm.get("attn_kind", 0))
                lt_cols[3][l_idx] = int(lm.get("mlp_kind", 1 if has_moe else 0))
                lt_cols[4][l_idx] = int(lm.get("win", 0))
                lt_cols[5][l_idx] = int(lm.get("nsink", 0))
                lt_cols[6][l_idx] = has_qk
                lt_cols[7][l_idx] = eq_ek
        cfg1 = np.zeros(prm.nc, dtype=np.int64)
        for f in range(4):
            cfg1[f * NV:(f + 1) * NV] = vt_cols[f]
        for f in range(8):
            cfg1[4 * NV + f * NL:4 * NV + (f + 1) * NL] = lt_cols[f]
        return cfg0, cfg1, tt_cols, vt_cols, lt_cols

    def prepare(self, log=print):
        sh = self.shared
        if "cd" in sh:
            return sh
        from .certified import kappa
        prm, L = self.L.prm, self.L
        n, nm = L.n_rows, L.n_meta
        H = prm.R_MAX // 2 if L.radius else prm.R_MAX
        t0 = time.time()
        raw = np.zeros((n, prm.nc), dtype=np.int32)
        cd = np.zeros((n, prm.nc), dtype=np.int32)
        rad = np.zeros((n, prm.nc), dtype=np.int32) if L.radius else None
        G = np.full(H, prm.G_LO, dtype=np.int64)
        G[:nm] = 0
        kap_row = np.zeros(H, dtype=np.int64)
        pn = np.zeros(L.n_pn * prm.nc, dtype=np.int64)
        ten_gs = {}
        for i in range(nm, n):
            rw = L.rows[i]
            c = self.raw_row(i)
            raw[i] = c
            g = row_exponent(c, prm.G_LO)
            if not prm.G_LO <= g < prm.G_LO + G_SPAN:
                raise ValueError(f"row {i} ({rw}): CD exponent {g} outside the profile range "
                                 f"[{prm.G_LO}, {prm.G_LO + G_SPAN})")
            w = cd_decode(c, g)
            G[i] = g
            cd[i] = w
            if rw["kind"] == "tile":
                tname = rw["tensor"]
                n_in = L.shapes[tname][1]
                kap = int(kappa(n_in))
                kap_row[i] = kap
                if (c & 0x7FFF).any():
                    ten_gs.setdefault(tname, []).append(int(g))
                if L.radius:
                    inex = tile_inexact(c, g)
                    wr = np.abs(w.astype(np.int64)) + kap * inex.astype(np.int64)
                    assert int(wr.max(initial=0)) < (1 << 23)
                    rad[i] = wr.astype(np.int32)
            w64 = w.astype(np.int64).reshape(prm.TO, prm.TI)
            pn[i * prm.TO:(i + 1) * prm.TO] = (w64 * w64).sum(axis=1)
        G_ten = {m: (max(ten_gs[m]) if ten_gs.get(m) else int(prm.G_LO)) for m in L.mats}
        scale_row = np.zeros(prm.nc, dtype=np.int64)
        scale_row[:H] = G
        if L.radius:
            scale_row[H:2 * H] = kap_row
        cd[META_SCALE] = scale_row.astype(np.int32)
        wide_rows = {META_SCALE: scale_row} if (scale_row != cd[META_SCALE].astype(np.int64)).any() else {}
        if L.cfg_rows:
            cfg0, cfg1, tt_cols, vt_cols, lt_cols = self._build_cfg_tables(G, kap_row, G_ten)
            for r_idx, crow in ((L.meta_cfg0, cfg0), (L.meta_cfg1, cfg1)):
                c32 = crow.astype(np.int32)
                if (c32.astype(np.int64) != crow).any():
                    wide_rows[r_idx] = crow
                    cd[r_idx] = 0
                else:
                    cd[r_idx] = c32
            sh.update(cfg0=cfg0, cfg1=cfg1, tt_cols=tt_cols, vt_cols=vt_cols, lt_cols=lt_cols)
        cd_pn = pn
        assert int(cd_pn.max(initial=0)) < (1 << 62)
        sh.update(raw=raw, cd=cd, rad=rad, G=G, kap_row=kap_row, G_ten=G_ten, scale_row=scale_row,
                  wide_rows=wide_rows, pn=cd_pn)
        log(f"  [decode] {n} rows decoded in {time.time() - t0:.1f}s (exponents {int(G[nm:n].min())}..{int(G[nm:n].max())})")
        return sh

    def rows(self, i0, i1):
        sh = self.prepare()
        prm, L = self.L.prm, self.L
        if L.radius and i0 >= prm.R_MAX // 2:
            assert not self.raw
            h = prm.R_MAX // 2
            return sh["rad"][i0 - h:i1 - h].astype(np.int64)
        if self.raw:
            return sh["raw"][i0:i1].astype(np.int64)
        out = sh["cd"][i0:i1].astype(np.int64)
        nc = prm.nc
        for k in range(L.n_pn):                          # int64 (PN entries exceed int32)
            if i0 <= META_PN + k < i1:
                out[META_PN + k - i0] = sh["pn"][k * nc:(k + 1) * nc]
        for r_idx, wrow in sh.get("wide_rows", {}).items():
            if i0 <= r_idx < i1:
                out[r_idx - i0] = wrow
        return out

    @property
    def Grow(self):
        return self.prepare()["G"]

    @property
    def Gten(self):
        return self.prepare()["G_ten"]


def aligned_mat(codes, prm, contract=None):
    """Decode a 2-D bf16 weight tensor tile-by-tile with per-row maximal exponents G_t = row_exponent(tile, prm.G_LO)
    and align all tiles (a, b) to G_ten = max_{nonzero tiles} G_{a,b} via s_t = 2^(G_ten - G_t) >= 1:
        W_aligned[a, b]  = s_t * cd_decode(tile, G_t)
        Wr_aligned[a, b] = s_t * (|cd_decode(tile, G_t)| + kappa(n_in) * tile_inexact(tile, G_t)).
    Returns (certified.Mat(W_aligned, G_ten, Wr_aligned, n_in), max_dG)."""
    from .certified import Mat, kappa, C0
    contract = C0 if contract is None else contract
    codes = np.asarray(codes, dtype=np.uint16)
    o, n_in = codes.shape
    nA, nB = -(-o // prm.TO), -(-n_in // prm.TI)
    kap = int(kappa(n_in, contract))
    tiles = {}
    nz_gs = []
    for a in range(nA):
        for b in range(nB):
            blk = np.zeros((prm.TO, prm.TI), dtype=np.uint16)
            sub = codes[a * prm.TO:(a + 1) * prm.TO, b * prm.TI:(b + 1) * prm.TI]
            blk[:sub.shape[0], :sub.shape[1]] = sub
            g = row_exponent(blk, prm.G_LO)
            if (blk & 0x7FFF).any():
                nz_gs.append(int(g))
            w = cd_decode(blk, g).astype(np.int64)
            wr = np.abs(w) + kap * tile_inexact(blk, g).astype(np.int64)
            tiles[(a, b)] = (int(g), w, wr, sub.shape)
    G_ten = max(nz_gs) if nz_gs else int(prm.G_LO)
    max_dG = max((G_ten - g for (g, _, _, _) in tiles.values()), default=0)
    W = np.zeros((o, n_in), dtype=np.int64)
    Wr = np.zeros((o, n_in), dtype=np.int64)
    for (a, b), (g, w, wr, sh) in tiles.items():
        s = 1 << (G_ten - g)
        W[a * prm.TO:a * prm.TO + sh[0], b * prm.TI:b * prm.TI + sh[1]] = s * w[:sh[0], :sh[1]]
        Wr[a * prm.TO:a * prm.TO + sh[0], b * prm.TI:b * prm.TI + sh[1]] = s * wr[:sh[0], :sh[1]]
    dt = np.int32 if max(int(np.abs(W).max(initial=0)), int(Wr.max(initial=0))) < (1 << 31) else np.int64
    return Mat(W.astype(dt), int(G_ten), Wr.astype(dt), n_in), int(max_dG)


class MatrixSource:
    """Rows given directly as an (nrows, n_c) signed integer array (tests)."""

    def __init__(self, W):
        self.W = np.asarray(W, dtype=np.int64)

    def rows(self, i0, i1):
        return self.W[i0:i1]


# ============================================================================ device helpers
def _is_gpu(B):
    return getattr(B, "name", "") == "cupy"


_K = {}


def _kernels():
    if not _K:
        import cupy as cp
        from .field_gpu import PRE
        src = PRE + r"""
extern "C" __global__ void mc_matvec(const int* __restrict__ W, const long long* __restrict__ tidx,
    const u64* __restrict__ h0, const u64* __restrict__ h1, const u64* __restrict__ h2, int R, int C,
    u64* __restrict__ o0, u64* __restrict__ o1, u64* __restrict__ o2){
  // o[b, r] = sum_c W[tidx[b]][r, c] h[b, c]     (one block per (r, b))
  int r = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int* w = W + (tidx[b] * (long long)R + r) * (long long)C;
  const u64* x0 = h0 + (long long)b * C; const u64* x1 = h1 + (long long)b * C; const u64* x2 = h2 + (long long)b * C;
  u64 s0 = 0, s1 = 0, s2 = 0;
  for (int c = tid; c < C; c += blockDim.x) {
    u64 wv = fromi((long long)w[c]);
    s0 = addm(s0, mulm(wv, x0[c])); s1 = addm(s1, mulm(wv, x1[c])); s2 = addm(s2, mulm(wv, x2[c]));
  }
  __shared__ u64 a0[256], a1[256], a2[256];
  a0[tid] = s0; a1[tid] = s1; a2[tid] = s2; __syncthreads();
  for (int h = blockDim.x >> 1; h > 0; h >>= 1) {
    if (tid < h) { a0[tid] = addm(a0[tid], a0[tid + h]); a1[tid] = addm(a1[tid], a1[tid + h]); a2[tid] = addm(a2[tid], a2[tid + h]); }
    __syncthreads();
  }
  if (tid == 0) { o0[(long long)b * R + r] = a0[0]; o1[(long long)b * R + r] = a1[0]; o2[(long long)b * R + r] = a2[0]; }
}
extern "C" __global__ void mc_vecmat(const int* __restrict__ W, const long long* __restrict__ tidx,
    const u64* __restrict__ v0, const u64* __restrict__ v1, const u64* __restrict__ v2, int R, long long C,
    u64* __restrict__ o0, u64* __restrict__ o1, u64* __restrict__ o2){
  // o[b, c] = sum_r v[b, r] W[tidx[b]][r, c]
  long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y; if (c >= C) return;
  const int* w = W + tidx[b] * (long long)R * C + c;
  const u64* a0 = v0 + (long long)b * R; const u64* a1 = v1 + (long long)b * R; const u64* a2 = v2 + (long long)b * R;
  u64 s0 = 0, s1 = 0, s2 = 0;
  for (int r = 0; r < R; r++) {
    u64 wv = fromi((long long)w[(long long)r * C]);
    s0 = addm(s0, mulm(a0[r], wv)); s1 = addm(s1, mulm(a1[r], wv)); s2 = addm(s2, mulm(a2[r], wv));
  }
  o0[(long long)b * C + c] = s0; o1[(long long)b * C + c] = s1; o2[(long long)b * C + c] = s2;
}
"""
        mod = cp.RawModule(code=src)
        _K["matvec"] = mod.get_function("mc_matvec")
        _K["vecmat"] = mod.get_function("mc_vecmat")
    return _K


def tiles_matvec(B, Wt, tidx, H):
    """Wt: (ntile, TO, TI) int32; tidx: per-term tile index; H: (3, nterm, TI) E.  Returns (3, nterm, TO)."""
    nt, R, C = Wt.shape
    nb = len(tidx)
    if _is_gpu(B):
        import cupy as cp
        k = _kernels()["matvec"]
        Hd = cp.ascontiguousarray(cp.asarray(H, dtype=cp.uint64))
        out = cp.empty((3, nb, R), dtype=cp.uint64)
        if nb:
            k((R, nb), (256,), (Wt, cp.asarray(np.asarray(tidx, dtype=np.int64)), Hd[0], Hd[1], Hd[2],
                                np.int32(R), np.int32(C), out[0], out[1], out[2]))
        return out
    out = np.zeros((3, nb, R), dtype=np.uint64)
    for b, t in enumerate(tidx):
        Wf = CPU.fromi(np.asarray(Wt[t], dtype=np.int64))
        for l in (0, 1, 2):
            hb = np.broadcast_to(np.asarray(H[l, b], dtype=np.uint64), Wf.shape)
            prod = CPU.mul(Wf, hb)
            out[l, b] = [CPU.bsum(prod[r]) for r in range(R)]
    return out


def tiles_vecmat(B, Wt, tidx, V):
    """Wt: (ntile, R, C) int32; V: (3, nterm, R) E.  Returns (3, nterm, C): out[b, c] = sum_r V[b, r] W[tidx[b]][r, c]."""
    nt, R, C = Wt.shape
    nb = len(tidx)
    if _is_gpu(B):
        import cupy as cp
        k = _kernels()["vecmat"]
        Vd = cp.ascontiguousarray(cp.asarray(V, dtype=cp.uint64))
        out = cp.empty((3, nb, C), dtype=cp.uint64)
        if nb:
            k(((C + 255) // 256, nb), (256,), (Wt, cp.asarray(np.asarray(tidx, dtype=np.int64)), Vd[0], Vd[1], Vd[2],
                                               np.int32(R), np.int64(C), out[0], out[1], out[2]))
        return out
    out = np.zeros((3, nb, C), dtype=np.uint64)
    for b, t in enumerate(tidx):
        Wf = CPU.fromi(np.asarray(Wt[t], dtype=np.int64))
        for l in (0, 1, 2):
            acc = np.zeros(C, dtype=np.uint64)
            for r in range(R):
                acc = CPU.add(acc, CPU.mul(Wf[r], np.uint64(int(V[l, b, r]))))
            out[l, b] = acc
    return out


def _ntt(B, coeffs, n, out=None):
    if _is_gpu(B):
        return B.ntt(coeffs, n, out=out)
    return CPU.ntt(np.asarray(coeffs, dtype=np.uint64), n)


def _sync(B):
    if _is_gpu(B):
        import cupy as cp
        cp.cuda.Device().synchronize()


def _gpu_mem():
    try:
        import cupy as cp
        return cp.get_default_memory_pool().total_bytes()
    except Exception:
        return 0


# ============================================================================ commitment
def _leaf_init(B, prm, salts, cols=None):
    """Sponge state after the first block [j, s_j(4), R_MAX, 0, 0] for columns `cols` (default: all n)."""
    xp = B.xp
    js = np.arange(prm.n, dtype=np.uint64) if cols is None else np.asarray(cols, dtype=np.uint64)
    m = len(js)
    st = xp.zeros((PS.T, m), dtype=np.uint64)
    st[10] = np.uint64(DOM_COL); st[11] = np.uint64(PS.TAG_SPONGE)
    st[0] = B.asarray(js)
    st[1:5] = B.asarray(np.ascontiguousarray(salts if cols is None else salts[:, cols]))
    st[5] = np.uint64(prm.R_MAX)
    return PS.permute(st, B)


def leaf_hashes(prm, js, salts, cols):
    """Host reference: leaf digests of columns js (salts (4, m), cols (R_MAX, m) field elements) -> list of 4-tuples."""
    cols = np.asarray(cols, dtype=np.uint64)
    m = len(js)
    assert cols.shape == (prm.R_MAX, m)
    s = np.zeros((PS.T, m), dtype=np.uint64)
    s[10] = DOM_COL; s[11] = PS.TAG_SPONGE
    s[0] = np.asarray(js, dtype=np.uint64); s[1:5] = np.asarray(salts, dtype=np.uint64); s[5] = prm.R_MAX
    s = PS.permute(s, CPU)
    for b in range(prm.R_MAX // 8):
        s = s.copy()
        s[0:8] = CPU.add(s[0:8], cols[8 * b:8 * b + 8])
        s = PS.permute(s, CPU)
    return [tuple(int(v) for v in s[0:4, k]) for k in range(m)]


def _present(ranges, g0, g1):
    """rows of [g0, g1) inside the used row ranges; must be a prefix of the group (returns its length)."""
    rows = [i for i in range(g0, g1) if any(lo <= i < hi for lo, hi in ranges)]
    assert rows == list(range(g0, g0 + len(rows))), ("used rows must form a group prefix", g0, rows[:3], rows[-3:])
    return len(rows)


def commit_matrix(src, n_used, prm, B, store_dir=None, tag="W", salts=None, store_coeffs=False, log=print,
                  store_cw=True, cw_hook=None, row_ranges=None):
    """Streaming commitment of the rows src.rows(i0, i1) (signed ints), i < n_used (rows n_used..R_MAX-1 are zero), or
    of the rows inside row_ranges [(lo, hi), ...] (all other rows zero; used rows must form a prefix of each GROUP).
    store_dir: tree + salts (+ codeword row-group files if store_cw, + int32 coefficients if store_coeffs; a row that
    does not fit int32 -- the PN meta row -- is written as zeros there and saved whole in {tag}_wide.npz).
    cw_hook(g0, X): test hook that may modify a codeword row group (device array) before it is hashed.
    Returns dict(root, levels (host, levels[d] (4, n >> d)), salts (4, n), stats)."""
    wide = {}
    if row_ranges is None:
        row_ranges = [(0, n_used)]
    xp = B.xp
    gpu = _is_gpu(B)
    assert prm.GROUP % 8 == 0 and prm.R_MAX % prm.GROUP == 0
    st_t = {"decode_s": 0.0, "upload_s": 0.0, "ntt_s": 0.0, "sponge_s": 0.0, "store_s": 0.0, "merkle_s": 0.0}
    t_all = time.time()
    if salts is None:
        salts = rand_field(4 * prm.n).reshape(4, prm.n)
    t0 = time.time()
    st = _leaf_init(B, prm, salts)
    _sync(B); st_t["sponge_s"] += time.time() - t0
    files = []
    cf = None
    if store_dir is not None:
        os.makedirs(store_dir, exist_ok=True)
        if store_coeffs:
            cf = open(os.path.join(store_dir, f"{tag}_coeffs.i32"), "wb")
    ngroups = prm.R_MAX // prm.GROUP
    for g in range(ngroups):
        g0 = g * prm.GROUP
        nrow = _present(row_ranges, g0, g0 + prm.GROUP)
        X = None
        if nrow:
            t0 = time.time()
            C = src.rows(g0, g0 + nrow)                      # (nrow, nc) int64
            st_t["decode_s"] += time.time() - t0
            if cf is not None:
                t0 = time.time()
                C32 = np.ascontiguousarray(C, dtype=np.int32)
                for r in np.flatnonzero((C32.astype(np.int64) != C).any(axis=1)):
                    wide[str(g0 + int(r))] = np.asarray(C[r], dtype=np.int64); C32[r] = 0
                C32.tofile(cf); st_t["store_s"] += time.time() - t0
            t0 = time.time()
            Cf = B.asarray(CPU.fromi(C))
            _sync(B); st_t["upload_s"] += time.time() - t0
            t0 = time.time()
            X = xp.zeros((prm.GROUP, prm.n), dtype=np.uint64)
            for r in range(nrow):
                if gpu:
                    _ntt(B, Cf[r], prm.n, out=X[r])
                else:
                    X[r] = _ntt(B, Cf[r], prm.n)
            del Cf
            if cw_hook is not None:
                cw_hook(g0, X)
            _sync(B); st_t["ntt_s"] += time.time() - t0
        t0 = time.time()
        for b in range(prm.GROUP // 8):
            if X is not None and 8 * b < nrow:
                st = st.copy() if not gpu else st
                st[0:8] = B.add(st[0:8], X[8 * b:8 * b + 8])
            st = PS.permute(st, B)
        _sync(B); st_t["sponge_s"] += time.time() - t0
        if X is not None and store_dir is not None and store_cw:
            t0 = time.time()
            path = os.path.join(store_dir, f"{tag}_cw_g{g:02d}.u64")
            XT = xp.ascontiguousarray(X.T)                   # (n, GROUP): column j of the group is contiguous
            del X
            h = B.tonumpy(XT); del XT
            h.tofile(path); del h
            files.append(path)
            st_t["store_s"] += time.time() - t0
        X = None
        log(f"  [{tag}] group {g + 1}/{ngroups} rows {g0}..{g0 + prm.GROUP - 1} used {nrow}  t={time.time() - t_all:.1f}s")
    if cf is not None:
        cf.close()
    t0 = time.time()
    lv = [st[0:4].copy()]
    del st
    while lv[-1].shape[1] > 1:
        a = lv[-1]
        lv.append(PS.compress(xp.ascontiguousarray(a[:, 0::2]), xp.ascontiguousarray(a[:, 1::2]), B))
    levels = [B.tonumpy(x) for x in lv]
    del lv
    _sync(B); st_t["merkle_s"] += time.time() - t0
    root = tuple(int(v) for v in levels[-1][:, 0])
    st_t["total_s"] = time.time() - t_all
    st_t["codeword_files"] = files
    st_t["row_ranges"] = [list(r) for r in row_ranges]
    st_t["codeword_bytes"] = sum(os.path.getsize(f) for f in files)
    if store_dir is not None:
        t0 = time.time()
        np.save(os.path.join(store_dir, f"{tag}_salts.npy"), salts)
        np.save(os.path.join(store_dir, f"{tag}_tree.npy"), np.concatenate(levels, axis=1))
        st_t["store_s"] += time.time() - t0
        st_t["tree_bytes"] = os.path.getsize(os.path.join(store_dir, f"{tag}_tree.npy"))
        st_t["salt_bytes"] = os.path.getsize(os.path.join(store_dir, f"{tag}_salts.npy"))
        if store_coeffs:
            st_t["coeff_bytes"] = os.path.getsize(os.path.join(store_dir, f"{tag}_coeffs.i32"))
            if wide:
                np.savez(os.path.join(store_dir, f"{tag}_wide.npz"), **wide)
            st_t["wide_rows"] = sorted(int(k) for k in wide)
    return {"root": root, "levels": levels, "salts": salts, "stats": st_t}


def bytes_to_field(b):
    """7 bytes per element (injective with the length prefix)."""
    b = bytes(b)
    out = [len(b)]
    for i in range(0, len(b), 7):
        out.append(int.from_bytes(b[i:i + 7], "little"))
    while len(out) % 8:
        out.append(0)
    return np.array(out, dtype=np.uint64)


def sponge_host(elems, domain):
    e = np.asarray(elems, dtype=np.uint64)
    if len(e) % 8:
        e = np.concatenate([e, np.zeros(8 - len(e) % 8, dtype=np.uint64)])
    d = PS.sponge(e.reshape(-1, 8, 1), CPU, domain=domain)
    return tuple(int(v) for v in d[:, 0])


def model_id(salt0, c_w, c_raw, c_cfg):
    return sponge_host(list(salt0) + list(c_w) + list(c_raw) + list(c_cfg), DOM_MID)


def encode_volatile_cfg(vol_dict, P_max=P_MAX_CFG):
    """Encode volatile configuration M^vol (specs/09 §3.2) into a fixed-length (P_max * 8,) uint64 field array:
      block 0: [v_rel, v_cap, v_profile, k_prf[0..3], sys_len]
      block 1: [T_bits, p16, beta, L_real, V_real, cot_mode, flags, salt_cfg[0]]
      block 2: salt_cfg[1..3] (indices 16..18), zeros; blocks 4..P_max-1: sys_prompt token IDs from index VOL_SYS0 = 32
      (aligned, so an in-circuit view maps position bits directly), padded with zeros."""
    assert P_max >= 3
    d = dict(vol_dict or {})
    out = np.zeros(P_max * 8, dtype=np.uint64)
    k_prf = list(d.get("k_prf", (0, 0, 0, 0)))
    assert len(k_prf) == 4
    salt_cfg = list(d.get("salt_cfg", (0, 0, 0, 0)))
    assert len(salt_cfg) == 4
    sys_prompt = [int(t) for t in d.get("sys_prompt", ())]
    assert len(sys_prompt) <= P_max * 8 - VOL_SYS0, f"sys_prompt length {len(sys_prompt)} exceeds {P_max * 8 - VOL_SYS0}"
    out[0] = np.uint64(int(d.get("v_rel", 1)) % P)
    out[1] = np.uint64(int(d.get("v_cap", 1)) % P)
    out[2] = np.uint64(int(d.get("v_profile", 1)) % P)
    for k in range(4):
        out[3 + k] = np.uint64(int(k_prf[k]) % P)
    out[7] = np.uint64(len(sys_prompt))
    out[8] = np.uint64(int(d.get("T_bits", 0)) % P)
    out[9] = np.uint64(int(d.get("p16", 0)) % P)
    out[10] = np.uint64(int(d.get("beta", 0)) % P)
    out[11] = np.uint64(int(d.get("L_real", 0)) % P)
    out[12] = np.uint64(int(d.get("V_real", 0)) % P)
    out[13] = np.uint64(int(d.get("cot_mode", 0)) % P)
    out[14] = np.uint64(int(d.get("flags", 0)) % P)
    out[15] = np.uint64(int(salt_cfg[0]) % P)
    for k in range(3):
        out[16 + k] = np.uint64(int(salt_cfg[1 + k]) % P)
    for idx, tok in enumerate(sys_prompt):
        assert 0 <= tok < P
        out[VOL_SYS0 + idx] = np.uint64(tok)
    return out


def volatile_config_root(vol_elems):
    """C_cfg = H_in(DOM_CFG; Enc_{P_max}(M^vol)) (specs/09 §3.2)."""
    v = np.asarray(vol_elems, dtype=np.uint64)
    assert v.ndim == 1 and len(v) % 8 == 0 and len(v) > 0
    return sponge_host(v, DOM_CFG)


def config_root(layout, config, salt):
    if isinstance(config, dict) and ("k_prf" in config or "sys_prompt" in config or config.get("volatile")):
        vd = dict(config)
        vd.setdefault("salt_cfg", salt)
        return volatile_config_root(encode_volatile_cfg(vd))
    blob = json.dumps({"version": VERSION, "config": config, "layout": layout.table_json()}, sort_keys=True).encode()
    return sponge_host(list(salt) + list(bytes_to_field(blob)), DOM_CFG)


# ============================================================================ stored commitment (prover side)
class Store:
    def __init__(self, d, prm, tag="W"):
        self.d, self.prm, self.tag = d, prm, tag
        meta = json.load(open(os.path.join(d, "commit.json")))
        self.meta = meta
        self.ranges = [tuple(r) for r in meta[tag].get("row_ranges", [[0, meta[tag]["n_rows"]]])]
        self.n_used = self.ranges[-1][1]                        # 1 + highest used row
        self.salts = np.load(os.path.join(d, f"{tag}_salts.npy"), mmap_mode="r")
        tree = np.load(os.path.join(d, f"{tag}_tree.npy"), mmap_mode="r")
        self.levels, off = [], 0
        for k in range(prm.depth + 1):
            w = prm.n >> k
            self.levels.append(tree[:, off:off + w]); off += w
        self.root = tuple(int(v) for v in self.levels[-1][:, 0])
        npres = sum(hi - lo for lo, hi in self.ranges)
        self.coeffs = _SparseRows(np.memmap(os.path.join(d, f"{tag}_coeffs.i32"), dtype=np.int32, mode="r",
                                            shape=(npres, prm.nc)), self.ranges, prm.nc)
        wp = os.path.join(d, f"{tag}_wide.npz")
        self.wide = {int(k): v for k, v in np.load(wp).items()} if os.path.exists(wp) else {}   # rows beyond int32 (PN)
        pgs = [g for g in range(-(-self.n_used // prm.GROUP))
               if _present(self.ranges, g * prm.GROUP, (g + 1) * prm.GROUP)]
        cwp = {g: os.path.join(d, f"{tag}_cw_g{g:02d}.u64") for g in pgs}
        self.cw = ({g: np.memmap(f, dtype=np.uint64, mode="r", shape=(prm.n, prm.GROUP)) for g, f in cwp.items()}
                   if all(os.path.exists(f) for f in cwp.values()) else None)     # optional (commit --store-cw)

    def row(self, i):
        """Coefficient row i (int64), wide rows included."""
        if i in self.wide:
            return np.asarray(self.wide[i], dtype=np.int64)
        return np.asarray(self.coeffs[i], dtype=np.int64)

    def drop_cache(self):
        import mmap
        for mm in [self.coeffs.mm, self.salts] + list((self.cw or {}).values()) + (self.levels[:1] if self.levels else []):
            try:
                base_mm = getattr(mm, "_mmap", None) or getattr(getattr(mm, "base", None), "_mmap", None)
                if base_mm is not None and hasattr(mmap, "MADV_DONTNEED"):
                    base_mm.madvise(mmap.MADV_DONTNEED)
            except (OSError, AttributeError, ValueError):
                pass
        paths = [os.path.join(self.d, f"{self.tag}_cw_g{g:02d}.u64") for g in (self.cw or ())]
        for sfx in ("_coeffs.i32", "_tree.npy", "_salts.npy"):
            paths.append(os.path.join(self.d, f"{self.tag}{sfx}"))
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            except (OSError, AttributeError):
                pass

    def column(self, j):
        if self.cw is None:
            raise RuntimeError("codeword files not stored (commit with --store-cw)")
        col = np.zeros(self.prm.R_MAX, dtype=np.uint64)
        for g, mm in self.cw.items():
            col[g * self.prm.GROUP:(g + 1) * self.prm.GROUP] = mm[j]
        return col

    def columns(self, Q, B=CPU):
        """Columns M[:, Q] (R_MAX, len(Q)): read from codeword files if stored, else re-encoded in 64-row batches on B."""
        if self.cw is not None:
            return np.stack([self.column(int(j)) for j in Q], axis=1)
        Q_arr = np.asarray(Q, dtype=np.int64)
        nq = len(Q_arr)
        out = np.zeros((self.prm.R_MAX, nq), dtype=np.uint64)
        gpu = _is_gpu(B)
        q_dev = B.asarray(Q_arr) if gpu else Q_arr
        buf = B.xp.empty(self.prm.n, dtype=np.uint64) if gpu else None
        for (lo, hi), o in zip(self.coeffs.ranges, self.coeffs.off):
            for c0 in range(0, hi - lo, 64):
                c1 = min(hi - lo, c0 + 64)
                nr = c1 - c0
                C32 = np.ascontiguousarray(self.coeffs.mm[o + c0:o + c1])
                if gpu:
                    Cf = B.fromi(B.asarray(C32))
                    g_dev = B.xp.empty((nr, nq), dtype=np.uint64)
                    for k in range(nr):
                        r = lo + c0 + k
                        r_dev = B.asarray(CPU.fromi(self.wide[r])) if r in self.wide else Cf[k]
                        _ntt(B, r_dev, self.prm.n, out=buf)
                        g_dev[k] = buf[q_dev]
                    out[lo + c0:lo + c1] = B.tonumpy(g_dev)
                    del Cf, g_dev
                else:
                    for k in range(nr):
                        r = lo + c0 + k
                        cf = CPU.fromi(self.wide[r] if r in self.wide else C32[k])
                        out[r] = _ntt(B, cf, self.prm.n)[Q_arr]
        self.drop_cache()
        return out

    def path(self, j):
        out = []
        for d in range(self.prm.depth):
            out.append(tuple(int(v) for v in self.levels[d][:, j ^ 1])); j >>= 1
        return out


class _SparseRows:
    """row index -> coefficient row (zeros outside the used ranges)."""

    def __init__(self, mm, ranges, nc):
        self.mm, self.ranges, self.nc = mm, ranges, nc
        self.off = []
        o = 0
        for lo, hi in ranges:
            self.off.append(o); o += hi - lo

    def __getitem__(self, i):
        for (lo, hi), o in zip(self.ranges, self.off):
            if lo <= i < hi:
                return self.mm[o + i - lo]
        return np.zeros(self.nc, dtype=np.int32)


class MemStore:
    """In-memory store from commit_matrix (tests)."""

    def __init__(self, prm, W, com, B=CPU):
        self.prm = prm
        W64 = np.asarray(W, dtype=np.int64)
        self.coeffs = W64.astype(np.int32)
        self.wide = {int(r): W64[int(r)] for r in np.flatnonzero((self.coeffs.astype(np.int64) != W64).any(axis=1))}
        self.n_used = self.coeffs.shape[0]
        self.levels, self.salts, self.root = com["levels"], com["salts"], com["root"]
        Wf = CPU.fromi(W64)
        self.M = np.zeros((prm.R_MAX, prm.n), dtype=np.uint64)
        for r in range(self.n_used):
            self.M[r] = CPU.ntt(Wf[r], prm.n)

    def row(self, i):
        if i in self.wide:
            return np.asarray(self.wide[i], dtype=np.int64)
        return np.asarray(self.coeffs[i], dtype=np.int64) if i < self.n_used else np.zeros(self.prm.nc, dtype=np.int64)

    def column(self, j):
        return self.M[:, j].copy()

    def path(self, j):
        out = []
        for d in range(self.prm.depth):
            out.append(tuple(int(v) for v in self.levels[d][:, j ^ 1])); j >>= 1
        return out


# ============================================================================ claims
@dataclass
class GemmClaim:
    """y = sum_i W[tiles[i]] h_i over the tile rows (y: (TO,) or (2, TO); h_i: (TI,) or (2, TI))."""
    tiles: list
    hs: list
    y: object


@dataclass
class LookupClaim:
    """x = W[tile][r0, :] (x: (TI,))."""
    tile: int
    r0: int
    x: object


def _e(v):
    v = np.asarray(v)
    if v.ndim == 1:
        f = CPU.fromi(v) if v.dtype != np.uint64 else v
        z = np.zeros(len(v), dtype=np.uint64)
        return np.stack([f, z, z])
    return v.astype(np.uint64)


def mle(v, z):
    """MLE of a host vector (base or E) at z (variable 0 = LSB)."""
    v = _e(v)
    return CPU.edot_dev(CPU.eq_table(z), v)


def eq_vec(z):
    return CPU.eq_table(z)


def _bits_point(x, nb):
    return [((x >> b) & 1, 0, 0) for b in range(nb)]


def _uses(claims, prm, xi_p, zeta_p, lams):
    """Expand claims into sumcheck terms: list of (tile, g (3, TO) host, h (3, TI) host, gspec, hspec) and the total."""
    uses = []
    total = ZERO
    eqx = eq_vec(xi_p); eqz = eq_vec(zeta_p)
    for k, (cl, lam) in enumerate(zip(claims, lams)):
        if isinstance(cl, GemmClaim):
            g = CPU.escale(eqx, lam)
            for t, h in zip(cl.tiles, cl.hs):
                uses.append({"tile": int(t), "g": g, "h": _e(h), "gspec": ("eq", lam), "hspec": ("vec", k, len(uses))})
            total = hadd(total, hmul(lam, mle(cl.y, xi_p)))
        else:
            g = np.zeros((3, prm.TO), dtype=np.uint64)
            g[0, cl.r0] = lam[0]; g[1, cl.r0] = lam[1]; g[2, cl.r0] = lam[2]
            uses.append({"tile": int(cl.tile), "g": g, "h": eqz, "gspec": ("onehot", lam, cl.r0), "hspec": ("eqz",)})
            total = hadd(total, hmul(lam, mle(cl.x, zeta_p)))
    return uses, total


def _round3(B, A, Bv):
    """s(0), s(1), s(2) of sum_i A_i B_i over the first (LSB) variable of flattened (3, N) tables."""
    a0, a1 = A[:, 0::2], A[:, 1::2]; b0, b1 = Bv[:, 0::2], Bv[:, 1::2]
    s0 = B.esum(B.emul(a0, b0)); s1 = B.esum(B.emul(a1, b1))
    a2 = B.esub(B.eadd(a1, a1), a0); b2 = B.esub(B.eadd(b1, b1), b0)
    s2 = B.esum(B.emul(a2, b2))
    return [tuple(int(x) for x in s0), tuple(int(x) for x in s1), tuple(int(x) for x in s2)]


def _flat(B, arr3):
    """(3, m, L) -> contiguous (3, m L)."""
    xp = B.xp
    return xp.ascontiguousarray(arr3.reshape(3, -1))


def _absorb_u(tr, u_host):
    tr.absorb(b"u", hashlib.blake2b(np.ascontiguousarray(u_host).tobytes(), digest_size=32).digest())


# ============================================================================ prover
def prove(prm, store, claims, tr, B=CPU, timings=None):
    """Clear-text prover of the 09 per-proof path.  store: Store / MemStore.  Returns the proof dict."""
    T = timings if timings is not None else {}
    xp = B.xp
    tclk = lambda: (_sync(B), time.time())[1]
    t0 = tclk()
    xi_p = tr.exts(prm.ob); zeta_p = tr.exts(prm.ib); lams = tr.exts(len(claims))
    uses, total = _uses(claims, prm, xi_p, zeta_p, lams)
    tiles = sorted({u["tile"] for u in uses})
    tpos = {t: i for i, t in enumerate(tiles)}
    T["n_terms"] = len(uses); T["n_tiles"] = len(tiles)
    # -- touched tiles to the device (in production W^cd stays resident; here read from the int32 store)
    t1 = tclk()
    Wh = np.stack([np.asarray(store.coeffs[t]) for t in tiles]).reshape(len(tiles), prm.TO, prm.TI)
    T["tile_read_s"] = time.time() - t1
    t1 = tclk()
    Wd = xp.ascontiguousarray(B.xp.asarray(Wh)) if _is_gpu(B) else Wh
    T["tile_upload_s"] = tclk() - t1
    del Wh
    # -- phase 1: A_u = W_t h_u, sumcheck over the output-row variables
    t1 = tclk()
    tid = [tpos[u["tile"]] for u in uses]
    H = np.stack([u["h"] for u in uses], axis=1)                      # (2, U, TI)
    A = tiles_matvec(B, Wd, tid, B.asarray(H) if _is_gpu(B) else H)   # (2, U, TO)
    G = np.stack([u["g"] for u in uses], axis=1)
    Gd = _flat(B, B.asarray(G) if _is_gpu(B) else G); Ad = _flat(B, A)
    rounds = []
    claim = total
    xi = []
    for _ in range(prm.ob):
        s = _round3(B, Gd, Ad)
        rounds.append(s); tr.absorb(b"sc1", s)
        r = tr.ext(); xi.append(r)
        Gd = B.fold(Gd, r); Ad = B.fold(Ad, r)
        claim = lagrange_eval(s, r)
    gxi = B.tonumpy(Gd)                                                # (2, U): g~_u(xi)
    T["phase1_s"] = tclk() - t1
    # -- phase 2: Wxi_t = sum_r eq(xi, r) W_t[r, :]; H_t = sum_{u on t} g~_u(xi) h_u
    t1 = tclk()
    eqxi = CPU.eq_table(xi)
    V = np.broadcast_to(eqxi[:, None, :], (3, len(tiles), prm.TO)).copy()
    Wxi = tiles_vecmat(B, Wd, list(range(len(tiles))), B.asarray(V) if _is_gpu(B) else V)   # (3, nt, TI)
    Ht = np.zeros((3, len(tiles), prm.TI), dtype=np.uint64)
    for k, u in enumerate(uses):
        i = tpos[u["tile"]]
        Ht[:, i] = CPU.eadd(Ht[:, i], CPU.escale(u["h"], (int(gxi[0, k]), int(gxi[1, k]), int(gxi[2, k]))))
    Wf = _flat(B, Wxi); Hf = _flat(B, B.asarray(Ht) if _is_gpu(B) else Ht)
    rounds2 = []
    zeta = []
    for _ in range(prm.ib):
        s = _round3(B, Wf, Hf)
        rounds2.append(s); tr.absorb(b"sc2", s)
        r = tr.ext(); zeta.append(r)
        Wf = B.fold(Wf, r); Hf = B.fold(Hf, r)
    cvals = B.tonumpy(Wf)                                              # (3, nt): c_t = W~_t(xi, zeta)
    CL = [(t, (int(cvals[0, i]), int(cvals[1, i]), int(cvals[2, i]))) for i, t in enumerate(tiles)]
    tr.absorb(b"CL", CL)
    T["phase2_s"] = tclk() - t1
    # -- gamma, alpha, u = alpha^T W
    t1 = tclk()
    gam = tr.ext()
    gp = [ONE]
    for _ in range(len(tiles) - 1):
        gp.append(hmul(gp[-1], gam))
    gv = np.array(gp, dtype=np.uint64).T.reshape(3, 1, len(tiles))
    Wrow = Wd.reshape(1, len(tiles), prm.nc)
    u = tiles_vecmat(B, Wrow, [0], B.asarray(gv) if _is_gpu(B) else gv)[:, 0]       # (3, nc)
    u_h = B.tonumpy(u)
    T["u_s"] = tclk() - t1
    del Wd, Wrow, A, Gd, Ad, Wxi, Wf, Hf
    _absorb_u(tr, u_h)
    # -- column positions, openings
    Q = tr.indices(prm.MQ, prm.n)
    t1 = time.time()
    cols = np.stack([store.column(j) for j in Q], axis=1)              # (R_MAX, MQ)
    T["column_read_s"] = time.time() - t1
    t1 = time.time()
    salts = np.ascontiguousarray(np.stack([np.asarray(store.salts[:, j]) for j in Q], axis=1))
    paths = [store.path(j) for j in Q]
    T["path_read_s"] = time.time() - t1
    # -- batched column check <u, g>, g_k = sum_j beta_j w^{jk} (= NTT_n of the sparse beta vector, first n_c values)
    t1 = tclk()
    tr.absorb(b"cols", hashlib.blake2b(cols.tobytes(), digest_size=32).digest())
    beta = tr.exts(prm.MQ)
    bvec = np.zeros((3, prm.n), dtype=np.uint64)
    for j, b_ in zip(Q, beta):
        bvec[0, j] = (int(bvec[0, j]) + b_[0]) % P
        bvec[1, j] = (int(bvec[1, j]) + b_[1]) % P
        bvec[2, j] = (int(bvec[2, j]) + b_[2]) % P
    if _is_gpu(B):
        gq = xp.stack([B.ntt(B.asarray(bvec[l]), prm.n)[:prm.nc] for l in (0, 1, 2)])
        ud = B.asarray(u_h)
    else:
        gq = np.stack([CPU.ntt(bvec[l], prm.n)[:prm.nc] for l in (0, 1, 2)])
        ud = u_h
    gq = xp.ascontiguousarray(gq)
    rounds3 = []
    z = []
    for _ in range(prm.ob + prm.ib):
        s = _round3(B, ud, gq)
        rounds3.append(s); tr.absorb(b"sc3", s)
        r = tr.ext(); z.append(r)
        ud = B.fold(ud, r); gq = B.fold(gq, r)
    T["colcheck_sumcheck_s"] = tclk() - t1
    T["prove_total_s"] = tclk() - t0
    return {"rounds1": rounds, "rounds2": rounds2, "CL": CL, "u": u_h, "Q": Q, "cols": cols,
            "salts": salts, "paths": paths, "rounds3": rounds3}


# ============================================================================ verifier (clear)
class Reject(Exception):
    pass


def gtilde(prm, Q, beta, z):
    """g~(z) for g_k = sum_j beta_j w^{jk}, k < n_c: sum_j beta_j prod_b (1 - z_b + z_b w^{j 2^b}).  O(m_q log n_c)."""
    w = pow(7, (P - 1) // prm.n, P)              # the NTT's n-th root of unity (field.CPU.ntt / GpuBackend.ntt)
    out = ZERO
    for j, b_ in zip(Q, beta):
        acc = b_
        wj = pow(w, j, P)
        for zb in z:
            # 1 - z_b + z_b wj
            term = hadd(hsub(ONE, zb), hmul(zb, (wj, 0, 0)))
            acc = hmul(acc, term)
            wj = wj * wj % P
        out = hadd(out, acc)
    return out


def verify(prm, root, claims, pf, tr, n_used=None, check_columns_direct=True, timings=None):
    """Clear-text verifier.  The activation MLE claims (y~(xi'), h~_u(zeta), u~(.)) are computed here from the vectors
    the outer circuit would hold as hidden committed columns.  Raises Reject on failure."""
    T = timings if timings is not None else {}
    t0 = time.time()
    xi_p = tr.exts(prm.ob); zeta_p = tr.exts(prm.ib); lams = tr.exts(len(claims))
    uses, total = _uses(claims, prm, xi_p, zeta_p, lams)
    # phase 1
    claim = total; xi = []
    for s in pf["rounds1"]:
        if hadd(s[0], s[1]) != claim:
            raise Reject("phase-1 sumcheck round")
        tr.absorb(b"sc1", s); r = tr.ext(); xi.append(r); claim = lagrange_eval(s, r)
    zeta = []
    for s in pf["rounds2"]:
        if hadd(s[0], s[1]) != claim:
            raise Reject("phase-2 sumcheck round")
        tr.absorb(b"sc2", s); r = tr.ext(); zeta.append(r); claim = lagrange_eval(s, r)
    CL = [(int(t), tuple(c)) for t, c in pf["CL"]]
    tiles = [t for t, _ in CL]
    if tiles != sorted(set(tiles)) or set(tiles) != {u["tile"] for u in uses}:
        raise Reject("claim list does not match the touched tiles")
    cmap = dict(CL)
    # final check: claim == sum_u c_t g~_u(xi) h~_u(zeta)
    acc = ZERO
    for u in uses:
        gs = u["gspec"]
        if gs[0] == "eq":
            gx = hmul(gs[1], _eq_pt(xi_p, xi))
        else:
            gx = hmul(gs[1], _eq_pt(_bits_point(gs[2], prm.ob), xi))
        hz = mle(u["h"], zeta) if u["hspec"][0] == "vec" else _eq_pt(zeta_p, zeta)
        acc = hadd(acc, hmul(cmap[u["tile"]], hmul(gx, hz)))
    if acc != claim:
        raise Reject("final sumcheck check (weight claims)")
    tr.absorb(b"CL", pf["CL"])
    gam = tr.ext()
    alpha = {}
    gp = ONE
    lhs = ZERO
    for t, c in CL:
        alpha[t] = gp; lhs = hadd(lhs, hmul(gp, c)); gp = hmul(gp, gam)
    u = np.asarray(pf["u"], dtype=np.uint64)
    if mle(u, list(zeta) + list(xi)) != lhs:
        raise Reject("sum_j gamma^j c_j != u~(xi, zeta)")
    _absorb_u(tr, u)
    Q = tr.indices(prm.MQ, prm.n)
    if list(Q) != list(pf["Q"]):
        raise Reject("column positions")
    cols = np.asarray(pf["cols"], dtype=np.uint64)
    t1 = time.time()
    leaves = leaf_hashes(prm, Q, pf["salts"], cols)
    for k, j in enumerate(Q):
        if not PS.verify_path(root, leaves[k], j, pf["paths"][k]):
            raise Reject(f"Merkle path of column {j}")
    T["merkle_verify_s"] = time.time() - t1
    rows = np.array(tiles, dtype=np.int64)
    av = np.array([alpha[t] for t in tiles], dtype=np.uint64).T        # (3, J)
    # <alpha, M[:, j]> over touched rows (alpha is zero elsewhere)
    rhs = []
    for k in range(len(Q)):
        cj = cols[rows, k]
        rhs.append(CPU.edot_dev(av, CPU.ext(cj)))
    if check_columns_direct:
        t1 = time.time()
        ru = np.stack([CPU.ntt(u[l], prm.n) for l in (0, 1, 2)])
        bad = [j for k, j in enumerate(Q) if (int(ru[0, j]), int(ru[1, j]), int(ru[2, j])) != rhs[k]]
        T["direct_colcheck_s"] = time.time() - t1
        T["bad_columns"] = len(bad)
        if bad:
            raise Reject(f"column check failed on {len(bad)} of {len(Q)} columns")
    # batched column check (in-circuit form): sum_j beta_j RS(u)[j] = <u, g> via a sumcheck, g~ in closed form
    tr.absorb(b"cols", hashlib.blake2b(cols.tobytes(), digest_size=32).digest())
    beta = tr.exts(prm.MQ)
    claim = ZERO
    for b_, r_ in zip(beta, rhs):
        claim = hadd(claim, hmul(b_, r_))
    z = []
    for s in pf["rounds3"]:
        if hadd(s[0], s[1]) != claim:
            raise Reject("column-check sumcheck round")
        tr.absorb(b"sc3", s); r = tr.ext(); z.append(r); claim = lagrange_eval(s, r)
    if hmul(mle(u, z), gtilde(prm, Q, beta, z)) != claim:
        raise Reject("column-check sumcheck final (g~ closed form)")
    T["verify_total_s"] = time.time() - t0
    return True


def _eq_pt(a, b):
    out = ONE
    for x, y in zip(a, b):
        out = hmul(out, hadd(hmul(x, y), hmul(hsub(ONE, x), hsub(ONE, y))))
    return out


# ============================================================================ Qwen3-0.6B driver
def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def cmd_commit(a):
    from .field import gpu_backend
    B = gpu_backend()
    prm = QWEN_PARAMS_R if a.radius else QWEN_PARAMS
    ck = Checkpoint(a.model)
    shapes, aliases = model_shapes(ck)
    L = Layout(shapes, prm, radius=a.radius, aliases=aliases)
    print(f"aliases (committed once): {aliases}", flush=True)
    print(f"layout: {len(L.mats)} matrices, {len(L.vecs)} vectors, rows used {L.n_rows} (raw {L.raw_rows}, cfg {L.cfg_rows}) of R_MAX {prm.R_MAX}", flush=True)
    os.makedirs(a.store, exist_ok=True)
    res = {"model": os.path.basename(os.path.normpath(a.model)), "machine": a.machine or describe_machine(), "params": asdict(prm),
           "rows_used": L.n_rows, "raw_rows": L.raw_rows, "cfg_rows": bool(L.cfg_rows),
           "n_params": int(sum(int(np.prod(s)) for s in shapes.values())), "aliases": aliases}
    t0 = time.time()
    shared = {}
    srcW = RowSource(ck, L, raw=False, shared=shared)
    srcW.prepare()
    cw = commit_matrix(srcW, L.n_span, prm, B, store_dir=a.store, tag="W", store_coeffs=True, store_cw=a.store_cw,
                       row_ranges=L.row_ranges)
    res["W"] = {"n_rows": L.n_rows, "radius_rows": L.n_tiles if a.radius else 0, "root": list(cw["root"]),
                **{k: v for k, v in cw["stats"].items() if k != "codeword_files"}}
    print(json.dumps(res["W"]), flush=True)
    srcR = RowSource(ck, L, raw=True, shared=shared)
    cr = commit_matrix(srcR, L.raw_rows, prm, B, store_dir=a.store, tag="RAW", store_coeffs=True,
                       store_cw=a.store_cw and a.store_raw)
    res["RAW"] = {"n_rows": L.raw_rows, "root": list(cr["root"]), "codeword_stored": bool(a.store_cw and a.store_raw),
                  **{k: v for k, v in cr["stats"].items() if k != "codeword_files"}}
    print(json.dumps(res["RAW"]), flush=True)
    s0 = tuple(int(x) for x in rand_field(4)); scfg = tuple(int(x) for x in rand_field(4))
    kprf = tuple(int(x) for x in rand_field(4))
    vol_elems = encode_volatile_cfg({
        "k_prf": kprf, "salt_cfg": scfg,
        "L_real": int(ck.config.get("num_hidden_layers", 0)),
        "V_real": int(ck.config.get("vocab_size", 0)),
    })
    ccfg = volatile_config_root(vol_elems) if L.cfg_rows else config_root(L, {"hf_config": ck.config, "aliases": aliases}, scfg)
    mid = model_id(s0, cw["root"], cr["root"], ccfg)
    res["C_cfg"] = list(ccfg); res["model_id"] = list(mid)
    res["commit_total_s"] = time.time() - t0
    G = shared["G"]
    Gd = G[L.n_meta:L.n_rows]
    res["cd_row_exponents"] = {"min": int(Gd.min()), "max": int(Gd.max()),
                               "hist": {int(g): int((Gd == g).sum()) for g in np.unique(Gd)}}
    del shared
    res["peak_rss_gib"] = round(_rss_gib(), 2)
    # prover-private: the model_id preimage (flashprove_next.mc_zk opens it in-circuit); never part of the published result
    secret = {"s0": list(s0), "c_raw": list(cr["root"]), "c_cfg": list(ccfg),
              "vol_elems": [int(x) for x in vol_elems] if L.cfg_rows else None}
    json.dump({"W": {"n_rows": L.n_rows, "row_ranges": [list(r) for r in L.row_ranges], "radius": bool(a.radius),
                     "cfg_rows": bool(L.cfg_rows)},
               "RAW": {"n_rows": L.raw_rows}, "result": res, "secret": secret},
              open(os.path.join(a.store, "commit.json"), "w"), indent=1)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res), flush=True)


def _sample_blocks(L, k1, rng):
    """K1 GEMM blocks: (tensor) drawn with probability proportional to its size (cost weight), any layer."""
    mats = [m for m in L.mats if "layers." in m]
    sizes = np.array([np.prod(L.shapes[m]) for m in mats], dtype=np.float64)
    pick = rng.choice(len(mats), size=k1, p=sizes / sizes.sum())
    return [mats[i] for i in pick]


def build_claims(L, store, prm, k1, k2, rng, B):
    """Synthetic but exact witness: random integer activations (|a| < 2^24), outputs computed from W^cd in F_p."""
    claims = []
    emb = next(m for m in L.mats if "embed" in m)
    blocks = _sample_blocks(L, k1, rng)
    Wcache = {}

    def tile_dev(t):
        if t not in Wcache:
            w = np.asarray(store.coeffs[t]).reshape(1, prm.TO, prm.TI)
            Wcache[t] = B.xp.asarray(w) if _is_gpu(B) else w
        return Wcache[t]

    def gemv(t, h):
        H = _e(h).reshape(3, 1, prm.TI)
        return B.tonumpy(tiles_matvec(B, tile_dev(t), [0], B.asarray(H) if _is_gpu(B) else H))[:, 0]

    for name in blocks:
        o, i = L.shapes[name]
        nR, nC = -(-o // prm.TO), -(-i // prm.TI)
        a = rng.integers(-(1 << 24) + 1, 1 << 24, size=nC * prm.TI)
        a[i:] = 0
        for R in range(nR):
            ts = [L.tile_of[(name, R, C)] for C in range(nC)]
            hs = [a[C * prm.TI:(C + 1) * prm.TI] for C in range(nC)]
            y = np.zeros((3, prm.TO), dtype=np.uint64)
            for t, h in zip(ts, hs):
                y = CPU.eadd(y, gemv(t, h))
            claims.append(GemmClaim(ts, hs, y))
        Wcache.clear()
    # LM head: K2 steps, logits z_s = E h_s; batched by r_s (drawn after the trace is fixed; here a fixed seed stands in)
    V, d = L.shapes[emb]
    nR = -(-V // prm.TO)
    hsteps = rng.integers(-(1 << 24) + 1, 1 << 24, size=(k2, d))
    rs = [(int(x) % P, int(y) % P, int(z) % P) for x, y, z in rng.integers(0, 1 << 62, size=(k2, 3))]
    hbar = np.zeros((3, prm.TI), dtype=np.uint64)
    for s in range(k2):
        hbar = CPU.eadd(hbar, CPU.escale(_e(hsteps[s]), rs[s]))
    for R in range(nR):
        t = L.tile_of[(emb, R, 0)]
        y = gemv(t, hbar)          # = sum_s r_s z_s restricted to vocabulary tile R (the trace holds the z_s)
        claims.append(GemmClaim([t], [hbar], y))
        Wcache.clear()
    # embedding lookups of K2 sampled tokens
    for tok in rng.integers(0, V, size=k2):
        t = L.tile_of[(emb, int(tok) // prm.TO, 0)]
        x = np.asarray(store.coeffs[t]).reshape(prm.TO, prm.TI)[int(tok) % prm.TO].astype(np.int64)
        claims.append(LookupClaim(t, int(tok) % prm.TO, x))
    return claims, blocks


def cmd_prove(a):
    from .field import gpu_backend
    import cupy as cp
    B = gpu_backend()
    prm = QWEN_PARAMS
    ck = Checkpoint(a.model)
    L = Layout(model_shapes(ck)[0], prm)
    store = Store(a.store, prm, "W")
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    claims, blocks = build_claims(L, store, prm, a.k1, a.k2, rng, B)
    t_wit = time.time() - t0
    out = {"model": os.path.basename(os.path.normpath(a.model)), "machine": a.machine or describe_machine(), "k1": a.k1, "k2": a.k2,
           "params": asdict(prm), "witness_build_s": round(t_wit, 2), "n_claims": len(claims)}
    runs = []
    for rep in range(a.reps):
        if a.cold:
            # posix_fadvise(DONTNEED) evicts the codeword pages only while no mapping holds them: in practice only the
            # first repetition reads the columns from disk (later ones hit the page cache)
            store.drop_cache()
        cp.get_default_memory_pool().free_all_blocks()
        mp = cp.get_default_memory_pool()
        T = {"rep": rep, "cold_cache": bool(a.cold and rep == 0)}
        trp = Transcript(b"mc09"); trp.absorb(list(store.root))
        pf = prove(prm, store, claims, trp, B, T)
        T["gpu_pool_peak_gib"] = round(mp.total_bytes() / (1 << 30), 2)
        trv = Transcript(b"mc09"); trv.absorb(list(store.root))
        V = {}
        ok = verify(prm, store.root, claims, pf, trv, timings=V)
        T.update({"verify_" + k if not k.startswith("verify") else k: v for k, v in V.items()})
        T["accepted"] = ok
        runs.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in T.items()})
        print(json.dumps(runs[-1]), flush=True)
    out["runs"] = runs
    out["distinct_tiles"] = runs[0]["n_tiles"]
    out["sampled_blocks_by_kind"] = {k: int(v) for k, v in zip(*np.unique([b.split(".")[-2] for b in blocks], return_counts=True))}
    out["layers_hit"] = len({b.split(".")[2] for b in blocks})
    out["opening_payload_bytes"] = int(prm.MQ * (prm.R_MAX * 8 + 32 + prm.depth * 32) + prm.nc * 24)
    out["peak_rss_gib"] = round(_rss_gib(), 2)
    # one forged proof: the prover computes with an int4-like requantised copy of one touched tile
    if a.forge:
        out["forgery"] = forge_test(prm, store, claims, B)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}), flush=True)


class _ForgedStore:
    """Wraps a store so that one row's coefficients differ (the cheating prover's weights)."""

    def __init__(self, base, row, new):
        self.base, self.row, self.new = base, row, new
        self.prm, self.n_used, self.salts, self.root = base.prm, base.n_used, base.salts, base.root

        class _C:
            def __getitem__(s, t):
                return self.new if t == self.row else base.coeffs[t]
        self.coeffs = _C()

    def column(self, j): return self.base.column(j)
    def path(self, j): return self.base.path(j)


def requant(w, bits=4):
    """Symmetric per-tile requantisation to `bits` bits, mapped back to the integer grid (a cheap model)."""
    w = np.asarray(w, dtype=np.int64)
    s = max(1, int(np.abs(w).max()) // ((1 << (bits - 1)) - 1))
    return (np.round(w / s) * s).astype(np.int32)


def forge_test(prm, store, claims, B):
    """Cheating prover: its trace (claims) is computed with requantised weights of one tile; it runs the protocol
    with those weights.  Expected: the column check rejects (u is the combination of the forged rows)."""
    t = claims[0].tiles[0] if isinstance(claims[0], GemmClaim) else claims[0].tile
    forged = requant(np.asarray(store.coeffs[t]))
    fs = _ForgedStore(store, t, forged)
    # recompute the claims that use tile t with the forged weights
    fcl = []
    Wf = forged.reshape(prm.TO, prm.TI).astype(np.int64)
    for cl in claims:
        if isinstance(cl, GemmClaim) and t in cl.tiles:
            y = np.zeros((3, prm.TO), dtype=np.uint64)
            for tt, h in zip(cl.tiles, cl.hs):
                Wt = Wf if tt == t else np.asarray(store.coeffs[tt]).reshape(prm.TO, prm.TI)
                Wd = B.xp.asarray(Wt.reshape(1, prm.TO, prm.TI).astype(np.int32)) if _is_gpu(B) else Wt.reshape(1, prm.TO, prm.TI).astype(np.int32)
                H = _e(h).reshape(3, 1, prm.TI)
                y = CPU.eadd(y, B.tonumpy(tiles_matvec(B, Wd, [0], B.asarray(H) if _is_gpu(B) else H))[:, 0])
            fcl.append(GemmClaim(cl.tiles, cl.hs, y))
        elif isinstance(cl, LookupClaim) and cl.tile == t:
            fcl.append(LookupClaim(t, cl.r0, Wf[cl.r0]))
        else:
            fcl.append(cl)
    trp = Transcript(b"mc09"); trp.absorb(list(store.root))
    pf = prove(prm, fs, fcl, trp, B, {})
    trv = Transcript(b"mc09"); trv.absorb(list(store.root))
    V = {}
    try:
        verify(prm, store.root, fcl, pf, trv, timings=V)
        res = "ACCEPTED (bad)"
    except Reject as e:
        res = "rejected: " + str(e)
    return {"tile": int(t), "changed_entries": int((forged != np.asarray(store.coeffs[t])).sum()), "result": res,
            "bad_columns": V.get("bad_columns")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit")
    c.add_argument("--model", required=True); c.add_argument("--store", required=True)
    c.add_argument("--store-raw", action="store_true"); c.add_argument("--out", default=None)
    c.add_argument("--store-cw", action="store_true", help="also write the codeword row-group files (needed by mc_zk/wsum_zk)")
    c.add_argument("--radius", action="store_true", help="also commit W^r at rows R_MAX/2 + t (R_MAX = 2048)")
    c.add_argument("--machine", default=None, help="host description (default: detected)")
    p = sub.add_parser("prove")
    p.add_argument("--model", required=True); p.add_argument("--store", required=True)
    p.add_argument("--k1", type=int, default=78); p.add_argument("--k2", type=int, default=32)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--reps", type=int, default=2)
    p.add_argument("--cold", action="store_true"); p.add_argument("--forge", action="store_true")
    p.add_argument("--out", default=None); p.add_argument("--machine", default=None, help="host description (default: detected)")
    a = ap.parse_args(argv)
    {"commit": cmd_commit, "prove": cmd_prove}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
