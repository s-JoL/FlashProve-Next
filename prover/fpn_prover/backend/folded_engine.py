# SPDX-License-Identifier: Apache-2.0
"""Unified Folded ZK Engine (`folded_engine`) for `LLM_INFERENCE_V2` (`FPN2` Relation 2).

Replaces the legacy Stage-1 `zk_flat` prover/verifier (Phase-2 LogUp commitments, 168-query rate-1/4
per-group Basefold, JSON transcripts, 149.8 MB proofs) with ONE unified high-speed folded backend:

  1. Raw-byte `Fpn2Transcript` (BLAKE2b-512, specs/11 §4) for all core engine messages;
  2. ZERO Phase-2 LogUp inverse commitments: ALL `cir.lookups` (range checks, BF16/rsqrt/exp/Gumbel/RoPE
     tables, token chain, weighted HCO LogUp) are packed into ONE ZK-masked `multi_gkr` binary fraction
     tree whose leaf MLE evaluations `(num_eval, den_eval)` are bound directly to Phase-1 committed
     vectors (and tiny degree-2 selector sumzeros when `sel.degree() > 1`);
  3. ONE batched rate-1/8 coset-4 folded ZK-Basefold opening (`multi_pcs`) covering EVERY committed group
     across Phase 1, interactive `cir.rounds`, and the tiny Mask group in a single fold chain -> compact
     fixed-structure proofs (<= 4-8 MB even at 100M+ cells) with fast CPU verification;
  4. Seamless CPU / GPU (`CudaEngine`) execution with bit-identical proofs and transcripts.
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass

import numpy as np

from flashprove_next import zk_flat as Z
from flashprove_next.field import (
    CPU, ONE, P, ZERO, eq1, hadd, hf, hinv, hmul, hneg, hsub, lagrange_eval,
)

from . import cuda_engine as CE
from . import multi_gkr as MG
from . import multi_pcs as MP
from .fpn2 import Fpn2Transcript, dec3, enc3, enc3s

DEFAULT_QUERIES = 100       # rate-1/8 coset-4 Basefold queries (production default)
DEFAULT_GKR_MASKS = 16      # cancelling mask leaf pairs in the unified GKR tree
MIN_PRODUCTION_QUERIES = 72 # rate-1/8 coset-4 query floor (~104-bit soundness; matches Relation 2 STANDARD)
MIN_GKR_MASKS = 2           # >= 1 ordinary numerator mask + 1 zero-num denominator blinder slot (BLOCKER #1)


class FoldedEngineError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# ----------------------------------------------------------------------------- Transcript adapter
class Fpn2TranscriptAdapter:
    """Adapter presenting the 4-method `zk_flat.Transcript` interface (`absorb`, `ext`, `exts`, `indices`)
    on top of the normative raw-byte `Fpn2Transcript` (BLAKE2b-512, specs/11 §4)."""

    def __init__(self, label: bytes = b"FPN2/llm-v2"):
        self.tr = Fpn2Transcript(label)

    @property
    def state(self) -> bytes:
        return self.tr.state

    def absorb_bytes(self, data: bytes):
        self.tr.absorb_bytes(data)

    def squeeze_fp3(self):
        return self.tr.squeeze_fp3()

    def squeeze_bytes(self, n: int) -> bytes:
        return self.tr.squeeze_bytes(n)

    # Compatibility surface for cir.rounds callbacks (hsel_zk, wsum_zk, mc_zk, attn_zk, gen_zk, api_zk)
    def absorb(self, *parts):
        from flashprove_next.transcript import canon
        for p in parts:
            if isinstance(p, (bytes, bytearray)):
                self.tr.absorb_bytes(b"B" + bytes(p))
            elif isinstance(p, np.ndarray):
                self.tr.absorb_bytes(b"A" + np.ascontiguousarray(p).tobytes())
            else:
                self.tr.absorb_bytes(b"J" + canon(p))

    def ext(self):
        return self.tr.squeeze_fp3()

    def exts(self, k: int):
        return [self.tr.squeeze_fp3() for _ in range(k)]

    def indices(self, t: int, n: int):
        out = []
        for _ in range(t):
            raw = self.tr.squeeze_bytes(8)
            out.append(int.from_bytes(raw, "little") % n)
        return out


# ----------------------------------------------------------------------------- Aligned group packing
@dataclass(frozen=True)
class VecPlacement:
    vec_id: int
    group_idx: int
    offset: int         # aligned offset in [0, data_capacity), offset % (1 << nbits) == 0
    nbits: int


def choose_group_nbits(vecs, queries: int, min_n: int = 10, max_n: int = 26) -> int:
    """Choose an even `n_group` such that every vector has `v.nbits <= n_group - 1` and
    large circuits pack into fewer groups (min_n=24 at >=32M cells, min_n=26 at >=256M cells)."""
    max_vb = max((v.nbits for v in vecs), default=0)
    tot_cells = sum(1 << v.nbits for v in vecs)
    if tot_cells >= (1 << 28):
        min_n = max(min_n, 26)
    elif tot_cells >= (1 << 25):
        min_n = max(min_n, 24)
    n = max(min_n, max_vb + 1)
    if n % 2:
        n += 1
    while n <= max_n:
        lay = MP.MultiPcsLayout(n=n, n_groups=1, queries=queries)
        if lay.data_capacity >= (1 << max_vb):
            return n
        n += 2
    raise ValueError(f"cannot fit vector of 2^{max_vb} cells within max_n={max_n}")


def pack_vectors_into_groups(vecs, n_group: int, data_capacity: int, start_group_idx: int = 0):
    """Deterministically pack `vecs` (largest-first, power-of-two aligned) into groups of capacity
    `data_capacity` inside `2^n_group`.

    Returns (group_lists, placements_dict) where:
      - `group_lists` is a list of lists of `(vec, offset)` pairs (1 per new group);
      - `placements_dict` maps `vec.id -> VecPlacement`.
    """
    if not vecs:
        return [], {}
    order = sorted(vecs, key=lambda v: (-v.nbits, v.id))
    groups = []     # list of [cur_off, [(vec, off), ...]]
    placements = {}
    for v in order:
        sz = 1 << v.nbits
        if sz > data_capacity:
            raise ValueError(f"vector {v.name} (2^{v.nbits}) exceeds group data_capacity={data_capacity}")
        placed = False
        for gi, g in enumerate(groups):
            off = (g[0] + sz - 1) // sz * sz
            if off + sz <= data_capacity:
                g[1].append((v, off))
                g[0] = off + sz
                placements[v.id] = VecPlacement(v.id, start_group_idx + gi, off, v.nbits)
                placed = True
                break
        if not placed:
            gi = len(groups)
            groups.append([sz, [(v, 0)]])
            placements[v.id] = VecPlacement(v.id, start_group_idx + gi, 0, v.nbits)
    return [g[1] for g in groups], placements


def _smallest_group_n(used_len: int, queries: int, n_max: int) -> int:
    """Smallest even n_g in [8, n_max] with `group_data_capacity(n_g, queries) >= used_len`."""
    tl = MP.tail_len_for(queries)
    n = 8
    while n < n_max and (1 << n) - tl < used_len:
        n += 2
    if (1 << n) - tl < used_len:
        raise ValueError(f"used_len={used_len} exceeds capacity at n_max={n_max}")
    return n


def pack_phase_adaptive(vecs, n_cap: int, queries: int, start_group_idx: int = 0):
    """Deterministically pack `vecs` into aligned groups, shrinking each group to its smallest fitting
    even `n_g` (and automatically widening `n_cap` by +2 if an interactive commit round emits a vector
    larger than the Phase-1 `n_cap`, e.g. `wsW.w` of 2^24 on 4B-class models).

    Returns `(group_lists, placements_dict, group_ns)`.
    """
    if not vecs:
        return [], {}, []
    max_vb = max(v.nbits for v in vecs)
    eff_cap = int(n_cap)
    while MP.group_data_capacity(eff_cap, queries) < (1 << max_vb):
        eff_cap += 2
        if eff_cap > 28:
            raise ValueError(f"vector 2^{max_vb} exceeds max group size 2^28")
    cap_max = MP.group_data_capacity(eff_cap, queries)
    g_lists, placements = pack_vectors_into_groups(vecs, eff_cap, cap_max, start_group_idx=start_group_idx)
    g_ns = []
    for items in g_lists:
        used = max(off + (1 << v.nbits) for v, off in items)
        g_ns.append(_smallest_group_n(used, queries, eff_cap))
    return g_lists, placements, g_ns


def _materialize_group_cells(group_items, data_capacity: int) -> np.ndarray:
    """Assemble host `(data_capacity, 3)` uint64 array for one committed group from its `(Vec, offset)` list."""
    cells = np.zeros((data_capacity, 3), dtype=np.uint64)
    for v, off in group_items:
        sz = 1 << v.nbits
        if v.is_ext:
            # _ext_rows returns (3, content_or_sz) uint64
            arr = Z._ext_rows(v, 0, sz)
            n_arr = arr.shape[1]
            cells[off:off + n_arr, 0] = arr[0]
            cells[off:off + n_arr, 1] = arr[1]
            cells[off:off + n_arr, 2] = arr[2]
            if n_arr < sz:
                cells[off + n_arr:off + sz] = MP._os_rand_fp3(sz - n_arr)
            elif v.content < sz:
                cells[off + v.content:off + sz] = MP._os_rand_fp3(sz - v.content)
        else:
            raw = Z.wit(v)
            if raw is None:
                raise FoldedEngineError("E_WITNESS", f"missing witness values for {v.name}")
            u = Z.field_u64(raw)
            cells[off:off + len(u), 0] = u
            # Note: any content..sz tail of the Vec itself stays 0 (group-level OS-CSPRNG tail blinds PCS!)
            # Wait: if a vector is queried directly in sumcheck at random r, its own tail rows (if any) are
            # filled with fresh OS-CSPRNG blinding so individual sumcheck claim values `v~(r)` are ALSO hidden!
            if v.content < sz:
                tail_n = sz - v.content
                cells[off + v.content:off + sz] = MP._os_rand_fp3(tail_n)
    return cells


# ----------------------------------------------------------------------------- Unified GKR Lookup Packing
@dataclass(frozen=True)
class GkrBlock:
    kind: str           # "T" (table) or "V" (value part)
    tid: int
    name: str
    part_idx: int       # -1 for table, >= 0 for value part
    nbits: int
    offset: int         # power-of-two aligned leaf offset in [0, 2^s)
    cols: tuple         # tuple of Ex (all linear, degree <= 1)
    num_ex: object      # Ex for numerator (None => constant 1 for V; table uses -View(mult)*SelLt)
    is_linear_num: bool # True if numerator is constant 1 or degree-1 Ex (no sumzero needed)
    mult_vec: object = None
    raw_vals: object = None
    raw_sel: object = None


def _pack_blocks_into_gkr(raw_blocks, m_mask: int, max_s: int = 28):
    """Try to pack `raw_blocks` (already sorted largest-first) into one `MultiGkrLayout` with `s <= max_s`.
    Returns `(layout, blocks)` or `None` if it doesn't fit within `max_s`."""
    if not raw_blocks:
        return None
    max_nb = raw_blocks[0][4]
    total_cells = sum(1 << b[4] for b in raw_blocks)
    s = max(4, max_nb + 1, (total_cells + 2 * m_mask - 1).bit_length())
    while s <= max_s:
        half = 1 << (s - 1)
        m_use = min(m_mask, 1 << (s - 2))
        lim_l = half - m_use
        lim_r = (1 << s) - m_use
        cur_l, cur_r = 0, half
        placed = []
        ok = True
        for b in raw_blocks:
            sz = 1 << b[4]
            off_l = (cur_l + sz - 1) // sz * sz
            if off_l + sz <= lim_l:
                placed.append((b, off_l))
                cur_l = off_l + sz
            else:
                off_r = (cur_r + sz - 1) // sz * sz
                if off_r + sz <= lim_r:
                    placed.append((b, off_r))
                    cur_r = off_r + sz
                else:
                    ok = False
                    break
        if ok:
            lay = MG.MultiGkrLayout(s=s, m_mask=m_use)
            blocks = [
                GkrBlock(
                    kind=b[0], tid=b[1], name=b[2], part_idx=b[3], nbits=b[4], offset=off,
                    cols=b[5], num_ex=b[6], is_linear_num=b[7], mult_vec=b[8],
                    raw_vals=b[9], raw_sel=b[10],
                )
                for b, off in placed
            ]
            return lay, blocks
        s += 1
    return None


def plan_unified_gkr_batches(cir, m_mask: int = DEFAULT_GKR_MASKS, max_s: int = 26):
    """Deterministically partition `cir.lookups` (each lookup table is self-contained!) into 1 or more
    `MultiGkrLayout` batches with height `s <= max_s` (bounded GPU VRAM even at 1B+ leaves!).

    Returns `list[(MultiGkrLayout, list[GkrBlock])]` (empty if `cir.lookups` is empty).
    """
    if not cir.lookups:
        return []

    per_table = []
    for tid, name in enumerate(sorted(cir.lookups)):
        lk = cir.lookups[name]
        if lk.table is None or not lk.parts:
            raise FoldedEngineError("E_LOOKUP", f"lookup {name}: table or parts missing")
        nvt, tcols = lk.table
        for c in tcols:
            if c.degree() > 1:
                raise FoldedEngineError("E_LOOKUP", f"lookup {name}: non-linear table column (deg {c.degree()})")
        mv = lk.mult
        if mv.content == (1 << nvt):
            num_ex = Z.Ex.atom(Z.View(mv)) * Z.Ex.const(P - 1)
            lin_num = True
        else:
            num_ex = Z.Ex.atom(Z.View(mv)) * Z.Ex.atom(Z.SelLt(list(range(nvt)), mv.content)) * Z.Ex.const(P - 1)
            lin_num = False
        tbl_blocks = [("T", tid, name, -1, nvt, tuple(tcols), num_ex, lin_num, mv, None, None)]
        for pi, (nvp, cols, sel, vals, sel_vals) in enumerate(lk.parts):
            for c in cols:
                if c.degree() > 1:
                    raise FoldedEngineError("E_LOOKUP", f"lookup {name} part {pi}: non-linear col (deg {c.degree()})")
            lin_num_p = (sel is None) or (sel.degree() <= 1)
            tbl_blocks.append(("V", tid, name, pi, nvp, tuple(cols), sel, lin_num_p, None, vals, sel_vals))
        tbl_cells = sum(1 << b[4] for b in tbl_blocks)
        per_table.append((tbl_cells, tid, tbl_blocks))

    # Fast path: try packing all tables into ONE batch of height <= max_s
    all_raw = sorted([b for _, _, bs in per_table for b in bs], key=lambda b: (-b[4], b[1], b[3]))
    single = _pack_blocks_into_gkr(all_raw, m_mask, max_s=max_s)
    if single is not None:
        return [single]

    # Multi-batch greedy bin packing (largest tables first, keeping each table intact!)
    per_table.sort(key=lambda x: (-x[0], x[1]))
    batches_raw = []  # list of list[raw_block]
    for _cells, _tid, bs in per_table:
        bs_sorted = sorted(bs, key=lambda b: (-b[4], b[1], b[3]))
        placed = False
        for bi in range(len(batches_raw)):
            cand = sorted(batches_raw[bi] + bs_sorted, key=lambda b: (-b[4], b[1], b[3]))
            if _pack_blocks_into_gkr(cand, m_mask, max_s=max_s) is not None:
                batches_raw[bi] = cand
                placed = True
                break
        if not placed:
            batches_raw.append(bs_sorted)

    out = []
    for br in batches_raw:
        packed = _pack_blocks_into_gkr(br, m_mask, max_s=28)
        if packed is None:
            raise FoldedEngineError("E_LOOKUP", "single lookup table exceeds maximum GKR height s=28")
        out.append(packed)
    return out


def plan_unified_gkr(cir, m_mask: int = DEFAULT_GKR_MASKS):
    """Legacy single-batch wrapper (returns `(lay, blocks)` or `(None, [])`)."""
    batches = plan_unified_gkr_batches(cir, m_mask=m_mask, max_s=28)
    if not batches:
        return None, []
    return batches[0]


def _eval_ex_on_cube(ex, nbits: int, B=CPU) -> np.ndarray:
    """Evaluate an `Ex` (base or E) over all 2^nbits points of its cube, returning `(2^nbits, 3)` uint64 on host
    (or device array when `B` is GPU if `return_dev=True`). Uses `B` (CPU or GPU) for vectorized evaluation."""
    N = 1 << nbits
    vcache = {}
    tabs = Z._materialize(B, ex, nbits, vcache)
    out = B.zeros_e(N)
    for c, f in ex.terms:
        if not f:
            if c != ZERO:
                out = B.eaddc(out, c)
            continue
        acc = None
        for l, p in f:
            off, arr = tabs[l.key]
            if arr.shape[1] == N and off == 0:
                full = arr
            else:
                full = B.zeros_e(N)
                full[:, off:off + arr.shape[1]] = arr
            cur = full
            for _ in range(p - 1):
                cur = B.emul(cur, full)
            acc = cur if acc is None else B.emul(acc, cur)
        if c != ONE:
            acc = B.escale(acc, c)
        out = B.eadd(out, acc)
    if B is not CPU:
        out_h = B.tonumpy(out)
        return np.ascontiguousarray(out_h.T, dtype=np.uint64)
    return np.ascontiguousarray(out.T, dtype=np.uint64)


def _eval_ex_on_cube_dev(ex, nbits: int, B):
    """Evaluate an `Ex` over 2^nbits points directly on backend `B`, returning `(3, 2^nbits)` array on `B`."""
    N = 1 << nbits
    vcache = {}
    tabs = Z._materialize(B, ex, nbits, vcache)
    out = B.zeros_e(N)
    for c, f in ex.terms:
        if not f:
            if c != ZERO:
                out = B.eaddc(out, c)
            continue
        acc = None
        for l, p in f:
            off, arr = tabs[l.key]
            if arr.shape[1] == N and off == 0:
                full = arr
            else:
                full = B.zeros_e(N)
                full[:, off:off + arr.shape[1]] = arr
            cur = full
            for _ in range(p - 1):
                cur = B.emul(cur, full)
            acc = cur if acc is None else B.emul(acc, cur)
        if c != ONE:
            acc = B.escale(acc, c)
        out = B.eadd(out, acc)
    return out


def _eval_ex_on_cube_host(ex, nbits: int, B=CPU) -> np.ndarray:
    return _eval_ex_on_cube(ex, nbits, B=B)


def _block_ind_at(off: int, nbits: int, r_gkr) -> tuple[int, int, int]:
    """eq indicator on the high bits [nbits .. s-1] of `r_gkr` for aligned offset `off`."""
    acc = ONE
    for b in range(nbits, len(r_gkr)):
        rb = r_gkr[b]
        acc = hmul(acc, rb if (off >> b) & 1 else hsub(ONE, rb))
    return acc


def _build_gkr_leaves_direct(
    gkr_lay: MG.MultiGkrLayout, blocks: list[GkrBlock], gamma_lu, beta_lu, mask_a, mask_d, *, B=CPU, engine=None,
):
    """Build the full `(2^s, 6)` uint64 leaf array directly at aligned block offsets (GPU-accelerated when B!=CPU)."""
    N = gkr_lay.n_leaves
    use_dev = engine is not None and hasattr(engine, "cp") and B is not CPU
    if use_dev:
        cp = engine.cp
        leaves = cp.zeros((N, 6), dtype=cp.uint64)
        leaves[:, 3] = 1
        for blk in blocks:
            sz = 1 << blk.nbits
            off = blk.offset
            if blk.num_ex is None:
                leaves[off:off + sz, 0] = 1
                leaves[off:off + sz, 1:3] = 0
            else:
                num_d = _eval_ex_on_cube_dev(blk.num_ex, blk.nbits, B)  # (3, sz)
                leaves[off:off + sz, 0] = num_d[0]
                leaves[off:off + sz, 1] = num_d[1]
                leaves[off:off + sz, 2] = num_d[2]
            fp = B.zeros_e(sz)
            if blk.tid % P:
                fp[0, :] = np.uint64(blk.tid % P)
            bp = beta_lu
            for col_ex in blk.cols:
                c_d = _eval_ex_on_cube_dev(col_ex, blk.nbits, B)
                fp = B.eadd(fp, B.escale(c_d, bp))
                bp = hmul(bp, beta_lu)
            gam_d = cp.empty((3, sz), dtype=cp.uint64)
            gam_d[0, :] = np.uint64(gamma_lu[0])
            gam_d[1, :] = np.uint64(gamma_lu[1])
            gam_d[2, :] = np.uint64(gamma_lu[2])
            den_d = B.esub(gam_d, fp)
            leaves[off:off + sz, 3] = den_d[0]
            leaves[off:off + sz, 4] = den_d[1]
            leaves[off:off + sz, 5] = den_d[2]
        left_idx, right_idx = gkr_lay.mask_leaf_indices
        mask_l = np.zeros((gkr_lay.m_mask, 6), dtype=np.uint64)
        mask_r = np.zeros((gkr_lay.m_mask, 6), dtype=np.uint64)
        last = gkr_lay.m_mask - 1
        for m in range(gkr_lay.m_mask):
            am = tuple(int(v) for v in mask_a[m])
            dm = tuple(int(v) for v in mask_d[m])
            if m < last:
                mask_l[m, :3], mask_l[m, 3:] = am, dm
                mask_r[m, :3], mask_r[m, 3:] = hsub(ZERO, am), dm
            else:
                mask_l[m, :3], mask_l[m, 3:] = ZERO, dm
                mask_r[m, :3], mask_r[m, 3:] = ZERO, am
        leaves[left_idx[0]:left_idx[-1] + 1] = cp.asarray(mask_l)
        leaves[right_idx[0]:right_idx[-1] + 1] = cp.asarray(mask_r)
        return leaves

    leaves = np.zeros((N, 6), dtype=np.uint64)
    leaves[:, 3] = 1  # default unused slots: (0, 1)

    for blk in blocks:
        sz = 1 << blk.nbits
        off = blk.offset
        if blk.num_ex is None:
            num = np.zeros((sz, 3), dtype=np.uint64)
            num[:, 0] = 1
        else:
            num = _eval_ex_on_cube_host(blk.num_ex, blk.nbits)

        fp = np.zeros((3, sz), dtype=np.uint64)
        fp[0, :] = blk.tid % P
        bp = beta_lu
        for col_ex in blk.cols:
            c_arr = _eval_ex_on_cube_host(col_ex, blk.nbits).T
            fp = CPU.eadd(fp, CPU.escale(c_arr, bp))
            bp = hmul(bp, beta_lu)
        gam_arr = np.empty((3, sz), dtype=np.uint64)
        for k in range(3):
            gam_arr[k, :] = gamma_lu[k]
        den = CPU.esub(gam_arr, fp).T

        leaves[off:off + sz, :3] = num
        leaves[off:off + sz, 3:] = den

    left_idx, right_idx = gkr_lay.mask_leaf_indices
    last = gkr_lay.m_mask - 1
    for m in range(gkr_lay.m_mask):
        am = tuple(int(v) for v in mask_a[m])
        dm = tuple(int(v) for v in mask_d[m])
        if m < last:
            leaves[left_idx[m], :3] = am
            leaves[left_idx[m], 3:] = dm
            leaves[right_idx[m], :3] = hsub(ZERO, am)
            leaves[right_idx[m], 3:] = dm
        else:
            leaves[left_idx[m], :3] = ZERO
            leaves[left_idx[m], 3:] = dm
            leaves[right_idx[m], :3] = ZERO
            leaves[right_idx[m], 3:] = am
    return leaves


def _eval_linear_ex_at_pt(ex, r_loc, claim_recorder, prover_eval_fn=None):
    """Evaluate a degree-<=1 `Ex` at `r_loc`, recording every `View` atom into `claim_recorder`."""
    atoms = Z._view_atoms(ex)
    cv = {}
    for a in atoms:
        val = claim_recorder(a, r_loc, prover_eval_fn)
        cv[a.kid] = val
    return Z._eval_terms(ex, r_loc, cv)


# ----------------------------------------------------------------------------- Claim weight builders
def _claim_weight_table(spec, nbits: int):
    """Build local `(2^nbits, 3)` uint64 weight vector for one claim spec on a `Vec` of `nbits`."""
    kind = spec[0]
    if kind == "pt":
        _, _, r = spec
        return MP._eq_table_fp3(r)
    if kind == "sh":
        _, _, r, delta = spec
        w = Z.shift_table(CPU, list(r), delta, nbits)  # (3, 2^nbits)
        return np.ascontiguousarray(w.T, dtype=np.uint64)
    if kind == "pw":
        _, _, r, D = spec
        w = Z._pow_weights(list(r), D, nbits)          # (3, 2^nbits)
        return np.ascontiguousarray(w.T, dtype=np.uint64)
    if kind == "linear_idx":
        # Direct sparse linear combination inside a vector: ("linear_idx", vid, tuple((idx, coeff), ...))
        _, _, entries = spec
        w = np.zeros((1 << nbits, 3), dtype=np.uint64)
        for idx, coeff in entries:
            cur = (int(w[idx, 0]), int(w[idx, 1]), int(w[idx, 2]))
            w[idx] = hadd(cur, coeff)
        return w
    raise ValueError(f"unknown claim spec kind: {kind!r}")


def _claim_weight_mle(spec, nbits: int, r_sub):
    """Verifier-side O(nbits) MLE evaluation of a single claim spec's local weight at `r_sub`."""
    kind = spec[0]
    if kind == "pt":
        _, _, r = spec
        return MP.eq_eval_pt(r, r_sub)
    if kind == "sh":
        _, _, r, delta = spec
        return Z.mle_shift_signed(list(r), list(r_sub), delta)
    if kind == "pw":
        _, _, r, D = spec
        # w[i*(D+1) + k] = r[i]^k
        acc = ZERO
        for i, ri in enumerate(r):
            pw = ONE
            for k in range(D + 1):
                idx = i * (D + 1) + k
                eq_v = ONE
                for b in range(nbits):
                    rb = r_sub[b]
                    eq_v = hmul(eq_v, rb if (idx >> b) & 1 else hsub(ONE, rb))
                acc = hadd(acc, hmul(pw, eq_v))
                pw = hmul(pw, ri)
        return acc
    if kind == "linear_idx":
        _, _, entries = spec
        acc = ZERO
        for idx, coeff in entries:
            eq_v = ONE
            for b in range(nbits):
                rb = r_sub[b]
                eq_v = hmul(eq_v, rb if (idx >> b) & 1 else hsub(ONE, rb))
            acc = hadd(acc, hmul(coeff, eq_v))
        return acc
    raise ValueError(f"unknown claim spec kind: {kind!r}")


def _eval_vec_claim_prover(cells_group: np.ndarray, pl: VecPlacement, spec):
    """Exact prover-side evaluation of one claim spec on a committed group's host cells."""
    sz = 1 << pl.nbits
    sub = cells_group[pl.offset:pl.offset + sz]
    w = _claim_weight_table(spec, pl.nbits)
    t = MP._pair_sumcheck_terms(None, sub, w)
    return hadd(t[0], t[1])


# ----------------------------------------------------------------------------- Prover & Verifier
def _encode_stmt_bytes(statement) -> bytes:
    from flashprove_next.transcript import canon
    if isinstance(statement, (bytes, bytearray)):
        return bytes(statement)
    return canon(statement)


def _sync_vec_vals_from_groups(cir, groups: list[MP.GroupCommitment], placements: dict[int, VecPlacement]):
    """Point every Vec's `vals` to a zero-copy `(3, sz)` view of its committed (tail-blinded!) slice."""
    for v in cir.vecs:
        pl = placements.get(v.id)
        if pl is None:
            continue
        sz = 1 << pl.nbits
        sub = groups[pl.group_idx].cells[pl.offset:pl.offset + sz]
        v.is_ext = True
        v.vals = sub.T  # zero-copy view of committed host cells!


def prove_folded(
    cir,
    statement,
    *,
    engine=None,
    queries: int = DEFAULT_QUERIES,
    gkr_masks: int = DEFAULT_GKR_MASKS,
    gkr_max_s: int = 26,
    n_group: int | None = None,
    label: bytes = b"FPN2/llm-v2",
    rng=None,
):
    """Prove any `zk_flat.Circuit` using the unified Folded ZK Engine (0 Phase-2 commitments,
    1 unified `multi_gkr`, 1 batched rate-1/8 coset-4 `multi_pcs`).

    Returns `(proof_dict, info_dict)` where `proof_dict` contains canonical bytes payloads and exact
    byte sizes (ready for `FPN2` fixed-size container framing).
    """
    t_start = time.perf_counter()
    tr = Fpn2TranscriptAdapter(label)
    stmt_bytes = _encode_stmt_bytes(statement)
    tr.absorb_bytes(stmt_bytes)

    # 1. Compute lookup multiplicities (Phase 1)
    for name in sorted(cir.lookups):
        Z._logup_mult(cir, cir.lookups[name])

    # Determine max group n_group once (each phase adaptively shrinks its groups to n_g <= n_group!)
    ph1 = [v for v in cir.vecs if v.phase == 1]
    if n_group is None:
        n_group = choose_group_nbits(ph1, queries=queries, min_n=10)

    # 2. Pack & commit Phase-1 groups (adaptive n_g <= n_group)
    t_c0 = time.perf_counter()
    groups: list[MP.GroupCommitment] = []
    group_ns: list[int] = []
    placements: dict[int, VecPlacement] = {}
    ph1_lists, ph1_pl, ph1_ns = pack_phase_adaptive(ph1, n_group, queries, start_group_idx=0)
    placements.update(ph1_pl)
    root1 = []
    vram_keep_budget = 16 << 30
    vram_kept = 0
    for g_items, ng in zip(ph1_lists, ph1_ns):
        cap_g = MP.group_data_capacity(ng, queries)
        cells = _materialize_group_cells(g_items, cap_g)
        grp = MP.commit_data_group(ng, cells, queries=queries, engine=engine, rng=rng)
        est_bytes = (8 << ng) * 32
        if engine is not None and hasattr(engine, "cp") and (vram_kept + est_bytes > vram_keep_budget):
            grp.release_device()
            engine.cp.get_default_memory_pool().free_all_blocks()
        else:
            vram_kept += est_bytes
        groups.append(grp)
        group_ns.append(ng)
        root1.append(grp.root)
        tr.absorb_bytes(grp.root)
    cir.root1 = list(root1)

    # 3. Interactive commit rounds (cir.rounds: WSum, HCO, HSel, Gen, API, Attn)
    round_pf = {}
    cir.pf = round_pf
    rootx = []
    for k, fn in enumerate(cir.rounds):
        ph = 10 + k
        cir.cur_phase = ph
        try:
            fn(tr)
        finally:
            cir.cur_phase = 1
        r_vecs = [v for v in cir.vecs if v.phase == ph]
        r_lists, r_pl, r_ns = pack_phase_adaptive(r_vecs, n_group, queries, start_group_idx=len(groups))
        placements.update(r_pl)
        r_roots = []
        for g_items, ng in zip(r_lists, r_ns):
            cap_g = MP.group_data_capacity(ng, queries)
            cells = _materialize_group_cells(g_items, cap_g)
            grp = MP.commit_data_group(ng, cells, queries=queries, engine=engine, rng=rng)
            est_bytes = (8 << ng) * 32
            if engine is not None and hasattr(engine, "cp") and (vram_kept + est_bytes > vram_keep_budget):
                grp.release_device()
                engine.cp.get_default_memory_pool().free_all_blocks()
            else:
                vram_kept += est_bytes
            groups.append(grp)
            group_ns.append(ng)
            r_roots.append(grp.root)
            tr.absorb_bytes(grp.root)
        rootx.append(r_roots)
    t_commit_s = time.perf_counter() - t_c0

    # Sync Vec values to exact committed (tail-blinded) cells so GKR & Sumchecks match PCS 100%
    _sync_vec_vals_from_groups(cir, groups, placements)

    # 4. Plan batched GKR + sumcheck sizes/degrees BEFORE committing the tiny Mask group!
    gkr_batches = plan_unified_gkr_batches(cir, m_mask=gkr_masks, max_s=gkr_max_s)
    extra_sz_degs: dict[int, int] = {}
    for _lay, blks in gkr_batches:
        for blk in blks:
            if not blk.is_linear_num:
                d_need = blk.num_ex.degree() + 1  # EqP(r_loc) * num_ex
                extra_sz_degs[blk.nbits] = max(extra_sz_degs.get(blk.nbits, 0), d_need)

    sizes = sorted(set(cir.zeros) | set(cir.sums) | set(extra_sz_degs))
    degs = {}
    for nv in sizes:
        d0 = Z._degree(cir, nv) if (nv in cir.zeros or nv in cir.sums) else 0
        degs[nv] = max(d0, extra_sz_degs.get(nv, 0))

    # Allocate Mask group: [per-batch GKR aux (libra_coeffs, mask_a, mask_d) | sumcheck_libra_masks...]
    gkr_batch_masks = []
    gkr_aux_vecs = []
    sc_masks = {}
    mask_vecs = []
    for bi, (glay, _blks) in enumerate(gkr_batches):
        lc, ma, md = MG.sample_gkr_masks(glay, rng=rng)
        gkr_batch_masks.append((lc, ma, md))
        n_aux = glay.libra_coeff_count + 2 * glay.m_mask
        aux_nb = max(1, (n_aux - 1).bit_length())
        aux_v = cir.vec(f"mask:gkr_aux:{bi}", aux_nb, n_aux, phase=20, tail=0)
        aux_v.is_ext = True
        aux_arr = np.zeros((1 << aux_nb, 3), dtype=np.uint64)
        n_l = glay.libra_coeff_count
        m_m = glay.m_mask
        aux_arr[:n_l] = lc
        aux_arr[n_l:n_l + m_m] = ma
        aux_arr[n_l + m_m:n_l + 2 * m_m] = md
        aux_v.vals = np.ascontiguousarray(aux_arr.T, dtype=np.uint64)
        gkr_aux_vecs.append(aux_v)
        mask_vecs.append(aux_v)

    for nv in sizes:
        D = degs[nv]
        cnt = nv * (D + 1)
        nb = max(1, (cnt - 1).bit_length())
        mv = cir.vec(f"mask:sc:{nv}", nb, cnt, phase=20, tail=0)
        mv.is_ext = True
        arr = np.zeros((1 << nb, 3), dtype=np.uint64)
        arr[:cnt] = MP._os_rand_fp3(cnt, rng=rng)
        mv.vals = np.ascontiguousarray(arr.T, dtype=np.uint64)
        sc_masks[nv] = mv
        mask_vecs.append(mv)

    mask_lists, mask_pl, mask_ns = pack_phase_adaptive(mask_vecs, n_group, queries, start_group_idx=len(groups))
    placements.update(mask_pl)
    root_mask = []
    for g_items, ng in zip(mask_lists, mask_ns):
        cap_g = MP.group_data_capacity(ng, queries)
        cells = _materialize_group_cells(g_items, cap_g)
        grp = MP.commit_data_group(ng, cells, queries=queries, engine=engine, rng=rng)
        est_bytes = (8 << ng) * 32
        if engine is not None and hasattr(engine, "cp") and (vram_kept + est_bytes > vram_keep_budget):
            grp.release_device()
            engine.cp.get_default_memory_pool().free_all_blocks()
        else:
            vram_kept += est_bytes
        groups.append(grp)
        group_ns.append(ng)
        root_mask.append(grp.root)
        tr.absorb_bytes(grp.root)

    # Collect per-group PCS claims: dict[group_idx] -> Z._Claims()
    claims = {gi: Z._Claims() for gi in range(len(groups))}

    # Determine GPU acceleration backend once
    sc_B = CPU
    if engine is not None and hasattr(engine, "cp"):
        from flashprove_next import field_gpu as FG
        sc_B = FG.GpuBackend

    # 5. Batched ZK-masked FracLogUp-GKR (ZERO Phase-2 commitments; bounded VRAM across arbitrary scale!)
    t_gkr0 = time.perf_counter()
    gkr_proofs_list = []
    gkr_leaf_claims_list = []
    if gkr_batches:
        gamma_lu = tr.squeeze_fp3()
        beta_lu = tr.squeeze_fp3()
        gkr_vcache = {}
        for bi, ((gkr_lay, gkr_blocks), (gkr_libra_coeffs, gkr_mask_a, gkr_mask_d), gkr_aux_vec) in enumerate(
            zip(gkr_batches, gkr_batch_masks, gkr_aux_vecs)
        ):
            leaves = _build_gkr_leaves_direct(
                gkr_lay, gkr_blocks, gamma_lu, beta_lu, gkr_mask_a, gkr_mask_d, B=sc_B, engine=engine,
            )
            gkr_res = _prove_gkr_on_packed_leaves(
                gkr_lay, leaves, tr.tr,
                libra_coeffs=gkr_libra_coeffs, mask_a=gkr_mask_a, mask_d=gkr_mask_d, engine=engine,
            )
            del leaves
            if engine is not None and hasattr(engine, "cp"):
                engine.cp.get_default_memory_pool().free_all_blocks()
            gkr_proofs_list.append(gkr_res.proof)
            r_gkr = gkr_res.r_gkr

            gkr_claim_vals = []
            seen_specs = {}

            def _rec_view_prover(atom, r_loc, _unused):
                spec = (
                    ("sh", atom.vec.id, tuple(atom.local_point(r_loc)), atom.shift)
                    if atom.shift else
                    ("pt", atom.vec.id, tuple(atom.local_point(r_loc)))
                )
                if spec in seen_specs:
                    return seen_specs[spec]
                pl = placements[atom.vec.id]
                if sc_B is not CPU:
                    val = Z._claim_value_prover(sc_B, atom, r_loc, gkr_vcache)
                else:
                    val = _eval_vec_claim_prover(groups[pl.group_idx].cells, pl, spec)
                seen_specs[spec] = val
                claims[pl.group_idx].add(spec, val)
                gkr_claim_vals.append(val)
                return val

            for blk in gkr_blocks:
                r_loc = r_gkr[:blk.nbits]
                for col_ex in blk.cols:
                    _eval_linear_ex_at_pt(col_ex, r_loc, _rec_view_prover)
                if blk.num_ex is None:
                    pass
                elif blk.is_linear_num:
                    _eval_linear_ex_at_pt(blk.num_ex, r_loc, _rec_view_prover)
                else:
                    if sc_B is not CPU:
                        num_d = _eval_ex_on_cube_dev(blk.num_ex, blk.nbits, sc_B)
                        eq_d = sc_B.eq_table(list(r_loc))
                        n_blk = sc_B.esum(sc_B.emul(num_d, eq_d))
                    else:
                        num_arr = _eval_ex_on_cube_host(blk.num_ex, blk.nbits)
                        eq_arr = MP._eq_table_fp3(r_loc)
                        tb = MP._pair_sumcheck_terms(None, num_arr, eq_arr)
                        n_blk = hadd(tb[0], tb[1])
                    gkr_claim_vals.append(n_blk)
                    ind0 = Z.Ex.atom(Z.SelLt(list(range(blk.nbits)), 1))
                    eq_ex = Z.Ex.atom(Z.EqP(list(range(blk.nbits)), list(r_loc)))
                    cir.sumzero(blk.nbits, eq_ex * blk.num_ex - ind0 * Z.Ex.const(n_blk), eq_vars=(),
                                tag=f"gkr-num:{bi}:{blk.name}:{blk.part_idx}")

            left_idx, right_idx = gkr_lay.mask_leaf_indices
            n_l = gkr_lay.libra_coeff_count
            m_m = gkr_lay.m_mask
            last = m_m - 1
            a_entries, d_entries = [], []
            for m in range(m_m):
                el, er = ONE, ONE
                il, ir = left_idx[m], right_idx[m]
                for b in range(gkr_lay.s):
                    rb = r_gkr[b]
                    el = hmul(el, rb if (il >> b) & 1 else hsub(ONE, rb))
                    er = hmul(er, rb if (ir >> b) & 1 else hsub(ONE, rb))
                if m < last:
                    a_entries.append((n_l + m, hsub(el, er)))
                    d_entries.append((n_l + m_m + m, hadd(el, er)))
                else:
                    d_entries.append((n_l + m_m + m, el))
                    d_entries.append((n_l + m, er))

            aux_pl = placements[gkr_aux_vec.id]
            spec_a = ("linear_idx", gkr_aux_vec.id, tuple(a_entries))
            spec_d = ("linear_idx", gkr_aux_vec.id, tuple(d_entries))
            val_a = _eval_vec_claim_prover(groups[aux_pl.group_idx].cells, aux_pl, spec_a)
            val_d = _eval_vec_claim_prover(groups[aux_pl.group_idx].cells, aux_pl, spec_d)
            claims[aux_pl.group_idx].add(spec_a, val_a)
            claims[aux_pl.group_idx].add(spec_d, val_d)
            gkr_claim_vals.append(val_a)
            gkr_claim_vals.append(val_d)

            for k, (r_k, g_r) in enumerate(zip(gkr_res.layer_points, gkr_res.layer_libra_evals), start=1):
                off_l = gkr_lay.layer_libra_offset(k)
                entries_k = []
                for i, ri in enumerate(r_k):
                    r2 = hmul(ri, ri)
                    r3 = hmul(r2, ri)
                    for j, pw in enumerate((ONE, ri, r2, r3)):
                        entries_k.append((off_l + 4 * i + j, pw))
                claims[aux_pl.group_idx].add(("linear_idx", gkr_aux_vec.id, tuple(entries_k)), g_r)

            cb_i = enc3s(gkr_claim_vals)
            gkr_leaf_claims_list.append(cb_i)
            tr.absorb_bytes(cb_i)
        gkr_vcache.clear()
        if sc_B is not CPU:
            Z._free(sc_B)
    t_gkr_s = time.perf_counter() - t_gkr0

    # 6. Libra-masked Sumchecks per cube size `nv` (pure raw-byte Fpn2Transcript; GPU-accelerated when available!)
    t_sc0 = time.perf_counter()
    sc_proofs = []
    vcache = {}
    for nv in sizes:
        D = degs[nv]
        r_zc = tr.exts(nv)
        lambdas = tr.exts(Z._n_constraints(cir, nv))
        total = Z._instance_terms(cir, nv, r_zc, lambdas)
        mv = sc_masks[nv]
        cnt = nv * (D + 1)
        coef = Z._mask_coef(mv, cnt)
        G_sum = Z._mask_sum(coef, nv, D)
        tr.absorb_bytes(struct.pack("<I", nv) + enc3(G_sum))
        rho = tr.squeeze_fp3()

        tables = Z._materialize(sc_B, total, nv, vcache)
        msgs_raw, r = _sumcheck_prove_fpn2(sc_B, total, nv, D, tables, coef, rho, tr.tr)
        del tables

        atoms = Z._view_atoms(total)
        if sc_B is not CPU:
            old_ccb = Z.CLAIM_CHUNK_BITS
            Z.CLAIM_CHUNK_BITS = 28  # RTX 6000D / H100 fits single 2^25 vectors directly; avoids legacy v.grp path
            try:
                cvals = [tuple(int(x) for x in v) for v in Z._claim_values(sc_B, atoms, r, vcache)]
            finally:
                Z.CLAIM_CHUNK_BITS = old_ccb
            for a, val in zip(atoms, cvals):
                spec = ("sh", a.vec.id, tuple(a.local_point(r)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r)))
                pl = placements[a.vec.id]
                claims[pl.group_idx].add(spec, val)
        else:
            cvals = []
            for a in atoms:
                spec = ("sh", a.vec.id, tuple(a.local_point(r)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r)))
                pl = placements[a.vec.id]
                val = _eval_vec_claim_prover(groups[pl.group_idx].cells, pl, spec)
                cvals.append(val)
                claims[pl.group_idx].add(spec, val)

        gval = ZERO
        for i in range(nv):
            gval = hadd(gval, Z._gi(coef, i, D, r[i]))
        mv_pl = placements[mv.id]
        claims[mv_pl.group_idx].add(("pw", mv.id, tuple(r), D), gval)

        claims_bytes = enc3s(cvals) + enc3(gval)
        tr.absorb_bytes(claims_bytes)
        sc_proofs.append({
            "nv": nv,
            "G": enc3(G_sum),
            "msgs": msgs_raw,
            "claims": claims_bytes,
        })
        if sc_B is not CPU and nv >= 20:
            vcache.clear()
            Z._free(sc_B)
    vcache.clear()
    if sc_B is not CPU:
        Z._free(sc_B)
    t_sc_s = time.perf_counter() - t_sc0

    # 7. ONE batched `multi_pcs.prove_multi_open` across ALL groups (multi-size layer injection!)
    t_open0 = time.perf_counter()
    n_groups = len(groups)
    n_max = max(group_ns)
    pcs_lay = MP.MultiPcsLayout(n=n_max, n_groups=n_groups, queries=queries, group_ns=tuple(group_ns))
    weights = []
    group_claims = []
    eq_cache = {}
    for gi in range(n_groups):
        cl = claims[gi]
        ng = group_ns[gi]
        glen = 1 << ng
        tr.absorb_bytes(struct.pack("<I", len(cl.items)))
        mus = [tr.squeeze_fp3() for _ in range(len(cl.items))]
        if sc_B is not CPU and ng >= 16:
            W_d = sc_B.zeros_e(glen)
            y_g = ZERO
            for spec, val, mu in zip(cl.items, cl.vals, mus):
                pl = placements[spec[1]]
                assert pl.group_idx == gi
                sz = 1 << pl.nbits
                kind = spec[0]
                if kind == "pt":
                    ck = ("pt", pl.nbits, spec[2])
                    lw_d = eq_cache.get(ck)
                    if lw_d is None:
                        lw_d = sc_B.eq_table(list(spec[2]))
                        if len(eq_cache) < 256:
                            eq_cache[ck] = lw_d
                elif kind == "sh":
                    lw_d = Z.shift_table(sc_B, list(spec[2]), spec[3], pl.nbits)
                else:
                    lw_h = _claim_weight_table(spec, pl.nbits)
                    lw_d = sc_B.asarray(np.ascontiguousarray(lw_h.T, dtype=np.uint64))
                sc_B.eaxpy_(W_d[:, pl.offset:pl.offset + sz], lw_d, mu)
                y_g = hadd(y_g, hmul(mu, val))
            W_g = np.ascontiguousarray(sc_B.tonumpy(W_d).T, dtype=np.uint64)
            del W_d
        else:
            W_g = np.zeros((glen, 3), dtype=np.uint64)
            y_g = ZERO
            for spec, val, mu in zip(cl.items, cl.vals, mus):
                pl = placements[spec[1]]
                assert pl.group_idx == gi
                lw = _claim_weight_table(spec, pl.nbits)
                scaled = MP._fp3_mul_scalar_host(lw, mu)
                sz = 1 << pl.nbits
                W_g[pl.offset:pl.offset + sz] = MP._fp3_add_dev(None, W_g[pl.offset:pl.offset + sz], scaled)
                y_g = hadd(y_g, hmul(mu, val))
        weights.append(W_g)
        group_claims.append(y_g)
    eq_cache.clear()

    if n_max >= 24 and engine is not None and hasattr(engine, "cp"):
        for grp in groups:
            grp.release_device()
        engine.cp.get_default_memory_pool().free_all_blocks()

    pcs_proof_bytes = MP.prove_multi_open(
        pcs_lay, groups, weights, group_claims, tr.tr, engine=engine, rng=rng,
    )
    t_open_s = time.perf_counter() - t_open0

    proof = {
        "version": "fpn2-folded-v1",
        "n_group": n_group,
        "queries": queries,
        "gkr_masks": gkr_masks,
        "gkr_max_s": gkr_max_s,
        "root1": root1,
        "rootx": rootx,
        "root_mask": root_mask,
        "round_pf": round_pf,
        "gkr_proofs": gkr_proofs_list,
        "gkr_leaf_claims": gkr_leaf_claims_list,
        "sc": sc_proofs,
        "pcs_proof": pcs_proof_bytes,
    }
    gkr_proof_total_b = sum(len(x) for x in gkr_proofs_list)
    gkr_claims_total_b = sum(len(x) for x in gkr_leaf_claims_list)
    raw_proof_bytes = (
        len(root1) * 32
        + sum(len(rr) for rr in rootx) * 32
        + len(root_mask) * 32
        + gkr_proof_total_b
        + gkr_claims_total_b
        + sum(len(s["G"]) + len(s["msgs"]) + len(s["claims"]) for s in sc_proofs)
        + len(pcs_proof_bytes)
    )
    info = {
        "n_group": n_group,
        "n_groups": n_groups,
        "group_ns": tuple(group_ns),
        "queries": queries,
        "gkr_s": tuple(lay.s for lay, _ in gkr_batches),
        "total_cells": sum(v.N for v in cir.vecs),
        "phase2_cells": 0,
        "raw_proof_bytes": raw_proof_bytes,
        "pcs_proof_bytes": len(pcs_proof_bytes),
        "gkr_proof_bytes": gkr_proof_total_b,
        "commit_s": round(t_commit_s, 3),
        "gkr_s_time": round(t_gkr_s, 3),
        "sumcheck_s": round(t_sc_s, 3),
        "open_s": round(t_open_s, 3),
        "total_s": round(time.perf_counter() - t_start, 3),
    }
    return proof, info


def _prove_gkr_on_packed_leaves(layout: MG.MultiGkrLayout, leaves: np.ndarray, transcript: Fpn2Transcript, *,
                                libra_coeffs, mask_a, mask_d, engine=None):
    """Run `multi_gkr` on an already-packed `(2^s, 6)` leaf array (aligned blocks + cancelling mask pairs)."""
    s = layout.s
    use_dev = engine is not None and hasattr(engine, "cp")
    if use_dev:
        tree = CE.gkr_tree_dev(engine, leaves)
        root_row = CE._host(engine, tree[-1])[0]
    elif engine is not None:
        tree = CE.gkr_tree(engine, leaves)
        root_row = tree[-1][0]
    else:
        cur = leaves
        tree = [cur]
        while len(cur) > 1:
            pairs = len(cur) // 2
            nxt = np.empty((pairs, 6), dtype=np.uint64)
            for j in range(pairs):
                nl = (int(cur[2 * j, 0]), int(cur[2 * j, 1]), int(cur[2 * j, 2]))
                dl = (int(cur[2 * j, 3]), int(cur[2 * j, 4]), int(cur[2 * j, 5]))
                nr = (int(cur[2 * j + 1, 0]), int(cur[2 * j + 1, 1]), int(cur[2 * j + 1, 2]))
                dr = (int(cur[2 * j + 1, 3]), int(cur[2 * j + 1, 4]), int(cur[2 * j + 1, 5]))
                n_, d_ = MG.combine_frac((nl, dl), (nr, dr))
                nxt[j, :3], nxt[j, 3:] = n_, d_
            cur = nxt
            tree.append(cur)
        root_row = tree[-1][0]

    root_num = (int(root_row[0]), int(root_row[1]), int(root_row[2]))
    root_den = (int(root_row[3]), int(root_row[4]), int(root_row[5]))

    out = bytearray()
    root_bytes = enc3(root_num) + enc3(root_den)
    out += root_bytes
    transcript.absorb_bytes(root_bytes)

    l1 = CE._host(engine, tree[s - 1]) if use_dev else tree[s - 1]
    nl0 = (int(l1[0, 0]), int(l1[0, 1]), int(l1[0, 2]))
    dl0 = (int(l1[0, 3]), int(l1[0, 4]), int(l1[0, 5]))
    nr0 = (int(l1[1, 0]), int(l1[1, 1]), int(l1[1, 2]))
    dr0 = (int(l1[1, 3]), int(l1[1, 4]), int(l1[1, 5]))
    ch0_bytes = enc3s([nl0, dl0, nr0, dr0])
    out += ch0_bytes
    transcript.absorb_bytes(ch0_bytes)

    br0 = transcript.squeeze_fp3()
    pt = [br0]
    claim_num = hadd(nl0, hmul(br0, hsub(nr0, nl0)))
    claim_den = hadd(dl0, hmul(br0, hsub(dr0, dl0)))

    layer_points = []
    layer_libra_evals = []

    for k in range(1, s):
        alpha = transcript.squeeze_fp3()
        off_l = layout.layer_libra_offset(k)
        l_coeffs = libra_coeffs[off_l:off_l + 4 * k]
        g_sum = MG._libra3_sum(l_coeffs, k)
        g_sum_bytes = enc3(g_sum)
        out += g_sum_bytes
        transcript.absorb_bytes(g_sum_bytes)
        lam = transcript.squeeze_fp3()

        child_layer = tree[s - k - 1]
        eq_host = MG._eq_table_host(pt)
        if use_dev:
            cp = engine.cp
            w_col = engine.dev(eq_host)
            nl_col = cp.ascontiguousarray(child_layer[0::2, :3])
            dl_col = cp.ascontiguousarray(child_layer[0::2, 3:])
            nr_col = cp.ascontiguousarray(child_layer[1::2, :3])
            dr_col = cp.ascontiguousarray(child_layer[1::2, 3:])
        else:
            w_col = eq_host
            nl_col = np.ascontiguousarray(child_layer[0::2, :3])
            dl_col = np.ascontiguousarray(child_layer[0::2, 3:])
            nr_col = np.ascontiguousarray(child_layer[1::2, :3])
            dr_col = np.ascontiguousarray(child_layer[1::2, 3:])

        cols = [w_col, nl_col, dl_col, nr_col, dr_col]
        r_k = []
        libra_prefix = ZERO
        for i in range(k):
            if use_dev:
                raw_arr = engine.gkr_round(cols, alpha)
                base_vals = [tuple(int(v) for v in raw_arr[t]) for t in range(4)]
            else:
                base_vals = MG._gkr_round_host(cols, alpha)
            l_vals = MG._libra3_round_vals(l_coeffs, k, i, libra_prefix)
            vals = [hadd(base_vals[t], hmul(lam, l_vals[t])) for t in range(4)]
            msg = enc3s(vals)
            out += msg
            transcript.absorb_bytes(msg)
            ri = transcript.squeeze_fp3()
            r_k.append(ri)
            libra_prefix = hadd(libra_prefix, MG._libra3_eval_var(l_coeffs, i, ri))
            cols = [MG._halve_col(engine if use_dev else None, c, ri) for c in cols]

        finals = []
        for idx_c in (1, 2, 3, 4):
            row = CE._host(engine, cols[idx_c])[0] if use_dev else cols[idx_c][0]
            finals.append((int(row[0]), int(row[1]), int(row[2])))
        nl, dl, nr, dr = finals
        g_r = libra_prefix
        fin_bytes = enc3s([nl, dl, nr, dr, g_r])
        out += fin_bytes
        transcript.absorb_bytes(fin_bytes)

        layer_points.append(r_k)
        layer_libra_evals.append(g_r)

        br_k = transcript.squeeze_fp3()
        pt = [br_k] + r_k
        claim_num = hadd(nl, hmul(br_k, hsub(nr, nl)))
        claim_den = hadd(dl, hmul(br_k, hsub(dr, dl)))

    assert len(out) == layout.proof_bytes
    return MG.MultiGkrProofResult(
        proof=bytes(out), r_gkr=pt, num_eval=claim_num, den_eval=claim_den,
        layer_points=layer_points, layer_libra_evals=layer_libra_evals,
        libra_coeffs=libra_coeffs, mask_a=mask_a, mask_d=mask_d,
    )


def _sumcheck_prove_fpn2(B, total, nv, D, tables, mask_coef, rho, tr: Fpn2Transcript):
    """Exact Libra-masked sumcheck prover absorbing raw canonical Fp3 bytes into `Fpn2Transcript`."""
    terms = Z._sc_terms(total)
    fused = Z._sc_fused(B, D, terms)
    cur_tabs = dict(tables)
    r = []
    pre = ZERO
    msgs_raw = bytearray()
    for i in range(nv):
        cur_tabs = {k: Z._expand2(B, e) for k, e in cur_tabs.items()}
        h = 1 << (nv - i - 1)
        vals = Z._sc_round_vals(B, terms, cur_tabs, D, fused, h)
        mvals = Z._mask_round(mask_coef, i, nv, D, pre, rho)
        round_vals = [hadd(vals[t], mvals[t]) for t in range(D + 1)]
        raw = enc3s(round_vals)
        msgs_raw += raw
        tr.absorb_bytes(raw)
        a = tr.squeeze_fp3()
        r.append(a)
        pre = hadd(pre, Z._gi(mask_coef, i, D, a))
        cur_tabs = {k: Z._fold_block(B, ent, a) for k, ent in cur_tabs.items()}
    return bytes(msgs_raw), r


# ----------------------------------------------------------------------------- Verifier
def verify_folded(
    cir,
    statement,
    proof: dict,
    *,
    label: bytes = b"FPN2/llm-v2",
    min_queries: int = MIN_PRODUCTION_QUERIES,
):
    """Verify a `prove_folded` proof against a verify-mode `Circuit` (zero witness).

    Enforces `queries >= min_queries` (default 72, matching Relation 2 STANDARD) and `gkr_masks >= 2`
    (required for independent layer-0 denominator blinding).  Raises `FoldedEngineError` (or
    `MultiPcsError` / `MultiGkrError`) on any violation; returns True on pass.
    """
    if not isinstance(proof, dict) or proof.get("version") != "fpn2-folded-v1":
        raise FoldedEngineError("E_VERSION", "unsupported folded proof version")
    n_group = proof.get("n_group")
    queries = proof.get("queries")
    gkr_masks = proof.get("gkr_masks")
    gkr_max_s = proof.get("gkr_max_s", 26)
    if type(n_group) is not int or not (10 <= n_group <= 28) or (n_group % 2 != 0):
        raise FoldedEngineError("E_NONCANONICAL", f"invalid n_group={n_group!r}")
    if type(queries) is not int or queries < 1 or queries > 512:
        raise FoldedEngineError("E_NONCANONICAL", f"invalid queries={queries!r}")
    if queries < int(min_queries):
        raise FoldedEngineError("E_REDUCED_SECURITY", f"queries={queries} < min_queries={min_queries}")
    if type(gkr_masks) is not int or gkr_masks < MIN_GKR_MASKS or gkr_masks > 256:
        raise FoldedEngineError("E_GKR", f"gkr_masks={gkr_masks!r} must be in [{MIN_GKR_MASKS}, 256]")
    if type(gkr_max_s) is not int or not (4 <= gkr_max_s <= 28):
        raise FoldedEngineError("E_NONCANONICAL", f"invalid gkr_max_s={gkr_max_s!r}")

    tr = Fpn2TranscriptAdapter(label)
    tr.absorb_bytes(_encode_stmt_bytes(statement))

    ph1 = [v for v in cir.vecs if v.phase == 1]

    placements: dict[int, VecPlacement] = {}
    group_roots: list[bytes] = []
    group_ns: list[int] = []

    ph1_lists, ph1_pl, ph1_ns = pack_phase_adaptive(ph1, n_group, queries, start_group_idx=0)
    placements.update(ph1_pl)
    root1 = proof["root1"]
    if len(root1) != len(ph1_lists):
        raise FoldedEngineError("E_COMMIT", "phase-1 group count mismatch")
    for r_, ng in zip(root1, ph1_ns):
        if type(r_) is not bytes or len(r_) != 32:
            raise FoldedEngineError("E_COMMIT", "bad phase-1 root")
        group_roots.append(r_)
        group_ns.append(ng)
        tr.absorb_bytes(r_)
    cir.root1 = list(root1)

    # Interactive commit rounds
    cir.pf = proof.get("round_pf", {})
    rootx = proof["rootx"]
    if len(rootx) != len(cir.rounds):
        raise FoldedEngineError("E_COMMIT", "commit round count mismatch")
    for k, fn in enumerate(cir.rounds):
        ph = 10 + k
        cir.cur_phase = ph
        try:
            fn(tr)
        finally:
            cir.cur_phase = 1
        r_vecs = [v for v in cir.vecs if v.phase == ph]
        r_lists, r_pl, r_ns = pack_phase_adaptive(r_vecs, n_group, queries, start_group_idx=len(group_roots))
        placements.update(r_pl)
        if len(rootx[k]) != len(r_lists):
            raise FoldedEngineError("E_COMMIT", f"round {k} group count mismatch")
        for r_, ng in zip(rootx[k], r_ns):
            if type(r_) is not bytes or len(r_) != 32:
                raise FoldedEngineError("E_COMMIT", f"bad round-{k} root")
            group_roots.append(r_)
            group_ns.append(ng)
            tr.absorb_bytes(r_)

    # Plan batched GKR + sumcheck sizes/degrees (identical public structure to prover!)
    gkr_max_s = int(proof.get("gkr_max_s", 26))
    gkr_batches = plan_unified_gkr_batches(cir, m_mask=gkr_masks, max_s=gkr_max_s)
    extra_sz_degs: dict[int, int] = {}
    for _lay, blks in gkr_batches:
        for blk in blks:
            if not blk.is_linear_num:
                extra_sz_degs[blk.nbits] = max(extra_sz_degs.get(blk.nbits, 0), blk.num_ex.degree() + 1)

    sizes = sorted(set(cir.zeros) | set(cir.sums) | set(extra_sz_degs))
    degs = {
        nv: max(Z._degree(cir, nv) if (nv in cir.zeros or nv in cir.sums) else 0, extra_sz_degs.get(nv, 0))
        for nv in sizes
    }

    mask_vecs = []
    gkr_aux_vecs = []
    for bi, (glay, _blks) in enumerate(gkr_batches):
        n_aux = glay.libra_coeff_count + 2 * glay.m_mask
        aux_nb = max(1, (n_aux - 1).bit_length())
        aux_v = cir.vec(f"mask:gkr_aux:{bi}", aux_nb, n_aux, phase=20, tail=0)
        aux_v.is_ext = True
        gkr_aux_vecs.append(aux_v)
        mask_vecs.append(aux_v)

    sc_masks = {}
    for nv in sizes:
        D = degs[nv]
        cnt = nv * (D + 1)
        nb = max(1, (cnt - 1).bit_length())
        mv = cir.vec(f"mask:sc:{nv}", nb, cnt, phase=20, tail=0)
        mv.is_ext = True
        sc_masks[nv] = mv
        mask_vecs.append(mv)

    mask_lists, mask_pl, mask_ns = pack_phase_adaptive(mask_vecs, n_group, queries, start_group_idx=len(group_roots))
    placements.update(mask_pl)
    root_mask = proof["root_mask"]
    if len(root_mask) != len(mask_lists):
        raise FoldedEngineError("E_COMMIT", "mask group count mismatch")
    for r_, ng in zip(root_mask, mask_ns):
        if type(r_) is not bytes or len(r_) != 32:
            raise FoldedEngineError("E_COMMIT", "bad mask root")
        group_roots.append(r_)
        group_ns.append(ng)
        tr.absorb_bytes(r_)

    claims = {gi: Z._Claims() for gi in range(len(group_roots))}

    # Batched GKR verification
    if gkr_batches:
        gkr_proofs_list = proof["gkr_proofs"]
        gkr_leaf_claims_list = proof["gkr_leaf_claims"]
        if len(gkr_proofs_list) != len(gkr_batches) or len(gkr_leaf_claims_list) != len(gkr_batches):
            raise FoldedEngineError("E_GKR", "GKR batch count mismatch")
        gamma_lu = tr.squeeze_fp3()
        beta_lu = tr.squeeze_fp3()
        for bi, ((gkr_lay, gkr_blocks), gkr_aux_vec, gpf_b, raw_gc) in enumerate(
            zip(gkr_batches, gkr_aux_vecs, gkr_proofs_list, gkr_leaf_claims_list)
        ):
            ver_gkr = MG.verify_multi_gkr(gkr_lay, gpf_b, tr.tr)
            r_gkr = ver_gkr.r_gkr

            gc_off = [0]

            def _next_fp3():
                o = gc_off[0]
                if o + 24 > len(raw_gc):
                    raise FoldedEngineError("E_LENGTH", "truncated gkr_leaf_claims")
                try:
                    val = dec3(raw_gc[o:o + 24])
                except ValueError as e:
                    raise FoldedEngineError("E_NONCANONICAL", str(e)) from None
                gc_off[0] = o + 24
                return val

            seen_specs = {}

            def _rec_view_verifier(atom, r_loc, _unused):
                spec = (
                    ("sh", atom.vec.id, tuple(atom.local_point(r_loc)), atom.shift)
                    if atom.shift else
                    ("pt", atom.vec.id, tuple(atom.local_point(r_loc)))
                )
                if spec in seen_specs:
                    return seen_specs[spec]
                val = _next_fp3()
                seen_specs[spec] = val
                pl = placements[atom.vec.id]
                claims[pl.group_idx].add(spec, val)
                return val

            acc_num = ZERO
            acc_den = ZERO
            sum_blk_ind = ZERO
            for blk in gkr_blocks:
                ind_blk = _block_ind_at(blk.offset, blk.nbits, r_gkr)
                sum_blk_ind = hadd(sum_blk_ind, ind_blk)
                r_loc = r_gkr[:blk.nbits]

                fp_eval = hf(blk.tid % P)
                bp = beta_lu
                for col_ex in blk.cols:
                    c_eval = _eval_linear_ex_at_pt(col_ex, r_loc, _rec_view_verifier)
                    fp_eval = hadd(fp_eval, hmul(bp, c_eval))
                    bp = hmul(bp, beta_lu)
                den_blk = hsub(gamma_lu, fp_eval)
                acc_den = hadd(acc_den, hmul(ind_blk, den_blk))

                if blk.num_ex is None:
                    n_blk = ONE
                elif blk.is_linear_num:
                    n_blk = _eval_linear_ex_at_pt(blk.num_ex, r_loc, _rec_view_verifier)
                else:
                    n_blk = _next_fp3()
                    ind0 = Z.Ex.atom(Z.SelLt(list(range(blk.nbits)), 1))
                    eq_ex = Z.Ex.atom(Z.EqP(list(range(blk.nbits)), list(r_loc)))
                    cir.sumzero(blk.nbits, eq_ex * blk.num_ex - ind0 * Z.Ex.const(n_blk), eq_vars=(),
                                tag=f"gkr-num:{bi}:{blk.name}:{blk.part_idx}")
                acc_num = hadd(acc_num, hmul(ind_blk, n_blk))

            left_idx, right_idx = gkr_lay.mask_leaf_indices
            n_l = gkr_lay.libra_coeff_count
            m_m = gkr_lay.m_mask
            last = m_m - 1
            a_entries, d_entries = [], []
            sum_mask_ind = ZERO
            for m in range(m_m):
                el, er = ONE, ONE
                il, ir = left_idx[m], right_idx[m]
                for b in range(gkr_lay.s):
                    rb = r_gkr[b]
                    el = hmul(el, rb if (il >> b) & 1 else hsub(ONE, rb))
                    er = hmul(er, rb if (ir >> b) & 1 else hsub(ONE, rb))
                if m < last:
                    a_entries.append((n_l + m, hsub(el, er)))
                    d_entries.append((n_l + m_m + m, hadd(el, er)))
                else:
                    d_entries.append((n_l + m_m + m, el))
                    d_entries.append((n_l + m, er))
                sum_mask_ind = hadd(sum_mask_ind, hadd(el, er))

            val_a = _next_fp3()
            val_d = _next_fp3()
            if gc_off[0] != len(raw_gc):
                raise FoldedEngineError("E_LENGTH", "trailing bytes in gkr_leaf_claims")

            aux_pl = placements[gkr_aux_vec.id]
            claims[aux_pl.group_idx].add(("linear_idx", gkr_aux_vec.id, tuple(a_entries)), val_a)
            claims[aux_pl.group_idx].add(("linear_idx", gkr_aux_vec.id, tuple(d_entries)), val_d)

            unused_ind = hsub(ONE, hadd(sum_blk_ind, sum_mask_ind))
            expect_num = hadd(acc_num, val_a)
            expect_den = hadd(hadd(acc_den, unused_ind), val_d)
            if expect_num != ver_gkr.num_eval or expect_den != ver_gkr.den_eval:
                raise FoldedEngineError("E_GKR", "GKR leaf MLE mismatch against committed vectors")

            for k, (r_k, g_r) in enumerate(zip(ver_gkr.layer_points, ver_gkr.layer_libra_evals), start=1):
                off_l = gkr_lay.layer_libra_offset(k)
                entries_k = []
                for i, ri in enumerate(r_k):
                    r2 = hmul(ri, ri)
                    r3 = hmul(r2, ri)
                    for j, pw in enumerate((ONE, ri, r2, r3)):
                        entries_k.append((off_l + 4 * i + j, pw))
                claims[aux_pl.group_idx].add(("linear_idx", gkr_aux_vec.id, tuple(entries_k)), g_r)

            tr.absorb_bytes(raw_gc)

    # Sumcheck verification per cube size
    sc_proofs = proof["sc"]
    if len(sc_proofs) != len(sizes):
        raise FoldedEngineError("E_SUMCHECK", "sumcheck instance count mismatch")
    for nv, scp in zip(sizes, sc_proofs):
        D = degs[nv]
        if int(scp["nv"]) != nv:
            raise FoldedEngineError("E_SUMCHECK", "sumcheck nv mismatch")
        r_zc = tr.exts(nv)
        lambdas = tr.exts(Z._n_constraints(cir, nv))
        total = Z._instance_terms(cir, nv, r_zc, lambdas)

        G_bytes = scp["G"]
        if type(G_bytes) is not bytes or len(G_bytes) != 24:
            raise FoldedEngineError("E_LENGTH", "bad Libra G_sum encoding")
        try:
            G_sum = dec3(G_bytes)
        except ValueError as e:
            raise FoldedEngineError("E_NONCANONICAL", str(e)) from None
        tr.absorb_bytes(struct.pack("<I", nv) + G_bytes)
        rho = tr.squeeze_fp3()
        cl = hmul(rho, G_sum)

        msgs_raw = scp["msgs"]
        stride = (D + 1) * 24
        if type(msgs_raw) is not bytes or len(msgs_raw) != nv * stride:
            raise FoldedEngineError("E_LENGTH", "bad sumcheck round bytes length")
        r = []
        for i in range(nv):
            chunk = msgs_raw[i * stride:(i + 1) * stride]
            try:
                vals = [dec3(chunk[j * 24:(j + 1) * 24]) for j in range(D + 1)]
            except ValueError as e:
                raise FoldedEngineError("E_NONCANONICAL", str(e)) from None
            if hadd(vals[0], vals[1]) != cl:
                raise FoldedEngineError("E_SUMCHECK", f"sumcheck round mismatch (cube 2^{nv})")
            tr.absorb_bytes(chunk)
            a = tr.squeeze_fp3()
            r.append(a)
            cl = lagrange_eval(vals, a)

        atoms = Z._view_atoms(total)
        cb = scp["claims"]
        if type(cb) is not bytes or len(cb) != (len(atoms) + 1) * 24:
            raise FoldedEngineError("E_LENGTH", "bad sumcheck claims length")
        cv = {}
        for idx, a in enumerate(atoms):
            try:
                val = dec3(cb[idx * 24:(idx + 1) * 24])
            except ValueError as e:
                raise FoldedEngineError("E_NONCANONICAL", str(e)) from None
            cv[a.kid] = val
            spec = ("sh", a.vec.id, tuple(a.local_point(r)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r)))
            pl = placements[a.vec.id]
            claims[pl.group_idx].add(spec, val)
        try:
            gval = dec3(cb[len(atoms) * 24:])
        except ValueError as e:
            raise FoldedEngineError("E_NONCANONICAL", str(e)) from None
        mv = sc_masks[nv]
        mv_pl = placements[mv.id]
        claims[mv_pl.group_idx].add(("pw", mv.id, tuple(r), D), gval)
        tr.absorb_bytes(cb)

        lhs = hadd(Z._eval_terms(total, r, cv), hmul(rho, gval))
        if lhs != cl:
            raise FoldedEngineError("E_SUMCHECK", f"sumcheck final eval mismatch (cube 2^{nv})")

    # Batched multi_pcs verification (O(claims * n) verifier weight eval — zero length-2^n allocations!)
    n_groups = len(group_roots)
    n_max = max(group_ns)
    pcs_lay = MP.MultiPcsLayout(n=n_max, n_groups=n_groups, queries=queries, group_ns=tuple(group_ns))
    group_mus = []
    group_claims = []
    for gi in range(n_groups):
        cl = claims[gi]
        tr.absorb_bytes(struct.pack("<I", len(cl.items)))
        mus = [tr.squeeze_fp3() for _ in range(len(cl.items))]
        y_g = ZERO
        for val, mu in zip(cl.vals, mus):
            y_g = hadd(y_g, hmul(mu, val))
        group_mus.append(mus)
        group_claims.append(y_g)

    def w_eval_fn(gi, r_ip):
        cl = claims[gi]
        mus = group_mus[gi]
        acc = ZERO
        for spec, mu in zip(cl.items, mus):
            pl = placements[spec[1]]
            ind = _block_ind_at(pl.offset, pl.nbits, r_ip)
            loc = _claim_weight_mle(spec, pl.nbits, r_ip[:pl.nbits])
            acc = hadd(acc, hmul(mu, hmul(ind, loc)))
        return acc

    MP.verify_multi_open(pcs_lay, group_roots, group_claims, w_eval_fn, proof["pcs_proof"], tr.tr)
    return True
