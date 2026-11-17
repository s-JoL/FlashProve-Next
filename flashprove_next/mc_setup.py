# SPDX-License-Identifier: Apache-2.0
"""One-time model setup proof (specs/09 §2) pi_setup, in ZERO KNOWLEDGE, with ONE joint column test for all row blocks.

Public: model_id, the profile (modelcommit.Params + SetupParams), and pi_setup.  Proven (specs/09 §2.2, layout v2 of
flashprove_next.modelcommit): there exist W^cd, W^raw (R_MAX x n_c), salts and the model_id preimage (s_0, C_W, C_raw, C_cfg) with
model_id = H(DOM_MID; s_0, C_W, C_raw, C_cfg), such that
  (a) proximity / binding: every row of the codeword matrices committed in C_W and C_raw agrees with RS(W^cd_i),
      RS(W^raw_i) on a COMMON set of >= (1 - delta) n columns (delta = 0.3), jointly over all rows of both trees;
  (b) legality and range: every raw entry of every data row (i >= 2) is a finite bf16 code and the decoded entry is
      cd = floor(sigma 2^(E + G_i)) with |cd| < 2^22 and G_i in [G_LO, G_LO + 16); G_i is MAXIMAL (some |cd| >= 2^21 in
      the row, unless the row's codes are all 0x0000), so a provider cannot quantise a row by under-scaling it;
  (c) decode consistency: W^cd = CD(W^raw) entry-wise with the exponents of (b), and G_i = SCALE[i] (meta row 0);
  (d) norms: PN[i T_O + r] = sum_c W^cd[i, r T_I + c]^2 for every data row and tile row (meta row 1), PN = 0 elsewhere.

Structure (and why the split into sub-proofs is sound):
  * B row-block sub-proofs, block b = rows [7 b, 7 b + 7) on a cube (i: 3 bits, k: 20 bits); cube row 7 is a spare
    row that holds the public decode table (2^20 entries, LogUp) and every vector's T1 tail.  Phase 1 of block b
    commits its MESSAGE rows W^cd_b, W^raw_b (coefficients), the big/multiplicity column, the per-row exponents, empty
    flags, count inverses, partial norms and a uniform mask m_b.
  * seed1 = H(statement, phase-1 roots of ALL blocks)  ->  the m_s = 206 column positions Q (shared by every block and
    both trees).  This is drawn ONCE, after every row block is committed.
  * one column sub-proof: opens the 206 hidden columns of both trees, hashes them in-circuit (Poseidon2 leaves + paths)
    to the hidden C_W, C_raw, which it opens from model_id in-circuit.  seed2 = H(seed1, its phase-1 root) -> the
    batching challenges r^cd, r^raw (one per row), beta (one per column) and the link points z_G, z_N.
  * every sub-proof x proves T_x = m_x + S_x for a PUBLIC scalar T_x, where
        S_b   = sum_{i in b} (r^cd_i <W^cd_i, g> + r^raw_i <W^raw_i, g>) + sum_i eq(z_G, i) G_i + sum_{i,r} eq(z_N, (i,r)) s_{i,r}
                [- sum_k eq(z_G, k) SCALE[k] - sum_x eq(z_N, x) PN[x]   in block 0]
        S_col = - sum_q beta_q sum_i (r^cd_i M^cd[i, j_q] + r^raw_i M^raw[i, j_q]),
    g_k = sum_q beta_q w^{j_q k} (so <W_i, g> = sum_q beta_q RS(W_i)[j_q]); the verifier checks sum_x T_x = 0.
  Soundness of (a): all message rows are fixed before Q.  Let D = {j : some row i of either tree has RS(W_i)[j] !=
  M[i, j]}.  If |D| >= delta n, Q misses D with probability <= (1 - delta)^206 <= 2^-106.0.  Otherwise A = complement(D)
  is a common agreement set of size > (1 - delta) n > (1 + rate)/2 n, so every committed row is uniquely decoded to the
  committed message row (no correlated-agreement theorem needed: the messages are committed, not extracted).  If Q
  hits D, the batched difference sum_q beta_q sum_i r_i (RS(W_i)[j_q] - M[i, j_q]) is a nonzero bilinear form in
  (r, beta), drawn after everything it depends on (incl. sum_x m_x), so sum_x T_x = 0 with probability <= 2/|E|.  The
  G and PN links are linear forms in eq(z, .) drawn after the G_i, s, SCALE, PN commitments: <= log(n)/|E|.
  Zero knowledge: every witness value is a T1-blinded committed vector; the public T_x are uniform subject to
  sum_x T_x = 0 (the masks are uniform with sum 0), Q and the challenges are public coins; all shapes (block count,
  cube sizes, column count) are functions of the profile.
  Two-pass prover: Q needs every block's phase-1 root, so each block's phase 1 is computed twice (before Q for its
  root, after the column proof for the full proof), reproduced bit-for-bit under field.seeded_rng with a per-block
  secret key.

usage (CUDA GPU):
  python -m flashprove_next.mc_setup qwen --model PATH/TO/Qwen3-0.6B --store DIR --out FILE [--blocks N]
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import zk_flat as Z
from . import gadgets as G
from . import poseidon2 as PS
from . import modelcommit as MC
from . import mc_zk as MZ
from .zk_flat import Ex, EqP, EqZC, SelLt, IndexP, TB
from .field import P, CPU, ZERO, ONE, hadd, hsub, hmul, hneg, rand_ext, seeded_rng
from .hwinfo import describe as describe_machine
from .transcript import Transcript, dumps, loads

VERSION = "fpn-mcsetup-1"
LBL_BLK, LBL_COL = b"fpn-mcsetup-blk-v1", b"fpn-mcsetup-col-v1"
BIG, LIM = 1 << 21, 1 << 22
GBITS = 4                    # 16 exponent values: table = 2^(GBITS + 16) entries = one spare row of n_c = 2^20


@dataclass(frozen=True)
class SetupParams:
    MQS: int = 206           # joint column queries: 0.7^206 = 2^-106.0 (delta = 0.3 < (1 - rate)/2)
    SR_BITS: int = 3         # block cube rows (2^SR_BITS - 1 data rows + 1 spare row); 3: cd, raw, big share one 2^25 group
    DELTA: float = 0.3
    RADIUS: bool = False     # True: prove W^r at [R_MAX/2, R_MAX) paired with [0, R_MAX/2), raw upper-half zero
    CFG_ROWS: bool = False   # True: prove stable config rows CFG0 (Tt) & CFG1 (Vt + Lt) in block 0
    GBITS: int = GBITS       # exponent table bits (4 -> 16 exponents in [G_LO, G_LO + 16); small CPU tests may use 1..2)


def mqs_for(delta=0.3, bits=106):
    return math.ceil(bits / -math.log2(1.0 - delta))


def _lg(n):
    return max(0, int(n - 1).bit_length())


class SShape:
    """Public layout constants (functions of the profile only)."""

    def __init__(self, prm, sp):
        self.prm, self.sp = prm, sp
        assert prm.nc & (prm.nc - 1) == 0 and prm.R_MAX & (prm.R_MAX - 1) == 0
        self.KB = prm.nc.bit_length() - 1
        self.gbits = sp.GBITS
        self.TB_BITS = self.gbits + 16
        self.inline_table = (self.KB == self.TB_BITS)
        assert self.KB == self.TB_BITS or prm.nc >= 1024, "nc must either equal 2^(gbits+16) or be >= 1024"
        self.radius = bool(sp.RADIUS)
        self.cfg_rows = bool(sp.CFG_ROWS)
        self.IB = prm.R_MAX.bit_length() - 1
        self.H = (prm.R_MAX // 2) if self.radius else prm.R_MAX
        self.HB = self.H.bit_length() - 1
        self.PNB = self.HB + prm.ob
        assert self.PNB <= self.KB, "layout v2 with one PN row"
        self.n_pn = 1
        self.n_cfg = 2 if self.cfg_rows else 0
        self.meta_cfg0 = (1 + self.n_pn) if self.cfg_rows else None
        self.meta_cfg1 = (2 + self.n_pn) if self.cfg_rows else None
        self.n_meta = 1 + self.n_pn + self.n_cfg
        if self.cfg_rows:
            self.NT, self.NV, self.NL = MC.cfg_sizes(prm)
        else:
            self.NT = self.NV = self.NL = 0
        self.SRB = sp.SR_BITS
        self.SR = 1 << sp.SR_BITS
        self.RB = self.SR - 1
        assert not self.cfg_rows or self.n_meta <= self.RB, "meta+cfg rows must fit inside block 0"
        assert prm.TO >= TB, "partial-norm vector needs room for its tail"
        self.nblocks = -(-self.H // self.RB)
        self.MQS = sp.MQS
        self.QB = _lg(sp.MQS + 1)
        assert sp.MQS + 1 < (1 << self.QB), "need a spare column block for the model_id rows and the tail"
        self.A = prm.R_MAX // 8
        self.depth = prm.depth
        self.CB = _lg(self.A + self.depth + 1)
        self.INIT = (1 << self.CB) - 1
        self.ROOT = self.A + self.depth - 1
        self.MID = sp.MQS << self.CB
        assert self.A == 1 << (self.IB - 3) and self.IB - 3 < self.CB

    def rows_of(self, b):
        i0 = b * self.RB
        return i0, max(0, min(self.RB, self.H - i0))


# ============================================================================ public decode table
_TAB = {}


def cd_table(G_LO, gbits=GBITS):
    """x = gi 2^16 + code (gi < 2^gbits): cdT[x] = floor(sigma 2^(E + G_LO + gi)), absT[x] = |cdT[x]|,
    bigT[x] = [|cd| >= 2^21], inexT[x] = [bf16(code) * 2^(G_LO + gi) not in Z] for legal entries (finite code,
    |cd| < 2^22); illegal entries have cdT = 0, absT = 0, bigT = 2, inexT = 0 (never matched: data big bits are boolean)."""
    key = (int(G_LO), int(gbits))
    if key not in _TAB:
        sig_t, E_t, ok_t = MC._bf_tables()
        n = 1 << (gbits + 16)
        cdT = np.zeros(n, dtype=np.int64)
        absT = np.zeros(n, dtype=np.int64)
        bigT = np.full(n, 2, dtype=np.int64)
        inexT = np.zeros(n, dtype=np.int64)
        for gi in range(1 << gbits):
            k = E_t + G_LO + gi
            cd = np.where(k >= 0, sig_t << np.clip(k, 0, 40), sig_t >> np.clip(-k, 0, 63))
            legal = ok_t & (np.abs(cd) < LIM)
            inex = k < 0
            if inex.any():
                sh = np.clip(-k, 0, 62)
                inex = (inex & ((np.abs(sig_t) & ((np.int64(1) << sh) - 1)) != 0)) | (inex & (sh > 20))
            sl = slice(gi << 16, (gi + 1) << 16)
            cd_leg = np.where(legal, cd, 0)
            cdT[sl] = cd_leg
            absT[sl] = np.abs(cd_leg)
            bigT[sl] = np.where(legal, (np.abs(cd) >= BIG).astype(np.int64), 2)
            inexT[sl] = np.where(legal, inex.astype(np.int64), 0)
        assert (bigT[-TB:] == 2).all()                 # the table's T1 tail only covers illegal (NaN) entries
        _TAB[key] = (cdT, absT, bigT, inexT,
                     CPU.fromi(cdT), absT.astype(np.uint64), bigT.astype(np.uint64), inexT.astype(np.uint64))
    return _TAB[key]


# ============================================================================ seeds and shared challenges
def statement0(prm, sp, mid):
    return {"v": VERSION, "params": asdict(prm), "setup": asdict(sp), "model_id": [int(x) for x in mid]}


def stmt_block(st0, b):
    return dict(st0, kind="block", b=int(b))


def stmt_col(st0, seed1):
    return dict(st0, kind="col", seed1=seed1.hex())


def seed1_of(st0, roots1):
    tr = Transcript(b"fpn-mcsetup-seed1")
    tr.absorb(st0)
    for rr in roots1:
        tr.absorb(len(rr))
        for r in rr:
            tr.absorb(bytes(r))
    return tr._next()[:32]


def seed2_of(seed1, root1_col):
    tr = Transcript(b"fpn-mcsetup-seed2")
    tr.absorb(seed1, len(root1_col))
    for r in root1_col:
        tr.absorb(bytes(r))
    return tr._next()[:32]


def q_of(sh, seed1):
    tr = Transcript(b"fpn-mcsetup-Q")
    tr.absorb(seed1)
    return tr.indices(sh.MQS, sh.prm.n)


def glob_of(sh, seed1, seed2):
    tr = Transcript(b"fpn-mcsetup-glob")
    tr.absorb(seed2)
    R = sh.prm.R_MAX
    g = {"seed1": seed1, "seed2": seed2, "Q": q_of(sh, seed1), "rcd": tr.exts(R), "rraw": tr.exts(R),
         "beta": tr.exts(sh.MQS), "zG": tr.exts(sh.HB), "zN": tr.exts(sh.PNB)}
    if sh.radius:
        g["zK"] = tr.exts(sh.HB)
        g["eqK"] = CPU.eq_table(list(g["zK"]))                 # (3, H)
    g["eqG"] = CPU.eq_table(list(g["zG"]))                     # (3, H)
    g["eqN"] = CPU.eq_table(list(g["zN"]))                     # (3, 2^PNB)
    return g


def _ev(vals, n):
    out = np.zeros((3, n), dtype=np.uint64)
    for i, v in enumerate(vals):
        out[0, i], out[1, i], out[2, i] = int(v[0]) % P, int(v[1]) % P, int(v[2]) % P
    return out


def _esum(xs):
    acc = ZERO
    for x in xs:
        acc = hadd(acc, x)
    return acc


def _bdot(vals, e):
    """sum_x vals[x] e[x] (vals signed ints, e (2, n) E)."""
    return Z._dot_base(CPU.fromi(np.asarray(vals, dtype=np.int64)), np.ascontiguousarray(e))


_GQ = {}


def gq_table(sh, glob, B):
    key = (id(glob), getattr(B, "name", ""))
    if key not in _GQ:
        _GQ.clear()
        a = MZ.GQ(sh.prm, glob["Q"], glob["beta"], list(range(sh.KB)))
        _GQ[key] = np.asarray(B.tonumpy(a.local_table(B)) if getattr(B, "name", "") == "cupy" else a.local_table(B))
    return _GQ[key]


# ============================================================================ block witness
def block_witness(sh, b, src):
    """Phase-1 data of row block b from a source with cd_row(i), raw_row(i) (signed int64, zero past the used rows)."""
    prm = sh.prm; nc = prm.nc; SR = sh.SR; H = sh.H
    ntab = 1 << sh.TB_BITS
    i0, nr = sh.rows_of(b)
    cd = np.zeros((SR, nc), dtype=np.int64); raw = np.zeros((SR, nc), dtype=np.int64)
    wr = np.zeros((SR, nc), dtype=np.int64) if sh.radius else None
    for i in range(nr):
        cd[i] = src.cd_row(i0 + i); raw[i] = src.raw_row(i0 + i)
        if sh.radius:
            wr[i] = src.cd_row(H + i0 + i)
    scale = src.cd_row(MC.META_SCALE)
    data = [i for i in range(nr) if i0 + i >= sh.n_meta]
    meta = [i for i in range(nr) if i0 + i < sh.n_meta]
    viol = []
    for i in meta:
        if raw[i].any():
            viol.append(f"meta/cfg row {i0 + i}: W^raw must be all zero")
        if sh.radius and wr[i].any():
            viol.append(f"meta/cfg row {i0 + i}: upper-half W^r row {H + i0 + i} must be all zero")
    Gr = np.zeros(SR, dtype=np.int64)
    kap = np.zeros(SR, dtype=np.int64) if sh.radius else None
    tfl = np.zeros(SR, dtype=np.int64) if sh.radius else None
    big = np.zeros((SR, nc), dtype=np.int64)
    abs_cd = np.zeros((SR, nc), dtype=np.int64) if sh.radius else None
    inex = np.zeros((SR, nc), dtype=np.int64) if sh.radius else None
    mult = np.zeros(ntab, dtype=np.int64)
    e = np.zeros(SR, dtype=np.int64); cnt = np.zeros(SR, dtype=np.int64); inv = np.zeros(SR, dtype=np.uint64)
    s = np.zeros((SR, prm.TO), dtype=np.int64)
    cdT, absT, bigT, inexT = cd_table(prm.G_LO, sh.gbits)[:4]
    for i in data:
        gi_row = i0 + i
        Gr[i] = int(scale[gi_row])
        big[i] = (np.abs(cd[i]) >= BIG).astype(np.int64)
        if sh.radius:
            kap[i] = int(scale[H + gi_row])
            tfl[i] = int(kap[i] > 0)
            abs_cd[i] = np.abs(cd[i])
        gi = Gr[i] - prm.G_LO
        if not 0 <= gi < (1 << sh.gbits):
            viol.append(f"row {gi_row}: exponent {Gr[i]} outside the table range")
            gi = min(max(gi, 0), (1 << sh.gbits) - 1)
        rc = np.clip(raw[i], 0, (1 << 16) - 1)
        if not np.array_equal(rc, raw[i]):
            viol.append(f"row {gi_row}: raw entry outside 16 bits")
        x = (gi << 16) + rc
        mult += np.bincount(x, minlength=ntab)
        if sh.radius:
            inex[i] = MC.tile_inexact(rc, Gr[i]).astype(np.int64)
            bad = (cdT[x] != cd[i]) | (absT[x] != abs_cd[i]) | (bigT[x] != big[i]) | (inexT[x] != inex[i])
            if bad.any():
                viol.append(f"row {gi_row}: {int(bad.sum())} entries are not (code, G, cd, |cd|, big, inex) table rows")
            exp_wr = tfl[i] * (abs_cd[i] + kap[i] * inex[i])
            if not np.array_equal(wr[i], exp_wr):
                viol.append(f"row {gi_row}: W^r row {H + gi_row} != is_tile * (|cd| + kap * inex)")
        else:
            bad = (cdT[x] != cd[i]) | (bigT[x] != big[i])
            if bad.any():
                viol.append(f"row {gi_row}: {int(bad.sum())} entries are not (code, G, CD(code), big) table rows")
        e[i] = int(not raw[i].any())
        cnt[i] = int(big[i].sum())
        if cnt[i]:
            inv[i] = pow(int(cnt[i]), P - 2, P)
        elif not e[i]:
            viol.append(f"row {gi_row}: exponent not maximal (no |cd| >= 2^21 in a nonzero row)")
        w = cd[i].reshape(prm.TO, prm.TI)
        s[i] = (w * w).sum(axis=1)
    bm = big.copy()
    if sh.inline_table:
        bm[SR - 1] = mult
    out = {"b": b, "i0": i0, "nr": nr, "data": data, "cd": cd, "raw": raw, "wr": wr, "bm": bm, "mult": mult,
           "abs_cd": abs_cd, "inex": inex, "G": Gr, "kap": kap, "tfl": tfl, "e": e, "inv": inv, "cnt": cnt, "s": s,
           "viol": viol}
    if b == 0 and sh.cfg_rows:
        NT, NV = sh.NT, sh.NV
        cfg0 = cd[sh.meta_cfg0]
        cfg1 = cd[sh.meta_cfg1]
        tt = [cfg0[f * NT:(f + 1) * NT].copy() for f in range(8)]
        vt = [cfg1[f * NV:(f + 1) * NV].copy() for f in range(4)]
        out["tt"], out["vt"] = tt, vt
        out["g_meta"] = scale[:H].copy()
        out["k_meta"] = scale[H:2 * H].copy() if sh.radius else np.zeros(H, dtype=np.int64)
        if int(scale[H - 1]) != prm.G_LO:
            viol.append(f"SCALE[zero_row={H - 1}] = {int(scale[H - 1])} != G_LO={prm.G_LO}")
        if sh.radius and int(out["k_meta"][H - 1]) != 0:
            viol.append(f"KAPPA_META[zero_row={H - 1}] = {int(out['k_meta'][H - 1])} != 0")
        if (tt[0][1:] <= tt[0][:-1]).any():
            viol.append("CFG0 (Tt): keys are not strictly increasing")
        if (vt[0][1:] <= vt[0][:-1]).any():
            viol.append("CFG1 (Vt): keys are not strictly increasing")
        for idx in range(NT):
            trw = int(tt[1][idx])
            if not 0 <= trw < H:
                viol.append(f"CFG0[{idx}]: t_row {trw} outside [0, {H})")
            else:
                if int(tt[2][idx]) != int(out["g_meta"][trw]):
                    viol.append(f"CFG0[{idx}]: G_t {int(tt[2][idx])} != SCALE[{trw}]={int(out['g_meta'][trw])}")
                if sh.radius and int(tt[3][idx]) != int(out["k_meta"][trw]):
                    viol.append(f"CFG0[{idx}]: kappa_t {int(tt[3][idx])} != KAPPA_META[{trw}]={int(out['k_meta'][trw])}")
            fl = int(tt[7][idx])
            if fl not in (0, 1):
                viol.append(f"CFG0[{idx}]: flag {fl} not boolean")
            elif fl == 0 and (trw != H - 1 or int(tt[2][idx]) != prm.G_LO or int(tt[3][idx]) != 0 or int(tt[4][idx]) != 1):
                viol.append(f"CFG0[{idx}]: sentinel entry (flag=0) violates (t_row=H-1, G_t=G_LO, kap=0, s_align=1)")
            elif fl == 1 and sh.radius and int(tt[3][idx]) == 0:
                viol.append(f"CFG0[{idx}]: real tile entry (flag=1) points to non-tile row {trw} (kappa=0)")
    return out


def block_T(sh, w, glob, m):
    """T_b = m_b + S_b (prover)."""
    prm = sh.prm; H = sh.H
    gq = gq_table(sh, glob, CPU) if "_gq" not in glob else glob["_gq"]
    i0, nr = w["i0"], w["nr"]
    acc = m
    for i in range(nr):
        acc = hadd(acc, hmul(glob["rcd"][i0 + i], _bdot(w["cd"][i], gq)))
        acc = hadd(acc, hmul(glob["rraw"][i0 + i], _bdot(w["raw"][i], gq)))
        if sh.radius:
            acc = hadd(acc, hmul(glob["rcd"][H + i0 + i], _bdot(w["wr"][i], gq)))
    for i in w["data"]:
        gi = i0 + i
        acc = hadd(acc, hmul((int(glob["eqG"][0, gi]), int(glob["eqG"][1, gi]), int(glob["eqG"][2, gi])),
                             (int(w["G"][i]) % P, 0, 0)))
        if sh.radius:
            acc = hadd(acc, hmul((int(glob["eqK"][0, gi]), int(glob["eqK"][1, gi]), int(glob["eqK"][2, gi])),
                                 (int(w["kap"][i]) % P, 0, 0)))
        acc = hadd(acc, _bdot(w["s"][i], glob["eqN"][:, gi * prm.TO:(gi + 1) * prm.TO]))
    if w["b"] == 0:
        acc = hsub(acc, _bdot(w["cd"][MC.META_SCALE][:H], glob["eqG"]))
        if sh.radius:
            acc = hsub(acc, _bdot(w["cd"][MC.META_SCALE][H:2 * H], glob["eqK"]))
        acc = hsub(acc, _bdot(w["cd"][MC.META_PN][:1 << sh.PNB], glob["eqN"]))
    return acc


def block_h(sh, w, mu, lam):
    """LogUp helper: 1/(lam - key) on data rows of S, -mult/(lam - keyT) on the table rows (spare row of S when
    inline_table, else separate table space)."""
    prm = sh.prm; nc = prm.nc; SR = sh.SR; ntab = 1 << sh.TB_BITS
    mu2 = hmul(mu, mu); mu3 = hmul(mu2, mu)
    mu4 = hmul(mu3, mu); mu5 = hmul(mu4, mu)
    cdT, absT, bigT, inexT, cdTf, absTf, bigTf, inexTf = cd_table(prm.G_LO, sh.gbits)
    x = np.arange(ntab, dtype=np.uint64)
    code = x & np.uint64(0xFFFF); gT = (x >> np.uint64(16)) + np.uint64(prm.G_LO)

    def keyE(c, g, cdf, bf, af=None, ixf=None):
        k = CPU.ext(c)
        k = CPU.eadd(k, CPU.escale(CPU.ext(g), mu))
        k = CPU.eadd(k, CPU.escale(CPU.ext(cdf), mu2))
        k = CPU.eadd(k, CPU.escale(CPU.ext(bf), mu3))
        if sh.radius:
            k = CPU.eadd(k, CPU.escale(CPU.ext(af), mu4))
            k = CPU.eadd(k, CPU.escale(CPU.ext(ixf), mu5))
        return k

    lamv = lambda n: np.stack([np.full(n, lam[l], dtype=np.uint64) for l in (0, 1, 2)])
    invT = CPU.einv(CPU.esub(lamv(ntab), keyE(code, gT, cdTf, bigTf, absTf, inexTf)))
    h = np.zeros((3, SR * nc), dtype=np.uint64)
    gmask = (1 << sh.gbits) - 1
    for i in w["data"]:
        gi = int(w["G"][i]) - prm.G_LO
        rc = np.clip(w["raw"][i], 0, (1 << 16) - 1)
        xi = (min(max(gi, 0), gmask) << 16) + rc
        hi = invT[:, xi]
        bad_cond = (cdT[xi] != w["cd"][i]) | (bigT[xi] != w["bm"][i]) | (rc != w["raw"][i]) | (gi != (xi >> 16))
        if sh.radius:
            bad_cond = bad_cond | (absT[xi] != w["abs_cd"][i]) | (inexT[xi] != w["inex"][i])
        bad = np.flatnonzero(bad_cond)
        if len(bad):                                  # cheating witness: exact inverse of its own key
            kb = keyE(CPU.fromi(w["raw"][i][bad]), np.full(len(bad), int(w["G"][i]) % P, dtype=np.uint64),
                      CPU.fromi(w["cd"][i][bad]), CPU.fromi(w["bm"][i][bad]),
                      CPU.fromi(w["abs_cd"][i][bad]) if sh.radius else None,
                      CPU.fromi(w["inex"][i][bad]) if sh.radius else None)
            hi[:, bad] = CPU.einv(CPU.esub(lamv(len(bad)), kb))
        h[:, i * nc:(i + 1) * nc] = hi
    mlt = CPU.ext(CPU.fromi(w["mult"]))
    h_tab = CPU.esub(CPU.zeros_e(ntab), CPU.emul(mlt, invT))
    if sh.inline_table:
        h[:, (SR - 1) * nc:SR * nc] = h_tab
        return h[:, :(SR * nc) - TB], None
    return h[:, :(SR * nc) - TB], h_tab[:, :ntab - TB]


# ============================================================================ block circuit
def build_block(sh, b, stmt, wit=None, glob=None, tval=None, out=None, B=CPU):
    """Row-block sub-proof b.  Prover: wit = block_witness(...) + "m" (mask); glob (shared challenges) is needed only
    by the commit round (absent in the phase-1-only pass).  Verifier: wit None, glob and tval (= T_b) given."""
    prm = sh.prm; H = sh.H
    pv = wit is not None
    ctx = G.Ctx(pv)
    cir = ctx.cir
    SRB, SR, KB, nc = sh.SRB, sh.SR, sh.KB, prm.nc
    i0, nr = sh.rows_of(b)
    out = {} if out is None else out
    S = ctx.space("suB", [("i", SRB), ("k", KB)], padded=False)
    GS = ctx.space("suG", [("i", SRB), ("x", 5)], padded=False)
    NS = ctx.space("suN", [("i", SRB), ("r", prm.ob)], padded=False)
    KS = ctx.space("suK", [("x", 6)], padded=False)
    iv, kv = S.vars("i"), S.vars("k")
    if pv:
        for v in wit["viol"]:
            ctx.violation(v)
    # ------------------------------------------------------------------ phase 1
    flat = lambda a: a.reshape(-1)[:S.content] if pv else None
    cdc = S.col("cd", flat(wit["cd"]) if pv else None)
    rawc = S.col("raw", flat(wit["raw"]) if pv else None)
    bmc = S.col("bm", flat(wit["bm"]) if pv else None)
    wrc = absc = inexc = multc = None
    if sh.radius:
        wrc = S.col("wr", flat(wit["wr"]) if pv else None)
        absc = S.col("abs", flat(wit["abs_cd"]) if pv else None)
        inexc = S.col("inex", flat(wit["inex"]) if pv else None)
    for c in (cdc, rawc, bmc, wrc, absc, inexc):
        if c is not None:
            c.v = None                                    # the committed copy lives in vec.vals (no second host copy)
    n_gs = SR - 1
    rep = lambda a: np.repeat(np.asarray(a)[:n_gs], 32) if pv else None
    gc_ = GS.col("g", rep(wit["G"]) if pv else None)
    invc = GS.col("inv", rep(wit["inv"]) if pv else None)
    ec = GS.col("e", rep(wit["e"]) if pv else None)
    kapc = tflc = kinv_c = None
    if sh.radius:
        kapc = GS.col("kap", rep(wit["kap"]) if pv else None)
        tflc = GS.col("tfl", rep(wit["tfl"]) if pv else None)
        kinv_v = np.array([pow(int(x), P - 2, P) if int(x) else 0 for x in wit["kap"][:n_gs]], dtype=np.uint64) if pv else None
        kinv_c = GS.col("kinv", rep(kinv_v) if pv else None)
    sv_ = None
    if pv:
        sv_ = np.zeros(NS.content, dtype=np.int64)
        sv_[:n_gs * prm.TO] = wit["s"][:n_gs].reshape(-1)
    sc = NS.col("s", sv_)
    mc = MZ.ecol(KS, "m", np.array([[wit["m"][0]], [wit["m"][1]], [wit["m"][2]]], dtype=np.uint64) if pv else None)
    on_i = [("i", t) for t in range(SRB)]
    # Layout note: with SR = 8 row slots and TO >= 32, GS (n_gs * 32 rows) and NS (SR * TO rows) both fit a non-padded
    # space whose content is (SR - 1) * 32 rows plus a 32-row tail, so row i = SR - 1 never holds a data or meta row and
    # every view below can read GS / NS directly on S.
    gv = MZ.view(gc_, S, {"i": on_i, "x": [("k", t) for t in range(5)]})
    invv = MZ.view(invc, S, {"i": on_i, "x": [("k", t) for t in range(5)]})
    ev = MZ.view(ec, S, {"i": on_i, "x": [("k", t) for t in range(5)]})
    sv = MZ.view(sc, S, {"i": on_i, "r": [("k", prm.ib + t) for t in range(prm.ob)]})
    mv = MZ.view(mc, S, {"x": [("k", t) for t in range(6)]})
    lo = max(0, sh.n_meta - i0)
    datarow = (S.lt("i", nr) - S.lt("i", lo)) if lo else S.lt("i", nr)
    metarow = S.lt("i", min(nr, lo)) if lo else None
    k0 = S.ind(k=0)
    c0 = Ex.atom(SelLt([], 1, fixed=tuple((v, 0) for v in kv[:prm.ib])))
    one = Ex.const(1)
    eqI = Ex.atom(EqZC(list(iv)))
    eqIR = Ex.atom(EqZC(list(kv[prm.ib:]) + list(iv)))
    S.zero(bmc.ex * (bmc.ex - one), sel=datarow, tag="su.big.bool")
    S.sumzero((cdc.ex * cdc.ex - c0 * sv) * eqIR, sel=datarow, tag="su.norm")
    S.sumzero((bmc.ex * invv - (one - ev) * k0) * eqI, sel=datarow, tag="su.max")
    S.sumzero(ev * rawc.ex * eqI, sel=datarow, tag="su.empty")
    if metarow is not None:
        # specs/09 §2: W^raw is zero on meta/cfg rows, and W^r is zero on meta/cfg rows
        S.zero(rawc.ex, sel=metarow, tag="su.raw.meta_zero")
        if sh.radius:
            S.zero(wrc.ex, sel=metarow, tag="su.wr.meta_zero")
    kapv = None
    if sh.radius:
        kapv = MZ.view(kapc, S, {"i": on_i, "x": [("k", t) for t in range(5)]})
        tflv = MZ.view(tflc, S, {"i": on_i, "x": [("k", t) for t in range(5)]})
        # (v) Radius matrix W^r consistency: wr = tfl * (|cd| + kap * inex) on data rows, 0 on unused rows in block
        S.zero(wrc.ex - tflv * (absc.ex + kapv * inexc.ex), sel=datarow, tag="su.wr.def")
        if nr < SR - 1:
            S.zero(wrc.ex, sel=S.rng("i", nr, SR - 1), tag="su.wr.unused_zero")
        for c in (kapc, tflc, kinv_c):
            GS.zero(c.ex - c.shifted(-1), sel=one - GS.ind(x=0), tag="su.rep.kap")
        GS.zero(tflc.ex * (tflc.ex - one), tag="su.tfl.bool")
        GS.zero((one - tflc.ex) * kapc.ex, tag="su.tfl.zero")
        GS.zero(tflc.ex * (kapc.ex * kinv_c.ex - one), tag="su.tfl.nz")
    if b == 0:
        scale_lim = (2 * H) if sh.radius else H
        if scale_lim < nc:
            S.zero(cdc.ex, sel=S.ind(i=MC.META_SCALE) * (one - Ex.atom(SelLt(list(kv), scale_lim))), tag="su.scale.pad")
        # Zero padding row H - 1 must have SCALE[H - 1] = G_LO (and KAPPA_META[H - 1] = 0 when radius=True)
        zpt_g = S.ind(i=MC.META_SCALE, k=H - 1)
        S.zero(cdc.ex - Ex.const(prm.G_LO), sel=zpt_g, tag="su.scale.zrow")
        if sh.radius:
            S.zero(cdc.ex, sel=S.ind(i=MC.META_SCALE, k=2 * H - 1), tag="su.kap.zrow")
        if sh.cfg_rows:
            # (vi) Stable configuration row checks on CFG0 (Tt) and CFG1 (Vt)
            NT, NV, NL = sh.NT, sh.NV, sh.NL
            NB, VB = (NT - 1).bit_length(), (NV - 1).bit_length()
            TTS = ctx.space("suTT", [("idx", NB)])
            VTS = ctx.space("suVT", [("idx", VB)])
            tt_cols = [TTS.col(f"f{f}", wit["tt"][f] if pv else None) for f in range(8)]
            vt_cols = [VTS.col(f"f{f}", wit["vt"][f] if pv else None) for f in range(4)]
            HB_sp = max(5, sh.HB)
            HN = 1 << HB_sp
            HS = ctx.space("suHS", [("t", HB_sp)])
            gm_v = km_v = None
            if pv:
                gm_v = np.zeros(HN, dtype=np.int64); gm_v[:H] = wit["g_meta"]
                km_v = np.zeros(HN, dtype=np.int64); km_v[:H] = wit["k_meta"]
            g_meta_c = HS.col("gm", gm_v)
            k_meta_c = HS.col("km", km_v)
            #Note: meta_cfg0 = 2 (binary 010) and meta_cfg1 = 3 (binary 011) and META_SCALE = 0 (binary 000)
            # all have i < 4, so bit ("i", SRB - 1) is ALWAYS 0 on those rows (SRB >= 3).
            pad_bit = ("i", SRB - 1)
            for f in range(8):
                vs = MZ.view(tt_cols[f], S, {"idx": [("k", s) for s in range(NB)]}, pad_to=pad_bit)
                S.zero(cdc.ex - vs, sel=S.ind(i=sh.meta_cfg0) * S.rng("k", f * NT, (f + 1) * NT), tag=f"su.cfg0.f{f}")
            if 8 * NT < nc:
                S.zero(cdc.ex, sel=S.ind(i=sh.meta_cfg0) * S.rng("k", 8 * NT, nc), tag="su.cfg0.pad")
            for f in range(4):
                vs = MZ.view(vt_cols[f], S, {"idx": [("k", s) for s in range(VB)]}, pad_to=pad_bit)
                S.zero(cdc.ex - vs, sel=S.ind(i=sh.meta_cfg1) * S.rng("k", f * NV, (f + 1) * NV), tag=f"su.cfg1.vt{f}")
            if 4 * NV + 8 * NL < nc:
                S.zero(cdc.ex, sel=S.ind(i=sh.meta_cfg1) * S.rng("k", 4 * NV + 8 * NL, nc), tag="su.cfg1.pad")
            tm = [("k", s) for s in range(sh.HB)] + [("c", 0)] * (HB_sp - sh.HB)
            S.zero(cdc.ex - MZ.view(g_meta_c, S, {"t": tm}, pad_to=pad_bit),
                   sel=S.ind(i=MC.META_SCALE) * S.rng("k", 0, H), tag="su.cfg.gm")
            if sh.radius:
                S.zero(cdc.ex - MZ.view(k_meta_c, S, {"t": tm}, pad_to=pad_bit),
                       sel=S.ind(i=MC.META_SCALE) * S.rng("k", H, 2 * H), tag="su.cfg.km")
            # Strictly increasing keys: key[idx + 1] - key[idx] - 1 in [0, 2^44)
            dt_v = dv_v = None
            if pv:
                dt_v = np.zeros(NT, dtype=np.int64); dt_v[:NT - 1] = wit["tt"][0][1:] - wit["tt"][0][:-1] - 1
                dv_v = np.zeros(NV, dtype=np.int64); dv_v[:NV - 1] = wit["vt"][0][1:] - wit["vt"][0][:-1] - 1
            G.range_check(TTS, tt_cols[0].shifted(1) - tt_cols[0].ex - one, dt_v, 44, "su.tt.key_inc",
                          sel=TTS.lt("idx", NT - 1),
                          sel_vals=np.array([1] * (NT - 1) + [0], dtype=np.int64) if pv else None)
            G.range_check(VTS, vt_cols[0].shifted(1) - vt_cols[0].ex - one, dv_v, 44, "su.vt.key_inc",
                          sel=VTS.lt("idx", NV - 1),
                          sel_vals=np.array([1] * (NV - 1) + [0], dtype=np.int64) if pv else None)
            # Lookup (t_row, G_t, kappa_t) of every Tt entry against (t, SCALE[t], KAPPA_META[t])
            if sh.radius:
                ctx.cir.lookup_table("su.gk", HS.nv, [HS.idx("t"), g_meta_c.ex, k_meta_c.ex], H,
                                     vals=[np.arange(H, dtype=np.int64), gm_v[:H], km_v[:H]] if pv else None)
                G.wire_lookup(ctx, "su.gk", TTS, [tt_cols[1].ex, tt_cols[2].ex, tt_cols[3].ex],
                              [wit["tt"][1], wit["tt"][2], wit["tt"][3]] if pv else None)
            else:
                ctx.cir.lookup_table("su.gk", HS.nv, [HS.idx("t"), g_meta_c.ex], H,
                                     vals=[np.arange(H, dtype=np.int64), gm_v[:H]] if pv else None)
                G.wire_lookup(ctx, "su.gk", TTS, [tt_cols[1].ex, tt_cols[2].ex],
                              [wit["tt"][1], wit["tt"][2]] if pv else None)
            # Flags boolean, sentinel rows canonical, and real tile rows have kappa > 0 (when radius=True)
            fl_ex = tt_cols[7].ex
            TTS.zero(fl_ex * (fl_ex - one), tag="su.tt.flag_bool")
            TTS.zero((one - fl_ex) * (tt_cols[1].ex - Ex.const(H - 1)), tag="su.tt.sent_t")
            TTS.zero((one - fl_ex) * (tt_cols[2].ex - Ex.const(prm.G_LO)), tag="su.tt.sent_g")
            TTS.zero((one - fl_ex) * tt_cols[3].ex, tag="su.tt.sent_kap")
            TTS.zero((one - fl_ex) * (tt_cols[4].ex - one), tag="su.tt.sent_sal")
            if sh.radius:
                tkinv_v = np.array([pow(int(x), P - 2, P) if int(x) else 0 for x in wit["tt"][3]], dtype=np.uint64) if pv else None
                tkinv_c = TTS.col("kinv", tkinv_v)
                TTS.zero(fl_ex * (tt_cols[3].ex * tkinv_c.ex - one), tag="su.tt.real_kap_nz")
    for c in (gc_, invc, ec):
        GS.zero(c.ex - c.shifted(-1), sel=one - GS.ind(x=0), tag="su.rep")
    GS.zero(ec.ex * (ec.ex - one), tag="su.e.bool")
    TS = ctx.space("suT", [("t", sh.TB_BITS)], padded=False)
    cdTf, absTf, bigTf, inexTf = cd_table(prm.G_LO, sh.gbits)[4:]
    if sh.inline_table:
        spare = S.ind(i=SR - 1)
        cdT_ex = TS.pub("cdT", cdTf).on(S, {"t": "k"})
        bigT_ex = TS.pub("bigT", bigTf).on(S, {"t": "k"})
        absT_ex = TS.pub("absT", absTf).on(S, {"t": "k"}) if sh.radius else None
        inexT_ex = TS.pub("inexT", inexTf).on(S, {"t": "k"}) if sh.radius else None
    else:
        multc = TS.col("mult", wit["mult"][:TS.content] if pv else None)
        cdT_ex = TS.pub("cdT", cdTf).ex
        bigT_ex = TS.pub("bigT", bigTf).ex
        absT_ex = TS.pub("absT", absTf).ex if sh.radius else None
        inexT_ex = TS.pub("inexT", inexTf).ex if sh.radius else None

    # ------------------------------------------------------------------ round A: shared challenges, LogUp helper, link
    def round_a(tr):
        t0 = time.time()
        tr.absorb(glob["seed2"])
        mu = tr.ext(); lam = tr.ext()
        mu2 = hmul(mu, mu); mu3 = hmul(mu2, mu)
        mu4 = hmul(mu3, mu); mu5 = hmul(mu4, mu)
        hv, htv = block_h(sh, wit, mu, lam) if pv else (None, None)
        h = MZ.ecol(S, "h", hv)
        key = rawc.ex + gv * Ex.const(mu) + cdc.ex * Ex.const(mu2) + bmc.ex * Ex.const(mu3)
        if sh.radius:
            key = key + absc.ex * Ex.const(mu4) + inexc.ex * Ex.const(mu5)
        S.zero(h.ex * (Ex.const(lam) - key) - one, sel=datarow, tag="su.lu.data")
        if sh.inline_table:
            codeT = Ex.atom(IndexP(list(kv[:16])))
            gT = Ex.const(prm.G_LO) + Ex.atom(IndexP(list(kv[16:16 + sh.gbits])))
            keyT = codeT + gT * Ex.const(mu) + cdT_ex * Ex.const(mu2) + bigT_ex * Ex.const(mu3)
            if sh.radius:
                keyT = keyT + absT_ex * Ex.const(mu4) + inexT_ex * Ex.const(mu5)
            S.zero(h.ex * (Ex.const(lam) - keyT) + bmc.ex, sel=spare, tag="su.lu.table")
            S.sumzero(h.ex, sel=datarow + spare, tag="su.lu.sum")
        else:
            tv = TS.vars("t")
            codeT = Ex.atom(IndexP(list(tv[:16])))
            gT = Ex.const(prm.G_LO) + Ex.atom(IndexP(list(tv[16:16 + sh.gbits])))
            keyT = codeT + gT * Ex.const(mu) + cdT_ex * Ex.const(mu2) + bigT_ex * Ex.const(mu3)
            if sh.radius:
                keyT = keyT + absT_ex * Ex.const(mu4) + inexT_ex * Ex.const(mu5)
            ht = MZ.ecol(TS, "ht", htv)
            TS.zero(ht.ex * (Ex.const(lam) - keyT) + multc.ex, tag="su.lu.table")
            sumS_v = CPU.esum(hv[:, np.concatenate([np.arange(i * nc, (i + 1) * nc) for i in wit["data"]])]) if pv else None
            sumS_c = MZ.ecol(KS, "sumS", np.array([[sumS_v[0]], [sumS_v[1]], [sumS_v[2]]], dtype=np.uint64) if pv else None)
            S.sumzero(datarow * h.ex - S.ind(i=0, k=0) * MZ.view(sumS_c, S, {"x": [("k", t) for t in range(6)]}), tag="su.lu.sumS")
            TS.sumzero(TS.sel() * ht.ex + TS.ind(t=0) * MZ.view(sumS_c, TS, {"x": [("t", t) for t in range(6)]}), tag="su.lu.sumT")
        T = block_T(sh, wit, glob, wit["m"]) if pv else tuple(int(x) % P for x in tval)
        out["T"] = T
        tr.absorb([int(T[0]), int(T[1]), int(T[2])])
        rc = np.zeros((3, SR), dtype=np.uint64); rr = np.zeros((3, SR), dtype=np.uint64)
        rwr = np.zeros((3, SR), dtype=np.uint64) if sh.radius else None
        wG = np.zeros((3, SR), dtype=np.uint64); wN = np.zeros((3, SR * prm.TO), dtype=np.uint64)
        wK = np.zeros((3, SR), dtype=np.uint64) if sh.radius else None
        for i in range(nr):
            rc[:, i] = glob["rcd"][i0 + i]; rr[:, i] = glob["rraw"][i0 + i]
            if sh.radius:
                rwr[:, i] = glob["rcd"][H + i0 + i]
            if i0 + i >= sh.n_meta:
                wG[:, i] = glob["eqG"][:, i0 + i]
                if sh.radius:
                    wK[:, i] = glob["eqK"][:, i0 + i]
                wN[:, i * prm.TO:(i + 1) * prm.TO] = glob["eqN"][:, (i0 + i) * prm.TO:(i0 + i + 1) * prm.TO]
        gq = Ex.atom(MZ.GQ(prm, glob["Q"], glob["beta"], list(kv)))
        at0 = S.ind(i=0, k=0)
        link = (Ex.atom(MZ.EVec("rc", rc, list(iv))) * gq * cdc.ex
                + Ex.atom(MZ.EVec("rr", rr, list(iv))) * gq * rawc.ex
                + Ex.atom(MZ.EVec("wG", wG, list(iv))) * k0 * gv
                + Ex.atom(MZ.EVec("wN", wN, list(kv[prm.ib:]) + list(iv))) * c0 * sv
                + at0 * mv - at0 * Ex.const(T))
        if sh.radius:
            link = (link + Ex.atom(MZ.EVec("rwr", rwr, list(iv))) * gq * wrc.ex
                    + Ex.atom(MZ.EVec("wK", wK, list(iv))) * k0 * kapv)
        if b == 0:
            eqG = Ex.atom(EqP(list(kv[:sh.HB]), glob["zG"], fixed=tuple((v, 0) for v in kv[sh.HB:])))
            eqN = Ex.atom(EqP(list(kv[:sh.PNB]), glob["zN"], fixed=tuple((v, 0) for v in kv[sh.PNB:])))
            link = link - S.ind(i=MC.META_SCALE) * eqG * cdc.ex - S.ind(i=MC.META_PN) * eqN * cdc.ex
            if sh.radius:
                fixedK = ((kv[sh.HB], 1),) + tuple((v, 0) for v in kv[sh.HB + 1:])
                eqK = Ex.atom(EqP(list(kv[:sh.HB]), glob["zK"], fixed=fixedK))
                link = link - S.ind(i=MC.META_SCALE) * eqK * cdc.ex
        S.sumzero(link, tag="su.link")
        out["round_a_s"] = round(time.time() - t0, 3)

    cir.rounds = [round_a]
    return ctx


# ============================================================================ column sub-proof
def col_witness(sh, src, Q):
    """Hidden openings of both trees at Q: columns (R_MAX, m) per tree, salts (4, m), paths (m, depth, 4)."""
    cols = [src.columns(t, Q) for t in (0, 1)]
    salts = [np.ascontiguousarray(np.stack([np.asarray(src.salts(t)[:, j], dtype=np.uint64) for j in Q], axis=1))
             for t in (0, 1)]
    paths = [[src.path(t, j) for j in Q] for t in (0, 1)]
    return {"cols": cols, "salts": salts, "paths": paths, "pre": src.pre}


def ps_inputs_setup(sh, Q, w):
    """Permutation inputs (12, content) of the column proof's Poseidon2 rows; returns (inp, roots[t], model_id)."""
    prm = sh.prm
    n = (1 << (1 + sh.QB + sh.CB)) - TB
    inp = np.zeros((12, n), dtype=np.uint64)
    mq = sh.MQS
    Qa = np.asarray(Q, dtype=np.uint64)
    roots = []
    for t in (0, 1):
        base = (t << (sh.QB + sh.CB)) + (np.arange(mq, dtype=np.int64) << sh.CB)
        st = np.zeros((12, mq), dtype=np.uint64)
        st[0] = Qa; st[1:5] = w["salts"][t]; st[5] = prm.R_MAX; st[10] = MC.DOM_COL; st[11] = PS.TAG_SPONGE
        inp[:, base + sh.INIT] = st
        st = PS.permute(st, CPU)
        cols = w["cols"][t]
        for bb in range(sh.A):
            st = st.copy()
            st[0:8] = CPU.add(st[0:8], np.ascontiguousarray(cols[8 * bb:8 * bb + 8]))
            inp[:, base + bb] = st
            st = PS.permute(st, CPU)
        cur = st[0:4].copy()
        sib = np.asarray(w["paths"][t], dtype=np.uint64)            # (m, depth, 4)
        for d in range(sh.depth):
            bit = ((Qa >> np.uint64(d)) & np.uint64(1)).astype(bool)
            s = np.ascontiguousarray(sib[:, d, :].T)
            st = np.zeros((12, mq), dtype=np.uint64)
            st[0:4] = np.where(bit, s, cur); st[4:8] = np.where(bit, cur, s); st[11] = PS.TAG_NODE
            inp[:, base + sh.A + d] = st
            cur = PS.permute(st, CPU)[0:4]
        roots.append(cur)
    pre = w["pre"]
    stA = np.zeros((12, 1), dtype=np.uint64)
    stA[0:4, 0] = pre["s0"]; stA[4:8, 0] = roots[0][:, mq - 1]; stA[10] = MC.DOM_MID; stA[11] = PS.TAG_SPONGE
    inp[:, sh.MID] = stA[:, 0]
    stB = PS.permute(stA, CPU).copy()
    stB[0:4, 0] = CPU.add(stB[0:4, 0], roots[1][:, mq - 1].copy())
    stB[4:8, 0] = CPU.add(stB[4:8, 0], np.asarray(pre["c_cfg"], dtype=np.uint64))
    inp[:, sh.MID + 1] = stB[:, 0]
    mid = tuple(int(v) for v in PS.permute(stB, CPU)[0:4, 0])
    return inp, roots, mid


def col_T(sh, w, glob, m):
    acc = m
    for t, r in ((0, glob["rcd"]), (1, glob["rraw"])):
        cols = w["cols"][t]                                          # (R_MAX, m) field elements
        rv = _ev(r, sh.prm.R_MAX)
        for q, bq in enumerate(glob["beta"]):
            acc = hsub(acc, hmul(bq, CPU.edot_dev(rv, CPU.ext(np.ascontiguousarray(cols[:, q])))))
    return acc


def build_col(sh, stmt, wit=None, tval=None, out=None, B=CPU):
    """Column sub-proof (one per pi_setup).  out receives seed2, glob and T_col (computed inside the commit round)."""
    prm = sh.prm
    pv = wit is not None
    ctx = G.Ctx(pv)
    cir = ctx.cir
    IB, QB, CB, mq = sh.IB, sh.QB, sh.CB, sh.MQS
    out = {} if out is None else out
    seed1 = bytes.fromhex(stmt["seed1"])
    Q = q_of(sh, seed1)
    MS = ctx.space("scM", [("t", 1), ("q", QB), ("i", IB)])
    M_v = None
    if pv:
        M_v = np.zeros((2, 1 << QB, 1 << IB), dtype=np.uint64)
        for t in (0, 1):
            M_v[t, :mq] = wit["cols"][t].T
        M_v = M_v.reshape(-1)
    M = MS.col("M", M_v)
    KS = ctx.space("scK", [("x", 6)], padded=False)
    mk = MZ.ecol(KS, "m", np.array([[wit["m"][0]], [wit["m"][1]], [wit["m"][2]]], dtype=np.uint64) if pv else None)
    inp = None
    if pv:
        t1 = time.time()
        if sh.radius and wit["cols"][1][sh.H:].any():
            ctx.violation("upper-half M^raw[H:, j_q] is non-zero")
        inp, roots, mid = ps_inputs_setup(sh, Q, wit)
        out["ps_inputs_s"] = round(time.time() - t1, 3)
        if tuple(int(x) for x in roots[0][:, mq - 1]) != tuple(wit["pre"]["c_w"]) or \
                any(tuple(int(x) for x in roots[0][:, q]) != tuple(wit["pre"]["c_w"]) for q in range(mq)):
            ctx.violation("opened W column does not authenticate to C_W")
        if any(tuple(int(x) for x in roots[1][:, q]) != tuple(wit["pre"]["c_raw"]) for q in range(mq)):
            ctx.violation("opened raw column does not authenticate to C_raw")
        if mid != tuple(int(x) for x in stmt["model_id"]):
            ctx.violation("model_id preimage mismatch")
    if sh.radius:
        # specs/09 §2: W^raw is zero on the upper half [H, R_MAX) -> constrain M^raw[H:, j_q] == 0
        MS.zero(M.ex, sel=MS.ind(t=1) * MS.lt("q", mq) * MS.rng("i", sh.H, prm.R_MAX), tag="sc.raw.upper_zero")
    PSp = ctx.space("scPS", [("t", 1), ("q", QB), ("b", CB)], padded=False)
    reg = G.PoseRegion(PSp, inp, B=B if pv else None)
    I = lambda k: reg.I(k).ex
    O = lambda k: reg.O(k).ex
    Osh = lambda k, d: reg.O(k).shifted(d)
    Ish = lambda k, d: reg.I(k).shifted(d)
    cst = lambda c: Ex.const(c % P)
    nv = PSp.nv
    allv = list(range(nv))

    def pt(x):
        return Ex.atom(SelLt([], 1, fixed=tuple((v, (x >> v) & 1) for v in allv)))

    jq = np.zeros(1 << nv, dtype=np.int64); pb = np.zeros(1 << nv, dtype=np.int64)
    for t in (0, 1):
        for q, j in enumerate(Q):
            x = (t << (QB + CB)) + (q << CB)
            jq[x + sh.INIT] = j
            for d in range(sh.depth):
                pb[x + sh.A + d] = (j >> d) & 1
    jq_ex = PSp.pub("jq", jq).ex; pb_ex = PSp.pub("pbit", pb).ex
    colsel = PSp.lt("q", mq)
    s_init = colsel * PSp.ind(b=sh.INIT)
    PSp.zero(I(0) - jq_ex, sel=s_init, tag="sc.leaf.idx")
    PSp.zero(I(5) - cst(prm.R_MAX), sel=s_init, tag="sc.leaf.rmax")
    for k in (6, 7, 8, 9):
        PSp.zero(I(k), sel=s_init, tag="sc.leaf.zero")
    PSp.zero(I(10) - cst(MC.DOM_COL), sel=s_init, tag="sc.leaf.dom")
    PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=s_init, tag="sc.leaf.tag")
    first = PSp.ind(b=0); absorb = PSp.lt("b", sh.A); rest = absorb - first
    s_abs = colsel * absorb
    qmap = [("q", s) for s in range(QB)]
    for k in range(12):
        prev = first * Osh(k, sh.INIT) + rest * Osh(k, -1)
        e = I(k) - prev
        if k < 8:
            imap = [("c", (k >> s) & 1) for s in range(3)] + [("b", s) for s in range(IB - 3)]
            e = e - MZ.view(M, PSp, {"t": [("t", 0)], "q": qmap, "i": imap}, pad_to=("b", IB - 3))
        PSp.zero(e, sel=s_abs, tag="sc.leaf.absorb")
    s_path = colsel * PSp.rng("b", sh.A, sh.A + sh.depth)
    for k in range(4):
        cur = Osh(k, -1)
        PSp.zero((Ex.const(1) - pb_ex) * (I(k) - cur) + pb_ex * (I(4 + k) - cur), sel=s_path, tag="sc.path")
    for k in (8, 9, 10):
        PSp.zero(I(k), sel=s_path, tag="sc.path.pad")
    PSp.zero(I(11) - cst(PS.TAG_NODE), sel=s_path, tag="sc.path.tag")
    x0 = ((mq - 1) << CB) + sh.ROOT
    x1 = (1 << (QB + CB)) + x0
    for k in range(4):
        PSp.zero(Osh(k, 1 << CB) - O(k), sel=PSp.lt("q", mq - 1) * PSp.ind(b=sh.ROOT), tag="sc.root.chain")
        PSp.zero(O(k) - Ish(4 + k, sh.MID - x0), sel=pt(x0), tag="sc.root.cw")
        PSp.zero(I(k) - Osh(k, -1) - Osh(k, x1 - (sh.MID + 1)), sel=pt(sh.MID + 1), tag="sc.root.craw")
    for k in (8, 9):
        PSp.zero(I(k), sel=pt(sh.MID), tag="sc.mid.zero")
    PSp.zero(I(10) - cst(MC.DOM_MID), sel=pt(sh.MID), tag="sc.mid.dom")
    PSp.zero(I(11) - cst(PS.TAG_SPONGE), sel=pt(sh.MID), tag="sc.mid.tag")
    for k in range(8, 12):
        PSp.zero(I(k) - Osh(k, -1), sel=pt(sh.MID + 1), tag="sc.mid.cap")
    for k in range(4):
        PSp.zero(O(k) - cst(int(stmt["model_id"][k])), sel=pt(sh.MID + 1), tag="sc.mid.out")

    def round_a(tr):
        seed2 = seed2_of(seed1, cir.root1)
        tr.absorb(seed2)
        glob = glob_of(sh, seed1, seed2)
        out["seed2"], out["glob"] = seed2, glob
        T = col_T(sh, wit, glob, wit["m"]) if pv else tuple(int(x) % P for x in tval)
        out["T"] = T
        tr.absorb([int(T[0]), int(T[1]), int(T[2])])
        bq = _ev(glob["beta"], 1 << QB)
        rt = np.concatenate([_ev(glob["rcd"], 1 << IB), _ev(glob["rraw"], 1 << IB)], axis=1)
        at0 = MS.ind(t=0, q=0, i=0)
        xm = ([("i", t) for t in range(IB)] + [("q", t) for t in range(QB)])[:6]
        mvk = MZ.view(mk, MS, {"x": xm})
        e = (Ex.const(hneg(ONE)) * Ex.atom(MZ.EVec("beta", bq, MS.vars("q"))) * Ex.atom(MZ.EVec("r", rt, MS.vars("i") + MS.vars("t")))
             * M.ex + at0 * mvk - at0 * Ex.const(T))
        MS.sumzero(e, tag="sc.link")

    cir.rounds = [round_a]
    return ctx


# ============================================================================ sources (prover side)
def columns_from_coeffs(prm, row_fn, n_used, Q, B, batch=32):
    """M[:, Q] (R_MAX, m) for the rows 0..n_used-1 given by row_fn(i) (int64), recomputed by NTT (no stored codeword)."""
    out = np.zeros((prm.R_MAX, len(Q)), dtype=np.uint64)
    Qd = np.asarray(Q, dtype=np.int64)
    gpu = MC._is_gpu(B)
    for a in range(0, n_used, batch):
        rows = list(range(a, min(n_used, a + batch)))
        C = np.stack([row_fn(i) for i in rows])
        Cf = CPU.fromi(C)
        if gpu:
            import cupy as cp
            Cd = cp.asarray(Cf); Qc = cp.asarray(Qd)
            for k, i in enumerate(rows):
                out[i] = cp.asnumpy(B.ntt(Cd[k], prm.n)[Qc])
            del Cd
        else:
            for k, i in enumerate(rows):
                out[i] = CPU.ntt(Cf[k], prm.n)[Qd]
    return out


class MemSource:
    """In-memory commitment of explicit W^cd / W^raw matrices (tests and small models): both trees built with
    modelcommit.commit_matrix on B.  rand_rows: {row: seed} replaces that codeword row of C_W by a uniformly random
    vector (a non-codeword row; tamper test).  rand_raw_rows: {row: seed} same on C_raw."""

    def __init__(self, prm, Wcd, Wraw, B=CPU, rand_rows=None, rand_raw_rows=None, c_cfg=None):
        self.prm, self.B = prm, B
        self.Wcd = np.asarray(Wcd, dtype=np.int64); self.Wraw = np.asarray(Wraw, dtype=np.int64)
        self.n_used = self.Wcd.shape[0]
        self.n_raw_used = self.Wraw.shape[0]
        self.rand_rows = dict(rand_rows or {})
        self.rand_raw_rows = dict(rand_raw_rows or {})
        self._rand = {}

        def hook(g0, X):
            for r, sd in self.rand_rows.items():
                if g0 <= r < g0 + prm.GROUP:
                    X[r - g0] = B.asarray(self.rand_row(("w", r), sd))

        def hook_raw(g0, X):
            for r, sd in self.rand_raw_rows.items():
                if g0 <= r < g0 + prm.GROUP:
                    X[r - g0] = B.asarray(self.rand_row(("r", r), sd))

        nolog = lambda *a: None
        n_w_commit = max(self.n_used, max(self.rand_rows, default=-1) + 1)
        n_r_commit = max(self.n_raw_used, max(self.rand_raw_rows, default=-1) + 1)
        Wcd_pad = self.Wcd if n_w_commit == self.n_used else np.pad(self.Wcd, ((0, n_w_commit - self.n_used), (0, 0)))
        Wraw_pad = self.Wraw if n_r_commit == self.n_raw_used else np.pad(self.Wraw, ((0, n_r_commit - self.n_raw_used), (0, 0)))
        self.com = [MC.commit_matrix(MC.MatrixSource(Wcd_pad), n_w_commit, prm, B, log=nolog,
                                     cw_hook=hook if self.rand_rows else None),
                    MC.commit_matrix(MC.MatrixSource(Wraw_pad), n_r_commit, prm, B, log=nolog,
                                     cw_hook=hook_raw if self.rand_raw_rows else None)]
        s0 = tuple(int(x) for x in CPU.asarray(MC.rand_field(4)))
        c_cfg = tuple(int(x) for x in MC.rand_field(4)) if c_cfg is None else tuple(c_cfg)
        self.pre = {"s0": s0, "c_w": self.com[0]["root"], "c_raw": self.com[1]["root"], "c_cfg": c_cfg}
        self.model_id = MC.model_id(s0, self.pre["c_w"], self.pre["c_raw"], c_cfg)

    def rand_row(self, key, seed):
        if key not in self._rand:
            with seeded_rng(hashlib.blake2b(f"randrow:{key}:{seed}".encode(), digest_size=32).digest()):
                self._rand[key] = MC.rand_field(self.prm.n)
        return self._rand[key]

    def cd_row(self, i):
        return self.Wcd[i] if i < self.n_used else np.zeros(self.prm.nc, dtype=np.int64)

    def raw_row(self, i):
        return self.Wraw[i] if i < self.n_raw_used else np.zeros(self.prm.nc, dtype=np.int64)

    def columns(self, t, Q):
        W, nu = (self.Wcd, self.n_used) if t == 0 else (self.Wraw, self.n_raw_used)
        out = columns_from_coeffs(self.prm, lambda i: W[i], nu, Q, self.B)
        if t == 0:
            for r, sd in self.rand_rows.items():
                out[r] = self.rand_row(("w", r), sd)[np.asarray(Q)]
        else:
            for r, sd in self.rand_raw_rows.items():
                out[r] = self.rand_row(("r", r), sd)[np.asarray(Q)]
        return out

    def salts(self, t):
        return self.com[t]["salts"]

    def path(self, t, j):
        lv = self.com[t]["levels"]
        o = []
        for d in range(self.prm.depth):
            o.append(tuple(int(v) for v in lv[d][:, j ^ 1])); j >>= 1
        return o


class DiskSource:
    """modelcommit store directory (commit with layout v2): W / RAW coefficients, trees, salts, model_id preimage."""

    def __init__(self, d, prm, B):
        self.prm, self.B = prm, B
        self.st = [MC.Store(d, prm, "W"), MC.Store(d, prm, "RAW")]
        meta = json.load(open(os.path.join(d, "commit.json")))
        sec = meta["secret"]
        self.n_used = self.st[0].n_used
        self.pre = {"s0": tuple(sec["s0"]), "c_w": tuple(self.st[0].root), "c_raw": tuple(self.st[1].root),
                    "c_cfg": tuple(sec["c_cfg"])}
        self.model_id = tuple(meta["result"]["model_id"])
        assert MC.model_id(self.pre["s0"], self.pre["c_w"], self.pre["c_raw"], self.pre["c_cfg"]) == self.model_id

    def cd_row(self, i):
        return self.st[0].row(i) if i < self.n_used else np.zeros(self.prm.nc, dtype=np.int64)

    def raw_row(self, i):
        return self.st[1].row(i) if i < self.st[1].n_used else np.zeros(self.prm.nc, dtype=np.int64)

    def columns(self, t, Q):
        st = self.st[t]
        return columns_from_coeffs(self.prm, st.row, st.n_used, Q, self.B)

    def salts(self, t):
        return self.st[t].salts

    def path(self, t, j):
        return self.st[t].path(j)


# ============================================================================ prove / verify
def _free(B):
    gc.collect()
    if MC._is_gpu(B):
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def _cells(cir):
    out = {}
    for v in cir.vecs:
        sp = v.name.split(".")[0]
        out[sp] = out.get(sp, 0) + v.N
    return out


def prove_setup(prm, sp, src, B=CPU, out_dir=None, log=print, blocks=None, tamper=None):
    """Two-pass prover.  Returns (bundle, info).  out_dir: block proofs are written there (bundle holds file names)
    instead of being kept in RAM.  blocks: prove only these block indices in pass 2 (cost measurement; the bundle is
    then incomplete and does not verify).  tamper(stage, obj): test hook (e.g. alter a block witness)."""
    sh = SShape(prm, sp)
    st0 = statement0(prm, sp, src.model_id)
    info = {"blocks": sh.nblocks, "pass1": [], "pass2": []}
    K = os.urandom(32)
    kd = lambda *t: hashlib.blake2b(K + json.dumps(t).encode(), digest_size=32).digest()
    masks = []
    for b in range(sh.nblocks):
        with seeded_rng(kd("mask", b)):
            masks.append(rand_ext())
    m_col = hneg(_esum(masks))
    t_all = time.time()
    roots1 = []
    for b in range(sh.nblocks):
        t0 = time.time()
        w = block_witness(sh, b, src); w["m"] = masks[b]
        if tamper:
            tamper("witness", w)
        tw = time.time() - t0
        with seeded_rng(kd("blk", b)):
            ctx = build_block(sh, b, stmt_block(st0, b), w, B=B)
            r = Z.prove(ctx.cir, stmt_block(st0, b), B, label=LBL_BLK, phase1_only=True)
        roots1.append(r["root1"])
        info["pass1"].append({"b": b, "witness_s": round(tw, 2), "s": round(time.time() - t0, 2)})
        log(f"[setup] pass 1 block {b + 1}/{sh.nblocks}: {time.time() - t0:.1f}s (witness {tw:.1f}s)")
        del ctx, w, r
        _free(B)
    seed1 = seed1_of(st0, roots1)
    Q = q_of(sh, seed1)
    t0 = time.time()
    wc = col_witness(sh, src, Q); wc["m"] = m_col
    tcw = time.time() - t0
    oc = {}
    sc = stmt_col(st0, seed1)
    ctx = build_col(sh, sc, wc, out=oc, B=B)
    with Z.PoolPeak() as pk:
        pf_col = Z.prove(ctx.cir, sc, B, label=LBL_COL)
    glob = oc["glob"]
    glob["_gq"] = gq_table(sh, glob, B)
    info["col"] = {"witness_s": round(tcw, 2), "s": round(time.time() - t0, 2), "cells": int(sum(_cells(ctx.cir).values())),
                   "cells_by_space": _cells(ctx.cir), "pool_peak_gib": round(pk.peak / (1 << 30), 3),
                   "proof_bytes": len(dumps(pf_col)), "violations": list(ctx.notes)}
    log(f"[setup] column proof: {info['col']['s']}s, {info['col']['cells']} cells")
    del ctx, wc
    _free(B)
    T = [None] * sh.nblocks
    pfs = [None] * sh.nblocks
    for b in (range(sh.nblocks) if blocks is None else blocks):
        t0 = time.time()
        w = block_witness(sh, b, src); w["m"] = masks[b]
        if tamper:
            tamper("witness", w)
        ob = {}
        with seeded_rng(kd("blk", b)):
            ctx = build_block(sh, b, stmt_block(st0, b), w, glob=glob, out=ob, B=B)
            with Z.PoolPeak() as pk:
                pf = Z.prove(ctx.cir, stmt_block(st0, b), B, label=LBL_BLK)
        if pf["root1"] != roots1[b]:
            raise RuntimeError(f"block {b}: phase-1 commitment not reproduced in pass 2")
        T[b] = ob["T"]
        blob = dumps(pf)
        rec = {"b": b, "s": round(time.time() - t0, 2), "cells": int(sum(_cells(ctx.cir).values())),
               "cells_by_space": _cells(ctx.cir), "pool_peak_gib": round(pk.peak / (1 << 30), 3),
               "proof_bytes": len(blob), "violations": list(ctx.notes), "rss_gib": round(_rss_gib(), 2)}
        if out_dir:
            fn = os.path.join(out_dir, f"block{b:03d}.pf")
            with open(fn, "wb") as fh:
                fh.write(blob)
            pfs[b] = fn
        else:
            pfs[b] = pf
        info["pass2"].append(rec)
        log(f"[setup] pass 2 block {b + 1}/{sh.nblocks}: {rec['s']}s, {rec['cells']} cells, pool {rec['pool_peak_gib']} GiB, "
            f"{rec['proof_bytes']} B")
        del ctx, w, pf, blob
        _free(B)
    info["total_s"] = round(time.time() - t_all, 2)
    bundle = {"v": VERSION, "stmt0": st0, "roots1": roots1, "col": pf_col, "T_col": list(oc["T"]),
              "T": [list(t) if t is not None else None for t in T], "blocks": pfs}
    return bundle, info


def _load(x):
    if isinstance(x, str):
        with open(x, "rb") as fh:
            return loads(fh.read())
    return x


def verify_setup(prm, sp, model_id, bundle, log=None):
    """Raises Z.ProofError on rejection; returns per-part CPU times."""
    sh = SShape(prm, sp)
    st0 = statement0(prm, sp, model_id)
    if bundle.get("v") != VERSION or bundle.get("stmt0") != json.loads(json.dumps(st0)):
        raise Z.ProofError("statement mismatch")
    if len(bundle["blocks"]) != sh.nblocks or len(bundle["T"]) != sh.nblocks or any(t is None for t in bundle["T"]):
        raise Z.ProofError("wrong number of row-block proofs")
    times = {}
    t0 = time.time()
    pfs = [_load(x) for x in bundle["blocks"]]
    roots1 = [pf["root1"] for pf in pfs]
    seed1 = seed1_of(st0, roots1)
    sc = stmt_col(st0, seed1)
    oc = {}
    ctx = build_col(sh, sc, None, tval=bundle["T_col"], out=oc)
    Z.verify(ctx.cir, sc, bundle["col"], label=LBL_COL)
    glob = oc["glob"]
    times["col_s"] = round(time.time() - t0, 3)
    tb = []
    for b in range(sh.nblocks):
        t1 = time.time()
        ctx = build_block(sh, b, stmt_block(st0, b), None, glob=glob, tval=bundle["T"][b])
        Z.verify(ctx.cir, stmt_block(st0, b), pfs[b], label=LBL_BLK)
        tb.append(round(time.time() - t1, 3))
        if log:
            log(f"[verify] block {b + 1}/{sh.nblocks} {tb[-1]}s")
    times["blocks_s"] = tb
    tot = _esum([tuple(int(x) % P for x in t) for t in bundle["T"]] + [tuple(int(x) % P for x in bundle["T_col"])])
    if tot != ZERO:
        raise Z.ProofError("joint column / exponent / norm link: sum of the sub-proof scalars T is not zero")
    times["total_s"] = round(time.time() - t0, 3)
    return times


def try_verify(prm, sp, model_id, bundle):
    try:
        verify_setup(prm, sp, model_id, bundle)
        return "accepted"
    except Z.ProofError as e:
        return "rejected: " + str(e)


def load_bundle(proof_dir):
    """Load a bundle saved by cmd_qwen from proof_dir/bundle.pf, resolving relative block paths."""
    with open(os.path.join(proof_dir, "bundle.pf"), "rb") as fh:
        bundle = loads(fh.read())
    bundle["blocks"] = [
        os.path.join(proof_dir, p) if isinstance(p, str) and not os.path.isabs(p) else p
        for p in bundle["blocks"]
    ]
    return bundle


def bundle_bytes(bundle):
    n = len(dumps({k: v for k, v in bundle.items() if k != "blocks"}))
    for x in bundle["blocks"]:
        n += os.path.getsize(x) if isinstance(x, str) else len(dumps(x))
    return n


def _rss_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


# ============================================================================ full-model driver
def cmd_qwen(a):
    from .field import gpu_backend
    B = gpu_backend()
    meta = json.load(open(os.path.join(a.store, "commit.json")))
    rad = bool(meta["W"].get("radius"))
    cfgr = bool(meta["W"].get("cfg_rows"))
    prm = MC.QWEN_PARAMS_R if rad else MC.QWEN_PARAMS
    sp = SetupParams(RADIUS=rad, CFG_ROWS=cfgr)
    src = DiskSource(a.store, prm, B)
    sh = SShape(prm, sp)
    os.makedirs(a.proof_dir, exist_ok=True)
    blocks = None if a.blocks is None else list(range(a.blocks))
    t0 = time.time()
    bundle, info = prove_setup(prm, sp, src, B, out_dir=a.proof_dir, blocks=blocks)
    wall = time.time() - t0
    res = {"model": os.path.basename(os.path.normpath(a.model)), "machine": a.machine or describe_machine(), "version": VERSION,
           "params": asdict(prm), "setup": asdict(sp), "rows_used": src.n_used, "blocks": sh.nblocks,
           "blocks_proven_pass2": len(info["pass2"]), "prove_wall_s": round(wall, 1), "info": info,
           "peak_rss_gib": round(_rss_gib(), 2)}
    if blocks is None:
        bundle_disk = dict(bundle, blocks=[os.path.basename(p) if isinstance(p, str) else p for p in bundle["blocks"]])
        with open(os.path.join(a.proof_dir, "bundle.pf"), "wb") as fh:
            fh.write(dumps(bundle_disk))
        res["proof_bytes"] = bundle_bytes(bundle)
        t1 = time.time()
        try:
            res["verify_times"] = verify_setup(prm, sp, src.model_id, bundle)
            res["verify"] = "accepted"
        except Z.ProofError as e:
            res["verify"] = "rejected: " + str(e)
        res["verify_cpu_s"] = round(time.time() - t1, 1)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "info"}, default=str), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qwen")
    q.add_argument("--model", required=True); q.add_argument("--store", required=True)
    q.add_argument("--proof-dir", default="setup_proof"); q.add_argument("--out", default=None)
    q.add_argument("--blocks", type=int, default=None, help="prove only the first N blocks in pass 2 (cost probe)")
    q.add_argument("--machine", default=None, help="host description (default: detected)")
    a = ap.parse_args(argv)
    {"qwen": cmd_qwen}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
