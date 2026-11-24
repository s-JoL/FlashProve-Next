# SPDX-License-Identifier: Apache-2.0
"""Detached pure-Python reference verifier for `FPN2` Relation 2 (`LLM_INFERENCE_V2`).

Bit-exact twin of `core/src/relation2.rs` (zero imports of `prover/` or `flashprove_next/`):
  * Independent left/right GKR denominator blinders (`m = m_mask - 1`);
  * `MIN_STANDARD_QUERIES = 72` enforced under `STANDARD` tier;
  * Pin-before-decode (`expected_circuit_root` & `expected_stmt_digest` checked before parsing `raw_desc`);
  * Full bounds/alignment/non-overlap validation inside `decode_circuit_desc`;
  * Only linear GKR numerator tags `0` and `1` admitted (`>= 2` rejected).
"""
import hashlib
import struct
from typing import List, Optional

from .field import ONE3, P, ZERO3, NonCanonical, e_add, e_decode, e_mul, e_sub, lagrange_eval
from .folded_gkr import RATE, _eq, _eq1, _leaf, _merkle_ok, fold4 as _fold4
from .profile import HEADER_LEN, REL_LLM_INFERENCE_V2, TIER_REDUCED, TIER_STANDARD, lookup
from .transcript import LABEL_V2, Fpn2Transcript
from .wire import Reject, parse_header

DESC_MAGIC = b"FPN2DESC"
DESC_VERSION = 1
PERSONAL_CDESC = b"FPN2-CDESC-v2\x00\x00\x00"
PERSONAL_STMT = b"FPN2-STMT-v2\x00\x00\x00\x00"
MIN_STANDARD_QUERIES = 72


def circuit_desc_root(raw_desc: bytes) -> bytes:
    return hashlib.blake2b(raw_desc, digest_size=32, person=PERSONAL_CDESC).digest()


def stmt_digest_bytes(raw_stmt: bytes) -> bytes:
    return hashlib.blake2b(raw_stmt, digest_size=32, person=PERSONAL_STMT).digest()


def _tail_len_for(queries: int) -> int:
    return ((4 * queries + 16) + 15) & ~15


def _libra_group_n(n_max: int, queries: int) -> int:
    need = 3 * n_max + _tail_len_for(queries)
    n = 8
    while (1 << n) < need:
        n += 2
    return n


def _e_encode(v) -> bytes:
    return v[0].to_bytes(8, "little") + v[1].to_bytes(8, "little") + v[2].to_bytes(8, "little")


def _mix(a, b, r):
    return e_add(a, e_mul(r, e_sub(b, a)))


def _combine_frac(nl, dl, nr, dr):
    return e_add(e_mul(nl, dr), e_mul(nr, dl)), e_mul(dl, dr)


class _Cur:
    def __init__(self, b: bytes):
        self.b = b
        self.off = 0

    def remaining(self) -> int:
        return len(self.b) - self.off

    def check_count(self, count: int, min_elem_bytes: int) -> int:
        if count * min_elem_bytes > self.remaining():
            raise Reject("E_LENGTH")
        return count

    def take(self, n: int) -> bytes:
        if self.off + n > len(self.b):
            raise Reject("E_LENGTH")
        s = self.b[self.off:self.off + n]
        self.off += n
        return s

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def fp3(self):
        try:
            return e_decode(self.take(24))
        except NonCanonical:
            raise Reject("E_NONCANONICAL") from None


def _validate_place(place, group_ns: List[int], queries: int):
    gi, offset, nbits = place
    if gi >= len(group_ns):
        raise Reject("E_NONCANONICAL")
    ng = group_ns[gi]
    if nbits >= ng:
        raise Reject("E_NONCANONICAL")
    sz = 1 << nbits
    cap = (1 << ng) - _tail_len_for(queries)
    if offset % sz != 0 or offset + sz > cap:
        raise Reject("E_NONCANONICAL")


def _decode_place(c: _Cur, group_ns: List[int], queries: int):
    pl = (c.u16(), c.u32(), c.u8())
    _validate_place(pl, group_ns, queries)
    return pl


def _decode_view_ref(c: _Cur, group_ns: List[int], queries: int, cube_nv: int):
    place = _decode_place(c, group_ns, queries)
    shift = c.i32()
    if abs(shift) >= (1 << place[2]):
        raise Reject("E_NONCANONICAL")
    vlen = c.u8()
    if vlen != place[2]:
        raise Reject("E_NONCANONICAL")
    c.check_count(vlen, 2)
    varmap = []
    for _ in range(vlen):
        tag, val = c.u8(), c.u8()
        if tag == 0:
            if val >= cube_nv:
                raise Reject("E_NONCANONICAL")
            varmap.append(("var", val))
        elif tag == 1 and val <= 1:
            varmap.append(("const", val))
        else:
            raise Reject("E_NONCANONICAL")
    flen = c.u8()
    c.check_count(flen, 2)
    fixed = []
    for _ in range(flen):
        v, b = c.u8(), c.u8()
        if v >= cube_nv or b > 1:
            raise Reject("E_NONCANONICAL")
        fixed.append((v, b))
    return (place, shift, tuple(varmap), tuple(fixed))


def _decode_ast(c: _Cur, cube_nv: int, n_views: int, allow_eqzc: bool, depth: int = 0):
    if depth > 64:
        raise Reject("E_LENGTH")
    tag = c.u8()
    if tag == 0:
        return ("const", c.fp3())
    if tag == 1:
        idx = c.u32()
        if idx >= n_views:
            raise Reject("E_NONCANONICAL")
        return ("view", idx)
    if tag == 2:
        if not allow_eqzc:
            raise Reject("E_NONCANONICAL")
        n = c.check_count(c.u8(), 1)
        vs = tuple(c.u8() for _ in range(n))
        if any(v >= cube_nv for v in vs):
            raise Reject("E_NONCANONICAL")
        return ("eqzc", vs)
    if tag == 3:
        n = c.u8()
        if n > 63:
            raise Reject("E_NONCANONICAL")
        c.check_count(n, 1)
        vs = tuple(c.u8() for _ in range(n))
        if any(v >= cube_nv for v in vs):
            raise Reject("E_NONCANONICAL")
        lim = c.u64()
        fn = c.check_count(c.u8(), 2)
        fx = []
        for _ in range(fn):
            v, b = c.u8(), c.u8()
            if v >= cube_nv or b > 1:
                raise Reject("E_NONCANONICAL")
            fx.append((v, b))
        return ("sellt", vs, lim, tuple(fx))
    if tag == 4:
        n = c.u8()
        if n > 63:
            raise Reject("E_NONCANONICAL")
        c.check_count(n, 1)
        vs = tuple(c.u8() for _ in range(n))
        if any(v >= cube_nv for v in vs):
            raise Reject("E_NONCANONICAL")
        sc = c.fp3()
        fn = c.check_count(c.u8(), 2)
        fx = []
        for _ in range(fn):
            v, b = c.u8(), c.u8()
            if v >= cube_nv or b > 1:
                raise Reject("E_NONCANONICAL")
            fx.append((v, b))
        return ("indexp", vs, sc, tuple(fx))
    if tag == 5:
        return ("add", _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1),
                _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1))
    if tag == 6:
        return ("sub", _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1),
                _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1))
    if tag == 7:
        return ("mul", _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1),
                _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1))
    if tag == 8:
        s = c.fp3()
        return ("scale", s, _decode_ast(c, cube_nv, n_views, allow_eqzc, depth + 1))
    raise Reject("E_NONCANONICAL")


def decode_circuit_desc(raw: bytes) -> dict:
    c = _Cur(raw)
    if c.take(8) != DESC_MAGIC:
        raise Reject("E_MAGIC")
    if c.u16() != DESC_VERSION:
        raise Reject("E_VERSION")
    n_max = c.u8()
    queries = c.u16()
    n_groups = c.check_count(c.u16(), 1)
    group_ns = [c.u8() for _ in range(n_groups)]
    if not (8 <= n_max <= 28 and n_max % 2 == 0 and 1 <= queries <= 4096 and 1 <= n_groups <= 128):
        raise Reject("E_LENGTH")
    tl = _tail_len_for(queries)
    for ng in group_ns:
        if not (8 <= ng <= n_max and ng % 2 == 0 and tl < (1 << ng)):
            raise Reject("E_LENGTH")

    n_gkr = c.check_count(c.u16(), 18)
    gkr_batches = []
    for _ in range(n_gkr):
        s = c.u8()
        m_mask = c.u16()
        if not (3 <= s <= 28 and 2 <= m_mask <= (1 << (s - 2))):
            raise Reject("E_LENGTH")
        aux_place = _decode_place(c, group_ns, queries)
        if (1 << aux_place[2]) < 2 * (s - 1) * s + 2 * m_mask:
            raise Reject("E_NONCANONICAL")
        n_views = c.check_count(c.u32(), 13)
        view_refs = [_decode_view_ref(c, group_ns, queries, s) for _ in range(n_views)]
        n_blocks = c.check_count(c.u32(), 15)
        blocks = []
        intervals = []
        half = 1 << (s - 1)
        lim_l = half - m_mask
        lim_r = (1 << s) - m_mask
        for _ in range(n_blocks):
            offset = c.u32()
            nbits = c.u8()
            if nbits >= s:
                raise Reject("E_NONCANONICAL")
            sz = 1 << nbits
            end = offset + sz
            if offset % sz != 0 or not (end <= lim_l or (offset >= half and end <= lim_r)):
                raise Reject("E_NONCANONICAL")
            intervals.append((offset, end))
            tid = c.u64()
            n_cols = c.check_count(c.u8(), 1)
            col_exprs = [_decode_ast(c, nbits, n_views, False) for _ in range(n_cols)]
            nk = c.u8()
            if nk == 0:
                num_expr = None
            elif nk == 1:
                num_expr = _decode_ast(c, nbits, n_views, False)
            else:
                raise Reject("E_NONCANONICAL")
            blocks.append((offset, nbits, tid, col_exprs, num_expr))
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i - 1][1] > intervals[i][0]:
                raise Reject("E_NONCANONICAL")
        gkr_batches.append((s, m_mask, aux_place, view_refs, blocks))

    n_sc = c.check_count(c.u16(), 17)
    sumchecks = []
    for _ in range(n_sc):
        nv = c.u8()
        deg = c.u8()
        if not (1 <= nv <= 28 and 1 <= deg <= 16):
            raise Reject("E_NONCANONICAL")
        n_c = c.check_count(c.u32(), 1)
        mask_place = _decode_place(c, group_ns, queries)
        if (1 << mask_place[2]) < nv * (deg + 1):
            raise Reject("E_NONCANONICAL")
        n_views = c.check_count(c.u32(), 13)
        view_refs = [_decode_view_ref(c, group_ns, queries, nv) for _ in range(n_views)]
        constraints = [_decode_ast(c, nv, n_views, True) for _ in range(n_c)]
        sumchecks.append((nv, deg, n_c, mask_place, view_refs, constraints))

    if c.off != len(raw):
        raise Reject("E_LENGTH")
    return {
        "n_max": n_max, "queries": queries, "group_ns": group_ns,
        "gkr_batches": gkr_batches, "sumchecks": sumchecks,
    }


def _ind_fixed(fixed, r):
    acc = ONE3
    for v, b in fixed:
        acc = e_mul(acc, r[v] if b else e_sub(ONE3, r[v]))
    return acc


def _mle_lt(z, n):
    bits = len(z)
    if bits == 0:
        return ONE3 if n > 0 else ZERO3
    if n == 0:
        return ZERO3
    if n >= (1 << bits):
        return ONE3
    acc, pref = ZERO3, ONE3
    for i in range(bits - 1, -1, -1):
        zi = z[i]
        if (n >> i) & 1:
            acc = e_add(acc, e_mul(pref, e_sub(ONE3, zi)))
            pref = e_mul(pref, zi)
        else:
            pref = e_mul(pref, e_sub(ONE3, zi))
    return acc


def _mle_shift(r, alpha, delta):
    n = len(r)
    if delta >= (1 << n):
        return ZERO3
    f = [ONE3, ZERO3]
    for i in range(n):
        d = (delta >> i) & 1
        g = [ZERO3, ZERO3]
        for c in (0, 1):
            if f[c] == ZERO3:
                continue
            for xb in (0, 1):
                s = xb + d + c
                yb, c2 = s & 1, s >> 1
                rx = r[i] if xb else e_sub(ONE3, r[i])
                ay = alpha[i] if yb else e_sub(ONE3, alpha[i])
                g[c2] = e_add(g[c2], e_mul(f[c], e_mul(rx, ay)))
        f = g
    return f[0]


def _mle_shift_signed(r, alpha, delta):
    return _mle_shift(r, alpha, delta) if delta >= 0 else _mle_shift(alpha, r, -delta)


def _eval_ast(node, r, r_zc, view_vals):
    tag = node[0]
    if tag == "const":
        return node[1]
    if tag == "view":
        return view_vals[node[1]]
    if tag == "eqzc":
        acc = ONE3
        for v in node[1]:
            acc = e_mul(acc, _eq1(r_zc[v], r[v]))
        return acc
    if tag == "sellt":
        _, vs, lim, fx = node
        return e_mul(_ind_fixed(fx, r), _mle_lt([r[v] for v in vs], lim))
    if tag == "indexp":
        _, vs, sc, fx = node
        acc = ZERO3
        for k, v in enumerate(vs):
            acc = e_add(acc, e_mul(((1 << k) % P, 0, 0), r[v]))
        return e_mul(_ind_fixed(fx, r), e_mul(sc, acc))
    if tag == "add":
        return e_add(_eval_ast(node[1], r, r_zc, view_vals), _eval_ast(node[2], r, r_zc, view_vals))
    if tag == "sub":
        return e_sub(_eval_ast(node[1], r, r_zc, view_vals), _eval_ast(node[2], r, r_zc, view_vals))
    if tag == "mul":
        return e_mul(_eval_ast(node[1], r, r_zc, view_vals), _eval_ast(node[2], r, r_zc, view_vals))
    if tag == "scale":
        return e_mul(node[1], _eval_ast(node[2], r, r_zc, view_vals))
    raise Reject("E_NONCANONICAL")


def _block_ind_at(offset: int, nbits: int, r_full):
    acc = ONE3
    for b in range(nbits, len(r_full)):
        rb = r_full[b]
        acc = e_mul(acc, rb if (offset >> b) & 1 else e_sub(ONE3, rb))
    return acc


def _gkr_proof_bytes(s: int) -> int:
    return 48 + 96 + sum(24 + 96 * k + 96 + 24 for k in range(1, s))


def _pcs_proof_bytes(n_max: int, queries: int, group_ns: List[int]) -> int:
    g = len(group_ns)
    folds = n_max // 2
    step_a = 24 + n_max * 72 + (g + 2) * 24
    step_b_hdr = n_max * 72 + (folds - 1) * 32 + 24
    main_q = (96 + (n_max + 1) * 32) + sum(96 + (ng + 1) * 32 for ng in group_ns) + sum(
        96 + (n_max + 1 - 2 * t) * 32 for t in range(1, folds)
    )
    nl = _libra_group_n(n_max, queries)
    fl = nl // 2
    lib_b = (nl * 72 + (fl - 1) * 32 + 24) + queries * sum(96 + (nl + 1 - 2 * t) * 32 for t in range(fl))
    return 64 + step_a + lib_b + step_b_hdr + queries * main_q


def body_used(desc: dict) -> int:
    n = 64 + 32 * len(desc["group_ns"])
    for s, _m, _aux, vrefs, _blks in desc["gkr_batches"]:
        n += _gkr_proof_bytes(s) + (len(vrefs) + 2) * 24
    for nv, deg, _nc, _mp, vrefs, _cs in desc["sumchecks"]:
        n += 24 + nv * (deg + 1) * 24 + (len(vrefs) + 1) * 24
    n += _pcs_proof_bytes(desc["n_max"], desc["queries"], desc["group_ns"])
    return n


def _verify_gkr_batch(s: int, proof: bytes, tr: Fpn2Transcript):
    c = _Cur(proof)
    r_start = c.off
    rn, rd = c.fp3(), c.fp3()
    if rn != ZERO3 or rd == ZERO3:
        raise Reject("E_GKR_ROOT")
    tr.absorb_bytes(proof[r_start:c.off])

    c0_start = c.off
    nl0, dl0, nr0, dr0 = c.fp3(), c.fp3(), c.fp3(), c.fp3()
    if _combine_frac(nl0, dl0, nr0, dr0) != (rn, rd):
        raise Reject("E_GKR")
    tr.absorb_bytes(proof[c0_start:c.off])

    br0 = tr.squeeze_fp3()
    pt = [br0]
    cn, cd = _mix(nl0, nr0, br0), _mix(dl0, dr0, br0)
    layer_pts, layer_g = [], []
    for k in range(1, s):
        alpha = tr.squeeze_fp3()
        gs_s = c.off
        g_sum = c.fp3()
        tr.absorb_bytes(proof[gs_s:c.off])
        lam = tr.squeeze_fp3()
        target = e_add(e_add(cn, e_mul(alpha, cd)), e_mul(lam, g_sum))
        rk = []
        for _ in range(k):
            rs = c.off
            vs = [c.fp3() for _ in range(4)]
            if e_add(vs[0], vs[1]) != target:
                raise Reject("E_GKR")
            tr.absorb_bytes(proof[rs:c.off])
            ri = tr.squeeze_fp3()
            rk.append(ri)
            target = lagrange_eval(vs, ri)
        fs = c.off
        nl, dl, nr, dr, gr = c.fp3(), c.fp3(), c.fp3(), c.fp3(), c.fp3()
        tr.absorb_bytes(proof[fs:c.off])
        cnum, cden = _combine_frac(nl, dl, nr, dr)
        exp = e_add(e_mul(_eq(pt, rk), e_add(cnum, e_mul(alpha, cden))), e_mul(lam, gr))
        if exp != target:
            raise Reject("E_GKR")
        layer_pts.append(rk)
        layer_g.append(gr)
        brk = tr.squeeze_fp3()
        pt = [brk] + rk
        cn, cd = _mix(nl, nr, brk), _mix(dl, dr, brk)
    return pt, cn, cd, layer_pts, layer_g


def _merkle_raw_ok(root, raw_leaf, idx, sibs):
    return _merkle_ok(root, _leaf(raw_leaf), idx, sibs)


def _read_coset_path(c: _Cur, depth: int):
    raw_leaf = c.take(96)
    try:
        vals = [e_decode(raw_leaf[24 * k:24 * k + 24]) for k in range(4)]
    except NonCanonical:
        raise Reject("E_NONCANONICAL") from None
    sibs = [c.take(32) for _ in range(depth)]
    return vals, raw_leaf, sibs


def _libra_w_mle(r_ip, r_lib):
    n_lib = len(r_lib)
    acc = ZERO3
    k = 0
    for r in r_ip:
        for w in (ONE3, r, e_mul(r, r)):
            ev = ONE3
            for b in range(n_lib):
                rb = r_lib[b]
                ev = e_mul(ev, rb if (k >> b) & 1 else e_sub(ONE3, rb))
            acc = e_add(acc, e_mul(w, ev))
            k += 1
    return acc


def _verify_single_open(n: int, root: bytes, claim, w_fn, queries: int, c: _Cur, tr: Fpn2Transcript):
    folds = n // 2
    code_len = RATE << n
    target = claim
    r_bf, fold_roots = [], []
    for i in range(n):
        rs = c.off
        ps = [c.fp3() for _ in range(3)]
        if e_add(ps[0], ps[1]) != target:
            raise Reject("E_SUMCHECK")
        tr.absorb_bytes(c.b[rs:c.off])
        ri = tr.squeeze_fp3()
        r_bf.append(ri)
        target = lagrange_eval(ps, ri)
        if i % 2 == 1 and (i // 2) + 1 < folds:
            rt = c.take(32)
            tr.absorb_bytes(rt)
            fold_roots.append(rt)
    fs = c.off
    fin = c.fp3()
    tr.absorb_bytes(c.b[fs:c.off])
    if e_mul(fin, w_fn(r_bf)) != target:
        raise Reject("E_FINAL")

    q_mask = (code_len // 4) - 1
    positions = [int.from_bytes(tr.squeeze_bytes(8), "little") & q_mask for _ in range(queries)]
    omega = pow(7, (P - 1) // code_len, P)
    inv_iota = pow(pow(7, (P - 1) // 4, P), P - 2, P)
    for p0 in positions:
        cur_p = p0
        prev = None
        for t in range(folds):
            dt = n + 1 - 2 * t
            coset, raw_leaf, sibs = _read_coset_path(c, dt)
            if t == 0:
                if not _merkle_raw_ok(root, raw_leaf, cur_p, sibs):
                    raise Reject("E_MERKLE")
            else:
                q_t = (code_len >> (2 * t)) // 4
                slot = cur_p // q_t
                cur_p %= q_t
                if not _merkle_raw_ok(fold_roots[t - 1], raw_leaf, cur_p, sibs):
                    raise Reject("E_MERKLE")
                if coset[slot] != prev:
                    raise Reject("E_FOLD")
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(pow(omega_t, cur_p, P), P - 2, P)
            prev = _fold4(coset, x_inv, inv_iota, r_bf[2 * t], r_bf[2 * t + 1])
        if prev != fin:
            raise Reject("E_FOLD")


def verify_relation2(
    raw: bytes,
    raw_desc: bytes,
    *,
    allow_reduced: bool = False,
    expected_profile_id: Optional[bytes] = None,
    expected_circuit_root: Optional[bytes] = None,
    expected_x_root: Optional[bytes] = None,
    expected_stmt_digest: Optional[bytes] = None,
) -> str:
    raw = bytes(raw)
    hdr = parse_header(raw)
    prof = lookup(hdr.profile_id)
    if prof is None:
        raise Reject("E_PROFILE_ID")
    if expected_profile_id is not None and bytes(expected_profile_id) != hdr.profile_id:
        raise Reject("E_PROFILE_PINNED")
    if prof.relation_id != REL_LLM_INFERENCE_V2:
        raise Reject("E_PROFILE_ID")
    if prof.security_tier == TIER_REDUCED and not allow_reduced:
        raise Reject("E_REDUCED_SECURITY")
    if len(raw) != prof.c_p:
        raise Reject("E_LENGTH")
    if expected_x_root is not None:
        raise Reject("E_X_ROOT_PINNED")

    # Pin-before-decode: check circuit_root and stmt_digest BEFORE decoding raw_desc!
    if expected_circuit_root is None:
        raise Reject("E_CIRCUIT_UNPINNED")
    d_root = circuit_desc_root(raw_desc)
    if bytes(expected_circuit_root) != d_root:
        raise Reject("E_CIRCUIT_PINNED")
    if expected_stmt_digest is None:
        raise Reject("E_STMT_UNPINNED")
    if len(raw) < HEADER_LEN + 64:
        raise Reject("E_LENGTH")
    if raw[HEADER_LEN:HEADER_LEN + 32] != d_root:
        raise Reject("E_CIRCUIT_PINNED")
    if raw[HEADER_LEN + 32:HEADER_LEN + 64] != bytes(expected_stmt_digest):
        raise Reject("E_STMT_PINNED")

    desc = decode_circuit_desc(raw_desc)
    if prof.security_tier == TIER_STANDARD and desc["queries"] < MIN_STANDARD_QUERIES:
        raise Reject("E_REDUCED_SECURITY")

    used = body_used(desc)
    body_end = HEADER_LEN + used
    if body_end > len(raw):
        raise Reject("E_LENGTH")
    if any(raw[body_end:]):
        raise Reject("E_PADDING")

    body = raw[HEADER_LEN:body_end]
    c = _Cur(body)
    c_root = c.take(32)
    s_dig = c.take(32)
    g_cnt = len(desc["group_ns"])
    group_roots = [c.take(32) for _ in range(g_cnt)]

    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(raw[:HEADER_LEN])
    tr.absorb_bytes(c_root)
    tr.absorb_bytes(s_dig)
    for r in group_roots:
        tr.absorb_bytes(r)

    claims = [[] for _ in range(g_cnt)]

    # 1. GKR batches
    if desc["gkr_batches"]:
        gamma_lu = tr.squeeze_fp3()
        beta_lu = tr.squeeze_fp3()
        for s, m_mask, aux_place, vrefs, blks in desc["gkr_batches"]:
            gpf = c.take(_gkr_proof_bytes(s))
            r_gkr, num_eval, den_eval, lpts, lgs = _verify_gkr_batch(s, gpf, tr)

            gc_start = c.off
            view_vals = []
            for place, shift, varmap, fixed in vrefs:
                val = c.fp3()
                loc = [r_gkr[v] if k == "var" else (v, 0, 0) for k, v in varmap]
                spec = ("sh", loc, shift) if shift != 0 else ("pt", loc)
                claims[place[0]].append((place, spec, val))
                view_vals.append(e_mul(val, _ind_fixed(fixed, r_gkr)))

            acc_num, acc_den, sum_blk_ind = ZERO3, ZERO3, ZERO3
            for offset, nbits, tid, col_exprs, num_expr in blks:
                ind_b = _block_ind_at(offset, nbits, r_gkr)
                sum_blk_ind = e_add(sum_blk_ind, ind_b)
                r_loc = r_gkr[:nbits]
                fp_ev = (tid % P, 0, 0)
                bp = beta_lu
                for ce in col_exprs:
                    fp_ev = e_add(fp_ev, e_mul(bp, _eval_ast(ce, r_loc, [], view_vals)))
                    bp = e_mul(bp, beta_lu)
                acc_den = e_add(acc_den, e_mul(ind_b, e_sub(gamma_lu, fp_ev)))
                n_blk = _eval_ast(num_expr, r_loc, [], view_vals) if num_expr is not None else ONE3
                acc_num = e_add(acc_num, e_mul(ind_b, n_blk))

            val_a, val_d = c.fp3(), c.fp3()
            gc_raw = body[gc_start:c.off]

            half = 1 << (s - 1)
            n_leaves = 1 << s
            n_l = 2 * (s - 1) * s
            last = m_mask - 1
            a_ent, d_ent = [], []
            sum_mask_ind = ZERO3
            for m in range(m_mask):
                il, ir = half - m_mask + m, n_leaves - m_mask + m
                el, er = ONE3, ONE3
                for bit, rb in enumerate(r_gkr):
                    el = e_mul(el, rb if (il >> bit) & 1 else e_sub(ONE3, rb))
                    er = e_mul(er, rb if (ir >> bit) & 1 else e_sub(ONE3, rb))
                if m < last:
                    a_ent.append((n_l + m, e_sub(el, er)))
                    d_ent.append((n_l + m_mask + m, e_add(el, er)))
                else:
                    d_ent.append((n_l + m_mask + m, el))
                    d_ent.append((n_l + m, er))
                sum_mask_ind = e_add(sum_mask_ind, e_add(el, er))
            claims[aux_place[0]].append((aux_place, ("lin", a_ent), val_a))
            claims[aux_place[0]].append((aux_place, ("lin", d_ent), val_d))

            unused_ind = e_sub(ONE3, e_add(sum_blk_ind, sum_mask_ind))
            if e_add(acc_num, val_a) != num_eval or e_add(e_add(acc_den, unused_ind), val_d) != den_eval:
                raise Reject("E_GKR")

            for idx_k, (rk, gr) in enumerate(zip(lpts, lgs)):
                k = idx_k + 1
                off_l = 2 * (k - 1) * k
                ent_k = []
                for i, ri in enumerate(rk):
                    r2 = e_mul(ri, ri)
                    r3 = e_mul(r2, ri)
                    ent_k.extend([(off_l + 4 * i, ONE3), (off_l + 4 * i + 1, ri), (off_l + 4 * i + 2, r2), (off_l + 4 * i + 3, r3)])
                claims[aux_place[0]].append((aux_place, ("lin", ent_k), gr))
            tr.absorb_bytes(gc_raw)

    # 2. Sumchecks
    for nv, deg, n_c, mask_place, vrefs, constraints in desc["sumchecks"]:
        r_zc = [tr.squeeze_fp3() for _ in range(nv)]
        lambdas = [tr.squeeze_fp3() for _ in range(n_c)]
        gs = c.off
        g_sum = c.fp3()
        tr.absorb_bytes(struct.pack("<I", nv) + body[gs:c.off])
        rho = tr.squeeze_fp3()
        cl = e_mul(rho, g_sum)
        r = []
        for _ in range(nv):
            rs = c.off
            vs = [c.fp3() for _ in range(deg + 1)]
            if e_add(vs[0], vs[1]) != cl:
                raise Reject("E_SUMCHECK")
            tr.absorb_bytes(body[rs:c.off])
            a = tr.squeeze_fp3()
            r.append(a)
            cl = lagrange_eval(vs, a)
        cs = c.off
        view_vals = []
        for place, shift, varmap, fixed in vrefs:
            val = c.fp3()
            loc = [r[v] if k == "var" else (v, 0, 0) for k, v in varmap]
            spec = ("sh", loc, shift) if shift != 0 else ("pt", loc)
            claims[place[0]].append((place, spec, val))
            view_vals.append(e_mul(val, _ind_fixed(fixed, r)))
        gval = c.fp3()
        claims[mask_place[0]].append((mask_place, ("pw", r, deg), gval))
        tr.absorb_bytes(body[cs:c.off])
        lhs = e_mul(rho, gval)
        for ast, lam in zip(constraints, lambdas):
            lhs = e_add(lhs, e_mul(lam, _eval_ast(ast, r, r_zc, view_vals)))
        if lhs != cl:
            raise Reject("E_SUMCHECK")

    # 3. Batched multi-size multi_pcs
    group_mus, group_claims = [], []
    for gi in range(g_cnt):
        gcl = claims[gi]
        tr.absorb_bytes(struct.pack("<I", len(gcl)))
        mus = [tr.squeeze_fp3() for _ in range(len(gcl))]
        yg = ZERO3
        for (_pl, _sp, val), mu in zip(gcl, mus):
            yg = e_add(yg, e_mul(mu, val))
        group_mus.append(mus)
        group_claims.append(yg)

    n_max, queries, group_ns = desc["n_max"], desc["queries"], desc["group_ns"]
    root_b = c.take(32)
    root_l = c.take(32)
    g_sum = c.fp3()
    for r in group_roots:
        tr.absorb_bytes(r)
    tr.absorb_bytes(root_b)
    tr.absorb_bytes(root_l)
    tr.absorb_bytes(b"".join(_e_encode(x) for x in group_claims))
    tr.absorb_bytes(_e_encode(g_sum))

    mu, lam = tr.squeeze_fp3(), tr.squeeze_fp3()
    mu_pows = [ONE3]
    for _ in range(1, g_cnt):
        mu_pows.append(e_mul(mu_pows[-1], mu))
    target = e_mul(lam, g_sum)
    for g in range(g_cnt):
        target = e_add(target, e_mul(mu_pows[g], group_claims[g]))

    r_ip = []
    for _ in range(n_max):
        rs = c.off
        ps = [c.fp3() for _ in range(3)]
        if e_add(ps[0], ps[1]) != target:
            raise Reject("E_SUMCHECK")
        tr.absorb_bytes(body[rs:c.off])
        ri = tr.squeeze_fp3()
        r_ip.append(ri)
        target = lagrange_eval(ps, ri)

    fs = c.off
    v_groups = [c.fp3() for _ in range(g_cnt)]
    v_b = c.fp3()
    e_lib = c.fp3()
    tr.absorb_bytes(body[fs:c.off])

    def _w_eval(gi, r_ip_g):
        acc = ZERO3
        for (pl, spec, _val), mu_i in zip(claims[gi], group_mus[gi]):
            _, off_v, nb_v = pl
            ind = _block_ind_at(off_v, nb_v, r_ip_g)
            r_sub = r_ip_g[:nb_v]
            k = spec[0]
            if k == "pt":
                loc = _eq(spec[1], r_sub)
            elif k == "sh":
                loc = _mle_shift_signed(spec[1], r_sub, spec[2])
            elif k == "pw":
                loc = ZERO3
                for i, ri in enumerate(spec[1]):
                    pw = ONE3
                    for j in range(spec[2] + 1):
                        idx = i * (spec[2] + 1) + j
                        ev = ONE3
                        for b in range(nb_v):
                            rb = r_sub[b]
                            ev = e_mul(ev, rb if (idx >> b) & 1 else e_sub(ONE3, rb))
                        loc = e_add(loc, e_mul(pw, ev))
                        pw = e_mul(pw, ri)
            else:
                loc = ZERO3
                for idx, coeff in spec[1]:
                    ev = ONE3
                    for b in range(nb_v):
                        rb = r_sub[b]
                        ev = e_mul(ev, rb if (idx >> b) & 1 else e_sub(ONE3, rb))
                    loc = e_add(loc, e_mul(coeff, ev))
            acc = e_add(acc, e_mul(mu_i, e_mul(ind, loc)))
        return acc

    exp_a = e_mul(lam, e_lib)
    for g in range(g_cnt):
        sg = n_max - group_ns[g]
        zlo = ONE3
        for rj in r_ip[:sg]:
            zlo = e_mul(zlo, e_sub(ONE3, rj))
        exp_a = e_add(exp_a, e_mul(e_mul(mu_pows[g], zlo), e_mul(v_groups[g], _w_eval(g, r_ip[sg:]))))
    if exp_a != target:
        raise Reject("E_SUMCHECK")

    n_lib = _libra_group_n(n_max, queries)
    _verify_single_open(n_lib, root_l, e_lib, lambda r_lib: _libra_w_mle(r_ip, r_lib), queries, c, tr)

    theta = tr.squeeze_fp3()
    theta_pows = []
    tp = theta
    for _ in range(g_cnt):
        theta_pows.append(tp)
        tp = e_mul(tp, theta)

    folds = n_max // 2
    by_layer = [[] for _ in range(folds)]
    for g in range(g_cnt):
        by_layer[(n_max - group_ns[g]) // 2].append(g)

    target_b = v_b
    r_bf, fold_roots = [], []
    c_lo = ONE3
    for i in range(n_max):
        if i % 2 == 0:
            for g in by_layer[i // 2]:
                target_b = e_add(target_b, e_mul(theta_pows[g], e_mul(c_lo, v_groups[g])))
        rs = c.off
        ps = [c.fp3() for _ in range(3)]
        if e_add(ps[0], ps[1]) != target_b:
            raise Reject("E_SUMCHECK")
        tr.absorb_bytes(body[rs:c.off])
        ri = tr.squeeze_fp3()
        r_bf.append(ri)
        c_lo = e_mul(c_lo, _eq1(r_ip[i], ri))
        target_b = lagrange_eval(ps, ri)
        if i % 2 == 1 and (i // 2) + 1 < folds:
            rt = c.take(32)
            tr.absorb_bytes(rt)
            fold_roots.append(rt)

    fs = c.off
    fin = c.fp3()
    tr.absorb_bytes(body[fs:c.off])
    if e_mul(fin, c_lo) != target_b:
        raise Reject("E_FINAL")

    code_len = RATE << n_max
    q_mask = (code_len // 4) - 1
    positions = [int.from_bytes(tr.squeeze_bytes(8), "little") & q_mask for _ in range(queries)]
    omega = pow(7, (P - 1) // code_len, P)
    inv_iota = pow(pow(7, (P - 1) // 4, P), P - 2, P)

    for p0 in positions:
        cur_p = p0
        prev = None
        for t in range(folds):
            dt = n_max + 1 - 2 * t
            base_c, base_raw, base_sibs = _read_coset_path(c, dt)
            if t == 0:
                if not _merkle_raw_ok(root_b, base_raw, cur_p, base_sibs):
                    raise Reject("E_MERKLE")
            else:
                q_t = (code_len >> (2 * t)) // 4
                slot = cur_p // q_t
                cur_p %= q_t
                if not _merkle_raw_ok(fold_roots[t - 1], base_raw, cur_p, base_sibs):
                    raise Reject("E_MERKLE")
                if base_c[slot] != prev:
                    raise Reject("E_FOLD")
            comb = list(base_c)
            for g in by_layer[t]:
                gc, graw, gsibs = _read_coset_path(c, dt)
                if not _merkle_raw_ok(group_roots[g], graw, cur_p, gsibs):
                    raise Reject("E_MERKLE")
                tg = theta_pows[g]
                for k in range(4):
                    comb[k] = e_add(comb[k], e_mul(tg, gc[k]))
            omega_t = pow(omega, 4 ** t, P)
            x_inv = pow(pow(omega_t, cur_p, P), P - 2, P)
            prev = _fold4(comb, x_inv, inv_iota, r_bf[2 * t], r_bf[2 * t + 1])
        if prev != fin:
            raise Reject("E_FOLD")

    assert c.off == len(body)
    return "VERIFIED"
