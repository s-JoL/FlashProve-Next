# SPDX-License-Identifier: Apache-2.0
"""`FPN2` Relation 2 (`LLM_INFERENCE_V2`) canonical `CircuitDesc` compiler and fixed-size container prover.

Bit-exact counterpart of `core/src/relation2.rs` (`fpn-verify --circuit-desc desc.bin --circuit-root HEX64`):
  * Compiles a `zk_flat.Circuit` (Phase-1 vectors + lookups + zerochecks/sumzeros, no ad-hoc JSON rounds)
    into a canonical binary `CircuitDesc` whose 32-byte `circuit_desc_root = B256("FPN2-CDESC-v2", desc)`
    is pinned by the verifier;
  * Proves the circuit using `multi_gkr` (0 Phase-2 commitments!) + Libra Sumchecks + `multi_pcs`
    (single-chain multi-size rate-1/8 coset-4 folded ZK-Basefold);
  * Packs the exact fixed-size zero-padded `FPN2` container (`fpn2-llm-std-v1`, `C_p = 8,388,608` B = 8 MiB)
    with ZERO inner length headers, ZERO compression, and ZERO JSON in the transcript.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from flashprove_next import zk_flat as Z
from flashprove_next.field import CPU, ONE, P, ZERO, hadd, hf, hmul, hsub

from . import folded_engine as FE
from . import multi_gkr as MG
from . import multi_pcs as MP
from .fpn2 import Fpn2Transcript, LABEL_V2, enc3, enc3s, header, profile_id

DESC_MAGIC = b"FPN2DESC"
DESC_VERSION = 1
PERSONAL_CDESC = b"FPN2-CDESC-v2\x00\x00\x00"
PERSONAL_STMT = b"FPN2-STMT-v2\x00\x00\x00\x00"

# Canonical `fpn2-llm-std-v1` profile (bit-exact with `core/src/profile.rs::LLM_STD_V1`)
LLM_STD_V1 = {
    "name": "fpn2-llm-std-v1",
    "security_tier": 1,
    "relation_id": 2,
    "security_bits": 100,
    "max_layers": 128,
    "max_hidden": 16384,
    "max_seq_len": 131072,
    "max_vocab": 262144,
    "max_new_tokens": 32768,
    "stop_ids": (151643, 151645),
    "n_vars": 0,
    "sc_degree": 0,
    "n_queries": 0,
    "c_p": 8 * 1024 * 1024,
}


def llm_profile_bytes(p: dict) -> bytes:
    """Exact 128-byte canonical profile encoding matching `core/src/profile.rs::Profile::canonical_bytes`."""
    b = bytearray(128)
    b[0:8] = b"FPN2PROF"
    b[8:10] = struct.pack("<H", 1)
    nm = p["name"].encode("ascii")
    b[10:10 + len(nm)] = nm
    b[42] = p["security_tier"]
    b[43] = p["relation_id"]
    b[44:46] = struct.pack("<H", p["security_bits"])
    b[46:66] = struct.pack(
        "<5I", p["max_layers"], p["max_hidden"], p["max_seq_len"], p["max_vocab"], p["max_new_tokens"]
    )
    sids = p["stop_ids"]
    b[66] = len(sids)
    for k, sid in enumerate(sids):
        b[67 + 4 * k:71 + 4 * k] = struct.pack("<I", sid)
    b[99] = p["n_vars"]
    b[100] = p["sc_degree"]
    b[101:103] = struct.pack("<H", p["n_queries"])
    b[103:111] = struct.pack("<Q", p["c_p"])
    return bytes(b)


def llm_profile_id(p: dict) -> bytes:
    return hashlib.blake2b(llm_profile_bytes(p), digest_size=32).digest()
def circuit_desc_root(raw_desc: bytes) -> bytes:
    return hashlib.blake2b(raw_desc, digest_size=32, person=PERSONAL_CDESC).digest()


def stmt_digest(stmt_bytes: bytes) -> bytes:
    return hashlib.blake2b(stmt_bytes, digest_size=32, person=PERSONAL_STMT).digest()


# ----------------------------------------------------------------------------- Binary encoders (exact match to relation2.rs)
def _enc_place(pl: FE.VecPlacement) -> bytes:
    return struct.pack("<HIB", pl.group_idx, pl.offset, pl.nbits)


def _enc_view_ref(atom: Z.View, pl: FE.VecPlacement) -> bytes:
    out = bytearray(_enc_place(pl))
    out += struct.pack("<i", int(atom.shift))
    out += struct.pack("<B", len(atom.varmap))
    for v in atom.varmap:
        if isinstance(v, tuple):
            out += struct.pack("<BB", 1, int(v[1]))
        else:
            out += struct.pack("<BB", 0, int(v))
    out += struct.pack("<B", len(atom.fixed))
    for var, bit in atom.fixed:
        out += struct.pack("<BB", int(var), int(bit))
    return bytes(out)


def _enc_atom_ast(atom, view_index_map: dict) -> bytes:
    if isinstance(atom, Z.View):
        if atom.public:
            raise ValueError("PubVec views must be lowered to structured atoms or constants for Relation 2")
        return struct.pack("<BI", 1, view_index_map[atom.kid])
    if isinstance(atom, Z.EqZC):
        vs = list(atom.varmap)
        return struct.pack("<BB", 2, len(vs)) + bytes(int(v) for v in vs)
    if isinstance(atom, Z.SelLt):
        vs = list(atom.varmap)
        out = bytearray(struct.pack("<BB", 3, len(vs)))
        out += bytes(int(v) for v in vs)
        out += struct.pack("<QB", int(atom.n), len(atom.fixed))
        for var, bit in atom.fixed:
            out += struct.pack("<BB", int(var), int(bit))
        return bytes(out)
    if isinstance(atom, Z.IndexP):
        vs = list(atom.varmap)
        out = bytearray(struct.pack("<BB", 4, len(vs)))
        out += bytes(int(v) for v in vs)
        out += enc3(hf(int(atom.scale) % P))
        out += struct.pack("<B", len(atom.fixed))
        for var, bit in atom.fixed:
            out += struct.pack("<BB", int(var), int(bit))
        return bytes(out)
    raise ValueError(f"unsupported public atom in Relation 2 AST: {type(atom).__name__}")


def _enc_lin_ast(lin: Z.Lin, view_index_map: dict) -> bytes:
    parts = []
    if lin.const != ZERO:
        parts.append(struct.pack("<B", 0) + enc3(lin.const))
    for a, co in sorted(lin.items.values(), key=lambda t: repr(t[0].key)):
        node = _enc_atom_ast(a, view_index_map)
        if co != ONE:
            node = struct.pack("<B", 8) + enc3(co) + node
        parts.append(node)
    if not parts:
        return struct.pack("<B", 0) + enc3(ZERO)
    acc = parts[0]
    for nxt in parts[1:]:
        acc = struct.pack("<B", 5) + acc + nxt
    return acc


def _enc_ex_ast(ex: Z.Ex, view_index_map: dict) -> bytes:
    term_nodes = []
    for c, f in ex.terms:
        if not f:
            term_nodes.append(struct.pack("<B", 0) + enc3(c))
            continue
        factors = []
        for l, p in f:
            ln = _enc_lin_ast(l, view_index_map)
            for _ in range(p):
                factors.append(ln)
        prod = factors[0]
        for nxt in factors[1:]:
            prod = struct.pack("<B", 7) + prod + nxt
        if c != ONE:
            prod = struct.pack("<B", 8) + enc3(c) + prod
        term_nodes.append(prod)
    if not term_nodes:
        return struct.pack("<B", 0) + enc3(ZERO)
    acc = term_nodes[0]
    for nxt in term_nodes[1:]:
        acc = struct.pack("<B", 5) + acc + nxt
    return acc


# ----------------------------------------------------------------------------- Prover + Container Packer
@dataclass
class Relation2Artifacts:
    container: bytes        # exact C_p-byte fixed-size FPN2 container
    desc: bytes             # canonical public CircuitDesc bytes
    circuit_root: bytes     # 32-byte BLAKE2b-256 of desc (pin for --circuit-root)
    stmt_digest: bytes      # 32-byte BLAKE2b-256 of canonical statement bytes
    body_used: int
    profile_name: str


MIN_STANDARD_QUERIES = 72


def prove_relation2(
    cir,
    stmt_bytes: bytes,
    *,
    profile: dict = LLM_STD_V1,
    engine=None,
    queries: int = 72,
    gkr_masks: int = 8,
    gkr_max_s: int = 26,
    n_group: int | None = None,
    rng=None,
    _allow_weak_queries_for_test: bool = False,
) -> Relation2Artifacts:
    """Compile `cir` into canonical `CircuitDesc`, prove it with 0 Phase-2 commitments and 1 batched
    multi-size Coset-4 Basefold opening, and emit a fixed-size `FPN2` Relation 2 container verifiable
    directly by standalone Rust `fpn-verify --circuit-desc ... --circuit-root ... --stmt-digest ...`."""
    if cir.rounds:
        raise ValueError("Relation 2 direct container expects self-contained circuits without legacy cir.rounds")
    if profile["security_tier"] == 1 and queries < MIN_STANDARD_QUERIES and not _allow_weak_queries_for_test:
        raise ValueError(f"STANDARD profile requires queries >= {MIN_STANDARD_QUERIES}, got {queries}")

    # 1. Lookup multiplicities (Phase 1)
    for name in sorted(cir.lookups):
        Z._logup_mult(cir, cir.lookups[name])

    ph1 = [v for v in cir.vecs if v.phase == 1]
    if n_group is None:
        n_group = FE.choose_group_nbits(ph1, queries=queries, min_n=10)

    # 2. Pack & commit Phase-1 groups
    groups: list[MP.GroupCommitment] = []
    group_ns: list[int] = []
    placements: dict[int, FE.VecPlacement] = {}

    ph1_lists, ph1_pl, ph1_ns = FE.pack_phase_adaptive(ph1, n_group, queries, start_group_idx=0)
    placements.update(ph1_pl)
    for g_items, ng in zip(ph1_lists, ph1_ns):
        cap_g = MP.group_data_capacity(ng, queries)
        cells = FE._materialize_group_cells(g_items, cap_g)
        grp = MP.commit_data_group(ng, cells, queries=queries, engine=engine, rng=rng)
        groups.append(grp)
        group_ns.append(ng)

    FE._sync_vec_vals_from_groups(cir, groups, placements)

    # 3. Plan GKR batches & Sumchecks
    gkr_batches = FE.plan_unified_gkr_batches(cir, m_mask=gkr_masks, max_s=gkr_max_s)
    # For clean Relation-2 standalone AST, require all lookup numerators to be linear (no post-GKR sumzeros)
    for _glay, blks in gkr_batches:
        for blk in blks:
            if not blk.is_linear_num:
                raise ValueError(f"Relation 2 direct container requires linear lookup numerators (got {blk.name})")

    sizes = sorted(set(cir.zeros) | set(cir.sums))
    degs = {nv: Z._degree(cir, nv) for nv in sizes}

    # 4. Allocate & commit Mask vectors
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
        n_l, m_m = glay.libra_coeff_count, glay.m_mask
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

    mask_lists, mask_pl, mask_ns = FE.pack_phase_adaptive(mask_vecs, n_group, queries, start_group_idx=len(groups))
    placements.update(mask_pl)
    for g_items, ng in zip(mask_lists, mask_ns):
        cap_g = MP.group_data_capacity(ng, queries)
        cells = FE._materialize_group_cells(g_items, cap_g)
        grp = MP.commit_data_group(ng, cells, queries=queries, engine=engine, rng=rng)
        groups.append(grp)
        group_ns.append(ng)

    # 5. Build canonical binary `CircuitDesc` BEFORE starting transcript!
    n_max = max(group_ns)
    desc_buf = bytearray()
    desc_buf += DESC_MAGIC
    desc_buf += struct.pack("<HBHH", DESC_VERSION, n_max, queries, len(group_ns))
    desc_buf += bytes(group_ns)

    # GKR batch descriptors (with deterministic view_ref deduplication per batch)
    desc_buf += struct.pack("<H", len(gkr_batches))
    gkr_batch_views = []
    for (glay, blks), aux_v in zip(gkr_batches, gkr_aux_vecs):
        # Collect unique View atoms across all blocks of this batch in deterministic order
        v_list = []
        v_map = {}
        for blk in blks:
            for ce in blk.cols:
                for a in Z._view_atoms(ce):
                    if a.kid not in v_map:
                        v_map[a.kid] = len(v_list)
                        v_list.append(a)
            if blk.num_ex is not None:
                for a in Z._view_atoms(blk.num_ex):
                    if a.kid not in v_map:
                        v_map[a.kid] = len(v_list)
                        v_list.append(a)
        gkr_batch_views.append((v_list, v_map))
        desc_buf += struct.pack("<BH", glay.s, glay.m_mask)
        desc_buf += _enc_place(placements[aux_v.id])
        desc_buf += struct.pack("<I", len(v_list))
        for a in v_list:
            desc_buf += _enc_view_ref(a, placements[a.vec.id])
        desc_buf += struct.pack("<I", len(blks))
        for blk in blks:
            desc_buf += struct.pack("<IBQB", blk.offset, blk.nbits, blk.tid, len(blk.cols))
            for ce in blk.cols:
                desc_buf += _enc_ex_ast(ce, v_map)
            if blk.num_ex is None:
                desc_buf += struct.pack("<B", 0)
            else:
                desc_buf += struct.pack("<B", 1) + _enc_ex_ast(blk.num_ex, v_map)

    # Sumcheck descriptors (per cube size nv)
    desc_buf += struct.pack("<H", len(sizes))
    sc_views_per_nv = []
    sc_constraints_per_nv = []
    for nv in sizes:
        D = degs[nv]
        # Build list of individual constraints in exact `Z._instance_terms` order:
        #   first zerochecks `cir.zeros.get(nv, [])` multiplied by EqZC(0..nv-1), then sumzeros `cir.sums.get(nv, [])`!
        c_list = []
        eq_all = Z.Ex.atom(Z.EqZC(list(range(nv))))
        for ex, _tag in cir.zeros.get(nv, ()):
            c_list.append(eq_all * ex)
        for ex, eq_vars, _tag in cir.sums.get(nv, ()):
            if eq_vars:
                c_list.append(Z.Ex.atom(Z.EqZC(list(eq_vars))) * ex)
            else:
                c_list.append(ex)
        sc_constraints_per_nv.append(c_list)

        # Deterministic View atom order across constraints (matching `Z._view_atoms` sorted by repr(key))
        seen = {}
        for cex in c_list:
            for a in Z._view_atoms(cex):
                seen[a.kid] = a
        v_list = sorted(seen.values(), key=lambda a: repr(a.key))
        v_map = {a.kid: i for i, a in enumerate(v_list)}
        sc_views_per_nv.append((v_list, v_map))

        desc_buf += struct.pack("<BBI", nv, D, len(c_list))
        desc_buf += _enc_place(placements[sc_masks[nv].id])
        desc_buf += struct.pack("<I", len(v_list))
        for a in v_list:
            desc_buf += _enc_view_ref(a, placements[a.vec.id])
        for cex in c_list:
            desc_buf += _enc_ex_ast(cex, v_map)

    raw_desc = bytes(desc_buf)
    c_root = circuit_desc_root(raw_desc)
    s_dig = stmt_digest(stmt_bytes)

    # 6. Start normative `Fpn2Transcript` matching `core/src/relation2.rs` byte for byte!
    hdr = header(llm_profile_id(profile))
    tr = Fpn2Transcript(LABEL_V2)
    tr.absorb_bytes(hdr)
    tr.absorb_bytes(c_root)
    tr.absorb_bytes(s_dig)
    for grp in groups:
        tr.absorb_bytes(grp.root)

    body = bytearray()
    body += c_root
    body += s_dig
    for grp in groups:
        body += grp.root

    claims = {gi: Z._Claims() for gi in range(len(groups))}
    sc_B = CPU
    if engine is not None and hasattr(engine, "cp"):
        from flashprove_next import field_gpu as FG
        sc_B = FG.GpuBackend

    # ---- GKR batches
    if gkr_batches:
        gamma_lu = tr.squeeze_fp3()
        beta_lu = tr.squeeze_fp3()
        for (glay, blks), (lc, ma, md), aux_v, (v_list, _v_map) in zip(
            gkr_batches, gkr_batch_masks, gkr_aux_vecs, gkr_batch_views
        ):
            leaves = FE._build_gkr_leaves_direct(glay, blks, gamma_lu, beta_lu, ma, md, B=sc_B, engine=engine)
            gkr_res = FE._prove_gkr_on_packed_leaves(glay, leaves, tr, libra_coeffs=lc, mask_a=ma, mask_d=md, engine=engine)
            del leaves
            body += gkr_res.proof
            r_gkr = gkr_res.r_gkr

            gc_vals = []
            for a in v_list:
                spec = ("sh", a.vec.id, tuple(a.local_point(r_gkr)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r_gkr)))
                pl = placements[a.vec.id]
                val = FE._eval_vec_claim_prover(groups[pl.group_idx].cells, pl, spec)
                claims[pl.group_idx].add(spec, val)
                gc_vals.append(val)

            left_idx, right_idx = glay.mask_leaf_indices
            n_l, m_m = glay.libra_coeff_count, glay.m_mask
            last = m_m - 1
            a_entries, d_entries = [], []
            for m in range(m_m):
                el, er = ONE, ONE
                il, ir = left_idx[m], right_idx[m]
                for bit in range(glay.s):
                    rb = r_gkr[bit]
                    el = hmul(el, rb if (il >> bit) & 1 else hsub(ONE, rb))
                    er = hmul(er, rb if (ir >> bit) & 1 else hsub(ONE, rb))
                if m < last:
                    a_entries.append((n_l + m, hsub(el, er)))
                    d_entries.append((n_l + m_m + m, hadd(el, er)))
                else:
                    d_entries.append((n_l + m_m + m, el))
                    d_entries.append((n_l + m, er))
            aux_pl = placements[aux_v.id]
            val_a = FE._eval_vec_claim_prover(groups[aux_pl.group_idx].cells, aux_pl, ("linear_idx", aux_v.id, tuple(a_entries)))
            val_d = FE._eval_vec_claim_prover(groups[aux_pl.group_idx].cells, aux_pl, ("linear_idx", aux_v.id, tuple(d_entries)))
            claims[aux_pl.group_idx].add(("linear_idx", aux_v.id, tuple(a_entries)), val_a)
            claims[aux_pl.group_idx].add(("linear_idx", aux_v.id, tuple(d_entries)), val_d)
            gc_vals.extend([val_a, val_d])

            for k, (r_k, g_r) in enumerate(zip(gkr_res.layer_points, gkr_res.layer_libra_evals), start=1):
                off_l = glay.layer_libra_offset(k)
                entries_k = []
                for i, ri in enumerate(r_k):
                    r2 = hmul(ri, ri)
                    r3 = hmul(r2, ri)
                    for j, pw in enumerate((ONE, ri, r2, r3)):
                        entries_k.append((off_l + 4 * i + j, pw))
                claims[aux_pl.group_idx].add(("linear_idx", aux_v.id, tuple(entries_k)), g_r)

            gc_raw = enc3s(gc_vals)
            body += gc_raw
            tr.absorb_bytes(gc_raw)

    # ---- Sumchecks
    vcache = {}
    for nv, (v_list, _v_map), c_list in zip(sizes, sc_views_per_nv, sc_constraints_per_nv):
        D = degs[nv]
        r_zc = [tr.squeeze_fp3() for _ in range(nv)]
        lambdas = [tr.squeeze_fp3() for _ in range(len(c_list))]
        total = Z._instance_terms(cir, nv, r_zc, lambdas)
        mv = sc_masks[nv]
        cnt = nv * (D + 1)
        coef = Z._mask_coef(mv, cnt)
        G_sum = Z._mask_sum(coef, nv, D)
        G_bytes = enc3(G_sum)
        body += G_bytes
        tr.absorb_bytes(struct.pack("<I", nv) + G_bytes)
        rho = tr.squeeze_fp3()

        tables = Z._materialize(sc_B, total, nv, vcache)
        msgs_raw, r = FE._sumcheck_prove_fpn2(sc_B, total, nv, D, tables, coef, rho, tr)
        del tables
        body += msgs_raw

        cvals = []
        for a in v_list:
            spec = ("sh", a.vec.id, tuple(a.local_point(r)), a.shift) if a.shift else ("pt", a.vec.id, tuple(a.local_point(r)))
            pl = placements[a.vec.id]
            val = FE._eval_vec_claim_prover(groups[pl.group_idx].cells, pl, spec)
            cvals.append(val)
            claims[pl.group_idx].add(spec, val)
        gval = ZERO
        for i in range(nv):
            gval = hadd(gval, Z._gi(coef, i, D, r[i]))
        mv_pl = placements[mv.id]
        claims[mv_pl.group_idx].add(("pw", mv.id, tuple(r), D), gval)

        cb = enc3s(cvals) + enc3(gval)
        body += cb
        tr.absorb_bytes(cb)
    vcache.clear()

    # ---- Batched multi-size Coset-4 Basefold opening
    pcs_lay = MP.MultiPcsLayout(n=n_max, n_groups=len(groups), queries=queries, group_ns=tuple(group_ns))
    weights = []
    group_claims = []
    for gi in range(len(groups)):
        cl = claims[gi]
        ng = group_ns[gi]
        tr.absorb_bytes(struct.pack("<I", len(cl.items)))
        mus = [tr.squeeze_fp3() for _ in range(len(cl.items))]
        W_g = np.zeros((1 << ng, 3), dtype=np.uint64)
        y_g = ZERO
        for spec, val, mu in zip(cl.items, cl.vals, mus):
            pl = placements[spec[1]]
            lw = FE._claim_weight_table(spec, pl.nbits)
            scaled = MP._fp3_mul_scalar_host(lw, mu)
            sz = 1 << pl.nbits
            W_g[pl.offset:pl.offset + sz] = MP._fp3_add_dev(None, W_g[pl.offset:pl.offset + sz], scaled)
            y_g = hadd(y_g, hmul(mu, val))
        weights.append(W_g)
        group_claims.append(y_g)

    pcs_pf = MP.prove_multi_open(pcs_lay, groups, weights, group_claims, tr, engine=engine, rng=rng)
    body += pcs_pf

    # 7. Exact fixed-size zero-padded FPN2 container (`C_p` bytes!)
    c_p = int(profile["c_p"])
    used = len(body)
    total_len = len(hdr) + used
    if total_len > c_p:
        raise ValueError(f"Relation 2 body {total_len} B exceeds profile C_p = {c_p} B")
    container = bytes(hdr) + bytes(body) + bytes(c_p - total_len)
    assert len(container) == c_p
    return Relation2Artifacts(
        container=container,
        desc=raw_desc,
        circuit_root=c_root,
        stmt_digest=s_dig,
        body_used=used,
        profile_name=profile["name"],
    )
