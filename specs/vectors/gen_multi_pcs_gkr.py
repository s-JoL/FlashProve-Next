# SPDX-License-Identifier: Apache-2.0
"""Generate cross-language golden vectors for `multi_gkr` and `multi_pcs` (Python -> Rust parity)."""
from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flashprove_next.field import ONE, P, ZERO, hadd, hf, hmul, hsub
from prover.fpn_prover.backend import multi_gkr as MG
from prover.fpn_prover.backend import multi_pcs as MP
from prover.fpn_prover.backend.fpn2 import Fpn2Transcript


def _dot_fp3(a_np, b_np):
    acc = ZERO
    for i in range(len(a_np)):
        u = (int(a_np[i, 0]), int(a_np[i, 1]), int(a_np[i, 2]))
        v = (int(b_np[i, 0]), int(b_np[i, 1]), int(b_np[i, 2]))
        acc = hadd(acc, hmul(u, v))
    return acc


def build_vectors():
    out = {"multi_gkr": [], "multi_pcs": []}

    # --- 1. multi_gkr honest + tamper + false-root vectors
    glay = MG.MultiGkrLayout(s=6, m_mask=4)
    rng = np.random.default_rng(20261124)
    lc, ma, md = MG.sample_gkr_masks(glay, rng=rng)
    tr0 = Fpn2Transcript(b"vec/multi_gkr")
    beta = tr0.squeeze_fp3()
    qs = [0, 3, 7, 2, 3, 5, 1, 7, 0, 4, 6, 2]
    ms = [qs.count(t) for t in range(8)]
    rn = np.array([ONE] * len(qs) + [hsub(ZERO, hf(m)) for m in ms], dtype=np.uint64)
    rd = np.array([hsub(beta, hf(x)) for x in qs] + [hsub(beta, hf(t)) for t in range(8)], dtype=np.uint64)

    res = MG.prove_multi_gkr(glay, rn, rd, Fpn2Transcript(b"vec/gkr/h"), libra_coeffs=lc, mask_a=ma, mask_d=md)
    ver = MG.verify_multi_gkr(glay, res.proof, Fpn2Transcript(b"vec/gkr/h"))
    out["multi_gkr"].append({
        "name": "gkr_s6_honest",
        "s": glay.s, "m_mask": glay.m_mask, "label": "vec/gkr/h",
        "proof_hex": res.proof.hex(),
        "expected": "VERIFIED",
        "num_eval": list(ver.num_eval),
        "den_eval": list(ver.den_eval),
    })

    # Tampered layer-1 round -> E_GKR
    bad_gkr = bytearray(res.proof)
    bad_gkr[48 + 96 + 24] ^= 1
    out["multi_gkr"].append({
        "name": "gkr_s6_bad_round",
        "s": glay.s, "m_mask": glay.m_mask, "label": "vec/gkr/h",
        "proof_hex": bytes(bad_gkr).hex(),
        "expected": "E_GKR",
    })

    # Out-of-table query -> E_GKR_ROOT
    qs_bad = list(qs)
    qs_bad[0] = 9
    rd_bad = np.array([hsub(beta, hf(x)) for x in qs_bad] + [hsub(beta, hf(t)) for t in range(8)], dtype=np.uint64)
    res_bad = MG.prove_multi_gkr(glay, rn, rd_bad, Fpn2Transcript(b"vec/gkr/bad"), libra_coeffs=lc, mask_a=ma, mask_d=md)
    out["multi_gkr"].append({
        "name": "gkr_s6_bad_root",
        "s": glay.s, "m_mask": glay.m_mask, "label": "vec/gkr/bad",
        "proof_hex": res_bad.proof.hex(),
        "expected": "E_GKR_ROOT",
    })

    # --- 2. multi_pcs mixed-size layer-injection honest + negative vectors
    gns = (12, 10, 8)
    play = MP.MultiPcsLayout(n=12, n_groups=3, queries=4, group_ns=gns)
    rng_p = np.random.default_rng(7777)
    groups, weights, claims, pts = [], [], [], []
    for ng in gns:
        cap = MP.group_data_capacity(ng, play.queries)
        data = rng_p.integers(0, 5000, size=cap, dtype=np.uint64)
        grp = MP.commit_data_group(ng, data, queries=play.queries, rng=rng_p)
        pt = [tuple(int(x) for x in rng_p.integers(0, P, size=3, dtype=np.uint64)) for _ in range(ng)]
        w = MP._eq_table_fp3(pt)
        claims.append(_dot_fp3(grp.cells, w))
        groups.append(grp)
        weights.append(w)
        pts.append(pt)

    tr_p = Fpn2Transcript(b"vec/pcs/mixed")
    pf = MP.prove_multi_open(play, groups, weights, claims, tr_p, rng=rng_p)
    tr_v = Fpn2Transcript(b"vec/pcs/mixed")
    MP.verify_multi_open(
        play, [g.root for g in groups], claims,
        lambda g, r_ip_g: MP.eq_eval_pt(pts[g], r_ip_g),
        pf, tr_v,
    )

    base_entry = {
        "n": play.n, "queries": play.queries, "group_ns": list(gns),
        "label": "vec/pcs/mixed",
        "roots_hex": [g.root.hex() for g in groups],
        "claims": [list(c) for c in claims],
        "pts": [[list(x) for x in pt] for pt in pts],
    }
    out["multi_pcs"].append({**base_entry, "name": "pcs_mixed_12_10_8_honest", "proof_hex": pf.hex(), "expected": "VERIFIED"})

    # False claim on smallest n=8 group -> E_SUMCHECK
    bad_c = [list(c) for c in claims]
    bad_c[2] = list(hadd(claims[2], ONE))
    out["multi_pcs"].append({**base_entry, "name": "pcs_mixed_bad_claim", "claims": bad_c, "proof_hex": pf.hex(), "expected": "E_SUMCHECK"})

    # Tampered FINAL -> E_FINAL
    fin_off = 64 + play.step_a_bytes + play.libra_open_bytes + play.step_b_header_bytes - 24
    bad_fin = bytearray(pf)
    bad_fin[fin_off] ^= 1
    out["multi_pcs"].append({**base_entry, "name": "pcs_mixed_bad_final", "proof_hex": bytes(bad_fin).hex(), "expected": "E_FINAL"})

    # Tampered query coset -> E_MERKLE
    q_off = 64 + play.step_a_bytes + play.libra_open_bytes + play.step_b_header_bytes
    bad_q = bytearray(pf)
    bad_q[q_off] ^= 1
    out["multi_pcs"].append({**base_entry, "name": "pcs_mixed_bad_merkle", "proof_hex": bytes(bad_q).hex(), "expected": "E_MERKLE"})

    # Noncanonical limb -> E_NONCANONICAL
    bad_nc = bytearray(pf)
    bad_nc[64:72] = P.to_bytes(8, "little")
    out["multi_pcs"].append({**base_entry, "name": "pcs_mixed_noncanonical", "proof_hex": bytes(bad_nc).hex(), "expected": "E_NONCANONICAL"})

    return out


def main():
    vecs = build_vectors()
    path = os.path.join(os.path.dirname(__file__), "fpn2_multi_pcs_gkr.json")
    with open(path, "w") as f:
        json.dump(vecs, f, indent=1)
    print(f"wrote {path}: {len(vecs['multi_gkr'])} GKR + {len(vecs['multi_pcs'])} PCS vectors")


if __name__ == "__main__":
    main()
