# SPDX-License-Identifier: Apache-2.0
""": live vLLM 0.33 -> fpn_capture PageStore -> pre-challenge ZK-blinded x-chunk commitment
(`runtime/x_commit.py`) -> C_pre -> external seed -> ONE relation 6 (`fpn2-folded-vec-bound-n22-v1`) on CudaEngine ->
zero-plaintext offline verification (Rust `fpn-verify --weight-root --x-root` + detached Python).

    FPN_GIT_SHA=$(git rev-parse HEAD) python bench/tier2/bench_live_vllm_rel6_zk_e2e.py --parity-only
    FPN_GIT_SHA=$(git rev-parse HEAD) python bench/tier2/bench_live_vllm_rel6_zk_e2e.py --model .../Qwen3-0.6B
    FPN_GIT_SHA=$(git rev-parse HEAD) python bench/tier2/bench_live_vllm_rel6_zk_e2e.py --model .../Qwen3-30B-A3B-FP8

Each invocation merges its section into `--out` (default benchmarks/prover/results/live_vllm_rel6_zk_rtx6000d.json).

Protocol (per model): serve the / 12A two-turn chat + side request (FULL_NO_KV capture, prefix caching,
`allow_bf16_reduced_precision_reduction = False`); the PUBLIC row schedule is every fresh (non-cached) turn-2 token
position x every layer x {x_attn, x_mlp} x every 1024-column chunk, in that order; each row is the capped-window
integer encoding of the captured BF16 words (`weight_commit.window_ints(.., 23)`, the shift stays private); all rows
are committed chunk by chunk (B = 64) with `folded_vec_bound.commit_x` on CudaEngine (OS CSPRNG) into
`x_chunks_root` -> `C_pre` BEFORE the external nonce; the challenge picks chunk c* and per row one 32 x 1024 weight
tile (PUBLIC tile rule below) of `SetupCertificate.weight_commitment_root` (`commit_model_weights`, lossless
block-128 FP8 for the MoE); one relation-6 container proves Y = W X for the 64 rows; the verifier receives the
x-chunk opening (x_root + Merkle path), the tile opening, the container and -- MoE only -- the challenged chunk's
captured top-8 expert ids (routing metadata), never an activation value.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bench_live_vllm_e2e as B  # noqa: E402  (sets sys.path + vLLM env defaults)

import numpy as np  # noqa: E402

NONCE = hashlib.blake2b(b"v1 rel6 zk external challenge nonce", digest_size=32).digest()
WINDOW_BITS = 23
SLOTS = ("x_attn", "x_mlp")
DENSE = {"layers": 28, "d_model": 1024, "attn": (("q_proj", 2048), ("k_proj", 1024), ("v_proj", 1024)),
         "mlp": (("gate_proj", 3072), ("up_proj", 3072))}
MOE = {"layers": 48, "d_model": 2048, "attn": (("q_proj", 4096), ("k_proj", 512), ("v_proj", 512)),
       "mlp": (("gate_proj", 768), ("up_proj", 768)), "topk": 8, "experts": 128}


def git_rev():
    return os.environ.get("FPN_GIT_SHA") or B.git_rev()


def u64(b):
    return int.from_bytes(b[:8], "little")


def make_tile_rule(geo, moe):
    """PUBLIC deterministic tile rule (row descriptor, 32-byte draw, routing row) -> (tensor, r0, c0).

    x_attn rows (slot 0) -> q / k / v_proj (draw[0] % 3); x_mlp rows (slot 1) -> dense gate / up_proj, or for the MoE
    the ROUTED expert routing[draw[1] % 8] (the row's captured top-8 ids, committed before the challenge) gate / up
    (draw[0] % 2); r0 = 32 * (u64(draw[8:16]) mod rows / 32); c0 = 1024 * column chunk.  Pad rows -> layer 0 q_proj."""
    from fpn_prover.runtime.x_commit import PAD_ROW

    def rule(desc, draw, routing):
        pos, layer, slot, chunk = desc
        if tuple(desc) == PAD_ROW:
            layer, slot, chunk = 0, 0, 0
        if slot == 0:
            proj, rows = geo["attn"][draw[0] % 3]
            name = f"model.layers.{layer}.self_attn.{proj}.weight"
        else:
            proj, rows = geo["mlp"][draw[0] % 2]
            if moe:
                e = int(routing[draw[1] % geo["topk"]])
                if not 0 <= e < geo["experts"]:
                    raise ValueError("routed expert id out of range")
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
            else:
                name = f"model.layers.{layer}.mlp.{proj}.weight"
        return (name, 32 * (u64(draw[8:16]) % (rows // 32)), 1024 * chunk)
    return rule


def merge_out(path, key, value, top):
    doc = json.load(open(path)) if os.path.exists(path) else {}
    doc.update(top)
    doc.setdefault("runs", {})
    if key == "engine_parity":
        doc["engine_parity"] = value
    else:
        doc["runs"][key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)


# ===================================================================================== engine parity (item 2)
class SeededRng:
    """Explicit test rng (bulk numpy draws) -- parity only; production commit_x uses the OS CSPRNG."""

    def __init__(self, s):
        self.g = np.random.default_rng(s)

    def fp3_array(self, c):
        from flashprove_next.field import P
        return self.g.integers(0, P, size=(c, 3), dtype=np.uint64)


def engine_parity(args):
    from fpn_prover.backend import cuda_engine as CE
    from fpn_prover.backend import folded_vec as FV
    from fpn_prover.backend import folded_vec_bound as FB
    from fpn_prover.backend.fpn2 import FOLDED_VEC_BOUND_N16_V1 as N16, FOLDED_VEC_BOUND_N22_V1 as N22
    from fpn_prover.runtime import x_commit as XC
    out = {}
    for tag, prof, engines in (("n16", N16, ("cpu", "host", "cuda")), ("n22", N22, ("host", "cuda"))):
        lay = FB.layout_of(prof)
        g = np.random.default_rng(11)
        Wi = g.integers(-(1 << 15), 1 << 15, size=(lay.B, FV.ROWS, lay.K))
        Xi = g.integers(-(1 << 15), 1 << 15, size=(lay.B, lay.K))
        Yi, _ = FV.exact_products(Wi, Xi)
        W, X, Y = FV.to_field(Wi), FV.to_field(Xi), FV.to_field(Yi)
        shas = {}
        for e in engines:
            eng = None if e == "cpu" else CE.HostEngine() if e == "host" else CE.CudaEngine()
            FV._W_CACHE.clear()
            t = time.perf_counter()
            xc = FB.commit_x(lay, X, rng=SeededRng(1), engine=eng)
            tc = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            cont = FB.prove_bound(prof, W, X, Y, xc, n_s=b"\x07" * 32, rng=SeededRng(3), engine=eng)
            tp = (time.perf_counter() - t) * 1e3
            wr = FB.weight_root(lay, W, eng)
            FV._W_CACHE.clear()
            rv = XC.rust_verify_rel6(args.fpn_verify, cont, wr, xc.root)
            shas[e] = hashlib.sha256(cont).hexdigest()
            out[f"{tag}_{e}"] = {"sha256": shas[e], "x_root": xc.root.hex(), "commit_x_ms": round(tc, 1),
                                 "prove_bound_ms": round(tp, 1), "rust": XC._code(rv[0]), "bytes": len(cont)}
            print(json.dumps({tag + "_" + e: out[f"{tag}_{e}"]}), flush=True)
        out[f"{tag}_byte_identical"] = len(set(shas.values())) == 1
    out["note"] = ("seeded (explicit numpy rng) commit_x + prove_bound: n16 pure-Python CPU / HostEngine / CudaEngine and "
                   "n22 HostEngine / CudaEngine give byte-identical containers; pure-Python CPU at n22 was not run "
                   "(hours); production commit_x draws its blinding from the OS CSPRNG")
    return out


# ===================================================================================== live run
def chunk_with_root(xcc, c, new_root, XC):
    """(forgery) x_chunks_root rebuilt with chunk c's x_root replaced (honest everything else)."""
    leaves = [XC._leaf(i, new_root if i == c else xcc.x_roots[i], xcc.metas[i]) for i in range(xcc.n_chunks)]
    lv = XC._levels(leaves)
    root = XC.chunks_root(lv[-1][0], n_chunks=xcc.n_chunks, B=xcc.B, K=xcc.K, n_rows=xcc.n_rows,
                          int_bits=xcc.int_bits, profile=xcc.profile, layout_fingerprint=xcc.layout_fingerprint,
                          sched_digest=XC.schedule_digest(xcc.row_desc))
    op = XC.XChunkOpening(c, xcc.n_chunks, xcc.B, xcc.K, new_root, xcc.metas[c],
                          tuple(lv[k][(c >> k) ^ 1] for k in range(len(lv) - 1)))
    return root, op.encode()


def live(args):
    moe = "FP8" in os.path.basename(args.model.rstrip("/"))
    geo = MOE if moe else DENSE
    model_name = os.path.basename(args.model.rstrip("/"))
    res = {"model": model_name, "kind": "moe_fp8" if moe else "dense_bf16", "evidence": "MEASURED"}
    if moe:
        os.environ["VLLM_USE_DEEP_GEMM"] = "0"          # DeepGEMM JIT needs NVCC >= 12.9; rtx6000d-node ships CUDA 12.8
        import bench_live_vllm_moe_e2e as M
        lv = M.serve(args, res)
        lv["pages"] = list(lv["store"].handle(lv["key"]).iter_verified_pages(lv["token_ids"]))
    else:
        lv = B.serve(args, res)

    from fpn_prover.backend import cuda_engine as CE
    from fpn_prover.backend import folded_vec as FV
    from fpn_prover.backend import folded_vec_bound as FB
    from fpn_prover.backend.fpn2 import FOLDED_VEC_BOUND_N22_V1 as PROF, FOLDED_VEC_N22_V1
    from fpn_prover.runtime import weight_commit as WC
    from fpn_prover.runtime import x_commit as XC
    from fpn_prover.runtime.setup_cert import SetupCertificate

    # ---------------------------------------------------------------- one-time setup (not per request)
    t = time.perf_counter()
    ck = WC.SafetensorsCheckpoint(args.model)
    wcom = WC.commit_model_weights(ck, threads=args.commit_threads, window_int_bits=WINDOW_BITS)
    t_wc = (time.perf_counter() - t) * 1e3
    setup = SetupCertificate(wcom.root, hashlib.blake2b(b"v1/rel6-only", digest_size=32).digest(),
                             hashlib.blake2b(b"window23" + (b"/fp8_e4m3_block128" if moe else b"/bf16"),
                                             digest_size=32).digest(),
                             hashlib.blake2b(model_name.encode(), digest_size=32).digest(), 0).encode()
    res["setup"] = {"setup_cert_id": SetupCertificate.decode(setup).id, "weight_commitment_root": wcom.root.hex(),
                    "weight_commit_stats": {k: v for k, v in wcom.stats.items() if k != "skipped"},
                    "weight_commit_wall_ms": round(t_wc, 1),
                    "note": "one-time setup over the checkpoint safetensors (FP8 matrices tiled losslessly with their "
                            "block-128 scales); MoE F32 routers are in the residual digest (not drawn here)"}
    print(json.dumps(res["setup"], default=str)[:2000], flush=True)

    # ---------------------------------------------------------------- PUBLIC row schedule + exact integer rows
    n_tok = len(lv["token_ids"])
    positions = list(range(lv["cached"], n_tok))
    if args.max_positions and len(positions) > args.max_positions:
        positions = positions[-args.max_positions:]
    chunks_per_row = geo["d_model"] // 1024
    t = time.perf_counter()
    rows = B.read_rows(lv["store"], lv["pages"], positions)
    t_read = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    desc, X, routing = [], [], []
    for p in positions:
        for l in range(geo["layers"]):
            ids = None
            if moe:
                ids = np.asarray(rows[p][("expert_ids", l)], dtype=np.int64)[:geo["topk"]]
                if ids.min() < 0 or ids.max() >= geo["experts"]:
                    raise SystemExit("captured expert id out of range")
            for s, slot in enumerate(SLOTS):
                words = rows[p][(slot, l)]
                for c in range(chunks_per_row):
                    xi, _ = WC.window_ints(WC.bf16_values(words[1024 * c:1024 * (c + 1)]), WINDOW_BITS)
                    desc.append((p, l, s, c))
                    X.append(xi)
                    routing.append(ids if (moe and s == 1) else np.zeros(geo.get("topk", 8), dtype=np.int64))
    X = np.array(X, dtype=np.int64)
    R = np.array(routing, dtype=np.uint8) if moe else None
    t_rows = (time.perf_counter() - t) * 1e3
    del rows
    fprint = lv["sess"].layout.fingerprint()
    engine = CE.CudaEngine()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # ---------------------------------------------------------------- pre-challenge commitment -> C_pre
    t = time.perf_counter()
    xcc = XC.commit_x_chunks(PROF, X, desc, fprint, engine=engine, routing_rows=R)
    t_commit = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    c_pre = XC.precommit_x_chunks(setup, xcc.root, lv["out"])
    t_cpre = (time.perf_counter() - t) * 1e3
    per = xcc.timings["commit_x_per_chunk_ms"]
    res["precommit"] = {
        "row_schedule": {"positions": [positions[0], positions[-1] + 1], "n_positions": len(positions),
                         "layers": geo["layers"], "slots": list(SLOTS), "column_chunks_per_row": chunks_per_row,
                         "n_rows": xcc.n_rows, "schedule_digest": XC.schedule_digest(desc).hex(),
                         "cached_prefix_positions_excluded": lv["cached"], "trace_tokens": n_tok},
        "profile": PROF["name"], "B": xcc.B, "K": xcc.K, "int_bits": xcc.int_bits, "n_chunks": xcc.n_chunks,
        "padded_rows_in_last_chunk": xcc.n_chunks * xcc.B - xcc.n_rows, "x_chunks_root": xcc.root.hex(),
        "x_chunks_tree_depth": len(xcc.levels) - 1, "layout_fingerprint": fprint.hex(), "c_pre": c_pre.hex(),
        "routing_metadata_bound": moe,
        "timing_ms": {"pagestore_read_rows": round(t_read, 1), "window_rows": round(t_rows, 1),
                      "commit_x_chunks_total": round(t_commit, 1),
                      "commit_x_total_sum": round(xcc.timings["commit_x_total_ms"], 1),
                      "commit_x_per_chunk": {"median": round(statistics.median(per), 1),
                                             "mean": round(statistics.mean(per), 1), "min": round(min(per), 1),
                                             "max": round(max(per), 1), "first": round(per[0], 1)},
                      "x_chunks_tree": round(xcc.timings["tree_ms"], 2), "precommit_x_chunks": round(t_cpre, 3)},
        "host_cells_retained_mib": round(
            (sum(c.nbytes for c in xcc.cells if c is not None)
             + sum(x.nbytes for x in (xcc.chunk_x_vals or ()) if x is not None)
             + 32 * len(xcc.chunk_seeds or ())) / 2**20,
            2,
        ),
        "maxrss_mib_before_commit": round(rss0, 1),
        "maxrss_mib_after_commit": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    print(json.dumps(res["precommit"]), flush=True)

    # ---------------------------------------------------------------- challenge -> ONE relation 6 (GPU)
    import secrets as _secrets
    nonce = NONCE if args.fixed_nonce else _secrets.token_bytes(32)
    challenge_mode = "fixed_nonce_benchmark" if args.fixed_nonce else "fresh_os_nonce_after_cpre"
    rule = make_tile_rule(geo, moe)
    kw = dict(setup_cert=setup, weight_commitment=wcom, checkpoint=ck, token_ids=lv["token_ids"],
              response_tokens=lv["out"], external_nonce=nonce, tile_rule=rule)
    t = time.perf_counter()
    bun = XC.prove_rel6_zk_bundle(xcc, **kw)
    reps = [{"rep": 0, "wall_ms": round((time.perf_counter() - t) * 1e3, 1), "fresh_blinding": True,
             **{k: (round(v, 1) if isinstance(v, float) else {a: round(b, 1) for a, b in v.items()})
                for k, v in bun.report["timings_ms"].items()}}]
    print(json.dumps(reps[-1]), flush=True)
    c = bun.report["chunk_idx"]
    # Additional timing repetitions use independently blinded commitments so chunk c's blinding is never reused.
    to = WC.TileOpening.decode(bun.tile_opening)
    W_int_rep = np.stack([w for w, _ in to.weights_int(WINDOW_BITS)]).astype(np.int64)
    Wf_rep = FV.to_field(W_int_rep)
    X_int_rep = XC.field_to_int(xcc.chunk_x(c))
    Yf_rep = FV.to_field(np.array([FV.exact_products(W_int_rep[b:b + 1], X_int_rep[b:b + 1])[0][0]
                                   for b in range(xcc.B)], dtype=object))
    for rep in range(1, args.prove_reps):
        t = time.perf_counter()
        xcom_fresh = FB.commit_x(xcc.layout, xcc.chunk_x(c), engine=engine)
        t_cm = (time.perf_counter() - t) * 1e3
        ph = {"timings": {}}
        FV._W_CACHE.clear()
        t_p = time.perf_counter()
        FB.prove_bound(PROF, Wf_rep, xcom_fresh.x_values, Yf_rep, xcom_fresh,
                       n_s=bun.container[160:192], engine=engine, hooks=ph)
        p_ms = (time.perf_counter() - t_p) * 1e3
        reps.append({"rep": rep, "wall_ms": round((time.perf_counter() - t) * 1e3, 1), "fresh_blinding": True,
                     "recommit_chunk_ms": round(t_cm, 1), "prove_bound_ms": round(p_ms, 1),
                     "prove_bound_phases_ms": {a: round(b, 1) for a, b in ph["timings"].items()}})
        print(json.dumps(reps[-1]), flush=True)
    vkw = dict(c_pre=c_pre, token_ids=lv["token_ids"], response_tokens=lv["out"], external_nonce=nonce,
               profile=PROF, row_desc=desc, layout_fingerprint=fprint, x_chunks_root=bun.x_chunks_root,
               chunk_opening=bun.chunk_opening, routing_opening=bun.routing_opening, tile_opening=bun.tile_opening,
               container=bun.container, tile_rule=rule, rust_binary=args.fpn_verify, check_python=True,
               engine=engine, rust_runs=args.verify_runs)
    res["challenge_mode"] = challenge_mode
    t = time.perf_counter()
    ver = XC.verify_rel6_zk_bundle(setup, **vkw)
    t_ver = (time.perf_counter() - t) * 1e3
    print(json.dumps({k: v for k, v in ver.items() if k != "tiles"}), flush=True)
    rows_c = XC.chunk_rows(desc, xcc.B, c)
    to = WC.TileOpening.decode(bun.tile_opening)
    kinds = {}
    for (n, _, _), tl in zip(bun.report["coords"], to.tiles):
        k = ("fp8_expert" if ".experts." in n else "fp8_attention") if tl.kind == WC.KIND_FP8 else \
            ("bf16_attention" if "self_attn" in n else "bf16_mlp")
        kinds[k] = kinds.get(k, 0) + 1
    steady = [r["prove_bound_ms"] for r in reps[1:]] or [reps[0]["prove_bound_ms"]]
    res["relation6"] = {
        "profile": PROF["name"], "container_bytes": len(bun.container), "chunk_idx": c,
        "rows_in_chunk": [list(r) for r in rows_c[:4]] + (["..."] if len(rows_c) > 4 else []),
        "padded_rows_in_chunk": bun.report["padded_rows"], "tile_kinds": kinds,
        "coords_sample": bun.report["coords"][:6], "integer_bits_max": bun.report["integer_bits_max"],
        "integer_faithful": f"{sum(bun.report['integer_faithful'])}/{len(bun.report['integer_faithful'])}",
        "engine": "CudaEngine", "prove_reps": reps,
        "prove_bound_n22_ms_steady_median": round(statistics.median(steady), 1),
        "container_sha256": hashlib.sha256(bun.container).hexdigest()}
    res["verify"] = {
        "verdict": ver["verdict"], "rust": ver.get("rust"), "python": ver.get("python"),
        "rust_verify_ms_median_of_5": round(statistics.median(ver.get("rust_wall_ms", [0])), 2),
        "rust_wall_ms": ver.get("rust_wall_ms"), "python_wall_ms": ver.get("python_wall_ms"),
        "x_chunk_opening_bytes": len(bun.chunk_opening), "routing_opening_bytes": len(bun.routing_opening),
        "weight_tile_opening_bytes": len(bun.tile_opening), "container_bytes": len(bun.container),
        "plaintext_activation_bytes_disclosed": 0,
        "plaintext_activation_bytes_disclosed_note": "the verifier API has no activation parameter; the reply carries "
                                                     "x_root + path (x-chunk opening), weight tiles, the container"
                                                     + (" and the challenged chunk's top-8 expert ids" if moe else ""),
        "tiles_check": ver.get("tiles"), "timings_ms": ver.get("timings_ms"),
        "verify_rel6_zk_bundle_wall_ms": round(t_ver, 1)}

    # ---------------------------------------------------------------- forgeries (all must be rejected)
    def code(**over):
        return XC.verify_rel6_zk_bundle(setup, **{**vkw, "rust_runs": 1, **over})["verdict"]

    W_int = np.stack([w for w, _ in to.weights_int(WINDOW_BITS)]).astype(np.int64)
    Wf = FV.to_field(W_int)
    Xs = XC.field_to_int(xcc.chunk_x(c)).copy()
    b0 = next(b for b, r in enumerate(rows_c) if r != XC.PAD_ROW)
    Xs[b0, 7] += 1
    Ys = FV.to_field(FV.exact_products(W_int, Xs)[0])
    Xsf = FV.to_field(Xs)
    n_s = bun.container[160:192]
    forg = {}
    xcs = FB.commit_x(xcc.layout, Xsf, engine=engine)
    FV._W_CACHE.clear()
    f_a1 = FB.prove_bound(PROF, Wf, Xsf, Ys, xcs, n_s=n_s, engine=engine, hooks={"stmt_x_root": xcc.x_roots[c]})
    forg["a1_case_c_substitution_claims_honest_x_root"] = code(container=f_a1)
    FV._W_CACHE.clear()
    f_a2 = FB.prove_bound(PROF, Wf, Xsf, Ys, xcc.x_commitment(c, _allow_reuse=True), n_s=n_s, engine=engine,
                          hooks={"allow_x_mismatch": True})
    forg["a2_case_c_sumcheck_on_substituted_x_honest_x_com"] = code(container=f_a2)
    FV._W_CACHE.clear()
    f_b = FB.prove_bound(PROF, Wf, Xsf, Ys, xcs, n_s=n_s, engine=engine)
    honest_op = XC.XChunkOpening.decode(bun.chunk_opening)
    sub = XC.XChunkOpening(c, honest_op.n_chunks, honest_op.B, honest_op.K, xcs.root, honest_op.meta,
                           honest_op.path).encode()
    forg["b1_post_challenge_x_root_prime"] = code(chunk_opening=sub, container=f_b)
    root2, op2 = chunk_with_root(xcc, c, xcs.root, XC)
    forg["b2_rebuilt_x_chunks_root"] = code(x_chunks_root=root2, chunk_opening=op2, container=f_b)
    del xcs
    FV._W_CACHE.clear()
    Yh = FV.to_field(FV.exact_products(W_int, XC.field_to_int(xcc.chunk_x(c)))[0])
    r5 = FV.prove(Wf, xcc.chunk_x(c), Yh, n_s=n_s, profile=FOLDED_VEC_N22_V1, engine=engine)
    FV._W_CACHE.clear()
    forg["c_relation5_downgrade_bundle"] = code(container=r5)
    w_root, x_root = bytes.fromhex(ver["weight_root"]), xcc.x_roots[c]
    forg["c_relation5_downgrade_rust_cli"] = XC._code(XC.rust_verify_rel6(args.fpn_verify, r5, w_root, x_root)[0])
    forg["c_relation5_downgrade_python_cli"] = XC._code(XC.python_verify_rel6(r5, w_root, x_root)[0])
    path = list(honest_op.path)
    sb = bytearray(path[0])
    sb[3] ^= 1
    path[0] = bytes(sb)
    forg["d_tampered_x_chunk_sibling"] = code(chunk_opening=XC.XChunkOpening(
        c, honest_op.n_chunks, honest_op.B, honest_op.K, honest_op.x_root, honest_op.meta, tuple(path)).encode())
    forg["d_wrong_chunk_index"] = code(chunk_opening=xcc.opening((c + 1) % xcc.n_chunks).encode(),
                                       routing_opening=(xcc.routing_opening((c + 1) % xcc.n_chunks).encode()
                                                        if moe else b""))
    wb = bytearray(bun.tile_opening)
    wb[-5] ^= 1
    forg["d_tampered_weight_tile"] = code(tile_opening=bytes(wb))
    forg["d_missing_c_pre"] = code(c_pre=None)
    if moe:
        ro = XC.RoutingOpening.decode(bun.routing_opening)
        forg["moe_tampered_routing_expert_id"] = code(routing_opening=XC.RoutingOpening(
            ro.salt, ro.width, ro.data[:8] + bytes([ro.data[8] ^ 1]) + ro.data[9:]).encode())
    forg["stale_n_s"] = code(container=FB.prove_bound(PROF, Wf, xcc.chunk_x(c), Yh,
                                                      xcc.x_commitment(c, _allow_reuse=True),
                                                      n_s=bytes(32), engine=engine))
    FV._W_CACHE.clear()
    expected = {"a1_case_c_substitution_claims_honest_x_root": "E_MERKLE",
                "a2_case_c_sumcheck_on_substituted_x_honest_x_com": "E_SUMCHECK",
                "b1_post_challenge_x_root_prime": "E_X_CHUNK_PATH", "b2_rebuilt_x_chunks_root": "E_CPRE_MISMATCH",
                "c_relation5_downgrade_bundle": "E_X_ROOT_PINNED", "c_relation5_downgrade_rust_cli": "E_X_ROOT_PINNED",
                "c_relation5_downgrade_python_cli": "E_X_ROOT_PINNED", "d_tampered_x_chunk_sibling": "E_X_CHUNK_PATH",
                "d_wrong_chunk_index": "E_X_CHUNK_INDEX", "d_tampered_weight_tile": "E_TILE_PATH",
                "d_missing_c_pre": "E_CPRE_REQUIRED", "moe_tampered_routing_expert_id": "E_X_CHUNK_META",
                "stale_n_s": "E_NS_MISMATCH"}
    res["forgeries"] = {k: {"verdict": v, "expected": expected[k],
                            "rejected_as_expected": v.removeprefix("REJECTED: ") == expected[k]}
                        for k, v in forg.items()}
    res["all_forgeries_rejected_as_expected"] = all(f["rejected_as_expected"] for f in res["forgeries"].values())
    print(json.dumps(res["forgeries"], indent=1), flush=True)
    res["summary_measured"] = {
        "commit_x_chunks_ms": res["precommit"]["timing_ms"]["commit_x_chunks_total"],
        "commit_x_per_chunk_ms_median": res["precommit"]["timing_ms"]["commit_x_per_chunk"]["median"],
        "n_chunks": xcc.n_chunks, "prove_bound_n22_ms": res["relation6"]["prove_bound_n22_ms_steady_median"],
        "rust_verify_ms": res["verify"]["rust_verify_ms_median_of_5"],
        "x_chunk_opening_bytes": len(bun.chunk_opening), "plaintext_activation_bytes_disclosed": 0,
        "integer_faithful": res["relation6"]["integer_faithful"], "verdict": ver["verdict"],
        "all_forgeries_rejected_as_expected": res["all_forgeries_rejected_as_expected"]}
    print(json.dumps(res["summary_measured"]), flush=True)
    return model_name, res


HONEST_LIMITS = [
    "The x-chunk commitment binds the committed X BEFORE the challenge but does not prove X equals the served "
    "activation: captured rows enter commit_x from the prover's PageStore (host), with no in-kernel GPU commitment "
    "during serving, and the plaintext trace_payload_root opening that tied X to the capture in is what "
    "this zero-plaintext path replaces (a prover that commits different rows is caught only by other relations, e.g. "
    "the relation-4 chain, not by relation 6).",
    "commit_x cost scales linearly with the number of pre-committed chunks; lazy ephemeral AES-256-CTR seeding "
    "retains only (~512 KiB of X + 32 B OS-CSPRNG key) per 64-row chunk (~192x less host RAM than storing full "
    "96 MiB blinded cells) and regenerates the challenged chunk on demand.",
    "Only x_attn / x_mlp rows are committed (o_proj / down_proj inputs are not in the schedule); the schedule covers "
    "the fresh turn-2 positions only (cached-prefix positions get no draw, as in ).",
    "X and Y carry no range proof inside relation 6: |x| <= 2^23 is enforced by the honest prover's partition_rows, "
    "not by the circuit; integer faithfulness (sum |W||X| < 2^62) is reported, not proved.",
    "Relation 6 proves Y = W X for the WINDOWED operands (capped exponent window, 23 bits); the window tail vs the "
    "exact BF16 / FP8 operands and the served kernel's own FP8 activation quantisation (MoE) are outside the proof, "
    "as disclosed since ; Y is private and is not compared to the served output here.",
    "MoE: the challenged chunk's captured top-8 expert ids (routing metadata, 8 bytes per row) are disclosed to the "
    "verifier so it can re-derive the routed-expert tile; they are bound pre-challenge by a salted per-chunk digest "
    "in the x_chunks tree, but are not themselves proven to be the router's argmax.",
    "One chunk of n_chunks is drawn uniformly per request: k affected chunks are sampled with probability "
    "k / n_chunks (miss probability 1 - k / n_chunks). Turning this sampling rate into end-to-end cheating "
    "detection requires additionally binding X to the served GPU capture and Y to the served output (neither is "
    "enforced by relation 6 alone).",
    "Single-proof blinding budget per chunk: prove_rel6_zk_bundle enforces at most one proof per committed chunk "
    "(identical retries return the cached bundle; distinct challenges raise E_CHUNK_BLINDING_EXHAUSTED); "
    "benchmark timing repetitions allocate independently blinded commitments.",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="<MODELS_DIR>/Qwen3-0.6B")
    ap.add_argument("--fpn-verify", default=os.path.join(ROOT, "core", "target", "release", "fpn-verify"))
    ap.add_argument("--out", default=os.path.join(ROOT, "internal", "results", "m3", "live_vllm_rel6_zk_rtx6000d.json"))
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--max-positions", type=int, default=0, help="0 = every fresh turn-2 position")
    ap.add_argument("--ring-mib", type=int, default=1536)
    ap.add_argument("--max-batched-tokens", type=int, default=0, help="0 = 512 dense / 128 MoE")
    ap.add_argument("--gpu-mem-util", type=float, default=0.0, help="0 = 0.35 dense / 0.6 MoE")
    ap.add_argument("--prove-reps", type=int, default=3)
    ap.add_argument("--verify-runs", type=int, default=5)
    ap.add_argument("--commit-threads", type=int, default=16)
    ap.add_argument("--fixed-nonce", action="store_true",
                    help="use the deterministic literal NONCE (labels challenge_mode='fixed_nonce_benchmark')")
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args(argv)
    moe = "FP8" in args.model
    args.max_batched_tokens = args.max_batched_tokens or (128 if moe else 512)
    args.gpu_mem_util = args.gpu_mem_util or (0.6 if moe else 0.35)
    from flashprove_next.field import native, native_threads
    if native() is None:
        raise SystemExit("flashprove_next native library unavailable")
    top = {"bench": "tier2/bench_live_vllm_rel6_zk_e2e", "wave": "14A", "git_rev": git_rev(),
           "machine": {"host": platform.node(), "python": platform.python_version(), "native_threads": native_threads()},
           "honest_limits": HONEST_LIMITS}
    try:
        top["machine"]["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    t0 = time.perf_counter()
    if args.parity_only:
        merge_out(args.out, "engine_parity", {**engine_parity(args), "git_rev": git_rev(),
                                              "wall_s": round(time.perf_counter() - t0, 1)}, top)
        return
    name, res = live(args)
    res["git_rev"], res["command"] = git_rev(), " ".join(sys.argv)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    merge_out(args.out, name, res, top)
    os._exit(0)                                          # vLLM worker teardown can hang after a finished run


if __name__ == "__main__":
    main()
