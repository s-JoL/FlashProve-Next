#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous `reply_zk` prover daemon (`spool/pending` -> GPU proof -> isolated CPU self-check -> `spool/ready`).

  python integrations/vllm/prover_daemon.py --model PATH/TO/Qwen3-0.6B --store STORE_DIR --spool SPOOL \
      [--m-c 210] [--m-g 105] [--profile-name qwen3-0.6b-sec315] [--vram-gb 11.5]

For every job written by `fpn_vllm_server.py`:
  1. Atomically claims `pending/<id>` -> `working/<owner>/<id>`.
  2. Commits `H` (`RZ.make_wit`) and proves `RZ.prove` on GPU (`FPN_OOC=1`).
  3. Packages `FPNZK1` bundle (`flashprove_next.reply_bundle.pack_bundle`).
  4. Runs `verifier/fpn_verify.py` in an isolated CPU-only subprocess (`python -I`, `CUDA_VISIBLE_DEVICES=""`)
     against the published `model_id` and public `(request.json, reply.json, nonce, --tokenizer)`.
  5. Moves the proof to `ready/` ONLY on exit code 0 + `"VERIFIED"`; otherwise records `failed/<id>` (`HTTP 409 withheld`).
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import heapq
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from spool import Spool  # noqa: E402
from flashprove_next import modelcommit as MC  # noqa: E402
from flashprove_next import reply_bundle as RB  # noqa: E402
from flashprove_next import reply_zk as RZ  # noqa: E402
from flashprove_next import sampling_v3 as S3  # noqa: E402
from flashprove_next import zk_flat as Z  # noqa: E402
from flashprove_next.field import CPU, gpu_backend  # noqa: E402

VERIFY_CLI = os.path.join(ROOT, "verifier", "fpn_verify.py")
STOP = {"n": 0}


def _sig(signum, frame):
    STOP["n"] += 1
    if STOP["n"] > 1:
        os._exit(130)


class Releaser(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="fpn-releaser")
        self.cv = threading.Condition()
        self.heap, self.seq = [], 0

    def schedule(self, when: float, fn):
        with self.cv:
            heapq.heappush(self.heap, (when, self.seq, fn))
            self.seq += 1
            self.cv.notify()

    def pending(self) -> int:
        with self.cv:
            return len(self.heap)

    def run(self):
        while True:
            with self.cv:
                while not self.heap:
                    self.cv.wait()
                when, _, fn = self.heap[0]
                dt = when - time.time()
                if dt > 0:
                    self.cv.wait(timeout=dt)
                    continue
                heapq.heappop(self.heap)
            try:
                fn()
            except Exception as e:
                print(f"[daemon] release failed: {e}", flush=True)


def release_time(created: float, tick: float) -> float:
    now = time.time()
    if tick <= 0:
        return now
    k = max(1, math.ceil((now - created) / tick - 1e-9))
    return created + k * tick


def _peak_rss_gib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / (1 << 20), 3)
    except OSError:
        pass
    return 0.0


def self_check(workdir: str, model_id_hex: str, model_dir: str, job: dict, blob: bytes,
               timeout: float, profile_name: str, allow_reduced_security: bool = False) -> tuple[int, str, float]:
    os.makedirs(workdir, mode=0o700, exist_ok=True)
    rq_p, rp_p, pf_p = (os.path.join(workdir, fn) for fn in ("request.json", "reply.json", "reply.proof"))
    api = job["api"]
    am = api.get("api_meta") or {}
    st_s = job["st_stmt"]
    if job.get("raw_req") is not None:
        req_doc = dict(job["raw_req"])
    else:
        req_doc = {
            "tokens": api["req_tokens"],
            "turn": int(st_s["turn"]),
            "St_prev": st_s["St_prev"],
            "max_tokens": am.get("max_tokens"),
            "schema": am.get("schema"),
        }
    rep_doc = {
        "tokens": api["vis_tokens"],
        "n_reason": int(api["n_reason"]),
        "St_new": st_s["St_new"],
        "finish_reason": am.get("finish_reason"),
        "tool_segments": am.get("tool_segments"),
    }
    with open(rq_p, "w") as f:
        json.dump(req_doc, f)
    with open(rp_p, "w") as f:
        json.dump(rep_doc, f)
    with open(pf_p, "wb") as f:
        f.write(blob)

    cmd = [
        sys.executable, "-I", VERIFY_CLI,
        "--model-id", model_id_hex,
        "--request", rq_p,
        "--reply", rp_p,
        "--nonce", job["nonce"],
        "--proof", pf_p,
        "--profile", profile_name,
        "--tokenizer", model_dir,
    ]
    if allow_reduced_security:
        cmd.append("--allow-reduced-security")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        lines = p.stdout.strip().splitlines()
        out = lines[-1] if lines else (p.stderr.strip()[-300:] or "REJECTED: empty verifier output")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc, out = 1, "REJECTED: self-check timed out"
    for fp in (rq_p, rp_p, pf_p):
        try:
            os.remove(fp)
        except OSError:
            pass
    return rc, out, round(time.perf_counter() - t0, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--spool", required=True)
    ap.add_argument("--backend", default="cupy", choices=["cupy", "cpu"])
    ap.add_argument("--m-c", type=int, default=210)
    ap.add_argument("--m-g", type=int, default=105)
    ap.add_argument("--t-pad", type=int, default=256)
    ap.add_argument("--profile-name", default="qwen3-0.6b-sec315")
    ap.add_argument("--vram-gb", type=float, default=11.5)
    ap.add_argument("--poll", type=float, default=0.3)
    ap.add_argument("--exit-when-idle", type=float, default=0.0)
    ap.add_argument("--verify-timeout", type=float, default=600.0)
    ap.add_argument("--release-tick-s", type=float, default=0.0)
    ap.add_argument("--force-unsound-for-test", action="store_true")
    ap.add_argument("--pad-proof-bytes", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    if args.force_unsound_for_test:
        if os.environ.get("FPN_ALLOW_UNSOUND_TEST") != "1":
            raise SystemExit("ERROR: --force-unsound-for-test requires FPN_ALLOW_UNSOUND_TEST=1")

    os.environ["FPN_OOC"] = "1"
    Z.OOC = True
    if not os.environ.get("FPN_SPILL_DIR"):
        os.environ["FPN_SPILL_DIR"] = os.path.abspath(os.path.join(args.spool, "spill"))
    os.makedirs(os.environ["FPN_SPILL_DIR"], mode=0o700, exist_ok=True)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    B = gpu_backend() if args.backend == "cupy" else CPU
    if B.name == "cupy":
        import cupy as cp
        cp.get_default_memory_pool().set_limit(size=int(args.vram_gb * (1 << 30)))

    ck = MC.Checkpoint(args.model)
    lazy_t = RZ.LazyCommitTensors(ck, MC.QWEN_PARAMS_R.TI)
    setup_meta = json.load(open(os.path.join(args.store, "setup_meta.json")))
    serving = json.load(open(os.path.join(args.store, "serving.json")))
    model_id_hex = serving["model_id"]
    rd = serving["rule"]
    rule = S3.Rule(float(rd["temperature"]), float(rd["top_p"]), int(rd.get("beta", 32)))

    lay = MC.Layout(lazy_t.shapes(), prm=MC.QWEN_PARAMS_R, radius=True, cfg_rows=True)
    store = MC.Store(args.store, MC.QWEN_PARAMS_R, tag="W")
    com = {
        "layout": lay,
        "store": store,
        "pre": {k: tuple(v) if isinstance(v, list) else v for k, v in setup_meta["pre"].items()},
        "mid": tuple(setup_meta["mid"]),
        "vol": np.asarray(setup_meta["vol"], dtype=np.uint64),
    }
    assert RB.mid_to_hex(com["mid"]) == model_id_hex

    prof_spec = RB.profile_spec_from_hf_config(
        ck.config,
        name=args.profile_name,
        vec_pad=int(setup_meta["vec_pad"]),
        t_pad=args.t_pad,
        m_c=args.m_c,
        m_g=args.m_g,
        sys_cap=int(serving.get("sys_cap", 32)),
        hist_cap=int(serving.get("hist_cap", args.t_pad)),
        tool_cap=int(serving.get("tool_cap", 32)),
        rule=rule,
    )
    tok_bytes = RB.token_bytes_from_dir(args.model, int(ck.config["vocab_size"]))
    prf = RB.build_profile(prof_spec, token_bytes=tok_bytes)

    sp = Spool(args.spool)
    owner = sp.attach_worker()
    sp.requeue_stale()
    rel = Releaser()
    rel.start()
    log_lock = threading.Lock()
    print(f"[daemon] ready: model_id={model_id_hex[:16]}.. profile={args.profile_name} "
          f"(M_C={args.m_c}, M_G={args.m_g}, T_pad={args.t_pad}) worker={owner}", flush=True)

    def write_log(rec: dict):
        line = json.dumps(rec, default=str)
        print("[daemon]", line, flush=True)
        if args.log:
            data = (line + "\n").encode()
            with log_lock:
                fd = os.open(args.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    os.write(fd, data)
                finally:
                    os.close(fd)

    def finalize(pid, job, rec, action):
        created = job.get("created", time.time())
        rec["self_check_done_after_s"] = round(time.time() - created, 3)

        def go():
            action()
            rec["released_after_s"] = round(time.time() - created, 3)
            write_log(rec)

        if args.release_tick_s > 0:
            rel.schedule(release_time(created, args.release_tick_s), go)
        else:
            go()

    idle_since = time.time()
    while STOP["n"] == 0:
        got = sp.claim_next()
        if got is None:
            if args.exit_when_idle and time.time() - idle_since > args.exit_when_idle:
                break
            time.sleep(args.poll)
            continue
        pid, job = got
        idle_since = time.time()
        rec = {
            "proof_id": pid,
            "profile": args.profile_name,
            "worker": owner,
            "queued_s": round(time.time() - job.get("created", time.time()), 3),
            "meta": job.get("meta"),
        }
        if job.get("unprovable"):
            rec["outcome"] = "trace_unavailable"
            finalize(pid, job, rec, lambda p_=pid, r_=rec, u_=job["unprovable"]: sp.finish_failed(p_, u_, info=r_))
            continue
        if job.get("model_id") != model_id_hex:
            rec["outcome"] = "wrong_model_id"
            finalize(pid, job, rec, lambda p_=pid, r_=rec: sp.finish_failed(p_, "wrong model_id", info=r_))
            continue

        t1 = time.perf_counter()
        Z.FORCE = bool(args.force_unsound_for_test)
        try:
            R_root, wit = RZ.make_wit(
                prf, lazy_t, job["cap"], com,
                tokens=job["full_tokens"], sys_len=int(job["sys_len"]), B=B,
            )
            wit["hs_wit"]["release_h"] = True
            wit["st"] = job["st_wit"]
            stmt = RZ.statement(prf, com, R_root, job["api"], job["st_stmt"])
            pf, info = RZ.prove(prf, stmt, wit, B=B)
            blob = RB.pack_bundle(prof_spec, stmt, pf, pad_to_bytes=args.pad_proof_bytes)
        except Exception as e:
            rec.update(prove_s=round(time.perf_counter() - t1, 3), outcome="prover_refused",
                       error=f"{type(e).__name__}: {e}"[:400])
            finalize(pid, job, rec, lambda p_=pid, r_=rec: sp.finish_failed(p_, "prover refused", info=r_))
            continue
        finally:
            Z.FORCE = False
            gc.collect()
            if B.name == "cupy":
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()

        rec.update(
            prove_total_s=round(time.perf_counter() - t1, 3),
            build_s=info["build_s"],
            prove_s=info["prove_s"],
            cells=info["cells"],
            proof_bytes=len(blob),
            gadget_fails=info["gadget_fails"],
            violations=info["violations"][:4],
            peak_rss_gib=_peak_rss_gib(),
        )
        wdir = sp.wpath(pid, ".chk")
        rc, out, vt = self_check(
            wdir, model_id_hex, args.model, job, blob, args.verify_timeout, args.profile_name,
            allow_reduced_security=(prof_spec.get("security_tier") != "production"),
        )
        try:
            os.rmdir(wdir)
        except OSError:
            pass
        rec.update(self_check=out, self_check_s=vt)
        if rc == 0 and out == "VERIFIED":
            rec["outcome"] = "ready"
            finalize(pid, job, rec, lambda p_=pid, b_=blob, r_=rec: sp.finish_ready(p_, b_, r_))
        else:
            rec["outcome"] = "withheld"
            finalize(pid, job, rec, lambda p_=pid, b_=blob, r_=rec, o_=out: sp.finish_failed(
                p_, "withheld: " + o_, blob=b_, info=r_
            ))
        idle_since = time.time()

    while rel.pending():
        time.sleep(0.1)
    sp.detach_worker()
    print("[daemon] stopped", flush=True)


if __name__ == "__main__":
    main()
