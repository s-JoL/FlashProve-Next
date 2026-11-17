#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live end-to-end suite for `vLLM 0.33` + `flashprove_next.reply_zk` on the full, unsliced `Qwen3-0.6B`.

Starts the serving process (`fpn_vllm_server.py`) and the prover daemon (`prover_daemon.py`), sends real API requests,
and checks every reply with the standalone CPU verifier (`verifier/fpn_verify.py`) from the client side, including
white-box forgeries that must be withheld and rejected.

  python integrations/vllm/end_to_end_vllm.py --model PATH/TO/Qwen3-0.6B --work WORK_DIR \
      [--serve-python VLLM_ENV/bin/python] [--prove-python CUPY_ENV/bin/python] \
      --out integrations/vllm/results/e2e_reply_zk_qwen3_06b.json [--clean-store]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flashprove_next.hwinfo import describe as describe_machine, nvidia_smi  # noqa: E402

VERIFY_CLI = os.path.join(ROOT, "verifier", "fpn_verify.py")
_SMI = nvidia_smi()
ENV_BASE = {
    **os.environ,
    "PATH": (os.path.dirname(_SMI) + os.pathsep if _SMI else "") + os.environ.get("PATH", ""),
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


class VramMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = False
        self.peak_vram_mib = None
        self.peak_host_used_gib = 0.0

    def run(self):
        smi = _SMI
        while not self.stop_flag:
            try:
                if smi is None:               # no nvidia-smi (set FPN_NVIDIA_SMI): VRAM stays None
                    raise FileNotFoundError("nvidia-smi")
                out = subprocess.check_output(
                    [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=3,
                ).decode().strip()
                self.peak_vram_mib = max(self.peak_vram_mib or 0, int(float(out.splitlines()[0])))
            except Exception:
                pass
            try:
                mem = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        k, v = line.split(":", 1)
                        mem[k] = int(v.split()[0])
                used_gib = (mem["MemTotal"] - mem["MemAvailable"]) / (1 << 20)
                self.peak_host_used_gib = max(self.peak_host_used_gib, used_gib)
            except Exception:
                pass
            time.sleep(1.0)


class Proc:
    def __init__(self, name: str, cmd: list[str], log_path: str, env: dict | None = None):
        self.name = name
        self.log_path = log_path
        self.logf = open(log_path, "w")
        self.p = subprocess.Popen(
            cmd, stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env or ENV_BASE, cwd=ROOT, start_new_session=True,
        )

    def stop(self, timeout: float = 30.0):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.p.wait()
        self.logf.close()
        return self.p.returncode


def wait_health(base_url: str, proc: Proc, timeout: float = 420.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.p.poll() is not None:
            tail = open(proc.log_path).read()[-2000:]
            raise RuntimeError(f"{proc.name} exited early ({proc.p.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return round(time.perf_counter() - t0, 2)
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"{proc.name} health timeout after {timeout}s")


def http_post(url: str, body: dict, stream: bool = False, timeout: float = 180.0) -> tuple[dict, dict]:
    body = dict(body)
    body["stream"] = bool(stream)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"content-type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("content-type", "")
        if not stream or "text/event-stream" not in ctype:
            return json.loads(r.read().decode("utf-8")), {"reply_s": round(time.perf_counter() - t0, 4)}
        streamed_ids: list[int] = []
        streamed_text = []
        ttft = None
        final_ev = None
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "error" in ev:
                raise RuntimeError(f"SSE error: {ev['error']}")
            if ev.get("fpn") is not None:
                final_ev = ev
            else:
                ch = ev["choices"][0]
                streamed_ids.extend(ch.get("token_ids") or [])
                streamed_text.append((ch.get("delta") or {}).get("content") or "")
                if ttft is None:
                    ttft = round(time.perf_counter() - t0, 4)
        assert final_ev is not None, "missing final SSE event with the fpn envelope"
        final_ev["streamed_token_ids"] = streamed_ids
        final_ev["streamed_text"] = "".join(streamed_text)
        return final_ev, {"ttft_s": ttft, "reply_s": round(time.perf_counter() - t0, 4)}


def poll_proof(base_url: str, proof_id: str, timeout: float = 900.0) -> tuple[int, bytes, float]:
    t0 = time.perf_counter()
    url = f"{base_url}/v1/proofs/{proof_id}"
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                if r.status == 200:
                    return 200, r.read(), round(time.perf_counter() - t0, 3)
        except urllib.error.HTTPError as e:
            if e.code == 409:
                return 409, e.read(), round(time.perf_counter() - t0, 3)
            if e.code != 202:
                raise
        time.sleep(1.0)
    raise TimeoutError(f"proof {proof_id} not ready after {timeout}s")


def run_standalone_verifier(
    prove_py: str,
    workdir: str,
    model_id_hex: str,
    model_dir: str,
    nonce_hex: str,
    req_doc: dict,
    rep_doc: dict,
    proof_blob: bytes,
    profile_name: str,
) -> tuple[int, str, float]:
    os.makedirs(workdir, exist_ok=True)
    rq_p = os.path.join(workdir, "request.json")
    rp_p = os.path.join(workdir, "reply.json")
    pf_p = os.path.join(workdir, "reply.proof")
    json.dump(req_doc, open(rq_p, "w"))
    json.dump(rep_doc, open(rp_p, "w"))
    open(pf_p, "wb").write(proof_blob)
    env = {k: v for k, v in ENV_BASE.items() if not k.startswith("PYTHON")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        prove_py, "-I", VERIFY_CLI,
        "--model-id", model_id_hex,
        "--request", rq_p,
        "--reply", rp_p,
        "--nonce", nonce_hex,
        "--proof", pf_p,
        "--profile", profile_name,
        "--tokenizer", model_dir,
    ]
    from flashprove_next import reply_bundle as RB
    if profile_name in RB.REGISTERED_PROFILES and RB.REGISTERED_PROFILES[profile_name].get("security_tier") != "production":
        cmd.append("--allow-reduced-security")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    vt = round(time.perf_counter() - t0, 3)
    out = (p.stdout.strip().splitlines() or [p.stderr.strip()[-300:]])[-1]
    return p.returncode, out, vt


def setup_commitment(prove_py: str, serve_py: str, model_dir: str, store_dir: str, public_json: str,
                     profile_name: str, t_pad: int = 256) -> dict:
    """Commit unsliced Qwen3-0.6B `W` once on GPU + write `serving.json` and `public.json`."""
    os.makedirs(store_dir, exist_ok=True)
    # Tokenize the hidden system prompt with HF AutoTokenizer in serve_py
    sys_text = "You are a concise, accurate assistant."
    tok_out = subprocess.check_output([
        serve_py, "-c",
        f"import json; from transformers import AutoTokenizer; "
        f"t = AutoTokenizer.from_pretrained({model_dir!r}); "
        f"ids = t.encode('<|im_start|>system\\n{sys_text}<|im_end|>\\n', add_special_tokens=False); "
        f"print(json.dumps(ids))"
    ], env=ENV_BASE).decode().strip()
    sys_tokens = json.loads(tok_out)
    assert len(sys_tokens) <= 32, len(sys_tokens)

    setup_script = f"""
import json, os, time
import numpy as np
from dataclasses import asdict
from flashprove_next import mc_zk as MZ, modelcommit as MC, reply_bundle as RB, reply_zk as RZ, sampling_v3 as S3
from flashprove_next.field import gpu_backend

B = gpu_backend()
assert B.name == 'cupy'
ck = MC.Checkpoint({model_dir!r})
lazy_t = RZ.LazyCommitTensors(ck, MC.QWEN_PARAMS_R.TI)
prm = MC.QWEN_PARAMS_R
lay = MC.Layout(lazy_t.shapes(), prm, radius=True, cfg_rows=True)
vec_pad = 1 << max(1, (lay.vec_len - 1).bit_length())
store_dir = {store_dir!r}
cjson = {{'W': {{'n_rows': lay.n_rows, 'row_ranges': [list(r) for r in lay.row_ranges], 'radius': True, 'cfg_rows': True}}}}
if os.path.exists(os.path.join(store_dir, 'W_tree.npy')) and os.path.exists(os.path.join(store_dir, 'W_coeffs.i32')):
    json.dump(cjson, open(os.path.join(store_dir, 'commit.json'), 'w'), indent=1)
    w_root = MC.Store(store_dir, prm, tag='W').root
    commit_s = 0.0
else:
    src = MC.RowSource(lazy_t, lay)
    t0 = time.perf_counter()
    cw = MC.commit_matrix(src, lay.n_span, prm, B, store_dir=store_dir, tag='W',
                          store_coeffs=True, store_cw=False, row_ranges=lay.row_ranges)
    commit_s = round(time.perf_counter() - t0, 3)
    w_root = cw['root']
    json.dump(cjson, open(os.path.join(store_dir, 'commit.json'), 'w'), indent=1)

kprf = [0x0123456789ABCDEF % ((1 << 64) - (1 << 32) + 1), 0x11112222, 0x33334444, 0x55556666]
rng = np.random.default_rng(7)
salt_cfg = [int(x) for x in rng.integers(1, 1 << 30, size=4)]
sys_tokens = {sys_tokens!r}
rule = S3.Rule(0.6, 0.95, 32)
vol = MC.encode_volatile_cfg({{
    'k_prf': kprf,
    'salt_cfg': salt_cfg,
    'sys_prompt': sys_tokens,
    'L_real': int(ck.config['num_hidden_layers']),
    'V_real': int(ck.config['vocab_size']),
    'T_bits': rule.T_bits,
    'p16': rule.p16,
    'beta': rule.beta,
}})
ccfg = MC.volatile_config_root(vol)
pre, mid = MZ.model_preimage(w_root, c_cfg=ccfg)
mid_hex = RB.mid_to_hex(mid)

setup_meta = {{
    'model': 'Qwen3-0.6B',
    'prm': asdict(prm),
    'vec_pad': vec_pad,
    'n_rows': lay.n_rows,
    'row_ranges': [list(r) for r in lay.row_ranges],
    'meta_cfg0': lay.meta_cfg0,
    'meta_cfg1': lay.meta_cfg1,
    'mid': [int(x) for x in mid],
    'pre': {{k: ([int(x) for x in v] if isinstance(v, (list, tuple)) else int(v)) for k, v in pre.items()}},
    'vol': [int(x) for x in vol],
    'commit_w_s': commit_s,
}}
json.dump(setup_meta, open(os.path.join(store_dir, 'setup_meta.json'), 'w'), indent=1)
rule = S3.Rule(0.6, 0.95, 32)
serving = {{
    'model_id': mid_hex,
    'sys_tokens': sys_tokens,
    'sys_cap': 32,
    'hist_cap': {t_pad},
    'tool_cap': 32,
    'kprf': kprf,
    'rule': rule.as_dict(),
    'wrap_key_hex': ('77' * 32),
}}
json.dump(serving, open(os.path.join(store_dir, 'serving.json'), 'w'), indent=1)
json.dump({{'model_id': mid_hex, 'profile': {profile_name!r}}}, open({public_json!r}, 'w'), indent=1)
print(json.dumps({{'model_id': mid_hex, 'commit_w_s': commit_s, 'n_rows': lay.n_rows}}))
"""
    out = subprocess.check_output([prove_py, "-c", setup_script], env=ENV_BASE, cwd=ROOT).decode().strip()
    return json.loads(out.splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Hugging Face snapshot of Qwen3-0.6B")
    ap.add_argument("--work", required=True)
    ap.add_argument("--serve-python", default=sys.executable, help="interpreter of the vLLM serving environment")
    ap.add_argument("--prove-python", default=sys.executable, help="interpreter of the CuPy prover environment")
    ap.add_argument("--port", type=int, default=8299)
    ap.add_argument("--m-c", type=int, default=4)
    ap.add_argument("--m-g", type=int, default=2)
    ap.add_argument("--t-pad", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-store", action="store_true")
    ap.add_argument("--only-cases", default="", help="Optional comma-separated subset of case groups (e.g. 6,7,8,9)")
    args = ap.parse_args()
    only = {s.strip() for s in args.only_cases.split(",") if s.strip()}
    want = lambda k: (not only) or (str(k) in only)

    os.makedirs(args.work, exist_ok=True)
    store_dir = os.path.join(args.work, "store")
    spool_dir = os.path.join(args.work, "spool")
    logs_dir = os.path.join(args.work, "logs")
    vfy_dir = os.path.join(args.work, "verify")
    public_json = os.path.join(args.work, "public.json")
    shutil.rmtree(spool_dir, ignore_errors=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(vfy_dir, exist_ok=True)

    prof_name = f"qwen3-0.6b-mc{args.m_c}-mg{args.m_g}"
    base_url = f"http://127.0.0.1:{args.port}"
    mon = VramMonitor()
    mon.start()

    results: dict = {
        "model": "Qwen3-0.6B",
        "machine": describe_machine(),
        "profile": prof_name,
        "M_C": args.m_c,
        "M_G": args.m_g,
        "T_pad": args.t_pad,
        "cases": {},
    }

    srv: Proc | None = None
    dmn: Proc | None = None
    try:
        print("[e2e] 1. Committing real unsliced Qwen3-0.6B...", flush=True)
        cinfo = setup_commitment(
            args.prove_python, args.serve_python, args.model, store_dir, public_json, prof_name, t_pad=args.t_pad
        )
        model_id_hex = cinfo["model_id"]
        results["setup"] = cinfo
        # Verify public.json contains ONLY {model_id, profile} (zero knowledge: nothing else is published)
        pub_doc = json.load(open(public_json))
        assert set(pub_doc.keys()) == {"model_id", "profile"}

        print("[e2e] 2. Starting live vLLM 0.33 (--cudagraph, bf16 + fp32 head, prefix cache) + prover_daemon...", flush=True)
        srv = Proc("vllm-server", [
            args.serve_python, os.path.join(HERE, "fpn_vllm_server.py"),
            "--model", args.model, "--store", store_dir, "--public", public_json, "--spool", spool_dir,
            "--port", str(args.port), "--gpu-memory-utilization", "0.22",
            "--max-num-seqs", "8", "--max-model-len", str(args.t_pad), "--cudagraph",
        ], os.path.join(logs_dir, "server.log"))
        dmn_jsonl = os.path.join(logs_dir, "daemon.jsonl")
        if os.path.exists(dmn_jsonl):
            os.remove(dmn_jsonl)
        dmn = Proc("prover-daemon", [
            args.prove_python, os.path.join(HERE, "prover_daemon.py"),
            "--model", args.model, "--store", store_dir, "--spool", spool_dir,
            "--m-c", str(args.m_c), "--m-g", str(args.m_g), "--t-pad", str(args.t_pad),
            "--profile-name", prof_name, "--vram-gb", "10.0",
            "--log", dmn_jsonl,
        ], os.path.join(logs_dir, "daemon.log"))
        boot_s = wait_health(base_url, srv)
        results["server_boot_s"] = boot_s
        print(f"[e2e] vLLM 0.33 CUDA-Graph server ready in {boot_s}s", flush=True)

        def read_daemon_meta(proof_id: str) -> dict:
            if not os.path.exists(dmn_jsonl):
                return {}
            with open(dmn_jsonl) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    rec = json.loads(ln)
                    if rec.get("proof_id") == proof_id:
                        return rec.get("meta") or {}
            return {}

        def verify_response(case_tag: str, orig_req_body: dict, resp: dict, nonce_hex: str,
                            stream: bool = False) -> dict:
            tenv = resp["fpn"]
            assert tenv["provable"] is True, tenv
            # Zero-knowledge envelope hygiene: operator meta (T, prefix_cache_hits, max_cobatch) never reaches the HTTP reply
            assert "meta" not in tenv, f"ZK envelope leak in fpn response: {list(tenv.keys())}"
            pid = tenv["proof_id"]
            st_code, blob, wait_s = poll_proof(base_url, pid)
            assert st_code == 200, f"{case_tag}: expected HTTP 200 proof, got {st_code}"
            vis_tokens = resp["fpn_reply_token_ids"] if stream else resp["choices"][0]["token_ids"]
            fr = resp["choices"][0]["finish_reason"]
            # Client passes its OWN original raw request JSON (NO server-echoed prompt_token_ids).
            req_doc = {k: v for k, v in orig_req_body.items() if k != "fpn"}
            ta_in = orig_req_body.get("fpn") or {}
            req_doc["fpn"] = {
                k: v for k, v in ta_in.items() if k in ("thinking", "turn", "St_prev", "server_tools")
            }
            rep_doc = {
                "tokens": vis_tokens,
                "n_reason": tenv["n_reason"],
                "St_new": tenv["St_new"],
                "finish_reason": fr,
                "tool_segments": tenv.get("tool_segments"),
            }
            rc, vout, vs = run_standalone_verifier(
                args.prove_python, os.path.join(vfy_dir, case_tag),
                model_id_hex, args.model, nonce_hex, req_doc, rep_doc, blob, prof_name,
            )
            assert rc == 0 and vout == "VERIFIED", f"{case_tag}: verifier failed ({rc}, {vout})"
            dmeta = read_daemon_meta(pid)
            return {
                "pass": True,
                "proof_id": pid,
                "proof_bytes": len(blob),
                "proof_wait_s": wait_s,
                "verify_cpu_s": vs,
                "verdict": vout,
                "finish_reason": fr,
                "n_req": len(resp["prompt_token_ids"]),
                "n_reason": tenv["n_reason"],
                "n_vis": len(vis_tokens),
                "operator_meta": dmeta,
            }

        # -------------------------------------------------------------- Case 1: Concurrent SSE Streaming + CUDA Graph + Non-Blocking
        if want(1):
            print("[e2e] Case 1: 4 concurrent SSE streaming requests + non-blocking probe during GPU proving...", flush=True)
            prompts_4 = [
                "Name the capital of France in one word.",
                "What is 6 times 7?",
                "Name the largest planet in our solar system.",
                "What color is the clear daytime sky?",
            ]
            nonces_4 = [secrets.token_hex(16) for _ in range(4)]

            bodies_4 = [
                {
                    "messages": [{"role": "user", "content": prompts_4[idx]}],
                    "max_tokens": 12,
                    "fpn": {"nonce": nonces_4[idx], "thinking": False},
                }
                for idx in range(4)
            ]

            def fire_one(idx: int):
                return http_post(base_url + "/v1/chat/completions", bodies_4[idx], stream=True)

            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(fire_one, i) for i in range(4)]
                res_4 = [f.result() for f in futs]

            # Immediately fire a non-blocking latency probe WHILE prover_daemon is busy proving the 4 queued jobs on GPU.
            time.sleep(2.0)
            probe_nonce = secrets.token_hex(16)
            body_probe = {
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 4,
                "fpn": {"nonce": probe_nonce, "thinking": False},
            }
            probe_resp, probe_timing = http_post(base_url + "/v1/chat/completions", body_probe, stream=True)
            assert probe_timing["reply_s"] < 5.0, f"inference blocked by proving! reply_s={probe_timing['reply_s']}"

            c1_items = []
            max_cb = 0
            full_replays = 0
            for idx, ((resp_i, tim_i), nc_i) in enumerate(zip(res_4, nonces_4)):
                assert resp_i["streamed_token_ids"] == resp_i["fpn_reply_token_ids"]
                vi = verify_response(f"c1_stream_{idx}", bodies_4[idx], resp_i, nc_i, stream=True)
                vi.update(tim_i)
                max_cb = max(max_cb, int(vi["operator_meta"].get("max_cobatch", 1)))
                full_replays = max(full_replays, int(vi["operator_meta"].get("full_graph_replays_total", 0)))
                c1_items.append(vi)
            # Verify the probe proof too.
            vp = verify_response("c1_nonblocking_probe", body_probe, probe_resp, probe_nonce, stream=True)
            assert max_cb >= 2, f"expected continuous batching max_cobatch >= 2, got {max_cb}"
            assert full_replays > 0, f"expected FULL CUDA-graph replays > 0, got {full_replays}"
            results["cases"]["concurrent_streaming_cudagraph"] = {
                "pass": True,
                "max_cobatch": max_cb,
                "full_cudagraph_replays": full_replays,
                "nonblocking_probe_during_proving_s": probe_timing["reply_s"],
                "requests": c1_items,
                "probe": vp,
            }
            print(f"  -> PASS concurrent_streaming_cudagraph (max_cobatch={max_cb}, "
                  f"full_replays={full_replays}, probe_reply_s={probe_timing['reply_s']}s)", flush=True)

        # -------------------------------------------------------------- Case 2: Prefix Caching Hit
        if want(2):
            print("[e2e] Case 2: Prefix cache hit across shared long prefix...", flush=True)
            shared_ctx = (
                "Context paragraph for prefix cache verification: FlashProve-Next proves native floating-point LLM replies in zero "
                "knowledge on stock vLLM 0.33 without modifying model weights or disabling CUDA graphs. "
            )
            nc_p1, nc_p2 = secrets.token_hex(16), secrets.token_hex(16)
            body_p1 = {
                "messages": [{"role": "user", "content": shared_ctx + "Question A: What engine version is named?"}],
                "max_tokens": 8,
                "fpn": {"nonce": nc_p1, "thinking": False},
            }
            body_p2 = {
                "messages": [{"role": "user", "content": shared_ctx + "Question B: Are weights modified?"}],
                "max_tokens": 8,
                "fpn": {"nonce": nc_p2, "thinking": False},
            }
            r_p1, _ = http_post(base_url + "/v1/chat/completions", body_p1)
            r_p2, _ = http_post(base_url + "/v1/chat/completions", body_p2)
            vp1 = verify_response("c2_prefix_1", body_p1, r_p1, nc_p1)
            vp2 = verify_response("c2_prefix_2", body_p2, r_p2, nc_p2)
            assert int(vp2["operator_meta"].get("prefix_cache_hits", 0)) >= 16, vp2["operator_meta"]
            results["cases"]["prefix_cache"] = {
                "pass": True,
                "prefix_cache_hits_tokens": vp2["operator_meta"]["prefix_cache_hits"],
                "req1": vp1,
                "req2": vp2,
            }
            print(f"  -> PASS prefix_cache (cached_tokens={vp2['operator_meta']['prefix_cache_hits']})", flush=True)

        # -------------------------------------------------------------- Case 3 & 4: Hidden Sys + Hidden CoT + EOS Stop + Multi-Turn State Chain
        if want(3) or want(4):
            print("[e2e] Case 3 & 4: Hidden system prompt + hidden <think> CoT + EOS stop + Turn 0->1 state chain...", flush=True)
            nc_t0 = secrets.token_hex(16)
            body_t0 = {
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "max_tokens": 160,
                "fpn": {"nonce": nc_t0, "thinking": True},
            }
            r_t0, _ = http_post(base_url + "/v1/chat/completions", body_t0, stream=True)
            assert "<think>" not in r_t0["streamed_text"] and "</think>" not in r_t0["streamed_text"]
            assert r_t0["fpn"]["n_reason"] > 0, r_t0["fpn"]
            assert r_t0["choices"][0]["finish_reason"] == "stop", r_t0["choices"][0]
            vt0 = verify_response("c3_cot_eos_turn0", body_t0, r_t0, nc_t0, stream=True)
            results["cases"]["hidden_sys_cot_eos_stop"] = vt0
            print(f"  -> PASS hidden_sys_cot_eos_stop (n_reason={vt0['n_reason']}, vis={r_t0['streamed_text']!r})", flush=True)

            # Turn 1 multi-turn continuation using St_new + encrypted state_blob.
            tenv0 = r_t0["fpn"]
            nc_t1 = secrets.token_hex(16)
            body_t1 = {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": r_t0["streamed_text"]},
                    {"role": "user", "content": "Which country is that city in?"},
                ],
                "max_tokens": 12,
                "fpn": {
                    "nonce": nc_t1,
                    "thinking": False,
                    "turn": 1,
                    "sess": tenv0["sess"],
                    "St_prev": tenv0["St_new"],
                    "state_blob": tenv0["state_blob"],
                },
            }
            r_t1, _ = http_post(base_url + "/v1/chat/completions", body_t1)
            vt1 = verify_response("c4_multiturn_turn1", body_t1, r_t1, nc_t1)
            # Negative check: wrong St_prev must be REJECTED by standalone verifier.
            _, blob_t1, _ = poll_proof(base_url, r_t1["fpn"]["proof_id"])
            bad_st_prev = [(tenv0["St_new"][0] + 1), tenv0["St_new"][1], tenv0["St_new"][2], tenv0["St_new"][3]]
            rc_bad, out_bad, _ = run_standalone_verifier(
                args.prove_python, os.path.join(vfy_dir, "c4_bad_st_prev"),
                model_id_hex, args.model, nc_t1,
                {"messages": body_t1["messages"], "max_tokens": 12, "fpn": {"thinking": False, "turn": 1, "St_prev": bad_st_prev}},
                {"tokens": r_t1["choices"][0]["token_ids"], "n_reason": r_t1["fpn"]["n_reason"],
                 "St_new": r_t1["fpn"]["St_new"], "finish_reason": r_t1["choices"][0]["finish_reason"]},
                blob_t1, prof_name,
            )
            assert rc_bad != 0 and out_bad.startswith("REJECTED"), (rc_bad, out_bad)
            results["cases"]["multiturn_state_chain"] = {
                "pass": True,
                "turn1": vt1,
                "wrong_st_prev_rejected": out_bad,
            }
            print(f"  -> PASS multiturn_state_chain (wrong St_prev -> {out_bad})", flush=True)

        # -------------------------------------------------------------- Case 5: Constrained Decoding (JSON Schema & Strict Tool + Server Tool Witness)
        if want(5):
            print("[e2e] Case 5: Constrained decoding (Structured Outputs JSON Schema)...", flush=True)
            json_schema = {
                "type": "object",
                "properties": {
                    "city": {"enum": ["Paris", "Lyon", "Nice"]},
                    "capital": {"type": "boolean"},
                },
                "required": ["city", "capital"],
                "additionalProperties": False,
            }
            nc_js = secrets.token_hex(16)
            body_js = {
                "messages": [{"role": "user", "content": "Return JSON for France's capital."}],
                "max_tokens": 24,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "city_info", "strict": True, "schema": json_schema},
                },
                "fpn": {"nonce": nc_js, "thinking": False},
            }
            r_js, _ = http_post(base_url + "/v1/chat/completions", body_js)
            js_txt = r_js["choices"][0]["message"]["content"]
            parsed_js = json.loads(js_txt)
            assert parsed_js["city"] in ("Paris", "Lyon", "Nice") and isinstance(parsed_js["capital"], bool)
            v_js = verify_response("c5_json_schema", body_js, r_js, nc_js)
            v_js["parsed_json"] = parsed_js
            results["cases"]["constrained_json_schema"] = v_js
            print(f"  -> PASS constrained_json_schema (json={parsed_js})", flush=True)

        if want(6):
            print("[e2e] Case 6: Strict tool call (`finish_reason='tool_use'`) + in-turn server tool (`z_tool` witness)...", flush=True)
            strict_tools = [{
                "type": "function",
                "function": {
                    "name": "multiply",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "minimum": 15, "maximum": 20},
                            "b": {"type": "integer", "minimum": 20, "maximum": 25},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }]
            # (6a) Client strict tool_use (stops at </tool_call>, finish_reason == 'tool_use')
            nc_tu = secrets.token_hex(16)
            body_tu = {
                "messages": [{"role": "user", "content": "Call multiply with a=17 and b=23."}],
                "max_tokens": 32,
                "tools": strict_tools,
                "fpn": {"nonce": nc_tu, "thinking": False},
            }
            r_tu, _ = http_post(base_url + "/v1/chat/completions", body_tu)
            assert r_tu["choices"][0]["finish_reason"] == "tool_use", r_tu["choices"][0]
            v_tu = verify_response("c6a_client_strict_tool_use", body_tu, r_tu, nc_tu)

            # (6b) In-turn server tool execution: G^(1) -> committed z_tool witness -> G^(2).
            nc_st = secrets.token_hex(16)
            body_st = {
                "messages": [{"role": "user", "content": "Use multiply(17, 23) and state the product."}],
                "max_tokens": 64,
                "tools": strict_tools,
                "fpn": {"nonce": nc_st, "thinking": False, "server_tools": True},
            }
            r_st, _ = http_post(base_url + "/v1/chat/completions", body_st)
            assert r_st["fpn"]["tool_segments"] and len(r_st["fpn"]["tool_segments"]) == 1
            assert r_st["fpn"]["tool_segments"][0]["n_ret"] == 32
            v_st = verify_response("c6b_server_tool_witness", body_st, r_st, nc_st)
            results["cases"]["tool_calls_and_witness"] = {
                "pass": True,
                "client_tool_use": v_tu,
                "server_tool_committed_witness": v_st,
            }
            print("  -> PASS tool_calls_and_witness (both client tool_use & server z_tool witness VERIFIED)", flush=True)

        # -------------------------------------------------------------- Case 7: max_tokens = 0 & length truncation
        if want(7):
            print("[e2e] Case 7: max_tokens=0 empty generation boundary...", flush=True)
            nc_z = secrets.token_hex(16)
            body_z = {
                "messages": [{"role": "user", "content": "Boundary max_tokens=0 check."}],
                "max_tokens": 0,
                "fpn": {"nonce": nc_z, "thinking": False},
            }
            r_z, _ = http_post(base_url + "/v1/chat/completions", body_z)
            assert r_z["choices"][0]["token_ids"] == [] and r_z["choices"][0]["finish_reason"] == "length"
            v_z = verify_response("c7_max_tokens_zero", body_z, r_z, nc_z)
            results["cases"]["max_tokens_zero"] = v_z
            print("  -> PASS max_tokens_zero", flush=True)

        # -------------------------------------------------------------- Case 8: Public refusal of custom string stop
        if want(8):
            print("[e2e] Case 8: Public refusal of custom string `stop`...", flush=True)
            r_stop, _ = http_post(base_url + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Count to ten."}],
                "max_tokens": 16,
                "stop": ["five"],
                "fpn": {"nonce": secrets.token_hex(16)},
            })
            assert r_stop["fpn"]["provable"] is False
            assert r_stop["fpn"]["proof_id"] is None
            assert r_stop["fpn"]["refusal_reason"] == "unsupported_public_policy:custom_stop_sequence"
            results["cases"]["custom_stop_public_refusal"] = {
                "pass": True,
                "fpn": r_stop["fpn"],
            }
            print(f"  -> PASS custom_stop_public_refusal ({r_stop['fpn']['refusal_reason']})", flush=True)

        # Stop honest server & daemon cleanly before Case 9 (int4 downgrade red-team test)
        srv.stop()
        dmn.stop()
        srv = dmn = None

        # -------------------------------------------------------------- Case 9: Dishonest int4 Downgraded Server Withheld & Rejected
        if want(9):
            print("[e2e] Case 9: Dishonest int4-quantized server + forced daemon -> withheld (409) & REJECTED...", flush=True)
            spool_bad = os.path.join(args.work, "spool_bad")
            shutil.rmtree(spool_bad, ignore_errors=True)
            env_bad = {**ENV_BASE, "FPN_TEST_INT4_DOWNGRADE": "1", "FPN_ALLOW_UNSOUND_TEST": "1"}
            srv = Proc("vllm-int4-bad", [
                args.serve_python, os.path.join(HERE, "fpn_vllm_server.py"),
                "--model", args.model, "--store", store_dir, "--public", public_json, "--spool", spool_bad,
                "--port", str(args.port), "--gpu-memory-utilization", "0.22",
                "--max-num-seqs", "4", "--max-model-len", str(args.t_pad), "--cudagraph",
            ], os.path.join(logs_dir, "server_bad.log"), env=env_bad)
            dmn = Proc("daemon-forced-bad", [
                args.prove_python, os.path.join(HERE, "prover_daemon.py"),
                "--model", args.model, "--store", store_dir, "--spool", spool_bad,
                "--m-c", str(args.m_c), "--m-g", str(args.m_g), "--t-pad", str(args.t_pad),
                "--profile-name", prof_name, "--vram-gb", "10.0", "--force-unsound-for-test",
            ], os.path.join(logs_dir, "daemon_bad.log"), env=env_bad)
            wait_health(base_url, srv)
            nc_bad = secrets.token_hex(16)
            r_bad, _ = http_post(base_url + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Explain photosynthesis briefly."}],
                "max_tokens": 8,
                "fpn": {"nonce": nc_bad, "thinking": False},
            })
            pid_bad = r_bad["fpn"]["proof_id"]
            st_bad, _, _ = poll_proof(base_url, pid_bad)
            assert st_bad == 409, f"expected HTTP 409 withheld on downgraded int4 server, got {st_bad}"
            failed_pf = os.path.join(spool_bad, "failed", f"{pid_bad}.proof")
            assert os.path.exists(failed_pf), "expected withheld proof bytes in spool_bad/failed/"
            rc_b, out_b, _ = run_standalone_verifier(
                args.prove_python, os.path.join(vfy_dir, "c9_int4_bad"),
                model_id_hex, args.model, nc_bad,
                {
                    "messages": [{"role": "user", "content": "Explain photosynthesis briefly."}],
                    "max_tokens": 8,
                    "fpn": {"thinking": False},
                },
                {"tokens": r_bad["choices"][0]["token_ids"], "n_reason": r_bad["fpn"]["n_reason"],
                 "St_new": r_bad["fpn"]["St_new"], "finish_reason": r_bad["choices"][0]["finish_reason"]},
                open(failed_pf, "rb").read(), prof_name,
            )
            assert rc_b != 0 and out_b.startswith("REJECTED"), (rc_b, out_b)
            results["cases"]["downgraded_int4_withheld"] = {
                "pass": True,
                "http_status": st_bad,
                "standalone_verifier": out_b,
            }
            print(f"  -> PASS downgraded_int4_withheld (HTTP 409 + standalone verifier: {out_b})", flush=True)

        results["all_pass"] = all(c.get("pass") for c in results["cases"].values())
    finally:
        if srv is not None:
            srv.stop()
        if dmn is not None:
            dmn.stop()
        mon.stop_flag = True
        mon.join(timeout=3)
        results["peak_vram_gib"] = None if mon.peak_vram_mib is None else round(mon.peak_vram_mib / 1024.0, 3)
        results["peak_host_used_gib"] = round(mon.peak_host_used_gib, 3)
        if args.clean_store:
            for fn in glob.glob(os.path.join(store_dir, "W_*")):
                try:
                    os.remove(fn)
                except OSError:
                    pass
            shutil.rmtree(spool_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(args.work, "spool_bad"), ignore_errors=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"[e2e] ALL PASS = {results.get('all_pass')}  peak_vram={results['peak_vram_gib']} GiB  "
          f"peak_ram={results['peak_host_used_gib']} GiB -> {args.out}", flush=True)
    if not results.get("all_pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
