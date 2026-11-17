#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate real Qwen3-0.6B proof bundles (via live vLLM 0.33 + GPU prover) for `demo/bundles/`.

Run on a CUDA workstation with vLLM 0.33 and CuPy:
  python demo/generate_qwen_bundles.py \
      --model /path/to/Qwen3-0.6B \
      --work /var/tmp/fpn-demo-scratch \
      --serve-python /path/to/vllm_venv/bin/python \
      --prove-python /path/to/cupy_venv/bin/python
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import gzip
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERIFY_CLI = ROOT / "verifier" / "fpn_verify.py"
for p in (str(ROOT), str(ROOT / "integrations" / "vllm")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _verify_bundle(
    prove_py: str,
    model_id: str,
    profile: str,
    nonce: str,
    req_doc: dict,
    rep_doc: dict,
    proof_path: Path,
    tok_dir: Path,
    work: Path,
    allow_reduced: bool = True,
) -> tuple[int, str, float]:
    work.mkdir(parents=True, exist_ok=True)
    rq_p = work / "request.json"
    rp_p = work / "reply.json"
    rq_p.write_text(json.dumps(req_doc), encoding="utf-8")
    rp_p.write_text(json.dumps(rep_doc), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        prove_py, str(VERIFY_CLI),
        "--model-id", model_id,
        "--request", str(rq_p),
        "--reply", str(rp_p),
        "--nonce", nonce,
        "--proof", str(proof_path),
        "--profile", profile,
        "--tokenizer", str(tok_dir),
    ]
    if allow_reduced:
        cmd.append("--allow-reduced-security")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    elapsed = round(time.perf_counter() - t0, 3)
    lines = p.stdout.strip().splitlines()
    verdict = lines[-1] if lines else (p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "EMPTY")
    return p.returncode, verdict, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to unsliced Qwen3-0.6B checkpoint")
    ap.add_argument("--work", required=True, help="Scratch directory (intermediate stores/spools cleaned in finally)")
    ap.add_argument("--serve-python", default=sys.executable)
    ap.add_argument("--prove-python", default=sys.executable)
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--profile", default="qwen3-0.6b-demo")
    args = ap.parse_args()

    from flashprove_next import hwinfo, reply_bundle as RB
    import end_to_end_vllm as E2E

    spec = RB.get_registered_profile(args.profile)
    m_c = int(spec["M_C"])
    m_g = int(spec["M_G"])
    t_pad = int(spec["T_pad"])
    sec_tier = str(spec["security_tier"])

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache_root = work / "compile_cache"
    for sub, var in (
        ("inductor", "TORCHINDUCTOR_CACHE_DIR"),
        ("triton", "TRITON_CACHE_DIR"),
        ("vllm", "VLLM_CACHE_ROOT"),
    ):
        p_sub = cache_root / sub
        p_sub.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(p_sub)
        E2E.ENV_BASE[var] = str(p_sub)
    tmp_bundles = work / "bundles.tmp"
    shutil.rmtree(tmp_bundles, ignore_errors=True)
    tmp_bundles.mkdir(parents=True, exist_ok=True)

    store_dir = str(work / "store")
    spool_dir = str(work / "spool")
    spool_bad = str(work / "spool_bad")
    logs_dir = str(work / "logs")
    public_json = str(work / "public.json")
    shutil.rmtree(spool_dir, ignore_errors=True)
    shutil.rmtree(spool_bad, ignore_errors=True)
    os.makedirs(logs_dir, exist_ok=True)

    vllm_ver = subprocess.check_output(
        [args.serve_python, "-c", "import vllm; print(vllm.__version__)"],
        env=E2E.ENV_BASE, text=True,
    ).strip()

    base_url = f"http://127.0.0.1:{args.port}"
    srv = dmn = None
    try:
        print(f"[1/6] Committing unsliced model weights at {args.model}...", flush=True)
        cinfo = E2E.setup_commitment(
            args.prove_python, args.serve_python, args.model, store_dir, public_json, args.profile, t_pad=t_pad
        )
        model_id = cinfo["model_id"]
        print(f"      model_id = {model_id} (commit_w_s={cinfo['commit_w_s']}s)", flush=True)

        srv = E2E.Proc("vllm-server", [
            args.serve_python, str(ROOT / "integrations" / "vllm" / "fpn_vllm_server.py"),
            "--model", args.model, "--store", store_dir, "--public", public_json, "--spool", spool_dir,
            "--port", str(args.port), "--gpu-memory-utilization", "0.22",
            "--max-num-seqs", "4", "--max-model-len", str(t_pad), "--cudagraph",
        ], os.path.join(logs_dir, "server.log"))
        dmn = E2E.Proc("prover-daemon", [
            args.prove_python, str(ROOT / "integrations" / "vllm" / "prover_daemon.py"),
            "--model", args.model, "--store", store_dir, "--spool", spool_dir,
            "--m-c", str(m_c), "--m-g", str(m_g), "--t-pad", str(t_pad),
            "--profile-name", args.profile, "--vram-gb", "8.0",
        ], os.path.join(logs_dir, "daemon.log"))
        E2E.wait_health(base_url, srv)
        print("[2/6] vLLM 0.33 server + GPU prover daemon ready; proving Case 1 (hidden prompt + CoT)...", flush=True)

        # Case 1: Honest hidden system prompt + hidden <think> CoT (Turn 0)
        r_cot = nc_cot = req_cot = blob_cot = None
        for attempt in range(4):
            nc_try = secrets.token_hex(16)
            req_try = {
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "max_tokens": 160,
                "fpn": {"nonce": nc_try, "thinking": True},
            }
            resp_try, _ = E2E.http_post(base_url + "/v1/chat/completions", req_try, stream=True)
            vis_ids = resp_try.get("fpn_reply_token_ids") or []
            vis_txt = resp_try.get("streamed_text") or ""
            fr = resp_try["choices"][0]["finish_reason"]
            n_rsn = int(resp_try["fpn"]["n_reason"])
            if "<think>" not in vis_txt and "</think>" not in vis_txt and n_rsn > 0 and fr == "stop" and len(vis_ids) >= 2:
                st1, blob1, _ = E2E.poll_proof(base_url, resp_try["fpn"]["proof_id"])
                if st1 != 200:
                    raise SystemExit(f"Case 1 proof poll returned HTTP {st1}")
                nc_cot, req_cot, r_cot, blob_cot = nc_try, req_try, resp_try, blob1
                break
        if r_cot is None or blob_cot is None:
            raise SystemExit("Failed to obtain a clean closed-<think> EOS rollout in 4 attempts")
        pf_cot_path = tmp_bundles / "qwen3_06b_honest_cot.proof"
        pf_cot_path.write_bytes(blob_cot)
        print(f"      Case 1 ready ({len(blob_cot)} B, n_reason={r_cot['fpn']['n_reason']}); proving Case 2 (Turn-1 multi-turn)...", flush=True)

        # Case 1b: Turn-1 multi-turn continuation chaining St_new + AEAD state_blob from Case 1
        env0 = r_cot["fpn"]
        nc_t1 = secrets.token_hex(16)
        req_t1 = {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": r_cot["streamed_text"]},
                {"role": "user", "content": "Which country is that city in?"},
            ],
            "max_tokens": 12,
            "fpn": {
                "nonce": nc_t1,
                "thinking": False,
                "turn": 1,
                "sess": env0["sess"],
                "St_prev": env0["St_new"],
                "state_blob": env0["state_blob"],
            },
        }
        r_t1, _ = E2E.http_post(base_url + "/v1/chat/completions", req_t1)
        if not r_t1["choices"][0]["token_ids"]:
            raise SystemExit("Turn-1 multi-turn reply returned empty token_ids")
        st_t1, blob_t1, _ = E2E.poll_proof(base_url, r_t1["fpn"]["proof_id"])
        if st_t1 != 200:
            raise SystemExit(f"Case 1b multi-turn proof poll returned HTTP {st_t1}")
        pf_t1_path = tmp_bundles / "qwen3_06b_multiturn_turn1.proof"
        pf_t1_path.write_bytes(blob_t1)
        print(f"      Case 2 ready ({len(blob_t1)} B); proving Case 3 (strict JSON Schema)...", flush=True)

        # Case 2: Honest strict JSON Schema constrained decoding
        schema = {
            "type": "object",
            "properties": {
                "city": {"enum": ["Paris", "Lyon", "Nice"]},
                "capital": {"type": "boolean"},
            },
            "required": ["city", "capital"],
            "additionalProperties": False,
        }
        nc_js = secrets.token_hex(16)
        req_js = {
            "messages": [{"role": "user", "content": "Return JSON for France's capital."}],
            "max_tokens": 24,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "city_info", "strict": True, "schema": schema},
            },
            "fpn": {"nonce": nc_js, "thinking": False},
        }
        r_js, _ = E2E.http_post(base_url + "/v1/chat/completions", req_js)
        js_txt = r_js["choices"][0]["message"]["content"]
        parsed_js = json.loads(js_txt)
        if parsed_js.get("city") not in ("Paris", "Lyon", "Nice") or not isinstance(parsed_js.get("capital"), bool):
            raise SystemExit(f"Case 2 JSON output failed schema check: {js_txt!r}")
        st2, blob_js, _ = E2E.poll_proof(base_url, r_js["fpn"]["proof_id"])
        if st2 != 200:
            raise SystemExit(f"Case 2 JSON Schema proof poll returned HTTP {st2}")
        pf_js_path = tmp_bundles / "qwen3_06b_json_schema.proof"
        pf_js_path.write_bytes(blob_js)
        print(f"      Case 3 ready ({len(blob_js)} B, json={js_txt!r}); restarting for Case 4 (int4 downgrade)...", flush=True)

        srv.stop()
        dmn.stop()
        srv = dmn = None

        # Case 3: Dishonest int4-quantized server behind the committed bf16 model_id
        env_bad = {**E2E.ENV_BASE, "FPN_TEST_INT4_DOWNGRADE": "1", "FPN_ALLOW_UNSOUND_TEST": "1"}
        srv = E2E.Proc("vllm-int4-bad", [
            args.serve_python, str(ROOT / "integrations" / "vllm" / "fpn_vllm_server.py"),
            "--model", args.model, "--store", store_dir, "--public", public_json, "--spool", spool_bad,
            "--port", str(args.port), "--gpu-memory-utilization", "0.22",
            "--max-num-seqs", "4", "--max-model-len", str(t_pad), "--cudagraph",
        ], os.path.join(logs_dir, "server_bad.log"), env=env_bad)
        dmn = E2E.Proc("daemon-bad", [
            args.prove_python, str(ROOT / "integrations" / "vllm" / "prover_daemon.py"),
            "--model", args.model, "--store", store_dir, "--spool", spool_bad,
            "--m-c", str(m_c), "--m-g", str(m_g), "--t-pad", str(t_pad),
            "--profile-name", args.profile, "--vram-gb", "8.0", "--force-unsound-for-test",
        ], os.path.join(logs_dir, "daemon_bad.log"), env=env_bad)
        E2E.wait_health(base_url, srv)

        nc_bad = secrets.token_hex(16)
        req_bad = {
            "messages": [{"role": "user", "content": "Explain photosynthesis briefly."}],
            "max_tokens": 8,
            "fpn": {"nonce": nc_bad, "thinking": False},
        }
        r_bad, _ = E2E.http_post(base_url + "/v1/chat/completions", req_bad)
        pid_bad = r_bad["fpn"]["proof_id"]
        st3, _, _ = E2E.poll_proof(base_url, pid_bad)
        if st3 != 409:
            raise SystemExit(f"Expected HTTP 409 withheld from int4 server, got {st3}")
        blob_bad = Path(spool_bad, "failed", f"{pid_bad}.proof").read_bytes()
        pf_bad_path = tmp_bundles / "qwen3_06b_int4_cheat.proof"
        pf_bad_path.write_bytes(blob_bad)
        print(f"[5/6] Case 4 int4 withheld proof captured ({len(blob_bad)} B); running standalone CPU self-checks...", flush=True)

        srv.stop()
        dmn.stop()
        srv = dmn = None

        # Compress Qwen3 tokenizer.json (Apache-2.0) so laptop CPU demo works 100% offline
        tok_src = Path(args.model) / "tokenizer.json"
        tok_gz_path = tmp_bundles / "qwen3_tokenizer.json.gz"
        with open(tok_src, "rb") as f_in, gzip.open(tok_gz_path, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)

        cases = {
            "honest_hidden_cot": {
                "proof_file": "qwen3_06b_honest_cot.proof",
                "proof_bytes": len(blob_cot),
                "nonce": nc_cot,
                "visible_text": r_cot["streamed_text"],
                "request": {
                    "messages": req_cot["messages"],
                    "max_tokens": req_cot["max_tokens"],
                    "fpn": {"thinking": True},
                },
                "reply": {
                    "tokens": r_cot["fpn_reply_token_ids"],
                    "n_reason": r_cot["fpn"]["n_reason"],
                    "St_new": r_cot["fpn"]["St_new"],
                    "finish_reason": r_cot["choices"][0]["finish_reason"],
                },
            },
            "honest_multiturn_turn1": {
                "proof_file": "qwen3_06b_multiturn_turn1.proof",
                "proof_bytes": len(blob_t1),
                "nonce": nc_t1,
                "visible_text": r_t1["choices"][0]["message"]["content"],
                "request": {
                    "messages": req_t1["messages"],
                    "max_tokens": req_t1["max_tokens"],
                    "fpn": {
                        "thinking": False,
                        "turn": 1,
                        "St_prev": env0["St_new"],
                    },
                },
                "reply": {
                    "tokens": r_t1["choices"][0]["token_ids"],
                    "n_reason": r_t1["fpn"]["n_reason"],
                    "St_new": r_t1["fpn"]["St_new"],
                    "finish_reason": r_t1["choices"][0]["finish_reason"],
                },
            },
            "honest_json_schema": {
                "proof_file": "qwen3_06b_json_schema.proof",
                "proof_bytes": len(blob_js),
                "nonce": nc_js,
                "visible_text": js_txt,
                "request": {
                    "messages": req_js["messages"],
                    "max_tokens": req_js["max_tokens"],
                    "response_format": req_js["response_format"],
                    "fpn": {"thinking": False},
                },
                "reply": {
                    "tokens": r_js["choices"][0]["token_ids"],
                    "n_reason": r_js["fpn"]["n_reason"],
                    "St_new": r_js["fpn"]["St_new"],
                    "finish_reason": r_js["choices"][0]["finish_reason"],
                },
            },
            "dishonest_int4_quantized": {
                "proof_file": "qwen3_06b_int4_cheat.proof",
                "proof_bytes": len(blob_bad),
                "nonce": nc_bad,
                "request": {
                    "messages": req_bad["messages"],
                    "max_tokens": req_bad["max_tokens"],
                    "fpn": {"thinking": False},
                },
                "reply": {
                    "tokens": r_bad["choices"][0]["token_ids"],
                    "n_reason": r_bad["fpn"]["n_reason"],
                    "St_new": r_bad["fpn"]["St_new"],
                    "finish_reason": r_bad["choices"][0]["finish_reason"],
                },
            },
        }

        # Self-verify all exported bundles using the exact gzipped tokenizer directory before installing!
        with tempfile.TemporaryDirectory(prefix="fpn-gen-chk-") as chk_s:
            chk = Path(chk_s)
            tok_dir = chk / "tok"
            tok_dir.mkdir()
            with gzip.open(tok_gz_path, "rb") as f_in, open(tok_dir / "tokenizer.json", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            for key in ("honest_hidden_cot", "honest_multiturn_turn1", "honest_json_schema"):
                c = cases[key]
                rc, vout, vs = _verify_bundle(
                    args.prove_python, model_id, args.profile, c["nonce"],
                    c["request"], c["reply"], tmp_bundles / c["proof_file"], tok_dir, chk / key,
                )
                if rc != 0 or vout != "VERIFIED":
                    raise SystemExit(f"Self-check failed on honest bundle {key}: rc={rc}, out={vout!r}")
                c["verify_verdict"] = vout
                c["verify_cpu_s"] = vs

            c_bad = cases["dishonest_int4_quantized"]
            rc_b, vout_b, vs_b = _verify_bundle(
                args.prove_python, model_id, args.profile, c_bad["nonce"],
                c_bad["request"], c_bad["reply"], tmp_bundles / c_bad["proof_file"], tok_dir, chk / "int4_bad",
            )
            if rc_b == 0 or not vout_b.startswith("REJECTED"):
                raise SystemExit(f"Self-check failed to reject int4 bundle: rc={rc_b}, out={vout_b!r}")
            c_bad["verify_verdict"] = vout_b
            c_bad["verify_cpu_s"] = vs_b

            # Verify default production-only policy rejects demo_fast when --allow-reduced-security is omitted
            c0 = cases["honest_hidden_cot"]
            rc_dg, vout_dg, _ = _verify_bundle(
                args.prove_python, model_id, args.profile, c0["nonce"],
                c0["request"], c0["reply"], tmp_bundles / c0["proof_file"], tok_dir, chk / "downgrade_chk",
                allow_reduced=False,
            )
            if rc_dg == 0 or "--allow-reduced-security" not in vout_dg:
                raise SystemExit(f"Default production-only policy failed to reject demo tier: {rc_dg}, {vout_dg!r}")

        manifest = {
            "model": "Qwen/Qwen3-0.6B",
            "engine": f"vLLM {vllm_ver} (CUDA graphs enabled, native bf16 + fp32 head)",
            "machine": hwinfo.describe(),
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_id": model_id,
            "profile": args.profile,
            "security_tier": sec_tier,
            "M_C": m_c,
            "M_G": m_g,
            "T_pad": t_pad,
            "note": (
                "Reduced demo_fast budget (M_C=4, M_G=2) keeps bundles compact (~10.1 MiB) for seconds-scale laptop "
                "inspection and requires explicit --allow-reduced-security; production 2^-40 lifetime soundness uses "
                "qwen3-0.6b-sec315 (M_C=210, M_G=105 = 315 draws)."
            ),
            "cases": cases,
        }
        (tmp_bundles / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Atomic directory replacement
        out_dir = HERE / "bundles"
        old_dir = HERE / "bundles.old"
        shutil.rmtree(old_dir, ignore_errors=True)
        if out_dir.exists():
            os.replace(out_dir, old_dir)
        os.replace(tmp_bundles, out_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
        for fn in glob.glob(os.path.join(store_dir, "W_*")):
            try:
                os.remove(fn)
            except OSError:
                pass
        shutil.rmtree(cache_root, ignore_errors=True)
        print(f"OK: generated and verified all demo bundles in {out_dir}", flush=True)
        return 0
    finally:
        if srv is not None:
            srv.stop()
        if dmn is not None:
            dmn.stop()
        shutil.rmtree(spool_dir, ignore_errors=True)
        shutil.rmtree(spool_bad, ignore_errors=True)
        shutil.rmtree(tmp_bundles, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
