#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Laptop CPU verification demo for FlashProve-Next (zero GPU, zero model weights required).

Verifies real proof bundles produced by unsliced Qwen3-0.6B served on vLLM 0.33 (CUDA graphs + native bf16):
  1. Honest Turn-0 reply (hidden system prompt + hidden <think> CoT)          -> VERIFIED
  2. Honest Turn-1 multi-turn continuation (chaining St_prev -> St_new)       -> VERIFIED
  3. Honest structured output (strict JSON Schema byte-DFA mask over F_{p^3}) -> VERIFIED
  4. Dishonest provider serving int4-quantized weights under bf16 model_id    -> REJECTED
  5. Security-downgrade & tampering checks (omitting --allow-reduced-security,
     modified visible token, replayed nonce, tampered St_prev, schema inject) -> REJECTED
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERIFY_CLI = ROOT / "verifier" / "fpn_verify.py"
BUNDLES_DIR = HERE / "bundles"


def _check_dependencies() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        try:
            import transformers  # noqa: F401
        except ImportError:
            print(
                'ERROR: Public request tokenization requires `tokenizers`. Run:\n'
                '  pip install -e ".[tokenizers]"',
                file=sys.stderr,
            )
            raise SystemExit(1)


def _prepare_tokenizer_dir(tmp_root: Path) -> Path:
    tok_gz = BUNDLES_DIR / "qwen3_tokenizer.json.gz"
    tok_plain = BUNDLES_DIR / "tokenizer.json"
    out_dir = tmp_root / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "tokenizer.json"
    if tok_gz.exists():
        with gzip.open(tok_gz, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return out_dir
    if tok_plain.exists():
        shutil.copy2(tok_plain, dst)
        return out_dir
    raise FileNotFoundError(
        f"Missing demo tokenizer at {tok_gz}; generate bundles via demo/generate_qwen_bundles.py"
    )


def _run_verifier(
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
        sys.executable,
        str(VERIFY_CLI),
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
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    dt = time.perf_counter() - t0
    lines = p.stdout.strip().splitlines()
    verdict = lines[-1] if lines else (p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "EMPTY")
    return p.returncode, verdict, dt


def main() -> int:
    _check_dependencies()
    manifest_path = BUNDLES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Demo bundle manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_id = manifest["model_id"]
    profile = manifest["profile"]
    tier = manifest.get("security_tier", "demo_fast")

    print("=" * 78)
    print("FlashProve-Next — Standalone CPU Verification Demo (CUDA_VISIBLE_DEVICES=\"\")")
    print(f"  Model checkpoint : {manifest['model']} (weights & architecture NOT present here)")
    print(f"  Published ID     : {model_id[:24]}...{model_id[-8:]}")
    print(f"  Profile & tier   : {profile} [{tier}: M_C={manifest['M_C']}, M_G={manifest['M_G']}, T_pad={manifest['T_pad']}]")
    print("                     (uses --allow-reduced-security; full 2^-40 tier = qwen3-0.6b-sec315, 210+105 draws)")
    print("=" * 78)

    fails: list[str] = []
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fpn-demo-") as tmp_s:
        tmp = Path(tmp_s)
        tok_dir = _prepare_tokenizer_dir(tmp)

        # 1. Honest Turn-0 reply with hidden system prompt + hidden <think> CoT
        c_cot = manifest["cases"]["honest_hidden_cot"]
        pf_cot = BUNDLES_DIR / c_cot["proof_file"]
        rc, out, dt = _run_verifier(
            model_id, profile, c_cot["nonce"], c_cot["request"], c_cot["reply"], pf_cot, tok_dir, tmp / "c1"
        )
        ok1 = (rc == 0 and out == "VERIFIED")
        print(f"\n[1/5] Honest Turn-0 Qwen3-0.6B reply (hidden system prompt + {c_cot['reply']['n_reason']} hidden CoT tokens)")
        print(f"      Client prompt : {c_cot['request']['messages'][0]['content']!r}")
        print(f"      Visible reply : {c_cot['visible_text'].strip()!r}  (CoT text & weights stay private)")
        print(f"      Proof size    : {pf_cot.stat().st_size / (1 << 20):.2f} MiB  |  CPU verify: {dt:.2f} s  ->  {out}")
        if not ok1:
            fails.append(f"honest_hidden_cot ({rc}, {out})")

        # 2. Honest Turn-1 multi-turn continuation chaining St_prev -> St_new
        c_t1 = manifest["cases"]["honest_multiturn_turn1"]
        pf_t1 = BUNDLES_DIR / c_t1["proof_file"]
        rc, out, dt = _run_verifier(
            model_id, profile, c_t1["nonce"], c_t1["request"], c_t1["reply"], pf_t1, tok_dir, tmp / "c2"
        )
        ok2 = (rc == 0 and out == "VERIFIED")
        print("\n[2/5] Honest Turn-1 multi-turn continuation (chaining verified anchor St_prev -> St_new)")
        print(f"      Follow-up     : {c_t1['request']['messages'][-1]['content']!r}")
        print(f"      Visible reply : {c_t1['visible_text'].strip()!r}")
        print(f"      Proof size    : {pf_t1.stat().st_size / (1 << 20):.2f} MiB  |  CPU verify: {dt:.2f} s  ->  {out}")
        if not ok2:
            fails.append(f"honest_multiturn_turn1 ({rc}, {out})")

        # 3. Honest reply under strict JSON Schema constrained decoding
        c_js = manifest["cases"]["honest_json_schema"]
        pf_js = BUNDLES_DIR / c_js["proof_file"]
        rc, out, dt = _run_verifier(
            model_id, profile, c_js["nonce"], c_js["request"], c_js["reply"], pf_js, tok_dir, tmp / "c3"
        )
        ok3 = (rc == 0 and out == "VERIFIED")
        print("\n[3/5] Honest Qwen3-0.6B structured output (strict JSON Schema byte-DFA mask bound over F_{p^3})")
        print(f"      Client prompt : {c_js['request']['messages'][0]['content']!r}")
        print(f"      Visible JSON  : {c_js['visible_text']!r}")
        print(f"      Proof size    : {pf_js.stat().st_size / (1 << 20):.2f} MiB  |  CPU verify: {dt:.2f} s  ->  {out}")
        if not ok3:
            fails.append(f"honest_json_schema ({rc}, {out})")

        # 4. Dishonest provider serving int4-quantized weights behind the bf16 model_id
        c_i4 = manifest["cases"]["dishonest_int4_quantized"]
        pf_i4 = BUNDLES_DIR / c_i4["proof_file"]
        rc, out, dt = _run_verifier(
            model_id, profile, c_i4["nonce"], c_i4["request"], c_i4["reply"], pf_i4, tok_dir, tmp / "c4"
        )
        ok4 = (rc != 0 and out.startswith("REJECTED"))
        print("\n[4/5] Dishonest provider serving int4-quantized Qwen3-0.6B under the committed bf16 model_id")
        print(f"      CPU verify    : {dt:.2f} s  ->  {out}")
        if not ok4:
            fails.append(f"dishonest_int4_quantized ({rc}, {out})")

        # 5. Security-downgrade & tampering checks
        print("\n[5/5] Security-downgrade, replay, and tampering checks:")
        # 5a. Omitting --allow-reduced-security rejects non-production demo tier by default
        rc_dg, out_dg, _ = _run_verifier(
            model_id, profile, c_cot["nonce"], c_cot["request"], c_cot["reply"],
            pf_cot, tok_dir, tmp / "c5a", allow_reduced=False,
        )
        if rc_dg == 0 or not out_dg.startswith("REJECTED"):
            fails.append("downgrade_without_flag")
        print(f"      - Demo bundle without opt-in flag ->  {out_dg}")

        # 5b. Modify 1 visible output token
        rep_bad_tok = copy.deepcopy(c_cot["reply"])
        rep_bad_tok["tokens"] = list(rep_bad_tok["tokens"])
        if not rep_bad_tok["tokens"]:
            fails.append("empty_vis_tokens")
        else:
            rep_bad_tok["tokens"][-1] = int(rep_bad_tok["tokens"][-1]) ^ 1
        rc_tok, out_tok, _ = _run_verifier(
            model_id, profile, c_cot["nonce"], c_cot["request"], rep_bad_tok, pf_cot, tok_dir, tmp / "c5b"
        )
        if rc_tok == 0 or not out_tok.startswith("REJECTED"):
            fails.append("tamper_reply_token")
        print(f"      - Modified visible reply token    ->  {out_tok}")

        # 5c. Replay old proof under a fresh client nonce
        bad_nonce = ("ff" if not c_cot["nonce"].startswith("ff") else "ee") + c_cot["nonce"][2:]
        rc_nc, out_nc, _ = _run_verifier(
            model_id, profile, bad_nonce, c_cot["request"], c_cot["reply"], pf_cot, tok_dir, tmp / "c5c"
        )
        if rc_nc == 0 or not out_nc.startswith("REJECTED"):
            fails.append("tamper_nonce_replay")
        print(f"      - Replayed under fresh nonce      ->  {out_nc}")

        # 5d. Tampered multi-turn state anchor St_prev on Turn 1
        req_bad_st = copy.deepcopy(c_t1["request"])
        st_prev_bad = list(req_bad_st["fpn"]["St_prev"])
        st_prev_bad[0] = int(st_prev_bad[0]) + 1
        req_bad_st["fpn"]["St_prev"] = st_prev_bad
        rc_st, out_st, _ = _run_verifier(
            model_id, profile, c_t1["nonce"], req_bad_st, c_t1["reply"], pf_t1, tok_dir, tmp / "c5d"
        )
        if rc_st == 0 or not out_st.startswith("REJECTED"):
            fails.append("tamper_st_prev")
        print(f"      - Tampered multi-turn St_prev     ->  {out_st}")

        # 5e. Server-injected schema check (verifying schema proof against a plain request without schema)
        req_plain_for_js = {
            "messages": c_js["request"]["messages"],
            "max_tokens": c_js["request"]["max_tokens"],
            "fpn": c_js["request"].get("fpn", {"thinking": False}),
        }
        rc_sch, out_sch, _ = _run_verifier(
            model_id, profile, c_js["nonce"], req_plain_for_js, c_js["reply"], pf_js, tok_dir, tmp / "c5e"
        )
        if rc_sch == 0 or not out_sch.startswith("REJECTED"):
            fails.append("tamper_schema_injection")
        print(f"      - Unrequested schema injection    ->  {out_sch}")

    wall = time.perf_counter() - t_total
    print("\n" + "=" * 78)
    if fails:
        print(f"DEMO FAILED: {fails}", file=sys.stderr)
        return 1
    print(f"ALL DEMO VERIFICATIONS PASSED in {wall:.1f} s (3 honest VERIFIED, 6 dishonest/tampered REJECTED)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
