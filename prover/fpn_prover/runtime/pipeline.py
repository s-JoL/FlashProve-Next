# SPDX-License-Identifier: Apache-2.0
"""Read private sampled activation rows from a settled capture trace and drive Relation-4 proving."""

import hashlib
import struct
from dataclasses import dataclass

from ..circuit import e2e_prover as E2E
from ..circuit import lower as LW
from ..circuit import protocol
from . import trace_commit as TC
from .setup_cert import SetupCertificate

_PORTS = frozenset(("h_in", "h_mid", "q", "o", "lse", "a", "h_L"))
_CPRE_PERSON = b"FPN2-CPRE-v1\0\0\0\0"


def commit_trace_payload(store, trace, request_id, token_ids):
    """Canonical 32-byte `trace_payload_root`: the Merkle root over EVERY captured token row (full pages and the
    partial tail) bound to the token count and the capture-layout fingerprint (`trace_commit.commit_trace_merkle`).
    Identical logical rows give identical roots whatever the page assembly; drawn rows open against it."""
    return TC.commit_trace_merkle(store, trace, request_id, token_ids, retain_rows=False).root


def drawn_positions(draws):
    """Token positions of every nonempty compute draw (fresh-input + generation strata) -- the rows to open."""
    return sorted({d.position for s in (protocol.STRATUM_FRESH_INPUT, protocol.STRATUM_GENERATION)
                   for d in draws[s] if not d.empty})


def _exact_u32_tokens(tokens, name):
    out = []
    for t in tokens:
        if type(t) is not int:
            raise TypeError(f"{name} must contain exact ints, got {type(t).__name__}")
        if not 0 <= t <= 0xFFFFFFFF:
            raise ValueError(f"{name} element {t} out of u32 range")
        out.append(t)
    return out


def request_seed(c_pre, token_ids, response_tokens, challenge_mode=protocol.MODE_EXTERNAL_SEED, external_nonce=None):
    """Protocol v2 challenge seed from PUBLIC data (prompt/forwarded token ids, response, C_pre, nonce)."""
    toks = _exact_u32_tokens(token_ids, "token_ids")
    resps = _exact_u32_tokens(response_tokens, "response_tokens")
    prompt_bytes = struct.pack(f"<{len(toks)}I", *toks)
    resp_bytes = struct.pack(f"<{len(resps)}I", *resps)
    if challenge_mode == protocol.MODE_EXTERNAL_SEED:
        if external_nonce is None or len(external_nonce) != 32:
            raise ValueError("external_seed mode requires a 32-byte external_nonce")
        return protocol.external_seed(prompt_bytes, resp_bytes, c_pre, external_nonce)
    if challenge_mode == protocol.MODE_NI:
        return protocol.ni_seed(prompt_bytes, resp_bytes, c_pre)
    raise ValueError(f"unknown challenge mode: {challenge_mode!r}")


def precommit_request(setup_cert_bytes, trace_payload_root, response_tokens, policy_bytes=b""):
    """32-byte pre-challenge commitment C_pre binding setup, full activation payload digest, response and policy."""
    if type(trace_payload_root) is not bytes or len(trace_payload_root) != 32:
        raise ValueError("trace_payload_root must be the 32-byte Merkle root from commit_trace_payload()")
    resps = _exact_u32_tokens(response_tokens, "response_tokens")
    cert = SetupCertificate.decode(setup_cert_bytes)
    h = hashlib.blake2b(digest_size=32, person=_CPRE_PERSON)
    h.update(bytes.fromhex(cert.id))
    h.update(trace_payload_root)
    h.update(struct.pack("<Q", len(resps)))
    for t in resps:
        h.update(struct.pack("<I", t))
    h.update(struct.pack("<Q", len(policy_bytes)))
    h.update(policy_bytes)
    return h.digest()


@dataclass(frozen=True)
class CapturedProofResult:
    receipt: bytes
    c_pre: bytes
    trace_payload_root: bytes
    seed: bytes
    challenge_mode: str
    setup_cert_id: str
    circuit_root: bytes
    circuit_digest: bytes
    draws: dict
    extracted_count: int
    proven_blocks: int
    trace_row_openings: bytes = b""     # TraceRowOpenings.encode() of every drawn position (verify_drawn_trace_rows)


def extract_samples_from_trace(store, trace, request_id, *, token_ids, draws):
    """Return full captured ports for each nonempty compute draw.

    The supplied token IDs cover the mounted prefix and every forwarded row.
    Capture page keys authenticate the token chain only. Tensor payloads lack
    capture-time authentication in this format.
    """
    handle = store.handle(request_id)
    if (
        handle is None or handle.state != handle.FINISHED or handle.finish() != trace
        or trace.request_id != request_id
    ):
        raise ValueError("expected a matching settled full-width trace")

    positions = {}
    for stratum in (protocol.STRATUM_FRESH_INPUT, protocol.STRATUM_GENERATION):
        for draw in draws[stratum]:
            if not draw.empty:
                if not 0 <= draw.position < trace.n_tokens or not 0 <= draw.layer < store.layout.dims.n_layers:
                    raise ValueError("sample outside the captured trace")
                positions.setdefault(draw.position, []).append(draw)

    offset = 0
    result = {}
    for _, count, rows in handle.iter_verified_pages(token_ids):
        page_draws = (
            draw for position in range(offset, offset + count) for draw in positions.get(position, ())
        )
        for draw in page_draws:
            start = (draw.position - offset) * store.row_bytes
            sample = {}
            for slot in store.layout.slots:
                if slot.kind not in _PORTS or (slot.kind != "h_L" and slot.layer != draw.layer):
                    continue
                byte_offset = start + 2 * slot.offset
                if slot.kind == "lse":
                    values = struct.unpack_from(f"<{slot.width // 2}f", rows, byte_offset)
                else:
                    words = struct.unpack_from(f"<{slot.width}H", rows, byte_offset)
                    values = tuple(struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words)
                sample[slot.kind] = values
            result[draw] = sample
        offset += count
    return result


def prove_captured_request(
    store,
    trace,
    request_id,
    *,
    token_ids,
    response_tokens,
    cache_read_input_tokens,
    setup_cert_bytes,
    sampler_profile,
    weights_provider,
    challenge_mode=protocol.MODE_EXTERNAL_SEED,
    external_nonce=None,
    policy_bytes=b"",
    opcode="dense_gqa",
    model_name="captured-model",
    arch="dense_gqa_qknorm",
    draws_per_pool=3,
    rng=None,
):
    """End-to-end bridge: settled PageStore trace -> payload C_pre -> Protocol v2 sampler -> Relation-4 FPN2 receipt.

    Scope (explicit trust boundaries):
    * Commits ALL full-page and partial-tail captured activation rows into the Merkle `trace_payload_root` and
      `C_pre` BEFORE deriving the sampling challenge seed (prevents post-challenge activation swapping), and
      returns inclusion proofs of every drawn row (`trace_row_openings`, checked by `verify_drawn_trace_rows`).
    * In `fpn2-folded-air-v1` (K = 512), proves 1 reduced sample (`proven_blocks = 1`) out of
      `extracted_count` draws; multi-block / full-width proving requires `fpn2-folded-air-n20-v1`
      and Relation 5 (`FOLDED_VEC_MATMUL`).
    * `SetupCertificate` publisher authenticity and binding `weights_provider` to `weight_commitment_root`
      remain caller/setup-protocol responsibilities outside this single-block helper.
    """
    cert = SetupCertificate.decode(setup_cert_bytes)
    tree = TC.commit_trace_merkle(store, trace, request_id, token_ids)
    trace_payload_root = tree.root
    c_pre = precommit_request(setup_cert_bytes, trace_payload_root, response_tokens, policy_bytes)
    seed = request_seed(c_pre, token_ids, response_tokens, challenge_mode, external_nonce)

    input_tokens = len(token_ids) - len(response_tokens) + 1
    draws = protocol.sample_request(
        seed,
        sampler_profile,
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        output_tokens=len(response_tokens),
        draws=draws_per_pool,
    )
    samples = extract_samples_from_trace(store, trace, request_id, token_ids=token_ids, draws=draws)
    openings = TC.open_trace_rows(tree, drawn_positions(draws)).encode()
    if not samples:
        raise ValueError("no nonempty compute draw extracted from captured trace")

    first_draw, ports = next(iter(samples.items()))
    h_in = ports["h_in"]
    if len(h_in) < 2:
        raise ValueError("captured h_in width must be >= 2")
    w_info = weights_provider(first_draw, ports)
    q_vals = ports.get("q", h_in)
    o_vals = ports.get("o", h_in)
    slot_vals = w_info.get("slot") or {
        "normalize": 1,
        "query": float(q_vals[0]),
        "key0": float(h_in[0]),
        "key1": float(h_in[1]),
        "value0": float(o_vals[0]),
        "value1": float(o_vals[min(1, len(o_vals) - 1)]),
    }
    e2e_samples = E2E.Samples(
        arch=arch,
        model=model_name,
        norm_input=(E2E.bf16_word(h_in[0]), E2E.bf16_word(h_in[1])),
        norm_gain=tuple(w_info["norm_gain"]),
        gemm_x=E2E.bf16_word(h_in[0]),
        gemm_w=int(w_info["gemm_w"]),
        gemm_observed=int(w_info["gemm_observed"]),
        opcode=opcode,
        slot=slot_vals,
    )
    proof = E2E.prove_samples(e2e_samples, rng=rng, server_nonce=c_pre)
    if proof.circuit.root != cert.circuit_root:
        raise ValueError("setup certificate circuit_root does not match the lowered Relation-4 circuit")
    return CapturedProofResult(
        receipt=proof.container,
        c_pre=c_pre,
        trace_payload_root=trace_payload_root,
        seed=seed,
        challenge_mode=challenge_mode,
        setup_cert_id=cert.id,
        circuit_root=proof.circuit.root,
        circuit_digest=proof.circuit.digest,
        draws=draws,
        extracted_count=len(samples),
        proven_blocks=1,
        trace_row_openings=openings,
    )


def verify_drawn_trace_rows(
    setup_cert_bytes,
    trace_payload_root,
    openings,
    *,
    token_ids,
    response_tokens,
    cache_read_input_tokens,
    sampler_profile,
    layout_fingerprint,
    row_bytes,
    challenge_mode=protocol.MODE_EXTERNAL_SEED,
    external_nonce=None,
    policy_bytes=b"",
    draws_per_pool=3,
    expected_c_pre,
):
    """Offline check of the activation rows a reply was proven on, from PUBLIC data plus the opening bundle.

    `expected_c_pre` is MANDATORY: the reply's C_pre as published BEFORE the challenge nonce existed.  Recomputes
    C_pre = precommit_request(setup, trace_payload_root, response, policy) and requires equality (a root rebuilt
    after the challenge over a tampered payload changes C_pre -> E_CPRE_MISMATCH), re-derives the Protocol v2 seed and draws independently, and requires the bundle to
    open EXACTLY the drawn positions against `trace_payload_root` (n_tokens = len(token_ids)).  Returns the
    `verify_trace_row_openings` verdict extended with c_pre / seed / draws; `rows` holds the authenticated rows."""
    if type(expected_c_pre) is not bytes or len(expected_c_pre) != 32:
        return {"verdict": "E_CPRE_REQUIRED", "detail": "the reply's pre-challenge C_pre is required", "rows": {}}
    c_pre = precommit_request(setup_cert_bytes, trace_payload_root, response_tokens, policy_bytes)
    if expected_c_pre != c_pre:
        return {"verdict": "E_CPRE_MISMATCH", "detail": "C_pre does not recompute from the trace root", "rows": {}}
    seed = request_seed(c_pre, token_ids, response_tokens, challenge_mode, external_nonce)
    draws = protocol.sample_request(
        seed,
        sampler_profile,
        input_tokens=len(token_ids) - len(response_tokens) + 1,
        cache_read_input_tokens=cache_read_input_tokens,
        output_tokens=len(response_tokens),
        draws=draws_per_pool,
    )
    out = TC.verify_trace_row_openings(trace_payload_root, len(token_ids), layout_fingerprint, openings,
                                       row_bytes=row_bytes, expected_positions=drawn_positions(draws))
    out.update(c_pre=c_pre, seed=seed, draws=draws)
    return out


def verify_reply_with_setup(
    setup_cert_bytes,
    receipt,
    *,
    expected_c_pre,
    circuit_digest=None,
    rust_binary=None,
    check_python=True,
):
    """Verify a fixed-size Relation-4 FPN2 reply receipt pinned to a SetupCertificate and mandatory C_pre."""
    if type(expected_c_pre) is not bytes or len(expected_c_pre) != 32:
        return {"verified": False, "error": "E_CPRE_REQUIRED", "rust": None, "python": None}
    cert = SetupCertificate.decode(setup_cert_bytes)
    expected_ns = (
        hashlib.blake2b(LW.AIR_DOMAIN + bytes(expected_c_pre) + bytes(circuit_digest), digest_size=32).digest()
        if circuit_digest is not None
        else bytes(expected_c_pre)
    )
    if bytes(receipt[LW.AIR_STMT_NS]) != expected_ns:
        return {"verified": False, "error": "E_CPRE_MISMATCH", "rust": None, "python": None}
    rs_verdict = None
    py_verdict = None
    if rust_binary is not None:
        rs_line, _, _ = E2E.rust_verify(rust_binary, receipt, cert.circuit_root)
        rs_verdict = E2E.verdict_code(rs_line)
    if check_python:
        py_line, _, _ = E2E.python_verify_detached(receipt, cert.circuit_root)
        py_verdict = E2E.verdict_code(py_line)
    ok = (rs_verdict in (None, "VERIFIED")) and (py_verdict in (None, "VERIFIED")) and (
        rs_verdict == "VERIFIED" or py_verdict == "VERIFIED"
    )
    return {"verified": ok, "rust": rs_verdict, "python": py_verdict, "setup_cert_id": cert.id}
