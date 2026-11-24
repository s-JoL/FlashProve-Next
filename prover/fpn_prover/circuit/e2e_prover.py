# SPDX-License-Identifier: Apache-2.0
"""end-to-end relation-4 (`FOLDED_GKR_AIR`) prover driven by real checkpoint samples.

READ THIS FIRST -- exact scope (no overclaiming)
================================================
ONE `fpn2-folded-air-v1` container (n_vars = 14, K = 512 Plonkish rows / 512 variables, C_p = 262,144 B)
proves ONE fixed-shape `Program` (`build_program`) that is IDENTICAL for every architecture:

  * ``norm``  -- `rmsnorm_block` of width 2 (two sampled hidden channels of a real RMSNorm input);
  * ``gemm``  -- `gemm_tile_row_block` 1 x 1 (one real activation element times one real weight element);
  * ``slot``  -- the shared-column 5-opcode microprogram slot (tokens = 2, experts = 2, top_k = 2,
    window = 1, sink = 1, conv_width = 1, extent = 1) with the model's real opcode selected
    (`dense_gqa` with / without QK-norm, `gdn_lrec` for Gated DeltaNet).

Because the shape is identical the public circuit root `root_circ` is identical across architectures, so a
verifier pinning ONE root accepts proofs for all of them and cannot tell which opcode was selected from the
circuit.  The container length is 262,144 B BY CONSTRUCTION (fixed `c_p` zero padding); the substantive
architecture-hiding fact is the shared root.

What the verifiers attest for a real witness: every Plonkish row (Program equalities incl. the gemm centre
dot product / box difference, rmsnorm span, slot booleanity / one-hot / unused-lane guards, limb
recomposition and tightening), every copy constraint and every limb range.  `LookupSpec` reference
predicates (RMSNorm rsqrt enclosure, BF16 decoding, gemm radius / box tolerance, the five branch relations)
are checked ONLY by the prover-side reference interpreter (`Program.failures` inside `lower.assign_air`).

Provenance of every wire is reported (`Provenance`): ``extracted`` = a real checkpoint weight or a real
forward-pass activation (BF16 word or fixed-point 1/256 quantisation of it); ``recomputed`` = derived from
extracted values by the block's reference arithmetic (width-2 norm outputs and rsqrt endpoints, the gemm
centre / radius / box delta, slot outputs).  With extent = 1 the dense branch output is trivially v0.

This module is torch-free: the real-model bench (`bench/tier2/bench_e2e_real_model.py`) feeds it hooked
tensors converted to plain numbers, the unit tests feed it synthetic samples, so the same mapping runs in both.
"""
import hashlib
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction

from ..backend import folded_air as FA
from ..backend.fpn2 import FOLDED_AIR_N20_V1, FOLDED_AIR_V1, merkle_path
from ..ir.arith import bf16_decode, norm_rsqrt
from ..ir.format import get_contract
from ..ir.gadget_ir import Program
from . import lower as LW
from .blocks import gemm_tile_row_block, rmsnorm_block
from .microprogram import _exp_cell, install_microprogram

CONTRACT = "bf16_fp32acc"          # root-neutral vs mxfp4_e2m1 (gamma / eta enter LookupSpec closures only)
SLOT_SHAPE = dict(tokens=2, experts=2, top_k=2, window=1, sink=1, conv_width=1, extent=1)
SCALE = 256                        # microprogram fixed-point scale (lanes are round(256 * x))
OPCODES = ("dense_gqa", "swa_sink", "moe_router_expert", "mla_absorbed", "gdn_lrec")
DEFAULT_EPS = Fraction(1, 10**6)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ----------------------------------------------------------------------------- BF16 helpers
def bf16_word(x):
    """Round-to-nearest-even BF16 word (16-bit) of a Python float."""
    word = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


def bf16_int(word):
    """(signed significand, power) with value = significand * 2^power, exact."""
    b = bf16_decode(word)
    return (-1) ** b.sign * b.significand, b.power


def fixed(x, scale=SCALE):
    return int(round(float(x) * scale))


# ----------------------------------------------------------------------------- program
def build_program(epsilon=DEFAULT_EPS):
    """The canonical shared-shape Program.  Returns (program, names) with names[block] = wire names."""
    p = Program()
    norm = rmsnorm_block(p, "norm", 2, epsilon=epsilon)
    gemm = gemm_tile_row_block(p, "gemm", 1, 1, get_contract(CONTRACT), operand_bits=38)
    slot = install_microprogram(p, "slot", **SLOT_SHAPE)
    return p, {"norm": norm, "gemm": gemm, "slot": slot}


def build_program_multi(blocks, epsilon=DEFAULT_EPS):
    """`blocks` disjoint copies of the shared block set (names prefixed ``b<i>.``) -- the multi-block
    circuit for the production-scale `fpn2-folded-air-n20-v1` profile.  Returns (program, [names per block])."""
    if blocks < 1:
        raise ValueError("need at least one block")
    p = Program()
    out = []
    for b in range(blocks):
        norm = rmsnorm_block(p, f"b{b}.norm", 2, epsilon=epsilon)
        gemm = gemm_tile_row_block(p, f"b{b}.gemm", 1, 1, get_contract(CONTRACT), operand_bits=38)
        slot = install_microprogram(p, f"b{b}.slot", **SLOT_SHAPE)
        out.append({"norm": norm, "gemm": gemm, "slot": slot})
    return p, out


# ----------------------------------------------------------------------------- samples
@dataclass
class Samples:
    """Plain-number samples of one real (or synthetic) forward pass."""
    arch: str                       # e.g. "dense_gqa_qknorm", "dense_gqa", "gdn_hybrid"
    model: str
    norm_input: tuple               # 2 BF16 words (real RMSNorm input channels)
    norm_gain: tuple                # 2 BF16 words (effective gain, real checkpoint)
    gemm_x: int                     # BF16 word of the real activation feeding the projection
    gemm_w: int                     # BF16 word of the real weight element
    gemm_observed: int              # BF16 word of the device product x * w (GPU bf16 multiply)
    opcode: str                     # OPCODES entry
    slot: dict                      # real floats for the selected branch (see slot_lanes)
    epsilon: Fraction = DEFAULT_EPS
    meta: dict = field(default_factory=dict)


@dataclass
class Provenance:
    extracted: dict = field(default_factory=dict)
    recomputed: dict = field(default_factory=dict)

    def as_dict(self):
        return {"extracted": self.extracted, "recomputed": self.recomputed}


def norm_witness(names, s, prov):
    x = tuple(bf16_decode(w).value for w in s.norm_input)
    g = tuple(bf16_decode(w).value for w in s.norm_gain)
    squares = sum(v * v for v in x)
    ref = norm_rsqrt(squares / 2 + s.epsilon)
    rs = 1.0 / math.sqrt(float(squares / 2 + s.epsilon))
    y = tuple(bf16_word(float(a) * float(b) * rs) for a, b in zip(x, g, strict=True))
    vals = (*s.norm_input, *s.norm_gain, *y, ref.k, ref.low, ref.high, ref.high - ref.low)
    for n, v in zip(names[:4], vals[:4], strict=True):
        prov.extracted[n] = v
    for n, v in zip(names[4:], vals[4:], strict=True):
        prov.recomputed[n] = v
    return dict(zip(names, vals, strict=True))


def gemm_witness(names, s, prov):
    """Integer-unit 1 x 1 tile: x = sx * 2^ex, w = sw * 2^ew, all products in units 2^(ex + ew)."""
    gamma = get_contract(CONTRACT).dot_relative_bound(1)
    sx, ex = bf16_int(s.gemm_x)
    sw, ew = bf16_int(s.gemm_w)
    so, eo = bf16_int(s.gemm_observed)
    shift = eo - (ex + ew)
    if shift < 0:
        raise ValueError("observed product finer than the operand unit")
    observed = so << shift
    center = sx * sw
    delta = observed - center
    wr = abs(sw)
    # The activation is an exact BF16 value: input radius 0.  The device product's rounding must then fit the
    # contract's relative box gamma |x| |w| (+ eta); `Program.failures` (gemm.box.0) is the arbiter, so a
    # device product that disagrees with x * w beyond BF16 rounding is REJECTED, not absorbed.
    rx = 0
    z = gamma.numerator * abs(sx) + (gamma.denominator + gamma.numerator) * rx
    vals = (sx, rx, z, sw, wr, center, z * wr, observed, delta)
    kinds = ("x", "rx", "z", "w", "wr", "center", "radius", "observed", "delta")
    out = dict(zip(names, vals, strict=True))
    for n, k in zip(names, kinds, strict=True):
        (prov.extracted if k in ("x", "w", "observed") else prov.recomputed)[n] = out[n]
    prov.recomputed["gemm.unit_log2"] = ex + ew
    return out


def _nearest(v):
    return round(v)


def slot_lanes(opcode, sl):
    """Fixed-point lanes for the selected branch from real floats; returns (lanes, extracted_count)."""
    if opcode == "dense_gqa":
        q, k0, k1, v0, v1 = (fixed(sl[k]) for k in ("query", "key0", "key1", "value0", "value1"))
        # extent = 1: one score, softmax weight exp(0) = SCALE, output = v0 exactly.
        return (int(sl["normalize"]), q, k0, k1, v0, v1, v0), 6
    if opcode == "gdn_lrec":
        decay, beta = fixed(sl["decay"]), fixed(sl["beta"])
        key, kernel, history = fixed(sl["key"]), fixed(sl["kernel"]), fixed(sl["history"])
        previous = fixed(sl.get("previous", 0.0))
        conv = _nearest(Fraction(kernel * history, SCALE))
        value = fixed(sl["value_post"]) - conv          # value + convolution == real post-conv value v_t
        decayed = Fraction(decay * previous, SCALE)
        correction = Fraction(beta * key, SCALE ** 2) * (value + conv - key * decayed / SCALE)
        state = _nearest(decayed + correction)
        return (0, 1, previous, decay, beta, key, value, kernel, history, conv, state), 7
    if opcode == "swa_sink":
        # position 0 (< extent = 1), window = 1, sink = 1: the causal SWA+sink mask keeps token 0 only, so the
        # single softmax weight is exp(0) and the output lane is v0 by construction (as for dense_gqa).
        q, k0, k1, v0, v1 = (fixed(sl[k]) for k in ("query", "key0", "key1", "value0", "value1"))
        return (0, q, k0, k1, v0, v1, v0), (1, 2, 3, 4, 5)
    if opcode == "moe_router_expert":
        return moe_lanes(sl["scores"], sl["outputs"]), (0, 1, 8, 9)
    raise ValueError(f"opcode {opcode} has no real-sample mapping")


def moe_lanes(scores, outputs, scale=SCALE):
    """15 lanes of the 2-expert / top-2 MoE branch from real router logits and real expert outputs.

    Layout (`microprogram._lift`, experts = top_k = 2): score[2] | active[2] | argmax[2] | weight[2] | output[2]
    | index[2] | maximum | denominator | result.  weight_i = exp-cell of (score_i - max) at 1/256 (None -> the
    sample is rejected and the caller picks another one); result = nearest(sum w_i out_i / sum w_i), i.e. the
    softmax-renormalised mixture over the two selected experts."""
    s = tuple(fixed(x, scale) for x in scores)
    o = tuple(fixed(x, scale) for x in outputs)
    if len(s) != 2 or len(o) != 2:
        raise ValueError("the shared slot has exactly 2 experts")
    mx = max(s)
    best = s.index(mx)
    w = tuple(_exp_cell(v - mx, scale) for v in s)
    if None in w or min(w) < 1:
        raise ValueError(f"router logits {scores} have no exact 1/{scale} exp cell (or a zero weight)")
    den = sum(w)
    result = _nearest(Fraction(w[0] * o[0] + w[1] * o[1], den))
    return (*s, 1, 1, int(best == 0), int(best == 1), *w, *o, 0, 1, mx, den, result)


def _block_witness(s, names, prov):
    """Base witness (before `Program.complete`) of one (norm, gemm, slot) block for sample set `s`."""
    w = {}
    w.update(norm_witness(names["norm"], s, prov))
    w.update(gemm_witness(names["gemm"], s, prov))
    slot = names["slot"]
    op = OPCODES.index(s.opcode)
    lanes, ext = slot_lanes(s.opcode, s.slot)
    ext = set(range(ext)) if isinstance(ext, int) else set(ext)
    lanes = lanes + (0,) * (len(slot.lanes) - len(lanes))
    w.update(zip(slot.selectors, (int(i == op) for i in range(len(OPCODES))), strict=True))
    w.update(zip(slot.lanes, lanes, strict=True))
    for i, (n, v) in enumerate(zip(slot.lanes, lanes, strict=True)):
        (prov.extracted if i in ext else prov.recomputed)[n] = v
    for n in slot.selectors:
        prov.recomputed[n] = w[n]
    return w


def build_witness(s, program=None, names=None):
    """(program, witness, provenance) for one sample set; raises LoweringError if a predicate fails."""
    if program is None:
        program, names = build_program(s.epsilon)
    prov = Provenance()
    w = _block_witness(s, names, prov)
    w = program.complete(w)
    for k, v in w.items():
        if k not in prov.extracted and k not in prov.recomputed:
            prov.recomputed[k] = v
    bad = program.failures(w)
    if bad:
        raise LW.LoweringError(bad)
    return program, w, prov


# ----------------------------------------------------------------------------- prove / verify
@dataclass
class E2EProof:
    circuit: object
    assignment: object
    container: bytes
    provenance: Provenance
    timings_ms: dict
    record: dict


def _prove(program, witness, profile, rng, record, server_nonce, engine):
    t1 = time.perf_counter()
    circuit = LW.compile_air(program, profile, engine)          # structural cache (lower.air_program_fingerprint)
    t_c = time.perf_counter()
    assignment = LW.assign_air(circuit, program, witness)
    t2 = time.perf_counter()
    rec, phases = {}, {}
    hooks = {"record": rec} if record else {}
    if engine is not None:
        hooks["timings"] = phases
    data = LW.prove_air(circuit, assignment, server_nonce=server_nonce, rng=rng, hooks=hooks or None,
                        engine=engine)
    t3 = time.perf_counter()
    timings = {"lower_air": (t2 - t1) * 1e3, "lower_air_compile": (t_c - t1) * 1e3,
               "lower_air_assign": (t2 - t_c) * 1e3, "folded_air_prove": (t3 - t2) * 1e3}
    if phases:
        timings["folded_air_prove_phases"] = phases
    return circuit, assignment, data, timings, rec


def prove_samples(s, rng=None, record=False, server_nonce=b"\x00" * 32, engine=None):
    """One `fpn2-folded-air-v1` container for one sample set (`engine`: optional kernel executor, same bytes)."""
    t0 = time.perf_counter()
    program, witness, prov = build_witness(s)
    t1 = time.perf_counter()
    circuit, assignment, data, timings, rec = _prove(program, witness, FOLDED_AIR_V1, rng, record, server_nonce,
                                                     engine)
    return E2EProof(circuit, assignment, data, prov, {"witness_build": (t1 - t0) * 1e3, **timings}, rec)


def build_witness_multi(samples, program=None, names=None):
    """(program, witness, provenance) of the multi-block circuit, block i <- samples[i] (one shared epsilon)."""
    eps = {s.epsilon for s in samples}
    if len(eps) != 1:
        raise ValueError("all blocks of one multi-block program share one RMSNorm epsilon")
    if program is None:
        program, names = build_program_multi(len(samples), samples[0].epsilon)
    if len(names) != len(samples):
        raise ValueError("one sample set per block")
    prov = Provenance()
    w = {}
    for s, nm in zip(samples, names, strict=True):
        w.update(_block_witness(s, nm, prov))
    w = program.complete(w)
    for k, v in w.items():
        if k not in prov.extracted and k not in prov.recomputed:
            prov.recomputed[k] = v
    bad = program.failures(w)
    if bad:
        raise LW.LoweringError(bad)
    return program, w, prov


def prove_multi(samples, profile=FOLDED_AIR_N20_V1, rng=None, record=False, server_nonce=b"\x00" * 32,
                engine=None):
    """ONE container (default `fpn2-folded-air-n20-v1`, K = 32,768) proving len(samples) blocks at once."""
    t0 = time.perf_counter()
    program, witness, prov = build_witness_multi(samples)
    t1 = time.perf_counter()
    circuit, assignment, data, timings, rec = _prove(program, witness, profile, rng, record, server_nonce, engine)
    return E2EProof(circuit, assignment, data, prov, {"witness_build": (t1 - t0) * 1e3, **timings}, rec)


def _cpu_children():
    import resource
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _run_cli(cmd, data, env=None, cwd=None):
    with tempfile.NamedTemporaryFile(suffix=".fpn2", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        c0, w0 = _cpu_children(), time.perf_counter()
        r = subprocess.run(cmd + [path], capture_output=True, text=True, env=env, cwd=cwd)
        w1, c1 = time.perf_counter(), _cpu_children()
        return r.stdout.strip(), (w1 - w0) * 1e3, (c1 - c0) * 1e3
    finally:
        os.unlink(path)


def rust_verify(binary, data, root):
    """(verdict line, wall ms, child CPU ms) of `fpn-verify <file> --circuit-root <hex>`."""
    out, wall, cpu = _run_cli([binary, "--circuit-root", root.hex()], data)
    return out, wall, cpu


def python_verify_detached(data, root, python=None):
    """Detached Python reference verifier in an isolated interpreter (-I) with ONLY verifier/python on sys.path."""
    vdir = os.path.join(ROOT, "verifier", "python")
    boot = (
        "import runpy, sys; "
        f"sys.path.insert(0, {vdir!r}); "
        "sys.argv = ['fpn_verify_ref', *sys.argv[1:]]; "
        "runpy.run_module('fpn_verify_ref', run_name='__main__')"
    )
    out, wall, cpu = _run_cli([python or sys.executable, "-I", "-c", boot,
                               "--circuit-root", root.hex()], data, cwd=vdir)
    return out, wall, cpu


def verdict_code(line):
    return "VERIFIED" if line == "VERIFIED" else line.removeprefix("REJECTED: ")


# ----------------------------------------------------------------------------- byte sections (M2-3)
def section_map(layout, c_p=None):
    """[(name, start, end, public)] covering [0, c_p) exactly; public = independent of the private witness.

    header (64) | STMT root_0 (32, private) root_circ (32, public) N_s (32, public) | GKR + zerocheck +
    z_blind + PCS sumcheck / fold roots (private) | Q queries x [F opening (private), C opening (public:
    the pinned circuit codeword + its Merkle siblings), folded-layer openings (private, H = F + theta C)]
    | zero padding (public).  c_p defaults to the registered profile of `layout.n` (14 or 20)."""
    if c_p is None:
        c_p = {FOLDED_AIR_V1["n_vars"]: FOLDED_AIR_V1["c_p"], FOLDED_AIR_N20_V1["n_vars"]: FOLDED_AIR_N20_V1["c_p"]}[
            layout.n]
    secs = [("header", 0, 64, True), ("stmt.root_0", 64, 96, False), ("stmt.root_circ", 96, 128, True),
            ("stmt.N_s", 128, 160, True)]
    q_start = 64 + layout.body_used - layout.queries * layout.query_len
    secs.append(("gkr_zc_pcs_rounds", 160, q_start, False))
    pos = q_start
    open0 = 96 + 32 * layout.depth(0)
    for q in range(layout.queries):
        secs.append((f"q{q}.F", pos, pos + open0, False))
        secs.append((f"q{q}.C", pos + open0, pos + 2 * open0, True))
        pos += 2 * open0
        for t in range(1, layout.folds):
            ln = 96 + 32 * layout.depth(t)
            secs.append((f"q{q}.L{t}", pos, pos + ln, False))
            pos += ln
    if pos != 64 + layout.body_used:
        raise AssertionError("section map does not sum to body_used")
    secs.append(("padding", pos, c_p, True))
    return secs


def check_public_c_openings(circuit, data, positions, engine=None):
    """Assert every query's C opening equals the deterministic circuit codeword + Merkle path."""
    from ..backend.fpn2 import enc3s
    lay = circuit.layout
    comC = FA.commit_circuit(lay, circuit.table, engine)
    quarter = len(comC.code) // 4
    secs = {name: (a, b) for name, a, b, _ in section_map(lay, circuit.profile["c_p"])}
    for q, p in enumerate(positions):
        a, b = secs[f"q{q}.C"]
        if engine is not None:
            want = enc3s([comC.code_at(p + k * quarter) for k in range(4)]) + b"".join(comC.path(p))
        else:
            want = enc3s([comC.code[p + k * quarter] for k in range(4)]) + b"".join(merkle_path(comC.levels, p))
        if bytes(data[a:b]) != want:
            raise AssertionError(f"query {q}: C opening is not the public circuit opening")
    return True


def shared_blocks(a, b, layout, block=64):
    """Re-proof leakage report over 64-byte blocks, split public / private by `section_map`."""
    secs = section_map(layout)

    def private(off):
        return all(not pub for _, s, e, pub in secs if s < off + block and off < e)

    def public(off):
        return all(pub for _, s, e, pub in secs if s < off + block and off < e)

    offs = range(0, len(a), block)
    aligned_priv = [o for o in offs if private(o) and a[o:o + block] == b[o:o + block]]
    aligned_pub = [o for o in offs if public(o) and a[o:o + block] == b[o:o + block]]
    # Unaligned set intersection over private bytes only (64-byte windows at every 8-byte offset).
    priv_ranges = [(s, e) for _, s, e, pub in secs if not pub]

    def windows(x):
        out = set()
        for s, e in priv_ranges:
            for o in range(s, e - block + 1, 8):
                out.add(bytes(x[o:o + block]))
        return out

    inter = windows(a) & windows(b)
    priv_bytes = sum(e - s for s, e in priv_ranges)
    return {"block_bytes": block, "private_bytes": priv_bytes,
            "public_bytes": len(a) - priv_bytes,
            "private_aligned_blocks": sum(1 for o in offs if private(o)),
            "private_aligned_shared": len(aligned_priv),
            "private_window_set_intersection": len(inter),
            "public_aligned_blocks": sum(1 for o in offs if public(o)),
            "public_aligned_shared": len(aligned_pub),
            "total_aligned_shared_incl_public": sum(1 for o in offs if a[o:o + block] == b[o:o + block])}


# ----------------------------------------------------------------------------- ZK rank + cheats
def joint_rank(proof, claims_only=False):
    c = proof.circuit
    rows = list(FA._claim_rows(c.layout, proof.record)) if claims_only else None
    rep = FA.require_joint_mask_air(c.layout, c.table, proof.record, c.mask_variables, rows=rows,
                                    raise_on_fail=False)
    rep["claims_only"] = claims_only
    return rep


def cheat_gemm_weight(proof, delta=1, rng=None):
    """Perturb the real gemm weight by `delta` WITHOUT recomputing the centre product (bypasses assign_air):
    every limb stays in range, the `gemm.center_dot.0` Plonkish row fails -> E_ZEROCHECK."""
    c = proof.circuit
    base = proof.assignment.values
    w = next(n for n in base if n == "gemm.weight.0.0")
    witness = {name: (base[name] if base[name] < LW.P // 2 else base[name] - LW.P) for name, *_ in c.columns}
    witness[w] += delta
    values = LW.air_values(c, witness)
    fails = LW.air_row_failures(c, values)
    in_range = all(0 <= values[n] < 128 for pl in c.limb_plans for n in pl.names())
    data = LW.prove_air_unchecked(c, values, rng=rng)
    return data, {"perturbed": w, "delta": delta, "row_failures": fails, "limbs_in_range": in_range}


def cheat_multiplicity(proof, rng=None):
    """Shift one range multiplicity (total unchanged) on the real witness: only the GKR sees it -> E_GKR_ROOT."""
    c = proof.circuit

    def wrong_m(layout, table, V, r):
        m = FA.range_multiplicities(table, V) + [0] * (layout.K - FA.TABLE)
        t = next(i for i, x in enumerate(m) if x)
        m[t] -= 1
        m[(t + 1) % FA.TABLE] += 1
        return FA.build_witness(layout, table, V, r, m=m)

    data = LW.prove_air_unchecked(c, dict(proof.assignment.values), rng=rng, hooks={"witness": wrong_m})
    return data, {"perturbed": "range multiplicity m_t -= 1, m_(t+1) += 1"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()
