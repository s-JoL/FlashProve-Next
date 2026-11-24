# SPDX-License-Identifier: Apache-2.0
"""Lowering of a gadget-IR `Program` + witness into `FPN2` containers: relation 4 (primary) and relation 3.

READ THIS FIRST -- what the emitted proofs do and do NOT attest
===============================================================
Relation 4 (`FOLDED_GKR_AIR`, `compile_air` / `assign_air` / `prove_air`, specs/11-wire-format.md §5.5)
-----------------------------------------------------------------------------------------------------
ONE container per Program.  The public circuit -- Plonkish rows q_M a b + q_L a + q_R b + q_O c + q_C = 0,
the wiring (which variable feeds which gate cell) and the range flags -- is committed under `root_circ`,
which the verifier's caller MUST pin (`fpn-verify --circuit-root`, `expected_circuit_root`); the verifiers
then enforce, for an arbitrary (possibly adversarial) prover:

  * ``ALGEBRAIC_GATE``  -- VERIFIER-CHECKED.  Every `ConstraintSpec` and every wire recomposition
    u + 2^(B-1) = sum 128^i limb_i / tightening identity is a set of Plonkish rows (accumulator variables
    chain gates with more than one product or more than three terms); the zerocheck checks every row and
    the copy-constraint GKR checks every gate cell equals its variable.  `AIR_GATE_FAMILIES` below lists
    EVERY gate family the blocks emit and the exact condition it enforces ; the lifted conditions are NOT
    re-checked by the remaining predicates.
  * ``LOOKUP_RANGE``    -- VERIFIER-CHECKED.  Every limb / tightening variable is in [0, 128) (relation-4
    range lookup in the same GKR); with the recomposition rows this pins every wire to its signed range.
    Inequalities are range checks: `Program.nonneg` writes value = slack + 2^(B-1) for a B-bit slack wire
    with B a multiple of 7, so the lookup forces 0 <= value < 2^B.
  * ``REFERENCE_PREDICATE`` -- NOT verifier-checked.  What is left inside `LookupSpec` predicates, checked
    only by `Program.failures` at assignment time; a relation-4 proof says nothing about it.
    `AIR_PREDICATE_RESIDUE` below lists EVERY remaining predicate and exactly what it still contains.  In
    short: the table / transcendental parts (`bf16_decode`, `norm_rsqrt`, the `_exp` enclosures) and every
    nearest-rounding division `_nearest(Fraction(.))` built on them, plus three gemm relations that are
    algebraic but do not fit the 2^62 signed lift bound at full operand width (`operand` for wide
    rounding denominators, `products` unless the operand width is narrowed, `box` always).

Auxiliary wires (gated products s * lane, slack cells, one-hot index cells) are ordinary range-checked
STRICT columns of the public circuit; `Program.hints` / `complete_witness` only generate their honest
values (prover side, outside the digest) and are trusted by no verifier.

Mutation coverage of the lowered table (`tests/audit/test_lowered_air_mutation_audit.py`): every variable
perturbed, every row deleted, on the committed `AirTable` itself.

Field semantics: gates hold over F_p; `Program.check_bounds` (|difference| < 2^62 < P) makes that the
integer equality.  `prove_air_unchecked` / `air_values(..., overrides=...)` deliberately bypass every
check (witness-only forgery tests, `tests/unit/test_relation4_air_and_wbforge.py`).

Relation 3 (`FOLDED_GKR_LOOKUP`, `compile_program` / `assign` / `prove`) -- legacy, range checks only
----------------------------------------------------------------------------------------------------
Relation 3 as registered (`fpn2-folded-gkr-v1`, specs/11-wire-format.md §5.4) proves exactly one
statement per container:

    "the prover knows a vector F committed under root_0 whose query block of T = 2^(s-2) cells
     (T = 128 for v1) lies entirely in the public table [0, T)"

via a ZK-masked FracLogUp-GKR whose leaf claims (queries, table multiplicities, masks) are opened by
one fused folded ZK-Basefold opening.  The verifiers (Rust `fpn-verify`, Python `fpn_verify_ref`)
know NOTHING about the IR: they do not see wires, equality constraints, or which committed cell is
which limb of which wire.  Consequently this module partitions every relation of the Program into
exactly one of three buckets, recorded in `LoweredCircuit.manifest` (coverage is checked):

  1. ``LOOKUP_RANGE`` -- every wire's signed bound ``-2^(B-1) <= v < 2^(B-1)``.  The wire is shifted to
     ``u = v + 2^(B-1)`` and split into base-T limbs; a partial top limb (``r = B mod log2 T`` bits) is
     tightened by a second query ``top + (T - 2^r)``.  These limbs ARE the relation-3 queries and the
     table multiplicities m_t are packed into F; the verifiers check limb membership in [0, T)
     (FracLogUp root_num == 0) and nothing else.  The recomposition ``u = sum limb_i T^i`` and the
     tightening identity are linear gates checked ONLY by the prover (bucket 2 semantics).
  2. ``ALGEBRAIC_GATE`` -- every `ConstraintSpec` (incl. microprogram selector booleanity / exhaustive
     one-hot) is flattened into a sparse polynomial gate over Fp and evaluated on the witness mod P.
     `Program.check_bounds` guarantees |difference| < 2^62 < P, so the mod-P check is equivalent to
     the integer equality.  CHECKED BY THE PROVER AT LOWERING TIME ONLY: relation 3 has no
     arithmetic-gate component, so a prover that skips this module's check can emit containers for
     an in-range witness that violates a gate and BOTH verifiers will accept them
     (`tests/unit/test_ir_lowering_and_ntt.py` pins this limitation explicitly).
  3. ``REFERENCE_PREDICATE`` -- every `LookupSpec` (RMSNorm rsqrt enclosure, the GEMM box relation, the
     residue of the five microprogram branch relations, ...; `AIR_PREDICATE_RESIDUE`).  These are opaque exact-rational
     Python predicates; they are evaluated by the reference interpreter (`Program.failures`) and are
     NOT lowered to any table or gate.  Prover-side only.

So: forged witnesses are always rejected *by lowering*; out-of-table limbs and wrong multiplicities
are additionally rejected *by both verifiers*; violations of buckets 2/3 by an in-range witness are
rejected *only* by lowering.  Closing that gap needs a relation with an arithmetic/wiring component
(a new profile + verifier change), which is outside relation 3.

Binding: the public circuit (column specs, gates, limb plan, predicate names/columns, profile id) is
hashed into `LoweredCircuit.digest`; container i of c carries
``N_s = BLAKE2b-256("FPN2-IR-LOWER-v1" || server_nonce || digest || i || c)`` in its STMT, so the
containers are bound (through the relation-3 transcript) to this circuit and their position, and
`check_binding` lets a verifier that recompiles the circuit check it.  This binds *which* circuit the
prover claims; it does not make buckets 2/3 verifier-checked.
"""
import hashlib
import json
import struct
import threading
from dataclasses import dataclass, field, replace

from flashprove_next.field import P, hf

from ..backend import folded_air as FA
from ..backend import zk_lookup as ZL
from ..backend.fpn2 import FOLDED_GKR_V1, Layout, profile_bytes, profile_id
from ..ir.gadget_ir import Expr

LOOKUP_RANGE = "LOOKUP_RANGE"
ALGEBRAIC_GATE = "ALGEBRAIC_GATE"
REFERENCE_PREDICATE = "REFERENCE_PREDICATE"
VERIFIER_CHECKED = {
    LOOKUP_RANGE: "limb membership in [0, T) only (FracLogUp-GKR + folded ZK-Basefold)",
    ALGEBRAIC_GATE: "no (prover-side at lowering time)",
    REFERENCE_PREDICATE: "no (prover-side reference interpreter, not lowered)",
}
DOMAIN = b"FPN2-IR-LOWER-v1"
STMT_NS = slice(64 + 32, 64 + 64)     # header 64 | STMT = root_0 (32) || N_s (32)


class LoweringError(ValueError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__("lowering rejected the witness: " + ", ".join(self.failures))


# ----------------------------------------------------------------------------- algebraic gates
def flatten(expr):
    """Expr -> {monomial (sorted wire names): integer coefficient}, zero terms dropped."""
    if not isinstance(expr, Expr):
        raise TypeError("expected Expr")
    if expr.op == "wire":
        return {(expr.args[0],): 1}
    if expr.op == "const":
        return {(): expr.args[0]} if expr.args[0] else {}
    a, b = (flatten(arg) for arg in expr.args)
    out = {}
    if expr.op in "+-":
        sign = 1 if expr.op == "+" else -1
        for mono, c in a.items():
            out[mono] = out.get(mono, 0) + c
        for mono, c in b.items():
            out[mono] = out.get(mono, 0) + sign * c
    elif expr.op == "*":
        for ma, ca in a.items():
            for mb, cb in b.items():
                mono = tuple(sorted(ma + mb))
                out[mono] = out.get(mono, 0) + ca * cb
    else:
        raise ValueError(f"unknown Expr op {expr.op}")
    return {m: c for m, c in out.items() if c}


@dataclass(frozen=True)
class Gate:
    """sum_k coeff_k * prod(wires_k) == 0 over Fp (coefficients canonical in [0, P))."""
    name: str
    origin: str                 # "constraint" | "recompose" | "tighten"
    terms: tuple                # ((coeff, (wire, ...)), ...) sorted canonically

    @classmethod
    def from_poly(cls, name, origin, poly):
        terms = tuple(sorted(((c % P, mono) for mono, c in poly.items() if c % P), key=lambda t: t[1]))
        return cls(name, origin, terms)

    def evaluate(self, values):
        acc = 0
        for coeff, mono in self.terms:
            term = coeff
            for w in mono:
                term = term * (values[w] % P) % P
            acc = (acc + term) % P
        return acc

    @property
    def degree(self):
        return max((len(m) for _, m in self.terms), default=0)

    def canonical(self):
        return [self.name, self.origin, [[c, list(m)] for c, m in self.terms]]


# ----------------------------------------------------------------------------- lookup (range) plan
def limb_name(wire, i):
    return f"{wire}#limb{i}"


def tight_name(wire):
    return f"{wire}#tight"


@dataclass(frozen=True)
class LimbPlan:
    wire: str
    bits: int                   # signed bound bits B
    limb_bits: int              # log2 T
    limbs: int                  # ceil(B / log2 T)
    top_bits: int               # bits of the top limb (== limb_bits if exact)
    first_query: int            # index of the first query of this wire in the query stream

    @property
    def offset(self):
        return 1 << (self.bits - 1)

    @property
    def tightened(self):
        return self.top_bits < self.limb_bits

    @property
    def query_count(self):
        return self.limbs + (1 if self.tightened else 0)

    def decompose(self, value):
        """Unchecked limb queries of `value` (callers must check the recomposition gate)."""
        u = value + self.offset
        mask = (1 << self.limb_bits) - 1
        limbs = [(u >> (self.limb_bits * i)) & mask for i in range(self.limbs)]
        if self.tightened:
            limbs.append(limbs[-1] + (1 << self.limb_bits) - (1 << self.top_bits))
        return limbs

    def gates(self):
        t = 1 << self.limb_bits
        poly = {(self.wire,): 1, (): self.offset}
        for i in range(self.limbs):
            poly[(limb_name(self.wire, i),)] = -(t ** i)
        out = [Gate.from_poly(f"{self.wire}#recompose", "recompose", poly)]
        if self.tightened:
            out.append(Gate.from_poly(f"{self.wire}#tighten", "tighten", {
                (tight_name(self.wire),): 1, (limb_name(self.wire, self.limbs - 1),): -1,
                (): -(t - (1 << self.top_bits))}))
        return out

    def names(self):
        return [limb_name(self.wire, i) for i in range(self.limbs)] + (
            [tight_name(self.wire)] if self.tightened else [])

    def canonical(self):
        return [self.wire, self.bits, self.limb_bits, self.limbs, self.top_bits, self.first_query]


# ----------------------------------------------------------------------------- compiled circuit
@dataclass(frozen=True)
class LoweredCircuit:
    """Witness-independent (public) lowering of a Program onto a relation-3 profile."""
    profile: dict
    layout: Layout
    columns: tuple              # ((name, bits, semantic_type, producer, freedom), ...) in program order
    gates: tuple                # constraint gates followed by recompose/tighten gates
    limb_plans: tuple
    predicates: tuple           # ((name, columns), ...) reference predicates (not lowered)
    manifest: dict              # relation name -> bucket
    query_count: int
    chunks: int
    digest: bytes = field(repr=False)
    hints: dict = field(default_factory=dict, repr=False, compare=False)   # prover-side aux witness generation

    @property
    def table_size(self):
        return self.layout.quarter

    @property
    def mask_variables(self):
        """Variable indices whose value is uniform F_p randomness independent of the witness: the pad
        variable, the reserved `#zk` blinding variables and every unused index (the rank check's B side)."""
        named = {k for k, name in enumerate(self.variables) if name != PAD and not name.startswith(ZK_PREFIX)}
        return tuple(k for k in range(self.layout.K) if k not in named)

    def summary(self):
        counts = {}
        for bucket in self.manifest.values():
            counts[bucket] = counts.get(bucket, 0) + 1
        return {"profile": self.profile["name"], "table": self.table_size, "queries": self.query_count,
                "padding": self.chunks * self.table_size - self.query_count, "containers": self.chunks,
                "gates": len(self.gates), "max_gate_degree": max((g.degree for g in self.gates), default=0),
                "buckets": counts, "verifier_checked": VERIFIER_CHECKED}


def _canonical_digest(obj):
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.blake2b(data, digest_size=32, person=b"FPN2-IR-LOWER\x00\x00\x00").digest()


def _plan(program, limb_bits):
    """Witness-independent partition of the Program: columns, limb plans, gates, predicates, manifest."""
    program.check_bounds()
    names = [r.name for r in (*program.constraints, *program.lookups)]
    if len(set(names)) != len(names):
        raise ValueError("duplicate relation name")
    for relation in (*program.constraints, *program.lookups):
        if set(relation.columns) - program.columns.keys():
            raise ValueError(f"relation {relation.name} uses an unknown column")
    columns, plans, manifest = [], [], {}
    cursor = 0
    for name, spec in program.columns.items():
        if "#" in name:
            raise ValueError("column names may not contain '#'")
        columns.append((name, spec.signed_bound_bits, spec.semantic_type, spec.producer,
                        spec.freedom_kind.value))
        bits = spec.signed_bound_bits
        limbs = -(-bits // limb_bits)
        plan = LimbPlan(name, bits, limb_bits, limbs, bits - limb_bits * (limbs - 1), cursor)
        plans.append(plan)
        cursor += plan.query_count
        manifest[f"{name}#range"] = LOOKUP_RANGE
    gates = []
    for c in program.constraints:
        gates.append(Gate.from_poly(c.name, "constraint", flatten(c.difference)))
        manifest[c.name] = ALGEBRAIC_GATE
    for plan in plans:
        gates.extend(plan.gates())
    predicates = []
    for lk in program.lookups:
        predicates.append((lk.name, tuple(lk.columns)))
        manifest[lk.name] = REFERENCE_PREDICATE
    if set(manifest) != set(names) | {f"{n}#range" for n in program.columns}:
        raise AssertionError("manifest does not cover every relation exactly once")
    return columns, plans, gates, predicates, manifest, cursor


def compile_program(program, profile=FOLDED_GKR_V1):
    """Public lowering: gates, limb plan, predicate list, bucket manifest and circuit digest."""
    layout = Layout(profile["n_vars"], profile["n_queries"])
    table = layout.quarter
    limb_bits = table.bit_length() - 1
    columns, plans, gates, predicates, manifest, cursor = _plan(program, limb_bits)
    chunks = max(1, -(-cursor // table))
    digest = _canonical_digest({
        "domain": DOMAIN.decode(), "profile_id": profile_id(profile).hex(), "columns": columns,
        "gates": [g.canonical() for g in gates], "limbs": [p.canonical() for p in plans],
        "predicates": [[n, list(c)] for n, c in predicates], "chunks": chunks, "table": table})
    return LoweredCircuit(profile, layout, tuple(columns), tuple(gates), tuple(plans), tuple(predicates),
                          manifest, cursor, chunks, digest, dict(program.hints))


# ----------------------------------------------------------------------------- private assignment
@dataclass(frozen=True)
class Assignment:
    """Private lowering of one witness: limb queries (padded with 0) and per-chunk multiplicities."""
    values: dict                # wire / limb name -> integer
    queries: tuple              # chunks * T queries
    multiplicities: tuple       # per chunk: T counts

    def chunk(self, i, table):
        return list(self.queries[i * table:(i + 1) * table])


def multiplicities_of(queries, table):
    counts = [0] * table
    for q in queries:
        if not 0 <= q < table:
            raise LoweringError([f"limb {q} outside table [0, {table})"])
        counts[q] += 1
    return tuple(counts)


def complete_witness(circuit, witness):
    """Fill absent auxiliary wires (gated products, slack cells, ...) from the circuit's hints, in column
    order.  Prover-side witness generation only; present values are never overwritten."""
    out = dict(witness)
    for name, *_ in circuit.columns:
        if name not in out and name in circuit.hints:
            out[name] = circuit.hints[name](out)
    return out


def limb_queries_unchecked(circuit, witness):
    """Adversarial / audit helper: limb queries WITHOUT any IR check (never use to prove)."""
    witness = complete_witness(circuit, witness)
    out = []
    for plan in circuit.limb_plans:
        out.extend(plan.decompose(witness[plan.wire]))
    return out + [0] * (circuit.chunks * circuit.table_size - len(out))


def assign(circuit, program, witness):
    """Check the witness against every bucket and build the private relation-3 assignment."""
    failures = list(program.failures(witness))       # bounds, shape, constraints, reference predicates
    if failures:
        raise LoweringError(failures)
    witness = complete_witness(circuit, witness)
    values = dict(witness)
    queries = []
    for plan in circuit.limb_plans:
        limbs = plan.decompose(witness[plan.wire])
        values.update(zip(plan.names(), limbs))
        queries.extend(limbs)
    bad = [g.name for g in circuit.gates if g.evaluate(values)]
    if bad:
        raise LoweringError(bad)
    table = circuit.table_size
    if any(not 0 <= q < table for q in queries) or len(queries) != circuit.query_count:
        raise LoweringError(["limb outside table"])
    queries += [0] * (circuit.chunks * table - len(queries))
    mult = tuple(multiplicities_of(queries[i * table:(i + 1) * table], table) for i in range(circuit.chunks))
    return Assignment(values, tuple(queries), mult)


def lower(program, witness, profile=FOLDED_GKR_V1):
    circuit = compile_program(program, profile)
    return circuit, assign(circuit, program, witness)


# ----------------------------------------------------------------------------- proving and binding
def binding_nonce(circuit, index, server_nonce=b"\x00" * 32):
    if len(server_nonce) != 32:
        raise ValueError("server nonce must be 32 bytes")
    return hashlib.blake2b(DOMAIN + bytes(server_nonce) + circuit.digest + index.to_bytes(4, "little")
                           + circuit.chunks.to_bytes(4, "little"), digest_size=32).digest()


def _packed_witness(multiplicities):
    """Relation-3 F builder that packs the lowering's own multiplicities into the m block."""
    def build(layout, queries, rng):
        cells = ZL.build_witness(layout, queries, rng)
        for t, m in enumerate(multiplicities):
            cells[layout.cell(layout.off_m + t)] = hf(m % P)
        return cells
    return build


def prove(circuit, assignment, server_nonce=b"\x00" * 32, rng=None, unchecked=False):
    """Emit one canonical fixed-size FPN2 relation-3 container per chunk of T limb queries.

    Unless `unchecked` (adversarial tests only), the multiplicities must match the queries."""
    table = circuit.table_size
    if len(assignment.queries) != circuit.chunks * table or len(assignment.multiplicities) != circuit.chunks:
        raise LoweringError(["assignment shape"])
    out = []
    for i in range(circuit.chunks):
        queries, mult = assignment.chunk(i, table), assignment.multiplicities[i]
        if not unchecked and tuple(mult) != multiplicities_of(queries, table):
            raise LoweringError([f"chunk {i}: multiplicities do not match the limb multiset"])
        out.append(ZL.prove(queries, n_s=binding_nonce(circuit, i, server_nonce), layout=circuit.layout,
                            rng=rng, profile=circuit.profile, witness=_packed_witness(mult)))
    return tuple(out)


def check_binding(circuit, containers, server_nonce=b"\x00" * 32):
    """Verifier-side: right number of containers, each carrying this circuit's positional N_s."""
    if len(containers) != circuit.chunks:
        return False
    return all(bytes(c[STMT_NS]) == binding_nonce(circuit, i, server_nonce) for i, c in enumerate(containers))


# ============================================================================= relation 4 (FOLDED_GKR_AIR)
# Every Program equality (AIR_GATE_FAMILIES), every wire recomposition u + 2^(B-1) = sum 128^i limb_i and
# every tightening identity is lowered to REAL Plonkish rows  q_M a b + q_L a + q_R b + q_O c + q_C = 0
# of a public circuit table whose commitment `root_circ` the verifier pins; the wiring (which variable
# feeds which gate cell) is enforced by the relation-4 copy-constraint GKR, and every limb / tightening
# variable is range checked against [0, 128) by the same GKR.  What remains in reference predicates
# (`LookupSpec`, AIR_PREDICATE_RESIDUE) stays prover-side.
AIR_VERIFIER_CHECKED = {
    LOOKUP_RANGE: "yes: every limb and tightening cell in [0, 128) (relation-4 FracLogUp-GKR range lookup); "
                  "with the recomposition rows, every wire's signed range and every `nonneg` inequality",
    ALGEBRAIC_GATE: "yes: Plonkish rows (relation-4 zerocheck) + copy constraints (relation-4 wiring GKR); "
                    "families and conditions in AIR_GATE_FAMILIES",
    REFERENCE_PREDICATE: "no (prover-side reference interpreter, not lowered); exact residue in "
                         "AIR_PREDICATE_RESIDUE",
}


def relation_family(name):
    """Family key of a gate / predicate name: block prefix, tile / lane indices dropped."""
    parts = [part.rstrip("0123456789") for part in name.split(".")[1:] if not part.isdigit()]
    return ".".join(parts)


_OPCODES = ("dense_gqa", "swa_sink", "moe_router_expert", "mla_absorbed", "gdn_lrec")
_MOE, _GDN = "moe_router_expert.", "gdn_lrec."

# EVERY algebraic gate family emitted by circuit/blocks.py, circuit/microprogram.py and ir/gadget_ir.py
# (`relation_family` of the gate name) -> the condition the relation-4 verifier therefore checks.  s is the
# branch selector of the family's opcode; a "nonneg" family is the slack gate value = slack + 2^(B-1) and
# holds as an inequality through the range lookup of its slack wire.  Checked for completeness (no gate
# outside this table, no entry without a gate) by tests/audit/test_lowered_air_mutation_audit.py.
AIR_GATE_FAMILIES = {
    # ---- rmsnorm_block
    "span": "endpoint.1 - endpoint.0 = span",
    # ---- gemm_tile_row_block
    "center_dot": "center = x . w",
    "difference": "observed - center = delta",
    "input_radius_nonneg": "nonneg: r_x >= 0",
    "weight_radius_upper": "nonneg: wr - w >= 0",
    "weight_radius_lower": "nonneg: wr + w >= 0   (with the line above: wr >= |w|)",
    "abs_square": "a^2 = x^2 for the range-checked cell a >= 0, i.e. a = |x|   [only if `operand` is a gate]",
    "operand": "z = num a + (den + num) r_x   [only if within the 2^62 lift bound; else a predicate]",
    "radius_dot": "radius_numerator = z . wr   [only if the declared operand width keeps the lift bound, "
                  "e.g. operand_bits = 38 at inner = 1; else the `products` predicate]",
    # ---- MicroprogramSlot.install (every opcode)
    **{f"{op}.boolean": "s (s - 1) = 0" for op in _OPCODES},
    "exhaustive": "sum of the five selectors = 1",
    **{f"{op}.unused": "s * lane_i = 0 for every lane i >= the opcode's width" for op in _OPCODES},
    # ---- dense_gqa
    "dense_gqa.normalize_gate.def": "g = s * normalize",
    "dense_gqa.normalize_boolean": "g (normalize - 1) = 0: normalize in {0, 1} when selected",
    # ---- swa_sink
    "swa_sink.position_low": "nonneg: s * position >= 0",
    "swa_sink.position_high": "nonneg: s * (extent - 1 - position) >= 0",
    # ---- moe_router_expert (on_i = s * active_i, best_i = s * argmax_i)
    _MOE + "active_gate.def": "on_i = s * active_i",
    _MOE + "argmax_gate.def": "best_i = s * argmax_i",
    _MOE + "argmax_boolean": "best_i (argmax_i - 1) = 0",
    _MOE + "argmax_one": "sum_i best_i = s: exactly one attaining maximum",
    _MOE + "argmax_attains": "best_i (score_i - maximum) = 0",
    _MOE + "argmax_selected": "best_i (active_i - 1) = 0   [top_k < experts; otherwise every expert is selected]",
    _MOE + "maximum_bound": "nonneg: on_i (maximum - score_i) >= 0",
    _MOE + "weight_positive": "nonneg: on_i (weight_i - 1) >= 0",
    _MOE + "weight_zero": "(s - on_i) weight_i = 0   [top_k < experts]",
    _MOE + "denominator": "s (denominator - sum_i weight_i) = 0",
    _MOE + "index_onehot_boolean": "h_r,i (h_r,i - 1) = 0 for the top_k x experts one-hot index cells",
    _MOE + "index_row": "sum_i h_r,i = s",
    _MOE + "index_column": "sum_r h_r,i = on_i   (with the rows: bitmap boolean, exactly top_k selected)",
    _MOE + "index_value": "s * index_r = sum_i i h_r,i",
    _MOE + "index_order": "nonneg: s (index_r - index_(r-1) - 1) >= 0: index list = ascending selected bitmap",
    _MOE + "topk_margin": "nonneg: (2 on_i - s)(score_i - threshold) + s tau >= 0: unselected - selected "
                          "<= 2 tau   [top_k < experts]",
    # ---- gdn_lrec (in = s * in_range, out = s - in)
    _GDN + "in_range_gate.def": "in = s * in_range",
    _GDN + "out_range_gate.def": "out = s - in",
    _GDN + "in_range_boolean": "in (in_range - 1) = 0",
    _GDN + "position_low": "nonneg: s * position >= 0",
    _GDN + "position_high": "nonneg: s * (tokens - 1 - position) >= 0",
    _GDN + "in_range_compare": "nonneg: in (extent - 1 - position) + out (position - extent) >= 0: "
                               "in_range = [position < extent]",
    _GDN + "out_range_convolution": "out * convolution = 0",
    _GDN + "out_range_state": "out (state - previous) = 0",
    _GDN + "decay_low": "nonneg: in * decay >= 0",
    _GDN + "decay_high": "nonneg: in (scale - decay) >= 0",
    _GDN + "beta_low": "nonneg: in * beta >= 0",
    _GDN + "beta_high": "nonneg: in (scale - beta) >= 0",
    _GDN + "tail_reached_boolean": "t_j (t_j - 1) = 0   [conv_width > 1, j = 1 .. conv_width - 1]",
    _GDN + "tail_reached_selected": "t_j (s - 1) = 0",
    _GDN + "tail_compare": "nonneg: t_j (2 position - 2 j + 1) + s (j - 1 - position) >= 0: t_j = [position >= j]",
    _GDN + "tail_gate.def": "b_j = in - in * t_j",
    _GDN + "history_tail": "b_j * history_j = 0: zero in-range history tail beyond the position",
}

# EVERY remaining reference predicate (`relation_family` of its name) -> exactly what it still checks,
# prover side only.  [T] = table / transcendental, [R] = nearest-rounding division of exact rationals,
# [A] = algebraic but not lowered (reason given).
AIR_PREDICATE_RESIDUE = {
    "norm": "rmsnorm, entire predicate: [T] bf16_decode of every input / gain / output word; residual floor "
            "sum x^2 >= width * floor^2 on the decoded values; [T] (power, endpoint.0, endpoint.1) = "
            "norm_rsqrt(mean square + epsilon); [T] output enclosure from norm_rsqrt of the "
            "accumulation-widened radicand",
    "operand": "gemm, ONLY when the gate would exceed the lift bound (e.g. fp8_e4m3_block128 at a 62-bit "
               "operand): [A] z = num |x| + (den + num) r_x",
    "products": "gemm, ONLY at an operand width whose product exceeds the lift bound (the 62-bit default): "
                "[A] radius_numerator = z . wr",
    "box": "gemm, always: [A] |delta| den <= radius_numerator + eta den (61-bit delta times den exceeds the "
           "lift bound and its slack would not fit a 62-bit wire)",
    "dense_gqa.relation": "[A] sign normalisation of query / keys when normalize != 0; [R] score_j = "
                          "_nearest(q k_j / scale); [T] weight_j = _exp_cell(score_j - max score) (_exp "
                          "enclosure, both endpoints rounding to one cell); [R] result = "
                          "_nearest(sum w v / sum w)",
    "swa_sink.relation": "0 <= position < extent (totality guard; ALSO the position_low / position_high "
                         "gates); [A] the mask index set swa_indices(position, window, sink); [R] scores; "
                         "[T] _exp_cell weights; [R] result",
    "moe_router_expert.relation": "[T] weight_i = _exp_cell(score_i - maximum) for every selected expert; "
                                  "[R] result = _nearest(weights . outputs / denominator)",
    "mla_absorbed.relation": "entire predicate: [R] score_j = _nearest((q k_j + q_rope r_j) / scale); "
                             "[T] _exp_cell weights; [R] softmax result; [R] _nearest(result * absorbed / scale)",
    "gdn_lrec.relation": "in range only: [R] convolution = _nearest(kernel . history / scale); [R] state = "
                         "_nearest(decay previous / scale + beta key / scale^2 (value + convolution - key "
                         "decay previous / scale^2))",
    "gate": "swiglu, entire predicate (no gate lifted): [T] bf16_decode, [T] SiLU via _exp, store tolerances",
    "head": "gqa group attention, entire predicate (no gate lifted): [T] _exp / ln enclosures of the softmax box",
}
AIR_DOMAIN = b"FPN2-IR-AIR-v1"
AIR_LIMB_BITS = 7                 # relation-4 limb table is [0, 128) for every profile
PAD = "#pad"
ZK_PREFIX = "#zk"                 # reserved zero-knowledge blinding variables (uniform F_p, never range-checked)
AIR_STMT_ROOT = slice(64 + 32, 64 + 64)      # header 64 | STMT = root_0 || root_circ || N_s
AIR_STMT_NS = slice(64 + 64, 64 + 96)


@dataclass(frozen=True)
class AirRow:
    """One Plonkish row: q = (q_M, q_L, q_R, q_O, q_C) canonical in [0, P); wires = (a, b, c) names."""
    origin: str
    q: tuple
    wires: tuple


def _gate_rows(gate, fresh):
    """Gate (sum coeff * monomial, degree <= 2) -> Plonkish rows and accumulator definitions.

    Returns (rows, defs) where defs maps each new accumulator variable to ('lin', [(c, w), ...]) or
    ('prod', c, u, v), evaluated in row order by `air_values`."""
    quad = [(c, m) for c, m in gate.terms if len(m) == 2]
    lin = [(c, m[0]) for c, m in gate.terms if len(m) == 1]
    const = sum(c for c, m in gate.terms if not m) % P
    if any(len(m) > 2 for _, m in gate.terms):
        raise ValueError(f"gate {gate.name} has degree > 2")
    rows, defs = [], {}
    if len(quad) == 1:
        c, (u, v) = quad[0]
        others = [(cl, w) for cl, w in lin if w not in (u, v)]
        if len(others) <= 1:
            ql = sum(cl for cl, w in lin if w == u) % P
            qr = sum(cl for cl, w in lin if w == v and v != u) % P
            qo, wc = others[0] if others else (0, PAD)
            return [AirRow(gate.name, (c, ql, qr, qo, const), (u, v, wc))], defs
    for c, (u, v) in quad:
        p = fresh(gate.name)
        rows.append(AirRow(gate.name, (c, 0, 0, P - 1, 0), (u, v, p)))
        defs[p] = ("prod", c, u, v)
        lin.append((1, p))
    if len(lin) <= 3:
        lin = lin + [(0, PAD)] * (3 - len(lin))
        rows.append(AirRow(gate.name, (0, lin[0][0], lin[1][0], lin[2][0], const),
                           (lin[0][1], lin[1][1], lin[2][1])))
        return rows, defs
    acc = fresh(gate.name)
    rows.append(AirRow(gate.name, (0, lin[0][0], lin[1][0], P - 1, 0), (lin[0][1], lin[1][1], acc)))
    defs[acc] = ("lin", (lin[0], lin[1]))
    rest = lin[2:]
    while len(rest) > 2:
        nxt = fresh(gate.name)
        rows.append(AirRow(gate.name, (0, 1, rest[0][0], P - 1, 0), (acc, rest[0][1], nxt)))
        defs[nxt] = ("lin", ((1, acc), rest[0]))
        acc, rest = nxt, rest[1:]
    rows.append(AirRow(gate.name, (0, 1, rest[0][0], rest[1][0], const), (acc, rest[0][1], rest[1][1])))
    return rows, defs


@dataclass(frozen=True)
class AirCircuit:
    """Public (witness-independent) relation-4 lowering of a Program."""
    profile: dict
    layout: object               # backend.folded_air.AirLayout
    columns: tuple
    gates: tuple                 # IR-level gates (constraints + recompose/tighten), each -> >= 1 rows
    limb_plans: tuple
    predicates: tuple
    manifest: dict
    variables: tuple             # variable index k has wire id k + 1; index 0 is the pad variable
    rows: tuple                  # AirRow per used row (the table pads to K rows with zero selectors)
    #  The ZK blinding rows (`FA.ZK_ROWS` zero-selector rows wired to 3 * ZK_ROWS fresh `#zk` variables)
    #  live only in `table`, not in `rows` (they carry no IR gate).
    defs: dict                   # accumulator definitions
    table: object                # backend.folded_air.AirTable (public circuit table)
    root: bytes                  # root_circ: the commitment the verifier pins
    digest: bytes = field(repr=False)
    hints: dict = field(default_factory=dict, repr=False, compare=False)   # prover-side aux witness generation

    @property
    def mask_variables(self):
        """Variable indices whose value is uniform F_p randomness independent of the witness: the pad
        variable, the reserved `#zk` blinding variables and every unused index (the rank check's B side)."""
        named = {k for k, name in enumerate(self.variables) if name != PAD and not name.startswith(ZK_PREFIX)}
        return tuple(k for k in range(self.layout.K) if k not in named)

    def summary(self):
        counts = {}
        for bucket in self.manifest.values():
            counts[bucket] = counts.get(bucket, 0) + 1
        return {"profile": self.profile["name"], "rows": len(self.rows), "capacity": self.layout.K,
                "variables": len(self.variables), "range_checked": sum(self.table.flag),
                "zk_rows": FA.ZK_ROWS, "ir_gates": len(self.gates), "buckets": counts,
                "verifier_checked": AIR_VERIFIER_CHECKED,
                "circuit_root": self.root.hex()}


# ----------------------------------------------------------------------------- structural compile cache
# `compile_air` is a pure function of the public Program structure (columns in order with every ColumnSpec field,
# constraints in order with their exact Expr trees, lookup names / columns) and the profile.  The cache key is a
# BLAKE2b-256 digest of a length-prefixed, self-delimiting canonical encoding of exactly that structure -- never an
# object id() and never a shape summary -- so two programs share an entry only if they lower to the same circuit
# (Defect D12: same shape, different coefficient / wiring / bits => different key).  The fingerprint is recomputed
# on EVERY call, so in-place edits of an already-compiled Program can never hit a stale entry.  Prover-side `hints`
# (not part of the public circuit) are always taken from the CALLER's program; `manifest` / `defs` are copied.
_AIR_FP_PERSON = b"FPN2-AIRFP-v1\0\0\0"
_AIR_CACHE = {}
_AIR_CACHE_MAX = 16
_AIR_CACHE_LOCK = threading.Lock()
AIR_CACHE_STATS = {"hits": 0, "misses": 0, "uncacheable": 0}


class _Uncacheable(Exception):
    """Program contains something outside the canonical encoding: compile it uncached (same errors, same result)."""


def _expr_code(e):
    """Self-delimiting canonical code of an Expr tree: w<len>:<name> | k<int>; | (<op><a><b>)."""
    if type(e) is not Expr:
        raise _Uncacheable
    op, args = e.op, e.args
    if op == "wire":
        if len(args) != 1 or type(args[0]) is not str:
            raise _Uncacheable
        return f"w{len(args[0])}:{args[0]}"
    if op == "const":
        if len(args) != 1 or type(args[0]) is not int:
            raise _Uncacheable
        return f"k{args[0]};"
    if op in ("+", "-", "*") and len(args) == 2:
        return "(" + op + _expr_code(args[0]) + _expr_code(args[1]) + ")"
    raise _Uncacheable


def air_program_fingerprint(program, profile):
    """32-byte canonical fingerprint of everything `compile_air(program, profile)` reads (the cache key)."""
    h = hashlib.blake2b(digest_size=32, person=_AIR_FP_PERSON)

    def put(data):
        if type(data) is str:
            data = data.encode()
        h.update(struct.pack("<Q", len(data)))
        h.update(data)

    def count(n):
        h.update(struct.pack("<Q", n))

    put(AIR_DOMAIN)
    put(profile_bytes(profile))
    put(f"limb_bits={AIR_LIMB_BITS};zk_rows={FA.ZK_ROWS}")
    count(len(program.columns))
    for name, spec in program.columns.items():
        if type(name) is not str:
            raise _Uncacheable
        fk = spec.freedom_kind
        put(name)
        put(repr((spec.name, spec.domain_size, spec.semantic_type, spec.producer, spec.signed_bound_bits,
                  type(fk).__name__, getattr(fk, "value", fk))))
    count(len(program.constraints))
    for c in program.constraints:
        if type(c.name) is not str:
            raise _Uncacheable
        put(c.name)
        put(_expr_code(c.difference))
    count(len(program.lookups))
    for lk in program.lookups:
        cols = tuple(lk.columns)
        if type(lk.name) is not str or any(type(x) is not str for x in cols):
            raise _Uncacheable
        put(lk.name)
        count(len(cols))
        for x in cols:
            put(x)
    return h.digest()


def clear_air_cache():
    with _AIR_CACHE_LOCK:
        _AIR_CACHE.clear()


def compile_air(program, profile=None, engine=None):
    """Public relation-4 lowering: Plonkish rows, wiring, range flags and the pinned circuit root.

    Cached on `air_program_fingerprint(program, profile)` (see above); a hit returns the SAME immutable `table` /
    `rows` / `root` / `digest` with the caller's own `hints` and fresh copies of `manifest` / `defs`.
    `engine` (optional `cuda_engine` executor) only accelerates the circuit commitment; the root is identical."""
    from copy import deepcopy
    from ..backend.fpn2 import FOLDED_AIR_V1
    profile = deepcopy(profile or FOLDED_AIR_V1)
    try:
        key = air_program_fingerprint(program, profile)
    except _Uncacheable:
        key = None
    if key is not None:
        with _AIR_CACHE_LOCK:
            hit = _AIR_CACHE.get(key)
        if hit is not None:
            AIR_CACHE_STATS["hits"] += 1        # bounds / name checks are a function of the fingerprinted structure
            return replace(hit, profile=profile, manifest=dict(hit.manifest), defs=dict(hit.defs),
                           hints=dict(program.hints))
        AIR_CACHE_STATS["misses"] += 1
    else:
        AIR_CACHE_STATS["uncacheable"] += 1
    circuit = _compile_air_uncached(program, profile, engine)
    if key is not None:
        with _AIR_CACHE_LOCK:
            while len(_AIR_CACHE) >= _AIR_CACHE_MAX:
                _AIR_CACHE.pop(next(iter(_AIR_CACHE)))
            _AIR_CACHE[key] = replace(circuit, profile=deepcopy(profile), manifest=dict(circuit.manifest),
                                      defs=dict(circuit.defs), hints={})
    return circuit


def _compile_air_uncached(program, profile, engine=None):
    layout = FA.layout_of(profile)
    columns, plans, gates, predicates, manifest, _ = _plan(program, AIR_LIMB_BITS)
    variables = [PAD] + [c[0] for c in columns] + [n for plan in plans for n in plan.names()]
    counter = [0]

    def fresh(origin):
        counter[0] += 1
        name = f"{origin}#acc{counter[0]}"
        variables.append(name)
        return name

    rows, defs = [], {}
    for gate in gates:
        r, d = _gate_rows(gate, fresh)
        rows.extend(r)
        defs.update(d)
    # Zero-knowledge blinding rows .  The zerocheck reveals e_col = w_col~(r) for every gate
    # column; padded with rows wired (1, 1, 1) to the single pad variable, so e_a, e_b, e_c shared
    # ONE F_p mask and e_a - e_b was a witness-only linear functional (`folded_air.require_joint_mask_air`
    # fails on that layout).  ZK_ROWS zero-selector rows wired to 3 * ZK_ROWS fresh uniform F_p variables
    # give every column ZK_ROWS independent masks, i.e. an F_p-basis of F_p^3 at a random point.
    zk = [[f"{ZK_PREFIX}{j}.{c}" for c in "abc"] for j in range(FA.ZK_ROWS)]
    variables.extend(name for row in zk for name in row)
    K = layout.K
    if len(rows) + FA.ZK_ROWS > K or len(variables) > K:
        raise ValueError(f"circuit needs {len(rows)} + {FA.ZK_ROWS} ZK rows / {len(variables)} variables > K = {K}")
    index = {name: k for k, name in enumerate(variables)}
    if len(index) != len(variables):
        raise AssertionError("duplicate variable name")
    table_rows = [tuple(r.q) + tuple(index[w] + 1 for w in r.wires) for r in rows]
    table_rows += [(0, 0, 0, 0, 0) + tuple(index[w] + 1 for w in row) for row in zk]
    table_rows += [(0, 0, 0, 0, 0, 1, 1, 1)] * (K - len(table_rows))
    mult = [0] * K
    for row in table_rows:
        for sig in row[5:]:
            mult[sig - 1] += 1
    flagged = {n for plan in plans for n in plan.names()}
    flag = [1 if k < len(variables) and variables[k] in flagged else 0 for k in range(K)]
    table = FA.AirTable(tuple(table_rows), tuple(mult), tuple(flag))
    root = FA.circuit_root(layout, table, engine)
    digest = _canonical_digest({
        "domain": AIR_DOMAIN.decode(), "profile_id": profile_id(profile).hex(), "columns": columns,
        "gates": [g.canonical() for g in gates], "limbs": [p.canonical() for p in plans],
        "predicates": [[n, list(c)] for n, c in predicates], "variables": variables,
        "circuit_root": root.hex()})
    from copy import deepcopy
    return AirCircuit(deepcopy(profile), layout, tuple(columns), tuple(gates), tuple(plans), tuple(predicates), manifest,
                      tuple(variables), tuple(rows), defs, table, root, digest, dict(program.hints))


def air_values(circuit, witness, overrides=None):
    """UNCHECKED variable values (wires, limbs, tightening cells, accumulators) mod P.

    Auxiliary wires absent from `witness` are filled from the circuit's hints (so a forged base witness
    gets its honest-prover auxiliary cells); auxiliary wires present in `witness` are used as given.
    `overrides` (name -> int) replaces any wire / limb value BEFORE accumulators are derived; this is
    the witness-only forgery entry point (wbforge) and performs no IR check whatsoever."""
    values = complete_witness(circuit, {name: witness[name] for name, *_ in circuit.columns if name in witness})
    for plan in circuit.limb_plans:
        values.update(zip(plan.names(), plan.decompose(values[plan.wire])))
    values.update(overrides or {})
    for row in circuit.rows:
        for w in row.wires:
            if w in circuit.defs and w not in values:
                d = circuit.defs[w]
                if d[0] == "prod":
                    values[w] = d[1] * (values[d[2]] % P) * (values[d[3]] % P) % P
                else:
                    values[w] = sum(c * (values[x] % P) for c, x in d[1]) % P
    return {k: v % P for k, v in values.items()}


def air_vector(circuit, values, rng=None):
    """Variable vector V (K canonical F_p values); the pad / `#zk` / unused variables are uniform if `rng`."""
    K = circuit.layout.K
    if rng is not None and hasattr(rng, "fp3_array"):
        V = [int(v) for v in rng.fp3_array(K)[:, 0]]       # bulk uniform F_p (SecretRng); same distribution
    else:
        fill = (lambda: rng.fp3()[0]) if rng is not None else (lambda: 0)
        V = [fill() for _ in range(K)]
    for k, name in enumerate(circuit.variables):
        if name != PAD and not name.startswith(ZK_PREFIX):
            V[k] = values[name] % P
    return V


def air_row_failures(circuit, values):
    """Names of IR gates with a violated Plonkish row (prover-side mirror of the zerocheck)."""
    bad = []
    for row in circuit.rows:
        a, b, c = (values.get(w, 0) if w != PAD else 0 for w in row.wires)
        qm, ql, qr, qo, qc = row.q
        if (qm * a * b + ql * a + qr * b + qo * c + qc) % P:
            bad.append(row.origin)
    return list(dict.fromkeys(bad))


@dataclass(frozen=True)
class AirAssignment:
    values: dict                 # variable name -> canonical F_p value


def assign_air(circuit, program, witness):
    """Check the witness against every bucket and build the private relation-4 assignment."""
    failures = list(program.failures(witness))
    if failures:
        raise LoweringError(failures)
    values = air_values(circuit, witness)
    bad = air_row_failures(circuit, values)
    if bad:
        raise LoweringError(bad)
    for plan in circuit.limb_plans:
        if any(not 0 <= values[n] < (1 << AIR_LIMB_BITS) for n in plan.names()):
            raise LoweringError([f"{plan.wire}: limb outside [0, 128)"])
    return AirAssignment(values)


def lower_air(program, witness, profile=None, engine=None):
    circuit = compile_air(program, profile, engine)
    return circuit, assign_air(circuit, program, witness)


def air_binding_nonce(circuit, server_nonce=b"\x00" * 32):
    if len(server_nonce) != 32:
        raise ValueError("server nonce must be 32 bytes")
    return hashlib.blake2b(AIR_DOMAIN + bytes(server_nonce) + circuit.digest, digest_size=32).digest()


def prove_air(circuit, assignment, server_nonce=b"\x00" * 32, rng=None, hooks=None, engine=None):
    """One canonical fixed-size FPN2 relation-4 container for the whole lowered circuit (`engine`: see
    `folded_air.prove`; byte-identical output for the same rng stream)."""
    from ..backend import folded_air as FA
    V = air_vector(circuit, assignment.values, rng or FA.SecretRng())
    return FA.prove(circuit.table, V, n_s=air_binding_nonce(circuit, server_nonce), profile=circuit.profile,
                    rng=rng, hooks=hooks, engine=engine)


def prove_air_unchecked(circuit, values, server_nonce=b"\x00" * 32, rng=None, hooks=None, engine=None):
    """Witness-only forgery path (wbforge): prove arbitrary variable values with NO lowering check."""
    return prove_air(circuit, AirAssignment(values), server_nonce, rng, hooks, engine)


def check_binding_air(circuit, container, server_nonce=b"\x00" * 32):
    """Caller-side binding: the container commits to this circuit's root and positional N_s."""
    return (bytes(container[AIR_STMT_ROOT]) == circuit.root
            and bytes(container[AIR_STMT_NS]) == air_binding_nonce(circuit, server_nonce))
