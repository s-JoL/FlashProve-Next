# SPDX-License-Identifier: Apache-2.0
"""Committed block-sparse / DeepSeek-DSA attention gadget (M3-2), lowered to REAL relation-4 Plonkish rows.

One query attends to `N_blk` compressed KV blocks.  A committed per-block summary supplies a selection score
`score_j`, an attention mass `w_j` and a value lane `val_j` (all small signed cells; the summary digest below
binds them to the statement nonce).  The gadget proves, as `ALGEBRAIC_GATE` rows + `LOOKUP_RANGE` limbs only
(no `LookupSpec` predicate, nothing prover-side), that

  * `sel_j in {0, 1}`                                   sel_j (sel_j - 1) = 0
  * `mand_j` is the circuit-fixed sink / local mask     mand_j = c_j,  mand_j (mand_j - 1) = 0
  * mandatory blocks are always selected                mand_j (1 - sel_j) = 0
  * `d_j = sel_j - mand_j`  (selected NON-mandatory)    d_j = sel_j - mand_j     (so d_j in {0, 1})
  * exactly `B` non-mandatory blocks are selected       sum_j d_j = B
  * threshold `tau` (prover-chosen, existentially quantified) separates them:
        d_j (score_j - tau) = u_j >= 0      selected non-mandatory score >= tau
        (1 - sel_j)(tau - score_j) = v_j >= 0  every unselected block's score <= tau
    (`u_j`, `v_j` are the `nonneg` slack cells: value = slack + 2^(B-1), slack range-checked limb-wise)
  * sparse attention accumulation over the SELECTED blocks only
        a_j = sel_j w_j,   Z = sum_j a_j,   num = sum_j a_j val_j,   Z >= 1
  * normalisation within the nearest-rounding radius r_o = Z / round_den (signed-lift slack s_lo, s_hi >= 0)
        Z - round_den (Z o - num) >= 0     and     Z + round_den (Z o - num) >= 0

so the selected set is a valid top-`B` set (up to ties at `tau`) of the non-mandatory block scores and `o` is the
nearest-rounded softmax-weighted mean of the selected value lanes.  The weights / values / scores of UNSELECTED
blocks enter no gate other than their own range check and the `<= tau` bound on their score: the output does not
depend on them (tested).  Prover work in the SELECTED path (`a_j`, `Z`, `num`) is O(B) semantic work; the circuit
shape is fixed and therefore still carries O(N_blk) rows (a block that is not selected costs rows multiplied by
`sel_j = 0`).  N_blk counts COMPRESSED blocks (T / block_len), not tokens.

Scope limits (kept explicit; this is a prototype gadget, not a full DSA proof):
  * `w_j` is a committed attention mass (the exp table / softmax enclosure that produces it is the existing
    prover-side predicate residue of `dense_gqa.relation`); the gadget does NOT prove `w_j >= 0` nor `w_j =
    exp(...)`: a negative weight is a statement about the committed summary, bound only through the digest.
  * One scalar value lane per block (head-dim reduction is out of scope).
  * Relation-4 containers expose no public inputs besides `root_circ` and the statement nonce `N_s`.  The summary
    digest / Merkle root below is folded into `N_s` (`dsa_statement_nonce`), exactly like
    `trace_commit.activation_binding_digest`; the proof does NOT show that the in-circuit scores / weights /
    values are the ones that digest commits to (no in-circuit hash).  A verifier that recomputes the root from
    opened summary rows learns that the proof was produced for THAT commitment, nothing more.
  * The mandatory mask is a circuit constant (`mand_j - c_j` rows), so it is part of `root_circ`.
"""
import hashlib
import struct
from dataclasses import dataclass

from ..ir.gadget_ir import Program, expression

_LEAF_PERSON = b"FPN2-DSA-LEAF-v1\0"[:16]
_NODE_PERSON = b"FPN2-DSA-NODE-v1\0"[:16]
_ROOT_PERSON = b"FPN2-DSA-ROOT-v1\0"[:16]
_NONCE_PERSON = b"FPN2-DSA-NONC-v1\0"[:16]
_PRODUCER = "dsa_block_sparse"


@dataclass(frozen=True)
class DsaConfig:
    n_blocks: int = 16
    top_b: int = 4
    mandatory: tuple = (0, 14, 15)      # sink block 0 + local window blocks (circuit constants)
    score_bits: int = 7                 # signed width of score_j / tau (int7 indexer logits; use 14 at N20 scale)
    round_den: int = 2                  # |Z o - num| * round_den <= Z   (round_den = 2: nearest rounding)

    def __post_init__(self):
        if (type(self.n_blocks) is not int or type(self.top_b) is not int or type(self.mandatory) is not tuple
                or type(self.score_bits) is not int or type(self.round_den) is not int
                or any(type(m) is not int for m in self.mandatory)):
            raise TypeError("DsaConfig requires exact built-in field types")
        if self.n_blocks < 2 or self.score_bits < 7 or self.score_bits > 28 or self.round_den < 1:
            raise ValueError("invalid DsaConfig")
        if list(self.mandatory) != sorted(set(self.mandatory)) or any(
                not 0 <= m < self.n_blocks for m in self.mandatory):
            raise ValueError("mandatory blocks must be distinct, increasing and inside [0, n_blocks)")
        if not 1 <= self.top_b <= self.n_blocks - len(self.mandatory):
            raise ValueError("top_b must be in [1, number of non-mandatory blocks]")

    @property
    def small_bits(self):
        return 7

    @property
    def z_bits(self):
        return 7 * max(1, -(-((self.n_blocks * 63).bit_length() + 1) // 7))

    @property
    def num_bits(self):
        return 7 * max(1, -(-((self.n_blocks * 63 * 64).bit_length() + 1) // 7))

    @property
    def free_blocks(self):
        return tuple(j for j in range(self.n_blocks) if j not in self.mandatory)


def names(cfg):
    n = cfg.n_blocks
    return {
        "tau": "dsa.tau", "Z": "dsa.Z", "num": "dsa.num", "o": "dsa.o",
        "score": [f"dsa.score.{j}" for j in range(n)], "sel": [f"dsa.sel.{j}" for j in range(n)],
        "mand": [f"dsa.mand.{j}" for j in range(n)], "w": [f"dsa.w.{j}" for j in range(n)],
        "val": [f"dsa.val.{j}" for j in range(n)], "d": [f"dsa.d.{j}" for j in range(n)],
        "a": [f"dsa.a.{j}" for j in range(n)],
        "u": [f"dsa.u_thr.{j}.slack" for j in range(n)], "v": [f"dsa.v_thr.{j}.slack" for j in range(n)],
        "s_lo": "dsa.o_lo.slack", "s_hi": "dsa.o_hi.slack",
    }


def build_program(cfg=None):
    """(Program, column-name map).  Primary inputs: tau, o, score_j, sel_j, mand_j, w_j, val_j; everything else is
    an auxiliary wire with an honest-prover hint (no verifier trusts a hint)."""
    cfg = cfg or DsaConfig()
    nm = names(cfg)
    n, sb, small = cfg.n_blocks, cfg.score_bits, cfg.small_bits
    p = Program()
    tau = p.wire(nm["tau"], sb, "block threshold", _PRODUCER)
    score, sel, mand, w, val, d, a = ([] for _ in range(7))
    zmax = n * 63
    for j in range(n):
        score.append(p.wire(nm["score"][j], sb, "block summary score", _PRODUCER))
        sel.append(p.wire(nm["sel"][j], small, "block selection bit", _PRODUCER))
        mand.append(p.wire(nm["mand"][j], small, "mandatory (sink/local) bit", _PRODUCER))
        w.append(p.wire(nm["w"][j], small, "block attention mass", _PRODUCER))
        val.append(p.wire(nm["val"][j], small, "block value lane", _PRODUCER))
    const = set(cfg.mandatory)
    for j in range(n):
        s, m = sel[j], mand[j]
        p.equal(f"dsa.sel_boolean.{j}", s * (s - 1))
        p.equal(f"dsa.mand_boolean.{j}", m * (m - 1))
        p.equal(f"dsa.mand_public.{j}", m - int(j in const))
        p.equal(f"dsa.mand_selected.{j}", m * (expression(1) - s))
        sname, mname = nm["sel"][j], nm["mand"][j]
        d.append(p.aux(nm["d"][j], small, "selected non-mandatory", _PRODUCER,
                       lambda wit, sname=sname, mname=mname: wit[sname] - wit[mname]))
        p.equal(f"dsa.d_def.{j}", d[j] - (s - m))
        a.append(p.product(nm["a"][j], s, w[j], _PRODUCER, small))
    cardinality = expression(0)
    for j in range(n):
        cardinality = cardinality + d[j]
    p.equal("dsa.topb_card", cardinality - cfg.top_b)
    slack_bound = (1 << sb) - 1
    for j in range(n):
        p.nonneg(f"dsa.u_thr.{j}", d[j] * (score[j] - tau), slack_bound, _PRODUCER)
        p.nonneg(f"dsa.v_thr.{j}", (expression(1) - sel[j]) * (tau - score[j]), slack_bound, _PRODUCER)
    z_sum, num_sum = expression(0), expression(0)
    for j in range(n):
        z_sum = z_sum + a[j]
        num_sum = num_sum + a[j] * val[j]
    a_names, v_names = nm["a"], nm["val"]
    z = p.aux(nm["Z"], cfg.z_bits, "sparse normaliser", _PRODUCER, lambda wit: sum(wit[x] for x in a_names))
    nu = p.aux(nm["num"], cfg.num_bits, "sparse numerator", _PRODUCER,
               lambda wit: sum(wit[x] * wit[y] for x, y in zip(a_names, v_names)))
    p.equal("dsa.Z_def", z - z_sum)
    p.equal("dsa.num_def", nu - num_sum)
    o = p.wire(nm["o"], small, "sparse attention output", _PRODUCER)
    p.nonneg("dsa.Z_positive", z - 1, zmax, _PRODUCER)
    resid = z * o - nu
    p.nonneg("dsa.o_hi", z - resid * cfg.round_den, 2 * zmax, _PRODUCER)
    p.nonneg("dsa.o_lo", z + resid * cfg.round_den, 2 * zmax, _PRODUCER)
    return p, nm


def honest_witness(cfg, scores, weights, values, tau=None):
    """Primary-input witness of the honest prover: exactly the top-`B` NON-mandatory blocks (ties by lower index),
    tau = lowest selected score (or the caller's tau), o = nearest-rounded weighted mean of the selected values."""
    n = cfg.n_blocks
    if not len(scores) == len(weights) == len(values) == n:
        raise ValueError("one score / weight / value per block")
    ranked = sorted(cfg.free_blocks, key=lambda j: (-scores[j], j))
    chosen = set(ranked[:cfg.top_b])
    sel = [int(j in chosen or j in cfg.mandatory) for j in range(n)]
    if tau is None:
        tau = min(scores[j] for j in chosen)
    z = sum(s * x for s, x in zip(sel, weights))
    num = sum(s * x * y for s, x, y in zip(sel, weights, values))
    nm = names(cfg)
    wit = {nm["tau"]: tau, nm["o"]: (2 * num + z) // (2 * z) if z > 0 else 0}
    for j in range(n):
        wit[nm["score"][j]] = scores[j]
        wit[nm["sel"][j]] = sel[j]
        wit[nm["mand"][j]] = int(j in cfg.mandatory)
        wit[nm["w"][j]] = weights[j]
        wit[nm["val"][j]] = values[j]
    return wit


# ----------------------------------------------------------------------------- committed block-summary binding
def _leaf(j, score, weight, value):
    return hashlib.blake2b(struct.pack("<Qqqq", j, score, weight, value), digest_size=32, person=_LEAF_PERSON).digest()


def _exact_i64(val, name):
    if type(val) is not int:
        raise TypeError(f"{name} must be an exact int, got {type(val).__name__}")
    if not -(1 << 63) <= val < (1 << 63):
        raise ValueError(f"{name}={val} out of signed i64 range")
    return val


def block_summary_root(cfg, scores, weights, values):
    """Merkle root of the per-block summaries (leaf_j = H(j, score_j, w_j, val_j)), bound to (N_blk, B, mask)."""
    scores, weights, values = list(scores), list(weights), list(values)
    if not (len(scores) == len(weights) == len(values) == cfg.n_blocks):
        raise ValueError(
            f"summary vectors must each have length {cfg.n_blocks}, got "
            f"({len(scores)}, {len(weights)}, {len(values)})"
        )
    level = [
        _leaf(j, _exact_i64(scores[j], f"scores[{j}]"), _exact_i64(weights[j], f"weights[{j}]"),
              _exact_i64(values[j], f"values[{j}]"))
        for j in range(cfg.n_blocks)
    ]
    while len(level) > 1:
        if len(level) & 1:
            level.append(level[-1])
        level = [hashlib.blake2b(level[i] + level[i + 1], digest_size=32, person=_NODE_PERSON).digest()
                 for i in range(0, len(level), 2)]
    h = hashlib.blake2b(digest_size=32, person=_ROOT_PERSON)
    h.update(struct.pack("<QQQ", cfg.n_blocks, cfg.top_b, len(cfg.mandatory)))
    h.update(struct.pack(f"<{len(cfg.mandatory)}Q", *cfg.mandatory))
    h.update(level[0])
    return h.digest()


def dsa_statement_nonce(summary_root, c_pre):
    """32-byte server nonce folding the committed block-summary root into the container's `N_s` (via
    `lower.air_binding_nonce`).  Verifier side: recompute the root from opened summaries and call
    `lower.check_binding_air(circuit, container, dsa_statement_nonce(root, c_pre))`."""
    if len(summary_root) != 32 or len(c_pre) != 32:
        raise ValueError("summary root and C_pre must be 32 bytes")
    return hashlib.blake2b(bytes(c_pre) + bytes(summary_root), digest_size=32, person=_NONCE_PERSON).digest()
