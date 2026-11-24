"""Protocol v2 reference engine (gate M2-1): challenge source, sampling, lineage, LSE radius, padding.

Everything here is exact (``int`` / ``fractions.Fraction``; transcendental terms use the outward-rounded
enclosures of ``blocks.py``), reads no environment, and hashes only fixed-width byte fields.

Scope, stated plainly:

* The soundness numbers are the spec 02 §5-6 with-replacement spot-check bound ``Q (1 - s)^k <= eps_sp``
  with the canonical split ``eps_sp = eps / 2``.  They are *not* "100-bit" in the external-seed regimes:
  per attempt the miss probability is ``2^-51`` (``k = 51`` at 0.5, ``k = 100`` at 0.3) and the lifetime
  target there is ``eps = 2^-20`` over ``Q_samp <= 2^30`` observable attempts.  Only the non-interactive
  regime (``k = 105``) exceeds 100 per-attempt bits.  :func:`soundness_report` returns both quantities
  under separate names; nothing in this module relabels one as the other.
* The sampler, lineage closure and padding model are reference definitions for the circuit and the
  verifier profile.  They are not wired into an end-to-end prover here.
* ``RESIDUAL_RMS_FLOOR`` for deep layers is a model-calibrated completeness premise (M0-L1 verified the
  embedding rows of four models exhaustively; deep residuals are calibrated per model), not a theorem.
"""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from hashlib import blake2b, sha256
from itertools import accumulate
from math import log2  # noqa: F401  (re-exported for reports)

from .blocks import LINEAGE_DEPTH, RESIDUAL_RMS_FLOOR, _exp, _transcendental
from .microprogram import swa_indices

# ===================================================================================== challenge source
MODE_EXTERNAL_SEED = "external_seed"
MODE_NI = "ni"
Q_SAMP_EXTERNAL_LOG2 = 30      # observable committed attempts (GOAL §2.3)
Q_NI_LOG2 = 64                 # offline Fiat-Shamir grinding budget (spec 02 §6 default)
Q_NI_REDUCED_LOG2 = 50         # reduced NI budget; k = 105 then leaves 2^-54 total instead of 2^-40


def _framed(parts):
    if any(type(part) is not bytes or not part for part in parts):
        raise ValueError("transcript inputs must be nonempty bytes")
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def external_seed(x0, root, c_pre, challenge):
    """Bind the precommitment, including under retries with fresh blinding."""
    return sha256(b"FPN2 external sampling seed\x00" + _framed((x0, root, c_pre, challenge))).digest()


def ni_seed(x0, root, c_pre):
    """Non-interactive fallback: the seed is a function of the commitments alone (grindable, Q = 2^64)."""
    return sha256(b"FPN2 NI sampling seed\x00" + _framed((x0, root, c_pre))).digest()


def challenge_seed(mode, x0, root, c_pre, challenge=None):
    """``H_out(X_0, R, C_pre, c)`` or its NI fallback.  The two modes never share a seed domain."""
    if mode == MODE_EXTERNAL_SEED:
        if challenge is None:
            raise ValueError("external-seed mode needs the external challenge c")
        return external_seed(x0, root, c_pre, challenge)
    if mode == MODE_NI:
        if challenge is not None:
            raise ValueError("NI mode takes no external challenge")
        return ni_seed(x0, root, c_pre)
    raise ValueError(f"unknown challenge mode {mode!r}")


class RetryError(ValueError):
    """A retry tried to open a different C_pre for an already committed (X_0, R)."""


class PreCommitLedger:
    """Verifier-side record of ``(X_0, R) -> C_pre``.

    A retransmission or retry must open the same ``C_pre``; a second, different ``C_pre`` for the same
    response is rejected instead of re-rolling the draws.  The external challenge is fixed at the first
    request as well, so a provider cannot ask for a second ``c`` either.
    """

    def __init__(self):
        self._bound = {}

    def commit(self, x0, root, c_pre):
        key = sha256(_framed((x0, root))).digest()
        seen = self._bound.setdefault(key, [c_pre, None])
        if seen[0] != c_pre:
            raise RetryError("a different C_pre was already committed for this response")
        return key

    def seed(self, x0, root, c_pre, challenge):
        slot = self._bound.get(self.commit(x0, root, c_pre))
        if slot[1] is None:
            slot[1] = challenge
        elif slot[1] != challenge:
            raise RetryError("the external challenge of a committed response is fixed")
        return external_seed(x0, root, c_pre, challenge)


# ===================================================================================== soundness bits
def _fraction(value):
    out = Fraction(value)
    if not 0 < out <= 1:
        raise ValueError("defect threshold must lie in (0, 1]")
    return out


def miss_probability(threshold, draws):
    """All ``draws`` cost-weighted draws with replacement miss a bad cost fraction ``>= threshold``."""
    if draws < 0:
        raise ValueError("negative draw count")
    return (1 - _fraction(threshold)) ** draws


def hypergeometric_miss(units, bad, draws):
    """Exact probability that ``draws`` distinct cost units, drawn without replacement, are all good."""
    if not (0 <= bad <= units and 0 <= draws <= units):
        raise ValueError("invalid hypergeometric parameters")
    out = Fraction(1)
    for i in range(draws):
        out *= Fraction(units - bad - i, units - i) if units - bad - i > 0 else 0
    return out


def miss_bits_floor(threshold, draws):
    """Largest integer ``n`` with ``(1 - s)^draws <= 2^-n`` (exact integer arithmetic)."""
    left = 1 - _fraction(threshold)
    if left == 0:
        raise ValueError("s = 1 misses with probability zero")
    return (left.denominator**draws // left.numerator**draws).bit_length() - 1


def sufficient(threshold, draws, q_log2, eps_log2):
    """``Q (1 - s)^k <= eps / 2`` with ``Q = 2^q_log2`` and ``eps = 2^eps_log2`` (``eps_log2 < 0``)."""
    left = 1 - _fraction(threshold)
    return left.numerator**draws << (q_log2 - eps_log2 + 1) <= left.denominator**draws


def draws_required(threshold, q_log2, eps_log2):
    """Least ``k`` satisfying :func:`sufficient` (spec 02 eq. (2); the Lean-checked 16-cell table)."""
    if q_log2 < 0 or eps_log2 >= 0:
        raise ValueError("need Q >= 1 and eps < 1")
    threshold = _fraction(threshold)
    if threshold == 1:
        return 1
    k = max(1, int((q_log2 - eps_log2 + 1) / -log2(1 - threshold)) - 2)
    while not sufficient(threshold, k, q_log2, eps_log2):
        k += 1
    while k > 1 and sufficient(threshold, k - 1, q_log2, eps_log2):
        k -= 1
    return k


@dataclass(frozen=True)
class Regime:
    name: str
    mode: str
    sigma: Fraction
    rho: Fraction
    q_log2: int
    eps_log2: int       # lifetime target, log2
    draws: int          # per pool


REGIMES = {r.name: r for r in (
    Regime("external_tight", MODE_EXTERNAL_SEED, Fraction(3, 10), Fraction(3, 10), Q_SAMP_EXTERNAL_LOG2, -20, 100),
    Regime("external_light", MODE_EXTERNAL_SEED, Fraction(1, 2), Fraction(1, 2), Q_SAMP_EXTERNAL_LOG2, -20, 51),
    Regime("ni", MODE_NI, Fraction(1, 2), Fraction(1, 2), Q_NI_LOG2, -40, 105),
)}


@dataclass(frozen=True)
class SoundnessReport:
    regime: str
    draws: int
    draws_required: int
    per_attempt_bits_floor: int      # -log2 of one attempt's spot-check miss probability
    per_attempt_bits: float
    q_log2: int
    lifetime_spot_bits_floor: int    # per_attempt - log2 Q: union bound over all attempts
    lifetime_bits_floor: int         # after eps_sp = eps/2 (aux terms are premises of the backend)
    target_bits: int
    meets_target: bool
    per_attempt_at_least_100: bool


def soundness_report(regime, *, q_log2=None):
    """Exact bits for a regime.  ``q_log2`` overrides the attempt budget (e.g. ``Q_NI_REDUCED_LOG2``)."""
    if isinstance(regime, str):
        regime = REGIMES[regime]
    q = regime.q_log2 if q_log2 is None else q_log2
    threshold = min(regime.sigma, regime.rho)      # the weaker pool governs (spec 02 §5 max bound)
    floor = miss_bits_floor(threshold, regime.draws)
    left = 1 - threshold
    exact = regime.draws * (log2(left.denominator) - log2(left.numerator))
    lifetime = floor - q - 1
    return SoundnessReport(
        regime.name, regime.draws, draws_required(threshold, q, regime.eps_log2), floor, exact, q,
        floor - q, lifetime, -regime.eps_log2,
        sufficient(threshold, regime.draws, q, regime.eps_log2), floor >= 100)


def assert_regimes():
    """Every registered regime uses exactly the least sufficient draw count for its lifetime target."""
    reports = {name: soundness_report(regime) for name, regime in REGIMES.items()}
    for report in reports.values():
        if not report.meets_target or report.draws != report.draws_required:
            raise AssertionError(f"regime {report.regime}: {report}")
    return reports


# ===================================================================================== sampler
N_SMP = 192
STRATUM_FRESH_INPUT = 1
STRATUM_GENERATION = 2
POOL_TOKEN_SAMPLING = 3
TILE_ROWS = 32
_PERSON = b"FPN2-SAMPLE-v2"


def fresh_input_stratum(input_tokens, cache_read_input_tokens):
    if not 0 <= cache_read_input_tokens <= input_tokens:
        raise ValueError("invalid cache read extent")
    return range(cache_read_input_tokens, input_tokens)


def generation_stratum(input_tokens, output_tokens):
    """Positions whose logits a generation step consumes (rule Pi_2)."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("negative token count")
    if not output_tokens:
        return range(0)
    return range(max(input_tokens - 1, 0), input_tokens + output_tokens - 1)


def draw_words(seed, pool, count):
    """``count`` independent ``N_SMP``-bit integers for one pool (BLAKE2b, 1-byte pool separator)."""
    if type(seed) is not bytes or len(seed) != 32 or not 0 < pool < 256 or count < 0:
        raise ValueError("need a 32-byte seed, a pool tag in [1, 255] and a nonnegative count")
    return tuple(int.from_bytes(blake2b(seed + bytes((pool,)) + i.to_bytes(4, "big"),
                                        digest_size=N_SMP // 8, person=_PERSON).digest(), "big")
                 for i in range(count))


def unit_index(word, total):
    """``floor(U C / 2^n_smp)`` (spec 02 §4.2); selection bias per unit is below ``2^-n_smp``."""
    if not 0 <= word < 1 << N_SMP or total < 1:
        raise ValueError("invalid sampling word or empty pool")
    return word * total >> N_SMP


@dataclass(frozen=True)
class SamplerProfile:
    """Private cost geometry of one model.  ``gemms`` lists ``(inner, out_rows)`` per layer."""
    layers: int
    gemms: tuple
    q_heads: int
    kv_groups: int
    head_dim: int
    tile_rows: int = TILE_ROWS
    window: int = 0         # sliding-window attention; 0 = dense causal
    sink: int = 0
    attn_gemms: int = 4     # the first ``attn_gemms`` GEMMs (Q, K, V, O) belong to the attention block

    def __post_init__(self):
        if (self.layers < 1 or not self.gemms or min(self.q_heads, self.kv_groups, self.head_dim) < 1
                or self.q_heads % self.kv_groups or self.tile_rows < 1 or self.window < 0 or self.sink < 0
                or not 0 <= self.attn_gemms <= len(self.gemms)
                or any(inner < 1 or rows < 1 for inner, rows in self.gemms)):
            raise ValueError("invalid sampler profile")

    @property
    def heads_per_group(self):
        return self.q_heads // self.kv_groups

    @property
    def gemm_cost(self):
        return sum(inner * rows for inner, rows in self.gemms)

    def key_positions(self, position):
        """Active keys of a query at ``position`` in canonical order (dense causal, or window plus sinks)."""
        if not self.window:
            return range(position + 1)
        return swa_indices(position, self.window, self.sink)

    def keys(self, position):
        return len(self.key_positions(position))

    def group_cost(self, position):
        """QK and PV multiply-adds of every query head of one KV group at ``position``."""
        return self.heads_per_group * self.keys(position) * 2 * self.head_dim

    def position_cost(self, position):
        return self.layers * (self.gemm_cost + self.kv_groups * self.group_cost(position))


@dataclass(frozen=True)
class Draw:
    """One sampled atom.  Draws are witness data: positions, layers and shapes never enter a container."""
    stratum: int
    index: int
    empty: bool = False
    position: int = -1
    layer: int = -1
    kind: str = ""              # "gemm" | "attn"
    gemm: int = -1
    tile_start: int = -1
    tile_rows: int = 0
    kv_group: int = -1
    query_heads: tuple = ()
    key_position: int = -1      # the drawn key atom of an attention draw (its KV producer enters the lineage)
    cost: int = 0
    block_is_attn: bool = True

    @property
    def unit(self):
        """The ``(position, layer, block)`` unit that computed this atom."""
        if self.empty:
            raise ValueError("an empty-stratum bypass draw has no unit")
        return self.position, self.layer, BLOCK_ATTN if self.block_is_attn else BLOCK_MLP

    def lineage(self, depth=LINEAGE_DEPTH):
        """``Lin_h`` of the drawn atom, including the KV-producer closure of an attention key atom."""
        return lineage(self.unit, key_position=self.key_position if self.kind == "attn" else None, depth=depth)


def _locate(profile, stratum, index, position, unit):
    per_layer = profile.gemm_cost + profile.kv_groups * profile.group_cost(position)
    layer, unit = divmod(unit, per_layer)
    if unit < profile.gemm_cost:
        ends = tuple(accumulate(inner * rows for inner, rows in profile.gemms))
        op = bisect_right(ends, unit)
        inner, rows = profile.gemms[op]
        row = (unit - (ends[op - 1] if op else 0)) // inner
        start = row - row % profile.tile_rows
        height = min(profile.tile_rows, rows - start)
        return Draw(stratum, index, False, position, layer, "gemm", op, start, height, cost=inner * height,
                    block_is_attn=op < profile.attn_gemms)
    hpg = profile.heads_per_group
    group, unit = divmod(unit - profile.gemm_cost, profile.group_cost(position))
    key = profile.key_positions(position)[unit // (hpg * 2 * profile.head_dim)]
    return Draw(stratum, index, False, position, layer, "attn", kv_group=group,
                query_heads=tuple(range(group * hpg, (group + 1) * hpg)), key_position=key,
                cost=profile.group_cost(position))


def sample_stratum(seed, profile, stratum, positions, draws):
    """``draws`` cost-weighted atoms with replacement over ``positions`` (fixed count, no rejection).

    A cost unit is drawn uniformly and snapped to the 32-row output tile or KV group containing it, so
    an atom is hit with probability proportional to its cost.  An empty stratum yields ``draws`` bypass
    records so that the circuit shape does not depend on it.
    """
    positions = range(*positions) if isinstance(positions, tuple) else positions
    if not len(positions):
        return tuple(Draw(stratum, i, True) for i in range(draws))
    ends = tuple(accumulate(profile.position_cost(p) for p in positions))
    out = []
    for i, word in enumerate(draw_words(seed, stratum, draws)):
        unit = unit_index(word, ends[-1])
        at = bisect_right(ends, unit)
        out.append(_locate(profile, stratum, i, positions[at], unit - (ends[at - 1] if at else 0)))
    return tuple(out)


def sample_request(seed, profile, *, input_tokens, cache_read_input_tokens, output_tokens, draws):
    """The three v2 pools of one turn: fresh-input compute, generation compute, token-sampling steps.

    Prefix-cached positions ``[0, cache_read_input_tokens)`` were computed, committed and sampled in the
    turn that produced them; they carry no compute cost in this turn and receive no draws, so a long
    cached prefix cannot dilute the fresh tokens.  Their keys still enter the attention cost of every
    fresh position.

    Rule Pi_2 (spec 01, default two-stratum rule): a position belongs to the generation stratum iff its final activations
    produce logits consumed by a generation step.  Those are positions ``input - 1 .. input + output - 2``:
    the last input position emits the first output token, and the last output token is never fed forward.
    The last input position is therefore generation compute even when the whole prompt is a cache hit.
    """
    if output_tokens < 0:
        raise ValueError("negative output length")
    generated = generation_stratum(input_tokens, output_tokens)
    fresh = fresh_input_stratum(input_tokens, cache_read_input_tokens)
    fresh = range(fresh.start, min(fresh.stop, generated.start)) if len(generated) else fresh
    fresh = fresh if len(fresh) else range(0)
    steps = tuple(unit_index(word, output_tokens) if output_tokens else -1
                  for word in draw_words(seed, POOL_TOKEN_SAMPLING, draws))
    return {
        STRATUM_FRESH_INPUT: sample_stratum(seed, profile, STRATUM_FRESH_INPUT, fresh, draws),
        STRATUM_GENERATION: sample_stratum(seed, profile, STRATUM_GENERATION, generated, draws),
        POOL_TOKEN_SAMPLING: steps,
    }


def stratum_cost(profile, positions):
    return sum(profile.position_cost(p) for p in positions)


# ===================================================================================== lineage
BLOCK_ATTN, BLOCK_MLP = 0, 1
COMPLETENESS_PREMISES = {
    "residual_rms_floor": {
        "value": RESIDUAL_RMS_FLOOR,
        "embedding_layer": "exhaustive over the vocabularies of the four M0-L1 models (minimum 0.007043)",
        "deep_layers": "MODEL_CALIBRATED: a completeness premise checked per model, not proved",
    },
}


def _predecessor(unit):
    position, layer, block = unit
    if block == BLOCK_MLP:
        return position, layer, BLOCK_ATTN
    return (position, layer - 1, BLOCK_MLP) if layer else None


def lineage(unit, *, key_position=None, depth=LINEAGE_DEPTH):
    """``Lin_h`` of a unit ``(position, layer, block)`` along the residual producer chain.

    For an attention key atom the closure also contains the depth-``h`` closure of the unit that
    produced ``(k, v)`` at ``key_position`` for the same layer (spec 01 §4.5).
    """
    position, layer, block = unit
    if min(position, layer, depth) < 0 or block not in (BLOCK_ATTN, BLOCK_MLP):
        raise ValueError("invalid unit or depth")
    roots = [unit]
    if key_position is not None:
        if block != BLOCK_ATTN or not 0 <= key_position <= position:
            raise ValueError("a key atom belongs to an attention unit and a causal key position")
        producer = _predecessor((key_position, layer, BLOCK_ATTN))
        if producer is not None:
            roots.append(producer)
    out = []
    for node in roots:
        for _ in range(depth + 1):
            if node is None:
                break
            if node not in out:
                out.append(node)
            node = _predecessor(node)
    return tuple(out)


def lineage_bound(depth=LINEAGE_DEPTH):
    """``B_h``: units opened per draw, padded to this count regardless of position or layer."""
    return 2 * (depth + 1)


def lineage_defect(costs, bad, *, depth=LINEAGE_DEPTH):
    """``d_C = sum c_a V_a / C`` with ``V_a = max`` over ``Lin_h`` for unit atoms (no key atoms)."""
    total = sum(costs.values())
    if total <= 0:
        return Fraction(0)
    bad = frozenset(bad)
    return Fraction(sum(cost for unit, cost in costs.items()
                        if not bad.isdisjoint(lineage(unit, depth=depth))), total)


def residual_floor_holds(sum_squares, width):
    """Norm non-degeneracy ``sum x_i^2 >= width * floor^2`` (the ``h = 0`` zero-operand guard)."""
    if width < 1 or sum_squares < 0:
        raise ValueError("invalid residual statistics")
    return sum_squares >= width * RESIDUAL_RMS_FLOOR**2


# ===================================================================================== certified LSE radius
U_FP32 = Fraction(1, 2**24)
U_BF16 = Fraction(1, 2**8)
U_FP16 = Fraction(1, 2**11)
U_ELEM = Fraction(1, 2**21)
ETA_FP32 = Fraction(1, 2**126)
ETA_BF16 = Fraction(1, 2**126)      # bf16 shares the fp32 exponent range (FTZ floor)
ETA_FP16 = Fraction(1, 2**25)       # binary16 gradual underflow: half of the 2^-24 subnormal spacing
CIRCUIT_EXP_STEP = Fraction(1, 2**14)
CIRCUIT_NORM_LIFT = Fraction(1, 2**32)
CIRCUIT_LOG_STEP = Fraction(1, 2**14)
ATTN_KEY_TILE = 64
ATTN_SPLIT_STEPS = 8
_PRECISION = {"bf16": (U_BF16, ETA_BF16), "fp16": (U_FP16, ETA_FP16)}


class LseDomainError(ValueError):
    """The radius is not certifiable at this operating point; the caller must refuse, never clamp."""


@dataclass(frozen=True)
class LseRadius:
    tokens: int
    precision: str
    e_den: Fraction         # relative normalizer error of the floating-point kernel
    norm_rel: Fraction      # T * 2^-32 circuit normalizer lift (uncapped)
    floating: Fraction      # r_lse_fp
    circuit: Fraction       # r_lse_circuit: floating + exp table + lift + log table + fp32 store


def _power(u, n):
    """Upper bound of ``(1 + u)^n`` as ``exp(n ln(1 + u))``, rounded outward (keeps the rationals small)."""
    return _exp(n * _transcendental(1 + u, "ln")[1])[1]


def _neg_log1m(error):
    """Upper bound of ``-ln(1 - error)`` for ``0 <= error < 1``."""
    return -_transcendental(1 - error, "ln")[0] if error else Fraction(0)


def score_radius(abs_dot, score_abs, score_max_abs, head_dim):
    """Data-dependent score radius ``scale * gamma * <|q|,|k|> + 4 u32 (|z| + |z_max|)`` (M0-S1).

    ``scale`` is passed in through ``abs_dot``, which must already carry the ``1/sqrt(d_h)`` factor.
    """
    depth = -(-head_dim // 16) + 64
    gamma = depth * U_FP32 / (1 - depth * U_FP32)
    return gamma * Fraction(abs_dot) + 4 * U_FP32 * (Fraction(score_abs) + Fraction(score_max_abs))


def certified_lse_radius(tokens, *, score_radius_max=0, score_max_abs=0, precision="bf16"):
    """v2 certified LSE radius of one query head over ``tokens`` keys, formula of M0-S1 re-verification.

    ``score_radius_max`` is ``rbar = max_j`` :func:`score_radius`.  Fails closed with
    :class:`LseDomainError` when the normalizer denominator is not provably positive or the uncapped
    normalizer lift reaches 1.  Every transcendental is rounded outward.
    """
    if precision not in _PRECISION:
        raise ValueError(f"unknown attention precision {precision!r}")
    rbar, zmax = Fraction(score_radius_max), Fraction(score_max_abs)
    if tokens < 1 or rbar < 0 or zmax < 0:
        raise ValueError("invalid LSE inputs")
    u_p, eta_p = _PRECISION[precision]
    norm_rel = tokens * CIRCUIT_NORM_LIFT
    if norm_rel >= 1:
        raise LseDomainError(f"normalizer lift T * 2^-32 = {float(norm_rel):.3g} >= 1")
    tiles = -(-tokens // ATTN_KEY_TILE) + ATTN_SPLIT_STEPS
    depth_pv = -(-ATTN_KEY_TILE // 16) + 64
    emu = (1 + u_p) * _power(U_FP32, 2 * tiles + depth_pv) - 1
    emu2 = _power(U_FP32, 2 * tiles + ATTN_KEY_TILE) - 1
    one_s = _exp(2 * rbar)[1] * _power(U_ELEM, tiles + 1)
    eta_bar = _exp(rbar)[1] * ((tiles + 1) * (1 + emu) * ETA_FP32 + eta_p)
    e_den = one_s * (1 + emu2) - 1 + tokens * eta_bar
    if e_den >= 1:
        raise LseDomainError(f"normalizer denominator 1 - e_den = {float(1 - e_den):.3g} <= 0")
    floating = rbar + _neg_log1m(e_den) + 4 * U_ELEM
    log_t = _transcendental(Fraction(max(tokens, 2)), "ln")[1]
    circuit = (floating + CIRCUIT_EXP_STEP + _neg_log1m(norm_rel) + CIRCUIT_LOG_STEP
               + U_FP32 * (zmax + log_t))
    return LseRadius(tokens, precision, e_den, norm_rel, floating, circuit)


# ===================================================================================== padding and p_audit
LENGTH_BUCKETS = tuple(1 << e for e in range(8, 19))     # 256 .. 262144 total tokens
CONTAINER_ALIGN = 4096


def length_bucket(total_tokens):
    """Smallest public length bucket holding ``T_total`` (a function of API usage fields only)."""
    if total_tokens < 0:
        raise ValueError("negative token count")
    for bucket in LENGTH_BUCKETS:
        if total_tokens <= bucket:
            return bucket
    raise ValueError(f"T_total = {total_tokens} exceeds the largest length bucket {LENGTH_BUCKETS[-1]}")


@dataclass(frozen=True)
class ProofBudget:
    bucket: int
    limb_queries: int       # public cap for this (profile, bucket); never the hidden model's own count
    n_vars: int
    containers: int
    container_bytes: int
    registered: bool        # False: container_bytes is a layout estimate, not a registered C_p

    @property
    def total_bytes(self):
        return self.containers * self.container_bytes

    def padding(self, used_containers):
        """Dummy containers the prover must add.  Exceeding the budget is an error, never a truncation."""
        if not 0 <= used_containers <= self.containers:
            raise ValueError("proof exceeds the public budget of its length bucket")
        return self.containers - used_containers


def fpn2_budget(total_tokens, queries_per_token, *, fixed_queries=0, n_queries=72, max_n_vars=28):
    """Fixed relation-3 budget of a length bucket from public profile caps.

    ``queries_per_token`` and ``fixed_queries`` are the profile's limb-query caps (an envelope over every
    architecture the profile admits).  One container of ``2^n`` cells holds ``2^(n-5)`` queries; the
    smallest even ``n`` that fits is used, then whole containers at ``max_n_vars``.
    """
    from ..backend.fpn2 import FOLDED_GKR_V1, Layout    # imported lazily: needs the field package

    bucket = length_bucket(total_tokens)
    if queries_per_token < 0 or fixed_queries < 0 or queries_per_token + fixed_queries == 0:
        raise ValueError("invalid query caps")
    need = bucket * queries_per_token + fixed_queries
    n = 8
    while n < max_n_vars and 1 << (n - 5) < need:
        n += 2
    layout = Layout(n, n_queries)
    registered = (n, n_queries) == (FOLDED_GKR_V1["n_vars"], FOLDED_GKR_V1["n_queries"])
    size = (FOLDED_GKR_V1["c_p"] if registered
            else -(-(64 + layout.body_used) // CONTAINER_ALIGN) * CONTAINER_ALIGN)
    return ProofBudget(bucket, need, n, -(-need // layout.quarter), size, registered)


@dataclass(frozen=True)
class AuditCost:
    cost_ratio: Fraction        # prover GPU-s / serving GPU-s for a fully proved request
    commit_overhead: Fraction   # serving-side cost of committing (R, C_pre) on every response
    p_audit: Fraction
    overhead: Fraction          # steady-state total = commit + p_audit * ratio
    gate: Fraction
    passes: bool
    p_audit_max: Fraction       # largest audit rate that still meets the gate (capped at 1)


def audit_cost(cost_ratio, p_audit, *, commit_overhead=0, gate=Fraction(1, 20)):
    """Commit-all / prove-sampled steady-state cost model.  A model, not a measurement."""
    ratio, p, commit, gate = (Fraction(v) for v in (cost_ratio, p_audit, commit_overhead, gate))
    if ratio < 0 or commit < 0 or not 0 <= p <= 1 or gate <= 0:
        raise ValueError("invalid cost model inputs")
    overhead = commit + p * ratio
    if commit > gate:
        p_max = Fraction(0)
    else:
        p_max = min(Fraction(1), (gate - commit) / ratio) if ratio else Fraction(1)
    return AuditCost(ratio, commit, p, overhead, gate, overhead <= gate, p_max)
