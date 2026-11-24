# SPDX-License-Identifier: Apache-2.0
"""Hardware-agnostic numeric-format and rounding-contract IR (stdlib only, no hardware branches).

Five frozen data types describe *what* a deployment computes, never *which GPU* computes it:

* :class:`QuantBlockScale` -- block-scale metadata (``fp32`` 1x128 / 128x128, ``e8m0`` block-32, ...).
* :class:`NumFormat`       -- an element format (``bf16``, ``fp16``, ``fp8_e4m3``, ``fp4_e2m1``, ...), optionally
  carrying a :class:`QuantBlockScale`.  Exact storage width and exact dyadic-lift width are derived, not hand-typed.
* :class:`RoundingContract` -- the parameters of the specs/06 §1.1 error templates (``Store``, ``Dot``, ``Dot^sk``,
  ``AttnFA``, ``Reduce``): accumulator, ``promote_k``, ``split_k_max``, ``tp_reduce_max``, attention ``P`` / output
  formats.  ``contract <= profile_cap`` is a *componentwise partial order* (never lexicographic).
* :class:`ModelFormat`     -- weights / activations / KV / residual formats plus the per-operator contracts.
* :class:`KernelAdapter`   -- binds an engine kernel family on an architecture to a contract; ``certified``
  operators only read the contract, so there is no GPU-model branch anywhere downstream.

Honesty rules baked into the data (``Calibration``):

* ``bf16_fp32acc`` is pinned to the constants of ``flashprove_next.certified.Contract`` (specs/06 §1.2 ``R32``) and is
  the only entry measured on hardware (RTX 4070 Ti SUPER / sm_89; Hopper recalibration is gate M3-1).
* Every FP8 / MXFP8 / MXFP4 / FP16-attention entry is ``SCHEMA_ONLY``: it fixes the *shape* of the contract so that the
  rest of the stack can be written against it today, but its numeric parameters are NOT calibrated radii.
  :meth:`RoundingContract.require_calibrated` refuses to hand such a contract to a proof path.

``RoundingContract.dot_relative_bound`` is a conservative closed-form composition of the specs/06 templates, used for
ordering/regression tests and sizing -- it does not replace the certified evaluators.  All arithmetic is exact
(``int`` / ``fractions.Fraction``); no floats.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from fractions import Fraction
from typing import Any, Mapping, Optional

__all__ = [
    "Calibration",
    "ToleranceClass",
    "QuantBlockScale",
    "NumFormat",
    "RoundingContract",
    "ProfileCap",
    "ModelFormat",
    "KernelAdapter",
    "FormatError",
    "ContractCapError",
    "UncalibratedContractError",
    "NUM_FORMATS",
    "CONTRACTS",
    "MODEL_FORMATS",
    "KERNEL_ADAPTERS",
    "PROFILE_CAPS",
    "get_contract",
    "get_model_format",
    "register_contract",
    "GemmErrorBound",
    "gemm_error_bound",
    "weight_commitment",
    "exact_dequant_scale_exp_range",
]


class FormatError(ValueError):
    """Malformed format / contract description."""


class ContractCapError(FormatError):
    """A contract is looser than the public profile's tolerance-class upper bound."""


class UncalibratedContractError(FormatError):
    """A schema-only (not hardware-calibrated) contract was requested where a calibrated one is required."""


class Calibration(str, Enum):
    CALIBRATED = "calibrated"
    SCHEMA_ONLY = "schema_only"


class ToleranceClass(IntEnum):
    """Provisional ordinal of the worst-case error template family (lower = tighter).

    The ordinal is only the *first* component of the cap check; the numeric fields of the contract are compared
    componentwise as well (see :meth:`RoundingContract.within`).  The order of ``BLOCK_MX`` vs ``BLOCK_FP8`` is
    provisional until gate M3-1 calibrates both on Hopper.
    """

    FP16_STRICT = 0     # fp16 attention probabilities / outputs
    R32 = 1             # fp32 accumulation, fp32 split-K partial sums (specs/06 §1.2 default C_max entry)
    BLOCK_MX = 2        # MX block-32 e8m0 scaled formats
    BLOCK_FP8 = 3       # fp8 with limited tensor-core accumulation + periodic fp32 promotion
    BF16_SPLIT_K = 4    # PyTorch default allow_bf16_reduced_precision_reduction=True (violates R32 by 31-39x)


# ====================================================================================================== block scale
_E8M0 = "e8m0"
_SCALE_BITS = {"fp32": 32, "e8m0": 8, "e4m3": 8, "bf16": 16}


@dataclass(frozen=True)
class QuantBlockScale:
    """One scale per ``block_rows x block_cols`` elements (``block_rows == 1`` for per-row-group activations)."""

    scale_format: str
    block_rows: int
    block_cols: int

    def __post_init__(self) -> None:
        if self.scale_format not in _SCALE_BITS:
            raise FormatError(f"unknown scale format {self.scale_format!r}")
        if self.block_rows < 1 or self.block_cols < 1:
            raise FormatError("block dimensions must be >= 1")

    @property
    def block_elems(self) -> int:
        return self.block_rows * self.block_cols

    @property
    def scale_bits(self) -> int:
        return _SCALE_BITS[self.scale_format]

    @property
    def pow2(self) -> bool:
        """True when every scale is exactly ``2**e`` (``e8m0``): the scaled value stays a dyadic rational with a pure
        exponent shift, so the lift into the field needs no extra mantissa bits."""
        return self.scale_format == _E8M0

    @property
    def scale_exp_range(self) -> Optional[tuple[int, int]]:
        """Inclusive exponent range of an ``e8m0`` scale (code 255 is NaN); ``None`` for non-power-of-two scales."""
        return (-127, 127) if self.pow2 else None


# ======================================================================================================= element format
@dataclass(frozen=True)
class NumFormat:
    """An element format.  ``kind`` is ``"float"`` (IEEE-like sign/exp/mantissa) or ``"pow2"`` (``e8m0``)."""

    name: str
    kind: str
    bits: int
    exp_bits: int
    man_bits: int
    bias: int
    has_inf: bool
    max_finite: Fraction
    scale: Optional[QuantBlockScale] = None

    def __post_init__(self) -> None:
        if self.kind not in ("float", "pow2"):
            raise FormatError(f"bad kind {self.kind!r}")
        if self.bits < 1 or self.bits > 64:
            raise FormatError("bits out of range")
        if self.kind == "float" and 1 + self.exp_bits + self.man_bits != self.bits:
            raise FormatError(f"{self.name}: 1+exp+man != bits")

    # -- rounding ---------------------------------------------------------------------------------------------------
    @property
    def unit_roundoff_log2(self) -> Optional[int]:
        """``log2(u)`` of round-to-nearest (``bf16`` -8, ``fp16`` -11, ``fp32`` -24, ``e4m3`` -4, ``e5m2`` -3).
        ``None`` for ``pow2`` formats, which are exact on the dyadic grid."""
        return None if self.kind == "pow2" else -(self.man_bits + 1)

    @property
    def unit_roundoff(self) -> Optional[Fraction]:
        k = self.unit_roundoff_log2
        return None if k is None else Fraction(1, 1 << -k)

    # -- storage ----------------------------------------------------------------------------------------------------
    @property
    def storage_bits_per_element(self) -> Fraction:
        """Exact bits per element including the amortised block scale (``mxfp4``: 4 + 8/32 = 4.25)."""
        extra = Fraction(0) if self.scale is None else Fraction(self.scale.scale_bits, self.scale.block_elems)
        return self.bits + extra

    def storage_bytes(self, n_elements: int) -> int:
        """Exact bytes for ``n_elements`` (whole scale blocks, elements packed to the byte)."""
        if n_elements < 0:
            raise FormatError("negative element count")
        elem = -(-n_elements * self.bits // 8)
        if self.scale is None:
            return elem
        blocks = -(-n_elements // self.scale.block_elems)
        return elem + blocks * self.scale.scale_bits // 8

    def block_count(self, rows: int, cols: int) -> int:
        """Scale blocks of a ``rows x cols`` matrix (ragged edge blocks count as whole blocks; 0 if unscaled)."""
        if rows < 0 or cols < 0:
            raise FormatError("negative matrix dimension")
        if self.scale is None:
            return 0
        return -(-rows // self.scale.block_rows) * -(-cols // self.scale.block_cols)

    def storage_bytes_2d(self, rows: int, cols: int) -> int:
        """Exact bytes of a ``rows x cols`` matrix: packed elements plus one scale per 2-D block."""
        blocks = self.block_count(rows, cols)
        return -(-rows * cols * self.bits // 8) + (blocks * self.scale.scale_bits // 8 if self.scale else 0)

    @property
    def lift_mantissa_bits(self) -> int:
        """Significand bits of an exactly lifted (scaled) element: the element's own for a power-of-two scale,
        element + scale significands for a floating-point scale (the product is exact but wider)."""
        own = self.man_bits + 1
        if self.scale is None or self.scale.pow2:
            return own
        return own + {"fp32": 24, "bf16": 8, "e4m3": 4}[self.scale.scale_format]

    # -- exact dyadic lift ------------------------------------------------------------------------------------------
    @property
    def unit_log2(self) -> int:
        """``log2`` of the finest representable spacing (the subnormal step)."""
        if self.kind == "pow2":
            return -self.bias
        return 1 - self.bias - self.man_bits

    @property
    def element_integer_bits(self) -> int:
        """Signed bits of ``N`` such that every finite element equals ``N * 2**unit_log2`` exactly."""
        if self.kind == "pow2":
            return 8
        n_max = self.max_finite / (Fraction(1, 1 << -self.unit_log2) if self.unit_log2 < 0 else 1 << self.unit_log2)
        if n_max.denominator != 1:  # pragma: no cover - guarded by construction
            raise FormatError(f"{self.name}: max_finite not on the unit grid")
        return int(n_max).bit_length() + 1

    @property
    def exact_dyadic(self) -> bool:
        """Every finite value (including any ``e8m0``-scaled value) is a dyadic rational: lifts losslessly."""
        return True

    def with_scale(self, scale: Optional[QuantBlockScale], name: Optional[str] = None) -> "NumFormat":
        return replace(self, scale=scale, name=name or self.name)


def _float_fmt(name: str, e: int, m: int, bias: int, has_inf: bool, max_finite: Fraction) -> NumFormat:
    return NumFormat(name, "float", 1 + e + m, e, m, bias, has_inf, max_finite)


def _max_ieee(e: int, m: int, bias: int) -> Fraction:
    emax = (1 << e) - 2 - bias
    return (2 - Fraction(1, 1 << m)) * (Fraction(2) ** emax)


NUM_FORMATS: Mapping[str, NumFormat] = {
    "bf16": _float_fmt("bf16", 8, 7, 127, True, _max_ieee(8, 7, 127)),
    "fp16": _float_fmt("fp16", 5, 10, 15, True, Fraction(65504)),
    "fp32": _float_fmt("fp32", 8, 23, 127, True, _max_ieee(8, 23, 127)),
    "fp8_e4m3": _float_fmt("fp8_e4m3", 4, 3, 7, False, Fraction(448)),       # OCP e4m3fn: no inf, one NaN code
    "fp8_e5m2": _float_fmt("fp8_e5m2", 5, 2, 15, True, Fraction(57344)),
    "fp4_e2m1": _float_fmt("fp4_e2m1", 2, 1, 1, False, Fraction(6)),          # {0, .5, 1, 1.5, 2, 3, 4, 6}
    "e8m0": NumFormat("e8m0", "pow2", 8, 8, 0, 127, False, Fraction(2) ** 127),
}


# ======================================================================================================= rounding contract
def _u(log2: int) -> Fraction:
    return Fraction(1, 1 << -log2) if log2 <= 0 else Fraction(1 << log2)


def _gamma(h: int, u: Fraction) -> Fraction:
    hu = h * u
    if hu >= 1:
        raise FormatError("h*u >= 1: bound is vacuous")
    return hu / (1 - hu)


@dataclass(frozen=True)
class RoundingContract:
    """Parameters of the specs/06 §1.1 templates.  All ``*_u_log2`` are ``log2`` of a unit roundoff (more negative
    is tighter).  ``promote_k == 0`` means "no reduced-precision accumulation stage".  ``split_k_max == 1`` with an
    ``fp32`` partial store means the split is absorbed by the certified depth ``ceil(n/depth_div)+depth_add``
    (specs/06: ``Dot^sk`` collapses to ``Dot`` when ``k_p = fp32``)."""

    name: str
    tol_class: ToleranceClass
    calibration: Calibration
    calibrated_on: tuple[str, ...] = ()
    # -- GEMM / dot ------------------------------------------------------------------------------------------------
    accum_u_log2: int = -24
    depth_div: int = 16
    depth_add: int = 64
    promote_k: int = 0
    promote_u_log2: int = -24          # accumulator precision *inside* a promote_k window (ignored if promote_k == 0)
    split_k_max: int = 1
    split_partial_u_log2: int = -24    # store format of split-K partial sums (fp32 -24, bf16 -8)
    tp_reduce_max: int = 1             # fan-in cap of cross-device reductions; does not reveal the TP degree
    tp_reduce_u_log2: int = -24
    store_u_log2: int = -8             # output store format of GEMM-class operators
    elem_u_log2: int = -21             # silu / rsqrt / exp / log (4 ulp fp32)
    eta_log2: int = -126               # FTZ-safe additive floor
    # -- attention (FlashAttention-style) ------------------------------------------------------------------------
    p_u_log2: Optional[int] = None     # probability store format before PV
    attn_out_u_log2: Optional[int] = None
    attn_key_tile: int = 0             # B_c; 0 = not applicable
    split_kv_max: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise FormatError("contract needs a name")
        for f in ("depth_div", "split_k_max", "tp_reduce_max", "split_kv_max"):
            if getattr(self, f) < 1:
                raise FormatError(f"{f} must be >= 1")
        if self.promote_k < 0 or self.depth_add < 0 or self.attn_key_tile < 0:
            raise FormatError("negative parameter")
        for f in fields(self):
            if f.name.endswith("_u_log2"):
                v = getattr(self, f.name)
                if v is not None and not (-60 <= v <= 0):
                    raise FormatError(f"{f.name}={v} out of range")
        if self.calibration is Calibration.CALIBRATED and not self.calibrated_on:
            raise FormatError("a CALIBRATED contract must name its evidence")

    # -- derived constants --------------------------------------------------------------------------------------
    @property
    def u_acc(self) -> Fraction:
        return _u(self.accum_u_log2)

    @property
    def u_store(self) -> Fraction:
        return _u(self.store_u_log2)

    @property
    def effective_split_k(self) -> int:
        """Split-K only matters when the partial store is lossier than the accumulator."""
        return self.split_k_max if self.split_partial_u_log2 > self.accum_u_log2 else 1

    def depth(self, n: int) -> int:
        """Certified leaf-to-root depth ``h(n) = ceil(n/depth_div) + depth_add`` (as ``certified.Contract.h``)."""
        return -(-n // self.depth_div) + self.depth_add

    def dot_relative_bound(self, k: int) -> Fraction:
        """Conservative ``c`` with ``|y_hat - y| <= c * sum|a_i b_i|`` (excluding the additive floor) for a length-``k``
        reduction under this contract: accumulate (optionally in ``promote_k`` windows at ``promote_u``), split-K with
        partial store, TP reduction, final store.  Uses ``sum|y_s| <= <|a|,|b|>`` (specs/06 §1.1 note)."""
        if k < 1:
            raise FormatError("k must be >= 1")
        u_acc = self.u_acc
        if self.promote_k > 0:
            win = min(k, self.promote_k)
            n_win = -(-k // self.promote_k)
            g = (1 + _gamma(win, _u(self.promote_u_log2))) * (1 + _gamma(n_win, u_acc)) - 1
        else:
            g = _gamma(self.depth(k), u_acc)
        s = self.effective_split_k
        if s > 1:
            seg = -(-k // s)
            if self.promote_k > 0:
                g_seg = (1 + _gamma(min(seg, self.promote_k), _u(self.promote_u_log2))) \
                    * (1 + _gamma(-(-seg // self.promote_k), u_acc)) - 1
            else:
                g_seg = _gamma(self.depth(seg), u_acc)
            u_kp = _u(self.split_partial_u_log2)
            g_red = _gamma(s - 1, u_acc)
            g = (1 + g_red) * ((1 + u_kp) * g_seg + u_kp) + g_red
        if self.tp_reduce_max > 1:
            g_tp = _gamma(self.tp_reduce_max - 1, _u(self.tp_reduce_u_log2))
            g = g + g_tp * (1 + g)
        return (1 + g) * (1 + self.u_store) - 1

    # -- partial order ------------------------------------------------------------------------------------------
    def within(self, cap: "RoundingContract") -> bool:
        """``self <= cap``: *every* component of ``self`` is at least as tight as ``cap`` (componentwise partial
        order; two contracts may be incomparable).  A ``None`` attention field is "not applicable" and only fits a
        cap that is also not applicable."""
        return not self.violations(cap)

    def violations(self, cap: "RoundingContract") -> list[str]:
        v: list[str] = []

        def le(label: str, mine: Any, theirs: Any) -> None:
            if mine > theirs:
                v.append(f"{label}: {mine} > cap {theirs}")

        def le_opt(label: str, mine: Optional[int], theirs: Optional[int]) -> None:
            if mine is None:
                return
            if theirs is None:
                v.append(f"{label}: {mine} but cap has none")
            elif mine > theirs:
                v.append(f"{label}: {mine} > cap {theirs}")

        le("tol_class", int(self.tol_class), int(cap.tol_class))
        le("accum_u_log2", self.accum_u_log2, cap.accum_u_log2)
        le("promote_k", self.promote_k, cap.promote_k)
        if self.promote_k > 0 and cap.promote_k > 0:
            le("promote_u_log2", self.promote_u_log2, cap.promote_u_log2)
        le("effective_split_k", self.effective_split_k, cap.effective_split_k)
        if self.effective_split_k > 1:
            le("split_partial_u_log2", self.split_partial_u_log2, cap.split_partial_u_log2)
        le("tp_reduce_max", self.tp_reduce_max, cap.tp_reduce_max)
        if self.tp_reduce_max > 1:
            le("tp_reduce_u_log2", self.tp_reduce_u_log2, cap.tp_reduce_u_log2)
        le("store_u_log2", self.store_u_log2, cap.store_u_log2)
        le("elem_u_log2", self.elem_u_log2, cap.elem_u_log2)
        le("eta_log2", self.eta_log2, cap.eta_log2)
        le("depth_add", self.depth_add, cap.depth_add)
        le("split_kv_max", self.split_kv_max, cap.split_kv_max)
        le_opt("p_u_log2", self.p_u_log2, cap.p_u_log2)
        le_opt("attn_out_u_log2", self.attn_out_u_log2, cap.attn_out_u_log2)
        if self.attn_key_tile and cap.attn_key_tile and self.attn_key_tile < cap.attn_key_tile:
            v.append(f"attn_key_tile: {self.attn_key_tile} < cap {cap.attn_key_tile} (more steps = looser)")
        return v

    def __le__(self, cap: "RoundingContract") -> bool:
        if not isinstance(cap, RoundingContract):
            return NotImplemented
        return self.within(cap)

    def __ge__(self, other: "RoundingContract") -> bool:
        if not isinstance(other, RoundingContract):
            return NotImplemented
        return other.within(self)

    def assert_within(self, cap: "RoundingContract") -> None:
        bad = self.violations(cap)
        if bad:
            raise ContractCapError(f"contract {self.name!r} exceeds profile cap {cap.name!r}: " + "; ".join(bad))

    def require_calibrated(self) -> "RoundingContract":
        if self.calibration is not Calibration.CALIBRATED:
            raise UncalibratedContractError(
                f"contract {self.name!r} is a schema-only entry (no hardware calibration); calibrate it (gate M3-1) "
                "before using it on a proof path"
            )
        return self

    # -- data-driven schema ---------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            d[f.name] = val.name if isinstance(val, Enum) else (list(val) if isinstance(val, tuple) else val)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RoundingContract":
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise FormatError(f"unknown contract fields: {sorted(extra)}")
        kw = dict(d)
        tc = kw["tol_class"]
        kw["tol_class"] = ToleranceClass[tc] if isinstance(tc, str) else ToleranceClass(tc)
        cal = kw["calibration"]
        if not isinstance(cal, Calibration):
            cal = Calibration[cal] if cal in Calibration.__members__ else Calibration(cal)
        kw["calibration"] = cal
        kw["calibrated_on"] = tuple(kw.get("calibrated_on", ()))
        return cls(**kw)


@dataclass(frozen=True)
class ProfileCap:
    """What a public profile pins: a tolerance-class upper bound per operator family -- never the deployed precision."""

    name: str
    gemm: RoundingContract
    attn: RoundingContract


# --- built-in contracts -------------------------------------------------------------------------------------------
# bf16_fp32acc == flashprove_next.certified.Contract (R32): u_store 2^-8, u_acc 2^-24, h(n)=ceil(n/16)+64,
# u_elem 2^-21, eta 2^-126, u_P 2^-8, B_c 64, split_steps 8.
_BF16_FP32ACC = RoundingContract(
    name="bf16_fp32acc",
    tol_class=ToleranceClass.R32,
    calibration=Calibration.CALIBRATED,
    calibrated_on=(
        "sm_89 RTX 4070 Ti SUPER: Qwen3-0.6B 0/249384 bad blocks (benchmarks/certified/results)",
        "sm_89 RTX 4070 Ti SUPER: Llama-3.2-1B-Instruct 0/143544 bad blocks (benchmarks/certified/results)",
    ),
    accum_u_log2=-24, depth_div=16, depth_add=64, promote_k=0, split_k_max=1, split_partial_u_log2=-24,
    tp_reduce_max=1, store_u_log2=-8, elem_u_log2=-21, eta_log2=-126,
    p_u_log2=-8, attn_out_u_log2=-8, attn_key_tile=64, split_kv_max=8,
)

# fp16 attention P / output tier (design memo 03, gate M0-S1): P and the attention output are stored in fp16.
_FP16_ATTN_STRICT = replace(
    _BF16_FP32ACC, name="fp16_attn_strict", tol_class=ToleranceClass.FP16_STRICT,
    calibration=Calibration.SCHEMA_ONLY, calibrated_on=(),
    p_u_log2=-11, attn_out_u_log2=-11,
)

# H100 fp8 block-scaled GEMM: limited tensor-core accumulation, fp32 promotion every `promote_k` elements.
# promote_u_log2=-14 is the *reported* ~14-bit hardware accumulation width; it is a schema default, NOT a calibrated
# radius.  Every numeric parameter below is replaced by measurement in gate M3-1.
_FP8_E4M3_BLOCK128 = RoundingContract(
    name="fp8_e4m3_block128", tol_class=ToleranceClass.BLOCK_FP8, calibration=Calibration.SCHEMA_ONLY,
    accum_u_log2=-24, promote_k=128, promote_u_log2=-14, split_k_max=1, split_partial_u_log2=-24,
    tp_reduce_max=1, store_u_log2=-8, p_u_log2=-8, attn_out_u_log2=-8, attn_key_tile=64, split_kv_max=8,
)

_MXFP8_E4M3 = RoundingContract(
    name="mxfp8_e4m3", tol_class=ToleranceClass.BLOCK_MX, calibration=Calibration.SCHEMA_ONLY,
    accum_u_log2=-24, promote_k=0, split_k_max=1, split_partial_u_log2=-24,
    tp_reduce_max=1, store_u_log2=-8, p_u_log2=-8, attn_out_u_log2=-8, attn_key_tile=64, split_kv_max=8,
)

# mxfp4_e2m1: exact dyadic block-32 e8m0.  Weight lift is lossless; the *dot* contract is the same fp32-accumulate
# schema as mxfp8 until calibrated.
_MXFP4_E2M1 = replace(_MXFP8_E4M3, name="mxfp4_e2m1")

# --- production deployment contracts (Hopper sm_90 / Blackwell sm_120), gate M3-1 --------------------------------
# Both describe the path the serving engines actually run for these checkpoints: block-scaled weights are expanded
# to their exact scaled values, the GEMM accumulates in fp32 and the output is stored in bf16.  Neither has been
# measured on hardware yet, so both stay SCHEMA_ONLY; bench/tier2/bench_frontier_calibrate.py produces the
# evidence that a later change may cite in `calibrated_on`.  The architecture never enters a contract.
#
# fp8_e4m3_block128_fp32acc: e4m3 weights with one fp32 scale per 128x128 block, e4m3 activations with one fp32
# scale per token row group (Qwen3-30B-A3B-FP8 layout).  The activation quantisation is a separate Store step,
# accounted for by `gemm_error_bound`, not by the dot template.
_FP8_E4M3_BLOCK128_FP32ACC = RoundingContract(
    name="fp8_e4m3_block128_fp32acc", tol_class=ToleranceClass.BLOCK_FP8, calibration=Calibration.SCHEMA_ONLY,
    accum_u_log2=-24, promote_k=0, split_k_max=1, split_partial_u_log2=-24,
    tp_reduce_max=1, store_u_log2=-8, elem_u_log2=-21, eta_log2=-126,
    p_u_log2=-8, attn_out_u_log2=-8, attn_key_tile=64, split_kv_max=8,
)

# mxfp4_e2m1_block32_fp32acc: OCP MXFP4 weights (e2m1 elements, one ue8m0 scale per 32 elements) with bf16
# activations (gpt-oss-20b layout).  A scaled weight is an exact dyadic rational, so the only rounding is the fp32
# accumulation and the bf16 store.
_MXFP4_E2M1_BLOCK32_FP32ACC = RoundingContract(
    name="mxfp4_e2m1_block32_fp32acc", tol_class=ToleranceClass.BLOCK_MX, calibration=Calibration.SCHEMA_ONLY,
    accum_u_log2=-24, promote_k=0, split_k_max=1, split_partial_u_log2=-24,
    tp_reduce_max=1, store_u_log2=-8, elem_u_log2=-21, eta_log2=-126,
    p_u_log2=-8, attn_out_u_log2=-8, attn_key_tile=64, split_kv_max=8,
)

CONTRACTS: dict[str, RoundingContract] = {
    c.name: c for c in (_BF16_FP32ACC, _FP16_ATTN_STRICT, _FP8_E4M3_BLOCK128, _MXFP8_E4M3, _MXFP4_E2M1,
                        _FP8_E4M3_BLOCK128_FP32ACC, _MXFP4_E2M1_BLOCK32_FP32ACC)
}


def register_contract(c: RoundingContract, *, replace_existing: bool = False) -> RoundingContract:
    if c.name in CONTRACTS and not replace_existing:
        raise FormatError(f"contract {c.name!r} already registered")
    CONTRACTS[c.name] = c
    return c


def get_contract(name: str) -> RoundingContract:
    try:
        return CONTRACTS[name]
    except KeyError:
        raise FormatError(f"unknown contract {name!r}; known: {sorted(CONTRACTS)}") from None


# ============================================================================================================ model format
@dataclass(frozen=True)
class ModelFormat:
    name: str
    weight: NumFormat
    activation: NumFormat
    kv_cache: NumFormat
    residual: NumFormat
    gemm: RoundingContract
    attn: RoundingContract

    def contract_for(self, op: str) -> RoundingContract:
        if op == "gemm":
            return self.gemm
        if op == "attn":
            return self.attn
        raise FormatError(f"unknown operator family {op!r}")

    def weight_commit_ratio_vs(self, other: "ModelFormat") -> Fraction:
        """How many times smaller this model's weight commitment is than ``other``'s (storage bits per element)."""
        return other.weight.storage_bits_per_element / self.weight.storage_bits_per_element

    def check_within(self, cap: ProfileCap) -> None:
        self.gemm.assert_within(cap.gemm)
        self.attn.assert_within(cap.attn)

    def require_calibrated(self) -> "ModelFormat":
        self.gemm.require_calibrated()
        self.attn.require_calibrated()
        return self


_S_FP32_1x128 = QuantBlockScale("fp32", 1, 128)
_S_FP32_128x128 = QuantBlockScale("fp32", 128, 128)
_S_E8M0_32 = QuantBlockScale("e8m0", 1, 32)

_BF16 = NUM_FORMATS["bf16"]
_FP8_W = NUM_FORMATS["fp8_e4m3"].with_scale(_S_FP32_128x128, "fp8_e4m3_block128_w")
_FP8_A = NUM_FORMATS["fp8_e4m3"].with_scale(_S_FP32_1x128, "fp8_e4m3_block128_a")
_MXFP8 = NUM_FORMATS["fp8_e4m3"].with_scale(_S_E8M0_32, "mxfp8_e4m3")
_MXFP4 = NUM_FORMATS["fp4_e2m1"].with_scale(_S_E8M0_32, "mxfp4_e2m1")


def _mf(name: str, w: NumFormat, a: NumFormat, gemm: RoundingContract, attn: RoundingContract) -> ModelFormat:
    return ModelFormat(name, w, a, _BF16, _BF16, gemm, attn)


MODEL_FORMATS: dict[str, ModelFormat] = {
    m.name: m
    for m in (
        _mf("bf16", _BF16, _BF16, _BF16_FP32ACC, _BF16_FP32ACC),
        _mf("bf16_fp16attn", _BF16, _BF16, _BF16_FP32ACC, _FP16_ATTN_STRICT),
        _mf("fp8_e4m3_block128", _FP8_W, _FP8_A, _FP8_E4M3_BLOCK128, _BF16_FP32ACC),
        _mf("mxfp8_e4m3", _MXFP8, _MXFP8, _MXFP8_E4M3, _BF16_FP32ACC),
        _mf("mxfp4_e2m1", _MXFP4, _BF16, _MXFP4_E2M1, _BF16_FP32ACC),   # W4A16 (gpt-oss-style MoE weights)
        _mf("fp8_e4m3_block128_fp32acc", _FP8_W, _FP8_A, _FP8_E4M3_BLOCK128_FP32ACC, _BF16_FP32ACC),
        _mf("mxfp4_e2m1_block32_fp32acc", _MXFP4, _BF16, _MXFP4_E2M1_BLOCK32_FP32ACC, _BF16_FP32ACC),
    )
}


def get_model_format(name: str) -> ModelFormat:
    try:
        return MODEL_FORMATS[name]
    except KeyError:
        raise FormatError(f"unknown model format {name!r}; known: {sorted(MODEL_FORMATS)}") from None


# ============================================================================================================ per-GEMM bounds
@dataclass(frozen=True)
class GemmErrorBound:
    """``|y_hat - y| <= relative * <|w|,|x|> + operand_floor * scale * sum|w| + 2**eta_log2`` for one output element,
    where ``y`` is the exact dot product of the exactly scaled weights with the *unquantised* activations.

    ``arithmetic`` is the contract's dot template on the operands the kernel multiplies.  ``operand_store`` is the
    relative rounding of quantising the activation into its element format (0 when activations are consumed in the
    committed format).  ``operand_floor_log2`` is ``log2`` of the largest absolute activation rounding per unit block
    scale in the subnormal range (``None`` when there is no activation quantisation).  ``weight_exact`` is False when
    a scaled weight is wider than the accumulator significand, in which case ``weight_store`` is charged."""

    k: int
    arithmetic: Fraction
    operand_store: Fraction
    weight_store: Fraction
    relative: Fraction
    eta_log2: int
    operand_floor_log2: Optional[int]
    weight_exact: bool


def gemm_error_bound(mf: ModelFormat, k: int) -> GemmErrorBound:
    """Exact per-GEMM bound of a model format for an inner dimension ``k`` (activation quantisation included)."""
    c = mf.gemm
    arithmetic = c.dot_relative_bound(k)
    a, w = mf.activation, mf.weight
    quantised = a.scale is not None and a.kind == "float" and a.name != mf.residual.name
    operand = a.unit_roundoff if quantised else Fraction(0)
    weight_exact = w.lift_mantissa_bits <= -c.accum_u_log2
    weight = Fraction(0) if weight_exact else c.u_acc
    relative = (1 + arithmetic) * (1 + operand) * (1 + weight) - 1
    return GemmErrorBound(k, arithmetic, operand, weight, relative, c.eta_log2,
                          a.unit_log2 - 1 if quantised else None, weight_exact)


def exact_dequant_scale_exp_range(fmt: NumFormat, target: NumFormat) -> Optional[tuple[int, int]]:
    """Inclusive power-of-two scale exponents ``e`` for which *every* finite element of ``fmt`` times ``2**e`` is a
    normal number of ``target`` (so expanding the block is exact, with no subnormal or overflow case).  ``None``
    when the scale is not a power of two or the element significand does not fit."""
    if fmt.scale is None or not fmt.scale.pow2 or fmt.kind != "float" or target.kind != "float":
        return None
    if fmt.man_bits > target.man_bits:
        return None
    lo = (1 - target.bias) - fmt.unit_log2                       # smallest element 2**unit_log2 stays normal
    hi = 0
    while fmt.max_finite * (Fraction(2) ** (hi + 1)) <= target.max_finite:
        hi += 1
    declared = fmt.scale.scale_exp_range
    return (max(lo, declared[0]), min(hi, declared[1])) if declared else (lo, hi)


def weight_commitment(mf: ModelFormat, rows: int, cols: int) -> dict[str, Any]:
    """Commitment size of one ``rows x cols`` weight matrix: exact stored bytes, committed cells (elements plus
    block scales) and the signed integer width of an element on its own grid."""
    w = mf.weight
    blocks = w.block_count(rows, cols)
    return {
        "format": w.name,
        "elements": rows * cols,
        "scale_blocks": blocks,
        "bytes": w.storage_bytes_2d(rows, cols),
        "cells": rows * cols + blocks,
        "element_integer_bits": w.element_integer_bits,
        "scale_bits": w.scale.scale_bits if w.scale else 0,
        "bits_per_element": w.storage_bits_per_element,
    }


# Default public cap: the specs/06 §1.2 R32 entry for both operator families.
PROFILE_CAPS: dict[str, ProfileCap] = {
    "r32_default": ProfileCap("r32_default", _BF16_FP32ACC, _BF16_FP32ACC),
    "fp16_attn": ProfileCap("fp16_attn", _BF16_FP32ACC, _FP16_ATTN_STRICT),
}


# ============================================================================================================ kernel adapter
@dataclass(frozen=True)
class KernelAdapter:
    """Binds a serving-engine kernel family on an architecture to a contract.  Pure data: the proof side never
    branches on ``arch``; it only reads ``contract``."""

    name: str
    engine: str
    arch: str                    # informational ("sm_89", "sm_90", ...)
    op: str                      # "gemm" | "attn"
    weight_format: str
    activation_format: str
    contract: RoundingContract
    notes: str = ""

    def __post_init__(self) -> None:
        if self.op not in ("gemm", "attn"):
            raise FormatError(f"bad op {self.op!r}")

    def supports(self, mf: ModelFormat) -> bool:
        return (mf.weight.name == self.weight_format and mf.activation.name == self.activation_format
                and mf.contract_for(self.op) == self.contract)

    def satisfies(self, cap: ProfileCap) -> bool:
        return self.contract.within(cap.gemm if self.op == "gemm" else cap.attn)


KERNEL_ADAPTERS: dict[str, KernelAdapter] = {
    a.name: a
    for a in (
        KernelAdapter("vllm_cublas_bf16_r32", "vllm", "sm_89", "gemm", "bf16", "bf16", _BF16_FP32ACC,
                      "requires allow_bf16_reduced_precision_reduction=False"),
        KernelAdapter("vllm_fa2_bf16", "vllm", "sm_89", "attn", "bf16", "bf16", _BF16_FP32ACC),
        KernelAdapter("vllm_fa2_fp16_p", "vllm", "sm_89", "attn", "bf16", "bf16", _FP16_ATTN_STRICT,
                      "schema only: fp16 P/output tier not yet exercised"),
        KernelAdapter("vllm_deepgemm_fp8_block128", "vllm", "sm_90", "gemm", "fp8_e4m3_block128_w",
                      "fp8_e4m3_block128_a", _FP8_E4M3_BLOCK128, "schema only until M3-1 Hopper calibration"),
        KernelAdapter("vllm_mx_gemm_mxfp8", "vllm", "sm_100", "gemm", "mxfp8_e4m3", "mxfp8_e4m3", _MXFP8_E4M3,
                      "schema only"),
        KernelAdapter("vllm_mx_gemm_mxfp4", "vllm", "sm_90", "gemm", "mxfp4_e2m1", "bf16", _MXFP4_E2M1,
                      "schema only"),
        KernelAdapter("vllm_fp8_block128_fp32acc_sm90", "vllm", "sm_90", "gemm", "fp8_e4m3_block128_w",
                      "fp8_e4m3_block128_a", _FP8_E4M3_BLOCK128_FP32ACC, "schema only until M3-1 calibration"),
        KernelAdapter("vllm_fp8_block128_fp32acc_sm120", "vllm", "sm_120", "gemm", "fp8_e4m3_block128_w",
                      "fp8_e4m3_block128_a", _FP8_E4M3_BLOCK128_FP32ACC, "schema only until M3-1 calibration"),
        KernelAdapter("vllm_mxfp4_block32_fp32acc_sm90", "vllm", "sm_90", "gemm", "mxfp4_e2m1", "bf16",
                      _MXFP4_E2M1_BLOCK32_FP32ACC, "schema only until M3-1 calibration"),
        KernelAdapter("vllm_mxfp4_block32_fp32acc_sm120", "vllm", "sm_120", "gemm", "mxfp4_e2m1", "bf16",
                      _MXFP4_E2M1_BLOCK32_FP32ACC, "schema only until M3-1 calibration"),
    )
}
