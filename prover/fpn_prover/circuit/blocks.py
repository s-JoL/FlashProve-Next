"""Scalar reference relations for sampled Transformer blocks.

Lookup predicates are interpreted by Program. Their columns are materialized
here, but the predicates still require a checked backend translation.
Conditions lifted to `program.equal` gates (see `gemm_tile_row_block` and
`circuit/lower.py`) are NOT re-checked by the remaining predicates.
"""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

from ..ir.arith import bf16_decode, norm_rsqrt
from ..ir.format import get_contract
from ..ir.gadget_ir import AUX_BITS, LookupSpec

RESIDUAL_RMS_FLOOR = Fraction(1, 1000)
LINEAGE_DEPTH = 2
OBSERVATION = "FULL_NO_KV"
SILU_FLOOR = Fraction(2) ** -120


def _wires(program, prefix, label, count, bits):
    return tuple(program.wire(f"{prefix}.{label}.{i}", bits, label, prefix)
                 for i in range(count))


def _names(wires):
    return tuple(wire.args[0] for wire in wires)


def _lookup(program, name, wires, predicate):
    program.lookups.append(LookupSpec(name, _names(wires), predicate))


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right, strict=True))


def rmsnorm_block(program, prefix, width, *, epsilon=Fraction(1, 10**6),
                  contract=None, residual=True):
    """BF16 inputs, gains and observations with normalized rsqrt endpoint cells."""
    if width < 1 or epsilon <= 0:
        raise ValueError("invalid RMSNorm dimensions or epsilon")
    contract = contract or get_contract("bf16_fp32acc")
    x = _wires(program, prefix, "input", width, 17)
    gain = _wires(program, prefix, "gain", width, 17)
    y = _wires(program, prefix, "output", width, 17)
    k, = _wires(program, prefix, "power", 1, 12)
    lower, upper = _wires(program, prefix, "endpoint", 2, 26)
    span, = _wires(program, prefix, "span", 1, 2)
    program.equal(prefix + ".span", upper - lower, span)
    floor_sq = width * RESIDUAL_RMS_FLOOR**2 if residual else 0
    acc_contract = replace(contract, store_u_log2=contract.accum_u_log2)
    accumulation = acc_contract.dot_relative_bound(width)
    u = (1 + Fraction(2)**contract.elem_u_log2)**3 * (1+contract.u_store) - 1
    eta = Fraction(2)**contract.eta_log2

    def accepts(*row):
        try:
            inputs, gains, outputs = tuple(
                tuple(bf16_decode(w).value for w in row[i:i+width])
                for i in (0, width, 2*width)
            )
        except ValueError:
            return False
        squares = _dot(inputs, inputs)
        if squares < floor_sq:
            return False
        ref = norm_rsqrt(squares / width + epsilon)
        if row[-3:] != (ref.k, ref.low, ref.high):
            return False
        mean = squares / width
        low_rad = max(epsilon, mean*(1-accumulation)+epsilon-eta)
        high_rad = mean*(1+accumulation)+epsilon+eta
        lo = norm_rsqrt(high_rad).enclosure[0]
        hi = norm_rsqrt(low_rad).enclosure[1]
        for a, g, observed in zip(inputs, gains, outputs, strict=True):
            low, high = sorted((a*g*lo, a*g*hi))
            radius = max(abs(low), abs(high))*u + eta
            if not low-radius <= observed <= high+radius:
                return False
        return True

    _lookup(program, prefix + ".norm", x + gain + y + (k, lower, upper), accepts)
    return _names(x + gain + y + (k, lower, upper, span))


def gemm_tile_row_block(program, prefix, inner, rows, contract, *, unit=1, bits=24, operand_bits=62):
    """Check a contiguous tile's center and radius products in integer units.

    Weight rows and input boxes are opening inputs. X and Y have the same
    physical unit when W is integer-valued. The additive store floor is
    converted into Y units.

    Algebraic gates (relation-4 Plonkish rows; `<p>` is the prefix):
      * `<p>.input_radius_nonneg.<i>`      r_x,i >= 0 (slack wire);
      * `<p>.weight_radius_upper.<r>.<i>`  wr - w >= 0 and `<p>.weight_radius_lower.<r>.<i>`  wr + w >= 0,
        i.e. wr >= |w| (two slack wires);
      * `<p>.center_dot.<r>`  center = x . w  and  `<p>.difference.<r>`  observed - center = delta;
      * `<p>.abs_square.<i>`  a_i^2 = x_i^2 for the range-checked non-negative cell a_i (so a_i = |x_i|) and
        `<p>.operand.<i>`  z_i = num a_i + (den + num) r_x,i -- ONLY when that gate keeps the 2^62 signed
        lift bound over the declared ranges (`Program.fits`: true for the mxfp4 / bf16 contracts at any
        operand width, for fp8_e4m3_block128 only at a narrowed operand); otherwise the operand stays
        the reference predicate `<p>.operand.<i>`;
      * `<p>.radius_dot.<r>`  radius_numerator = z . wr -- ONLY when the declared operand width makes the
        product fit the lift bound (inner * 2^(`operand_bits` + `bits` - 2) < 2^61, e.g. `operand_bits=38`
        at inner = 1, 37 at inner = 2); at the default 62-bit operand -- what every caller other than the
        relation-4 demo circuit uses -- it stays the reference predicate `<p>.products.<r>`.
    Reference predicate that always remains: `<p>.box.<r>`  |delta| den <= radius_numerator + eta den
    (algebraic, but a 61-bit delta times den exceeds the lift bound and the slack would not fit a wire).
    """
    if inner < 1 or rows < 1 or unit <= 0 or not 1 <= operand_bits <= 62:
        raise ValueError("invalid tile dimensions or unit")
    x = _wires(program, prefix, "input", inner, bits)
    rx = _wires(program, prefix, "input_radius", inner, bits)
    z = _wires(program, prefix, "radius_operand", inner, operand_bits)
    gamma = contract.dot_relative_bound(inner)
    if gamma <= 0:
        raise ValueError("radius product requires a positive rounding coefficient")
    top = (1 << (bits - 1)) - 1
    num, den = gamma.numerator, gamma.denominator
    for i, (value, radius, scaled) in enumerate(zip(x, rx, z, strict=True)):
        program.nonneg(f"{prefix}.input_radius_nonneg.{i}", radius, top, prefix)
        abs_bits = AUX_BITS * -(-bits // AUX_BITS)
        half = 1 << (abs_bits - 1)
        if program.fits(scaled - (2*half - 1)*num - radius*(den + num)):
            # a = cell + half is range-checked into [0, 2^abs_bits); a^2 = x^2 then forces a = |x|.
            cell = program.aux(f"{prefix}.abs.{i}", abs_bits, "absolute value", prefix,
                               lambda w, name=value.args[0], half=half: abs(w[name]) - half)
            magnitude = cell + half
            program.equal(f"{prefix}.abs_square.{i}", magnitude*magnitude - value*value)
            program.equal(f"{prefix}.operand.{i}", scaled - magnitude*num - radius*(den + num))
        else:
            _lookup(program, f"{prefix}.operand.{i}", (value, radius, scaled),
                    lambda a, r, b: b == num*abs(a) + (den + num)*r)
    eta = Fraction(2)**contract.eta_log2 / Fraction(unit)
    result = x + rx + z
    for i in range(rows):
        w = _wires(program, prefix, f"weight.{i}", inner, bits)
        wr = _wires(program, prefix, f"radius_weight.{i}", inner, bits)
        center, = _wires(program, prefix, f"center.{i}", 1, 61)
        radius, = _wires(program, prefix, f"radius_numerator.{i}", 1, 62)
        observed, = _wires(program, prefix, f"observed.{i}", 1, 61)
        box_delta, = _wires(program, prefix, f"delta.{i}", 1, 61)
        for j, (a, b) in enumerate(zip(w, wr, strict=True)):
            program.nonneg(f"{prefix}.weight_radius_upper.{i}.{j}", b - a, 2*top + 1, prefix)
            program.nonneg(f"{prefix}.weight_radius_lower.{i}.{j}", b + a, 2*top + 1, prefix)
        # Interval bound of radius - z . wr over the declared ranges, in closed form (the expression is
        # `inner` terms deep, far past the recursion limit at production widths).
        if (1 << 61) + inner * (1 << (operand_bits - 1)) * (1 << (bits - 1)) < (1 << 62):
            radius_dot = radius
            for a, b in zip(z, wr, strict=True):
                radius_dot = radius_dot - a*b
            program.equal(f"{prefix}.radius_dot.{i}", radius_dot)
        else:
            _lookup(program, f"{prefix}.products.{i}", z + wr + (radius,),
                    lambda *a: a[-1] == _dot(a[:inner], a[inner:2*inner]))
        dot = center
        for a, b in zip(x, w, strict=True):
            dot = dot - a*b
        program.equal(f"{prefix}.center_dot.{i}", dot)
        program.equal(f"{prefix}.difference.{i}", observed-center, box_delta)
        _lookup(program, f"{prefix}.box.{i}", (box_delta, radius),
                lambda d, a: abs(d) <= Fraction(a, den) + eta)
        result += w + wr + (center, radius, observed, box_delta)
    return _names(result)


def _transcendental(value, operation):
    """Enclose a monotone Decimal exp/ln with adjacent representable endpoints."""
    with localcontext() as ctx:
        ctx.prec = 70
        ctx.Emin, ctx.Emax = -200, 200
        a = Decimal(value.numerator) / Decimal(value.denominator)
        lo = getattr(a.next_minus(), operation)()
        hi = getattr(a.next_plus(), operation)()
        return Fraction(lo.next_minus()), Fraction(hi.next_plus())


def _exp(x):
    # exp(-512) < 10**-222, below the reference context's useful range.
    if not x:
        return Fraction(1), Fraction(1)
    if x <= -512:
        return Fraction(0), Fraction(1, 10**222)
    lo, hi = _transcendental(x, "exp")
    return max(Fraction(0), lo), hi


def _silu(x):
    low, high = _exp(-abs(x))
    if x >= 0:
        return x/(1+high), x/(1+low)
    return x*high/(1+high), x*low/(1+low)


def swiglu_block(program, prefix, width, contract):
    """Independently observed BF16 SiLU and Hadamard stores."""
    if width < 1:
        raise ValueError("empty SwiGLU")
    u = Fraction(2)**contract.elem_u_log2
    eta = Fraction(2)**contract.eta_log2

    def accepts(gate, up, silu, output):
        try:
            g, v, s, y = (bf16_decode(w).value for w in (gate, up, silu, output))
        except ValueError:
            return False
        low, high = _silu(g)
        tolerance = max(abs(low), abs(high)) * ((1+u)*(1+contract.u_store)-1)
        tolerance += (1+contract.u_store)*SILU_FLOOR + eta
        return (low-tolerance <= s <= high+tolerance
                and abs(y-s*v) <= abs(s*v)*contract.u_store + eta)

    wires = tuple(_wires(program, prefix, role, width, 17)
                  for role in ("gate", "up", "silu", "output"))
    for i, row in enumerate(zip(*wires, strict=True)):
        _lookup(program, f"{prefix}.gate.{i}", row, accepts)
    return _names(sum(wires, ()))


def _attention_box(q, keys, values, contract, scale):
    scores = [_dot(q, key)*scale for key in keys]
    acc_contract = replace(contract, store_u_log2=contract.accum_u_log2)
    dot_error = acc_contract.dot_relative_bound(len(q))
    score_error = [
        sum(abs(a*b) for a, b in zip(q, key, strict=True))*scale
        * dot_error
        for key in keys
    ]
    eta = Fraction(2)**contract.eta_log2
    score_error = [error + eta for error in score_error]
    maximum = max(scores)
    weights = [_exp(score-maximum) for score in scores]
    low_sum = sum(low for low, _ in weights)
    high_sum = sum(high for _, high in weights)
    n = len(keys)
    tiles = -(-n // contract.attn_key_tile) + contract.split_kv_max
    elem = Fraction(2)**contract.elem_u_log2
    common = (1+elem)**(tiles+1)
    accum = (1+contract.u_acc)**(2*tiles+contract.depth(n))
    probability = Fraction(2)**contract.p_u_log2
    store = Fraction(2)**contract.attn_out_u_log2
    # Bound each shifted weight using the actual absolute QK product sum.
    # The shift is shared, so the LSE comparison uses score error once.
    weight_low = [_exp(s-maximum-e)[0] for s, e in zip(scores, score_error, strict=True)]
    upper_center = max(s+e for s, e in zip(scores, score_error, strict=True))
    weight_high = [_exp(s+e-upper_center)[1]
                   for s, e in zip(scores, score_error, strict=True)]
    score_slack = max(score_error)
    shift_factor = _exp(-score_slack)[0]
    if not shift_factor:
        return None
    additive = n*(tiles+1)*common*accum*eta
    normalizer_low = sum(weight_low)/(common*accum) - additive/shift_factor
    if normalizer_low <= 0:
        return None
    low = maximum + _transcendental(normalizer_low, "ln")[0]
    high = upper_center + _transcendental(sum(weight_high)*common*accum+additive, "ln")[1]
    lower_center = max(s-e for s, e in zip(scores, score_error, strict=True))
    log_error = max(abs(low-upper_center), abs(high-lower_center))*elem
    store_error = max(abs(low), abs(high))*contract.u_acc
    lse = low-log_error-store_error-eta, high+log_error+store_error+eta

    # For a nontrivial score interval use the shared-factor softmax bound.
    # Reject intervals whose denominator lower bound cannot be certified.
    sigma = _exp(-2*score_slack)[0]
    if not sigma:
        return None
    den_error = common*accum/sigma - 1
    num_error = (1+probability)*common*accum/sigma - 1
    # The fp16 tier assumes gradual underflow in the P store.
    eta_p = Fraction(2)**(-25 if contract.p_u_log2 == -11 else -126)
    eta_out = Fraction(2)**-25 if contract.attn_out_u_log2 == -11 else eta
    eta_weight = ((tiles+1)*common*accum*eta+eta_p)/shift_factor
    den_floor = 1-den_error-n*eta_weight
    if den_floor <= 0:
        return None
    boxes = []
    for vector in zip(*values, strict=True):
        low_num = sum(min(a*v, b*v) for (a, b), v in zip(weights, vector, strict=True))
        high_num = sum(max(a*v, b*v) for (a, b), v in zip(weights, vector, strict=True))
        candidates = [a/b for a in (low_num, high_num) for b in (low_sum, high_sum)]
        lo, hi = min(candidates), max(candidates)
        abs_center = max(abs(lo), abs(hi))
        variation = sum(b*max(abs(v-lo), abs(v-hi))
                        for (_, b), v in zip(weights, vector, strict=True))/low_sum
        radius = (num_error*variation + (num_error+den_error)*abs_center
                  + eta_weight*(sum(map(abs, vector))+n*abs_center))/den_floor
        radius += tiles*eta
        radius += (abs_center+radius)*((1+store)*(1+contract.u_acc)-1)+eta_out
        boxes.append((lo-radius, hi+radius))
    return lse, boxes


def gqa_group_attn_block(program, prefix, heads, tokens, key_width, value_width,
                         contract=None, *, unit=Fraction(1, 2**16), score_scale=1):
    """A single KV group with shared K/V and one output/LSE per query head.

    Inputs are exact fixed-point openings. The profile supplies a positive
    rational score scale and the active key set in canonical order.
    """
    contract = contract or get_contract("fp16_attn_strict")
    if (min(heads, tokens, key_width, value_width) < 1 or unit <= 0
            or score_scale <= 0 or contract.p_u_log2 not in (-8, -11)
            or contract.attn_out_u_log2 not in (-8, -11)
            or contract.attn_key_tile < 1):
        raise ValueError("invalid attention profile")
    unit = Fraction(unit)
    scale = Fraction(score_scale)*unit**2
    k = _wires(program, prefix, "key", tokens*key_width, 32)
    v = _wires(program, prefix, "value", tokens*value_width, 32)
    q = _wires(program, prefix, "query", heads*key_width, 32)
    y = _wires(program, prefix, "output", heads*value_width, 32)
    lse = _wires(program, prefix, "lse", heads, 32)

    def check(*row):
        q_row = row[:key_width]
        offset = key_width
        keys = tuple(row[i:i+key_width] for i in range(offset, offset+len(k), key_width))
        offset += len(k)
        values = tuple(tuple(x*unit for x in row[i:i+value_width])
                       for i in range(offset, offset+len(v), value_width))
        checked = _attention_box(q_row, keys, values, contract, scale)
        if checked is None:
            return False
        (lo, hi), outputs = checked
        return (lo <= row[-1]*unit <= hi
                and all(a <= x*unit <= b
                        for x, (a, b) in zip(row[-1-value_width:-1], outputs, strict=True)))

    for head in range(heads):
        _lookup(program, f"{prefix}.head.{head}",
                q[head*key_width:(head+1)*key_width] + k + v
                + y[head*value_width:(head+1)*value_width] + (lse[head],), check)
    return _names(k + v + q + y + lse)
