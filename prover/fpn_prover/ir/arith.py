"""Integer reference gadgets. Large magnitudes use a separate power of two."""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from .gadget_ir import LookupSpec, expression


@dataclass(frozen=True)
class BF16:
    sign: int
    exponent: int
    mantissa: int

    @property
    def significand(self):
        return self.mantissa + (128 if self.exponent else 0)

    @property
    def power(self):
        return max(1, self.exponent) - 134

    @property
    def value(self):
        return (-1)**self.sign * self.significand * Fraction(2)**self.power


def bf16_decode(word):
    if type(word) is not int or not 0 <= word < (1 << 16):
        raise ValueError("expected an unsigned BF16 word")
    result = BF16(word >> 15, (word >> 7) & 255, word & 127)
    if result.exponent == 255:
        raise ValueError("nonfinite BF16")
    return result


def install_bf16(program, prefix="bf16"):
    fields = (
        ("word", 17), ("sign", 2), ("exponent", 9), ("mantissa", 8),
        ("significand", 9), ("power", 9),
    )
    names = tuple(f"{prefix}.{name}" for name, _ in fields)
    word, sign, exp, mant, _, _ = [
        program.wire(name, bits, "BF16 component", prefix)
        for name, (_, bits) in zip(names, fields)
    ]
    program.equal(prefix + ".pack", word, sign * 32768 + exp * 128 + mant)
    program.lookups.append(LookupSpec(
        prefix + ".finite", names[1:],
        lambda s, e, m, significand, power: (
            s in (0, 1) and 0 <= e < 255 and 0 <= m < 128
            and significand == m + 128 * bool(e) and power == max(1, e) - 134
        ),
    ))
    return names


@dataclass(frozen=True)
class Rsqrt:
    k: int
    u: Fraction
    low: int
    high: int
    scale: int

    @property
    def enclosure(self):
        factor = Fraction(2)**(-self.k) / self.scale
        return self.low * factor, self.high * factor

    def accepts(self, x):
        return (
            self.scale > 0 and 1 <= self.u < 4
            and x == self.u * Fraction(2)**(2*self.k)
            and 0 <= self.low <= self.high <= self.scale
            and self.high - self.low <= 1
            and self.u * self.low**2 <= self.scale**2 <= self.u * self.high**2
        )


def norm_rsqrt(x, precision=24):
    x = Fraction(x)
    if x <= 0 or type(precision) is not int or precision < 1:
        raise ValueError("expected positive radicand and precision")
    k = (x.numerator.bit_length() - x.denominator.bit_length()) // 2
    u = x / Fraction(2)**(2*k)
    if u < 1:
        k -= 1
        u *= 4
    scale = 1 << precision
    low = isqrt((scale**2 * u.denominator) // u.numerator)
    high = low + (u * low**2 != scale**2)
    return Rsqrt(k, u, low, high, scale)


def signed_range_check(program, name, bits, limb_bits=8):
    """Offset-binary decomposition, deduplicated per wire and limb layout."""
    if (not 1 <= bits <= 60 or not 1 <= limb_bits <= bits
            or program.columns[name].signed_bound_bits < bits
            or program.columns[name].signed_bound_bits > 61):
        raise ValueError("unsupported signed range")
    offset = 1 << (bits - 1)
    prefix = f"{name}.range{bits}x{limb_bits}"
    names = tuple(f"{prefix}.{shift}" for shift in range(0, bits, limb_bits))
    if any(eq.name == prefix for eq in program.constraints):
        return names
    total = expression(0)
    for shift, limb in zip(range(0, bits, limb_bits), names):
        width = min(limb_bits, bits - shift)
        wire = program.wire(limb, width + 1, "unsigned limb", prefix)
        program.lookups.append(LookupSpec(limb + ".range", (limb,),
                                          lambda value, width=width: 0 <= value < 1 << width))
        total = total + wire * (1 << shift)
    program.equal(prefix, expression(0) + offset + program.wire(
        name, program.columns[name].signed_bound_bits, program.columns[name].semantic_type,
        program.columns[name].producer, program.columns[name].freedom_kind,
    ), total)
    return names


def install_rsqrt(program, prefix="rsqrt", precision=16):
    """Reference lowering for dyadic inputs with up to 22 significand bits."""
    if not 1 <= precision <= 16:
        raise ValueError("unsupported circuit precision")
    scale = 1 << precision
    fields = {
        "coefficient": 23, "power": 24, "k": 24, "u_num": 23, "u_den": 23,
        "low": precision+2, "high": precision+2, "output_power": 24,
        "lower_slack": 56, "upper_slack": 56,
    }
    wires = {key: program.wire(f"{prefix}.{key}", bits, "rsqrt component", prefix)
             for key, bits in fields.items()}
    c, p, k, n, d, low, high, out, lower_slack, upper_slack = wires.values()
    program.equal(prefix + ".lower", n * low * low + lower_slack, d * scale**2)
    program.equal(prefix + ".upper", d * scale**2 + upper_slack, n * high * high)
    program.equal(prefix + ".rescale", out + k, -precision)

    def normalized(coefficient, power, k, numerator, denominator):
        if not (coefficient > 0 and coefficient % 2 and denominator > 0
                and numerator == coefficient and denominator <= numerator < 4*denominator):
            return False
        return denominator.bit_count() == 1 and 2*k == power + denominator.bit_length() - 1

    program.lookups.append(LookupSpec(
        prefix + ".normalize", tuple(f"{prefix}.{key}" for key in list(fields)[:5]), normalized,
    ))
    program.lookups.append(LookupSpec(
        prefix + ".tight",
        tuple(f"{prefix}.{key}" for key in ("low", "high", "lower_slack", "upper_slack")),
        lambda l, h, a, b: (
            0 <= l <= h <= scale and a >= 0 and b >= 0
            and ((h == l and a == b == 0) or (h == l+1 and a > 0 and b > 0))
        ),
    ))
    return tuple(f"{prefix}.{key}" for key in fields)
