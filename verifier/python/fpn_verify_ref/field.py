# SPDX-License-Identifier: Apache-2.0
"""Goldilocks F_p (p = 2^64 - 2^32 + 1) and E = F_p[u]/(u^3 - 2), exact Python-int arithmetic.

Independent re-implementation (specs/11-wire-format.md §1); never imports the prover package.
Fp elements are ints in [0, P); Fp3 elements are 3-tuples (limb 0 = constant term).
"""

P = (1 << 64) - (1 << 32) + 1

ZERO3 = (0, 0, 0)
ONE3 = (1, 0, 0)


class NonCanonical(ValueError):
    """A field encoding with a limb >= P."""


# ----------------------------------------------------------------------------- Fp
def fp_add(a, b): return (a + b) % P
def fp_sub(a, b): return (a - b) % P
def fp_neg(a): return (-a) % P
def fp_mul(a, b): return (a * b) % P
def fp_pow(a, e): return pow(a, e, P)


def fp_inv(a):
    if a % P == 0:
        raise ZeroDivisionError("inverse of zero in F_p")
    return pow(a, P - 2, P)


def fp_encode(x):
    if not 0 <= x < P:
        raise NonCanonical("Fp value out of range")
    return x.to_bytes(8, "little")


def fp_decode(b):
    if len(b) != 8:
        raise ValueError("Fp encoding must be 8 bytes")
    x = int.from_bytes(b, "little")
    if x >= P:
        raise NonCanonical("Fp limb >= P")
    return x


# ----------------------------------------------------------------------------- Fp3
def e_add(a, b): return ((a[0] + b[0]) % P, (a[1] + b[1]) % P, (a[2] + b[2]) % P)
def e_sub(a, b): return ((a[0] - b[0]) % P, (a[1] - b[1]) % P, (a[2] - b[2]) % P)
def e_neg(a): return ((-a[0]) % P, (-a[1]) % P, (-a[2]) % P)
def e_from_fp(x): return (x % P, 0, 0)


def e_mul(a, b):
    return ((a[0] * b[0] + 2 * (a[1] * b[2] + a[2] * b[1])) % P,
            (a[0] * b[1] + a[1] * b[0] + 2 * a[2] * b[2]) % P,
            (a[0] * b[2] + a[1] * b[1] + a[2] * b[0]) % P)


def e_pow(a, e):
    r = ONE3
    while e:
        if e & 1:
            r = e_mul(r, a)
        a = e_mul(a, a)
        e >>= 1
    return r


def e_inv(a):
    t0 = (a[0] * a[0] - 2 * a[1] * a[2]) % P
    t1 = (2 * a[2] * a[2] - a[0] * a[1]) % P
    t2 = (a[1] * a[1] - a[0] * a[2]) % P
    n = (a[0] * t0 + 2 * (a[2] * t1 + a[1] * t2)) % P
    if n == 0:
        raise ZeroDivisionError("inverse of zero in E")
    ni = pow(n, P - 2, P)
    return (t0 * ni % P, t1 * ni % P, t2 * ni % P)


def e_encode(a):
    return b"".join(fp_encode(x) for x in a)


def e_decode(b):
    if len(b) != 24:
        raise ValueError("Fp3 encoding must be 24 bytes")
    return tuple(fp_decode(b[8 * i:8 * i + 8]) for i in range(3))


def lagrange_eval(vals, r):
    """Degree-<=d polynomial with g(j) = vals[j] (j = 0..d), evaluated at r in E."""
    d = len(vals) - 1
    out = ZERO3
    for i in range(d + 1):
        num, den = ONE3, 1
        for j in range(d + 1):
            if j != i:
                num = e_mul(num, e_sub(r, (j, 0, 0)))
                den = den * (i - j) % P
        out = e_add(out, e_mul(e_mul(vals[i], num), (pow(den, P - 2, P), 0, 0)))
    return out


def eq_pt(a, b):
    out = ONE3
    for x, y in zip(a, b):
        out = e_mul(out, e_add(e_mul(x, y), e_mul(e_sub(ONE3, x), e_sub(ONE3, y))))
    return out
