# SPDX-License-Identifier: Apache-2.0
"""Reference bf16 / fp32 arithmetic of a dense decoder engine: bf16 storage with round-to-nearest-even, fp32 accumulation.

These helpers define the engine semantics that the certified-tolerance gadgets (``flashprove_next.certified``) bound:
every activation is stored as a bf16 code (``np.uint16``) and every matrix product, normalization and activation is
evaluated in fp32.  They are shared by the deterministic test engine (``dense_engine.TinyDense``), the reply prover
(rotary tables) and the proof-bundle profile builder.  NumPy only.
"""
import numpy as np

EPS = float(np.float32(1e-6))
EPS_BITS = int(np.array([1e-6], dtype=np.float32).view(np.uint32)[0])


def bf16(x):
    """fp32 values -> bf16 codes (np.uint16), round to nearest even."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f32_of_bf16(c):
    """bf16 codes -> fp32 values (exact)."""
    return (np.asarray(c, dtype=np.uint32) << 16).view(np.float32)


def f32codes(x):
    """fp32 values -> their IEEE-754 bit patterns (np.uint32)."""
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def W32(codes):
    """bf16 weight codes -> fp32 matrix."""
    return f32_of_bf16(codes).astype(np.float32)


def rnd(rng, shape, scale=1.0):
    """Gaussian bf16 codes of the given shape and scale."""
    return bf16(rng.standard_normal(shape).astype(np.float32) * scale)


def eng_gemm_res(W, x, h):
    """bf16(bf16(W x) + h): projection followed by the residual add."""
    y = bf16(W32(W) @ f32_of_bf16(x))
    return bf16(f32_of_bf16(y) + f32_of_bf16(h))


def eng_norm(h, g, eps):
    """RMSNorm of one bf16 vector with gain g."""
    hf = f32_of_bf16(h)
    s = np.float32(1.0) / np.sqrt(np.mean(hf * hf, dtype=np.float32) + np.float32(eps))
    return bf16(f32_of_bf16(bf16(hf * s)) * f32_of_bf16(g))


def eng_gu(Wg, Wu, x):
    """SwiGLU: bf16(silu(bf16(Wg x)) * bf16(Wu x))."""
    xf = f32_of_bf16(x)
    g = f32_of_bf16(bf16(W32(Wg) @ xf))
    u = f32_of_bf16(bf16(W32(Wu) @ xf))
    s = f32_of_bf16(bf16(g / (np.float32(1) + np.exp(-g))))
    return bf16(s * u)


def eng_head(W, x):
    """LM head: fp32 logits as bit patterns (np.uint32)."""
    return f32codes(W32(W) @ f32_of_bf16(x))


def rmsnorm_eng(xb, g, eps):
    """RMSNorm over the last axis (batched over heads)."""
    xf = f32_of_bf16(xb)
    s = np.float32(1.0) / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps))
    return bf16(f32_of_bf16(bf16(xf * s)) * f32_of_bf16(g))


def rope_eng(xb, cos, sin):
    """Rotary embedding in the half-split (NeoX) layout."""
    xf = f32_of_bf16(xb)
    h = xf.shape[-1] // 2
    rot = np.concatenate([-xf[..., h:], xf[..., :h]], axis=-1)
    return bf16(f32_of_bf16(bf16(xf * cos)) + f32_of_bf16(bf16(rot * sin)))


def _dec_cos_sin(x_f64: float) -> tuple[object, object]:
    """60-digit Decimal (cos(x), sin(x)) for exact platform-independent bf16 tie-breaking."""
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 64
        twopi = Decimal("6.283185307179586476925286766559005768394338798750211641949889185")
        x = Decimal.from_float(float(x_f64)) % twopi
        x2 = x * x
        tc, c = Decimal(1), Decimal(1)
        ts, s = x, x
        for k in range(1, 48):
            tc = -tc * x2 / Decimal((2 * k - 1) * (2 * k))
            c += tc
            ts = -ts * x2 / Decimal((2 * k) * (2 * k + 1))
            s += ts
        return +c, +s


def _dec_to_bf16_code(d) -> int:
    """Exact IEEE-754 round-to-nearest-even of a Decimal `d` (`|d| <= 2`) into a uint16 bfloat16 code."""
    import math
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 64
        if d == 0:
            return 0
        sgn = 1 if d < 0 else 0
        a = -d if sgn else d
        e = int(math.floor(math.log2(float(a))))
        two = Decimal(2)
        while a >= (two ** (e + 1)):
            e += 1
        while a < (two ** e):
            e -= 1
        if e < -126:
            m = int((a * (two ** 133)).to_integral_value(rounding="ROUND_HALF_EVEN"))
            if m == 0:
                return sgn << 15
            if m == 128:
                return (sgn << 15) | (1 << 7)
            return (sgn << 15) | m
        scale = two ** (7 - e) if 7 - e >= 0 else Decimal(1) / (two ** (e - 7))
        sig = int((a * scale).to_integral_value(rounding="ROUND_HALF_EVEN"))
        if sig == 256:
            e += 1
            sig = 128
        return (sgn << 15) | ((e + 127) << 7) | (sig - 128)


def _exact_bf16_cos_sin(ang_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Platform-independent bf16 cos/sin of a 1-D float32 angle vector (disambiguates near-ties via 64-digit Decimal)."""
    ang64 = np.asarray(ang_f32, dtype=np.float64)
    c64 = np.cos(ang64)
    s64 = np.sin(ang64)
    c32 = c64.astype(np.float32)
    s32 = s64.astype(np.float32)
    cu = bf16(c32).copy()
    su = bf16(s32).copy()
    c_lo = c32.view(np.uint32) & 0xFFFF
    s_lo = s32.view(np.uint32) & 0xFFFF
    near = np.nonzero((np.abs(c_lo.astype(np.int64) - 0x8000) <= 16) | (np.abs(s_lo.astype(np.int64) - 0x8000) <= 16))[0]
    for idx in near:
        dc, ds = _dec_cos_sin(float(ang64[int(idx)]))
        cu[int(idx)] = _dec_to_bf16_code(dc)
        su[int(idx)] = _dec_to_bf16_code(ds)
    return cu, su


def _exact_rope_inv_f32(dh: int, base: float) -> np.ndarray:
    """Platform-independent float32 inverse-frequency vector `float32(1 / float32(base) ** float32(2j / dh))`."""
    import math
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 64
        two = Decimal(2)
        b_dec = Decimal.from_float(float(np.float32(base)))
        out = np.empty(dh // 2, dtype=np.float32)
        for k, i in enumerate(range(0, dh, 2)):
            ex_f32 = np.float32(i) / np.float32(dh)
            # Exact real value of b_dec ** ex_f32, rounded to IEEE binary32, then 1 / denom rounded to binary32
            val_pow = (b_dec.ln() * Decimal.from_float(float(ex_f32))).exp()
            # Round val_pow to exact IEEE binary32
            ep = int(math.floor(math.log2(float(val_pow))))
            while val_pow >= (two ** (ep + 1)):
                ep += 1
            while val_pow < (two ** ep):
                ep -= 1
            sc_p = two ** (23 - ep) if 23 - ep >= 0 else Decimal(1) / (two ** (ep - 23))
            sig_p = int((val_pow * sc_p).to_integral_value(rounding="ROUND_HALF_EVEN"))
            if sig_p == (1 << 24):
                ep += 1
                sig_p = 1 << 23
            pow_u32 = np.uint32(((ep + 127) << 23) | (sig_p - (1 << 23)))
            denom_f32 = pow_u32.view(np.float32)
            out[k] = np.float32(1.0) / denom_f32
        return out


def rope_table(pos, dh, base=1e6):
    """Reference rotary row of position ``pos``: cos and sin rounded to bf16, returned as fp32."""
    inv = _exact_rope_inv_f32(int(dh), float(base))
    ang = np.float32(pos) * inv
    cu, su = _exact_bf16_cos_sin(ang)
    cu2 = np.concatenate([cu, cu])
    su2 = np.concatenate([su, su])
    return f32_of_bf16(cu2), f32_of_bf16(su2)
