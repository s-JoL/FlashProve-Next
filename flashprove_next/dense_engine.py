# SPDX-License-Identifier: Apache-2.0
"""Deterministic numpy bf16/fp32 "engine" for a Qwen3-style dense decoder (plus optional Llama-style qk_norm=False).

Follows the exact fp32-accumulation / bf16-store rounding contract of ``flashprove_next.certified`` through the reference
helpers in ``flashprove_next.fp_ref``.  Exposes every intermediate per-port activation code as numpy arrays
indexed ``[t, l, :]`` so ZK reply provers and certified-tolerance evaluators can be tested end-to-end.
"""
from dataclasses import dataclass

import numpy as np

from .fp_ref import (
    EPS,
    EPS_BITS,
    W32,
    bf16,
    eng_gemm_res,
    eng_gu,
    eng_head,
    f32_of_bf16,
    f32codes,
    rnd,
    rmsnorm_eng,
    rope_eng,
    rope_table,
)
from .wsum_zk import int4_codes


@dataclass
class Capture:
    """Per-port activation codes captured from ``TinyDense.forward``.

    Layer ports have shape ``(T, L, *)``; final ports ``x_fn`` and ``logits`` have shape ``(T, d)`` and ``(T, V)``.
    All bf16 ports hold ``np.uint16`` codes; ``logits`` and ``lse`` hold ``np.uint32`` fp32 bit patterns.
    """
    tokens: np.ndarray
    h_in: np.ndarray
    x_attn: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    o: np.ndarray
    h_mid: np.ndarray
    x_mlp: np.ndarray
    a: np.ndarray
    h_out: np.ndarray
    x_fn: np.ndarray
    logits: np.ndarray
    lse: np.ndarray
    cheat: dict | None = None


def _int4_g32(codes):
    """RTN symmetric int4 group-32 re-quantization returning bf16 uint16 codes."""
    o, i = codes.shape
    if i % 32 == 0:
        return int4_codes(codes, G=32)
    pad = (-i) % 32
    padded = np.zeros((o, i + pad), dtype=np.uint16)
    padded[:, :i] = codes
    return int4_codes(padded, G=32)[:, :i]


class TinyDense:
    """Small deterministic Qwen3-style dense decoder (GQA + optional q/k RMSNorm + RoPE + SwiGLU)."""

    def __init__(self, d=64, n_q=4, n_kv=2, dh=16, dff=128, L=3, V=256, seed=0, qk_norm=True):
        if n_q % n_kv != 0:
            raise ValueError(f"n_q ({n_q}) must be divisible by n_kv ({n_kv})")
        if dh % 2 != 0:
            raise ValueError(f"dh ({dh}) must be even for RoPE")
        self.d = int(d)
        self.n_q = int(n_q)
        self.n_kv = int(n_kv)
        self.dh = int(dh)
        self.dff = int(dff)
        self.L = int(L)
        self.V = int(V)
        self.seed = int(seed)
        self.qk_norm = bool(qk_norm)

        rng = np.random.default_rng(self.seed)
        t = {
            "model.embed_tokens.weight": rnd(rng, (self.V, self.d), 0.5),
        }
        for l in range(self.L):
            p = f"model.layers.{l}."
            t[p + "input_layernorm.weight"] = bf16(1.0 + 0.2 * rng.standard_normal(self.d).astype(np.float32))
            t[p + "self_attn.q_proj.weight"] = rnd(rng, (self.n_q * self.dh, self.d), 0.1)
            t[p + "self_attn.k_proj.weight"] = rnd(rng, (self.n_kv * self.dh, self.d), 0.1)
            t[p + "self_attn.v_proj.weight"] = rnd(rng, (self.n_kv * self.dh, self.d), 0.1)
            t[p + "self_attn.o_proj.weight"] = rnd(rng, (self.d, self.n_q * self.dh), 0.05)
            if self.qk_norm:
                t[p + "self_attn.q_norm.weight"] = bf16(1.0 + 0.2 * rng.standard_normal(self.dh).astype(np.float32))
                t[p + "self_attn.k_norm.weight"] = bf16(1.0 + 0.2 * rng.standard_normal(self.dh).astype(np.float32))
            t[p + "post_attention_layernorm.weight"] = bf16(1.0 + 0.2 * rng.standard_normal(self.d).astype(np.float32))
            t[p + "mlp.gate_proj.weight"] = rnd(rng, (self.dff, self.d), 0.08)
            t[p + "mlp.up_proj.weight"] = rnd(rng, (self.dff, self.d), 0.08)
            t[p + "mlp.down_proj.weight"] = rnd(rng, (self.d, self.dff), 0.05)
        t["model.norm.weight"] = bf16(1.0 + 0.2 * rng.standard_normal(self.d).astype(np.float32))
        t["lm_head.weight"] = rnd(rng, (self.V, self.d), 0.05)
        self._tensors = t

    def tensors(self):
        """Return HF-named dict of honest bf16 uint16 weight arrays (compatible with ``flashprove_next.modelcommit``)."""
        return {k: v.copy() for k, v in self._tensors.items()}

    def forward(self, tokens, cheat=None):
        """Run deterministic prefill/decode forward pass and capture every port.

        ``cheat`` may contain:
          * ``"exec_seed": int`` -- execute with random weights from another seed while ``tensors()`` stays honest.
          * ``"int4": True`` -- re-quantize every 2-D weight matrix with RTN int4 group 32 before computing.
          * ``"skip_layers": list[int]`` -- skipped layers return ``h_out = h_in`` and zero intermediates.
        """
        toks = np.asarray(tokens, dtype=np.int64).reshape(-1)
        T = len(toks)
        if T == 0:
            raise ValueError("tokens must be non-empty")
        if (toks < 0).any() or (toks >= self.V).any():
            raise ValueError(f"token ids out of range [0, {self.V})")

        cheat_cfg = dict(cheat) if cheat else {}
        if "exec_seed" in cheat_cfg and cheat_cfg["exec_seed"] is not None:
            exec_model = TinyDense(
                d=self.d, n_q=self.n_q, n_kv=self.n_kv, dh=self.dh, dff=self.dff,
                L=self.L, V=self.V, seed=int(cheat_cfg["exec_seed"]), qk_norm=self.qk_norm,
            )
            W = exec_model.tensors()
        else:
            W = self.tensors()

        if cheat_cfg.get("int4"):
            for name, arr in list(W.items()):
                if arr.ndim == 2:
                    W[name] = _int4_g32(arr)

        skip_layers = set(int(x) for x in cheat_cfg.get("skip_layers", ()))
        trunc_kv = int(cheat_cfg.get("trunc_kv", 0))
        detach = bool(cheat_cfg.get("detach_residual", False))

        L, d, nq, nkv, dh, dff, V = self.L, self.d, self.n_q, self.n_kv, self.dh, self.dff, self.V
        nqd = nq * dh
        nkd = nkv * dh
        rep = nq // nkv
        scale = np.float32(1.0 / np.sqrt(dh))

        h_in = np.zeros((T, L, d), dtype=np.uint16)
        x_attn = np.zeros((T, L, d), dtype=np.uint16)
        q_arr = np.zeros((T, L, nqd), dtype=np.uint16)
        k_arr = np.zeros((T, L, nkd), dtype=np.uint16)
        v_arr = np.zeros((T, L, nkd), dtype=np.uint16)
        o_arr = np.zeros((T, L, nqd), dtype=np.uint16)
        h_mid = np.zeros((T, L, d), dtype=np.uint16)
        x_mlp = np.zeros((T, L, d), dtype=np.uint16)
        a_arr = np.zeros((T, L, dff), dtype=np.uint16)
        h_out = np.zeros((T, L, d), dtype=np.uint16)
        lse_arr = np.zeros((T, L, nq), dtype=np.uint32)
        x_fn = np.zeros((T, d), dtype=np.uint16)
        logits = np.zeros((T, V), dtype=np.uint32)

        h_in[:, 0, :] = W["model.embed_tokens.weight"][toks]

        rope_cache = [rope_table(t, dh) for t in range(T)]

        for l in range(L):
            if l in skip_layers:
                h_out[:, l, :] = h_in[:, l, :]
            else:
                p = f"model.layers.{l}."
                w_in_norm = W[p + "input_layernorm.weight"]
                Wq = W[p + "self_attn.q_proj.weight"]
                Wk = W[p + "self_attn.k_proj.weight"]
                Wv = W[p + "self_attn.v_proj.weight"]
                Wo = W[p + "self_attn.o_proj.weight"]
                w_q_norm = W[p + "self_attn.q_norm.weight"] if self.qk_norm else None
                w_k_norm = W[p + "self_attn.k_norm.weight"] if self.qk_norm else None
                w_post_norm = W[p + "post_attention_layernorm.weight"]
                Wgate = W[p + "mlp.gate_proj.weight"]
                Wup = W[p + "mlp.up_proj.weight"]
                Wdown = W[p + "mlp.down_proj.weight"]

                for t in range(T):
                    xa = rmsnorm_eng(h_in[t, l], w_in_norm, EPS)
                    x_attn[t, l] = xa
                    xf = f32_of_bf16(xa)

                    q_lin = bf16(W32(Wq) @ xf).reshape(nq, dh)
                    k_lin = bf16(W32(Wk) @ xf).reshape(nkv, dh)
                    if self.qk_norm:
                        q_lin = rmsnorm_eng(q_lin, w_q_norm, EPS)
                        k_lin = rmsnorm_eng(k_lin, w_k_norm, EPS)
                    cos, sin = rope_cache[t]
                    q_arr[t, l] = rope_eng(q_lin, cos, sin).reshape(-1)
                    k_arr[t, l] = rope_eng(k_lin, cos, sin).reshape(-1)
                    v_arr[t, l] = bf16(W32(Wv) @ xf).reshape(-1)

                    # Causal GQA softmax attention over keys 0..t in float64 -> bf16
                    qf = f32_of_bf16(q_arr[t, l].reshape(nq, dh)).astype(np.float64)
                    lo = max(0, t + 1 - trunc_kv) if trunc_kv else 0     # cheat: truncated KV window
                    Kc = k_arr[lo:t + 1, l].reshape(t + 1 - lo, nkv, dh).transpose(1, 0, 2)
                    Vc = v_arr[lo:t + 1, l].reshape(t + 1 - lo, nkv, dh).transpose(1, 0, 2)
                    Kf = np.repeat(f32_of_bf16(Kc).astype(np.float64), rep, axis=0)
                    Vf = np.repeat(f32_of_bf16(Vc).astype(np.float64), rep, axis=0)
                    z = scale * np.einsum("hd,htd->ht", qf, Kf)
                    m = z.max(-1, keepdims=True)
                    w = np.exp(z - m)
                    wsum = w.sum(-1, keepdims=True)
                    attn = np.einsum("ht,htd->hd", w / wsum, Vf)
                    o_arr[t, l] = bf16(attn.astype(np.float32)).reshape(-1)
                    lse_arr[t, l] = f32codes((m[:, 0] + np.log(wsum[:, 0])).astype(np.float32))

                    hm = eng_gemm_res(Wo, o_arr[t, l], h_in[t, l])
                    h_mid[t, l] = hm
                    xm = rmsnorm_eng(hm, w_post_norm, EPS)
                    x_mlp[t, l] = xm
                    act = eng_gu(Wgate, Wup, xm)
                    a_arr[t, l] = act
                    h_out[t, l] = eng_gemm_res(Wdown, act, hm)

            if l + 1 < L:
                h_in[:, l + 1, :] = h_out[:, l, :]
                if detach:          # cheat: the next layer reads a residual detached from h_out (halved)
                    h_in[:, l + 1, :] = bf16(f32_of_bf16(h_out[:, l, :]) * np.float32(0.5))

        w_final = W["model.norm.weight"]
        W_head = W["lm_head.weight"]
        for t in range(T):
            xfn = rmsnorm_eng(h_out[t, L - 1], w_final, EPS)
            x_fn[t] = xfn
            logits[t] = eng_head(W_head, xfn)

        return Capture(
            tokens=toks,
            h_in=h_in,
            x_attn=x_attn,
            q=q_arr,
            k=k_arr,
            v=v_arr,
            o=o_arr,
            h_mid=h_mid,
            x_mlp=x_mlp,
            a=a_arr,
            h_out=h_out,
            x_fn=x_fn,
            logits=logits,
            lse=lse_arr,
            cheat=cheat_cfg or None,
        )
