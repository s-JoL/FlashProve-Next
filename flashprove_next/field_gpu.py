# SPDX-License-Identifier: Apache-2.0
"""CuPy back end for flashprove_next.field (same API as the NumPy back end)."""
import os
import numpy as np
import cupy as cp
from .field import P, ONE, hsub, _bitrev, pow_table

PRE = r"""
typedef unsigned long long u64;
#define PP 0xFFFFFFFF00000001ULL
__device__ __forceinline__ u64 red128(u64 lo, u64 hi){
  u64 hh = hi >> 32, hl = hi & 0xFFFFFFFFULL;
  u64 t = lo - hh; if (lo < hh) t -= 0xFFFFFFFFULL;
  u64 m = hl * 0xFFFFFFFFULL;
  u64 r = t + m; if (r < t) r += 0xFFFFFFFFULL;
  if (r >= PP) r -= PP;
  return r;
}
__device__ __forceinline__ u64 mulm(u64 a, u64 b){ return red128(a*b, __umul64hi(a,b)); }
__device__ __forceinline__ u64 addm(u64 a, u64 b){ u64 r = a + b; if (r < a || r >= PP) r -= PP; return r; }
__device__ __forceinline__ u64 subm(u64 a, u64 b){ return a >= b ? a - b : a + (PP - b); }
__device__ __forceinline__ u64 fromi(long long v){ return v >= 0 ? (u64)v : PP - (u64)(-v); }
__device__ __forceinline__ u64 powm(u64 b, u64 e){ u64 r = 1; while (e) { if (e & 1ULL) r = mulm(r, b); b = mulm(b, b); e >>= 1; } return r; }
struct E3 { u64 a, b, c; };
__device__ __forceinline__ E3 e_add(E3 x, E3 y){ E3 r = { addm(x.a, y.a), addm(x.b, y.b), addm(x.c, y.c) }; return r; }
__device__ __forceinline__ E3 e_sub(E3 x, E3 y){ E3 r = { subm(x.a, y.a), subm(x.b, y.b), subm(x.c, y.c) }; return r; }
__device__ __forceinline__ E3 e_mul(E3 x, E3 y){
  u64 t0 = addm(mulm(x.b, y.c), mulm(x.c, y.b));
  u64 c0 = addm(mulm(x.a, y.a), addm(t0, t0));
  u64 t1 = mulm(x.c, y.c);
  u64 c1 = addm(addm(mulm(x.a, y.b), mulm(x.b, y.a)), addm(t1, t1));
  u64 c2 = addm(addm(mulm(x.a, y.c), mulm(x.b, y.b)), mulm(x.c, y.a));
  E3 r = { c0, c1, c2 }; return r;
}
"""

_mul = cp.ElementwiseKernel("uint64 a, uint64 b", "uint64 c", "c = mulm(a, b)", "tg_mul", preamble=PRE)
_add = cp.ElementwiseKernel("uint64 a, uint64 b", "uint64 c", "c = addm(a, b)", "tg_add", preamble=PRE)
_sub = cp.ElementwiseKernel("uint64 a, uint64 b", "uint64 c", "c = subm(a, b)", "tg_sub", preamble=PRE)
_fromi = cp.ElementwiseKernel("int64 a", "uint64 c", "c = fromi(a)", "tg_fromi", preamble=PRE)
_sum = cp.ReductionKernel("uint64 x", "uint64 y", "x", "addm(a, b)", "y = a", "0", "tg_sum", preamble=PRE)
_emul = cp.ElementwiseKernel("uint64 a0, uint64 a1, uint64 a2, uint64 b0, uint64 b1, uint64 b2",
                             "uint64 c0, uint64 c1, uint64 c2",
                             "E3 r = e_mul({a0, a1, a2}, {b0, b1, b2}); c0 = r.a; c1 = r.b; c2 = r.c;",
                             "tg_emul", preamble=PRE)
_eadd = cp.ElementwiseKernel("uint64 a0, uint64 a1, uint64 a2, uint64 b0, uint64 b1, uint64 b2",
                             "uint64 c0, uint64 c1, uint64 c2",
                             "c0 = addm(a0, b0); c1 = addm(a1, b1); c2 = addm(a2, b2);", "tg_eadd", preamble=PRE)
_esub = cp.ElementwiseKernel("uint64 a0, uint64 a1, uint64 a2, uint64 b0, uint64 b1, uint64 b2",
                             "uint64 c0, uint64 c1, uint64 c2",
                             "c0 = subm(a0, b0); c1 = subm(a1, b1); c2 = subm(a2, b2);", "tg_esub", preamble=PRE)
_eaxpy = cp.ElementwiseKernel("uint64 w0, uint64 w1, uint64 w2, uint64 m0, uint64 m1, uint64 m2",
                              "uint64 d0, uint64 d1, uint64 d2",
                              "E3 r = e_mul({w0, w1, w2}, {m0, m1, m2}); d0 = addm(d0, r.a); d1 = addm(d1, r.b); d2 = addm(d2, r.c);",
                              "tg_eaxpy", preamble=PRE)
_dot0 = cp.ReductionKernel("uint64 a0, uint64 a1, uint64 a2, uint64 b0, uint64 b1, uint64 b2", "uint64 y",
                           "e_mul({a0, a1, a2}, {b0, b1, b2}).a", "addm(a, b)", "y = a", "0", "tg_dot0", preamble=PRE)
_dot1 = cp.ReductionKernel("uint64 a0, uint64 a1, uint64 a2, uint64 b0, uint64 b1, uint64 b2", "uint64 y",
                           "e_mul({a0, a1, a2}, {b0, b1, b2}).b", "addm(a, b)", "y = a", "0", "tg_dot1", preamble=PRE)
_dot2 = cp.ReductionKernel("uint64 a0, uint64 a1, uint64 a2, uint64 b0, uint64 b1, uint64 b2", "uint64 y",
                           "e_mul({a0, a1, a2}, {b0, b1, b2}).c", "addm(a, b)", "y = a", "0", "tg_dot2", preamble=PRE)
_einv = cp.ElementwiseKernel("uint64 a0, uint64 a1, uint64 a2", "uint64 c0, uint64 c1, uint64 c2",
    """u64 a1a2 = mulm(a1, a2);
       u64 t0 = subm(mulm(a0, a0), addm(a1a2, a1a2));
       u64 a2a2 = mulm(a2, a2);
       u64 t1 = subm(addm(a2a2, a2a2), mulm(a0, a1));
       u64 t2 = subm(mulm(a1, a1), mulm(a0, a2));
       u64 s = addm(mulm(a2, t1), mulm(a1, t2));
       u64 n = addm(mulm(a0, t0), addm(s, s));
       u64 r = n ? powm(n, PP - 2) : 0ULL;
       c0 = mulm(t0, r); c1 = mulm(t1, r); c2 = mulm(t2, r);""", "tg_einv", preamble=PRE)
_epow = cp.ElementwiseKernel("uint64 a0, uint64 a1, uint64 a2, int32 e", "uint64 c0, uint64 c1, uint64 c2",
    """E3 r = {1ULL, 0ULL, 0ULL}, b = {a0, a1, a2}; int k = e;
       while (k) { if (k & 1) r = e_mul(r, b);
                   k >>= 1; if (k) b = e_mul(b, b); }
       c0 = r.a; c1 = r.b; c2 = r.c;""", "tg_epow", preamble=PRE)

_SRC = PRE + r"""
extern "C" __global__ void ntt_stage(u64* __restrict__ a, const u64* __restrict__ tw, long long n, int half, int twstride, long long nrows){
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long per = n >> 1; if (t >= per * nrows) return;
  long long row = t / per, k = t % per;
  long long blk = k / half, off = k % half;
  long long i = row * n + blk * 2 * half + off, j = i + half;
  u64 w = tw[(long long)off * twstride];
  u64 x = a[i], y = mulm(a[j], w);
  a[i] = addm(x, y); a[j] = subm(x, y);
}
"""
_BLAKE = r"""
typedef unsigned long long u64;
__constant__ u64 IV[8] = {0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,0x3c6ef372fe94f82bULL,0xa54ff53a5f1d36f1ULL,
  0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL};
__constant__ unsigned char SG[12][16] = {
 {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},{11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
 {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},{9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},{2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
 {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},{13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},{6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
 {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}};
__device__ __forceinline__ u64 rotr(u64 x, int n){ return (x >> n) | (x << (64 - n)); }
#define G(a,b,c,d,x,y) { v[a]=v[a]+v[b]+x; v[d]=rotr(v[d]^v[a],32); v[c]=v[c]+v[d]; v[b]=rotr(v[b]^v[c],24); \
  v[a]=v[a]+v[b]+y; v[d]=rotr(v[d]^v[a],16); v[c]=v[c]+v[d]; v[b]=rotr(v[b]^v[c],63); }
__device__ void compress(u64* h, const u64* m, u64 t, bool last){
  u64 v[16];
  for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i+8] = IV[i]; }
  v[12] ^= t; if (last) v[14] = ~v[14];
  for (int r = 0; r < 12; r++) {
    const unsigned char* s = SG[r];
    G(0,4,8,12,m[s[0]],m[s[1]]); G(1,5,9,13,m[s[2]],m[s[3]]); G(2,6,10,14,m[s[4]],m[s[5]]); G(3,7,11,15,m[s[6]],m[s[7]]);
    G(0,5,10,15,m[s[8]],m[s[9]]); G(1,6,11,12,m[s[10]],m[s[11]]); G(2,7,8,13,m[s[12]],m[s[13]]); G(3,4,9,14,m[s[14]],m[s[15]]);
  }
  for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i+8];
}
extern "C" __global__ void blake2b_rows(const u64* __restrict__ src, long long nmsg, int nw, u64* __restrict__ out){
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= nmsg) return;
  const u64* msg = src + i * (long long)nw;
  u64 h[8]; for (int k = 0; k < 8; k++) h[k] = IV[k];
  h[0] ^= 0x01010000ULL ^ 32ULL;
  long long nbytes = (long long)nw * 8; long long off = 0; u64 m[16];
  while (true) {
    long long rem = nbytes - off;
    if (rem > 128) { for (int k = 0; k < 16; k++) m[k] = msg[off/8 + k]; off += 128; compress(h, m, (u64)off, false); }
    else { for (int k = 0; k < 16; k++) m[k] = (k * 8 < rem) ? msg[off/8 + k] : 0ULL; compress(h, m, (u64)nbytes, true); break; }
  }
  for (int k = 0; k < 4; k++) out[i * 4 + k] = h[k];
}
// Streaming leaf hash of a rate-1/4 commitment (same digest as blake2b_rows on the 52-word leaf row): the leaf message is
// [ Big_0 (8w) | Big_1 (8w) | Big_2 (8w) | h_0 (8w) | h_1 (8w) | h_2 (8w) | 4 salt words ], i.e. three 128-byte BLAKE2b
// blocks + final 32-byte block, so at most two codeword limbs ever coexist on the device.
// leaf i, coset s reads L0[i + s m] into msg[s] and L1[i + s m] into msg[8 + s] (m = n0 / 8).
extern "C" __global__ void blake2b_leaf_absorb(const u64* __restrict__ L0, const u64* __restrict__ L1, long long m,
                                               u64* __restrict__ H, int first, long long t){
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
  u64 h[8];
  if (first) { for (int k = 0; k < 8; k++) h[k] = IV[k]; h[0] ^= 0x01010000ULL ^ 32ULL; }
  else { for (int k = 0; k < 8; k++) h[k] = H[i * 8 + k]; }
  u64 msg[16];
  for (int s = 0; s < 8; s++) { msg[s] = L0[i + s * m]; msg[8 + s] = L1[i + s * m]; }
  compress(h, msg, (u64)t, false);
  for (int k = 0; k < 8; k++) H[i * 8 + k] = h[k];
}
extern "C" __global__ void blake2b_leaf_final(const u64* __restrict__ H, const u64* __restrict__ salt, long long m, int sw,
                                              long long nbytes, u64* __restrict__ out){
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
  u64 h[8]; for (int k = 0; k < 8; k++) h[k] = H[i * 8 + k];
  u64 msg[16]; for (int k = 0; k < 16; k++) msg[k] = k < sw ? salt[i * sw + k] : 0ULL;
  compress(h, msg, (u64)nbytes, true);
  for (int k = 0; k < 4; k++) out[i * 4 + k] = h[k];
}
"""
_SRC += r"""
extern "C" __global__ void fold_table(const u64* T, long long ld, long long n, u64 a0, u64 a1, u64 a2, u64* out){
  long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x; long long h = n / 2; if (i >= h) return;
  E3 al = { a0, a1, a2 };
  E3 lo = { T[2*i], T[ld + 2*i], T[2*ld + 2*i] }, hi = { T[2*i+1], T[ld + 2*i+1], T[2*ld + 2*i+1] };
  E3 r = e_add(lo, e_mul(al, e_sub(hi, lo))); out[i] = r.a; out[h + i] = r.b; out[2*h + i] = r.c;
}
// out may alias C (in-place fold into the first 3*h words): thread i reads {k*h + i : k < 6} into registers and writes
// {0*h + i, 1*h + i, 2*h + i} -- positions disjoint across threads.  inv2x of a larger codeword is read with stride xs.
extern "C" __global__ void fold_code(const u64* C, long long n, const u64* inv2x, long long xs, u64 inv2, u64 a0, u64 a1, u64 a2, u64* out){
  long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x; long long h = n / 2; if (i >= h) return;
  E3 al = { a0, a1, a2 };
  E3 a = { C[i], C[n + i], C[2*n + i] }, b = { C[h + i], C[n + h + i], C[2*n + h + i] };
  E3 s = e_add(a, b), d = e_sub(a, b);
  u64 w = inv2x[i * xs];
  E3 pe = { mulm(s.a, inv2), mulm(s.b, inv2), mulm(s.c, inv2) };
  E3 po = { mulm(d.a, w), mulm(d.b, w), mulm(d.c, w) };
  E3 r = e_add(pe, e_mul(al, e_sub(po, pe))); out[i] = r.a; out[h + i] = r.b; out[2*h + i] = r.c;
}
"""
_SRC += r"""
// zero-pad a (length na) to n = 2^k and apply the bit-reversal permutation in one pass (no index table, no copy)
extern "C" __global__ void bitrev_pad(const u64* __restrict__ a, long long na, u64* __restrict__ X, long long n, int k){
  long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x; if (i >= n) return;
  long long j = (long long)(__brevll((unsigned long long)i) >> (64 - k));
  X[i] = j < na ? a[j] : 0ULL;
}
extern "C" __global__ void eq_level(u64* T, long long ld, long long h, u64 r0, u64 r1, u64 r2, u64 s0, u64 s1, u64 s2){
  long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x; if (i >= h) return;
  E3 v = { T[i], T[ld + i], T[2*ld + i] }; E3 r = { r0, r1, r2 }, s = { s0, s1, s2 };
  E3 a = e_mul(v, r), b = e_mul(v, s);
  T[h + i] = a.a; T[ld + h + i] = a.b; T[2*ld + h + i] = a.c;
  T[i] = b.a; T[ld + i] = b.b; T[2*ld + i] = b.c;
}
#define SC_MAXD 15
#define SC_MAXF 8
#define SC_TH 128
// One sumcheck term c * prod_f T_f(x)^{p_f} on its support: for every pair i, values at X = 0..D of the line through
// (T_f[2i], T_f[2i+1]); per-block partial sums go to part[(blk_off + blockIdx.x) * (D+1) * 3 + 3X + limb].
extern "C" __global__ void sc_term(u64 q0, u64 q1, u64 q2, u64 q3, u64 q4, u64 q5, u64 q6, u64 q7,
    long long l0, long long l1, long long l2, long long l3, long long l4, long long l5, long long l6, long long l7,
    u64 pws, int nf, long long npairs, int D, u64 c0, u64 c1, u64 c2, u64* part, long long blk_off){
  const u64* Ap[SC_MAXF] = { (const u64*)q0, (const u64*)q1, (const u64*)q2, (const u64*)q3, (const u64*)q4, (const u64*)q5, (const u64*)q6, (const u64*)q7 };
  long long Ald[SC_MAXF] = { l0, l1, l2, l3, l4, l5, l6, l7 };
  int Apw[SC_MAXF];
  for (int f = 0; f < SC_MAXF; f++) Apw[f] = (int)((pws >> (4 * f)) & 15ULL);
  E3 acc[SC_MAXD + 1];
  for (int X = 0; X <= D; X++) { acc[X].a = 0; acc[X].b = 0; acc[X].c = 0; }
  for (long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x; i < npairs; i += (long long)blockDim.x * gridDim.x) {
    E3 v[SC_MAXF], d[SC_MAXF];
    for (int f = 0; f < nf; f++) {
      const u64* T = Ap[f]; long long ld = Ald[f];
      E3 lo = { T[2*i], T[ld + 2*i], T[2*ld + 2*i] }, hi = { T[2*i+1], T[ld + 2*i+1], T[2*ld + 2*i+1] };
      v[f] = lo; d[f] = e_sub(hi, lo);
    }
    for (int X = 0; X <= D; X++) {
      E3 prod = { 1ULL, 0ULL, 0ULL };
      for (int f = 0; f < nf; f++) {
        E3 t = v[f]; E3 q = t;
        for (int k = 1; k < Apw[f]; k++) q = e_mul(q, t);
        prod = e_mul(prod, q);
        v[f] = e_add(v[f], d[f]);
      }
      acc[X] = e_add(acc[X], prod);
    }
  }
  __shared__ u64 sh[SC_TH * 3];
  E3 cc = { c0, c1, c2 };
  for (int X = 0; X <= D; X++) {
    E3 a = e_mul(acc[X], cc);
    sh[3*threadIdx.x] = a.a; sh[3*threadIdx.x+1] = a.b; sh[3*threadIdx.x+2] = a.c;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sh[3*threadIdx.x]   = addm(sh[3*threadIdx.x],   sh[3*(threadIdx.x+s)]);
        sh[3*threadIdx.x+1] = addm(sh[3*threadIdx.x+1], sh[3*(threadIdx.x+s)+1]);
        sh[3*threadIdx.x+2] = addm(sh[3*threadIdx.x+2], sh[3*(threadIdx.x+s)+2]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) { long long o = ((blk_off + blockIdx.x) * (D + 1) + X) * 3; part[o] = sh[0]; part[o+1] = sh[1]; part[o+2] = sh[2]; }
    __syncthreads();
  }
}
"""
_SRC += r"""
// Poseidon2 (t = 12, x^7, R_F = 8, R_P = 22): one thread per permutation, the whole state in registers.
// cst = RC (30 x 12) || DIAG (12).  state, out: (12, n) lane-major.  trace (optional): (130, n).
__device__ __forceinline__ u64 p2_sbox(u64 x){ u64 x2 = mulm(x, x); u64 x3 = mulm(x2, x); u64 x6 = mulm(x3, x3); return mulm(x6, x); }
__device__ __forceinline__ u64 red_small(u64 lo, u64 hi){ return red128(lo, hi); }
__device__ __forceinline__ u64 lin4(u64 a, u64 ca, u64 b, u64 cb, u64 c, u64 cc, u64 d, u64 cd){
  // a ca + b cb + c cc + d cd with small constants (< 8): exact 128-bit accumulation then one reduction
  u64 lo = 0, hi = 0, t;
  t = a * ca; hi += __umul64hi(a, ca); lo += t; hi += (lo < t);
  t = b * cb; hi += __umul64hi(b, cb); lo += t; hi += (lo < t);
  t = c * cc; hi += __umul64hi(c, cc); lo += t; hi += (lo < t);
  t = d * cd; hi += __umul64hi(d, cd); lo += t; hi += (lo < t);
  return red128(lo, hi);
}
__device__ __forceinline__ void p2_ext(u64* s){
  u64 t[12];
  for (int c = 0; c < 3; c++) {
    u64 a = s[4*c], b = s[4*c+1], cc = s[4*c+2], d = s[4*c+3];
    t[4*c]   = lin4(a, 2, b, 3, cc, 1, d, 1);
    t[4*c+1] = lin4(a, 1, b, 2, cc, 3, d, 1);
    t[4*c+2] = lin4(a, 1, b, 1, cc, 2, d, 3);
    t[4*c+3] = lin4(a, 3, b, 1, cc, 1, d, 2);
  }
  for (int i = 0; i < 4; i++) { u64 tot = addm(addm(t[i], t[4+i]), t[8+i]); s[i] = addm(tot, t[i]); s[4+i] = addm(tot, t[4+i]); s[8+i] = addm(tot, t[8+i]); }
}
extern "C" __global__ void p2_perm(const u64* __restrict__ state, long long n, const u64* __restrict__ cst, u64* __restrict__ out, u64* __restrict__ trace){
  long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x; if (i >= n) return;
  u64 s[12];
  for (int l = 0; l < 12; l++) s[l] = state[l * n + i];
  p2_ext(s);
  int xr = 0;
  for (int k = 0; k < 30; k++) {
    if (k < 4 || k >= 26) {
      for (int l = 0; l < 12; l++) s[l] = p2_sbox(addm(s[l], cst[12*k + l]));
      p2_ext(s);
    } else {
      s[0] = addm(s[0], cst[12*k]);
      if (trace) trace[(48 + xr) * n + i] = s[0];
      xr++;
      s[0] = p2_sbox(s[0]);
      u64 sm = 0; for (int l = 0; l < 12; l++) sm = addm(sm, s[l]);
      for (int l = 0; l < 12; l++) s[l] = addm(sm, mulm(cst[360 + l], s[l]));
    }
    if (trace) {
      int row = -1;
      if (k < 4) row = 12 * k; else if (k >= 25) row = 70 + 12 * (k - 25);
      if (row >= 0) for (int l = 0; l < 12; l++) trace[(row + l) * n + i] = s[l];
    }
  }
  for (int l = 0; l < 12; l++) out[l * n + i] = s[l];
}
"""
_SRC += r"""
// w^e from two small tables: lo[k] = w^k (k < 2^lob), hi[k] = w^(k 2^lob)
__device__ __forceinline__ u64 tw2(const u64* __restrict__ lo, const u64* __restrict__ hi, int lob, long long e){
  return mulm(lo[e & ((1LL << lob) - 1)], hi[e >> lob]);
}
// radix-2 DIT stage on a bit-reversed array; twiddle w_n^(off * twstride) computed on the fly (no n/2 table)
extern "C" __global__ void ntt_stage2(u64* __restrict__ a, const u64* __restrict__ lo, const u64* __restrict__ hi, int lob,
                                      long long n, int hb, long long twstride){
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (n >> 1)) return;
  long long half = 1LL << hb;
  long long blk = t >> hb, off = t & (half - 1);
  long long i = (blk << (hb + 1)) + off, j = i + half;
  u64 w = tw2(lo, hi, lob, off * twstride);
  u64 x = a[i], y = mulm(a[j], w);
  a[i] = addm(x, y); a[j] = subm(x, y);
}
// the first K stages (half = 1 .. 2^(K-1)) of 2^K-element blocks in shared memory: one global pass for K stages
extern "C" __global__ void ntt_smem(u64* __restrict__ a, const u64* __restrict__ lo, const u64* __restrict__ hi, int lob,
                                    long long n, int K){
  extern __shared__ u64 sh[];
  long long base = (long long)blockIdx.x << K;
  int S = 1 << K;
  for (int t = threadIdx.x; t < S; t += blockDim.x) sh[t] = a[base + t];
  __syncthreads();
  for (int s = 0; s < K; s++) {
    int half = 1 << s;
    long long twstride = n >> (s + 1);
    for (int t = threadIdx.x; t < (S >> 1); t += blockDim.x) {
      int blk = t >> s, off = t & (half - 1);
      int i = (blk << (s + 1)) + off, j = i + half;
      u64 w = tw2(lo, hi, lob, (long long)off * twstride);
      u64 x = sh[i], y = mulm(sh[j], w);
      sh[i] = addm(x, y); sh[j] = subm(x, y);
    }
    __syncthreads();
  }
  for (int t = threadIdx.x; t < S; t += blockDim.x) a[base + t] = sh[t];
}
// out[i] = scale * w^i  (two-level tables)
extern "C" __global__ void pow_fill(u64* __restrict__ out, long long n, const u64* __restrict__ lo, const u64* __restrict__ hi,
                                    int lob, u64 scale){
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  out[i] = mulm(tw2(lo, hi, lob, i), scale);
}
"""
_SRC += r"""
// ALL terms of one sumcheck round in ONE launch.  desc: per job SC_DW u64 words
//   [0..7] factor table pointers (already offset to the term's support), [8..15] limb strides, [16] 4-bit powers,
//   [17] #factors, [18] #pairs, [19..21] coefficient (E), [22] first block of the job, [23] #blocks of the job.
// bjob[b] = job of block b.  part[b][X][limb] = the block's partial sum (reduced on the device afterwards).
#define SC_DW 24
extern "C" __global__ void sc_multi(const u64* __restrict__ desc, const int* __restrict__ bjob, int D, u64* part){
  int job = bjob[blockIdx.x];
  const u64* dj = desc + (long long)job * SC_DW;
  const u64* Ap[SC_MAXF]; long long Ald[SC_MAXF]; int Apw[SC_MAXF];
  int nf = (int)dj[17]; long long npairs = (long long)dj[18];
  u64 pws = dj[16];
  for (int f = 0; f < SC_MAXF; f++) { Ap[f] = (const u64*)dj[f]; Ald[f] = (long long)dj[8 + f]; Apw[f] = (int)((pws >> (4 * f)) & 15ULL); }
  long long lb = (long long)blockIdx.x - (long long)dj[22], nb = (long long)dj[23];
  E3 acc[SC_MAXD + 1];
  for (int X = 0; X <= D; X++) { acc[X].a = 0; acc[X].b = 0; acc[X].c = 0; }
  for (long long i = lb * blockDim.x + threadIdx.x; i < npairs; i += nb * blockDim.x) {
    E3 v[SC_MAXF], d[SC_MAXF];
    for (int f = 0; f < nf; f++) {
      const u64* T = Ap[f]; long long ld = Ald[f];
      E3 lo = { T[2*i], T[ld + 2*i], T[2*ld + 2*i] }, hi = { T[2*i+1], T[ld + 2*i+1], T[2*ld + 2*i+1] };
      v[f] = lo; d[f] = e_sub(hi, lo);
    }
    for (int X = 0; X <= D; X++) {
      E3 prod = { 1ULL, 0ULL, 0ULL };
      for (int f = 0; f < nf; f++) {
        E3 t = v[f]; E3 q = t;
        for (int k = 1; k < Apw[f]; k++) q = e_mul(q, t);
        prod = e_mul(prod, q);
        v[f] = e_add(v[f], d[f]);
      }
      acc[X] = e_add(acc[X], prod);
    }
  }
  __shared__ u64 sh[SC_TH * 3];
  E3 cc = { dj[19], dj[20], dj[21] };
  for (int X = 0; X <= D; X++) {
    E3 a = e_mul(acc[X], cc);
    sh[3*threadIdx.x] = a.a; sh[3*threadIdx.x+1] = a.b; sh[3*threadIdx.x+2] = a.c;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sh[3*threadIdx.x]   = addm(sh[3*threadIdx.x],   sh[3*(threadIdx.x+s)]);
        sh[3*threadIdx.x+1] = addm(sh[3*threadIdx.x+1], sh[3*(threadIdx.x+s)+1]);
        sh[3*threadIdx.x+2] = addm(sh[3*threadIdx.x+2], sh[3*(threadIdx.x+s)+2]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) { long long o = ((long long)blockIdx.x * (D + 1) + X) * 3; part[o] = sh[0]; part[o+1] = sh[1]; part[o+2] = sh[2]; }
    __syncthreads();
  }
}
"""
_mod = cp.RawModule(code=_SRC, options=("-std=c++14",))
_k_foldt = _mod.get_function("fold_table")
_k_foldc = _mod.get_function("fold_code")
_k_ntt = _mod.get_function("ntt_stage")
_k_ntt2 = _mod.get_function("ntt_stage2")
_k_ntts = _mod.get_function("ntt_smem")
_k_powf = _mod.get_function("pow_fill")
_k_eqlv = _mod.get_function("eq_level")
_k_brp = _mod.get_function("bitrev_pad")
_k_sc = _mod.get_function("sc_term")
_k_scm = _mod.get_function("sc_multi")
_k_p2 = _mod.get_function("p2_perm")
_P2C = {}
SC_MAXD, SC_MAXF, SC_TH = 15, 8, 128
_bmod = cp.RawModule(code=_BLAKE, options=("-std=c++14",))
_k_blake = _bmod.get_function("blake2b_rows")
_k_babs = _bmod.get_function("blake2b_leaf_absorb")
_k_bfin = _bmod.get_function("blake2b_leaf_final")
_PU = np.uint64(P)
_XCH = 1 << 23            # host <-> device copies in 64 MiB pieces (see _h2d)


def _h2d(a, dtype=np.uint64):
    """Host -> device copy in 64 MiB pieces.  Some CUDA drivers (for example under virtualized GPU stacks) keep a pinned
    staging buffer as large as the largest pageable transfer ever made (RssShmem never shrinks: one 1 GiB cp.asarray
    pins 1 GiB of host RAM for the rest of the process); piecewise copies keep it at ~75 MiB and are also ~5x faster."""
    if isinstance(a, cp.ndarray):
        return a.astype(dtype, copy=False)
    a = np.asarray(a)
    if a.dtype != dtype:
        a = a.astype(dtype)
    if a.nbytes <= (_XCH << 3):
        return cp.asarray(a)
    a = np.ascontiguousarray(a)
    out = cp.empty(a.shape, dtype=dtype)
    fa = a.reshape(-1); fo = out.reshape(-1)
    for a0 in range(0, fa.size, _XCH):
        fo[a0:a0 + _XCH].set(fa[a0:a0 + _XCH])
    return out


def _d2h(x):
    """Device -> host copy in 64 MiB pieces (no large staging buffer)."""
    if not isinstance(x, cp.ndarray):
        return np.asarray(x)
    if x.nbytes <= (_XCH << 3):
        return cp.asnumpy(x)
    x = cp.ascontiguousarray(x)
    out = np.empty(x.shape, dtype=x.dtype)
    fx = x.reshape(-1); fo = out.reshape(-1)
    for a0 in range(0, fx.size, _XCH):
        fx[a0:a0 + _XCH].get(out=fo[a0:a0 + _XCH])
    return out


class GpuBackend:
    name = "cupy"
    xp = cp

    @staticmethod
    def mul(a, b): return _mul(cp.asarray(a, dtype=cp.uint64), cp.asarray(b, dtype=cp.uint64))
    @staticmethod
    def add(a, b): return _add(cp.asarray(a, dtype=cp.uint64), cp.asarray(b, dtype=cp.uint64))
    @staticmethod
    def sub(a, b): return _sub(cp.asarray(a, dtype=cp.uint64), cp.asarray(b, dtype=cp.uint64))

    @staticmethod
    def fromi(v):
        if isinstance(v, np.ndarray) and v.dtype == object:
            return cp.asarray(np.array([int(x) % P for x in v.ravel()], dtype=np.uint64).reshape(v.shape))
        return _fromi(cp.asarray(v).astype(cp.int64, copy=False))

    @staticmethod
    def bsum(a): return int(_sum(a))

    # extension ops write straight into one preallocated (3, n) array (no cp.stack copies)
    @staticmethod
    def emul(a, b):
        out = cp.empty((3, max(a.shape[1], b.shape[1])), dtype=cp.uint64)
        _emul(a[0], a[1], a[2], b[0], b[1], b[2], out[0], out[1], out[2]); return out

    @staticmethod
    def eadd(a, b):
        out = cp.empty((3, max(a.shape[1], b.shape[1])), dtype=cp.uint64)
        _eadd(a[0], a[1], a[2], b[0], b[1], b[2], out[0], out[1], out[2]); return out

    @staticmethod
    def esub(a, b):
        out = cp.empty((3, max(a.shape[1], b.shape[1])), dtype=cp.uint64)
        _esub(a[0], a[1], a[2], b[0], b[1], b[2], out[0], out[1], out[2]); return out

    @staticmethod
    def escale(a, c):
        out = cp.empty((3, a.shape[1]), dtype=cp.uint64)
        _emul(a[0], a[1], a[2], cp.uint64(c[0]), cp.uint64(c[1]), cp.uint64(c[2]), out[0], out[1], out[2]); return out

    @staticmethod
    def eaddc(a, c):
        out = cp.empty((3, a.shape[1]), dtype=cp.uint64)
        _eadd(a[0], a[1], a[2], cp.uint64(c[0]), cp.uint64(c[1]), cp.uint64(c[2]), out[0], out[1], out[2]); return out

    @staticmethod
    def eaxpy_(dst, w, c):
        """dst += c * w in place (dst may be a (3, n) slice view)."""
        _eaxpy(w[0], w[1], w[2], cp.uint64(c[0]), cp.uint64(c[1]), cp.uint64(c[2]), dst[0], dst[1], dst[2])

    @staticmethod
    def esum(a):
        r = cp.stack([_sum(a[0]), _sum(a[1]), _sum(a[2])]).get(); return (int(r[0]), int(r[1]), int(r[2]))

    @staticmethod
    def edot_dev(a, w):
        """<a, w> in E as a device (3,) array (no host sync)."""
        return cp.stack([_dot0(a[0], a[1], a[2], w[0], w[1], w[2]),
                         _dot1(a[0], a[1], a[2], w[0], w[1], w[2]),
                         _dot2(a[0], a[1], a[2], w[0], w[1], w[2])])

    @staticmethod
    def fetch_pairs(lst):
        if not lst:
            return []
        r = cp.stack(lst).get()
        return [(int(x[0]), int(x[1]), int(x[2])) for x in r]

    @staticmethod
    def epow(a, e):
        out = cp.empty((3, a.shape[1]), dtype=cp.uint64); _epow(a[0], a[1], a[2], np.int32(e), out[0], out[1], out[2]); return out

    @staticmethod
    def einv(a):
        out = cp.empty((3, a.shape[1]), dtype=cp.uint64); _einv(a[0], a[1], a[2], out[0], out[1], out[2]); return out

    @staticmethod
    def ext(a):
        a = cp.asarray(a, dtype=cp.uint64); z = cp.zeros_like(a); return cp.stack([a, z, z])

    @staticmethod
    def zeros_e(n): return cp.zeros((3, n), dtype=cp.uint64)

    @staticmethod
    def rand_e(n):
        """uniform (3, n) E elements: os.urandom words, rejection-sampled below p (exactly uniform, no PRG)."""
        from .field import rand_field
        return _h2d(rand_field(3 * n).reshape(3, n))

    @staticmethod
    def asarray(a, dtype=np.uint64): return _h2d(a, dtype)

    @staticmethod
    def tonumpy(a): return _d2h(a)

    @classmethod
    def eq_table(cls, r):
        """eq(r, x) for all x (variable 0 = LSB): one in-place doubling kernel per variable."""
        n = 1 << len(r)
        t = cp.empty((3, n), dtype=cp.uint64); t[0, 0] = 1; t[1, 0] = 0; t[2, 0] = 0
        h = 1
        for ri in r:
            s = hsub(ONE, ri)
            _k_eqlv(((h + 255) // 256,), (256,), (t, np.int64(n), np.int64(h),
                                                  np.uint64(ri[0]), np.uint64(ri[1]), np.uint64(ri[2]),
                                                  np.uint64(s[0]), np.uint64(s[1]), np.uint64(s[2])))
            h *= 2
        return t

    @staticmethod
    def p2_permute(st, rc, diag, trace=False):
        """Batched Poseidon2 permutation of the columns of st (12, n).  Returns (out, trace (130, n) or None)."""
        key = "c"
        if key not in _P2C:
            _P2C[key] = cp.asarray(np.concatenate([rc, diag]).astype(np.uint64))
        n = st.shape[1]
        out = cp.empty((12, n), dtype=cp.uint64)
        tr = cp.empty((130, n), dtype=cp.uint64) if trace else None
        if n:
            _k_p2(((n + 127) // 128,), (128,), (st, np.int64(n), _P2C[key], out, tr if trace else np.uint64(0)))
        return out, tr

    @staticmethod
    def sc_round(jobs, D):
        """Fused sumcheck round: jobs = [(coef, [(arr3xn, start, pow), ...], npairs)].  Every term of the round is
        evaluated by ONE kernel launch (a packed job descriptor table + block -> job map), the partial sums are
        reduced on the device, and the D + 1 round values (E triples) come back in ONE device-to-host transfer."""
        assert 0 <= D <= SC_MAXD, D
        nj = len(jobs)
        if nj == 0:
            return [(0, 0, 0)] * (D + 1)
        desc = np.zeros((nj, 24), dtype=np.uint64)
        nbl = np.empty(nj, dtype=np.int64)
        for q, (c, fs, npairs) in enumerate(jobs):
            assert len(fs) <= SC_MAXF, len(fs)
            row = desc[q]
            pws = 0
            for k, (arr, st, pw) in enumerate(fs):
                assert 1 <= pw <= 15, pw
                row[k] = arr.data.ptr + 8 * st; row[8 + k] = arr.shape[1]; pws |= pw << (4 * k)
            row[16] = pws; row[17] = len(fs); row[18] = npairs
            row[19] = c[0] % P; row[20] = c[1] % P; row[21] = c[2] % P
            nbl[q] = min(max(1, (npairs + SC_TH - 1) // SC_TH), 512)
        offs = np.concatenate([[0], np.cumsum(nbl)[:-1]])
        desc[:, 22] = offs.astype(np.uint64); desc[:, 23] = nbl.astype(np.uint64)
        tot = int(nbl.sum())
        bjob = np.repeat(np.arange(nj, dtype=np.int32), nbl)
        part = cp.empty((tot, D + 1, 3), dtype=cp.uint64)
        _k_scm((tot,), (SC_TH,), (cp.asarray(desc), cp.asarray(bjob), np.int32(D), part))
        s0 = _sum(part[:, :, 0], axis=0); s1 = _sum(part[:, :, 1], axis=0); s2 = _sum(part[:, :, 2], axis=0)
        r = cp.stack([s0, s1, s2], axis=1).get()
        return [(int(r[X, 0]), int(r[X, 1]), int(r[X, 2])) for X in range(D + 1)]

    @staticmethod
    def fold(T, a):
        n = T.shape[1]; h = n // 2
        if not T.flags.c_contiguous:
            T = cp.ascontiguousarray(T)
        out = cp.empty((3, h), dtype=cp.uint64)
        _k_foldt(((h + 255) // 256,), (256,), (T, np.int64(n), np.int64(n),
                                               np.uint64(a[0]), np.uint64(a[1]), np.uint64(a[2]), out))
        return out

    @staticmethod
    def fold_code(C, a, inv2x, inv2, xs=1, inplace=False):
        """fold the (3, n) codeword C with challenge a; inplace: the (3, n/2) result overwrites C's first 3 n/2 words
        (returned as a contiguous view, no second codeword-sized buffer)."""
        n = C.shape[1]; h = n // 2
        C = cp.ascontiguousarray(C)
        out = C.reshape(-1)[:3 * h].reshape(3, h) if inplace else cp.empty((3, h), dtype=cp.uint64)
        _k_foldc(((h + 255) // 256,), (256,), (C, np.int64(n), inv2x, np.int64(xs), np.uint64(inv2),
                                               np.uint64(a[0]), np.uint64(a[1]), np.uint64(a[2]), out))
        return out

    _tw = {}
    _br = {}
    LOB = 12
    NTT_K = 11            # stages done in shared memory (2^11 elements = 16 KiB per block)

    @classmethod
    def _tw2(cls, w, emax):
        """device two-level power tables of w for exponents e < emax: lo[k] = w^k (k < 2^LOB), hi[k] = w^(k 2^LOB)."""
        nh = ((max(1, int(emax)) - 1) >> cls.LOB) + 1
        key = (int(w), nh)
        if key not in cls._tw:
            lo = pow_table(int(w), 1 << cls.LOB)
            hi = pow_table(pow(int(w), 1 << cls.LOB, P), nh)
            cls._tw[key] = (cp.asarray(lo), cp.asarray(hi))
        return cls._tw[key]

    @classmethod
    def pow_dev(cls, w, n, scale=1):
        """device [scale * w^i for i < n] (no host table)."""
        lo, hi = cls._tw2(w, n)
        out = cp.empty(n, dtype=cp.uint64)
        if n:
            _k_powf(((n + 255) // 256,), (256,), (out, np.int64(n), lo, hi, np.int32(cls.LOB), np.uint64(scale % P)))
        return out

    @classmethod
    def ntt(cls, a, n, out=None):
        """Evaluations of the polynomial with coefficients a (zero-padded to n) at w_n^i, i < n (natural order)."""
        lo, hi = cls._tw2(pow(7, (P - 1) // n, P), max(1, n // 2))
        a = cp.ascontiguousarray(cp.asarray(a, dtype=cp.uint64))
        X = cp.empty(n, dtype=cp.uint64) if out is None else out
        lg = n.bit_length() - 1
        _k_brp(((n + 255) // 256,), (256,), (a, np.int64(a.shape[0]), X, np.int64(n), np.int32(lg)))
        K = min(cls.NTT_K, lg)
        if K >= 1:
            _k_ntts((n >> K,), (min(512, 1 << (K - 1)),), (X, lo, hi, np.int32(cls.LOB), np.int64(n), np.int32(K)),
                    shared_mem=8 << K)
        th = 256; tot = n // 2
        for hb in range(K, lg):
            _k_ntt2(((tot + th - 1) // th,), (th,), (X, lo, hi, np.int32(cls.LOB), np.int64(n), np.int32(hb), np.int64(n >> (hb + 1))))
        return X

    @classmethod
    def ntt_host(cls, row, n):
        """NTT of a host (or memory-mapped) coefficient row: one host -> device copy, transform on the device."""
        return cls.ntt(_h2d(row), n)

    @classmethod
    def clear_caches(cls):
        cls._tw.clear(); cls._br.clear()
        cp.get_default_memory_pool().free_all_blocks()

    @classmethod
    def encode(cls, f, rate):
        """device codeword (3, rate N) of the device (3, N) array f (all limbs transformed in place of `out`)."""
        n0 = rate * f.shape[1]
        out = cp.empty((3, n0), dtype=cp.uint64)
        for l in (0, 1, 2):
            cls.ntt(cp.ascontiguousarray(f[l]), n0, out=out[l])
        return out

    @classmethod
    def encode_host(cls, f, rate, out=None):
        """Codeword of f (3, N) as a HOST array (3, rate N) (or into `out`, e.g. a memmap): one limb at a time."""
        n0 = rate * f.shape[1]
        if out is None:
            out = np.zeros((3, n0), dtype=np.uint64)
        for l in (0, 1, 2):
            X = cls.ntt_host(f[l], n0)
            X.get(out=out[l]) if out[l].flags.c_contiguous else out.__setitem__(l, cp.asnumpy(X))
            del X
        return out

    @classmethod
    def inv2x(cls, n):
        """[INV2 * w_n^(-i) for i < n/2] on the device (codeword fold twiddles)."""
        w = pow(7, (P - 1) // n, P)
        return cls.pow_dev(pow(w, P - 2, P), n // 2, pow(2, P - 2, P))

    @staticmethod
    def leaf_absorb(L0, L1, m, H, first, t):
        """absorb one 128-byte block (8 cosets of limbs L0, L1) into the per-leaf BLAKE2b states H (m, 8)."""
        th = 128
        _k_babs(((m + th - 1) // th,), (th,), (L0, L1, np.int64(m), H, np.int32(1 if first else 0), np.int64(t)))

    @staticmethod
    def leaf_final(H, salt, nbytes):
        """final block (salt words, zero padded) of the per-leaf BLAKE2b states H (m, 8) -> digests (m, 4)."""
        m = H.shape[0]
        salt = cp.ascontiguousarray(cp.asarray(salt, dtype=cp.uint64))
        out = cp.empty((m, 4), dtype=cp.uint64); th = 128
        _k_bfin(((m + th - 1) // th,), (th,), (H, salt, np.int64(m), np.int32(salt.shape[1]), np.int64(nbytes), out))
        return out

    @staticmethod
    def blake_rows(A):
        A = cp.ascontiguousarray(cp.asarray(A, dtype=cp.uint64)); n, nw = A.shape
        out = cp.empty((n, 4), dtype=cp.uint64); th = 128
        _k_blake(((n + th - 1) // th,), (th,), (A, np.int64(n), np.int32(nw), out))
        return out
