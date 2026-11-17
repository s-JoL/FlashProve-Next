// SPDX-License-Identifier: Apache-2.0
// Native CPU accelerator for flashprove_next (optional; every entry point has a pure-NumPy fallback in Python).
// Build: python -m flashprove_next.build_ext  (g++ -O3 -shared -std=c++17 -fPIC _fast_cpu.cpp -o _fast_cpu.so)
// Goldilocks p = 2^64 - 2^32 + 1, cubic extension E = F_p[u]/(u^3 - 2).  All arrays are uint64 in [0, p).
#include <cstdint>
#include <cstring>
#include <vector>

typedef uint64_t u64;
typedef unsigned __int128 u128;
static const u64 PP = 0xFFFFFFFF00000001ULL;

static inline u64 red128(u128 x) {
  u64 lo = (u64)x, hi = (u64)(x >> 64);
  u64 hh = hi >> 32, hl = hi & 0xFFFFFFFFULL;
  u64 t = lo - hh; if (lo < hh) t -= 0xFFFFFFFFULL;
  u64 m = hl * 0xFFFFFFFFULL;
  u64 r = t + m; if (r < t) r += 0xFFFFFFFFULL;
  if (r >= PP) r -= PP;
  return r;
}
static inline u64 mulm(u64 a, u64 b) { return red128((u128)a * b); }
static inline u64 addm(u64 a, u64 b) { u64 r = a + b; if (r < a || r >= PP) r -= PP; return r; }
static inline u64 subm(u64 a, u64 b) { return a >= b ? a - b : a + (PP - b); }
static inline u64 powm(u64 b, u64 e) { u64 r = 1; while (e) { if (e & 1) r = mulm(r, b); b = mulm(b, b); e >>= 1; } return r; }
static inline u64 invm(u64 a) { return powm(a, PP - 2); }

struct E3 { u64 a, b, c; };
static inline E3 eadd(E3 x, E3 y) { return {addm(x.a, y.a), addm(x.b, y.b), addm(x.c, y.c)}; }
static inline E3 esub(E3 x, E3 y) { return {subm(x.a, y.a), subm(x.b, y.b), subm(x.c, y.c)}; }
static inline E3 emul(E3 x, E3 y) {
  // each 64x64 product is reduced separately (a sum of full 128-bit products could overflow u128)
  u64 t0 = addm(mulm(x.b, y.c), mulm(x.c, y.b));
  u64 c0 = addm(mulm(x.a, y.a), addm(t0, t0));
  u64 t1 = mulm(x.c, y.c);
  u64 c1 = addm(addm(mulm(x.a, y.b), mulm(x.b, y.a)), addm(t1, t1));
  u64 c2 = addm(addm(mulm(x.a, y.c), mulm(x.b, y.b)), mulm(x.c, y.a));
  return {c0, c1, c2};
}

extern "C" {

int tg_version() { return 6; }

// ------------------------------------------------------------------ base-field vectors
void gl_mul(const u64* a, const u64* b, u64* c, long n) { for (long i = 0; i < n; i++) c[i] = mulm(a[i], b[i]); }
void gl_add(const u64* a, const u64* b, u64* c, long n) { for (long i = 0; i < n; i++) c[i] = addm(a[i], b[i]); }
void gl_sub(const u64* a, const u64* b, u64* c, long n) { for (long i = 0; i < n; i++) c[i] = subm(a[i], b[i]); }
u64 gl_sum(const u64* a, long n) { u128 s = 0; for (long i = 0; i < n; i++) { s += a[i]; if ((i & 0xFFFF) == 0xFFFF) s = red128(s); } return red128(s); }

void gl_pow_table(u64 w, u64* out, long n) { u64 v = 1; for (long i = 0; i < n; i++) { out[i] = v; v = mulm(v, w); } }
// Montgomery batch inversion (zero entries map to zero)
void gl_batch_inv(const u64* a, u64* out, long n) {
  std::vector<u64> pre(n + 1); pre[0] = 1;
  for (long i = 0; i < n; i++) pre[i + 1] = a[i] ? mulm(pre[i], a[i]) : pre[i];
  u64 inv = invm(pre[n]);
  for (long i = n - 1; i >= 0; i--) {
    if (a[i]) { out[i] = mulm(inv, pre[i]); inv = mulm(inv, a[i]); } else out[i] = 0;
  }
}
// extension vectors (3, n): a0 = a, a1 = a + n, a2 = a + 2n
void ext_mul(const u64* a, const u64* b, u64* c, long n) {
  for (long i = 0; i < n; i++) {
    E3 r = emul({a[i], a[n + i], a[2 * n + i]}, {b[i], b[n + i], b[2 * n + i]});
    c[i] = r.a; c[n + i] = r.b; c[2 * n + i] = r.c;
  }
}
void ext_dot(const u64* a, const u64* b, long n, u64* out) {
  u128 s0 = 0, s1 = 0, s2 = 0;
  for (long i = 0; i < n; i++) {
    E3 r = emul({a[i], a[n + i], a[2 * n + i]}, {b[i], b[n + i], b[2 * n + i]});
    s0 += r.a; s1 += r.b; s2 += r.c;
    if ((i & 0xFFFF) == 0xFFFF) { s0 = red128(s0); s1 = red128(s1); s2 = red128(s2); }
  }
  out[0] = red128(s0); out[1] = red128(s1); out[2] = red128(s2);
}
// <base vals, ext w>
void base_ext_dot(const u64* v, const u64* w, long n, u64* out) {
  u128 s0 = 0, s1 = 0, s2 = 0;
  for (long i = 0; i < n; i++) {
    s0 += mulm(v[i], w[i]); s1 += mulm(v[i], w[n + i]); s2 += mulm(v[i], w[2 * n + i]);
    if ((i & 0xFFFF) == 0xFFFF) { s0 = red128(s0); s1 = red128(s1); s2 = red128(s2); }
  }
  out[0] = red128(s0); out[1] = red128(s1); out[2] = red128(s2);
}
// eq(r, x) over k variables (variable 0 = LSB); r: k triples (r0, r1, r2); out: (3, 2^k)
void eq_table(const u64* r, int k, u64* out) {
  long n = 1L << k;
  out[0] = 1; out[n] = 0; out[2 * n] = 0;
  long h = 1;
  for (int j = 0; j < k; j++) {
    E3 rr = {r[3 * j], r[3 * j + 1], r[3 * j + 2]}; E3 ss = esub({1, 0, 0}, rr);
    for (long i = 0; i < h; i++) {
      E3 v = {out[i], out[n + i], out[2 * n + i]};
      E3 a = emul(v, rr), b = emul(v, ss);
      out[h + i] = a.a; out[n + h + i] = a.b; out[2 * n + h + i] = a.c;
      out[i] = b.a; out[n + i] = b.b; out[2 * n + i] = b.c;
    }
    h *= 2;
  }
}
// eq(a, b) for two points of k extension coordinates
void eq_pt(const u64* a, const u64* b, int k, u64* out) {
  E3 acc = {1, 0, 0};
  for (int j = 0; j < k; j++) {
    E3 x = {a[3 * j], a[3 * j + 1], a[3 * j + 2]}, y = {b[3 * j], b[3 * j + 1], b[3 * j + 2]};
    E3 t = eadd(emul(x, y), emul(esub({1, 0, 0}, x), esub({1, 0, 0}, y)));
    acc = emul(acc, t);
  }
  out[0] = acc.a; out[1] = acc.b; out[2] = acc.c;
}
// Basefold codeword fold of a full codeword C (3, n) -> out (3, n/2), size n has root w (inv2x[i] = 1 / (2 w^i))
void fold_full(const u64* C, long n, u64 w, u64 a0, u64 a1, u64 a2, u64* out) {
  long h = n / 2; u64 wi = invm(w); u64 inv2 = invm(2); u64 xi = inv2; E3 al = {a0, a1, a2};
  for (long i = 0; i < h; i++) {
    E3 A = {C[i], C[n + i], C[2 * n + i]}, B = {C[h + i], C[n + h + i], C[2 * n + h + i]};
    E3 s = eadd(A, B), d = esub(A, B);
    E3 pe = {mulm(s.a, inv2), mulm(s.b, inv2), mulm(s.c, inv2)};
    E3 po = {mulm(d.a, xi), mulm(d.b, xi), mulm(d.c, xi)};
    E3 r = eadd(pe, emul(al, esub(po, pe)));
    out[i] = r.a; out[h + i] = r.b; out[2 * h + i] = r.c; xi = mulm(xi, wi);
  }
}
// fold one coset of 2^m values (triples, interleaved a0,a1,a2) at position j of a size-`size` codeword through
// `na` alphas (triples); w = root of unity of `size`.  Result (2^(m - na) triples) overwrites vals.
void fold_coset(u64* vals, int m, long j, long size, u64 w, const u64* alphas, int na) {
  long cnt = 1L << m; long step = size >> m; u64 inv2 = invm(2);
  for (int t = 0; t < na; t++) {
    long k = cnt / 2;
    for (long s = 0; s < k; s++) {
      u64 x = powm(w, (u64)(j + s * step));
      u64 i2x = invm(mulm(2, x));
      E3 A = {vals[3 * s], vals[3 * s + 1], vals[3 * s + 2]};
      E3 B = {vals[3 * (s + k)], vals[3 * (s + k) + 1], vals[3 * (s + k) + 2]};
      E3 sm = eadd(A, B), df = esub(A, B);
      E3 pe = {mulm(sm.a, inv2), mulm(sm.b, inv2), mulm(sm.c, inv2)};
      E3 po = {mulm(df.a, i2x), mulm(df.b, i2x), mulm(df.c, i2x)};
      E3 al = {alphas[3 * t], alphas[3 * t + 1], alphas[3 * t + 2]};
      E3 r = eadd(pe, emul(al, esub(po, pe)));
      vals[3 * s] = r.a; vals[3 * s + 1] = r.b; vals[3 * s + 2] = r.c;
    }
    cnt = k; size /= 2; w = mulm(w, w); step = size >> (m - t - 1);
  }
}

// ------------------------------------------------------------------ Poseidon2 (t = 12, x^7, R_F = 8, R_P = 22)
static u64 RC[30][12];
static u64 DIAGC[12];
static int P2_READY = 0;

void p2_init(const u64* rc, const u64* diag) {
  for (int k = 0; k < 30; k++) for (int i = 0; i < 12; i++) RC[k][i] = rc[12 * k + i];
  for (int i = 0; i < 12; i++) DIAGC[i] = diag[i];
  P2_READY = 1;
}
static inline u64 sbox(u64 x) { u64 x2 = mulm(x, x); u64 x3 = mulm(x2, x); u64 x6 = mulm(x3, x3); return mulm(x6, x); }
static inline void ext_layer(u64* s) {
  u64 t[12]; u128 tot[4] = {0, 0, 0, 0};
  for (int c = 0; c < 3; c++) {
    const u64 a = s[4 * c], b = s[4 * c + 1], cc = s[4 * c + 2], d = s[4 * c + 3];
    t[4 * c + 0] = red128((u128)a * 2 + (u128)b * 3 + (u128)cc + (u128)d);
    t[4 * c + 1] = red128((u128)a + (u128)b * 2 + (u128)cc * 3 + (u128)d);
    t[4 * c + 2] = red128((u128)a + (u128)b + (u128)cc * 2 + (u128)d * 3);
    t[4 * c + 3] = red128((u128)a * 3 + (u128)b + (u128)cc + (u128)d * 2);
  }
  for (int i = 0; i < 4; i++) tot[i] = (u128)t[i] + t[4 + i] + t[8 + i];
  for (int i = 0; i < 12; i++) s[i] = red128(tot[i & 3] + t[i]);
}
static inline void int_layer(u64* s) {
  u128 sum = 0; for (int i = 0; i < 12; i++) sum += s[i];
  u64 sm = red128(sum);
  for (int i = 0; i < 12; i++) s[i] = addm(sm, mulm(DIAGC[i], s[i]));
}
// state: (12, n) lane-major (state[lane * n + i]); in-place output.  trace (optional): (130, n) rows
// S1, S2, S3, S4 (48), x0..x21 (22), S26..S30 (60).
void p2_permute(u64* state, long n, u64* trace) {
  for (long i = 0; i < n; i++) {
    u64 s[12];
    for (int l = 0; l < 12; l++) s[l] = state[l * n + i];
    ext_layer(s);
    int xr = 0;
    for (int k = 0; k < 30; k++) {
      if (k < 4 || k >= 26) {
        for (int l = 0; l < 12; l++) s[l] = sbox(addm(s[l], RC[k][l]));
        ext_layer(s);
      } else {
        s[0] = addm(s[0], RC[k][0]);
        if (trace) trace[(48 + xr) * n + i] = s[0];
        xr++;
        s[0] = sbox(s[0]);
        int_layer(s);
      }
      if (trace) {
        int row = -1;
        if (k < 4) row = 12 * k; else if (k >= 25) row = 70 + 12 * (k - 25);
        if (row >= 0) for (int l = 0; l < 12; l++) trace[(row + l) * n + i] = s[l];
      }
    }
    for (int l = 0; l < 12; l++) state[l * n + i] = s[l];
  }
}
}
