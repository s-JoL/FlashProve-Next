// SPDX-License-Identifier: Apache-2.0
#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

using Word = unsigned long long;
constexpr Word Prime = 0xffffffff00000001ULL;
constexpr Word Epsilon = 0xffffffffULL;

__device__ Word add(Word a, Word b) {
    Word sum = a + b;
    return (sum < a || sum >= Prime) ? sum - Prime : sum;
}

__device__ Word sub(Word a, Word b) {
    return a >= b ? a - b : Prime - (b - a);
}

__device__ Word mul(Word a, Word b) {
    Word lo = a * b;
    Word hi = __umul64hi(a, b);
    Word upper = hi >> 32;
    Word base = lo - upper;
    if (lo < upper) base -= Epsilon;
    if (base >= Prime) base -= Prime;
    return add(base, (hi & Epsilon) * Epsilon);
}

template<int N>
__device__ void butterflies(Word* tile, const Word* twiddles) {
    __syncthreads();
    for (int half = 1; half < N; half *= 2) {
        for (int j = threadIdx.x; j < N / 2; j += blockDim.x) {
            int offset = j % half;
            int i = 2 * (j - offset) + offset;
            Word a = tile[i];
            Word b = mul(tile[i + half], twiddles[half - 1 + offset]);
            tile[i] = add(a, b);
            tile[i + half] = sub(a, b);
        }
        __syncthreads();
    }
}

// Launch one block per row with 65536 dynamic shared bytes. Opt in using
// cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize).
// All transforms use natural order and stage-packed roots.
extern "C" __global__ void ntt8192(
    const Word* input, Word* output, const Word* twiddles, int message_size
) {
    extern __shared__ Word tile[];
    size_t row = blockIdx.x;
    for (unsigned i = threadIdx.x; i < 8192; i += blockDim.x) {
        tile[__brev(i) >> 19] = i < message_size ? input[row * message_size + i] : 0;
    }
    butterflies<8192>(tile, twiddles);
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) output[row * 8192 + i] = tile[i];
}

// Both Bailey passes use grid (rows, 256) and distinct buffers.
// powers[j] = 7^(((Prime - 1) / 65536) * j) modulo Prime.
extern "C" __global__ void bailey_first(
    const Word* input, Word* intermediate, const Word* twiddles,
    const Word* powers, int message_size
) {
    __shared__ Word tile[256];
    size_t row = blockIdx.x;
    unsigned n1 = blockIdx.y;
    for (unsigned n0 = threadIdx.x; n0 < 256; n0 += blockDim.x) {
        unsigned i = 256 * n0 + n1;
        tile[__brev(n0) >> 24] = i < message_size ? input[row * message_size + i] : 0;
    }
    butterflies<256>(tile, twiddles);
    for (unsigned k0 = threadIdx.x; k0 < 256; k0 += blockDim.x) {
        intermediate[row * 65536 + 256 * k0 + n1] = mul(tile[k0], powers[k0 * n1]);
    }
}

extern "C" __global__ void bailey_second(
    const Word* intermediate, Word* output, const Word* twiddles
) {
    __shared__ Word tile[256];
    size_t row = blockIdx.x;
    unsigned k0 = blockIdx.y;
    for (unsigned n1 = threadIdx.x; n1 < 256; n1 += blockDim.x) {
        tile[__brev(n1) >> 24] = intermediate[row * 65536 + 256 * k0 + n1];
    }
    butterflies<256>(tile, twiddles);
    for (unsigned k1 = threadIdx.x; k1 < 256; k1 += blockDim.x) {
        output[row * 65536 + k0 + 256 * k1] = tile[k1];
    }
}

struct Cubic { Word a, b, c; };
struct FractionNode { Cubic numerator, denominator; };

__device__ Cubic add(Cubic x, Cubic y) {
    return {add(x.a, y.a), add(x.b, y.b), add(x.c, y.c)};
}

__device__ Cubic mul(Cubic x, Cubic y) {
    Word a = add(mul(x.b, y.c), mul(x.c, y.b));
    Word b = mul(x.c, y.c);
    return {
        add(mul(x.a, y.a), add(a, a)),
        add(add(mul(x.a, y.b), mul(x.b, y.a)), add(b, b)),
        add(add(mul(x.a, y.c), mul(x.b, y.b)), mul(x.c, y.a))
    };
}

// FracLogUp tree layer: out[i] = in[2i] (+) in[2i+1] with (n, d) (+) (n', d') = (n d' + n' d, d d').
// External layout is unchanged: (2 pairs, 6) u64 in, (pairs, 6) u64 out (48-byte nodes, AoS).
//
// Each block owns FRAC_TILE consecutive pairs (FRAC_TILE * 96 input bytes, contiguous).  Instead of
// every thread issuing twelve strided 8-byte loads of its own 96-byte pair (the 48-byte AoS stride
// splits each warp load over ~24 sectors, which stalled sm_120), the block streams the tile through
// shared memory with 16-byte `ulonglong2` read-only loads at consecutive addresses (a warp covers
// 512 contiguous bytes per instruction), computes from shared memory, and writes the output tile back
// with coalesced 16-byte stores.  Every loop is strided by blockDim.x, so the kernel is correct for
// any block size (including the sequential blockDim.x = 1 host emulation).  Buffers from cudaMalloc /
// CuPy are 256-byte aligned and every tile starts at a multiple of 96 (in) / 48 (out) bytes, hence
// 16-byte aligned.
constexpr unsigned FRAC_TILE = 128;
static_assert((FRAC_TILE * 12) % 2 == 0 && (FRAC_TILE * 6) % 2 == 0, "vector width");

__device__ inline ulonglong2 frac_load2(const ulonglong2* p) {
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}

extern "C" __global__ void fraction_layer(
    const FractionNode* input, FractionNode* output, size_t pairs
) {
    __shared__ Word tile_in[FRAC_TILE * 12];
    __shared__ Word tile_out[FRAC_TILE * 6];
    size_t first = size_t(blockIdx.x) * FRAC_TILE;
    if (first >= pairs) return;
    unsigned count = (unsigned)((pairs - first) < FRAC_TILE ? (pairs - first) : FRAC_TILE);
    const Word* src = reinterpret_cast<const Word*>(input) + first * 12;
    Word* dst = reinterpret_cast<Word*>(output) + first * 6;
    unsigned in_words = count * 12, out_words = count * 6;      // both even
    const ulonglong2* src2 = reinterpret_cast<const ulonglong2*>(src);
    for (unsigned k = threadIdx.x; k < in_words / 2; k += blockDim.x) {
        ulonglong2 v = frac_load2(src2 + k);
        tile_in[2 * k] = v.x;
        tile_in[2 * k + 1] = v.y;
    }
    __syncthreads();
    for (unsigned p = threadIdx.x; p < count; p += blockDim.x) {
        const Word* w = tile_in + 12 * p;
        Cubic ln = {w[0], w[1], w[2]}, ld = {w[3], w[4], w[5]};
        Cubic rn = {w[6], w[7], w[8]}, rd = {w[9], w[10], w[11]};
        Cubic num = add(mul(ln, rd), mul(rn, ld));
        Cubic den = mul(ld, rd);
        Word* o = tile_out + 6 * p;
        o[0] = num.a; o[1] = num.b; o[2] = num.c;
        o[3] = den.a; o[4] = den.b; o[5] = den.c;
    }
    __syncthreads();
    ulonglong2* dst2 = reinterpret_cast<ulonglong2*>(dst);
    for (unsigned k = threadIdx.x; k < out_words / 2; k += blockDim.x) {
        ulonglong2 v;
        v.x = tile_out[2 * k];
        v.y = tile_out[2 * k + 1];
        dst2[k] = v;
    }
}

// ============================================================================ relation-3 folded PCS kernels
// Portable CUDA C++ (NVRTC-clean: no headers under __CUDACC_RTC__). Bit-exact with
// prover/fpn_prover/backend/folded_pcs.py and the normative verifiers (specs/11-wire-format.md §5.4).

__device__ Word pow_mod(Word base, unsigned long long e) {
    Word acc = 1;
    while (e) {
        if (e & 1) acc = mul(acc, base);
        base = mul(base, base);
        e >>= 1;
    }
    return acc;
}

__device__ Cubic sub(Cubic x, Cubic y) {
    return {sub(x.a, y.a), sub(x.b, y.b), sub(x.c, y.c)};
}

__device__ Cubic scale(Cubic x, Word s) {
    return {mul(x.a, s), mul(x.b, s), mul(x.c, s)};
}

// Generic radix-2 NTT (natural-order output) for codewords of any power-of-two length n:
// ntt_bitrev once, then ntt_stage for half = 1, 2, ..., n/2 with stage-packed twiddles
// twiddles[half - 1 + j] = (7^((Prime - 1) / (2 half)))^j. Grid: (ceil(n/2 / block), rows).
extern "C" __global__ void ntt_bitrev(const Word* input, Word* output, int log_n, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t row = blockIdx.y;
    unsigned long long r = 0, v = i;
    for (int b = 0; b < log_n; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
    output[row * n + r] = input[row * n + i];
}

extern "C" __global__ void ntt_stage(Word* data, const Word* twiddles, size_t n, size_t half) {
    size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n / 2) return;
    size_t row = blockIdx.y;
    size_t offset = j % half;
    size_t i = 2 * (j - offset) + offset;
    Word* d = data + row * n;
    Word a = d[i];
    Word b = mul(d[i + half], twiddles[half - 1 + offset]);
    d[i] = add(a, b);
    d[i + half] = sub(a, b);
}

// ---------------------------------------------------------------------------- generalized Bailey SMEM NTT
// Any power-of-two length n = 2^log_n as a short sequence of shared-memory passes (Bailey's 4-step /
// 6-step decomposition in its in-place, transpose-free form).  The radix-2 DIT schedule above
// (bit reversal, then stages half = 2^0 .. 2^(log_n - 1)) is cut into consecutive groups of m stages;
// pass (s, m) executes stages s .. s+m-1.  Those stages only couple indices that agree outside bits
// [s, s+m), so the data splits into independent tiles
//     idx(lo, x, hi) = lo + (x << s) + (hi << (s + m)),   lo < 2^s, x < 2^m,
// each a length-2^m DIT transform held in shared memory.  The global twiddle of stage s+u at offset
// lo + (o << s) factors exactly (exact field arithmetic, so bit-identical) as
//     w_{2^(s+u+1)}^(lo + o 2^s) = c_u(lo) * w_{2^(u+1)}^o,   c_u(lo) = g^(2^(m-1-u)), g = w_{2^(s+m)}^lo,
// i.e. the 4-step inter-pass twiddle multiply is fused into each butterfly (c_u is the coset factor of
// the tile; c_u = 1 for s = 0).  w_{2^(u+1)}^o = local_tw[2^u - 1 + o] are the first 2^m - 1 entries of
// the stage-packed table of ANY length >= 2^m, so a single 4095-word table serves every pass (m <= 12).
// A block owns 2^c tiles with consecutive lo (c <= s) so global loads/stores are 2^c-word coalesced.
// The first pass gathers its tile straight from the natural-order input through the bit reversal.
// Pass plan: cuda_engine.bailey_plan(); dynamic shared bytes = 8 * ((2^m << c) + (m << c)).
// Grid: (n >> (m + c), rows), block <= 2^(m + c - 1) threads.  Passes after the first are in place
// (every tile reads exactly the cells it writes; tiles are disjoint).
extern "C" __global__ void bailey_pass(
    const Word* input, Word* data, const Word* local_tw, int log_n, int s, int m, int c, int bitrev_load
) {
    extern __shared__ Word tile[];
    const size_t n = size_t(1) << log_n;
    const size_t row = blockIdx.y;
    const unsigned cols = 1u << c, size = 1u << m;
    Word* fac = tile + (size_t(size) << c);           // fac[col * m + u] = c_u(lo0 + col)
    const size_t group = blockIdx.x;
    const size_t lo0 = (group & ((size_t(1) << (s - c)) - 1)) << c;
    const size_t hi = group >> (s - c);
    const size_t base = row * n + lo0 + (hi << (s + m));
    if (s > 0) {
        // g = w_{2^(s+m)}^lo, then c_{u-1} = c_u^2.
        const Word root = pow_mod(7, (Prime - 1) >> (s + m));
        for (unsigned col = threadIdx.x; col < cols; col += blockDim.x) {
            Word g = pow_mod(root, lo0 + col);
            for (int u = m - 1; u >= 0; --u) {
                fac[col * m + u] = g;
                g = mul(g, g);
            }
        }
    }
    for (unsigned e = threadIdx.x; e < (size << c); e += blockDim.x) {
        unsigned col = e & (cols - 1), x = e >> c;
        size_t idx = base + col + (size_t(x) << s);
        if (bitrev_load) {
            size_t local = idx - row * n;
            size_t src = log_n ? (size_t)(__brev((unsigned)local) >> (32 - log_n)) : 0;
            tile[e] = input[row * n + src];
        } else {
            tile[e] = data[idx];
        }
    }
    __syncthreads();
    for (int u = 0; u < m; ++u) {
        const unsigned half = 1u << u;
        for (unsigned j = threadIdx.x; j < ((size >> 1) << c); j += blockDim.x) {
            unsigned col = j & (cols - 1), jj = j >> c;
            unsigned off = jj & (half - 1);
            unsigned i = 2 * (jj - off) + off;
            Word w = local_tw[half - 1 + off];
            if (s > 0) w = mul(w, fac[col * m + u]);
            unsigned ia = (i << c) + col, ib = ((i + half) << c) + col;
            Word a = tile[ia];
            Word b = mul(tile[ib], w);
            tile[ia] = add(a, b);
            tile[ib] = sub(a, b);
        }
        __syncthreads();
    }
    for (unsigned e = threadIdx.x; e < (size << c); e += blockDim.x) {
        unsigned col = e & (cols - 1), x = e >> c;
        data[base + col + (size_t(x) << s)] = tile[e];
    }
}

// Hypercube values -> monomial coefficients, one variable per launch: for idx with bit set,
// data[idx] -= data[idx ^ bit]. One thread per pair (n/2 threads).
extern "C" __global__ void mobius_stage(Cubic* data, size_t n, size_t bit) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n / 2) return;
    size_t lo = i & (bit - 1);
    size_t idx = ((i - lo) << 1) | lo;
    data[idx | bit] = sub(data[idx | bit], data[idx]);
}

// Fused coset-4 fold of one layer of length 4*quarter (natural order, coset j = {j + k quarter}):
// f_e = x^-e / 4 * sum_k iota^-ek v_k with x^-1 = inv_omega^j, out[j] = f0 + ra f1 + rb f2 + ra rb f3.
// challenges = {ra.a, ra.b, ra.c, rb.a, rb.b, rb.c}.
extern "C" __global__ void fold4_fp3(
    const Cubic* input, Cubic* output, const Word* challenges, Word inv_omega, Word inv_iota,
    size_t quarter
) {
    size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= quarter) return;
    const Word inv4 = 0xbfffffff40000001ULL;
    Cubic v0 = input[j], v1 = input[j + quarter], v2 = input[j + 2 * quarter], v3 = input[j + 3 * quarter];
    Cubic a = sub(v0, v2);
    Cubic tb = scale(sub(v1, v3), inv_iota);
    Cubic s0 = add(add(v0, v1), add(v2, v3));
    Cubic s1 = add(a, tb);
    Cubic s2 = sub(add(v0, v2), add(v1, v3));
    Cubic s3 = sub(a, tb);
    Word x_inv = pow_mod(inv_omega, j);
    Word c0 = inv4, c1 = mul(c0, x_inv), c2 = mul(c1, x_inv), c3 = mul(c2, x_inv);
    Cubic ra = {challenges[0], challenges[1], challenges[2]};
    Cubic rb = {challenges[3], challenges[4], challenges[5]};
    Cubic f0 = scale(s0, c0), f1 = scale(s1, c1), f2 = scale(s2, c2), f3 = scale(s3, c3);
    output[j] = add(add(f0, mul(ra, f1)), add(mul(rb, f2), mul(mul(ra, rb), f3)));
}

// Basefold sumcheck: per pair i the three products (f0 + t df)(w0 + t dw) for t = 0, 1, 2;
// reduce with fp3_pair_sum. halve_fp3 fixes the lowest variable: out[i] = in[2i] + r (in[2i+1] - in[2i]).
extern "C" __global__ void basefold_round_terms(const Cubic* f, const Cubic* w, Cubic* terms, size_t pairs) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    Cubic f0 = f[2 * i], w0 = w[2 * i];
    Cubic df = sub(f[2 * i + 1], f0), dw = sub(w[2 * i + 1], w0);
    Cubic f2 = add(f[2 * i + 1], df), w2 = add(w[2 * i + 1], dw);
    terms[i] = mul(f0, w0);
    terms[pairs + i] = mul(f[2 * i + 1], w[2 * i + 1]);
    terms[2 * pairs + i] = mul(f2, w2);
}

// FracLogUp-GKR layer-sumcheck round (relation 4, `folded_air.masked_prove_np`): per pair i and t = 0..3 with
// x_t = x[2i] + t (x[2i+1] - x[2i]) for the columns w = eq(point, .), nl, dl, nr, dr,
//     terms[t pairs + i] = w_t (nl_t dr_t + nr_t dl_t + alpha dl_t dr_t);   reduce with fp3_pair_sum (rows = 4).
extern "C" __global__ void gkr_round_terms(const Cubic* w, const Cubic* nl, const Cubic* dl, const Cubic* nr,
                                           const Cubic* dr, const Word* alpha, Cubic* terms, size_t pairs) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    Cubic al = {alpha[0], alpha[1], alpha[2]};
    Cubic x0 = w[2 * i], x1 = nl[2 * i], x2 = dl[2 * i], x3 = nr[2 * i], x4 = dr[2 * i];
    Cubic s0 = sub(w[2 * i + 1], x0), s1 = sub(nl[2 * i + 1], x1), s2 = sub(dl[2 * i + 1], x2);
    Cubic s3 = sub(nr[2 * i + 1], x3), s4 = sub(dr[2 * i + 1], x4);
    for (int t = 0; t < 4; t++) {
        if (t) { x0 = add(x0, s0); x1 = add(x1, s1); x2 = add(x2, s2); x3 = add(x3, s3); x4 = add(x4, s4); }
        Cubic num = add(mul(x1, x4), mul(x3, x2));
        Cubic den = mul(x2, x4);
        terms[t * pairs + i] = mul(x0, add(num, mul(den, al)));
    }
}

extern "C" __global__ void fp3_pair_sum(const Cubic* input, Cubic* output, size_t half, size_t rows) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= half * rows) return;
    size_t row = i / half, k = i % half;
    output[i] = add(input[row * 2 * half + 2 * k], input[row * 2 * half + 2 * k + 1]);
}

extern "C" __global__ void halve_fp3(const Cubic* input, Cubic* output, const Word* r, size_t half) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= half) return;
    Cubic rr = {r[0], r[1], r[2]};
    output[i] = add(input[2 * i], mul(rr, sub(input[2 * i + 1], input[2 * i])));
}

// ---------------------------------------------------------------------------- BLAKE2b-256 Merkle
// RFC 7693 single-block BLAKE2b with personal "FPN2-MERKLE-v1\0\0" (messages <= 128 bytes).
__device__ __constant__ Word BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

__device__ __constant__ unsigned char BLAKE2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

constexpr Word MerklePersonal0 = 0x52454d2d324e5046ULL;   // "FPN2-MER"
constexpr Word MerklePersonal1 = 0x000031762d454c4bULL;   // "KLE-v1\0\0"

__device__ Word rotr64(Word x, int n) { return (x >> n) | (x << (64 - n)); }

__device__ void blake2b_g(Word* v, int a, int b, int c, int d, Word x, Word y) {
    v[a] = v[a] + v[b] + x; v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];     v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y; v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];     v[b] = rotr64(v[b] ^ v[c], 63);
}

// block: 128 message bytes (zero padded), len: message length; writes 32 digest bytes.
__device__ void blake2b256_one_block(const unsigned char* block, unsigned len, unsigned char* out) {
    Word h[8], v[16], m[16];
    for (int i = 0; i < 8; ++i) h[i] = BLAKE2B_IV[i];
    h[0] ^= 0x01010000ULL ^ 32ULL;
    h[6] ^= MerklePersonal0;
    h[7] ^= MerklePersonal1;
    for (int i = 0; i < 16; ++i) {
        Word w = 0;
        for (int k = 7; k >= 0; --k) w = (w << 8) | block[8 * i + k];
        m[i] = w;
    }
    for (int i = 0; i < 8; ++i) { v[i] = h[i]; v[i + 8] = BLAKE2B_IV[i]; }
    v[12] ^= (Word)len;
    v[14] = ~v[14];
    for (int r = 0; r < 12; ++r) {
        const unsigned char* s = BLAKE2B_SIGMA[r];
        blake2b_g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        blake2b_g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        blake2b_g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        blake2b_g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        blake2b_g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        blake2b_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2b_g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        blake2b_g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 4; ++i) {
        Word w = h[i] ^ v[i] ^ v[i + 8];
        for (int k = 0; k < 8; ++k) out[8 * i + k] = (unsigned char)(w >> (8 * k));
    }
}

// Coset leaf j of a codeword of length 4*quarter: B256(0x00 || enc(v_0) || ... || enc(v_3)).
extern "C" __global__ void merkle_leaf_blake2b(const Cubic* code, unsigned char* out, size_t quarter) {
    size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= quarter) return;
    unsigned char block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (int k = 0; k < 4; ++k) {
        Cubic c = code[j + k * quarter];
        Word limbs[3] = {c.a, c.b, c.c};
        for (int l = 0; l < 3; ++l)
            for (int b = 0; b < 8; ++b) block[1 + 24 * k + 8 * l + b] = (unsigned char)(limbs[l] >> (8 * b));
    }
    blake2b256_one_block(block, 97, out + 32 * j);
}

// One Merkle layer: node i = B256(0x01 || in[2i] || in[2i+1]).
extern "C" __global__ void merkle_layer_blake2b(const unsigned char* in, unsigned char* out, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned char block[128];
    for (int k = 0; k < 128; ++k) block[k] = 0;
    block[0] = 1;
    for (int k = 0; k < 64; ++k) block[1 + k] = in[64 * i + k];
    blake2b256_one_block(block, 65, out + 32 * i);
}
