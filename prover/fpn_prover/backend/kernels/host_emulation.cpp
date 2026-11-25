// SPDX-License-Identifier: Apache-2.0
// Host (CPU) emulation of goldilocks_ntt.cu for bit-exact parity tests on machines without CUDA.
// Every kernel is executed sequentially, thread by thread (blockDim.x = 1 for the shared-memory
// NTT kernels -- ntt8192 and the generalized Bailey passes `bailey_pass` -- so their
// __syncthreads() stages stay ordered).
//
//   c++ -std=c++17 -O2 -o host_emulation host_emulation.cpp
//   host_emulation <op>  < input.bin  > output.bin      (little-endian u64 words)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define __CUDACC_RTC__ 1
#define __device__
#define __global__
#define __constant__
#define __shared__

struct dim3 { unsigned x = 0, y = 0, z = 0; };
static dim3 threadIdx, blockIdx, blockDim;
static inline void __syncthreads() {}
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a * b) >> 64);
}
static inline unsigned __brev(unsigned x) {
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}
struct ulonglong2 { unsigned long long x, y; };          // CUDA vector type (fraction_layer)
extern "C" { unsigned long long tile[8192]; }   // backs `extern __shared__ Word tile[]` (ntt8192, bailey_pass)

#include "goldilocks_ntt.cu"

static std::vector<Word> in_words;
static size_t cursor = 0;
static Word next() { return in_words.at(cursor++); }
static const Word* take_ptr(size_t n) {
    if (cursor + n > in_words.size()) {
        fprintf(stderr, "short input\n");
        std::exit(2);
    }
    const Word* p = in_words.data() + cursor;
    cursor += n;
    return p;
}
static std::vector<Word> take(size_t n) {
    const Word* p = take_ptr(n);
    return std::vector<Word>(p, p + n);
}
static void emit(const void* p, size_t bytes) { fwrite(p, 1, bytes, stdout); }

template <class F>
static void launch(size_t gx, size_t gy, unsigned bx, F kernel) {
    blockDim.x = bx;
    for (size_t y = 0; y < gy; ++y)
        for (size_t x = 0; x < gx; ++x) {
            blockIdx.x = (unsigned)x;
            blockIdx.y = (unsigned)y;
            for (unsigned t = 0; t < bx; ++t) { threadIdx.x = t; kernel(); }
        }
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: host_emulation <op>\n"); return 2; }
    std::string op = argv[1];
    {
        Word buf[65536];
        size_t got;
        while ((got = fread(buf, 8, 65536, stdin)) > 0)
            in_words.insert(in_words.end(), buf, buf + got);
    }
    const unsigned B = 32;
    if (op == "fold4") {
        size_t quarter = next();
        Word inv_omega = next(), inv_iota = next();
        Word base_pow = next();
        const Word* ch = take_ptr(6);
        const Word* code = take_ptr(12 * quarter);
        std::vector<Word> out(3 * quarter);
        const Cubic* cc = (const Cubic*)code;
        Cubic* co = (Cubic*)out.data();
        const Word inv4 = 0xbfffffff40000001ULL;
        Cubic ra = {ch[0], ch[1], ch[2]}, rb = {ch[3], ch[4], ch[5]};
        Word x_inv = base_pow;
        for (size_t j = 0; j < quarter; ++j) {
            Cubic v0 = cc[j], v1 = cc[j + quarter], v2 = cc[j + 2 * quarter], v3 = cc[j + 3 * quarter];
            Cubic a = sub(v0, v2);
            Cubic tb = scale(sub(v1, v3), inv_iota);
            Cubic s0 = add(add(v0, v1), add(v2, v3));
            Cubic s1 = add(a, tb);
            Cubic s2 = sub(add(v0, v2), add(v1, v3));
            Cubic s3 = sub(a, tb);
            Word c0 = inv4, c1 = mul(c0, x_inv), c2 = mul(c1, x_inv), c3 = mul(c2, x_inv);
            Cubic f0 = scale(s0, c0), f1 = scale(s1, c1), f2 = scale(s2, c2), f3 = scale(s3, c3);
            co[j] = add(add(f0, mul(ra, f1)), add(mul(rb, f2), mul(mul(ra, rb), f3)));
            x_inv = mul(x_inv, inv_omega);
        }
        emit(out.data(), out.size() * 8);
    } else if (op == "leaf") {
        size_t quarter = next();
        const Word* code = take_ptr(12 * quarter);
        std::vector<unsigned char> out(32 * quarter);
        launch((quarter + B - 1) / B, 1, B, [&] { merkle_leaf_blake2b((const Cubic*)code, out.data(), quarter); });
        emit(out.data(), out.size());
    } else if (op == "layer") {
        size_t count = next();
        const Word* raw = take_ptr(8 * count);     // 64 bytes per node = 8 words
        std::vector<unsigned char> out(32 * count);
        launch((count + B - 1) / B, 1, B, [&] {
            merkle_layer_blake2b((const unsigned char*)raw, out.data(), count);
        });
        emit(out.data(), out.size());
    } else if (op == "ntt") {
        int log_n = (int)next();
        size_t rows = next(), n = size_t(1) << log_n;
        const Word* data = take_ptr(rows * n);
        const Word* tw = take_ptr(n - 1);
        std::vector<Word> out(rows * n);
        launch((n + B - 1) / B, rows, B, [&] { ntt_bitrev(data, out.data(), log_n, n); });
        for (size_t half = 1; half < n; half *= 2)
            launch((n / 2 + B - 1) / B, rows, B, [&] { ntt_stage(out.data(), tw, n, half); });
        emit(out.data(), out.size() * 8);
    } else if (op == "bailey") {
        // log_n, rows, passes, then (s, m, c) per pass; data; local stage twiddles (4095 words).
        int log_n = (int)next();
        size_t rows = next(), n = size_t(1) << log_n, passes = next();
        const Word* plan = take_ptr(3 * passes);
        const Word* data = take_ptr(rows * n);
        const Word* tw = take_ptr(4095);
        std::vector<Word> out(rows * n);
        for (size_t p = 0; p < passes; ++p) {
            int s = (int)plan[3 * p], m = (int)plan[3 * p + 1], c = (int)plan[3 * p + 2];
            if (m < 1 || m > 12 || c < 0 || c > s || ((size_t(1) << m) << c) > 4096) {
                fprintf(stderr, "invalid bailey pass\n");
                return 2;
            }
            launch(n >> (m + c), rows, 1, [&] {
                bailey_pass(data, out.data(), tw, log_n, s, m, c, p == 0 ? 1 : 0);
            });
        }
        emit(out.data(), out.size() * 8);
    } else if (op == "ntt8192") {
        int message_size = (int)next();
        const Word* data = take_ptr(message_size);
        const Word* tw = take_ptr(8191);
        std::vector<Word> out(8192);
        launch(1, 1, 1, [&] { ntt8192(data, out.data(), tw, message_size); });
        emit(out.data(), out.size() * 8);
    } else if (op == "mobius") {
        size_t n = next();
        auto data = take(3 * n);
        for (size_t bit = 1; bit < n; bit <<= 1)
            launch((n / 2 + B - 1) / B, 1, B, [&] { mobius_stage((Cubic*)data.data(), n, bit); });
        emit(data.data(), data.size() * 8);
    } else if (op == "frac") {
        size_t pairs = next();
        const Word* nodes = take_ptr(12 * pairs);
        std::vector<Word> out(6 * pairs);
        // One sequential thread per block: the tiled kernel stages FRAC_TILE pairs through
        // (here function-local) shared memory, so its block-strided loops run end to end.
        launch((pairs + FRAC_TILE - 1) / FRAC_TILE, 1, 1, [&] {
            fraction_layer((const FractionNode*)nodes, (FractionNode*)out.data(), pairs);
        });
        emit(out.data(), out.size() * 8);
    } else if (op == "halve") {
        size_t n = next();             // cells (even); pure halve without sumcheck
        const Word* r = take_ptr(3);
        const Word* f = take_ptr(3 * n);
        size_t pairs = n / 2;
        std::vector<Word> halved(3 * pairs);
        launch((pairs + B - 1) / B, 1, B, [&] {
            halve_fp3((const Cubic*)f, (Cubic*)halved.data(), r, pairs);
        });
        emit(halved.data(), halved.size() * 8);
    } else if (op == "sumcheck") {
        size_t n = next();             // cells; one Basefold round + halve
        const Word* r = take_ptr(3);
        const Word* f = take_ptr(3 * n);
        const Word* wv = take_ptr(3 * n);
        size_t pairs = n / 2;
        const Cubic* cf = (const Cubic*)f;
        const Cubic* cw = (const Cubic*)wv;
        Cubic acc[3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (size_t i = 0; i < pairs; ++i) {
            Cubic f0 = cf[2 * i], f1 = cf[2 * i + 1];
            Cubic w0 = cw[2 * i], w1 = cw[2 * i + 1];
            Cubic df = sub(f1, f0), dw = sub(w1, w0);
            Cubic f2 = add(f1, df), w2 = add(w1, dw);
            acc[0] = add(acc[0], mul(f0, w0));
            acc[1] = add(acc[1], mul(f1, w1));
            acc[2] = add(acc[2], mul(f2, w2));
        }
        std::vector<Word> halved(3 * pairs);
        launch((pairs + B - 1) / B, 1, B, [&] {
            halve_fp3(cf, (Cubic*)halved.data(), r, pairs);
        });
        emit(acc, 9 * 8);
        emit(halved.data(), halved.size() * 8);
    } else {
        fprintf(stderr, "unknown op %s\n", op.c_str());
        return 2;
    }
    return 0;
}
