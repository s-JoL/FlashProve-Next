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
static std::vector<Word> take(size_t n) {
    std::vector<Word> v(in_words.begin() + cursor, in_words.begin() + cursor + n);
    cursor += n;
    return v;
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
    Word w;
    while (fread(&w, 8, 1, stdin) == 1) in_words.push_back(w);
    const unsigned B = 32;
    if (op == "fold4") {
        size_t quarter = next();
        Word inv_omega = next(), inv_iota = next();
        auto ch = take(6);
        auto code = take(12 * quarter);
        std::vector<Word> out(3 * quarter);
        launch((quarter + B - 1) / B, 1, B, [&] {
            fold4_fp3((const Cubic*)code.data(), (Cubic*)out.data(), ch.data(), inv_omega, inv_iota, quarter);
        });
        emit(out.data(), out.size() * 8);
    } else if (op == "leaf") {
        size_t quarter = next();
        auto code = take(12 * quarter);
        std::vector<unsigned char> out(32 * quarter);
        launch((quarter + B - 1) / B, 1, B, [&] { merkle_leaf_blake2b((const Cubic*)code.data(), out.data(), quarter); });
        emit(out.data(), out.size());
    } else if (op == "layer") {
        size_t count = next();
        auto raw = take(8 * count);     // 64 bytes per node = 8 words
        std::vector<unsigned char> out(32 * count);
        launch((count + B - 1) / B, 1, B, [&] {
            merkle_layer_blake2b((const unsigned char*)raw.data(), out.data(), count);
        });
        emit(out.data(), out.size());
    } else if (op == "ntt") {
        int log_n = (int)next();
        size_t rows = next(), n = size_t(1) << log_n;
        auto data = take(rows * n);
        auto tw = take(n - 1);
        std::vector<Word> out(rows * n);
        launch((n + B - 1) / B, rows, B, [&] { ntt_bitrev(data.data(), out.data(), log_n, n); });
        for (size_t half = 1; half < n; half *= 2)
            launch((n / 2 + B - 1) / B, rows, B, [&] { ntt_stage(out.data(), tw.data(), n, half); });
        emit(out.data(), out.size() * 8);
    } else if (op == "bailey") {
        // log_n, rows, passes, then (s, m, c) per pass; data; local stage twiddles (4095 words).
        int log_n = (int)next();
        size_t rows = next(), n = size_t(1) << log_n, passes = next();
        std::vector<Word> plan = take(3 * passes);
        auto data = take(rows * n);
        auto tw = take(4095);
        std::vector<Word> out(rows * n);
        for (size_t p = 0; p < passes; ++p) {
            int s = (int)plan[3 * p], m = (int)plan[3 * p + 1], c = (int)plan[3 * p + 2];
            if (m < 1 || m > 12 || c < 0 || c > s || ((size_t(1) << m) << c) > 4096) {
                fprintf(stderr, "invalid bailey pass\n");
                return 2;
            }
            launch(n >> (m + c), rows, 1, [&] {
                bailey_pass(data.data(), out.data(), tw.data(), log_n, s, m, c, p == 0 ? 1 : 0);
            });
        }
        emit(out.data(), out.size() * 8);
    } else if (op == "ntt8192") {
        int message_size = (int)next();
        auto data = take(message_size);
        auto tw = take(8191);
        std::vector<Word> out(8192);
        launch(1, 1, 1, [&] { ntt8192(data.data(), out.data(), tw.data(), message_size); });
        emit(out.data(), out.size() * 8);
    } else if (op == "mobius") {
        size_t n = next();
        auto data = take(3 * n);
        for (size_t bit = 1; bit < n; bit <<= 1)
            launch((n / 2 + B - 1) / B, 1, B, [&] { mobius_stage((Cubic*)data.data(), n, bit); });
        emit(data.data(), data.size() * 8);
    } else if (op == "frac") {
        size_t pairs = next();
        auto nodes = take(12 * pairs);
        std::vector<Word> out(6 * pairs);
        // One sequential thread per block: the tiled kernel stages FRAC_TILE pairs through
        // (here function-local) shared memory, so its block-strided loops run end to end.
        launch((pairs + FRAC_TILE - 1) / FRAC_TILE, 1, 1, [&] {
            fraction_layer((const FractionNode*)nodes.data(), (FractionNode*)out.data(), pairs);
        });
        emit(out.data(), out.size() * 8);
    } else if (op == "sumcheck") {
        size_t n = next();             // cells; one Basefold round + halve
        auto r = take(3);
        auto f = take(3 * n), wv = take(3 * n);
        size_t pairs = n / 2;
        std::vector<Word> terms(9 * pairs);
        launch((pairs + B - 1) / B, 1, B, [&] {
            basefold_round_terms((const Cubic*)f.data(), (const Cubic*)wv.data(), (Cubic*)terms.data(), pairs);
        });
        for (size_t half = pairs / 2; half >= 1; half /= 2) {
            std::vector<Word> nxt(9 * half);
            launch((3 * half + B - 1) / B, 1, B, [&] {
                fp3_pair_sum((const Cubic*)terms.data(), (Cubic*)nxt.data(), half, 3);
            });
            terms.swap(nxt);
        }
        std::vector<Word> halved(3 * pairs);
        launch((pairs + B - 1) / B, 1, B, [&] {
            halve_fp3((const Cubic*)f.data(), (Cubic*)halved.data(), r.data(), pairs);
        });
        emit(terms.data(), 9 * 8);
        emit(halved.data(), halved.size() * 8);
    } else {
        fprintf(stderr, "unknown op %s\n", op.c_str());
        return 2;
    }
    return 0;
}
