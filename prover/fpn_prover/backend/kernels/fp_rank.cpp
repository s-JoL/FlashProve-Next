// SPDX-License-Identifier: Apache-2.0
// Goldilocks (p = 2^64 - 2^32 + 1) row-echelon rank of a dense matrix whose columns are ordered [B | A].
//
// Prover-side audit tool for the relation-4 zero-knowledge rank condition im(A) <= im(B)
// (`folded_air.require_joint_mask_air`).  NOT part of `fpn-core` or of any verifier.
//
//   fp_rank <matrix.bin>      prints "rank_B rank_BA"
//
// File: u64 magic 0x4b4e4152504e4650 ("FPNPRANK"), u64 rows, u64 cols, u64 nb, then rows*cols u64 (row-major,
// canonical in [0, p)).  Columns are eliminated left to right, so the pivots found among the first nb columns
// are exactly rank(B) and the total is rank([B | A]); im(A) <= im(B)  <=>  rank_B == rank_BA.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

typedef unsigned __int128 u128;
static const uint64_t P = 0xffffffff00000001ULL;

static inline uint64_t reduce128(u128 x) {
    uint64_t lo = (uint64_t)x, hi = (uint64_t)(x >> 64);
    uint64_t hh = hi >> 32, hl = hi & 0xffffffffULL;
    uint64_t t = lo - hh;
    if (lo < hh) t -= 0xffffffffULL;              // borrow: + p == - (2^32 - 1) mod 2^64
    uint64_t m = hl * 0xffffffffULL;
    uint64_t r = t + m;
    if (r < t) r += 0xffffffffULL;                 // carry
    if (r >= P) r -= P;
    return r;
}
static inline uint64_t mulm(uint64_t a, uint64_t b) { return reduce128((u128)a * b); }
static inline uint64_t subm(uint64_t a, uint64_t b) { return a >= b ? a - b : a + (P - b); }
static uint64_t powm(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; }
    return r;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: fp_rank matrix.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }
    uint64_t hdr[4];
    if (fread(hdr, 8, 4, f) != 4 || hdr[0] != 0x4b4e4152504e4650ULL) { fprintf(stderr, "bad header\n"); return 2; }
    const uint64_t R = hdr[1], C = hdr[2], NB = hdr[3];
    std::vector<uint64_t> a(R * C);
    if (fread(a.data(), 8, R * C, f) != R * C) { fprintf(stderr, "short matrix\n"); return 2; }
    fclose(f);
    for (uint64_t v : a) if (v >= P) { fprintf(stderr, "noncanonical entry\n"); return 2; }
    unsigned T = std::max(1u, std::min(64u, std::thread::hardware_concurrency()));
    uint64_t rank = 0, rank_b = 0;
    for (uint64_t col = 0; col < C && rank < R; ++col) {
        uint64_t piv = R;
        for (uint64_t i = rank; i < R; ++i) if (a[i * C + col]) { piv = i; break; }
        if (piv == R) { if (col + 1 == NB) rank_b = rank; continue; }
        if (piv != rank) std::swap_ranges(a.begin() + piv * C, a.begin() + piv * C + C, a.begin() + rank * C);
        uint64_t *prow = &a[rank * C];
        uint64_t inv = powm(prow[col], P - 2);
        for (uint64_t j = col; j < C; ++j) prow[j] = mulm(prow[j], inv);
        const uint64_t lo = rank + 1, n = R - lo;
        auto work = [&](uint64_t s, uint64_t e) {
            for (uint64_t i = s; i < e; ++i) {
                uint64_t *row = &a[i * C];
                uint64_t fct = row[col];
                if (!fct) continue;
                for (uint64_t j = col; j < C; ++j) if (prow[j]) row[j] = subm(row[j], mulm(fct, prow[j]));
            }
        };
        if (n * (C - col) < (1ULL << 20) || T == 1) {
            work(lo, R);
        } else {
            std::vector<std::thread> pool;
            uint64_t chunk = (n + T - 1) / T;
            for (unsigned t = 0; t < T; ++t) {
                uint64_t s = lo + t * chunk, e = std::min(R, s + chunk);
                if (s < e) pool.emplace_back(work, s, e);
            }
            for (auto &th : pool) th.join();
        }
        ++rank;
        if (col < NB) rank_b = rank;
    }
    if (C <= NB) rank_b = rank;
    printf("%llu %llu\n", (unsigned long long)rank_b, (unsigned long long)rank);
    return 0;
}
