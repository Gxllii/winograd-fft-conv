//
// Copyright (C) 2018 Aleksandar Zlateski <zlateski@mit.edu>
// Copyright (C) 2018 Zhen Jia <zhenj@princeton.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include "znn/types.hpp"
#include "znn/win/propagation.hpp"
#include <chrono>
#include <iomanip>
#include <limits>
#include <string>

namespace znn::win
{

struct configuration
{
    long_t threads;
    long_t row_block;
    bool   ht_transforms;
    long_t max_k;
    long_t max_nk;
    bool   apf1;
    bool   bpf1;
};

using measured = std::pair<configuration, double>;

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, configuration const& c)
{
    os << "pointwise_threads: " << c.threads << " row_block: " << c.row_block
       << " ht_transforms: " << (c.ht_transforms ? 1 : 0)
       << " max_k: " << c.max_k << " max_nk: " << c.max_nk
       << " apf1: " << (c.apf1 ? 1 : 0) << " bpf1: " << (c.bpf1 ? 1 : 0);
    return os;
}

template <typename F>
double function_time(F&& f)
{
    auto begin = std::chrono::high_resolution_clock::now();

    f();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    return static_cast<double>(duration) / 1000;
}

template <long_t Cores, long_t Threads, long_t RowBlock, long_t B, long_t C1,
          long_t C2, long_t D, long_t H, long_t W, class m_size, class k_size,
          bool TransformKernels = true, long_t MaxK = 128,
          long_t MaxNK = 128 * 256>
measured bench(bool pf1 = true, bool pf2 = true)
{
    static constexpr long_t OD = D - k_size::value[0] + 1;
    static constexpr long_t OH = H - k_size::value[1] + 1;
    static constexpr long_t OW = W - k_size::value[2] + 1;

    using idim = vek<B, C1, D, H, W>;
    using odim = vek<B, C2, OD, OH, OW>;

    using istrides =
        vek<C1 * D * H * W, D * H * W * CACHELINE_SIZE, H * W * CACHELINE_SIZE,
            W * CACHELINE_SIZE, CACHELINE_SIZE>;

    using ostrides =
        vek<C2 * OD * OH * OW, OD * OH * OW * CACHELINE_SIZE,
            OH * OW * CACHELINE_SIZE, OW * CACHELINE_SIZE, CACHELINE_SIZE>;

    using layer = layer_t<idim, istrides, odim, ostrides, m_size, k_size,
                          RowBlock, MaxK, MaxNK>;

    using transform_t =
        propagation<Cores, Threads, layer, TransformKernels, false>;

    transform_t tt(pf1, pf2);

    long_t ker_memory = TransformKernels ? C1 * C2 * k_size::value.prod() : 1;

    hbw_array<float> a(one_init, B * C1 * D * H * W);
    hbw_array<float> b(one_init, B * C2 * OD * OH * OW);
    hbw_array<float> buffer(one_init, transform_t::buffer_floats);
    hbw_array<float> ker(one_init, ker_memory);

    long_t iters = 5;

    for (long_t i = 0; i < iters; ++i)
    {
        tt.execute(a.data(), ker.data(), b.data(), buffer.data());
    }

    {
        vec<double, 3> total;

        auto begin = std::chrono::high_resolution_clock::now();

        for (long_t i = 0; i < iters; ++i)
        {
            total += tt.execute(a.data(), ker.data(), b.data(), buffer.data());
        }

        // f();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();
        auto msecs = static_cast<double>(duration) / 1000;

        total /= iters;

        configuration conf{Threads, RowBlock, false, MaxK, MaxNK, pf1, pf2};

        std::cout << conf << "\n\t" << total << " " << tt.actual_complexity()
                  << " " << tt.complexity() << ' ' << " : " << total.sum()
                  << " " << (msecs / iters) << " actual: "
                  << (tt.actual_complexity() / total *
                      static_cast<double>(1000))
                  << " effective: "
                  << (tt.complexity() / total * static_cast<double>(1000));

        return {conf, msecs / iters};
    }
}

inline void compare_to(measured& best, measured const& x)
{
    if (x.second < best.second)
    {
        best = x;
        std::cout << " NEW BEST (" << x.second << ")";
    }

    std::cout << std::endl;
}

constexpr long_t get_k(long_t k, long_t n, long_t MaxK, long_t MaxKTimesN)
{
    return (k / 2) % 16 ? k
                        : ((k <= MaxK || k * n <= MaxKTimesN)
                               ? k
                               : get_k(k / 2, n, MaxK, MaxKTimesN));
}

constexpr long_t get_n(long_t k, long_t n, long_t MaxK, long_t MaxKTimesN)
{
    return (n / 2) % 16
               ? n
               : (n * k <= MaxKTimesN ? n : get_n(k, n / 2, MaxK, MaxKTimesN));
}

template <long_t Cores, long_t Threads, long_t RowBlock, long_t B, long_t C1,
          long_t C2, long_t D, long_t H, long_t W, class m_size, class k_size,
          bool TransformKernels, long_t MaxK, long_t MaxNK>
void bench_row_block_loop(measured& best)
{
    static constexpr long_t K =
        (C2 <= MaxK * 2) ? C2 : get_k(C2, C1, MaxK, MaxNK);
    static constexpr long_t N = get_n(K, C1, MaxK, MaxNK);

    static constexpr long_t CACHE_REQUIRED =
        (K * N + RowBlock * (K + N) * 2) * sizeof(float) * Threads;

    // HACK!
    static constexpr long_t ZNN_FFT_L2_CACHE_SIZE = ZNN_CACHE_SIZE * 1024;

    if constexpr (RowBlock <= 32 &&
                  CACHE_REQUIRED < (ZNN_FFT_L2_CACHE_SIZE * 3 / 4))
    {
        // znn::win::avx512::clear_znn_gemms();

        compare_to(best,
                   bench<Cores, Threads, RowBlock, B, C1, C2, D, H, W, m_size,
                         k_size, TransformKernels, MaxK, MaxNK>(true, true));

        compare_to(best,
                   bench<Cores, Threads, RowBlock, B, C1, C2, D, H, W, m_size,
                         k_size, TransformKernels, MaxK, MaxNK>(true, false));

        compare_to(best,
                   bench<Cores, Threads, RowBlock, B, C1, C2, D, H, W, m_size,
                         k_size, TransformKernels, MaxK, MaxNK>(false, true));

        compare_to(best,
                   bench<Cores, Threads, RowBlock, B, C1, C2, D, H, W, m_size,
                         k_size, TransformKernels, MaxK, MaxNK>(false, false));

        bench_row_block_loop<Cores, Threads, RowBlock + 1, B, C1, C2, D, H, W,
                             m_size, k_size, TransformKernels, MaxK, MaxNK>(
            best);
    }
}

template <long_t Cores, long_t B, long_t C1, long_t C2, long_t D, long_t H,
          long_t W, class m_size, class k_size, bool TransformKernels = true>
void do_bench_real(measured& best)
{

    if constexpr (C1 * C2 > 256 * 256)
    {
        bench_row_block_loop<Cores, 1, 6, B, C1, C2, D, H, W, m_size, k_size,
                             TransformKernels, 256, 256 * 512>(best);
        bench_row_block_loop<Cores, 2, 6, B, C1, C2, D, H, W, m_size, k_size,
                             TransformKernels, 256, 256 * 512>(best);
    }

    if constexpr (C1 * C2 > 128 * 256)
    {
        bench_row_block_loop<Cores, 1, 6, B, C1, C2, D, H, W, m_size, k_size,
                             TransformKernels, 256, 256 * 256>(best);
        bench_row_block_loop<Cores, 2, 6, B, C1, C2, D, H, W, m_size, k_size,
                             TransformKernels, 256, 256 * 256>(best);
    }

    if constexpr (C1 * C2 > 128 * 128)
    {
        bench_row_block_loop<Cores, 1, 6, B, C1, C2, D, H, W, m_size, k_size,
                             TransformKernels, 128, 128 * 256>(best);
        bench_row_block_loop<Cores, 2, 6, B, C1, C2, D, H, W, m_size, k_size,
                             TransformKernels, 128, 128 * 256>(best);
    }

    bench_row_block_loop<Cores, 1, 6, B, C1, C2, D, H, W, m_size, k_size,
                         TransformKernels, 128, 128 * 128>(best);
    bench_row_block_loop<Cores, 2, 6, B, C1, C2, D, H, W, m_size, k_size,
                         TransformKernels, 128, 128 * 128>(best);
}

inline constexpr long_t ceil_div(long_t a, long_t b) { return (a + b - 1) / b; }

inline constexpr long_t padded_size(long_t L, long_t M, long_t K)
{
    return ceil_div(L - K + 1, M) * M + K - 1;
}

template <long_t Cores, long_t B, long_t C1, long_t C2, long_t D, long_t H,
          long_t W, class m_size, class k_size, bool TransformKernels = true>
double do_bench(std::string const& name = "")
{
    static constexpr long_t PADDED_D =
        padded_size(D, m_size::value[0], k_size::value[0]);
    static constexpr long_t PADDED_H =
        padded_size(H, m_size::value[1], k_size::value[1]);
    static constexpr long_t PADDED_W =
        padded_size(W, m_size::value[2], k_size::value[2]);

    std::cout << "Bench of layer " << name << ": " << B << ' ' << C1 << ' '
              << C2 << ' ' << PADDED_D << ' ' << PADDED_H << ' ' << PADDED_W
              << ' ' << " M_SIZE: " << m_size::value
              << " K_SIZE: " << k_size::value << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    measured best;
    best.second = std::numeric_limits<double>::max();

    do_bench_real<Cores, B, C1, C2, PADDED_D, PADDED_H, PADDED_W, m_size,
                  k_size, TransformKernels>(best);

    std::cout << "\n\n[DONE] Bench of layer" << name << ": " << B << ' ' << C1
              << ' ' << C2 << ' ' << PADDED_D << ' ' << PADDED_H << ' '
              << PADDED_W << ' ' << " M_SIZE: " << m_size::value
              << " K_SIZE: " << k_size::value << std::endl;

    std::cout << "\n[BEST] for " << name << " is " << best.second << "\n[FOR] "
              << best.first << "\n"
              << std::endl;

    return best.second;
}

} // namespace znn::win
