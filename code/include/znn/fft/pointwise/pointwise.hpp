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

#include "znn/fft/layer.hpp"
#include "znn/jit/cgemm/cgemm.hpp"
#include "znn/util/constexpr.hpp"
#include "znn/util/kernel_launcher.hpp"

namespace znn::fft::pointwise
{

template <long_t NCores, bool UseHT, class Layer>
class mmm
{
private:
    static constexpr long_t num_cores = NCores;
    using matrices                    = typename Layer::matrices;

    kernel_launcher&                                                  launcher_;
    std::array<std::function<void()>, ZNN_NUM_CORES * 2>              fns_;
    std::array<std::vector<std::function<void()>>, ZNN_NUM_CORES * 2> allfns_;

    bool al1_pf_;
    bool bl1_pf_;

    float const* a_;
    float const* b_;
    float*       c_;
    float*       c_final_;

    static constexpr long_t tile_stride =
        Layer::output_transform::flat_tile_stride;

    static constexpr long_t channel_stride =
        Layer::output_transform::flat_channel_stride;

    void schedule_serial(long_t thread, long_t n0, long_t n_len, long_t c0,
                         long_t c_len, long_t r0, long_t r_len)
    {
        using As = typename matrices::As;
        using Bs = typename matrices::Bs;
        using Cs = typename matrices::Cs;

        // std::cout << "Thread: " << thread << "(" << n0 << ":" << n_len
        //           << ") (" << c0 << ":" << c_len << ") (" << r0 << ":"
        //           << r_len << ")\n";

        znn::jit::shared_cgemm_t first =
            Bs::num_row_tiles == 1
                ? znn::jit::get_znn_cgemm(
                      As::tile_row_size, As::tile_col_size, Bs::tile_col_size,
                      As::row_stride, Bs::row_stride, Cs::row_stride, 1, 0, 1,
                      0, 1, al1_pf_, bl1_pf_, 0, 1, tile_stride, channel_stride)
                : znn::jit::get_znn_cgemm(As::tile_row_size, As::tile_col_size,
                                          Bs::tile_col_size, As::row_stride,
                                          Bs::row_stride, Cs::row_stride, 1, 0,
                                          1, 0, 1);
        znn::jit::shared_cgemm_t mid = znn::jit::get_znn_cgemm(
            As::tile_row_size, As::tile_col_size, Bs::tile_col_size,
            As::row_stride, Bs::row_stride, Cs::row_stride, 1, 1, 1, 0, 1,
            al1_pf_, bl1_pf_);

        znn::jit::shared_cgemm_t last =
            Bs::num_row_tiles == 1
                ? znn::jit::get_znn_cgemm(
                      As::tile_row_size, As::tile_col_size, Bs::tile_col_size,
                      As::row_stride, Bs::row_stride, Cs::row_stride, 1, 0, 1,
                      0, 1, al1_pf_, bl1_pf_, 0, 1, tile_stride, channel_stride)
                : znn::jit::get_znn_cgemm(As::tile_row_size, As::tile_col_size,
                                          Bs::tile_col_size, As::row_stride,
                                          Bs::row_stride, Cs::row_stride, 1, 1,
                                          1, 0, 1, al1_pf_, bl1_pf_, 0, 1,
                                          tile_stride, channel_stride);

        if (!UseHT)
        {
            thread = thread * 2;
        }

        allfns_[thread].push_back([=]() {
            for (long_t n = 0; n < n_len; ++n)
            {
                for (long_t c = 0; c < c_len; ++c)
                {
                    // first
                    for (long_t r = 0; r < r_len; ++r)
                    {

                        first(a_ + As::tile_offset(0, r0 + r, n0 + n),
                              b_ + Bs::tile_offset(c0 + c, 0, n0 + n),
                              c_ + Cs::tile_offset(c0 + c, r0 + r, n0 + n),
                              a_ + As::tile_offset(0, r0 + r + 1, n0 + n),
                              nullptr,
                              c_ + Cs::tile_offset(c0 + c, r0 + r + 1, n0 + n),
                              c_final_ +
                                  channel_stride *
                                      (Cs::tile_col_size / SIMD_WIDTH) *
                                      (c0 + c) +
                                  tile_stride * (Cs::tile_row_size) * (r0 + r) +
                                  (n0 + n) * SIMD_WIDTH * 2);
                    }

                    long_t k = 1;

                    for (k = 1; k < Bs::num_row_tiles - 1; ++k)
                    {
                        for (long_t r = 0; r < r_len; ++r)
                        {
                            mid(a_ + As::tile_offset(k, r0 + r, n0 + n),
                                b_ + Bs::tile_offset(c0 + c, k, n0 + n),
                                c_ + Cs::tile_offset(c0 + c, r0 + r, n0 + n),
                                a_ + As::tile_offset(k, r0 + r + 1, n0 + n),
                                nullptr,
                                c_ +
                                    Cs::tile_offset(c0 + c, r0 + r + 1, n0 + n),
                                nullptr);
                        }
                    }

                    if (Bs::num_row_tiles > 1)
                    {
                        for (long_t r = 0; r < r_len; ++r)
                        {
                            last(a_ + As::tile_offset(k, r0 + r, n0 + n),
                                 b_ + Bs::tile_offset(c0 + c, k, n0 + n),
                                 c_ + Cs::tile_offset(c0 + c, r0 + r, n0 + n),
                                 a_ + As::tile_offset(k, r0 + r + 1, n0 + n),
                                 nullptr,
                                 c_ + Cs::tile_offset(c0 + c, r0 + r + 1,
                                                      n0 + n),
                                 c_final_ +
                                     channel_stride *
                                         (Cs::tile_col_size / SIMD_WIDTH) *
                                         (c0 + c) +
                                     tile_stride * (Cs::tile_row_size) *
                                         (r0 + r) +
                                     (n0 + n) * SIMD_WIDTH * 2);
                        }
                    }
                }
            }
        });
    }

    void schedule_parallel_simple(long_t thread_from, long_t num_threads,
                                  long_t n0, long_t n_len, long_t c0,
                                  long_t c_len, long_t r0, long_t r_len)
    {
        long_t len  = n_len / num_threads;
        long_t full = n_len % num_threads;
        long_t i    = 0;

        for (; i < full; ++i)
        {
            schedule_serial(thread_from + i, n0 + i * (len + 1), len + 1, c0,
                            c_len, r0, r_len);
        }
        for (; i < num_threads; ++i)
        {
            schedule_serial(thread_from + i, n0 + i * len + full, len, c0,
                            c_len, r0, r_len);
        }
    }

    void schedule_parallel(long_t thread_from, long_t num_threads, long_t n0,
                           long_t n_len, long_t c0, long_t c_len, long_t r0,
                           long_t r_len)
    {
        if (num_threads == 1)
        {
            schedule_serial(thread_from, n0, n_len, c0, c_len, r0, r_len);
        }
        else
        {
            long_t n_way = smallest_prime_factor(num_threads);
            if ((n_len % n_way == 0) && (n_len >= n_way))
            {
                num_split_schedule(n_way, thread_from, num_threads, n0, n_len,
                                   c0, c_len, r0, r_len);
            }
            else if ((c_len % n_way == 0) && (c_len >= n_way))
            {
                col_split_schedule(n_way, thread_from, num_threads, n0, n_len,
                                   c0, c_len, r0, r_len);
            }
            // else if (n_len > num_threads * 17) // try dynamic value for 17
            // {
            //     schedule_parallel_simple(thread_from, num_threads, n0, n_len,
            //                              c0, c_len, r0, r_len);
            //     return;
            // }
            else if (n_way <= n_len)
            {
                num_split_schedule_with_tail(n_way, thread_from, num_threads,
                                             n0, n_len, c0, c_len, r0, r_len);
            }
            else if (r_len >= n_way)
            {
                row_split_schedule(n_way, thread_from, num_threads, n0, n_len,
                                   c0, c_len, r0, r_len);
            }
            else
            {
                schedule_serial(thread_from, n0, n_len, c0, c_len, r0, r_len);
            }
        }
    }

    void num_split_schedule(long_t n_way, long_t thread_from,
                            long_t num_threads, long_t n0, long_t n_len,
                            long_t c0, long_t c_len, long_t r0, long_t r_len)
    {
        STRONG_ASSERT(n_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(n_len % n_way == 0);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;

        for (long_t i = 0; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              n0 + i * (n_len / n_way), (n_len / n_way), c0,
                              c_len, r0, r_len);
        }
    }

    void num_split_schedule_with_tail(long_t n_way, long_t thread_from,
                                      long_t num_threads, long_t n0,
                                      long_t n_len, long_t c0, long_t c_len,
                                      long_t r0, long_t r_len)
    {
        STRONG_ASSERT(n_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;

        for (long_t i = 0; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              n0 + i * (n_len / n_way), (n_len / n_way), c0,
                              c_len, r0, r_len);
        }

        if (n_len % n_way)
        {
            STRONG_ASSERT((n_len - (n_len % n_way)) ==
                          ((n_len / n_way) * n_way));
            schedule_parallel(thread_from, num_threads,
                              n0 + ((n_len / n_way) * n_way), n_len % n_way, c0,
                              c_len, r0, r_len);
        }
    }

    void col_split_schedule(long_t n_way, long_t thread_from,
                            long_t num_threads, long_t n0, long_t n_len,
                            long_t c0, long_t c_len, long_t r0, long_t r_len)
    {
        STRONG_ASSERT(c_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(c_len % n_way == 0);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;

        for (long_t i = 0; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads, n0,
                              n_len, c0 + i * (c_len / n_way), (c_len / n_way),
                              r0, r_len);
        }
    }

    void row_split_schedule(long_t n_way, long_t thread_from,
                            long_t num_threads, long_t n0, long_t n_len,
                            long_t c0, long_t c_len, long_t r0, long_t r_len)
    {
        STRONG_ASSERT(r_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;
        long_t len         = r_len / n_way;
        long_t full        = r_len % n_way;
        long_t full_start  = r0 + (len + 1) * full;
        long_t i           = 0;

        for (; i < full; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads, n0,
                              n_len, c0, c_len, r0 + (len + 1) * i, len + 1);
        }
        for (; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads, n0,
                              n_len, c0, c_len, full_start + len * (i - full),
                              len);
        }
    }

public:
    long_t flops() const
    {
        return matrices::As::num_matrices * matrices::As::num_rows *
               matrices::As::num_cols * matrices::Cs::num_cols * 2 * 4;
    }

    double gflops() const { return static_cast<double>(flops()) / 1000000000; }

    long_t actual_flops() const
    {
        return matrices::As::num_matrices * matrices::As::actual_num_rows *
               matrices::As::num_cols * matrices::Cs::num_cols * 2 * 4;
    }

    double actual_gflops() const
    {
        return static_cast<double>(actual_flops()) / 1000000000;
    }

    mmm(kernel_launcher& kl, bool apf1, bool bpf1)
        : launcher_(kl)
        , al1_pf_(apf1)
        , bl1_pf_(bpf1)
    {
        schedule_parallel(
            0, UseHT ? NCores * 2 : NCores, 0, matrices::Cs::num_matrices, 0,
            matrices::Cs::num_col_tiles, 0, matrices::Cs::num_row_tiles);

        for (long_t i = 0; i < ZNN_NUM_CORES * 2; ++i)
        {
            fns_[i] = [i, this]() {
                for (auto& f : allfns_[i])
                {
                    f();
                }
            };
        }
    }

    void printmats() const { matrices::printme(); }

    long_t afloats() const { return matrices::As::memory_in_floats; }
    long_t bfloats() const { return matrices::Bs::memory_in_floats; }
    long_t cfloats() const { return matrices::Cs::memory_in_floats; }

    void operator()(float const* a, float const* b, float* c, float* c_final)
    {
        a_       = a;
        b_       = b;
        c_       = c;
        c_final_ = c_final;

        launcher_.template launch2<UseHT>(&(fns_[0]));
    }

    void execute(float const* a, float const* b, float* c, float* c_final)
    {
        this->operator()(a, b, c, c_final);
    }
};

} // namespace znn::fft::pointwise
