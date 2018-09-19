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

#include "znn/assert.hpp"
#include "znn/util/constexpr.hpp"
#include "znn/util/kernel_launcher.hpp"
#include "znn/win/filter_transform/filter_transform.hpp"
#include "znn/win/input_transform/image_transform.hpp"
#include "znn/win/layer.hpp"

namespace znn::win::input_transform
{

template <long_t Threads, class Layer, bool TransformKernels = true,
          bool UseHT = false, bool DivSchedule = false>
class transform
{
private:
    using problem  = typename Layer::input_transform;
    using kproblem = typename Layer::kernel_transform;

    static constexpr long_t hyperthreads = UseHT ? 2 : 1;

    std::array<std::vector<std::pair<long_t, long_t>>, Threads * hyperthreads>
        individual_in;
    std::array<std::vector<std::pair<long_t, long_t>>, Threads * hyperthreads>
        individual_ker;

    void
    schedule_serial_kernel(long_t thread, long_t t_from, long_t t_len,
                           std::vector<std::pair<long_t, long_t>> const& ts)
    {
        for (long_t t = t_from; t < t_from + t_len; ++t)
        {
            individual_ker[Threads * hyperthreads - 1 - thread].push_back(
                ts[t]);
        }
    }

    void schedule_kernels()
    {
        std::vector<std::pair<long_t, long_t>> all;

        for (long_t ifm = 0; ifm < kproblem::input_channels; ++ifm)
        {
            for (long_t ofm = 0;
                 ofm < kproblem::output_channels / CACHELINE_SIZE; ++ofm)
            {
                all.push_back({kproblem::tile_offset(ifm, ofm),
                               kproblem::matrix_offset(ifm, ofm)});
            }
        }

        long_t t_len = static_cast<long_t>(all.size());

        long_t len        = t_len / (Threads * hyperthreads);
        long_t full       = t_len % (Threads * hyperthreads);
        long_t full_start = (len + 1) * full;

        long_t i = 0;

        for (; i < full; ++i)
        {
            schedule_serial_kernel(i, (len + 1) * i, len + 1, all);
        }
        for (; i < Threads * hyperthreads; ++i)
        {
            schedule_serial_kernel(i, full_start + len * (i - full), len, all);
        }
    }

    void
    schedule_serial_images(long_t thread, long_t t_from, long_t t_len,
                           std::vector<std::pair<long_t, long_t>> const& ts)
    {
        for (long_t t = t_from; t < t_from + t_len; ++t)
        {
            individual_in[thread].push_back(ts[t]);
        }
    }

    void schedule_images()
    {
        std::vector<std::pair<long_t, long_t>> all;

        for (long_t b = 0; b < problem::size[0]; ++b)
        {
            for (long_t c = 0; c < problem::size[1] / CACHELINE_SIZE; ++c)
            {
                for (long_t d = 0; d < problem::num_tiles[0]; ++d)
                {
                    for (long_t h = 0; h < problem::num_tiles[1]; ++h)
                    {
                        for (long_t w = 0; w < problem::num_tiles[2]; ++w)
                        {
                            all.push_back({problem::tile_offset(
                                               vec<long_t, 5>{b, c, d, h, w}),
                                           problem::matrix_offset(
                                               vec<long_t, 5>{b, c, d, h, w})});
                        }
                    }
                }
            }
        }

        long_t t_len = static_cast<long_t>(all.size());

        long_t len        = t_len / (Threads * hyperthreads);
        long_t full       = t_len % (Threads * hyperthreads);
        long_t full_start = (len + 1) * full;

        long_t i = 0;

        for (; i < full; ++i)
        {
            schedule_serial_images(i, (len + 1) * i, len + 1, all);
        }
        for (; i < Threads * hyperthreads; ++i)
        {
            schedule_serial_images(i, full_start + len * (i - full), len, all);
        }
    }

    void schedule_serial(long_t thread, long_t b_from, long_t b_len,
                         long_t c_from, long_t c_len, long_t d_from,
                         long_t d_len, long_t h_from, long_t h_len,
                         long_t w_from, long_t w_len)
    {
        for (long_t b = b_from; b < b_from + b_len; ++b)
        {
            for (long_t c = c_from; c < (c_from + c_len); ++c)
            {
                for (long_t d = d_from; d < d_from + d_len; ++d)
                {
                    for (long_t h = h_from; h < h_from + h_len; ++h)
                    {
                        for (long_t w = w_from; w < w_from + w_len; ++w)
                        {
                            individual_in[thread].push_back(
                                {problem::tile_offset(
                                     vec<long_t, 5>{b, c, d, h, w}),
                                 problem::matrix_offset(
                                     vec<long_t, 5>{b, c, d, h, w})});
                        }
                    }
                }
            }
        }
    }

    void schedule_parallel(long_t thread_from, long_t num_threads,
                           long_t b_from, long_t b_len, long_t c_from,
                           long_t c_len, long_t d_from, long_t d_len,
                           long_t h_from, long_t h_len, long_t w_from,
                           long_t w_len)
    {
        if (num_threads == 1)
        {
            schedule_serial(thread_from, b_from, b_len, c_from, c_len, d_from,
                            d_len, h_from, h_len, w_from, w_len);
        }
        else
        {
            long_t n_way = smallest_prime_factor(num_threads);

            if (b_len >= n_way)
            {
                b_split_schedule(n_way, thread_from, num_threads, b_from, b_len,
                                 c_from, c_len, d_from, d_len, h_from, h_len,
                                 w_from, w_len);
            }
            else if (c_len >= n_way)
            {
                c_split_schedule(n_way, thread_from, num_threads, b_from, b_len,
                                 c_from, c_len, d_from, d_len, h_from, h_len,
                                 w_from, w_len);
            }
            else if (d_len >= n_way)
            {
                d_split_schedule(n_way, thread_from, num_threads, b_from, b_len,
                                 c_from, c_len, d_from, d_len, h_from, h_len,
                                 w_from, w_len);
            }
            else if (h_len >= n_way)
            {
                h_split_schedule(n_way, thread_from, num_threads, b_from, b_len,
                                 c_from, c_len, d_from, d_len, h_from, h_len,
                                 w_from, w_len);
            }
            else if (w_len >= n_way)
            {
                w_split_schedule(n_way, thread_from, num_threads, b_from, b_len,
                                 c_from, c_len, d_from, d_len, h_from, h_len,
                                 w_from, w_len);
            }
            else
            {
                schedule_serial(thread_from, b_from, b_len, c_from, c_len,
                                d_from, d_len, h_from, h_len, w_from, w_len);
            }
        }
    }

    void b_split_schedule(long_t n_way, long_t thread_from, long_t num_threads,
                          long_t b_from, long_t b_len, long_t c_from,
                          long_t c_len, long_t d_from, long_t d_len,
                          long_t h_from, long_t h_len, long_t w_from,
                          long_t w_len)
    {
        STRONG_ASSERT(b_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;

        for (long_t i = 0; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from + i * (b_len / n_way), (b_len / n_way),
                              c_from, c_len, d_from, d_len, h_from, h_len,
                              w_from, w_len);
        }
        if (b_len % n_way)
        {
            schedule_parallel(thread_from, num_threads,
                              b_from + n_way * (b_len / n_way), b_len % n_way,
                              c_from, c_len, d_from, d_len, h_from, h_len,
                              w_from, w_len);
        }
    }

    void c_split_schedule(long_t n_way, long_t thread_from, long_t num_threads,
                          long_t b_from, long_t b_len, long_t c_from,
                          long_t c_len, long_t d_from, long_t d_len,
                          long_t h_from, long_t h_len, long_t w_from,
                          long_t w_len)
    {
        STRONG_ASSERT(c_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;

        for (long_t i = 0; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from + i * (c_len / n_way),
                              (c_len / n_way), d_from, d_len, h_from, h_len,
                              w_from, w_len);
        }
        if (c_len % n_way)
        {
            schedule_parallel(thread_from, num_threads, b_from, b_len,
                              c_from + n_way * (c_len / n_way), c_len % n_way,
                              d_from, d_len, h_from, h_len, w_from, w_len);
        }
    }

    void d_split_schedule(long_t n_way, long_t thread_from, long_t num_threads,
                          long_t b_from, long_t b_len, long_t c_from,
                          long_t c_len, long_t d_from, long_t d_len,
                          long_t h_from, long_t h_len, long_t w_from,
                          long_t w_len)
    {
        STRONG_ASSERT(d_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;
        long_t len         = d_len / n_way;
        long_t full        = d_len % n_way;
        long_t full_start  = d_from + (len + 1) * full;
        long_t i           = 0;

        for (; i < full; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from, c_len,
                              d_from + (len + 1) * i, len + 1, h_from, h_len,
                              w_from, w_len);
        }
        for (; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from, c_len,
                              full_start + len * (i - full), len, h_from, h_len,
                              w_from, w_len);
        }
    }

    void h_split_schedule(long_t n_way, long_t thread_from, long_t num_threads,
                          long_t b_from, long_t b_len, long_t c_from,
                          long_t c_len, long_t d_from, long_t d_len,
                          long_t h_from, long_t h_len, long_t w_from,
                          long_t w_len)
    {
        STRONG_ASSERT(h_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;
        long_t len         = h_len / n_way;
        long_t full        = h_len % n_way;
        long_t full_start  = h_from + (len + 1) * full;
        long_t i           = 0;

        for (; i < full; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from, c_len, d_from, d_len,
                              h_from + (len + 1) * i, len + 1, w_from, w_len);
        }
        for (; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from, c_len, d_from, d_len,
                              full_start + len * (i - full), len, w_from,
                              w_len);
        }
    }

    void w_split_schedule(long_t n_way, long_t thread_from, long_t num_threads,
                          long_t b_from, long_t b_len, long_t c_from,
                          long_t c_len, long_t d_from, long_t d_len,
                          long_t h_from, long_t h_len, long_t w_from,
                          long_t w_len)
    {
        STRONG_ASSERT(w_len >= n_way);
        STRONG_ASSERT(num_threads >= n_way);
        STRONG_ASSERT(num_threads % n_way == 0);

        long_t sub_threads = num_threads / n_way;
        long_t len         = w_len / n_way;
        long_t full        = w_len % n_way;
        long_t full_start  = w_from + (len + 1) * full;
        long_t i           = 0;

        for (; i < full; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from, c_len, d_from, d_len,
                              h_from, h_len, w_from + (len + 1) * i, len + 1);
        }
        for (; i < n_way; ++i)
        {
            schedule_parallel(thread_from + i * sub_threads, sub_threads,
                              b_from, b_len, c_from, c_len, d_from, d_len,
                              h_from, h_len, full_start + len * (i - full),
                              len);
        }
    }

    kernel_launcher&                                     launcher_;
    std::array<std::function<void()>, ZNN_NUM_CORES * 2> fns_;

    float const* in_;
    float*       out_;
    float const* kin_;
    float*       kout_;

    using in_transform_fn =
        transform_image<problem::m_size[0], problem::m_size[1],
                        problem::m_size[2], problem::k_size[0],
                        problem::k_size[1], problem::k_size[2],
                        problem::stride[2], problem::stride[3],
                        problem::stride[4], problem::matrices::matrix_stride>;

    using ker_transform_fn = kernel_transform::transform_kernel<
        problem::m_size[0], problem::m_size[1], problem::m_size[2],
        problem::k_size[0], problem::k_size[1], problem::k_size[2],
        kproblem::stride[0], kproblem::stride[1], kproblem::stride[2],
        kproblem::matrices::matrix_stride>;

public:
    double input_transform_gbytes() const
    {
        double bytes = problem::size.prod();
        bytes += problem::size[0] * problem::size[1] * problem::t_size.prod() *
                 problem::num_tiles.prod();
        bytes *= 4;
        return bytes / 1000000000;
    }

    double kernel_transform_gbytes() const
    {
        double bytes = 0;
        if constexpr (TransformKernels)
        {
            bytes += problem::size[1] * Layer::output_transform::size[1] *
                     (kproblem::k_size.prod() + kproblem::t_size.prod());
        }

        bytes *= 4;
        return bytes / 1000000000;
    }

    double gbytes() const
    {
        return input_transform_gbytes() + kernel_transform_gbytes();
    }

    transform(kernel_launcher& kl)
        : launcher_(kl)
    {
        if constexpr (DivSchedule)
        {
            schedule_parallel(0, Threads * hyperthreads, 0, problem::size[0], 0,
                              problem::size[1] / CACHELINE_SIZE, 0,
                              problem::num_tiles[0], 0, problem::num_tiles[1],
                              0, problem::num_tiles[2]);
        }
        else
        {
            schedule_images();
        }

        if constexpr (TransformKernels)
        {
            schedule_kernels();
        }

        for (long_t i = 0; i < Threads * hyperthreads; ++i)
        {
            fns_[i * (3 - hyperthreads)] = [i, this]() {

                SIMD_FLOAT tmp1[problem::tile_elements]
                    __attribute__((aligned(64)));
                SIMD_FLOAT tmp2[problem::tile_elements]
                    __attribute__((aligned(64)));

                for (auto const& e : this->individual_in[i])
                {
                    in_transform_fn::execute(this->in_ + e.first,
                                             this->out_ + e.second,
                                             reinterpret_cast<float*>(tmp1),
                                             reinterpret_cast<float*>(tmp2));
                }

                if constexpr (TransformKernels)
                {
                    for (auto const& e : this->individual_ker[i])
                    {
                        ker_transform_fn::execute(
                            this->kin_ + e.first, this->kout_ + e.second,
                            reinterpret_cast<float*>(tmp1),
                            reinterpret_cast<float*>(tmp2));
                    }
                }
            };
        }
    }

    void execute(float const* __restrict in, float* __restrict out,
                 float const* __restrict kin = nullptr,
                 float* __restrict kout      = nullptr)
    {
        in_  = in;
        out_ = out;

        if constexpr (TransformKernels)
        {
            ZNN_ASSERT(kin != nullptr && kout != nullptr);

            kin_  = kin;
            kout_ = kout;
        }

        else
        {
            static_cast<void>(kin);
            static_cast<void>(kout);
        }

        launcher_.template launch2<UseHT>(&(fns_[0]));
    }
};

} // namespace znn::win::input_transform
