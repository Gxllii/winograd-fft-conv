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
#include "znn/fft/output_transform/image_transform.hpp"
#include "znn/util/kernel_launcher.hpp"

namespace znn::fft::output_transform
{

template <long_t Threads, class Layer, bool UseHT = true>
class transform
{
private:
    using problem = typename Layer::output_transform;

    static constexpr long_t hyperthreads = UseHT ? 2 : 1;

    std::array<std::vector<std::pair<long_t, long_t>>, Threads * hyperthreads>
        individual;

    void schedule_serial(long_t thread, long_t t_from, long_t t_len,
                         std::vector<std::pair<long_t, long_t>> const& ts)
    {
        // std::cout << "Thread " << thread << "(" << t_from << ":" << t_len
        //           << ")\n";
        for (long_t t = t_from; t < t_from + t_len; ++t)
        {
            individual[thread].push_back(ts[t]);
        }
    }

    void schedule_parallel()
    {
        std::vector<std::pair<long_t, long_t>> all_tasks;

        for (long_t c = 0; c < problem::size[1] / SIMD_WIDTH; ++c)
        {
            long_t i = 0;
            for (long_t b = 0; b < problem::size[0]; ++b)
            {
                for (long_t d = 0; d < problem::num_tiles[0]; ++d)
                {
                    for (long_t h = 0; h < problem::num_tiles[1]; ++h)
                    {
                        for (long_t w = 0; w < problem::num_tiles[2]; ++w, ++i)
                        {
                            all_tasks.push_back(std::make_pair(
                                problem::flat_tile_offset(c, i),
                                problem::tile_offset(
                                    vec<long_t, 5>{b, c, d, h, w})));
                        }
                    }
                }
            }
        }

        // std::cout << "Total tasks: " << all_tasks.size() << "\n";

        long_t t_len = static_cast<long_t>(all_tasks.size());

        long_t len        = t_len / (Threads * hyperthreads);
        long_t full       = t_len % (Threads * hyperthreads);
        long_t full_start = (len + 1) * full;

        long_t i = 0;

        for (; i < full; ++i)
        {
            schedule_serial(i, (len + 1) * i, len + 1, all_tasks);
        }
        for (; i < Threads * hyperthreads; ++i)
        {
            schedule_serial(i, full_start + len * (i - full), len, all_tasks);
        }
    }

    kernel_launcher&                                     launcher_;
    std::array<std::function<void()>, ZNN_NUM_CORES * 2> fns_;

    float* in_;
    float* out_;

    using transform_fn =
        output_fft<problem::m_size[0], problem::m_size[1], problem::m_size[2],
                   problem::stride[2], problem::stride[3], problem::stride[4],
                   problem::k_size[0], problem::k_size[1], problem::k_size[2],
                   problem::flat_tile_stride>;

public:
    transform(kernel_launcher& kl)
        : launcher_(kl)
    {
        schedule_parallel();
        for (long_t i = 0; i < Threads * hyperthreads; ++i)
        {
            fns_[i * (3 - hyperthreads)] = [i, this]() {

                SIMD_FLOAT tmp[problem::fft_tile_size.prod() * 2]
                    __attribute__((aligned(64)));

                for (auto const& e : this->individual[i])
                {
                    transform_fn::backward(this->in_ + e.first,
                                           this->out_ + e.second,
                                           reinterpret_cast<float*>(tmp));
                }
            };
        }
    }

    double gbytes() const
    {
        double bytes = problem::size.prod();
        bytes += problem::size[0] * problem::size[1] *
                 problem::fft_tile_size.prod() * problem::num_tiles.prod() * 2;
        bytes *= 4;
        return bytes / 1000000000;
    }

    void execute(float* __restrict in, float* __restrict out)
    {
        in_  = in;
        out_ = out;
        launcher_.template launch2<UseHT>(&(fns_[0]));
    }
};

} // namespace znn::fft::output_transform
