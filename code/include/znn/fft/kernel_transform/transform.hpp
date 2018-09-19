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
#include "znn/fft/kernel_transform/kernel_transform.hpp"
#include "znn/fft/layer.hpp"
#include "znn/util/kernel_launcher.hpp"

namespace znn::fft::kernel_transform
{

template <long_t Threads, class Layer>
class transform
{
private:
    using problem = typename Layer::kernel_transform;

    std::array<std::vector<std::pair<long_t, long_t>>, Threads> individual_ker;

    void
    schedule_serial_kernel(long_t thread, long_t t_from, long_t t_len,
                           std::vector<std::pair<long_t, long_t>> const& ts)
    {
        for (long_t t = t_from; t < t_from + t_len; ++t)
        {
            individual_ker[thread].push_back(ts[t]);
        }
    }

    void schedule_kernels()
    {
        std::vector<std::pair<long_t, long_t>> all;

        for (long_t ifm = 0; ifm < problem::input_channels; ++ifm)
        {
            for (long_t ofm = 0; ofm < problem::output_channels / SIMD_WIDTH;
                 ++ofm)
            {
                all.push_back({problem::tile_offset(ifm, ofm),
                               problem::matrix_offset(ifm, ofm)});
            }
        }

        long_t t_len = static_cast<long_t>(all.size());

        long_t len        = t_len / Threads;
        long_t full       = t_len % Threads;
        long_t full_start = (len + 1) * full;

        long_t i = 0;

        for (; i < full; ++i)
        {
            schedule_serial_kernel(i, (len + 1) * i, len + 1, all);
        }
        for (; i < Threads; ++i)
        {
            schedule_serial_kernel(i, full_start + len * (i - full), len, all);
        }
    }

    kernel_launcher&                   launcher_;
    std::vector<std::function<void()>> fns_;

    float const* in_;
    float*       out_;

    using kernel_transform_fn =
        kernel_fft<problem::k_size[0], problem::k_size[1], problem::k_size[2],
                   problem::t_size[0], problem::t_size[1], problem::t_size[2],
                   problem::stride[0], problem::stride[1], problem::stride[2],
                   problem::matrices::matrix_stride>;

public:
    transform(kernel_launcher& kl)
        : launcher_(kl)
        , fns_(512)
    {
        schedule_kernels();

        for (long_t i = 0; i < Threads; ++i)
        {
            fns_[i * 2] = [i, this]() {

                SIMD_FLOAT tmp[problem::fft_tile_size.prod() * 2]
                    __attribute__((aligned(64)));

                for (auto const& e : this->individual_ker[i])
                {
                    kernel_transform_fn::forward(this->in_ + e.first,
                                                 this->out_ + e.second,
                                                 reinterpret_cast<float*>(tmp));
                }
            };
        }
    }

    void execute(float const* __restrict in, float* __restrict out);
    {
        in_  = in;
        out_ = out;
        launcher_.template launch2<false>(&(fns_[0]));
    }
};

} // namespace znn::fft::input_transform
