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

#include "znn/util/kernel_launcher.hpp"
#include "znn/win/input_transform/transform.hpp"
#include "znn/win/output_transform/transform.hpp"
#include "znn/win/pointwise/pointwise.hpp"

namespace znn
{
namespace win
{

template <long_t Cores, long_t Threads, class Layer,
          bool TransformKernels = true, bool HTTransforms = false>
class propagation
{
private:
    static inline constexpr long_t cacheline_pad(long_t s)
    {
        return ((s + CACHELINE_SIZE - 1) / CACHELINE_SIZE) * CACHELINE_SIZE;
    }

public:
    static constexpr long_t o_offset =
        cacheline_pad(Layer::matrices::Bs::memory_in_floats);

    static constexpr long_t c_offset =
        o_offset + cacheline_pad(Layer::matrices::As::memory_in_floats);

    static constexpr long_t c_f_offset =
        c_offset + cacheline_pad(Layer::matrices::Cs::memory_in_floats);

    static constexpr long_t buffer_floats =
        c_f_offset + Layer::output_transform::buffer_memory;

    static constexpr long_t buffer_memory = buffer_floats * sizeof(float);

private:
    kernel_launcher launcher;

    input_transform::transform<Cores, Layer, TransformKernels, HTTransforms>
                                                            input_trans;
    pointwise::mmm<Cores, (Threads == 2), Layer>            pointwise_comp;
    output_transform::transform<Cores, Layer, HTTransforms> output_trans;

public:
    propagation(bool pf1 = true, bool pf2 = true)
        : launcher(Cores, 2)
        , input_trans(launcher)
        , pointwise_comp(launcher, pf1, pf2)
        , output_trans(launcher)
    {
    }

    vec<double, 3> complexity() const
    {
        return {input_trans.gbytes(), pointwise_comp.gflops(),
                output_trans.gbytes()};
    }

    vec<double, 3> actual_complexity() const
    {
        return {input_trans.gbytes(), pointwise_comp.actual_gflops(),
                output_trans.gbytes()};
    }

    vec<double, 3> execute(float const* __restrict in,
                           float const* __restrict ker, float* __restrict out,
                           float* __restrict buffer)
    {
        auto begin = std::chrono::high_resolution_clock::now();

        if constexpr (TransformKernels)
        {
            // input transform
            input_trans.execute(in, buffer + o_offset, ker, buffer);
        }
        else
        {
            // input transform
            input_trans.execute(in, buffer + o_offset, nullptr, nullptr);
            static_cast<void>(ker);
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto time1 = duration_in_ms(end - begin);

        // pointwise
        pointwise_comp(buffer + o_offset, buffer, buffer + c_offset,
                       buffer + c_f_offset);

        begin = std::exchange(end, std::chrono::high_resolution_clock::now());
        auto time2 = duration_in_ms(end - begin);

        // output
        output_trans.execute(buffer + c_f_offset, out);

        begin = std::exchange(end, std::chrono::high_resolution_clock::now());
        auto time3 = duration_in_ms(end - begin);

        return {time1, time2, time3};
    }
};

} // namespace win
} // namespace znn
