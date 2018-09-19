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
#include "znn/fft/input_transform/image_transform.hpp"
#include "znn/fft/kernel_transform/kernel_transform.hpp"
#include "znn/fft/output_transform/image_transform.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"
#include <chrono>
#include <iomanip>
#include <limits>
#include <string>

using namespace znn;
using namespace znn::fft;

int main()
{
    static constexpr long_t D = 1;
    static constexpr long_t H = 10;
    static constexpr long_t W = 10;

    image_fft<D, H, W, H * W * SIMD_WIDTH, W * SIMD_WIDTH, SIMD_WIDTH,
              SIMD_WIDTH * 2>
        it;
    output_fft<D, H, W, H * W * SIMD_WIDTH, W * SIMD_WIDTH, SIMD_WIDTH, 1, 1, 1,
               0>
        ot;

    kernel_fft<D, H, W, D, H, W, H * W * SIMD_WIDTH, W * SIMD_WIDTH, SIMD_WIDTH,
               2 * SIMD_WIDTH>
        kt;

    hbw_array<float> r1(rand_init, D * H * W * SIMD_WIDTH * 2);
    hbw_array<float> r2(rand_init, D * H * W * SIMD_WIDTH * 2);

    hbw_array<float> t(D * H * (W / 2 + 1) * SIMD_WIDTH * 2);

    hbw_array<float> tmp(2 * D * H * (W / 2 + 1) * SIMD_WIDTH);

    kt.forward(r1.data(), t.data(), tmp.data());
    it.forward(r1.data(), t.data(), tmp.data(), r1.data());
    ot.backward(t.data(), r2.data(), tmp.data());
    // r2cb<10, 0, 1, 2>(reinterpret_cast<SIMD_FLOAT*>(r2.data()),
    //                  reinterpret_cast<SIMD_FLOAT const*>(t.data()));

    float max_diff = 0;

    for (long_t i = 0; i < D * W * H * SIMD_WIDTH; ++i)
    {
        max_diff = std::max(max_diff, std::abs(r1.data()[i] - r2.data()[i]));
        std::cout << i << ' ' << r1.data()[i] << ' ' << r2.data()[i] << ' '
                  << max_diff << "\n";
    }

    std::cout << "MAX DIFF: " << max_diff << "\n";
}
