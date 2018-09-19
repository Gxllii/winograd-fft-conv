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
    static constexpr long_t H = 32;
    static constexpr long_t W = 32;

    static constexpr long_t KD = 1;
    static constexpr long_t KH = 3;
    static constexpr long_t KW = 3;

    static constexpr long_t OD = D + 1 - KD;
    static constexpr long_t OH = H + 1 - KH;
    static constexpr long_t OW = W + 1 - KW;

    image_fft<D, H, W, H * W * SIMD_WIDTH, W * SIMD_WIDTH, SIMD_WIDTH,
              SIMD_WIDTH * 2>
        it;
    output_fft<OD, OH, OW, OH * OW * SIMD_WIDTH, OW * SIMD_WIDTH, SIMD_WIDTH,
               KD, KH, KW, 0>
        ot;

    kernel_fft<KD, KH, KW, D, H, W, KH * KW * SIMD_WIDTH, KW * SIMD_WIDTH,
               SIMD_WIDTH, 2 * SIMD_WIDTH>
        kt;

    hbw_tensor<float, 4> id(rand_init, D, H, W, SIMD_WIDTH);
    hbw_tensor<float, 4> fd(rand_init, KD, KH, KW, SIMD_WIDTH);
    hbw_tensor<float, 4> od(one_init, OD, OH, OW, SIMD_WIDTH);
    hbw_tensor<float, 4> xd(one_init, OD, OH, OW, SIMD_WIDTH);

    hbw_tensor<float, 5> ti(D, H, (W / 2 + 1), 2, SIMD_WIDTH);
    hbw_tensor<float, 5> tk(D, H, (W / 2 + 1), 2, SIMD_WIDTH);
    hbw_tensor<float, 5> to(D, H, (W / 2 + 1), 2, SIMD_WIDTH);

    hbw_array<float> tmp(D * H * (W / 2 + 1) * SIMD_WIDTH * 2);

    it.forward(id.data(), ti.data(), tmp.data(), id.data());
    kt.forward(fd.data(), tk.data(), tmp.data());

    for (long_t d = 0; d < D; ++d)
    {
        for (long_t h = 0; h < H; ++h)
        {
            for (long_t w = 0; w < W / 2 + 1; ++w)
            {
                for (long_t s = 0; s < 16; ++s)
                {
                    to[d][h][w][0][s] = ti[d][h][w][0][s] * tk[d][h][w][0][s] -
                                        ti[d][h][w][1][s] * tk[d][h][w][1][s];
                    to[d][h][w][1][s] = ti[d][h][w][1][s] * tk[d][h][w][0][s] +
                                        ti[d][h][w][0][s] * tk[d][h][w][1][s];
                }
            }
        }
    }

    for (long_t s = 0; s < 16; ++s)
    {
        for (long_t d = 0; d < OD; ++d)
        {
            for (long_t h = 0; h < OH; ++h)
            {
                for (long_t w = 0; w < OW; ++w)
                {
                    xd[d][h][w][s] = 0;
                    for (long_t kd = 0; kd < KD; ++kd)
                    {
                        for (long_t kh = 0; kh < KH; ++kh)
                        {
                            for (long_t kw = 0; kw < KW; ++kw)
                            {
                                xd[d][h][w][s] +=
                                    id[d + kd][h + kh][w + kw][s] *
                                    fd[KD - kd - 1][KH - kh - 1][KW - kw - 1]
                                      [s];
                            }
                        }
                    }
                }
            }
        }
    }

    ot.backward(to.data(), od.data(), tmp.data());

    float max_diff = 0;

    for (long_t s = 0; s < 16; ++s)
    {
        for (long_t d = 0; d < OD; ++d)
        {
            for (long_t h = 0; h < OH; ++h)
            {
                for (long_t w = 0; w < OW; ++w)
                {
                    std::cout << (od[d][h][w][s] - xd[d][h][w][s]) << ' ';
                    max_diff = std::max(
                        max_diff, std::abs(od[d][h][w][s] - xd[d][h][w][s]));
                }
                std::cout << "\n";
            }
            std::cout << "\n\n";
        }
    }

    std::cout << "MAX DIFF: " << max_diff << "\n";
}
