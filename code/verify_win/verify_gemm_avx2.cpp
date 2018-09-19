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
#include "znn/asm/avx2.hpp"
#include "znn/direct_conv/simple_gemm.hpp"
#include "znn/intrin.hpp"
#include "znn/jit/gemm/gemm.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"
#include "znn/util/kernel_launcher.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <type_traits>

using namespace znn::win;
using namespace znn::phi;
using namespace znn::jit;

void verify_result(float* __restrict c1, float* __restrict c2, long_t OF,
                   long_t tile_num)
{
    float maxx = 0.f;
    float maxy = 0.f;

    for (long_t i = 0; i < tile_num * OF; i++)
    {
        maxx = std::max(std::abs(c1[i] - c2[i]), maxx);
        maxy = std::max(std::max(std::abs(c1[i]), std::abs(c2[i])), maxy);
        if (c1[i] != c2[i])
        {
            printf("wrong --------c1 %d is %f, c2 is %f\n", i, c1[i], c2[i]);
        }
    }

    std::cout << "MAX DIFF: " << maxx << "\n";
    std::cout << "MAX NUMB: " << maxy << "\n";
}

void list_data(float* data, long_t index)
{
    for (long_t i = 0; i < index; i++)
    {
        printf("i %d is %f \n", i, data[i]);
    }
}

int main()
{
    for (long_t tile_num = 1; tile_num < 2; ++tile_num)
    {
        for (long_t IF = 48; IF <= 48; IF += 16)
        {
            for (long_t OF = 16; OF <= 16; OF += 16)
            {

                // auto gemm = avx2::get_znn_gemm(tile_num, IF, OF, IF, OF, OF, 0,
                //                                false, false, false, true);
                auto gemm = get_znn_gemm(tile_num, IF, OF, IF, OF, OF, 1, 0,
                                         false, false, false);

                hbw_array<float> a(rand_init, tile_num * IF);
                hbw_array<float> b(rand_init, IF * OF);
                hbw_array<float> c1(zero_init, tile_num * OF);
                hbw_array<float> c2(zero_init, tile_num * OF);

                gemm(a.data(), b.data(), c1.data(), a.data(), b.data(),
                     c1.data(), nullptr);

                std::cout << "DONE GEMM" << std::endl;

                simple_gemm(a.data(), b.data(), c2.data(), tile_num, OF, IF, IF,
                            OF, OF);

                std::cout << "DONE NAIVE GEMM" << std::endl;

                verify_result(c1.data(), c2.data(), OF, tile_num);
            }
        }
    }
}
