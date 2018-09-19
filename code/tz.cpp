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

#include "znn/direct_conv/simple_gemm.hpp"
#include "znn/jit/gemm/gemm.hpp"
#include "znn/tensor/tensor.hpp"

#include <fstream>
#include <map>
#include <mutex>

#include <dlfcn.h>

using namespace znn::jit;
// using namespace znn::win;

void verify_result(float* __restrict c1, float* __restrict c2, long_t len)
{
    float maxx = 0.f;
    float maxy = 0.f;

    for (long_t i = 0; i < len; i++)
    {
        maxx = std::max(std::abs(c1[i] - c2[i]), maxx);
        maxy = std::max(std::max(std::abs(c1[i]), std::abs(c2[i])), maxy);
        if (c1[i] != c2[i])
        {
            printf("wrong --------c1 %d is %f, c2 is %f\n", i, c1[i], c2[i]);
        }
    }

    std::cout << "MAX DIFF: " << maxx << " :: ";
    std::cout << "MAX NUMB: " << maxy << "\n";
}

int main()
{

    for (int n = 16; n <= 256; n += 16)
    {
        for (int k = 16; k <= 256; k += 16)
        {
            for (int m = 1; m <= 47; ++m)
            {
                std::cout << "-----> M: " << m << " N: " << n << " K: " << k
                          << "\n";

                hbw_array<float> a(rand_init, m * n);
                hbw_array<float> b(rand_init, n * k);
                hbw_array<float> c1(rand_init, m * k);
                hbw_array<float> c2(zero_init, m * k);

                c2 = c1;

                // clear_znn_gemms();
                auto gemm10 = get_znn_gemm(m, n, k, n, k, k, 1, 0, false, false,
                                           false, 0, 0, 0, 1, k, 16);

                auto gemm_10 =
                    get_znn_gemm(m, n, k, n, k, k, -1, 0, false, false, false);

                auto gemm11 =
                    get_znn_gemm(m, n, k, n, k, k, 1, 1, false, false, false);

                auto gemm_11 =
                    get_znn_gemm(m, n, k, n, k, k, -1, 1, false, false, false);

                gemm10(a.data(), b.data(), c1.data(), a.data(), b.data(),
                       c1.data(), c1.data());
                simple_gemm(a.data(), b.data(), c2.data(), m, k, n, n, k, k, 1,
                            0);
                verify_result(c1.data(), c2.data(), m * k);

                gemm_10(a.data(), b.data(), c1.data(), a.data(), b.data(),
                        c1.data(), nullptr);
                simple_gemm(a.data(), b.data(), c2.data(), m, k, n, n, k, k, -1,
                            0);
                verify_result(c1.data(), c2.data(), m * k);

                gemm_11(a.data(), b.data(), c1.data(), a.data(), b.data(),
                        c1.data(), nullptr);
                simple_gemm(a.data(), b.data(), c2.data(), m, k, n, n, k, k, -1,
                            1);
                verify_result(c1.data(), c2.data(), m * k);

                gemm11(a.data(), b.data(), c1.data(), a.data(), b.data(),
                       c1.data(), nullptr);
                simple_gemm(a.data(), b.data(), c2.data(), m, k, n, n, k, k, 1,
                            1);
                verify_result(c1.data(), c2.data(), m * k);
            }
        }
    }
}
