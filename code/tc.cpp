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
#include "znn/jit/cgemm/cgemm.hpp"
#include "znn/tensor/tensor.hpp"

#include <fstream>
#include <map>
#include <mutex>

#include <dlfcn.h>

using namespace znn::jit;
// using namespace znn::win;

inline void simple_cgemm(float const* __restrict a, float const* __restrict b,
                         float* __restrict c, long_t M, long_t N, long_t K,
                         long_t LDA, long_t LDB, long_t LDC, long_t alpha = 1,
                         long_t beta = 1)
{
    for (long_t m = 0; m < M; ++m)
    {
        for (long_t n = 0; n < N; ++n)
        {
            long_t nr = (n / 16) * 32 + (n % 16);
            long_t ni = nr + 16;

            if (beta == 0)
            {
                c[m * LDC + nr] = 0;
                c[m * LDC + ni] = 0;
            }

            for (long_t k = 0; k < K; ++k)
            {
                long_t kr = (k / 16) * 32 + (k % 16);
                long_t ki = kr + 16;

                if (alpha == 1)
                {
                    c[m * LDC + nr] += a[m * LDA + kr] * b[k * LDB + nr];
                    c[m * LDC + nr] -= a[m * LDA + ki] * b[k * LDB + ni];

                    c[m * LDC + ni] += a[m * LDA + ki] * b[k * LDB + nr];
                    c[m * LDC + ni] += a[m * LDA + kr] * b[k * LDB + ni];
                }
                else
                {
                    c[m * LDC + nr] -= a[m * LDA + kr] * b[k * LDB + nr];
                    c[m * LDC + nr] += a[m * LDA + ki] * b[k * LDB + ni];

                    c[m * LDC + ni] -= a[m * LDA + ki] * b[k * LDB + nr];
                    c[m * LDC + ni] -= a[m * LDA + kr] * b[k * LDB + ni];
                }
            }
        }
    }
}

void verify_result(float* __restrict c1, float* __restrict c2, long_t len)
{
    float maxx = 0.f;
    float maxy = 0.f;

    for (long_t i = 0; i < len; i++)
    {
        maxx = std::max(std::abs(c1[i] - c2[i]), maxx);
        maxy = std::max(std::max(std::abs(c1[i]), std::abs(c2[i])), maxy);
        if (std::abs(c1[i] - c2[i]) > 5e-6)
        {
            printf("wrong --------c1 %d is %f, c2 is %f\n", i, c1[i], c2[i]);
        }
    }

    std::cout << "MAX DIFF: " << maxx << " :: ";
    std::cout << "MAX NUMB: " << maxy << "\n";
}

int main()
{

    for (int n = 32; n <= 256; n += 16)
    {
        for (int k = 32; k <= 256; k += 16)
        {
            for (int m = 1; m <= 47; ++m)
            {
                std::cout << "-----> M: " << m << " N: " << n << " K: " << k
                          << "\n";

                hbw_array<float> a(rand_init, m * n * 2);
                hbw_array<float> b(rand_init, n * k * 2);
                hbw_array<float> c1(rand_init, m * k * 2);
                hbw_array<float> c2(zero_init, m * k * 2);

                c2 = c1;

                auto cgemm10 =
                    get_znn_cgemm(m, n, k, n * 2, k * 2, k * 2, 1, 0, false,
                                  false, false, 0, 0, 0, 1, k * 2, 32);

                auto cgemm_10 = get_znn_cgemm(m, n, k, n * 2, k * 2, k * 2, -1,
                                              0, false, false, false);

                auto cgemm11 = get_znn_cgemm(m, n, k, n * 2, k * 2, k * 2, 1, 1,
                                             false, false, false);

                auto cgemm_11 = get_znn_cgemm(m, n, k, n * 2, k * 2, k * 2, -1,
                                              1, false, false, false);

                cgemm10(a.data(), b.data(), c1.data(), a.data(), b.data(),
                        c1.data(), c1.data());
                simple_cgemm(a.data(), b.data(), c2.data(), m, k, n, n * 2,
                             k * 2, k * 2, 1, 0);
                verify_result(c1.data(), c2.data(), m * k * 2);

                cgemm_10(a.data(), b.data(), c1.data(), a.data(), b.data(),
                         c1.data(), nullptr);
                simple_cgemm(a.data(), b.data(), c2.data(), m, k, n, n * 2,
                             k * 2, k * 2, -1, 0);
                verify_result(c1.data(), c2.data(), m * k * 2);

                cgemm_11(a.data(), b.data(), c1.data(), a.data(), b.data(),
                         c1.data(), nullptr);
                simple_cgemm(a.data(), b.data(), c2.data(), m, k, n, n * 2,
                             k * 2, k * 2, -1, 1);
                verify_result(c1.data(), c2.data(), m * k * 2);

                cgemm11(a.data(), b.data(), c1.data(), a.data(), b.data(),
                        c1.data(), nullptr);
                simple_cgemm(a.data(), b.data(), c2.data(), m, k, n, n * 2,
                             k * 2, k * 2, 1, 1);
                verify_result(c1.data(), c2.data(), m * k * 2);
            }
        }
    }
}
