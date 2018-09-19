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
#include "znn/asm/avx2_cgemm.hpp"
#include "znn/intrin.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"
#include <chrono>
#include <complex>
#include <iostream>
#include <thread>
#include <type_traits>

static const int S = 4;

using namespace znn;
using namespace znn::win;

inline void naive_cgemm(long_t M, long_t N, long_t K, long_t LDA, long_t LDB,
                        long_t LDC, float const* a, float const* b, float* c)
{
    for (long_t m = 0; m < M; ++m)
    {
        for (long_t k = 0; k < K; ++k)
        {
            for (long_t n = 0; n < N; ++n)
            {
                long_t nr = (n / 8) * 16 + (n % 8);
                long_t ni = nr + 8;
                long_t kr = (k / 8) * 16 + (k % 8);
                long_t ki = kr + 8;

                c[m * LDC + nr] += a[m * LDA + kr] * b[k * LDB + nr];
                c[m * LDC + nr] -= a[m * LDA + ki] * b[k * LDB + ni];

                c[m * LDC + ni] += a[m * LDA + ki] * b[k * LDB + nr];
                c[m * LDC + ni] += a[m * LDA + kr] * b[k * LDB + ni];
            }
        }
    }
}

void verify_result(float* c1, float* c2, long_t len)
{
    float maxx = 0.f;
    float maxy = 0.f;

    for (long_t i = 0; i < len; i++)
    {
        maxx = std::max(std::abs(c1[i] - c2[i]), maxx);
        maxy = std::max(std::max(std::abs(c1[i]), std::abs(c2[i])), maxy);
        // if (c1[i] != c2[i])
        {
            //  printf("wrong --------c1 %d is %f, c2 is %f\n", i, c1[i],
            //  c2[i]);
        }
    }

    std::cout << "MAX DIFF: " << maxx << "\n";
    std::cout << "MAX NUMB: " << maxy << "\n";
}

inline void print_cgemm(long_t R, long_t C, long_t LD, float* d)
{
    for (long_t r = 0; r < R; ++r)
    {
        for (long_t c = 0; c < C; ++c)
        {
            std::cout << std::complex<float>(
                             d[r * LD + (c / 8) * 16 + (c % 8)],
                             d[r * LD + (c / 8) * 16 + (c % 8) + 8])
                      << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

inline void test_gemm(long_t M, long_t N, long_t K)
{
    auto cgemm =
        avx2::get_znn_cgemm(M, N, K, N * 2, K * 2, K * 2, 1, 1, 0, 1, 1, 1);

    hbw_array<float> a(rand_init, M * N * 2);
    hbw_array<float> b(rand_init, N * K * 2);
    hbw_array<float> c(zero_init, M * K * 2);

    hbw_array<float> c2(zero_init, M * K * 2);

    std::cout << "Test for: " << M << ' ' << N << ' ' << K << "\n";
    naive_cgemm(M, K, N, N * 2, K * 2, K * 2, a.data(), b.data(), c2.data());

    cgemm(a.data(), b.data(), c.data(), a.data(), b.data(), c.data());

    verify_result(c.data(), c2.data(), K * M * 2);
}

int main()
{
    //test_gemm(2, 16, 16);
     for (long_t N = 96; N <= 256; N += 32)
     {
         for (long_t K = 96; K <= 256; K += 32)
         {
             if (N * K <= 128 * 128)
             {
                 for (long_t M = 6; M <= 32; ++M)
                 {
                     test_gemm(M, N, K);
                 }
             }
         }
     }
}
