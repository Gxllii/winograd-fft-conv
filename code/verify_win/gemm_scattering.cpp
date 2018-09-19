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
#include "znn/asm/avx512_with_scatter.hpp"
#include "znn/intrin.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"
#include "znn/util/kernel_launcher.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <type_traits>

using namespace znn::win;
using namespace znn::phi;
void verify_result(float* __restrict c1, float* __restrict c2, long_t OF,
                   long_t chunk_size, long_t tile_num)
{
    for (long_t i = 0; i < tile_num; i++)
    {
        for (long_t o = 0; o < OF; o++)
        {
            long_t c1_index = i * OF + o;
            long_t block    = o / SIMD_WIDTH;
            long_t out_o    = o % SIMD_WIDTH;
            long_t c2_index = block * tile_num * chunk_size * SIMD_WIDTH +
                              i * chunk_size * SIMD_WIDTH + out_o;
           // printf("compare:c1 %d is %f, c2 %d is %f\n", c1_index, c2_index,
           //        c1[c1_index], c2[c2_index]);
            if (c1[c1_index] != c2[c2_index])
            {
                printf("wrong --------c1 %d is %f, c2 is %f\n", c1_index,
                       c1[c1_index], c2[c2_index]);
            }
        }
    }
}
void list_data(float * data, long_t index)
{
	for(long_t i =0; i< index; i++)
	{
		printf("i %d is %f \n", i, data[i]);
	}
}
int main()
{
    // float *a, *b, *c;

	static const long_t tile_num = 30;
	static const long_t chunk_size = 2;
	static const long_t IF = 64;
	static const long_t OF = 128;

    static const long_t M = 30;
    static const long_t N = 64;
    static const long_t K = 128;

    static const long_t SC_LD0 = chunk_size * SIMD_WIDTH;
    static const long_t SC_LD1 = tile_num * chunk_size * SIMD_WIDTH;

    auto gemm =
        avx512::get_znn_gemm(M, N, K, N, K, K, 0, false, false, false, true);
    auto gemm_scatter =
        avx512::get_znn_gemm(M, N, K, N, K, K, 0, false, false, false, true,
                             true, false, true, SC_LD0, SC_LD1);
        // using e = gemm<3212, 1024, 1024, 256, 256, 256>;
        // e::execute(a, b, c);
    int const NTHREADS = 2;
    std::function<void()> threads[NTHREADS];

    hbw_array<float> a(rand_init, tile_num * IF);
    hbw_array<float> b(rand_init, IF*OF);
    hbw_array<float> c1(zero_init, tile_num*OF);
    hbw_array<float> c2(zero_init, tile_num*OF*chunk_size);
    threads[0] = [&]() {
        gemm(a.data(), b.data(), c1.data(), a.data(), b.data(), c1.data(),
             c1.data());
    };
    threads[1] = [&]() {
        gemm_scatter(a.data(), b.data(), c2.data(), a.data(), b.data(),
                     c2.data(), c2.data()+ SIMD_WIDTH);
    };

    kernel_launcher kl(2, 1);
    kl.launch(threads);
    verify_result(c1.data(), c2.data()+SIMD_WIDTH, OF, chunk_size, tile_num);
	//list_data(c2.data(),tile_num*OF*chunk_size);
}
