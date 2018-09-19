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
#include "znn/types.hpp"

using namespace znn;
void simple_gemm(float const* __restrict A, float const* __restrict B,
                 float* __restrict C, long_t M, long_t N, long_t K, long_t lda,
                 long_t ldb, long_t ldc, long_t alpha = 1, long_t beta = 1)
{

    for (int m = 0; m < M; m += 1)
    {
        for (int n = 0; n < N; n += 1)
        {
            if (beta == 0)
            {
                C[m * ldc + n] = 0;
            }

            for (int k = 0; k < K; k += 1)
            {
                if (alpha == 1)
                {
                    C[m * ldc + n] += A[m * lda + k] * B[k * ldb + n];
                }
                else
                {
                    C[m * ldc + n] -= A[m * lda + k] * B[k * ldb + n];
                }
            }
        }
    }
}
