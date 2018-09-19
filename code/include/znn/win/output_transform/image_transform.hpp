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

#include "znn/intrin.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"

namespace znn::win::output_transform
{

#include "znn/win/output_transform/formula.hpp"

template <long_t M_D, long_t M_H, long_t M_W, long_t R_D, long_t R_H,
          long_t R_W, long_t D_STRIDE, long_t H_STRIDE, long_t W_STRIDE>
struct transform_image
{
    static void execute(float const* __restrict in, float* __restrict out,
                        float* __restrict b1, float* __restrict b2)
    {
        SIMD_FLOAT* __restrict buffer1 = reinterpret_cast<SIMD_FLOAT*>(b1);
        SIMD_FLOAT* __restrict buffer2 = reinterpret_cast<SIMD_FLOAT*>(b2);

        static_cast<void>(buffer1);

        static const long_t D_TS = M_D + R_D - 1;
        static const long_t H_TS = M_H + R_H - 1;
        static const long_t W_TS = M_W + R_W - 1;

        static constexpr long_t D_Tile_stride = H_TS * W_TS * CACHELINE_SIZE;
        static constexpr long_t H_Tile_stride = W_TS * CACHELINE_SIZE;

        // transform along W (and gather)
        for (long_t d = 0; d < D_TS; ++d)
        {
#pragma unroll(H_TS)
            for (long_t h = 0; h < H_TS; ++h)
            {
                out_image_1d<M_W, R_W, 1, 1>(
                    buffer2 + d * H_TS * M_W + h * M_W,
                    reinterpret_cast<SIMD_FLOAT const*>(in + d * D_Tile_stride +
                                                        h * H_Tile_stride));
            }
        }

        if constexpr (D_TS == 1)
        {
        // transform along H and scatter
#pragma unroll(M_W)
            for (long_t w = 0; w < M_W; ++w)
            {
                out_image_1d_last<M_H, R_H, M_W, H_STRIDE>(out + w * W_STRIDE,
                                                           buffer2 + w);
            }
        }
        else
        {
            // transform along H
            for (long_t d = 0; d < D_TS; ++d)
            {
#pragma unroll(M_W)
                for (long_t w = 0; w < M_W; ++w)
                {
                    out_image_1d<M_H, R_H, M_W, M_W>(
                        buffer1 + d * M_H * M_W + w,
                        buffer2 + d * H_TS * M_W + w);
                }
            }

            // transform along D (and scatter)
            for (long_t h = 0; h < M_H; ++h)
            {
#pragma unroll(M_W)
                for (long_t w = 0; w < M_W; ++w)
                {
                    out_image_1d_last<M_D, R_D, M_W * M_H, D_STRIDE>(
                        out + h * H_STRIDE + w * W_STRIDE,
                        buffer1 + h * M_W + w);
                }
            }
        }
    }
};

} // namespace znn::win::output_transform
