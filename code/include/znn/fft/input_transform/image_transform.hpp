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

#include "znn/fft/codelets/codelets.hpp"

namespace znn::fft
{

template <long_t D, long_t H, long_t W, long_t DS, long_t HS, long_t WS,
          long_t OS>
struct image_fft
{
    static_assert(WS % SIMD_WIDTH == 0);

    static void forward(float const* __restrict in, float* __restrict out,
                        float* __restrict buf, float const* __restrict)
    {
        static constexpr long_t D_TS = D;
        static constexpr long_t H_TS = H;
        static constexpr long_t W_TS = W / 2 + 1;

        SIMD_FLOAT* __restrict buffer = reinterpret_cast<SIMD_FLOAT*>(buf);

        // First along width
        for (long_t d = 0; d < D_TS; ++d)
        {
#pragma unroll(H_TS)
            for (long_t h = 0; h < H_TS; ++h)
            {
                r2cf<W, W, WS / SIMD_WIDTH, 2>(
                    reinterpret_cast<SIMD_FLOAT const*>(in + d * DS + h * HS),
                    buffer + (d * H_TS * W_TS + h * W_TS) * 2);

#pragma unroll(W)
                for (long_t w = 0; w < W; ++w)
                {
                    // SIMD_PREFETCH_L2(nextin + d * DS + h * HS + w * WS);
                }
            }
        }

        for (long_t d = 0; d < D_TS; ++d)
        {
#pragma unroll(W_TS)
            for (long_t w = 0; w < W_TS; ++w)
            {
                if constexpr (H > 1)
                {
                    c2cf<H_TS, H_TS, W_TS * 2>(buffer +
                                               (d * H_TS * W_TS + w) * 2);
                }
                if constexpr (D == 1)
                {
#pragma unroll(H_TS)
                    for (long_t h = 0; h < H_TS; ++h)
                    {
                        SIMD_STREAM(
                            out + OS * (d * W_TS * H_TS + h * W_TS + w),
                            buffer[(d * W_TS * H_TS + h * W_TS + w) * 2]);
                        SIMD_STREAM(
                            out + OS * (d * W_TS * H_TS + h * W_TS + w) +
                                SIMD_WIDTH,
                            buffer[(d * W_TS * H_TS + h * W_TS + w) * 2 + 1]);
                    }
                }
            }
        }

        if constexpr (D > 1)
        {
            for (long_t h = 0; h < H_TS; ++h)
            {
#pragma unroll(W_TS)
                for (long_t w = 0; w < W_TS; ++w)
                {
                    c2cf<D, D, H_TS * W_TS * 2>(buffer + (h * W_TS + w) * 2);
#pragma unroll(D_TS)
                    for (long_t d = 0; d < D_TS; ++d)
                    {
                        SIMD_STREAM(
                            out + OS * (d * W_TS * H_TS + h * W_TS + w),
                            buffer[(d * W_TS * H_TS + h * W_TS + w) * 2]);
                        SIMD_STREAM(
                            out + OS * (d * W_TS * H_TS + h * W_TS + w) +
                                SIMD_WIDTH,
                            buffer[(d * W_TS * H_TS + h * W_TS + w) * 2 + 1]);
                    }
                }
            }
        }
    }
};

} // namespace znn::fft
