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

#include "znn/fft/codelets/codelets_v.hpp"

namespace znn::fft
{

template <long_t KD, long_t KH, long_t KW, long_t TD, long_t TH, long_t TW,
          long_t DS, long_t HS, long_t WS, long_t OS, typename T>
struct kernel_fft_v
{
    static void forward(T const* __restrict in, T* __restrict out,
                        T* __restrict buf)
    {
        static constexpr long_t D_TS = TD;
        static constexpr long_t H_TS = TH;
        static constexpr long_t W_TS = TW / 2 + 1;

        T* buffer = buf;
        // First along width

#pragma unroll(KD)
        for (long_t d = 0; d < KD; ++d)
        {
#pragma unroll(KH)
            for (long_t h = 0; h < KH; ++h)
            {
                r2cf<TW, KW, WS / SIMD_WIDTH, 2, T>(
                    reinterpret_cast<T const*>(in + d * DS + h * HS),
                    buffer + (d * H_TS * W_TS + h * W_TS) * 2 * SIMD_WIDTH);
            }
        }

        if constexpr (KH > 1)
        {
            for (long_t d = 0; d < KD; ++d)
            {
#pragma unroll(W_TS)
                for (long_t w = 0; w < W_TS; ++w)
                {
                    c2cf<H_TS, KH, W_TS * 2, T>(buffer + (d * H_TS * W_TS + w) *
                                                             2 * SIMD_WIDTH);
                }
            }
        }

        if constexpr (KD > 1)
        {
            for (long_t h = 0; h < H_TS; ++h)
            {
#pragma unroll(W_TS)
                for (long_t w = 0; w < W_TS; ++w)
                {
                    c2cf<TD, KD, H_TS * W_TS * 2, T>(
                        buffer + (h * W_TS + w) * 2 * SIMD_WIDTH);
                }
            }
        }

        // Stream the output
        static constexpr long_t TOTAL_ELEMENTS = D_TS * H_TS * W_TS;

#pragma unroll(TOTAL_ELEMENTS)
        for (long_t e = 0; e < TOTAL_ELEMENTS; ++e)
        {
            for (long_t i = 0; i < SIMD_WIDTH; i++)
            {
                out[e * OS + i] = buffer[e * 2 * SIMD_WIDTH + i];
                out[e * OS + SIMD_WIDTH + i] =
                    buffer[(e * 2 + 1) * SIMD_WIDTH + i];
            }
        }
    }
};
}
