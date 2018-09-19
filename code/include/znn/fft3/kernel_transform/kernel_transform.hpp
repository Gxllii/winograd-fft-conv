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

#include "znn/fft3/codelets/codelets.hpp"

namespace znn::fft3
{

template <long_t KD, long_t KH, long_t KW, long_t TD, long_t TH, long_t TW,
          long_t DS, long_t HS, long_t WS, long_t OS, long_t SUBO>
struct kernel_fft
{
    static void forward(float const* __restrict in, float* __restrict out,
                        float* __restrict buf, float const* __restrict)
    {
        static constexpr long_t D_TS = TD;
        static constexpr long_t H_TS = TH;
        static constexpr long_t W_TS = TW / 2 + 1;

        // static constexpr long_t T_ELEM = D_TS * H_TS * W_TS;

        SIMD_FLOAT* __restrict buffer = reinterpret_cast<SIMD_FLOAT*>(buf);

// First along width
#pragma unroll(KD)
        for (long_t d = 0; d < KD; ++d)
        {
#pragma unroll(KH)
            for (long_t h = 0; h < KH; ++h)
            {
                r2cf<TW, KW, WS / SIMD_WIDTH, 2>(
                    reinterpret_cast<SIMD_FLOAT const*>(in + d * DS + h * HS),
                    buffer + (d * H_TS * W_TS + h * W_TS) * 2);
#pragma unroll(KW)
                for (long_t w = 0; w < KW; ++w)
                {
                    // SIMD_PREFETCH_L1(nextin + d * DS + h * HS + w * WS);
                }
            }
        }

#define ZNN_STREAM_ELEMENT(e)                                                  \
    SIMD_STREAM(out + e * OS, SIMD_SUB(buffer[e * 2], buffer[e * 2 + 1]));     \
    SIMD_STREAM(out + e * OS + SUBO,                                           \
                SIMD_ADD(buffer[e * 2], buffer[e * 2 + 1]));                   \
    SIMD_STREAM(out + e * OS + 2 * SUBO, buffer[e * 2 + 1])

        for (long_t d = 0; d < KD; ++d)
        {
#pragma unroll(W_TS)
            for (long_t w = 0; w < W_TS; ++w)
            {
                if constexpr (KH > 1)
                {
                    c2cf<H_TS, KH, W_TS * 2>(buffer +
                                             (d * H_TS * W_TS + w) * 2);
                }

                if constexpr (KD == 1)
                {
#pragma unroll(H_TS)
                    for (long_t h = 0; h < H_TS; ++h)
                    {
                        long_t e = d * H_TS * W_TS + h * W_TS + w;
                        ZNN_STREAM_ELEMENT(e);
                    }
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
                    c2cf<TD, KD, H_TS * W_TS * 2>(buffer + (h * W_TS + w) * 2);
#pragma unroll(D_TS)
                    for (long_t d = 0; d < D_TS; ++d)
                    {
                        long_t e = d * H_TS * W_TS + h * W_TS + w;
                        ZNN_STREAM_ELEMENT(e);
                    }
                }
            }
        }

#undef ZNN_STREAM_ELEMENT

        // Stream the output
    }
};

} // namespace znn::fft3
