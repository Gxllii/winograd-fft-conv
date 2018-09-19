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

template <long_t D, long_t H, long_t W, long_t DS, long_t HS, long_t WS,
          long_t KD, long_t KH, long_t KW, long_t>
struct output_fft
{
    static constexpr long_t     D_TS  = D + KD - 1;
    static constexpr long_t     H_TS  = H + KH - 1;
    static constexpr long_t     W_TS  = (W + KW - 1) / 2 + 1;
    static constexpr long_t     SCALE = D_TS * H_TS * (W + KW - 1);
    static constexpr float      S     = static_cast<float>(1) / SCALE;
    static constexpr SIMD_FLOAT svec  = {S, S, S, S, S, S, S, S,
                                        S, S, S, S, S, S, S, S};

    static constexpr long_t T_ELEM = D_TS * H_TS * W_TS;

    static void backward(float* __restrict in, float* __restrict out,
                         float* __restrict buf)
    {

        SIMD_FLOAT* __restrict buffer = reinterpret_cast<SIMD_FLOAT*>(buf);
        SIMD_FLOAT* __restrict vin    = reinterpret_cast<SIMD_FLOAT*>(in);

        for (long_t h = 0; h < H_TS; ++h)
        {
#pragma unroll(W_TS)
            for (long_t w = 0; w < W_TS; ++w)
            {
#pragma unroll(D_TS)
                for (long_t d = 0; d < D_TS; ++d)
                {
                    buffer[(d * H_TS * W_TS + h * W_TS + w) * 2] =
                        SIMD_ADD(vin[(d * H_TS * W_TS + h * W_TS + w) * 3],
                                 vin[(d * H_TS * W_TS + h * W_TS + w) * 3 + 2]);

                    buffer[(d * H_TS * W_TS + h * W_TS + w) * 2 + 1] =
                        SIMD_ADD(vin[(d * H_TS * W_TS + h * W_TS + w) * 3 + 2],
                                 vin[(d * H_TS * W_TS + h * W_TS + w) * 3 + 1]);
                }

                if constexpr (D_TS > 1)
                {
                    c2cb<D_TS, KD - 1, H_TS * W_TS * 2>(
                        buffer + (h * W_TS + w) * 2,
                        buffer + (h * W_TS + w) * 2);
                }
            }
        }
        for (long_t d = 0; d < D; ++d)
        {
#pragma unroll(W_TS)
            for (long_t w = 0; w < W_TS; ++w)
            {

                if constexpr (H_TS > 1)
                {
                    c2cb<H_TS, KH - 1, W_TS * 2>(
                        buffer + (d * H_TS * W_TS + w) * 2,
                        buffer + (d * H_TS * W_TS + w) * 2);
                }
            }
        }

        for (long_t d = 0; d < D; ++d)
        {
#pragma unroll(H_TS)
            for (long_t h = 0; h < H; ++h)
            {
                r2cb<W + KW - 1, KW - 1, 1, 2>(
                    buffer + (d * H_TS * W_TS + h * W_TS) * 2,
                    buffer + (d * H_TS * W_TS + h * W_TS) * 2);
#pragma unroll(W)
                for (long_t w = 0; w < W; ++w)
                {
                    buffer[(d * H_TS * W_TS + h * W_TS) * 2 + w] =
                        buffer[(d * H_TS * W_TS + h * W_TS) * 2 + w] * svec;
                    SIMD_STREAM(out + d * DS + h * HS + w * WS,
                                buffer[(d * H_TS * W_TS + h * W_TS) * 2 + w]);
                }
            }
        }
    }
};

} // namespace znn::fft3
