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
#include "znn/types.hpp"
#include <type_traits>

namespace znn::fft_codelets
{

namespace detail
{

template <long_t, long_t>
struct r2cf_traits;
template <long_t, long_t>
struct r2cb_traits;
template <long_t, long_t>
struct c2cf_traits;
template <long_t, long_t>
struct c2cb_traits;

#define DK(name, val) static constexpr SIMD_FLOAT name = SIMD_SET1_CONST(val)

#define WS(a, b) ((a) * (b))

// Size 2
#include "detail/c2cb_2.hpp"
#include "detail/c2cf_2.hpp"
#include "detail/r2cb_2.hpp"
#include "detail/r2cf_2.hpp"
// Size 3
#include "detail/c2cb_3.hpp"
#include "detail/c2cf_3.hpp"
#include "detail/r2cb_3.hpp"
#include "detail/r2cf_3.hpp"
// Size 4
#include "detail/c2cb_4.hpp"
#include "detail/c2cf_4.hpp"
#include "detail/r2cb_4.hpp"
#include "detail/r2cf_4.hpp"
// Size 5
#include "detail/c2cb_5.hpp"
#include "detail/c2cf_5.hpp"
#include "detail/r2cb_5.hpp"
#include "detail/r2cf_5.hpp"
// Size 6
#include "detail/c2cb_6.hpp"
#include "detail/c2cf_6.hpp"
#include "detail/r2cb_6.hpp"
#include "detail/r2cf_6.hpp"
// Size 7
#include "detail/c2cb_7.hpp"
#include "detail/c2cf_7.hpp"
#include "detail/r2cb_7.hpp"
#include "detail/r2cf_7.hpp"
// Size 8
#include "detail/c2cb_8.hpp"
#include "detail/c2cf_8.hpp"
#include "detail/r2cb_8.hpp"
#include "detail/r2cf_8.hpp"
// Size 9
#include "detail/c2cb_9.hpp"
#include "detail/c2cf_9.hpp"
#include "detail/r2cb_9.hpp"
#include "detail/r2cf_9.hpp"
// Size 10
#include "detail/c2cb_10.hpp"
#include "detail/c2cf_10.hpp"
#include "detail/r2cb_10.hpp"
#include "detail/r2cf_10.hpp"
// Size 11
#include "detail/c2cb_11.hpp"
#include "detail/c2cf_11.hpp"
#include "detail/r2cb_11.hpp"
#include "detail/r2cf_11.hpp"
// Size 12
#include "detail/c2cb_12.hpp"
#include "detail/c2cf_12.hpp"
#include "detail/r2cb_12.hpp"
#include "detail/r2cf_12.hpp"
// Size 13
#include "detail/c2cb_13.hpp"
#include "detail/c2cf_13.hpp"
#include "detail/r2cb_13.hpp"
#include "detail/r2cf_13.hpp"
// Size 14
#include "detail/c2cb_14.hpp"
#include "detail/c2cf_14.hpp"
#include "detail/r2cb_14.hpp"
#include "detail/r2cf_14.hpp"
// Size 15
#include "detail/c2cb_15.hpp"
#include "detail/c2cf_15.hpp"
#include "detail/r2cb_15.hpp"
#include "detail/r2cf_15.hpp"
// Size 16
#include "detail/c2cb_16.hpp"
#include "detail/c2cf_16.hpp"
#include "detail/r2cb_16.hpp"
#include "detail/r2cf_16.hpp"
// Size 17
#include "detail/c2cb_17.hpp"
#include "detail/c2cf_17.hpp"
#include "detail/r2cb_17.hpp"
#include "detail/r2cf_17.hpp"
// Size 18
#include "detail/c2cb_18.hpp"
#include "detail/c2cf_18.hpp"
#include "detail/r2cb_18.hpp"
#include "detail/r2cf_18.hpp"
// Size 19
#include "detail/c2cb_19.hpp"
#include "detail/c2cf_19.hpp"
#include "detail/r2cb_19.hpp"
#include "detail/r2cf_19.hpp"
// Size 20
#include "detail/c2cb_20.hpp"
#include "detail/c2cf_20.hpp"
#include "detail/r2cb_20.hpp"
#include "detail/r2cf_20.hpp"
// Size 21
#include "detail/c2cb_21.hpp"
#include "detail/c2cf_21.hpp"
#include "detail/r2cb_21.hpp"
#include "detail/r2cf_21.hpp"
// Size 22
#include "detail/c2cb_22.hpp"
#include "detail/c2cf_22.hpp"
#include "detail/r2cb_22.hpp"
#include "detail/r2cf_22.hpp"
// Size 23
#include "detail/c2cb_23.hpp"
#include "detail/c2cf_23.hpp"
#include "detail/r2cb_23.hpp"
#include "detail/r2cf_23.hpp"
// Size 24
#include "detail/c2cb_24.hpp"
#include "detail/c2cf_24.hpp"
#include "detail/r2cb_24.hpp"
#include "detail/r2cf_24.hpp"
// Size 25
#include "detail/c2cb_25.hpp"
#include "detail/c2cf_25.hpp"
#include "detail/r2cb_25.hpp"
#include "detail/r2cf_25.hpp"
// Size 26
#include "detail/c2cb_26.hpp"
#include "detail/c2cf_26.hpp"
#include "detail/r2cb_26.hpp"
#include "detail/r2cf_26.hpp"
// Size 27
#include "detail/c2cb_27.hpp"
#include "detail/c2cf_27.hpp"
#include "detail/r2cb_27.hpp"
#include "detail/r2cf_27.hpp"
// Size 28
#include "detail/c2cb_28.hpp"
#include "detail/c2cf_28.hpp"
#include "detail/r2cb_28.hpp"
#include "detail/r2cf_28.hpp"
// Size 29
#include "detail/c2cb_29.hpp"
#include "detail/c2cf_29.hpp"
#include "detail/r2cb_29.hpp"
#include "detail/r2cf_29.hpp"
// Size 30
#include "detail/c2cb_30.hpp"
#include "detail/c2cf_30.hpp"
#include "detail/r2cb_30.hpp"
#include "detail/r2cf_30.hpp"
// Size 31
#include "detail/c2cb_31.hpp"
#include "detail/c2cf_31.hpp"
#include "detail/r2cb_31.hpp"
#include "detail/r2cf_31.hpp"
// Size 32
#include "detail/c2cb_32.hpp"
#include "detail/c2cf_32.hpp"
#include "detail/r2cb_32.hpp"
#include "detail/r2cf_32.hpp"
// Size 64
#include "detail/c2cb_64.hpp"
#include "detail/c2cf_64.hpp"
#include "detail/r2cb_64.hpp"
#include "detail/r2cf_64.hpp"

} // namespace detail

template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline)) void
r2cf_helper(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
            SIMD_FLOAT* __restrict Ci)
{
    detail::r2cf<TransformSize, ProvidedElements, rs, cs>(R0, Cr, Ci);

    if constexpr (ProvidedElements == 1)
    {
        static constexpr long_t to_fill = (TransformSize / 2) + 1;
#pragma unroll(to_fill)
        for (long_t i = 0; i < to_fill; ++i)
        {
            Ci[WS(cs, i)] = SIMD_ZERO();
        }
    }
    else
    {
        Ci[0] = SIMD_ZERO();
        if constexpr ((TransformSize % 2) == 0)
        {
            Ci[WS(cs, TransformSize / 2)] = SIMD_ZERO();
        }
    }
}

template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline)) void r2cf(SIMD_FLOAT const* __restrict R,
                                                SIMD_FLOAT* __restrict C)
{
    r2cf_helper<TransformSize, ProvidedElements, rs, cs>(R, C, C + 1);
}

template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline)) void r2cb(SIMD_FLOAT* __restrict R,
                                                SIMD_FLOAT const* __restrict C)
{
    detail::r2cb<TransformSize, SkippedOutputs, rs, cs>(R, C, C + 1);
}

template <long_t TransformSize, long_t ProvidedElements, long_t ios>
inline __attribute__((always_inline)) void c2cf(SIMD_FLOAT* __restrict d)
{
    detail::c2cf<TransformSize, ProvidedElements, ios, ios>(d, d + 1, d, d + 1);
}

template <long_t TransformSize, long_t SkippedOutputs, long_t ios>
inline __attribute__((always_inline)) void c2cb(SIMD_FLOAT* __restrict s,
                                                SIMD_FLOAT* __restrict d)
{
    detail::c2cb<TransformSize, SkippedOutputs, ios, ios>(s, s + 1, d, d + 1);
}

#undef DK
#undef WS

using detail::c2cb_traits;
using detail::c2cf_traits;
using detail::r2cb_traits;
using detail::r2cf_traits;

} // namespace znn::fft_codelets
