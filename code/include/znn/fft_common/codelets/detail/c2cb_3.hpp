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
template <>
struct c2cb_traits<3, 0>
{
    static constexpr long_t flops           = 18;
    static constexpr long_t operations      = 12;
    static constexpr long_t memory_accesses = 12;
    static constexpr long_t stack_vars      = 14;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 3 -skip-k 0 -name c2cb
 * -standalone */

/*
 * This function contains 12 FP additions, 6 FP multiplications,
 * (or, 6 additions, 0 multiplications, 6 fused multiply/add),
 * 14 stack variables, 2 constants, and 12 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 3 && SkippedOutputs == 0>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T9, T2, T3, T6, T7, T4, Tc, T8, Ta, T5, Tb;
        T1            = ri[0];
        T9            = ii[0];
        T2            = ri[WS(is, 1)];
        T3            = ri[WS(is, 2)];
        T6            = ii[WS(is, 1)];
        T7            = ii[WS(is, 2)];
        T4            = T2 + T3;
        Tc            = T3 - T2;
        T8            = T6 - T7;
        Ta            = T6 + T7;
        T5            = SIMD_FNMADD(KP500000000, T4, T1);
        ro[0]         = T1 + T4;
        Tb            = SIMD_FNMADD(KP500000000, Ta, T9);
        io[0]         = T9 + Ta;
        ro[WS(os, 1)] = SIMD_FMADD(KP866025403, T8, T5);
        ro[WS(os, 2)] = SIMD_FNMADD(KP866025403, T8, T5);
        io[WS(os, 2)] = SIMD_FNMADD(KP866025403, Tc, Tb);
        io[WS(os, 1)] = SIMD_FMADD(KP866025403, Tc, Tb);
    }
}

template <>
struct c2cb_traits<3, 1>
{
    static constexpr long_t flops           = 16;
    static constexpr long_t operations      = 10;
    static constexpr long_t memory_accesses = 10;
    static constexpr long_t stack_vars      = 14;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 3 -skip-k 1 -name c2cb
 * -standalone */

/*
 * This function contains 10 FP additions, 6 FP multiplications,
 * (or, 4 additions, 0 multiplications, 6 fused multiply/add),
 * 14 stack variables, 2 constants, and 10 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 3 && SkippedOutputs == 1>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T9, T2, T3, T6, T7, T4, Tc, T8, Ta, T5, Tb;
        T1            = ri[0];
        T9            = ii[0];
        T2            = ri[WS(is, 1)];
        T3            = ri[WS(is, 2)];
        T6            = ii[WS(is, 1)];
        T7            = ii[WS(is, 2)];
        T4            = T2 + T3;
        Tc            = T3 - T2;
        T8            = T6 - T7;
        Ta            = T6 + T7;
        T5            = SIMD_FNMADD(KP500000000, T4, T1);
        Tb            = SIMD_FNMADD(KP500000000, Ta, T9);
        ro[0]         = SIMD_FMADD(KP866025403, T8, T5);
        ro[WS(os, 1)] = SIMD_FNMADD(KP866025403, T8, T5);
        io[WS(os, 1)] = SIMD_FNMADD(KP866025403, Tc, Tb);
        io[0]         = SIMD_FMADD(KP866025403, Tc, Tb);
    }
}

template <>
struct c2cb_traits<3, 2>
{
    static constexpr long_t flops           = 12;
    static constexpr long_t operations      = 8;
    static constexpr long_t memory_accesses = 8;
    static constexpr long_t stack_vars      = 14;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 3 -skip-k 2 -name c2cb
 * -standalone */

/*
 * This function contains 8 FP additions, 4 FP multiplications,
 * (or, 4 additions, 0 multiplications, 4 fused multiply/add),
 * 14 stack variables, 2 constants, and 8 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 3 && SkippedOutputs == 2>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP577350269, +0.577350269189625764509148780501957455647601751);
    {
        SIMD_FLOAT T1, T9, T2, T3, T5, T6, T4, Tb, T7, Ta, Tc, T8;
        T1    = ri[0];
        T9    = ii[0];
        T2    = ii[WS(is, 2)];
        T3    = ii[WS(is, 1)];
        T5    = ri[WS(is, 1)];
        T6    = ri[WS(is, 2)];
        T4    = T2 - T3;
        Tb    = T3 + T2;
        T7    = T5 + T6;
        Ta    = T5 - T6;
        Tc    = SIMD_FNMADD(KP577350269, Tb, Ta);
        T8    = SIMD_FNMADD(KP577350269, T7, T4);
        io[0] = SIMD_FMADD(KP866025403, Tc, T9);
        ro[0] = SIMD_FMADD(KP866025403, T8, T1);
    }
}
