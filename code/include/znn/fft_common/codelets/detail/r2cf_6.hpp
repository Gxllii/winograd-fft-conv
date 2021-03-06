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
struct r2cf_traits<6, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 5;
    static constexpr long_t stack_vars      = 1;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 1 -name r2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 1 stack variables, 0 constants, and 5 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 1>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    {
        SIMD_FLOAT T1;
        T1            = R0[0];
        Cr[WS(cs, 2)] = T1;
        Cr[WS(cs, 1)] = T1;
        Cr[WS(cs, 3)] = T1;
        Cr[0]         = T1;
    }
}

template <>
struct r2cf_traits<6, 2>
{
    static constexpr long_t flops           = 8;
    static constexpr long_t operations      = 6;
    static constexpr long_t memory_accesses = 8;
    static constexpr long_t stack_vars      = 4;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 2 -name r2cf
 * -standalone */

/*
 * This function contains 4 FP additions, 4 FP multiplications,
 * (or, 2 additions, 2 multiplications, 2 fused multiply/add),
 * 4 stack variables, 2 constants, and 8 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 2>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        Cr[WS(cs, 3)] = T1 - T2;
        Cr[0]         = T1 + T2;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP500000000, T2, T1);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP500000000, T2, T1);
        Ci[WS(cs, 2)] = -(KP866025403 * T2);
        Ci[WS(cs, 1)] = -(KP866025403 * T2);
    }
}

template <>
struct r2cf_traits<6, 3>
{
    static constexpr long_t flops           = 10;
    static constexpr long_t operations      = 8;
    static constexpr long_t memory_accesses = 9;
    static constexpr long_t stack_vars      = 7;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 3 -name r2cf
 * -standalone */

/*
 * This function contains 6 FP additions, 4 FP multiplications,
 * (or, 4 additions, 2 multiplications, 2 fused multiply/add),
 * 7 stack variables, 2 constants, and 9 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 3>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T4, T1, T2, T5, T3;
        T4            = R0[0];
        T1            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 1)];
        T5            = T1 + T2;
        T3            = T1 - T2;
        Cr[0]         = T4 + T5;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP500000000, T5, T4);
        Ci[WS(cs, 1)] = -(KP866025403 * T5);
        Cr[WS(cs, 3)] = T4 + T3;
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP500000000, T3, T4);
        Ci[WS(cs, 2)] = KP866025403 * T3;
    }
}

template <>
struct r2cf_traits<6, 4>
{
    static constexpr long_t flops           = 12;
    static constexpr long_t operations      = 10;
    static constexpr long_t memory_accesses = 10;
    static constexpr long_t stack_vars      = 10;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 4 -name r2cf
 * -standalone */

/*
 * This function contains 8 FP additions, 4 FP multiplications,
 * (or, 6 additions, 2 multiplications, 2 fused multiply/add),
 * 10 stack variables, 2 constants, and 10 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 4>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    {
        SIMD_FLOAT T1, T2, T4, T5, T3, T8, T6, T7;
        T1            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 1)];
        T4            = R0[0];
        T5            = R0[WS(rs, 3)];
        T3            = T1 - T2;
        T8            = T1 + T2;
        T6            = T4 - T5;
        T7            = T4 + T5;
        Ci[WS(cs, 1)] = -(KP866025403 * T8);
        Ci[WS(cs, 2)] = KP866025403 * T3;
        Cr[0]         = T7 + T8;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP500000000, T8, T7);
        Cr[WS(cs, 3)] = T6 + T3;
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP500000000, T3, T6);
    }
}

template <>
struct r2cf_traits<6, 5>
{
    static constexpr long_t flops           = 16;
    static constexpr long_t operations      = 14;
    static constexpr long_t memory_accesses = 11;
    static constexpr long_t stack_vars      = 13;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 5 -name r2cf
 * -standalone */

/*
 * This function contains 12 FP additions, 4 FP multiplications,
 * (or, 10 additions, 2 multiplications, 2 fused multiply/add),
 * 13 stack variables, 2 constants, and 11 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 5>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    {
        SIMD_FLOAT T4, T5, T6, T1, T2, T9, T7, Ta, T3, Tb, T8;
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 4)];
        T6            = R0[WS(rs, 1)];
        T1            = R0[0];
        T2            = R0[WS(rs, 3)];
        T9            = T5 + T6;
        T7            = T5 - T6;
        Ta            = T1 + T2;
        T3            = T1 - T2;
        Tb            = T4 + T9;
        Ci[WS(cs, 2)] = KP866025403 * (T4 - T9);
        T8            = T4 + T7;
        Ci[WS(cs, 1)] = KP866025403 * (T7 - T4);
        Cr[0]         = Ta + Tb;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP500000000, Tb, Ta);
        Cr[WS(cs, 3)] = T3 + T8;
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP500000000, T8, T3);
    }
}

template <>
struct r2cf_traits<6, 6>
{
    static constexpr long_t flops           = 18;
    static constexpr long_t operations      = 16;
    static constexpr long_t memory_accesses = 12;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 6 -name r2cf
 * -standalone */

/*
 * This function contains 14 FP additions, 4 FP multiplications,
 * (or, 12 additions, 2 multiplications, 2 fused multiply/add),
 * 16 stack variables, 2 constants, and 12 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 6>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    {
        SIMD_FLOAT T1, T2, T7, T8, T4, T5, T3, Td, T9, Tc, T6, Tb, Te, Ta;
        T1            = R0[0];
        T2            = R0[WS(rs, 3)];
        T7            = R0[WS(rs, 4)];
        T8            = R0[WS(rs, 1)];
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 5)];
        T3            = T1 - T2;
        Td            = T1 + T2;
        T9            = T7 - T8;
        Tc            = T7 + T8;
        T6            = T4 - T5;
        Tb            = T4 + T5;
        Te            = Tb + Tc;
        Ci[WS(cs, 2)] = KP866025403 * (Tb - Tc);
        Ta            = T6 + T9;
        Ci[WS(cs, 1)] = KP866025403 * (T9 - T6);
        Cr[0]         = Td + Te;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP500000000, Te, Td);
        Cr[WS(cs, 3)] = T3 + Ta;
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP500000000, Ta, T3);
    }
}
