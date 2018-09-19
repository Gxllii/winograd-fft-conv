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
struct r2cb_traits<6, 0>
{
    static constexpr long_t flops           = 20;
    static constexpr long_t operations      = 14;
    static constexpr long_t memory_accesses = 12;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -sign 1 -n 6 -skip-k 0 -name r2cb
 * -standalone */

/*
 * This function contains 14 FP additions, 6 FP multiplications,
 * (or, 8 additions, 0 multiplications, 6 fused multiply/add),
 * 16 stack variables, 2 constants, and 12 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && SkippedOutputs == 0>::type
r2cb(SIMD_FLOAT* __restrict R0, SIMD_FLOAT const* __restrict Cr,
     SIMD_FLOAT const* __restrict Ci)
{
    DK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, Ta, Tb, T4, T5, T3, T7, Tc, Te, T6, T8, Td, T9;
        T1            = Cr[0];
        T2            = Cr[WS(cs, 3)];
        Ta            = Ci[WS(cs, 2)];
        Tb            = Ci[WS(cs, 1)];
        T4            = Cr[WS(cs, 2)];
        T5            = Cr[WS(cs, 1)];
        T3            = T1 + T2;
        T7            = T1 - T2;
        Tc            = Ta - Tb;
        Te            = Ta + Tb;
        T6            = T4 + T5;
        T8            = T5 - T4;
        Td            = T7 + T8;
        R0[WS(rs, 3)] = SIMD_FNMADD(KP2_000000000, T8, T7);
        T9            = T3 - T6;
        R0[0]         = SIMD_FMADD(KP2_000000000, T6, T3);
        R0[WS(rs, 5)] = SIMD_FMADD(KP1_732050807, Te, Td);
        R0[WS(rs, 1)] = SIMD_FNMADD(KP1_732050807, Te, Td);
        R0[WS(rs, 2)] = SIMD_FMADD(KP1_732050807, Tc, T9);
        R0[WS(rs, 4)] = SIMD_FNMADD(KP1_732050807, Tc, T9);
    }
}

template <>
struct r2cb_traits<6, 1>
{
    static constexpr long_t flops           = 18;
    static constexpr long_t operations      = 13;
    static constexpr long_t memory_accesses = 11;
    static constexpr long_t stack_vars      = 14;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -sign 1 -n 6 -skip-k 1 -name r2cb
 * -standalone */

/*
 * This function contains 13 FP additions, 5 FP multiplications,
 * (or, 8 additions, 0 multiplications, 5 fused multiply/add),
 * 14 stack variables, 2 constants, and 11 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && SkippedOutputs == 1>::type
r2cb(SIMD_FLOAT* __restrict R0, SIMD_FLOAT const* __restrict Cr,
     SIMD_FLOAT const* __restrict Ci)
{
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
    DK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
    {
        SIMD_FLOAT T8, T9, T1, T5, T2, T4, Ta, Tc, T3, T7, T6, Tb;
        T8            = Ci[WS(cs, 2)];
        T9            = Ci[WS(cs, 1)];
        T1            = Cr[0];
        T5            = Cr[WS(cs, 2)];
        T2            = Cr[WS(cs, 3)];
        T4            = Cr[WS(cs, 1)];
        Ta            = T8 - T9;
        Tc            = T8 + T9;
        T3            = T1 - T2;
        T7            = T1 + T2 - T4 - T5;
        T6            = T4 - T5;
        R0[WS(rs, 3)] = SIMD_FNMADD(KP1_732050807, Ta, T7);
        R0[WS(rs, 1)] = SIMD_FMADD(KP1_732050807, Ta, T7);
        Tb            = T3 + T6;
        R0[WS(rs, 2)] = SIMD_FNMADD(KP2_000000000, T6, T3);
        R0[WS(rs, 4)] = SIMD_FMADD(KP1_732050807, Tc, Tb);
        R0[0]         = SIMD_FNMADD(KP1_732050807, Tc, Tb);
    }
}

template <>
struct r2cb_traits<6, 2>
{
    static constexpr long_t flops           = 16;
    static constexpr long_t operations      = 12;
    static constexpr long_t memory_accesses = 10;
    static constexpr long_t stack_vars      = 12;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -sign 1 -n 6 -skip-k 2 -name r2cb
 * -standalone */

/*
 * This function contains 12 FP additions, 4 FP multiplications,
 * (or, 8 additions, 0 multiplications, 4 fused multiply/add),
 * 12 stack variables, 2 constants, and 10 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && SkippedOutputs == 2>::type
r2cb(SIMD_FLOAT* __restrict R0, SIMD_FLOAT const* __restrict Cr,
     SIMD_FLOAT const* __restrict Ci)
{
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
    DK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
    {
        SIMD_FLOAT T8, T9, T1, T5, T2, T4, Ta, T3, T7, T6;
        T8            = Ci[WS(cs, 2)];
        T9            = Ci[WS(cs, 1)];
        T1            = Cr[0];
        T5            = Cr[WS(cs, 2)];
        T2            = Cr[WS(cs, 3)];
        T4            = Cr[WS(cs, 1)];
        Ta            = T8 - T9;
        T3            = T1 - T2;
        T7            = T1 + T2 - T4 - T5;
        T6            = T4 - T5;
        R0[WS(rs, 2)] = SIMD_FNMADD(KP1_732050807, Ta, T7);
        R0[0]         = SIMD_FMADD(KP1_732050807, Ta, T7);
        R0[WS(rs, 3)] = T3 + SIMD_FMADD(KP1_732050807, T8 + T9, T6);
        R0[WS(rs, 1)] = SIMD_FNMADD(KP2_000000000, T6, T3);
    }
}

template <>
struct r2cb_traits<6, 3>
{
    static constexpr long_t flops           = 15;
    static constexpr long_t operations      = 11;
    static constexpr long_t memory_accesses = 9;
    static constexpr long_t stack_vars      = 10;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -sign 1 -n 6 -skip-k 3 -name r2cb
 * -standalone */

/*
 * This function contains 11 FP additions, 4 FP multiplications,
 * (or, 7 additions, 0 multiplications, 4 fused multiply/add),
 * 10 stack variables, 2 constants, and 9 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && SkippedOutputs == 3>::type
r2cb(SIMD_FLOAT* __restrict R0, SIMD_FLOAT const* __restrict Cr,
     SIMD_FLOAT const* __restrict Ci)
{
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
    DK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
    {
        SIMD_FLOAT T7, T4, T5, T8, T1, T2, T6, T3;
        T7            = Ci[WS(cs, 1)];
        T4            = Cr[WS(cs, 1)];
        T5            = Cr[WS(cs, 2)];
        T8            = Ci[WS(cs, 2)];
        T1            = Cr[0];
        T2            = Cr[WS(cs, 3)];
        T6            = T4 - T5;
        R0[WS(rs, 1)] = T1 + SIMD_FMADD(KP1_732050807, T7, T2) -
                        (SIMD_FMADD(KP1_732050807, T8, T4)) - T5;
        T3            = T1 - T2;
        R0[WS(rs, 2)] = T3 + SIMD_FMADD(KP1_732050807, T7 + T8, T6);
        R0[0]         = SIMD_FNMADD(KP2_000000000, T6, T3);
    }
}

template <>
struct r2cb_traits<6, 4>
{
    static constexpr long_t flops           = 8;
    static constexpr long_t operations      = 6;
    static constexpr long_t memory_accesses = 8;
    static constexpr long_t stack_vars      = 9;
    static constexpr long_t constants       = 1;
};
/* Generated by: ./gen_r2cb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -sign 1 -n 6 -skip-k 4 -name r2cb
 * -standalone */

/*
 * This function contains 6 FP additions, 2 FP multiplications,
 * (or, 4 additions, 0 multiplications, 2 fused multiply/add),
 * 9 stack variables, 1 constants, and 8 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && SkippedOutputs == 4>::type
r2cb(SIMD_FLOAT* __restrict R0, SIMD_FLOAT const* __restrict Cr,
     SIMD_FLOAT const* __restrict Ci)
{
    DK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
    {
        SIMD_FLOAT T1, T3, T2, T5, T7, T6, T4, T8;
        T1            = Cr[0];
        T3            = Cr[WS(cs, 2)];
        T2            = Ci[WS(cs, 1)];
        T5            = Cr[WS(cs, 1)];
        T7            = Cr[WS(cs, 3)];
        T6            = Ci[WS(cs, 2)];
        T4            = SIMD_FMADD(KP1_732050807, T2, T1) - T3;
        T8            = SIMD_FMADD(KP1_732050807, T6, T5) - T7;
        R0[WS(rs, 1)] = T8 + T4;
        R0[0]         = T4 - T8;
    }
}

template <>
struct r2cb_traits<6, 5>
{
    static constexpr long_t flops           = 6;
    static constexpr long_t operations      = 5;
    static constexpr long_t memory_accesses = 7;
    static constexpr long_t stack_vars      = 7;
    static constexpr long_t constants       = 1;
};
/* Generated by: ./gen_r2cb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -sign 1 -n 6 -skip-k 5 -name r2cb
 * -standalone */

/*
 * This function contains 5 FP additions, 1 FP multiplications,
 * (or, 4 additions, 0 multiplications, 1 fused multiply/add),
 * 7 stack variables, 1 constants, and 7 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && SkippedOutputs == 5>::type
r2cb(SIMD_FLOAT* __restrict R0, SIMD_FLOAT const* __restrict Cr,
     SIMD_FLOAT const* __restrict Ci)
{
    DK(KP1_732050807, +1.732050807568877293527446341505872366942805254);
    {
        SIMD_FLOAT T1, T3, T6, T5, T4, T2;
        T1    = Cr[0];
        T3    = Ci[WS(cs, 2)];
        T6    = Cr[WS(cs, 2)];
        T5    = Cr[WS(cs, 3)];
        T4    = Ci[WS(cs, 1)];
        T2    = Cr[WS(cs, 1)];
        R0[0] = T1 + SIMD_FMADD(KP1_732050807, T3 + T4, T2) - (T5 + T6);
    }
}