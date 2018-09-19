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
struct r2cf_traits<5, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 4;
    static constexpr long_t stack_vars      = 1;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 5 -first-k 1 -name r2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 1 stack variables, 0 constants, and 4 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 5 && ProvidedElements == 1>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    {
        SIMD_FLOAT T1;
        T1            = R0[0];
        Cr[WS(cs, 1)] = T1;
        Cr[WS(cs, 2)] = T1;
        Cr[0]         = T1;
    }
}

template <>
struct r2cf_traits<5, 2>
{
    static constexpr long_t flops           = 7;
    static constexpr long_t operations      = 5;
    static constexpr long_t memory_accesses = 7;
    static constexpr long_t stack_vars      = 6;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 5 -first-k 2 -name r2cf
 * -standalone */

/*
 * This function contains 3 FP additions, 4 FP multiplications,
 * (or, 1 additions, 2 multiplications, 2 fused multiply/add),
 * 6 stack variables, 4 constants, and 7 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 5 && ProvidedElements == 2>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP809016994, +0.809016994374947424102293417182819058860154590);
    DK(KP309016994, +0.309016994374947424102293417182819058860154590);
    DK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T2;
        T1            = R0[WS(rs, 1)];
        T2            = R0[0];
        Ci[WS(cs, 1)] = -(KP951056516 * T1);
        Ci[WS(cs, 2)] = -(KP587785252 * T1);
        Cr[0]         = T2 + T1;
        Cr[WS(cs, 1)] = SIMD_FMADD(KP309016994, T1, T2);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP809016994, T1, T2);
    }
}

template <>
struct r2cf_traits<5, 3>
{
    static constexpr long_t flops           = 15;
    static constexpr long_t operations      = 10;
    static constexpr long_t memory_accesses = 8;
    static constexpr long_t stack_vars      = 10;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 5 -first-k 3 -name r2cf
 * -standalone */

/*
 * This function contains 8 FP additions, 7 FP multiplications,
 * (or, 3 additions, 2 multiplications, 5 fused multiply/add),
 * 10 stack variables, 4 constants, and 8 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 5 && ProvidedElements == 3>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T3, T1, T2, T6, T4, T5;
        T3            = R0[0];
        T1            = R0[WS(rs, 1)];
        T2            = R0[WS(rs, 2)];
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, T1, T2));
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FMADD(KP618033988, T2, T1)));
        T6            = T1 - T2;
        T4            = T1 + T2;
        T5            = SIMD_FNMADD(KP250000000, T4, T3);
        Cr[0]         = T3 + T4;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, T6, T5);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, T6, T5);
    }
}

template <>
struct r2cf_traits<5, 4>
{
    static constexpr long_t flops           = 17;
    static constexpr long_t operations      = 12;
    static constexpr long_t memory_accesses = 9;
    static constexpr long_t stack_vars      = 13;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 5 -first-k 4 -name r2cf
 * -standalone */

/*
 * This function contains 10 FP additions, 7 FP multiplications,
 * (or, 5 additions, 2 multiplications, 5 fused multiply/add),
 * 13 stack variables, 4 constants, and 9 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 5 && ProvidedElements == 4>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T5, T1, T2, T3, T6, T4, T7, T9, T8;
        T5            = R0[0];
        T1            = R0[WS(rs, 1)];
        T2            = R0[WS(rs, 2)];
        T3            = R0[WS(rs, 3)];
        T6            = T2 + T3;
        T4            = T2 - T3;
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, T1, T4));
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FMADD(KP618033988, T4, T1)));
        T7            = T1 + T6;
        T9            = T1 - T6;
        T8            = SIMD_FNMADD(KP250000000, T7, T5);
        Cr[0]         = T5 + T7;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, T9, T8);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, T9, T8);
    }
}

template <>
struct r2cf_traits<5, 5>
{
    static constexpr long_t flops           = 19;
    static constexpr long_t operations      = 14;
    static constexpr long_t memory_accesses = 10;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 5 -first-k 5 -name r2cf
 * -standalone */

/*
 * This function contains 12 FP additions, 7 FP multiplications,
 * (or, 7 additions, 2 multiplications, 5 fused multiply/add),
 * 16 stack variables, 4 constants, and 10 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 5 && ProvidedElements == 5>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T7, T1, T2, T4, T5, T3, T8, T6, T9, Tc, Ta, Tb;
        T7            = R0[0];
        T1            = R0[WS(rs, 4)];
        T2            = R0[WS(rs, 1)];
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 3)];
        T3            = T1 - T2;
        T8            = T2 + T1;
        T6            = T4 - T5;
        T9            = T4 + T5;
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FMADD(KP618033988, T3, T6));
        Ci[WS(cs, 1)] = KP951056516 * (SIMD_FNMADD(KP618033988, T6, T3));
        Tc            = T8 - T9;
        Ta            = T8 + T9;
        Tb            = SIMD_FNMADD(KP250000000, Ta, T7);
        Cr[0]         = T7 + Ta;
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Tc, Tb);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Tc, Tb);
    }
}