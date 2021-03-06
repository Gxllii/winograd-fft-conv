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
struct r2cf_traits<10, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 7;
    static constexpr long_t stack_vars      = 1;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 1 -name r2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 1 stack variables, 0 constants, and 7 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 1>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    {
        SIMD_FLOAT T1;
        T1            = R0[0];
        Cr[WS(cs, 2)] = T1;
        Cr[WS(cs, 3)] = T1;
        Cr[WS(cs, 4)] = T1;
        Cr[WS(cs, 5)] = T1;
        Cr[WS(cs, 1)] = T1;
        Cr[0]         = T1;
    }
}

template <>
struct r2cf_traits<10, 2>
{
    static constexpr long_t flops           = 14;
    static constexpr long_t operations      = 10;
    static constexpr long_t memory_accesses = 12;
    static constexpr long_t stack_vars      = 6;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 2 -name r2cf
 * -standalone */

/*
 * This function contains 6 FP additions, 8 FP multiplications,
 * (or, 2 additions, 4 multiplications, 4 fused multiply/add),
 * 6 stack variables, 4 constants, and 12 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 2>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DK(KP309016994, +0.309016994374947424102293417182819058860154590);
    DK(KP809016994, +0.809016994374947424102293417182819058860154590);
    {
        SIMD_FLOAT T1, T2;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        Cr[WS(cs, 5)] = T1 - T2;
        Cr[WS(cs, 1)] = SIMD_FMADD(KP809016994, T2, T1);
        Cr[WS(cs, 4)] = SIMD_FNMADD(KP809016994, T2, T1);
        Cr[0]         = T1 + T2;
        Cr[WS(cs, 2)] = SIMD_FMADD(KP309016994, T2, T1);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP309016994, T2, T1);
        Ci[WS(cs, 2)] = -(KP951056516 * T2);
        Ci[WS(cs, 4)] = -(KP587785252 * T2);
        Ci[WS(cs, 3)] = -(KP951056516 * T2);
        Ci[WS(cs, 1)] = -(KP587785252 * T2);
    }
}

template <>
struct r2cf_traits<10, 3>
{
    static constexpr long_t flops           = 28;
    static constexpr long_t operations      = 18;
    static constexpr long_t memory_accesses = 13;
    static constexpr long_t stack_vars      = 11;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 3 -name r2cf
 * -standalone */

/*
 * This function contains 14 FP additions, 14 FP multiplications,
 * (or, 4 additions, 4 multiplications, 10 fused multiply/add),
 * 11 stack variables, 4 constants, and 13 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 3>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T2, T3, T5, T4, T6, T7;
        T1            = R0[0];
        T2            = R0[WS(rs, 2)];
        T3            = R0[WS(rs, 1)];
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FMADD(KP618033988, T3, T2)));
        Ci[WS(cs, 3)] = KP951056516 * (SIMD_FMSUB(KP618033988, T2, T3));
        Ci[WS(cs, 2)] = -(KP951056516 * (SIMD_FMADD(KP618033988, T2, T3)));
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FNMADD(KP618033988, T3, T2));
        T5            = T2 + T3;
        T4            = T2 - T3;
        T6            = SIMD_FNMADD(KP250000000, T5, T1);
        Cr[0]         = T1 + T5;
        T7            = SIMD_FNMADD(KP250000000, T4, T1);
        Cr[WS(cs, 5)] = T1 + T4;
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, T4, T6);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, T4, T6);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, T5, T7);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, T5, T7);
    }
}

template <>
struct r2cf_traits<10, 4>
{
    static constexpr long_t flops           = 32;
    static constexpr long_t operations      = 22;
    static constexpr long_t memory_accesses = 14;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 4 -name r2cf
 * -standalone */

/*
 * This function contains 18 FP additions, 14 FP multiplications,
 * (or, 8 additions, 4 multiplications, 10 fused multiply/add),
 * 16 stack variables, 4 constants, and 14 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 4>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T5, T2, T3, T4, T9, Ta, Tc, T6, T8, Tb, T7;
        T1            = R0[0];
        T5            = R0[WS(rs, 1)];
        T2            = R0[WS(rs, 2)];
        T3            = R0[WS(rs, 3)];
        T4            = T2 - T3;
        T9            = T2 + T3;
        Ta            = T9 + T5;
        Tc            = T9 - T5;
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FMADD(KP618033988, T5, T9)));
        Ci[WS(cs, 3)] = KP951056516 * (SIMD_FMSUB(KP618033988, T9, T5));
        T6            = T4 - T5;
        T8            = T4 + T5;
        Ci[WS(cs, 2)] = -(KP951056516 * (SIMD_FMADD(KP618033988, T4, T5)));
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FNMADD(KP618033988, T5, T4));
        Tb            = SIMD_FNMADD(KP250000000, Ta, T1);
        Cr[0]         = T1 + Ta;
        T7            = SIMD_FNMADD(KP250000000, T6, T1);
        Cr[WS(cs, 5)] = T1 + T6;
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Tc, Tb);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Tc, Tb);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, T8, T7);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, T8, T7);
    }
}

template <>
struct r2cf_traits<10, 5>
{
    static constexpr long_t flops           = 34;
    static constexpr long_t operations      = 24;
    static constexpr long_t memory_accesses = 15;
    static constexpr long_t stack_vars      = 19;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 5 -name r2cf
 * -standalone */

/*
 * This function contains 20 FP additions, 14 FP multiplications,
 * (or, 10 additions, 4 multiplications, 10 fused multiply/add),
 * 19 stack variables, 4 constants, and 15 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 5>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T7, T1, T2, T4, T5, Tb, T3, Tc, T6, Tf, Td, Ta, T8, Te, T9;
        T7            = R0[0];
        T1            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 3)];
        T4            = R0[WS(rs, 4)];
        T5            = R0[WS(rs, 1)];
        Tb            = T1 + T2;
        T3            = T1 - T2;
        Tc            = T4 + T5;
        T6            = T4 - T5;
        Tf            = Tb - Tc;
        Td            = Tb + Tc;
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FMADD(KP618033988, Tc, Tb)));
        Ci[WS(cs, 3)] = -(KP951056516 * (SIMD_FNMADD(KP618033988, Tb, Tc)));
        Ta            = T3 - T6;
        T8            = T3 + T6;
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, T3, T6));
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FMADD(KP618033988, T6, T3));
        Te            = SIMD_FNMADD(KP250000000, Td, T7);
        Cr[0]         = T7 + Td;
        T9            = SIMD_FNMADD(KP250000000, T8, T7);
        Cr[WS(cs, 5)] = T7 + T8;
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Tf, Te);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Tf, Te);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, Ta, T9);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Ta, T9);
    }
}

template <>
struct r2cf_traits<10, 6>
{
    static constexpr long_t flops           = 36;
    static constexpr long_t operations      = 26;
    static constexpr long_t memory_accesses = 16;
    static constexpr long_t stack_vars      = 22;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 6 -name r2cf
 * -standalone */

/*
 * This function contains 22 FP additions, 14 FP multiplications,
 * (or, 12 additions, 4 multiplications, 10 fused multiply/add),
 * 22 stack variables, 4 constants, and 16 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 6>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T7, T8, T1, T2, T4, T5, T9, Tf, T3, Td, T6, Te, Tg, Ti, Ta;
        SIMD_FLOAT Tc, Th, Tb;
        T7            = R0[0];
        T8            = R0[WS(rs, 5)];
        T1            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 3)];
        T4            = R0[WS(rs, 4)];
        T5            = R0[WS(rs, 1)];
        T9            = T7 - T8;
        Tf            = T7 + T8;
        T3            = T1 - T2;
        Td            = T1 + T2;
        T6            = T4 - T5;
        Te            = T4 + T5;
        Tg            = Td + Te;
        Ti            = Td - Te;
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FMADD(KP618033988, Te, Td)));
        Ci[WS(cs, 3)] = -(KP951056516 * (SIMD_FNMADD(KP618033988, Td, Te)));
        Ta            = T3 + T6;
        Tc            = T3 - T6;
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, T3, T6));
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FMADD(KP618033988, T6, T3));
        Th            = SIMD_FNMADD(KP250000000, Tg, Tf);
        Cr[0]         = Tf + Tg;
        Tb            = SIMD_FNMADD(KP250000000, Ta, T9);
        Cr[WS(cs, 5)] = T9 + Ta;
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Ti, Th);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Ti, Th);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, Tc, Tb);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Tc, Tb);
    }
}

template <>
struct r2cf_traits<10, 7>
{
    static constexpr long_t flops           = 40;
    static constexpr long_t operations      = 30;
    static constexpr long_t memory_accesses = 17;
    static constexpr long_t stack_vars      = 27;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 7 -name r2cf
 * -standalone */

/*
 * This function contains 26 FP additions, 14 FP multiplications,
 * (or, 16 additions, 4 multiplications, 10 fused multiply/add),
 * 27 stack variables, 4 constants, and 17 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 7>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T2, T4, T5, T7, T8, T9, Tj, T3, Tf, T6, Th, Ta, Ti, Tk;
        SIMD_FLOAT Tg, Tb, Tl, Tn, Tc, Te, Tm, Td;
        T1            = R0[0];
        T2            = R0[WS(rs, 5)];
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 3)];
        T7            = R0[WS(rs, 4)];
        T8            = R0[WS(rs, 6)];
        T9            = R0[WS(rs, 1)];
        Tj            = T1 + T2;
        T3            = T1 - T2;
        Tf            = T4 + T5;
        T6            = T4 - T5;
        Th            = T8 + T9;
        Ta            = T8 - T9;
        Ti            = T7 - Th;
        Tk            = T7 + Th;
        Tg            = Ta - T7;
        Tb            = T7 + Ta;
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FMADD(KP618033988, Ti, T6));
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, T6, Ti));
        Tl            = Tf + Tk;
        Tn            = Tf - Tk;
        Ci[WS(cs, 3)] = KP951056516 * (SIMD_FMADD(KP618033988, Tf, Tg));
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FNMADD(KP618033988, Tg, Tf)));
        Tc            = T6 + Tb;
        Te            = T6 - Tb;
        Cr[0]         = Tj + Tl;
        Tm            = SIMD_FNMADD(KP250000000, Tl, Tj);
        Cr[WS(cs, 5)] = T3 + Tc;
        Td            = SIMD_FNMADD(KP250000000, Tc, T3);
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Tn, Tm);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Tn, Tm);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, Te, Td);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Te, Td);
    }
}

template <>
struct r2cf_traits<10, 8>
{
    static constexpr long_t flops           = 44;
    static constexpr long_t operations      = 34;
    static constexpr long_t memory_accesses = 18;
    static constexpr long_t stack_vars      = 32;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 8 -name r2cf
 * -standalone */

/*
 * This function contains 30 FP additions, 14 FP multiplications,
 * (or, 20 additions, 4 multiplications, 10 fused multiply/add),
 * 32 stack variables, 4 constants, and 18 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 8>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T2, T9, Ta, Tb, T7, T4, T5, T3, Tn, Tc, Tj, T6, Tl, Tp;
        SIMD_FLOAT Tk, Td, Ti, To, Tm, T8, Th, Ts, Tq, Tg, Te, Tr, Tf;
        T1            = R0[0];
        T2            = R0[WS(rs, 5)];
        T9            = R0[WS(rs, 4)];
        Ta            = R0[WS(rs, 6)];
        Tb            = R0[WS(rs, 1)];
        T7            = R0[WS(rs, 3)];
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 7)];
        T3            = T1 - T2;
        Tn            = T1 + T2;
        Tc            = Ta - Tb;
        Tj            = Ta + Tb;
        T6            = T4 - T5;
        Tl            = T4 + T5;
        Tp            = T9 + Tj;
        Tk            = T9 - Tj;
        Td            = T9 + Tc;
        Ti            = Tc - T9;
        To            = Tl + T7;
        Tm            = Tl - T7;
        T8            = T6 - T7;
        Th            = T6 + T7;
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FMADD(KP618033988, Tk, Tm));
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, Tm, Tk));
        Ts            = To - Tp;
        Tq            = To + Tp;
        Ci[WS(cs, 3)] = KP951056516 * (SIMD_FMADD(KP618033988, Th, Ti));
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FNMADD(KP618033988, Ti, Th)));
        Tg            = T8 - Td;
        Te            = T8 + Td;
        Cr[0]         = Tn + Tq;
        Tr            = SIMD_FNMADD(KP250000000, Tq, Tn);
        Cr[WS(cs, 5)] = T3 + Te;
        Tf            = SIMD_FNMADD(KP250000000, Te, T3);
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Ts, Tr);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Ts, Tr);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, Tg, Tf);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Tg, Tf);
    }
}

template <>
struct r2cf_traits<10, 9>
{
    static constexpr long_t flops           = 46;
    static constexpr long_t operations      = 36;
    static constexpr long_t memory_accesses = 19;
    static constexpr long_t stack_vars      = 35;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 9 -name r2cf
 * -standalone */

/*
 * This function contains 32 FP additions, 14 FP multiplications,
 * (or, 22 additions, 4 multiplications, 10 fused multiply/add),
 * 35 stack variables, 4 constants, and 19 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 9>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T2, Tb, Tc, Td, T4, T5, T7, T8, Tq, T3, Tl, Te, Tn, T6;
        SIMD_FLOAT To, T9, Tm, Ts, Tk, Tf, Tp, Tr, Tj, Ta, Tt, Tv, Tg, Ti, Tu;
        SIMD_FLOAT Th;
        T1            = R0[0];
        T2            = R0[WS(rs, 5)];
        Tb            = R0[WS(rs, 4)];
        Tc            = R0[WS(rs, 6)];
        Td            = R0[WS(rs, 1)];
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 7)];
        T7            = R0[WS(rs, 8)];
        T8            = R0[WS(rs, 3)];
        Tq            = T1 + T2;
        T3            = T1 - T2;
        Tl            = Tc + Td;
        Te            = Tc - Td;
        Tn            = T4 + T5;
        T6            = T4 - T5;
        To            = T7 + T8;
        T9            = T7 - T8;
        Tm            = Tb - Tl;
        Ts            = Tb + Tl;
        Tk            = Te - Tb;
        Tf            = Tb + Te;
        Tp            = Tn - To;
        Tr            = Tn + To;
        Tj            = T6 - T9;
        Ta            = T6 + T9;
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FMADD(KP618033988, Tm, Tp));
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, Tp, Tm));
        Tt            = Tr + Ts;
        Tv            = Tr - Ts;
        Ci[WS(cs, 3)] = KP951056516 * (SIMD_FMADD(KP618033988, Tj, Tk));
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FNMADD(KP618033988, Tk, Tj)));
        Tg            = Ta + Tf;
        Ti            = Ta - Tf;
        Cr[0]         = Tq + Tt;
        Tu            = SIMD_FNMADD(KP250000000, Tt, Tq);
        Cr[WS(cs, 5)] = T3 + Tg;
        Th            = SIMD_FNMADD(KP250000000, Tg, T3);
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Tv, Tu);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Tv, Tu);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, Ti, Th);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Ti, Th);
    }
}

template <>
struct r2cf_traits<10, 10>
{
    static constexpr long_t flops           = 48;
    static constexpr long_t operations      = 38;
    static constexpr long_t memory_accesses = 20;
    static constexpr long_t stack_vars      = 38;
    static constexpr long_t constants       = 4;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 10 -first-k 10 -name r2cf
 * -standalone */

/*
 * This function contains 34 FP additions, 14 FP multiplications,
 * (or, 24 additions, 4 multiplications, 10 fused multiply/add),
 * 38 stack variables, 4 constants, and 20 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 10 && ProvidedElements == 10>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        SIMD_FLOAT T1, T2, Tb, Tc, Te, Tf, T4, T5, T7, T8, T3, Tt, Td, Tn, Tg;
        SIMD_FLOAT To, T6, Tq, T9, Tr, Tv, Tp, Th, Tm, Tu, Ts, Ta, Tl, Ty, Tw;
        SIMD_FLOAT Tk, Ti, Tx, Tj;
        T1            = R0[0];
        T2            = R0[WS(rs, 5)];
        Tb            = R0[WS(rs, 4)];
        Tc            = R0[WS(rs, 9)];
        Te            = R0[WS(rs, 6)];
        Tf            = R0[WS(rs, 1)];
        T4            = R0[WS(rs, 2)];
        T5            = R0[WS(rs, 7)];
        T7            = R0[WS(rs, 8)];
        T8            = R0[WS(rs, 3)];
        T3            = T1 - T2;
        Tt            = T1 + T2;
        Td            = Tb - Tc;
        Tn            = Tb + Tc;
        Tg            = Te - Tf;
        To            = Te + Tf;
        T6            = T4 - T5;
        Tq            = T4 + T5;
        T9            = T7 - T8;
        Tr            = T7 + T8;
        Tv            = Tn + To;
        Tp            = Tn - To;
        Th            = Td + Tg;
        Tm            = Tg - Td;
        Tu            = Tq + Tr;
        Ts            = Tq - Tr;
        Ta            = T6 + T9;
        Tl            = T6 - T9;
        Ci[WS(cs, 4)] = KP951056516 * (SIMD_FMADD(KP618033988, Tp, Ts));
        Ci[WS(cs, 2)] = KP951056516 * (SIMD_FNMADD(KP618033988, Ts, Tp));
        Ty            = Tu - Tv;
        Tw            = Tu + Tv;
        Ci[WS(cs, 3)] = KP951056516 * (SIMD_FMADD(KP618033988, Tl, Tm));
        Ci[WS(cs, 1)] = -(KP951056516 * (SIMD_FNMADD(KP618033988, Tm, Tl)));
        Tk            = Ta - Th;
        Ti            = Ta + Th;
        Cr[0]         = Tt + Tw;
        Tx            = SIMD_FNMADD(KP250000000, Tw, Tt);
        Cr[WS(cs, 5)] = T3 + Ti;
        Tj            = SIMD_FNMADD(KP250000000, Ti, T3);
        Cr[WS(cs, 4)] = SIMD_FMADD(KP559016994, Ty, Tx);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP559016994, Ty, Tx);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP559016994, Tk, Tj);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP559016994, Tk, Tj);
    }
}
