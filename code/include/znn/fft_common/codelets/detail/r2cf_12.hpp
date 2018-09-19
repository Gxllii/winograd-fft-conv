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
struct r2cf_traits<12, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 8;
    static constexpr long_t stack_vars      = 1;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 1 -name r2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 1 stack variables, 0 constants, and 8 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 1>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    {
        SIMD_FLOAT T1;
        T1            = R0[0];
        Cr[WS(cs, 4)] = T1;
        Cr[WS(cs, 2)] = T1;
        Cr[WS(cs, 3)] = T1;
        Cr[WS(cs, 5)] = T1;
        Cr[WS(cs, 6)] = T1;
        Cr[WS(cs, 1)] = T1;
        Cr[0]         = T1;
    }
}

template <>
struct r2cf_traits<12, 2>
{
    static constexpr long_t flops           = 14;
    static constexpr long_t operations      = 10;
    static constexpr long_t memory_accesses = 14;
    static constexpr long_t stack_vars      = 4;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 2 -name r2cf
 * -standalone */

/*
 * This function contains 6 FP additions, 8 FP multiplications,
 * (or, 2 additions, 4 multiplications, 4 fused multiply/add),
 * 4 stack variables, 2 constants, and 14 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 2>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    {
        SIMD_FLOAT T1, T2;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        Cr[WS(cs, 3)] = T1;
        Cr[WS(cs, 5)] = SIMD_FNMADD(KP866025403, T2, T1);
        Cr[WS(cs, 2)] = SIMD_FMADD(KP500000000, T2, T1);
        Cr[WS(cs, 6)] = T1 - T2;
        Cr[0]         = T1 + T2;
        Cr[WS(cs, 4)] = SIMD_FNMADD(KP500000000, T2, T1);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP866025403, T2, T1);
        Ci[WS(cs, 5)] = -(KP500000000 * T2);
        Ci[WS(cs, 4)] = -(KP866025403 * T2);
        Ci[WS(cs, 1)] = -(KP500000000 * T2);
        Ci[WS(cs, 2)] = -(KP866025403 * T2);
        Ci[WS(cs, 3)] = -T2;
    }
}

template <>
struct r2cf_traits<12, 3>
{
    static constexpr long_t flops           = 26;
    static constexpr long_t operations      = 18;
    static constexpr long_t memory_accesses = 15;
    static constexpr long_t stack_vars      = 9;
    static constexpr long_t constants       = 3;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 3 -name r2cf
 * -standalone */

/*
 * This function contains 14 FP additions, 12 FP multiplications,
 * (or, 6 additions, 4 multiplications, 8 fused multiply/add),
 * 9 stack variables, 3 constants, and 15 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 3>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    DK(KP577350269, +0.577350269189625764509148780501957455647601751);
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    {
        SIMD_FLOAT T3, T1, T2, T5, T6, T4;
        T3            = R0[WS(rs, 1)];
        T1            = R0[0];
        T2            = R0[WS(rs, 2)];
        Ci[WS(cs, 3)] = -T3;
        Ci[WS(cs, 2)] = -(KP866025403 * (T2 + T3));
        Ci[WS(cs, 5)] = KP866025403 * (SIMD_FNMADD(KP577350269, T3, T2));
        Ci[WS(cs, 1)] = -(KP866025403 * (SIMD_FMADD(KP577350269, T3, T2)));
        Ci[WS(cs, 4)] = KP866025403 * (T2 - T3);
        Cr[WS(cs, 3)] = T1 - T2;
        T5            = SIMD_FNMADD(KP500000000, T2, T1);
        T6            = SIMD_FMADD(KP500000000, T2, T1);
        T4            = T1 + T2;
        Cr[WS(cs, 2)] = SIMD_FMADD(KP500000000, T3, T5);
        Cr[WS(cs, 4)] = SIMD_FNMADD(KP500000000, T3, T5);
        Cr[WS(cs, 5)] = SIMD_FNMADD(KP866025403, T3, T6);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP866025403, T3, T6);
        Cr[WS(cs, 6)] = T4 - T3;
        Cr[0]         = T4 + T3;
    }
}

template <>
struct r2cf_traits<12, 4>
{
    static constexpr long_t flops           = 28;
    static constexpr long_t operations      = 20;
    static constexpr long_t memory_accesses = 16;
    static constexpr long_t stack_vars      = 12;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 4 -name r2cf
 * -standalone */

/*
 * This function contains 18 FP additions, 10 FP multiplications,
 * (or, 10 additions, 2 multiplications, 8 fused multiply/add),
 * 12 stack variables, 2 constants, and 16 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 4>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, T3, T4, T8, T9, T6, T7, Ta, T5;
        T1            = R0[0];
        T2            = R0[WS(rs, 2)];
        T3            = R0[WS(rs, 3)];
        T4            = R0[WS(rs, 1)];
        Cr[WS(cs, 3)] = T1 - T2;
        T8            = SIMD_FMADD(KP500000000, T2, T1);
        T9            = T1 + T2;
        T6            = SIMD_FNMADD(KP500000000, T2, T1);
        Ci[WS(cs, 2)] = -(KP866025403 * (T2 + T4));
        Ci[WS(cs, 4)] = KP866025403 * (T2 - T4);
        Ci[WS(cs, 3)] = T3 - T4;
        T7            = SIMD_FNMADD(KP500000000, T4, T3);
        Ta            = T3 + T4;
        T5            = SIMD_FMADD(KP500000000, T4, T3);
        Cr[WS(cs, 5)] = SIMD_FNMADD(KP866025403, T4, T8);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP866025403, T4, T8);
        Cr[WS(cs, 2)] = T6 - T7;
        Cr[WS(cs, 4)] = T6 + T7;
        Cr[WS(cs, 6)] = T9 - Ta;
        Cr[0]         = T9 + Ta;
        Ci[WS(cs, 5)] = SIMD_FMSUB(KP866025403, T2, T5);
        Ci[WS(cs, 1)] = -(SIMD_FMADD(KP866025403, T2, T5));
    }
}

template <>
struct r2cf_traits<12, 5>
{
    static constexpr long_t flops           = 33;
    static constexpr long_t operations      = 24;
    static constexpr long_t memory_accesses = 17;
    static constexpr long_t stack_vars      = 17;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 5 -name r2cf
 * -standalone */

/*
 * This function contains 22 FP additions, 11 FP multiplications,
 * (or, 13 additions, 2 multiplications, 9 fused multiply/add),
 * 17 stack variables, 2 constants, and 17 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 5>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T6, T7, T1, T4, T2, T8, Tf, Tc, T9, T5, Ta, T3, Tb, Td, Te;
        T6            = R0[WS(rs, 1)];
        T7            = R0[WS(rs, 3)];
        T1            = R0[0];
        T4            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 4)];
        Ci[WS(cs, 3)] = T7 - T6;
        T8            = SIMD_FMADD(KP500000000, T6, T7);
        Tf            = T7 + T6;
        Tc            = SIMD_FNMADD(KP500000000, T6, T7);
        T9            = T2 + T4;
        T5            = T2 - T4;
        Ta            = SIMD_FNMADD(KP500000000, T2, T1);
        T3            = T1 + T2;
        Ci[WS(cs, 5)] = SIMD_FMSUB(KP866025403, T9, T8);
        Ci[WS(cs, 1)] = -(SIMD_FMADD(KP866025403, T9, T8));
        Ci[WS(cs, 4)] = -(KP866025403 * (T5 + T6));
        Ci[WS(cs, 2)] = KP866025403 * (T5 - T6);
        Tb            = SIMD_FNMADD(KP500000000, T4, Ta);
        Td            = SIMD_FMADD(KP500000000, T4, Ta);
        Te            = T3 + T4;
        Cr[WS(cs, 3)] = T3 - T4;
        Cr[WS(cs, 4)] = Tb + Tc;
        Cr[WS(cs, 2)] = Tb - Tc;
        Cr[WS(cs, 1)] = SIMD_FMADD(KP866025403, T6, Td);
        Cr[WS(cs, 5)] = SIMD_FNMADD(KP866025403, T6, Td);
        Cr[0]         = Te + Tf;
        Cr[WS(cs, 6)] = Te - Tf;
    }
}

template <>
struct r2cf_traits<12, 6>
{
    static constexpr long_t flops           = 35;
    static constexpr long_t operations      = 26;
    static constexpr long_t memory_accesses = 18;
    static constexpr long_t stack_vars      = 20;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 6 -name r2cf
 * -standalone */

/*
 * This function contains 24 FP additions, 11 FP multiplications,
 * (or, 15 additions, 2 multiplications, 9 fused multiply/add),
 * 20 stack variables, 2 constants, and 18 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 6>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T5, T6, T7, T1, T4, T2, T8, Th, T3, Tb, Ta, Ti, T9, Td, Tf;
        SIMD_FLOAT Tg, Tc, Te;
        T5            = R0[WS(rs, 3)];
        T6            = R0[WS(rs, 1)];
        T7            = R0[WS(rs, 5)];
        T1            = R0[0];
        T4            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 4)];
        T8            = T6 + T7;
        Th            = T7 - T6;
        T3            = T1 + T2;
        Tb            = SIMD_FNMADD(KP500000000, T2, T1);
        Ta            = T2 + T4;
        Ti            = T2 - T4;
        T9            = SIMD_FMADD(KP500000000, T8, T5);
        Td            = SIMD_FNMADD(KP500000000, T8, T5);
        Tf            = T5 + T8;
        Ci[WS(cs, 3)] = T5 - T8;
        Tg            = SIMD_FMADD(KP500000000, T4, Tb);
        Tc            = SIMD_FNMADD(KP500000000, T4, Tb);
        Ci[WS(cs, 4)] = KP866025403 * (Th - Ti);
        Ci[WS(cs, 2)] = KP866025403 * (Ti + Th);
        Te            = T3 + T4;
        Cr[WS(cs, 3)] = T3 - T4;
        Ci[WS(cs, 5)] = SIMD_FMSUB(KP866025403, Ta, T9);
        Ci[WS(cs, 1)] = -(SIMD_FMADD(KP866025403, Ta, T9));
        Cr[WS(cs, 4)] = Tc + Td;
        Cr[WS(cs, 2)] = Tc - Td;
        Cr[WS(cs, 5)] = SIMD_FMADD(KP866025403, Th, Tg);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP866025403, Th, Tg);
        Cr[0]         = Te + Tf;
        Cr[WS(cs, 6)] = Te - Tf;
    }
}

template <>
struct r2cf_traits<12, 7>
{
    static constexpr long_t flops           = 36;
    static constexpr long_t operations      = 28;
    static constexpr long_t memory_accesses = 19;
    static constexpr long_t stack_vars      = 23;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 7 -name r2cf
 * -standalone */

/*
 * This function contains 26 FP additions, 10 FP multiplications,
 * (or, 18 additions, 2 multiplications, 8 fused multiply/add),
 * 23 stack variables, 2 constants, and 19 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 7>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T7, T8, T9, T1, T5, T2, T4, Tk, Ta, Tc, Tl, T3, Td, Te, T6;
        SIMD_FLOAT Ti, Tg, Tb, Tf, Tj, Th;
        T7            = R0[WS(rs, 3)];
        T8            = R0[WS(rs, 1)];
        T9            = R0[WS(rs, 5)];
        T1            = R0[0];
        T5            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 4)];
        T4            = R0[WS(rs, 6)];
        Tk            = T9 - T8;
        Ta            = T8 + T9;
        Tc            = T2 + T5;
        Tl            = T2 - T5;
        T3            = T1 + T2;
        Td            = SIMD_FNMADD(KP500000000, T2, T1);
        Te            = SIMD_FNMADD(KP500000000, T5, T4);
        T6            = T4 + T5;
        Ti            = T7 + Ta;
        Tg            = SIMD_FNMADD(KP500000000, Ta, T7);
        Tb            = SIMD_FMADD(KP500000000, Ta, T7);
        Ci[WS(cs, 3)] = T7 - Ta;
        Ci[WS(cs, 4)] = KP866025403 * (Tk - Tl);
        Ci[WS(cs, 2)] = KP866025403 * (Tl + Tk);
        Tf            = Td + Te;
        Tj            = Td - Te;
        Th            = T3 + T6;
        Cr[WS(cs, 3)] = T3 - T6;
        Ci[WS(cs, 5)] = SIMD_FMSUB(KP866025403, Tc, Tb);
        Ci[WS(cs, 1)] = -(SIMD_FMADD(KP866025403, Tc, Tb));
        Cr[WS(cs, 4)] = Tf + Tg;
        Cr[WS(cs, 2)] = Tf - Tg;
        Cr[WS(cs, 5)] = SIMD_FMADD(KP866025403, Tk, Tj);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP866025403, Tk, Tj);
        Cr[0]         = Th + Ti;
        Cr[WS(cs, 6)] = Th - Ti;
    }
}

template <>
struct r2cf_traits<12, 8>
{
    static constexpr long_t flops           = 41;
    static constexpr long_t operations      = 32;
    static constexpr long_t memory_accesses = 20;
    static constexpr long_t stack_vars      = 28;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 8 -name r2cf
 * -standalone */

/*
 * This function contains 30 FP additions, 11 FP multiplications,
 * (or, 21 additions, 2 multiplications, 9 fused multiply/add),
 * 28 stack variables, 2 constants, and 20 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 8>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T7, T8, Ta, Tb, T1, T5, T2, T4, T9, Td, Tc, Tn, Tp, Tf, Tg;
        SIMD_FLOAT T3, T6, Th, Tq, To, Tl, Tj, Te, Ti, Tm, Tk;
        T7            = R0[WS(rs, 3)];
        T8            = R0[WS(rs, 7)];
        Ta            = R0[WS(rs, 1)];
        Tb            = R0[WS(rs, 5)];
        T1            = R0[0];
        T5            = R0[WS(rs, 2)];
        T2            = R0[WS(rs, 4)];
        T4            = R0[WS(rs, 6)];
        T9            = T7 + T8;
        Td            = SIMD_FNMADD(KP500000000, T8, T7);
        Tc            = Ta + Tb;
        Tn            = Tb - Ta;
        Tp            = T2 - T5;
        Tf            = T2 + T5;
        Tg            = SIMD_FNMADD(KP500000000, T2, T1);
        T3            = T1 + T2;
        T6            = T4 + T5;
        Th            = SIMD_FNMADD(KP500000000, T5, T4);
        Tq            = Tn - T8;
        To            = T8 + Tn;
        Tl            = T9 + Tc;
        Tj            = SIMD_FNMADD(KP500000000, Tc, Td);
        Te            = SIMD_FMADD(KP500000000, Tc, Td);
        Ci[WS(cs, 3)] = T9 - Tc;
        Ti            = Tg + Th;
        Tm            = Tg - Th;
        Tk            = T3 + T6;
        Cr[WS(cs, 3)] = T3 - T6;
        Ci[WS(cs, 4)] = KP866025403 * (Tq - Tp);
        Ci[WS(cs, 2)] = KP866025403 * (Tp + Tq);
        Ci[WS(cs, 5)] = SIMD_FMSUB(KP866025403, Tf, Te);
        Ci[WS(cs, 1)] = -(SIMD_FMADD(KP866025403, Tf, Te));
        Cr[WS(cs, 5)] = SIMD_FMADD(KP866025403, To, Tm);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP866025403, To, Tm);
        Cr[WS(cs, 4)] = Ti + Tj;
        Cr[WS(cs, 2)] = Ti - Tj;
        Cr[0]         = Tk + Tl;
        Cr[WS(cs, 6)] = Tk - Tl;
    }
}

template <>
struct r2cf_traits<12, 9>
{
    static constexpr long_t flops           = 43;
    static constexpr long_t operations      = 34;
    static constexpr long_t memory_accesses = 21;
    static constexpr long_t stack_vars      = 31;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 9 -name r2cf
 * -standalone */

/*
 * This function contains 32 FP additions, 11 FP multiplications,
 * (or, 23 additions, 2 multiplications, 9 fused multiply/add),
 * 31 stack variables, 2 constants, and 21 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 9>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T6, T7, Tc, Td, T9, Ta, T1, T2, T3, Tk, T8, Tq, Te, Tf, Tb;
        SIMD_FLOAT Th, T4, Ts, Tr, Tm, Tg, To, Tt, Ti, Tj, T5, Tl, Tp, Tn;
        T6            = R0[WS(rs, 6)];
        T7            = R0[WS(rs, 2)];
        Tc            = R0[WS(rs, 1)];
        Td            = R0[WS(rs, 5)];
        T9            = R0[WS(rs, 3)];
        Ta            = R0[WS(rs, 7)];
        T1            = R0[0];
        T2            = R0[WS(rs, 4)];
        T3            = R0[WS(rs, 8)];
        Tk            = SIMD_FNMADD(KP500000000, T7, T6);
        T8            = T6 + T7;
        Tq            = Td - Tc;
        Te            = Tc + Td;
        Tf            = SIMD_FNMADD(KP500000000, Ta, T9);
        Tb            = T9 + Ta;
        Th            = T3 - T2;
        T4            = T2 + T3;
        Ts            = Tq - Ta;
        Tr            = Ta + Tq;
        Tm            = SIMD_FNMADD(KP500000000, Te, Tf);
        Tg            = SIMD_FMADD(KP500000000, Te, Tf);
        To            = Tb + Te;
        Ci[WS(cs, 3)] = Tb - Te;
        Tt            = Th + T7;
        Ti            = Th - T7;
        Tj            = SIMD_FNMADD(KP500000000, T4, T1);
        T5            = T1 + T4;
        Ci[WS(cs, 4)] = KP866025403 * (Tt + Ts);
        Ci[WS(cs, 2)] = KP866025403 * (Ts - Tt);
        Ci[WS(cs, 5)] = -(SIMD_FMADD(KP866025403, Ti, Tg));
        Ci[WS(cs, 1)] = SIMD_FMSUB(KP866025403, Ti, Tg);
        Tl            = Tj + Tk;
        Tp            = Tj - Tk;
        Tn            = T5 + T8;
        Cr[WS(cs, 3)] = T5 - T8;
        Cr[WS(cs, 4)] = Tl + Tm;
        Cr[WS(cs, 2)] = Tl - Tm;
        Cr[WS(cs, 5)] = SIMD_FMADD(KP866025403, Tr, Tp);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP866025403, Tr, Tp);
        Cr[0]         = Tn + To;
        Cr[WS(cs, 6)] = Tn - To;
    }
}

template <>
struct r2cf_traits<12, 10>
{
    static constexpr long_t flops           = 44;
    static constexpr long_t operations      = 36;
    static constexpr long_t memory_accesses = 22;
    static constexpr long_t stack_vars      = 34;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 10 -name r2cf
 * -standalone */

/*
 * This function contains 34 FP additions, 10 FP multiplications,
 * (or, 26 additions, 2 multiplications, 8 fused multiply/add),
 * 34 stack variables, 2 constants, and 22 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 10>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T6, T7, Te, Tf, T1, T2, T3, T9, Ta, Tb, T8, Tl, Tg, To, T4;
        SIMD_FLOAT Ti, Tc, Tt, Tj, Tw, T5, Tk, Tu, Tv, Td, Tp, Ts, Tm, Tq, Tr;
        SIMD_FLOAT Th, Tn;
        T6            = R0[WS(rs, 6)];
        T7            = R0[WS(rs, 2)];
        Te            = R0[WS(rs, 3)];
        Tf            = R0[WS(rs, 7)];
        T1            = R0[0];
        T2            = R0[WS(rs, 4)];
        T3            = R0[WS(rs, 8)];
        T9            = R0[WS(rs, 9)];
        Ta            = R0[WS(rs, 1)];
        Tb            = R0[WS(rs, 5)];
        T8            = T6 + T7;
        Tl            = SIMD_FNMADD(KP500000000, T7, T6);
        Tg            = SIMD_FNMADD(KP500000000, Tf, Te);
        To            = Te + Tf;
        T4            = T2 + T3;
        Ti            = T3 - T2;
        Tc            = Ta + Tb;
        Tt            = Tb - Ta;
        Tj            = Ti - T7;
        Tw            = Ti + T7;
        T5            = T1 + T4;
        Tk            = SIMD_FNMADD(KP500000000, T4, T1);
        Tu            = Tf + Tt;
        Tv            = Tt - Tf;
        Td            = SIMD_FNMADD(KP500000000, Tc, T9);
        Tp            = T9 + Tc;
        Ts            = Tk - Tl;
        Tm            = Tk + Tl;
        Tq            = T5 + T8;
        Cr[WS(cs, 3)] = T5 - T8;
        Ci[WS(cs, 4)] = KP866025403 * (Tw + Tv);
        Ci[WS(cs, 2)] = KP866025403 * (Tv - Tw);
        Tr            = To + Tp;
        Ci[WS(cs, 3)] = To - Tp;
        Th            = Td - Tg;
        Tn            = Tg + Td;
        Cr[WS(cs, 5)] = SIMD_FMADD(KP866025403, Tu, Ts);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP866025403, Tu, Ts);
        Cr[0]         = Tq + Tr;
        Cr[WS(cs, 6)] = Tq - Tr;
        Cr[WS(cs, 4)] = Tm + Tn;
        Cr[WS(cs, 2)] = Tm - Tn;
        Ci[WS(cs, 5)] = SIMD_FNMADD(KP866025403, Tj, Th);
        Ci[WS(cs, 1)] = SIMD_FMADD(KP866025403, Tj, Th);
    }
}

template <>
struct r2cf_traits<12, 11>
{
    static constexpr long_t flops           = 46;
    static constexpr long_t operations      = 38;
    static constexpr long_t memory_accesses = 23;
    static constexpr long_t stack_vars      = 37;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 11 -name r2cf
 * -standalone */

/*
 * This function contains 36 FP additions, 10 FP multiplications,
 * (or, 28 additions, 2 multiplications, 8 fused multiply/add),
 * 37 stack variables, 2 constants, and 23 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 11>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT Tg, Th, Tb, Tc, Td, T1, T2, T3, T6, T7, T8, Tr, Ti, Tw, Te;
        SIMD_FLOAT Tk, T4, Tl, T9, Ty, Tx, Ts, Tf, Tn, T5, Tz, Tm, To, Ta, Tu;
        SIMD_FLOAT Tq, Tj, Tp, Tv, Tt;
        Tg            = R0[WS(rs, 3)];
        Th            = R0[WS(rs, 7)];
        Tb            = R0[WS(rs, 9)];
        Tc            = R0[WS(rs, 1)];
        Td            = R0[WS(rs, 5)];
        T1            = R0[0];
        T2            = R0[WS(rs, 4)];
        T3            = R0[WS(rs, 8)];
        T6            = R0[WS(rs, 6)];
        T7            = R0[WS(rs, 10)];
        T8            = R0[WS(rs, 2)];
        Tr            = Tg + Th;
        Ti            = SIMD_FNMADD(KP500000000, Th, Tg);
        Tw            = Td - Tc;
        Te            = Tc + Td;
        Tk            = T3 - T2;
        T4            = T2 + T3;
        Tl            = T8 - T7;
        T9            = T7 + T8;
        Ty            = Tw - Th;
        Tx            = Th + Tw;
        Ts            = Tb + Te;
        Tf            = SIMD_FNMADD(KP500000000, Te, Tb);
        Tn            = SIMD_FNMADD(KP500000000, T4, T1);
        T5            = T1 + T4;
        Tz            = Tk + Tl;
        Tm            = Tk - Tl;
        To            = SIMD_FNMADD(KP500000000, T9, T6);
        Ta            = T6 + T9;
        Tu            = Tr + Ts;
        Ci[WS(cs, 3)] = Tr - Ts;
        Tq            = Ti + Tf;
        Tj            = Tf - Ti;
        Ci[WS(cs, 4)] = KP866025403 * (Tz + Ty);
        Ci[WS(cs, 2)] = KP866025403 * (Ty - Tz);
        Tp            = Tn + To;
        Tv            = Tn - To;
        Tt            = T5 + Ta;
        Cr[WS(cs, 3)] = T5 - Ta;
        Ci[WS(cs, 5)] = SIMD_FNMADD(KP866025403, Tm, Tj);
        Ci[WS(cs, 1)] = SIMD_FMADD(KP866025403, Tm, Tj);
        Cr[WS(cs, 4)] = Tp + Tq;
        Cr[WS(cs, 2)] = Tp - Tq;
        Cr[WS(cs, 5)] = SIMD_FMADD(KP866025403, Tx, Tv);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP866025403, Tx, Tv);
        Cr[0]         = Tt + Tu;
        Cr[WS(cs, 6)] = Tt - Tu;
    }
}

template <>
struct r2cf_traits<12, 12>
{
    static constexpr long_t flops           = 48;
    static constexpr long_t operations      = 40;
    static constexpr long_t memory_accesses = 24;
    static constexpr long_t stack_vars      = 40;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 12 -first-k 12 -name r2cf
 * -standalone */

/*
 * This function contains 38 FP additions, 10 FP multiplications,
 * (or, 30 additions, 2 multiplications, 8 fused multiply/add),
 * 40 stack variables, 2 constants, and 24 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 12 && ProvidedElements == 12>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, T3, Tg, Th, Ti, T6, T7, T8, Tb, Tc, Td, T4, Tm, Tj;
        SIMD_FLOAT Ty, T9, Tn, Te, Tz, T5, Tp, Tk, Tt, To, TC, Ta, Tq, TA, TB;
        SIMD_FLOAT Tf, Tu, Tx, Tr, Tv, Tw, Tl, Ts;
        T1            = R0[0];
        T2            = R0[WS(rs, 4)];
        T3            = R0[WS(rs, 8)];
        Tg            = R0[WS(rs, 3)];
        Th            = R0[WS(rs, 7)];
        Ti            = R0[WS(rs, 11)];
        T6            = R0[WS(rs, 6)];
        T7            = R0[WS(rs, 10)];
        T8            = R0[WS(rs, 2)];
        Tb            = R0[WS(rs, 9)];
        Tc            = R0[WS(rs, 1)];
        Td            = R0[WS(rs, 5)];
        T4            = T2 + T3;
        Tm            = T3 - T2;
        Tj            = Th + Ti;
        Ty            = Ti - Th;
        T9            = T7 + T8;
        Tn            = T8 - T7;
        Te            = Tc + Td;
        Tz            = Td - Tc;
        T5            = T1 + T4;
        Tp            = SIMD_FNMADD(KP500000000, T4, T1);
        Tk            = SIMD_FNMADD(KP500000000, Tj, Tg);
        Tt            = Tg + Tj;
        To            = Tm - Tn;
        TC            = Tm + Tn;
        Ta            = T6 + T9;
        Tq            = SIMD_FNMADD(KP500000000, T9, T6);
        TA            = Ty - Tz;
        TB            = Ty + Tz;
        Tf            = SIMD_FNMADD(KP500000000, Te, Tb);
        Tu            = Tb + Te;
        Tx            = Tp - Tq;
        Tr            = Tp + Tq;
        Tv            = T5 + Ta;
        Cr[WS(cs, 3)] = T5 - Ta;
        Ci[WS(cs, 4)] = KP866025403 * (TC + TB);
        Ci[WS(cs, 2)] = KP866025403 * (TB - TC);
        Tw            = Tt + Tu;
        Ci[WS(cs, 3)] = Tt - Tu;
        Tl            = Tf - Tk;
        Ts            = Tk + Tf;
        Cr[WS(cs, 1)] = SIMD_FMADD(KP866025403, TA, Tx);
        Cr[WS(cs, 5)] = SIMD_FNMADD(KP866025403, TA, Tx);
        Cr[0]         = Tv + Tw;
        Cr[WS(cs, 6)] = Tv - Tw;
        Cr[WS(cs, 4)] = Tr + Ts;
        Cr[WS(cs, 2)] = Tr - Ts;
        Ci[WS(cs, 5)] = SIMD_FNMADD(KP866025403, To, Tl);
        Ci[WS(cs, 1)] = SIMD_FMADD(KP866025403, To, Tl);
    }
}
