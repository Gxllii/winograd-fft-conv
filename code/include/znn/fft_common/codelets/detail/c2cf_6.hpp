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
struct c2cf_traits<6, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 14;
    static constexpr long_t stack_vars      = 2;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 1 -name c2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 2 stack variables, 0 constants, and 14 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 1>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    {
        SIMD_FLOAT T1, T2;
        T1            = ri[0];
        T2            = ii[0];
        ro[WS(os, 2)] = T1;
        ro[WS(os, 3)] = T1;
        ro[WS(os, 4)] = T1;
        ro[WS(os, 5)] = T1;
        ro[WS(os, 1)] = T1;
        ro[0]         = T1;
        io[WS(os, 2)] = T2;
        io[WS(os, 3)] = T2;
        io[WS(os, 4)] = T2;
        io[WS(os, 5)] = T2;
        io[WS(os, 1)] = T2;
        io[0]         = T2;
    }
}

template <>
struct c2cf_traits<6, 2>
{
    static constexpr long_t flops           = 28;
    static constexpr long_t operations      = 16;
    static constexpr long_t memory_accesses = 16;
    static constexpr long_t stack_vars      = 10;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 2 -name c2cf
 * -standalone */

/*
 * This function contains 16 FP additions, 12 FP multiplications,
 * (or, 4 additions, 0 multiplications, 12 fused multiply/add),
 * 10 stack variables, 2 constants, and 16 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 2>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, T3, T4, T5, T6, T7, T8;
        T1            = ri[0];
        T2            = ri[WS(is, 1)];
        T3            = ii[0];
        T4            = ii[WS(is, 1)];
        ro[0]         = T1 + T2;
        ro[WS(os, 3)] = T1 - T2;
        T5            = SIMD_FNMADD(KP500000000, T2, T1);
        T6            = SIMD_FMADD(KP500000000, T2, T1);
        io[0]         = T3 + T4;
        io[WS(os, 3)] = T3 - T4;
        T7            = SIMD_FNMADD(KP500000000, T4, T3);
        T8            = SIMD_FMADD(KP500000000, T4, T3);
        ro[WS(os, 1)] = SIMD_FMADD(KP866025403, T4, T6);
        ro[WS(os, 5)] = SIMD_FNMADD(KP866025403, T4, T6);
        ro[WS(os, 2)] = SIMD_FMADD(KP866025403, T4, T5);
        ro[WS(os, 4)] = SIMD_FNMADD(KP866025403, T4, T5);
        io[WS(os, 1)] = SIMD_FNMADD(KP866025403, T2, T8);
        io[WS(os, 5)] = SIMD_FMADD(KP866025403, T2, T8);
        io[WS(os, 2)] = SIMD_FNMADD(KP866025403, T2, T7);
        io[WS(os, 4)] = SIMD_FMADD(KP866025403, T2, T7);
    }
}

template <>
struct c2cf_traits<6, 3>
{
    static constexpr long_t flops           = 32;
    static constexpr long_t operations      = 20;
    static constexpr long_t memory_accesses = 18;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 3 -name c2cf
 * -standalone */

/*
 * This function contains 20 FP additions, 12 FP multiplications,
 * (or, 8 additions, 0 multiplications, 12 fused multiply/add),
 * 16 stack variables, 2 constants, and 18 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 3>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T6, T2, T3, T7, T8, T4, T5, T9, Tc, Tb, Te, Td, Ta;
        T1            = ri[0];
        T6            = ii[0];
        T2            = ri[WS(is, 2)];
        T3            = ri[WS(is, 1)];
        T7            = ii[WS(is, 2)];
        T8            = ii[WS(is, 1)];
        T4            = T2 - T3;
        T5            = T2 + T3;
        T9            = T7 - T8;
        Tc            = T7 + T8;
        Tb            = SIMD_FNMADD(KP500000000, T5, T1);
        ro[0]         = T1 + T5;
        Te            = SIMD_FNMADD(KP500000000, T4, T1);
        ro[WS(os, 3)] = T1 + T4;
        Td            = SIMD_FNMADD(KP500000000, Tc, T6);
        io[0]         = T6 + Tc;
        Ta            = SIMD_FNMADD(KP500000000, T9, T6);
        io[WS(os, 3)] = T6 + T9;
        ro[WS(os, 2)] = SIMD_FNMADD(KP866025403, T9, Tb);
        ro[WS(os, 4)] = SIMD_FMADD(KP866025403, T9, Tb);
        ro[WS(os, 1)] = SIMD_FMADD(KP866025403, Tc, Te);
        ro[WS(os, 5)] = SIMD_FNMADD(KP866025403, Tc, Te);
        io[WS(os, 4)] = SIMD_FNMADD(KP866025403, T4, Td);
        io[WS(os, 2)] = SIMD_FMADD(KP866025403, T4, Td);
        io[WS(os, 5)] = SIMD_FMADD(KP866025403, T5, Ta);
        io[WS(os, 1)] = SIMD_FNMADD(KP866025403, T5, Ta);
    }
}

template <>
struct c2cf_traits<6, 4>
{
    static constexpr long_t flops           = 36;
    static constexpr long_t operations      = 24;
    static constexpr long_t memory_accesses = 20;
    static constexpr long_t stack_vars      = 22;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 4 -name c2cf
 * -standalone */

/*
 * This function contains 24 FP additions, 12 FP multiplications,
 * (or, 12 additions, 0 multiplications, 12 fused multiply/add),
 * 22 stack variables, 2 constants, and 20 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 4>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, T9, Ta, T4, T5, Tc, Td, T3, T7, Tb, Th, T6, T8, Te;
        SIMD_FLOAT Ti, Tg, Tk, Tj, Tf;
        T1            = ri[0];
        T2            = ri[WS(is, 3)];
        T9            = ii[0];
        Ta            = ii[WS(is, 3)];
        T4            = ri[WS(is, 2)];
        T5            = ri[WS(is, 1)];
        Tc            = ii[WS(is, 2)];
        Td            = ii[WS(is, 1)];
        T3            = T1 - T2;
        T7            = T1 + T2;
        Tb            = T9 - Ta;
        Th            = T9 + Ta;
        T6            = T4 - T5;
        T8            = T4 + T5;
        Te            = Tc - Td;
        Ti            = Tc + Td;
        Tg            = SIMD_FNMADD(KP500000000, T8, T7);
        ro[0]         = T7 + T8;
        Tk            = SIMD_FNMADD(KP500000000, T6, T3);
        ro[WS(os, 3)] = T3 + T6;
        Tj            = SIMD_FNMADD(KP500000000, Ti, Th);
        io[0]         = Th + Ti;
        Tf            = SIMD_FNMADD(KP500000000, Te, Tb);
        io[WS(os, 3)] = Tb + Te;
        ro[WS(os, 2)] = SIMD_FNMADD(KP866025403, Te, Tg);
        ro[WS(os, 4)] = SIMD_FMADD(KP866025403, Te, Tg);
        ro[WS(os, 1)] = SIMD_FMADD(KP866025403, Ti, Tk);
        ro[WS(os, 5)] = SIMD_FNMADD(KP866025403, Ti, Tk);
        io[WS(os, 4)] = SIMD_FNMADD(KP866025403, T6, Tj);
        io[WS(os, 2)] = SIMD_FMADD(KP866025403, T6, Tj);
        io[WS(os, 5)] = SIMD_FMADD(KP866025403, T8, Tf);
        io[WS(os, 1)] = SIMD_FNMADD(KP866025403, T8, Tf);
    }
}

template <>
struct c2cf_traits<6, 5>
{
    static constexpr long_t flops           = 44;
    static constexpr long_t operations      = 32;
    static constexpr long_t memory_accesses = 22;
    static constexpr long_t stack_vars      = 32;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 5 -name c2cf
 * -standalone */

/*
 * This function contains 32 FP additions, 12 FP multiplications,
 * (or, 20 additions, 0 multiplications, 12 fused multiply/add),
 * 32 stack variables, 2 constants, and 22 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 5>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, Ti, Tj, T4, T5, T6, Td, Te, Tf, T3, T9, Tk, Tr, T7;
        SIMD_FLOAT Ta, Tg, Tp, Tb, Tu, T8, Tn, Ts, Tq, Tl, Th, To, Tc, Tt, Tm;
        T1            = ri[0];
        T2            = ri[WS(is, 3)];
        Ti            = ii[0];
        Tj            = ii[WS(is, 3)];
        T4            = ri[WS(is, 2)];
        T5            = ri[WS(is, 4)];
        T6            = ri[WS(is, 1)];
        Td            = ii[WS(is, 2)];
        Te            = ii[WS(is, 4)];
        Tf            = ii[WS(is, 1)];
        T3            = T1 - T2;
        T9            = T1 + T2;
        Tk            = Ti - Tj;
        Tr            = Ti + Tj;
        T7            = T5 - T6;
        Ta            = T5 + T6;
        Tg            = Te - Tf;
        Tp            = Te + Tf;
        Tb            = T4 + Ta;
        Tu            = Ta - T4;
        T8            = T4 + T7;
        Tn            = T7 - T4;
        Ts            = Td + Tp;
        Tq            = Td - Tp;
        Tl            = Td + Tg;
        Th            = Td - Tg;
        To            = SIMD_FNMADD(KP500000000, Tb, T9);
        ro[0]         = T9 + Tb;
        Tc            = SIMD_FNMADD(KP500000000, T8, T3);
        ro[WS(os, 3)] = T3 + T8;
        Tt            = SIMD_FNMADD(KP500000000, Ts, Tr);
        io[0]         = Tr + Ts;
        Tm            = SIMD_FNMADD(KP500000000, Tl, Tk);
        io[WS(os, 3)] = Tk + Tl;
        ro[WS(os, 4)] = SIMD_FMADD(KP866025403, Tq, To);
        ro[WS(os, 2)] = SIMD_FNMADD(KP866025403, Tq, To);
        ro[WS(os, 1)] = SIMD_FMADD(KP866025403, Th, Tc);
        ro[WS(os, 5)] = SIMD_FNMADD(KP866025403, Th, Tc);
        io[WS(os, 4)] = SIMD_FMADD(KP866025403, Tu, Tt);
        io[WS(os, 2)] = SIMD_FNMADD(KP866025403, Tu, Tt);
        io[WS(os, 5)] = SIMD_FNMADD(KP866025403, Tn, Tm);
        io[WS(os, 1)] = SIMD_FMADD(KP866025403, Tn, Tm);
    }
}

template <>
struct c2cf_traits<6, 6>
{
    static constexpr long_t flops           = 48;
    static constexpr long_t operations      = 36;
    static constexpr long_t memory_accesses = 24;
    static constexpr long_t stack_vars      = 38;
    static constexpr long_t constants       = 2;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 6 -first-k 6 -name c2cf
 * -standalone */

/*
 * This function contains 36 FP additions, 12 FP multiplications,
 * (or, 24 additions, 0 multiplications, 12 fused multiply/add),
 * 38 stack variables, 2 constants, and 24 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 6 && ProvidedElements == 6>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        SIMD_FLOAT T1, T2, Tn, To, T4, T5, T7, T8, Tg, Th, Tj, Tk, T3, Tb, Tp;
        SIMD_FLOAT Tx, T6, Tc, T9, Td, Ti, Tu, Tl, Tv, Te, TA, Ta, Ts, Ty, Tw;
        SIMD_FLOAT Tq, Tm, Tt, Tf, Tz, Tr;
        T1            = ri[0];
        T2            = ri[WS(is, 3)];
        Tn            = ii[0];
        To            = ii[WS(is, 3)];
        T4            = ri[WS(is, 2)];
        T5            = ri[WS(is, 5)];
        T7            = ri[WS(is, 4)];
        T8            = ri[WS(is, 1)];
        Tg            = ii[WS(is, 2)];
        Th            = ii[WS(is, 5)];
        Tj            = ii[WS(is, 4)];
        Tk            = ii[WS(is, 1)];
        T3            = T1 - T2;
        Tb            = T1 + T2;
        Tp            = Tn - To;
        Tx            = Tn + To;
        T6            = T4 - T5;
        Tc            = T4 + T5;
        T9            = T7 - T8;
        Td            = T7 + T8;
        Ti            = Tg - Th;
        Tu            = Tg + Th;
        Tl            = Tj - Tk;
        Tv            = Tj + Tk;
        Te            = Tc + Td;
        TA            = Td - Tc;
        Ta            = T6 + T9;
        Ts            = T9 - T6;
        Ty            = Tu + Tv;
        Tw            = Tu - Tv;
        Tq            = Ti + Tl;
        Tm            = Ti - Tl;
        Tt            = SIMD_FNMADD(KP500000000, Te, Tb);
        ro[0]         = Tb + Te;
        Tf            = SIMD_FNMADD(KP500000000, Ta, T3);
        ro[WS(os, 3)] = T3 + Ta;
        Tz            = SIMD_FNMADD(KP500000000, Ty, Tx);
        io[0]         = Tx + Ty;
        Tr            = SIMD_FNMADD(KP500000000, Tq, Tp);
        io[WS(os, 3)] = Tp + Tq;
        ro[WS(os, 4)] = SIMD_FMADD(KP866025403, Tw, Tt);
        ro[WS(os, 2)] = SIMD_FNMADD(KP866025403, Tw, Tt);
        ro[WS(os, 1)] = SIMD_FMADD(KP866025403, Tm, Tf);
        ro[WS(os, 5)] = SIMD_FNMADD(KP866025403, Tm, Tf);
        io[WS(os, 4)] = SIMD_FMADD(KP866025403, TA, Tz);
        io[WS(os, 2)] = SIMD_FNMADD(KP866025403, TA, Tz);
        io[WS(os, 5)] = SIMD_FNMADD(KP866025403, Ts, Tr);
        io[WS(os, 1)] = SIMD_FMADD(KP866025403, Ts, Tr);
    }
}
