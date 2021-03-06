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
struct c2cf_traits<7, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 16;
    static constexpr long_t stack_vars      = 2;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 1 -name c2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 2 stack variables, 0 constants, and 16 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 1>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    {
        SIMD_FLOAT T1, T2;
        T1            = ri[0];
        T2            = ii[0];
        ro[WS(os, 4)] = T1;
        ro[WS(os, 2)] = T1;
        ro[WS(os, 3)] = T1;
        ro[WS(os, 5)] = T1;
        ro[WS(os, 6)] = T1;
        ro[WS(os, 1)] = T1;
        ro[0]         = T1;
        io[WS(os, 4)] = T2;
        io[WS(os, 2)] = T2;
        io[WS(os, 3)] = T2;
        io[WS(os, 5)] = T2;
        io[WS(os, 6)] = T2;
        io[WS(os, 1)] = T2;
        io[0]         = T2;
    }
}

template <>
struct c2cf_traits<7, 2>
{
    static constexpr long_t flops           = 38;
    static constexpr long_t operations      = 20;
    static constexpr long_t memory_accesses = 18;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 2 -name c2cf
 * -standalone */

/*
 * This function contains 20 FP additions, 18 FP multiplications,
 * (or, 2 additions, 0 multiplications, 18 fused multiply/add),
 * 16 stack variables, 6 constants, and 18 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 2>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP433883739, +0.433883739117558120475768332848358754609990728);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP781831482, +0.781831482468029808708444526674057750232334519);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP222520933, +0.222520933956314404288902564496794759466355569);
    DK(KP623489801, +0.623489801858733530525004884004239810632274731);
    {
        SIMD_FLOAT T1, T2, T3, T4, T8, Ta, T6, T7, T9, T5;
        T1            = ri[0];
        T2            = ri[WS(is, 1)];
        T3            = ii[0];
        T4            = ii[WS(is, 1)];
        ro[0]         = T1 + T2;
        T8            = SIMD_FMADD(KP623489801, T2, T1);
        Ta            = SIMD_FNMADD(KP222520933, T2, T1);
        T6            = SIMD_FNMADD(KP900968867, T2, T1);
        io[0]         = T3 + T4;
        T7            = SIMD_FMADD(KP623489801, T4, T3);
        T9            = SIMD_FNMADD(KP900968867, T4, T3);
        T5            = SIMD_FNMADD(KP222520933, T4, T3);
        ro[WS(os, 1)] = SIMD_FMADD(KP781831482, T4, T8);
        ro[WS(os, 6)] = SIMD_FNMADD(KP781831482, T4, T8);
        ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, T4, Ta);
        ro[WS(os, 2)] = SIMD_FMADD(KP974927912, T4, Ta);
        ro[WS(os, 3)] = SIMD_FMADD(KP433883739, T4, T6);
        ro[WS(os, 4)] = SIMD_FNMADD(KP433883739, T4, T6);
        io[WS(os, 1)] = SIMD_FNMADD(KP781831482, T2, T7);
        io[WS(os, 6)] = SIMD_FMADD(KP781831482, T2, T7);
        io[WS(os, 3)] = SIMD_FNMADD(KP433883739, T2, T9);
        io[WS(os, 4)] = SIMD_FMADD(KP433883739, T2, T9);
        io[WS(os, 5)] = SIMD_FMADD(KP974927912, T2, T5);
        io[WS(os, 2)] = SIMD_FNMADD(KP974927912, T2, T5);
    }
}

template <>
struct c2cf_traits<7, 3>
{
    static constexpr long_t flops           = 64;
    static constexpr long_t operations      = 34;
    static constexpr long_t memory_accesses = 20;
    static constexpr long_t stack_vars      = 34;
    static constexpr long_t constants       = 10;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 3 -name c2cf
 * -standalone */

/*
 * This function contains 34 FP additions, 30 FP multiplications,
 * (or, 4 additions, 0 multiplications, 30 fused multiply/add),
 * 34 stack variables, 10 constants, and 20 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 3>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP781831482, +0.781831482468029808708444526674057750232334519);
    DK(KP623489801, +0.623489801858733530525004884004239810632274731);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP246979603, +0.246979603717467061050009768008479621264549462);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP445041867, +0.445041867912628808577805128993589518932711138);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, T9, T2, T3, T6, T7, Tf, Tl, Tm, Tc, T4, T8, Ta, To, Tj;
        SIMD_FLOAT Ti, Td, Th, Tg, Tn, T5, Te, Tk, Tb;
        T1            = ri[0];
        T9            = ii[0];
        T2            = ri[WS(is, 1)];
        T3            = ri[WS(is, 2)];
        T6            = ii[WS(is, 1)];
        T7            = ii[WS(is, 2)];
        ro[0]         = T1 + T2 + T3;
        Tf            = SIMD_FNMADD(KP554958132, T2, T3);
        Tg            = SIMD_FMSUB(KP692021471, T3, T2);
        Tl            = SIMD_FMSUB(KP445041867, T3, T2);
        Tm            = SIMD_FNMADD(KP356895867, T3, T2);
        Tc            = SIMD_FMADD(KP801937735, T2, T3);
        T4            = SIMD_FMADD(KP246979603, T2, T3);
        io[0]         = T9 + T6 + T7;
        T8            = SIMD_FNMADD(KP445041867, T7, T6);
        Ta            = SIMD_FNMADD(KP356895867, T7, T6);
        To            = SIMD_FMADD(KP801937735, T6, T7);
        Tj            = SIMD_FMADD(KP246979603, T6, T7);
        Ti            = SIMD_FMSUB(KP554958132, T6, T7);
        Td            = SIMD_FMSUB(KP692021471, T7, T6);
        Th            = SIMD_FMADD(KP900968867, Tg, T1);
        Tn            = SIMD_FMADD(KP623489801, Tm, T1);
        T5            = SIMD_FNMADD(KP900968867, T4, T1);
        Te            = SIMD_FMADD(KP900968867, Td, T9);
        Tk            = SIMD_FNMADD(KP900968867, Tj, T9);
        Tb            = SIMD_FMADD(KP623489801, Ta, T9);
        ro[WS(os, 3)] = SIMD_FMADD(KP781831482, Ti, Th);
        ro[WS(os, 4)] = SIMD_FNMADD(KP781831482, Ti, Th);
        ro[WS(os, 1)] = SIMD_FMADD(KP974927912, To, Tn);
        ro[WS(os, 6)] = SIMD_FNMADD(KP974927912, To, Tn);
        ro[WS(os, 2)] = SIMD_FMADD(KP974927912, T8, T5);
        ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, T8, T5);
        io[WS(os, 4)] = SIMD_FNMADD(KP781831482, Tf, Te);
        io[WS(os, 3)] = SIMD_FMADD(KP781831482, Tf, Te);
        io[WS(os, 5)] = SIMD_FNMADD(KP974927912, Tl, Tk);
        io[WS(os, 2)] = SIMD_FMADD(KP974927912, Tl, Tk);
        io[WS(os, 6)] = SIMD_FMADD(KP974927912, Tc, Tb);
        io[WS(os, 1)] = SIMD_FNMADD(KP974927912, Tc, Tb);
    }
}

template <>
struct c2cf_traits<7, 4>
{
    static constexpr long_t flops           = 90;
    static constexpr long_t operations      = 48;
    static constexpr long_t memory_accesses = 22;
    static constexpr long_t stack_vars      = 44;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 4 -name c2cf
 * -standalone */

/*
 * This function contains 48 FP additions, 42 FP multiplications,
 * (or, 6 additions, 0 multiplications, 42 fused multiply/add),
 * 44 stack variables, 6 constants, and 22 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 4>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    {
        SIMD_FLOAT T1, T5, T2, T4, T3, T6, T8, T7;
        T1 = ri[0];
        T5 = ii[0];
        T2 = ri[WS(is, 1)];
        T4 = ri[WS(is, 3)];
        T3 = ri[WS(is, 2)];
        T6 = ii[WS(is, 1)];
        T8 = ii[WS(is, 3)];
        T7 = ii[WS(is, 2)];
        {
            SIMD_FLOAT Tx, Tp, Tn, Td, Tz, Tf, Ti, Tk, Ts, TC, Tu, Ta, To, Tw,
                Tm;
            SIMD_FLOAT Ty, Te, Tc, Th, Tj, T9, Tr, TB, Tt, Tq, TA, Tg, Tl, Tv,
                Tb;
            To            = SIMD_FNMADD(KP356895867, T2, T4);
            Tw            = SIMD_FMADD(KP554958132, T4, T2);
            ro[0]         = T1 + T2 + T3 + T4;
            Tm            = SIMD_FMADD(KP554958132, T3, T4);
            Ty            = SIMD_FNMADD(KP356895867, T4, T3);
            Te            = SIMD_FNMADD(KP356895867, T3, T2);
            Tc            = SIMD_FMSUB(KP554958132, T2, T3);
            Th            = SIMD_FMADD(KP554958132, T8, T6);
            Tj            = SIMD_FNMADD(KP356895867, T6, T8);
            io[0]         = T5 + T6 + T7 + T8;
            T9            = SIMD_FNMADD(KP356895867, T8, T7);
            Tr            = SIMD_FMADD(KP554958132, T7, T8);
            TB            = SIMD_FMSUB(KP554958132, T6, T7);
            Tt            = SIMD_FNMADD(KP356895867, T7, T6);
            Tx            = SIMD_FMADD(KP801937735, Tw, T3);
            Tp            = SIMD_FNMADD(KP692021471, To, T3);
            Tn            = SIMD_FNMADD(KP801937735, Tm, T2);
            Td            = SIMD_FMADD(KP801937735, Tc, T4);
            Tz            = SIMD_FNMADD(KP692021471, Ty, T2);
            Tf            = SIMD_FNMADD(KP692021471, Te, T4);
            Ti            = SIMD_FMADD(KP801937735, Th, T7);
            Tk            = SIMD_FNMADD(KP692021471, Tj, T7);
            Ts            = SIMD_FNMADD(KP801937735, Tr, T6);
            TC            = SIMD_FMADD(KP801937735, TB, T8);
            Tu            = SIMD_FNMADD(KP692021471, Tt, T8);
            Ta            = SIMD_FNMADD(KP692021471, T9, T6);
            Tq            = SIMD_FNMADD(KP900968867, Tp, T1);
            TA            = SIMD_FNMADD(KP900968867, Tz, T1);
            Tg            = SIMD_FNMADD(KP900968867, Tf, T1);
            Tl            = SIMD_FNMADD(KP900968867, Tk, T5);
            Tv            = SIMD_FNMADD(KP900968867, Tu, T5);
            Tb            = SIMD_FNMADD(KP900968867, Ta, T5);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Ts, Tq);
            ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, Ts, Tq);
            ro[WS(os, 3)] = SIMD_FMADD(KP974927912, TC, TA);
            ro[WS(os, 4)] = SIMD_FNMADD(KP974927912, TC, TA);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, Ti, Tg);
            ro[WS(os, 6)] = SIMD_FNMADD(KP974927912, Ti, Tg);
            io[WS(os, 5)] = SIMD_FMADD(KP974927912, Tn, Tl);
            io[WS(os, 2)] = SIMD_FNMADD(KP974927912, Tn, Tl);
            io[WS(os, 6)] = SIMD_FMADD(KP974927912, Tx, Tv);
            io[WS(os, 1)] = SIMD_FNMADD(KP974927912, Tx, Tv);
            io[WS(os, 4)] = SIMD_FMADD(KP974927912, Td, Tb);
            io[WS(os, 3)] = SIMD_FNMADD(KP974927912, Td, Tb);
        }
    }
}

template <>
struct c2cf_traits<7, 5>
{
    static constexpr long_t flops           = 94;
    static constexpr long_t operations      = 52;
    static constexpr long_t memory_accesses = 24;
    static constexpr long_t stack_vars      = 50;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 5 -name c2cf
 * -standalone */

/*
 * This function contains 52 FP additions, 42 FP multiplications,
 * (or, 10 additions, 0 multiplications, 42 fused multiply/add),
 * 50 stack variables, 6 constants, and 24 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 5>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    {
        SIMD_FLOAT T1, Tr, T2, T3, T4, T5, Ta, Tb, Tc, Td;
        T1 = ri[0];
        Tr = ii[0];
        T2 = ri[WS(is, 1)];
        T3 = ri[WS(is, 2)];
        T4 = ri[WS(is, 3)];
        T5 = ri[WS(is, 4)];
        Ta = ii[WS(is, 2)];
        Tb = ii[WS(is, 1)];
        Tc = ii[WS(is, 3)];
        Td = ii[WS(is, 4)];
        {
            SIMD_FLOAT T7, Tx, T6, Tw, TE, Tp, Te, Ts, Ty, Tt, Tz, TF, Tq, Tk,
                Tf;
            SIMD_FLOAT TD, TI, Tn, Ti, T9, TC, TH, Tm, Th, T8, Tu, TA, TG, Tl,
                Tg;
            SIMD_FLOAT To, Tj, Tv, TB;
            T7            = SIMD_FNMADD(KP356895867, T3, T2);
            Tx            = SIMD_FNMADD(KP554958132, T2, T3);
            T6            = T4 + T5;
            Tw            = T5 - T4;
            TE            = SIMD_FNMADD(KP356895867, Ta, Tb);
            Tp            = SIMD_FMSUB(KP554958132, Tb, Ta);
            Te            = Tc - Td;
            Ts            = Tc + Td;
            Ty            = SIMD_FMADD(KP801937735, Tx, Tw);
            TC            = SIMD_FNMADD(KP554958132, T3, Tw);
            TH            = SIMD_FNMADD(KP554958132, Tw, T2);
            ro[0]         = T1 + T2 + T3 + T6;
            Tm            = SIMD_FNMADD(KP356895867, T6, T3);
            Th            = SIMD_FNMADD(KP356895867, T2, T6);
            T8            = SIMD_FNMADD(KP692021471, T7, T6);
            io[0]         = Tr + Tb + Ta + Ts;
            Tt            = SIMD_FNMADD(KP356895867, Ts, Ta);
            Tz            = SIMD_FNMADD(KP356895867, Tb, Ts);
            TF            = SIMD_FNMADD(KP692021471, TE, Ts);
            Tq            = SIMD_FMADD(KP801937735, Tp, Te);
            Tk            = SIMD_FMADD(KP554958132, Ta, Te);
            Tf            = SIMD_FMADD(KP554958132, Te, Tb);
            TD            = SIMD_FMADD(KP801937735, TC, T2);
            TI            = SIMD_FMADD(KP801937735, TH, T3);
            Tn            = SIMD_FNMADD(KP692021471, Tm, T2);
            Ti            = SIMD_FNMADD(KP692021471, Th, T3);
            T9            = SIMD_FNMADD(KP900968867, T8, T1);
            Tu            = SIMD_FNMADD(KP692021471, Tt, Tb);
            TA            = SIMD_FNMADD(KP692021471, Tz, Ta);
            TG            = SIMD_FNMADD(KP900968867, TF, Tr);
            Tl            = SIMD_FNMADD(KP801937735, Tk, Tb);
            Tg            = SIMD_FMADD(KP801937735, Tf, Ta);
            To            = SIMD_FNMADD(KP900968867, Tn, T1);
            Tj            = SIMD_FNMADD(KP900968867, Ti, T1);
            Tv            = SIMD_FNMADD(KP900968867, Tu, Tr);
            TB            = SIMD_FNMADD(KP900968867, TA, Tr);
            io[WS(os, 6)] = SIMD_FMADD(KP974927912, TI, TG);
            io[WS(os, 1)] = SIMD_FNMADD(KP974927912, TI, TG);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, Tg, T9);
            ro[WS(os, 6)] = SIMD_FNMADD(KP974927912, Tg, T9);
            ro[WS(os, 3)] = SIMD_FMADD(KP974927912, Tq, To);
            ro[WS(os, 4)] = SIMD_FNMADD(KP974927912, Tq, To);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Tl, Tj);
            ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, Tl, Tj);
            io[WS(os, 4)] = SIMD_FNMADD(KP974927912, Ty, Tv);
            io[WS(os, 3)] = SIMD_FMADD(KP974927912, Ty, Tv);
            io[WS(os, 5)] = SIMD_FMADD(KP974927912, TD, TB);
            io[WS(os, 2)] = SIMD_FNMADD(KP974927912, TD, TB);
        }
    }
}

template <>
struct c2cf_traits<7, 6>
{
    static constexpr long_t flops           = 98;
    static constexpr long_t operations      = 56;
    static constexpr long_t memory_accesses = 26;
    static constexpr long_t stack_vars      = 56;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 6 -name c2cf
 * -standalone */

/*
 * This function contains 56 FP additions, 42 FP multiplications,
 * (or, 14 additions, 0 multiplications, 42 fused multiply/add),
 * 56 stack variables, 6 constants, and 26 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 6>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, Tv, T2, T6, T7, T3, T4, Tf, Tg, Th, Tc, Td, T8, TB, T5;
        SIMD_FLOAT TC, Ti, Tx, Te, Tw;
        T1 = ri[0];
        Tv = ii[0];
        T2 = ri[WS(is, 1)];
        T6 = ri[WS(is, 3)];
        T7 = ri[WS(is, 4)];
        T3 = ri[WS(is, 2)];
        T4 = ri[WS(is, 5)];
        Tf = ii[WS(is, 1)];
        Tg = ii[WS(is, 3)];
        Th = ii[WS(is, 4)];
        Tc = ii[WS(is, 2)];
        Td = ii[WS(is, 5)];
        T8 = T6 + T7;
        TB = T7 - T6;
        T5 = T3 + T4;
        TC = T4 - T3;
        Ti = Tg - Th;
        Tx = Tg + Th;
        Te = Tc - Td;
        Tw = Tc + Td;
        {
            SIMD_FLOAT TO, Tm, TE, TJ, Tr, Ta, TG, Tk, Tz, TL, Tu, Tp, TN, Tl,
                TD;
            SIMD_FLOAT TI, Tq, T9, TF, Tj, Ty, TK, Tt, To, Tn, Ts, Tb, TH, TA,
                TM;
            TN            = SIMD_FMSUB(KP554958132, TB, T2);
            Tl            = SIMD_FNMADD(KP356895867, T2, T8);
            TD            = SIMD_FMADD(KP554958132, T2, TC);
            TI            = SIMD_FMADD(KP554958132, TC, TB);
            ro[0]         = T1 + T2 + T5 + T8;
            Tq            = SIMD_FNMADD(KP356895867, T8, T5);
            T9            = SIMD_FNMADD(KP356895867, T5, T2);
            TF            = SIMD_FNMADD(KP356895867, Tf, Tx);
            Tj            = SIMD_FMADD(KP554958132, Ti, Tf);
            io[0]         = Tv + Tf + Tw + Tx;
            Ty            = SIMD_FNMADD(KP356895867, Tx, Tw);
            TK            = SIMD_FNMADD(KP356895867, Tw, Tf);
            Tt            = SIMD_FNMADD(KP554958132, Tf, Te);
            To            = SIMD_FMADD(KP554958132, Te, Ti);
            TO            = SIMD_FMADD(KP801937735, TN, TC);
            Tm            = SIMD_FNMADD(KP692021471, Tl, T5);
            TE            = SIMD_FNMADD(KP801937735, TD, TB);
            TJ            = SIMD_FMADD(KP801937735, TI, T2);
            Tr            = SIMD_FNMADD(KP692021471, Tq, T2);
            Ta            = SIMD_FNMADD(KP692021471, T9, T8);
            TG            = SIMD_FNMADD(KP692021471, TF, Tw);
            Tk            = SIMD_FMADD(KP801937735, Tj, Te);
            Tz            = SIMD_FNMADD(KP692021471, Ty, Tf);
            TL            = SIMD_FNMADD(KP692021471, TK, Tx);
            Tu            = SIMD_FNMADD(KP801937735, Tt, Ti);
            Tp            = SIMD_FNMADD(KP801937735, To, Tf);
            Tn            = SIMD_FNMADD(KP900968867, Tm, T1);
            Ts            = SIMD_FNMADD(KP900968867, Tr, T1);
            Tb            = SIMD_FNMADD(KP900968867, Ta, T1);
            TH            = SIMD_FNMADD(KP900968867, TG, Tv);
            TA            = SIMD_FNMADD(KP900968867, Tz, Tv);
            TM            = SIMD_FNMADD(KP900968867, TL, Tv);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Tp, Tn);
            ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, Tp, Tn);
            ro[WS(os, 3)] = SIMD_FMADD(KP974927912, Tu, Ts);
            ro[WS(os, 4)] = SIMD_FNMADD(KP974927912, Tu, Ts);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, Tk, Tb);
            ro[WS(os, 6)] = SIMD_FNMADD(KP974927912, Tk, Tb);
            io[WS(os, 5)] = SIMD_FMADD(KP974927912, TJ, TH);
            io[WS(os, 2)] = SIMD_FNMADD(KP974927912, TJ, TH);
            io[WS(os, 4)] = SIMD_FNMADD(KP974927912, TE, TA);
            io[WS(os, 3)] = SIMD_FMADD(KP974927912, TE, TA);
            io[WS(os, 6)] = SIMD_FNMADD(KP974927912, TO, TM);
            io[WS(os, 1)] = SIMD_FMADD(KP974927912, TO, TM);
        }
    }
}

template <>
struct c2cf_traits<7, 7>
{
    static constexpr long_t flops           = 102;
    static constexpr long_t operations      = 60;
    static constexpr long_t memory_accesses = 28;
    static constexpr long_t stack_vars      = 62;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 7 -name c2cf
 * -standalone */

/*
 * This function contains 60 FP additions, 42 FP multiplications,
 * (or, 18 additions, 0 multiplications, 42 fused multiply/add),
 * 62 stack variables, 6 constants, and 28 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 7>::type
c2cf(SIMD_FLOAT const* __restrict ri, SIMD_FLOAT const* __restrict ii,
     SIMD_FLOAT* __restrict ro, SIMD_FLOAT* __restrict io)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, Tz, T2, T3, T8, T9, T5, T6, Te, Tf, Tk, Tl, Th, Ti, T4;
        SIMD_FLOAT TI, Ta, TG, T7, TH, Tg, TB, Tm, TC, Tj, TA;
        T1 = ri[0];
        Tz = ii[0];
        T2 = ri[WS(is, 1)];
        T3 = ri[WS(is, 6)];
        T8 = ri[WS(is, 3)];
        T9 = ri[WS(is, 4)];
        T5 = ri[WS(is, 2)];
        T6 = ri[WS(is, 5)];
        Te = ii[WS(is, 2)];
        Tf = ii[WS(is, 5)];
        Tk = ii[WS(is, 3)];
        Tl = ii[WS(is, 4)];
        Th = ii[WS(is, 1)];
        Ti = ii[WS(is, 6)];
        T4 = T2 + T3;
        TI = T3 - T2;
        Ta = T8 + T9;
        TG = T9 - T8;
        T7 = T5 + T6;
        TH = T6 - T5;
        Tg = Te - Tf;
        TB = Te + Tf;
        Tm = Tk - Tl;
        TC = Tk + Tl;
        Tj = Th - Ti;
        TA = Th + Ti;
        {
            SIMD_FLOAT TU, Tq, TK, TP, Tv, Tc, TE, Tt, TM, TR, Ty, To, TT, Tp,
                TJ;
            SIMD_FLOAT TO, Tu, Tb, TD, Ts, TL, TQ, Tx, Tn, Tr, Tw, Td, TF, TN,
                TS;
            TT            = SIMD_FMADD(KP554958132, TG, TI);
            Tp            = SIMD_FNMADD(KP356895867, T4, Ta);
            TJ            = SIMD_FNMADD(KP554958132, TI, TH);
            TO            = SIMD_FMADD(KP554958132, TH, TG);
            ro[0]         = T1 + T4 + T7 + Ta;
            Tu            = SIMD_FNMADD(KP356895867, Ta, T7);
            Tb            = SIMD_FNMADD(KP356895867, T7, T4);
            TD            = SIMD_FNMADD(KP356895867, TC, TB);
            Ts            = SIMD_FMADD(KP554958132, Tg, Tm);
            io[0]         = Tz + TA + TB + TC;
            TL            = SIMD_FNMADD(KP356895867, TA, TC);
            TQ            = SIMD_FNMADD(KP356895867, TB, TA);
            Tx            = SIMD_FNMADD(KP554958132, Tj, Tg);
            Tn            = SIMD_FMADD(KP554958132, Tm, Tj);
            TU            = SIMD_FMADD(KP801937735, TT, TH);
            Tq            = SIMD_FNMADD(KP692021471, Tp, T7);
            TK            = SIMD_FNMADD(KP801937735, TJ, TG);
            TP            = SIMD_FNMADD(KP801937735, TO, TI);
            Tv            = SIMD_FNMADD(KP692021471, Tu, T4);
            Tc            = SIMD_FNMADD(KP692021471, Tb, Ta);
            TE            = SIMD_FNMADD(KP692021471, TD, TA);
            Tt            = SIMD_FNMADD(KP801937735, Ts, Tj);
            TM            = SIMD_FNMADD(KP692021471, TL, TB);
            TR            = SIMD_FNMADD(KP692021471, TQ, TC);
            Ty            = SIMD_FNMADD(KP801937735, Tx, Tm);
            To            = SIMD_FMADD(KP801937735, Tn, Tg);
            Tr            = SIMD_FNMADD(KP900968867, Tq, T1);
            Tw            = SIMD_FNMADD(KP900968867, Tv, T1);
            Td            = SIMD_FNMADD(KP900968867, Tc, T1);
            TF            = SIMD_FNMADD(KP900968867, TE, Tz);
            TN            = SIMD_FNMADD(KP900968867, TM, Tz);
            TS            = SIMD_FNMADD(KP900968867, TR, Tz);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Tt, Tr);
            ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, Tt, Tr);
            ro[WS(os, 3)] = SIMD_FMADD(KP974927912, Ty, Tw);
            ro[WS(os, 4)] = SIMD_FNMADD(KP974927912, Ty, Tw);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, To, Td);
            ro[WS(os, 6)] = SIMD_FNMADD(KP974927912, To, Td);
            io[WS(os, 4)] = SIMD_FNMADD(KP974927912, TK, TF);
            io[WS(os, 3)] = SIMD_FMADD(KP974927912, TK, TF);
            io[WS(os, 5)] = SIMD_FNMADD(KP974927912, TP, TN);
            io[WS(os, 2)] = SIMD_FMADD(KP974927912, TP, TN);
            io[WS(os, 6)] = SIMD_FNMADD(KP974927912, TU, TS);
            io[WS(os, 1)] = SIMD_FMADD(KP974927912, TU, TS);
        }
    }
}
