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
struct c2cb_traits<7, 0>
{
    static constexpr long_t flops           = 102;
    static constexpr long_t operations      = 60;
    static constexpr long_t memory_accesses = 28;
    static constexpr long_t stack_vars      = 62;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 0 -name c2cb
 * -standalone */

/*
 * This function contains 60 FP additions, 42 FP multiplications,
 * (or, 18 additions, 0 multiplications, 42 fused multiply/add),
 * 62 stack variables, 6 constants, and 28 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 0>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
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

template <>
struct c2cb_traits<7, 1>
{
    static constexpr long_t flops           = 96;
    static constexpr long_t operations      = 54;
    static constexpr long_t memory_accesses = 26;
    static constexpr long_t stack_vars      = 62;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 1 -name c2cb
 * -standalone */

/*
 * This function contains 54 FP additions, 42 FP multiplications,
 * (or, 12 additions, 0 multiplications, 42 fused multiply/add),
 * 62 stack variables, 6 constants, and 26 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 1>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, Tz, T2, T3, T8, T9, T5, T6, Te, Tf, Tk, Tl, Th, Ti, T4;
        SIMD_FLOAT TI, Ta, TG, T7, TH, Tg, TC, Tm, TA, Tj, TB;
        T1 = ri[0];
        Tz = ii[0];
        T2 = ri[WS(is, 3)];
        T3 = ri[WS(is, 4)];
        T8 = ri[WS(is, 2)];
        T9 = ri[WS(is, 5)];
        T5 = ri[WS(is, 1)];
        T6 = ri[WS(is, 6)];
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
        TC = Te + Tf;
        Tm = Tk - Tl;
        TA = Tk + Tl;
        Tj = Th - Ti;
        TB = Th + Ti;
        {
            SIMD_FLOAT TP, Tv, TK, TU, Tq, Tc, TR, Tt, TE, TM, Ty, To, TO, Tu,
                TJ;
            SIMD_FLOAT TT, Tp, Tb, TQ, Ts, TD, TL, Tx, Tn, Tw, Tr, Td, TS, TF,
                TN;
            TO            = SIMD_FMADD(KP554958132, TG, TI);
            Tu            = SIMD_FNMADD(KP356895867, T4, Ta);
            TJ            = SIMD_FMADD(KP554958132, TI, TH);
            TT            = SIMD_FNMADD(KP554958132, TH, TG);
            Tp            = SIMD_FNMADD(KP356895867, T7, T4);
            Tb            = SIMD_FNMADD(KP356895867, Ta, T7);
            TQ            = SIMD_FNMADD(KP356895867, TA, TC);
            Ts            = SIMD_FMADD(KP554958132, Tg, Tm);
            TD            = SIMD_FNMADD(KP356895867, TC, TB);
            TL            = SIMD_FNMADD(KP356895867, TB, TA);
            Tx            = SIMD_FNMADD(KP554958132, Tj, Tg);
            Tn            = SIMD_FMADD(KP554958132, Tm, Tj);
            TP            = SIMD_FNMADD(KP801937735, TO, TH);
            Tv            = SIMD_FNMADD(KP692021471, Tu, T7);
            TK            = SIMD_FMADD(KP801937735, TJ, TG);
            TU            = SIMD_FNMADD(KP801937735, TT, TI);
            Tq            = SIMD_FNMADD(KP692021471, Tp, Ta);
            Tc            = SIMD_FNMADD(KP692021471, Tb, T4);
            TR            = SIMD_FNMADD(KP692021471, TQ, TB);
            Tt            = SIMD_FNMADD(KP801937735, Ts, Tj);
            TE            = SIMD_FNMADD(KP692021471, TD, TA);
            TM            = SIMD_FNMADD(KP692021471, TL, TC);
            Ty            = SIMD_FNMADD(KP801937735, Tx, Tm);
            To            = SIMD_FMADD(KP801937735, Tn, Tg);
            Tw            = SIMD_FNMADD(KP900968867, Tv, T1);
            Tr            = SIMD_FNMADD(KP900968867, Tq, T1);
            Td            = SIMD_FNMADD(KP900968867, Tc, T1);
            TS            = SIMD_FNMADD(KP900968867, TR, Tz);
            TF            = SIMD_FNMADD(KP900968867, TE, Tz);
            TN            = SIMD_FNMADD(KP900968867, TM, Tz);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Ty, Tw);
            ro[WS(os, 3)] = SIMD_FNMADD(KP974927912, Ty, Tw);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, Tt, Tr);
            ro[WS(os, 4)] = SIMD_FNMADD(KP974927912, Tt, Tr);
            ro[0]         = SIMD_FMADD(KP974927912, To, Td);
            ro[WS(os, 5)] = SIMD_FNMADD(KP974927912, To, Td);
            io[WS(os, 3)] = SIMD_FNMADD(KP974927912, TU, TS);
            io[WS(os, 2)] = SIMD_FMADD(KP974927912, TU, TS);
            io[WS(os, 5)] = SIMD_FNMADD(KP974927912, TK, TF);
            io[0]         = SIMD_FMADD(KP974927912, TK, TF);
            io[WS(os, 4)] = SIMD_FNMADD(KP974927912, TP, TN);
            io[WS(os, 1)] = SIMD_FMADD(KP974927912, TP, TN);
        }
    }
}

template <>
struct c2cb_traits<7, 2>
{
    static constexpr long_t flops           = 92;
    static constexpr long_t operations      = 52;
    static constexpr long_t memory_accesses = 24;
    static constexpr long_t stack_vars      = 67;
    static constexpr long_t constants       = 11;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 2 -name c2cb
 * -standalone */

/*
 * This function contains 52 FP additions, 40 FP multiplications,
 * (or, 12 additions, 0 multiplications, 40 fused multiply/add),
 * 67 stack variables, 11 constants, and 24 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 2>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP924138961, +0.924138961091093314542963749712412773983469263);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP867767478, +0.867767478235116240951536665696717509219981456);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP797473388, +0.797473388882403961415688254214434877743320657);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP695895486, +0.695895486700943376465352387651397991244687582);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP512858431, +0.512858431636276949746649808137724830241051169);
    {
        SIMD_FLOAT T1, Tz, T2, T3, Tk, Tl, T5, T6, Th, Ti, Te, Tf, T8, T9, T4;
        SIMD_FLOAT TI, Tm, TA, T7, TH, Tj, TB, Tg, TC, Ta, TG;
        T1 = ri[0];
        Tz = ii[0];
        T2 = ri[WS(is, 2)];
        T3 = ri[WS(is, 5)];
        Tk = ii[WS(is, 5)];
        Tl = ii[WS(is, 2)];
        T5 = ri[WS(is, 3)];
        T6 = ri[WS(is, 4)];
        Th = ii[WS(is, 4)];
        Ti = ii[WS(is, 3)];
        Te = ii[WS(is, 1)];
        Tf = ii[WS(is, 6)];
        T8 = ri[WS(is, 1)];
        T9 = ri[WS(is, 6)];
        T4 = T2 + T3;
        TI = T2 - T3;
        Tm = Tk - Tl;
        TA = Tk + Tl;
        T7 = T5 + T6;
        TH = T5 - T6;
        Tj = Th - Ti;
        TB = Th + Ti;
        Tg = Te - Tf;
        TC = Te + Tf;
        Ta = T8 + T9;
        TG = T9 - T8;
        {
            SIMD_FLOAT TQ, TJ, Tu, TL, Tp, Tn, TD, Tx, TO, Tb, TR, TK, Tv, TM,
                Tq;
            SIMD_FLOAT To, TE, Ty, TP, Tc, TS, Tw, TN, Tr, TF, Td, TT, Ts, TU,
                Tt;
            TQ            = SIMD_FNMADD(KP512858431, TA, TH);
            TJ            = SIMD_FMADD(KP554958132, TI, TH);
            Tu            = SIMD_FNMADD(KP356895867, T7, T4);
            TL            = SIMD_FNMADD(KP356895867, TB, TA);
            Tp            = SIMD_FNMADD(KP512858431, T4, Tj);
            Tn            = SIMD_FMADD(KP554958132, Tm, Tj);
            TD            = SIMD_FNMADD(KP356895867, TC, TB);
            Tx            = SIMD_FMADD(KP554958132, Tg, Tm);
            TO            = SIMD_FMADD(KP554958132, TG, TI);
            Tb            = SIMD_FNMADD(KP356895867, Ta, T7);
            TR            = SIMD_FMADD(KP695895486, TQ, TC);
            TK            = SIMD_FMADD(KP801937735, TJ, TG);
            Tv            = SIMD_FNMADD(KP692021471, Tu, Ta);
            TM            = SIMD_FNMADD(KP692021471, TL, TC);
            Tq            = SIMD_FMADD(KP695895486, Tp, Ta);
            To            = SIMD_FMADD(KP801937735, Tn, Tg);
            TE            = SIMD_FNMADD(KP692021471, TD, TA);
            Ty            = SIMD_FNMADD(KP801937735, Tx, Tj);
            TP            = SIMD_FNMADD(KP801937735, TO, TH);
            Tc            = SIMD_FNMADD(KP692021471, Tb, T4);
            TS            = SIMD_FNMADD(KP797473388, TR, TG);
            Tw            = SIMD_FNMADD(KP900968867, Tv, T1);
            TN            = SIMD_FNMADD(KP900968867, TM, Tz);
            Tr            = SIMD_FNMADD(KP797473388, Tq, Tg);
            TF            = SIMD_FNMADD(KP900968867, TE, Tz);
            Td            = SIMD_FNMADD(KP900968867, Tc, T1);
            TT            = SIMD_FMADD(KP867767478, TS, TB);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Ty, Tw);
            ro[WS(os, 1)] = SIMD_FNMADD(KP974927912, Ty, Tw);
            io[WS(os, 2)] = SIMD_FMADD(KP974927912, TP, TN);
            io[WS(os, 1)] = SIMD_FNMADD(KP974927912, TP, TN);
            Ts            = SIMD_FMADD(KP867767478, Tr, T7);
            io[WS(os, 3)] = SIMD_FNMADD(KP974927912, TK, TF);
            io[0]         = SIMD_FMADD(KP974927912, TK, TF);
            ro[0]         = SIMD_FMADD(KP974927912, To, Td);
            ro[WS(os, 3)] = SIMD_FNMADD(KP974927912, To, Td);
            TU            = SIMD_FNMADD(KP924138961, TT, TI);
            Tt            = SIMD_FNMADD(KP924138961, Ts, Tm);
            io[WS(os, 4)] = SIMD_FMADD(KP974927912, TU, Tz);
            ro[WS(os, 4)] = SIMD_FMADD(KP974927912, Tt, T1);
        }
    }
}

template <>
struct c2cb_traits<7, 3>
{
    static constexpr long_t flops           = 88;
    static constexpr long_t operations      = 50;
    static constexpr long_t memory_accesses = 22;
    static constexpr long_t stack_vars      = 67;
    static constexpr long_t constants       = 11;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 3 -name c2cb
 * -standalone */

/*
 * This function contains 50 FP additions, 38 FP multiplications,
 * (or, 12 additions, 0 multiplications, 38 fused multiply/add),
 * 67 stack variables, 11 constants, and 22 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 3>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP924138961, +0.924138961091093314542963749712412773983469263);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP867767478, +0.867767478235116240951536665696717509219981456);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP797473388, +0.797473388882403961415688254214434877743320657);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP695895486, +0.695895486700943376465352387651397991244687582);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP512858431, +0.512858431636276949746649808137724830241051169);
    {
        SIMD_FLOAT T1, Tz, T2, T3, Th, Ti, T5, T6, Te, Tf, Tb, Tc, T8, T9, T4;
        SIMD_FLOAT TA, Tj, TI, T7, TH, Tg, TB, Td, TG, Ta, TC;
        T1 = ri[0];
        Tz = ii[0];
        T2 = ii[WS(is, 6)];
        T3 = ii[WS(is, 1)];
        Th = ri[WS(is, 1)];
        Ti = ri[WS(is, 6)];
        T5 = ri[WS(is, 2)];
        T6 = ri[WS(is, 5)];
        Te = ii[WS(is, 5)];
        Tf = ii[WS(is, 2)];
        Tb = ri[WS(is, 3)];
        Tc = ri[WS(is, 4)];
        T8 = ii[WS(is, 3)];
        T9 = ii[WS(is, 4)];
        T4 = T2 - T3;
        TA = T3 + T2;
        Tj = Th + Ti;
        TI = Th - Ti;
        T7 = T5 + T6;
        TH = T5 - T6;
        Tg = Te - Tf;
        TB = Te + Tf;
        Td = Tb + Tc;
        TG = Tc - Tb;
        Ta = T8 - T9;
        TC = T8 + T9;
        {
            SIMD_FLOAT TL, TJ, Tk, Ts, TQ, Tp, TD, Tu, TM, TK, Tl, Tt, TR, Tq,
                TE;
            SIMD_FLOAT Tv, TN, Tm, TS, Tr, TF, Tw, TO, Tn, TT, Tx, TP, To, TU,
                Ty;
            TL            = SIMD_FMADD(KP512858431, TA, TH);
            TJ            = SIMD_FNMADD(KP554958132, TI, TH);
            Tk            = SIMD_FMADD(KP512858431, Tj, Tg);
            Ts            = SIMD_FNMADD(KP554958132, T4, Tg);
            TQ            = SIMD_FMADD(KP512858431, TB, TG);
            Tp            = SIMD_FNMADD(KP356895867, Td, T7);
            TD            = SIMD_FNMADD(KP356895867, TC, TB);
            Tu            = SIMD_FMADD(KP512858431, T7, Ta);
            TM            = SIMD_FNMADD(KP695895486, TL, TC);
            TK            = SIMD_FMADD(KP801937735, TJ, TG);
            Tl            = SIMD_FNMADD(KP695895486, Tk, Td);
            Tt            = SIMD_FMADD(KP801937735, Ts, Ta);
            TR            = SIMD_FNMADD(KP695895486, TQ, TA);
            Tq            = SIMD_FNMADD(KP692021471, Tp, Tj);
            TE            = SIMD_FNMADD(KP692021471, TD, TA);
            Tv            = SIMD_FNMADD(KP695895486, Tu, Tj);
            TN            = SIMD_FMADD(KP797473388, TM, TG);
            Tm            = SIMD_FMADD(KP797473388, Tl, Ta);
            TS            = SIMD_FMADD(KP797473388, TR, TI);
            Tr            = SIMD_FNMADD(KP900968867, Tq, T1);
            TF            = SIMD_FNMADD(KP900968867, TE, Tz);
            Tw            = SIMD_FMADD(KP797473388, Tv, T4);
            TO            = SIMD_FNMADD(KP867767478, TN, TB);
            Tn            = SIMD_FNMADD(KP867767478, Tm, T7);
            TT            = SIMD_FNMADD(KP867767478, TS, TC);
            ro[0]         = SIMD_FMADD(KP974927912, Tt, Tr);
            ro[WS(os, 1)] = SIMD_FNMADD(KP974927912, Tt, Tr);
            io[WS(os, 1)] = SIMD_FNMADD(KP974927912, TK, TF);
            io[0]         = SIMD_FMADD(KP974927912, TK, TF);
            Tx            = SIMD_FNMADD(KP867767478, Tw, Td);
            TP            = SIMD_FNMADD(KP924138961, TO, TI);
            To            = SIMD_FNMADD(KP924138961, Tn, T4);
            TU            = SIMD_FNMADD(KP924138961, TT, TH);
            Ty            = SIMD_FNMADD(KP924138961, Tx, Tg);
            io[WS(os, 2)] = SIMD_FMADD(KP974927912, TP, Tz);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, To, T1);
            io[WS(os, 3)] = SIMD_FMADD(KP974927912, TU, Tz);
            ro[WS(os, 3)] = SIMD_FMADD(KP974927912, Ty, T1);
        }
    }
}

template <>
struct c2cb_traits<7, 4>
{
    static constexpr long_t flops           = 84;
    static constexpr long_t operations      = 48;
    static constexpr long_t memory_accesses = 20;
    static constexpr long_t stack_vars      = 62;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 4 -name c2cb
 * -standalone */

/*
 * This function contains 48 FP additions, 36 FP multiplications,
 * (or, 12 additions, 0 multiplications, 36 fused multiply/add),
 * 62 stack variables, 6 constants, and 20 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 4>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP924138961, +0.924138961091093314542963749712412773983469263);
    DK(KP867767478, +0.867767478235116240951536665696717509219981456);
    DK(KP797473388, +0.797473388882403961415688254214434877743320657);
    DK(KP695895486, +0.695895486700943376465352387651397991244687582);
    DK(KP512858431, +0.512858431636276949746649808137724830241051169);
    {
        SIMD_FLOAT T1, Tz, T2, T3, Th, Ti, T5, T6, Te, Tf, Tb, Tc, T8, T9, T4;
        SIMD_FLOAT TF, Tj, TA, T7, TE, Tg, TB, Td, TC, Ta, TD;
        T1 = ri[0];
        Tz = ii[0];
        T2 = ii[WS(is, 4)];
        T3 = ii[WS(is, 3)];
        Th = ri[WS(is, 3)];
        Ti = ri[WS(is, 4)];
        T5 = ri[WS(is, 1)];
        T6 = ri[WS(is, 6)];
        Te = ii[WS(is, 6)];
        Tf = ii[WS(is, 1)];
        Tb = ri[WS(is, 2)];
        Tc = ri[WS(is, 5)];
        T8 = ii[WS(is, 5)];
        T9 = ii[WS(is, 2)];
        T4 = T2 - T3;
        TF = T3 + T2;
        Tj = Th + Ti;
        TA = Th - Ti;
        T7 = T5 + T6;
        TE = T5 - T6;
        Tg = Te - Tf;
        TB = Tf + Te;
        Td = Tb + Tc;
        TC = Tb - Tc;
        Ta = T8 - T9;
        TD = T8 + T9;
        {
            SIMD_FLOAT TG, Tk, TL, Tu, TQ, Tp, TH, Tl, TM, Tv, TR, Tq, TI, Tm,
                TN;
            SIMD_FLOAT Tw, TS, Tr, TJ, Tn, TO, Tx, TT, Ts, TK, To, TP, Ty, TU,
                Tt;
            TG            = SIMD_FNMADD(KP512858431, TF, TE);
            Tk            = SIMD_FNMADD(KP512858431, Tj, Tg);
            TL            = SIMD_FMADD(KP512858431, TB, TC);
            Tu            = SIMD_FNMADD(KP512858431, Td, T4);
            TQ            = SIMD_FNMADD(KP512858431, TD, TA);
            Tp            = SIMD_FMADD(KP512858431, T7, Ta);
            TH            = SIMD_FMADD(KP695895486, TG, TD);
            Tl            = SIMD_FMADD(KP695895486, Tk, Td);
            TM            = SIMD_FNMADD(KP695895486, TL, TF);
            Tv            = SIMD_FMADD(KP695895486, Tu, T7);
            TR            = SIMD_FMADD(KP695895486, TQ, TB);
            Tq            = SIMD_FNMADD(KP695895486, Tp, Tj);
            TI            = SIMD_FNMADD(KP797473388, TH, TC);
            Tm            = SIMD_FNMADD(KP797473388, Tl, Ta);
            TN            = SIMD_FNMADD(KP797473388, TM, TA);
            Tw            = SIMD_FMADD(KP797473388, Tv, Tg);
            TS            = SIMD_FMADD(KP797473388, TR, TE);
            Tr            = SIMD_FNMADD(KP797473388, Tq, T4);
            TJ            = SIMD_FMADD(KP867767478, TI, TB);
            Tn            = SIMD_FMADD(KP867767478, Tm, T7);
            TO            = SIMD_FMADD(KP867767478, TN, TD);
            Tx            = SIMD_FNMADD(KP867767478, Tw, Tj);
            TT            = SIMD_FNMADD(KP867767478, TS, TF);
            Ts            = SIMD_FMADD(KP867767478, Tr, Td);
            TK            = SIMD_FNMADD(KP924138961, TJ, TA);
            To            = SIMD_FNMADD(KP924138961, Tn, T4);
            TP            = SIMD_FNMADD(KP924138961, TO, TE);
            Ty            = SIMD_FNMADD(KP924138961, Tx, Ta);
            TU            = SIMD_FNMADD(KP924138961, TT, TC);
            Tt            = SIMD_FNMADD(KP924138961, Ts, Tg);
            io[0]         = SIMD_FMADD(KP974927912, TK, Tz);
            ro[0]         = SIMD_FMADD(KP974927912, To, T1);
            io[WS(os, 1)] = SIMD_FMADD(KP974927912, TP, Tz);
            ro[WS(os, 2)] = SIMD_FMADD(KP974927912, Ty, T1);
            io[WS(os, 2)] = SIMD_FMADD(KP974927912, TU, Tz);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, Tt, T1);
        }
    }
}

template <>
struct c2cb_traits<7, 5>
{
    static constexpr long_t flops           = 60;
    static constexpr long_t operations      = 36;
    static constexpr long_t memory_accesses = 18;
    static constexpr long_t stack_vars      = 52;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 5 -name c2cb
 * -standalone */

/*
 * This function contains 36 FP additions, 24 FP multiplications,
 * (or, 12 additions, 0 multiplications, 24 fused multiply/add),
 * 52 stack variables, 6 constants, and 18 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 5>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP924138961, +0.924138961091093314542963749712412773983469263);
    DK(KP867767478, +0.867767478235116240951536665696717509219981456);
    DK(KP797473388, +0.797473388882403961415688254214434877743320657);
    DK(KP695895486, +0.695895486700943376465352387651397991244687582);
    DK(KP512858431, +0.512858431636276949746649808137724830241051169);
    {
        SIMD_FLOAT T1, Tu, T2, T3, Th, Ti, T5, T6, Te, Tf, Tb, Tc, T8, T9;
        T1 = ri[0];
        Tu = ii[0];
        T2 = ii[WS(is, 6)];
        T3 = ii[WS(is, 1)];
        Th = ri[WS(is, 1)];
        Ti = ri[WS(is, 6)];
        T5 = ri[WS(is, 2)];
        T6 = ri[WS(is, 5)];
        Te = ii[WS(is, 5)];
        Tf = ii[WS(is, 2)];
        Tb = ri[WS(is, 3)];
        Tc = ri[WS(is, 4)];
        T8 = ii[WS(is, 4)];
        T9 = ii[WS(is, 3)];
        {
            SIMD_FLOAT TA, T4, Tj, Tv, T7, Tz, Tg, Tw, Td, Tx, Ta, Ty, TB, Tk,
                TG;
            SIMD_FLOAT Tp, TC, Tl, TH, Tq, TD, Tm, TI, Tr, TE, Tn, TJ, Ts, TF,
                To;
            SIMD_FLOAT TK, Tt;
            TA            = T3 + T2;
            T4            = T2 - T3;
            Tj            = Th + Ti;
            Tv            = Th - Ti;
            T7            = T5 + T6;
            Tz            = T5 - T6;
            Tg            = Te - Tf;
            Tw            = Te + Tf;
            Td            = Tb + Tc;
            Tx            = Tb - Tc;
            Ta            = T8 - T9;
            Ty            = T8 + T9;
            TB            = SIMD_FMADD(KP512858431, TA, Tz);
            Tk            = SIMD_FMADD(KP512858431, Tj, Tg);
            TG            = SIMD_FNMADD(KP512858431, Tw, Tx);
            Tp            = SIMD_FNMADD(KP512858431, T7, Ta);
            TC            = SIMD_FNMADD(KP695895486, TB, Ty);
            Tl            = SIMD_FNMADD(KP695895486, Tk, Td);
            TH            = SIMD_FMADD(KP695895486, TG, TA);
            Tq            = SIMD_FMADD(KP695895486, Tp, Tj);
            TD            = SIMD_FNMADD(KP797473388, TC, Tx);
            Tm            = SIMD_FNMADD(KP797473388, Tl, Ta);
            TI            = SIMD_FMADD(KP797473388, TH, Tv);
            Tr            = SIMD_FMADD(KP797473388, Tq, T4);
            TE            = SIMD_FMADD(KP867767478, TD, Tw);
            Tn            = SIMD_FMADD(KP867767478, Tm, T7);
            TJ            = SIMD_FNMADD(KP867767478, TI, Ty);
            Ts            = SIMD_FNMADD(KP867767478, Tr, Td);
            TF            = SIMD_FNMADD(KP924138961, TE, Tv);
            To            = SIMD_FNMADD(KP924138961, Tn, T4);
            TK            = SIMD_FNMADD(KP924138961, TJ, Tz);
            Tt            = SIMD_FNMADD(KP924138961, Ts, Tg);
            io[0]         = SIMD_FMADD(KP974927912, TF, Tu);
            ro[0]         = SIMD_FMADD(KP974927912, To, T1);
            io[WS(os, 1)] = SIMD_FMADD(KP974927912, TK, Tu);
            ro[WS(os, 1)] = SIMD_FMADD(KP974927912, Tt, T1);
        }
    }
}

template <>
struct c2cb_traits<7, 6>
{
    static constexpr long_t flops           = 38;
    static constexpr long_t operations      = 24;
    static constexpr long_t memory_accesses = 16;
    static constexpr long_t stack_vars      = 40;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_notwb.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -skip-k 6 -name c2cb
 * -standalone */

/*
 * This function contains 24 FP additions, 14 FP multiplications,
 * (or, 10 additions, 0 multiplications, 14 fused multiply/add),
 * 40 stack variables, 6 constants, and 16 memory accesses
 */
template <long_t TransformSize, long_t SkippedOutputs, long_t is, long_t os>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && SkippedOutputs == 6>::type
c2cb(SIMD_FLOAT const* __restrict ii, SIMD_FLOAT const* __restrict ri,
     SIMD_FLOAT* __restrict io, SIMD_FLOAT* __restrict ro)
{
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP924138961, +0.924138961091093314542963749712412773983469263);
    DK(KP797473388, +0.797473388882403961415688254214434877743320657);
    DK(KP695895486, +0.695895486700943376465352387651397991244687582);
    DK(KP512858431, +0.512858431636276949746649808137724830241051169);
    DK(KP867767478, +0.867767478235116240951536665696717509219981456);
    {
        SIMD_FLOAT T1, To, T2, T3, T7, T9, T5, T6, Ta, Tb, Tg, Th, Td, Te, Tt;
        SIMD_FLOAT T4, Tr, T8, Ts, Tc, Ti, Tp, Tq, Tf, Tu, Tj, Tv, Tk, Tw, Tl;
        SIMD_FLOAT Tx, Tm, Ty, Tn;
        T1    = ri[0];
        To    = ii[0];
        T2    = ii[WS(is, 5)];
        T3    = ii[WS(is, 2)];
        T7    = ii[WS(is, 1)];
        T9    = ii[WS(is, 6)];
        T5    = ri[WS(is, 4)];
        T6    = ri[WS(is, 3)];
        Ta    = ri[WS(is, 1)];
        Tb    = ri[WS(is, 6)];
        Tg    = ri[WS(is, 2)];
        Th    = ri[WS(is, 5)];
        Td    = ii[WS(is, 4)];
        Te    = ii[WS(is, 3)];
        Tt    = T2 + T3;
        T4    = T2 - T3;
        Tr    = T7 + T9;
        T8    = T5 + (SIMD_FMADD(KP867767478, T7, T6));
        Ts    = T6 - T5;
        Tc    = Ta + Tb;
        Ti    = Tg + Th;
        Tp    = Tg - Th;
        Tq    = (SIMD_FMADD(KP867767478, Tb, Td)) + Te;
        Tf    = Td - Te;
        Tu    = SIMD_FNMADD(KP512858431, Tt, Ts);
        Tj    = SIMD_FNMADD(KP512858431, Ti, Tf);
        Tv    = SIMD_FMADD(KP695895486, Tu, Tr);
        Tk    = SIMD_FMADD(KP695895486, Tj, Tc);
        Tw    = SIMD_FMADD(KP797473388, Tv, Ta);
        Tl    = SIMD_FMADD(KP797473388, Tk, T9);
        Tx    = SIMD_FNMADD(KP867767478, Tw, Tq);
        Tm    = SIMD_FNMADD(KP867767478, Tl, T8);
        Ty    = SIMD_FNMADD(KP924138961, Tx, Tp);
        Tn    = SIMD_FNMADD(KP924138961, Tm, T4);
        io[0] = SIMD_FMADD(KP974927912, Ty, To);
        ro[0] = SIMD_FMADD(KP974927912, Tn, T1);
    }
}