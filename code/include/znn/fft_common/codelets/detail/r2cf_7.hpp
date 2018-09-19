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
struct r2cf_traits<7, 1>
{
    static constexpr long_t flops           = 0;
    static constexpr long_t operations      = 0;
    static constexpr long_t memory_accesses = 5;
    static constexpr long_t stack_vars      = 1;
    static constexpr long_t constants       = 0;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 1 -name r2cf
 * -standalone */

/*
 * This function contains 0 FP additions, 0 FP multiplications,
 * (or, 0 additions, 0 multiplications, 0 fused multiply/add),
 * 1 stack variables, 0 constants, and 5 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 1>::type
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
struct r2cf_traits<7, 2>
{
    static constexpr long_t flops           = 10;
    static constexpr long_t operations      = 7;
    static constexpr long_t memory_accesses = 9;
    static constexpr long_t stack_vars      = 8;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 2 -name r2cf
 * -standalone */

/*
 * This function contains 4 FP additions, 6 FP multiplications,
 * (or, 1 additions, 3 multiplications, 3 fused multiply/add),
 * 8 stack variables, 6 constants, and 9 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 2>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP623489801, +0.623489801858733530525004884004239810632274731);
    DK(KP222520933, +0.222520933956314404288902564496794759466355569);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP433883739, +0.433883739117558120475768332848358754609990728);
    DK(KP781831482, +0.781831482468029808708444526674057750232334519);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    {
        SIMD_FLOAT T1, T2;
        T1            = R0[WS(rs, 1)];
        T2            = R0[0];
        Ci[WS(cs, 2)] = -(KP974927912 * T1);
        Ci[WS(cs, 1)] = -(KP781831482 * T1);
        Ci[WS(cs, 3)] = -(KP433883739 * T1);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP900968867, T1, T2);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP222520933, T1, T2);
        Cr[0]         = T2 + T1;
        Cr[WS(cs, 1)] = SIMD_FMADD(KP623489801, T1, T2);
    }
}

template <>
struct r2cf_traits<7, 3>
{
    static constexpr long_t flops           = 23;
    static constexpr long_t operations      = 14;
    static constexpr long_t memory_accesses = 10;
    static constexpr long_t stack_vars      = 16;
    static constexpr long_t constants       = 10;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 3 -name r2cf
 * -standalone */

/*
 * This function contains 11 FP additions, 12 FP multiplications,
 * (or, 2 additions, 3 multiplications, 9 fused multiply/add),
 * 16 stack variables, 10 constants, and 10 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 3>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP623489801, +0.623489801858733530525004884004239810632274731);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP246979603, +0.246979603717467061050009768008479621264549462);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP445041867, +0.445041867912628808577805128993589518932711138);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP781831482, +0.781831482468029808708444526674057750232334519);
    {
        SIMD_FLOAT T1, T2, T3, T5, T6, T4;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        T3            = R0[WS(rs, 2)];
        Cr[0]         = T1 + T2 + T3;
        Ci[WS(cs, 3)] = KP781831482 * (SIMD_FNMADD(KP554958132, T2, T3));
        Ci[WS(cs, 2)] = KP974927912 * (SIMD_FMSUB(KP445041867, T3, T2));
        Ci[WS(cs, 1)] = -(KP974927912 * (SIMD_FMADD(KP801937735, T2, T3)));
        T5            = SIMD_FMSUB(KP692021471, T3, T2);
        T6            = SIMD_FMADD(KP246979603, T2, T3);
        T4            = SIMD_FNMADD(KP356895867, T3, T2);
        Cr[WS(cs, 3)] = SIMD_FMADD(KP900968867, T5, T1);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP900968867, T6, T1);
        Cr[WS(cs, 1)] = SIMD_FMADD(KP623489801, T4, T1);
    }
}

template <>
struct r2cf_traits<7, 4>
{
    static constexpr long_t flops           = 36;
    static constexpr long_t operations      = 21;
    static constexpr long_t memory_accesses = 11;
    static constexpr long_t stack_vars      = 19;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 4 -name r2cf
 * -standalone */

/*
 * This function contains 18 FP additions, 18 FP multiplications,
 * (or, 3 additions, 3 multiplications, 15 fused multiply/add),
 * 19 stack variables, 6 constants, and 11 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 4>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, T2, T4, T3, T7, T8, Ta, T5, Td, Tb, T9, Tc, T6;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        T4            = R0[WS(rs, 3)];
        T3            = R0[WS(rs, 2)];
        T7            = SIMD_FMADD(KP554958132, T4, T2);
        T8            = SIMD_FNMADD(KP356895867, T2, T4);
        Cr[0]         = T1 + T2 + T3 + T4;
        Ta            = SIMD_FMADD(KP554958132, T3, T4);
        T5            = SIMD_FNMADD(KP356895867, T4, T3);
        Td            = SIMD_FNMADD(KP554958132, T2, T3);
        Tb            = SIMD_FNMADD(KP356895867, T3, T2);
        T9            = SIMD_FNMADD(KP692021471, T8, T3);
        Ci[WS(cs, 1)] = -(KP974927912 * (SIMD_FMADD(KP801937735, T7, T3)));
        Ci[WS(cs, 2)] = -(KP974927912 * (SIMD_FNMADD(KP801937735, Ta, T2)));
        Tc            = SIMD_FNMADD(KP692021471, Tb, T4);
        Ci[WS(cs, 3)] = -(KP974927912 * (SIMD_FNMADD(KP801937735, Td, T4)));
        T6            = SIMD_FNMADD(KP692021471, T5, T2);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP900968867, T9, T1);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP900968867, Tc, T1);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP900968867, T6, T1);
    }
}

template <>
struct r2cf_traits<7, 5>
{
    static constexpr long_t flops           = 38;
    static constexpr long_t operations      = 23;
    static constexpr long_t memory_accesses = 12;
    static constexpr long_t stack_vars      = 22;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 5 -name r2cf
 * -standalone */

/*
 * This function contains 20 FP additions, 18 FP multiplications,
 * (or, 5 additions, 3 multiplications, 15 fused multiply/add),
 * 22 stack variables, 6 constants, and 12 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 5>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    {
        SIMD_FLOAT T1, T2, T3, T4, T5, Tb, Tg, Td, T6, Tf, Te, Tc, T9, T7, Ta;
        SIMD_FLOAT T8;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        T3            = R0[WS(rs, 2)];
        T4            = R0[WS(rs, 3)];
        T5            = R0[WS(rs, 4)];
        Tb            = SIMD_FNMADD(KP356895867, T3, T2);
        Tg            = SIMD_FNMADD(KP554958132, T2, T3);
        Td            = T5 - T4;
        T6            = T4 + T5;
        Tf            = SIMD_FMSUB(KP554958132, Td, T2);
        Te            = SIMD_FNMADD(KP554958132, T3, Td);
        Ci[WS(cs, 3)] = KP974927912 * (SIMD_FMADD(KP801937735, Tg, Td));
        Tc            = SIMD_FNMADD(KP692021471, Tb, T6);
        Cr[0]         = T1 + T2 + T3 + T6;
        T9            = SIMD_FNMADD(KP356895867, T2, T6);
        T7            = SIMD_FNMADD(KP356895867, T6, T3);
        Ci[WS(cs, 1)] = -(KP974927912 * (SIMD_FNMADD(KP801937735, Tf, T3)));
        Ci[WS(cs, 2)] = -(KP974927912 * (SIMD_FMADD(KP801937735, Te, T2)));
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP900968867, Tc, T1);
        Ta            = SIMD_FNMADD(KP692021471, T9, T3);
        T8            = SIMD_FNMADD(KP692021471, T7, T2);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP900968867, Ta, T1);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP900968867, T8, T1);
    }
}

template <>
struct r2cf_traits<7, 6>
{
    static constexpr long_t flops           = 40;
    static constexpr long_t operations      = 25;
    static constexpr long_t memory_accesses = 13;
    static constexpr long_t stack_vars      = 25;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 6 -name r2cf
 * -standalone */

/*
 * This function contains 22 FP additions, 18 FP multiplications,
 * (or, 7 additions, 3 multiplications, 15 fused multiply/add),
 * 25 stack variables, 6 constants, and 13 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 6>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, T2, T6, T7, T3, T4, T8, Tf, T5, Tg, Ti, Tb, Th, Tj, Td;
        SIMD_FLOAT T9, Tc, Te, Ta;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        T6            = R0[WS(rs, 3)];
        T7            = R0[WS(rs, 4)];
        T3            = R0[WS(rs, 2)];
        T4            = R0[WS(rs, 5)];
        T8            = T6 + T7;
        Tf            = T7 - T6;
        T5            = T3 + T4;
        Tg            = T4 - T3;
        Ti            = SIMD_FMSUB(KP554958132, Tf, T2);
        Tb            = SIMD_FNMADD(KP356895867, T2, T8);
        Th            = SIMD_FMADD(KP554958132, Tg, Tf);
        Tj            = SIMD_FMADD(KP554958132, T2, Tg);
        Cr[0]         = T1 + T2 + T5 + T8;
        Td            = SIMD_FNMADD(KP356895867, T5, T2);
        T9            = SIMD_FNMADD(KP356895867, T8, T5);
        Ci[WS(cs, 1)] = KP974927912 * (SIMD_FMADD(KP801937735, Ti, Tg));
        Tc            = SIMD_FNMADD(KP692021471, Tb, T5);
        Ci[WS(cs, 2)] = -(KP974927912 * (SIMD_FMADD(KP801937735, Th, T2)));
        Ci[WS(cs, 3)] = KP974927912 * (SIMD_FNMADD(KP801937735, Tj, Tf));
        Te            = SIMD_FNMADD(KP692021471, Td, T8);
        Ta            = SIMD_FNMADD(KP692021471, T9, T2);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP900968867, Tc, T1);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP900968867, Te, T1);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP900968867, Ta, T1);
    }
}

template <>
struct r2cf_traits<7, 7>
{
    static constexpr long_t flops           = 42;
    static constexpr long_t operations      = 27;
    static constexpr long_t memory_accesses = 14;
    static constexpr long_t stack_vars      = 28;
    static constexpr long_t constants       = 6;
};
/* Generated by: ./gen_r2cf.native -fma -reorder-insns -schedule-for-pipeline
 * -compact -variables 30 -pipeline-latency 12 -n 7 -first-k 7 -name r2cf
 * -standalone */

/*
 * This function contains 24 FP additions, 18 FP multiplications,
 * (or, 9 additions, 3 multiplications, 15 fused multiply/add),
 * 28 stack variables, 6 constants, and 14 memory accesses
 */
template <long_t TransformSize, long_t ProvidedElements, long_t rs, long_t cs>
inline __attribute__((always_inline))
typename std::enable_if<TransformSize == 7 && ProvidedElements == 7>::type
r2cf(SIMD_FLOAT const* __restrict R0, SIMD_FLOAT* __restrict Cr,
     SIMD_FLOAT* __restrict Ci)
{
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    {
        SIMD_FLOAT T1, T2, T3, T8, T9, T5, T6, Th, T4, Ti, Ta, Tj, T7, Tl, Td;
        SIMD_FLOAT Tk, Tm, Tf, Tb, Te, Tg, Tc;
        T1            = R0[0];
        T2            = R0[WS(rs, 1)];
        T3            = R0[WS(rs, 6)];
        T8            = R0[WS(rs, 3)];
        T9            = R0[WS(rs, 4)];
        T5            = R0[WS(rs, 2)];
        T6            = R0[WS(rs, 5)];
        Th            = T3 - T2;
        T4            = T2 + T3;
        Ti            = T9 - T8;
        Ta            = T8 + T9;
        Tj            = T6 - T5;
        T7            = T5 + T6;
        Tl            = SIMD_FMADD(KP554958132, Ti, Th);
        Td            = SIMD_FNMADD(KP356895867, T4, Ta);
        Tk            = SIMD_FMADD(KP554958132, Tj, Ti);
        Tm            = SIMD_FNMADD(KP554958132, Th, Tj);
        Cr[0]         = T1 + T4 + T7 + Ta;
        Tf            = SIMD_FNMADD(KP356895867, T7, T4);
        Tb            = SIMD_FNMADD(KP356895867, Ta, T7);
        Ci[WS(cs, 1)] = KP974927912 * (SIMD_FMADD(KP801937735, Tl, Tj));
        Te            = SIMD_FNMADD(KP692021471, Td, T7);
        Ci[WS(cs, 2)] = KP974927912 * (SIMD_FNMADD(KP801937735, Tk, Th));
        Ci[WS(cs, 3)] = KP974927912 * (SIMD_FNMADD(KP801937735, Tm, Ti));
        Tg            = SIMD_FNMADD(KP692021471, Tf, Ta);
        Tc            = SIMD_FNMADD(KP692021471, Tb, T4);
        Cr[WS(cs, 2)] = SIMD_FNMADD(KP900968867, Te, T1);
        Cr[WS(cs, 1)] = SIMD_FNMADD(KP900968867, Tg, T1);
        Cr[WS(cs, 3)] = SIMD_FNMADD(KP900968867, Tc, T1);
    }
}
