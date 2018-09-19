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

#include "znn/types.hpp"
#include <x86intrin.h>

#if defined(ZNN_AVX512)

#define SIMD_WIDTH 16
#define SIMD_FLOAT __m512

#define SIMD_MUL _mm512_mul_ps
#define SIMD_ADD _mm512_add_ps
#define SIMD_SUB _mm512_sub_ps

#define SIMD_FMADD _mm512_fmadd_ps
#define SIMD_FMSUB _mm512_fmsub_ps
#define SIMD_FNMADD _mm512_fnmadd_ps
#define SIMD_FNMSUB _mm512_fnmsub_ps

#define SIMD_SET1 _mm512_set1_ps
#define SIMD_SET1_CONST(v)                                                     \
    {                                                                          \
        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v                         \
    }

#define SIMD_LOAD _mm512_load_ps
#define SIMD_STORE _mm512_store_ps
#define SIMD_STREAM _mm512_stream_ps
#define SIMD_ZERO _mm512_setzero_ps

#elif defined(ZNN_AVX2)

#define SIMD_WIDTH 16

struct SIMD_FLOAT
{
    __m256 a;
    __m256 b;
};

#define SIMD_MUL(x, y)                                                         \
    SIMD_FLOAT { _mm256_mul_ps(x.a, y.a), _mm256_mul_ps(x.b, y.b) }

#define SIMD_ADD(x, y)                                                         \
    SIMD_FLOAT { _mm256_add_ps(x.a, y.a), _mm256_add_ps(x.b, y.b) }

#define SIMD_SUB(x, y)                                                         \
    SIMD_FLOAT { _mm256_sub_ps(x.a, y.a), _mm256_sub_ps(x.b, y.b) }

__attribute__((always_inline)) inline SIMD_FLOAT
SIMD_FMADD(SIMD_FLOAT const& x, SIMD_FLOAT const& y, SIMD_FLOAT const& z)
{
    auto F = _mm256_fmadd_ps(x.a, y.a, z.a);
    auto S = _mm256_fmadd_ps(x.b, y.b, z.b);
    return SIMD_FLOAT{F, S};
}

__attribute__((always_inline)) inline SIMD_FLOAT
SIMD_FNMADD(SIMD_FLOAT const& x, SIMD_FLOAT const& y, SIMD_FLOAT const& z)
{
    auto F = _mm256_fnmadd_ps(x.a, y.a, z.a);
    auto S = _mm256_fnmadd_ps(x.b, y.b, z.b);
    return SIMD_FLOAT{F, S};
}

__attribute__((always_inline)) inline SIMD_FLOAT
SIMD_FMSUB(SIMD_FLOAT const& x, SIMD_FLOAT const& y, SIMD_FLOAT const& z)
{
    auto F = _mm256_fmsub_ps(x.a, y.a, z.a);
    auto S = _mm256_fmsub_ps(x.b, y.b, z.b);
    return SIMD_FLOAT{F, S};
}

__attribute__((always_inline)) inline SIMD_FLOAT
SIMD_FNMSUB(SIMD_FLOAT const& x, SIMD_FLOAT const& y, SIMD_FLOAT const& z)
{
    auto F = _mm256_fnmsub_ps(x.a, y.a, z.a);
    auto S = _mm256_fnmsub_ps(x.b, y.b, z.b);
    return SIMD_FLOAT{F, S};
}

#define SIMD_SET1(v)                                                           \
    SIMD_FLOAT { _mm256_set1_ps(v), _mm256_set1_ps(v) }

#define SIMD_SET1_CONST(v)                                                     \
    SIMD_FLOAT                                                                 \
    {                                                                          \
        {v, v, v, v, v, v, v, v}, { v, v, v, v, v, v, v, v }                   \
    }

__attribute__((always_inline)) inline SIMD_FLOAT SIMD_LOAD(float const* ptr)
{
    return SIMD_FLOAT{_mm256_load_ps(ptr), _mm256_load_ps(ptr + 8)};
}

__attribute__((always_inline)) inline void SIMD_STORE(float* dest, SIMD_FLOAT v)
{
    _mm256_store_ps(dest, v.a);
    _mm256_store_ps(dest + 8, v.b);
}

__attribute__((always_inline)) inline void SIMD_STREAM(float*     dest,
                                                       SIMD_FLOAT v)
{
    _mm256_stream_ps(dest, v.a);
    _mm256_stream_ps(dest + 8, v.b);
}

#define SIMD_ZERO()                                                            \
    SIMD_FLOAT { _mm256_setzero_ps(), _mm256_setzero_ps() }

__attribute__((always_inline)) inline SIMD_FLOAT operator+(SIMD_FLOAT const& a,
                                                           SIMD_FLOAT const& b)
{
    return SIMD_ADD(a, b);
}

__attribute__((always_inline)) inline SIMD_FLOAT operator-(SIMD_FLOAT const& a,
                                                           SIMD_FLOAT const& b)
{
    return SIMD_SUB(a, b);
}

__attribute__((always_inline)) inline SIMD_FLOAT operator-(SIMD_FLOAT const& a)
{
    return SIMD_SET1_CONST(0.f) - a;
}

__attribute__((always_inline)) inline SIMD_FLOAT operator*(SIMD_FLOAT const& a,
                                                           SIMD_FLOAT const& b)
{
    return SIMD_MUL(a, b);
}

#else

#error "THIS CODE REQUIRES SOME AVX2 || AVX512"

#endif

#define SIMD_PREFETCH_L1(address)                                              \
    _mm_prefetch(reinterpret_cast<char const*>(address), _MM_HINT_T0)

#define SIMD_PREFETCH_L2(address)                                              \
    _mm_prefetch(reinterpret_cast<char const*>(address), _MM_HINT_T1)

#define CACHELINE_SIZE 16
