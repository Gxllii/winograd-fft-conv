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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>

namespace zi2::vl
{

struct load_tag
{
};

namespace
{
load_tag load;
}

inline void ____use_load_tag() { static_cast<void>(load); }

template <class T, std::size_t N>
class vec : public std::array<T, N>
{
private:
    static_assert(N > 0, "vec<T,N> needs N > 0");
    static_assert(std::is_scalar_v<T>, "vec<T,N> needs scalar T");

    using type = vec<T, N>;
    using base = std::array<T, N>;

    using size_type       = typename base::size_type;
    using reference       = typename base::reference;
    using const_reference = typename base::const_reference;

public:
    template <typename V>
    constexpr vec(std::initializer_list<V> v) noexcept
    {
        static_assert(std::is_convertible_v<V, T>);
        size_type i = 0;
        for (auto x : v)
        {
            base::data()[i++] = x;
        }
        for (; i < N; ++i)
        {
            base::data()[i] = T();
        }
    }

    explicit constexpr vec(T const& v = T()) noexcept
    {
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = v;
        }
    }

    template <typename V>
    explicit constexpr vec(V const& v = V()) noexcept
    {
        static_assert(std::is_convertible_v<V, T>);
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = static_cast<T>(v);
        }
    }

    constexpr vec(vec<T, N> const& rcp) noexcept
    {
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = rcp[i];
        }
    }

    template <class V>
    constexpr vec(vec<V, N> const& rcp) noexcept
    {
        static_assert(std::is_convertible_v<V, T>);
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = static_cast<T>(rcp[i]);
        }
    }

    template <class T1, std::size_t N1, class T2, std::size_t N2>
    constexpr vec(vec<T1, N1> const& v1, vec<T2, N2> const& v2) noexcept
    {
        static_assert(std::is_convertible_v<T1, T>);
        static_assert(std::is_convertible_v<T2, T>);
        static_assert(N1 + N2 == N);

        size_type i = 0;
        for (size_type j = 0; j < N1; ++i, ++j)
        {
            base::data()[i] = static_cast<T>(v1[i]);
        }

        for (size_type j = 0; j < N2; ++i, ++j)
        {
            base::data()[i] = static_cast<T>(v2[j]);
        }
    }

    template <typename O>
    vec(load_tag, O const* data,
        typename std::enable_if<std::is_convertible<O, T>::value, void*>::type =
            0)
    {
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = data[i];
        }
    }

public:
// Some accessors
#define ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(name, idx)                          \
    constexpr const_reference name() const                                     \
    {                                                                          \
        static_assert(idx < N);                                                \
        return base::data()[idx];                                              \
    }                                                                          \
    constexpr reference name()                                                 \
    {                                                                          \
        static_assert(idx < N);                                                \
        return base::data()[idx];                                              \
    }

    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(x, 0)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(y, 1)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(z, 2)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(w, 3)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(t, 3)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(r, 0)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(g, 1)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(b, 2)
    ZI2_VL_NAMED_INDEX_ACCESSOR_MEMBER(a, 3)

    constexpr reference elem(size_type i) { return base::data()[i]; }

    constexpr const_reference elem(size_type i) const
    {
        return base::data()[i];
    }

    constexpr T sum() const
    {
        T r = base::data()[0];
        for (size_type i = 1; i < N; ++i)
        {
            r += base::data()[i];
        }
        return r;
    }

    constexpr T prod() const
    {
        T r = base::data()[0];
        for (size_type i = 1; i < N; ++i)
        {
            r *= base::data()[i];
        }
        return r;
    }

    // Min Max Elements
    // ---------------------------------------------------------------------

    constexpr T const& min() const
    {
        return *::std::min_element(base::begin(), base::end());
    }

    constexpr size_type min_index() const
    {
        return ::std::min_element(base::begin(), base::end()) - base::begin();
    }

    constexpr T const& max() const
    {
        return *::std::max_element(base::begin(), base::end());
    }

    constexpr size_type max_index() const
    {
        return ::std::max_element(base::begin(), base::end()) - base::begin();
    }

    constexpr T absmax() const
    {
        T m = -type::min();
        return ::std::max(type::max(), m);
    }

    constexpr size_type absmax_index() const
    {
        size_type mini = type::min_index();
        size_type maxi = type::max_index();
        return (-base::data()[mini] > base::data()[maxi]) ? mini : maxi;
    }

    // Assignments
    // ---------------------------------------------------------------------

    template <class O>
    vec<T, N>& operator=(vec<O, N> const& rhs)
    {
        static_assert(std::is_convertible_v<O, T>);
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = static_cast<T>(rhs[i]);
        }
        return *this;
    }

    vec<T, N>& operator=(vec<T, N> const& rhs)
    {
        for (size_type i = 0; i < N; ++i)
        {
            base::data()[i] = rhs[i];
        }
        return *this;
    }

    template <class O>
    vec<T, N>& operator=(O const& rhs)
    {
        static_assert(std::is_convertible_v<O, T>);
        std::fill_n(base::begin(), N, static_cast<T>(rhs));
        return *this;
    }

    vec<T, N>& operator=(T const& rhs)
    {
        std::fill_n(base::begin(), N, rhs);
        return *this;
    }

    void fill(T const& val) { std::fill_n(base::begin(), N, val); }

    void assign(T const& val) { std::fill_n(base::begin(), N, val); }

#define ZI2_VL_COMPOUND_OPERATOR_SCALAR(op)                                    \
                                                                               \
    vec<T, N>& operator op(T const& rhs)                                       \
    {                                                                          \
        for (size_type i = 0; i < N; ++i)                                      \
        {                                                                      \
            base::data()[i] op rhs;                                            \
        }                                                                      \
        return *this;                                                          \
    }

#define ZI2_VL_COMPOUND_OPERATOR_VECTOR(op)                                    \
                                                                               \
    template <class X>                                                         \
    vec<T, N>& operator op(vec<X, N> const& rhs)                               \
    {                                                                          \
        for (size_type i = 0; i < N; ++i)                                      \
        {                                                                      \
            base::data()[i] op rhs[i];                                         \
        }                                                                      \
        return *this;                                                          \
    }                                                                          \
                                                                               \
    vec<T, N>& operator op(vec<T, N> const& rhs)                               \
    {                                                                          \
        for (size_type i = 0; i < N; ++i)                                      \
        {                                                                      \
            base::data()[i] op rhs[i];                                         \
        }                                                                      \
        return *this;                                                          \
    }

#define ZI2_VL_COMPOUND_OPERATOR(op)                                           \
    ZI2_VL_COMPOUND_OPERATOR_SCALAR(op) ZI2_VL_COMPOUND_OPERATOR_VECTOR(op)

    ZI2_VL_COMPOUND_OPERATOR(+=)
    ZI2_VL_COMPOUND_OPERATOR(-=)
    ZI2_VL_COMPOUND_OPERATOR(*=)
    ZI2_VL_COMPOUND_OPERATOR(%=)
    ZI2_VL_COMPOUND_OPERATOR(/=)

#undef ZI2_VL_COMPOUND_OPERATOR
#undef ZI2_VL_COMPOUND_OPERATOR_SCALAR
#undef ZI2_VL_COMPOUND_OPERATOR_VECTOR

    void swap(vec<T, N>& rhs)
    {
        for (size_type i = 0; i < N; ++i)
        {
            ::std::swap(base::data()[i], rhs[i]);
        }
    }

    static constexpr vec<T, N> one  = vec<T, N>(1);
    static constexpr vec<T, N> zero = vec<T, N>(0);
};

#define ZI2_VL_VECTORDEF_VEC_VECTOR(len)                                       \
    typedef vec<int, len>         vec##len##i;                                 \
    typedef vec<long, len>        vec##len##l;                                 \
    typedef vec<long long, len>   vec##len##ll;                                \
    typedef vec<float, len>       vec##len##f;                                 \
    typedef vec<double, len>      vec##len##d;                                 \
    typedef vec<long double, len> vec##len##ld

ZI2_VL_VECTORDEF_VEC_VECTOR(1);
ZI2_VL_VECTORDEF_VEC_VECTOR(2);
ZI2_VL_VECTORDEF_VEC_VECTOR(3);
ZI2_VL_VECTORDEF_VEC_VECTOR(4);
ZI2_VL_VECTORDEF_VEC_VECTOR(5);
ZI2_VL_VECTORDEF_VEC_VECTOR(6);
ZI2_VL_VECTORDEF_VEC_VECTOR(7);
ZI2_VL_VECTORDEF_VEC_VECTOR(8);
ZI2_VL_VECTORDEF_VEC_VECTOR(9);
ZI2_VL_VECTORDEF_VEC_VECTOR(10);
ZI2_VL_VECTORDEF_VEC_VECTOR(11);
ZI2_VL_VECTORDEF_VEC_VECTOR(12);
ZI2_VL_VECTORDEF_VEC_VECTOR(13);
ZI2_VL_VECTORDEF_VEC_VECTOR(14);
ZI2_VL_VECTORDEF_VEC_VECTOR(15);
ZI2_VL_VECTORDEF_VEC_VECTOR(16);
ZI2_VL_VECTORDEF_VEC_VECTOR(17);
ZI2_VL_VECTORDEF_VEC_VECTOR(18);
ZI2_VL_VECTORDEF_VEC_VECTOR(19);
ZI2_VL_VECTORDEF_VEC_VECTOR(20);

#undef ZI2_VL_VECTORDEF_VEC_VECTOR

template <class T, std::size_t N>
inline constexpr bool operator==(vec<T, N> const& a, vec<T, N> const& b)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

template <class T, class O, std::size_t N>
inline constexpr bool operator==(vec<T, N> const& a, vec<O, N> const& b)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

template <class T, std::size_t N>
inline constexpr bool operator!=(vec<T, N> const& a, vec<T, N> const& b)
{
    return !(a == b);
}

template <class T, class O, std::size_t N>
inline constexpr bool operator!=(vec<T, N> const& a, vec<O, N> const& b)
{
    return !(a == b);
}

template <class T, std::size_t N>
inline constexpr bool operator<(vec<T, N> const& a, vec<T, N> const& b)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if (a[i] < b[i])
            return true;
        if (a[i] > b[i])
            return false;
    }
    return false;
}

template <class T, class O, std::size_t N>
inline constexpr bool operator<(vec<T, N> const& a, vec<O, N> const& b)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if (a[i] < b[i])
            return true;
        if (a[i] > b[i])
            return false;
    }
    return false;
}

template <class T, std::size_t N>
inline constexpr bool operator>(vec<T, N> const& a, vec<T, N> const& b)
{
    return (b < a);
}

template <class T, class O, std::size_t N>
inline constexpr bool operator>(vec<T, N> const& a, vec<O, N> const& b)
{
    return (b < a);
}

template <class T, std::size_t N>
inline constexpr bool operator>=(vec<T, N> const& a, vec<T, N> const& b)
{
    return !(a < b);
}

template <class T, class O, std::size_t N>
inline constexpr bool operator>=(vec<T, N> const& a, vec<O, N> const& b)
{
    return !(a < b);
}

template <class T, std::size_t N>
inline constexpr bool operator<=(vec<T, N> const& a, vec<T, N> const& b)
{
    return !(b < a);
}

template <class T, class O, std::size_t N>
inline constexpr bool operator<=(vec<T, N> const& a, vec<O, N> const& b)
{
    return !(b < a);
}

template <class T, std::size_t N, class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, vec<T, N> const& v)
{
    os << v[0];
    for (std::size_t i = 1; i < N; ++i)
    {
        os << "," << v[i];
    }
    return os;
}

    //
    // Basic arithmetic
    // ---------------------------------------------------------------------
    //

#define ZI2_VL_INLINE_BINARY_OPERATOR(op, name)                                \
                                                                               \
    template <class T, std::size_t N, std::size_t... Ints>                     \
    inline constexpr vec<T, N> ____##name##_helper1(                           \
        T const& lhs, vec<T, N> const& rhs, std::index_sequence<Ints...>)      \
    {                                                                          \
        return {(lhs op rhs.elem(Ints))...};                                   \
    }                                                                          \
                                                                               \
    template <class T1, class T2, std::size_t N, std::size_t... Ints>          \
    constexpr vec<decltype(std::declval<T1>() op std::declval<T2>()), N>       \
        ____##name##_helper2(T1 const& lhs, vec<T2, N> const& rhs,             \
                             std::index_sequence<Ints...>,                     \
                             std::enable_if_t<std::is_scalar_v<T1>, void*> =   \
                                 0)                                            \
    {                                                                          \
        return {(lhs op rhs.elem(Ints))...};                                   \
    }                                                                          \
                                                                               \
    template <class T, std::size_t N>                                          \
    inline constexpr auto operator op(T const& lhs, vec<T, N> const& rhs)      \
    {                                                                          \
        return ____##name##_helper1(lhs, rhs, std::make_index_sequence<N>());  \
    }                                                                          \
                                                                               \
    template <class O, class T, std::size_t N>                                 \
    inline constexpr std::enable_if_t<                                         \
        std::is_scalar_v<T>,                                                   \
        vec<decltype(std::declval<O>() op std::declval<T>()), N>>              \
    operator op(T const& lhs, vec<O, N> const& rhs)                            \
    {                                                                          \
        return ____##name##_helper2(lhs, rhs, std::make_index_sequence<N>());  \
    }                                                                          \
                                                                               \
    template <class T, std::size_t N, std::size_t... Ints>                     \
    inline constexpr vec<T, N> ____##name##_helper1(                           \
        vec<T, N> const& lhs, T const& rhs, std::index_sequence<Ints...>)      \
    {                                                                          \
        return {(lhs.elem(Ints) op rhs)...};                                   \
    }                                                                          \
                                                                               \
    template <class T1, class T2, std::size_t N, std::size_t... Ints>          \
    constexpr auto ____##name##_helper2(                                       \
        vec<T1, N> const& lhs, T2 const& rhs, std::index_sequence<Ints...>,    \
        std::enable_if_t<std::is_scalar_v<T2>, void*> = 0)                     \
        ->vec<decltype(std::declval<T1>() op std::declval<T2>()), N>           \
    {                                                                          \
        return {(lhs.elem(Ints) op rhs)...};                                   \
    }                                                                          \
                                                                               \
    template <class T, std::size_t N>                                          \
    inline constexpr auto operator op(vec<T, N> const& lhs, T const& rhs)      \
    {                                                                          \
        return ____##name##_helper1(lhs, rhs, std::make_index_sequence<N>());  \
    }                                                                          \
                                                                               \
    template <class T, class O, std::size_t N>                                 \
    inline constexpr std::enable_if_t<                                         \
        std::is_scalar_v<O>,                                                   \
        vec<decltype(std::declval<T>() op std::declval<O>()), N>>              \
    operator op(vec<T, N> const& lhs, O const& rhs)                            \
    {                                                                          \
        return ____##name##_helper2(lhs, rhs, std::make_index_sequence<N>());  \
    }                                                                          \
                                                                               \
    template <class T, std::size_t N, std::size_t... Ints>                     \
    inline constexpr vec<T, N> ____##name##_helper1(                           \
        vec<T, N> const& lhs, vec<T, N> const& rhs,                            \
        std::index_sequence<Ints...>)                                          \
    {                                                                          \
        return {(lhs.elem(Ints) op rhs.elem(Ints))...};                        \
    }                                                                          \
                                                                               \
    template <class T1, class T2, std::size_t N, std::size_t... Ints>          \
    constexpr auto ____##name##_helper2(vec<T1, N> const& lhs,                 \
                                        vec<T2, N> const& rhs,                 \
                                        std::index_sequence<Ints...>)          \
        ->vec<decltype(std::declval<T1>() op std::declval<T2>()), N>           \
    {                                                                          \
        return {(lhs.elem(Ints) op rhs.elem(Ints))...};                        \
    }                                                                          \
                                                                               \
    template <class T, std::size_t N>                                          \
    inline constexpr auto operator op(vec<T, N> const& lhs,                    \
                                      vec<T, N> const& rhs)                    \
    {                                                                          \
        return ____##name##_helper1(lhs, rhs, std::make_index_sequence<N>());  \
    }                                                                          \
                                                                               \
    template <class T, class O, std::size_t N>                                 \
    inline constexpr auto operator op(vec<T, N> const& lhs,                    \
                                      vec<O, N> const& rhs)                    \
    {                                                                          \
        return ____##name##_helper2(lhs, rhs, std::make_index_sequence<N>());  \
    }

ZI2_VL_INLINE_BINARY_OPERATOR(+, plus)
ZI2_VL_INLINE_BINARY_OPERATOR(-, minus)
ZI2_VL_INLINE_BINARY_OPERATOR(*, times)
ZI2_VL_INLINE_BINARY_OPERATOR(/, divs)
ZI2_VL_INLINE_BINARY_OPERATOR(%, percent)

#undef ZI2_VL_INLINE_BINARY_OPERATOR

template <class T, std::size_t N>
inline constexpr vec<T, N> operator+(vec<T, N> const& rhs)
{
    return rhs;
}

template <class T, std::size_t N, std::size_t... Ints>
inline constexpr vec<T, N> ____unary_minus_helper(vec<T, N> const& rhs,
                                                  std::index_sequence<Ints...>)
{
    return {(-rhs.elem(Ints))...};
}

template <class T, std::size_t N>
inline constexpr auto operator-(vec<T, N> const& rhs)
{
    return ____unary_minus_helper(rhs, std::make_index_sequence<N>());
}

template <std::size_t First, std::size_t Len, class T, std::size_t N,
          std::size_t... Ints>
inline constexpr vec<T, Len> ____subvec_helper(vec<T, N> const& v,
                                               std::index_sequence<Ints...>)
{
    return {v.elem(First + Ints)...};
}

template <std::size_t First, std::size_t Len, class T, std::size_t N>
inline constexpr auto subvec(vec<T, N> const& v)
{
    static_assert(First < N);
    static_assert(First + Len <= N);
    return ____subvec_helper<First, Len>(v, std::make_index_sequence<Len>());
}

template <class T, std::size_t N>
inline constexpr T dot(vec<T, N> const& a, vec<T, N> const& b)
{
    T r = a[0] * b[0];
    for (std::size_t i = 1; i < N; ++i)
    {
        r += a[i] * b[i];
    }
    return r;
}

// Some functions

template <class R, class T, std::size_t N>
inline constexpr R sqrlen(vec<T, N> const& v)
{
    R r = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        r += static_cast<R>(v[i]) * v[i];
    }
    return r;
}

template <class T, std::size_t N>
inline constexpr vec<T, N + 1> operator,(T const& v, vec<T, N> const& vr)
{
    return vec<T, N + 1>(vec<T, 1>(v), vr);
}

template <class T, std::size_t N>
inline constexpr vec<T, N + 1> operator,(vec<T, N> const& vl, T const& v)
{
    return vec<T, N + 1>(vl, vec<T, 1>(v));
}

template <class T, std::size_t N1, std::size_t N2>
inline constexpr vec<T, N1 + N2> operator,(vec<T, N1> const& v1,
                                           vec<T, N2> const& v2)
{
    return vec<T, N1 + N2>(v1, v2);
}

template <class T, T... Ints>
struct vec_type
{
    static constexpr vec<T, sizeof...(Ints)> value = {Ints...};
};

} // namespace zi2::vl
