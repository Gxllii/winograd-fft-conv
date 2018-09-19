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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "compiler.hpp"
#include "znn/vec.hpp"

#if !defined(ZNN_CACHE_SIZE)
#define ZNN_CACHE_SIZE 512
#endif

namespace znn
{

typedef std::int64_t long_t;
typedef std::size_t  size_t;

using zi2::vl::vec;

typedef vec<long_t, 2> vec2i;
typedef vec<long_t, 3> vec3i;
typedef vec<long_t, 4> vec4i;
typedef vec<long_t, 5> vec5i;

using zi2::vl::subvec;

template <long_t... Ints>
using vek = zi2::vl::vec_type<long_t, Ints...>;

typedef std::lock_guard<std::mutex> guard;

template <typename T>
inline double duration_in_ms(T const& interval)
{
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::microseconds>(interval)
                   .count()) /
           1000;
}

} // namespace znn
