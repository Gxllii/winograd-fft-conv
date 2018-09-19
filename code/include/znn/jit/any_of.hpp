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

#include <string>
#include <type_traits>

namespace znn::jit
{

template <class... Ts>
struct is_any_of_types : std::false_type
{
};

template <class A, class B, class... Ts>
struct is_any_of_types<A, B, Ts...>
    : std::bool_constant<std::is_same<A, B>::value ||
                         is_any_of_types<A, Ts...>::value>
{
};

template <class... Ts>
class any_of
{
private:
    std::string str_;

public:
    template <class T,
              class V = std::enable_if_t<is_any_of_types<T, Ts...>::value>>
    any_of(T const& v)
        : str_(v.to_arg())
    {
    }

    any_of(any_of const& other)
        : str_(other.str_)
    {
    }

    any_of& operator=(any_of const& other)
    {
        str_ = other.str_;
        return *this;
    }

    std::string to_arg() const { return str_; }
};

} // namespace znn::jit
