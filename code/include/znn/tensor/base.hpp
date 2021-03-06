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

#include "znn/tensor/device_reference.hpp"
#include "znn/tensor/tags.hpp"
#include "znn/types.hpp"

namespace znn::detail::tensor
{

template <typename, size_t, typename>
class tensor;

template <typename, size_t, typename>
class sub_tensor;

template <typename, size_t, typename>
class const_sub_tensor;

template <typename T, size_t NumDims, typename Arch>
class value_accessor
{
public:
    static const size_t dimensionality = NumDims;

    typedef T                                      element;
    typedef tensor<T, NumDims - 1, Arch>           value_type;
    typedef sub_tensor<T, NumDims - 1, Arch>       reference;
    typedef const_sub_tensor<T, NumDims - 1, Arch> const_reference;
    typedef Arch                                   architecture;

protected:
    template <typename Reference>
    Reference access(type_t<Reference>, long_t index, T* base,
                     long_t const* extents, long_t const* strides) const
        noexcept
    {
        ZNN_ASSERT(index < extents[0]);
        return Reference(base + index * strides[1], extents + 1, strides + 1);
    }
};

template <typename T>
class value_accessor<T, 1, host_tag>
{
public:
    static const size_t dimensionality = 1;

    typedef T        element;
    typedef T        value_type;
    typedef T&       reference;
    typedef T const& const_reference;
    typedef host_tag architecture;

protected:
    template <typename Reference>
    Reference access(type_t<Reference>, long_t index, T* base,
                     long_t const*, long_t const* strides) const noexcept
    {
        // ZNN_ASSERT(index<extents[0]);
        return *(base + index * strides[1]);
    }
};

template <typename T>
class value_accessor<T, 1, hbw_tag>
{
public:
    static const size_t dimensionality = 1;

    typedef T        element;
    typedef T        value_type;
    typedef T&       reference;
    typedef T const& const_reference;
    typedef hbw_tag  architecture;

protected:
    template <typename Reference>
    Reference access(type_t<Reference>, long_t index, T* base,
                     long_t const*, long_t const* strides) const noexcept
    {
        // ZNN_ASSERT(index<extents[0]);
        return *(base + index * strides[1]);
    }
};

template <typename T>
class value_accessor<T, 1, device_tag>
{
public:
    static const size_t dimensionality = 1;

    typedef T                         element;
    typedef T                         value_type;
    typedef device_reference<T>       reference;
    typedef const_device_reference<T> const_reference;
    typedef device_tag                architecture;

protected:
    template <typename Reference>
    Reference access(type_t<Reference>, long_t index, T* base,
                     long_t const*, long_t const* strides) const noexcept
    {
        // ZNN_ASSERT(index<extents[0]);
        return Reference(base + index * strides[1]);
    }
};

} // namespace znn::detail::tensor
