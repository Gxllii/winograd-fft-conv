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

#include "znn/tensor/memory.hpp"
#include "znn/tensor/tags.hpp"
#include "znn/types.hpp"

namespace znn
{
namespace detail
{
namespace tensor
{

template <typename T>
class const_device_reference
{
protected:
    T* p;

public:
    explicit const_device_reference(T* t) noexcept
        : p(t)
    {
    }

    const_device_reference(const_device_reference const& other) noexcept
        : p(other.p)
    {
    }

    operator T() const noexcept
    {
        return detail::tensor::load(p, device_tag());
    }

    const_device_reference& operator=(const_device_reference const&) = delete;
};

template <typename T>
class device_reference : public const_device_reference<T>
{
private:
    typedef const_device_reference<T> super_type;

public:
    explicit device_reference(T* t) noexcept
        : super_type(t)
    {
    }

    device_reference(device_reference const& other) noexcept
        : super_type(other.p)
    {
    }

    device_reference& operator=(device_reference const& other) noexcept
    {
        T x = detail::tensor::load(other.p, device_tag());
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template <typename O>
    device_reference& operator=(O const& other) noexcept
    {
        detail::tensor::store(super_type::p, static_cast<T>(other),
                              device_tag());
        return *this;
    }

    template <typename O>
    device_reference& operator+=(O const& other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x += static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template <typename O>
    device_reference& operator-=(O const& other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x -= static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template <typename O>
    device_reference& operator*=(O const& other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x *= static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template <typename O>
    device_reference& operator/=(O const& other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x /= static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }
};
}
}
} // namespace znn::detail::tensor
