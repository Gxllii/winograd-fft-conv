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

#include <iostream>

namespace znn
{

namespace detail
{

struct host_pointer_tag
{
};
struct device_pointer_tag
{
};
struct hbw_pointer_tag
{
};

template <typename T, typename Tag>
class pointer
{
public:
    typedef T  value_type;
    typedef T* pointer_type;

private:
    pointer_type ptr_;

public:
    explicit pointer(pointer_type p = nullptr)
        : ptr_(p)
    {
    }

    pointer(std::nullptr_t)
        : ptr_(nullptr)
    {
    }

    pointer(pointer const& other)
        : ptr_(other.get())
    {
    }

    template <typename O>
    pointer(pointer<O, Tag> const& other,
            typename std::enable_if<std::is_convertible<O, T>::value,
                                    void*>::type = 0)
        : ptr_(other.get())
    {
    }

    pointer& operator=(pointer const& other)
    {
        ptr_ = other.ptr_;
        return *this;
    }

    template <typename O>
    typename std::enable_if<std::is_convertible<O, T>::value, pointer&>::type
    operator=(pointer<O, Tag> const& other)
    {
        ptr_ = other.get();
        return *this;
    }

    pointer_type get() const { return ptr_; }

    operator bool() const { return ptr_ != nullptr; }
};

template <typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>&  os,
           pointer<T, host_pointer_tag> const& p)
{
    os << "h[" << p.get() << "]";
    return os;
}

template <typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>& os,
           pointer<T, hbw_pointer_tag> const& p)
{
    os << "x[" << p.get() << "]";
    return os;
}

template <typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>&    os,
           pointer<T, device_pointer_tag> const& p)
{
    os << "d[" << p.get() << "]";
    return os;
}

} //  Namespace detail

template <typename T>
using host_ptr = detail::pointer<T, detail::host_pointer_tag>;

template <typename T>
using hbw_ptr = detail::pointer<T, detail::hbw_pointer_tag>;

template <typename T>
using device_ptr = detail::pointer<T, detail::device_pointer_tag>;

} // namespace znn
