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

#include <dlfcn.h>
#include <memory>
#include <utility>

namespace znn::jit
{

extern "C" {
typedef void (*znn_cgemm_t)(float const* const, float const* const,
                            float* const, float const* const,
                            float const* const, float const* const,
                            float* const);
}

namespace detail
{

class library_deleter
{
private:
    void* dl_;

public:
    library_deleter(void* dl)
        : dl_(dl)
    {
    }

    library_deleter(library_deleter const&) = delete;
    library_deleter& operator=(library_deleter const&) = delete;

    ~library_deleter() { dlclose(dl_); }
};

} // namespace detail

class weak_cgemm_t;

class shared_cgemm_t
{
private:
    znn_cgemm_t                              fn_ = nullptr;
    std::shared_ptr<detail::library_deleter> deleter_;

    friend class weak_cgemm_t;

    shared_cgemm_t(znn_cgemm_t                                   fn,
                   std::weak_ptr<detail::library_deleter> const& del)
        : fn_(nullptr)
        , deleter_(del.lock())
    {
        if (deleter_)
        {
            fn_ = fn;
        }
    }

public:
    shared_cgemm_t() {}

    shared_cgemm_t(znn_cgemm_t fn, void* dl)
        : fn_(fn)
        , deleter_(std::make_shared<detail::library_deleter>(dl))
    {
    }

    shared_cgemm_t(shared_cgemm_t const& other)
        : fn_(other.fn_)
        , deleter_(other.deleter_)
    {
    }

    shared_cgemm_t& operator=(shared_cgemm_t const& other)
    {
        fn_      = other.fn_;
        deleter_ = other.deleter_;
        return *this;
    }

    shared_cgemm_t(shared_cgemm_t&& other)
        : fn_(other.fn_)
        , deleter_(std::move(other.deleter_))
    {
        other.fn_ = nullptr;
    }

    shared_cgemm_t& operator=(shared_cgemm_t&& other)
    {
        fn_      = std::exchange(other.fn_, nullptr);
        deleter_ = std::move(other.deleter_);
        return *this;
    }

    __attribute__((always_inline)) void
    operator()(float const* const a1, float const* const a2, float* const a3,
               float const* const a4, float const* const a5,
               float const* const a6, float* const a7) const
    {
        ZNN_JIT_DEBUG_ASSERT(fn_);
        fn_(a1, a2, a3, a4, a5, a6, a7);
    }

    explicit operator bool() const { return fn_ != nullptr; }
};

class weak_cgemm_t
{
private:
    znn_cgemm_t                            fn_ = nullptr;
    std::weak_ptr<detail::library_deleter> deleter_;

public:
    weak_cgemm_t() {}

    weak_cgemm_t(weak_cgemm_t const& other)
        : fn_(other.fn_)
        , deleter_(other.deleter_)
    {
    }

    weak_cgemm_t& operator=(weak_cgemm_t const& other)
    {
        fn_      = other.fn_;
        deleter_ = other.deleter_;
        return *this;
    }

    weak_cgemm_t(weak_cgemm_t&& other)
        : fn_(other.fn_)
        , deleter_(std::move(other.deleter_))
    {
        other.fn_ = nullptr;
    }

    weak_cgemm_t& operator=(weak_cgemm_t&& other)
    {
        fn_      = std::exchange(other.fn_, nullptr);
        deleter_ = std::move(other.deleter_);
        return *this;
    }

    weak_cgemm_t(shared_cgemm_t const& shared)
        : fn_(shared.fn_)
        , deleter_(shared.deleter_)
    {
    }

    weak_cgemm_t& operator=(shared_cgemm_t const& shared)
    {
        fn_      = shared.fn_;
        deleter_ = shared.deleter_;

        return *this;
    }

    shared_cgemm_t lock() const { return shared_cgemm_t(fn_, deleter_); }
};

} // namespace znn::jit
