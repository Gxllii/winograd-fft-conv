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

#include "znn/tensor/tags.hpp"
#include "znn/types.hpp"

#include <cstdlib>

#if !defined(ZNN_NO_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(ZNN_USE_HBW)
#include <hbwmalloc.h>
#else
#define hbw_malloc std::malloc
#define hbw_free std::free
#endif

namespace znn
{
namespace detail
{
namespace tensor
{

inline void* malloc(size_t required_bytes, host_tag)
{
#if 1
    if (required_bytes == 0)
    {
        return nullptr;
    }

    void*  p1; // original block
    void** p2; // aligned block

    size_t alignment = 64;

    int offset = alignment - 1 + sizeof(void*);

    if ((p1 = (void*)std::malloc(required_bytes + offset)) == NULL)
    {
        DIE("std::bad_alloc()");
        // throw std::bad_alloc();
    }

    p2     = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
#else
    void* ret = std::malloc(required_bytes);
    if (ret == NULL)
    {
        DIE("std::bad_alloc()");
        // throw std::bad_alloc();
    }
    return ret;
#endif
}

//#if defined(ZNN_USE_HBW)

inline void* malloc(size_t required_bytes, hbw_tag)
{
#if 1
    if (required_bytes == 0)
    {
        return nullptr;
    }

    void*  p1; // original block
    void** p2; // aligned block

    size_t alignment = 64;

    int offset = alignment - 1 + sizeof(void*);

    if ((p1 = (void*)hbw_malloc(required_bytes + offset)) == NULL)
    {
        DIE("std::bad_alloc()");
        // throw std::bad_alloc();
    }

    p2     = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
#else
    void* ret = hbw_malloc(required_bytes);
    if (ret == NULL)
    {
        DIE("std::bad_alloc()");
        // throw std::bad_alloc();
    }
    return ret;
#endif
}

//#endif

inline void free(void* p, host_tag)
{
#if 1
    std::free(((void**)p)[-1]);
#else
    std::free(p);
#endif
}

//#if defined(ZNN_USE_HBW)

inline void free(void* p, hbw_tag)
{
#if 1
    hbw_free(((void**)p)[-1]);
#else
    hbw_free(p);
#endif
}

//#endif

#if !defined(ZNN_NO_CUDA)

inline void* malloc(size_t required_bytes, device_tag)
{
    void* p = nullptr;

    if (required_bytes > 0)
    {
        auto status = cudaMalloc(&p, required_bytes);
        if (status != 0)
        {
            DIE("std::bad_alloc()");
            // throw std::bad_alloc();
        }
    }

    return p;
}

inline void free(void* p, device_tag)
{
    auto status = cudaFree(p);
    if (status != 0)
    {
        DIE(cudaGetErrorString(status));
        // throw std::logic_error(cudaGetErrorString(status));
    }
}

#endif

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, host_tag, host_tag) noexcept
{
    std::copy_n(in, n, out);
}

template <typename T, typename U>
inline typename std::enable_if<!std::is_same<T, U>::value>::type
copy_cast_n(T const* in, size_t n, U* out, host_tag, host_tag) noexcept
{
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = static_cast<U>(in[i]);
    }
}

//#if defined(ZNN_USE_HBW)

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, hbw_tag, hbw_tag) noexcept
{
    std::copy_n(in, n, out);
}

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, host_tag, hbw_tag) noexcept
{
    std::copy_n(in, n, out);
}

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, hbw_tag, host_tag) noexcept
{
    std::copy_n(in, n, out);
}

//#endif

#if !defined(ZNN_NO_CUDA)

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, host_tag, device_tag) noexcept
{
    auto status = cudaMemcpy(out, in, n * sizeof(T), cudaMemcpyHostToDevice);
    if (status != 0)
    {
        DIE(cudaGetErrorString(status));
        // throw std::logic_error(cudaGetErrorString(status));
    }
}

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, device_tag,
                   device_tag) noexcept
{
    auto status = cudaMemcpy(out, in, n * sizeof(T), cudaMemcpyDeviceToDevice);
    if (status != 0)
    {
        DIE(cudaGetErrorString(status));
        // throw std::logic_error(cudaGetErrorString(status));
    }
}

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, device_tag, host_tag) noexcept
{
    auto status = cudaMemcpy(out, in, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != 0)
    {
        DIE(cudaGetErrorString(status));
        // throw std::logic_error(cudaGetErrorString(status));
    }
}

//#if defined(ZNN_USE_HBW)

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, hbw_tag, device_tag) noexcept
{
    auto status = cudaMemcpy(out, in, n * sizeof(T), cudaMemcpyHostToDevice);
    if (status != 0)
    {
        DIE(cudaGetErrorString(status));
        // throw std::logic_error(cudaGetErrorString(status));
    }
}

template <typename T>
inline void copy_n(T const* in, size_t n, T* out, device_tag, hbw_tag) noexcept
{
    auto status = cudaMemcpy(out, in, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != 0)
    {
        DIE(cudaGetErrorString(status));
        // throw std::logic_error(cudaGetErrorString(status));
    }
}

//#endif

#endif

template <typename T>
inline T load(T const* in, host_tag) noexcept
{
    return *in;
}

template <typename T>
inline void store(T* in, const T& val, host_tag) noexcept
{
    *in = val;
}

//#if defined(ZNN_USE_HBW)

template <typename T>
inline T load(T const* in, hbw_tag) noexcept
{
    return *in;
}

template <typename T>
inline void store(T* in, const T& val, hbw_tag) noexcept
{
    *in = val;
}

//#endif

#if !defined(ZNN_NO_CUDA)

template <typename T>
inline T load(T const* in, device_tag) noexcept
{
    T r;
    copy_n(in, 1, &r, device_tag(), host_tag());
    return r;
}

template <typename T>
inline void store(T* in, const T& val, device_tag) noexcept
{
    copy_n(&val, 1, in, host_tag(), device_tag());
}

#endif
}
}
} // namespace znn::detail::tensor
