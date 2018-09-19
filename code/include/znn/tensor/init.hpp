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

#include <cstddef>
#include <mutex>
#include <random>

namespace znn::detail::tensor
{

template <typename T>
void random_initialize(T* ptr, std::size_t n,
                       T d = static_cast<T>(0.1)) noexcept
{
    static std::mt19937 rng = std::mt19937(1234);
    static std::mutex   m;

    std::uniform_real_distribution<T> dis(-d, d);

    {
        guard g(m);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr[i] = dis(rng);
        }
    }
}

template <typename T>
void random_initialize_normal(T* ptr, std::size_t n, T mean, T stdev) noexcept
{
    static std::mt19937 rng = std::mt19937(1234);
    static std::mutex   m;

    std::normal_distribution<T> dis(mean, stdev);

    {
        guard g(m);

        for (std::size_t i = 0; i < n; ++i)
        {
            ptr[i] = dis(rng);
        }
    }
}

} // namespace znn::detail::tensor
