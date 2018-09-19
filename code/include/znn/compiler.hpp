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

#include <sstream>
#include <string>

// GCC version
#if defined(__GNUC__) && !defined(GCC_VERSION) && !defined(__clang__)
#define GCC_VERSION                                                            \
    ((__GNUC__)*10000 + (__GNUC_MINOR__)*100 + (__GNUC_PATCHLEVEL__))
#endif

// Clang version
#if defined(__clang__)
#define CLANG_VERSION                                                          \
    ((__clang_major__)*10000 + (__clang_minor__)*100 + (__clang_patchlevel__))
#endif

// ICC version
#if defined(__INTEL_COMPILER)
#define ICC_VERSION __INTEL_COMPILER
#endif

inline std::string ___znn__compiler_verson(int v)
{
    return std::to_string(v / 10000) + "." + std::to_string((v % 10000) / 100) +
           "." + std::to_string(v % 100);
}

namespace znn
{

inline std::string compiler_version()
{
    std::string ret;

#if defined(__INTEL_COMPILER)
    ret += "icc " + std::to_string(__INTEL_COMPILER) + " compat with ";
#endif

#if defined(__GNUC__) && !defined(__clang__)
    ret += "gcc ";
    ret += ___znn__compiler_verson(GCC_VERSION);
#elif defined(__clang__)
    ret += "clang ";
    ret += ___znn__compiler_verson(CLANG_VERSION);
#endif
    return ret;
}

} // namespace znn
