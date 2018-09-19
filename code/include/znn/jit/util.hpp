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

#include <cstdlib>
#include <iostream>

#define ZNN_JIT_STRINGIFY_0(s) #s
#define ZNN_JIT_STRINGIFY(s) ZNN_JIT_STRINGIFY_0(s)

#define ZNN_JIT_ASSERT(condition)                                              \
    if (!(condition))                                                          \
    {                                                                          \
        std::cout << "Assertion " << ZNN_JIT_STRINGIFY(condition)              \
                  << " failed "                                                \
                  << "file: " << __FILE__ << " line: " << __LINE__             \
                  << std::endl;                                                \
        std::abort();                                                          \
    }                                                                          \
    static_cast<void>(0)

#if defined(NDEBUG) || defined(ZNN_NO_DEBUG)

#define ZNN_JIT_DEBUG_ASSERT(condition) static_cast<void>(0)

#else

#define ZNN_JIT_DEBUG_ASSERT(condition) ZNN_JIT_ASSERT(condition)

#endif
