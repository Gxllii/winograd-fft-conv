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

#define ZNN_STRINGIFY_0(s) #s
#define ZNN_STRINGIFY(s) ZNN_STRINGIFY_0(s)

#define DIE(message)                                                           \
    {                                                                          \
        std::cout << message << std::endl                                      \
                  << "file: " << __FILE__ << " line: " << __LINE__             \
                  << std::endl;                                                \
        abort();                                                               \
    }                                                                          \
    static_cast<void>(0)

#define UNIMPLEMENTED()                                                        \
    {                                                                          \
        std::cout << "unimplemented function" << std::endl                     \
                  << "file: " << __FILE__ << " line: " << __LINE__             \
                  << std::endl;                                                \
        abort();                                                               \
    }                                                                          \
    static_cast<void>(0)

#define STRONG_ASSERT(condition)                                               \
    if (!(condition))                                                          \
    {                                                                          \
        std::cout << "Assertion " << ZNN_STRINGIFY(condition) << " failed "    \
                  << "file: " << __FILE__ << " line: " << __LINE__             \
                  << std::endl;                                                \
        abort();                                                               \
    }                                                                          \
    static_cast<void>(0)

#if defined(NDEBUG) || defined(ZNN_NO_DEBUG)
#define WEAK_ASSERT(condition) static_cast<void>(0)
#else
#define WEAK_ASSERT(condition) STRONG_ASSERT(condition)
#endif

#define ZNN_ASSERT(condition) WEAK_ASSERT(condition)

template <typename F, typename L>
inline std::string ___this_file_this_line(const F& f, const L& l)
{
    std::ostringstream oss;
    oss << "\nfile: " << f << "\nline: " << l << "\n";
    return oss.str();
}

#define HERE() ___this_file_this_line(__FILE__, __LINE__)
