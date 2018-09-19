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

#include "znn/types.hpp"

namespace znn
{

inline constexpr long_t div_ceil(long_t a, long_t b)
{
    return a / b + (a % b ? 1 : 0);
}

inline constexpr long_t smallest_prime_factor(long_t a)
{
    return (a % 2 == 0)
               ? 2
               : ((a % 3 == 0)
                      ? 3
                      : ((a % 5 == 0)
                             ? 5
                             : ((a % 7 == 0)
                                    ? 7
                                    : ((a % 11 == 0)
                                           ? 11
                                           : ((a % 13 == 0) ? 13 : a)))));
}

inline constexpr long_t gcd(long_t a, long_t b)
{
    return (b == 0) ? a : gcd(b, a % b);
}

inline constexpr long_t ceil(long_t a, long_t b)
{
    return a / b + (a % b ? 1 : 0);
}

struct maximize_ai_result
{
    long_t      cache;
    long_t      R, C;
    long_t      r, c;
    long double ai;
};

inline constexpr maximize_ai_result maximize_ai(long_t cache, long_t R,
                                                long_t C, long_t beta = 1)
{
    float              best = 0;
    maximize_ai_result ret{0, 0, 0, 0, 0, 0.0};
    for (long_t r = 16; r <= R; r += 16)
    {
        if (R % r == 0)
        {
            for (long_t c = 16; c <= C; c += 16)
            {
                if (C % c == 0)
                {
                    if (beta * 4 * c * r <= cache * 1024 * 2 / 4)
                    {
                        long_t      alpha = (r < R) ? 2 : 1;
                        long double ai =
                            static_cast<long double>(beta * c * r) / 2 /
                            (r + alpha * c);

                        if (ai > best)
                        {
                            best = ai;
                            ret  = {cache, R, C, r, c, ai};
                        }
                    }
                }
            }
        }
    }

    return ret;
}

} // namespace znn
