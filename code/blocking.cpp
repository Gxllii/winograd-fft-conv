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
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "znn/types.hpp"

using v2i = zi2::vl::vec<int, 2>;

struct result
{
    int         cache;
    int         R, C;
    int         r, c;
    long double ai;
};

inline constexpr result maximize_ai(int cache, int R, int C, int beta = 1)
{
    float  best = 0;
    result ret{0, 0, 0, 0, 0, 0.0};
    for (int r = 16; r <= R; r += 16)
    {
        if (R % r == 0)
        {
            for (int c = 16; c <= C; c += 16)
            {
                if (C % c == 0)
                {
                    if (beta * 4 * c * r <= cache * 1024 * 2 / 4)
                    {
                        int         alpha = (r < R) ? 2 : 1;
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

inline void print(result const& r)
{
    std::cout << "BEST FOR " << r.R << " x " << r.C << " " << r.cache
              << "kb is: " << r.ai << " (" << r.r << " x " << r.c << ")\n";
}

template<int X>
inline void static_print()
{
    std::cout << "X: " << X << "\n";
}

int main()
{
    constexpr auto xx = maximize_ai(256, 256, 256, 2);

    static_print<xx.r>();
    static_print<xx.c>();

    std::cout << xx.ai << "\n";

    // std::cout << "cache,candr,ai,cgemm\n";
    // // for (int r = 64; r <= 1024; r *=
    // // 2)
    // {
    //     for (int c = 64; c <= 1024; c *= 2)
    //     {
    //         for (int cache = 32; cache <= 2048; ++cache)
    //         {
    //             auto gemm = maximize_ai(cache, c, c, 1);
    //             std::cout << cache << "," << c << "," << gemm.ai << ",0\n";
    //             auto cgemm = maximize_ai(cache, c, c, 2);
    //             std::cout << cache << "," << c << "," << cgemm.ai << ",1\n";
    //         }
    //     }
    // }
}
