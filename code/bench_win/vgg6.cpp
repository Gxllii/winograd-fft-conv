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
#include "znn/win/bench.hpp"

using namespace znn;
using namespace znn::win;

int main()
{
    using M = vek<1, 6, 6>;
    using K = vek<1, 3, 3>;

    do_bench<ZNN_USE_CORES, 64, 64, 64, 1, 226, 226, M, K>("VGG 1.2");
    do_bench<ZNN_USE_CORES, 64, 64, 128, 1, 114, 114, M, K>("VGG 2.1");
    do_bench<ZNN_USE_CORES, 64, 128, 128, 1, 114, 114, M, K>("VGG 2.2");
    do_bench<ZNN_USE_CORES, 64, 128, 256, 1, 58, 58, M, K>("VGG 3.1");
    do_bench<ZNN_USE_CORES, 64, 256, 256, 1, 58, 58, M, K>("VGG 3.2");
    do_bench<ZNN_USE_CORES, 64, 256, 512, 1, 30, 30, M, K>("VGG 4.1");
    do_bench<ZNN_USE_CORES, 64, 512, 512, 1, 30, 30, M, K>("VGG 4.2");
    do_bench<ZNN_USE_CORES, 64, 512, 512, 1, 16, 16, M, K>("VGG 5");
}
