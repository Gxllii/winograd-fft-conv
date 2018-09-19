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
    using M = vek<1, 5, 5>;
    using K = vek<1, 3, 3>;

    do_bench<ZNN_USE_CORES, 1, 64, 64, 1, 640, 640, M, K>("FusionNet 1.2");
    do_bench<ZNN_USE_CORES, 1, 128, 128, 1, 320, 320, M, K>("FusionNet 1.3");
    do_bench<ZNN_USE_CORES, 1, 256, 256, 1, 160, 160, M, K>("FusionNet 1.4");
    do_bench<ZNN_USE_CORES, 1, 512, 512, 1, 80, 80, M, K>("FusionNet 1.5");
    do_bench<ZNN_USE_CORES, 1, 1024, 1024, 1, 40, 40, M, K>("FusionNet 1.5");
}
