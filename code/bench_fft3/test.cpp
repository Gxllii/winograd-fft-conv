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
#include "znn/fft3/bench.hpp"

using namespace znn::fft3;
using namespace znn;

int main()
{
    using F32 = vek<1, 32, 32>;
    using F16 = vek<1, 16, 16>;
    using K3  = vek<1, 3, 3>;
    using K5  = vek<1, 5, 5>;

    do_bench<ZNN_USE_CORES, 256, 96, 256, 1, 32, 32, F32, K5, false>(
        "AlexNet 2");
    do_bench<ZNN_USE_CORES, 256, 256, 384, 1, 16, 16, F16, K3, false>(
        "AlexNet 3");
    do_bench<ZNN_USE_CORES, 256, 384, 256, 1, 16, 16, F16, K3, false>(
        "AlexNet 4");
    do_bench<ZNN_USE_CORES, 256, 256, 256, 1, 16, 16, F16, K3, false>(
        "AlexNet 5");
}
