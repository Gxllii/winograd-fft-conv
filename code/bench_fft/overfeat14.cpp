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
#include "znn/fft/bench.hpp"

using namespace znn::fft;
using namespace znn;

int main()
{
    using F  = vek<1, 14, 14>;
    using K3 = vek<1, 3, 3>;
    using K5 = vek<1, 5, 5>;

    do_bench<ZNN_USE_CORES, 128, 96, 256, 1, 28, 28, F, K5>("OverFeat 2");
    do_bench<ZNN_USE_CORES, 128, 256, 512, 1, 14, 14, F, K3>("OverFeat 3");
    do_bench<ZNN_USE_CORES, 128, 512, 1024, 1, 14, 14, F, K3>("OverFeat 4");
    do_bench<ZNN_USE_CORES, 128, 1024, 1024, 1, 14, 14, F, K3>("OverFeat 5");
}
