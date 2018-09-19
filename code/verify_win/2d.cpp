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
#include "znn/direct_conv/direct.hpp"
#include "znn/win/propagation.hpp"
#include <chrono>
#include <string>

using namespace znn;
using namespace znn::win;

template <long_t Cores, long_t B, long_t C1, long_t C2, long_t D, long_t H,
          long_t W, class m_size, class k_size>
void bench()
{
    constexpr long_t RowBlock = 16;

    static constexpr long_t OD = D - k_size::value[0] + 1;
    static constexpr long_t OH = H - k_size::value[1] + 1;
    static constexpr long_t OW = W - k_size::value[2] + 1;

    using idim = vek<B, C1, D, H, W>;
    using odim = vek<B, C2, OD, OH, OW>;

    using istrides =
        vek<C1 * D * H * W, D * H * W * CACHELINE_SIZE, H * W * CACHELINE_SIZE,
            W * CACHELINE_SIZE, CACHELINE_SIZE>;

    using ostrides =
        vek<C2 * OD * OH * OW, OD * OH * OW * CACHELINE_SIZE,
            OH * OW * CACHELINE_SIZE, OW * CACHELINE_SIZE, CACHELINE_SIZE>;

    using layer =
        layer_t<idim, istrides, odim, ostrides, m_size, k_size, RowBlock>;

    using transform_t = propagation<Cores, 1, layer, true, false>;

    long_t ker_memory = C1 * C2 * k_size::value.prod();
    // rand_init
    hbw_array<float> a(rand_init, B * C1 * D * H * W);
    hbw_array<float> b(zero_init, B * C2 * OD * OH * OW);
    hbw_array<float> buffer(zero_init, transform_t::buffer_floats);
    hbw_array<float> ker(rand_init, ker_memory);
    hbw_array<float> test_out(zero_init, B * C2 * OD * OH * OW);

    {
        transform_t tt(true, true);

        tt.execute(a.data(), ker.data(), b.data(), buffer.data());

        std::cout << "DONE WIN" << std::endl;
    }

    direct_conv<B, C1, D, H, W, C2, k_size::value[0], k_size::value[1],
                k_size::value[2]>(a.data(), ker.data(), test_out.data());
    test_result(b.data(), test_out.data(), B * C2 * OD * OH * OW);
}

int main()
{
    bench<ZNN_NUM_CORES, 1, 64, 64, 1, 230, 230, vek<1, 6, 6>, vek<1, 3, 3>>();

    bench<ZNN_NUM_CORES, 12, 512, 512, 14, 14, 14, vek<6, 6, 6>,
          vek<3, 3, 3>>();
}
