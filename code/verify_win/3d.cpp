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
#include "znn/win/propagation2.hpp"
#include <chrono>
#include <string>

using namespace znn::win;

template <long_t Threads, long_t B, long_t C1, long_t C2, long_t D, long_t H,
          long_t W>
void bench(std::string const& name = "")
{
    using idim = tensor_size<B, C1, D, H, W>;
    using odim = tensor_size<B, C2, D - 2, H - 2, W - 2>;

    using istrides =
        tensor_size<C1 * D * H * W, D * H * W * SIMD_WIDTH, H * W * SIMD_WIDTH,
                    W * SIMD_WIDTH, SIMD_WIDTH>;

    using ostrides = tensor_size<C2*(D - 2) * (H - 2) * (W - 2),
                                 (D - 2) * (H - 2) * (W - 2) * SIMD_WIDTH,
                                 (H - 2) * (W - 2) * SIMD_WIDTH,
                                 (W - 2) * SIMD_WIDTH, SIMD_WIDTH>;

    using layer = layer_t<idim, istrides, odim, ostrides, size3d<4, 4, 4>,
                          size3d<3, 3, 3>>;

    using transform_t = propagation<Threads, layer>;

    kernel_launcher kl(Threads, 1);

    transform_t tt(kl);

    long_t ker_memory = C1 * C2 * 3 * 3 * 3;
    // rand_init
    hbw_array<float> a(rand_init, B * C1 * D * H * W);
    hbw_array<float> b(zero_init, B * C2 * D * H * W);
    hbw_array<float> buffer(one_init, transform_t::buffer_memory / 4);
    hbw_array<float> ker(rand_init, ker_memory);
    hbw_array<float> test_out(zero_init, B * C2 * D * H * W);

    tt.execute(a.data(), ker.data(), b.data(), buffer.data());
    direct_conv<B, C1, D, H, W, C2, 3, 3, 3>(a.data(), ker.data(),
                                             test_out.data());
    test_result(b.data(), test_out.data(), B * C2 * D * H * W);
}

int main()
{
    bench<1, 2, 32, 32, 10, 22, 10>("test");

    // bench<64, 64, 64, 128, 1, 114, 114>("VGG C2");
    // bench<64, 64, 128, 256, 1, 58, 58>("VGG C3");
    // bench<64, 64, 256, 256, 1, 58, 58>("VGG C4");
    // bench<64, 64, 256, 512, 1, 30, 30>("VGG C5");
    // bench<64, 64, 512, 512, 1, 30, 30>("VGG C6");

  //  bench<64, 1, 64, 64, 1, 570, 570>("UNET C1b");
  //  bench<64, 1, 128, 128, 1, 282, 282>("UNET C2b");
  //  bench<64, 1, 256, 256, 1, 138, 138>("UNET C3b");
  //  bench<64, 1, 512, 512, 1, 66, 66>("UNET C4b");
  //  bench<64, 1, 1024, 1024, 1, 30, 30>("UNET C5b");
}
