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
#if (!defined(ZNN_FFT_PROPAGATION_HEADER) || !defined(ZNN_FFT_NAMESPACE))
#error "Need to define the propagation header and fft namespace"
#endif

#include ZNN_FFT_PROPAGATION_HEADER

#include "znn/direct_conv/direct.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"
#include <chrono>
#include <iomanip>
#include <limits>
#include <string>

namespace ZNN_FFT_NAMESPACE
{

using znn::vek;

template <long_t Cores, long_t Threads, long_t RowBlock, long_t B, long_t C1,
          long_t C2, long_t D, long_t H, long_t W, class t_size, class k_size,
          bool TransformKernels = true, bool HTTransforms = true>
void verify(bool pf1 = true, bool pf2 = true)
{
    static constexpr long_t OD = D - k_size::value[0] + 1;
    static constexpr long_t OH = H - k_size::value[1] + 1;
    static constexpr long_t OW = W - k_size::value[2] + 1;

    using idim = vek<B, C1, D, H, W>;
    using odim = vek<B, C2, OD, OH, OW>;

    using istrides = vek<C1 * D * H * W, D * H * W * SIMD_WIDTH,
                         H * W * SIMD_WIDTH, W * SIMD_WIDTH, SIMD_WIDTH>;

    using ostrides = vek<C2 * OD * OH * OW, OD * OH * OW * SIMD_WIDTH,
                         OH * OW * SIMD_WIDTH, OW * SIMD_WIDTH, SIMD_WIDTH>;

    using layer =
        layer_t<idim, istrides, odim, ostrides, t_size, k_size, RowBlock>;

    using transform_t =
        propagation<Cores, Threads, layer, TransformKernels, HTTransforms>;

    transform_t tt(pf1, pf2);

    long_t ker_memory = TransformKernels ? C1 * C2 * k_size::value.prod() : 1;

    hbw_array<float> a(rand_init, B * C1 * D * H * W);
    hbw_array<float> b(zero_init, B * C2 * OD * OH * OW);
    hbw_array<float> buffer(one_init, transform_t::buffer_floats);
    hbw_array<float> ker(one_init, ker_memory);
    hbw_array<float> test_out(zero_init, B * C2 * OD * OH * OW);

    tt.execute(a.data(), ker.data(), b.data(), buffer.data());
    std::cout << "fft finished, start to execute direct one " << std::endl;
    direct_conv<B, C1, D, H, W, C2, k_size::value[0], k_size::value[1],
                k_size::value[2], float>(a.data(), ker.data(), test_out.data());
    std::cout << "direct finished, verifying result " << std::endl;
    test_result(b.data(), test_out.data(), B * C2 * OD * OH * OW);
}

inline constexpr long_t ceil_div(long_t a, long_t b) { return (a + b - 1) / b; }

inline constexpr long_t padded_size(long_t L, long_t T, long_t K)
{
    return ceil_div(L - K + 1, T - K + 1) * (T - K + 1) + K - 1;
}

template <long_t Cores, long_t B, long_t C1, long_t C2, long_t D, long_t H,
          long_t W, class t_size, class k_size, bool TransformKernels = true,
          bool HTTransforms>
void do_verify(std::string const& = "", bool pf1 = true, bool pf2 = true)
{
    static constexpr long_t PADDED_D =
        padded_size(D, t_size::value[0], k_size::value[0]);
    static constexpr long_t PADDED_H =
        padded_size(H, t_size::value[1], k_size::value[1]);
    static constexpr long_t PADDED_W =
        padded_size(W, t_size::value[2], k_size::value[2]);

    std::cout << "Bench of: " << B << ' ' << C1 << ' ' << C2 << ' ' << PADDED_D
              << ' ' << PADDED_H << ' ' << PADDED_W << std::endl;

    verify<Cores, 1, 30, B, C1, C2, PADDED_D, PADDED_H, PADDED_W, t_size,
           k_size, TransformKernels, HTTransforms>(pf1, pf2);
}

} // namespace ZNN_FFT_NAMESPACE
