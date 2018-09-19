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
#include "znn/fft/propagation.hpp"
#include "znn/tensor/tensor.hpp"

#include <bitset>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

using namespace znn::fft;
using namespace znn;

inline constexpr long_t ceil_div(long_t a, long_t b) { return (a + b - 1) / b; }

inline constexpr long_t padded_size(long_t L, long_t T, long_t K)
{
    return ceil_div(L - K + 1, T - K + 1) * (T - K + 1) + K - 1;
}

template <typename T1, typename T2>
void cast_data(T1* From, T2* To, long_t length)
{
    for (long_t i = 0; i < length; i++)
    {
        To[i] = static_cast<T2>(From[i]);
    }
}

using measurement = std::tuple<long double, long double, long double>;

measurement gather_statistics(long double* x, float* y, long_t len)
{
    measurement ret{0.0, 0.0, 0.0};

    for (long_t i = 0; i < len; ++i)
    {
        std::get<0>(ret) = std::max(
            std::get<0>(ret), std::abs(x[i] - static_cast<long double>(y[i])));
        std::get<1>(ret) += std::abs(x[i] - static_cast<long double>(y[i]));
        std::get<2>(ret) += 1;
    }

    return ret;
}

measurement merge_statistics(measurement const& a, measurement const& b)
{
    return {std::max(std::get<0>(a), std::get<0>(b)),
            std::get<1>(a) + std::get<1>(b), std::get<2>(a) + std::get<2>(b)};
}

template <long_t B, long_t C1, long_t C2, long_t D, long_t H, long_t W,
          class k_size>
measurement measure_error_direct(std::string const& name    = "",
                                 std::string const& weights = "")
{
    constexpr long_t OD = D - k_size::value[0] + 1;
    constexpr long_t OH = H - k_size::value[1] + 1;
    constexpr long_t OW = W - k_size::value[2] + 1;

    std::cout << "DIRECT CONVOLUTION ERROR MEASUREMENT FOR: " << name
              << std::endl;

    long_t ker_memory = C1 * C2 * k_size::value.prod();
    ;

    hbw_array<long double> dker(zero_init, ker_memory);
    hbw_array<float>       fker;

    if (weights != "")
    {
        fker = hbw_array<float>(zero_init, ker_memory);
        std::ifstream ifs(weights.c_str());
        for (long_t i = 0; i < ker_memory; ++i)
        {
            STRONG_ASSERT(ifs >> dker.data()[i]);
        }
        cast_data<long double, float>(dker.data(), fker.data(), ker_memory);
    }
    else
    {
        fker = hbw_array<float>(norm_init, 0.0f,
                                std::sqrt(2.f / (C1 * k_size::value.prod())),
                                ker_memory);

        cast_data<float, long double>(fker.data(), dker.data(), ker_memory);
    }

    // rand_init
    hbw_array<float> fa(rand_init, B * C1 * D * H * W);

    hbw_array<long double> da(zero_init, B * C1 * D * H * W);
    cast_data<float, long double>(fa.data(), da.data(), B * C1 * D * H * W);

    hbw_array<float>       fb(zero_init, B * C2 * OD * OH * OW);
    hbw_array<long double> db(zero_init, B * C2 * OD * OH * OW);

    std::cout << "Computing ground truth using long doubles..." << std::flush;

    direct_conv_parallel<B, C1, D, H, W, C2, k_size::value[0], k_size::value[1],
                         k_size::value[2], long double>(da.data(), dker.data(),
                                                        db.data());

    std::cout << " DONE\n"
              << "Computing using floats..." << std::flush;

    direct_conv_parallel<B, C1, D, H, W, C2, k_size::value[0], k_size::value[1],
                         k_size::value[2], float>(fa.data(), fker.data(),
                                                  fb.data());

    std::cout << " DONE\n";

    auto ret = gather_statistics(db.data(), fb.data(), db.num_elements());

    std::cout << "MAX ERROR: " << std::get<0>(ret)
              << ", AVG ERROR: " << (std::get<1>(ret) / std::get<2>(ret))
              << std::endl;

    return ret;
}

template <long_t B, long_t C1, long_t C2, long_t ID, long_t IH, long_t IW,
          class k_size, class t_size, long_t Cores = ZNN_NUM_CORES,
          long_t HT = 1, long_t RB = 16>
measurement measure_error_fft(std::string const& name,
                              std::string const& weights = "")
{

    std::cout << "FFT CONVOLUTION ERROR MEASUREMENT FOR: " << name << std::endl;

    // znn::win::avx512::clear_znn_gemms();

    static constexpr long_t D =
        padded_size(ID, t_size::value[0], k_size::value[0]);
    static constexpr long_t H =
        padded_size(IH, t_size::value[1], k_size::value[1]);
    static constexpr long_t W =
        padded_size(IW, t_size::value[2], k_size::value[2]);

    long_t ker_memory =
        C1 * C2 * k_size::value[0] * k_size::value[2] * k_size::value[1];
    hbw_array<long double> dker(zero_init, ker_memory);
    hbw_array<float>       fker;

    if (weights != "")
    {
        fker = hbw_array<float>(zero_init, ker_memory);
        std::ifstream ifs(weights.c_str());
        for (long_t i = 0; i < ker_memory; ++i)
        {
            STRONG_ASSERT(ifs >> dker.data()[i]);
        }
        cast_data<long double, float>(dker.data(), fker.data(), ker_memory);
    }
    else
    {
        fker = hbw_array<float>(
            norm_init, 0.0f,
            std::sqrt(2.f / (C1 * k_size::value[0] * k_size::value[1] *
                             k_size::value[2])),
            ker_memory);

        cast_data<float, long double>(fker.data(), dker.data(), ker_memory);
    }

    constexpr long_t OD = D - k_size::value[0] + 1;
    constexpr long_t OH = H - k_size::value[1] + 1;
    constexpr long_t OW = W - k_size::value[2] + 1;

    using idim = vek<B, C1, D, H, W>;
    using odim = vek<B, C2, OD, OH, OW>;

    using istrides = vek<C1 * D * H * W, D * H * W * SIMD_WIDTH,
                         H * W * SIMD_WIDTH, W * SIMD_WIDTH, SIMD_WIDTH>;

    using ostrides = vek<C2 * OD * OH * OW, OD * OH * OW * SIMD_WIDTH,
                         OH * OW * SIMD_WIDTH, OW * SIMD_WIDTH, SIMD_WIDTH>;

    using layer = layer_t<idim, istrides, odim, ostrides, t_size, k_size, RB>;

    using transform_t = propagation<Cores, HT, layer, true>;

    transform_t tt;

    hbw_array<float> fa(rand_init, B * C1 * D * H * W);

    hbw_array<long double> da(zero_init, B * C1 * D * H * W);
    cast_data<float, long double>(fa.data(), da.data(), B * C1 * D * H * W);

    hbw_array<float>       fb(zero_init, B * C2 * OD * OH * OW);
    hbw_array<long double> db(zero_init, B * C2 * OD * OH * OW);

    hbw_array<float> buffer(one_init, transform_t::buffer_memory / 4);

    std::cout << "Computing ground truth using long doubles..." << std::flush;
    direct_conv_parallel<B, C1, D, H, W, C2, k_size::value[0], k_size::value[1],
                         k_size::value[2], long double>(da.data(), dker.data(),
                                                        db.data());

    std::cout << " DONE\n"
              << "Computing using fft..." << std::flush;

    tt.execute(fa.data(), fker.data(), fb.data(), buffer.data());

    std::cout << " DONE\n";

    auto ret = gather_statistics(db.data(), fb.data(), db.num_elements());

    std::cout << "MAX ERROR: " << std::get<0>(ret)
              << ", AVG ERROR: " << (std::get<1>(ret) / std::get<2>(ret))
              << std::endl;

    return ret;
}

template <class M>
measurement measure_vgg_fft(bool inference = false)
{
    using K = vek<1, 3, 3>;

    std::cout << "F(" << M::value[1] << "x" << M::value[2] << ", 3x3)"
              << std::endl;
    auto m = measure_error_fft<1, 64, 64, 1, 226, 226, K, M>(
        "vgg 1.2", inference ? "./trained_kernels/vgg-c1_2" : "");

    m = merge_statistics(
        m, measure_error_fft<1, 64, 128, 1, 114, 114, K, M>(
               "vgg 2.1", inference ? "./trained_kernels/vgg-c2_1" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 128, 128, 1, 114, 114, K, M>(
               "vgg 2.2", inference ? "./trained_kernels/vgg-c2_2" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 128, 256, 1, 58, 58, K, M>(
               "vgg 3.1", inference ? "./trained_kernels/vgg-c3_1" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 256, 256, 1, 58, 58, K, M>(
               "vgg 3.2", inference ? "./trained_kernels/vgg-c3_2" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 256, 512, 1, 30, 30, K, M>(
               "vgg 4.1", inference ? "./trained_kernels/vgg-c4_1" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 512, 512, 1, 30, 30, K, M>(
               "vgg 4.2", inference ? "./trained_kernels/vgg-c4_2" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 512, 512, 1, 16, 16, K, M>(
               "vgg 5", inference ? "./trained_kernels/vgg-c5_1" : ""));

    std::cout << ">>>>> MAX ERROR: " << std::get<0>(m)
              << ", AVG ERROR: " << (std::get<1>(m) / std::get<2>(m))
              << std::endl
              << std::endl;

    return m;
}

measurement measure_vgg_direct(bool inference = false)
{
    using K = vek<1, 3, 3>;

    auto m = measure_error_direct<1, 64, 64, 1, 226, 226, K>(
        "vgg 1.2", inference ? "./trained_kernels/vgg-c1_2" : "");

    m = merge_statistics(
        m, measure_error_direct<1, 64, 128, 1, 114, 114, K>(
               "vgg 2.1", inference ? "./trained_kernels/vgg-c2_1" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 128, 128, 1, 114, 114, K>(
               "vgg 2.2", inference ? "./trained_kernels/vgg-c2_2" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 128, 256, 1, 58, 58, K>(
               "vgg 3.1", inference ? "./trained_kernels/vgg-c3_1" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 256, 256, 1, 58, 58, K>(
               "vgg 3.2", inference ? "./trained_kernels/vgg-c3_2" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 256, 512, 1, 30, 30, K>(
               "vgg 4.1", inference ? "./trained_kernels/vgg-c4_1" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 512, 512, 1, 30, 30, K>(
               "vgg 4.2", inference ? "./trained_kernels/vgg-c4_2" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 512, 512, 1, 16, 16, K>(
               "vgg 5", inference ? "./trained_kernels/vgg-c5_1" : ""));

    std::cout << ">>>>> MAX ERROR: " << std::get<0>(m)
              << ", AVG ERROR: " << (std::get<1>(m) / std::get<2>(m))
              << std::endl
              << std::endl;

    return m;
}

template <class M>
measurement measure_c3d_fft(bool inference = false)
{
    using K = vek<3, 3, 3>;

    std::cout << "F(" << M::value[0] << "x" << M::value[1] << "x" << M::value[2]
              << ", 3x3x3)" << std::endl;
    auto m = measure_error_fft<1, 64, 128, 18, 58, 58, K, M>(
        "c3d C2a", inference ? "./trained_kernels/C3D-c2a" : "");

    m = merge_statistics(
        m, measure_error_fft<1, 128, 256, 10, 30, 30, K, M>(
               "c3d C3a", inference ? "./trained_kernels/C3D-c3a" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 256, 256, 10, 30, 30, K, M>(
               "c3d C3b", inference ? "./trained_kernels/C3D-c3b" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 256, 512, 6, 16, 16, K, M>(
               "c3d C4a", inference ? "./trained_kernels/C3D-c4a" : ""));

    m = merge_statistics(
        m, measure_error_fft<1, 512, 512, 6, 16, 16, K, M>(
               "c3d C4b", inference ? "./trained_kernels/C3D-c4b" : ""));

    std::cout << ">>>>> MAX ERROR: " << std::get<0>(m)
              << ", AVG ERROR: " << (std::get<1>(m) / std::get<2>(m))
              << std::endl
              << std::endl;

    return m;
}

measurement measure_c3d_direct(bool inference = false)
{
    using K = vek<3, 3, 3>;

    auto m = measure_error_direct<1, 64, 128, 18, 58, 58, K>(
        "c3d C2a", inference ? "./trained_kernels/C3D-c2a" : "");

    m = merge_statistics(
        m, measure_error_direct<1, 128, 256, 10, 30, 30, K>(
               "c3d C3a", inference ? "./trained_kernels/C3D-c3a" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 256, 256, 10, 30, 30, K>(
               "c3d C3b", inference ? "./trained_kernels/C3D-c3b" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 256, 512, 6, 16, 16, K>(
               "c3d C4a", inference ? "./trained_kernels/C3D-c4a" : ""));

    m = merge_statistics(
        m, measure_error_direct<1, 512, 512, 6, 16, 16, K>(
               "c3d C4b", inference ? "./trained_kernels/C3D-c4b" : ""));

    std::cout << ">>>>> MAX ERROR: " << std::get<0>(m)
              << ", AVG ERROR: " << (std::get<1>(m) / std::get<2>(m))
              << std::endl
              << std::endl;

    return m;
}

int main()
{

    std::cout << std::scientific << std::setprecision(2);

    std::ostringstream c3ds, vggs;
    // c3ds << std::scientific << std::setprecision(2);
    vggs << std::scientific << std::setprecision(2);

    // clang-format off
    // c3ds << "+-----------+--------------+--------------+--------------+--------------+--------------+--------------+\n";
    // c3ds << "| C3D       |    DIRECT    | F(4x4x4,3^3) | F(4x4x4,3^3) | F(4x6x6,3^3) | F(6x6x6,3^3) | F(8x6x6,3^3) |\n";
    // c3ds << "+-----------+--------------+--------------+--------------+--------------+--------------+--------------+\n";

    vggs << "+-----------+--------------+--------------+--------------+--------------+--------------+--------------+\n";
    vggs << "| VGG       |    DIRECT    |  F(8x8,3x3)  |  F(16^2,3x3) |  F(32^2,3x3) |  F(17^2,3x3) |  F(29^2,3x3) |\n";
    vggs << "+-----------+--------------+--------------+--------------+--------------+--------------+--------------+\n";

    // clang-format on

    {
        measurement r[6];

        r[0] = measure_vgg_direct(false);
        r[1] = measure_vgg_fft<vek<1, 8, 8>>(false);
        r[2] = measure_vgg_fft<vek<1, 16, 16>>(false);
        r[3] = measure_vgg_fft<vek<1, 32, 32>>(false);
        r[4] = measure_vgg_fft<vek<1, 17, 17>>(false);
        r[5] = measure_vgg_fft<vek<1, 29, 29>>(false);

        vggs << "| TrainMax  |";

        for (int i = 0; i < 6; ++i)
        {
            vggs << "   " << std::get<0>(r[i]) << "   |";
        }

        vggs << "\n";
        vggs << "| TrainAvg  |";

        for (int i = 0; i < 6; ++i)
        {
            vggs << "   " << (std::get<1>(r[i]) / std::get<2>(r[i])) << "   |";
        }

        vggs << "\n";

        r[0] = measure_vgg_direct(true);
        r[1] = measure_vgg_fft<vek<1, 8, 8>>(true);
        r[2] = measure_vgg_fft<vek<1, 16, 16>>(true);
        r[3] = measure_vgg_fft<vek<1, 32, 32>>(true);
        r[4] = measure_vgg_fft<vek<1, 17, 17>>(true);
        r[5] = measure_vgg_fft<vek<1, 29, 29>>(true);

        vggs << "| InferMax  |";

        for (int i = 0; i < 6; ++i)
        {
            vggs << "   " << std::get<0>(r[i]) << "   |";
        }

        vggs << "\n";
        vggs << "| InferAvg  |";

        for (int i = 0; i < 6; ++i)
        {
            vggs << "   " << (std::get<1>(r[i]) / std::get<2>(r[i])) << "   |";
        }

        vggs << "\n";

        vggs << "+-----------+--------------+--------------+--------------+----"
                "----------+----"
                "----------+--------------+\n";
    }

    // {
    //     measurement r[6];

    //     r[0] = measure_c3d_direct(false);
    //     r[1] = measure_c3d_fft<vek<2, 2, 2>>(false);
    //     r[2] = measure_c3d_fft<vek<4, 4, 4>>(false);
    //     r[3] = measure_c3d_fft<vek<4, 6, 6>>(false);
    //     r[4] = measure_c3d_fft<vek<6, 6, 6>>(false);
    //     r[5] = measure_c3d_fft<vek<8, 6, 6>>(false);

    //     c3ds << "| TrainMax  |";

    //     for (int i = 0; i < 6; ++i)
    //     {
    //         c3ds << "   " << std::get<0>(r[i]) << "   |";
    //     }

    //     c3ds << "\n";
    //     c3ds << "| TrainAvg  |";

    //     for (int i = 0; i < 6; ++i)
    //     {
    //         c3ds << "   " << (std::get<1>(r[i]) / std::get<2>(r[i])) << " |";
    //     }

    //     c3ds << "\n";

    //     r[0] = measure_c3d_direct(true);
    //     r[1] = measure_c3d_fft<vek<2, 2, 2>>(true);
    //     r[2] = measure_c3d_fft<vek<4, 4, 4>>(true);
    //     r[3] = measure_c3d_fft<vek<4, 6, 6>>(true);
    //     r[4] = measure_c3d_fft<vek<6, 6, 6>>(true);
    //     r[5] = measure_c3d_fft<vek<8, 6, 6>>(true);

    //     c3ds << "| InferMax  |";

    //     for (int i = 0; i < 6; ++i)
    //     {
    //         c3ds << "   " << std::get<0>(r[i]) << "   |";
    //     }

    //     c3ds << "\n";
    //     c3ds << "| InferAvg  |";

    //     for (int i = 0; i < 6; ++i)
    //     {
    //         c3ds << "   " << (std::get<1>(r[i]) / std::get<2>(r[i])) << " |";
    //     }

    //     c3ds << "\n";

    //     c3ds <<
    //     "+-----------+--------------+--------------+--------------+----"
    //             "----------+----"
    //             "----------+--------------+\n";
    // }

    std::cout << vggs.str() << "\n\n";
    // std::cout << c3ds.str() << "\n\n";

    std::cerr << vggs.str() << "\n\n";
    // std::cerr << c3ds.str() << "\n\n";
}
