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
#include "znn/intrin.hpp"
#include "znn/types.hpp"
#include "znn/util/kernel_launcher.hpp"

using namespace znn;

template <const long_t batch, const long_t IF, const long_t D, const long_t H,
          const long_t W, const long_t OF, const long_t KD, const long_t KH,
          const long_t KW, typename T>
void direct_corr(T* D0, T* F, T* O)
{
    constexpr long_t batch_stride  = IF * D * H * W;
    constexpr long_t inf_stride    = D * H * W * SIMD_WIDTH;
    constexpr long_t D_stride      = H * W * SIMD_WIDTH;
    constexpr long_t H_stride      = W * SIMD_WIDTH;
    constexpr long_t K_inf_stride  = OF * KD * KH * KW;
    constexpr long_t K_of_stride   = KD * KH * KW * SIMD_WIDTH;
    constexpr long_t oD            = D - KD + 1;
    constexpr long_t oW            = W - KW + 1;
    constexpr long_t oH            = H - KH + 1;
    constexpr long_t out_B_stride  = OF * oD * oH * oW;
    constexpr long_t out_of_stride = oD * oH * oW * SIMD_WIDTH;
    constexpr long_t out_D_stride  = oH * oW * SIMD_WIDTH;
    constexpr long_t out_H_stride  = oW * SIMD_WIDTH;

    long_t inf, d, h, w;
    for (long_t b = 0; b < batch; b++)
    {
        for (inf = 0; inf < IF / SIMD_WIDTH; inf++)
        {
            for (d = 0; d < oD; d++)
            {
                for (h = 0; h < oH; h++)
                {
                    for (w = 0; w < oW; w++)
                    {
                        for (long_t outf = 0; outf < OF / SIMD_WIDTH; outf++)
                        {
                            for (long_t ofs = 0; ofs < SIMD_WIDTH; ofs++)
                                for (long_t ifs = 0; ifs < SIMD_WIDTH; ifs++)
                                {
                                    for (long_t z = 0; z < KD; z++)
                                    {
                                        for (long_t y = 0; y < KH; y++)
                                        {
                                            for (long_t x = 0; x < KW; x++)
                                            {
                                                long_t index =
                                                    outf * out_of_stride +
                                                    b * out_B_stride +
                                                    d * out_D_stride +
                                                    h * out_H_stride +
                                                    w * SIMD_WIDTH + ofs;
                                                long_t data_index =
                                                    inf * inf_stride +
                                                    b * batch_stride +
                                                    (d + z) * D_stride +
                                                    (h + y) * H_stride +
                                                    (w + x) * SIMD_WIDTH + ifs;
                                                long_t kernel_index =
                                                    (inf * SIMD_WIDTH + ifs) *
                                                        K_inf_stride +
                                                    outf * K_of_stride +
                                                    z * KH * KW * SIMD_WIDTH +
                                                    y * KW * SIMD_WIDTH +
                                                    x * SIMD_WIDTH + ofs;
                                                O[index] += F[kernel_index] *
                                                            D0[data_index];
                                            }
                                        }
                                    }
                                }
                        }
                    }
                }
            }
        }
    }
}

template <const long_t batch, const long_t IF, const long_t D, const long_t H,
          const long_t W, const long_t OF, const long_t KD, const long_t KH,
          const long_t KW, typename T>
void direct_corr_parallel(T* D0, T* F, T* O)
{
    constexpr long_t batch_stride  = IF * D * H * W;
    constexpr long_t inf_stride    = D * H * W * SIMD_WIDTH;
    constexpr long_t D_stride      = H * W * SIMD_WIDTH;
    constexpr long_t H_stride      = W * SIMD_WIDTH;
    constexpr long_t K_inf_stride  = OF * KD * KH * KW;
    constexpr long_t K_of_stride   = KD * KH * KW * SIMD_WIDTH;
    constexpr long_t oD            = D - KD + 1;
    constexpr long_t oW            = W - KW + 1;
    constexpr long_t oH            = H - KH + 1;
    constexpr long_t out_B_stride  = OF * oD * oH * oW;
    constexpr long_t out_of_stride = oD * oH * oW * SIMD_WIDTH;
    constexpr long_t out_D_stride  = oH * oW * SIMD_WIDTH;
    constexpr long_t out_H_stride  = oW * SIMD_WIDTH;

    std::array<std::vector<std::function<void()>>, ZNN_NUM_CORES * 2> work;
    std::array<std::function<void()>, ZNN_NUM_CORES * 2>              fns;

    kernel_launcher launcher(ZNN_NUM_CORES, 2);

    for (long_t b = 0; b < batch; b++)
    {
        for (long_t outf = 0; outf < OF / SIMD_WIDTH; outf++)
        {
            for (long_t h = 0; h < oH; h++)
            {
                work[(outf * oH + h) % (ZNN_NUM_CORES * 2)].push_back([=]() {
                    for (long_t d = 0; d < oD; d++)
                    {
                        for (long_t w = 0; w < oW; w++)
                        {
                            for (long_t inf = 0; inf < IF / SIMD_WIDTH; inf++)
                            {
                                for (long_t ofs = 0; ofs < SIMD_WIDTH; ofs++)
                                    for (long_t ifs = 0; ifs < SIMD_WIDTH;
                                         ifs++)
                                    {
                                        for (long_t z = 0; z < KD; z++)
                                        {
                                            for (long_t y = 0; y < KH; y++)
                                            {
                                                for (long_t x = 0; x < KW; x++)
                                                {
                                                    long_t index =
                                                        outf * out_of_stride +
                                                        b * out_B_stride +
                                                        d * out_D_stride +
                                                        h * out_H_stride +
                                                        w * SIMD_WIDTH + ofs;
                                                    long_t data_index =
                                                        inf * inf_stride +
                                                        b * batch_stride +
                                                        (d + z) * D_stride +
                                                        (h + y) * H_stride +
                                                        (w + x) * SIMD_WIDTH +
                                                        ifs;
                                                    long_t kernel_index =
                                                        (inf * SIMD_WIDTH +
                                                         ifs) *
                                                            K_inf_stride +
                                                        outf * K_of_stride +
                                                        z * KH * KW *
                                                            SIMD_WIDTH +
                                                        y * KW * SIMD_WIDTH +
                                                        x * SIMD_WIDTH + ofs;
                                                    O[index] +=
                                                        F[kernel_index] *
                                                        D0[data_index];
                                                }
                                            }
                                        }
                                    }
                            }
                        }
                    }
                });
            }
        }
    }

    for (int i = 0; i < ZNN_NUM_CORES * 2; ++i)
    {
        fns[i] = [&, i]() {
            for (auto& f : work[i])
            {
                f();
            }
        };
    }

    launcher.template launch2<true>(&(fns[0]));
}

template <const long_t batch, const long_t IF, const long_t D, const long_t H,
          const long_t W, const long_t OF, const long_t KD, const long_t KH,
          const long_t KW, typename T>
void direct_conv(T* D0, T* F, T* O)
{
    constexpr long_t batch_stride  = IF * D * H * W;
    constexpr long_t inf_stride    = D * H * W * SIMD_WIDTH;
    constexpr long_t D_stride      = H * W * SIMD_WIDTH;
    constexpr long_t H_stride      = W * SIMD_WIDTH;
    constexpr long_t K_inf_stride  = OF * KD * KH * KW;
    constexpr long_t K_of_stride   = KD * KH * KW * SIMD_WIDTH;
    constexpr long_t oD            = D - KD + 1;
    constexpr long_t oW            = W - KW + 1;
    constexpr long_t oH            = H - KH + 1;
    constexpr long_t out_B_stride  = OF * oD * oH * oW;
    constexpr long_t out_of_stride = oD * oH * oW * SIMD_WIDTH;
    constexpr long_t out_D_stride  = oH * oW * SIMD_WIDTH;
    constexpr long_t out_H_stride  = oW * SIMD_WIDTH;

    long_t inf, d, h, w;
    for (long_t b = 0; b < batch; b++)
    {
        for (inf = 0; inf < IF / SIMD_WIDTH; inf++)
        {
            for (d = 0; d < oD; d++)
            {
                for (h = 0; h < oH; h++)
                {
                    for (w = 0; w < oW; w++)
                    {
                        for (long_t outf = 0; outf < OF / SIMD_WIDTH; outf++)
                        {
                            for (long_t ofs = 0; ofs < SIMD_WIDTH; ofs++)
                                for (long_t ifs = 0; ifs < SIMD_WIDTH; ifs++)
                                {
                                    for (long_t z = 0; z < KD; z++)
                                    {
                                        for (long_t y = 0; y < KH; y++)
                                        {
                                            for (long_t x = 0; x < KW; x++)
                                            {
                                                long_t index =
                                                    outf * out_of_stride +
                                                    b * out_B_stride +
                                                    d * out_D_stride +
                                                    h * out_H_stride +
                                                    w * SIMD_WIDTH + ofs;
                                                long_t data_index =
                                                    inf * inf_stride +
                                                    b * batch_stride +
                                                    (d + z) * D_stride +
                                                    (h + y) * H_stride +
                                                    (w + x) * SIMD_WIDTH + ifs;
                                                long_t kernel_index =
                                                    (inf * SIMD_WIDTH + ifs) *
                                                        K_inf_stride +
                                                    outf * K_of_stride +
                                                    (KD - z - 1) * KH * KW *
                                                        SIMD_WIDTH +
                                                    (KH - y - 1) * KW *
                                                        SIMD_WIDTH +
                                                    (KW - x - 1) * SIMD_WIDTH +
                                                    ofs;
                                                O[index] += F[kernel_index] *
                                                            D0[data_index];
                                            }
                                        }
                                    }
                                }
                        }
                    }
                }
            }
        }
    }
}

template <const long_t batch, const long_t IF, const long_t D, const long_t H,
          const long_t W, const long_t OF, const long_t KD, const long_t KH,
          const long_t KW, typename T>
void direct_conv_parallel(T* D0, T* F, T* O)
{
    constexpr long_t batch_stride  = IF * D * H * W;
    constexpr long_t inf_stride    = D * H * W * SIMD_WIDTH;
    constexpr long_t D_stride      = H * W * SIMD_WIDTH;
    constexpr long_t H_stride      = W * SIMD_WIDTH;
    constexpr long_t K_inf_stride  = OF * KD * KH * KW;
    constexpr long_t K_of_stride   = KD * KH * KW * SIMD_WIDTH;
    constexpr long_t oD            = D - KD + 1;
    constexpr long_t oW            = W - KW + 1;
    constexpr long_t oH            = H - KH + 1;
    constexpr long_t out_B_stride  = OF * oD * oH * oW;
    constexpr long_t out_of_stride = oD * oH * oW * SIMD_WIDTH;
    constexpr long_t out_D_stride  = oH * oW * SIMD_WIDTH;
    constexpr long_t out_H_stride  = oW * SIMD_WIDTH;

    std::array<std::vector<std::function<void()>>, ZNN_NUM_CORES * 2> work;
    std::array<std::function<void()>, ZNN_NUM_CORES * 2>              fns;

    kernel_launcher launcher(ZNN_NUM_CORES, 2);

    for (long_t b = 0; b < batch; b++)
    {
        for (long_t outf = 0; outf < OF / SIMD_WIDTH; outf++)
        {
            for (long_t h = 0; h < oH; h++)
            {
                work[(outf * oH + h) % (ZNN_NUM_CORES * 2)].push_back([=]() {
                    for (long_t d = 0; d < oD; d++)
                    {
                        for (long_t w = 0; w < oW; w++)
                        {
                            for (long_t inf = 0; inf < IF / SIMD_WIDTH; inf++)
                            {
                                for (long_t ofs = 0; ofs < SIMD_WIDTH; ofs++)
                                    for (long_t ifs = 0; ifs < SIMD_WIDTH;
                                         ifs++)
                                    {
                                        for (long_t z = 0; z < KD; z++)
                                        {
                                            for (long_t y = 0; y < KH; y++)
                                            {
                                                for (long_t x = 0; x < KW; x++)
                                                {
                                                    long_t index =
                                                        outf * out_of_stride +
                                                        b * out_B_stride +
                                                        d * out_D_stride +
                                                        h * out_H_stride +
                                                        w * SIMD_WIDTH + ofs;
                                                    long_t data_index =
                                                        inf * inf_stride +
                                                        b * batch_stride +
                                                        (d + z) * D_stride +
                                                        (h + y) * H_stride +
                                                        (w + x) * SIMD_WIDTH +
                                                        ifs;
                                                    long_t kernel_index =
                                                        (inf * SIMD_WIDTH +
                                                         ifs) *
                                                            K_inf_stride +
                                                        outf * K_of_stride +
                                                        (KD - z - 1) * KH * KW *
                                                            SIMD_WIDTH +
                                                        (KH - y - 1) * KW *
                                                            SIMD_WIDTH +
                                                        (KW - x - 1) *
                                                            SIMD_WIDTH +
                                                        ofs;
                                                    O[index] +=
                                                        F[kernel_index] *
                                                        D0[data_index];
                                                }
                                            }
                                        }
                                    }
                            }
                        }
                    }
                });
            }
        }
    }

    for (int i = 0; i < ZNN_NUM_CORES * 2; ++i)
    {
        fns[i] = [&, i]() {
            for (auto& f : work[i])
            {
                f();
            }
        };
    }

    launcher.template launch2<true>(&(fns[0]));
}

// static void test_result(float* __restrict out, float* __restrict vout,
//                         long_t len)
// {
//     float max_abs_diff  = 0;
//     float max_abs_value = 0;
//     for (long_t i = 0; i < len; i++)
//     {
//         max_abs_diff  = std::max(max_abs_diff, std::abs(out[i] - vout[i]));
//         max_abs_value = std::max(max_abs_value, std::abs(out[i]));
//         if (out[i] != vout[i])
//             if (std::abs(out[i] - vout[i]) > 0.001)
//             {
//                 printf("wrong---------- :i is %d, out is %f, vout is %f\n",
//                 i,
//                        out[i], vout[i]);
//                 break;
//             }
//     }
//     std::cout << "max abs diff: " << max_abs_diff
//               << " max abs val: " << max_abs_value << std::endl;
// }
