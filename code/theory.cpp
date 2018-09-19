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
#include "znn/fft_common/codelets/codelets.hpp"
#include "znn/vec.hpp"
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using zi2::vl::vec;
using znn::long_t;

using vec3l = vec<long_t, 3>;

struct complexity
{
    long_t flops           = 0;
    long_t operations      = 0;
    long_t memory_accesses = 0;
    long_t stack_vars      = 0;
    long_t constants       = 0;
};

std::vector<std::vector<complexity>> r2cf_complexities(33);
std::vector<std::vector<complexity>> r2cb_complexities(33);
std::vector<std::vector<complexity>> c2cf_complexities(33);
std::vector<std::vector<complexity>> c2cb_complexities(33);

std::vector<complexity>              win_complexities(13);
std::vector<std::vector<complexity>> wker_complexities(10);
std::vector<std::vector<complexity>> wout_complexities(10);

inline void load_win()
{
    for (long_t i = 0; i < 10; ++i)
    {
        wker_complexities[i].resize(10);
        wout_complexities[i].resize(10);
    }

    win_complexities[4]  = {4, 4, 0, 0, 0};
    win_complexities[6]  = {20, 13, 0, 0, 3};
    win_complexities[8]  = {44, 26, 0, 0, 14};
    win_complexities[10] = {75, 43, 0, 0, 26};
    win_complexities[12] = {114, 64, 0, 0, 43};

    wout_complexities[2][3] = {4, 4, 0, 0, 0};
    wout_complexities[4][3] = {20, 14, 0, 0, 3};
    wout_complexities[6][3] = {52, 32, 0, 0, 10};
    wout_complexities[8][3] = {100, 58, 0, 0, 18};
    wout_complexities[2][5] = {10, 8, 0, 0, 1};
    wout_complexities[4][5] = {34, 22, 0, 0, 6};
    wout_complexities[6][5] = {74, 44, 0, 0, 13};
    wout_complexities[8][5] = {130, 74, 0, 0, 25};

    wker_complexities[2][3] = {7, 4, 0, 0, 1};
    wker_complexities[4][3] = {15, 9, 0, 0, 4};
    wker_complexities[6][3] = {22, 13, 0, 0, 8};
    wker_complexities[8][3] = {29, 17, 0, 0, 10};
    wker_complexities[2][5] = {21, 15, 0, 0, 6};
    wker_complexities[4][5] = {31, 22, 0, 0, 12};
    wker_complexities[6][5] = {41, 29, 0, 0, 16};
    wker_complexities[8][5] = {51, 36, 0, 0, 22};
}

template <long_t N, long_t X>
inline void load_fwd()
{

    if constexpr (X == 1)
    {
        r2cf_complexities[N].resize(33);
        c2cf_complexities[N].resize(33);
    }

    using r = znn::fft_codelets::r2cf_traits<N, X>;
    using c = znn::fft_codelets::c2cf_traits<N, X>;

    r2cf_complexities[N][X] = {r::flops, r::operations, r::memory_accesses,
                               r::stack_vars, r::constants};

    c2cf_complexities[N][X] = {c::flops, c::operations, c::memory_accesses,
                               c::stack_vars, c::constants};

    if constexpr (X < N)
    {
        load_fwd<N, X + 1>();
    }
    else
    {
        if constexpr (N < 32)
        {
            load_fwd<N + 1, 1>();
        }
    }
}

template <long_t N, long_t X>
inline void load_bwd()
{

    if constexpr (X == 0)
    {
        r2cb_complexities[N].resize(33);
        c2cb_complexities[N].resize(33);
    }

    using r = znn::fft_codelets::r2cb_traits<N, X>;
    using c = znn::fft_codelets::c2cb_traits<N, X>;

    r2cb_complexities[N][X] = {r::flops, r::operations, r::memory_accesses,
                               r::stack_vars, r::constants};

    c2cb_complexities[N][X] = {c::flops, c::operations, c::memory_accesses,
                               c::stack_vars, c::constants};

    if constexpr (X < N - 1)
    {
        load_bwd<N, X + 1>();
    }
    else
    {
        if constexpr (N < 32)
        {
            load_bwd<N + 1, 0>();
        }
    }
}

struct layer_complexity
{
    long_t input_transform_flops = 0;
    long_t input_transform_memory;

    long_t kernel_transform_flops;
    long_t kernel_transform_memory;

    long_t pointwise_flops;
    long_t pointwise_memory;
    bool   pointwise_cgemm;

    long_t output_transform_flops;
    long_t output_transform_memory;

    long_t fin, fout;
};

inline constexpr long_t ceil_div(long_t a, long_t b) { return (a + b - 1) / b; }

inline constexpr vec<long_t, 3> ceil_div(vec<long_t, 3> a, vec<long_t, 3> b)
{
    return {(a[0] + b[0] - 1) / b[0], (a[1] + b[1] - 1) / b[1],
            (a[2] + b[2] - 1) / b[2]};
}

inline constexpr vec<long_t, 3>
    fft_padded_size(vec<long_t, 3> L, vec<long_t, 3> T, vec<long_t, 3> K)
{
    return ceil_div(L - K + 1, T - K + 1) * (T - K + 1) + K - 1;
}

inline constexpr vec<long_t, 3>
    win_padded_size(vec<long_t, 3> L, vec<long_t, 3> T, vec<long_t, 3> K)
{
    return ceil_div(L - K + 1, T) * T + K - 1;
}

inline layer_complexity get_fft_layer_complexities(long_t B, long_t C1,
                                                   long_t C2, vec<long_t, 3> Ix,
                                                   vec<long_t, 3> K,
                                                   vec<long_t, 3> F)
{

    auto I = fft_padded_size(Ix, F, K);

    // std::cout << "Padded size: " << I << "\n";

    auto Tiles = (I - K + vec<long_t, 3>::one) / (F - K + vec<long_t, 3>::one);

    // std::cout << "Per image tiles: " << Tiles << "\n";

    long_t FFT2 = F[2] / 2 + 1;

    long_t in_tile_flops = F[0] * F[1] * r2cf_complexities[F[2]][F[2]].flops +
                           F[0] * FFT2 * c2cf_complexities[F[1]][F[1]].flops +
                           F[1] * FFT2 * c2cf_complexities[F[0]][F[0]].flops;

    // std::cout << "IN TILE FLOPS: " << in_tile_flops << std::endl;

    long_t out_tile_flops =
        F[0] * F[1] * r2cb_complexities[F[2]][K[2] - 1].flops +
        F[0] * FFT2 * c2cb_complexities[F[1]][K[1] - 1].flops +
        F[1] * FFT2 * c2cb_complexities[F[0]][K[0] - 1].flops;

    // std::cout << "OUT TILE FLOPS: " << out_tile_flops << std::endl;

    long_t ker_tile_flops = F[0] * F[1] * r2cf_complexities[F[2]][K[2]].flops +
                            F[0] * FFT2 * c2cf_complexities[F[1]][K[1]].flops +
                            F[1] * FFT2 * c2cf_complexities[F[0]][K[0]].flops;

    long_t input_transform_flops  = Tiles.prod() * B * C1 * in_tile_flops;
    long_t output_transform_flops = Tiles.prod() * B * C2 * out_tile_flops;
    long_t kernel_transform_flops = C1 * C2 * ker_tile_flops;
    long_t pointwise_flops =
        F[0] * F[1] * (F[2] / 2 + 1) * Tiles.prod() * B * C1 * C2 * 4 * 2;

    long_t input_transform_memory =
        I.prod() * C1 * B +
        F[0] * F[1] * (F[2] / 2 + 1) * Tiles.prod() * B * C1 * 2;

    auto O = I + vec<long_t, 3>::one - K;

    long_t output_transform_memory =
        O.prod() * C2 * B +
        F[0] * F[1] * (F[2] / 2 + 1) * Tiles.prod() * B * C2 * 2;

    long_t kernel_transform_memory =
        K.prod() * C1 * C2 + C1 * C2 * F[0] * F[1] * (F[2] / 2 + 1) * 2;

    long_t pointwise_memory =
        F[0] * F[1] * (F[2] / 2 + 1) *
        (Tiles.prod() * B * C1 * 2 + Tiles.prod() * B * C2 * 2 + C1 * C2 * 2);

    return {input_transform_flops,
            input_transform_memory * 4,
            kernel_transform_flops,
            kernel_transform_memory * 4,
            pointwise_flops,
            pointwise_memory * 4,
            true,
            output_transform_flops,
            output_transform_memory * 4,
            C1,
            C2};
}

inline layer_complexity
get_fft3_layer_complexities(long_t B, long_t C1, long_t C2, vec<long_t, 3> Ix,
                            vec<long_t, 3> K, vec<long_t, 3> F)
{
    auto I = fft_padded_size(Ix, F, K);

    auto Tiles = (I - K + vec<long_t, 3>::one) / (F - K + vec<long_t, 3>::one);

    long_t FFT2 = F[2] / 2 + 1;

    long_t in_tile_flops = F[0] * F[1] * r2cf_complexities[F[2]][F[2]].flops +
                           F[0] * FFT2 * c2cf_complexities[F[1]][F[1]].flops +
                           F[1] * FFT2 * c2cf_complexities[F[0]][F[0]].flops;

    in_tile_flops += F[0] * F[1] * FFT2;

    // std::cout << "IN TILE FLOPS: " << in_tile_flops << std::endl;

    long_t out_tile_flops =
        F[0] * F[1] * r2cb_complexities[F[2]][K[2] - 1].flops +
        F[0] * FFT2 * c2cb_complexities[F[1]][K[1] - 1].flops +
        F[1] * FFT2 * c2cb_complexities[F[0]][K[0] - 1].flops;

    out_tile_flops += F[0] * F[1] * FFT2 * 2;

    // std::cout << "OUT TILE FLOPS: " << out_tile_flops << std::endl;

    long_t ker_tile_flops = F[0] * F[1] * r2cf_complexities[F[2]][K[2]].flops +
                            F[0] * FFT2 * c2cf_complexities[F[1]][K[1]].flops +
                            F[1] * FFT2 * c2cf_complexities[F[0]][K[0]].flops;

    ker_tile_flops += F[0] * F[1] * FFT2 * 2;

    long_t input_transform_flops  = Tiles.prod() * B * C1 * in_tile_flops;
    long_t output_transform_flops = Tiles.prod() * B * C2 * out_tile_flops;
    long_t kernel_transform_flops = C1 * C2 * ker_tile_flops;
    long_t pointwise_flops =
        F[0] * F[1] * (F[2] / 2 + 1) * Tiles.prod() * B * C1 * C2 * 3 * 2;

    long_t input_transform_memory =
        I.prod() * C1 * B +
        F[0] * F[1] * (F[2] / 2 + 1) * Tiles.prod() * B * C1 * 3;

    auto O = I + vec<long_t, 3>::one - K;

    long_t output_transform_memory =
        O.prod() * C2 * B +
        F[0] * F[1] * (F[2] / 2 + 1) * Tiles.prod() * B * C2 * 3;

    long_t kernel_transform_memory =
        K.prod() * C1 * C2 + C1 * C2 * F[0] * F[1] * (F[2] / 2 + 1) * 3;

    long_t pointwise_memory =
        F[0] * F[1] * (F[2] / 2 + 1) *
        (Tiles.prod() * B * C1 * 3 + Tiles.prod() * B * C2 * 2 + C1 * C2 * 3);

    return {input_transform_flops,
            input_transform_memory * 4,
            kernel_transform_flops,
            kernel_transform_memory * 4,
            pointwise_flops,
            pointwise_memory * 4,
            false,
            output_transform_flops,
            output_transform_memory * 4,
            C1,
            C2};
}

inline void print(layer_complexity const& c)
{
    std::cout //<< "IN    FLOPS : " <<
              // static_cast<double>(c.input_transform_flops) / 1000000000 <<
              //'\n'
        << "IN    MEMORY: " << c.input_transform_memory * 4
        << '\n'
        //<< "KER   FLOPS : " << static_cast<double>(c.kernel_transform_flops) /
        // 1000000000 << '\n'
        << "KER   MEMORY: " << c.kernel_transform_memory * 4
        << '\n'
        //<< "POINTWISE FLOPS : " << static_cast<double>(c.pointwise_flops) /
        // 1000000000
        //<< '\n'
        ///<< "POINTWISE MEMORY: " << c.pointwise_memory << '\n'
        //<< "OUT   FLOPS : " << static_cast<double>(c.output_transform_flops) /
        // 1000000000 << '\n'
        << "OUT   MEMORY: " << c.output_transform_memory * 4 << std::endl
        << std::endl;

    // std::cout << "IN    RATIO: "
    //           << static_cast<long double>(c.input_transform_flops) /
    //                  c.input_transform_memory
    //           << '\n'
    //           << "KER   RATIO: "
    //           << static_cast<long double>(c.kernel_transform_flops) /
    //                  c.kernel_transform_memory
    //           << '\n'
    //           << "POINTWISE RATIO: "
    //           << static_cast<long double>(c.pointwise_flops) /
    //           c.pointwise_memory
    //           << '\n'
    //           << "OUT   RATIO: "
    //           << static_cast<long double>(c.output_transform_flops) /
    //                  c.output_transform_memory
    //           << '\n';
}

inline layer_complexity get_win_layer_complexities(long_t B, long_t C1,
                                                   long_t C2, vec<long_t, 3> Ix,
                                                   vec<long_t, 3> K,
                                                   vec<long_t, 3> M)
{
    auto I = win_padded_size(Ix, M, K);

    // std::cout << "Padded size: " << I << "\n";

    auto Tiles = (I - K + vec<long_t, 3>::one) / M;

    // std::cout << "Per image tiles: " << Tiles << "\n";

    auto F = M + K - vec<long_t, 3>::one;

    long_t in_tile_flops = F[0] * F[1] * win_complexities[F[2]].flops +
                           F[0] * F[2] * win_complexities[F[1]].flops +
                           F[1] * F[2] * win_complexities[F[0]].flops;

    // std::cout << "IN TILE FLOPS: " << in_tile_flops << std::endl;

    long_t out_tile_flops = F[0] * F[1] * wout_complexities[M[2]][K[2]].flops +
                            F[0] * F[2] * wout_complexities[M[1]][K[1]].flops +
                            F[1] * F[2] * wout_complexities[M[0]][K[0]].flops;

    // std::cout << "OUT TILE FLOPS: " << out_tile_flops << std::endl;

    long_t ker_tile_flops = F[0] * F[1] * wker_complexities[M[2]][K[2]].flops +
                            F[0] * F[2] * wker_complexities[M[1]][K[1]].flops +
                            F[1] * F[2] * wker_complexities[M[0]][K[0]].flops;

    long_t input_transform_flops  = Tiles.prod() * B * C1 * in_tile_flops;
    long_t output_transform_flops = Tiles.prod() * B * C2 * out_tile_flops;
    long_t kernel_transform_flops = C1 * C2 * ker_tile_flops;
    long_t pointwise_flops =
        F[0] * F[1] * F[2] * Tiles.prod() * B * C1 * C2 * 2;

    long_t input_transform_memory =
        I.prod() * C1 * B + F[0] * F[1] * F[2] * Tiles.prod() * B * C1;

    auto O = I + vec<long_t, 3>::one - K;

    long_t output_transform_memory =
        O.prod() * C2 * B + F[0] * F[1] * F[2] * Tiles.prod() * B * C2;

    long_t kernel_transform_memory =
        K.prod() * C1 * C2 + C1 * C2 * F[0] * F[1] * F[2];

    long_t pointwise_memory =
        F[0] * F[1] * F[2] *
        (Tiles.prod() * B * C1 + Tiles.prod() * B * C2 + C1 * C2);

    return {input_transform_flops,
            input_transform_memory * 4,
            kernel_transform_flops,
            kernel_transform_memory * 4,
            pointwise_flops,
            pointwise_memory * 4,
            false,
            output_transform_flops,
            output_transform_memory * 4,
            C1,
            C2};
}

inline long_t get_gemm_fout_block(long_t cache_size, long_t bytes_per_number)
{
    long_t fout = 64;

    while (fout * fout * bytes_per_number * 2 <= cache_size * 3 / 4)
    {
        fout *= 2;
    }

    return fout;
}

inline std::pair<long_t, long_t> get_gemm_blocks(long_t fin, long_t fout,
                                                 long_t cache_size,
                                                 long_t bytes_per_number)
{

    long_t max_fout = get_gemm_fout_block(cache_size, bytes_per_number);

    if (fout >= max_fout * 2)
    {
        while (((fout / 2) % 16 == 0) && (fout > max_fout) &&
               (fin * fout * bytes_per_number > cache_size * 3 / 4))
        {
            fout /= 2;
        }
    }

    while (((fin / 2) % 16 == 0) &&
           (fin * fout * bytes_per_number > cache_size * 3 / 4))
    {
        fin /= 2;
    }

    return {fin, fout};
}

inline long double get_time(layer_complexity const& l, long double r,
                            long_t cache_size = 512 * 1024)
{
    if (l.input_transform_memory == 0)
    {
        return std::numeric_limits<long double>::max();
    }

    auto blocks =
        get_gemm_blocks(l.fin, l.fout, cache_size, l.pointwise_cgemm ? 8 : 4);

    auto flops = blocks.first * blocks.second * (l.pointwise_cgemm ? 8 : 2);

    auto mem =
        (blocks.first + blocks.second * (blocks.second == l.fout ? 1 : 2)) *
        (l.pointwise_cgemm ? 8 : 4);

    auto pointwise_r = std::min(r, static_cast<long double>(flops) / mem);

    return std::max(static_cast<long double>(l.input_transform_flops) / r,
                    static_cast<long double>(l.input_transform_memory)) +
           std::max(static_cast<long double>(l.kernel_transform_flops) / r,
                    static_cast<long double>(l.kernel_transform_memory)) +
           std::max(static_cast<long double>(l.pointwise_flops) / pointwise_r,
                    static_cast<long double>(l.pointwise_memory)) +
           std::max(static_cast<long double>(l.output_transform_flops) / r,
                    static_cast<long double>(l.output_transform_memory));
}

inline long double get_time_inference(layer_complexity const& l, long double r,
                                      long_t cache_size = 512 * 1024)
{
    if (l.input_transform_memory == 0)
    {
        return std::numeric_limits<long double>::max();
    }

    auto blocks =
        get_gemm_blocks(l.fin, l.fout, cache_size, l.pointwise_cgemm ? 8 : 4);

    auto flops = blocks.first * blocks.second * (l.pointwise_cgemm ? 8 : 2);

    auto mem =
        (blocks.first + blocks.second * (blocks.second == l.fout ? 1 : 2)) *
        (l.pointwise_cgemm ? 8 : 4);

    auto pointwise_r = std::min(r, static_cast<long double>(flops) / mem);

    return std::max(static_cast<long double>(l.input_transform_flops) / r,
                    static_cast<long double>(l.input_transform_memory)) +
           std::max(static_cast<long double>(l.pointwise_flops) / pointwise_r,
                    static_cast<long double>(l.pointwise_memory)) +
           std::max(static_cast<long double>(l.output_transform_flops) / r,
                    static_cast<long double>(l.output_transform_memory));
}

template <typename T>
inline void analyze_single(T const& time_fn, std::string const& name, long_t B,
                           long_t C1, long_t C2, vec<long_t, 3> I,
                           vec<long_t, 3> K, long_t cache_kb = 512)
{
    std::array<layer_complexity, 20> wino;

    for (int i = 2; i < 10; ++i)
    {
        if ((i + K[1] - 1) <= 8)
        {
            wino[i + K[1] - 1] =
                get_win_layer_complexities(B, C1, C2, I, K, {1, i, i});
        }
    }

    std::array<layer_complexity, 33> fft;
    std::array<layer_complexity, 33> fft3;

    for (int i = 8; i <= 32; ++i)
    {
        fft[i]  = get_fft_layer_complexities(B, C1, C2, I, K, {1, i, i});
        fft3[i] = get_fft3_layer_complexities(B, C1, C2, I, K, {1, i, i});
    }

    for (long_t r = 1; r <= 100; ++r)
    {
        long double W6   = std::numeric_limits<long double>::max();
        long double W8   = W6;
        long double FFT  = W6;
        long double FFT3 = W6;

        std::cout << "\"" << name << "\"," << r << ",\"" << cache_kb
                  << " kb\",";

        for (long_t i = 2; i <= 6; ++i)
        {
            if (wino[i].input_transform_memory)
            {
                W6 = std::min(W6, time_fn(wino[i], r, cache_kb * 1024));
            }
        }

        for (long_t i = 2; i <= 8; ++i)
        {
            if (wino[i].input_transform_memory)
            {
                W8 = std::min(W8, time_fn(wino[i], r, cache_kb * 1024));
            }
        }

        for (long_t i = 8; i <= 32; ++i)
        {
            if (fft[i].input_transform_memory)
            {
                FFT = std::min(FFT, time_fn(fft[i], r, cache_kb * 1024));
            }
            if (fft3[i].input_transform_memory)
            {
                FFT3 = std::min(FFT3, time_fn(fft3[i], r, cache_kb * 1024));
            }
        }

        std::cout << (W8 / FFT) << "," << (W8 / FFT3) << "," << (W6 / FFT)
                  << "," << (W6 / FFT3) << "," << cache_kb << "\n";
    }
}

template <typename T>
inline void analyze(T const& fn, std::string const& name, long_t B, long_t C1,
                    long_t C2, vec<long_t, 3> I, vec<long_t, 3> K)
{
    analyze_single(fn, name, B, C1, C2, I, K, 256);
    analyze_single(fn, name, B, C1, C2, I, K, 512);
    analyze_single(fn, name, B, C1, C2, I, K, 1024);
}

int main()
{
    r2cf_complexities[0].resize(33);
    r2cf_complexities[1].resize(33);
    r2cb_complexities[0].resize(33);
    r2cb_complexities[1].resize(33);
    c2cf_complexities[0].resize(33);
    c2cf_complexities[1].resize(33);
    c2cb_complexities[0].resize(33);
    c2cb_complexities[1].resize(33);

    load_fwd<2, 1>();
    load_bwd<2, 0>();

    load_win();

    std::cout
        << "layer,ratio,cache,fft_vs_w8,fft3_vs_w8,fft_vs_w6,fft3_vs_w6,ord\n";

    auto fn = get_time;

    int x;
    std::cin >> x;

    if (x == 1)
    {
        fn = get_time_inference;
    }

    analyze(fn, "vgg-1.2", 64, 64, 64, {1, 226, 226}, {1, 3, 3});
    analyze(fn, "vgg-2.1", 64, 64, 128, {1, 114, 114}, {1, 3, 3});
    analyze(fn, "vgg-2.2", 64, 128, 128, {1, 114, 114}, {1, 3, 3});
    analyze(fn, "vgg-3.1", 64, 128, 256, {1, 58, 58}, {1, 3, 3});
    analyze(fn, "vgg-3.2", 64, 256, 256, {1, 58, 58}, {1, 3, 3});
    analyze(fn, "vgg-4.1", 64, 256, 512, {1, 30, 30}, {1, 3, 3});
    analyze(fn, "vgg-4.2", 64, 512, 512, {1, 30, 30}, {1, 3, 3});
    analyze(fn, "vgg-5", 64, 512, 512, {1, 16, 16}, {1, 3, 3});

    analyze(fn, "AlexNet-2", 128, 64, 192, {1, 31, 31}, {1, 5, 5});
    analyze(fn, "AlexNet-3", 128, 192, 384, {1, 15, 15}, {1, 3, 3});
    analyze(fn, "AlexNet-4", 128, 384, 256, {1, 15, 15}, {1, 3, 3});
    analyze(fn, "AlexNet-5", 128, 256, 256, {1, 15, 15}, {1, 3, 3});

    analyze(fn, "OverFeat-2", 128, 96, 256, {1, 28, 28}, {1, 5, 5});
    analyze(fn, "OverFeat-3", 128, 256, 384, {1, 14, 14}, {1, 3, 3});
    analyze(fn, "OverFeat-4", 128, 384, 256, {1, 14, 14}, {1, 3, 3});
    analyze(fn, "OverFeat-5", 128, 256, 256, {1, 14, 14}, {1, 3, 3});
}
