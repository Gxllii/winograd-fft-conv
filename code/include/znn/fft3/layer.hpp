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
#pragma once

#include "znn/fft3/matrices.hpp"
#include "znn/intrin.hpp"
#include "znn/types.hpp"

namespace znn::fft3
{

using zi2::vl::dot;

template <class Layer>
struct input_transform_t
{
    using layer = Layer;

    static constexpr auto size   = layer::in_size;
    static constexpr auto stride = layer::in_stride;
    static constexpr auto t_size = layer::t_size;
    static constexpr auto k_size = layer::k_size;
    static constexpr auto m_size = layer::m_size;

    using matrices = typename layer::matrices::As;

    static constexpr auto tiles_stride =
        vec<long_t, 5>(subvec<0, 2>(stride), m_size*(subvec<2, 3>(stride)));

    static constexpr auto num_tiles = layer::num_tiles;

    static constexpr long_t tile_elements = t_size.prod();

    static inline constexpr long_t tile_offset(vec<long_t, 5> const& x)
    {
        return dot(tiles_stride, x);
    }

    static inline constexpr long_t matrix_offset(vec<long_t, 5> const& x)
    {
        return matrices::offset(x[1] * SIMD_WIDTH,
                                x[0] * num_tiles.prod() +
                                    x[2] * num_tiles[1] * num_tiles[2] +
                                    x[3] * num_tiles[2] + x[4],
                                0);
    }

    static constexpr auto fft_tile_size = layer::fft_tile_size;
};

template <class Layer>
struct output_transform_t
{
    using layer = Layer;

    static constexpr auto size      = layer::out_size;
    static constexpr auto stride    = layer::out_stride;
    static constexpr auto t_size    = layer::t_size;
    static constexpr auto k_size    = layer::k_size;
    static constexpr auto m_size    = layer::m_size;
    static constexpr auto num_tiles = layer::num_tiles;

    static constexpr auto tiles_stride =
        vec<long_t, 5>(subvec<0, 2>(stride), m_size*(subvec<2, 3>(stride)));

    static inline constexpr long_t tile_offset(vec<long_t, 5> const& x)
    {
        return dot(tiles_stride, x);
    }

    static constexpr auto fft_tile_size = layer::fft_tile_size;

    static constexpr long_t tile_elements = fft_tile_size.prod();

    static constexpr long_t flat_subtile_stride = tile_elements * SIMD_WIDTH;
    static constexpr long_t flat_tile_stride = tile_elements * SIMD_WIDTH * 3;

    static constexpr long_t effective_rows =
        ((num_tiles.prod() * size[0] + layer::row_tile_size - 1) /
         layer::row_tile_size) *
        layer::row_tile_size;

    static constexpr long_t channel_tiles = num_tiles.prod() * size[0];
    static constexpr long_t flat_channel_stride =
        flat_tile_stride * effective_rows;

    static constexpr long_t buffer_floats =
        flat_channel_stride * (size[1] / SIMD_WIDTH);

    static constexpr long_t buffer_memory = buffer_floats * sizeof(float);

    static inline constexpr long_t flat_tile_offset(long_t vc, long_t i)
    {
        return vc * flat_channel_stride + i * flat_tile_stride;
    }
};

template <class Layer>
struct kernel_transform_t
{
    using layer = Layer;

    static constexpr auto m_size = layer::m_size;
    static constexpr auto k_size = layer::k_size;
    static constexpr auto t_size = layer::t_size;

    static constexpr vec3i stride =
        vec3i{k_size[1] * k_size[2] * SIMD_WIDTH, k_size[2] * SIMD_WIDTH,
              (long_t)(SIMD_WIDTH)};

    static constexpr long_t input_channels  = layer::in_size[1];
    static constexpr long_t output_channels = layer::out_size[1];

    static constexpr long_t vofm_stride = k_size.prod() * SIMD_WIDTH;
    static constexpr long_t ifm_stride  = k_size.prod() * output_channels;

    using matrices = typename layer::matrices::Bs;

    static inline constexpr long_t tile_offset(long_t ifm, long_t vofm)
    {
        return ifm * ifm_stride + vofm * vofm_stride;
    }

    static inline constexpr long_t matrix_offset(long_t ifm, long_t vofm)
    {
        return matrices::offset(vofm * SIMD_WIDTH, ifm, 0);
    }

    static constexpr auto fft_tile_size = layer::fft_tile_size;
};

template <class InSize, class InStride, class OutSize, class OutStride,
          class TSize, class KSize, long_t RowBlock = 30, long_t MaxK = 256,
          long_t MaxKTimesN = 256 * 256>
struct layer_t
{
    static constexpr auto in_size    = InSize::value;
    static constexpr auto in_stride  = InStride::value;
    static constexpr auto out_size   = OutSize::value;
    static constexpr auto out_stride = OutStride::value;
    static constexpr auto t_size     = TSize::value;
    static constexpr auto k_size     = KSize::value;
    static constexpr auto m_size     = t_size - k_size + vec3i::one;

    static constexpr auto in_memory = in_size.prod();

    static constexpr long_t row_tile_size = RowBlock;

    static_assert(in_size[0] == out_size[0]);

    static_assert((subvec<2, 3>(in_size) - k_size + vec3i::one) % m_size ==
                  vec3i::zero);

    static_assert(in_size[1] % CACHELINE_SIZE == 0);

    static constexpr auto num_tiles =
        (subvec<2, 3>(in_size) - k_size + vec3i::one) / m_size;

    static constexpr vec3i fft_tile_size =
        vec3i{t_size[0], t_size[1], t_size[2] / 2 + 1};

    using matrices =
        matrices_t<fft_tile_size.prod(), num_tiles.prod() * in_size[0],
                   in_size[1], out_size[1], RowBlock, MaxK, MaxKTimesN>;

    using input_transform  = input_transform_t<layer_t>;
    using output_transform = output_transform_t<layer_t>;
    using kernel_transform = kernel_transform_t<layer_t>;
};

} // namespace znn::fft3
