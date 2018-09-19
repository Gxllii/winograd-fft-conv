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

#include "znn/intrin.hpp"
#include "znn/types.hpp"
#include "znn/util/constexpr.hpp"

namespace znn::win
{

template <long_t NMats, long_t Rows, long_t Cols, long_t RowTileSize,
          long_t ColTileSize>
struct tiled_matrices_t
{
private:
    static_assert(Cols % 16 == 0, "Cols has to be divisible by 16");
    static_assert(Cols % ColTileSize == 0,
                  "Cols needs to be divisible by ColTIleSize");

public:
    static constexpr long_t num_matrices = NMats;

    static constexpr long_t num_rows = Rows;
    static constexpr long_t num_cols = Cols;

    static constexpr long_t tile_row_size = RowTileSize;
    static constexpr long_t tile_col_size = ColTileSize;

    static constexpr long_t num_row_tiles = div_ceil(Rows, RowTileSize);
    static constexpr long_t num_col_tiles = Cols / ColTileSize;

    static constexpr long_t actual_num_rows = num_row_tiles * RowTileSize;

    static constexpr long_t col_stride = 1;
    static constexpr long_t row_stride = ColTileSize;
    static constexpr long_t matrix_stride =
        row_stride * tile_row_size + CACHELINE_SIZE;

    static constexpr long_t tile_row_stride =
        matrix_stride * NMats + CACHELINE_SIZE;

    static constexpr long_t tile_col_stride =
        tile_row_stride * num_row_tiles + CACHELINE_SIZE;

    static constexpr long_t memory_in_floats = tile_col_stride * num_col_tiles;

    static constexpr long_t memory = memory_in_floats * sizeof(float);

    static constexpr long_t tile_offset(long_t col, long_t row, long_t mat)
    {
        return col * tile_col_stride + row * tile_row_stride +
               mat * matrix_stride;
    }

    static constexpr long_t offset(long_t col, long_t row, long_t mat)
    {
        auto r = tile_offset(col / ColTileSize, row / RowTileSize, mat) +
                 (row % RowTileSize) * row_stride + (col % ColTileSize);
        if (r >= memory_in_floats)
        {
            std::cout << "col: " << col << " ts: " << ColTileSize
                      << " tot: " << num_cols << "\n";
            std::cout << "row: " << row << " ts: " << RowTileSize
                      << " tot: " << num_rows << "\n";

            std::cout << " r: " << r << " mem: " << memory_in_floats
                      << " row_stride: " << row_stride << "\n";
        }
        return r;
    }

    static void printme()
    {
        std::cout << "num_rows: " << num_rows << "\n"
                  << "num_cols: " << num_cols << "\n"
                  << "tile_row_size: " << tile_row_size << "\n"
                  << "tile_col_size: " << tile_col_size << "\n"
                  << "num_row_tiles; " << num_row_tiles << "\n"
                  << "num_col_tiles: " << num_col_tiles << "\n"
                  << "col_stride: " << col_stride << "\n"
                  << "row_stride: " << row_stride << "\n"
                  << "matrix_stride: " << matrix_stride << "\n"
                  << "tile_row_stride: " << tile_row_stride << "\n"
                  << "tile_col_stride: " << tile_col_stride << "\n"
                  << "memory_in_floats: " << memory_in_floats << "\n"
                  << "memory: " << memory << "\n\n";
    }
};

template <long_t NMats, long_t M, long_t N, long_t K, long_t RowBlock,
          long_t MaxK = 128, long_t MaxKTimesN = 128 * 256>
struct matrices_t
{
private:
    static_assert(N % 16 == 0, "N has to be divisible by 16");
    static_assert(K % 16 == 0, "K has to be divisible by 16");

private:
    static constexpr long_t get_submatrix_k(long_t k, long_t n)
    {
        return (k / 2) % 16 ? k
                            : ((k <= MaxK || k * n <= MaxKTimesN)
                                   ? k
                                   : get_submatrix_k(k / 2, n));
    }

    static constexpr long_t get_submatrix_n(long_t k, long_t n)
    {
        return (n / 2) % 16
                   ? n
                   : (n * k <= MaxKTimesN ? n : get_submatrix_n(k, n / 2));
    }

    static constexpr long_t submatrix_k =
        (K <= 2 * MaxK) ? K : get_submatrix_k(K, N);
    static constexpr long_t submatrix_n = get_submatrix_n(submatrix_k, N);
    static constexpr long_t submatrix_m = RowBlock; // ? mabe dynamic

public:
    using As = tiled_matrices_t<NMats, M, N, submatrix_m, submatrix_n>;
    using Bs = tiled_matrices_t<NMats, N, K, submatrix_n, submatrix_k>;
    using Cs = tiled_matrices_t<NMats, M, K, submatrix_m, submatrix_k>;

    static void printme()
    {
        std::cout << "submatrix_m: " << submatrix_m << "\n"
                  << "submatrix_n: " << submatrix_n << "\n"
                  << "submatrix_k: " << submatrix_k << "\n\n";
        As::printme();
        Bs::printme();
        Cs::printme();
    }
};

} // namespace znn::win
