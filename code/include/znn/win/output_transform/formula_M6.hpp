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

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<M == 6 && N == 3>::type
out_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C27 = SIMD_SET1(27.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C81 = SIMD_SET1(81.0f);
SIMD_FLOAT C32 = SIMD_SET1(32.0f);
SIMD_FLOAT C243 = SIMD_SET1(243.0f);

out[0] = SIMD_ADD(in[0],in[IS]);
out[0] = SIMD_ADD(out[0],in[IS * 2]);
out[0] = SIMD_ADD(out[0],in[IS * 3]);
out[0] = SIMD_ADD(out[0],in[IS * 4]);
out[0] = SIMD_ADD(out[0],in[IS * 5]);
out[0] = SIMD_ADD(out[0],in[IS * 6]);

out[OS] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS] = SIMD_FMADD(in[IS * 3],C2,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 4],C2,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 5],C3,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 6],C3,out[OS]);

out[OS * 2] = SIMD_ADD(in[IS],in[IS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 3],C4,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 4],C4,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 5],C9,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 6],C9,out[OS * 2]);

out[OS * 3] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS * 3] = SIMD_FMADD(in[IS * 3],C8,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 4],C8,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 5],C27,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 6],C27,out[OS * 3]);

out[OS * 4] = SIMD_ADD(in[IS],in[IS * 2]);
out[OS * 4] = SIMD_FMADD(in[IS * 3],C16,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 4],C16,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 5],C81,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 6],C81,out[OS * 4]);

out[OS * 5] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS * 5] = SIMD_FMADD(in[IS * 3],C32,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 4],C32,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 5],C243,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 6],C243,out[OS * 5]);
out[OS * 5] = SIMD_ADD(out[OS * 5],in[IS * 7]);


}

template <long_t M, long_t N, long_t IS, long_t OSTRIDE>
inline __attribute__((always_inline))
typename std::enable_if<M == 6 && N == 3>::type
out_image_1d_last(float* __restrict output, SIMD_FLOAT const* __restrict in)
{
SIMD_FLOAT out[M] __attribute__((aligned(64)));
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C27 = SIMD_SET1(27.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C81 = SIMD_SET1(81.0f);
SIMD_FLOAT C32 = SIMD_SET1(32.0f);
SIMD_FLOAT C243 = SIMD_SET1(243.0f);

out[0] = SIMD_ADD(in[0],in[IS]);
out[0] = SIMD_ADD(out[0],in[IS * 2]);
out[0] = SIMD_ADD(out[0],in[IS * 3]);
out[0] = SIMD_ADD(out[0],in[IS * 4]);
out[0] = SIMD_ADD(out[0],in[IS * 5]);
out[0] = SIMD_ADD(out[0],in[IS * 6]);

out[1] = SIMD_SUB(in[IS],in[IS * 2]);
out[1] = SIMD_FMADD(in[IS * 3],C2,out[1]);
out[1] = SIMD_FNMADD(in[IS * 4],C2,out[1]);
out[1] = SIMD_FMADD(in[IS * 5],C3,out[1]);
out[1] = SIMD_FNMADD(in[IS * 6],C3,out[1]);

out[2] = SIMD_ADD(in[IS],in[IS * 2]);
out[2] = SIMD_FMADD(in[IS * 3],C4,out[2]);
out[2] = SIMD_FMADD(in[IS * 4],C4,out[2]);
out[2] = SIMD_FMADD(in[IS * 5],C9,out[2]);
out[2] = SIMD_FMADD(in[IS * 6],C9,out[2]);

out[3] = SIMD_SUB(in[IS],in[IS * 2]);
out[3] = SIMD_FMADD(in[IS * 3],C8,out[3]);
out[3] = SIMD_FNMADD(in[IS * 4],C8,out[3]);
out[3] = SIMD_FMADD(in[IS * 5],C27,out[3]);
out[3] = SIMD_FNMADD(in[IS * 6],C27,out[3]);

out[4] = SIMD_ADD(in[IS],in[IS * 2]);
out[4] = SIMD_FMADD(in[IS * 3],C16,out[4]);
out[4] = SIMD_FMADD(in[IS * 4],C16,out[4]);
out[4] = SIMD_FMADD(in[IS * 5],C81,out[4]);
out[4] = SIMD_FMADD(in[IS * 6],C81,out[4]);

out[5] = SIMD_SUB(in[IS],in[IS * 2]);
out[5] = SIMD_FMADD(in[IS * 3],C32,out[5]);
out[5] = SIMD_FNMADD(in[IS * 4],C32,out[5]);
out[5] = SIMD_FMADD(in[IS * 5],C243,out[5]);
out[5] = SIMD_FNMADD(in[IS * 6],C243,out[5]);
out[5] = SIMD_ADD(out[5],in[IS * 7]);


#pragma unroll(M)
for (long_t i = 0; i < M; i++)
    {
		SIMD_STREAM(output + i * OSTRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<M == 6 && N == 4>::type
out_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C27 = SIMD_SET1(27.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C81 = SIMD_SET1(81.0f);
SIMD_FLOAT C256 = SIMD_SET1(256.0f);
SIMD_FLOAT C32 = SIMD_SET1(32.0f);
SIMD_FLOAT C243 = SIMD_SET1(243.0f);
SIMD_FLOAT C1024 = SIMD_SET1(1024.0f);

out[0] = SIMD_ADD(in[0],in[IS]);
out[0] = SIMD_ADD(out[0],in[IS * 2]);
out[0] = SIMD_ADD(out[0],in[IS * 3]);
out[0] = SIMD_ADD(out[0],in[IS * 4]);
out[0] = SIMD_ADD(out[0],in[IS * 5]);
out[0] = SIMD_ADD(out[0],in[IS * 6]);
out[0] = SIMD_ADD(out[0],in[IS * 7]);

out[OS] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS] = SIMD_FMADD(in[IS * 3],C2,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 4],C2,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 5],C3,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 6],C3,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 7],C4,out[OS]);

out[OS * 2] = SIMD_ADD(in[IS],in[IS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 3],C4,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 4],C4,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 5],C9,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 6],C9,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 7],C16,out[OS * 2]);

out[OS * 3] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS * 3] = SIMD_FMADD(in[IS * 3],C8,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 4],C8,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 5],C27,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 6],C27,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 7],C64,out[OS * 3]);

out[OS * 4] = SIMD_ADD(in[IS],in[IS * 2]);
out[OS * 4] = SIMD_FMADD(in[IS * 3],C16,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 4],C16,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 5],C81,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 6],C81,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 7],C256,out[OS * 4]);

out[OS * 5] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS * 5] = SIMD_FMADD(in[IS * 3],C32,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 4],C32,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 5],C243,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 6],C243,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 7],C1024,out[OS * 5]);
out[OS * 5] = SIMD_ADD(out[OS * 5],in[IS * 8]);


}

template <long_t M, long_t N, long_t IS, long_t OSTRIDE>
inline __attribute__((always_inline))
typename std::enable_if<M == 6 && N == 4>::type
out_image_1d_last(float* __restrict output, SIMD_FLOAT const* __restrict in)
{
SIMD_FLOAT out[M] __attribute__((aligned(64)));
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C27 = SIMD_SET1(27.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C81 = SIMD_SET1(81.0f);
SIMD_FLOAT C256 = SIMD_SET1(256.0f);
SIMD_FLOAT C32 = SIMD_SET1(32.0f);
SIMD_FLOAT C243 = SIMD_SET1(243.0f);
SIMD_FLOAT C1024 = SIMD_SET1(1024.0f);

out[0] = SIMD_ADD(in[0],in[IS]);
out[0] = SIMD_ADD(out[0],in[IS * 2]);
out[0] = SIMD_ADD(out[0],in[IS * 3]);
out[0] = SIMD_ADD(out[0],in[IS * 4]);
out[0] = SIMD_ADD(out[0],in[IS * 5]);
out[0] = SIMD_ADD(out[0],in[IS * 6]);
out[0] = SIMD_ADD(out[0],in[IS * 7]);

out[1] = SIMD_SUB(in[IS],in[IS * 2]);
out[1] = SIMD_FMADD(in[IS * 3],C2,out[1]);
out[1] = SIMD_FNMADD(in[IS * 4],C2,out[1]);
out[1] = SIMD_FMADD(in[IS * 5],C3,out[1]);
out[1] = SIMD_FNMADD(in[IS * 6],C3,out[1]);
out[1] = SIMD_FMADD(in[IS * 7],C4,out[1]);

out[2] = SIMD_ADD(in[IS],in[IS * 2]);
out[2] = SIMD_FMADD(in[IS * 3],C4,out[2]);
out[2] = SIMD_FMADD(in[IS * 4],C4,out[2]);
out[2] = SIMD_FMADD(in[IS * 5],C9,out[2]);
out[2] = SIMD_FMADD(in[IS * 6],C9,out[2]);
out[2] = SIMD_FMADD(in[IS * 7],C16,out[2]);

out[3] = SIMD_SUB(in[IS],in[IS * 2]);
out[3] = SIMD_FMADD(in[IS * 3],C8,out[3]);
out[3] = SIMD_FNMADD(in[IS * 4],C8,out[3]);
out[3] = SIMD_FMADD(in[IS * 5],C27,out[3]);
out[3] = SIMD_FNMADD(in[IS * 6],C27,out[3]);
out[3] = SIMD_FMADD(in[IS * 7],C64,out[3]);

out[4] = SIMD_ADD(in[IS],in[IS * 2]);
out[4] = SIMD_FMADD(in[IS * 3],C16,out[4]);
out[4] = SIMD_FMADD(in[IS * 4],C16,out[4]);
out[4] = SIMD_FMADD(in[IS * 5],C81,out[4]);
out[4] = SIMD_FMADD(in[IS * 6],C81,out[4]);
out[4] = SIMD_FMADD(in[IS * 7],C256,out[4]);

out[5] = SIMD_SUB(in[IS],in[IS * 2]);
out[5] = SIMD_FMADD(in[IS * 3],C32,out[5]);
out[5] = SIMD_FNMADD(in[IS * 4],C32,out[5]);
out[5] = SIMD_FMADD(in[IS * 5],C243,out[5]);
out[5] = SIMD_FNMADD(in[IS * 6],C243,out[5]);
out[5] = SIMD_FMADD(in[IS * 7],C1024,out[5]);
out[5] = SIMD_ADD(out[5],in[IS * 8]);


#pragma unroll(M)
for (long_t i = 0; i < M; i++)
    {
		SIMD_STREAM(output + i * OSTRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<M == 6 && N == 5>::type
out_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C27 = SIMD_SET1(27.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C81 = SIMD_SET1(81.0f);
SIMD_FLOAT C256 = SIMD_SET1(256.0f);
SIMD_FLOAT C32 = SIMD_SET1(32.0f);
SIMD_FLOAT C243 = SIMD_SET1(243.0f);
SIMD_FLOAT C1024 = SIMD_SET1(1024.0f);

out[0] = SIMD_ADD(in[0],in[IS]);
out[0] = SIMD_ADD(out[0],in[IS * 2]);
out[0] = SIMD_ADD(out[0],in[IS * 3]);
out[0] = SIMD_ADD(out[0],in[IS * 4]);
out[0] = SIMD_ADD(out[0],in[IS * 5]);
out[0] = SIMD_ADD(out[0],in[IS * 6]);
out[0] = SIMD_ADD(out[0],in[IS * 7]);
out[0] = SIMD_ADD(out[0],in[IS * 8]);

out[OS] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS] = SIMD_FMADD(in[IS * 3],C2,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 4],C2,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 5],C3,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 6],C3,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 7],C4,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 8],C4,out[OS]);

out[OS * 2] = SIMD_ADD(in[IS],in[IS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 3],C4,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 4],C4,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 5],C9,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 6],C9,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 7],C16,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 8],C16,out[OS * 2]);

out[OS * 3] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS * 3] = SIMD_FMADD(in[IS * 3],C8,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 4],C8,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 5],C27,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 6],C27,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 7],C64,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 8],C64,out[OS * 3]);

out[OS * 4] = SIMD_ADD(in[IS],in[IS * 2]);
out[OS * 4] = SIMD_FMADD(in[IS * 3],C16,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 4],C16,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 5],C81,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 6],C81,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 7],C256,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 8],C256,out[OS * 4]);

out[OS * 5] = SIMD_SUB(in[IS],in[IS * 2]);
out[OS * 5] = SIMD_FMADD(in[IS * 3],C32,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 4],C32,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 5],C243,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 6],C243,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 7],C1024,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 8],C1024,out[OS * 5]);
out[OS * 5] = SIMD_ADD(out[OS * 5],in[IS * 9]);


}

template <long_t M, long_t N, long_t IS, long_t OSTRIDE>
inline __attribute__((always_inline))
typename std::enable_if<M == 6 && N == 5>::type
out_image_1d_last(float* __restrict output, SIMD_FLOAT const* __restrict in)
{
SIMD_FLOAT out[M] __attribute__((aligned(64)));
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C27 = SIMD_SET1(27.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C81 = SIMD_SET1(81.0f);
SIMD_FLOAT C256 = SIMD_SET1(256.0f);
SIMD_FLOAT C32 = SIMD_SET1(32.0f);
SIMD_FLOAT C243 = SIMD_SET1(243.0f);
SIMD_FLOAT C1024 = SIMD_SET1(1024.0f);

out[0] = SIMD_ADD(in[0],in[IS]);
out[0] = SIMD_ADD(out[0],in[IS * 2]);
out[0] = SIMD_ADD(out[0],in[IS * 3]);
out[0] = SIMD_ADD(out[0],in[IS * 4]);
out[0] = SIMD_ADD(out[0],in[IS * 5]);
out[0] = SIMD_ADD(out[0],in[IS * 6]);
out[0] = SIMD_ADD(out[0],in[IS * 7]);
out[0] = SIMD_ADD(out[0],in[IS * 8]);

out[1] = SIMD_SUB(in[IS],in[IS * 2]);
out[1] = SIMD_FMADD(in[IS * 3],C2,out[1]);
out[1] = SIMD_FNMADD(in[IS * 4],C2,out[1]);
out[1] = SIMD_FMADD(in[IS * 5],C3,out[1]);
out[1] = SIMD_FNMADD(in[IS * 6],C3,out[1]);
out[1] = SIMD_FMADD(in[IS * 7],C4,out[1]);
out[1] = SIMD_FNMADD(in[IS * 8],C4,out[1]);

out[2] = SIMD_ADD(in[IS],in[IS * 2]);
out[2] = SIMD_FMADD(in[IS * 3],C4,out[2]);
out[2] = SIMD_FMADD(in[IS * 4],C4,out[2]);
out[2] = SIMD_FMADD(in[IS * 5],C9,out[2]);
out[2] = SIMD_FMADD(in[IS * 6],C9,out[2]);
out[2] = SIMD_FMADD(in[IS * 7],C16,out[2]);
out[2] = SIMD_FMADD(in[IS * 8],C16,out[2]);

out[3] = SIMD_SUB(in[IS],in[IS * 2]);
out[3] = SIMD_FMADD(in[IS * 3],C8,out[3]);
out[3] = SIMD_FNMADD(in[IS * 4],C8,out[3]);
out[3] = SIMD_FMADD(in[IS * 5],C27,out[3]);
out[3] = SIMD_FNMADD(in[IS * 6],C27,out[3]);
out[3] = SIMD_FMADD(in[IS * 7],C64,out[3]);
out[3] = SIMD_FNMADD(in[IS * 8],C64,out[3]);

out[4] = SIMD_ADD(in[IS],in[IS * 2]);
out[4] = SIMD_FMADD(in[IS * 3],C16,out[4]);
out[4] = SIMD_FMADD(in[IS * 4],C16,out[4]);
out[4] = SIMD_FMADD(in[IS * 5],C81,out[4]);
out[4] = SIMD_FMADD(in[IS * 6],C81,out[4]);
out[4] = SIMD_FMADD(in[IS * 7],C256,out[4]);
out[4] = SIMD_FMADD(in[IS * 8],C256,out[4]);

out[5] = SIMD_SUB(in[IS],in[IS * 2]);
out[5] = SIMD_FMADD(in[IS * 3],C32,out[5]);
out[5] = SIMD_FNMADD(in[IS * 4],C32,out[5]);
out[5] = SIMD_FMADD(in[IS * 5],C243,out[5]);
out[5] = SIMD_FNMADD(in[IS * 6],C243,out[5]);
out[5] = SIMD_FMADD(in[IS * 7],C1024,out[5]);
out[5] = SIMD_FNMADD(in[IS * 8],C1024,out[5]);
out[5] = SIMD_ADD(out[5],in[IS * 9]);


#pragma unroll(M)
for (long_t i = 0; i < M; i++)
    {
		SIMD_STREAM(output + i * OSTRIDE, out[i]);
	}
}
	
