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
typename std::enable_if<(M + N - 1) == 4>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 

out[0] = SIMD_SUB(in[0],in[IS * 2]);

out[OS] = SIMD_ADD(in[IS],in[IS * 2]);

out[OS * 2] = SIMD_SUB(in[IS * 2],in[IS]);

out[OS * 3] = SIMD_SUB(in[IS * 3],in[IS]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 4>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));


out[0] = SIMD_SUB(in[0],in[IS * 2]);

out[1] = SIMD_ADD(in[IS],in[IS * 2]);

out[2] = SIMD_SUB(in[IS * 2],in[IS]);

out[3] = SIMD_SUB(in[IS * 3],in[IS]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 5>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);

out[0] = SIMD_FMADD(in[0],C2,in[IS * 3]);
out[0] = SIMD_SUB(out[0],in[IS]);
out[0] = SIMD_FNMADD(in[IS * 2],C2,out[0]);

out[OS] = SIMD_FNMADD(in[IS],C2,in[IS * 3]);
out[OS] = SIMD_SUB(out[OS],in[IS * 2]);

out[OS * 2] = SIMD_FMADD(in[IS],C2,in[IS * 3]);
out[OS * 2] = SIMD_FNMADD(in[IS * 2],C3,out[OS * 2]);

out[OS * 3] = SIMD_SUB(in[IS * 3],in[IS]);

out[OS * 4] = SIMD_FMADD(in[IS],C2,in[IS * 4]);
out[OS * 4] = SIMD_SUB(out[OS * 4],in[IS * 2]);
out[OS * 4] = SIMD_FNMADD(in[IS * 3],C2,out[OS * 4]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 5>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);

out[0] = SIMD_FMADD(in[0],C2,in[IS * 3]);
out[0] = SIMD_SUB(out[0],in[IS]);
out[0] = SIMD_FNMADD(in[IS * 2],C2,out[0]);

out[1] = SIMD_FNMADD(in[IS],C2,in[IS * 3]);
out[1] = SIMD_SUB(out[1],in[IS * 2]);

out[2] = SIMD_FMADD(in[IS],C2,in[IS * 3]);
out[2] = SIMD_FNMADD(in[IS * 2],C3,out[2]);

out[3] = SIMD_SUB(in[IS * 3],in[IS]);

out[4] = SIMD_FMADD(in[IS],C2,in[IS * 4]);
out[4] = SIMD_SUB(out[4],in[IS * 2]);
out[4] = SIMD_FNMADD(in[IS * 3],C2,out[4]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 6>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);

out[0] = SIMD_FMADD(in[0],C4,in[IS * 4]);
out[0] = SIMD_FNMADD(in[IS * 2],C5,out[0]);

SIMD_FLOAT V12S = SIMD_FNMADD(in[IS * 2],C4,in[IS * 4]);

SIMD_FLOAT V12R = SIMD_FNMADD(in[IS],C4,in[IS * 3]);

out[OS] = SIMD_ADD(V12S,V12R);

out[OS * 2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_SUB(in[IS * 4],in[IS * 2]);

SIMD_FLOAT V34R = SIMD_MUL(in[IS * 3],C2);
V34R = SIMD_FNMADD(in[IS],C2,V34R);

out[OS * 3] = SIMD_ADD(V34S,V34R);

out[OS * 4] = SIMD_SUB(V34S,V34R);
 
out[OS * 5] = SIMD_FMADD(in[IS],C4,in[IS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 3],C5,out[OS * 5]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 6>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);

out[0] = SIMD_FMADD(in[0],C4,in[IS * 4]);
out[0] = SIMD_FNMADD(in[IS * 2],C5,out[0]);

SIMD_FLOAT V12S = SIMD_FNMADD(in[IS * 2],C4,in[IS * 4]);

SIMD_FLOAT V12R = SIMD_FNMADD(in[IS],C4,in[IS * 3]);

out[1] = SIMD_ADD(V12S,V12R);

out[2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_SUB(in[IS * 4],in[IS * 2]);

SIMD_FLOAT V34R = SIMD_MUL(in[IS * 3],C2);
V34R = SIMD_FNMADD(in[IS],C2,V34R);

out[3] = SIMD_ADD(V34S,V34R);

out[4] = SIMD_SUB(V34S,V34R);
 
out[5] = SIMD_FMADD(in[IS],C4,in[IS * 5]);
out[5] = SIMD_FNMADD(in[IS * 3],C5,out[5]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 7>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C12 = SIMD_SET1(12.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C15 = SIMD_SET1(15.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C7 = SIMD_SET1(7.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C6 = SIMD_SET1(6.0f);

out[0] = SIMD_FMSUB(in[0],C12,in[IS * 5]);
out[0] = SIMD_FNMADD(in[IS],C4,out[0]);
out[0] = SIMD_FNMADD(in[IS * 2],C15,out[0]);
out[0] = SIMD_FMADD(in[IS * 3],C5,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C3,out[0]);

out[OS] = SIMD_FMADD(in[IS],C12,in[IS * 5]);
out[OS] = SIMD_FMADD(in[IS * 2],C8,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 3],C7,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 4],C2,out[OS]);

out[OS * 2] = SIMD_FNMADD(in[IS],C12,in[IS * 5]);
out[OS * 2] = SIMD_FMADD(in[IS * 2],C16,out[OS * 2]);
out[OS * 2] = SIMD_SUB(out[OS * 2],in[IS * 3]);
out[OS * 2] = SIMD_FNMADD(in[IS * 4],C4,out[OS * 2]);

out[OS * 3] = SIMD_FMADD(in[IS],C6,in[IS * 2]);
out[OS * 3] = SIMD_FNMADD(in[IS * 3],C7,out[OS * 3]);
out[OS * 3] = SIMD_SUB(out[OS * 3],in[IS * 4]);
out[OS * 3] = SIMD_ADD(out[OS * 3],in[IS * 5]);

out[OS * 4] = SIMD_FNMADD(in[IS],C6,in[IS * 5]);
out[OS * 4] = SIMD_FMADD(in[IS * 2],C5,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 3],C5,out[OS * 4]);
out[OS * 4] = SIMD_FNMADD(in[IS * 4],C5,out[OS * 4]);

out[OS * 5] = SIMD_FMADD(in[IS],C4,in[IS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 3],C5,out[OS * 5]);

out[OS * 6] = SIMD_FNMADD(in[IS],C12,in[IS * 6]);
out[OS * 6] = SIMD_FMADD(in[IS * 2],C4,out[OS * 6]);
out[OS * 6] = SIMD_FMADD(in[IS * 3],C15,out[OS * 6]);
out[OS * 6] = SIMD_FNMADD(in[IS * 4],C5,out[OS * 6]);
out[OS * 6] = SIMD_FNMADD(in[IS * 5],C3,out[OS * 6]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 7>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C12 = SIMD_SET1(12.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C15 = SIMD_SET1(15.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C8 = SIMD_SET1(8.0f);
SIMD_FLOAT C7 = SIMD_SET1(7.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C6 = SIMD_SET1(6.0f);

out[0] = SIMD_FMSUB(in[0],C12,in[IS * 5]);
out[0] = SIMD_FNMADD(in[IS],C4,out[0]);
out[0] = SIMD_FNMADD(in[IS * 2],C15,out[0]);
out[0] = SIMD_FMADD(in[IS * 3],C5,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C3,out[0]);

out[1] = SIMD_FMADD(in[IS],C12,in[IS * 5]);
out[1] = SIMD_FMADD(in[IS * 2],C8,out[1]);
out[1] = SIMD_FNMADD(in[IS * 3],C7,out[1]);
out[1] = SIMD_FNMADD(in[IS * 4],C2,out[1]);

out[2] = SIMD_FNMADD(in[IS],C12,in[IS * 5]);
out[2] = SIMD_FMADD(in[IS * 2],C16,out[2]);
out[2] = SIMD_SUB(out[2],in[IS * 3]);
out[2] = SIMD_FNMADD(in[IS * 4],C4,out[2]);

out[3] = SIMD_FMADD(in[IS],C6,in[IS * 2]);
out[3] = SIMD_FNMADD(in[IS * 3],C7,out[3]);
out[3] = SIMD_SUB(out[3],in[IS * 4]);
out[3] = SIMD_ADD(out[3],in[IS * 5]);

out[4] = SIMD_FNMADD(in[IS],C6,in[IS * 5]);
out[4] = SIMD_FMADD(in[IS * 2],C5,out[4]);
out[4] = SIMD_FMADD(in[IS * 3],C5,out[4]);
out[4] = SIMD_FNMADD(in[IS * 4],C5,out[4]);

out[5] = SIMD_FMADD(in[IS],C4,in[IS * 5]);
out[5] = SIMD_FNMADD(in[IS * 3],C5,out[5]);

out[6] = SIMD_FNMADD(in[IS],C12,in[IS * 6]);
out[6] = SIMD_FMADD(in[IS * 2],C4,out[6]);
out[6] = SIMD_FMADD(in[IS * 3],C15,out[6]);
out[6] = SIMD_FNMADD(in[IS * 4],C5,out[6]);
out[6] = SIMD_FNMADD(in[IS * 5],C3,out[6]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 8>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C36 = SIMD_SET1(36.0f);
SIMD_FLOAT C49 = SIMD_SET1(49.0f);
SIMD_FLOAT C14 = SIMD_SET1(14.0f);
SIMD_FLOAT C13 = SIMD_SET1(13.0f);
SIMD_FLOAT C18 = SIMD_SET1(18.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C20 = SIMD_SET1(20.0f);
SIMD_FLOAT C10 = SIMD_SET1(10.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C12 = SIMD_SET1(12.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C15 = SIMD_SET1(15.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);

out[0] = SIMD_FMSUB(in[0],C36,in[IS * 6]);
out[0] = SIMD_FNMADD(in[IS * 2],C49,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C14,out[0]);

SIMD_FLOAT V12S = SIMD_FMADD(in[IS * 2],C36,in[IS * 6]);
V12S = SIMD_FNMADD(in[IS * 4],C13,V12S);

SIMD_FLOAT V12R = SIMD_FMADD(in[IS],C36,in[IS * 5]);
V12R = SIMD_FNMADD(in[IS * 3],C13,V12R);

out[OS] = SIMD_ADD(V12S,V12R);

out[OS * 2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_FMADD(in[IS * 2],C9,in[IS * 6]);
V34S = SIMD_FNMADD(in[IS * 4],C10,V34S);

SIMD_FLOAT V34R = SIMD_MUL(in[IS],C18);
V34R = SIMD_FNMADD(in[IS * 3],C20,V34R);
V34R = SIMD_FMADD(in[IS * 5],C2,V34R);

out[OS * 3] = SIMD_ADD(V34S,V34R);

out[OS * 4] = SIMD_SUB(V34S,V34R);
 
SIMD_FLOAT V56S = SIMD_FMADD(in[IS * 2],C4,in[IS * 6]);
V56S = SIMD_FNMADD(in[IS * 4],C5,V56S);

SIMD_FLOAT V56R = SIMD_MUL(in[IS],C12);
V56R = SIMD_FNMADD(in[IS * 3],C15,V56R);
V56R = SIMD_FMADD(in[IS * 5],C3,V56R);

out[OS * 5] = SIMD_ADD(V56S,V56R);

out[OS * 6] = SIMD_SUB(V56S,V56R);
 
out[OS * 7] = SIMD_FNMADD(in[IS],C36,in[IS * 7]);
out[OS * 7] = SIMD_FMADD(in[IS * 3],C49,out[OS * 7]);
out[OS * 7] = SIMD_FNMADD(in[IS * 5],C14,out[OS * 7]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 8>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C36 = SIMD_SET1(36.0f);
SIMD_FLOAT C49 = SIMD_SET1(49.0f);
SIMD_FLOAT C14 = SIMD_SET1(14.0f);
SIMD_FLOAT C13 = SIMD_SET1(13.0f);
SIMD_FLOAT C18 = SIMD_SET1(18.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C20 = SIMD_SET1(20.0f);
SIMD_FLOAT C10 = SIMD_SET1(10.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C12 = SIMD_SET1(12.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C15 = SIMD_SET1(15.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);

out[0] = SIMD_FMSUB(in[0],C36,in[IS * 6]);
out[0] = SIMD_FNMADD(in[IS * 2],C49,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C14,out[0]);

SIMD_FLOAT V12S = SIMD_FMADD(in[IS * 2],C36,in[IS * 6]);
V12S = SIMD_FNMADD(in[IS * 4],C13,V12S);

SIMD_FLOAT V12R = SIMD_FMADD(in[IS],C36,in[IS * 5]);
V12R = SIMD_FNMADD(in[IS * 3],C13,V12R);

out[1] = SIMD_ADD(V12S,V12R);

out[2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_FMADD(in[IS * 2],C9,in[IS * 6]);
V34S = SIMD_FNMADD(in[IS * 4],C10,V34S);

SIMD_FLOAT V34R = SIMD_MUL(in[IS],C18);
V34R = SIMD_FNMADD(in[IS * 3],C20,V34R);
V34R = SIMD_FMADD(in[IS * 5],C2,V34R);

out[3] = SIMD_ADD(V34S,V34R);

out[4] = SIMD_SUB(V34S,V34R);
 
SIMD_FLOAT V56S = SIMD_FMADD(in[IS * 2],C4,in[IS * 6]);
V56S = SIMD_FNMADD(in[IS * 4],C5,V56S);

SIMD_FLOAT V56R = SIMD_MUL(in[IS],C12);
V56R = SIMD_FNMADD(in[IS * 3],C15,V56R);
V56R = SIMD_FMADD(in[IS * 5],C3,V56R);

out[5] = SIMD_ADD(V56S,V56R);

out[6] = SIMD_SUB(V56S,V56R);
 
out[7] = SIMD_FNMADD(in[IS],C36,in[IS * 7]);
out[7] = SIMD_FMADD(in[IS * 3],C49,out[7]);
out[7] = SIMD_FNMADD(in[IS * 5],C14,out[7]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 9>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C144 = SIMD_SET1(144.0f);
SIMD_FLOAT C36 = SIMD_SET1(36.0f);
SIMD_FLOAT C196 = SIMD_SET1(196.0f);
SIMD_FLOAT C49 = SIMD_SET1(49.0f);
SIMD_FLOAT C56 = SIMD_SET1(56.0f);
SIMD_FLOAT C14 = SIMD_SET1(14.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C108 = SIMD_SET1(108.0f);
SIMD_FLOAT C88 = SIMD_SET1(88.0f);
SIMD_FLOAT C39 = SIMD_SET1(39.0f);
SIMD_FLOAT C17 = SIMD_SET1(17.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C180 = SIMD_SET1(180.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C65 = SIMD_SET1(65.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C72 = SIMD_SET1(72.0f);
SIMD_FLOAT C18 = SIMD_SET1(18.0f);
SIMD_FLOAT C89 = SIMD_SET1(89.0f);
SIMD_FLOAT C20 = SIMD_SET1(20.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C54 = SIMD_SET1(54.0f);
SIMD_FLOAT C71 = SIMD_SET1(71.0f);
SIMD_FLOAT C60 = SIMD_SET1(60.0f);
SIMD_FLOAT C6 = SIMD_SET1(6.0f);
SIMD_FLOAT C48 = SIMD_SET1(48.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C28 = SIMD_SET1(28.0f);
SIMD_FLOAT C35 = SIMD_SET1(35.0f);
SIMD_FLOAT C7 = SIMD_SET1(7.0f);

out[0] = SIMD_FMADD(in[0],C144,in[IS * 7]);
out[0] = SIMD_FNMADD(in[IS],C36,out[0]);
out[0] = SIMD_FNMADD(in[IS * 2],C196,out[0]);
out[0] = SIMD_FMADD(in[IS * 3],C49,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C56,out[0]);
out[0] = SIMD_FNMADD(in[IS * 5],C14,out[0]);
out[0] = SIMD_FNMADD(in[IS * 6],C4,out[0]);

out[OS] = SIMD_FNMADD(in[IS],C144,in[IS * 7]);
out[OS] = SIMD_FNMADD(in[IS * 2],C108,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 3],C88,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 4],C39,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 5],C17,out[OS]);
out[OS] = SIMD_FNMADD(in[IS * 6],C3,out[OS]);

out[OS * 2] = SIMD_FMADD(in[IS],C144,in[IS * 7]);
out[OS * 2] = SIMD_FNMADD(in[IS * 2],C180,out[OS * 2]);
out[OS * 2] = SIMD_FNMADD(in[IS * 3],C16,out[OS * 2]);
out[OS * 2] = SIMD_FMADD(in[IS * 4],C65,out[OS * 2]);
out[OS * 2] = SIMD_FNMADD(in[IS * 5],C9,out[OS * 2]);
out[OS * 2] = SIMD_FNMADD(in[IS * 6],C5,out[OS * 2]);

out[OS * 3] = SIMD_FNMADD(in[IS],C72,in[IS * 7]);
out[OS * 3] = SIMD_FNMADD(in[IS * 2],C18,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 3],C89,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 4],C20,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 5],C18,out[OS * 3]);
out[OS * 3] = SIMD_FNMADD(in[IS * 6],C2,out[OS * 3]);

out[OS * 4] = SIMD_FMADD(in[IS],C72,in[IS * 7]);
out[OS * 4] = SIMD_FNMADD(in[IS * 2],C54,out[OS * 4]);
out[OS * 4] = SIMD_FNMADD(in[IS * 3],C71,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 4],C60,out[OS * 4]);
out[OS * 4] = SIMD_FNMADD(in[IS * 5],C2,out[OS * 4]);
out[OS * 4] = SIMD_FNMADD(in[IS * 6],C6,out[OS * 4]);

out[OS * 5] = SIMD_FNMADD(in[IS],C48,in[IS * 7]);
out[OS * 5] = SIMD_FNMADD(in[IS * 2],C4,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 3],C64,out[OS * 5]);
out[OS * 5] = SIMD_FMADD(in[IS * 4],C5,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS * 5],C17,out[OS * 5]);
out[OS * 5] = SIMD_SUB(out[OS * 5],in[IS * 6]);

out[OS * 6] = SIMD_FMADD(in[IS],C48,in[IS * 7]);
out[OS * 6] = SIMD_FNMADD(in[IS * 2],C28,out[OS * 6]);
out[OS * 6] = SIMD_FNMADD(in[IS * 3],C56,out[OS * 6]);
out[OS * 6] = SIMD_FMADD(in[IS * 4],C35,out[OS * 6]);
out[OS * 6] = SIMD_FMADD(in[IS * 5],C7,out[OS * 6]);
out[OS * 6] = SIMD_FNMADD(in[IS * 6],C7,out[OS * 6]);

out[OS * 7] = SIMD_FNMADD(in[IS],C36,in[IS * 7]);
out[OS * 7] = SIMD_FMADD(in[IS * 3],C49,out[OS * 7]);
out[OS * 7] = SIMD_FNMADD(in[IS * 5],C14,out[OS * 7]);

out[OS * 8] = SIMD_FMADD(in[IS],C144,in[IS * 8]);
out[OS * 8] = SIMD_FNMADD(in[IS * 2],C36,out[OS * 8]);
out[OS * 8] = SIMD_FNMADD(in[IS * 3],C196,out[OS * 8]);
out[OS * 8] = SIMD_FMADD(in[IS * 4],C49,out[OS * 8]);
out[OS * 8] = SIMD_FMADD(in[IS * 5],C56,out[OS * 8]);
out[OS * 8] = SIMD_FNMADD(in[IS * 6],C14,out[OS * 8]);
out[OS * 8] = SIMD_FNMADD(in[IS * 7],C4,out[OS * 8]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 9>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C144 = SIMD_SET1(144.0f);
SIMD_FLOAT C36 = SIMD_SET1(36.0f);
SIMD_FLOAT C196 = SIMD_SET1(196.0f);
SIMD_FLOAT C49 = SIMD_SET1(49.0f);
SIMD_FLOAT C56 = SIMD_SET1(56.0f);
SIMD_FLOAT C14 = SIMD_SET1(14.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);
SIMD_FLOAT C108 = SIMD_SET1(108.0f);
SIMD_FLOAT C88 = SIMD_SET1(88.0f);
SIMD_FLOAT C39 = SIMD_SET1(39.0f);
SIMD_FLOAT C17 = SIMD_SET1(17.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C180 = SIMD_SET1(180.0f);
SIMD_FLOAT C16 = SIMD_SET1(16.0f);
SIMD_FLOAT C65 = SIMD_SET1(65.0f);
SIMD_FLOAT C9 = SIMD_SET1(9.0f);
SIMD_FLOAT C5 = SIMD_SET1(5.0f);
SIMD_FLOAT C72 = SIMD_SET1(72.0f);
SIMD_FLOAT C18 = SIMD_SET1(18.0f);
SIMD_FLOAT C89 = SIMD_SET1(89.0f);
SIMD_FLOAT C20 = SIMD_SET1(20.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C54 = SIMD_SET1(54.0f);
SIMD_FLOAT C71 = SIMD_SET1(71.0f);
SIMD_FLOAT C60 = SIMD_SET1(60.0f);
SIMD_FLOAT C6 = SIMD_SET1(6.0f);
SIMD_FLOAT C48 = SIMD_SET1(48.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C28 = SIMD_SET1(28.0f);
SIMD_FLOAT C35 = SIMD_SET1(35.0f);
SIMD_FLOAT C7 = SIMD_SET1(7.0f);

out[0] = SIMD_FMADD(in[0],C144,in[IS * 7]);
out[0] = SIMD_FNMADD(in[IS],C36,out[0]);
out[0] = SIMD_FNMADD(in[IS * 2],C196,out[0]);
out[0] = SIMD_FMADD(in[IS * 3],C49,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C56,out[0]);
out[0] = SIMD_FNMADD(in[IS * 5],C14,out[0]);
out[0] = SIMD_FNMADD(in[IS * 6],C4,out[0]);

out[1] = SIMD_FNMADD(in[IS],C144,in[IS * 7]);
out[1] = SIMD_FNMADD(in[IS * 2],C108,out[1]);
out[1] = SIMD_FMADD(in[IS * 3],C88,out[1]);
out[1] = SIMD_FMADD(in[IS * 4],C39,out[1]);
out[1] = SIMD_FNMADD(in[IS * 5],C17,out[1]);
out[1] = SIMD_FNMADD(in[IS * 6],C3,out[1]);

out[2] = SIMD_FMADD(in[IS],C144,in[IS * 7]);
out[2] = SIMD_FNMADD(in[IS * 2],C180,out[2]);
out[2] = SIMD_FNMADD(in[IS * 3],C16,out[2]);
out[2] = SIMD_FMADD(in[IS * 4],C65,out[2]);
out[2] = SIMD_FNMADD(in[IS * 5],C9,out[2]);
out[2] = SIMD_FNMADD(in[IS * 6],C5,out[2]);

out[3] = SIMD_FNMADD(in[IS],C72,in[IS * 7]);
out[3] = SIMD_FNMADD(in[IS * 2],C18,out[3]);
out[3] = SIMD_FMADD(in[IS * 3],C89,out[3]);
out[3] = SIMD_FMADD(in[IS * 4],C20,out[3]);
out[3] = SIMD_FNMADD(in[IS * 5],C18,out[3]);
out[3] = SIMD_FNMADD(in[IS * 6],C2,out[3]);

out[4] = SIMD_FMADD(in[IS],C72,in[IS * 7]);
out[4] = SIMD_FNMADD(in[IS * 2],C54,out[4]);
out[4] = SIMD_FNMADD(in[IS * 3],C71,out[4]);
out[4] = SIMD_FMADD(in[IS * 4],C60,out[4]);
out[4] = SIMD_FNMADD(in[IS * 5],C2,out[4]);
out[4] = SIMD_FNMADD(in[IS * 6],C6,out[4]);

out[5] = SIMD_FNMADD(in[IS],C48,in[IS * 7]);
out[5] = SIMD_FNMADD(in[IS * 2],C4,out[5]);
out[5] = SIMD_FMADD(in[IS * 3],C64,out[5]);
out[5] = SIMD_FMADD(in[IS * 4],C5,out[5]);
out[5] = SIMD_FNMADD(in[IS * 5],C17,out[5]);
out[5] = SIMD_SUB(out[5],in[IS * 6]);

out[6] = SIMD_FMADD(in[IS],C48,in[IS * 7]);
out[6] = SIMD_FNMADD(in[IS * 2],C28,out[6]);
out[6] = SIMD_FNMADD(in[IS * 3],C56,out[6]);
out[6] = SIMD_FMADD(in[IS * 4],C35,out[6]);
out[6] = SIMD_FMADD(in[IS * 5],C7,out[6]);
out[6] = SIMD_FNMADD(in[IS * 6],C7,out[6]);

out[7] = SIMD_FNMADD(in[IS],C36,in[IS * 7]);
out[7] = SIMD_FMADD(in[IS * 3],C49,out[7]);
out[7] = SIMD_FNMADD(in[IS * 5],C14,out[7]);

out[8] = SIMD_FMADD(in[IS],C144,in[IS * 8]);
out[8] = SIMD_FNMADD(in[IS * 2],C36,out[8]);
out[8] = SIMD_FNMADD(in[IS * 3],C196,out[8]);
out[8] = SIMD_FMADD(in[IS * 4],C49,out[8]);
out[8] = SIMD_FMADD(in[IS * 5],C56,out[8]);
out[8] = SIMD_FNMADD(in[IS * 6],C14,out[8]);
out[8] = SIMD_FNMADD(in[IS * 7],C4,out[8]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	

template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 10>::type
transform_image_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C576 = SIMD_SET1(576.0f);
SIMD_FLOAT C820 = SIMD_SET1(820.0f);
SIMD_FLOAT C273 = SIMD_SET1(273.0f);
SIMD_FLOAT C30 = SIMD_SET1(30.0f);
SIMD_FLOAT C244 = SIMD_SET1(244.0f);
SIMD_FLOAT C29 = SIMD_SET1(29.0f);
SIMD_FLOAT C288 = SIMD_SET1(288.0f);
SIMD_FLOAT C144 = SIMD_SET1(144.0f);
SIMD_FLOAT C338 = SIMD_SET1(338.0f);
SIMD_FLOAT C169 = SIMD_SET1(169.0f);
SIMD_FLOAT C52 = SIMD_SET1(52.0f);
SIMD_FLOAT C26 = SIMD_SET1(26.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C192 = SIMD_SET1(192.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C252 = SIMD_SET1(252.0f);
SIMD_FLOAT C84 = SIMD_SET1(84.0f);
SIMD_FLOAT C63 = SIMD_SET1(63.0f);
SIMD_FLOAT C21 = SIMD_SET1(21.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C36 = SIMD_SET1(36.0f);
SIMD_FLOAT C196 = SIMD_SET1(196.0f);
SIMD_FLOAT C49 = SIMD_SET1(49.0f);
SIMD_FLOAT C56 = SIMD_SET1(56.0f);
SIMD_FLOAT C14 = SIMD_SET1(14.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);

out[0] = SIMD_FMADD(in[0],C576,in[IS * 8]);
out[0] = SIMD_FNMADD(in[IS * 2],C820,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C273,out[0]);
out[0] = SIMD_FNMADD(in[IS * 6],C30,out[0]);

SIMD_FLOAT V12S = SIMD_FNMADD(in[IS * 2],C576,in[IS * 8]);
V12S = SIMD_FMADD(in[IS * 4],C244,V12S);
V12S = SIMD_FNMADD(in[IS * 6],C29,V12S);

SIMD_FLOAT V12R = SIMD_FNMADD(in[IS],C576,in[IS * 7]);
V12R = SIMD_FMADD(in[IS * 3],C244,V12R);
V12R = SIMD_FNMADD(in[IS * 5],C29,V12R);

out[OS] = SIMD_ADD(V12S,V12R);

out[OS * 2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_FNMADD(in[IS * 2],C144,in[IS * 8]);
V34S = SIMD_FMADD(in[IS * 4],C169,V34S);
V34S = SIMD_FNMADD(in[IS * 6],C26,V34S);

SIMD_FLOAT V34R = SIMD_MUL(in[IS * 3],C338);
V34R = SIMD_FNMADD(in[IS],C288,V34R);
V34R = SIMD_FNMADD(in[IS * 5],C52,V34R);
V34R = SIMD_FMADD(in[IS * 7],C2,V34R);

out[OS * 3] = SIMD_ADD(V34S,V34R);

out[OS * 4] = SIMD_SUB(V34S,V34R);
 
SIMD_FLOAT V56S = SIMD_FNMADD(in[IS * 2],C64,in[IS * 8]);
V56S = SIMD_FMADD(in[IS * 4],C84,V56S);
V56S = SIMD_FNMADD(in[IS * 6],C21,V56S);

SIMD_FLOAT V56R = SIMD_MUL(in[IS * 3],C252);
V56R = SIMD_FNMADD(in[IS],C192,V56R);
V56R = SIMD_FNMADD(in[IS * 5],C63,V56R);
V56R = SIMD_FMADD(in[IS * 7],C3,V56R);

out[OS * 5] = SIMD_ADD(V56S,V56R);

out[OS * 6] = SIMD_SUB(V56S,V56R);
 
SIMD_FLOAT V78S = SIMD_FNMADD(in[IS * 2],C36,in[IS * 8]);
V78S = SIMD_FMADD(in[IS * 4],C49,V78S);
V78S = SIMD_FNMADD(in[IS * 6],C14,V78S);

SIMD_FLOAT V78R = SIMD_MUL(in[IS * 3],C196);
V78R = SIMD_FNMADD(in[IS],C144,V78R);
V78R = SIMD_FNMADD(in[IS * 5],C56,V78R);
V78R = SIMD_FMADD(in[IS * 7],C4,V78R);

out[OS * 7] = SIMD_ADD(V78S,V78R);

out[OS * 8] = SIMD_SUB(V78S,V78R);
 
out[OS * 9] = SIMD_FMADD(in[IS],C576,in[IS * 9]);
out[OS * 9] = SIMD_FNMADD(in[IS * 3],C820,out[OS * 9]);
out[OS * 9] = SIMD_FMADD(in[IS * 5],C273,out[OS * 9]);
out[OS * 9] = SIMD_FNMADD(in[IS * 7],C30,out[OS * 9]);


 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<(M + N - 1) == 10>::type
transform_image_1d_last(float* __restrict output,
      SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C576 = SIMD_SET1(576.0f);
SIMD_FLOAT C820 = SIMD_SET1(820.0f);
SIMD_FLOAT C273 = SIMD_SET1(273.0f);
SIMD_FLOAT C30 = SIMD_SET1(30.0f);
SIMD_FLOAT C244 = SIMD_SET1(244.0f);
SIMD_FLOAT C29 = SIMD_SET1(29.0f);
SIMD_FLOAT C288 = SIMD_SET1(288.0f);
SIMD_FLOAT C144 = SIMD_SET1(144.0f);
SIMD_FLOAT C338 = SIMD_SET1(338.0f);
SIMD_FLOAT C169 = SIMD_SET1(169.0f);
SIMD_FLOAT C52 = SIMD_SET1(52.0f);
SIMD_FLOAT C26 = SIMD_SET1(26.0f);
SIMD_FLOAT C2 = SIMD_SET1(2.0f);
SIMD_FLOAT C192 = SIMD_SET1(192.0f);
SIMD_FLOAT C64 = SIMD_SET1(64.0f);
SIMD_FLOAT C252 = SIMD_SET1(252.0f);
SIMD_FLOAT C84 = SIMD_SET1(84.0f);
SIMD_FLOAT C63 = SIMD_SET1(63.0f);
SIMD_FLOAT C21 = SIMD_SET1(21.0f);
SIMD_FLOAT C3 = SIMD_SET1(3.0f);
SIMD_FLOAT C36 = SIMD_SET1(36.0f);
SIMD_FLOAT C196 = SIMD_SET1(196.0f);
SIMD_FLOAT C49 = SIMD_SET1(49.0f);
SIMD_FLOAT C56 = SIMD_SET1(56.0f);
SIMD_FLOAT C14 = SIMD_SET1(14.0f);
SIMD_FLOAT C4 = SIMD_SET1(4.0f);

out[0] = SIMD_FMADD(in[0],C576,in[IS * 8]);
out[0] = SIMD_FNMADD(in[IS * 2],C820,out[0]);
out[0] = SIMD_FMADD(in[IS * 4],C273,out[0]);
out[0] = SIMD_FNMADD(in[IS * 6],C30,out[0]);

SIMD_FLOAT V12S = SIMD_FNMADD(in[IS * 2],C576,in[IS * 8]);
V12S = SIMD_FMADD(in[IS * 4],C244,V12S);
V12S = SIMD_FNMADD(in[IS * 6],C29,V12S);

SIMD_FLOAT V12R = SIMD_FNMADD(in[IS],C576,in[IS * 7]);
V12R = SIMD_FMADD(in[IS * 3],C244,V12R);
V12R = SIMD_FNMADD(in[IS * 5],C29,V12R);

out[1] = SIMD_ADD(V12S,V12R);

out[2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_FNMADD(in[IS * 2],C144,in[IS * 8]);
V34S = SIMD_FMADD(in[IS * 4],C169,V34S);
V34S = SIMD_FNMADD(in[IS * 6],C26,V34S);

SIMD_FLOAT V34R = SIMD_MUL(in[IS * 3],C338);
V34R = SIMD_FNMADD(in[IS],C288,V34R);
V34R = SIMD_FNMADD(in[IS * 5],C52,V34R);
V34R = SIMD_FMADD(in[IS * 7],C2,V34R);

out[3] = SIMD_ADD(V34S,V34R);

out[4] = SIMD_SUB(V34S,V34R);
 
SIMD_FLOAT V56S = SIMD_FNMADD(in[IS * 2],C64,in[IS * 8]);
V56S = SIMD_FMADD(in[IS * 4],C84,V56S);
V56S = SIMD_FNMADD(in[IS * 6],C21,V56S);

SIMD_FLOAT V56R = SIMD_MUL(in[IS * 3],C252);
V56R = SIMD_FNMADD(in[IS],C192,V56R);
V56R = SIMD_FNMADD(in[IS * 5],C63,V56R);
V56R = SIMD_FMADD(in[IS * 7],C3,V56R);

out[5] = SIMD_ADD(V56S,V56R);

out[6] = SIMD_SUB(V56S,V56R);
 
SIMD_FLOAT V78S = SIMD_FNMADD(in[IS * 2],C36,in[IS * 8]);
V78S = SIMD_FMADD(in[IS * 4],C49,V78S);
V78S = SIMD_FNMADD(in[IS * 6],C14,V78S);

SIMD_FLOAT V78R = SIMD_MUL(in[IS * 3],C196);
V78R = SIMD_FNMADD(in[IS],C144,V78R);
V78R = SIMD_FNMADD(in[IS * 5],C56,V78R);
V78R = SIMD_FMADD(in[IS * 7],C4,V78R);

out[7] = SIMD_ADD(V78S,V78R);

out[8] = SIMD_SUB(V78S,V78R);
 
out[9] = SIMD_FMADD(in[IS],C576,in[IS * 9]);
out[9] = SIMD_FNMADD(in[IS * 3],C820,out[9]);
out[9] = SIMD_FMADD(in[IS * 5],C273,out[9]);
out[9] = SIMD_FNMADD(in[IS * 7],C30,out[9]);


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}
	
