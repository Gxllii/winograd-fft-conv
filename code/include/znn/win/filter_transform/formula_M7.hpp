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
typename std::enable_if<M == 7 && N == 3>::type
transform_filter_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C1D144 = SIMD_SET1(0.00694444444444f);
SIMD_FLOAT CN1D144 = SIMD_SET1(-0.00694444444444f);
SIMD_FLOAT C1D240 = SIMD_SET1(0.00416666666667f);
SIMD_FLOAT C1D120 = SIMD_SET1(0.00833333333333f);
SIMD_FLOAT C1D60 = SIMD_SET1(0.0166666666667f);
SIMD_FLOAT C1D720 = SIMD_SET1(0.00138888888889f);
SIMD_FLOAT C1D360 = SIMD_SET1(0.00277777777778f);
SIMD_FLOAT C1D180 = SIMD_SET1(0.00555555555556f);
SIMD_FLOAT CN1D80 = SIMD_SET1(-0.0125f);
SIMD_FLOAT C1D5040 = SIMD_SET1(0.000198412698413f);
SIMD_FLOAT C1D1680 = SIMD_SET1(0.000595238095238f);
SIMD_FLOAT C1D560 = SIMD_SET1(0.00178571428571f);
SIMD_FLOAT C1D1260 = SIMD_SET1(0.000793650793651f);
SIMD_FLOAT C1D315 = SIMD_SET1(0.0031746031746f);

out[0] = SIMD_MUL(in[0],C1D144);

out[OS] = SIMD_MUL(in[0],CN1D144);
out[OS] = SIMD_FMADD(in[IS],CN1D144,out[OS]);
out[OS] = SIMD_FMADD(in[IS * 2],CN1D144,out[OS]);

out[OS * 2] = SIMD_MUL(in[IS],C1D240);
out[OS * 2] = SIMD_FNMADD(in[0],C1D240,out[OS * 2]);
out[OS * 2] = SIMD_FNMADD(in[IS * 2],C1D240,out[OS * 2]);

out[OS * 3] = SIMD_MUL(in[0],C1D240);
out[OS * 3] = SIMD_FMADD(in[IS],C1D120,out[OS * 3]);
out[OS * 3] = SIMD_FMADD(in[IS * 2],C1D60,out[OS * 3]);

out[OS * 4] = SIMD_MUL(in[0],C1D720);
out[OS * 4] = SIMD_FNMADD(in[IS],C1D360,out[OS * 4]);
out[OS * 4] = SIMD_FMADD(in[IS * 2],C1D180,out[OS * 4]);

out[OS * 5] = SIMD_MUL(in[IS * 2],CN1D80);
out[OS * 5] = SIMD_FNMADD(in[0],C1D720,out[OS * 5]);
out[OS * 5] = SIMD_FNMADD(in[IS],C1D240,out[OS * 5]);

out[OS * 6] = SIMD_MUL(in[IS],C1D1680);
out[OS * 6] = SIMD_FNMADD(in[0],C1D5040,out[OS * 6]);
out[OS * 6] = SIMD_FNMADD(in[IS * 2],C1D560,out[OS * 6]);

out[OS * 7] = SIMD_MUL(in[0],C1D5040);
out[OS * 7] = SIMD_FMADD(in[IS],C1D1260,out[OS * 7]);
out[OS * 7] = SIMD_FMADD(in[IS * 2],C1D315,out[OS * 7]);

out[OS * 8] = in[IS * 2];
 

 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<M == 7 && N == 3>::type
transform_filter_1d_last(float* __restrict output,
                        SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C1D144 = SIMD_SET1(0.00694444444444f);
SIMD_FLOAT CN1D144 = SIMD_SET1(-0.00694444444444f);
SIMD_FLOAT C1D240 = SIMD_SET1(0.00416666666667f);
SIMD_FLOAT C1D120 = SIMD_SET1(0.00833333333333f);
SIMD_FLOAT C1D60 = SIMD_SET1(0.0166666666667f);
SIMD_FLOAT C1D720 = SIMD_SET1(0.00138888888889f);
SIMD_FLOAT C1D360 = SIMD_SET1(0.00277777777778f);
SIMD_FLOAT C1D180 = SIMD_SET1(0.00555555555556f);
SIMD_FLOAT CN1D80 = SIMD_SET1(-0.0125f);
SIMD_FLOAT C1D5040 = SIMD_SET1(0.000198412698413f);
SIMD_FLOAT C1D1680 = SIMD_SET1(0.000595238095238f);
SIMD_FLOAT C1D560 = SIMD_SET1(0.00178571428571f);
SIMD_FLOAT C1D1260 = SIMD_SET1(0.000793650793651f);
SIMD_FLOAT C1D315 = SIMD_SET1(0.0031746031746f);

out[0] = SIMD_MUL(in[0],C1D144);

out[1] = SIMD_MUL(in[0],CN1D144);
out[1] = SIMD_FMADD(in[IS],CN1D144,out[1]);
out[1] = SIMD_FMADD(in[IS * 2],CN1D144,out[1]);

out[2] = SIMD_MUL(in[IS],C1D240);
out[2] = SIMD_FNMADD(in[0],C1D240,out[2]);
out[2] = SIMD_FNMADD(in[IS * 2],C1D240,out[2]);

out[3] = SIMD_MUL(in[0],C1D240);
out[3] = SIMD_FMADD(in[IS],C1D120,out[3]);
out[3] = SIMD_FMADD(in[IS * 2],C1D60,out[3]);

out[4] = SIMD_MUL(in[0],C1D720);
out[4] = SIMD_FNMADD(in[IS],C1D360,out[4]);
out[4] = SIMD_FMADD(in[IS * 2],C1D180,out[4]);

out[5] = SIMD_MUL(in[IS * 2],CN1D80);
out[5] = SIMD_FNMADD(in[0],C1D720,out[5]);
out[5] = SIMD_FNMADD(in[IS],C1D240,out[5]);

out[6] = SIMD_MUL(in[IS],C1D1680);
out[6] = SIMD_FNMADD(in[0],C1D5040,out[6]);
out[6] = SIMD_FNMADD(in[IS * 2],C1D560,out[6]);

out[7] = SIMD_MUL(in[0],C1D5040);
out[7] = SIMD_FMADD(in[IS],C1D1260,out[7]);
out[7] = SIMD_FMADD(in[IS * 2],C1D315,out[7]);

out[8] = in[IS * 2];
 


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}


template <long_t M, long_t N, long_t OS, long_t IS>
inline __attribute__((always_inline))
typename std::enable_if<M == 7 && N == 4>::type
transform_filter_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C1D576 = SIMD_SET1(0.00173611111111f);
SIMD_FLOAT CN1D720 = SIMD_SET1(-0.00138888888889f);
SIMD_FLOAT C1D1440 = SIMD_SET1(0.000694444444444f);
SIMD_FLOAT C1D360 = SIMD_SET1(0.00277777777778f);
SIMD_FLOAT C1D180 = SIMD_SET1(0.00555555555556f);
SIMD_FLOAT CN1D5040 = SIMD_SET1(-0.000198412698413f);
SIMD_FLOAT C1D1680 = SIMD_SET1(0.000595238095238f);
SIMD_FLOAT C1D560 = SIMD_SET1(0.00178571428571f);
SIMD_FLOAT C3D560 = SIMD_SET1(0.00535714285714f);
SIMD_FLOAT C1D40320 = SIMD_SET1(2.48015873016e-05f);
SIMD_FLOAT C1D10080 = SIMD_SET1(9.92063492063e-05f);
SIMD_FLOAT C1D2520 = SIMD_SET1(0.000396825396825f);
SIMD_FLOAT C1D630 = SIMD_SET1(0.0015873015873f);

out[0] = SIMD_MUL(in[0],C1D576);

SIMD_FLOAT V12S = SIMD_MUL(in[0],CN1D720);
V12S = SIMD_FMADD(in[IS * 2],CN1D720,V12S);

SIMD_FLOAT V12R = SIMD_MUL(in[IS],CN1D720);
V12R = SIMD_FMADD(in[IS * 3],CN1D720,V12R);

out[OS] = SIMD_ADD(V12S,V12R);

out[OS * 2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_MUL(in[0],C1D1440);
V34S = SIMD_FMADD(in[IS * 2],C1D360,V34S);

SIMD_FLOAT V34R = SIMD_MUL(in[IS * 3],C1D180);
V34R = SIMD_FNMADD(in[IS],CN1D720,V34R);

out[OS * 3] = SIMD_ADD(V34S,V34R);

out[OS * 4] = SIMD_SUB(V34S,V34R);
 
SIMD_FLOAT V56S = SIMD_MUL(in[0],CN1D5040);
V56S = SIMD_FNMADD(in[IS * 2],C1D560,V56S);

SIMD_FLOAT V56R = SIMD_MUL(in[IS],C1D1680);
V56R = SIMD_FMADD(in[IS * 3],C3D560,V56R);

out[OS * 5] = SIMD_SUB(V56S,V56R);

out[OS * 6] = SIMD_ADD(V56S,V56R);
 
SIMD_FLOAT V78S = SIMD_MUL(in[0],C1D40320);
V78S = SIMD_FMADD(in[IS * 2],C1D2520,V78S);

SIMD_FLOAT V78R = SIMD_MUL(in[IS],C1D10080);
V78R = SIMD_FMADD(in[IS * 3],C1D630,V78R);

out[OS * 7] = SIMD_ADD(V78S,V78R);

out[OS * 8] = SIMD_SUB(V78S,V78R);
 
out[OS * 9] = in[IS * 3];
 

 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<M == 7 && N == 4>::type
transform_filter_1d_last(float* __restrict output,
                        SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C1D576 = SIMD_SET1(0.00173611111111f);
SIMD_FLOAT CN1D720 = SIMD_SET1(-0.00138888888889f);
SIMD_FLOAT C1D1440 = SIMD_SET1(0.000694444444444f);
SIMD_FLOAT C1D360 = SIMD_SET1(0.00277777777778f);
SIMD_FLOAT C1D180 = SIMD_SET1(0.00555555555556f);
SIMD_FLOAT CN1D5040 = SIMD_SET1(-0.000198412698413f);
SIMD_FLOAT C1D1680 = SIMD_SET1(0.000595238095238f);
SIMD_FLOAT C1D560 = SIMD_SET1(0.00178571428571f);
SIMD_FLOAT C3D560 = SIMD_SET1(0.00535714285714f);
SIMD_FLOAT C1D40320 = SIMD_SET1(2.48015873016e-05f);
SIMD_FLOAT C1D10080 = SIMD_SET1(9.92063492063e-05f);
SIMD_FLOAT C1D2520 = SIMD_SET1(0.000396825396825f);
SIMD_FLOAT C1D630 = SIMD_SET1(0.0015873015873f);

out[0] = SIMD_MUL(in[0],C1D576);

SIMD_FLOAT V12S = SIMD_MUL(in[0],CN1D720);
V12S = SIMD_FMADD(in[IS * 2],CN1D720,V12S);

SIMD_FLOAT V12R = SIMD_MUL(in[IS],CN1D720);
V12R = SIMD_FMADD(in[IS * 3],CN1D720,V12R);

out[1] = SIMD_ADD(V12S,V12R);

out[2] = SIMD_SUB(V12S,V12R);
 
SIMD_FLOAT V34S = SIMD_MUL(in[0],C1D1440);
V34S = SIMD_FMADD(in[IS * 2],C1D360,V34S);

SIMD_FLOAT V34R = SIMD_MUL(in[IS * 3],C1D180);
V34R = SIMD_FNMADD(in[IS],CN1D720,V34R);

out[3] = SIMD_ADD(V34S,V34R);

out[4] = SIMD_SUB(V34S,V34R);
 
SIMD_FLOAT V56S = SIMD_MUL(in[0],CN1D5040);
V56S = SIMD_FNMADD(in[IS * 2],C1D560,V56S);

SIMD_FLOAT V56R = SIMD_MUL(in[IS],C1D1680);
V56R = SIMD_FMADD(in[IS * 3],C3D560,V56R);

out[5] = SIMD_SUB(V56S,V56R);

out[6] = SIMD_ADD(V56S,V56R);
 
SIMD_FLOAT V78S = SIMD_MUL(in[0],C1D40320);
V78S = SIMD_FMADD(in[IS * 2],C1D2520,V78S);

SIMD_FLOAT V78R = SIMD_MUL(in[IS],C1D10080);
V78R = SIMD_FMADD(in[IS * 3],C1D630,V78R);

out[7] = SIMD_ADD(V78S,V78R);

out[8] = SIMD_SUB(V78S,V78R);
 
out[9] = in[IS * 3];
 


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}

