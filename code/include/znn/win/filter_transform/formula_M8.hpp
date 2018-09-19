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
typename std::enable_if<M == 8 && N == 3>::type
transform_filter_1d(SIMD_FLOAT* __restrict out, SIMD_FLOAT const* __restrict in)
{ 
SIMD_FLOAT C1D576 = SIMD_SET1(0.00173611111111f);
SIMD_FLOAT CN1D720 = SIMD_SET1(-0.00138888888889f);
SIMD_FLOAT C1D1440 = SIMD_SET1(0.000694444444444f);
SIMD_FLOAT C1D360 = SIMD_SET1(0.00277777777778f);
SIMD_FLOAT CN1D5040 = SIMD_SET1(-0.000198412698413f);
SIMD_FLOAT C1D1680 = SIMD_SET1(0.000595238095238f);
SIMD_FLOAT C1D560 = SIMD_SET1(0.00178571428571f);
SIMD_FLOAT C1D40320 = SIMD_SET1(2.48015873016e-05f);
SIMD_FLOAT C1D10080 = SIMD_SET1(9.92063492063e-05f);
SIMD_FLOAT C1D2520 = SIMD_SET1(0.000396825396825f);

out[0] = SIMD_MUL(in[0],C1D576);

SIMD_FLOAT V12S = SIMD_MUL(in[0],CN1D720);
V12S = SIMD_FMADD(in[IS * 2],CN1D720,V12S);

out[OS] = SIMD_FMADD(CN1D720,in[IS],V12S);

out[OS * 2] = SIMD_FNMADD(CN1D720,in[IS],V12S);

SIMD_FLOAT V34S = SIMD_MUL(in[0],C1D1440);
V34S = SIMD_FMADD(in[IS * 2],C1D360,V34S);

out[OS * 3] = SIMD_FNMADD(CN1D720,in[IS],V34S);

out[OS * 4] = SIMD_FMADD(CN1D720,in[IS],V34S);

SIMD_FLOAT V56S = SIMD_MUL(in[0],CN1D5040);
V56S = SIMD_FNMADD(in[IS * 2],C1D560,V56S);

out[OS * 5] = SIMD_FNMADD(C1D1680,in[IS],V56S);

out[OS * 6] = SIMD_FMADD(C1D1680,in[IS],V56S);

SIMD_FLOAT V78S = SIMD_MUL(in[0],C1D40320);
V78S = SIMD_FMADD(in[IS * 2],C1D2520,V78S);

out[OS * 7] = SIMD_FMADD(C1D10080,in[IS],V78S);

out[OS * 8] = SIMD_FNMADD(C1D10080,in[IS],V78S);

out[OS * 9] = in[IS * 2];
 

 }

template <long_t M, long_t N, long_t O_STRIDE, long_t IS, long_t STRIDE>
inline __attribute__((always_inline))
typename std::enable_if<M == 8 && N == 3>::type
transform_filter_1d_last(float* __restrict output,
                        SIMD_FLOAT const* __restrict in, long_t base)
{
static const long_t TS = M + N - 1;
SIMD_FLOAT out[TS] __attribute__((aligned(64)));

SIMD_FLOAT C1D576 = SIMD_SET1(0.00173611111111f);
SIMD_FLOAT CN1D720 = SIMD_SET1(-0.00138888888889f);
SIMD_FLOAT C1D1440 = SIMD_SET1(0.000694444444444f);
SIMD_FLOAT C1D360 = SIMD_SET1(0.00277777777778f);
SIMD_FLOAT CN1D5040 = SIMD_SET1(-0.000198412698413f);
SIMD_FLOAT C1D1680 = SIMD_SET1(0.000595238095238f);
SIMD_FLOAT C1D560 = SIMD_SET1(0.00178571428571f);
SIMD_FLOAT C1D40320 = SIMD_SET1(2.48015873016e-05f);
SIMD_FLOAT C1D10080 = SIMD_SET1(9.92063492063e-05f);
SIMD_FLOAT C1D2520 = SIMD_SET1(0.000396825396825f);

out[0] = SIMD_MUL(in[0],C1D576);

SIMD_FLOAT V12S = SIMD_MUL(in[0],CN1D720);
V12S = SIMD_FMADD(in[IS * 2],CN1D720,V12S);

out[1] = SIMD_FMADD(CN1D720,in[IS],V12S);

out[2] = SIMD_FNMADD(CN1D720,in[IS],V12S);

SIMD_FLOAT V34S = SIMD_MUL(in[0],C1D1440);
V34S = SIMD_FMADD(in[IS * 2],C1D360,V34S);

out[3] = SIMD_FNMADD(CN1D720,in[IS],V34S);

out[4] = SIMD_FMADD(CN1D720,in[IS],V34S);

SIMD_FLOAT V56S = SIMD_MUL(in[0],CN1D5040);
V56S = SIMD_FNMADD(in[IS * 2],C1D560,V56S);

out[5] = SIMD_FNMADD(C1D1680,in[IS],V56S);

out[6] = SIMD_FMADD(C1D1680,in[IS],V56S);

SIMD_FLOAT V78S = SIMD_MUL(in[0],C1D40320);
V78S = SIMD_FMADD(in[IS * 2],C1D2520,V78S);

out[7] = SIMD_FMADD(C1D10080,in[IS],V78S);

out[8] = SIMD_FNMADD(C1D10080,in[IS],V78S);

out[9] = in[IS * 2];
 


#pragma unroll(TS)
    for (long_t i = 0; i < TS; i++)
	{
	  SIMD_STREAM(output + (base + i * STRIDE) * O_STRIDE, out[i]);
	}
}

