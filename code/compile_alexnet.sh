#! /bin/bash

mkdir -p compiled;

make clean;
make OPT=1 CORES=40 USE_CORES=20 CACHE_SIZE=1024 ./bin/avx$1/bench_win/test.bin ; mv ./bin/avx$1/bench_win/test.bin ./compiled/win_20

make clean;
make OPT=1 CORES=40 USE_CORES=1 CACHE_SIZE=1024 ./bin/avx$1/bench_win/test.bin ; mv ./bin/avx$1/bench_win/test.bin ./compiled/win_1


make clean;
make OPT=1 CORES=40 USE_CORES=20 CACHE_SIZE=1024 ./bin/avx$1/bench_fft/test.bin ; mv ./bin/avx$1/bench_fft/test.bin ./compiled/fft_20

make clean;
make OPT=1 CORES=40 USE_CORES=1 CACHE_SIZE=1024 ./bin/avx$1/bench_fft/test.bin ; mv ./bin/avx$1/bench_fft/test.bin ./compiled/fft_1


make clean;
make OPT=1 CORES=40 USE_CORES=20 CACHE_SIZE=1024 ./bin/avx$1/bench_fft3/test.bin ; mv ./bin/avx$1/bench_fft3/test.bin ./compiled/fft3_20

make clean;
make OPT=1 CORES=40 USE_CORES=1 CACHE_SIZE=1024 ./bin/avx$1/bench_fft3/test.bin ; mv ./bin/avx$1/bench_fft3/test.bin ./compiled/fft3_1
