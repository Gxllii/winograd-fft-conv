#! /bin/bash

mkdir -p compiled;

#make clean;

make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/vgg30.bin ; mv ./bin/avx$1/bench_fft/vgg30.bin ./compiled/vgg_fft_30_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/vgg27.bin ; mv ./bin/avx$1/bench_fft/vgg27.bin ./compiled/vgg_fft_27_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/vgg25.bin ; mv ./bin/avx$1/bench_fft/vgg25.bin ./compiled/vgg_fft_25_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/vgg21.bin ; mv ./bin/avx$1/bench_fft/vgg21.bin ./compiled/vgg_fft_21_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/vgg16.bin ; mv ./bin/avx$1/bench_fft/vgg16.bin ./compiled/vgg_fft_16_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/vgg9.bin  ; mv ./bin/avx$1/bench_fft/vgg9.bin  ./compiled/vgg_fft_9_avx$1_cores_$2_use_cores_$3_cache_$4

make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/alexnet31.bin ; mv ./bin/avx$1/bench_fft/alexnet31.bin ./compiled/alexnet_fft_31_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/alexnet18.bin ; mv ./bin/avx$1/bench_fft/alexnet18.bin ./compiled/alexnet_fft_18_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/alexnet15.bin ; mv ./bin/avx$1/bench_fft/alexnet15.bin ./compiled/alexnet_fft_15_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/alexnet11.bin ; mv ./bin/avx$1/bench_fft/alexnet11.bin ./compiled/alexnet_fft_11_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/alexnet9.bin  ; mv ./bin/avx$1/bench_fft/alexnet9.bin  ./compiled/alexnet_fft_9_avx$1_cores_$2_use_cores_$3_cache_$4

#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/overfeat8.bin  ; mv ./bin/avx$1/bench_fft/overfeat8.bin  ./compiled/overfeat_fft_8_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/overfeat10.bin ; mv ./bin/avx$1/bench_fft/overfeat10.bin ./compiled/overfeat_fft_10_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/overfeat14.bin ; mv ./bin/avx$1/bench_fft/overfeat14.bin ./compiled/overfeat_fft_14_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft/overfeat28.bin ; mv ./bin/avx$1/bench_fft/overfeat28.bin ./compiled/overfeat_fft_28_avx$1_cores_$2_use_cores_$3_cache_$4
