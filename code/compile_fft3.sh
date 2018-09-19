#! /bin/bash

mkdir -p compiled;

#make clean;

make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/vgg30.bin ; mv ./bin/avx$1/bench_fft3/vgg30.bin ./compiled/vgg_fft3_30_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/vgg27.bin ; mv ./bin/avx$1/bench_fft3/vgg27.bin ./compiled/vgg_fft3_27_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/vgg25.bin ; mv ./bin/avx$1/bench_fft3/vgg25.bin ./compiled/vgg_fft3_25_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/vgg21.bin ; mv ./bin/avx$1/bench_fft3/vgg21.bin ./compiled/vgg_fft3_21_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/vgg16.bin ; mv ./bin/avx$1/bench_fft3/vgg16.bin ./compiled/vgg_fft3_16_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/vgg9.bin  ; mv ./bin/avx$1/bench_fft3/vgg9.bin  ./compiled/vgg_fft3_9_avx$1_cores_$2_use_cores_$3_cache_$4

make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/alexnet31.bin ; mv ./bin/avx$1/bench_fft3/alexnet31.bin ./compiled/alexnet_fft3_31_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/alexnet18.bin ; mv ./bin/avx$1/bench_fft3/alexnet18.bin ./compiled/alexnet_fft3_18_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/alexnet15.bin ; mv ./bin/avx$1/bench_fft3/alexnet15.bin ./compiled/alexnet_fft3_15_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/alexnet11.bin ; mv ./bin/avx$1/bench_fft3/alexnet11.bin ./compiled/alexnet_fft3_11_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/alexnet9.bin  ; mv ./bin/avx$1/bench_fft3/alexnet9.bin  ./compiled/alexnet_fft3_9_avx$1_cores_$2_use_cores_$3_cache_$4

#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/overfeat8.bin  ; mv ./bin/avx$1/bench_fft3/overfeat8.bin  ./compiled/overfeat_fft3_8_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/overfeat10.bin ; mv ./bin/avx$1/bench_fft3/overfeat10.bin ./compiled/overfeat_fft3_10_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/overfeat14.bin ; mv ./bin/avx$1/bench_fft3/overfeat14.bin ./compiled/overfeat_fft3_14_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_fft3/overfeat28.bin ; mv ./bin/avx$1/bench_fft3/overfeat28.bin ./compiled/overfeat_fft3_28_avx$1_cores_$2_use_cores_$3_cache_$4
