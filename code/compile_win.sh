#! /bin/bash

mkdir -p compiled;

#make clean;

make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/vgg6.bin ; mv ./bin/avx$1/bench_win/vgg6.bin ./compiled/vgg_win_6_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/vgg5.bin ; mv ./bin/avx$1/bench_win/vgg5.bin ./compiled/vgg_win_5_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/vgg4.bin ; mv ./bin/avx$1/bench_win/vgg4.bin ./compiled/vgg_win_4_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/vgg3.bin ; mv ./bin/avx$1/bench_win/vgg3.bin ./compiled/vgg_win_3_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/vgg2.bin ; mv ./bin/avx$1/bench_win/vgg2.bin ./compiled/vgg_win_2_avx$1_cores_$2_use_cores_$3_cache_$4

make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/alexnet6.bin ; mv ./bin/avx$1/bench_win/alexnet6.bin ./compiled/alexnet_win_6_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/alexnet5.bin ; mv ./bin/avx$1/bench_win/alexnet5.bin ./compiled/alexnet_win_5_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/alexnet4.bin ; mv ./bin/avx$1/bench_win/alexnet4.bin ./compiled/alexnet_win_4_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/alexnet3.bin ; mv ./bin/avx$1/bench_win/alexnet3.bin ./compiled/alexnet_win_3_avx$1_cores_$2_use_cores_$3_cache_$4
make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/alexnet2.bin ; mv ./bin/avx$1/bench_win/alexnet2.bin ./compiled/alexnet_win_2_avx$1_cores_$2_use_cores_$3_cache_$4

#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/overfeat2.bin ; mv ./bin/avx$1/bench_win/overfeat2.bin ./compiled/overfeat_win_2_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/overfeat3.bin ; mv ./bin/avx$1/bench_win/overfeat3.bin ./compiled/overfeat_win_3_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/overfeat4.bin ; mv ./bin/avx$1/bench_win/overfeat4.bin ./compiled/overfeat_win_4_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/overfeat5.bin ; mv ./bin/avx$1/bench_win/overfeat5.bin ./compiled/overfeat_win_5_avx$1_cores_$2_use_cores_$3_cache_$4
#make OPT=1 CORES=$2 USE_CORES=$3 CACHE_SIZE=$4 ./bin/avx$1/bench_win/overfeat6.bin ; mv ./bin/avx$1/bench_win/overfeat6.bin ./compiled/overfeat_win_6_avx$1_cores_$2_use_cores_$3_cache_$4
