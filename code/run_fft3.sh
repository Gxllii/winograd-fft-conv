#! /bin/bash

mkdir -p ./logs
mkdir -p ./logs/fft3

numactl --membind=$5 ./compiled/vgg_fft3_30_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/vgg_fft3_30_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft3_27_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/vgg_fft3_27_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft3_25_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/vgg_fft3_25_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft3_21_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/vgg_fft3_21_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft3_16_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/vgg_fft3_16_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft3_9_avx$1_cores_$2_use_cores_$3_cache_$4  > ./logs/fft3/vgg_fft3_9_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6

numactl --membind=$5 ./compiled/alexnet_fft3_31_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/alexnet_fft3_31_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft3_18_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/alexnet_fft3_18_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft3_15_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/alexnet_fft3_15_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft3_11_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/alexnet_fft3_11_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft3_9_avx$1_cores_$2_use_cores_$3_cache_$4  > ./logs/fft3/alexnet_fft3_9_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6

#numactl --membind=$5 ./compiled/overfeat_fft3_8_avx$1_cores_$2_use_cores_$3_cache_$4  > ./logs/fft3/overfeat_fft3_8_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_fft3_10_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/overfeat_fft3_10_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_fft3_14_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/overfeat_fft3_14_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_fft3_28_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft3/overfeat_fft3_28_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
