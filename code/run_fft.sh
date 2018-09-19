#! /bin/bash

mkdir -p ./logs
mkdir -p ./logs/fft

numactl --membind=$5 ./compiled/vgg_fft_30_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/vgg_fft_30_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft_27_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/vgg_fft_27_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft_25_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/vgg_fft_25_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft_21_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/vgg_fft_21_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft_16_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/vgg_fft_16_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_fft_9_avx$1_cores_$2_use_cores_$3_cache_$4  > ./logs/fft/vgg_fft_9_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6

numactl --membind=$5 ./compiled/alexnet_fft_31_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/alexnet_fft_31_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft_18_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/alexnet_fft_18_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft_15_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/alexnet_fft_15_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft_11_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/alexnet_fft_11_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_fft_9_avx$1_cores_$2_use_cores_$3_cache_$4  > ./logs/fft/alexnet_fft_9_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6

#numactl --membind=$5 ./compiled/overfeat_fft_8_avx$1_cores_$2_use_cores_$3_cache_$4  > ./logs/fft/overfeat_fft_8_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_fft_10_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/overfeat_fft_10_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_fft_14_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/overfeat_fft_14_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_fft_28_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/fft/overfeat_fft_28_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
