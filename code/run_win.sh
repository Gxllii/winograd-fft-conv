#! /bin/bash

mkdir -p ./logs
mkdir -p ./logs/win

numactl --membind=$5 ./compiled/vgg_win_6_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/vgg_win_6_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_win_5_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/vgg_win_5_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_win_4_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/vgg_win_4_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_win_3_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/vgg_win_3_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/vgg_win_2_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/vgg_win_2_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6

numactl --membind=$5 ./compiled/alexnet_win_6_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/alexnet_win_6_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_win_5_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/alexnet_win_5_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_win_4_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/alexnet_win_4_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_win_3_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/alexnet_win_3_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
numactl --membind=$5 ./compiled/alexnet_win_2_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/alexnet_win_2_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6

#numactl --membind=$5 ./compiled/overfeat_win_2_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/overfeat_win_2_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_win_3_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/overfeat_win_3_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_win_4_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/overfeat_win_4_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_win_5_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/overfeat_win_5_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
#numactl --membind=$5 ./compiled/overfeat_win_6_avx$1_cores_$2_use_cores_$3_cache_$4 > ./logs/win/overfeat_win_6_avx$1_cores_$2_use_cores_$3_cache_$4_note_$6
