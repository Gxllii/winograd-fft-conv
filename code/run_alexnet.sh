#! /bin/bash

mkdir -p compiled;

numactl --membind=1 ./compiled/win_20 > ./logs/win_20.log
numactl --membind=1 ./compiled/fft_20 > ./logs/fft_20.log
numactl --membind=1 ./compiled/fft3_20 > ./logs/fft3_20.log

numactl --membind=1 ./compiled/win_1 > ./logs/win_1.log
numactl --membind=1 ./compiled/fft_1 > ./logs/fft_1.log
numactl --membind=1 ./compiled/fft3_1 > ./logs/fft3_1.log
