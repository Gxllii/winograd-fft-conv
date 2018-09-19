#! /bin/bash

# XEON SERVER
# ./compile_all_for_machine.sh 512 40 20 1024

# KNL
./compile_all_for_machine.sh 512 64 64 512

# sarma
./compile_all_for_machine.sh 2 10 10 1024

# SKX
./compile_all_for_machine.sh 512 10 10 1024

# 48 core
./compile_all_for_machine.sh 512 64 48 512

# 18 core Haswell
./compile_all_for_machine.sh 2 72 18 256

# 8 core Haswell
./compile_all_for_machine.sh 2 72 8 256

# 16 core Haswell
./compile_all_for_machine.sh 2 72 16 256

# 5 core Haswell
./compile_all_for_machine.sh 2 72 5 256
