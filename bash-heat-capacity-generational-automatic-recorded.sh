#!/bin/bash

#EXAMPLE INPUT:
#bash bash-heat-capacity-generational-automatic.sh "sim-20200327-215417-g_8000_-b_1_-ref_2000_-a_500_1000_2000_4000_6000_8000_-n_4_sensors" "0 3000" 5


# COMMAND LINE INPUTS:
sim=$1
generations=$2
cores=$3
beta_num=$4


args=("$@")

gens=(save/$sim/isings/*) 


parallel --bar --eta -j${cores} "python3 compute-heat-capacity-generational-2-recorded.py ${sim} {1} {2}" ::: $(seq ${beta_num}) ::: $2
