repeats=15
cores=15


subfolder_add="simpletask_random_time"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"


command="python3 train.py -rand_ts -rand_ts_lim 500 1500 -b_linspace {1} -1 1 ${repeats} -g 4001 -a 4000 -compress -noplt -rec_c 250 -c_props 10 50 -4 2 100 40 -c 1 -subfolder ${subfolder} -no_commands -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 
