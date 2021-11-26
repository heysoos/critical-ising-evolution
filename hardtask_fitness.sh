repeats=20
cores=20

# subfolder="plot_many_functions"
subfolder_add=""
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -NES -NES_elitism -NES_mutation_rate 0.5 -v_eat_max 0.005 -b_linspace {1} -1.5 1.5 ${repeats} -g 4001 -a 4000 -t 2000 -compress -noplt -rec_c 250 -c_props 3 10 -2 2 100 40 -c 1 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 
