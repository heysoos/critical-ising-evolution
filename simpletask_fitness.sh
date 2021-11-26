repeats=4
cores=4

# subfolder="plot_many_functions"
subfolder_add="simpletask_NES_elitism_noAnneal_noClip_size50"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py  -NES -NES_elitism -NES_mutation_rate 0.5 -b_linspace {1} -1.5 1.5 ${repeats} -g 4001 -a 4000 -t 2000 -compress -noplt -rec_c 250 -c_props 50 50 -4 2 150 40 -c 1 -subfolder ${subfolder} -no_commands -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 
