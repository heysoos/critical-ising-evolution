import matplotlib
matplotlib.use('Agg')
from helper_functions.automatic_plot_helper import all_sim_names_in_parallel_folder
from helper_functions.automatic_plot_helper import detect_all_isings
from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import choose_copied_isings
from helper_functions.heat_capacity_parameter import calc_heat_cap_param_main
from scipy.interpolate import interp1d
import numpy as np
# from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.lines import Line2D
import matplotlib.colors as colors_package
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

class OneSimPlotData:
    def __init__(self, sim_name, delta_dict, delta_list_dict, gens_list, fitnesses_at_gen):
        self.sim_name = sim_name
        self.delta_dict = delta_dict
        self.delta_list_dict = delta_list_dict
        self.fitnesses_at_gen = fitnesses_at_gen
        self.gens_list = gens_list


def save_plot_data(folder_name, attrs_lists):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    save_name = 'plot_delta_vs_fitness.pickle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(attrs_lists, pickle_out)
    pickle_out.close()


def load_plot_data(folder_name):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    save_name = 'plot_delta_vs_fitness.pickle'
    print('Load plot data from: {}{}'.format(save_dir, save_name))

    file = open(save_dir+save_name, 'rb')
    attrs_lists = pickle.load(file)
    file.close()

    return attrs_lists


def plot(sim_plot_data_list, plot_settings):

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.grid()

    # load in data from data class
    fitnesses = [sim_data.fitnesses_at_gen for sim_data in sim_plot_data_list]
    deltas = [list(sim_data.delta_list_dict.values()) for sim_data in sim_plot_data_list]

    # sort them all according to which generation they are from
    fitness_gens_list = [sim_data.gens_list for sim_data in sim_plot_data_list]
    fitness_gens_idx = [np.argsort(fg) for fg in fitness_gens_list]
    sorted_fitness_gens_list = [[fgl[idx] for idx in fitness_gens_idx[i]] for i, fgl in enumerate(fitness_gens_list)]
    sorted_fitnesses = [[f[idx] for idx in fitness_gens_idx[i]] for i, f in enumerate(fitnesses)]

    delta_gens = [[int(k) for k in sim_data.delta_dict.keys()] for sim_data in sim_plot_data_list]
    delta_gens_idx = [np.argsort(dg) for dg in delta_gens]
    sorted_deltas = [[d[idx] for idx in delta_gens_idx[i]] for i, d in enumerate(deltas)]

    # iterate over different runs
    for i in range(len(sorted_deltas)):

        delta = sorted_deltas[i]
        fitness = sorted_fitnesses[i]

        ax.plot(delta, fitness, '-ok', alpha=0.1, lw=1, markersize=1)

    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Avg. Fitness $\langle E \rangle$')
    ax.set_title(r'$\delta$ vs. Fitness')


    save_dir = 'save/{}/figs/'.format(folder_name, '')
    save_name = 'delta_vs_fitness_all_sims'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_dir+save_name+'.png', bbox_inches='tight', dpi=300)
    # plt.savefig(save_dir+save_name+'.pdf', bbox_inches='tight')

def load_data_from_sims(folder_name, plot_settings):
    sim_names = all_sim_names_in_parallel_folder(folder_name)
    sim_plot_data_list = []
    for sim_name in sim_names:
        module_settings = {}
        mean_log_beta_distance_dict, log_beta_distance_dict, beta_distance_dict, beta_index_max, betas_max_gen_dict, \
        heat_caps_max_dict, smoothed_heat_caps = calc_heat_cap_param_main(sim_name, module_settings, gaussian_kernel=True)
        delta_dict = mean_log_beta_distance_dict
        delta_list_dict = log_beta_distance_dict

        gens_list = [int(k) for k in delta_dict.keys()]
        fitnesses_at_gen = load_fitness_gen_list(sim_name, gens_list, plot_settings)


        sim_plot_data_list.append(OneSimPlotData(sim_name=sim_name,
                                                 delta_dict=delta_dict,
                                                 delta_list_dict=delta_list_dict,
                                                 gens_list=gens_list,
                                                 fitnesses_at_gen=fitnesses_at_gen))

        # settings_list.append(load_settings(dir))
    # delta_dicts_all_sims --> men of each generation, deltas_dicts_all_sims --> each individual in a list
    return sim_plot_data_list

def load_fitness_gen_list(sim_name, gens_list, plot_settings):
    generation = detect_all_isings(sim_name)[gens_list]
    isings_list = load_isings_from_list(sim_name, generation, decompress=plot_settings['decompress'], verbose=False)
    # fitness_at_gen = [np.max([I.avg_energy for I in isings]) for isings in isings_list]
    fitness_at_gen = [[I.avg_energy for I in isings] for isings in isings_list]

    return fitness_at_gen

if __name__ == '__main__':
    plot_settings = {}
    # Only plot loads previously saved plotting file instead of loading all simulations to save time
    plot_settings['only_plot'] = False
    plot_settings['decompress'] = True

    folder_name = 'sim-20211123-152231_parallel_simpletask_original'

    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 23, 'serif': ['computer modern roman']}
    plt.rc('font', **font)

    if not plot_settings['only_plot']:
        sim_plot_data_list = load_data_from_sims(folder_name, plot_settings)
        save_plot_data(folder_name, sim_plot_data_list)
    else:
        sim_plot_data_list = load_plot_data(folder_name)

    plot(sim_plot_data_list, plot_settings)
