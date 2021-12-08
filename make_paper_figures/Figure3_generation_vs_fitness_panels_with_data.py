import matplotlib
matplotlib.use('Agg')
from helper_functions.automatic_plot_helper import all_folders_in_dir_with
from helper_functions.automatic_plot_helper import load_isings_specific_path
from helper_functions.automatic_plot_helper import attribute_from_isings
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import choose_copied_isings
from helper_functions.automatic_plot_helper import calc_normalized_fitness
from helper_functions.automatic_plot_helper import load_isings_from_list
import numpy as np

from Figure4_dynamical_range_parameter_fitness_coloring_with_data import OneSimPlotData

import matplotlib.pyplot as plt
import os
import pickle
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


def main_plot_parallel_sims(folder_name, plot_settings):
#     path = Path(os.getcwd())
#     os.chdir(path.parent.absolute())

    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'size': 36, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    plt.rc('axes', axisbelow=True)
    if plot_settings['only_copied']:
        plot_settings['only_copied_str'] = '_only_copied_orgs'
    else:
        plot_settings['only_copied_str'] = '_all_orgs'

    if plot_settings['only_plot_certain_generations']:
        plot_settings['plot_generations_str'] = 'gen_{}_to_{}' \
            .format(plot_settings['lowest_and_highest_generations_to_be_plotted'][0],
                    plot_settings['lowest_and_highest_generations_to_be_plotted'][1])
    else:
        plot_settings['plot_generations_str'] = 'gen_all'

    if not plot_settings['only_plot']:
        attrs_lists = load_attrs(folder_name, plot_settings)
        save_plot_data(folder_name, attrs_lists, plot_settings)
    else:
        attrs_lists = load_plot_data(folder_name, plot_settings)

    plot(attrs_lists, plot_settings)


def save_plot_data(folder_name, attrs_lists, plot_settings):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    save_name = 'plot_data_{}{}_min_ts{}_min_food{}_{}.pickle' \
        .format(plot_settings['attr'], plot_settings['only_copied_str'], plot_settings['min_ts_for_plot'],
                plot_settings['min_food_for_plot'], plot_settings['plot_generations_str'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(attrs_lists, pickle_out)
    pickle_out.close()


def load_plot_data(folder_name, plot_settings):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    save_name = 'plot_data_{}{}_min_ts{}_min_food{}_{}.pickle'. \
        format(plot_settings['attr'], plot_settings['only_copied_str'], plot_settings['min_ts_for_plot'],
               plot_settings['min_food_for_plot'], plot_settings['plot_generations_str'])
    print('Load plot data from: {}{}'.format(save_dir, save_name))
    try:
        file = open(save_dir+save_name, 'rb')
        attrs_lists = pickle.load(file)
        file.close()
    except FileNotFoundError:
        print('Did not find original plot file where all generations are plotted...looking for older version file')
        if not plot_settings['only_plot_certain_generations']:
            save_name = 'plot_data_{}{}_min_ts{}_min_food{}.pickle'. \
                format(plot_settings['attr'], plot_settings['only_copied_str'], plot_settings['min_ts_for_plot'],
                       plot_settings['min_food_for_plot'])
            file = open(save_dir+save_name, 'rb')
            attrs_lists = pickle.load(file)
            file.close()

    # try to load the delta data from Figure 4 if it exists.
    # try:
    #     delta_save_name = 'plot_dynamic_range_param_data_with_fitness_last_gen.pickle'
    #     file = open(save_dir+delta_save_name, 'rb')
    #     delta_attrs_list = pickle.load(file)
    #     file.close()
    # except FileNotFoundError:
    #     print('No Delta savefile. Generate Figure 4 to create plot data for colourmap.')
    #     delta_attrs_list = []

    return attrs_lists


def plot(attrs_lists, plot_settings):
    # plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.grid()

    # colors = sns.color_palette("dark", len(attrs_lists))
    cmap = LinearSegmentedColormap.from_list('my_cmap', plot_settings['color_list'])

    # if delta data exists, use it to colour the segmented lines
    # if len(delta_attrs_list) > 0:
    #     delta_dicts = [D.delta_dict for D in delta_attrs_list]
    #     delta_gens = [ [int(k) for k in list(D.keys())] for D in delta_dicts]
    #     deltas = [ [v for v in list(D.values())] for D in delta_dicts]
    #     dmin = np.min(deltas)
    #     dmax= np.max(deltas)
    #     normed_deltas = [(d - dmin) / (dmax - dmin) for d in deltas]
    #     # colors = [cmap(nd) for nd in normed_deltas]
    #     colors = normed_deltas
    # else:
    color_norm_getters = np.linspace(0, 1, len(attrs_lists))
    colors = [cmap(color_norm_getter) for color_norm_getter in color_norm_getters]
    if not plot_settings['fitness_2']:
        colors.reverse()


    for isim, (attrs_list, color) in enumerate(zip(attrs_lists, colors)):
        generations = np.arange(len(attrs_list))
        highscore_list = [np.max(attrs_gen) for attrs_gen in attrs_list]
        # mean_attrs_lilst = [np.nanmean(gen_attrs) for gen_attrs in attrs_list]
        plot_attrs_list = highscore_list

        if plot_settings['attr'] == 'norm_avg_energy':
            plot_attrs_list = [2000* f for f in plot_attrs_list]

        # if len(delta_attrs_list) > 0:
        #     delta_colors = []
        #     color_idx = 0
        #     for g in generations:
        #         if g > delta_gens[isim][color_idx]:
        #             color_idx += 1
        #         delta_colors.append(color[color_idx])


        # removed scatters
        # plt.scatter(generations, mean_attrs_list, s=2, alpha=0.05, c=color) #alpha = .15
        if plot_settings['sliding_window']:
            slided_plot_attrs_list, slided_x_axis = slide_window(plot_attrs_list, plot_settings['sliding_window_size'])
            # plt.plot(slided_x_axis, slided_mean_attrs_list, alpha=0.8, linewidth=2, c=color)
            plt.plot(slided_x_axis, slided_plot_attrs_list, alpha=1, linewidth=2, c=color)
        if plot_settings['smooth']:
            '''
            Trying to make some sort of regression, that smoothes and interpolates 
            Trying to find an alternative to moving average, where boundary values are cut off
            '''
            # smoothed_mean_attrs_list = gaussian_kernel_smoothing(mean_attrs_list)
            # Savitzky-Golay filter:
            smoothed_plot_attrs_list = savgol_filter(plot_attrs_list, plot_settings['savegol_window'], 3) # window size, polynomial order
            # plt.plot(generations, smoothed_mean_attrs_list, c=color)

            if plot_settings['interpolate']:
                # Uncommand the following, if interpolation shall be applied to smoothed data
                f_interpolate = interp1d(generations, smoothed_plot_attrs_list, kind='cubic')
                x_interp = np.linspace(np.min(generations), np.max(generations), num=4000, endpoint=True)
                y_interp = f_interpolate(x_interp)

                # if len(delta_attrs_list) > 0:
                #     xy = np.stack([x_interp, y_interp], axis=1)
                #     xy = xy.reshape(-1, 1, 2)
                #     segments = np.hstack([xy[:-1], xy[1:]])
                #     line_segments = LineCollection(segments, cmap='Spectral')
                #     line_segments.set_array(np.array(delta_colors))
                #     ax.add_collection(line_segments)
                #     ax.autoscale_view()
                # else:
                plt.plot(x_interp, y_interp, c=color, alpha=0.8, linewidth=2)
            else:
                plt.plot(generations, smoothed_plot_attrs_list, c=color, alpha=0.8, linewidth=2)
        else:
            plt.plot(generations, plot_attrs_list, '.', alpha=0.5)

        # plt.scatter(generations, mean_attrs_list, s=20, alpha=1)
    if plot_settings['fitness_2']:
        plt.axhline(2, color='black', alpha=1, linewidth=3, linestyle=(0, (2, 4)))
    if not plot_settings['remove_x_ticks']:
        plt.xlabel('Generation')
    # plt.ylabel(plot_settings['attr'])
    if not plot_settings['remove_y_ticks']:
        plt.ylabel(r'Fitness, $\langle E \rangle$')
    # plt.ylim(plot_settings['ylim'])
    if plot_settings['legend']:
        create_legend()

    if plot_settings['ylim'][-1] == 10:
        plt.yticks(np.arange(0, 10+1, 2.0))

    if plot_settings['remove_x_ticks']:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    if plot_settings['remove_y_ticks']:
        pass
        # plt.tick_params(
        #     axis='y',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     left=False,      # ticks along the bottom edge are off
        #     labelleft=False) # labels along the bottom edge are off



    # save_dir = 'save/{}/figs/several_plots{}/'.format(folder_name, plot_settings['add_save_name'])
    save_dir = 'save/{}/figs/several_plots{}/'.format(plot_settings['folder_name'], plot_settings['add_save_name'])
    # save_name = 'several_sims_criticial_{}{}_{}_min_ts{}_min_food{}_{}.png'. \
    #     format(plot_settings['attr'], plot_settings['only_copied_str'], plot_settings['folder_name'],
    #            plot_settings['min_ts_for_plot'], plot_settings['min_food_for_plot'],
    #            plot_settings['plot_generations_str'])
    # save_name = '{}several_sims_criticial_{}{}_{}_min_ts{}_min_food{}_{}'. \
    #     format(folder_name, plot_settings['attr'], plot_settings['only_copied_str'], plot_settings['folder_name'],
    #            plot_settings['min_ts_for_plot'], plot_settings['min_food_for_plot'],
    #            plot_settings['plot_generations_str'])
    save_name = 'Gens_vs_fitness_panel_{}'.format(plot_settings['save_fig_add'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    # plt.savefig(save_dir+save_name+'.png', bbox_inches='tight', dpi=300)
    # plt.savefig(save_dir+save_name+'.pdf', bbox_inches='tight')
    bbox_inches = 'tight'
    plt.savefig(save_dir+save_name+'.png', dpi=300, bbox_inches=bbox_inches)
    plt.savefig(save_dir+save_name+'.pdf',bbox_inches=bbox_inches)


def create_legend():
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=15, alpha=0.0001, label=r'$10$ Simulations'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=15, alpha=0.75, label=r'One Generation'),
        Line2D([0], [0], color='b', lw=4, c='grey', alpha=0.7, label='One Simulation\nSmoothed'),
    ]

    plt.legend(handles=legend_elements, fontsize=36)


def load_attrs(folder_name, plot_settings):
    folder_dir = 'save/{}'.format(folder_name)
    dir_list = all_folders_in_dir_with(folder_dir, 'sim')
    attrs_list_all_sims = []
    settings_list = []
    for dir in dir_list:
        sim_name = dir[(dir.rfind('save/')+5):]
        settings = load_settings(sim_name)

        if plot_settings['only_plot_certain_generations']:
            load_generations = np.arange(plot_settings['lowest_and_highest_generations_to_be_plotted'][0],
                                         plot_settings['lowest_and_highest_generations_to_be_plotted'][1]+1)
            isings_list = load_isings_from_list(sim_name, load_generations, decompress=plot_settings['decompress'],
                                                verbose=False)
        else:
            isings_list = load_isings_specific_path('{}/isings'.format(dir), decompress=plot_settings['decompress'],
                                                    verbose=False)

        if plot_settings['only_copied']:
            isings_list = [choose_copied_isings(isings) for isings in isings_list]
        if plot_settings['attr'] == 'norm_avg_energy' or plot_settings['attr'] == 'norm_food_and_ts_avg_energy':

            calc_normalized_fitness(isings_list, plot_settings, settings)

        isings_list = below_threshold_nan(isings_list, settings)
        attrs_list = [attribute_from_isings(isings, plot_settings['attr']) if isings is not None else np.nan
                      for isings in isings_list]
        attrs_list_all_sims.append(attrs_list)
        del isings_list
        # settings_list.append(load_settings(dir))
    return attrs_list_all_sims


def below_threshold_nan(isings_list, sim_settings):
    for i, isings in enumerate(isings_list):
        if isings[0].time_steps < plot_settings['min_ts_for_plot']:
            isings_list[i] = None
        if sim_settings['random_food_seasons']:
            if isings[0].food_in_env < plot_settings['min_food_for_plot']:
                isings_list[i] = None
    return isings_list

def slide_window(iterable, win_size):
    slided = []
    x_axis_gens = []
    n = 0
    while n+win_size < len(iterable)-1:
        mean = np.nanmean(iterable[n:n+win_size])
        slided.append(mean)
        x_axis_gens.append(n+int(win_size/2))
        n += 1
    return slided, x_axis_gens

# These two functions are also in automatic_plot_helper, import them in future!!
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def color_shadings(color, lightness=1.5, darkness=0.5, num_colors=3):
    lightness_amount_vals = np.linspace(lightness, darkness, num_colors)
    return [adjust_lightness(color, lightness_amount_val) for lightness_amount_val in lightness_amount_vals]

if __name__ == '__main__':
    # folder_name = 'sim-20201020-181300_parallel_TEST'
    plot_settings = {}
    # Only plot loads previously saved plotting file instead of loading all simulations to save time
    plot_settings['only_plot'] = True
    plot_settings['decompress'] = True

    plot_settings['add_save_name'] = ''
    # plot_settings['attr'] = 'avg_energy' #'norm_food_and_ts_avg_energy' #'norm_avg_energy'
    plot_settings['attr'] = 'norm_avg_energy'  # 'norm_food_and_ts_avg_energy' #'norm_avg_energy'
    # plot_settings['only_plot_fittest']
    # if plot_settings['attr'] == 'norm_food_and_ts_avg_energy':
    #     plot_settings['ylim'] = (-0.0001, 0.00025)
    # else:
    #     plot_settings['ylim'] = (-0.001, 0.015)

    # plot_settings['ylim'] = (-0.000001, 0.00007)

    # This only plots individuals that have not been mutated in previous generation (thus were lfittest in previous generation)
    plot_settings['only_copied'] = True
    plot_settings['sliding_window'] = False
    ##### savgol smoothing ####
    plot_settings['smooth'] = True
    plot_settings['savegol_window'] = 201 # odd number
    plot_settings['interpolate'] = False # used only with smoothing
    #############################
    plot_settings['sliding_window_size'] = 100

    # ONLY PLOT HAS TO BE FALSE FOR FOLLOWING SETTINGS to work:
    plot_settings['min_ts_for_plot'] = 0
    plot_settings['min_food_for_plot'] = 0

    plot_settings['only_plot_certain_generations'] = False
    plot_settings['lowest_and_highest_generations_to_be_plotted'] = [0, 1000]
    plot_settings['title'] = ''
    plot_settings['legend'] = True

    plot_settings['savefolder_name'] = 'fitness_vs_gens_panel_{}' \
        .format(time.strftime("%Y%m%d-%H%M%S"))

    plot_settings['our_colors'] = {'lblue': '#8da6cbff', 'iblue': '#5e81b5ff', 'sblue': '#344e73ff',
                                   'lgreen': '#b6d25cff', 'igreen': '#8fb032ff', 'sgreen': '#5e7320ff',
                                   'lred': '#f2977aff', 'ired': '#eb6235ff', 'sred': '#c03e13ff'}

    ########################################
    ###          FOLDER NAME             ###
    folder_names = ['sim-20211206-201341_parallel_simpletask_NES_random_time']
    ########################################
    ########################################

    remove_x_ticks_list = [False]
    remove_y_ticks_list = [False]
    plot_legend_list = [False]
    save_fig_add_list = ['1']
    base_colors = ['igreen']
    fitness_2_list = [False]
    ylims = [(-1, 40)]
    for i, (folder_name, remove_x_ticks, remove_y_ticks, plot_legend, save_fig_add, base_color, ylim, fitness_2) in\
            enumerate(zip(folder_names, remove_x_ticks_list, remove_y_ticks_list, plot_legend_list, save_fig_add_list, base_colors, ylims, fitness_2_list)):
        plot_settings['folder_name'] = folder_name
        plot_settings['remove_x_ticks'] = remove_x_ticks
        plot_settings['remove_y_ticks'] = remove_y_ticks
        plot_settings['legend'] = plot_legend
        plot_settings['title_color'] = ''
        plot_settings['save_fig_add'] = save_fig_add
        plot_settings['ylim'] = ylim
        plot_settings['fitness_2'] = fitness_2

        # intermediate_colors = [plot_settings['our_colors']['igreen'], plot_settings['our_colors']['iblue'], plot_settings['our_colors']['ired']]
        color_list = color_shadings(plot_settings['our_colors'][base_color], lightness=1.5, darkness=0.5, num_colors=100)

        plot_settings['color_list'] = color_list



        main_plot_parallel_sims(folder_name, plot_settings)