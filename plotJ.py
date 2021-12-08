import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

import matplotlib

from helper_functions.automatic_plot_helper import all_folders_in_dir_with
from helper_functions.automatic_plot_helper import load_isings_specific_path
from helper_functions.automatic_plot_helper import load_isings_from_list
from helper_functions.automatic_plot_helper import load_settings


def load_Js(folder_name, plot_settings):
    folder_dir = 'save/{}'.format(folder_name)
    dir_sims = all_folders_in_dir_with(folder_dir, 'sim')
    if len(dir_sims) == 0: # if not loading from parallel folder
        dir_sims = [folder_dir]

    Js_all_sims = []
    gen_num_all_sims = []
    highscores_all_sims = []
    run_num_all_sims = []


    for dir in dir_sims:
        sim_name = dir[(dir.rfind('save/') + 5):]
        if not plot_settings['verbose']:
            print(f'Loading files from {sim_name}')
        if len(plot_settings['plot_gens']) > 0:
            assert len(plot_settings['plot_gens']) == 3, "plot_gens needs to have format [min, max, step] for np.arange()"

            load_generations = np.arange(plot_settings['plot_gens'][0],
                                         plot_settings['plot_gens'][1],
                                         plot_settings['plot_gens'][2])
            isings_list = load_isings_from_list(sim_name, load_generations,
                                                decompress=plot_settings['decompress'],
                                                verbose=plot_settings['verbose'])
            sim_settings = load_settings(sim_name)
        else:
            isings_list = load_isings_specific_path('{}/isings'.format(dir),
                                                    decompress=plot_settings['decompress'],
                                                    verbose=plot_settings['verbose'])
            sim_settings = load_settings(sim_name)

        # J_sim = [[I.J[I.maskJ] for I in I_gens] for I_gens in isings_list]
        J_sim = [[I.J[np.triu_indices_from(I.J, k=1)] for I in I_gens] for I_gens in isings_list]
        gen_sim = [I_gens[0].generation for I_gens in isings_list]
        highscore_sim = [ np.max([I.avg_energy for I in I_gens]) for I_gens in isings_list]

        if sim_settings['beta_linspace'] is not None:
            run_num = str(int(sim_settings['beta_linspace'][0])).zfill(2)
        else:
            run_num = ''


        Js_all_sims.append(J_sim)
        gen_num_all_sims.append(gen_sim)
        highscores_all_sims.append(highscore_sim)
        run_num_all_sims.append(run_num)

    save_dict = {'J': Js_all_sims,
                 'gen': gen_num_all_sims,
                 'highscore': highscores_all_sims,
                 'run': run_num_all_sims}

    save_plot_data(folder_name, save_dict, plot_settings)

    return save_dict

def save_plot_data(folder_name, attrs_lists, plot_settings):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)

    if len(plot_settings['plot_gens']) > 0:
        plot_gens_str = f"{plot_settings['plot_gens'][0]}_{plot_settings['plot_gens'][1]}_{plot_settings['plot_gens'][2]}"
    else:
        plot_gens_str = 'all'
    save_name = 'plot_data_J_plot_gens_' + plot_gens_str + '.pickle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle_out = open(save_dir + save_name, 'wb')
    pickle.dump(attrs_lists, pickle_out)
    pickle_out.close()

def load_plot_data(folder_name, plot_settings):
    save_dir = 'save/{}/one_pop_plot_data/'.format(folder_name)
    if len(plot_settings['plot_gens']) > 0:
        plot_gens_str = f"{plot_settings['plot_gens'][0]}_{plot_settings['plot_gens'][1]}_{plot_settings['plot_gens'][2]}"
    else:
        plot_gens_str = 'all'
    save_name = 'plot_data_J_plot_gens_' + plot_gens_str + '.pickle'
    try:
        file = open(save_dir+save_name, 'rb')
        attrs_lists = pickle.load(file)
        file.close()
    except FileNotFoundError:
        print('Did not file save file:' + save_name)


    return attrs_lists

def plot(data_dict, plot_settings):
    print('Generating plots...\n')
    for i in tqdm(range(len(data_dict['J']))):
        fig, ax = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]}, sharex=True)

        J_arr = np.array(data_dict['J'][i])
        gen_arr = np.array(data_dict['gen'][i])
        highscore_arr = np.array(data_dict['highscore'][i])
        run_num = np.array(data_dict['run'][i])

        ax[0].plot(gen_arr, highscore_arr, 'k.', markersize=2, alpha=1.)
        ax[0].set_ylabel('Highscores (Avg. E)')

        colors = plt.cm.Spectral(np.linspace(0, 1, J_arr.shape[-1]))
        if plot_settings['scatter']:
            for j in range(J_arr.shape[-1]):
                ax[1].plot(gen_arr, J_arr[...,j], '.',
                           markersize=3, alpha=0.1, color=colors[j])
        else:
            ax[1].plot(gen_arr, np.mean(J_arr, axis=1))
        ax[1].set_xlabel('Generations')
        ax[1].set_ylabel('Edge Weights')

        save_dir = 'save/{}/figs/Js/J_{}_{}_{}/'.format(plot_settings['folder_name'],
                                                        plot_settings['plot_gens'][0],
                                                        plot_settings['plot_gens'][1],
                                                        plot_settings['plot_gens'][2],
                                                        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = save_dir + 'J_run-{}.png'.format(str(run_num).zfill(2))
        plt.savefig(save_name, dpi=300)
        plt.close()
        # plt.show()



if __name__ == '__main__':
    folder_name = 'NES_mutation_rate_experiments'

    plot_settings = {}
    # empty list plots all generations, otherwise in the format of np.arange(min, max, step)
    plot_settings['plot_gens'] = [0, 4000, 2]
    plot_settings['scatter'] = True
    plot_settings['decompress'] = True
    plot_settings['only_plot'] = False
    plot_settings['verbose'] = False
    plot_settings['folder_name'] = folder_name

    if plot_settings['only_plot']:
        data_dict = load_plot_data(folder_name, plot_settings)
    else:
        data_dict = load_Js(folder_name, plot_settings)

    matplotlib.use('TkAgg')
    plot(data_dict, plot_settings)