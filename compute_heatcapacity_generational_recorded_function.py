import numpy as np
from os import path, makedirs

import pickle
import random
from numba import jit
from helper_functions.automatic_plot_helper import load_settings
from helper_functions.automatic_plot_helper import decompress_pickle
from other_experiments_and_plotting_scripts.ising_net_fitness_landscape import all_states
from other_experiments_and_plotting_scripts.ising_net_fitness_landscape import calculate_energies


def save_compute_heat_capacity(loadfile, bind, iterNum):
    settings = load_settings(loadfile)

    R, thermal_time, beta_low, beta_high, beta_num, y_lim_high = settings['heat_capacity_props']

    #Is there a bug here? Nbetas = 101 originally Nbetas = 102 solves index error?
    Nbetas = beta_num
    betas = 10 ** np.linspace(beta_low, beta_high, Nbetas)
    loadstr = 'save/' + loadfile +  '/isings/gen[' + str(iterNum) + ']-isings.pickle'

    try:
        file = open(loadstr, 'rb')
        isings = pickle.load(file)
        file.close()
    except FileNotFoundError:
        # Looking for compressed ising file in case normal pickle file is not found
        isings = decompress_pickle(loadstr)

    size = isings[0].size # get size from first agent
    numAgents = len(isings)

    C = np.zeros((R, numAgents))
    # tqdm(range(R))
    for rep in range(R):
        agentNum = 0

        for I in isings:

            betaVec = betas * I.Beta  # scale by org's local temperature
            # print(agentNum)
            beta_new = betaVec[bind]
            #I.randomize_state()
            #  Initialize sensors with randoms set of sensor values that have been recorded during simulation
            initialize_sensors_from_record_randomize_neurons(I)

            # Initialize network state with lowest energy network energy state
            # TODO: This eats up most of the computation, make this JIT!!!
            if settings['minimal_energy_initializatin_heat_cap']:
                sensor_vals = I.s[0:(settings['nSensors'])]
                permutated_states, permutated_states_with_sensors = all_states(I, settings, sensor_vals)
                energies_perm = calculate_energies(I, settings, permutated_states_with_sensors)
                i_min_energy = np.argmin(energies_perm)
                min_energy_state = permutated_states_with_sensors[i_min_energy]
                I.s = np.array(min_energy_state)

            if settings['anneal'] is not None:
                exp_beta_min_anneal, exp_beta_max_anneal, n_internal_anneal = settings['anneal']
                anneal_beta_exps = np.linspace(exp_beta_min_anneal, exp_beta_max_anneal, n_internal_anneal)
                anneal_betas = [10 ** beta_exp for beta_exp in anneal_beta_exps]
                anneal_betas = np.array(anneal_betas)
                # Initialize at last state of thermalization process:
                # I.s = SequentialGlauberStepFastAnnealing(I.s, I.h, I.J, beta_new, I.Ssize, I.size, anneal_betas)
                # Initialize lowest energy state during thermalization process:
                final_s, states, energies = SequentialGlauberStepFastAnnealing_calc_energy(I.s, I.h, I.J, I.Ssize, I.size,
                                                                              anneal_betas)

                argmin_energy = np.argmin(energies)
                I.s = states[argmin_energy]

            #  Measuring energy between Glaubersteps
            I.s, Em, E2m = SequentialGlauberStepFast_calc_energy(thermal_time, I.s, I.h, I.J, beta_new, I.Ssize, I.size)

            #  Claculate heat capacity
            C[rep, agentNum] = beta_new ** 2 * (E2m - Em ** 2) / size
            agentNum += 1

    # print(np.mean(C, 0))
    # TODO: CHANGE THIS SO THERE IS NO CONFLICT WITH OTHER DREAM HEAT CAP CALCULATION
    folder = 'save/' + loadfile + '/C_recorded' + '/C_' + str(iterNum) + '/'
    file = 'C-size_' + str(size) + '-Nbetas_' + \
           str(Nbetas) + '-bind_' + str(bind) + '.npy'
    filename = folder + file

    if not path.exists(folder):
        makedirs(folder)

    np.save(filename, C)

def initialize_sensors_from_record_randomize_neurons(I):
    '''
    Initialize sensors with randoms set of sensor values that have been recorded during simulation
    Randomize all other neurons
    '''
    s = np.random.randint(0, 2, I.size) * 2 - 1
    s = np.array(s, dtype=float)
    #all_recorded_inputs = from_list_of_arrs_to_arr(I.all_recorded_inputs)
    rand_index = random.randint(0, len(I.all_recorded_inputs)-1)
    chosen_sens_inputs = I.all_recorded_inputs[rand_index]
    I.s = s
    for i in range(len(chosen_sens_inputs)):
        I.s[i] = chosen_sens_inputs[i]


    if not len(chosen_sens_inputs) == I.Ssize:
        raise Exception('''For some reason the number of sensors that
        recorded values exist for is different from the sensor size saved in the settings''')

@jit(nopython=True)
def SequentialGlauberStepFast_calc_energy(thermalTime, s, h, J, Beta, Ssize, size):
    '''
    Energy calculation each thermal time step
    '''
    # TODO: After figuring the effect of thermalize sensors out delete this shit, slows everything down!
    thermalize_sensors = False

    if thermalize_sensors:
        all_neurons_except_sens = np.arange(0, size)
    else:
        all_neurons_except_sens = np.arange(Ssize, size)
    #perms_list = np.array([np.random.permutation(np.arange(Ssize, size)) for j in range(thermalTime)])
    random_vars = np.random.rand(thermalTime, len(all_neurons_except_sens)) #[np.random.rand() for i in perms]

    Em = 0
    E2m = 0

    for i in range(thermalTime):
        #perms = perms_list[i]
        #Prepare a matrix of random variables for later use

        # TODO: In previous dream heat cap calculation, the sensors were thermalized as well, while here they remain to have their values
        if thermalize_sensors:
            perms = np.random.permutation(np.arange(0, size))
            #np.random.permutation(size)
        else:
            perms = np.random.permutation(np.arange(Ssize, size))

        for j, perm in enumerate(perms):
            rand = random_vars[i, j]
            eDiff = 2 * s[perm] * (h[perm] + np.dot(J[perm, :] + J[:, perm], s))
            #deltaE = E_f - E_i = -2 E_i = -2 * - SUM{J_ij*s_i*s_j}
            #self.J[i, :] + self.J[:, i] are added because value in one of both halfs of J seperated by the diagonal is zero

            if Beta * eDiff < np.log(1.0 / rand - 1):
                #transformed  P = 1/(1+e^(deltaE* Beta)
                s[perm] = -s[perm]

        # Record/Measure energy:
        E = -(np.dot(s, h) + np.dot(np.dot(s, J), s))
        Em += E / float(thermalTime)   # <-- mean calculation??
        E2m += E ** 2 / float(thermalTime)

    return s, Em, E2m


@jit(nopython=True)
def SequentialGlauberStepFastAnnealing(s, h, J, Beta, Ssize, size, anneal_betas):
    thermalize_sensors = False
    if thermalize_sensors:
        all_neurons_except_sens = np.arange(0, size)
    else:
        all_neurons_except_sens = np.arange(Ssize, size)

    #perms_list = np.array([np.random.permutation(np.arange(Ssize, size)) for j in range(thermalTime)])
    random_vars = np.random.rand(len(anneal_betas), len(all_neurons_except_sens)) #[np.random.rand() for i in perms]
    for i, anneal_beta in enumerate(anneal_betas):
        # TODO: Use anneals betas directly as beta (currrently implemented)
        #  or use as beta factor and multiply with net's beta value??
        Beta = anneal_beta
        #perms = perms_list[i]
        #Prepare a matrix of random variables for later use

        #In previous dream heat cap calculation, the sensors were thermalized as well, while here they remain to have their values
        if thermalize_sensors:
            perms = np.random.permutation(np.arange(0, size))
            #perms = np.random.permutation(size)
        else:
            perms = np.random.permutation(np.arange(Ssize, size))

        for j, perm in enumerate(perms):
            rand = random_vars[i, j]
            eDiff = 2 * s[perm] * (h[perm] + np.dot(J[perm, :] + J[:, perm], s))
            #deltaE = E_f - E_i = -2 E_i = -2 * - SUM{J_ij*s_i*s_j}
            #self.J[i, :] + self.J[:, i] are added because value in one of both halfs of J seperated by the diagonal is zero

            if Beta * eDiff < np.log(1.0 / rand - 1):
                #transformed  P = 1/(1+e^(deltaE* Beta)
                s[perm] = -s[perm]

    return s

@jit(nopython=True)
def SequentialGlauberStepFastAnnealing_calc_energy(s, h, J, Ssize, size, anneal_betas):
    thermalize_sensors = False
    if thermalize_sensors:
        all_neurons_except_sens = np.arange(0, size)
    else:
        all_neurons_except_sens = np.arange(Ssize, size)

    #perms_list = np.array([np.random.permutation(np.arange(Ssize, size)) for j in range(thermalTime)])
    random_vars = np.random.rand(len(anneal_betas), len(all_neurons_except_sens)) #[np.random.rand() for i in perms]
    energies = np.empty(len(anneal_betas))
    states = np.empty((len(anneal_betas),len(s)))
    for i, anneal_beta in enumerate(anneal_betas):
        # TODO: Use anneals betas directly as beta (currrently implemented)
        #  or use as beta factor and multiply with net's beta value??
        Beta = anneal_beta
        #perms = perms_list[i]
        #Prepare a matrix of random variables for later use

        #In previous dream heat cap calculation, the sensors were thermalized as well, while here they remain to have their values
        if thermalize_sensors:
            perms = np.random.permutation(np.arange(0, size))
            #perms = np.random.permutation(size)
        else:
            perms = np.random.permutation(np.arange(Ssize, size))

        for j, perm in enumerate(perms):
            rand = random_vars[i, j]
            eDiff = 2 * s[perm] * (h[perm] + np.dot(J[perm, :] + J[:, perm], s))
            #deltaE = E_f - E_i = -2 E_i = -2 * - SUM{J_ij*s_i*s_j}
            #self.J[i, :] + self.J[:, i] are added because value in one of both halfs of J seperated by the diagonal is zero

            if Beta * eDiff < np.log(1.0 / rand - 1):
                #transformed  P = 1/(1+e^(deltaE* Beta)
                s[perm] = -s[perm]
        # Record/Measure energy:
        E = -(np.dot(s, h) + np.dot(np.dot(s, J), s))

        energies[i] = E
        states[i, 0:len(s)] = s
    return s, states, energies

@jit(nopython=True)
def SequentialGlauberStepFast(thermalTime, s, h, J, Beta, Ssize, size):
    thermalize_sensors = False
    if thermalize_sensors:
        all_neurons_except_sens = np.arange(0, size)
    else:
        all_neurons_except_sens = np.arange(Ssize, size)

    #perms_list = np.array([np.random.permutation(np.arange(Ssize, size)) for j in range(thermalTime)])
    random_vars = np.random.rand(thermalTime, len(all_neurons_except_sens)) #[np.random.rand() for i in perms]
    for i in range(thermalTime):
        #perms = perms_list[i]
        #Prepare a matrix of random variables for later use

        # TODO: In previous dream heat cap calculation, the sensors were thermalized as well, while here they remain to have their values
        if thermalize_sensors:
            perms = np.random.permutation(np.arange(0, size))
            #perms = np.random.permutation(size)
        else:
            perms = np.random.permutation(np.arange(Ssize, size))

        for j, perm in enumerate(perms):
            rand = random_vars[i, j]
            eDiff = 2 * s[perm] * (h[perm] + np.dot(J[perm, :] + J[:, perm], s))
            #deltaE = E_f - E_i = -2 E_i = -2 * - SUM{J_ij*s_i*s_j}
            #self.J[i, :] + self.J[:, i] are added because value in one of both halfs of J seperated by the diagonal is zero

            if Beta * eDiff < np.log(1.0 / rand - 1):
                #transformed  P = 1/(1+e^(deltaE* Beta)
                s[perm] = -s[perm]

    return s
