import numpy as np
from helper_functions.utils import compressed_pickle

class nes:
    def __init__(self, POP=50, SIGMA=0.5, CHUNK_SIZE=10,
                 ELITISM=False, NUM_ELITE=6,
                 MUTATION_RATE=1, ANNEALING=False, CLIP=None):

        # learning params
        self.ANNEALING = ANNEALING
        self.DECAY = 0.99
        self.SIGMA = SIGMA
        self.LEARNING_RATE = 0.2
        self.CHUNK_SIZE = CHUNK_SIZE
        self.POP = POP

        self.elitism = ELITISM
        self.num_elite = NUM_ELITE
        self.mutation_rate = MUTATION_RATE

        # param settings
        self.CLIP = CLIP

    def compute_ranks(self, x):
        """
        Returns rank as a vector of len(x) with integers from 0 to len(x)
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def compute_centered_ranks(self, x):
        """
        Maps x to [-0.5, 0.5] and returns the rank
        """
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y

    def init_coeffs(self, beta_init, num_edges):
        w = np.random.randn(num_edges) * self.SIGMA
        return np.append(beta_init, w)

    def random_block_sizes(self, i_start, num_params):
        '''
        Returns a list of random-sized steps summing to num_params - i_start
        '''
        blocks = []

        sum = 0
        while sum < (num_params - i_start):
            blocks.append(np.random.randint(1, self.CHUNK_SIZE))
            sum = np.sum(blocks)
        blocks[-1] = (num_params - i_start) - np.sum(blocks[:-1])

        return blocks

    def get_noise_population(self, num_params):

        population = []
        for i in range(int(self.POP / 2)):
            # generate a random number and include both its positive/negative values to get good gradients
            x = []
            x2 = []

            # since Beta is the first param, let it be its own 'chunk'
            rb = np.random.randn() / 10
            # x.append(rb)
            # x2.append(-rb)

            # the rest of the connectivity matrix gets grouped into sizes of CHUNK_SIZE
            # randomized chunk_size
            Ns = self.random_block_sizes(1, num_params)
            for j in Ns:
                n = np.min([j, num_params - j])
                if np.random.rand() > self.mutation_rate:
                    r = np.zeros(n)
                else:
                    r = np.random.randn(n)
                x.append(r)
                x2.append(-r)

            x = np.concatenate(x)
            x2 = np.concatenate(x2)
            # shuffle the mirrored chunks around
            # this way parameters that are far from each other in indices can still be grouped together as chunks
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x = [rb] + [x[i] for i in indices]
            x2 = [-rb] + [x2[i] for i in indices]


            # for j in range(1, num_params, self.CHUNK_SIZE):
            #     n = np.min([self.CHUNK_SIZE, num_params - j])
            #     r = np.random.randn(n)
            #     x.append(r)
            #     x2.append(-r)

            population.append(np.array(x))
            population.append(np.array(x2))

        return population

    def make_solutions(self, coeffs, noise_population):
        if self.CLIP is not None:
            return [np.clip(coeffs + p * self.SIGMA, self.CLIP[0], self.CLIP[1]) for p in noise_population]
        else:
            return [coeffs + p * self.SIGMA for p in noise_population]

    def initIsingNES(self, isings, beta_init):
        # num_edges = settings['size'] * settings['size']
        num_edges = isings[0].maskJ.sum()
        coeffs = self.init_coeffs(beta_init, num_edges)

        noise_population = self.get_noise_population(len(coeffs))  # make noise
        try_coeffs = self.make_solutions(coeffs, noise_population)  # make population of coeffs
        if self.CLIP is not None:
            noise_population = np.array(try_coeffs) - np.array(coeffs)  # get the post-clipped noise vectors
        self.set_solutions(isings, try_coeffs)

        return coeffs, noise_population, try_coeffs

    def loadIsingNES(self, isings):

        self.LEARNING_RATE = 0.001
        self.SIGMA = 0.01

        fitness = [I.avg_energy for I in isings]
        idx_best = np.argmax(fitness)

        try_coeffs = [np.append(I.Beta, I.J[np.where(I.maskJ)]) for I in isings]
        coeffs = try_coeffs[idx_best]
        noise_population = np.array(try_coeffs) - coeffs

        return coeffs, noise_population, try_coeffs

    def update_coeffs(self, rewards, noise_population, coeffs):

        rewards = self.compute_centered_ranks(np.array(rewards))
        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')
        rewards = (rewards - rewards.mean()) / std

        noise_population = np.array(noise_population)

        update_factor = self.LEARNING_RATE / (self.POP * self.SIGMA)
        reward_weighted_noise = update_factor * np.dot(noise_population.T,
                                                       rewards).T  # solutions with high rewards dominate the update
        coeffs = coeffs + reward_weighted_noise

        self.coeffs = coeffs
        self.reward_grad = reward_weighted_noise

        if self.ANNEALING:
            if self.LEARNING_RATE > 0.001:
                self.LEARNING_RATE *= self.DECAY

            if self.SIGMA > 0.01:
                self.SIGMA *= 0.999

        return coeffs

    def set_solutions(self, agents, solutions):

        # f_c = lambda x: 2 * np.tanh(x)  # connectivity encoding function
        f_c = lambda x: x # connectivity encoding function
        f_b = lambda x: 10 ** x         # beta encoding function

        for i, a in enumerate(agents):

            new_beta = solutions[i][0]
            a.Beta = f_b(new_beta)

            # respecting the mask
            new_conn = solutions[i][1:]
            new_conn = f_c(new_conn)

            ix, iy = np.where(a.maskJ)
            for j in range(len(ix)):
                a.J[ix[j], iy[j]] = new_conn[j]

        return agents

    def save_nes(self, folder, nes, gen):

        filename = folder + 'nes_logs/gen[' + str(gen) + ']-isings.pickle'

        compressed_pickle(filename, nes)

