import time
import pickle
import numpy as np

from pymoo.util import plotting
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.rand import random
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.crossover.uniform_crossover import BinaryUniformCrossover
from pymoo.optimize import minimize

from pyevonas.nasbench_evaluator import NASBench

# file where a snapshot will be saved
fname = "algorithm.dat"


# define a callback function that prints the X and F value of the best individual
def my_callback(algorithm):

    # get the best individual and print it
    best = algorithm.pop[0]
    # print(algorithm.n_gen, 1-best.F)

    time_spent, _ = algorithm.problem.nasbench.get_budget_counters()
    print('Estimated time spent = {} hours'.format(time_spent/3600))
    # # pickle the algorithm if it might fail for whatever reason
    # with open(fname, 'wb') as f:
    #     pickle.dump(algorithm, f)


def repair(problem, pop, **kwargs):
    pop.set("X", np.round(pop.get("X")).astype(np.int))
    return pop


def is_duplicate(pop, *other, epsilon=1e-20, **kwargs):

    X = pop.get("X")

    # value to finally return
    is_duplicate = np.full(len(pop), True)

    for i in range(X.shape[0]):
        if kwargs['algorithm'].problem.is_valid(X[i, :]):
            is_duplicate[i] = False

    return is_duplicate


class IntegerBitflipMutation(Mutation):
    def __init__(self, prob_mut=None):
        super().__init__()
        self.p_mut = prob_mut

    def _do(self, problem, pop, **kwargs):
        if self.p_mut is None:
            self.p_mut = 1.0 / problem.n_var

        X = pop.get("X")
        _X = np.full(X.shape, np.inf)

        for k in range(X.shape[0]):
            for i in range(X.shape[1]):
                if random.random() < self.p_mut:
                    available_choices = np.arange(problem.xl[i], problem.xu[i]+1).tolist()
                    available_choices.remove(X[k, i])  # removes the current value
                    _X[k, i] = np.random.choice(available_choices)
                else:
                    _X[k, i] = X[k, i]

        return pop.new("X", _X.astype(np.int))


class RandomSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def sample(self, problem, pop, n_samples, **kwargs):

        # loop until n_samples are met
        m, counter = problem.n_var, 0
        val = np.full((n_samples, m), np.nan)
        while True:
            for i in range(m):
                val[counter, i] = random.randint(low=problem.xl[i],
                                                 high=problem.xu[i]+1)
            if problem.is_valid(val[counter, :]):
                counter += 1

            if counter >= n_samples:
                break

        return pop.new("X", val)


start = time.time()
problem = NASBench()
pf = problem.pareto_front()
res = minimize(problem,
               seed=0,
               method='nsga2',
               method_args={
                   'pop_size': 100,
                   'sampling': RandomSampling(),
                   # 'crossover': SimulatedBinaryCrossover(prob_cross=0.0, eta_cross=3),
                   'crossover': BinaryUniformCrossover(),
                   'mutation': IntegerBitflipMutation(prob_mut=0.1),
                   'eliminate_duplicates': is_duplicate,
                   'func_repair': repair
               },
               termination=('n_gen', 100),
               # callback=my_callback,
               disp=True
               )

results = {
    'times': problem.times,
    'best_valids': problem.best_valids,
    'best_tests': problem.best_tests,
    'hash_archive': problem.hash_archive,
    'Function value': res.opt.get('F'),
    'Variable value': res.opt.get('X'),

}

# with open('nsga2.dat', 'wb') as handle:
#     pickle.dump(results, handle)

plot = True
if plot:
    plotting.plot(pf, res.opt.get('F'), labels=["Pareto-front", "F"])

# print("Time elapsed: {}".format(time.time() - start))
# print("Function value: %s" % res.F)
# print("Total time spent = {} hours, best valid acc = {}, best test acc = {}"
#       .format(problem.times[-1]/3600, problem.best_valids[-1], problem.best_tests[-1]))
# print("{}% of the model sampled are unique".format(len(list(set(problem.hash_archive))) /
#                                                    len(problem.hash_archive)*100))

