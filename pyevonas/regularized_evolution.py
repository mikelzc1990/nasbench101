# Initialize the NASBench object which parses the raw data into memory (this
# should only be run once as it takes up to a few minutes).

import sys
project_root_pth = "/home/zhichao/nasbench101"
sys.path.insert(0, project_root_pth)

from nasbench import api

# Standard imports
import os
import copy
import numpy as np
import random
import time
import argparse
import pickle

from pyevonas.util.loader import load


parser = argparse.ArgumentParser("Random search on NASBench 101")
parser.add_argument('--seed', type=int, default=4, help='random seed')
parser.add_argument('--n_runs', type=int, default=11, help='number of independent runs')
parser.add_argument('--deduplicate', action='store_true', default=True, help='remove duplicates')
parser.add_argument('--save', type=str, default='../REvolution', help='experiment name')
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--tournament_size', type=int, default=5, help='tournament size')
parser.add_argument('--selection_epochs', type=int, default=108,
                    help='selection of models based on acc @ this epoch')
parser.add_argument('--FEs', type=int, default=1000, help='maximum # of model samples')

args = parser.parse_args()

if not args.deduplicate:
    args.save = 'nasbench-{}-{}-{}'.format(args.save, 'isomorphic', time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = 'nasbench-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

abs_pth = os.path.dirname(os.path.abspath(__file__))
args.save = os.path.join(abs_pth, args.save)
os.makedirs(args.save)

# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = load(use_pickle=True, full=True)

# Best mean test accuracy
BEST_MEAN_TEST_ACC = 0.9442107081413269
BEST_MEAN_VALID_ACC = 0.9505542318026224

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


def extract_mean_statistics(stats):
    valid_acc_sum, test_acc_sum, train_time_sum = 0, 0, 0
    for sample in stats:
        valid_acc_sum += sample['final_validation_accuracy']
        test_acc_sum += sample['final_test_accuracy']
        train_time_sum += sample['final_training_time']

    return valid_acc_sum / len(stats), test_acc_sum / len(stats), train_time_sum / len(stats)


def random_spec(hash_archive):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)

        if nasbench.is_valid(spec):
            model_hash = nasbench._hash_spec(spec)
            if not (model_hash in hash_archive):
                return spec, model_hash


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))

    return tuple(pool[i] for i in indices)


def mutate_spec(old_spec, hash_archive, mutation_rate=1.0):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)

        # In expectation, V edges flipped (note that most end up being pruned).
        edge_mutation_prob = mutation_rate / NUM_VERTICES
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                if random.random() < edge_mutation_prob:
                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

        # In expectation, one op is resampled.
        op_mutation_prob = mutation_rate / OP_SPOTS
        for ind in range(1, NUM_VERTICES - 1):
            if random.random() < op_mutation_prob:
                available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
                new_ops[ind] = random.choice(available)

        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            model_hash = nasbench._hash_spec(new_spec)
            if not (model_hash in hash_archive):
                return new_spec, model_hash


# ------------------------------ Methods Main Routine ------------------------------- #
def run_evolution_search(seed=0,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0,
                         deduplicates=True,  # if true, no re-evaluate if isomorphic networks
                         selection_epochs=108,  # select architecture based on acc@selection_epochs
                         ):
    """Run a single roll-out of regularized evolution to a specified termination condition."""

    np.random.seed(seed)
    random.seed(seed)
    nasbench.reset_budget_counters()
    times, best_valids, best_tests, best_valid_acc_regret = [0.0], [0.0], [0.0], [BEST_MEAN_VALID_ACC]
    population = []  # (validation, spec) tuples
    archive = []  # stores hash tag of every evaluated network
    n_model_sampled = 0

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    for _ in range(population_size):
        spec, spec_hash = random_spec(archive)
        # keep track of the hash tags of evaluated networks
        # if you don't want isomorphic networks to be re-evaluated
        if deduplicates:
            archive.append(spec_hash)

        data = nasbench.query(spec, epochs=selection_epochs)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))
        n_model_sampled += 1

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        # offline checking
        fixed, computed = nasbench.get_metrics_from_hash(spec_hash)
        mean_valid_acc, _, mean_train_time = extract_mean_statistics(computed[108])
        valid_acc_regret = (BEST_MEAN_VALID_ACC - mean_valid_acc)

        if valid_acc_regret < best_valid_acc_regret[-1]:
            best_valid_acc_regret.append(valid_acc_regret)
        else:
            best_valid_acc_regret.append(best_valid_acc_regret[-1])

        # check termination criterion
        if n_model_sampled >= args.FEs:
            return times, best_valids, best_tests, best_valid_acc_regret

    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]

        new_spec, new_spec_hash = mutate_spec(best_spec, archive, mutation_rate)
        # keep track of the hash tags of evaluated networks
        # if you don't want isomorphic networks to be re-evaluated
        if deduplicates:
            archive.append(new_spec_hash)

        data = nasbench.query(new_spec, epochs=selection_epochs)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        n_model_sampled += 1

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        # offline checking just for termination
        fixed, computed = nasbench.get_metrics_from_hash(new_spec_hash)
        mean_valid_acc, _, mean_train_time = extract_mean_statistics(computed[108])
        valid_acc_regret = (BEST_MEAN_VALID_ACC - mean_valid_acc)

        if valid_acc_regret < best_valid_acc_regret[-1]:
            best_valid_acc_regret.append(valid_acc_regret)
        else:
            best_valid_acc_regret.append(best_valid_acc_regret[-1])

        # check termination criterion
        if n_model_sampled >= args.FEs:
            return times, best_valids, best_tests, best_valid_acc_regret


def main(seed):
    # execute one run of the evolution search
    results = run_evolution_search(seed=seed, population_size=args.pop_size,
                                   tournament_size=args.tournament_size,
                                   mutation_rate=1.0, deduplicates=args.deduplicate,
                                   selection_epochs=args.selection_epochs)
    return results


def experiment():
    import multiprocessing as mp

    np.random.seed(args.seed)
    seeds = np.random.permutation(500)[:args.n_runs].tolist()

    pool = mp.Pool(mp.cpu_count())

    data = pool.map(main, seeds)

    pool.close()

    with open(os.path.join(args.save, 'data.pkl'), 'wb') as handle:
        pickle.dump(data, handle, protocol=0)

    # save args to file
    with open(os.path.join(args.save, 'args.pkl'), 'wb') as handle:
        pickle.dump(args, handle, protocol=0)

    return


if __name__ == '__main__':
    # main(0)
    experiment()
