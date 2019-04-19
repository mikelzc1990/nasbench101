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
parser.add_argument('--selection_epochs', type=int, default=108,
                    help='selection of models based on acc @ this epoch')
parser.add_argument('--FEs', type=int, default=1000, help='maximum # of model samples')
parser.add_argument('--save', type=str, default='params-RSearch', help='experiment name')

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

# # Best mean accuracy
BEST_MEAN_TEST_ACC = 0.9442107081413269
BEST_MEAN_VALID_ACC = 0.9505542318026224

# Best mean test accuracy given params < 800,000
# BEST_MEAN_TEST_ACC = 0.8922609488169352
# BEST_MEAN_VALID_ACC = 0.8984708984692892

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


def is_valid(spec, hash_archive):
    # define what qualify as a valid model
    if nasbench.is_valid(spec):
        model_hash = nasbench._hash_spec(spec)
        fixed, _ = nasbench.get_metrics_from_hash(model_hash)
        if fixed['trainable_parameters'] < 800000:
            if not (model_hash in hash_archive):
                return spec, model_hash
    return None


def random_spec(hash_archive):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)

        check = is_valid(spec, hash_archive)

        if check is not None:
            return check[0], check[1]

        # if nasbench.is_valid(spec):
        #     model_hash = nasbench._hash_spec(spec)
        #     if not (model_hash in hash_archive):
        #         return spec, model_hash


# ------------------------------ Methods Main Routine ------------------------------- #
def run_random_search(seed=0,
                      deduplicates=True,  # if true, no re-evaluate if isomorphic networks
                      selection_epochs=108,  # select architecture based on acc@selection_epochs
                      ):
    """Run a single roll-out of random search to a specified termination condition."""
    np.random.seed(seed)
    random.seed(seed)
    nasbench.reset_budget_counters()
    times, best_valids, best_tests, best_valid_acc_regret = [0.0], [0.0], [0.0], [BEST_MEAN_VALID_ACC]
    archive = []  # stores hash tag of every evaluated network
    n_model_sampled = 0

    while True:
        spec, spec_hash = random_spec(archive)
        # keep track of the hash tags of evaluated networks
        # if you don't want isomorphic networks to be re-evaluated
        if deduplicates:
            archive.append(spec_hash)

        data = nasbench.query(spec, epochs=selection_epochs)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        n_model_sampled += 1

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        # offline checking just for termination
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


def main(seed):
    # execute one run of the random search
    results = run_random_search(seed=seed,
                                deduplicates=args.deduplicate,
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
    experiment()
