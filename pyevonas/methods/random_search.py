# Initialize the NASBench object which parses the raw data into memory (this
# should only be run once as it takes up to a few minutes).

import sys
project_root_pth = "/Users/zhichao.lu/Dropbox/2019/pyevonas"
sys.path.insert(0, project_root_pth)

import argparse
# Standard imports
import os
import pickle
import time

import numpy as np
from nasbench import api

from pyevonas.util.loader import load

parser = argparse.ArgumentParser("Random search on NASBench 101")
parser.add_argument('--n_runs', type=int, default=51, help='number of independent runs')
parser.add_argument('--max_time_budget', type=int, default=1e7, help='max time budget')
parser.add_argument('--deduplicate', action='store_true', default=True, help='remove duplicates')
parser.add_argument('--save', type=str, default='RSearch', help='experiment name')
args = parser.parse_args()

if args.deduplicate:
    args.save = '{}-{}-{}'.format(args.save, 'Deduplicate', time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

abs_pth = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(abs_pth, args.save))

# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = load(use_pickle=True, full=False)

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


# ------------------------------ Operators ------------------------------- #
def random_spec():
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec


def random_spec_deduplicates(hash_archive):
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


# ------------------------------ Methods Main Routine ------------------------------- #
def run_random_search(seed=0, max_time_budget=5e6):
    """Run a single roll-out of random search to a fixed time budget."""
    np.random.seed(seed)
    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    while True:
        spec = random_spec()
        data = nasbench.query(spec)

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        if time_spent > max_time_budget:
            # Break the first time we exceed the budget.
            break

    return times, best_valids, best_tests


def run_random_search_deduplicates(seed=0, max_time_budget=5e6):
    """Run a single roll-out of random search to a fixed time budget."""
    """Removes duplicates network sampled """
    np.random.seed(seed)
    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    archive = []  # stores hash tag of every evaluated network

    while True:
        spec, spec_hash = random_spec_deduplicates(archive)
        archive.append(spec_hash)
        data = nasbench.query(spec)

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        if time_spent > max_time_budget:
            # Break the first time we exceed the budget.
            break

    return times, best_valids, best_tests


def run_random_search_deduplicates_threshold(seed=0, threshold=0.01):
    """Run a single roll-out of random search to a fixed time budget."""
    """Removes duplicates network sampled """
    np.random.seed(seed)
    nasbench.reset_budget_counters()
    archive = []  # stores hash tag of every evaluated network
    train_time = 0

    while True:
        spec, spec_hash = random_spec_deduplicates(archive)
        archive.append(spec_hash)
        # data = nasbench.query(spec)

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        fixed, computed = nasbench.get_metrics_from_hash(spec_hash)
        mean_valid_acc, mean_test_acc, mean_train_time = extract_mean_statistics(computed[108])

        train_time += mean_train_time
        valid_acc_regret = (BEST_MEAN_VALID_ACC - mean_valid_acc)

        # terminate if the current valid accuracy exceeds the threshold
        if valid_acc_regret < threshold:
            break

    return len(archive), train_time


def main():
    results = []
    for run in range(args.n_runs):
        if args.deduplicate:
            times, best_valid, best_test = run_random_search_deduplicates(seed=run,
                                                                          max_time_budget=args.max_time_budget)
        else:
            times, best_valid, best_test = run_random_search(seed=run,
                                                             max_time_budget=args.max_time_budget)
        results.append((times, best_valid, best_test))

    with open(os.path.join(args.save, 'rsearch.pkl'), 'wb') as handle:
        pickle.dump(results, handle, protocol=0)

    return


def main_count():
    results = []
    threshold = [0.01, 0.005]
    for thr in threshold:
        result = []
        for run in range(args.n_runs):
            trials, time_spent = run_random_search_deduplicates_threshold(seed=run, threshold=thr)
            result.append((trials, time_spent))
        results.append((thr, result))

    with open(os.path.join(args.save, 'rsearch.pkl'), 'wb') as handle:
        pickle.dump(results, handle, protocol=0)

    return


if __name__ == '__main__':
    main_count()