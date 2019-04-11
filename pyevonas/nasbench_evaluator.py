import os
import numpy as np
from pymop.problem import Problem
from pymop.util import load_pareto_front_from_file

from nasbench import api
from pyevonas.util.loader import load

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


ops_mappings = {
    0: CONV3X3,
    1: CONV1X1,
    2: MAXPOOL3X3,
}

adjacency_matrix_mappings = {
    # 7 x 7 connectivity matrix
    # first row
    0: (0, 1),
    1: (0, 2),
    2: (0, 3),
    3: (0, 4),
    4: (0, 5),
    5: (0, 6),
    # second row
    6: (1, 2),
    7: (1, 3),
    8: (1, 4),
    9: (1, 5),
    10: (1, 6),
    # third row
    11: (2, 3),
    12: (2, 4),
    13: (2, 5),
    14: (2, 6),
    # forth row
    15: (3, 4),
    16: (3, 5),
    17: (3, 6),
    # fifth row,
    18: (4, 5),
    19: (4, 6),
    # sixth row,
    20: (5, 6)
}


class NASBench(Problem):
    def __init__(self):
        super().__init__(n_var=26, n_obj=2, n_constr=0, type_var=np.int)
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        self.xu[:5] = 2 * np.ones(5)
        self.nasbench = load(use_pickle=True, full=False)
        self.nasbench.reset_budget_counters()
        self.hash_archive = []  # stores the hash tags of every evaluated model
        self.times, self.best_valids, self.best_tests = [0.0], [0.0], [0.0]

    def _translate(self, _x):
        """
        x is an integer array of 26 variables.
        first 5 elements indicates the options
        the remaining 21 indicates the upper triangular matrix
        """
        ops = []
        for i in range(5):
            ops.append(ops_mappings[_x[i]])
        ops = [INPUT] + ops + [OUTPUT]

        matrix_locs = _x[5:]
        matrix = np.zeros((7, 7), dtype=int)
        for i in range(len(matrix_locs)):
            if _x[i] > 0:
                matrix[adjacency_matrix_mappings[i]] = int(1)

        return ops, matrix

    def is_valid(self, _x):
        ops, matrix = self._translate(_x)
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if self.nasbench.is_valid(spec):
            # let's save this anyway to see how many times evaluations are spent on duplicates
            # self.hash_archive.append(self.nasbench._hash_spec(spec))
            # return True
            hash_tag = self.nasbench._hash_spec(spec)
            if not(hash_tag in self.hash_archive):
                self.hash_archive.append(hash_tag)
                return True

        return False

    def query(self, spec):
        data = self.nasbench.query(spec)
        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if data['validation_accuracy'] > self.best_valids[-1]:
            self.best_valids.append(data['validation_accuracy'])
            self.best_tests.append(data['test_accuracy'])
        else:
            self.best_valids.append(self.best_valids[-1])
            self.best_tests.append(self.best_tests[-1])

        time_spent, _ = self.nasbench.get_budget_counters()
        self.times.append(time_spent)

        return data['validation_accuracy'], data['training_time']

    def _evaluate(self, x, out, *args, **kwargs):

        f1 = np.full((x.shape[0], 1), np.inf)
        f2 = np.full((x.shape[0], 1), np.inf)

        for idx in range(x.shape[0]):
            ops, matrix = self._translate(x[idx])
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            metric = self.query(spec)
            f1[idx, 0] = 1 - metric[0]
            f2[idx, 0] = metric[1]

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self):
        pf_pth = os.path.join('pfs', 'acc_vs_time.pf')
        return load_pareto_front_from_file(pf_pth)


if __name__ == '__main__':
    problem = NASBench()
    x = np.array([1, 0, 0, 0, 2,
                   1, 1, 1, 0, 1, 0,
                   0, 0, 0, 0, 1,
                   0, 0, 0, 1,
                   1, 0, 0,
                   0, 1,
                   1])
    problem.evaluate(x)
