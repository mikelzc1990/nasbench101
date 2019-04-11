import pickle
import time

import os.path
from nasbench import api


DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "data")
NASBENCH_TFRECORD_FULL = '%s/nasbench_full.%s' % (DATA, "%s")
NASBENCH_TFRECORD_PARTIAL = '%s/nasbench_only108.%s' % (DATA, "%s")


def load(use_pickle=True, full=False):
    fname = NASBENCH_TFRECORD_FULL if full else NASBENCH_TFRECORD_PARTIAL

    start = time.time()
    if use_pickle:

        if not os.path.isfile(fname % 'dat'):
            nasbench = api.NASBench(fname % 'tfrecord')
            pickle.dump(nasbench, open(fname % 'dat', "wb"))

        nasbench = pickle.load(open(fname % 'dat', "rb"))

    else:
        nasbench = api.NASBench(fname % 'tfrecord')

    end = time.time()
    print("Data loaded in %s seconds" % (end - start))
    return nasbench
