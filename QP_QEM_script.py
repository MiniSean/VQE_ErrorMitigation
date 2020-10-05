'''
This is a very messy implementation of quasi-probability error cancellation.
As discussed, please only refer to it as a 'practical' guide for Simon Benjamin's paper.

Eleanor
'''

import openfermion as of
import openfermioncirq as ofc
import cirq
import numpy as np
import itertools as it
import math
import json

from scipy.sparse.linalg import expm, svds
from scipy import sparse
from typing import Sequence, Tuple, Dict, Any
from tqdm import tqdm
import time

import QP_QEM_lib as qpl
from importlib import reload
reload(qpl)

if __name__ == '__main__':
    n = 2
    error_type = 'depolarize'
    error = 1e-2

    circuit_qpl = qpl.swap_circuit(n, cirq.LineQubit.range(2*n+1), True)
    circuit_vis = qpl.swap_circuit(n, cirq.LineQubit.range(2*n+1), False)

    print(circuit_vis)

    experiment = qpl.QPSamp(circuit_qpl, error_type, error)
    #experiment.run_experiment(meas_reps=10000, sample_reps=100, stat_reps=100)

    start_time = time.time()
    print(start_time)
    experiment.run_experiment(meas_reps=1000, sample_reps=100, stat_reps=1)
    print(f'Run time: {time.time()-start_time} sec')
