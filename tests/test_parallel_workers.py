import random
import numpy as np

from marble import Neuronenblitz
from marble_core import Core
from tests.test_core_functions import minimal_params


def test_train_in_parallel_single_worker_deterministic():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["plasticity_threshold"] = float("inf")
    examples = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]

    core_seq = Core(params)
    nb_seq = Neuronenblitz(core_seq, parallel_wanderers=1, plasticity_threshold=float("inf"))
    for ex in examples:
        nb_seq.train_example(*ex)
    weights_seq = [s.weight for s in core_seq.synapses]

    random.seed(0)
    np.random.seed(0)
    core_par = Core(params)
    nb_par = Neuronenblitz(core_par, parallel_wanderers=1, plasticity_threshold=float("inf"))
    nb_par.train_in_parallel(examples)
    weights_par = [s.weight for s in core_par.synapses]

    assert weights_seq == weights_par


def test_train_in_parallel_multiple_workers_metrics():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["plasticity_threshold"] = float("inf")
    core = Core(params)
    nb = Neuronenblitz(core, parallel_wanderers=2, plasticity_threshold=float("inf"))
    examples = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    summaries = nb.train_in_parallel(examples, max_workers=2)
    total = sum(s["examples"] for s in summaries.values())
    assert total == len(examples)
    assert len(summaries) == 2
