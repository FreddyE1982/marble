import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from distributed_training import DistributedTrainer


def test_distributed_training_spawns():
    params = minimal_params()
    trainer = DistributedTrainer(params, world_size=1)
    trainer._worker(0, [(0.1, 0.2)])
