import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from n_dimensional_topology import NDimensionalTopologyManager


def test_decrease_representation_size():
    params = minimal_params()
    core = Core(params)
    initial = core.rep_size
    core.increase_representation_size(2)
    core.decrease_representation_size(1)
    assert core.rep_size == initial + 1
    for n in core.neurons:
        assert len(n.representation) == core.rep_size


def test_nd_topology_add_remove():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    manager = NDimensionalTopologyManager(
        core,
        nb,
        enabled=True,
        target_dimensions=core.rep_size + 1,
        attention_threshold=0.0,
        loss_improve_threshold=0.1,
        stagnation_epochs=1,
    )
    manager.evaluate(1.0)  # stagnation triggers add
    assert core.rep_size == params["representation_size"] + 1
    manager.evaluate(1.0)  # no improvement -> remove
    assert core.rep_size == params["representation_size"]
