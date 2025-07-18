import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_parameter_controller import MetaParameterController
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_meta_controller_adjusts_threshold():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, plasticity_threshold=5.0)
    mpc = MetaParameterController(adjustment=1.0, min_threshold=1.0, max_threshold=10.0)

    mpc.record_loss(1.0)
    mpc.record_loss(2.0)
    mpc.adjust(nb)
    assert nb.plasticity_threshold == 4.0

    mpc.record_loss(1.5)
    mpc.adjust(nb)
    assert nb.plasticity_threshold == 5.0
