import tensor_backend as tb
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_compute_gradient_prune_mask_selects_low_gradients():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    for i, syn in enumerate(core.synapses):
        nb._prev_gradients[syn] = float(i)
    mask = nb.compute_gradient_prune_mask(0.5)
    assert len(mask) == len(core.synapses)
    assert sum(mask) == len(core.synapses) // 2
    assert mask[0]
    assert not mask[-1]
