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


def test_apply_gradient_prune_mask_removes_flagged_synapses():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    original = list(core.synapses)
    for i, syn in enumerate(original):
        nb._prev_gradients[syn] = float(i)
    mask = nb.compute_gradient_prune_mask(0.5)
    removed = nb.apply_gradient_prune_mask(mask)
    assert removed == len([m for m in mask if m])
    remaining = [s for s, m in zip(original, mask) if not m]
    assert core.synapses == remaining
    for s in remaining:
        assert s in nb._prev_gradients
    pruned = [s for s, m in zip(original, mask) if m]
    for s in pruned:
        assert s not in nb._prev_gradients


def test_train_runs_with_gradient_pruning():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(
        core, gradient_prune_ratio=0.5, structural_plasticity_enabled=False
    )
    for syn in core.synapses:
        nb._prev_gradients[syn] = 1.0
    initial = len(core.synapses)
    nb.train([(0.1, 0.2)], epochs=1)
    assert len(core.synapses) <= initial
