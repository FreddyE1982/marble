import random
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_mutate_synapses_changes_weights():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    random.seed(0)
    weights_before = [s.weight for s in core.synapses]
    mutated = brain.mutate_synapses(mutation_rate=1.0, mutation_strength=0.1)
    weights_after = [s.weight for s in core.synapses]
    assert mutated == len(core.synapses)
    assert weights_before != weights_after


def test_prune_weak_synapses_removes_small_weights():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    for syn in core.synapses:
        syn.weight = 0.001
    initial_count = len(core.synapses)
    pruned = brain.prune_weak_synapses(threshold=0.01)
    assert pruned == initial_count
    assert len(core.synapses) == 0


def test_evolve_combines_mutation_and_prune():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    for syn in core.synapses:
        syn.weight = 0.0
    initial_count = len(core.synapses)
    mutated, pruned = brain.evolve(mutation_rate=1.0, mutation_strength=0.0, prune_threshold=0.01)
    assert mutated == initial_count
    assert pruned == initial_count
    assert len(core.synapses) == 0


def test_evolve_uses_config_defaults():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    for syn in core.synapses:
        syn.weight = 0.0
    random.seed(0)
    brain = Brain(core, nb, DataLoader(),
                  mutation_rate=1.0,
                  mutation_strength=0.0,
                  prune_threshold=0.01)
    initial = len(core.synapses)
    mutated, pruned = brain.evolve()
    assert mutated == initial
    assert pruned == initial
    assert len(core.synapses) == 0
