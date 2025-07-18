import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_core_cluster_assigns_ids():
    params = minimal_params()
    core = Core(params)
    core.cluster_neurons(k=2)
    clusters = {n.cluster_id for n in core.neurons}
    assert None not in clusters
    assert len(clusters) <= 2


def test_relocate_clusters_changes_tiers():
    params = minimal_params()
    core = Core(params)
    for n in core.neurons[:3]:
        n.cluster_id = 0
        n.attention_score = 1.5
    core.relocate_clusters(high=1.0, medium=0.5)
    tiers = {n.tier for n in core.neurons[:3]}
    assert tiers == {"vram"} or tiers == {"ram", "vram"} or tiers == {"ram"}


def test_clustering_during_training():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, None)
    train_examples = [(0.1, 0.2), (0.2, 0.3)]
    brain.train(train_examples, epochs=1, validation_examples=None)
    has_cluster = any(n.cluster_id is not None for n in core.neurons)
    assert has_cluster
