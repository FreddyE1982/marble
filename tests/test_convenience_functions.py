import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_imports
import marble_brain
import marble_main
from marble_base import MetricsVisualizer
from tqdm import tqdm as std_tqdm
from tests.test_core_functions import minimal_params

from marble_interface import (
    new_marble_system,
    save_core_json_file,
    load_core_json_file,
    add_neuron_to_marble,
    add_synapse_to_marble,
    freeze_synapses_fraction,
    expand_marble_core,
    run_core_message_passing,
    increase_marble_representation,
    decrease_marble_representation,
    enable_marble_rl,
    disable_marble_rl,
    marble_select_action,
    marble_update_q,
    cluster_marble_neurons,
    relocate_marble_clusters,
    extract_submarble,
    get_marble_status,
    reset_core_representations,
    randomize_core_representations,
    count_marble_synapses,
)


def _create_marble(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return new_marble_system(str(cfg_path))


def test_save_core_json_file(tmp_path):
    m = _create_marble(tmp_path)
    p = tmp_path / "core.json"
    save_core_json_file(m, p)
    assert p.exists()


def test_load_core_json_file(tmp_path):
    m = _create_marble(tmp_path)
    p = tmp_path / "core.json"
    save_core_json_file(m, p)
    m2 = load_core_json_file(p)
    assert len(m2.get_core().neurons) == len(m.get_core().neurons)


def test_add_neuron_to_marble(tmp_path):
    m = _create_marble(tmp_path)
    n_before = len(m.get_core().neurons)
    nid = add_neuron_to_marble(m)
    assert len(m.get_core().neurons) == n_before + 1
    assert nid == n_before


def test_add_synapse_to_marble(tmp_path):
    m = _create_marble(tmp_path)
    add_neuron_to_marble(m)
    add_synapse_to_marble(m, 0, 1, weight=0.5)
    assert any(s.source == 0 and s.target == 1 for s in m.get_core().synapses)


def test_freeze_synapses_fraction(tmp_path):
    m = _create_marble(tmp_path)
    expand_marble_core(m, num_new_neurons=2, num_new_synapses=4)
    freeze_synapses_fraction(m, 0.5)
    frozen = sum(s.frozen for s in m.get_core().synapses)
    assert frozen > 0


def test_expand_marble_core(tmp_path):
    m = _create_marble(tmp_path)
    n_before = len(m.get_core().neurons)
    expand_marble_core(m, num_new_neurons=2, num_new_synapses=1)
    assert len(m.get_core().neurons) == n_before + 2


def test_run_core_message_passing(tmp_path):
    m = _create_marble(tmp_path)
    delta = run_core_message_passing(m, iterations=1)
    assert isinstance(delta, float)


def test_increase_marble_representation(tmp_path):
    m = _create_marble(tmp_path)
    rep = m.get_core().rep_size
    increase_marble_representation(m, 1)
    assert m.get_core().rep_size == rep + 1


def test_decrease_marble_representation(tmp_path):
    m = _create_marble(tmp_path)
    increase_marble_representation(m, 1)
    rep = m.get_core().rep_size
    decrease_marble_representation(m, 1)
    assert m.get_core().rep_size == rep - 1


def test_enable_marble_rl(tmp_path):
    m = _create_marble(tmp_path)
    enable_marble_rl(m)
    assert m.get_core().rl_enabled


def test_disable_marble_rl(tmp_path):
    m = _create_marble(tmp_path)
    enable_marble_rl(m)
    disable_marble_rl(m)
    assert not m.get_core().rl_enabled


def test_marble_select_action(tmp_path):
    m = _create_marble(tmp_path)
    enable_marble_rl(m)
    action = marble_select_action(m, "state", 2)
    assert isinstance(action, int)


def test_marble_update_q(tmp_path):
    m = _create_marble(tmp_path)
    enable_marble_rl(m)
    marble_update_q(m, "s", 1, 1.0, "ns", False, n_actions=2)


def test_cluster_marble_neurons(tmp_path):
    m = _create_marble(tmp_path)
    cluster_marble_neurons(m, k=2)
    assert all(n.cluster_id is not None for n in m.get_core().neurons)


def test_relocate_marble_clusters(tmp_path):
    m = _create_marble(tmp_path)
    cluster_marble_neurons(m, k=2)
    relocate_marble_clusters(m)


def test_extract_submarble(tmp_path):
    m = _create_marble(tmp_path)
    sub = extract_submarble(m, [0])
    assert len(sub.get_core().neurons) == 1


def test_get_marble_status(tmp_path):
    m = _create_marble(tmp_path)
    status = get_marble_status(m)
    assert isinstance(status, dict)


def test_reset_core_representations(tmp_path):
    m = _create_marble(tmp_path)
    randomize_core_representations(m, 0.1)
    reset_core_representations(m)
    assert all(np.allclose(n.representation, 0) for n in m.get_core().neurons)


def test_randomize_core_representations(tmp_path):
    m = _create_marble(tmp_path)
    randomize_core_representations(m, 0.1)
    assert any(np.any(n.representation != 0) for n in m.get_core().neurons)


def test_count_marble_synapses(tmp_path):
    m = _create_marble(tmp_path)
    cnt = count_marble_synapses(m)
    assert cnt == len(m.get_core().synapses)


def test_convenience_functions_together(tmp_path):
    m = _create_marble(tmp_path)
    add_neuron_to_marble(m)
    add_synapse_to_marble(m, 0, 1)
    randomize_core_representations(m, 0.1)
    increase_marble_representation(m)
    run_core_message_passing(m)
    status = get_marble_status(m)
    assert isinstance(status, dict)
