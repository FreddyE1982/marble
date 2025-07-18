import os
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain

from tests.test_core_functions import minimal_params


def combine(x, w):
    return max(x * w, 0)


def loss_fn(target, output):
    return target - output


def weight_update_fn(source, error, path_len):
    return (error * source) / (path_len + 1)


def test_brain_save_and_load(tmp_path):
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, combine_fn=combine, loss_fn=loss_fn,
                       weight_update_fn=weight_update_fn)
    brain = Brain(core, nb, DataLoader(), save_dir=str(tmp_path))

    brain.save_model()
    assert len(brain.saved_model_paths) == 1
    saved_path = brain.saved_model_paths[0]
    assert os.path.exists(saved_path)

    core.expand(num_new_neurons=1, num_new_synapses=1)
    old_count = len(core.neurons)
    brain.load_model(saved_path)
    assert isinstance(brain.core, Core)
    assert len(brain.core.neurons) != old_count


def test_metrics_visualizer_update():
    from marble_base import MetricsVisualizer
    mv = MetricsVisualizer()
    mv.update({'loss': 0.5, 'vram_usage': 0.1})
    assert mv.metrics['loss'][-1] == 0.5
    assert mv.metrics['vram_usage'][-1] == 0.1
