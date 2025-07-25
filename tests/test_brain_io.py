import os
from pathlib import Path
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from neuromodulatory_system import NeuromodulatorySystem
import torch

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
    mv = MetricsVisualizer(log_dir="tb_logs", csv_log_path="metrics.csv")
    mv.update({'loss': 0.5, 'vram_usage': 0.1, 'arousal': 0.2,
               'stress': 0.1, 'reward': 0.3,
               'plasticity_threshold': 5.0,
               'message_passing_change': 0.05,
               'compression_ratio': 0.8})
    assert mv.metrics['loss'][-1] == 0.5
    assert mv.metrics['vram_usage'][-1] == 0.1
    assert mv.metrics['arousal'][-1] == 0.2
    assert mv.metrics['plasticity_threshold'][-1] == 5.0
    mv.close()
    assert os.path.exists("metrics.csv")
    assert any(p.name.startswith("events.out.tfevents") for p in Path("tb_logs").iterdir())


def test_brain_neuromodulatory_system_integration():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    brain = Brain(core, nb, DataLoader(), neuromodulatory_system=ns, save_dir="saved_models")
    ns.update_signals(arousal=0.2)
    assert brain.neuromodulatory_system.get_context()['arousal'] == 0.2


def test_brain_dream_defaults():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), dream_num_cycles=3, dream_interval=2)
    assert brain.dream_num_cycles == 3
    assert brain.dream_interval == 2


def test_train_with_pytorch_dataloader_and_infer(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), save_dir=str(tmp_path))

    dataset = torch.utils.data.TensorDataset(
        torch.tensor([0.1, 0.2], dtype=torch.float32),
        torch.tensor([0.2, 0.4], dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    brain.train(loader, epochs=1)
    brain.save_model()
    path = brain.saved_model_paths[0]

    new_core = Core(params)
    new_nb = Neuronenblitz(new_core)
    new_brain = Brain(new_core, new_nb, DataLoader(), save_dir=str(tmp_path))
    new_brain.load_model(path)
    out = new_brain.infer(0.1)
    assert isinstance(out, float)
