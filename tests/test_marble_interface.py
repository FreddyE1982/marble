import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_imports
import marble_brain
import marble_main
from marble_base import MetricsVisualizer
from tqdm import tqdm as std_tqdm
from tests.test_core_functions import minimal_params

from marble_interface import (
    new_marble_system,
    configure_marble_system,
    save_marble_system,
    load_marble_system,
    infer_marble_system,
    train_marble_system,
    set_dreaming,
    set_autograd,
)


def test_save_and_load_marble(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    m = new_marble_system(str(cfg_path))
    train_marble_system(m, [(0.1, 0.2), (0.2, 0.4)], epochs=1)
    acts = m.get_neuronenblitz().global_activation_count
    save_path = tmp_path / "marble.pkl"
    save_marble_system(m, str(save_path))

    loaded = load_marble_system(str(save_path))
    assert len(loaded.get_core().neurons) == len(m.get_core().neurons)
    assert loaded.get_neuronenblitz().global_activation_count == acts
    out = infer_marble_system(loaded, 0.1)
    assert isinstance(out, float)


def test_toggle_features(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    m = new_marble_system(str(cfg_path))
    set_dreaming(m, True)
    assert m.get_brain().dreaming_active
    set_dreaming(m, False)
    assert not m.get_brain().dreaming_active
    set_autograd(m, True)
    assert m.get_autograd_layer() is not None
    set_autograd(m, False)
    assert m.get_autograd_layer() is None
